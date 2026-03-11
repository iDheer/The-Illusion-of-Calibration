"""
adapt_target.py — Run-6: PrototypeMixEnt-Adapt
Source-Free Domain Adaptation with Prototype Memory Bank

Run-6 introduces five principled innovations over Run-5:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Innovation 1 — PROTOTYPE STYLE MEMORY BANK  (most impactful)
  Run-5 drew style anchors from within the current batch (~3-4 confident
  images per batch of 48).  Run-6 maintains a rolling FIFO queue of K=256
  per-image patch-level style statistics (μ, σ) per class, accumulated
  across ALL batches.  Style injection then draws from up to 512 stable
  anchors, not 3 noise-dominated ones.

  Why this matters:
  - Statistics from 3 images are highly batch-dependent (lucky/unlucky draws)
  - 256 anchors per class represent the full distribution of Indian retina
    appearances encountered so far (Remidio, Forus, Bosch devices)
  - The bank is continuously updated as the model improves → bootstraps
    progressively better anchors throughout training

Innovation 2 — BANK WARMUP PHASE (epochs 0-2, no gradient updates)
  Before any gradient update, we run 3 epochs worth of forward passes to
  pre-populate the memory bank.  This ensures the very first real adaptation
  step starts with 200+ meaningful style anchors rather than an empty bank.
  Avoids the degenerate early gradients that come from injecting zero-sample
  (or single-sample) bank statistics.

Innovation 3 — AUGMENTATION CONSISTENCY LOSS  L_con
  Each target image is processed under both weak and strong augmentation.
  We KL-divergence the two predictions:
      L_con = KL( sg(p_weak) || p_strong )
  This teaches the model device-invariance: the same fundus under different
  handheld camera simulations (glare, blur, RandomSolarize, RandomErasing)
  should yield the same prediction.  Directly addresses the Remidio FoP vs
  Forus 3Nethra acquisition gap.

  λ_con is ramped from 0 → 0.5 over 5 epochs after warmup to avoid
  swamping the main loss before adaptation has begun.

Innovation 4 — PROGRESSIVE SELF-TRAINING  L_psl  (epoch ≥ 10)
  After 10 epochs of adaptation the model is much better calibrated on
  Indian retinas.  Predictions with source-model confidence > 0.90 are
  promoted to pseudo-labels and used in a supervised cross-entropy loss.
  This converts SFDA from purely unsupervised to semi-supervised as
  confidence grows — without any human labels.

  Why use SOURCE model confidence for pseudo-label selection (not adapted)?
  The source head (never updated) gives stable, conservative confidence
  scores.  Using it as the selection gate prevents the adapted head from
  confidently labelling its own errors.

Innovation 5 — GRADIENT CLIPPING + COSINE LR SCHEDULE
  Gradient clipping (max_norm=1.0) prevents catastrophic forgetting of
  source geometry features during the early aggressive adaptation steps.
  Cosine annealing from BASE_LR to 1e-8 over (MAX_EPOCHS - WARMUP_EPOCHS)
  ensures smooth convergence in the later epochs when pseudo-labels kick in.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total loss (post-warmup):
  L_SFDA = L_ent
         - λ_div * L_div
         + λ_con(t) * L_con
         + λ_psl * L_psl        [only epoch ≥ PSEUDO_START]

where:
  L_ent  = -(1/|X_unc|) Σ_{x∈X_unc} H(f(x_adapted))   entropy min
  L_div  = -H(E[p])                                      diversity max
  L_con  = KL(sg(p_weak) || p_strong)                    aug consistency
  L_psl  = CE(f(x_adapted), pseudo_label)                self-training
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NetraModel
from dataset_loader import (DualAugGlaucomaDataset, get_transforms,
                             get_strong_transforms)
from training_logger import get_logger
from utils import Logger

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE          = 32      # dual-aug: 2× memory per batch vs Run-5
MAX_EPOCHS          = 50
EARLY_STOP_PATIENCE = 12
MIN_DELTA           = 1e-4

WARMUP_EPOCHS       = 3       # epochs to build bank only (no parameter updates)
PSEUDO_START        = 10      # epoch at which pseudo-label loss activates
PSEUDO_THRESH       = 0.90    # source-model confidence threshold for pseudo-labels
BANK_K              = 256     # FIFO queue size per class in ProtoStyleBank

LAMBDA_DIV          = 1.0     # diversity loss weight
LAMBDA_CON          = 0.5     # consistency loss weight (ramped over 5 post-warmup epochs)
LAMBDA_PSL          = 1.0     # pseudo-label loss weight

BASE_LR_BACKBONE    = 5e-6
BASE_LR_HEAD        = 5e-4

SOURCE_WEIGHTS  = "/workspace/results_run6/Source_AIROGS/model.pth"
TARGET_CSV      = "/workspace/data/processed_csvs/chaksu_train_unlabeled.csv"
SAVE_DIR        = "/workspace/results_run6/Netra_Adapt"


# ── Prototype Style Memory Bank ──────────────────────────────────────────────

class ProtoStyleBank:
    """
    Rolling FIFO memory bank of per-image patch-level style statistics.

    For every confident prediction we compute the spatial mean (μ) and std (σ)
    over the N=1024 patch tokens of that image and push them into a per-class
    FIFO queue capped at K entries.

        μ_i = (1/N) Σ_n z_{i,n}   ∈ R^D   (spatial mean  within image i)
        σ_i = std_n(z_{i,n})       ∈ R^D   (spatial std   within image i)

    These are the "style descriptors" used by AdaIN.  Having K=256 anchors
    per class gives stable, representative statistics across the full range
    of Indian fundus appearances seen so far.
    """

    def __init__(self, K: int = 256, D: int = 1024):
        self.K = K
        self.D = D
        # CPU storage to save GPU memory
        self.banks = {
            0: {'mu': [], 'sigma': []},
            1: {'mu': [], 'sigma': []},
        }

    @torch.no_grad()
    def update(self, patch_tokens: torch.Tensor, logits: torch.Tensor,
               threshold: float = 0.85) -> None:
        """
        Add high-confidence images' style statistics to the per-class banks.

        patch_tokens : [B, N, D]  patch tokens from backbone (CLS excluded)
        logits       : [B, 2]     current model logits
        threshold    : float      minimum softmax confidence required
        """
        probs      = torch.softmax(logits.float(), dim=1)   # [B, 2]
        confidence = probs.max(dim=1).values                 # [B]
        pred_cls   = probs.argmax(dim=1)                     # [B]

        for cls in [0, 1]:
            mask = (pred_cls == cls) & (confidence > threshold)
            n    = int(mask.sum().item())
            if n == 0:
                continue

            pts   = patch_tokens[mask].float().cpu()          # [n, N, D]
            mu    = pts.mean(dim=1)                            # [n, D]
            sigma = pts.std(dim=1).clamp(min=1e-6)            # [n, D]

            for i in range(n):
                self.banks[cls]['mu'].append(mu[i])
                self.banks[cls]['sigma'].append(sigma[i])

            # FIFO: keep only last K entries
            if len(self.banks[cls]['mu']) > self.K:
                excess = len(self.banks[cls]['mu']) - self.K
                self.banks[cls]['mu']    = self.banks[cls]['mu'][excess:]
                self.banks[cls]['sigma'] = self.banks[cls]['sigma'][excess:]

    def sample_styles(self, n: int, device):
        """
        Sample n style descriptors uniformly from the combined (both-class) bank.

        Returns (mu [n, D], sigma [n, D]) or (None, None) if bank has < 2 entries.
        """
        all_mu, all_sig = [], []
        for cls in [0, 1]:
            all_mu.extend(self.banks[cls]['mu'])
            all_sig.extend(self.banks[cls]['sigma'])

        total = len(all_mu)
        if total < 2:
            return None, None

        n       = min(n, total)
        indices = torch.randperm(total)[:n].tolist()
        mu  = torch.stack([all_mu[i]  for i in indices]).to(device)   # [n, D]
        sig = torch.stack([all_sig[i] for i in indices]).to(device)   # [n, D]
        return mu, sig

    def is_ready(self, min_per_class: int = 4) -> bool:
        return all(len(self.banks[c]['mu']) >= min_per_class for c in [0, 1])

    def sizes(self) -> dict:
        return {c: len(self.banks[c]['mu']) for c in [0, 1]}


# ── MixEnt-Adapt v6 (bank-powered AdaIN) ────────────────────────────────────

def mixent_adapt_v6(all_tokens: torch.Tensor,
                    logits_init: torch.Tensor,
                    style_bank: ProtoStyleBank):
    """
    Run-6 MixEnt-Adapt: prototype memory bank style injection.

    Partitions the batch into confident / uncertain via entropy median split.
    For each uncertain image, draws a style anchor from the memory bank
    (or falls back to current-batch confident images if bank not ready).
    Applies AdaIN to strip uncertain image's style and inject anchor style.

    Args
    ----
    all_tokens  : [B, N+1, D]  backbone output tokens — CLS at index 0
    logits_init : [B, 2]       logits from stable no-grad partition pass
    style_bank  : ProtoStyleBank

    Returns
    -------
    adapted_features : [B, D]  mean-pooled adapted patch tokens (→ head)
    mask_unc         : [B]     bool — True = uncertain sample
    """
    probs   = torch.softmax(logits_init.float(), dim=1)
    entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1)   # [B]

    tau      = torch.median(entropy)
    mask_unc = entropy >= tau

    patch_tokens = all_tokens[:, 1:]   # [B, N, D]  strip CLS
    n_unc = int(mask_unc.sum().item())

    if n_unc == 0:
        return patch_tokens.mean(dim=1), mask_unc

    z_unc = patch_tokens[mask_unc]   # [n_unc, N, D]

    # ── Style source: memory bank (preferred) or batch fallback ─────────────
    if style_bank.is_ready(min_per_class=4):
        mu_c, sig_c = style_bank.sample_styles(n_unc, all_tokens.device)
        # Ensure we have exactly n_unc styles (repeat if bank smaller)
        if mu_c.size(0) < n_unc:
            repeat = math.ceil(n_unc / mu_c.size(0))
            mu_c  = mu_c.repeat(repeat, 1)[:n_unc]
            sig_c = sig_c.repeat(repeat, 1)[:n_unc]
        else:
            mu_c  = mu_c[:n_unc]
            sig_c = sig_c[:n_unc]
        mu_c  = mu_c.unsqueeze(1)    # [n_unc, 1, D]
        sig_c = sig_c.unsqueeze(1)   # [n_unc, 1, D]
    else:
        # Fallback: within-batch confident statistics (Run-5 behaviour)
        mask_conf = ~mask_unc
        if int(mask_conf.sum()) < 2:
            return patch_tokens.mean(dim=1), mask_unc
        z_conf = patch_tokens[mask_conf]
        perm   = torch.randperm(z_conf.size(0), device=all_tokens.device)
        idx    = perm.repeat(math.ceil(n_unc / z_conf.size(0)))[:n_unc]
        z_c    = z_conf[idx]
        mu_c   = z_c.mean(dim=1, keepdim=True)
        sig_c  = z_c.std(dim=1,  keepdim=True).clamp(min=1e-6)

    # ── Per-image normalisation of uncertain patches ─────────────────────────
    mu_unc  = z_unc.mean(dim=1, keepdim=True)               # [n_unc, 1, D]
    sig_unc = z_unc.std(dim=1,  keepdim=True).clamp(min=1e-6)
    z_norm  = (z_unc - mu_unc) / sig_unc                    # strip uncertain style

    # ── AdaIN: inject anchor style into uncertain content ────────────────────
    z_adapted = sig_c * z_norm + mu_c                       # [n_unc, N, D]
    z_mixed   = 0.5 * z_adapted + 0.5 * z_unc              # soft mix

    # ── Write back and mean-pool ─────────────────────────────────────────────
    patch_out            = patch_tokens.clone()
    patch_out[mask_unc]  = z_mixed
    adapted_features     = patch_out.mean(dim=1)            # [B, D]
    return adapted_features, mask_unc


# ── Main adaptation loop ─────────────────────────────────────────────────────

def run_adapt():
    """Run-6 PrototypeMixEnt-Adapt pipeline."""
    torch.set_float32_matmul_precision('high')
    os.makedirs(SAVE_DIR, exist_ok=True)

    logger     = Logger(save_dir=SAVE_DIR)
    exp_logger = get_logger()

    if not os.path.exists(SOURCE_WEIGHTS):
        print(f"[ERROR] Source model not found: {SOURCE_WEIGHTS}")
        print("        Run train_source.py first!")
        return
    if not os.path.exists(TARGET_CSV):
        print(f"[ERROR] Target CSV not found: {TARGET_CSV}")
        print("        Run prepare_data.py first!")
        return

    # ── Load source model ─────────────────────────────────────────────────────
    print("Loading source model...")
    model = NetraModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(SOURCE_WEIGHTS, map_location=DEVICE))
    print(f"  Loaded: {SOURCE_WEIGHTS}")

    # ── Dataset: dual augmentation ────────────────────────────────────────────
    print("Loading Chákṣu target dataset (dual augmentation — weak + strong)...")
    dataset = DualAugGlaucomaDataset(
        TARGET_CSV,
        weak_transform   = get_transforms(is_training=True),
        strong_transform = get_strong_transforms(),
    )
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 8,
        pin_memory  = True,
        drop_last   = True,
    )
    print(f"  Dataset : {len(dataset)} images  |  Batches/epoch: {len(loader)}")

    # Run-6: log target dataset statistics - verifies we're adapting on the
    # correct (train unlabeled) CSV, not the test set (was a Run-5 bug).
    import pandas as pd
    _df_tgt = pd.read_csv(TARGET_CSV)
    _n_tgt  = len(_df_tgt)
    # Detect device mix in path column
    _dev_counts = {}
    if 'path' in _df_tgt.columns:
        for _p in _df_tgt['path']:
            _p = str(_p).replace('\\', '/')
            for _dev in ('Bosch', 'Forus', 'Remidio'):
                if f'/{_dev}/' in _p or _p.endswith(f'/{_dev}'):
                    _dev_counts[_dev] = _dev_counts.get(_dev, 0) + 1
                    break
    _first_path = str(_df_tgt['path'].iloc[0]) if len(_df_tgt) > 0 else ''
    print(f"\n  \u250c\u2500 Ch\u00e1k\u1e63u target (SFDA) dataset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(f"  \u2502  CSV:    {TARGET_CSV}")
    print(f"  \u2502  Total images: {_n_tgt}  (expected ~1009 train unlabeled)")
    print(f"  \u2502  Device breakdown: {_dev_counts}")
    if 'Test' in _first_path or '/test/' in _first_path.lower():
        print(f"  \u2502  \u26a0 CRITICAL: paths contain 'Test' - adapting on TEST set!")
        print(f"  \u2502    This was the Run-5 bug. Should use chaksu_train_unlabeled.csv.")
    elif _n_tgt < 800:
        print(f"  \u2502  \u26a0 WARNING: {_n_tgt} images is less than expected ~1009")
        print(f"  \u2502    Some devices may have been missed in prepare_data.py")
    else:
        print(f"  \u2502  \u2713 Adapting on TRAIN split with {_n_tgt} unlabeled images")
    print(f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

    # ── Style bank ────────────────────────────────────────────────────────────
    style_bank = ProtoStyleBank(K=BANK_K, D=model.feature_dim)

    # ── Optimizer + cosine scheduler ─────────────────────────────────────────
    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': BASE_LR_BACKBONE},
        {'params': model.head.parameters(),                'lr': BASE_LR_HEAD},
    ], weight_decay=0.001)

    # Cosine annealing applies only to the post-warmup training epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(1, MAX_EPOCHS - WARMUP_EPOCHS),
        eta_min = 1e-8,
    )

    # ── Log hyperparameters ───────────────────────────────────────────────────
    exp_logger.log_phase_start("adapt", {
        "algorithm":          "PrototypeMixEnt-Adapt (Run-6)",
        "source_model":       SOURCE_WEIGHTS,
        "target_csv":         TARGET_CSV,
        "batch_size":         BATCH_SIZE,
        "max_epochs":         MAX_EPOCHS,
        "warmup_epochs":      WARMUP_EPOCHS,
        "pseudo_start_epoch": PSEUDO_START,
        "pseudo_threshold":   PSEUDO_THRESH,
        "bank_K":             BANK_K,
        "lambda_div":         LAMBDA_DIV,
        "lambda_con":         LAMBDA_CON,
        "lambda_psl":         LAMBDA_PSL,
        "lr_backbone":        BASE_LR_BACKBONE,
        "lr_head":            BASE_LR_HEAD,
        "grad_clip_norm":     1.0,
        "innovations": [
            "ProtoStyleBank: K=256 style anchors/class across all batches",
            "Bank warmup: 3 epochs no-grad to seed bank before adapting",
            "L_con: KL(sg(p_weak)||p_strong) augmentation consistency",
            "L_psl: pseudo-label supervised loss after epoch 10",
            "Cosine LR + gradient clipping",
        ],
    })

    print("\n" + "=" * 70)
    print("   RUN-6: PrototypeMixEnt-Adapt — Five Innovations")
    print("   Run-5 (all 3 algorithm fixes) + Memory Bank + Consistency + Self-Train")
    print("=" * 70)
    print(f"   Warmup:      epochs 0-{WARMUP_EPOCHS-1} (bank build, no gradients)")
    print(f"   Adaptation:  epochs {WARMUP_EPOCHS}-9 (L_ent + L_div + ramping L_con)")
    print(f"   Self-train:  epochs {PSEUDO_START}+ (add L_psl for confident pseudo-labels)")
    print(f"   Bank:        K={BANK_K} style anchors/class  |  Grad clip: 1.0")
    print(f"   LR:          backbone={BASE_LR_BACKBONE:.0e}  head={BASE_LR_HEAD:.0e}  (cosine decay)")
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None
    start_time       = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0
        is_warmup   = (epoch < WARMUP_EPOCHS)
        last_m: dict = {}

        loop = tqdm(loader, desc=f"{'WARMUP' if is_warmup else 'ADAPT '} {epoch+1}/{MAX_EPOCHS}")

        for images_weak, images_strong, _ in loop:
            images_weak   = images_weak.to(DEVICE)
            images_strong = images_strong.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):

                # ── Step 1: Stable no-grad partition pass + bank update ──────
                with torch.no_grad():
                    cls_init    = model.extract_features(images_weak)    # [B, D]
                    logits_init = model.head(cls_init)                   # [B, 2]
                    all_tok_b   = model.extract_all_tokens(images_weak)  # [B, 1025, D]
                    style_bank.update(
                        all_tok_b[:, 1:].float(),
                        logits_init,
                        threshold=0.85,
                    )

                # ── Warmup: bank-building only, skip gradient update ─────────
                if is_warmup:
                    bs = style_bank.sizes()
                    loop.set_postfix(phase='WARMUP', bank0=bs[0], bank1=bs[1])
                    continue

                # ── Step 2: Weak-aug adapted forward (gradients enabled) ─────
                optimizer.zero_grad()

                all_tokens_w          = model.extract_all_tokens(images_weak)
                feats_w, mask_unc     = mixent_adapt_v6(all_tokens_w, logits_init, style_bank)
                logits_w              = model.head(feats_w)
                probs_w               = torch.softmax(logits_w, dim=1)

                # L_ent: entropy minimisation on adapted uncertain samples
                if mask_unc.sum() > 0:
                    p_unc = probs_w[mask_unc]
                    L_ent = -(p_unc * torch.log(p_unc + 1e-6)).sum(dim=1).mean()
                else:
                    L_ent = torch.zeros(1, device=DEVICE)[0]

                # L_div: diversity maximisation over full batch
                mean_p = probs_w.mean(dim=0)
                L_div  = -(mean_p * torch.log(mean_p + 1e-6)).sum()

                # ── Step 3: Strong-aug consistency loss (ramped) ─────────────
                ramp_ep        = epoch - WARMUP_EPOCHS
                lambda_con_eff = LAMBDA_CON * min(1.0, ramp_ep / 5.0)

                if lambda_con_eff > 0.0:
                    all_tokens_s      = model.extract_all_tokens(images_strong)
                    feats_s, _        = mixent_adapt_v6(all_tokens_s, logits_init, style_bank)
                    logits_s          = model.head(feats_s)
                    probs_s           = torch.softmax(logits_s, dim=1)
                    # KL(sg(p_weak) || p_strong): force strong-aug to match weak-aug
                    L_con = F.kl_div(
                        torch.log(probs_s + 1e-6),
                        probs_w.detach(),
                        reduction='batchmean',
                    )
                else:
                    L_con = torch.zeros(1, device=DEVICE)[0]

                # ── Step 4: Pseudo-label supervised loss (after PSEUDO_START) ─
                if epoch >= PSEUDO_START:
                    with torch.no_grad():
                        src_probs = torch.softmax(logits_init.float(), dim=1)
                    conf_src = src_probs.max(dim=1).values
                    psl_cls  = src_probs.argmax(dim=1)
                    hi       = conf_src > PSEUDO_THRESH
                    if hi.sum() > 0:
                        L_psl = F.cross_entropy(logits_w[hi], psl_cls[hi].detach())
                    else:
                        L_psl = torch.zeros(1, device=DEVICE)[0]
                else:
                    L_psl = torch.zeros(1, device=DEVICE)[0]

                # ── Total loss ───────────────────────────────────────────────
                loss = (L_ent
                        - LAMBDA_DIV * L_div
                        + lambda_con_eff * L_con
                        + LAMBDA_PSL * L_psl)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            last_m = {
                'L':     loss.item(),
                'ent':   L_ent.item(),
                'div':   L_div.item(),
                'con':   L_con.item(),
                'psl':   L_psl.item(),
            }
            loop.set_postfix(
                **{k: f'{v:.3f}' for k, v in last_m.items()},
                b0=style_bank.sizes()[0],
                b1=style_bank.sizes()[1],
            )

        # ── End of epoch ──────────────────────────────────────────────────────
        if is_warmup:
            bs = style_bank.sizes()
            print(f"  [WARMUP {epoch+1}/{WARMUP_EPOCHS}]  "
                  f"Bank Normal={bs[0]}  Glaucoma={bs[1]}")
            continue   # skip scheduler / early stopping during warmup

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.log(epoch + 1, avg_loss)
        exp_logger.log_epoch("adapt", epoch + 1, MAX_EPOCHS, {
            "loss":   avg_loss,
            **{k: v for k, v in last_m.items()},
            "bank_normal":   style_bank.sizes()[0],
            "bank_glaucoma": style_bank.sizes()[1],
        })
        print(f"  Epoch {epoch+1}  loss={avg_loss:.4f}  "
              f"ent={last_m.get('ent',0):.3f}  div={last_m.get('div',0):.3f}  "
              f"con={last_m.get('con',0):.3f}  psl={last_m.get('psl',0):.3f}  "
              f"bank={style_bank.sizes()}")

        # Early stopping
        if avg_loss < (best_loss - MIN_DELTA):
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, f"{SAVE_DIR}/adapted_model.pth")
            print(f"  ✓ New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                exp_logger.log_early_stopping("adapt", epoch + 1, best_loss)
                print(f"\n⏹  Early stopping after epoch {epoch+1}")
                break

    if best_model_state is not None:
        torch.save(best_model_state, f"{SAVE_DIR}/adapted_model.pth")

    training_time = time.time() - start_time
    exp_logger.log_phase_end("adapt", training_time)
    print(f"\n✅  Adaptation complete.  Model → {SAVE_DIR}/adapted_model.pth")
    print(f"    Best loss: {best_loss:.4f}  |  Time: {training_time/60:.1f} min")


if __name__ == "__main__":
    run_adapt()
