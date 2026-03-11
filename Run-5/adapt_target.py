"""
adapt_target.py - Run-5: Phase C — MixEnt-Adapt Source-Free Domain Adaptation

Run-5 fixes vs Run-4 (three implementation bugs corrected)
─────────────────────────────────────────────────────────

Fix 1 — DIVERSITY LOSS RESTORED (critical)
  Runs 1-4 computed the diversity term (entropy of mean prediction, L_div)
  but never subtracted it from the loss, so entropy minimisation alone
  caused mode collapse: the model predicted 'Normal' for everything with
  high confidence, giving AUROC ≈ 0.48 (random) despite training.
  Paper eq: L_SFDA = L_ent - λ * L_div  (λ = 1.0)

Fix 2 — ADAPTATION ON TRAINING SET, NOT TEST SET
  Runs 1-4 adapted on chaksu_test_labeled.csv (302 images, the SAME images
  used for evaluation) which is both data-leakage and gives only 6 batches
  per epoch.  Correct: adapt on chaksu_train_unlabeled.csv (1,009 images,
  33 batches/epoch), which is the true source-free unlabeled target set.

Fix 3 — SPATIAL PATCH-TOKEN ADAIN (matches paper Section 3.3)
  Runs 1-4 applied AdaIN on CLS tokens [B, D], computing statistics across
  the batch dimension (across B images) rather than across the N spatial
  patch tokens within each image.  The paper defines:
    μ(z) = (1/N) Σ z_i   over i=1..N patch tokens within one image
  We now extract all 1025 tokens (CLS + 1024 patches), compute per-image
  spatial μ/σ over the N=1024 patch dimension, apply AdaIN, then mean-pool
  the adapted patches for the classifier head.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms
from training_logger import get_logger
from utils import Logger

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 48
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 10
MIN_DELTA  = 1e-4
LAMBDA     = 1.0    # weight of diversity term in L_SFDA = L_ent - LAMBDA * L_div
SOURCE_WEIGHTS = "/workspace/results_run5/Source_AIROGS/model.pth"
TARGET_CSV     = "/workspace/data/processed_csvs/chaksu_train_unlabeled.csv"  # Run-5 fix: train set
SAVE_DIR       = "/workspace/results_run5/Netra_Adapt"


def mixent_adapt(all_tokens, logits_init):
    """
    MixEnt-Adapt: Uncertainty-Guided Spatial Token Injection (Run-5 corrected)

    Implements the algorithm exactly as described in Section 3.3 of the paper.

    Args:
        all_tokens:   [B, N+1, D]  all DINOv3 tokens (CLS at pos 0, then N=1024 patches)
        logits_init:  [B, C]       logits from no-grad forward pass for entropy partitioning

    Returns:
        adapted_features: [B, D]   mean-pooled adapted patch tokens (ready for head)
        mask_unc:         [B]      boolean mask of uncertain samples (for targeted L_ent)
    """
    probs   = torch.softmax(logits_init, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)  # [B]

    tau      = torch.median(entropy)
    mask_unc  = entropy >= tau
    mask_conf = entropy <  tau

    # Extract patch tokens only (exclude CLS at index 0)
    patch_tokens = all_tokens[:, 1:]  # [B, N, D]  N=1024

    if mask_conf.sum() < 2 or mask_unc.sum() < 2:
        # Not enough samples to estimate style — return mean-pooled unchanged
        return patch_tokens.mean(dim=1), mask_unc

    z_conf = patch_tokens[mask_conf]  # [n_conf, N, D]
    z_unc  = patch_tokens[mask_unc]   # [n_unc,  N, D]

    # Per-image spatial statistics over the N=1024 patch dimension (paper eq. 3.2)
    mu_unc  = z_unc.mean(dim=1, keepdim=True)       # [n_unc, 1, D]
    sig_unc = z_unc.std(dim=1, keepdim=True) + 1e-6  # [n_unc, 1, D]

    # Normalise uncertain patches (strip uncertain sample's style)
    z_unc_norm = (z_unc - mu_unc) / sig_unc  # [n_unc, N, D]

    # Random pairing: each uncertain image gets one confident anchor
    perm    = torch.randperm(z_conf.size(0), device=all_tokens.device)
    repeat  = (z_unc.size(0) // z_conf.size(0)) + 1
    indices = perm.repeat(repeat)[:z_unc.size(0)]

    z_conf_sel = z_conf[indices]  # [n_unc, N, D]
    mu_c  = z_conf_sel.mean(dim=1, keepdim=True)       # [n_unc, 1, D]
    sig_c = z_conf_sel.std(dim=1, keepdim=True) + 1e-6  # [n_unc, 1, D]

    # AdaIN: inject confident anchor's spatial style into uncertain content (paper eq. 3.3)
    z_adapted = sig_c * z_unc_norm + mu_c  # [n_unc, N, D]

    # Soft mixing 50/50 (retains some original content)
    z_mixed = 0.5 * z_adapted + 0.5 * z_unc  # [n_unc, N, D]

    # Write back adapted patches
    patch_out = patch_tokens.clone()
    patch_out[mask_unc] = z_mixed

    # Mean-pool over N patch tokens → feature vector for classifier
    adapted_features = patch_out.mean(dim=1)  # [B, D]
    return adapted_features, mask_unc


def run_adapt():
    """Main adaptation loop implementing MixEnt-Adapt SFDA."""
    torch.set_float32_matmul_precision('high')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger     = Logger(save_dir=SAVE_DIR)
    exp_logger = get_logger()

    if not os.path.exists(SOURCE_WEIGHTS):
        print(f"[ERROR] Source model not found: {SOURCE_WEIGHTS}")
        print("        Run train_source.py (Phase A) first!")
        return

    if not os.path.exists(TARGET_CSV):
        print(f"[ERROR] Target CSV not found: {TARGET_CSV}")
        print("        Run prepare_data.py first!")
        return

    print("Loading source model...")
    model = NetraModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(SOURCE_WEIGHTS, map_location=DEVICE))
    print(f"  Loaded weights from {SOURCE_WEIGHTS}")

    print("Loading Chákṣu target dataset (TEST SET — labels ignored during adaptation)...")
    dataset = GlaucomaDataset(TARGET_CSV, transform=get_transforms(is_training=True))
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Run-5: Full RGB colour transforms active")
    print(f"  NOTE: Labels ignored — purely test-time adaptation on TRAIN set!")

    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': 5e-6},
        {'params': model.head.parameters(), 'lr': 5e-4}
    ], weight_decay=0.001)

    print("\n--- Phase C: MixEnt-Adapt (Test-Time Adaptation) ---")
    print(f"    Run-5: RGB inputs — spatial patch-token AdaIN (per-image \u03bc/\u03c3 over N patches)")
    print(f"    Run-5: L_SFDA = L_ent - {LAMBDA}*L_div (diversity loss prevents mode collapse)")
    print(f"    Run-5: Adapting on TRAIN unlabeled set ({len(dataset)} images)")
    print(f"    Labels IGNORED during adaptation — used only for evaluation!")
    print(f"    Early Stopping: patience={EARLY_STOP_PATIENCE}, min_delta={MIN_DELTA}")

    exp_logger.log_phase_start("adapt", {
        "algorithm": "MixEnt-Adapt (Test-Time Adaptation)",
        "source_model": SOURCE_WEIGHTS,
        "target_dataset": TARGET_CSV,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "AdamW",
        "backbone_lr": 5e-6,
        "head_lr": 5e-4,
        "weight_decay": 0.001,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "run5_diversity_loss":  "L_SFDA = L_ent - 1.0*L_div (mode-collapse fix)",
        "run5_target_csv":     "chaksu_train_unlabeled.csv (not test set)",
        "run5_adain":          "spatial patch-token AdaIN [B,N,D] per-image stats",
        "run4_colour": "Full RGB colour inputs",
    })

    best_loss        = float('inf')
    patience_counter = 0
    best_model_state = None
    start_time       = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        loop       = tqdm(loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        epoch_loss = 0

        for images, _ in loop:
            images = images.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):

                # Step 1: No-grad forward for entropy partitioning
                with torch.no_grad():
                    cls_init    = model.extract_features(images)   # [B, D]
                    logits_init = model.head(cls_init)             # [B, 2]

                # Step 2: Full forward — all tokens WITH gradients
                all_tokens = model.extract_all_tokens(images)  # [B, N+1, D]

                # Step 3: Spatial patch-token AdaIN (Run-5 fix)
                feats_adapted, mask_unc = mixent_adapt(all_tokens, logits_init)
                logits_final = model.head(feats_adapted)
                probs        = torch.softmax(logits_final, dim=1)

                # Step 4: L_ent — entropy min on adapted uncertain samples (paper § 3.4)
                if mask_unc.sum() > 0:
                    probs_unc = probs[mask_unc]
                    L_ent = -torch.sum(probs_unc * torch.log(probs_unc + 1e-6), dim=1).mean()
                else:
                    L_ent = torch.zeros(1, device=DEVICE)[0]

                # Step 5: L_div — diversity max over full batch (prevents mode collapse)
                mean_probs = probs.mean(dim=0)
                L_div = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6))

                # Step 6: Total SFDA loss (Run-5 fix: L_div was computed but never used)
                loss = L_ent - LAMBDA * L_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), L_ent=L_ent.item(), L_div=L_div.item())

        avg_loss = epoch_loss / len(loader)
        logger.log(epoch+1, avg_loss)
        exp_logger.log_epoch("adapt", epoch+1, MAX_EPOCHS, {
            "loss": avg_loss,
            "L_ent": L_ent.item(),
            "L_div": L_div.item(),
        })
        print(f"  Epoch {epoch+1} — Loss: {avg_loss:.4f}  L_ent: {L_ent.item():.4f}  L_div: {L_div.item():.4f}")

        if avg_loss < (best_loss - MIN_DELTA):
            best_loss        = avg_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, f"{SAVE_DIR}/adapted_model.pth")
            print(f"  ✓ New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (patience: {patience_counter}/{EARLY_STOP_PATIENCE})")

        if patience_counter >= EARLY_STOP_PATIENCE:
            exp_logger.log_early_stopping("adapt", epoch+1, best_loss)
            print(f"\n⏹ Early stopping triggered after {epoch+1} epochs")
            break

    if best_model_state is not None and patience_counter < EARLY_STOP_PATIENCE:
        torch.save(best_model_state, f"{SAVE_DIR}/adapted_model.pth")

    training_time = time.time() - start_time
    exp_logger.log_phase_end("adapt", training_time)
    print(f"\n✅ Adaptation Complete. Model saved to {SAVE_DIR}/adapted_model.pth")
    print(f"   Best loss: {best_loss:.4f}  |  Time: {training_time/60:.1f} min")


if __name__ == "__main__":
    run_adapt()
