"""
adapt_target.py - Run-3: Phase C — MixEnt-Adapt Source-Free Domain Adaptation

Run-3 changes vs Run-1/2
─────────────────────────
1. GRAYSCALE: get_transforms() from dataset_loader.py now applies
   GrayscaleToRGB as the first step — no explicit code change needed.
   The MixEnt algorithm operates on feature-space statistics, which are
   now derived from grayscale-aware features.  Because colour-based
   domain shift has been removed at the input level, the entropy partition
   in MixEnt is driven by remaining geometric/structural uncertainty,
   making the style injection more meaningful.

2. RESULT PATHS: Updated to /workspace/results_run3/ for clean separation.

The MixEnt-Adapt algorithm itself is unchanged: we still perform test-time
adaptation on the unlabeled Chákṣu test set using entropy-guided AdaIN.
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
MIN_DELTA = 1e-5
SOURCE_WEIGHTS = "/workspace/results_run3/Source_AIROGS/model.pth"
TARGET_CSV     = "/workspace/data/processed_csvs/chaksu_test_labeled.csv"
SAVE_DIR       = "/workspace/results_run3/Netra_Adapt"


def mixent_adapt(features, logits):
    """
    MixEnt-Adapt: Uncertainty-Guided Token Injection

    Partition batch by entropy → inject confident feature statistics
    (mean/std) into uncertain samples via AdaIN.  In Run-3 the features
    are derived from grayscale images, so style statistics reflect
    structural variation rather than pigmentation variation.

    Args:
        features: [B, D] CLS-token embeddings from DINOv3
        logits:   [B, C] classification logits for entropy computation

    Returns:
        features with AdaIN-style-injected uncertain samples
    """
    probs   = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)  # [B]

    tau = torch.median(entropy)
    mask_unc  = entropy >= tau
    mask_conf = entropy < tau

    if mask_conf.sum() < 2 or mask_unc.sum() < 2:
        return features

    z_conf = features[mask_conf]
    z_unc  = features[mask_unc]

    # Statistics across feature dimension for CLS token
    mu_conf  = z_conf.mean(dim=0, keepdim=True)
    sig_conf = z_conf.std(dim=0, keepdim=True) + 1e-6

    # Per-sample normalisation of uncertain features
    mu_unc  = z_unc.mean(dim=1, keepdim=True)
    sig_unc = z_unc.std(dim=1, keepdim=True) + 1e-6
    z_unc_norm = (z_unc - mu_unc) / sig_unc

    # Random pairing: each uncertain sample gets a confident anchor
    perm   = torch.randperm(z_conf.size(0))
    repeat = (z_unc.size(0) // z_conf.size(0)) + 1
    indices = perm.repeat(repeat)[:z_unc.size(0)]

    z_conf_selected = z_conf[indices]
    mu_c_sel  = z_conf_selected.mean(dim=1, keepdim=True)
    sig_c_sel = z_conf_selected.std(dim=1, keepdim=True) + 1e-6

    # AdaIN: z_adapted = sigma_c * ((z_u - mu_u) / sigma_u) + mu_c
    z_adapted = sig_c_sel * z_unc_norm + mu_c_sel

    # Soft mixing (50/50)
    z_mixed = 0.5 * z_adapted + 0.5 * z_unc

    features_out = features.clone()
    features_out[mask_unc] = z_mixed
    return features_out


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
    # Run-3: get_transforms() already applies GrayscaleToRGB
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
    print(f"  Run-3: Grayscale-aware transforms active")
    print(f"  NOTE: Labels exist but are NOT used — purely test-time adaptation!")

    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': 5e-6},
        {'params': model.head.parameters(), 'lr': 5e-4}
    ], weight_decay=0.001)

    print("\n--- Phase C: MixEnt-Adapt (Test-Time Adaptation) ---")
    print(f"    Run-3: Grayscale inputs — style injection targets structural uncertainty")
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
        # Run-3 additions
        "run3_grayscale": "GrayscaleToRGB in transforms",
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

                with torch.no_grad():
                    feats_init  = model.extract_features(images)
                    logits_init = model.head(feats_init)

                feats_train   = model.extract_features(images)
                feats_adapted = mixent_adapt(feats_train, logits_init)
                logits_final  = model.head(feats_adapted)
                probs         = torch.softmax(logits_final, dim=1)

                # Entropy Minimisation loss
                entropy_per_sample = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                loss = torch.mean(entropy_per_sample)

                mean_probs    = probs.mean(dim=0)
                entropy_batch = -torch.sum(mean_probs * torch.log(mean_probs + 1e-6))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), ent=loss.item(), batch_div=entropy_batch.item())

        avg_loss = epoch_loss / len(loader)
        logger.log(epoch+1, avg_loss)
        exp_logger.log_epoch("adapt", epoch+1, MAX_EPOCHS, {
            "loss": avg_loss,
            "entropy": avg_loss,
            "batch_diversity": entropy_batch.item()
        })
        print(f"  Epoch {epoch+1} — Avg Entropy: {avg_loss:.4f}  (Batch Div: {entropy_batch.item():.4f})")

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
