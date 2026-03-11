"""
train_oracle.py - Run-4: Phase B — Oracle Training on Chákṣu (Upper Bound)

Run-4 changes vs Run-3
───────────────────────
1. COLOUR RESTORED: GrayscaleToRGB removed in dataset_loader.py.  Run-3
   Oracle achieved AUROC=0.497 (random) because 907 images is too small to
   learn glaucoma features from grayscale.  Colour cues (cup pallor, RNFL
   defects) are essential at this scale.

2. AUROC VALIDATION: A 10% held-out split of chaksu_train_labeled.csv is
   used to compute AUROC every 5 epochs.  Early stopping now uses AUROC
   (not loss) as the primary signal.  If AUROC < 0.55 after 20 epochs the
   run aborts with a clear diagnostic error before wasting GPU time.

3. CLASS BALANCE: WeightedRandomSampler + weighted CrossEntropyLoss kept
   from Run-3 — these were correct and are unchanged.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import time
from models import NetraModel
from dataset_loader import GlaucomaDataset, get_transforms, compute_class_weights, compute_sample_weights
from utils import Logger
from training_logger import get_logger

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 12
MIN_DELTA = 1e-4       # Run-4: slightly looser delta to avoid noise-driven saves
AUROC_ABORT_THRESHOLD = 0.55   # If AUROC < this after 20 epochs, abort
AUROC_CHECK_EVERY = 5          # Check val AUROC every N epochs
CSV_PATH = "/workspace/data/processed_csvs/chaksu_train_labeled.csv"
SAVE_DIR = "/workspace/results_run5/Oracle_Chaksu"


def train():
    torch.set_float32_matmul_precision('high')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger     = Logger(save_dir=SAVE_DIR)
    exp_logger = get_logger()

    # ── Run-3: compute class weights BEFORE building the loader ────────────
    print("Computing class weights for Chákṣu (Run-3 imbalance fix)...")
    class_weights  = compute_class_weights(CSV_PATH).to(DEVICE)
    sample_weights = compute_sample_weights(CSV_PATH)
    print(f"  CrossEntropyLoss class weights: Normal={class_weights[0]:.4f}, "
          f"Glaucoma={class_weights[1]:.4f}")
    # ───────────────────────────────────────────────────────────────────────

    hyperparameters = {
        "dataset": "Chákṣu (Labeled)",
        "train_csv": CSV_PATH,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_DELTA,
        "lr_backbone": 2e-5,
        "lr_head": 2e-3,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "loss": "CrossEntropyLoss (weighted)",
        "device": DEVICE,
        "run4_class_balance": "WeightedRandomSampler + weighted loss",
        "run4_colour": "Full RGB (grayscale removed)",
        "run4_auroc_monitoring": f"check every {AUROC_CHECK_EVERY} epochs, abort if <{AUROC_ABORT_THRESHOLD} at epoch 20",
    }
    exp_logger.log_phase_start("oracle", hyperparameters)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Training CSV not found: {CSV_PATH}")
        print("        Run prepare_data.py first!")
        return

    print("Loading Chákṣu labeled dataset...")
    full_dataset = GlaucomaDataset(CSV_PATH, transform=get_transforms(is_training=True))

    # ── Run-4: 90/10 train/val split for AUROC monitoring ─────────────────
    n_total = len(full_dataset)
    n_val   = max(int(n_total * 0.10), 20)   # at least 20 val samples
    n_train = n_total - n_val
    indices = list(range(n_total))
    # Deterministic split (no shuffle — stratification handled by sampler)
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    train_subset = Subset(full_dataset, train_indices)
    # Val uses eval transforms (no augmentation)
    val_dataset  = GlaucomaDataset(CSV_PATH, transform=get_transforms(is_training=False))
    val_subset   = Subset(val_dataset, val_indices)

    print(f"  Train split: {len(train_subset)} images")
    print(f"  Val   split: {len(val_subset)} images (AUROC monitoring)")
    # ──────────────────────────────────────────────────────────────────────────────

    # Recompute sample_weights for the train subset only
    import pandas as pd
    df_full = pd.read_csv(CSV_PATH)
    df_full = df_full[df_full['label'] >= 0].reset_index(drop=True)
    counts  = df_full['label'].value_counts().to_dict()
    n_cls   = len(counts)
    sw_all  = [n_total / (n_cls * counts[int(df_full.iloc[i]['label'])]) for i in range(n_total)]
    train_sample_weights = [sw_all[i] for i in train_indices]

    dataset = train_subset  # use subset for training

    # ── Run-4: Balanced sampler on train subset ───────────────────────────
    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Dataset size: {len(dataset)} images (train subset)")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Run-4: WeightedRandomSampler active — balanced 50/50 batches")
    print(f"  Run-4: Val AUROC checked every {AUROC_CHECK_EVERY} epochs (abort if <{AUROC_ABORT_THRESHOLD} at ep 20)")

    model = NetraModel(num_classes=2).to(DEVICE)

    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': 2e-5},
        {'params': model.head.parameters(), 'lr': 2e-3}
    ], weight_decay=0.01)

    # ── Run-3: Weighted loss ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ───────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("   PHASE B: ORACLE TRAINING (UPPER BOUND BASELINE)")
    print("   Run-4: Balanced sampling + weighted loss (class balance fix)")
    print("   Run-4: Full RGB colour inputs (grayscale removed)")
    print("   Run-4: AUROC validation monitoring to catch silent failures")
    print("   Note: Uses labeled Chákṣu data — NOT source-free!")
    print("="*60)

    best_auroc = 0.0
    best_loss  = float('inf')
    patience_counter = 0
    best_model_path = f"{SAVE_DIR}/best_oracle_model.pth"
    start_time = time.time()

    def compute_val_auroc(mdl):
        """Run inference on val_loader, return AUROC (-1 on error)."""
        mdl.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    logits = mdl(imgs)
                    probs  = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(lbls.cpu().numpy())
                all_probs.extend(probs.float().cpu().numpy())
        lbls_arr  = np.array(all_labels)
        probs_arr = np.array(all_probs)
        valid = lbls_arr >= 0
        if len(set(lbls_arr[valid])) < 2:
            return -1.0
        try:
            return float(roc_auc_score(lbls_arr[valid], probs_arr[valid]))
        except Exception:
            return -1.0

    for epoch in range(MAX_EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        avg_loss = epoch_loss / len(loader)
        accuracy = 100. * correct / total
        logger.log(epoch+1, avg_loss)
        exp_logger.log_epoch("oracle", epoch+1, MAX_EPOCHS, {"loss": avg_loss, "accuracy": accuracy})
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        # ── Run-4: AUROC validation check ──────────────────────────────────
        if (epoch + 1) % AUROC_CHECK_EVERY == 0:
            val_auroc = compute_val_auroc(model)
            model.train()  # restore train mode after eval
            print(f"  [val] AUROC = {val_auroc:.4f}")
            exp_logger.log_epoch("oracle", epoch+1, MAX_EPOCHS, {"val_auroc": val_auroc})

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ New best val AUROC: {best_auroc:.4f} (saved)")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No AUROC improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")

            # Abort-early check at epoch 20
            if (epoch + 1) == 20 and val_auroc < AUROC_ABORT_THRESHOLD:
                print(f"\n[ABORT] Val AUROC {val_auroc:.4f} < {AUROC_ABORT_THRESHOLD} "
                      f"after 20 epochs — pipeline bug suspected.")
                print("  Possible causes: wrong labels, wrong CSV paths, transform error.")
                print("  Check 'Sample label_map keys' in prepare_data.py output above.")
                break

            if patience_counter >= EARLY_STOP_PATIENCE:
                exp_logger.log_early_stopping("oracle", epoch+1, best_auroc)
                print(f"\n⚠ Early stopping triggered (AUROC plateau)!")
                break
        # ──────────────────────────────────────────────────────────────────────────────

    # Finalise: copy best → oracle_model.pth
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, f"{SAVE_DIR}/oracle_model.pth")

    training_time = time.time() - start_time
    exp_logger.log_phase_end("oracle", training_time)
    print(f"\n✅ Oracle Training Complete. Model saved to {SAVE_DIR}/oracle_model.pth")
    print(f"   Best val AUROC: {best_auroc:.4f}  |  Time: {training_time/60:.1f} min")


if __name__ == "__main__":
    train()
