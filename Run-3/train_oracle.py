"""
train_oracle.py - Run-3: Phase B — Oracle Training on Chákṣu (Upper Bound)

Run-3 changes vs Run-1/2
─────────────────────────
1. CLASS BALANCE: WeightedRandomSampler + weighted CrossEntropyLoss.
   This is particularly important for the Oracle because Chákṣu is a small
   dataset (~1,009 train images) where class imbalance can cause the model
   to trivially predict the majority class.  In Run-1 the Oracle achieved
   only AUROC=0.586 — barely above chance — almost certainly because of
   uncompensated imbalance causing near-constant "Normal" predictions.

2. WEIGHTED LOSS: Same inverse-frequency weighting as Source training.

3. GRAYSCALE: get_transforms() from dataset_loader.py now applies
   GrayscaleToRGB as the first step.  No explicit code change needed here.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
MIN_DELTA = 1e-5
CSV_PATH = "/workspace/data/processed_csvs/chaksu_train_labeled.csv"
SAVE_DIR = "/workspace/results_run3/Oracle_Chaksu"


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
        # Run-3 additions
        "run3_class_balance": "WeightedRandomSampler + weighted loss",
        "run3_grayscale": "GrayscaleToRGB in transforms",
    }
    exp_logger.log_phase_start("oracle", hyperparameters)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Training CSV not found: {CSV_PATH}")
        print("        Run prepare_data.py first!")
        return

    print("Loading Chákṣu labeled dataset...")
    dataset = GlaucomaDataset(CSV_PATH, transform=get_transforms(is_training=True))

    # ── Run-3: Balanced sampler ─────────────────────────────────────────────
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
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
    # ───────────────────────────────────────────────────────────────────────

    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Run-3: WeightedRandomSampler active — balanced 50/50 batches")

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
    print("   Run-3: Balanced sampling + weighted loss (class balance fix)")
    print("   Run-3: Grayscale inputs (colour bias removal)")
    print("   Note: Uses labeled Chákṣu data — NOT source-free!")
    print("="*60)

    best_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{SAVE_DIR}/best_oracle_model.pth"
    start_time = time.time()

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

        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best loss: {best_loss:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                exp_logger.log_early_stopping("oracle", epoch+1, best_loss)
                print(f"\n⚠ Early stopping triggered!")
                break

    # Finalise: copy best → oracle_model.pth
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, f"{SAVE_DIR}/oracle_model.pth")

    training_time = time.time() - start_time
    exp_logger.log_phase_end("oracle", training_time)
    print(f"\n✅ Oracle Training Complete. Model saved to {SAVE_DIR}/oracle_model.pth")
    print(f"   Best loss: {best_loss:.4f}  |  Time: {training_time/60:.1f} min")


if __name__ == "__main__":
    train()
