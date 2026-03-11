"""
train_source.py - Run-3: Phase A — Source Training on AIROGS

   Run-4 changes vs Run-3
───────────────────────
1. CLASS BALANCE: WeightedRandomSampler + weighted CrossEntropyLoss kept
   from Run-3 — these were correct.

2. COLOUR RESTORED: GrayscaleToRGB removed in dataset_loader.py.
   Run-3 showed grayscale destroys diagnostic signal on the small
   Chákṣu dataset (Oracle AUROC=0.497). Colour inputs restored.
   Full colour jitter (saturation + hue) also restored.
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
BATCH_SIZE = 48
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 8
MIN_DELTA = 1e-5
CSV_PATH  = "/workspace/data/processed_csvs/airogs_train.csv"
SAVE_DIR  = "/workspace/results_run5/Source_AIROGS"


def train():
    torch.set_float32_matmul_precision('high')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger     = Logger(save_dir=SAVE_DIR)
    exp_logger = get_logger()

    # ── Run-3: compute class weights BEFORE building the loader ────────────
    print("Computing class weights for AIROGS (Run-3 imbalance fix)...")
    class_weights  = compute_class_weights(CSV_PATH).to(DEVICE)   # [w0, w1]
    sample_weights = compute_sample_weights(CSV_PATH)              # per-sample weights

    n0 = sample_weights.count(sample_weights[0])  # rough count — just for logging
    print(f"  CrossEntropyLoss class weights: Normal={class_weights[0]:.4f}, "
          f"Glaucoma={class_weights[1]:.4f}")
    # ───────────────────────────────────────────────────────────────────────

    hyperparameters = {
        "dataset": "AIROGS",
        "train_csv": CSV_PATH,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_DELTA,
        "lr_backbone": 1e-5,
        "lr_head": 1e-3,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "loss": "CrossEntropyLoss (weighted)",
        "device": DEVICE,
        "run4_class_balance": "WeightedRandomSampler + weighted loss",
        "run4_colour": "Full RGB (grayscale removed)",
    }
    exp_logger.log_phase_start("source", hyperparameters)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Training CSV not found: {CSV_PATH}")
        print("        Run prepare_data.py first!")
        return

    # Load dataset
    print("Loading AIROGS dataset...")
    dataset = GlaucomaDataset(CSV_PATH, transform=get_transforms(is_training=True))

    # ── Run-3: Balanced sampler ─────────────────────────────────────────────
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True    # with replacement so minority class is over-sampled
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,       # NOTE: shuffle=True must be removed when using a sampler
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    # ───────────────────────────────────────────────────────────────────────

    print(f"  Dataset size: {len(dataset)} images")
    print(f"  Batches per epoch: {len(loader)}")
    print(f"  Run-3: WeightedRandomSampler active — balanced 50/50 batches")

    # Initialize model
    model = NetraModel(num_classes=2).to(DEVICE)

    optimizer = optim.AdamW([
        {'params': model.backbone.layer[-2:].parameters(), 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)

    # ── Run-3: Weighted loss ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ───────────────────────────────────────────────────────────────────────

    print("\n" + "="*60)
    print("   PHASE A: SOURCE TRAINING ON AIROGS (WESTERN EYES)")
    print("   Run-3: Balanced sampling + weighted loss (class balance fix)")
    print("   Run-5: Full RGB colour inputs (grayscale removed in Run-4)")
    print("   Early Stopping: patience={}, min_delta={}".format(EARLY_STOP_PATIENCE, MIN_DELTA))
    print("="*60)

    best_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{SAVE_DIR}/best_model.pth"
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
        exp_logger.log_epoch("source", epoch+1, MAX_EPOCHS, {"loss": avg_loss, "accuracy": accuracy})
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
                exp_logger.log_early_stopping("source", epoch+1, best_loss)
                print(f"\n⚠ Early stopping triggered!")
                break

    # Finalise: copy best → model.pth for downstream scripts
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, f"{SAVE_DIR}/model.pth")

    training_time = time.time() - start_time
    exp_logger.log_phase_end("source", training_time)
    print(f"\n✅ Source Training Complete. Model saved to {SAVE_DIR}/model.pth")
    print(f"   Best loss: {best_loss:.4f}  |  Time: {training_time/60:.1f} min")


if __name__ == "__main__":
    train()
