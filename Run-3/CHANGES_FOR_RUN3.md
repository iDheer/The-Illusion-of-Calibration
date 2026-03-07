# Run-3: Class Balance + Grayscale Changes

## Summary of Changes

Two key improvements added in Run-3 based on reviewer suggestions:

---

## Change 1: Class Imbalance Removal

### Problem
The AIROGS training set has a natural class imbalance (~10:1 NRG:RG in the full dataset). This biases the source model toward predicting "Normal", which then makes the adapted model overfit to the majority class. The Chákṣu oracle training set also has imbalance.

### Fix
1. **`prepare_data.py`**: Reports precise class distribution for all splits with an imbalance ratio.
2. **`dataset_loader.py`**: Added `compute_class_weights(csv_path)` utility to compute per-class inverse-frequency weights.
3. **`train_source.py`**: Uses `torch.utils.data.WeightedRandomSampler` so each batch sees ~50/50 class distribution, AND uses `nn.CrossEntropyLoss(weight=...)` with class weights for a double fix.
4. **`train_oracle.py`**: Same as above for the Oracle (Chákṣu) training — especially important since Chákṣu has fewer glaucoma images.

### Effect Expected
- Source model should no longer be biased toward predicting "Normal"
- Better calibrated predictions → better entropy split in MixEnt-Adapt
- Oracle should see dramatic AUROC improvement (Run-1 oracle AUROC was only 0.586 — far below expected for supervised training, suggesting the imbalance was causing near-constant predictions)

---

## Change 2: Grayscale Images

### Problem
The primary domain gap is **colour/pigmentation**: Indian retinas have higher melanin → darker fundus tessellation. The model conflates pigment darkness with pathological features. However, glaucoma diagnosis is fundamentally a **geometric** task (Cup-to-Disc Ratio, neuroretinal rim), not a colour task.

### Fix
- **`dataset_loader.py`**: Added `GrayscaleToRGB` transform that converts an image to grayscale and replicates it across the three RGB channels (shape stays `(3, H, W)`, compatible with DINOv3 RGB backbone).
- **Normalization**: Switched to grayscale-calibrated ImageNet stats:
  - `mean = [0.449, 0.449, 0.449]`  (vs original per-channel `[0.485, 0.456, 0.406]`)
  - `std  = [0.226, 0.226, 0.226]`  (vs original per-channel `[0.229, 0.224, 0.225]`)
  - These are the standard grayscale ImageNet statistics (luminance-weighted average of the per-channel values)
- Grayscale conversion happens **after** `robust_circle_crop` and **before** all colour augmentations (colour augmentations are dropped since there's no colour signal).

### Why this is valid
- DINOv3 is pretrained on RGB images, so we keep the 3-channel input format.
- Replicating a single grayscale channel to 3 channels is standard practice for feeding grayscale to RGB pretrained models (e.g., medical X-ray → ImageNet models).
- Colour jitter / saturation / hue augmentations are removed (they are now no-ops on grayscale), replaced by more **geometric** augmentations.
- All four model variants (Pretrained, Source, Oracle, Adapted) see grayscale inputs for a fair comparison.

### Effect Expected
- Model forced to learn **shape** features (disc/cup boundary) rather than **colour** features
- Reduces domain gap: both Western and Indian images look similar in grayscale
- Potentially better adaptation since the main domain shift (pigmentation) is removed at the input level

---

## Files Changed vs Netra_Adapt (Run-1/2)

| File | Changed? | What changed |
|------|----------|--------------|
| `dataset_loader.py` | ✅ YES | Added `GrayscaleToRGB`, `compute_class_weights()`, updated `get_transforms()` |
| `prepare_data.py`   | ✅ YES | Added class balance reporting + imbalance ratio |
| `train_source.py`   | ✅ YES | `WeightedRandomSampler` + weighted `CrossEntropyLoss` |
| `train_oracle.py`   | ✅ YES | `WeightedRandomSampler` + weighted `CrossEntropyLoss` |
| `adapt_target.py`   | ✅ YES | Uses updated grayscale-aware `get_transforms()` from dataset_loader |
| `evaluate.py`       | ✅ YES | Uses updated grayscale-aware `get_transforms()` from dataset_loader |
| `models.py`         | ❌ NO  | Architecture unchanged |
| `utils.py`          | ❌ NO  | Unchanged |
| `training_logger.py`| ❌ NO  | Unchanged |
| `run_full_pipeline.py` | ✅ YES | Updated paths to Run-3 results |
| `run_everything.sh` | ✅ YES | Updated result dirs, added Run-3 banner |

---

## Run Commands (vast.ai)

```bash
# 1. Clone or sync code to /workspace/Run-3
# 2. Copy all files from this folder to wherever you run from

cd /workspace/Run-3
pip install -r requirements.txt

# Full pipeline:
bash run_everything.sh

# Or step-by-step:
python prepare_data.py
python train_source.py
python train_oracle.py
python adapt_target.py
python evaluate.py
```

Results will be saved to `/workspace/results_run3/`.
