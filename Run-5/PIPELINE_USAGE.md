# Netra-Adapt Run-3 — Pipeline Usage Guide (vast.ai)

## What's new in Run-3

| Feature | Run-1/2 | Run-3 |
|---------|---------|-------|
| Class imbalance | Ignored | `WeightedRandomSampler` + weighted `CrossEntropyLoss` |
| Input colour | RGB (colour-biased) | Grayscale-stacked (3×Gray) |
| Normalisation stats | ImageNet per-channel | Grayscale ImageNet (`mean=0.449, std=0.226`) |
| Augmentation | Strong colour jitter | Geometric + mild brightness/contrast only |
| Result directory | `/workspace/results/` | `/workspace/results_run3/` |

---

## Quick Start on vast.ai

### 1. Instance setup
Recommended: **RTX 3090 / A5000 / A6000** with PyTorch image  
(`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` or similar)

### 2. Upload code
```bash
# From your local machine, sync this folder to /workspace/Run-3/
scp -r Run-3/ root@<vast-ip>:<port>:/workspace/Run-3/
```

### 3. Upload / point to your downloaded data

You don't need to rename anything manually.  Run the setup script and it will
find and reorganise your data automatically, regardless of how you downloaded it:

```bash
cd /workspace/Run-3
bash setup_data.sh
```

It searches common download locations for:

| Dataset | What it looks for |
|---------|-------------------|
| **AIROGS** (Kaggle) | `glaucoma-dataset-eyepacs-airogs-light-v2.zip` or <br> any folder containing `RG/` and `NRG/` sub-folders |
| **Chákṣu** (Figshare) | `Train.zip` + `Test.zip` at `/workspace/data/` or<br> already-extracted `Train/` and `Test/` folders anywhere under `/workspace/` |

The script prints a summary at the end — check the image counts before continuing.

> **Manual override**: if the auto-detection misses your folders, just run:
> ```bash
> mkdir -p /workspace/data/raw_airogs/RG /workspace/data/raw_airogs/NRG
> cp -r /path/to/your/RG   /workspace/data/raw_airogs/RG
> cp -r /path/to/your/NRG  /workspace/data/raw_airogs/NRG
> cp -r /path/to/your/Train /workspace/data/raw_chaksu/
> cp -r /path/to/your/Test  /workspace/data/raw_chaksu/
> ```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the full pipeline
```bash
bash run_everything.sh
```

Or step by step:
```bash
python prepare_data.py    # Creates CSVs + reports class balance
python train_source.py    # Phase A: AIROGS balanced training (~30-60 min)
python train_oracle.py    # Phase B: Chákṣu oracle training (~20-40 min)
python adapt_target.py    # Phase C: MixEnt-Adapt (~15-30 min)
python evaluate.py        # Phase D: Metrics + plots
```

---

## Expected Output

Results are saved to `/workspace/results_run3/`:
```
results_run3/
├── Source_AIROGS/
│   ├── model.pth            ← source model for adaptation
│   ├── best_model.pth
│   └── log.csv
├── Oracle_Chaksu/
│   ├── oracle_model.pth
│   └── log.csv
├── Netra_Adapt/
│   ├── adapted_model.pth    ← final output
│   └── log.csv
└── evaluation/
    ├── results_table.csv    ← main metrics table
    ├── results_table.tex    ← LaTeX-ready
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── metrics_comparison.png
```

---

## Understanding the class balance fix

`prepare_data.py` will print something like:
```
┌─ Class Balance Report: AIROGS full set ─────────────────────
│  Total     :   4000
│  Normal (0):   3600  (90.0%)
│  Glaucoma(1):   400  ( 10.0%)
│  Imbalance ratio (Normal:Glaucoma): 9.00:1
│  ⚠ Significant imbalance — WeightedRandomSampler ACTIVE
└────────────────────────────────────────────────────────────
```

`train_source.py` then automatically applies:
- `WeightedRandomSampler` → each batch has ~50% Normal / ~50% Glaucoma
- `CrossEntropyLoss(weight=[0.556, 5.0])` → glaucoma samples upweighted in loss

---

## Understanding the grayscale fix

- `GrayscaleToRGB()` in `dataset_loader.py` converts each PIL RGB image to luminance grayscale then replicates it across all 3 channels.
- The resulting tensor shape is unchanged: `(3, 512, 512)`.
- DINOv3 model architecture is **unchanged** — it still receives 3-channel input.
- The normalisation uses grayscale ImageNet statistics: `mean=0.449, std=0.226`.
- This strips the pigmentation-driven colour bias that caused AUROC to be near 0.5 in Run-1.

---

## HuggingFace token (for DINOv3 download)

If the model is gated:
```bash
export HF_TOKEN=hf_your_token_here
# or
huggingface-cli login
```
