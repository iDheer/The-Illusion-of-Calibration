# Run-5 Results — Netra-Adapt

**Date**: 2026-03-11  
**Total runtime**: 1h 25m 42s  
**GPU**: NVIDIA GeForce RTX 5090 (sm_120)

---

## Final Evaluation Results (Chákṣu Test Set)

| Model | AUROC | Sensitivity | Specificity | Precision | F1-Score | Accuracy | Sens@95%Spec |
|---|---|---|---|---|---|---|---|
| Pretrained → Chákṣu (zero-shot) | 0.9750 | 1.0000 | 0.9750 | 0.5000 | 0.6667 | 0.9756 | 1.0000 |
| AIROGS → Chákṣu (source only) | 0.9750 | 1.0000 | 0.9750 | 0.5000 | 0.6667 | 0.9756 | 1.0000 |
| Chákṣu → Chákṣu (Oracle) | 0.8250 | 1.0000 | 0.8250 | 0.1250 | 0.2222 | 0.8293 | 0.0000 |
| AIROGS+Adapt → Chákṣu (Netra-Adapt) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**AIROGS sanity check (source domain)**: AUROC = 0.9790 ✅

---

## Training Summary

### Phase A — Source Training (AIROGS)
- Epochs: 60/60 (ran to completion — no early stopping triggered)
- Best train loss: 0.0783 at epoch 57
- Final train accuracy: 96.4%
- Time: 61 min

### Phase B — Oracle Training (Chákṣu labeled)
- Train set: 84 images (69 normal, 15 glaucoma — 80% of labeled train split)
- Val set: 20 images
- Best val AUROC: **0.7969** (epoch 60)
- Epochs: 80/80 (ran to completion)
- Time: 3.6 min

### Phase C — Adaptation (MixEnt-Adapt)
- Target CSV: `chaksu_train_unlabeled.csv` (1009 images, 21 batches/epoch)
- Epochs: 40/40 (ran to completion)
- Best loss: -0.0105 (epoch 38)
- L_ent trajectory: 0.685 → 0.635 (small but consistent entropy reduction)
- Time: 19.8 min

---

## Critical Analysis of Results

### ⚠️ WARNING: Evaluation is statistically unreliable

**Root cause: Test set contains only 1 glaucoma case out of 41 total.**

```
Chákṣu test split:
  Total     :  41
  Normal (0): 40  (97.6%)
  Glaucoma(1): 1   (2.4%)   ← ONLY 1 POSITIVE SAMPLE
```

With 1 positive sample:
- AUROC can only take values in {0, 0.025, 0.050, ..., 0.975, 1.000} (40 discrete steps)
- A single rank change flips AUROC by 0.025
- There is no statistical confidence — no p-values are valid
- All Precision/F1/Sens@95Spec metrics are dominated by this single case

### Why this happened (Root Cause)
The test set has only 1 glaucoma because `prepare_data.py` failed to match labels for **Remidio and Forus images** (1200/1345 images). Only **Bosch images** (145 total) were correctly matched. This is a label-parsing bug:

1. The `img_col` detection loop used `img_col = col` on every iteration, so it picked the **last** column with "image" in its name — likely an expert annotation column, not the base filename.
2. Forus image columns produce keys like `"image101.jpg-image101-1.jpg"` which don't match actual filenames `"image101.jpg"`.

Fixed in Run-6.

### Why Oracle < Source (AUROC 0.825 vs 0.975)
With only 1 glaucoma in the test set, AUROC = (rank of that 1 glaucoma among 41 samples). The Oracle's adapted head learned to call 7 normal eyes as glaucoma (Precision=0.125), which pushed the glaucoma ranking down. This is an artefact of overfitting on a 84-image train set with 2 batches/epoch, heavy class imbalance, and evaluation on 1 positive sample. Not meaningful.

### What IS meaningful
- **AIROGS sanity check AUROC = 0.9790** — source training definitely worked ✅
- **Oracle val AUROC = 0.7969** — the held-out val set (from the same distribution) shows the Oracle is genuinely learning, plateau is from tiny dataset not a failure ✅
- **Adaptation L_ent declining** — the adaptation is genuinely reducing uncertainty, not collapsing ✅
- **AIROGS+Adapt AUROC = 1.0** — the adapted model ranks the single glaucoma above all 40 normals, consistent with good adaptation. But unreliable given N=1 positive.

---

## Bug Fixes Applied vs Previous Runs

| Bug | Run where fixed |
|---|---|
| GrayscaleToRGB removing diagnostic colour signal | Run-4 |
| fname.split('-') truncating Remidio filenames → wrong labels | Run-4 |
| L_div computed but never subtracted (L_SFDA = L_ent only) | Run-5 |
| Adapting on test set CSV instead of train unlabeled | Run-5 |
| CLS token batch-level AdaIN instead of patch spatial AdaIN | Run-5 |
| CLS eval mismatch (trained on mean-pool, evaluated on CLS) | Run-5 |
| **img_col picks last "image" column → Remidio/Forus labels lost** | **Run-6** |
| **Forus key format "base.jpg-base-1.jpg" not matching filenames** | **Run-6** |

---

## Run-6 Improvements

In addition to the data-parsing bug fix, Run-6 adds 5 algorithmic innovations:
1. **ProtoStyleBank** — K=256 FIFO style anchors per class (vs ~3 from batch)
2. **Bank warmup** — 3 epochs building bank before gradient updates
3. **L_con** — augmentation consistency loss (KL weak vs strong), ramped over 5 epochs
4. **L_psl** — progressive self-training from epoch 10 (source confidence > 0.90)
5. **Cosine LR + gradient clipping** — max_norm=1.0
