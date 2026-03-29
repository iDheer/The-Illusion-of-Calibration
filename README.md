# Netra-Adapt: The Illusion of Calibration in Cross-Ethnic Glaucoma Screening

**An Empirical Investigation into the Vulnerabilities of Unsupervised Test-Time Adaptation**

Netra-Adapt initially set out to adapt foundation vision models trained on Western fundus images (AIROGS) to Indian eyes (Chákṣu) **without any labeled target data**. However, our rigorous experiments uncovered a profound finding: Foundation models exhibit exceptional *zero-shot* cross-ethnic generalization, while applying state-of-the-art unsupervised Test-Time Adaptation (SFDA) causes catastrophic model collapse by conflating demographic statistics with clinical decision boundaries. 

This repository houses the full pipeline (training, adaptation attempts, and oracle baselines) that verifies these claims.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 📖 Overview

Glaucoma remains the leading cause of irreversible blindness worldwide. While deep learning has achieved expert-level performance on color fundus photography, deploying models across international populations presents massive **phenotypic and acquisition shifts** (e.g., melanin pigmentation in Indian retinography vs Western data). 

We sought to mitigate this using **Source-Free Domain Adaptation (SFDA)**, requiring no labels. 

### Key Findings from Run-7 Evaluation

1. **High Zero-Shot Generalization**: A DINOv3 foundation model trained strictly on European data achieved a staggering **0.847 AUROC** on rural Indian hand-held camera data (outperforming prior literature assumptions).
2. **Oracle Fallacy**: Training directly on the localized Indian data did not mathematically beat the pure European generalist framework.
3. **The TTA Collapse**: Applying unsupervised Test-Time Adaptation—even with strict class-prior matching—utterly fails (plummeting to **0.371 AUROC**). The model exploits the structural differences in demographics to satisfy statistical priors rather than learning clinical disease properties.

---

## 🎯 Key Features

- ✅ **Robust Zero-Shot Evaluation**: AIROGS $\to$ Chákṣu cross-domain benchmark.
- ✅ **Negative Result Verification**: See exactly how entropy-based IM leads to mode-collapse in healthcare contexts.
- ✅ **Foundation Model**: DINOv3 ViT-L/16 evaluations
- ✅ **Cross-Ethnic Setting**: Western (AIROGS) $\to$ Indian (Chákṣu) fundus images

---
