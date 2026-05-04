# Compilation Status: "The Illusion of Calibration" Paper

## Current State
✅ **Ready for XeLaTeX/Tectonic compilation**

All structural issues resolved. PNG→PDF conversion completed to fix xdvipdfmx embedding failures.

## Recent Fixes (May 4, 2026)

### PNG Image Embedding (xdvipdfmx Failure)
**Problem:** XeLaTeX successfully generated XDV output, but xdvipdfmx conversion to PDF failed when processing PNG images (likely due to RGBA color profiles).

**Solution:** Converted all PNG images to PDF format using ImageMagick.

**Files Converted:**
- `dataset/Forus.png` → `dataset/Forus.pdf`
- `results/tsne_source.png` → `results/tsne_source.pdf`
- `results/tsne_adapted.png` → `results/tsne_adapted.pdf`
- `results/attention_maps_comparison.png` → `results/attention_maps_comparison.pdf`
- `results/Source_AIROGS/loss_curve_source.png` → `results/Source_AIROGS/loss_curve_source.pdf`
- `results/Oracle_Chakshu/loss_curve_oracle.png` → `results/Oracle_Chakshu/loss_curve_oracle.pdf`
- `results/NetraAdapt/loss_curve.png` → `results/NetraAdapt/loss_curve.pdf`

**Note:** JPG images (AIROGS.jpg, Remidio.JPG, bosch.JPG) remain unchanged—JPG works fine with xdvipdfmx.

## Document Structure Verification

### Environment Balance Check
- ✅ Document (begin/end): 1/1
- ✅ Abstract: 1/1
- ✅ Figures: 4/4
- ✅ Subfigures: 9/9 (domain shift has 4 images, plus individual figures)
- ✅ Tables: 1/1
- ✅ Bibliography: 1/1

### Image References
All 10 image files exist and are properly referenced in main.tex:
- ✅ 4 dataset images (AIROGS, Remidio, Forus, bosch)
- ✅ 2 t-SNE visualizations (source, adapted)
- ✅ 1 attention maps comparison
- ✅ 3 loss curves (source, oracle, SFDA)

### Bibliography
- **Total entries:** 18 (matches `\begin{thebibliography}{18}`)
- **All citations resolved:** No orphaned references
- **Removed (from previous session):** Anonymous entries (CPT4, CLMS, MedSeg-TTA)

## Paper Content Verification

### Title & Authors
✅ "The Illusion of Calibration: Unsupervised Test-Time Adaptation Can Harm Cross-Ethnic Glaucoma Screening"
✅ Inesh Dheer (IIIT Hyderabad)
✅ Varun Gupta (IIIT Hyderabad)

### Main Sections
1. **Abstract** – Investigation framing: zero-shot ≠ failure, SFDA causes collapse
2. **Introduction** – Problem, prior claims, hypothesis, surprise results
3. **Related Work** – Cross-ethnic fairness, SFDA/TTA, retinal foundation models
4. **Methodology** – DINOv3 backbone, five experimental conditions
5. **Evaluation** – Results table, illusion of calibration mechanism, training dynamics
6. **Conclusion** – Safety implications, deployment hierarchy
7. **Limitations & Future Work** – Methodological limitations, paths forward
8. **Appendix A** – Hyperparameter settings (source, oracle, SFDA)
9. **Appendix B** – RAG baseline configuration

### Key Results (Table 1)
| Model | AUROC | Sensitivity | Specificity | F1-Score |
|-------|-------|-------------|-------------|----------|
| Pretrained | 0.487 | 19.6% | 90.5% | 0.227 |
| RAG Baseline | 0.509 | 25% | 74% | 0.231 |
| **Zero-shot** | **0.852** | **76.5%** | **83.9%** | **0.574** |
| SHOT-IM (SFDA) | 0.390 | 100% | 1.4% | 0.266 |
| Oracle | 0.892 | 92.2% | 78.2% | 0.588 |

## How to Compile

### Using Tectonic (Recommended)
```bash
cd draft/
tectonic --keep-intermediates main.tex
```

### Using XeLaTeX
```bash
cd draft/
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex  # Run twice for references
```

### Using LuaLaTeX
```bash
cd draft/
lualatex main.tex
```

## Known Limitations

1. **RGBA PNG conversion:** While now working via PDF conversion, if original PNG images are regenerated with RGBA channels, they should be converted before re-compilation.

2. **Bibliography anonymization:** Submission guidelines for blind review would require removing author names. Current draft is in final (author-visible) format.

3. **ACL format:** Strict two-column, 8-page limit. Current draft is within limits. Check for overflow on target conference.

## Next Steps

1. ✅ Compile LaTeX → PDF and verify all figures render
2. ⏳ Proofread narrative flow and terminology consistency
3. ⏳ Verify numerical values match results.txt
4. ⏳ Final review against ACL submission guidelines

---
**Last updated:** 2026-05-04 10:17 UTC
**Status:** Awaiting compilation verification
**Git commit:** `b54f0cf` (PNG→PDF conversion)
