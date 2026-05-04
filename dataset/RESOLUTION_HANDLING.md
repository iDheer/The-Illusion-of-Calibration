# Resolution Handling Strategy for AIROGS and Chákṣu Datasets

## Overview

This document explains how Netra-Adapt handles the significant resolution differences between the source (AIROGS) and target (Chákṣu) datasets to ensure proper model training and adaptation.

---

## Dataset Specifications

### AIROGS (Source Dataset)
- **Source:** EyePACS AIROGS Light v2
- **Resolution:** **512 × 512 pixels** (already preprocessed)
- **Device:** Desktop fundus cameras (tabletop)
- **Number of Images:** 4,000 training images
- **Preprocessing:** Already applied CROP methodology (black background removed)
- **Aspect Ratio:** Perfect square (1:1)
- **Quality:** Professional, well-lit, centered fundus

### Chákṣu (Target Dataset)
- **Source:** Indian ethnicity fundus images
- **Resolutions:** **Mixed** (varies by device)
  - **Remidio FoP:** 2448 × 3264 pixels (1,074 images) - 4:3 aspect ratio
  - **Forus 3Nethra:** 2048 × 1536 pixels (126 images) - 4:3 aspect ratio
  - **Bosch Handheld:** 1920 × 1440 pixels (145 images) - 4:3 aspect ratio
- **Total Images:** 1,345
- **Preprocessing:** **Raw images** with black borders and varied aspect ratios
- **Acquisition Challenges:** Handheld devices introduce glare, motion blur, and centering issues

---

## The Resolution Challenge

### Problem Statement

1. **Size Mismatch:** AIROGS is 512×512, while Chákṣu ranges from 1920×1440 to 2448×3264
2. **Aspect Ratio Difference:** AIROGS is square (1:1), Chákṣu is rectangular (4:3)
3. **Black Borders:** Chákṣu images contain large black regions around the circular fundus
4. **Quality Variance:** Handheld devices produce images with artifacts absent in AIROGS

**If not handled properly:**
- Model trained on 512×512 will receive improperly scaled 2448×3264 images
- Black borders waste computational resources and confuse the model
- Aspect ratio distortion can deform fundus features (optic disc/cup)

---

## Our Solution: Intelligent Preprocessing Pipeline

### Stage 1: Dataset-Aware Loading

```python
dataset_type='airogs'  # Minimal processing (already clean)
dataset_type='chakshu'  # Aggressive preprocessing (raw images)
```

### Stage 2: AIROGS Processing (Minimal)

Since AIROGS images are already:
- Cropped to remove black backgrounds
- Resized to 512×512
- Centered and normalized

**Action:** Direct loading → Resize (if needed) → Normalize

```python
if self.dataset_type == 'airogs':
    # Already preprocessed - minimal processing
    return Image.fromarray(img_cv)
```

### Stage 3: Chákṣu Processing (Aggressive)

#### 3.1 Center-Circle-Crop Heuristic

For high-resolution images (>1500px), extract the center square:

```python
if h_orig > 1500 or w_orig > 1500:
    size = min(h_orig, w_orig)  # Use the smaller dimension
    y_start = (h_orig - size) // 2
    x_start = (w_orig - size) // 2
    img_cv = img_cv[y_start:y_start+size, x_start:x_start+size]
```

**Example:** 
- Input: 2448×3264 (Remidio)
- Output: 2448×2448 (center square containing full fundus)

#### 3.2 Intensity-Based Circle Detection

Remove remaining black borders using computer vision:

1. **Convert to grayscale** for mask detection
2. **Threshold at 10** (fundus is brighter than black borders)
3. **Find contours** to locate the fundus circle
4. **Extract bounding box** of the largest contour (the fundus)
5. **Add 2% padding** to avoid cutting fundus edges
6. **Crop to square** for consistent aspect ratio

```python
# Detect fundus region
gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
_, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract fundus bounding box
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Add padding and crop
padding = int(0.02 * min(w, h))
img_cv = img_cv[y-padding:y+h+padding, x-padding:x+w+padding]
```

#### 3.3 Final Resize to 512×512

After cropping, both AIROGS and Chákṣu images are square (1:1 aspect ratio).

PyTorch transforms then resize to the target 512×512:

```python
transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE))  # cfg.IMG_SIZE = 512
```

---

## Visual Pipeline

```
AIROGS Pipeline:
─────────────────
[512×512 Clean Image] → [Minimal Processing] → [512×512 Normalized]
                                                        ↓
                                                   DINOv3 Model

Chákṣu Pipeline:
────────────────
[2448×3264 Raw] → [Center Crop → 2448×2448] → [Circle Detection → 2200×2200]
                                                        ↓
                                          [Resize → 512×512] → [Normalize]
                                                        ↓
                                                   DINOv3 Model
```

---

## Benefits of This Approach

1. **Preserves Fundus Geometry:** Square crops maintain circular fundus shape
2. **Removes Noise:** Black borders don't waste model capacity
3. **Unified Resolution:** Both datasets feed 512×512 images to the model
4. **Efficient:** No unnecessary upscaling of AIROGS images
5. **Robust:** Handles varied Chákṣu device resolutions automatically

---

## Configuration

All resolution settings are centralized in [config.py](config.py):

```python
IMG_SIZE = 512  # Must be divisible by 16 (ViT patch size)
```

To experiment with different resolutions (e.g., 224, 384, 768):

```python
IMG_SIZE = 384  # Update this value
```

**Note:** Higher resolutions improve detail but increase VRAM usage exponentially.

---

## Validation

To verify preprocessing is working correctly:

1. **Visual Inspection:**
   ```python
   python -c "from dataset_loader import NetraDataset; import matplotlib.pyplot as plt; ..."
   ```

2. **Check Shapes:**
   ```python
   for img, label in dataloader:
       print(f"Batch shape: {img.shape}")  # Should be [B, 3, 512, 512]
   ```

3. **Compare Distributions:**
   - AIROGS and Chákṣu images should have similar mean/std after normalization
   - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## Troubleshooting

### Issue: Chákṣu images look distorted

**Solution:** Verify the circle detection is finding the fundus correctly. Add debug visualization:

```python
cv2.imwrite('debug_mask.png', binary_mask)
cv2.drawContours(img_cv, [largest_contour], -1, (0,255,0), 3)
cv2.imwrite('debug_contour.png', img_cv)
```

### Issue: Black borders still visible

**Solution:** Lower the threshold value in intensity detection:

```python
_, binary_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)  # Try 5 instead of 10
```

### Issue: Fundus cut off at edges

**Solution:** Increase padding percentage:

```python
padding = int(0.05 * min(w, h))  # Increase from 2% to 5%
```

---

## References

- [AIROGS Dataset Documentation](AIROGS.txt)
- [Chákṣu Dataset Documentation](CHAKSHU.txt)
- [Paper Methodology](paper_draft.txt) - Section 4.1
