"""
dataset_loader.py - Run-4: Class Weight Utilities (colour input restored)

Run-4 Changes vs Run-3:
1. REMOVED GrayscaleToRGB — Run-3 showed this destroys diagnostic signal on
   the small Chákṣu dataset (907 images).  Oracle AUROC was 0.497 (worse than
   random) because cup pallor, RNFL defects and peripapillary atrophy are only
   distinguishable via colour on small-N training sets.  Colour is restored.
2. Full colour jitter restored (brightness, contrast, saturation, hue) with
   slightly wider saturation/hue range to simulate cross-device colour variance.
3. Normalisation reverted to standard RGB ImageNet stats (mean/std per channel).
4. compute_class_weights() and compute_sample_weights() unchanged — still used
   for WeightedRandomSampler + weighted CrossEntropyLoss (Run-3 fix kept).
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# DINOv3 ViT-L/16 requires input size divisible by 16 (patch size)
# Using 512x512 (512 = 16 * 32 patches)
DINOV3_INPUT_SIZE = 512

# ── Standard RGB ImageNet statistics ─────────────────────────────────────
# Run-4: restored from standard per-channel ImageNet values.
# GrayscaleToRGB + GRAY_MEAN/GRAY_STD removed — colour inputs required.
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]


# ── Preprocessing ──────────────────────────────────────────────────────────

def robust_circle_crop(image_path, target_size=DINOV3_INPUT_SIZE):
    """
    Handles resolution variance (2448×3264 vs 1920×1440).
    Removes 'OD' text overlay via contour filtering.
    Ensures proper size for DINOv3 ViT-L/16 (512×512).
    Returns a PIL RGB image (grayscale conversion happens in transforms).
    """
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found: {image_path}")
        return Image.new('RGB', (target_size, target_size))

    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Failed to read: {image_path}")
        return Image.new('RGB', (target_size, target_size))

    h_orig, w_orig = img.shape[:2]

    # Step 1: For high-res images, centre-crop to square first
    if h_orig > 1500 or w_orig > 1500:
        size = min(h_orig, w_orig)
        y_start = (h_orig - size) // 2
        x_start = (w_orig - size) // 2
        img = img[y_start:y_start+size, x_start:x_start+size]

    # Step 2: Convert to gray & threshold for fundus detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Step 3: Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter: keep only large contours — removes small 'OD' text in Remidio images
        valid_contours = [c for c in contours if cv2.contourArea(c) > 5000]
        if valid_contours:
            c = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            pad = int(0.02 * min(w, h))
            y1 = max(0, y - pad)
            x1 = max(0, x - pad)
            y2 = min(img.shape[0], y + h + pad)
            x2 = min(img.shape[1], x + w + pad)
            img = img[y1:y2, x1:x2]

            # Make square by centre-cropping
            h_new, w_new = img.shape[:2]
            if h_new != w_new:
                size = min(h_new, w_new)
                y_start = (h_new - size) // 2
                x_start = (w_new - size) // 2
                img = img[y_start:y_start+size, x_start:x_start+size]

    # Step 4: Resize to DINOv3 input size (512×512) and return as RGB PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    return pil_img.resize((target_size, target_size), Image.Resampling.BICUBIC)


# ── Dataset ────────────────────────────────────────────────────────────────

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = robust_circle_crop(row['path'])
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ── Class-balance utilities ────────────────────────────────────────────────

def compute_class_weights(csv_path: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Given a CSV with a 'label' column (values 0 and 1), returns a 2-element
    tensor [w0, w1] where w_c = N / (num_classes * count_c).

    Example
    -------
    Normal: 800 samples  →  w0 = 1000 / (2 * 800) = 0.625
    Glaucoma: 200 samples → w1 = 1000 / (2 * 200) = 2.500
    → The loss contribution of each glaucoma sample is 4× that of a normal one.
    """
    df = pd.read_csv(csv_path)
    # Filter to labeled rows only (label != -1)
    df = df[df['label'] >= 0]
    counts = df['label'].value_counts().sort_index()   # index 0, 1
    n_total = len(df)
    n_classes = len(counts)
    weights = torch.tensor(
        [n_total / (n_classes * counts[c]) for c in range(n_classes)],
        dtype=torch.float32
    )
    return weights


def compute_sample_weights(csv_path: str) -> list:
    """
    Compute per-sample weights for WeightedRandomSampler.

    Each sample is assigned the inverse-frequency weight of its class,
    so that when the sampler draws len(dataset) samples, each class is
    seen approximately equally often.
    """
    df = pd.read_csv(csv_path)
    df = df[df['label'] >= 0].reset_index(drop=True)
    counts = df['label'].value_counts().to_dict()
    n_total = len(df)
    n_classes = len(counts)

    sample_weights = [
        n_total / (n_classes * counts[int(row['label'])])
        for _, row in df.iterrows()
    ]
    return sample_weights


# ── Transforms ────────────────────────────────────────────────────────────

def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Run-4 transforms — colour input restored.

    Key differences from Run-3
    --------------------------
    • GrayscaleToRGB REMOVED.  Run-3 showed that removing colour destroyed
      diagnostic signal on the 907-image Oracle dataset (AUROC = 0.497).
      Cup pallor, RNFL defects and peripapillary atrophy require colour.
    • Full colour jitter restored: brightness, contrast, saturation AND hue.
      Slight saturation/hue jitter simulates cross-device colour variance
      between AIROGS (Topcon/Canon, Western clinics) and Chákṣu (Remidio/Forus/
      Bosch, Indian clinics) without discarding the diagnostic colour signal.
    • Normalisation uses standard per-channel RGB ImageNet stats.
    """
    if is_training:
        return transforms.Compose([
            # ── Geometric augmentations (anatomically valid) ───────────────
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),
                shear=5
            ),

            # ── Colour augmentation — simulates cross-device variance ──────
            # Saturation/hue jitter is intentionally mild: enough to teach
            # device-invariance, not so strong as to destroy clinical signal.
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.15,
                hue=0.05
            ),

            # Quality-variance simulation (glare, blur from handheld devices)
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3
            ),

            transforms.ToTensor(),
            transforms.Normalize(RGB_MEAN, RGB_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(RGB_MEAN, RGB_STD),
        ])
