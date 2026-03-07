"""
dataset_loader.py - Run-3: Grayscale + Class Weight Utilities

Run-3 Changes vs Run-1/2:
1. GrayscaleToRGB transform: converts fundus images to luminance grayscale,
   then replicates to 3 channels so DINOv3 (RGB model) still works unchanged.
   Mean/std updated to grayscale ImageNet statistics.
   --- WHY: glaucoma is a GEOMETRY task (cup-to-disc ratio), not a colour task.
       Indian retinas are darker (higher melanin) which causes colour bias.
       Removing colour forces the model to learn shape/structure features,
       and collapses the main source of cross-ethnic domain shift.
2. compute_class_weights(): computes per-class inverse-frequency weights
   from a CSV file, used by train_source.py / train_oracle.py for
   WeightedRandomSampler and weighted CrossEntropyLoss.
   --- WHY: AIROGS and Chákṣu both have significant class imbalance.
       The source model biased toward "Normal" → bad entropy split in MixEnt.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

# DINOv3 ViT-L/16 requires input size divisible by 16 (patch size)
# Using 512x512 (512 = 16 * 32 patches)
DINOV3_INPUT_SIZE = 512

# ── Grayscale ImageNet statistics ──────────────────────────────────────────
# Standard ImageNet per-channel mean/std: [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
# Luminance-weighted grayscale equivalent (Y = 0.299R + 0.587G + 0.114B):
#   mean = 0.299*0.485 + 0.587*0.456 + 0.114*0.406 ≈ 0.449
#   std  = sqrt(0.299²*0.229² + 0.587²*0.224² + 0.114²*0.225²) ≈ 0.226
# We replicate to each channel since all 3 channels are identical after conversion.
GRAY_MEAN = [0.449, 0.449, 0.449]
GRAY_STD  = [0.226, 0.226, 0.226]


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


class GrayscaleToRGB:
    """
    Run-3 core transform: RGB fundus → luminance grayscale → 3-channel grayscale.

    Steps
    -----
    1. Convert PIL RGB image to luminance grayscale (single channel).
    2. Replicate the single channel to produce a (3, H, W) tensor where all
       three channels are identical.

    This keeps the tensor shape compatible with DINOv3's RGB input while
    stripping the colour information that causes cross-ethnic domain shift.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to grayscale (L mode = single channel)
        gray = ImageOps.grayscale(img)          # PIL 'L' image
        # Convert back to RGB so subsequent transforms (ToTensor etc.) work
        # PIL 'L' → 'RGB' replicates the channel across R, G, B
        return gray.convert('RGB')

    def __repr__(self):
        return "GrayscaleToRGB()"


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
    Run-3 transforms.

    Key differences from Run-1/2
    ----------------------------
    • Grayscale conversion (GrayscaleToRGB) is the FIRST transform so that
      all subsequent steps operate on a colour-agnostic image.
    • Colour jitter (brightness, contrast, saturation, hue) is REMOVED — it
      provides no useful augmentation on a grayscale image, and saturation/hue
      are literally no-ops.  We compensate by reinforcing geometric augmentation.
    • GaussianBlur probability raised to 0.4 to simulate quality variance
      that is now the primary remaining domain-shift signal.
    • Normalisation uses grayscale ImageNet stats (mean=0.449, std=0.226).
    """
    if is_training:
        return transforms.Compose([
            # ── Run-3: Strip colour FIRST ──────────────────────────────────
            GrayscaleToRGB(),

            # ── Geometric augmentations (anatomically valid) ───────────────
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),          # Slightly wider scale range
                shear=5                       # Slight shear for perspective variety
            ),

            # ── Appearance augmentations (grayscale-safe) ─────────────────
            # Brightness / contrast only — no saturation/hue on grayscale
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.0,              # No-op on grayscale but harmless
                hue=0.0                      # No-op on grayscale but harmless
            ),

            # Quality-variance simulation (glare, blur from handheld devices)
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.4
            ),

            transforms.ToTensor(),
            transforms.Normalize(GRAY_MEAN, GRAY_STD),
        ])
    else:
        return transforms.Compose([
            # ── Run-3: Strip colour for evaluation too ─────────────────────
            GrayscaleToRGB(),

            transforms.ToTensor(),
            transforms.Normalize(GRAY_MEAN, GRAY_STD),
        ])
