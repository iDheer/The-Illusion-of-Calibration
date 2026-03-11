"""
prepare_data.py - Run-3: Intelligent Data Preparation for Netra-Adapt

Run-3 addition: reports class distribution and imbalance ratio for every
split so we know exactly what WeightedRandomSampler is compensating for.

Handles:
1. AIROGS (Kaggle) - Simple RG/NRG folder structure
2. Chákṣu (Figshare) - Complex nested structure with Train/Test splits
   - 1.0_Original_Fundus_Images/[Bosch|Forus|Remidio]
   - 6.0_Glaucoma_Decision/Glaucoma_decision_comparision/*_majority.csv

The Figshare download creates a nested structure that we handle automatically.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from PIL import Image as _PIL_Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# --- CONFIG ---
BASE_DIR  = "/workspace/data"
AIROGS_DIR = os.path.join(BASE_DIR, "raw_airogs")
CHAKSU_DIR = os.path.join(BASE_DIR, "raw_chaksu")
CSV_OUT_DIR = os.path.join(BASE_DIR, "processed_csvs")

# ── Ground-truth dataset sizes (from the Chákṣu paper, 2022) ──────────────
# These are the AUTHORITATIVE expected values. Any significant deviation
# during data preparation must be flagged as a potential data issue.
# Source: Chákṣu IMAGE database paper (Kumar et al.)
KNOWN_CHAKSU = {
    'total':   1345,   # All images in the full dataset
    'train':   1009,   # Official train split
    'test':    336,    # Official test split
    # Per-device totals (independent of train/test split)
    'Remidio': 1074,   # 2448×3264 px, non-mydriatic Fundus-on-Phone
    'Forus':   126,    # 2048×1536 px, 3Nethra Classic non-mydriatic
    'Bosch':   145,    # 1920×1440 px, handheld fundus camera
}

# Known native resolutions per device (W×H as reported by PIL, i.e. landscape W > H)
KNOWN_RESOLUTIONS = {
    'Remidio': (2448, 3264),   # portrait: W=2448, H=3264 (PIL reports width×height)
    'Forus':   (2048, 1536),
    'Bosch':   (1920, 1440),
}

# ── GROUND-TRUTH split counts (approximate, 3:1 ratio per device) ─────────
# These are approximate. Exact counts depend on how the dataset was split.
# Train ≈ 75%, Test ≈ 25% per device.
KNOWN_CHAKSU_SPLITS = {
    ('Remidio', 'Train'): 810,
    ('Remidio', 'Test'):  264,
    ('Forus',   'Train'): 95,
    ('Forus',   'Test'):  31,
    ('Bosch',   'Train'): 109,
    ('Bosch',   'Test'):  36,
}


# ── Helper utilities ───────────────────────────────────────────────────────

def detect_device(path: str) -> str:
    """Return 'Bosch' | 'Forus' | 'Remidio' | 'Unknown' from an image path."""
    p = path.replace('\\', '/')
    for dev in ('Bosch', 'Forus', 'Remidio'):
        if f'/{dev}/' in p or p.endswith(f'/{dev}'):
            return dev
    # fallback: check all path components case-insensitively
    parts = p.lower().split('/')
    for dev in ('Bosch', 'Forus', 'Remidio'):
        if dev.lower() in parts:
            return dev
    return 'Unknown'


def detect_split(path: str) -> str:
    """Return 'Train' | 'Test' | 'Unknown' from an image path."""
    p = path.replace('\\', '/')
    if '/Train/' in p:
        return 'Train'
    if '/Test/' in p:
        return 'Test'
    return 'Unknown'


def sample_image_dims(image_paths: list, n: int = 3) -> list:
    """
    Open the first n images and return [(filename, W, H), ...] for logging.
    Falls back gracefully if PIL is not available.
    """
    results = []
    for path in image_paths[:n]:
        try:
            if _PIL_AVAILABLE:
                with _PIL_Image.open(path) as im:
                    w, h = im.size
                    results.append((os.path.basename(path), w, h))
            else:
                results.append((os.path.basename(path), '?', '?'))
        except Exception as e:
            results.append((os.path.basename(path), f'ERR:{e}', ''))
    return results


def find_folder(base_path, folder_name, recursive=True):
    """Find a folder anywhere within base_path."""
    if recursive:
        for root, dirs, files in os.walk(base_path):
            if folder_name in dirs:
                return os.path.join(root, folder_name)
    direct = os.path.join(base_path, folder_name)
    if os.path.exists(direct):
        return direct
    return None


def print_class_balance(records, name: str):
    """
    Run-3: Pretty-print class distribution and imbalance ratio.
    This tells us how aggressive the WeightedRandomSampler needs to be.
    """
    labels = [r['label'] for r in records if r.get('label', -1) >= 0]
    if not labels:
        print(f"  [{name}] No labeled records to analyse.")
        return
    n_total   = len(labels)
    n_normal  = labels.count(0)
    n_glaucoma = labels.count(1)
    ratio = n_normal / n_glaucoma if n_glaucoma > 0 else float('inf')
    pct_glaucoma = 100 * n_glaucoma / n_total if n_total > 0 else 0

    print(f"\n  ┌─ Class Balance Report: {name} ─────────────────────")
    print(f"  │  Total     : {n_total:>6}")
    print(f"  │  Normal (0): {n_normal:>6}  ({100*n_normal/n_total:.1f}%)")
    print(f"  │  Glaucoma(1): {n_glaucoma:>5}  ({pct_glaucoma:.1f}%)")
    print(f"  │  Imbalance ratio (Normal:Glaucoma): {ratio:.2f}:1")
    if ratio > 2.0:
        print(f"  │  ⚠ Significant imbalance — WeightedRandomSampler ACTIVE")
    else:
        print(f"  │  ✓ Reasonably balanced")
    print(f"  └────────────────────────────────────────────────────")


# ── AIROGS ─────────────────────────────────────────────────────────────────

def prepare_airogs():
    """
    Process AIROGS dataset with RG/NRG folder structure.

    Creates TWO CSVs:
    - airogs_train.csv: For training the Source model
    - airogs_test.csv:  For sanity-checking Source model on source domain

    Uses stratified 80-20 split to preserve class ratio in both splits.
    """
    print(f"\n--- Processing AIROGS ---")
    records = []

    rg_dir  = find_folder(AIROGS_DIR, "RG")
    nrg_dir = find_folder(AIROGS_DIR, "NRG")

    if rg_dir:
        rg_files = glob.glob(os.path.join(rg_dir, "*.jpg"))
        for f in rg_files:
            records.append({"path": f, "label": 1})
        print(f"  Found {len(rg_files)} RG (glaucoma) images")

    if nrg_dir:
        nrg_files = glob.glob(os.path.join(nrg_dir, "*.jpg"))
        for f in nrg_files:
            records.append({"path": f, "label": 0})
        print(f"  Found {len(nrg_files)} NRG (normal) images")

    if not records:
        print("[ERROR] No AIROGS images found!")
        print(f"  Expected structure: {AIROGS_DIR}/RG/*.jpg and {AIROGS_DIR}/NRG/*.jpg")
        return

    # Run-3: report imbalance before split
    print_class_balance(records, "AIROGS full set")

    df = pd.DataFrame(records)

    # Stratified 80-20 split (preserve class ratio in both splits)
    df_pos = df[df['label'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    df_neg = df[df['label'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)

    split_pos = int(len(df_pos) * 0.8)
    split_neg = int(len(df_neg) * 0.8)

    df_train = pd.concat([df_pos[:split_pos], df_neg[:split_neg]]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test  = pd.concat([df_pos[split_pos:], df_neg[split_neg:]]).sample(frac=1, random_state=42).reset_index(drop=True)

    print_class_balance(df_train.to_dict('records'), "AIROGS train split")
    print_class_balance(df_test.to_dict('records'),  "AIROGS test split")

    train_path = os.path.join(CSV_OUT_DIR, "airogs_train.csv")
    df_train.to_csv(train_path, index=False)
    print(f"\n  ✓ Saved {train_path} ({len(df_train)} images)")

    test_path = os.path.join(CSV_OUT_DIR, "airogs_test.csv")
    df_test.to_csv(test_path, index=False)
    print(f"  ✓ Saved {test_path} ({len(df_test)} images)")


# ── Chákṣu ─────────────────────────────────────────────────────────────────

def parse_chaksu_labels():
    """
    Process Chákṣu dataset with complex nested structure.

    Figshare download structure:
    raw_chaksu/
    ├── Train/
    │   ├── 1.0_Original_Fundus_Images/
    │   │   ├── Bosch/
    │   │   ├── Forus/
    │   │   └── Remidio/
    │   └── 6.0_Glaucoma_Decision/
    │       └── Glaucoma_decision_comparision/
    │           └── *_majority.csv
    └── Test/
        └── (same structure)
    """
    print(f"\n--- Processing Chákṣu ---")

    label_map = {}

    # Step 1: Find all majority decision CSV files
    csv_patterns = [
        os.path.join(CHAKSU_DIR, "**", "6.0_Glaucoma_Decision", "**", "*majority*.csv"),
        os.path.join(CHAKSU_DIR, "**", "Glaucoma_decision*", "*majority*.csv"),
        os.path.join(CHAKSU_DIR, "6.0_Glaucoma_Decision", "**", "*majority*.csv"),
    ]

    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))
    csv_files = list(set(csv_files))
    print(f"  Found {len(csv_files)} label CSV files")

    # Step 2: Parse each CSV to build label_map
    # Run-6: log ALL columns in each CSV + which img_col/dec_col was selected.
    # In Run-5 the wrong img_col was silently picked (last 'image' col instead
    # of first), which caused Remidio + Forus labels to be completely lost.
    # These verbose logs let you catch that failure immediately.
    print(f"  {'CSV file':<55} {'rows':>5}  {'img_col':<35} {'dec_col':<30}")
    print(f"  {'-'*55} {'-'*5}  {'-'*35} {'-'*30}")
    for csv_file in csv_files:
        try:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except Exception:
                df = pd.read_csv(csv_file, encoding='latin-1')

            df.columns = [c.strip() for c in df.columns]
            img_col = None
            dec_col = None
            all_image_cols = []    # track ALL 'image' columns found

            for col in df.columns:
                col_lower = col.lower()
                # Run-6 FIX: use FIRST column with 'image' in name (not last).
                # Remidio/Forus CSVs have multiple image-related columns; the
                # first one is the base filename, later ones are expert annotations.
                if 'image' in col_lower:
                    all_image_cols.append(col)
                    if img_col is None:
                        img_col = col
                if 'majority' in col_lower or 'decision' in col_lower:
                    dec_col = col

            if img_col is None or dec_col is None:
                print(f"  [SKIP] {os.path.basename(csv_file):<53} - img_col={img_col} dec_col={dec_col}")
                print(f"         All columns: {list(df.columns)}")
                continue

            # Show all 'image' columns so we can verify img_col is the base filename
            extra_img_note = ""
            if len(all_image_cols) > 1:
                extra_img_note = f"  ← {len(all_image_cols)} 'image' cols total; others: {all_image_cols[1:]}"

            # Sample the first value of img_col to verify it looks like a filename
            first_val = str(df[img_col].iloc[0]).strip() if len(df) > 0 else ''
            n_glaucoma = 0
            n_normal   = 0
            n_parsed   = 0

            for idx, row in df.iterrows():
                raw_name = str(row[img_col]).strip()
                decision = str(row[dec_col]).upper().strip()

                parts = raw_name.replace('\\', '/').split('/')
                fname = parts[-1]
                # Run-4 FIX: do NOT split on '-'.
                # Run-3 had fname.split('-')[0] here which truncated
                # filenames like "REMIDIO-0001.jpg" to "REMIDIO", causing
                # all Remidio images (76% of Chákṣu) to map to a single key
                # and receive wrong labels. Use the full basename.
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fname += ".jpg"

                fname_lower = fname.lower()

                if "NORMAL" in decision:
                    label = 0
                    n_normal += 1
                elif "GLAUCOMA" in decision or "SUSPECT" in decision:
                    label = 1
                    n_glaucoma += 1
                else:
                    continue
                n_parsed += 1

                label_map[fname_lower] = label
                label_map[fname] = label

                # Run-6 FIX (extended): register base filenames for compound
                # CSV keys like:
                #   Forus:   "1.jpg-1-1.jpg"     → actual file "1.png"  (ext mismatch!)
                #   Bosch:   "Image101.jpg-Image101-1.jpg" → actual "Image101.jpg" ✓
                #   Remidio: "17521.tif-17521-1.tif"       → actual "17521.JPG"
                # Key insight: the CSV separator can be .jpg-, .jpeg-, .png-, or .tif-.
                # The actual file on disk may use a DIFFERENT extension.
                # Solution: register the stem (no extension) with ALL likely extensions.
                for sep_ext in ('.jpg', '.jpeg', '.png', '.tif'):
                    if sep_ext + '-' in fname_lower:
                        base_stem = fname_lower.split(sep_ext + '-')[0]
                        # Register with every likely on-disk extension
                        for reg_ext in ('.jpg', '.jpeg', '.png', '.tif'):
                            label_map[base_stem + reg_ext] = label
                            # also capitalised variant (e.g. .JPG)
                            label_map[base_stem + reg_ext.upper()] = label
                            label_map[(base_stem + reg_ext).capitalize()] = label
                        break

            print(f"  [OK]   {os.path.basename(csv_file):<53} {len(df):>5}  "
                  f"img_col='{img_col}' (sample: '{first_val[:30]}')")
            if extra_img_note:
                print(f"         NOTE: multiple 'image' cols — Run-6 picks FIRST.{extra_img_note}")
            print(f"         Parsed: {n_parsed} rows → Normal={n_normal}, Glaucoma={n_glaucoma}")

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(csv_file)}: {e}")

    print(f"\n  Total unique keys in label_map: {len(label_map)}")
    # Show sample keys grouped by apparent type
    if label_map:
        keys = list(label_map.keys())
        # Filter to unique lowercase keys (no duplicate case variants)
        lower_keys = sorted(set(k for k in keys if k == k.lower()))[:10]
        print(f"  Sample label_map keys (lowercase, first 10):")
        for k in lower_keys:
            print(f"    '{k}'  → label={label_map[k]}")
        # Check for compound keys (Forus .jpg-, Remidio .tif-)
        compound_keys = [k for k in lower_keys
                         if any(ext+'-' in k for ext in ('.jpg','.jpeg','.png','.tif'))]
        if compound_keys:
            print(f"  ⚠ Compound keys present in label_map (normal — base stems also registered):")
            for k in compound_keys[:5]:
                for sep in ('.jpg-', '.jpeg-', '.png-', '.tif-'):
                    if sep in k:
                        stem = k.split(sep)[0]
                        print(f"    '{k}' → stem '{stem}' registered with .jpg/.png/.tif variants")
                        break
        else:
            print(f"  ✓ No compound keys — all filenames look clean")

    # Step 3: Find all images and log per-device per-split counts
    # Run-6: track per-device counts and compare against KNOWN_CHAKSU sizes.
    # In Run-5 all_images=1345 was correct but only 145 got labels — this
    # per-device table reveals such mismatches immediately.
    image_patterns = [
        os.path.join(CHAKSU_DIR, "**", "1.0_Original*", "Bosch", "*"),
        os.path.join(CHAKSU_DIR, "**", "1.0_Original*", "Forus", "*"),
        os.path.join(CHAKSU_DIR, "**", "1.0_Original*", "Remidio", "*"),
        os.path.join(CHAKSU_DIR, "Bosch", "*"),
        os.path.join(CHAKSU_DIR, "Forus", "*"),
        os.path.join(CHAKSU_DIR, "Remidio", "*"),
    ]

    all_images = []
    for pattern in image_patterns:
        found = glob.glob(pattern, recursive=True)
        found = [f for f in found if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(f)]
        all_images.extend(found)
    all_images = list(set(all_images))

    # Per-device per-split counts and dimension samples
    dev_split_imgs = {}   # (device, split) → [paths]
    for img_path in all_images:
        dev   = detect_device(img_path)
        split = detect_split(img_path)
        key   = (dev, split)
        if key not in dev_split_imgs:
            dev_split_imgs[key] = []
        dev_split_imgs[key].append(img_path)

    print(f"\n  ┌─ Image Scan Results vs Expected (from Ch\u00e1k\u1e63u paper) ─────────────────")
    print(f"  │  {'Device':<10} {'Split':<7} {'Found':>6}  {'Expected':>8}  {'Status'}")
    print(f"  │  {'-'*10} {'-'*7} {'-'*6}  {'-'*8}  {'-'*20}")
    total_found = 0
    total_warn  = False
    for dev in ('Remidio', 'Forus', 'Bosch'):
        for split in ('Train', 'Test'):
            key     = (dev, split)
            found_n = len(dev_split_imgs.get(key, []))
            total_found += found_n
            expected_n  = KNOWN_CHAKSU_SPLITS.get(key, '?')
            if isinstance(expected_n, int):
                delta = found_n - expected_n
                if abs(delta) <= 2:
                    status = '\u2713 OK'
                else:
                    status = f'\u26a0 DIFF {delta:+d} ({100*delta/expected_n:.0f}%)'
                    total_warn = True
            else:
                status = '? (unknown)'
            print(f"  │  {dev:<10} {split:<7} {found_n:>6}  {str(expected_n):>8}  {status}")

        # After both splits for this device, print dimension sample
        dev_imgs_all = (dev_split_imgs.get((dev,'Train'), []) +
                        dev_split_imgs.get((dev,'Test'),  []))
        if dev_imgs_all:
            dim_samples = sample_image_dims(dev_imgs_all, n=2)
            for fname_s, w, h in dim_samples:
                expected_wh = KNOWN_RESOLUTIONS.get(dev, ('?', '?'))
                dim_ok = '\u2713' if (w, h) == expected_wh else f'\u26a0 expected {expected_wh[0]}\u00d7{expected_wh[1]}'
                print(f"  │    dim sample: {fname_s}  → {w}\u00d7{h} {dim_ok}")
    print(f"  │")
    print(f"  │  {'TOTAL':<10} {'':>7} {total_found:>6}  {KNOWN_CHAKSU['total']:>8}  "
          f"{'\u2713 OK' if total_found == KNOWN_CHAKSU['total'] else '\u26a0 MISMATCH'}")
    if total_warn:
        print(f"  │  ⚠ Per-device count mismatch detected — check download integrity")
    print(f"  └─────────────────────────────────────────────────────────────────────")

    unknown_imgs = dev_split_imgs.get(('Unknown', 'Unknown'), [])
    if unknown_imgs:
        print(f"  ⚠ {len(unknown_imgs)} images with unknown device/split — check folder structure")
        for p in unknown_imgs[:3]:
            print(f"    {p}")

    # Step 4: Match images with labels — track per-device match rates
    # Run-6: per-device breakdown reveals instantly if one device has 0% match.
    # In Run-5 this would have shown Remidio=0%, Forus=0%, Bosch=100%.
    labeled_records   = []
    unlabeled_records = []
    # per-device per-split match tracking
    match_stats = {}   # (device, split) → {'matched': int, 'unmatched': int, 'glaucoma': int, 'normal': int}

    for img_path in all_images:
        fname      = os.path.basename(img_path)
        fname_lower = fname.lower()
        dev        = detect_device(img_path)
        split      = detect_split(img_path)
        key        = (dev, split)
        if key not in match_stats:
            match_stats[key] = {'matched': 0, 'unmatched': 0, 'glaucoma': 0, 'normal': 0}

        if fname_lower in label_map:
            lbl = label_map[fname_lower]
        elif fname in label_map:
            lbl = label_map[fname]
        else:
            lbl = None
            # Fallback: substring match (slower but catches edge cases)
            for mkey, mlbl in label_map.items():
                if mkey.lower() in fname_lower or fname_lower in mkey.lower():
                    lbl = mlbl
                    break

        if lbl is not None:
            labeled_records.append({"path": img_path, "label": lbl})
            match_stats[key]['matched'] += 1
            if lbl == 1:
                match_stats[key]['glaucoma'] += 1
            else:
                match_stats[key]['normal'] += 1
        else:
            unlabeled_records.append({"path": img_path, "label": -1})
            match_stats[key]['unmatched'] += 1

    # Print per-device per-split match table
    total_imgs    = len(all_images)
    total_labeled = len(labeled_records)
    total_unlabeled = len(unlabeled_records)
    print(f"\n  ┌─ Label Matching Results ──────────────────────────────────────────────")
    print(f"  │  {'Device':<10} {'Split':<7} {'Total':>6}  {'Matched':>7}  {'Rate':>6}  {'Normal':>7}  {'Glaucoma':>8}")
    print(f"  │  {'-'*10} {'-'*7} {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*8}")
    any_zero_match = False
    for dev in ('Remidio', 'Forus', 'Bosch', 'Unknown'):
        for split in ('Train', 'Test', 'Unknown'):
            key  = (dev, split)
            stat = match_stats.get(key)
            if stat is None:
                continue
            tot  = stat['matched'] + stat['unmatched']
            rate = f"{100*stat['matched']/tot:.0f}%" if tot > 0 else 'N/A'
            flag = ''
            if tot > 0 and stat['matched'] == 0:
                flag = '  ← ⚠ 0% MATCH'
                any_zero_match = True
            elif tot > 0 and stat['matched'] < tot * 0.5:
                flag = f"  ← ⚠ <50%"
            print(f"  │  {dev:<10} {split:<7} {tot:>6}  {stat['matched']:>7}  {rate:>6}  "
                  f"{stat['normal']:>7}  {stat['glaucoma']:>8}{flag}")
    total_rate = f"{100*total_labeled/total_imgs:.0f}%" if total_imgs > 0 else 'N/A'
    print(f"  │  {'TOTAL':<10} {'':>7} {total_imgs:>6}  {total_labeled:>7}  {total_rate:>6}")
    if any_zero_match:
        print(f"  │")
        print(f"  │  ⚠ ONE OR MORE DEVICES HAVE 0% LABEL MATCH.")
        print(f"  │    This was the Run-5 bug: Forus+Remidio had 0% match.")
        print(f"  │    Check: (1) img_col is the base filename column in each CSV")
        print(f"  │           (2) Forus compound key fix is working")
    else:
        print(f"  │  ✓ All devices with images have >0% label match")
    print(f"  └─────────────────────────────────────────────────────────────────────")
    print(f"  Total matched: {total_labeled} labeled, {total_unlabeled} unlabeled")

    # Step 5: Separate Train vs Test based on folder structure
    train_labeled = []
    test_labeled  = []
    train_all     = []
    test_all      = []

    for rec in labeled_records:
        if '/Train/' in rec['path'] or '\\Train\\' in rec['path']:
            train_labeled.append(rec)
            train_all.append(rec)
        elif '/Test/' in rec['path'] or '\\Test\\' in rec['path']:
            test_labeled.append(rec)
            test_all.append(rec)
        else:
            train_labeled.append(rec)
            train_all.append(rec)

    for rec in unlabeled_records:
        if '/Train/' in rec['path'] or '\\Train\\' in rec['path']:
            train_all.append(rec)
        elif '/Test/' in rec['path'] or '\\Test\\' in rec['path']:
            test_all.append(rec)
        else:
            train_all.append(rec)

    # Run-3/6: report imbalance for both splits
    print_class_balance(train_labeled, "Chákṣu train split (Oracle training)")
    print_class_balance(test_labeled,  "Chákṣu test split  (Evaluation)")

    # Run-6: Final sanity check comparing test glaucoma count vs expectation
    n_test_pos = sum(1 for r in test_labeled if r.get('label') == 1)
    n_test_tot = len(test_labeled)
    print(f"\n  ┌─ TEST SET SANITY CHECK (critical for valid AUROC) ─────────────────")
    print(f"  │  Test set total:    {n_test_tot:>4}  (expected ≥ {KNOWN_CHAKSU['test']} if all labels matched)")
    print(f"  │  Glaucoma positives:{n_test_pos:>4}")
    if n_test_pos < 5:
        print(f"  │  ⚠ CRITICAL: only {n_test_pos} glaucoma case(s) in test set!")
        print(f"  │    AUROC will be meaningless. This was the Run-5 failure.")
        print(f"  │    Expected ~50-80 glaucoma cases if all labels are matched.")
        print(f"  │    Fix: check img_col + Forus compound key fix output above.")
    elif n_test_pos < 20:
        print(f"  │  ⚠ WARNING: {n_test_pos} glaucoma case(s) is statistically marginal.")
        print(f"  │    AUROC CIs will be wide. Investigate missing labels above.")
    else:
        print(f"  │  ✓ {n_test_pos} glaucoma cases — statistically meaningful evaluation")
    print(f"  └────────────────────────────────────────────────────────────────────")

    # Step 6: Save CSVs
    if train_labeled:
        df_train = pd.DataFrame(train_labeled)
        out_path = os.path.join(CSV_OUT_DIR, "chaksu_train_labeled.csv")
        df_train.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved {out_path} ({len(df_train)} images) — FOR ORACLE TRAINING")

    if test_labeled:
        df_test = pd.DataFrame(test_labeled)
        out_path = os.path.join(CSV_OUT_DIR, "chaksu_test_labeled.csv")
        df_test.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_test)} images) — FOR ALL EVALUATIONS")

    # For Phase C (Adaptation), use ALL training images (labels ignored during SFDA)
    if train_all:
        df_all = pd.DataFrame(train_all)
        df_all['label'] = -1   # Force unlabeled
        out_path = os.path.join(CSV_OUT_DIR, "chaksu_train_unlabeled.csv")
        df_all.to_csv(out_path, index=False)
        print(f"  ✓ Saved {out_path} ({len(df_all)} images) — FOR NETRA-ADAPT")


# ── Validation ─────────────────────────────────────────────────────────────

def validate_data():
    """Validate prepared data and print a comprehensive summary table."""
    print(f"\n--- Validation & Final Summary ---")
    expected_csvs = [
        ("airogs_train.csv",          "Source training",        {'min': 7000, 'max': 8000}),
        ("airogs_test.csv",           "Source sanity check",    {'min': 1700, 'max': 2000}),
        ("chaksu_train_labeled.csv",  "Oracle training",        {'min': 150, 'max': None}),
        ("chaksu_test_labeled.csv",   "Evaluation (CRITICAL)",  {'min': 150, 'max': None}),
        ("chaksu_train_unlabeled.csv","Netra-Adapt (SFDA)",     {'min': 900, 'max': 1050}),
    ]
    any_fail = False
    print(f"  {'CSV':<35} {'Rows':>6}  {'Valid':>5}  {'Glaucoma':>8}  {'Purpose'}")
    print(f"  {'-'*35} {'-'*6}  {'-'*5}  {'-'*8}  {'-'*25}")
    for csv_name, purpose, size_check in expected_csvs:
        csv_path = os.path.join(CSV_OUT_DIR, csv_name)
        if os.path.exists(csv_path):
            df    = pd.read_csv(csv_path)
            valid = sum(1 for p in df['path'] if os.path.exists(p))
            n_glaucoma = int((df['label'] == 1).sum()) if 'label' in df.columns else -1
            n_rows = len(df)
            # Size sanity check
            lo = size_check.get('min')
            hi = size_check.get('max')
            if lo and n_rows < lo:
                status = f'⚠ TOO FEW (expected ≥{lo})'
                any_fail = True
            elif hi and n_rows > hi:
                status = f'⚠ TOO MANY (expected ≤{hi})'
            else:
                status = '✓'
            gl_str = str(n_glaucoma) if n_glaucoma >= 0 else 'N/A'
            print(f"  {csv_name:<35} {n_rows:>6}  {valid:>5}  {gl_str:>8}  {purpose}  {status}")
        else:
            print(f"  {csv_name:<35} {'NOT FOUND':>6}  —  {'—':>8}  {purpose}  ⚠ MISSING")
            any_fail = True

    print(f"")
    if any_fail:
        print(f"  ⚠ VALIDATION ISSUES DETECTED — review warnings above before training")
    else:
        print(f"  ✓ All outputs look healthy. Ready to proceed to training.")

    # Quick re-check of the critical test set glaucoma count
    test_csv = os.path.join(CSV_OUT_DIR, "chaksu_test_labeled.csv")
    if os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv)
        n_pos  = int((df_test['label'] == 1).sum())
        n_tot  = len(df_test)
        pct    = 100 * n_pos / n_tot if n_tot > 0 else 0
        print(f"")
        print(f"  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  FINAL CHECK: Chákṣu test set = {n_tot:>3} images, {n_pos:>2} glaucoma ({pct:.1f}%)")
        if n_pos < 5:
            print(f"  ║  ⚠ CRITICAL — evaluation will be meaningless (run-5 had 1)  ║")
        elif n_pos < 20:
            print(f"  ║  ⚠ marginal — results unreliable (CI too wide)               ║")
        else:
            print(f"  ║  ✓ statistically valid for AUROC evaluation                  ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    prepare_airogs()
    parse_chaksu_labels()
    validate_data()
