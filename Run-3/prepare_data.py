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

# --- CONFIG ---
BASE_DIR  = "/workspace/data"
AIROGS_DIR = os.path.join(BASE_DIR, "raw_airogs")
CHAKSU_DIR = os.path.join(BASE_DIR, "raw_chaksu")
CSV_OUT_DIR = os.path.join(BASE_DIR, "processed_csvs")


# ── Helper utilities ───────────────────────────────────────────────────────

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
    for csv_file in csv_files:
        try:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except Exception:
                df = pd.read_csv(csv_file, encoding='latin-1')

            df.columns = [c.strip() for c in df.columns]
            img_col = None
            dec_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'image' in col_lower:
                    img_col = col
                if 'majority' in col_lower or 'decision' in col_lower:
                    dec_col = col

            if img_col is None or dec_col is None:
                print(f"    [SKIP] {os.path.basename(csv_file)} - columns not found")
                continue

            for idx, row in df.iterrows():
                raw_name = str(row[img_col]).strip()
                decision = str(row[dec_col]).upper().strip()

                parts = raw_name.replace('\\', '/').split('/')
                fname = parts[-1]
                if '-' in fname and fname.count('-') > 0:
                    fname = fname.split('-')[0]
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fname += ".jpg"

                fname_lower = fname.lower()

                if "NORMAL" in decision:
                    label = 0
                elif "GLAUCOMA" in decision or "SUSPECT" in decision:
                    label = 1
                else:
                    continue

                label_map[fname_lower] = label
                label_map[fname] = label

            print(f"    [OK] {os.path.basename(csv_file)} - {len(df)} entries")

        except Exception as e:
            print(f"    [ERROR] {os.path.basename(csv_file)}: {e}")

    print(f"  Total labels parsed: {len(label_map)}")

    # Step 3: Find all images
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
    print(f"  Found {len(all_images)} image files")

    # Step 4: Match images with labels
    labeled_records   = []
    unlabeled_records = []

    for img_path in all_images:
        fname = os.path.basename(img_path)
        fname_lower = fname.lower()

        if fname_lower in label_map:
            labeled_records.append({"path": img_path, "label": label_map[fname_lower]})
        elif fname in label_map:
            labeled_records.append({"path": img_path, "label": label_map[fname]})
        else:
            matched = False
            for key, lbl in label_map.items():
                if key.lower() in fname_lower or fname_lower in key.lower():
                    labeled_records.append({"path": img_path, "label": lbl})
                    matched = True
                    break
            if not matched:
                unlabeled_records.append({"path": img_path, "label": -1})

    print(f"  Matched: {len(labeled_records)} labeled, {len(unlabeled_records)} unlabeled")

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

    # Run-3: report imbalance for both splits
    print_class_balance(train_labeled, "Chákṣu train split (Oracle training)")
    print_class_balance(test_labeled,  "Chákṣu test split  (Evaluation)")

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
    """Quick validation of prepared data."""
    print(f"\n--- Validation ---")
    expected_csvs = [
        "airogs_train.csv",
        "airogs_test.csv",
        "chaksu_train_labeled.csv",
        "chaksu_test_labeled.csv",
        "chaksu_train_unlabeled.csv",
    ]
    for csv_name in expected_csvs:
        csv_path = os.path.join(CSV_OUT_DIR, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            valid = sum(1 for p in df['path'] if os.path.exists(p))
            print(f"  {csv_name}: {len(df)} entries, {valid} valid paths")
        else:
            print(f"  {csv_name}: NOT FOUND")


if __name__ == "__main__":
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    prepare_airogs()
    parse_chaksu_labels()
    validate_data()
