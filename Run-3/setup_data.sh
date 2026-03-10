#!/bin/bash
# =============================================================================
# setup_data.sh — Organise raw downloads into the structure run_everything.sh
#                  expects.  Run this ONCE before run_everything.sh.
#
# Handles whatever you downloaded:
#
#   AIROGS (Kaggle):
#     • The .zip still at /workspace/glaucoma-dataset-eyepacs-airogs-light-v2.zip
#     • Already unzipped as  /workspace/glaucoma_dataset/   (Kaggle default name)
#     • Already unzipped as  /workspace/data/glaucoma_dataset/
#     • Any folder that contains RG/ and NRG/ sub-folders
#
#   Chákṣu (Figshare):
#     • Train.zip / Test.zip still at /workspace/data/
#     • Already unzipped — Train/ and Test/ folders under /workspace/data/
#     • Nested inside another directory after unzip
# =============================================================================

set -euo pipefail

BASE_DIR="/workspace"
DATA_DIR="$BASE_DIR/data"
RAW_AIROGS="$DATA_DIR/raw_airogs"
RAW_CHAKSU="$DATA_DIR/raw_chaksu"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_ok()    { echo -e "${GREEN}  ✓ $1${NC}"; }
print_warn()  { echo -e "${YELLOW}  ⚠ $1${NC}"; }
print_error() { echo -e "${RED}  ✗ $1${NC}"; }
print_step()  { echo -e "${BLUE}[*] $1${NC}"; }

echo "========================================================"
echo "   RUN-3 DATA SETUP"
echo "========================================================"

mkdir -p "$DATA_DIR" "$RAW_AIROGS/RG" "$RAW_AIROGS/NRG" "$RAW_CHAKSU" \
         "$DATA_DIR/processed_csvs"

# ===========================================================================
# AIROGS
# ===========================================================================
print_step "Setting up AIROGS..."

# Already done?
EXISTING_RG=$(ls "$RAW_AIROGS/RG" 2>/dev/null | wc -l)
EXISTING_NRG=$(ls "$RAW_AIROGS/NRG" 2>/dev/null | wc -l)
if [ "$EXISTING_RG" -gt 100 ] && [ "$EXISTING_NRG" -gt 100 ]; then
    print_ok "raw_airogs already populated (${EXISTING_RG} RG, ${EXISTING_NRG} NRG) — skipping"
else
    # Try to unzip the Kaggle download if it exists and hasn't been extracted yet
    KAGGLE_ZIP="$BASE_DIR/glaucoma-dataset-eyepacs-airogs-light-v2.zip"
    if [ -f "$KAGGLE_ZIP" ] && [ ! -d "$BASE_DIR/glaucoma_dataset" ]; then
        print_step "Unzipping Kaggle AIROGS zip..."
        unzip -q "$KAGGLE_ZIP" -d "$BASE_DIR"
        print_ok "Extracted to $BASE_DIR/glaucoma_dataset/"
    fi

    # Search for the folder that holds RG/ and NRG/ sub-folders
    # (works whether it's glaucoma_dataset/eyepac-light-v2-512-jpg/ or any other name)
    AIROGS_SOURCE=""
    for CANDIDATE in \
        "$BASE_DIR/glaucoma_dataset/eyepac-light-v2-512-jpg" \
        "$BASE_DIR/glaucoma_dataset" \
        "$DATA_DIR/glaucoma_dataset/eyepac-light-v2-512-jpg" \
        "$DATA_DIR/glaucoma_dataset" \
        "$DATA_DIR/airogs" \
        "$BASE_DIR/airogs"
    do
        if [ -d "$CANDIDATE/RG" ] || [ -d "$CANDIDATE/train/RG" ]; then
            AIROGS_SOURCE="$CANDIDATE"
            break
        fi
    done

    # Fallback: walk the whole workspace looking for a folder named RG
    if [ -z "$AIROGS_SOURCE" ]; then
        FOUND_RG=$(find "$BASE_DIR" -maxdepth 6 -type d -name "RG" 2>/dev/null | head -1)
        if [ -n "$FOUND_RG" ]; then
            AIROGS_SOURCE=$(dirname "$FOUND_RG")
        fi
    fi

    if [ -n "$AIROGS_SOURCE" ]; then
        print_step "Found AIROGS source: $AIROGS_SOURCE — copying RG/NRG images..."
        find "$AIROGS_SOURCE" -path "*/RG/*.jpg" -exec cp {} "$RAW_AIROGS/RG/" \; 2>/dev/null || true
        find "$AIROGS_SOURCE" -path "*/RG/*.jpeg" -exec cp {} "$RAW_AIROGS/RG/" \; 2>/dev/null || true
        find "$AIROGS_SOURCE" -path "*/NRG/*.jpg" -exec cp {} "$RAW_AIROGS/NRG/" \; 2>/dev/null || true
        find "$AIROGS_SOURCE" -path "*/NRG/*.jpeg" -exec cp {} "$RAW_AIROGS/NRG/" \; 2>/dev/null || true

        RG_COUNT=$(ls "$RAW_AIROGS/RG" 2>/dev/null | wc -l)
        NRG_COUNT=$(ls "$RAW_AIROGS/NRG" 2>/dev/null | wc -l)
        print_ok "Copied $RG_COUNT RG + $NRG_COUNT NRG images → $RAW_AIROGS"
    else
        print_error "Could not find AIROGS data anywhere under $BASE_DIR"
        echo "  Please place your downloaded AIROGS data (with RG/ and NRG/ sub-folders)"
        echo "  under $BASE_DIR or $DATA_DIR and re-run this script."
        echo "  Or rename it so the parent folder is: $RAW_AIROGS"
    fi
fi

# ===========================================================================
# CHAKSHU
# ===========================================================================
print_step "Setting up Chákṣu..."

# Already done?
EXISTING_TRAIN=$([ -d "$RAW_CHAKSU/Train" ] && echo "yes" || echo "no")
EXISTING_TEST=$([ -d "$RAW_CHAKSU/Test" ] && echo "yes" || echo "no")
if [ "$EXISTING_TRAIN" = "yes" ] && [ "$EXISTING_TEST" = "yes" ]; then
    print_ok "raw_chaksu already populated (Train/ + Test/) — skipping"
else
    TEMP="$DATA_DIR/temp_chaksu"
    mkdir -p "$TEMP"

    # Unzip Train.zip / Test.zip if present and not yet extracted
    for SPLIT in Train Test; do
        ZIP="$DATA_DIR/${SPLIT}.zip"
        if [ -f "$ZIP" ]; then
            if [ ! -d "$DATA_DIR/$SPLIT" ] && [ ! -d "$TEMP/$SPLIT" ]; then
                print_step "Extracting ${SPLIT}.zip..."
                unzip -q "$ZIP" -d "$TEMP"
                print_ok "${SPLIT}.zip extracted"
            fi
        fi
    done

    # Handle nested zips (Figshare sometimes double-zips)
    find "$TEMP" -name "*.zip" -exec unzip -q {} -d "$TEMP" \; 2>/dev/null || true

    # Now look for Train/ and Test/ folders in several possible locations
    for SPLIT in Train Test; do
        DEST="$RAW_CHAKSU/$SPLIT"
        if [ -d "$DEST" ]; then
            print_ok "$SPLIT/ already at $DEST — skipping"
            continue
        fi

        SRC=""
        for CANDIDATE in \
            "$TEMP/$SPLIT" \
            "$DATA_DIR/$SPLIT" \
            "$BASE_DIR/$SPLIT" \
            "$BASE_DIR/chaksu/$SPLIT" \
            "$DATA_DIR/chaksu/$SPLIT"
        do
            if [ -d "$CANDIDATE" ]; then
                SRC="$CANDIDATE"
                break
            fi
        done

        # Fallback: search under workspace
        if [ -z "$SRC" ]; then
            SRC=$(find "$BASE_DIR" -maxdepth 5 -type d -name "$SPLIT" 2>/dev/null | head -1)
        fi

        if [ -n "$SRC" ]; then
            print_step "Copying $SPLIT from $SRC → $DEST"
            cp -r "$SRC" "$DEST"
            print_ok "$SPLIT/ copied"
        else
            print_warn "Could not find Chákṣu $SPLIT/ folder"
            echo "  Expected: $DATA_DIR/Train.zip  OR  $DATA_DIR/Train/"
            echo "  Please upload Train.zip and Test.zip to $DATA_DIR and re-run."
        fi
    done

    # Clean up temp
    rm -rf "$TEMP" 2>/dev/null || true
fi

# ===========================================================================
# SUMMARY
# ===========================================================================
echo ""
echo "========================================================"
echo "  DATA LAYOUT SUMMARY"
echo "========================================================"
echo "  $RAW_AIROGS/"
echo "    RG/  : $(ls $RAW_AIROGS/RG 2>/dev/null | wc -l) images"
echo "    NRG/ : $(ls $RAW_AIROGS/NRG 2>/dev/null | wc -l) images"
echo ""
echo "  $RAW_CHAKSU/"
for d in Train Test; do
    if [ -d "$RAW_CHAKSU/$d" ]; then
        COUNT=$(find "$RAW_CHAKSU/$d" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo "    $d/ : $COUNT image files"
    else
        echo "    $d/ : MISSING"
    fi
done
echo "========================================================"
echo ""
echo "If all counts look correct, run:"
echo "  bash run_everything.sh"
echo ""
