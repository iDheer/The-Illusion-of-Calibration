#!/bin/bash

###############################################################################
#                       NETRA-ADAPT: RUN-3 PIPELINE                         #
#                                                                             #
#  Key changes vs Run-1/2:                                                    #
#  1. Class balance: WeightedRandomSampler + weighted CrossEntropyLoss        #
#  2. Grayscale: GrayscaleToRGB strips colour before all training passes      #
#                                                                             #
#  Usage: bash run_everything.sh                                             #
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

BASE_DIR="/workspace"
DATA_DIR="$BASE_DIR/data"
CSV_DIR="$DATA_DIR/processed_csvs"
RESULTS_DIR="$BASE_DIR/results_run3"
TOTAL_START=$(date +%s)

print_header() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}\n"
}
print_progress() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning()  { echo -e "${YELLOW}[!]${NC} $1"; }
print_error()    { echo -e "${RED}[✗]${NC} $1"; }
print_phase()    {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}
elapsed_time() {
    local DIFF=$(( $(date +%s) - $1 ))
    local H=$(( DIFF / 3600 ))
    local M=$(( (DIFF % 3600) / 60 ))
    local S=$(( DIFF % 60 ))
    [ $H -gt 0 ] && echo "${H}h ${M}m ${S}s" || ([ $M -gt 0 ] && echo "${M}m ${S}s" || echo "${S}s")
}

###############################################################################
clear
print_header "NETRA-ADAPT RUN-3: GRAYSCALE + BALANCED TRAINING"
echo -e "${BOLD}Date:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${BOLD}Workspace:${NC} $BASE_DIR"
echo -e "${BOLD}Results dir:${NC} $RESULTS_DIR"
echo ""
echo -e "${YELLOW}Run-3 changes:${NC}"
echo -e "  1. ${BOLD}Class balance${NC} — WeightedRandomSampler + weighted CrossEntropyLoss"
echo -e "  2. ${BOLD}Grayscale${NC}     — GrayscaleToRGB strips colour bias at input"
echo ""

# ── HuggingFace Token ───────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.huggingface/token)
        print_progress "Loaded HF_TOKEN from ~/.huggingface/token"
    else
        print_warning "HF_TOKEN not set. DINOv3 download may fail if model is gated."
        print_warning "Set via: export HF_TOKEN=hf_..."
    fi
fi

# ── Prerequisite checks ─────────────────────────────────────────────────────
print_phase "PREREQUISITES"

if ! python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    print_error "PyTorch not found. Install with: pip install torch torchvision"
    exit 1
fi

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -gt "0" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    print_progress "GPU detected: $GPU_NAME ($GPU_COUNT device(s))"
else
    print_warning "No GPU detected. Training will be very slow on CPU."
fi

PYTHON_PACKAGES=("transformers" "timm" "sklearn" "cv2" "seaborn" "tqdm")
for pkg in "${PYTHON_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        print_progress "Package: $pkg"
    else
        print_warning "Missing package: $pkg — installing..."
        pip install $pkg -q
    fi
done

# ── Data check ──────────────────────────────────────────────────────────────
print_phase "DATA VALIDATION"

if [ ! -d "$DATA_DIR/raw_airogs" ]; then
    print_error "AIROGS data not found at $DATA_DIR/raw_airogs"
    echo "  Please upload AIROGS data first (see PIPELINE_USAGE.md)"
    exit 1
fi
if [ ! -d "$DATA_DIR/raw_chaksu" ]; then
    print_error "Chákṣu data not found at $DATA_DIR/raw_chaksu"
    echo "  Please upload Chákṣu data first (see PIPELINE_USAGE.md)"
    exit 1
fi
print_progress "Data directories found"

mkdir -p "$CSV_DIR"
mkdir -p "$RESULTS_DIR/Source_AIROGS"
mkdir -p "$RESULTS_DIR/Oracle_Chaksu"
mkdir -p "$RESULTS_DIR/Netra_Adapt"
mkdir -p "$RESULTS_DIR/evaluation"
print_progress "Result directories created under $RESULTS_DIR/"

# ── Data Preparation ────────────────────────────────────────────────────────
print_phase "DATA PREPARATION (WITH CLASS BALANCE REPORTING)"

PREP_START=$(date +%s)
python prepare_data.py
print_progress "Data prepared in $(elapsed_time $PREP_START)"

# verify key CSVs
for csv in "airogs_train.csv" "chaksu_train_labeled.csv" "chaksu_test_labeled.csv"; do
    if [ -f "$CSV_DIR/$csv" ]; then
        ROWS=$(wc -l < "$CSV_DIR/$csv")
        print_progress "$csv — $ROWS rows"
    else
        print_error "$csv not found after prepare_data.py"
        exit 1
    fi
done

# ── Phase A: Source Training ─────────────────────────────────────────────────
print_phase "PHASE A: SOURCE TRAINING — AIROGS (BALANCED + GRAYSCALE)"

SRC_START=$(date +%s)
python train_source.py
print_progress "Source training complete in $(elapsed_time $SRC_START)"

if [ ! -f "$RESULTS_DIR/Source_AIROGS/model.pth" ]; then
    print_error "Source model not saved to $RESULTS_DIR/Source_AIROGS/model.pth"
    exit 1
fi

# ── Phase B: Oracle Training ─────────────────────────────────────────────────
print_phase "PHASE B: ORACLE TRAINING — CHÁKṢU (BALANCED + GRAYSCALE)"

ORC_START=$(date +%s)
python train_oracle.py || print_warning "Oracle training failed — continuing (it's a baseline)"
print_progress "Oracle training step done in $(elapsed_time $ORC_START)"

# ── Phase C: MixEnt-Adapt ────────────────────────────────────────────────────
print_phase "PHASE C: MIXENT-ADAPT — TEST-TIME ADAPTATION (GRAYSCALE-AWARE)"

ADAPT_START=$(date +%s)
python adapt_target.py
print_progress "Adaptation complete in $(elapsed_time $ADAPT_START)"

if [ ! -f "$RESULTS_DIR/Netra_Adapt/adapted_model.pth" ]; then
    print_error "Adapted model not found at $RESULTS_DIR/Netra_Adapt/adapted_model.pth"
    exit 1
fi

# ── Phase D: Evaluation ──────────────────────────────────────────────────────
print_phase "PHASE D: COMPREHENSIVE EVALUATION"

EVAL_START=$(date +%s)
python evaluate.py
print_progress "Evaluation complete in $(elapsed_time $EVAL_START)"

# ── Final Summary ─────────────────────────────────────────────────────────────
print_header "NETRA-ADAPT RUN-3 — PIPELINE COMPLETE"
echo -e "${BOLD}Total time:${NC} $(elapsed_time $TOTAL_START)"
echo ""

if [ -f "$RESULTS_DIR/evaluation/results_table.csv" ]; then
    echo -e "${BOLD}Results table:${NC}"
    cat "$RESULTS_DIR/evaluation/results_table.csv"
    echo ""
    print_progress "Full results in $RESULTS_DIR/evaluation/"
fi

echo -e "\n${GREEN}${BOLD}Run-3 complete! ✓${NC}"
