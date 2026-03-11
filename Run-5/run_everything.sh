#!/bin/bash

###############################################################################
#                       NETRA-ADAPT: RUN-5 PIPELINE                         #
#                                                                             #
#  All 4 bugs fixed vs Runs 1-4:                                             #
#  1. Colour inputs restored (GrayscaleToRGB removed)                        #
#  2. Chaksu label matching fixed (fname.split('-') bug removed)             #
#  3. Oracle uses AUROC early stopping + val split                           #
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
RESULTS_DIR="$BASE_DIR/results_run5"
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
print_header "NETRA-ADAPT RUN-5: COLOUR + DIVERSITY LOSS + CORRECT CSV + PATCH ADAIN"
echo -e "${BOLD}Date:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${BOLD}Workspace:${NC} $BASE_DIR"
echo -e "${BOLD}Results dir:${NC} $RESULTS_DIR"
echo ""
echo -e "${YELLOW}Run-5 fixes vs Runs 1-4:${NC}"
echo -e "  1. ${BOLD}Diversity loss${NC}  — L_SFDA = L_ent - 1.0*L_div (was computed but never subtracted)"
echo -e "  2. ${BOLD}Correct CSV${NC}     — Adapt on chaksu_train_unlabeled.csv, not test set"
echo -e "  3. ${BOLD}Patch AdaIN${NC}     — Spatial [B,N,D] per-image stats (not CLS batch stats)"
echo -e "  4. ${BOLD}Colour + labels${NC} — GrayscaleToRGB removed, fname fix, AUROC monitoring (Run-4)"
echo ""

# ── HuggingFace Token ───────────────────────────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.huggingface/token)
        print_progress "Loaded HF_TOKEN from ~/.huggingface/token"
    else
        print_error "HF_TOKEN is not set — required to download DINOv3 (gated model)."
        echo ""
        echo "  Run this before starting the pipeline:"
        echo "    export HF_TOKEN=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"
        echo ""
        exit 1
    fi
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # older transformers versions use this name
export HUGGINGFACE_TOKEN="$HF_TOKEN"

# ── Resolve python binary ───────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    print_error "No python/python3 found. Install Python 3 first."
    exit 1
fi
print_progress "Using: $(command -v $PYTHON) ($("$PYTHON" --version 2>&1))"

# ── Prerequisite checks ─────────────────────────────────────────────────────
print_phase "PREREQUISITES"

# Detect GPU compute capability from nvidia-smi (before importing torch)
GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | head -1 | tr -d '.' | xargs || echo "")
print_progress "GPU compute capability: sm_${GPU_SM:-unknown}"

# Decide which PyTorch build is needed
#   sm_120 / sm_121 = Blackwell (RTX 5090) — needs nightly cu128
#   everything else — stable cu121/cu118/cu117
if [[ "$GPU_SM" == "120" || "$GPU_SM" == "121" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
    TORCH_EXTRA="--pre"
    print_warning "Blackwell GPU (sm_${GPU_SM}) detected — nightly cu128 build required"
else
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1 || \
               nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || \
               echo "12")
    if [[ "$CUDA_VER" == 11.8* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    elif [[ "$CUDA_VER" == 11.* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu117"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
    TORCH_EXTRA=""
fi

# Install or force-reinstall PyTorch with the correct build
NEEDS_INSTALL=false
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
    NEEDS_INSTALL=true
    print_warning "PyTorch not found — installing..."
else
    # Already installed — verify it actually works on this GPU
    TORCH_OK=$("$PYTHON" -c "
import torch, warnings
warnings.filterwarnings('ignore')
try:
    t = torch.zeros(1).cuda()
    print('ok')
except Exception as e:
    print('fail')
" 2>/dev/null || echo "fail")
    if [[ "$TORCH_OK" != "ok" ]]; then
        print_warning "Installed PyTorch cannot use this GPU — reinstalling correct build..."
        NEEDS_INSTALL=true
    fi
fi

if $NEEDS_INSTALL; then
    print_warning "Installing: pip install $TORCH_EXTRA torch torchvision --index-url $TORCH_INDEX"
    pip install $TORCH_EXTRA torch torchvision --index-url "$TORCH_INDEX" --force-reinstall
fi

"$PYTHON" -c "
import torch
print('PyTorch ' + torch.__version__ + ', CUDA available: ' + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    t = torch.zeros(1).cuda()
    print('GPU tensor test: OK (' + torch.cuda.get_device_name(0) + ')')
else:
    print('WARNING: CUDA not available — training will run on CPU')
"

GPU_COUNT=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -gt "0" ]; then
    GPU_NAME=$("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    print_progress "GPU detected: $GPU_NAME ($GPU_COUNT device(s))"
else
    print_warning "No GPU detected. Training will be very slow on CPU."
fi

PYTHON_PACKAGES=("transformers" "timm" "sklearn" "cv2" "seaborn" "tqdm" "pandas" "PIL")
declare -A PIP_NAME=(["sklearn"]="scikit-learn" ["cv2"]="opencv-python" ["PIL"]="pillow")
for pkg in "${PYTHON_PACKAGES[@]}"; do
    if "$PYTHON" -c "import $pkg" 2>/dev/null; then
        print_progress "Package: $pkg"
    else
        pip_pkg="${PIP_NAME[$pkg]:-$pkg}"
        print_warning "Missing package: $pkg — installing $pip_pkg..."
        pip install "$pip_pkg" -q
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
"$PYTHON" prepare_data.py
print_progress "Data prepared in $(elapsed_time $PREP_START)"

# verify key CSVs
for csv in "airogs_train.csv" "chaksu_train_labeled.csv" "chaksu_test_labeled.csv"; do
    if [ -f "$CSV_DIR/$csv" ]; then
        ROWS=$(wc -l < "$CSV_DIR/$csv")
        print_progress "$csv — $ROWS rows"
    else
        print_error "$csv not found after prepare_data.py — check output above"
        exit 1
    fi
done

# ── Phase A: Source Training ─────────────────────────────────────────────────
print_phase "PHASE A: SOURCE TRAINING — AIROGS (BALANCED + COLOUR)"

SRC_START=$(date +%s)
"$PYTHON" train_source.py
print_progress "Source training complete in $(elapsed_time $SRC_START)"

if [ ! -f "$RESULTS_DIR/Source_AIROGS/model.pth" ]; then
    print_error "Source model not saved to $RESULTS_DIR/Source_AIROGS/model.pth"
    exit 1
fi

# ── Phase B: Oracle Training ─────────────────────────────────────────────────
print_phase "PHASE B: ORACLE TRAINING — CHÁKṢU (BALANCED + COLOUR + AUROC MONITORING)"

ORC_START=$(date +%s)
"$PYTHON" train_oracle.py || print_warning "Oracle training failed — continuing (it's a baseline)"
print_progress "Oracle training step done in $(elapsed_time $ORC_START)"

# ── Phase C: MixEnt-Adapt ────────────────────────────────────────────────────
print_phase "PHASE C: MIXENT-ADAPT — TEST-TIME ADAPTATION (COLOUR)"

ADAPT_START=$(date +%s)
"$PYTHON" adapt_target.py
print_progress "Adaptation complete in $(elapsed_time $ADAPT_START)"

if [ ! -f "$RESULTS_DIR/Netra_Adapt/adapted_model.pth" ]; then
    print_error "Adapted model not found at $RESULTS_DIR/Netra_Adapt/adapted_model.pth"
    exit 1
fi

# ── Phase D: Evaluation ──────────────────────────────────────────────────────
print_phase "PHASE D: COMPREHENSIVE EVALUATION"

EVAL_START=$(date +%s)
"$PYTHON" evaluate.py
print_progress "Evaluation complete in $(elapsed_time $EVAL_START)"

# ── Final Summary ─────────────────────────────────────────────────────────────
print_header "NETRA-ADAPT RUN-5 — PIPELINE COMPLETE"
echo -e "${BOLD}Total time:${NC} $(elapsed_time $TOTAL_START)"
echo ""

if [ -f "$RESULTS_DIR/evaluation/results_table.csv" ]; then
    echo -e "${BOLD}Results table:${NC}"
    cat "$RESULTS_DIR/evaluation/results_table.csv"
    echo ""
    print_progress "Full results in $RESULTS_DIR/evaluation/"
fi

echo -e "\n${GREEN}${BOLD}Run-5 complete! ✓${NC}"
