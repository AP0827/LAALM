#!/bin/bash
# Auto-AVSR Quick Start Commands
# Source this file or copy commands as needed

# Set the auto_avsr directory
export AUTOAVSR_DIR="/home/asish/LAALM/auto_avsr"
export PYTHON_BIN="$AUTOAVSR_DIR/.venv/bin/python"

# Activate virtual environment
alias avsr-activate="cd $AUTOAVSR_DIR && source .venv/bin/activate"

# Verify setup
alias avsr-verify="cd $AUTOAVSR_DIR && $PYTHON_BIN verify_setup.py"

# Quick training command (customize as needed)
avsr-train() {
    cd $AUTOAVSR_DIR
    $PYTHON_BIN train.py \
        --exp-dir=./exp \
        --exp-name="${1:-my_experiment}" \
        --modality="${2:-video}" \
        --root-dir="${3:-/path/to/data}" \
        --train-file="${4:-train.csv}" \
        --num-nodes=1
}

# Quick evaluation command (customize as needed)
avsr-eval() {
    cd $AUTOAVSR_DIR
    $PYTHON_BIN eval.py \
        --modality="${1:-video}" \
        --root-dir="${2:-/path/to/data}" \
        --test-file="${3:-test.csv}" \
        --pretrained-model-path="${4:-./exp/my_experiment/model_avg_10.pth}"
}

echo "Auto-AVSR commands loaded!"
echo ""
echo "Available commands:"
echo "  avsr-activate    - Activate the virtual environment"
echo "  avsr-verify      - Verify installation"
echo "  avsr-train       - Quick train command"
echo "  avsr-eval        - Quick eval command"
echo ""
echo "Examples:"
echo "  avsr-verify"
echo "  avsr-train my_model video /data/lrs3 train.csv"
echo "  avsr-eval video /data/lrs3 test.csv ./exp/my_model/model.pth"
