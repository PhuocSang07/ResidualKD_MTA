#!/bin/bash
# Run all distillation experiments sequentially (small → large).
# Usage: bash run.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

run() {
    local script="$PWD/$1"
    local log="$SCRIPT_DIR/logs/${1%.sh}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$log")"
    echo "===== START: $1 =====" | tee -a "$log"
    bash "$script" 2>&1 | tee -a "$log"
    echo "===== DONE:  $1 =====" | tee -a "$log"
}

# run install.sh

# Activate the venv installed above so torchrun is on PATH for all sub-scripts
# source "$SCRIPT_DIR/.venv/bin/activate"

cd MTA/
# Phase 1: Pretrain Projectors
run distillm-master/scripts/pretrain/stage1-mistral-7B-projectors.sh
run distillm-master/scripts/pretrain/stage1-qwen1.8B-projectors.sh
run distillm-master/scripts/pretrain/stage1-qwen2.5-7B-projectors.sh

# Phase 2: Distill
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen2.5-7B_gpt2-1.5B-paper.sh
run distillm-master/scripts/llama/spanresidual/stage2-mistral-7B_tinyllama-1.1B-paper.sh
run distillm-master/scripts/opt/spanresidual/stage2-qwen2.5-7B_opt-2.7B-paper.sh
