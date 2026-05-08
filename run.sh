#!/bin/bash
# Run all distillation experiments + evaluation sequentially.
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

# Skip install + venv activation if a Python environment is already active
# (e.g. conda env). Otherwise install dependencies into a project-local venv.
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    run install.sh
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "Skipping install.sh / .venv (active env: ${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV})"
fi

cd MTA/

# ========== Phase 1: Pretrain Projectors (run once per teacher) ==========
# run distillm-master/scripts/pretrain/stage1-mistral-7B-projectors.sh
# run distillm-master/scripts/pretrain/stage1-qwen1.8B-projectors.sh
# run distillm-master/scripts/pretrain/stage1-qwen2.5-7B-projectors.sh

# ========== Phase 2: Distillation ==========
# Pattern: paper (residual only) | mta (residual + span) | mta-entropy (+ entropy weight)

# --- OPT-2.7B (Qwen2.5-7B teacher) ---
# run distillm-master/scripts/opt/spanresidual/stage2-qwen2.5-7B_opt-2.7B-paper.sh
# run distillm-master/scripts/opt/spanresidual/stage2-qwen2.5-7B_opt-2.7B-mta.sh
run distillm-master/scripts/opt/spanresidual/stage2-qwen2.5-7B_opt-2.7B-paper-mta-entropy.sh

# --- GPT2-XL 1.5B (Qwen2.5-7B teacher) ---
# run distillm-master/scripts/gpt2/spanresidual/stage2-qwen2.5-7B_gpt2-1.5B-paper.sh
# run distillm-master/scripts/gpt2/spanresidual/stage2-qwen2.5-7B_gpt2-1.5B-mta.sh
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen2.5-7B_gpt2-1.5B-mta-entropy.sh

# --- GPT2-Medium 0.35B (Qwen1.8B teacher) ---
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen1.8B_gpt2-345M-paper.sh
# run distillm-master/scripts/gpt2/spanresidual/stage2-qwen1.8B_gpt2-345M-mta.sh
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen1.8B_gpt2-345M-mta-entropy.sh

# --- GPT2-Small 0.1B / 120M (Qwen1.8B teacher) ---
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen1.8B_gpt2-120M-paper.sh
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen1.8B_gpt2-120M-mta.sh
run distillm-master/scripts/gpt2/spanresidual/stage2-qwen1.8B_gpt2-120M-mta-entropy.sh

# --- TinyLlama-1.1B (Mistral-7B teacher) ---
run distillm-master/scripts/llama/spanresidual/stage2-mistral-7B_tinyllama-1.1B-paper.sh
# run distillm-master/scripts/llama/spanresidual/stage2-mistral-7B_tinyllama-1.1B-mta.sh
run distillm-master/scripts/llama/spanresidual/stage2-mistral-7B_tinyllama-1.1B-mta-entropy.sh

# ========== Phase 3: Evaluation ==========
# Eval scripts read LoRA checkpoints from results/<arch>/train/spanresidual_<variant>_<...>/<CKPT_STEP>/
# CKPT_STEP=14290 by default — adjust inside each eval_*.sh if total steps differ.

# --- OPT-2.7B ---
run scripts/eval_opt_2.7B_residual_paper.sh
# run scripts/eval_opt_2.7B_residual_mta.sh
run scripts/eval_opt_2.7B_residual_mta_entropy.sh

# --- GPT2-XL 1.5B ---
run scripts/eval_gpt2_1.5B_residual_paper.sh
# run scripts/eval_gpt2_1.5B_residual_mta.sh
run scripts/eval_gpt2_1.5B_residual_mta_entropy.sh

# --- GPT2-Medium 0.35B ---
run scripts/eval_gpt2_0.35B_residual_paper.sh
# run scripts/eval_gpt2_0.35B_residual_mta.sh
run scripts/eval_gpt2_0.35B_residual_mta_entropy.sh

# --- GPT2-Small 0.1B / 120M ---
run scripts/eval_gpt2_0.1B_residual_paper.sh
run scripts/eval_gpt2_0.1B_residual_mta.sh
run scripts/eval_gpt2_0.1B_residual_mta_entropy.sh

# --- TinyLlama-1.1B ---
run scripts/eval_llama_1.1B_residual_paper.sh
# run scripts/eval_llama_1.1B_residual_mta.sh
run scripts/eval_llama_1.1B_residual_mta_entropy.sh
