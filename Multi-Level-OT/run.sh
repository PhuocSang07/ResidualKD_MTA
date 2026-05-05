#!/bin/bash
# Run all distillation experiments sequentially (small → large).
# Usage: bash run.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

run() {
    local script="$SCRIPT_DIR/$1"
    local log="$SCRIPT_DIR/logs/${1%.sh}_$(date +%Y%m%d_%H%M%S).log"
    echo "===== START: $1 =====" | tee -a "$log"
    bash "$script" 2>&1 | tee -a "$log"
    echo "===== DONE:  $1 =====" | tee -a "$log"
}

# run install.sh

# Activate the venv installed above so torchrun is on PATH for all sub-scripts
# source "$SCRIPT_DIR/.venv/bin/activate"

# Qwen1.5-1.8B → GPT-2 120M
run run_qwen1.5_1.8B_to_gpt2_120M_mta.sh
# run run_qwen1.5_1.8B_to_gpt2_120M_mta_ew.sh

# Qwen1.5-1.8B → GPT-2 340M
run run_qwen1.5_1.8B_to_gpt2_340M_mta.sh
# run run_qwen1.5_1.8B_to_gpt2_340M_mta_ew.sh

# Qwen2.5-7B → GPT-2 1.5B
# run run_qwen2.5_7B_to_gpt2_1.5B_mta.sh
# run run_qwen2.5_7B_to_gpt2_1.5B_mta_ew.sh

# Qwen2.5-7B → OPT-2.7B
# run run_qwen2.5_7B_to_opt_2.7B_mta.sh
# run run_qwen2.5_7B_to_opt_2.7B_mta_ew.sh

# Mistral-7B → TinyLlama-1.1B
# run run_mistral_7B_to_tinyllama_1.1B_mta.sh
# run run_mistral_7B_to_tinyllama_1.1B_mta_ew.sh
