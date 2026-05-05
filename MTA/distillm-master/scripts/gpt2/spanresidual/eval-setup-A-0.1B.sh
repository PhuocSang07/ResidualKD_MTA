#!/bin/bash
# Eval — SpanResidual Setup A: GPT2-120M checkpoint
# Uses Multi-Level-OT eval_generate.py (4 benchmarks, 3 seeds, ROUGE-L).
#
# Usage:
#   bash eval-setup-A-0.1B.sh <run_dir> [ckpt_step]
#
# Examples:
#   # Auto-pick latest checkpoint:
#   bash eval-setup-A-0.1B.sh results/gpt2/train/spanresidual_baseline_A_0.1B_qwen1.8B
#   # Specific step:
#   bash eval-setup-A-0.1B.sh results/gpt2/train/spanresidual_baseline_A_0.1B_qwen1.8B 8900

set -e

RUN_DIR="${1:?Usage: $0 <run_dir> [ckpt_step]}"
CKPT_STEP="${2:-}"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLJ_ROOT="$(realpath "${SCRIPT_DIR}/../../../../../")"
EVAL_SCRIPT="${MLJ_ROOT}/Multi-Level-OT/eval_generate.py"
MTA_DATA_DIR="${MLJ_ROOT}/MTA/data"

# Resolve checkpoint path
if [ -n "$CKPT_STEP" ]; then
    CKPT_PATH="${RUN_DIR}/${CKPT_STEP}"
else
    CKPT_PATH=$(ls -d "${RUN_DIR}"/[0-9]* 2>/dev/null | sort -V | tail -1)
fi

if [ -z "$CKPT_PATH" ] || [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: No checkpoint found in ${RUN_DIR}"
    echo "  Expected dirs named by step number, e.g. ${RUN_DIR}/8900"
    exit 1
fi

CKPT_STEP_NUM=$(basename "$CKPT_PATH")
# Mirror Multi-Level-OT eval_results structure
RUN_NAME=$(basename "$RUN_DIR")
OUTPUT_DIR="${MLJ_ROOT}/Multi-Level-OT/eval_results/spanresidual/${RUN_NAME}/${CKPT_STEP_NUM}"
mkdir -p "$OUTPUT_DIR"

echo "=== SpanResidual Eval: Setup A (GPT2-120M) ==="
echo "  Checkpoint : ${CKPT_PATH}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Data       : ${MTA_DATA_DIR}"
echo ""

python "${EVAL_SCRIPT}" \
    --model_path   "${CKPT_PATH}" \
    --tokenizer_name openai-community/gpt2 \
    --mta_data_dir  "${MTA_DATA_DIR}" \
    --output_dir    "${OUTPUT_DIR}" \
    --max_new_tokens 128 \
    --batch_size 16 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "Results saved to ${OUTPUT_DIR}/scores.json"
