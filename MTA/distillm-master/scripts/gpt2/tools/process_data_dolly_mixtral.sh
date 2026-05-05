#!/bin/bash
# Process Dolly with Mixtral-8x7B tokenizer for cross-tokenizer KD.
# Output: processed_data/dolly/full/mixtral/

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ./data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --data-process-workers 32 \
    --max-prompt-length 128 \
    --dev-num 1000 \
    --model-type mistral
