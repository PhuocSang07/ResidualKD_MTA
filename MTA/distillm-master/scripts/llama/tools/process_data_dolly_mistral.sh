#!/bin/bash
# Tokenise dolly data with the Mistral/LLaMA SentencePiece tokenizer.
# Produces: processed_data/dolly/full/mistral/{train,valid}_0.{bin,idx}
# Usage: bash process_data_dolly_mistral.sh   (run from distillm-master/ or anywhere)

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do
    BASE_PATH="$(dirname "$BASE_PATH")"
done

export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path VoCuc/Mistral7B_Dolly_SFT \
    --data-process-workers 16 \
    --max-prompt-length 128 \
    --dev-num 1000 \
    --model-type mistral
