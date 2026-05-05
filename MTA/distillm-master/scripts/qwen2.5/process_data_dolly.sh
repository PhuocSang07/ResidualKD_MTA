BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3


# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ./data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path Qwen/Qwen1.5-0.5B \
    --data-process-workers 32 \
    --max-prompt-length 128 \
    --dev-num 1000 \
    --model-type qwen
