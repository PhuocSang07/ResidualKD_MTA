#! /bin/bash

SEED=42
CKPT_STEP=14290

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH="distillm-master/results/llama/train/spanresidual_paper_tinyllama-1.1B_mistral-7B/${CKPT_STEP}"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/tinyllama-1.1B-spanresidual-paper-${CKPT_STEP}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"

# training
OPTS+=" --val_batch_size 64"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
OPTS+=" --lora_path ${LORA_PATH}"
OPTS+=" --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ==== Gọi Python ====
python src/run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log 2>&1
