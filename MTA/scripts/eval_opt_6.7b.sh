#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="MiniLLM/SFT-OPT-6.7B"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"
OPTS+=" --teacher_layers_mapping 32"
OPTS+=" --student_encoder_layers_finetuned 22"

# training
OPTS+=" --val_batch_size 32"

# devices
OPTS+=" --student_device cuda:1"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
OPTS+=" --tokenizer MiniLLM/SFT-OPT-6.7B"

# ==== Gọi Python ====
python src/run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log 2>&1
