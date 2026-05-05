#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="Qwen/Qwen3-0.6B"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}-sft"


mkdir -p ${OUTPUT_DIR}

OPTS=""

OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"
OPTS+=" --teacher_layers_mapping 32"
OPTS+=" --student_encoder_layers_finetuned 22"

# training
OPTS+=" --val_batch_size 128"

# devices
OPTS+=" --student_device cuda:1"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path distillm-master/results/qwen3/4b_base_sft/e5-bs8-lr0.0005-G1-N2-NN1-lora-16-64-0.05/2142"
OPTS+=" --tokenizer Qwen/Qwen3-0.6B"

# ==== Gọi Python ====
python src/run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log 2>&1
