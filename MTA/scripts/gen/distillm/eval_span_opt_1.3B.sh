#! /bin/bash

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")y

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="facebook/opt-1.3b"
OUTPUT_DIR="${BASE_PATH}/eval_gen_outputs/${MODEL_PATH}-spandistillm-3570-v2"


mkdir -p ${OUTPUT_DIR}

OPTS=""

OPTS+=" --train_data ${BASE_PATH}/data/dolly/train.jsonl"
OPTS+=" --val_data ${BASE_PATH}/data/dolly/dev.jsonl"
OPTS+=" --test_data ${BASE_PATH}/data/dolly/valid.jsonl"
OPTS+=" --teacher_layers_mapping 32"
OPTS+=" --student_encoder_layers_finetuned 22"

# training
OPTS+=" --val_batch_size 64"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --output_dir ${OUTPUT_DIR}"

# extra arguments
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
OPTS+=" --lora_path distillm-master/results/opt/train/spandistillm_1.3B_6.7B-v2/3570"
OPTS+=" --tokenizer facebook/opt-350m"

# ==== Gọi Python ====
python src/run_get_eval_answer.py ${OPTS}
