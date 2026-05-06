#!/bin/bash
# Stage 2 — SpanResidual + MTA + Entropy Weight, Setup E
# Cross-tokenizer: Qwen2.5-7B (teacher) -> GPT2-XL 1.5B (student)

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=68$(($RANDOM%90+10))
NNODES=1; NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done

CKPT="openai-community/gpt2-xl";          CKPT_NAME="gpt2-xl"
TEACHER_CKPT="VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT"; TEACHER_CKPT_NAME="qwen2.5-7B-dolly-sft"
PROJECTOR_PATH="${BASE_PATH}/results/qwen2.5/projectors/spanresidual_qwen2.5-7B/projector_best.pt"
STUDENT_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
TEACHER_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen/"

BATCH_SIZE=8; LR=1e-4; GRAD_ACC=2; EVAL_BATCH_SIZE=8; EPOCHS=10; MAX_LENGTH=256
LAMBDA_RES=0.5; LAMBDA_RES_WARMUP=100; GAMMA_SPAN=1.0; W_SPAN_LOSS=2.0
SAVE_PATH="${BASE_PATH}/results/gpt2/train/spanresidual_mta_entropy_E_1.5B_qwen2.5-7B"; SEED=42

OPTS=""
OPTS+=" --base-path ${BASE_PATH} --model-path ${CKPT} --model-type gpt2 --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT} --teacher-ckpt-name ${TEACHER_CKPT_NAME} --teacher-model-type qwen --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE} --projector-load-path ${PROJECTOR_PATH} --d-bottleneck 64"
OPTS+=" --lambda-res ${LAMBDA_RES} --lambda-res-warmup-steps ${LAMBDA_RES_WARMUP} --gamma-span ${GAMMA_SPAN}"
OPTS+=" --data-dir ${STUDENT_DATA_DIR} --teacher-data-dir ${TEACHER_DATA_DIR}"
OPTS+=" --num-workers 1 --dev-num 1000 --lr ${LR} --batch-size ${BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC} --warmup-iters 0 --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2 --clip-grad 1.0 --epochs ${EPOCHS} --kd-ratio 1.0 --warmup-ratio 0.1"
OPTS+=" --w-span-loss ${W_SPAN_LOSS} --max-length ${MAX_LENGTH} --max-prompt-length 128"
OPTS+=" --do-train --do-valid --save-interval -1 --eval-interval -1 --eval-gen"
OPTS+=" --log-interval 10 --mid-log-num -1 --save ${SAVE_PATH} --type adaptive-srkl --seed ${SEED}"
OPTS+=" --deepspeed --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
OPTS+=" --do-sample --top-k 0 --top-p 1.0 --temperature 1.0 --gen-num-beams 1 --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --student-gen"
OPTS+=" --entropy_weight"
OPTS+=" --peft lora --peft-lora-r 256 --peft-lora-alpha 8 --peft-lora-dropout 0.1"
OPTS+=" --teacher_layer_mapping 9 19 28 --student_layer_mapping 16 32 48 --split_layer_mapping 0 1 3 3"

export NCCL_DEBUG="" WANDB_DISABLED=True TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_residual_finetune.py ${OPTS} $@"
echo ${CMD}; mkdir -p ${SAVE_PATH}; CODE_BASE=HF ${CMD}
