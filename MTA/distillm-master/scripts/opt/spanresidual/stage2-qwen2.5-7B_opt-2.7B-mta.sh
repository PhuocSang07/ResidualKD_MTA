#!/bin/bash
# Stage 2 — SpanResidual + MTA (no entropy weight): Qwen2.5-7B → OPT-2.7B
# Cross-tokenizer: Qwen2.5 tiktoken (151936) ≠ OPT/GPT2 BPE (50272)
# Teacher: VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT (28L, d_T=3584)
# Student: facebook/opt-2.7b                    (32L, d_S=2560)
# Pre-requisite: scripts/pretrain/stage1-qwen2.5-7B-projectors.sh

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=71$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done

CKPT="facebook/opt-2.7b"
CKPT_NAME="opt-2.7b"

TEACHER_CKPT="VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT"
TEACHER_CKPT_NAME="qwen2.5-7B-dolly-sft"

PROJECTOR_PATH="${BASE_PATH}/results/qwen2.5/projectors/spanresidual_qwen2.5-7B/projector_best.pt"

STUDENT_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/opt/"
TEACHER_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen/"

BATCH_SIZE=8
LR=1e-3
GRAD_ACC=1
EVAL_BATCH_SIZE=16
EPOCHS=10
MAX_LENGTH=256

LAMBDA_RES=0.5
LAMBDA_RES_WARMUP=100
GAMMA_SPAN=1
W_SPAN_LOSS=2

SAVE_PATH="${BASE_PATH}/results/opt/train/spanresidual_mta_opt-2.7B_qwen2.5-7B"
SEED=42

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type opt"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-type qwen"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --projector-load-path ${PROJECTOR_PATH}"
OPTS+=" --d-bottleneck 64"
OPTS+=" --lambda-res ${LAMBDA_RES}"
OPTS+=" --lambda-res-warmup-steps ${LAMBDA_RES_WARMUP}"
OPTS+=" --gamma-span ${GAMMA_SPAN}"
OPTS+=" --data-dir ${STUDENT_DATA_DIR}"
OPTS+=" --teacher-data-dir ${TEACHER_DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
OPTS+=" --warmup-ratio 0.1"
OPTS+=" --w-span-loss ${W_SPAN_LOSS}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 128"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --eval-gen"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --type adaptive-srkl"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
OPTS+=" --student-gen"
# LoRA
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 256"
OPTS+=" --peft-lora-alpha 8"
OPTS+=" --peft-lora-dropout 0.1"
# Qwen2.5-7B 28L → OPT-2.7B 32L; upper-region anchors (Paradigm B-3L)
# NOTE: NO --entropy_weight flag (this is the no-entropy MTA variant)
OPTS+=" --teacher_layer_mapping 19 23 28"
OPTS+=" --student_layer_mapping 21 27 32"
OPTS+=" --split_layer_mapping 0 1 3 3"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_residual_finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
