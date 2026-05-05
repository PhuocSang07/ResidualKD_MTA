#!/bin/bash
# Stage 1 — Projector pretraining for Mistral-7B teacher.
# d_T=4096 (Mistral-7B hidden size), d_A=64.
# Data: processed_data/dolly/full/mistral/ (already tokenised)

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=67$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done

TEACHER_CKPT="VoCuc/Mistral7B_Dolly_SFT"
TEACHER_CKPT_NAME="mistral-7B-dolly-sft"
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/mistral/"

BATCH_SIZE=64
EVAL_BATCH_SIZE=64
GRAD_ACC=1
D_BOTTLENECK=64
PROJECTOR_EPOCHS=10
PROJECTOR_LR=1e-3

SAVE_PATH="${BASE_PATH}/results/mistral/projectors/spanresidual_mistral7B_v2"
SEED=42

OPTS=""
OPTS+=" --model-path ${TEACHER_CKPT}"
OPTS+=" --model-type mistral"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
OPTS+=" --train-num -1"
OPTS+=" --train-ratio 1"
OPTS+=" --dev-ratio 1"
OPTS+=" --lr ${PROJECTOR_LR}"
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-pretrain-epochs ${PROJECTOR_EPOCHS}"
OPTS+=" --d-bottleneck ${D_BOTTLENECK}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --weight-decay 1e-4"
OPTS+=" --clip-grad 1.0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-min 1e-6"
OPTS+=" --max-length 256"
OPTS+=" --max-prompt-length 128"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --type projector-pretrain"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 50"
OPTS+=" --mid-log-num -1"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_residual_pretrain.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
