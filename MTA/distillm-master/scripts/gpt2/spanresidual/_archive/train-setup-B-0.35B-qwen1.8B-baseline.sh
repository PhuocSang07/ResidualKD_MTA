#!/bin/bash
# Setup B BASELINE — SpanResidual KD (no MTA): Qwen1.5-1.8B -> GPT2-medium 345M
# Includes lambda_res warmup to fix early training instability.
# Requires: projector_best.pt from pretrain-qwen1.8B-projectors.sh (v2)

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=69$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=./distillm-master

# Student: GPT2-medium (345M, hidden=1024, 24 layers)
CKPT_NAME="gpt2-medium"
CKPT="openai-community/gpt2-medium"

TEACHER_CKPT="VoCuc/Qwen1.5_1.8B_SFT_Dolly"
TEACHER_CKPT_NAME="qwen1.5-1.8B-sft-dolly"

# Same projector as Setup B MTA
PROJECTOR_PATH="${BASE_PATH}/results/qwen/projectors/spanresidual_qwen1.8B_v2/projector_best.pt"

STUDENT_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
TEACHER_DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen/"

BATCH_SIZE=16
LR=1e-4
GRAD_ACC=1
EVAL_BATCH_SIZE=32
EPOCHS=10
MAX_LENGTH=256

LAMBDA_RES=0.5
LAMBDA_RES_WARMUP=500   # ramp 0→0.5 over 500 steps
GAMMA_SPAN=0.0
W_SPAN_LOSS=0.0

SAVE_PATH="${BASE_PATH}/results/gpt2/train/spanresidual_baseline_B_0.35B_qwen1.8B"
SEED=42

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type gpt2"
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
# Qwen 24 layers -> GPT2-medium 24 layers (1:1 mapping)
OPTS+=" --teacher_layer_mapping 8 16 24"
OPTS+=" --student_layer_mapping 8 16 24"
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
