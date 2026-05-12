#!/bin/bash
# Stage 2 — SpanResidual KD (paper-faithful): Mistral-7B → TinyLLaMA-1.1B
# DIFFERENT tokenizers: Mistral-7B and TinyLlama use DIFFERENT SentencePiece models
# despite both having vocab=32000. Data MUST be tokenized with TinyLlama tokenizer.
# → Requires: bash scripts/llama/tools/process_data_dolly_tinyllama.sh first
# → Data dir: processed_data/dolly/full/llama/ (NOT mistral/)
# Teacher: VoCuc/Mistral7B_Dolly_SFT (32L, d_T=4096)
# Student: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (22L, d_S=2048)
# Pre-requisite: scripts/pretrain/stage1-mistral-7B-projectors.sh

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=72$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done

CKPT="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CKPT_NAME="tinyllama-1.1b"

TEACHER_CKPT="VoCuc/Mistral7B_Dolly_SFT"
TEACHER_CKPT_NAME="mistral-7B-dolly-sft"

PROJECTOR_PATH="${BASE_PATH}/results/mistral/projectors/spanresidual_mistral7B_v2/projector_best.pt"

# TinyLlama tokenizer data dir (processed with TinyLlama tokenizer, NOT mistral/)
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama/"


BATCH_SIZE=2
LR=1e-3
GRAD_ACC=2
EVAL_BATCH_SIZE=2
EPOCHS=10
MAX_LENGTH=256

LAMBDA_RES=0.5
LAMBDA_RES_WARMUP=100
GAMMA_SPAN=0.0
W_SPAN_LOSS=0.0

SAVE_PATH="${BASE_PATH}/results/llama/train/spanresidual_paper_tinyllama-1.1B_mistral-7B"
SEED=42

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --model-type llama"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-type mistral"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --projector-load-path ${PROJECTOR_PATH}"
OPTS+=" --d-bottleneck 64"
OPTS+=" --lambda-res ${LAMBDA_RES}"
OPTS+=" --lambda-res-warmup-steps ${LAMBDA_RES_WARMUP}"
OPTS+=" --gamma-span ${GAMMA_SPAN}"
OPTS+=" --data-dir ${DATA_DIR}"
# NO --teacher-data-dir: teacher is only used for logits/residuals, not data loading
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
# LoRA: student 1.1B — reduce trainable params and GPU memory
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 256"
OPTS+=" --peft-lora-alpha 8"
OPTS+=" --peft-lora-dropout 0.1"
# Mistral-7B 32L → TinyLLaMA 22L; anchor at thirds
OPTS+=" --teacher_layer_mapping 10 21 32"
OPTS+=" --student_layer_mapping 7 14 22"
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
