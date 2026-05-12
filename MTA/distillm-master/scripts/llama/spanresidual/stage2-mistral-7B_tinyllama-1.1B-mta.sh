#!/bin/bash
# Stage 2 — SpanResidual + MTA, Mistral-7B -> TinyLLaMA-1.1B
# Same-tokenizer (LLaMA SentencePiece, vocab=32000) — không cần teacher_data_dir

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=72$(($RANDOM%90+10))
NNODES=1; NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$(basename "$BASE_PATH")" != "distillm-master" ]] && [[ "$BASE_PATH" != "/" ]]; do BASE_PATH="$(dirname "$BASE_PATH")"; done

CKPT="TinyLlama/TinyLlama-1.1B-Chat-v1.0"; CKPT_NAME="tinyllama-1.1b"
TEACHER_CKPT="VoCuc/Mistral7B_Dolly_SFT";  TEACHER_CKPT_NAME="mistral-7B-dolly-sft"
PROJECTOR_PATH="${BASE_PATH}/results/mistral/projectors/spanresidual_mistral7B_v2/projector_best.pt"
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama/"

BATCH_SIZE=2; LR=1e-3; GRAD_ACC=2; EVAL_BATCH_SIZE=2; EPOCHS=10; MAX_LENGTH=256
LAMBDA_RES=0.5; LAMBDA_RES_WARMUP=100; GAMMA_SPAN=1.0; W_SPAN_LOSS=2.0
SAVE_PATH="${BASE_PATH}/results/llama/train/spanresidual_mta_tinyllama-1.1B_mistral-7B"; SEED=42

OPTS=""
OPTS+=" --base-path ${BASE_PATH} --model-path ${CKPT} --model-type llama --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT} --teacher-ckpt-name ${TEACHER_CKPT_NAME} --teacher-model-type mistral --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE} --projector-load-path ${PROJECTOR_PATH} --d-bottleneck 64"
OPTS+=" --lambda-res ${LAMBDA_RES} --lambda-res-warmup-steps ${LAMBDA_RES_WARMUP} --gamma-span ${GAMMA_SPAN}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1 --dev-num 1000 --lr ${LR} --batch-size ${BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC} --warmup-iters 0 --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2 --clip-grad 1.0 --epochs ${EPOCHS} --kd-ratio 1.0 --warmup-ratio 0.1"
OPTS+=" --w-span-loss ${W_SPAN_LOSS} --max-length ${MAX_LENGTH} --max-prompt-length 128"
OPTS+=" --do-train --do-valid --save-interval -1 --eval-interval -1 --eval-gen"
OPTS+=" --log-interval 10 --mid-log-num -1 --save ${SAVE_PATH} --type adaptive-srkl --seed ${SEED}"
OPTS+=" --deepspeed --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
OPTS+=" --do-sample --top-k 0 --top-p 1.0 --temperature 1.0 --gen-num-beams 1 --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --student-gen"
OPTS+=" --peft lora --peft-lora-r 256 --peft-lora-alpha 8 --peft-lora-dropout 0.1"
OPTS+=" --teacher_layer_mapping 21 27 32 --student_layer_mapping 15 18 22 --split_layer_mapping 0 1 3 3"

export NCCL_DEBUG="" WANDB_DISABLED=True TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_residual_finetune.py ${OPTS} $@"
echo ${CMD}; mkdir -p ${SAVE_PATH}; CODE_BASE=HF ${CMD}
