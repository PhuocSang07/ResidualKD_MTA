#! /bin/bash

GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=./distillm-master

CKPT_NAME="opt-1.3B"
CKPT="facebook/opt-1.3b"
TEACHER_CKPT_NAME="opt-1.3B"
TEACHER_CKPT="MiniLLM/SFT-OPT-6.7B"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/opt/"
# hp
BATCH_SIZE=8
LR=0.0005
GRAD_ACC=2
EVAL_BATCH_SIZE=64
EPOCHS=5
# length
MAX_LENGTH=256
# runtime
SAVE_PATH="${BASE_PATH}/results/opt/train/spanfdd_1.3B_6.7B-v2"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type opt"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
# OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
# hp
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
# OPTS+=" --w-span-loss 3.0"
OPTS+=" --w-span-loss 2.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 128"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
OPTS+=" --do-train"
# OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
# OPTS+=" --peft-path ${PEFT_CKPT}"
# OPTS+=" --teacher-peft-name ${TEACHER_PEFT_CKPT_NAME}"
# OPTS+=" --teacher-peft-path ${TEACHER_PEFT_CKPT}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type adaptive-srkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# distillm
OPTS+=" --student-gen"

# OPTS+=" --teacher_layer_mapping 17 20 23 26 29 32"
# OPTS+=" --student_layer_mapping 14 16 18 20 22 24"
# OPTS+=" --split_layer_mapping 0 1 6 6"
OPTS+=" --teacher_layer_mapping 20 23 26 29 32"
OPTS+=" --student_layer_mapping 16 18 20 22 24"
OPTS+=" --split_layer_mapping 0 1 5 5"

OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_fdd_finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
