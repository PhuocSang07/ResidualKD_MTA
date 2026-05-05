#! /bin/bash

GPUS=(0)

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
CKPT_NAME="opt-350m"
CKPT="facebook/opt-350m"
TEACHER_CKPT_NAME="2.7B-sft"
TEACHER_CKPT="HoangTran223/MCW_KD_OPT2.7B_SFT"
# MP_SIZE=4
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/opt/"
# LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/opt/512/1M/"
# hp
BATCH_SIZE=16
LR=0.00005
GRAD_ACC=1
EVAL_BATCH_SIZE=32
EPOCHS=10
# length
MAX_LENGTH=256
# runtime
SAVE_PATH="${BASE_PATH}/results/opt/train/spandistillm/0.3B_2.7B_v3"
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
OPTS+=" --gradient-checkpointing"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${MP_SIZE}"
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
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
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

OPTS+=" --teacher_layer_mapping 24 36 48"
OPTS+=" --student_layer_mapping 6 9 12"
OPTS+=" --split_layer_mapping 0 1 3 3"

OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/span_finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
