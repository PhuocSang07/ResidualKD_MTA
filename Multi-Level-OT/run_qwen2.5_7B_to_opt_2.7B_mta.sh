#!/bin/bash
# Distill VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT (teacher, 3584-dim, 28 layers)
#        → facebook/opt-2.7b (student, 2560-dim, 32 layers)
# Variant: MTA (OT + Span loss, no entropy weight)

GPUS=(0 1 2 3 4 5 6 7)
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPTS=""
OPTS+=" --model_name facebook/opt-2.7b"
OPTS+=" --dataset.file $SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py"
OPTS+=" --lr 1e-6"
OPTS+=" --num_epochs 10"
OPTS+=" --batch_size_training 2"
OPTS+=" --gradient_accumulation_steps 1"
OPTS+=" --val_batch_size 16"
OPTS+=" --output_dir $SCRIPT_DIR/output/qwen2.5-7B-to-opt-2.7B/mta"
OPTS+=" --distillation"
OPTS+=" --distillation_config_model_name VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT"
OPTS+=" --distillation_config_distil_factor 0.15"
OPTS+=" --distillation_config_cross_entropy_factor 1.0"
OPTS+=" --distillation_config_student_temperature 1.0"
OPTS+=" --distillation_config_teacher_temperature 2.0"
OPTS+=" --distillation_config_pure_bf16"
OPTS+=" --student_device cuda:0"
OPTS+=" --teacher_device cuda:0"
OPTS+=" --save_step 2500"
OPTS+=" --f 1"
OPTS+=" --span_loss_weight 3.0"
OPTS+=" --student_layer_mapping 16,19,22,26,29,32"
OPTS+=" --teacher_layer_mapping 14,17,20,22,25,28"
OPTS+=" --split_layer_mapping 0,1,6"
OPTS+=" --use_phrase_spans"
OPTS+=" --context_length 1024"
OPTS+=" --student_hidden_size 2560"
OPTS+=" --teacher_hidden_size 3584"
OPTS+=" --use_peft"
OPTS+=" --lora_r 256"
OPTS+=" --lora_alpha 8"
OPTS+=" --lora_dropout 0.1"

export NCCL_DEBUG=""
export WANDB_DISABLED=False
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=$SCRIPT_DIR

CMD="torchrun ${DISTRIBUTED_ARGS} $SCRIPT_DIR/finetuning.py ${OPTS} $@"
echo ${CMD}
mkdir -p $SCRIPT_DIR/output/qwen2.5-7B-to-opt-2.7B/mta
${CMD}
