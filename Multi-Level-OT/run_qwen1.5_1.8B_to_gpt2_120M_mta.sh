#!/bin/bash
# Distill VoCuc/Qwen1.5_1.8B_SFT_Dolly (teacher, 2048-dim, 24 layers)
#        → openai-community/gpt2 (student, 768-dim, 12 layers)
# Variant: MTA (OT + Span loss, no entropy weight)

# GPUS=(0 1 2 3 4 5 6 7)
GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export DS_IGNORE_CUDA_DETECTION=1

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
OPTS+=" --model_name openai-community/gpt2"
OPTS+=" --dataset.file $SCRIPT_DIR/llm_distillation/datasets/loader/dolly.py"
OPTS+=" --lr 1e-5"
OPTS+=" --num_epochs 10"
OPTS+=" --batch_size_training 8"
OPTS+=" --gradient_accumulation_steps 2"
OPTS+=" --val_batch_size 8"
OPTS+=" --output_dir $SCRIPT_DIR/output/qwen1.5-1.8B-to-gpt2-120M/mta"
OPTS+=" --distillation"
OPTS+=" --distillation_config_model_name VoCuc/Qwen1.5_1.8B_SFT_Dolly"
OPTS+=" --distillation_config_distil_factor 0.15"
OPTS+=" --distillation_config_cross_entropy_factor 1.0"
OPTS+=" --distillation_config_student_temperature 1.0"
OPTS+=" --distillation_config_teacher_temperature 2.0"
OPTS+=" --distillation_config_pure_bf16"
OPTS+=" --student_device cuda:0"
OPTS+=" --teacher_device cuda:0"
OPTS+=" --save_step 5000"
OPTS+=" --f 1"
OPTS+=" --span_loss_weight 0.1"
OPTS+=" --student_layer_mapping 6,9,12"
OPTS+=" --teacher_layer_mapping 12,18,24"
OPTS+=" --split_layer_mapping 0,1,3"
OPTS+=" --use_phrase_spans"
OPTS+=" --context_length 1024"
OPTS+=" --student_hidden_size 768"
OPTS+=" --teacher_hidden_size 2048"

export NCCL_DEBUG=""
export WANDB_DISABLED=False
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=$SCRIPT_DIR

CMD="torchrun ${DISTRIBUTED_ARGS} $SCRIPT_DIR/finetuning.py ${OPTS} $@"
echo ${CMD}
mkdir -p $SCRIPT_DIR/output/qwen1.5-1.8B-to-gpt2-120M/mta
${CMD}
