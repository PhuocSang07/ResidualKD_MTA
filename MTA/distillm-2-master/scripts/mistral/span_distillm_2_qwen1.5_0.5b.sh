#!/bin/bash

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# --- Accelerate launch ---
accelerate launch \
  --config_file distillm-2-master/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  distillm-2-master/src/run_span_distillm.py \
  distillm-2-master/training_configs/mistral-7b-1.8b-span-distillm2.yaml