#!/bin/bash

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=0

# --- Accelerate launch ---
accelerate launch \
  --config_file distillm-2-master/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  distillm-2-master/src/run_distillm.py \
  distillm-2-master/training_configs/qwen1.5-1.8b-0.5b-distillm2.yaml