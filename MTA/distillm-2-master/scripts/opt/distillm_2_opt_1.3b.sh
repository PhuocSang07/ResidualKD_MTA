#!/bin/bash

# --- GPU selection ---
export CUDA_VISIBLE_DEVICES=1

# --- Accelerate launch ---
accelerate launch \
  --config_file distillm-2-master/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  distillm-2-master/src/run_distillm.py \
  distillm-2-master/training_configs/opt-6.7b-1.3b-distillm2.yaml