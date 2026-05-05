#!/bin/bash
# Eval: Qwen1.5-1.8B → GPT-2 120M

export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DS_IGNORE_CUDA_DETECTION=1

RUN_DIR="$SCRIPT_DIR/output/qwen1.5-1.8B-to-gpt2-120M/mta"

# Đặt số step thủ công, hoặc để trống để tự lấy checkpoint lớn nhất
CKPT_STEP=""

if [ -n "$CKPT_STEP" ]; then
    CKPT_PATH="$RUN_DIR/$CKPT_STEP"
else
    CKPT_PATH=$(ls -d "$RUN_DIR"/[0-9]* 2>/dev/null | sort -V | tail -1)
fi

CKPT_STEP_NUM=$(basename "$CKPT_PATH")
OUTPUT_DIR="$SCRIPT_DIR/eval_results/qwen1.5-1.8B-to-gpt2-120M/mta/$CKPT_STEP_NUM"
mkdir -p "$OUTPUT_DIR"

echo "Using checkpoint: $CKPT_PATH"
echo "Output dir: $OUTPUT_DIR"

python eval_generate.py \
  --model_path "$CKPT_PATH" \
  --tokenizer_name openai-community/gpt2 \
  --mta_data_dir "$SCRIPT_DIR/../MTA/data" \
  --output_dir "$OUTPUT_DIR" \
  --max_new_tokens 128 \
  --batch_size 16 \
  >> "$OUTPUT_DIR/eval.log" 2>&1
