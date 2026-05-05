#!/bin/bash
# Eval: Qwen2.5-7B-Instruct → GPT-2 XL 1.5B

export CUDA_VISIBLE_DEVICES=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="openai-community/gpt2-xl"
RUN_DIR="$SCRIPT_DIR/output/qwen2.5-7B-to-gpt2-1.5B"

CKPT_STEP=""

if [ -n "$CKPT_STEP" ]; then
    CKPT_PATH="$RUN_DIR/$CKPT_STEP"
else
    CKPT_PATH=$(ls -d "$RUN_DIR"/[0-9]* 2>/dev/null | sort -V | tail -1)
fi

CKPT_STEP_NUM=$(basename "$CKPT_PATH")
OUTPUT_DIR="$SCRIPT_DIR/eval_results/qwen2.5-7B-to-gpt2-1.5B/$CKPT_STEP_NUM"
mkdir -p "$OUTPUT_DIR"

echo "Using checkpoint: $CKPT_PATH"
echo "Output dir: $OUTPUT_DIR"

# Auto-detect: LoRA checkpoint chỉ chứa adapter weights, không có full model
if [ -f "$CKPT_PATH/adapter_config.json" ]; then
    echo "Detected LoRA checkpoint → loading base model + adapter"
    MODEL_ARGS="--model_path $BASE_MODEL --lora_path $CKPT_PATH"
else
    echo "Detected full model checkpoint"
    MODEL_ARGS="--model_path $CKPT_PATH"
fi

python eval_generate.py \
  $MODEL_ARGS \
  --tokenizer_name "$BASE_MODEL" \
  --mta_data_dir "$SCRIPT_DIR/../MTA/data" \
  --output_dir "$OUTPUT_DIR" \
  --max_new_tokens 128 \
  --batch_size 4 \
  >> "$OUTPUT_DIR/eval.log" 2>&1
