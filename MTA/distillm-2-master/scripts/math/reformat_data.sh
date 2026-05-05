
TEACHER_DIR=data/dpo/Qwen/Qwen2.5-Math-1.5B
STUDENT_DIR=data/dpo/Qwen/Qwen2.5-0.5B
OUTPUT_DIR=distillm-2-master/data/reformatted/qwen_math

python distillm-2-master/generate/reformat_math.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR