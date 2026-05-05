
TEACHER_DIR=data/dpo/VoCuc/Qwen1.5_1.8B_SFT
STUDENT_DIR=data/dpo/Qwen/Qwen1.5-0.5B
OUTPUT_DIR=distillm-2-master/data/reformatted/qwen1.5

python distillm-2-master/generate/reformat.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR