
TEACHER_DIR=data/dpo/VoCuc/Mistral7B_Dolly_SFT
STUDENT_DIR=data/dpo/h2oai/h2o-danube2-1.8b-base
OUTPUT_DIR=distillm-2-master/data/reformatted/mistral

python distillm-2-master/generate/reformat.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR