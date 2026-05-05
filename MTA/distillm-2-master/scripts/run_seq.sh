echo "start"

bash distillm-2-master/scripts/gpt2/distillm_2_gpt2_0.1b.sh > distillm-2-master/outputs/gpt2-0.1b-distillm2-train.log 2>&1
echo "done gpt2 distillm"


bash distillm-2-master/scripts/gpt2/span_distillm_2_gpt2_0.1b.sh > distillm-2-master/outputs/span-gpt2-0.1b-distillm2-train.log 2>&1
echo "done gpt2 span distillm"
bash ./scripts/distillm2/eval_span_gpt2_0.1B.sh


bash distillm-2-master/scripts/qwen1.5/distillm_2_qwen1.5_0.5b.sh > distillm-2-master/outputs/qwen1.5-0.5b-distillm2-train.log 2>&1
echo "done qwen1.5 distillm"

bash distillm-2-master/scripts/qwen1.5/span_distillm_2_qwen1.5_0.5b.sh > distillm-2-master/outputs/span-qwen1.5-0.5b-distillm2-train.log 2>&1
echo "done qwen1.5 span distillm"
bash ./scripts/distillm2/eval_span_qwen1.5_0.5B.sh

# bash ./scripts/gen/eval_opt_1.3b.sh
# bash ./scripts/gen/eval_opt_6.7b.sh

bash ./distillm-2-master/scripts/opt/reformat_data.sh

bash distillm-2-master/scripts/opt/distillm_2_opt_1.3b.sh > distillm-2-master/outputs/distillm_2_opt_1.3b.log 2>&1
echo "done opt distillm"
bash ./scripts/distillm2/eval_opt_1.3B.sh

bash distillm-2-master/scripts/opt/span_distillm_2_opt_1.3b.sh > distillm-2-master/outputs/span_distillm_2_opt_1.3b.log 2>&1
echo "done opt span distillm"
bash ./scripts/distillm2/eval_span_opt_1.3B.sh


echo "done"