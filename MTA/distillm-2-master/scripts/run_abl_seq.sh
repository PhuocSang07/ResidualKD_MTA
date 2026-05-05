echo "start"

bash distillm-2-master/scripts/ablation_gpt2/dsa.sh > distillm-2-master/outputs/ablation-gpt2-dsa.log 2>&1
echo "done gpt2 dsa"

bash distillm-2-master/scripts/ablation_gpt2/hs.sh > distillm-2-master/outputs/ablation-gpt2-hs.log 2>&1
echo "done gpt2 hs"

bash distillm-2-master/scripts/ablation_gpt2/phrases.sh > distillm-2-master/outputs/ablation-gpt2-phrases.log 2>&1
echo "done gpt2 phrases"

bash distillm-2-master/scripts/ablation_gpt2/word.sh > distillm-2-master/outputs/ablation-gpt2-word.log 2>&1
echo "done gpt2 word"

# bash ./scripts/distillm2/eval_opt_1.3B.sh

echo "start eval"
bash ./scripts/ablation/eval_span2_gpt2_dsa.sh
bash ./scripts/ablation/eval_span2_gpt2_hs.sh
bash ./scripts/ablation/eval_span2_gpt2_phrases.sh
bash ./scripts/ablation/eval_span2_gpt2_word.sh

bash ./scripts/ablation/eval_span2_gpt2_dsa.sh
bash ./scripts/ablation/eval_span2_gpt2_hs.sh
bash ./scripts/ablation/eval_span2_gpt2_dsa.sh

bash scripts/gen/fdd/eval_fdd_qwen1.5_0.5B.sh
bash scripts/gen/fdd/eval_spanfdd_qwen1.5_0.5B.sh


echo "done"