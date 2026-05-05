from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-1.5B')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='generated/iter1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dir', type=str, default='UCLA-AGI/SPIN_iter0')
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]    
    for prompt_batch in batches:
        texts = tokenizer.apply_chat_template(
            prompt_batch,
            add_generation_prompt=True,
            tokenize=False 
        )

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        batches_tok.append(inputs)  # dict: input_ids + attention_mask

    return batches_tok

def main():
    args = parse_arguments()
    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # load data
    data = load_dataset('VoCuc/MetaMathQA-50k-256', split='train')['query'][:20000]

    prompts_all = [
        [{"role": "system", "content": "Put your final answer within \\boxed{}."},
         {"role": "user", "content": prompt}] for prompt in data
    ]

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()   

    # divide the prompt list onto the available GPUs
    # with accelerator.split_between_processes(prompts_all) as prompts:
    results = []
    prompt_batches=prepare_prompts(prompts_all, tokenizer, batch_size=args.batch_size)

    for prompts_tokenized in tqdm(prompt_batches):
        # set max_new_tokens smaller for faster inference
        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)

        # remove prompt from gen. tokens
        outputs_tokenized=[tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(prompts_tokenized['input_ids'], outputs_tokenized) ] 
        # decode gen. tokens 
        outputs=tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
        results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered=gather_object(results)
    results = [r.replace("</s>","").lstrip() for r in results_gathered]
    # import pdb; pdb.set_trace()

    # if accelerator.is_local_main_process:
    # timediff=time.time()-start
    # print(f"time elapsed: {timediff}")

    # collecting data
    output_data = []
    for i, generated_text in tqdm(enumerate(results)):
        output_data.append({
            'prompt': data[i],
            'generated_text': generated_text,
        })
    
    output_file = f'generated_train.jsonl'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, output_file), 'w') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")

if __name__ == "__main__":
    main()