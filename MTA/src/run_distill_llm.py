import argparse
from arguments import Arguments
from teacher_llm import TeacherQwen, TeacherMistral7B, TeacherGPT2
from student import LLMModel
from types import SimpleNamespace

from evaluator import Evaluator
from llm_train import Trainer, train

from transformers import HfArgumentParser
from huggingface_hub import login

import torch
import json
import numpy as np
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    hf_parser = HfArgumentParser(Arguments)
    args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args: Arguments = args
    args.knowledge_distillation = True
    args.weight_decay = 0.01
    args.warmup_ratio = 0.0
    args.finetune_embedding = True

    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--seed", type=int, default=42)
    extra_parser.add_argument("--teacher_sft", type=str, default=None)
    extra_parser.add_argument("--student_sft", type=str, default=None)
    extra_parser.add_argument("--model_type", type=str, default=None)
    extra_parser.add_argument("--use_lora", type=bool, default=False)
    extra_parser.add_argument("--grad_accum_steps", type=int, default=1)

    extras = extra_parser.parse_args(remaining)

    set_seed(extras.seed)
    if args.hf_token:
        login(args.hf_token)

    if extras.model_type == 'qwen':
        TeacherLLM = TeacherQwen
    elif extras.model_type == 'gpt2':
        TeacherLLM = TeacherGPT2
    elif extras.model_type == 'mistral':
        TeacherLLM = TeacherMistral7B
    else:
        raise ValueError(f'unknown teacher model type: {extras.model_type}')

    load_model_kwargs = {
        'torch_dtype': torch.float16,
        'quantization_config': None,
        'device_map': args.teach_device,
        'trust_remote_code': True,
        'output_hidden_states': args.finetune_hidden_states,
        'output_attentions': args.output_attentions,
        'attn_implementation': 'sdpa',
        'token': args.hf_token or None,
    }

    teacher_model = TeacherLLM(model_name=args.teacher_model,
                               load_model_kwargs=load_model_kwargs,
                               sft_path=extras.teacher_sft)

    load_student_model_kwargs = {
        'device_map': args.student_device,
        'output_hidden_states': args.finetune_hidden_states,
        'output_attentions': args.output_attentions,
        'attn_implementation': 'eager' if args.output_attentions else 'sdpa',
    }

    if extras.use_lora:
        lora_config = SimpleNamespace(lora_rank=256, lora_alpha=8, lora_dropout=0.1)
    else:
        lora_config = None

    llm_model = LLMModel(model_name=args.student_model,
                         load_model_kwargs=load_student_model_kwargs,
                         lora_conf=lora_config,
                         sft_path=extras.student_sft)

    trainer = Trainer(llm_model, extras.model_type, args=args, teacher_model=teacher_model)

    evaluator = Evaluator(tokenizer_path=args.student_tokenizer,
                          model_path=None, sft_lora=None, distilled_lora=None,
                          device=llm_model.device, seeds=[10])

    train(args, trainer, evaluator, grad_accum_steps=extras.grad_accum_steps)

    if extras.use_lora:
        evaluator = Evaluator(
            tokenizer_path=args.student_tokenizer,
            model_path=args.student_model,
            distilled_lora=args.output_dir,
            seeds=[10, 20, 30, 40, 50]
        )
    else:
        evaluator = Evaluator(
            tokenizer_path=args.student_tokenizer,
            model_path=args.output_dir,
            seeds=[10, 20, 30, 40, 50]
        )

    benchmark_configs = {
        'dolly': './data/dolly/valid.jsonl',
        'self_instruct': './data/self-inst/valid.jsonl',
        'vicuna': './data/vicuna/valid.jsonl',
        'sni': './data/sinst/11_/valid.jsonl',
    }

    results = evaluator.evaluate_multiple_benchmarks(
        benchmark_configs=benchmark_configs,
        batch_size=64,
        max_seq_length=256,
        max_new_tokens=512
    )

    with open(args.output_dir + "/eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
