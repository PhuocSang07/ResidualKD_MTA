"""
Evaluation script for Multi-Level-OT distilled models.
Evaluates on 4 benchmarks (dolly, sni, self-instruct, vicuna) matching MTA's eval setup.
Computes ROUGE-L averaged over seeds [30, 40, 50], plus ROUGE-1/2/Lsum on dolly.

Usage:
    python eval_generate.py \
        --model_path output/qwen1.5-1.8B-to-gpt2-120M/500 \
        --tokenizer_name openai-community/gpt2 \
        --mta_data_dir ../MTA/data \
        --output_dir eval_results/qwen1.5-1.8B-to-gpt2-120M \
        --max_new_tokens 128 \
        --batch_size 8
"""

import os
import json
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from rouge_score import rouge_scorer as rs_lib
import evaluate as hf_evaluate


BENCHMARKS = {
    "dolly":         "dolly/valid.jsonl",
    "sni":           "sinst/11_/valid.jsonl",
    "self_instruct": "self-inst/valid.jsonl",
    "vicuna":        "vicuna/valid.jsonl",
}

SEEDS = [30, 40, 50]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Full model checkpoint dir (full fine-tune) OR base HF model name (LoRA)")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Tokenizer name/path. Defaults to --model_path if omitted")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter directory. When set, --model_path must be the base HF model name")
    parser.add_argument("--mta_data_dir", type=str, required=True,
                        help="Path to MTA/data directory (contains dolly/, sinst/, self-inst/, vicuna/)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Limit samples per benchmark (-1 = all)")
    # kept for backward compat
    parser.add_argument("--data_path", type=str, default=None)
    return parser.parse_args()


def load_records(data_path, max_samples=-1):
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_samples > 0:
        records = records[:max_samples]
    return records


def batch_generate(model, tokenizer, prompts, max_new_tokens, device):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def evaluate_benchmark(model, tokenizer, data_path, args, device):
    """
    Evaluate on a single benchmark.
    Returns:
        rouge_l_avg: ROUGE-L F1 averaged over SEEDS (matches MTA metric)
        rouge_full:  ROUGE-1/2/L/Lsum from hf evaluate on seed=30 run (extra info)
        predictions, references from seed=30
    """
    records = load_records(data_path, args.max_samples)
    prompts    = [r["prompt"] for r in records]
    references = [r["output"] if isinstance(r["output"], str)
                  else r["output"][0] for r in records]

    scorer = rs_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_per_seed = []
    first_seed_preds = None

    for seed in SEEDS:
        set_seed(seed)
        preds = []
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"seed={seed}", leave=False):
            batch_preds = batch_generate(
                model, tokenizer,
                prompts[i: i + args.batch_size],
                args.max_new_tokens, device,
            )
            preds.extend(batch_preds)

        if first_seed_preds is None:
            first_seed_preds = preds

        scores = []
        for pred, ref in zip(preds, references):
            ref = ref.replace("<pad>", "").replace("<|endoftext|>", "").strip()
            if pred and ref:
                scores.append(scorer.score(pred, ref)["rougeL"].fmeasure)
        rouge_l = (sum(scores) / len(scores) * 100) if scores else 0.0
        rouge_l_per_seed.append(rouge_l)
        print(f"  seed={seed}  ROUGE-L: {rouge_l:.2f}")

    rouge_l_avg = sum(rouge_l_per_seed) / len(rouge_l_per_seed)

    # Full ROUGE on first-seed predictions (bonus info)
    rouge_full = {}
    try:
        hf_rouge = hf_evaluate.load("rouge")
        rouge_full = hf_rouge.compute(
            predictions=first_seed_preds,
            references=references,
            use_aggregator=True,
        )
        rouge_full = {k: round(v * 100, 4) for k, v in rouge_full.items()}
    except Exception:
        pass

    return rouge_l_avg, rouge_full, first_seed_preds, references


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_src = args.tokenizer_name or args.model_path
    print(f"Loading tokenizer: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    if args.lora_path is not None:
        print(f"Merging LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
        print("LoRA merged.")

    model.eval()

    all_scores = {"model_path": args.model_path}

    for key, rel_path in BENCHMARKS.items():
        data_path = os.path.join(args.mta_data_dir, rel_path)
        if not os.path.exists(data_path):
            print(f"[SKIP] {key}: {data_path} not found")
            all_scores[key] = {"status": "not_found"}
            continue

        print(f"\n=== {key.upper()} ({data_path}) ===")
        rouge_l_avg, rouge_full, preds, refs = evaluate_benchmark(
            model, tokenizer, data_path, args, device)

        print(f"  → ROUGE-L avg (seeds {SEEDS}): {rouge_l_avg:.2f}")

        # Save predictions
        pred_path = os.path.join(args.output_dir, f"{key}_predictions.jsonl")
        with open(pred_path, "w") as f:
            for rec, pred in zip(load_records(data_path, args.max_samples), preds):
                f.write(json.dumps({
                    "prompt":     rec.get("instruction", rec.get("prompt", "")),
                    "reference":  rec["output"] if isinstance(rec["output"], str) else rec["output"][0],
                    "prediction": pred,
                }, ensure_ascii=False) + "\n")

        all_scores[key] = {
            "rouge_l_avg": round(rouge_l_avg, 4),
            **rouge_full,
        }

    # Save aggregated scores
    score_path = os.path.join(args.output_dir, "scores.json")
    with open(score_path, "w") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)

    print("\n=== Final Scores ===")
    for key, val in all_scores.items():
        if key == "model_path":
            continue
        if isinstance(val, dict) and "rouge_l_avg" in val:
            print(f"  {key:15s}  ROUGE-L: {val['rouge_l_avg']:.2f}")
    print(f"\nScores saved to {score_path}")


if __name__ == "__main__":
    main()
