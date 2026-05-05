"""
Aggregate and compare SpanResidual eval results.

Reads scores.json from eval_results/spanresidual/ and prints a Rouge-L comparison table.

Usage:
    # From MTA/ directory:
    python distillm-master/scripts/gpt2/spanresidual/eval-aggregate.py

    # Or specify a custom eval_results root:
    python distillm-master/scripts/gpt2/spanresidual/eval-aggregate.py \
        --eval_dir /path/to/Multi-Level-OT/eval_results/spanresidual
"""
import os
import json
import argparse
from pathlib import Path


BENCHMARKS = ["dolly", "sni", "self_instruct", "vicuna"]

# Pretty display order — baseline first, then MTA
RUN_ORDER = [
    "spanresidual_baseline_A_0.1B_qwen1.8B",
    "spanresidual_setup_A_0.1B_qwen1.8B",
    "spanresidual_baseline_B_0.35B_qwen1.8B",
    "spanresidual_setup_B_0.35B_qwen1.8B",
]

LABELS = {
    "spanresidual_baseline_A_0.1B_qwen1.8B": "Baseline (γ=0) | GPT2-120M",
    "spanresidual_setup_A_0.1B_qwen1.8B":    "SpanResidual  | GPT2-120M",
    "spanresidual_baseline_B_0.35B_qwen1.8B": "Baseline (γ=0) | GPT2-345M",
    "spanresidual_setup_B_0.35B_qwen1.8B":    "SpanResidual  | GPT2-345M",
}


def find_scores(eval_dir: Path) -> dict:
    """Walk eval_dir, collect all scores.json → {run_name: scores}."""
    results = {}
    if not eval_dir.exists():
        return results
    for run_dir in sorted(eval_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        # Each run may have multiple checkpoint subdirs — pick latest
        ckpt_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if not ckpt_dirs:
            # Flat structure: scores.json directly in run_dir
            score_file = run_dir / "scores.json"
        else:
            score_file = ckpt_dirs[-1] / "scores.json"

        if score_file.exists():
            with open(score_file) as f:
                data = json.load(f)
            results[run_dir.name] = data
    return results


def print_table(results: dict):
    col_w = 32
    bench_w = 10

    # Header
    header = f"{'Model':<{col_w}}" + "".join(f"{b:>{bench_w}}" for b in BENCHMARKS) + f"{'AVG':>{bench_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Print known runs in order, then any extras
    shown = set()
    ordered = RUN_ORDER + [k for k in results if k not in RUN_ORDER]

    for run_name in ordered:
        if run_name not in results:
            continue
        shown.add(run_name)
        data = results[run_name]
        label = LABELS.get(run_name, run_name)

        scores = []
        row = f"{label:<{col_w}}"
        for b in BENCHMARKS:
            if b in data and "rouge_l_avg" in data[b]:
                v = data[b]["rouge_l_avg"]
                scores.append(v)
                row += f"{v:>{bench_w}.2f}"
            else:
                row += f"{'—':>{bench_w}}"

        avg = sum(scores) / len(scores) if scores else 0.0
        row += f"{avg:>{bench_w}.2f}"
        print(row)

    print("=" * len(header))
    print(f"Benchmarks: dolly / sni / self_instruct / vicuna  |  ROUGE-L avg over seeds [30,40,50]")

    # Delta rows
    baseline_A = results.get("spanresidual_baseline_A_0.1B_qwen1.8B")
    mta_A      = results.get("spanresidual_setup_A_0.1B_qwen1.8B")
    if baseline_A and mta_A:
        print("\n  Δ MTA vs Baseline (GPT2-120M):")
        deltas = []
        for b in BENCHMARKS:
            if b in baseline_A and b in mta_A and "rouge_l_avg" in baseline_A[b] and "rouge_l_avg" in mta_A[b]:
                d = mta_A[b]["rouge_l_avg"] - baseline_A[b]["rouge_l_avg"]
                deltas.append(d)
                sign = "+" if d >= 0 else ""
                print(f"    {b:15s}: {sign}{d:.2f}")
        if deltas:
            avg_d = sum(deltas) / len(deltas)
            sign = "+" if avg_d >= 0 else ""
            print(f"    {'AVG':15s}: {sign}{avg_d:.2f}")

    baseline_B = results.get("spanresidual_baseline_B_0.35B_qwen1.8B")
    mta_B      = results.get("spanresidual_setup_B_0.35B_qwen1.8B")
    if baseline_B and mta_B:
        print("\n  Δ MTA vs Baseline (GPT2-345M):")
        deltas = []
        for b in BENCHMARKS:
            if b in baseline_B and b in mta_B and "rouge_l_avg" in baseline_B[b] and "rouge_l_avg" in mta_B[b]:
                d = mta_B[b]["rouge_l_avg"] - baseline_B[b]["rouge_l_avg"]
                deltas.append(d)
                sign = "+" if d >= 0 else ""
                print(f"    {b:15s}: {sign}{d:.2f}")
        if deltas:
            avg_d = sum(deltas) / len(deltas)
            sign = "+" if avg_d >= 0 else ""
            print(f"    {'AVG':15s}: {sign}{avg_d:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        default=None,
        help="Path to eval_results/spanresidual/ dir. "
             "Defaults to Multi-Level-OT/eval_results/spanresidual/ relative to this script.",
    )
    args = parser.parse_args()

    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
    else:
        # Default: relative to this script → ../../../../Multi-Level-OT/eval_results/spanresidual
        script_dir = Path(__file__).resolve().parent
        eval_dir = script_dir.parents[3] / "Multi-Level-OT" / "eval_results" / "spanresidual"

    print(f"Scanning: {eval_dir}")
    results = find_scores(eval_dir)

    if not results:
        print("No scores.json found. Run eval-setup-A-0.1B.sh first.")
        return

    print_table(results)


if __name__ == "__main__":
    main()
