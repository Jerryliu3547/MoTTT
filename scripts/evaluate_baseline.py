#!/usr/bin/env python3
"""Evaluate an external baseline model (e.g. Qwen2.5, LLaMA, Mistral) on the modified distractor GSM8K benchmark."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src/ is on sys.path even if not installed via pip install -e .
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mottt.data.dataset_exporter import load_distractor_jsonl


def extract_predicted_answer(generated_text: str) -> str:
    """Extract numeric answer from model generation."""
    # Priority 1: Match '#### [answer]' pattern
    if "####" in generated_text:
        ans = generated_text.split("####")[-1].strip()
        ans = re.sub(r"[,$]", "", ans).split()[0].strip()
        return ans

    # Priority 2: 'The answer is [number]'
    match = re.search(r"(?:the\s+answer\s+is\s+|is\s+|equal\s+to\s+)([-+]?\d+(?:\.\d+)?)", generated_text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "").strip()

    # Priority 3: Last number in the text
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", generated_text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate external baseline LLM on distractor-injected GSM8K dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to distractor JSONL file (e.g. data/gsm8k_distractor/gsm8k_distractor_all.jsonl)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face model identifier or local path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (simulates model inference without GPU/model download)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate per answer (default: 512)",
    )
    parser.add_argument(
        "--output_results",
        type=str,
        default=None,
        help="Optional path to save JSON evaluation results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("MoTTT Baseline Model Evaluator")
    print("=" * 70)
    print(f"Dataset:    {args.dataset_path}")
    print(f"Model:      {args.model_name_or_path} {'[MOCK MODE]' if args.mock else ''}")
    print("=" * 70)

    # 1. Load data
    records = load_distractor_jsonl(args.dataset_path)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"Loaded {len(records)} evaluation records.")

    # 2. Setup model
    pipeline = None
    if not args.mock:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
            print(f"Loading model {args.model_name_or_path}...")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.max_length = None
                model.generation_config.max_new_tokens = args.max_new_tokens
            pipeline = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            print(f"Failed to load model {args.model_name_or_path}: {e}", file=sys.stderr)
            print("To test the evaluation pipeline without downloading models, run with --mock flag.")
            sys.exit(1)

    # 3. Evaluation loop
    depth_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total_correct = 0
    detailed_results = []

    print("\nRunning evaluation...")
    for idx, rec in enumerate(records):
        prompt = rec["full_prompt"]
        gold_ans = str(rec["gold_answer"]).strip()
        depth = round(rec["needle_depth_ratio"], 2)

        if args.mock:
            # Mock prediction simulation: 50% accurate for demonstration
            pred_ans = gold_ans if (idx % 2 == 0) else "0"
            gen_text = f"Step 1: ... #### {pred_ans}"
        else:
            outputs = pipeline(prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen_text = outputs[0]["generated_text"][len(prompt):]
            pred_ans = extract_predicted_answer(gen_text)

        is_correct = (pred_ans == gold_ans)
        if is_correct:
            total_correct += 1
        depth_stats[depth]["total"] += 1
        if is_correct:
            depth_stats[depth]["correct"] += 1

        detailed_results.append({
            "id": rec["id"],
            "depth": depth,
            "gold_answer": gold_ans,
            "pred_answer": pred_ans,
            "correct": is_correct,
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            print(f"  Processed {idx + 1}/{len(records)} | Overall Accuracy: {total_correct / (idx + 1) * 100:.1f}%")

    # 4. Summary report
    overall_acc = (total_correct / len(records) * 100) if records else 0.0
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated:  {len(records)}")
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{len(records)})")
    print("\nAccuracy Breakdown by Needle Depth Ratio (Robustness against Lost-in-the-Middle):")
    for depth in sorted(depth_stats.keys()):
        stat = depth_stats[depth]
        d_acc = (stat["correct"] / stat["total"] * 100) if stat["total"] > 0 else 0.0
        bar = "█" * int(d_acc / 5)
        print(f"  Depth {depth:.2f}: {d_acc:5.1f}% ({stat['correct']}/{stat['total']}) | {bar}")
    print("=" * 70)

    if args.output_results:
        out_p = Path(args.output_results)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model_name_or_path,
                "dataset": args.dataset_path,
                "total_records": len(records),
                "overall_accuracy": overall_acc,
                "depth_accuracy": {str(k): (v["correct"] / v["total"]) for k, v in depth_stats.items()},
                "results": detailed_results,
            }, f, indent=2)
        print(f"Detailed results written to: {args.output_results}")


if __name__ == "__main__":
    main()
