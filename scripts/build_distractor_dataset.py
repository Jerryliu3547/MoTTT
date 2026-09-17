#!/usr/bin/env python3
"""CLI script to build and export distractor-injected GSM8K datasets for MoTTT and external baseline models."""

import argparse
import sys
from pathlib import Path
from typing import List

from mottt.data.gsm8k_loader import load_gsm8k_dataset
from mottt.data.distractor_generator import (
    PremiseQueryDecomposer,
    DistractorNeedleSynthesizer,
    LongContextGSM8KRecord,
)
from mottt.data.dataset_exporter import export_dataset_bundle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build distractor-injected long-context GSM8K benchmark dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="GSM8K split to process (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of GSM8K examples to process (default: all)",
    )
    parser.add_argument(
        "--depth_ratios",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated needle depth ratios delta in [0.1, 0.9] (default: 0.1,0.3,0.5,0.7,0.9)",
    )
    parser.add_argument(
        "--target_context_tokens",
        type=int,
        default=4096,
        help="Target context length L_ctx in tokens (default: 4096)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Chunk token size K for test-time inner loop (default: 256)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/gsm8k_distractor",
        help="Output directory to save dataset copies (default: data/gsm8k_distractor)",
    )
    parser.add_argument(
        "--no_hf_dataset",
        action="store_true",
        help="Disable saving as Hugging Face Dataset format",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    depth_ratios: List[float] = [float(d.strip()) for d in args.depth_ratios.split(",")]

    print("=" * 70)
    print("MoTTT: Distractor-Injected Long-Context GSM8K Generator")
    print("=" * 70)
    print(f"Split:                {args.split}")
    print(f"Max Samples:          {args.max_samples if args.max_samples is not None else 'All'}")
    print(f"Depth Ratios:         {depth_ratios}")
    print(f"Context Tokens:       {args.target_context_tokens}")
    print(f"Output Directory:     {args.output_dir}")
    print("=" * 70)

    # 1. Load GSM8K from Hugging Face
    print("\n[1/3] Loading GSM8K from Hugging Face (openai/gsm8k)...")
    try:
        examples = load_gsm8k_dataset(split=args.split, max_samples=args.max_samples)
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(examples)} examples.")

    # 2. Decompose and synthesize needle distractors
    print("\n[2/3] Decomposing premises and embedding needles into long contexts...")
    synthesizer = DistractorNeedleSynthesizer(chunk_size=args.chunk_size)
    all_records: List[LongContextGSM8KRecord] = []

    for i, ex in enumerate(examples):
        decomposed = PremiseQueryDecomposer.decompose(
            problem_text=ex.question,
            solution_text=ex.solution,
            gold_answer=ex.gold_answer,
        )

        for depth in depth_ratios:
            rec = synthesizer.create_record(
                example_id=ex.example_id,
                original_question=ex.question,
                premise=decomposed.premise,
                query=decomposed.query,
                solution=ex.solution,
                gold_answer=ex.gold_answer,
                depth_ratio=depth,
                target_token_count=args.target_context_tokens,
            )
            all_records.append(rec)

        if (i + 1) % 50 == 0 or (i + 1) == len(examples):
            print(f"  Processed {i + 1}/{len(examples)} problems ({len(all_records)} total records)")

    # 3. Export dataset bundle for MoTTT and external baselines
    print(f"\n[3/3] Exporting dataset bundle to {args.output_dir}...")
    manifest = export_dataset_bundle(
        records=all_records,
        output_dir=args.output_dir,
        save_hf_dataset=not args.no_hf_dataset,
    )

    print("\n" + "=" * 70)
    print("Dataset generation COMPLETE!")
    print(f"Total exported records: {manifest['total_records']}")
    print(f"Combined JSONL file:    {args.output_dir}/{manifest['combined_file']}")
    print("Per-depth files:")
    for depth_str, info in manifest["records_per_depth"].items():
        print(f"  - Depth {depth_str}: {args.output_dir}/{info['file']} ({info['count']} samples)")
    if manifest.get("hf_dataset_directory"):
        print(f"HuggingFace Dataset:    {args.output_dir}/{manifest['hf_dataset_directory']}")
    print(f"Dataset Manifest:       {args.output_dir}/dataset_manifest.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
