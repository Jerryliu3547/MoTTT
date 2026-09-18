#!/usr/bin/env python3
"""Data preparation script for GSM8K distractor-injected long-context experiment."""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# Ensure src/ is on sys.path even if not installed via pip install -e .
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mottt.data.gsm8k_loader import GSM8KExample, load_gsm8k_dataset
from mottt.data.distractor_generator import (
    PremiseQueryDecomposer,
    DistractorNeedleSynthesizer,
    LongContextGSM8KRecord,
)
from mottt.data.dataset_exporter import export_records_to_jsonl, export_dataset_bundle



MOCK_GSM8K_SAMPLES = [
    {
        "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "solution": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18",
        "gold_answer": "18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "solution": "It takes 2 / 2 = <<2/2=1>>1 bolt of white fiber.\nIn total it takes 2 + 1 = <<2+1=3>>3 bolts.\n#### 3",
        "gold_answer": "3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. He sells the house for $150,000. How much profit did he make?",
        "solution": "Total cost is 80000 + 50000 = <<80000+50000=130000>>130000.\nProfit is 150000 - 130000 = <<150000-130000=20000>>20000.\n#### 20000",
        "gold_answer": "20000",
    },
    {
        "question": "James decides to run 3 miles every day for 2 weeks. How many miles does he run altogether?",
        "solution": "2 weeks has 2 * 7 = <<2*7=14>>14 days.\nJames runs 14 * 3 = <<14*3=42>>42 miles.\n#### 42",
        "gold_answer": "42",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare distractor-injected long-context GSM8K datasets for MoTTT"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Maximum training samples to process (default: all)",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=None,
        help="Maximum test samples to process (default: all)",
    )
    parser.add_argument(
        "--depth_ratios",
        type=str,
        default="0.1,0.3,0.5,0.7,0.9",
        help="Comma-separated needle depth ratios in [0.1, 0.9]",
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
        default="experiments/gsm8k/data",
        help="Output directory to save processed datasets",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock GSM8K examples offline without calling Hugging Face",
    )
    parser.add_argument(
        "--random_train_depth",
        action="store_true",
        help="Randomly select a single depth ratio per training example instead of duplicating across all ratios",
    )
    parser.add_argument(
        "--random_mode",
        type=str,
        default="choice",
        choices=["choice", "uniform"],
        help="Random depth mode: 'choice' selects randomly from --depth_ratios, 'uniform' samples uniformly in [0.1, 0.9] (default: choice)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible depth selection (default: 42)",
    )
    return parser.parse_args()


def process_split(
    examples: List[GSM8KExample],
    split_name: str,
    depth_ratios: List[float],
    target_context_tokens: int,
    chunk_size: int,
    random_depth: bool = False,
    random_mode: str = "choice",
    depth_range: Tuple[float, float] = (0.1, 0.9),
) -> List[LongContextGSM8KRecord]:
    synthesizer = DistractorNeedleSynthesizer(chunk_size=chunk_size)
    records: List[LongContextGSM8KRecord] = []

    pbar = tqdm(examples, desc=f"Synthesizing {split_name} split", unit="problem")
    for ex in pbar:
        decomposed = PremiseQueryDecomposer.decompose(
            problem_text=ex.question,
            solution_text=ex.solution,
            gold_answer=ex.gold_answer,
        )

        if random_depth:
            if random_mode == "uniform":
                chosen_depths = [round(random.uniform(depth_range[0], depth_range[1]), 2)]
            else:
                chosen_depths = [random.choice(depth_ratios)]
        else:
            chosen_depths = depth_ratios

        for depth in chosen_depths:
            rec = synthesizer.create_record(
                example_id=ex.example_id,
                original_question=ex.question,
                premise=decomposed.premise,
                query=decomposed.query,
                solution=ex.solution,
                gold_answer=ex.gold_answer,
                depth_ratio=depth,
                target_token_count=target_context_tokens,
            )
            records.append(rec)

    return records


def main():
    args = parse_args()
    random.seed(args.seed)
    depth_ratios: List[float] = [float(d.strip()) for d in args.depth_ratios.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GSM8K Experiment: Data Preparation Pipeline")
    print("=" * 70)
    print(f"Output Directory:     {out_dir}")
    print(f"Target Context:       {args.target_context_tokens} tokens")
    print(f"Chunk Size (K):       {args.chunk_size}")
    print(f"Depth Ratios:         {depth_ratios}")
    print(f"Random Train Depth:   {args.random_train_depth} (mode: {args.random_mode}, seed: {args.seed})")
    print(f"Mock Mode:            {args.mock}")
    print("=" * 70)

    # 1. Load data
    if args.mock:
        print("\nLoading mock GSM8K dataset offline...")
        train_examples = [
            GSM8KExample(
                example_id=f"gsm8k_train_{i}",
                question=s["question"],
                solution=s["solution"],
                gold_answer=s["gold_answer"],
            )
            for i, s in enumerate(MOCK_GSM8K_SAMPLES[:2])
        ]
        test_examples = [
            GSM8KExample(
                example_id=f"gsm8k_test_{i}",
                question=s["question"],
                solution=s["solution"],
                gold_answer=s["gold_answer"],
            )
            for i, s in enumerate(MOCK_GSM8K_SAMPLES[2:])
        ]
    else:
        print("\n[1/2] Loading GSM8K train split from Hugging Face...")
        train_examples = load_gsm8k_dataset(split="train", max_samples=args.train_samples)
        print(f"Loaded {len(train_examples)} train examples.")

        print("\n[2/2] Loading GSM8K test split from Hugging Face...")
        test_examples = load_gsm8k_dataset(split="test", max_samples=args.test_samples)
        print(f"Loaded {len(test_examples)} test examples.")

    # 2. Process and synthesize distractor contexts
    print("\nSynthesizing distractor-injected training set...")
    train_records = process_split(
        train_examples,
        split_name="train",
        depth_ratios=depth_ratios,
        target_context_tokens=args.target_context_tokens,
        chunk_size=args.chunk_size,
        random_depth=args.random_train_depth,
        random_mode=args.random_mode,
    )
    train_file = out_dir / "train_distractor.jsonl"
    export_records_to_jsonl(train_records, str(train_file))
    print(f"Saved {len(train_records)} training records to {train_file}")

    print("\nSynthesizing distractor-injected test set...")
    test_records = process_split(
        test_examples,
        split_name="test",
        depth_ratios=depth_ratios,
        target_context_tokens=args.target_context_tokens,
        chunk_size=args.chunk_size,
    )

    # Export test bundle (combined, per-depth, and manifest)
    manifest = export_dataset_bundle(
        records=test_records,
        output_dir=str(out_dir / "test_bundle"),
        save_hf_dataset=False,
    )
    test_combined_file = out_dir / "test_distractor.jsonl"
    export_records_to_jsonl(test_records, str(test_combined_file))
    print(f"Saved {len(test_records)} test records to {test_combined_file}")

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print(f"Train File:           {train_file}")
    print(f"Test File:            {test_combined_file}")
    print(f"Per-Depth Directory:  {out_dir}/test_bundle")
    print("=" * 70)


if __name__ == "__main__":
    main()
