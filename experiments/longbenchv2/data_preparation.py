#!/usr/bin/env python3
"""Data preparation script for LongBench-v2 benchmark tasks."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare LongBench-v2 tasks for MoTTT evaluation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="THUDM/LongBench-v2",
        help="Hugging Face repo for LongBench-v2",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Chunk size K for MoTTT test-time inner loop (default: 256)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/longbenchv2/data",
        help="Output directory",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate synthetic mock LongBench-v2 tasks offline",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LongBench-v2 Data Preparation Pipeline")
    print("=" * 70)
    print(f"Output Directory: {out_dir}")
    print(f"Chunk Size (K):   {args.chunk_size}")
    print(f"Mock Mode:        {args.mock}")
    print("=" * 70)

    records: List[Dict] = []
    if args.mock:
        print("Generating mock LongBench-v2 task records...")
        for i in range(args.max_samples or 5):
            records.append({
                "id": f"longbenchv2_mock_{i}",
                "task_category": "multi_doc_qa",
                "context": f"Document {i}: Extensive background text spanning multiple sections...",
                "query": f"Based on Document {i}, what is the critical finding?",
                "answers": ["Key finding"],
                "context_token_length": 8192,
                "num_chunks": 8192 // args.chunk_size,
            })
    else:
        try:
            from datasets import load_dataset
            print(f"Loading {args.dataset_name} from Hugging Face...")
            ds = load_dataset(args.dataset_name, split=args.split)
            if args.max_samples:
                ds = ds.select(range(min(len(ds), args.max_samples)))
            for i, item in enumerate(ds):
                records.append({
                    "id": item.get("id", f"lb_{i}"),
                    "context": item.get("context", ""),
                    "query": item.get("input", item.get("question", "")),
                    "answers": item.get("answers", [item.get("answer", "")]),
                    "task_category": item.get("dataset", "general"),
                })
        except Exception as e:
            print(f"Warning: Failed to load from Hugging Face ({e}). Creating mock template records.")
            for i in range(5):
                records.append({
                    "id": f"longbenchv2_mock_{i}",
                    "context": "Sample long context...",
                    "query": "Sample question?",
                    "answers": ["Sample answer"],
                })

    out_file = out_dir / "longbenchv2_chunked.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} LongBench-v2 records to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
