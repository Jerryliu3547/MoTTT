#!/usr/bin/env python3
"""Compare evaluation results between MoTTT and Full Fine-Tuning (SFT) baseline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MoTTT vs Full Fine-Tuning (SFT) baseline performance on distractor GSM8K"
    )
    parser.add_argument(
        "--mottt_report",
        type=str,
        default="experiments/gsm8k/results/gsm8k_eval_report.json",
        help="Path to MoTTT evaluation report JSON",
    )
    parser.add_argument(
        "--sft_report",
        type=str,
        default="experiments/gsm8k/results/baseline_sft_eval_report.json",
        help="Path to Full Fine-Tuning (SFT) evaluation report JSON",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="experiments/gsm8k/results/comparison_summary.json",
        help="Path to save output comparison summary JSON",
    )
    return parser.parse_args()


def load_report(path_str: str) -> Optional[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return None


def main():
    args = parse_args()
    print("=" * 78)
    print("MoTTT vs. Full Fine-Tuning (SFT) Baseline Performance Comparison")
    print("=" * 78)

    mottt_data = load_report(args.mottt_report)
    sft_data = load_report(args.sft_report)

    if mottt_data is None:
        print(f"Notice: MoTTT report not found at {args.mottt_report}")
        print("Run MoTTT test first via: python experiments/gsm8k/model_test.py")
    if sft_data is None:
        print(f"Notice: SFT baseline report not found at {args.sft_report}")
        print("Run SFT test first via: python experiments/gsm8k/test_full_finetune.py")

    if mottt_data is None and sft_data is None:
        print("\nNeither report was found. Exiting.")
        sys.exit(1)

    mottt_acc = mottt_data.get("overall_accuracy") if mottt_data else None
    sft_acc = sft_data.get("overall_accuracy") if sft_data else None

    print("\n--- 1. OVERALL ACCURACY ---")
    if mottt_acc is not None:
        print(f"MoTTT Overall Accuracy:           {mottt_acc:>6.2f}%")
    else:
        print("MoTTT Overall Accuracy:           [Report Missing]")

    if sft_acc is not None:
        print(f"Full Fine-Tuning (SFT) Accuracy:  {sft_acc:>6.2f}%")
    else:
        print("Full Fine-Tuning (SFT) Accuracy:  [Report Missing]")

    if mottt_acc is not None and sft_acc is not None:
        diff = mottt_acc - sft_acc
        sign = "+" if diff >= 0 else ""
        print(f"Difference (MoTTT - SFT):         {sign}{diff:>5.2f}%")

    # Depth breakdown comparison
    mottt_depth = mottt_data.get("depth_breakdown", {}) if mottt_data else {}
    sft_depth = sft_data.get("depth_breakdown", {}) if sft_data else {}

    all_depths = sorted(list(set(list(mottt_depth.keys()) + list(sft_depth.keys()))))

    depth_comparison = []
    if all_depths:
        print("\n--- 2. NEEDLE DEPTH BREAKDOWN (Lost-in-the-Middle Analysis) ---")
        print(f"{'Needle Depth':<14} | {'MoTTT Acc':<12} | {'Full SFT Acc':<14} | {'Delta (MoTTT - SFT)':<20}")
        print("-" * 70)
        for d in all_depths:
            m_info = mottt_depth.get(d)
            s_info = sft_depth.get(d)

            m_str = f"{m_info['accuracy']:.2f}%" if m_info else "N/A"
            s_str = f"{s_info['accuracy']:.2f}%" if s_info else "N/A"

            if m_info and s_info:
                delta = m_info["accuracy"] - s_info["accuracy"]
                d_str = f"{'+' if delta >= 0 else ''}{delta:.2f}%"
            else:
                delta = None
                d_str = "N/A"

            label = f"{d} (Middle)" if d in ("0.50", "0.5") else d
            print(f"{label:<14} | {m_str:<12} | {s_str:<14} | {d_str:<20}")

            depth_comparison.append({
                "depth": d,
                "mottt_acc": m_info["accuracy"] if m_info else None,
                "sft_acc": s_info["accuracy"] if s_info else None,
                "delta": delta,
            })
        print("-" * 70)

    # MoTTT specific metrics if present
    if mottt_data:
        print("\n--- 3. MoTTT ARCHITECTURE UTILIZATION ---")
        sp_mean = mottt_data.get("scratchpad_gate_mean")
        r_mean = mottt_data.get("reasoning_gate_mean")
        if sp_mean is not None:
            print(f"Scratchpad Expert Gate Mean (E_0): {sp_mean:.4f}")
        if r_mean is not None:
            print(f"Reasoning Experts Gate Mean (E_R): {r_mean:.4f}")

    # Save summary
    summary = {
        "mottt_overall_accuracy": mottt_acc,
        "sft_overall_accuracy": sft_acc,
        "accuracy_delta": (mottt_acc - sft_acc) if (mottt_acc is not None and sft_acc is not None) else None,
        "depth_comparison": depth_comparison,
    }

    out_file = Path(args.output_summary)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved comparison summary to: {out_file}")
    print("=" * 78)


if __name__ == "__main__":
    main()
