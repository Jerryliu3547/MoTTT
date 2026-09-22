"""Integration test for full fine-tuning (SFT) baseline and comparison scripts."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


def test_baseline_sft_and_comparison_cycle():
    python_bin = sys.executable

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_dir = tmp_path / "checkpoints_sft"
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Test train_full_finetune.py in mock mode
        train_cmd = [
            python_bin,
            "experiments/gsm8k/train_full_finetune.py",
            "--mock",
            "--output_dir", str(ckpt_dir),
            "--epochs", "1",
            "--batch_size", "2",
            "--grad_accum_steps", "1",
        ]
        res_train = subprocess.run(train_cmd, capture_output=True, text=True)
        assert res_train.returncode == 0, f"train_full_finetune failed:\n{res_train.stderr}"
        assert (ckpt_dir / "sft_training_config.json").exists()
        assert (ckpt_dir / "pytorch_model.bin").exists()

        with open(ckpt_dir / "sft_training_config.json", "r") as f:
            cfg = json.load(f)
        assert cfg["is_mock"] is True
        assert cfg["model_type"] == "full_finetune_sft"

        # 2. Test test_full_finetune.py in mock mode
        test_cmd = [
            python_bin,
            "experiments/gsm8k/test_full_finetune.py",
            "--mock",
            "--checkpoint_dir", str(ckpt_dir),
            "--output_dir", str(results_dir),
            "--output_file", "baseline_sft_eval_report.json",
        ]
        res_test = subprocess.run(test_cmd, capture_output=True, text=True)
        assert res_test.returncode == 0, f"test_full_finetune failed:\n{res_test.stderr}"

        sft_report_file = results_dir / "baseline_sft_eval_report.json"
        assert sft_report_file.exists()

        with open(sft_report_file, "r") as f:
            sft_report = json.load(f)
        assert "overall_accuracy" in sft_report
        assert "depth_breakdown" in sft_report
        assert len(sft_report["detailed_predictions"]) > 0

        # 3. Create dummy MoTTT report for comparison testing
        mottt_report_file = results_dir / "gsm8k_eval_report.json"
        mottt_report_data = {
            "overall_accuracy": 72.5,
            "scratchpad_gate_mean": 0.35,
            "reasoning_gate_mean": 0.16,
            "depth_breakdown": {
                "0.10": {"accuracy": 75.0, "correct": 3, "total": 4},
                "0.50": {"accuracy": 70.0, "correct": 7, "total": 10},
                "0.90": {"accuracy": 75.0, "correct": 3, "total": 4},
            },
        }
        with open(mottt_report_file, "w", encoding="utf-8") as f:
            json.dump(mottt_report_data, f, indent=2)

        # 4. Test compare_results.py
        summary_file = results_dir / "comparison_summary.json"
        comp_cmd = [
            python_bin,
            "experiments/gsm8k/compare_results.py",
            "--mottt_report", str(mottt_report_file),
            "--sft_report", str(sft_report_file),
            "--output_summary", str(summary_file),
        ]
        res_comp = subprocess.run(comp_cmd, capture_output=True, text=True)
        assert res_comp.returncode == 0, f"compare_results failed:\n{res_comp.stderr}"
        assert summary_file.exists()

        with open(summary_file, "r") as f:
            summary = json.load(f)
        assert summary["mottt_overall_accuracy"] == 72.5
        assert summary["sft_overall_accuracy"] is not None
        assert len(summary["depth_comparison"]) > 0
