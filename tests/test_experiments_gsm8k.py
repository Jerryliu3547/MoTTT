"""Integration test for experiments/gsm8k: data_preparation, model_train, and model_test."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


def test_experiments_gsm8k_end_to_end_cycle():
    python_bin = sys.executable

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_dir = tmp_path / "data"
        ckpt_dir = tmp_path / "checkpoints"
        results_dir = tmp_path / "results"

        # 1. Test data_preparation.py in mock mode
        prep_cmd = [
            python_bin,
            "experiments/gsm8k/data_preparation.py",
            "--mock",
            "--target_context_tokens", "128",
            "--chunk_size", "32",
            "--depth_ratios", "0.2,0.8",
            "--output_dir", str(data_dir),
        ]
        res_prep = subprocess.run(prep_cmd, capture_output=True, text=True)
        assert res_prep.returncode == 0, f"data_preparation failed:\n{res_prep.stderr}"

        train_jsonl = data_dir / "train_distractor.jsonl"
        test_jsonl = data_dir / "test_distractor.jsonl"
        assert train_jsonl.exists()
        assert test_jsonl.exists()

        # 2. Test model_train.py in mock mode
        train_cmd = [
            python_bin,
            "experiments/gsm8k/model_train.py",
            "--mock",
            "--data_path", str(train_jsonl),
            "--output_dir", str(ckpt_dir),
            "--num_experts", "4",
            "--rank", "8",
            "--alpha", "8.0",
            "--epochs", "1",
            "--batch_size", "2",
        ]
        res_train = subprocess.run(train_cmd, capture_output=True, text=True)
        assert res_train.returncode == 0, f"model_train failed:\n{res_train.stderr}"

        assert (ckpt_dir / "mottt_router.pt").exists()
        assert (ckpt_dir / "reasoning_experts.pt").exists()
        assert (ckpt_dir / "training_config.json").exists()

        # Verify training config contents
        with open(ckpt_dir / "training_config.json", "r") as f:
            cfg = json.load(f)
        assert cfg["num_reasoning_experts"] == 4
        assert cfg["is_mock"] is True

        # 3. Test model_test.py in mock mode
        test_cmd = [
            python_bin,
            "experiments/gsm8k/model_test.py",
            "--mock",
            "--test_data", str(test_jsonl),
            "--checkpoint_dir", str(ckpt_dir),
            "--output_dir", str(results_dir),
        ]
        res_test = subprocess.run(test_cmd, capture_output=True, text=True)
        assert res_test.returncode == 0, f"model_test failed:\n{res_test.stderr}"

        report_file = results_dir / "gsm8k_eval_report.json"
        assert report_file.exists()

        with open(report_file, "r") as f:
            report = json.load(f)
        assert "overall_accuracy" in report
        assert "scratchpad_gate_mean" in report
        assert "depth_breakdown" in report
        assert len(report["detailed_predictions"]) > 0
