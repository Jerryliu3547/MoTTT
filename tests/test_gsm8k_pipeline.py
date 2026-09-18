"""Tests for GSM8K HuggingFace integration, math filtering, and dataset export."""

import tempfile
from pathlib import Path
import pytest

from mottt.data.gsm8k_loader import extract_gold_answer
from mottt.data.distractor_generator import (
    PremiseQueryDecomposer,
    DistractorNeedleSynthesizer,
    is_non_numerical,
    format_baseline_prompt,
)
from mottt.data.dataset_exporter import (
    export_dataset_bundle,
    load_distractor_jsonl,
)


def test_extract_gold_answer():
    sol1 = "Janet makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18"
    assert extract_gold_answer(sol1) == "18"

    sol2 = "Total cost is $1,200.\n#### $1,200"
    assert extract_gold_answer(sol2) == "1200"

    sol3 = "The result is 42.5\n#### 42.5"
    assert extract_gold_answer(sol3) == "42.5"


def test_is_non_numerical_filter():
    assert is_non_numerical("The ancient lighthouse stood steadily against the coastal winds.") is True
    assert is_non_numerical("A quiet forest path wound gently through the towering trees.") is True

    # Reject sentences with digits
    assert is_non_numerical("There were 48 friends present.") is False
    # Reject math operators
    assert is_non_numerical("Total cost = expense + tax") is False
    # Reject spelled numbers
    assert is_non_numerical("There were four birds on the tree.") is False


def test_premise_query_decomposition():
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?"
    )
    solution = "Natalia sold 48 / 2 = <<48/2=24>>24 clips in May.\nAltogether she sold 48 + 24 = <<48+24=72>>72 clips.\n#### 72"

    decomposed = PremiseQueryDecomposer.decompose(
        problem_text=question,
        solution_text=solution,
        gold_answer="72",
    )

    assert "Natalia sold clips to 48" in decomposed.premise
    assert "How many clips did Natalia sell" in decomposed.query
    assert decomposed.gold_answer == "72"


def test_distractor_record_creation_and_export():
    synthesizer = DistractorNeedleSynthesizer(chunk_size=32)

    premise = "The secret vault passcode is 9876."
    query = "What is the passcode to the vault?"
    solution = "The passcode is 9876.\n#### 9876"
    gold_answer = "9876"

    records = []
    for depth in [0.1, 0.5, 0.9]:
        rec = synthesizer.create_record(
            example_id="gsm8k_mock_0",
            original_question=f"{premise} {query}",
            premise=premise,
            query=query,
            solution=solution,
            gold_answer=gold_answer,
            depth_ratio=depth,
            target_token_count=128,
        )
        records.append(rec)

    assert len(records) == 3

    # Check record fields
    r = records[1]
    assert r.needle_depth_ratio == 0.5
    assert premise in r.distractor_context
    assert query in r.full_prompt
    assert r.gold_answer == "9876"
    assert r.distractor_context[r.premise_char_start : r.premise_char_end] == premise

    # Test export bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = export_dataset_bundle(records=records, output_dir=tmpdir, save_hf_dataset=True)

        assert manifest["total_records"] == 3
        assert (Path(tmpdir) / "gsm8k_distractor_all.jsonl").exists()
        assert (Path(tmpdir) / "gsm8k_distractor_depth_0.10.jsonl").exists()
        assert (Path(tmpdir) / "gsm8k_distractor_depth_0.50.jsonl").exists()
        assert (Path(tmpdir) / "gsm8k_distractor_depth_0.90.jsonl").exists()
        assert (Path(tmpdir) / "hf_dataset").exists()
        assert (Path(tmpdir) / "dataset_manifest.json").exists()

        # Test loading JSONL back
        loaded = load_distractor_jsonl(str(Path(tmpdir) / "gsm8k_distractor_all.jsonl"))
        assert len(loaded) == 3
        assert loaded[1]["id"] == r.id
        assert loaded[1]["gold_answer"] == "9876"
        assert loaded[1]["query"] == query


def test_random_depth_selection():
    import subprocess
    import sys
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            "experiments/gsm8k/data_preparation.py",
            "--mock",
            "--random_train_depth",
            "--target_context_tokens", "128",
            "--chunk_size", "32",
            "--depth_ratios", "0.2,0.5,0.8",
            "--output_dir", str(tmpdir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Error:\n{res.stderr}"

        # In mock mode, 2 train examples. With --random_train_depth, exactly 2 records should be generated
        train_file = Path(tmpdir) / "train_distractor.jsonl"
        assert train_file.exists()
        loaded_train = load_distractor_jsonl(str(train_file))
        assert len(loaded_train) == 2
        for rec in loaded_train:
            assert rec["needle_depth_ratio"] in [0.2, 0.5, 0.8]


def test_extract_predicted_answer_edge_cases():
    """Verify that extract_predicted_answer never raises IndexError and extracts correctly."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "gsm8k"))
    from model_test import extract_predicted_answer

    # Edge Case 1: Trailing #### with only whitespace or empty
    assert extract_predicted_answer("The final answer is ####") == ""
    assert extract_predicted_answer("Step 1: 5 + 5 = 10\n####   \n") == "10"
    assert extract_predicted_answer("#### $,") == ""

    # Edge Case 2: Standard #### answer with currency/commas
    assert extract_predicted_answer("#### $1,400\nExtra text") == "1400"
    assert extract_predicted_answer("The answer is #### 42") == "42"

    # Edge Case 3: 'The answer is' pattern without ####
    assert extract_predicted_answer("Therefore, the answer is 250.") == "250"
    assert extract_predicted_answer("So it is equal to 99") == "99"

    # Edge Case 4: No pattern, fallback to last number
    assert extract_predicted_answer("He had 5 apples then bought 7 more so he has 12") == "12"

    # Edge Case 5: No numbers at all
    assert extract_predicted_answer("No solution found") == ""

