"""Hugging Face GSM8K dataset loader and answer parser."""

from dataclasses import dataclass
import re
from typing import List, Optional
from datasets import load_dataset


@dataclass
class GSM8KExample:
    example_id: str
    question: str
    solution: str
    gold_answer: str


def extract_gold_answer(solution_text: str) -> str:
    """Extract the final numerical answer from GSM8K solution (after '#### ').

    Example:
    'Janet makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18' -> '18'
    """
    if "####" in solution_text:
        ans = solution_text.split("####")[-1].strip()
        # Remove commas, currency symbols, and extra whitespaces
        ans = re.sub(r"[,$]", "", ans).strip()
        return ans
    # Fallback regex for numbers at the very end
    match = re.search(r"[-+]?\d+(?:\.\d+)?", solution_text.split("\n")[-1])
    if match:
        return match.group(0).strip()
    return ""


def load_gsm8k_dataset(
    split: str = "test",
    max_samples: Optional[int] = None,
    dataset_name: str = "openai/gsm8k",
    config_name: str = "main",
) -> List[GSM8KExample]:
    """Load GSM8K examples from Hugging Face datasets.

    Args:
        split: 'test' or 'train'
        max_samples: Optional limit on the number of examples to load
        dataset_name: Hugging Face repo name ('openai/gsm8k')
        config_name: Dataset configuration ('main')

    Returns:
        List of GSM8KExample dataclass instances
    """
    split_str = f"{split}[:{max_samples}]" if max_samples is not None else split
    hf_dataset = load_dataset(dataset_name, config_name, split=split_str)

    examples: List[GSM8KExample] = []
    for idx, item in enumerate(hf_dataset):
        q = item["question"].strip()
        sol = item["answer"].strip()
        gold = extract_gold_answer(sol)
        example_id = f"gsm8k_{split}_{idx}"
        examples.append(
            GSM8KExample(
                example_id=example_id,
                question=q,
                solution=sol,
                gold_answer=gold,
            )
        )

    return examples
