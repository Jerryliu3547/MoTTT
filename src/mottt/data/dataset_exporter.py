"""Dataset Exporter: Saves distractor-injected GSM8K datasets to standard JSONL and HF formats."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import Dataset

from mottt.data.distractor_generator import LongContextGSM8KRecord


def export_records_to_jsonl(
    records: List[LongContextGSM8KRecord],
    output_file: str,
) -> None:
    """Export a list of records to a JSON Lines file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def export_dataset_bundle(
    records: List[LongContextGSM8KRecord],
    output_dir: str,
    save_hf_dataset: bool = True,
) -> Dict[str, Any]:
    """Export complete dataset bundle: combined JSONL, per-depth JSONL, HF Dataset, and manifest.

    Args:
        records: List of synthesized LongContextGSM8KRecord
        output_dir: Target output directory (e.g. 'data/gsm8k_distractor')
        save_hf_dataset: Whether to also save as Hugging Face Dataset directory

    Returns:
        Manifest dictionary with summary metadata
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Export all records combined
    all_jsonl_path = out_path / "gsm8k_distractor_all.jsonl"
    export_records_to_jsonl(records, str(all_jsonl_path))

    # 2. Group records by depth ratio and export per-depth files
    depth_groups: Dict[float, List[LongContextGSM8KRecord]] = {}
    for rec in records:
        depth = round(rec.needle_depth_ratio, 2)
        depth_groups.setdefault(depth, []).append(rec)

    per_depth_files = {}
    for depth, group_recs in sorted(depth_groups.items()):
        filename = f"gsm8k_distractor_depth_{depth:.2f}.jsonl"
        depth_file_path = out_path / filename
        export_records_to_jsonl(group_recs, str(depth_file_path))
        per_depth_files[f"{depth:.2f}"] = {
            "file": filename,
            "count": len(group_recs),
        }

    # 3. Save as Hugging Face Dataset if requested
    hf_path_str = None
    if save_hf_dataset:
        hf_dataset_path = out_path / "hf_dataset"
        dict_records = [r.to_dict() for r in records]
        # Exclude mottt_chunks if None or keep as string/dict
        hf_ds = Dataset.from_list(dict_records)
        hf_ds.save_to_disk(str(hf_dataset_path))
        hf_path_str = str(hf_dataset_path.name)

    # 4. Generate Manifest
    sample_preview = records[0].to_dict() if records else {}
    manifest = {
        "dataset_name": "MoTTT_Distractor_GSM8K",
        "description": "Distractor-injected long-context GSM8K benchmark for MoTTT and baseline LLM evaluation",
        "total_records": len(records),
        "depth_ratios": [float(d) for d in sorted(depth_groups.keys())],
        "records_per_depth": per_depth_files,
        "combined_file": all_jsonl_path.name,
        "hf_dataset_directory": hf_path_str,
        "fields": list(sample_preview.keys()) if sample_preview else [],
        "sample_preview": {
            "id": sample_preview.get("id"),
            "query": sample_preview.get("query"),
            "needle_depth_ratio": sample_preview.get("needle_depth_ratio"),
            "gold_answer": sample_preview.get("gold_answer"),
            "full_prompt_snippet": sample_preview.get("full_prompt", "")[:300] + "...",
        },
    }

    manifest_path = out_path / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def load_distractor_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load records from an exported JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
