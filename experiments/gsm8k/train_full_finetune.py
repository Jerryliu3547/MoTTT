#!/usr/bin/env python3
"""Full Fine-Tuning (SFT) training script for Qwen2.5-0.5B on distractor GSM8K benchmark."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mottt.data.dataset_exporter import load_distractor_jsonl


def format_sft_prompt(context: str, query: str) -> str:
    """Format full prompt with background distractor context and query."""
    return (
        "Background Context:\n"
        f"{context.strip()}\n\n"
        "Question:\n"
        f"{query.strip()}\n\n"
        "Please solve the problem step by step and end your response with '#### [final numerical answer]'."
    )


class DistractorSFTDataset(Dataset):
    """Dataset for full fine-tuning with prompt-response masking."""

    def __init__(self, records: List[Dict[str, Any]], is_mock: bool = False, hidden_dim: int = 64):
        self.records = records
        self.is_mock = is_mock
        self.hidden_dim = hidden_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        if self.is_mock:
            seq_len = 32
            return {
                "input_ids": torch.randint(1, 100, (seq_len,)),
                "labels": torch.randint(1, 100, (seq_len,)),
                "attention_mask": torch.ones(seq_len, dtype=torch.long),
                "id": rec.get("id", f"mock_{idx}"),
            }
        return rec


def make_sft_collate_fn(tokenizer, max_length: int = 1024):
    """Dynamically tokenize context+query prompt and solution with loss masking."""
    def collate_fn(batch_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = []
        full_texts = []

        for r in batch_records:
            ctx = r.get("distractor_context", "")[:1500]
            q = r.get("query", "")
            sol = r.get("solution", "")
            prompt = format_sft_prompt(context=ctx, query=q)
            prompts.append(prompt)
            full_texts.append(f"{prompt}\n\nSolution: {sol}")

        # Tokenize prompt to determine prompt length for label masking
        prompt_lens = []
        for p in prompts:
            p_ids = tokenizer.encode(p, add_special_tokens=True)
            prompt_lens.append(len(p_ids))

        # Tokenize full prompt + solution
        enc = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        labels = enc.input_ids.clone()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Mask prompt tokens with -100 so loss is strictly computed on solution tokens
        for i, p_len in enumerate(prompt_lens):
            clamped_len = min(p_len, labels.shape[1])
            labels[i, :clamped_len] = -100

        labels[labels == pad_token_id] = -100

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
            "ids": [r.get("id", "") for r in batch_records],
        }

    return collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full parameter fine-tuning of Qwen2.5 on distractor-injected GSM8K benchmark"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/gsm8k_distractor_train/gsm8k_distractor_all.jsonl",
        help="Path to training distractor JSONL file",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face base model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/gsm8k/checkpoints_full_finetune",
        help="Directory to save fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Number of gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for full parameter fine-tuning (default: 2e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup steps ratio (default: 0.03)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum input sequence length in tokens (default: 1024)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode with synthetic dummy model on CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Baseline Comparison: Full Fine-Tuning (SFT) on Distractor GSM8K")
    print("=" * 70)
    print(f"Base Model:       {args.base_model_name} {'[MOCK MODE]' if args.mock else ''}")
    print(f"Training Data:    {args.data_path}")
    print(f"Output Checkpoint: {out_dir}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Batch Size:       {args.batch_size} (Grad Accum: {args.grad_accum_steps})")
    print(f"Epochs:           {args.epochs}")
    print(f"Max Seq Length:   {args.max_length}")
    print("=" * 70)

    device = "cuda" if (torch.cuda.is_available() and not args.mock) else "cpu"
    torch_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    model = None
    tokenizer = None
    collate_fn = None

    if args.mock:
        print("\n[Mock Mode] Creating synthetic mock model...")
        vocab_size = 100
        hidden_dim = 64

        class MockSFTModel(nn.Module):
            def __init__(self, vocab_sz, h_dim):
                super().__init__()
                self.embed = nn.Embedding(vocab_sz, h_dim)
                self.linear = nn.Linear(h_dim, h_dim)
                self.head = nn.Linear(h_dim, vocab_sz)

            def forward(self, input_ids, attention_mask=None, labels=None):
                h = self.linear(self.embed(input_ids))
                logits = self.head(h)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
                return type("MockOutput", (), {"logits": logits, "loss": loss})()

        model = MockSFTModel(vocab_size, hidden_dim).to(device)
        records = [{"id": f"mock_{i}"} for i in range(8)]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\nLoading base model & tokenizer: {args.base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Unfreeze all parameters for Full Fine-Tuning
        for p in model.parameters():
            p.requires_grad = True

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full Model Unfrozen: {total_trainable:,} trainable parameters.")

        collate_fn = make_sft_collate_fn(tokenizer, max_length=args.max_length)

        data_file = Path(args.data_path)
        if data_file.exists():
            records = load_distractor_jsonl(str(data_file))
            print(f"Loaded {len(records)} records from {args.data_path}")
        else:
            print(f"Notice: Data file {args.data_path} not found locally.")
            print("Falling back to fallback synthetic records for pipeline validation.")
            records = [{"id": f"synthetic_{i}"} for i in range(8)]

    dataset = DistractorSFTDataset(records, is_mock=args.mock)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(dataloader) * args.epochs // max(1, args.grad_accum_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def get_lr_multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    print("\nBeginning Full Fine-Tuning...")
    history = []
    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is None or torch.isnan(loss):
                continue

            loss_scaled = loss / args.grad_accum_steps
            loss_scaled.backward()
            epoch_loss += loss.item()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch} finished - Average Loss: {avg_loss:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss})

    # Save checkpoint
    print(f"\nSaving Full Fine-Tuned checkpoint to {out_dir}...")
    if not args.mock and hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
    else:
        # Save mock model weights
        torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    # Save training configuration
    cfg = {
        "model_type": "full_finetune_sft",
        "base_model_name": args.base_model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "is_mock": args.mock,
        "training_history": history,
    }
    with open(out_dir / "sft_training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Checkpoints and configuration successfully saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
