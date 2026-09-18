#!/usr/bin/env python3
"""Model training script for MoTTT on GSM8K with Qwen2.5-0.5B and Query-Aware Router."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src/ is on sys.path even if not installed via pip install -e .
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from typing import Any, Dict, List, Optional

from mottt.models.mottt_model import MoTTTModel
from mottt.models.molora import MoLoRALinear
from mottt.data.dataset_exporter import load_distractor_jsonl


class GSM8KTrainDataset(Dataset):
    """Dataset for training MoTTT Query-Aware Router and reasoning experts."""

    def __init__(self, records: List[Dict], hidden_dim: int = 896, is_mock: bool = False):
        self.records = records
        self.hidden_dim = hidden_dim
        self.is_mock = is_mock

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        if self.is_mock:
            seq_len = 16
            return {
                "hidden_states": torch.randn(seq_len, self.hidden_dim),
                "query_embedding": torch.randn(self.hidden_dim),
                "context_states": torch.randn(4, self.hidden_dim),
                "target_labels": torch.randint(0, 100, (seq_len,)),
                "id": rec.get("id", f"sample_{idx}"),
            }
        # In full GPU mode, return raw problem record for dynamic batch tokenization
        return rec


def make_collate_fn(tokenizer, max_length: int = 512, max_context_length: int = 256):
    """Collate function to dynamically tokenize queries, solutions, and contexts."""
    def collate_fn(batch_records: List[Dict]) -> Dict[str, Any]:
        queries = [r.get("query", "") for r in batch_records]
        solutions = [r.get("solution", "") for r in batch_records]
        contexts = [r.get("distractor_context", "")[:1500] for r in batch_records]

        prompts = [f"Question:\n{q}\n\nSolution:" for q in queries]
        full_texts = [f"{p} {s}" for p, s in zip(prompts, solutions)]

        ctx_enc = tokenizer(
            contexts,
            max_length=max_context_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        full_enc = tokenizer(
            full_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        labels = full_enc.input_ids.clone()
        prompt_lens = []
        for i, p in enumerate(prompts):
            p_len = len(tokenizer.encode(p, add_special_tokens=False))
            prompt_lens.append(p_len)
            labels[i, :p_len] = -100

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels[labels == pad_token_id] = -100

        return {
            "ctx_input_ids": ctx_enc.input_ids,
            "ctx_attention_mask": ctx_enc.attention_mask,
            "input_ids": full_enc.input_ids,
            "attention_mask": full_enc.attention_mask,
            "labels": labels,
            "prompt_lens": prompt_lens,
            "ids": [r.get("id", "") for r in batch_records],
        }

    return collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MoTTT on GSM8K with Qwen2.5-0.5B"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="experiments/gsm8k/data/train_distractor.jsonl",
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
        default="experiments/gsm8k/checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=4,
        help="Number of reasoning LoRA experts E in [4, 6] (default: 4)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank r (default: 16)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=16.0,
        help="LoRA alpha parameter (default: 16.0 for unit multiplier gamma=1.0)",
    )
    parser.add_argument(
        "--lambda_bal",
        type=float,
        default=0.01,
        help="Asymmetric load balancing weight lambda_bal (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for router and reasoning LoRAs (default: 1e-4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--inner_lr",
        type=float,
        default=1e-3,
        help="Inner-loop learning rate for dynamic scratchpad adaptation (default: 1e-3)",
    )
    parser.add_argument(
        "--inner_steps",
        type=int,
        default=1,
        help="Number of inner-loop adaptation steps per question/batch (default: 1)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/CPU mode with synthetic backbone",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MoTTT GSM8K: Model Training Pipeline")
    print("=" * 70)
    print(f"Base Model:       {args.base_model_name} {'[MOCK MODE]' if args.mock else ''}")
    print(f"Reasoning Experts: {args.num_experts}")
    print(f"LoRA Rank (r):    {args.rank}, Alpha: {args.alpha} (gamma={args.alpha/args.rank:.1f})")
    print(f"Lambda Bal:       {args.lambda_bal}")
    print(f"Output Checkpoint: {out_dir}")
    print("=" * 70)

    # 1. Determine model dimensions & device
    device = "cuda" if (torch.cuda.is_available() and not args.mock) else "cpu"
    hidden_dim = 64 if args.mock else 896

    # Optional: Load real base model and tokenizer when not in mock mode
    base_llm = None
    tokenizer = None
    collate_fn = None
    torch_dtype = torch.float32

    if not args.mock:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading base model & tokenizer: {args.base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        base_llm = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        base_llm.eval()
        for p in base_llm.parameters():
            p.requires_grad = False

        hidden_dim = base_llm.config.hidden_size
        vocab_size = base_llm.config.vocab_size
        collate_fn = make_collate_fn(tokenizer)
        print(f"Base LLM loaded (hidden_dim={hidden_dim}, vocab_size={vocab_size}, dtype={torch_dtype}).")

    # 2. Load training data
    if Path(args.data_path).exists():
        records = load_distractor_jsonl(args.data_path)
        print(f"Loaded {len(records)} records from {args.data_path}")
    else:
        print(f"Warning: Data file {args.data_path} not found. Using fallback mock records.")
        records = [{"id": f"mock_{i}"} for i in range(8)]

    dataset = GSM8KTrainDataset(records, hidden_dim=hidden_dim, is_mock=args.mock)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 3. Instantiate MoTTT Architecture
    print(f"\nInitializing MoTTT model on {device.upper()}...")
    if args.mock:
        class MockBackbone(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim)
            def forward(self, x):
                return self.q_proj(x)
        base_backbone = MockBackbone(hidden_dim)
    else:
        base_backbone = base_llm

    model = MoTTTModel(
        hidden_dim=hidden_dim,
        num_reasoning_experts=args.num_experts,
        rank=args.rank,
        alpha=args.alpha,
        lambda_bal=args.lambda_bal,
        base_backbone=base_backbone,
        all_linear=True,
    ).to(device)

    if not args.mock:
        model = model.to(device, dtype=torch_dtype)

    # 4. Optimizer: Optimize Router MLP and Reasoning LoRA Experts
    trainable_params = list(model.router.parameters())
    trainable_params.extend(model.get_reasoning_parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Inner-loop optimizer for dynamic scratchpad LoRA
    inner_params = model.get_scratchpad_parameters()
    inner_opt = torch.optim.SGD(inner_params, lr=args.inner_lr)

    # Prediction head and loss function
    if args.mock:
        head = nn.Linear(hidden_dim, 100).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Training Loop
    print("\nBeginning training...")
    history = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_epoch_loss = 0.0
        total_task_loss = 0.0
        total_bal_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_task = 0.0
            batch_bal = 0.0

            if not args.mock and base_llm is not None:
                # Real Qwen2.5-0.5B tokenized training path
                ctx_ids = batch["ctx_input_ids"].to(device)
                ctx_mask = batch["ctx_attention_mask"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                prompt_lens = batch["prompt_lens"]
                current_bsz = input_ids.shape[0]

                for b in range(current_bsz):
                    # -------------------------------------------------------------
                    # Step 1: Reset Scratchpad for sample b - isolated episodic memory
                    # Guarantees zero state leakage across samples within the batch
                    # -------------------------------------------------------------
                    model.reset_scratchpad()
                    inner_opt.zero_grad()
                    model.set_routing_gates(None)

                    # Step 2: Inner-loop adaptation exclusively on sample b's context
                    b_ctx_ids = ctx_ids[b : b + 1]
                    b_ctx_mask = ctx_mask[b : b + 1]
                    for _ in range(args.inner_steps):
                        ctx_out = base_llm(input_ids=b_ctx_ids, attention_mask=b_ctx_mask)
                        ctx_logits = ctx_out.logits
                        shift_ctx_logits = ctx_logits[..., :-1, :].contiguous()
                        shift_ctx_labels = b_ctx_ids[..., 1:].contiguous()
                        inner_loss = F.cross_entropy(shift_ctx_logits.view(-1, vocab_size), shift_ctx_labels.view(-1))
                        inner_opt.zero_grad()
                        inner_loss.backward()
                        inner_opt.step()

                    # Step 3: Outer-loop forward pass on sample b's query + solution
                    b_input_ids = input_ids[b : b + 1]
                    b_attn_mask = attention_mask[b : b + 1]
                    b_labels = labels[b : b + 1]

                    with torch.no_grad():
                        token_embs = base_llm.model.embed_tokens(b_input_ids)
                    p_len = prompt_lens[b]
                    p_len_clamped = max(1, min(p_len, token_embs.shape[1]))
                    q = token_embs[0, :p_len_clamped].mean(dim=0, keepdim=True)

                    gates, logits = model.router(token_embs, q)
                    model.set_routing_gates(gates)

                    outputs = base_llm(input_ids=b_input_ids, attention_mask=b_attn_mask)
                    vocab_logits = outputs.logits

                    shift_logits = vocab_logits[..., :-1, :].contiguous()
                    shift_labels = b_labels[..., 1:].contiguous()

                    task_loss = loss_fn(shift_logits.view(-1, vocab_size).float(), shift_labels.view(-1))
                    bal_loss = model.compute_auxiliary_loss(logits)

                    sample_loss = task_loss + bal_loss
                    # Accumulate outer gradients scaled by 1 / current_bsz for reasoning experts & router
                    (sample_loss / current_bsz).backward()

                    batch_loss += sample_loss.item() / current_bsz
                    batch_task += task_loss.item() / current_bsz
                    batch_bal += bal_loss.item() / current_bsz

                # Step outer optimizer once across the batch
                optimizer.step()
                model.reset_scratchpad()
            else:
                # Mock / CPU verification path
                ctx = batch["context_states"].to(device)
                h = batch["hidden_states"].to(device)
                q = batch["query_embedding"].to(device)
                targets = batch["target_labels"].to(device)
                current_bsz = h.shape[0]

                for b in range(current_bsz):
                    # Step 1: Reset Scratchpad for sample b
                    model.reset_scratchpad()
                    inner_opt.zero_grad()
                    model.set_routing_gates(None)

                    # Step 2: Inner-loop adaptation exclusively on sample b's context
                    b_ctx = ctx[b : b + 1]
                    for _ in range(args.inner_steps):
                        chunk_out = model.base_backbone(b_ctx)
                        inner_loss = F.mse_loss(chunk_out, torch.zeros_like(chunk_out))
                        inner_opt.zero_grad()
                        inner_loss.backward()
                        inner_opt.step()

                    # Step 3: Outer-loop forward on sample b
                    b_h = h[b : b + 1]
                    b_q = q[b : b + 1]
                    b_targets = targets[b : b + 1]

                    blended, gates, logits = model(b_h, b_q)
                    preds = head(blended)

                    task_loss = loss_fn(preds.view(-1, preds.shape[-1]), b_targets.view(-1))
                    bal_loss = model.compute_auxiliary_loss(logits)

                    sample_loss = task_loss + bal_loss
                    # Accumulate outer gradients scaled by 1 / current_bsz
                    (sample_loss / current_bsz).backward()

                    batch_loss += sample_loss.item() / current_bsz
                    batch_task += task_loss.item() / current_bsz
                    batch_bal += bal_loss.item() / current_bsz

                # Step outer optimizer once across the batch
                optimizer.step()
                model.reset_scratchpad()

            total_epoch_loss += batch_loss
            total_task_loss += batch_task
            total_bal_loss += batch_bal

            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "task": f"{batch_task:.4f}",
                "bal": f"{batch_bal:.5f}",
            })

        avg_loss = total_epoch_loss / len(dataloader)
        avg_task = total_task_loss / len(dataloader)
        avg_bal = total_bal_loss / len(dataloader)

        history.append({
            "epoch": epoch,
            "outer_loss": avg_loss,
            "task_loss": avg_task,
            "balance_loss": avg_bal,
        })
        print(f"Epoch {epoch}/{args.epochs} Complete | Outer Loss: {avg_loss:.4f} (Task: {avg_task:.4f}, Bal: {avg_bal:.6f})")

    # 6. Save Checkpoint
    print(f"\nSaving model checkpoints to {out_dir}...")
    torch.save(model.router.state_dict(), out_dir / "mottt_router.pt")

    if model.injected_linear_names:
        molora_states = {
            name: {
                "reasoning_lora_A": mod.reasoning_lora_A.state_dict(),
                "reasoning_lora_B": mod.reasoning_lora_B.state_dict(),
            }
            for name, mod in model.base_backbone.named_modules()
            if isinstance(mod, MoLoRALinear)
        }
        torch.save(molora_states, out_dir / "reasoning_experts.pt")
        print(f"Saved {len(molora_states)} all-linear reasoning expert modules to {out_dir / 'reasoning_experts.pt'}")
    else:
        torch.save(
            [exp.state_dict() for exp in model.reasoning_experts],
            out_dir / "reasoning_experts.pt",
        )

    config = {
        "base_model_name": args.base_model_name,
        "hidden_dim": hidden_dim,
        "num_reasoning_experts": args.num_experts,
        "rank": args.rank,
        "alpha": args.alpha,
        "lambda_bal": args.lambda_bal,
        "epochs": args.epochs,
        "is_mock": args.mock,
        "training_history": history,
    }
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("TRAINING COMPLETE & CHECKPOINTS SAVED!")
    print(f"  - Router weights:     {out_dir / 'mottt_router.pt'}")
    print(f"  - Reasoning experts:  {out_dir / 'reasoning_experts.pt'}")
    print(f"  - Training metadata:  {out_dir / 'training_config.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
