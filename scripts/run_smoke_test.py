#!/usr/bin/env python3
"""Run lightweight smoke verification of the MoTTT environment on CPU without downloading external datasets or models."""

import sys
import torch
import yaml
from pathlib import Path

from mottt.models.mottt_model import MoTTTModel
from mottt.data.distractor_generator import PremiseQueryDecomposer, DistractorNeedleSynthesizer


def main():
    print("=" * 60)
    print("MoTTT Environment Smoke Verification (CPU / Mock Mode)")
    print("=" * 60)

    # 1. Check Python & PyTorch
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} (CPU mode active)")

    # 2. Load smoke configuration
    config_path = Path(__file__).parent.parent / "src" / "mottt" / "configs" / "smoke_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config_path.name}")

    # 3. Instantiate MoTTT with mock dimensions
    hidden_dim = config["model"]["hidden_dim"]
    num_experts = config["model"]["num_reasoning_experts"]
    rank = config["model"]["rank"]
    alpha = config["model"]["alpha"]

    print(f"Initializing MoTTT model (d={hidden_dim}, E={num_experts}, rank={rank}, alpha={alpha})...")
    model = MoTTTModel(
        hidden_dim=hidden_dim,
        num_reasoning_experts=num_experts,
        rank=rank,
        alpha=alpha,
        lambda_bal=config["model"]["lambda_bal"],
    )

    # 4. Mock forward pass
    batch_size = config["training"]["batch_size"]
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_dim)
    q = torch.randn(batch_size, hidden_dim)

    blended, gates, logits = model(x, q)
    aux_loss = model.compute_auxiliary_loss(logits)

    print(f"Model forward pass successful!")
    print(f"  - Output shape: {blended.shape}")
    print(f"  - Gates shape: {gates.shape} (Expert 0=Scratchpad, 1..{num_experts}=Reasoning)")
    print(f"  - Asymmetric load balancing loss: {aux_loss.item():.6f}")

    # 5. Test premise decomposition & needle distractor synthesis
    print("Testing Premise/Query decomposition and distractor needle embedding...")
    sample_text = (
        "Janet has 15 apples. She gives 3 to Peter and then bakes half of the rest. "
        "How many apples did Janet bake?"
    )
    decomposed = PremiseQueryDecomposer.decompose(sample_text)
    print(f"  - Premise P: \"{decomposed.premise}\"")
    print(f"  - Query Q:   \"{decomposed.query}\"")

    synthesizer = DistractorNeedleSynthesizer(chunk_size=config["training"]["chunk_size"])
    context, depth = synthesizer.build_synthetic_context(
        premise=decomposed.premise,
        target_token_count=config["training"]["target_context_length"],
        depth_ratio=0.5,
    )
    print(f"  - Synthesized context length: {len(context.split())} words, needle embedded at depth {depth}")

    print("=" * 60)
    print("All smoke checks PASSED successfully! Environment is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
