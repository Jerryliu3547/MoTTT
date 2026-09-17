"""End-to-end integration test of MoTTTModel with mock tensors."""

import torch
import torch.nn as nn
from mottt.models.mottt_model import MoTTTModel


def test_mottt_model_forward_and_loss():
    batch_size = 2
    seq_len = 8
    hidden_dim = 64
    num_reasoning_experts = 4

    # Use a small linear layer as mock backbone
    mock_backbone = nn.Linear(hidden_dim, hidden_dim)

    model = MoTTTModel(
        hidden_dim=hidden_dim,
        num_reasoning_experts=num_reasoning_experts,
        rank=8,
        alpha=8.0,
        lambda_bal=0.01,
        base_backbone=mock_backbone,
    )

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    query_embedding = torch.randn(batch_size, hidden_dim)

    # Forward pass
    blended, gates, router_logits = model(hidden_states, query_embedding)

    assert blended.shape == (batch_size, seq_len, hidden_dim)
    assert gates.shape == (batch_size, seq_len, 1 + num_reasoning_experts)
    assert router_logits.shape == (batch_size, seq_len, 1 + num_reasoning_experts)

    # Check auxiliary asymmetric load balancing loss
    aux_loss = model.compute_auxiliary_loss(router_logits)
    assert aux_loss.dim() == 0  # Scalar
    assert aux_loss.item() > 0.0

    # Ensure backprop works through outer loss
    dummy_target = torch.randn_like(blended)
    task_loss = nn.functional.mse_loss(blended, dummy_target)
    total_loss = task_loss + aux_loss
    total_loss.backward()

    # Verify router received gradients
    for p in model.router.parameters():
        if p.requires_grad:
            assert p.grad is not None
