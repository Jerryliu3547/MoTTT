"""Tests for QueryAwareRouter and AsymmetricBalancingLoss."""

import torch
import pytest
from mottt.models.router import QueryAwareRouter, AsymmetricBalancingLoss


def test_router_forward_shape_and_simplex():
    batch_size = 2
    seq_len = 8
    hidden_dim = 64
    num_reasoning_experts = 4
    total_experts = 1 + num_reasoning_experts

    router = QueryAwareRouter(
        hidden_dim=hidden_dim,
        num_reasoning_experts=num_reasoning_experts,
    )

    token_hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    query_embedding = torch.randn(batch_size, hidden_dim)

    gates, logits = router(token_hidden_states, query_embedding)

    # Check output shapes
    assert gates.shape == (batch_size, seq_len, total_experts)
    assert logits.shape == (batch_size, seq_len, total_experts)

    # Gates must lie on the simplex (non-negative, sum to 1.0 along last dim)
    assert torch.all(gates >= 0.0)
    sums = torch.sum(gates, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_asymmetric_balancing_loss_frees_scratchpad():
    """Verify that Scratchpad (index 0) has ZERO gradient under AsymmetricBalancingLoss."""
    num_reasoning_experts = 4
    total_experts = 1 + num_reasoning_experts
    loss_fn = AsymmetricBalymmetricLoss = AsymmetricBalancingLoss(
        num_reasoning_experts=num_reasoning_experts,
        lambda_bal=0.01,
    )

    # Pre-softmax logits with requires_grad=True
    logits = torch.randn(10, 16, total_experts, requires_grad=True)

    loss = loss_fn(logits)
    loss.backward()

    assert logits.grad is not None
    # Index 0 is the dynamic Scratchpad: its gradient from asymmetric balancing loss MUST be 0!
    scratchpad_grad = logits.grad[..., 0]
    assert torch.allclose(scratchpad_grad, torch.zeros_like(scratchpad_grad), atol=1e-7), (
        "Scratchpad (expert 0) received non-zero gradient from AsymmetricBalancingLoss!"
    )

    # Indices 1..E are Reasoning experts: they should have non-zero gradients
    reasoning_grad = logits.grad[..., 1:]
    assert not torch.allclose(reasoning_grad, torch.zeros_like(reasoning_grad), atol=1e-7), (
        "Reasoning experts should receive non-zero balancing gradients."
    )


def test_asymmetric_balancing_loss_uniform_minimum():
    """Verify that perfectly balanced reasoning logits yield minimal loss."""
    num_reasoning_experts = 4
    total_experts = 1 + num_reasoning_experts
    loss_fn = AsymmetricBalancingLoss(
        num_reasoning_experts=num_reasoning_experts,
        lambda_bal=1.0,
    )

    # Case A: Perfectly balanced reasoning logits (all equal)
    balanced_logits = torch.zeros(100, total_experts)
    balanced_loss = loss_fn(balanced_logits).item()

    # Theoretical minimum: -E * log(1/E) = E * log(E)
    expected_min = num_reasoning_experts * torch.log(torch.tensor(float(num_reasoning_experts))).item()
    assert pytest.approx(balanced_loss, rel=1e-4) == expected_min

    # Case B: Heavily collapsed routing (one reasoning expert dominates)
    collapsed_logits = torch.zeros(100, total_experts)
    collapsed_logits[:, 1] = 50.0  # Expert 1 gets nearly 100% allocation
    collapsed_loss = loss_fn(collapsed_logits).item()

    # Collapsed loss should be substantially higher than balanced loss
    assert collapsed_loss > balanced_loss * 2.0
