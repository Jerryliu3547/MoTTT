"""End-to-end integration test of MoTTTModel with mock tensors."""

import torch
import torch.nn as nn
from mottt.models.mottt_model import MoTTTModel
from mottt.models.molora import MoLoRALinear


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


def test_molora_linear_forward_and_reset():
    in_features = 32
    out_features = 48
    rank = 4
    num_experts = 3

    base_layer = nn.Linear(in_features, out_features)
    molora = MoLoRALinear(
        base_layer=base_layer,
        num_reasoning_experts=num_experts,
        rank=rank,
        alpha=4.0,
    )

    # Base layer must be frozen
    assert not molora.base_layer.weight.requires_grad

    # Initial scratchpad B is zero -> delta is 0
    x = torch.randn(2, 5, in_features)
    base_out = molora.base_layer(x)
    out_init = molora(x)
    assert torch.allclose(base_out, out_init, atol=1e-6)

    # After perturbing scratchpad, forward should differ
    molora.scratchpad_lora_B.data.fill_(0.1)
    out_adapted = molora(x)
    assert not torch.allclose(base_out, out_adapted, atol=1e-5)

    # Calling reset_scratchpad restores zero delta
    molora.reset_scratchpad()
    out_reset = molora(x)
    assert torch.allclose(base_out, out_reset, atol=1e-6)

    # Test with gates: [batch, seq, 1 + E]
    gates = torch.zeros(2, 5, 1 + num_experts)
    gates[..., 1] = 1.0  # 100% to expert 1
    molora.reasoning_lora_B[0].data.fill_(0.2)
    out_expert = molora(x, gates=gates)
    assert not torch.allclose(base_out, out_expert, atol=1e-5)


def test_mottt_model_all_linear_injection_and_forward():
    class ToyTransformerBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.o_proj = nn.Linear(dim, dim)
            self.gate_proj = nn.Linear(dim, dim * 2)
            self.up_proj = nn.Linear(dim, dim * 2)
            self.down_proj = nn.Linear(dim * 2, dim)

        def forward(self, x):
            attn = self.o_proj(self.v_proj(self.q_proj(x)))
            mlp = self.down_proj(self.gate_proj(x) * self.up_proj(x))
            return x + attn + mlp

    dim = 32
    toy_backbone = ToyTransformerBlock(dim)

    model = MoTTTModel(
        hidden_dim=dim,
        num_reasoning_experts=4,
        rank=4,
        alpha=4.0,
        base_backbone=toy_backbone,
        all_linear=True,
    )

    # Verify all 7 linear layers are wrapped
    assert len(model.injected_linear_names) == 7
    for name in model.injected_linear_names:
        submod = dict(model.base_backbone.named_modules())[name]
        assert isinstance(submod, MoLoRALinear)

    # Test forward pass with query routing
    x = torch.randn(2, 6, dim)
    q = torch.randn(2, dim)
    blended, gates, logits = model(x, q)

    assert blended.shape == (2, 6, dim)
    assert gates.shape == (2, 6, 5)

    # Test scratchpad parameter accessor vs reasoning parameter accessor
    scratchpad_params = model.get_scratchpad_parameters()
    reasoning_params = model.get_reasoning_parameters()

    # 7 modules * 2 parameters (A, B) = 14 scratchpad params
    assert len(scratchpad_params) == 7 * 2
    # 7 modules * 4 experts * 2 params = 56 reasoning params
    assert len(reasoning_params) == 7 * 4 * 2

    # Reset scratchpad test: fill B matrices and assert they reset to 0
    for name in model.injected_linear_names:
        submod = dict(model.base_backbone.named_modules())[name]
        submod.scratchpad_lora_B.data.fill_(0.5)

    model.reset_scratchpad()

    for name in model.injected_linear_names:
        submod = dict(model.base_backbone.named_modules())[name]
        assert torch.all(submod.scratchpad_lora_B == 0.0)

