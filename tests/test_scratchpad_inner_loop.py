"""Tests for TestTimeScratchpadLoRA and inner-loop adaptation."""

import torch
import torch.nn as nn
from mottt.models.scratchpad import LoRALinear, TestTimeScratchpadLoRA


def test_lora_linear_unit_scaling():
    """Verify that rank 16 and alpha 16 produce an exact unit scaling multiplier."""
    in_dim, out_dim = 32, 32
    base = nn.Linear(in_dim, out_dim)
    lora = LoRALinear(base, rank=16, alpha=16.0)

    assert lora.scaling == 1.0

    # At initialization, B is zeroed out, so lora output == base output
    x = torch.randn(4, in_dim)
    with torch.no_grad():
        out_lora = lora(x)
        out_base = base(x)
    assert torch.allclose(out_lora, out_base, atol=1e-6)


def test_test_time_scratchpad_inner_step():
    """Verify that inner-loop gradient steps successfully reduce test loss on chunked data."""
    in_dim, out_dim = 16, 16
    base = nn.Linear(in_dim, out_dim)
    scratchpad = TestTimeScratchpadLoRA(rank=8, alpha=8.0, inner_lr=0.05)
    adapted_layer = scratchpad.attach_to_module("test_layer", base)

    # Synthetic chunk of tokens
    x = torch.randn(8, in_dim)
    target = torch.randn(8, out_dim)

    # Initial loss
    initial_out = adapted_layer(x)
    initial_loss = nn.functional.mse_loss(initial_out, target)

    # Take an inner-loop adaptation step
    optimizer = torch.optim.SGD(scratchpad.get_trainable_parameters(), lr=0.1)
    loss_val = scratchpad.inner_step(initial_loss, optimizer=optimizer)

    # Adapted loss
    updated_out = adapted_layer(x)
    updated_loss = nn.functional.mse_loss(updated_out, target)

    assert updated_loss.item() < initial_loss.item(), "Inner step should decrease loss on the chunk."

    # Test reset restores zero-delta state
    scratchpad.reset_all_adapters()
    reset_out = adapted_layer(x)
    assert torch.allclose(reset_out, base(x), atol=1e-6), "Reset should zero out delta B."
