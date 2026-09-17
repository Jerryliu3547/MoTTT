"""Dynamic Test-Time LoRA Scratchpad for episodic memory ingestion."""

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer with unit scaling multiplier.

    h = W_0 x + (alpha / r) * B A x
    With r = 16, alpha = 16, gamma = 1.0 for stable test-time inner loop gradient propagation.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 1.0 when alpha = rank = 16

        # Freeze base layer weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """Kaiming uniform for A, zero initialization for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


class TestTimeScratchpadLoRA(nn.Module):
    """Dynamic episodic memory adapter manager.

    Instantiates test-time LoRA parameters, executes inner-loop NLL adaptation on chunked contexts,
    and resets state between test instances.
    """

    __test__ = False

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        inner_lr: float = 1e-3,
        num_inner_steps: int = 1,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.adapters: nn.ModuleDict = nn.ModuleDict()

    def attach_to_module(self, name: str, linear_layer: nn.Linear) -> LoRALinear:
        """Attach a dynamic LoRA adapter to a specific target linear layer."""
        adapter = LoRALinear(
            base_layer=linear_layer,
            rank=self.rank,
            alpha=self.alpha,
        )
        self.adapters[name] = adapter
        return adapter

    def reset_all_adapters(self) -> None:
        """Reset dynamic LoRA parameters to zero-delta initial state."""
        for adapter in self.adapters.values():
            if isinstance(adapter, LoRALinear):
                adapter.reset_lora_parameters()

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return parameters that are updated during test-time inner loop."""
        params = []
        for adapter in self.adapters.values():
            if isinstance(adapter, LoRALinear):
                params.extend([adapter.lora_A, adapter.lora_B])
        return params

    def inner_step(
        self,
        loss: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Execute one inner loop test-time gradient step on chunked context loss."""
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # Fallback simple SGD update if no optimizer passed
            grads = torch.autograd.grad(
                loss,
                self.get_trainable_parameters(),
                retain_graph=False,
                create_graph=False,
            )
            with torch.no_grad():
                for param, grad in zip(self.get_trainable_parameters(), grads):
                    param.data -= self.inner_lr * grad
        return float(loss.detach().item())
