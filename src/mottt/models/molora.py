"""Mixture-of-LoRA (MoLoRA) Linear Layer and Model Injection.

Wraps target linear layers across all transformer blocks with:
1. Dynamic Scratchpad LoRA (Expert 0): Test-time episodic memory adapted on context tokens.
2. Pre-trained Reasoning LoRA Experts (Experts 1..E): Specialized multi-step arithmetic reasoning.
3. Query-Aware Gate Blending: Integrates with the Query-Aware Router.
"""

import math
from typing import Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class MoLoRALinear(nn.Module):
    """Linear layer wrapped with 1 Dynamic Scratchpad LoRA and E Reasoning LoRA Experts.

    Forward computation:
        y = W_0 x + g_pad * (alpha/r) * B_pad A_pad x + sum_{i=1}^E g_i * (alpha/r) * B_i A_i x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_reasoning_experts: int = 4,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_reasoning_experts = num_reasoning_experts
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 1.0 when alpha = rank = 16

        # Freeze base layer weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # 1. Dynamic Scratchpad LoRA (Expert 0)
        self.scratchpad_lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.scratchpad_lora_B = nn.Parameter(torch.empty(self.out_features, rank))

        # 2. Static Reasoning LoRA Experts (Experts 1..E)
        self.reasoning_lora_A = nn.ParameterList([
            nn.Parameter(torch.empty(rank, self.in_features))
            for _ in range(num_reasoning_experts)
        ])
        self.reasoning_lora_B = nn.ParameterList([
            nn.Parameter(torch.empty(self.out_features, rank))
            for _ in range(num_reasoning_experts)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.active_gates: Optional[torch.Tensor] = None

        self.reset_parameters()

    def reset_scratchpad(self) -> None:
        """Reset dynamic scratchpad LoRA to zero-delta initial state."""
        nn.init.kaiming_uniform_(self.scratchpad_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.scratchpad_lora_B)

    def reset_parameters(self) -> None:
        """Initialize all LoRA weights."""
        self.reset_scratchpad()
        for A, B in zip(self.reasoning_lora_A, self.reasoning_lora_B):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)

    def set_active_gates(self, gates: Optional[torch.Tensor]) -> None:
        """Set active routing gates for subsequent forward passes."""
        self.active_gates = gates

    def get_scratchpad_parameters(self) -> List[nn.Parameter]:
        """Return parameters for the dynamic scratchpad LoRA."""
        return [self.scratchpad_lora_A, self.scratchpad_lora_B]

    def get_reasoning_parameters(self) -> List[nn.Parameter]:
        """Return parameters for all reasoning expert LoRAs."""
        params: List[nn.Parameter] = []
        for A, B in zip(self.reasoning_lora_A, self.reasoning_lora_B):
            params.extend([A, B])
        return params

    def forward(
        self,
        x: torch.Tensor,
        gates: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through base linear layer + active LoRA pathways.

        Args:
            x: Input tensor [..., in_features]
            gates: Optional routing gate tensor [..., 1 + E] or [batch_size, 1, 1 + E].
                   If None, uses self.active_gates.
                   If both are None (e.g. during standalone inner-loop adaptation),
                   defaults to only the scratchpad LoRA pathway.
        """
        base_out = self.base_layer(x)
        effective_gates = gates if gates is not None else self.active_gates

        dropped_x = self.dropout(x)

        # Mode A: Context Adaptation (No gates set -> 100% scratchpad path)
        if effective_gates is None:
            scratchpad_delta = (dropped_x @ self.scratchpad_lora_A.T @ self.scratchpad_lora_B.T) * self.scaling
            return base_out + scratchpad_delta

        # Mode B: Query-Aware Blended Routing
        # effective_gates: [..., 1 + E]
        g_pad = effective_gates[..., 0:1]
        scratchpad_delta = (dropped_x @ self.scratchpad_lora_A.T @ self.scratchpad_lora_B.T) * self.scaling
        blended_out = base_out + g_pad * scratchpad_delta

        for i, (A, B) in enumerate(zip(self.reasoning_lora_A, self.reasoning_lora_B)):
            g_exp = effective_gates[..., (i + 1) : (i + 2)]
            exp_delta = (dropped_x @ A.T @ B.T) * self.scaling
            blended_out = blended_out + g_exp * exp_delta

        return blended_out


def inject_molora_to_model(
    model: nn.Module,
    num_reasoning_experts: int = 4,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, List[str]]:
    """Recursively replace matching linear layers in model with MoLoRALinear.

    Args:
        model: Base PyTorch model (e.g. Qwen2ForCausalLM).
        num_reasoning_experts: Number of reasoning LoRA experts E.
        rank: LoRA rank r.
        alpha: LoRA alpha scaling parameter.
        dropout: LoRA dropout probability.
        target_modules: Suffix names of linear modules to wrap.

    Returns:
        model: Model with replaced linear modules.
        injected_names: List of names of all replaced modules.
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    target_set: Set[str] = set(target_modules)
    injected_names: List[str] = []

    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Linear):
                # Check if child module name matches any target suffix
                is_target = any(
                    child_name == target or child_name.endswith(f".{target}")
                    for target in target_set
                )
                if is_target:
                    full_child_name = f"{name}.{child_name}" if name else child_name
                    molora_layer = MoLoRALinear(
                        base_layer=child_module,
                        num_reasoning_experts=num_reasoning_experts,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    # Preserve device and dtype
                    molora_layer = molora_layer.to(
                        device=child_module.weight.device,
                        dtype=child_module.weight.dtype,
                    )
                    setattr(module, child_name, molora_layer)
                    injected_names.append(full_child_name)

    return model, injected_names


def set_model_routing_gates(model: nn.Module, gates: Optional[torch.Tensor]) -> None:
    """Set the active routing gates on all MoLoRALinear layers in model."""
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            m.set_active_gates(gates)


def reset_model_scratchpad(model: nn.Module) -> None:
    """Reset the dynamic scratchpad LoRA across all MoLoRALinear layers in model."""
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            m.reset_scratchpad()


def get_model_scratchpad_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return all dynamic scratchpad parameters across all MoLoRALinear layers."""
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            params.extend(m.get_scratchpad_parameters())
    return params


def get_model_reasoning_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return all reasoning expert parameters across all MoLoRALinear layers."""
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, MoLoRALinear):
            params.extend(m.get_reasoning_parameters())
    return params
