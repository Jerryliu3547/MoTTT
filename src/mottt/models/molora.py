"""Mixture-of-LoRA (MoLoRA) Linear Layer and Model Injection.

Wraps target linear layers across all transformer blocks with:
1. Dynamic Scratchpad LoRA (Expert 0): Test-time episodic memory adapted on context tokens.
2. Pre-trained Reasoning LoRA Experts (Experts 1..E): Specialized multi-step arithmetic reasoning.
3. Query-Aware Gate Blending: Integrates with the Query-Aware Router.
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple, Union
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
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_reasoning_experts = num_reasoning_experts
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 1.0 when alpha = rank = 16
        self.layer_idx = layer_idx

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
        A_pad = self.scratchpad_lora_A.to(dtype=x.dtype) if self.scratchpad_lora_A.dtype != x.dtype else self.scratchpad_lora_A
        B_pad = self.scratchpad_lora_B.to(dtype=x.dtype) if self.scratchpad_lora_B.dtype != x.dtype else self.scratchpad_lora_B
        scratchpad_delta = (dropped_x @ A_pad.T @ B_pad.T) * self.scaling

        if effective_gates is None:
            return base_out + scratchpad_delta

        # Mode B: Query-Aware Blended Routing
        if effective_gates.dtype != x.dtype:
            effective_gates = effective_gates.to(dtype=x.dtype)

        g_pad = effective_gates[..., 0:1]
        blended_out = base_out + g_pad * scratchpad_delta

        for i, (A, B) in enumerate(zip(self.reasoning_lora_A, self.reasoning_lora_B)):
            g_exp = effective_gates[..., (i + 1) : (i + 2)]
            A_exp = A.to(dtype=x.dtype) if A.dtype != x.dtype else A
            B_exp = B.to(dtype=x.dtype) if B.dtype != x.dtype else B
            exp_delta = (dropped_x @ A_exp.T @ B_exp.T) * self.scaling
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
    layer_indices: List[int] = []

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
                    # Detect layer index from path (e.g. model.layers.7.self_attn.q_proj)
                    layer_match = re.search(r"(?:layers|layer|blocks|block|h)\.(\d+)", full_child_name)
                    layer_idx = int(layer_match.group(1)) if layer_match else 0
                    layer_indices.append(layer_idx)

                    molora_layer = MoLoRALinear(
                        base_layer=child_module,
                        num_reasoning_experts=num_reasoning_experts,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        layer_idx=layer_idx,
                    )
                    # Preserve device and dtype
                    molora_layer = molora_layer.to(
                        device=child_module.weight.device,
                        dtype=child_module.weight.dtype,
                    )
                    setattr(module, child_name, molora_layer)
                    injected_names.append(full_child_name)

    model.num_molora_layers = (max(layer_indices) + 1) if layer_indices else 1
    return model, injected_names


def set_model_routing_gates(model: nn.Module, gates: Optional[torch.Tensor]) -> None:
    """Set the active routing gates on all MoLoRALinear layers in model.

    Supports:
    - None: resets active gates (100% scratchpad path).
    - Global gates: [..., 1 + E] broadcast to all layers.
    - Layer-level gates: [..., num_layers, 1 + E], slices gates[..., layer_idx, :] for each layer.
    """
    molora_modules = [m for m in model.modules() if isinstance(m, MoLoRALinear)]
    if not molora_modules:
        return

    if gates is None:
        for m in molora_modules:
            m.set_active_gates(None)
        return

    max_layer_idx = max(m.layer_idx for m in molora_modules)
    num_layers_in_model = max_layer_idx + 1

    # Detect if gates tensor contains a layer dimension at dim=-2
    has_layer_dim = False
    if gates.dim() == 4:
        has_layer_dim = True
    elif gates.dim() == 3 and num_layers_in_model > 1 and gates.shape[-2] == num_layers_in_model:
        has_layer_dim = True

    if has_layer_dim:
        num_gate_layers = gates.shape[-2]
        for m in molora_modules:
            l_idx = min(m.layer_idx, num_gate_layers - 1)
            layer_gate = gates[..., l_idx, :]
            m.set_active_gates(layer_gate)
    else:
        for m in molora_modules:
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
