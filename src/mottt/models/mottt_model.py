"""Unified MoTTT Model: Base Model + Dynamic Scratchpad + Reasoning Experts + Router."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from mottt.models.router import QueryAwareRouter, AsymmetricBalancingLoss
from mottt.models.scratchpad import LoRALinear, TestTimeScratchpadLoRA
from mottt.models.molora import (
    MoLoRALinear,
    inject_molora_to_model,
    set_model_routing_gates,
    reset_model_scratchpad,
    get_model_scratchpad_parameters,
    get_model_reasoning_parameters,
    DEFAULT_TARGET_MODULES,
)


class ReasoningLoRAExpert(nn.Module):
    """Pre-trained offline LoRA reasoning expert module j in R = {1..E}."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Unit multiplier 1.0

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expert delta: (alpha / r) * B A x."""
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class MoTTTModel(nn.Module):
    """MoTTT Architecture Wrapper.

    Decouples factual episodic memory ingestion (Test-Time Scratchpad LoRA)
    from multi-step domain reasoning (Pre-trained Static LoRA Experts)
    using a Query-Aware MLP Router with Asymmetric Load Balancing.

    Supports two operation modes:
    1. Representation-level (all_linear=False): LoRA experts operate at the final hidden dimension.
    2. All-Linear Module (all_linear=True): LoRA experts are injected into all matching linear layers
       (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) across the backbone.
    """

    def __init__(
        self,
        hidden_dim: int = 896,
        num_reasoning_experts: int = 4,
        rank: int = 16,
        alpha: float = 16.0,
        lambda_bal: float = 0.01,
        temperature: float = 1.0,
        base_backbone: Optional[nn.Module] = None,
        all_linear: bool = False,
        target_modules: Optional[List[str]] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_experts = num_reasoning_experts
        self.rank = rank
        self.alpha = alpha
        self.lambda_bal = lambda_bal
        self.all_linear = all_linear
        self.target_modules = target_modules or DEFAULT_TARGET_MODULES

        # 1. Base model backbone (frozen)
        self.base_backbone = base_backbone or nn.Identity()
        self.injected_linear_names: List[str] = []

        if self.all_linear and not isinstance(self.base_backbone, nn.Identity):
            self.base_backbone, self.injected_linear_names = inject_molora_to_model(
                model=self.base_backbone,
                num_reasoning_experts=num_reasoning_experts,
                rank=rank,
                alpha=alpha,
                target_modules=self.target_modules,
            )

        # Determine number of transformer layers for layer-level gating
        if num_layers is not None:
            self.num_layers = num_layers
        elif self.all_linear and hasattr(self.base_backbone, "num_molora_layers"):
            self.num_layers = self.base_backbone.num_molora_layers
        elif hasattr(getattr(self.base_backbone, "config", None), "num_hidden_layers"):
            self.num_layers = self.base_backbone.config.num_hidden_layers
        else:
            self.num_layers = 1

        # 2. Dynamic Memory Scratchpad (Test-Time LoRA, index 0 in expert set)
        self.scratchpad_lora = ReasoningLoRAExpert(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rank=rank,
            alpha=alpha,
        )

        # 3. Static Pre-trained LoRA Reasoning Experts (indices 1..E)
        self.reasoning_experts = nn.ModuleList([
            ReasoningLoRAExpert(
                in_features=hidden_dim,
                out_features=hidden_dim,
                rank=rank,
                alpha=alpha,
            )
            for _ in range(num_reasoning_experts)
        ])

        # 4. Query-Aware MLP Router
        self.router = QueryAwareRouter(
            hidden_dim=hidden_dim,
            num_reasoning_experts=num_reasoning_experts,
            num_layers=self.num_layers,
            temperature=temperature,
        )

        # 5. Asymmetric Load Balancing Loss Criterion
        self.balance_criterion = AsymmetricBalancingLoss(
            num_reasoning_experts=num_reasoning_experts,
            num_layers=self.num_layers,
            lambda_bal=lambda_bal,
            temperature=temperature,
        )

    def reset_scratchpad(self) -> None:
        """Reset test-time dynamic scratchpad weights to zero delta across all layers."""
        self.scratchpad_lora.lora_A.data.normal_(0, 0.02)
        nn.init.kaiming_uniform_(self.scratchpad_lora.lora_A, a=5**0.5)
        nn.init.zeros_(self.scratchpad_lora.lora_B)

        if self.all_linear and self.injected_linear_names:
            reset_model_scratchpad(self.base_backbone)

    def set_routing_gates(self, gates: Optional[torch.Tensor]) -> None:
        """Set active routing gates on all injected linear modules in the backbone."""
        if self.all_linear and self.injected_linear_names:
            set_model_routing_gates(self.base_backbone, gates)

    def get_scratchpad_parameters(self) -> List[nn.Parameter]:
        """Get all dynamic scratchpad parameters across the entire model."""
        if self.all_linear and self.injected_linear_names:
            return get_model_scratchpad_parameters(self.base_backbone)
        return [self.scratchpad_lora.lora_A, self.scratchpad_lora.lora_B]

    def get_reasoning_parameters(self) -> List[nn.Parameter]:
        """Get all reasoning expert parameters across the entire model."""
        if self.all_linear and self.injected_linear_names:
            return get_model_reasoning_parameters(self.base_backbone)
        params: List[nn.Parameter] = []
        for exp in self.reasoning_experts:
            params.extend([exp.lora_A, exp.lora_B])
        return params

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through MoTTT.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim] base model token representations
            query_embedding: [batch_size, hidden_dim] pooled user query representation

        Returns:
            blended_output: [batch_size, seq_len, hidden_dim]
            gates: [batch_size, seq_len, 1 + E] routing probabilities
            router_logits: [batch_size, seq_len, 1 + E] pre-softmax logits
        """
        # Query-Aware Routing gates: gates[:, :, 0] is Scratchpad, 1..E are Reasoning
        gates, router_logits = self.router(hidden_states, query_embedding)

        if self.all_linear and self.injected_linear_names:
            # Set gates on all backbone linear modules
            self.set_routing_gates(gates)
            blended = self.base_backbone(hidden_states)
            return blended, gates, router_logits

        # Default representation-level blending
        base_h = self.base_backbone(hidden_states)

        # Dynamic Scratchpad forward pass (Expert 0)
        delta_pad = self.scratchpad_lora(base_h)  # [batch, seq, hidden_dim]
        pad_weight = gates[..., 0:1]  # [batch, seq, 1]

        # Accumulate blended output starting with base + scratchpad
        blended = base_h + pad_weight * delta_pad

        # Add Reasoning LoRA expert contributions (Experts 1..E)
        for i, expert in enumerate(self.reasoning_experts):
            delta_exp = expert(base_h)
            expert_weight = gates[..., (i + 1) : (i + 2)]
            blended = blended + expert_weight * delta_exp

        return blended, gates, router_logits

    def compute_auxiliary_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute the asymmetric load balancing loss across reasoning experts."""
        return self.balance_criterion(router_logits)
