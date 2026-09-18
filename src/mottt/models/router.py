"""Query-Aware MLP Router with Asymmetric Load Balancing for MoTTT."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryAwareRouter(nn.Module):
    """Query-Aware Router that gates between dynamic Scratchpad (C={0}) and Reasoning Experts (R={1..E}).

    Conditioned on both the current token representation h_t and the global pooled embedding of the query.
    """

    def __init__(
        self,
        hidden_dim: int = 896,
        num_reasoning_experts: int = 4,
        num_layers: int = 1,
        router_hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_experts = num_reasoning_experts
        self.num_total_experts = 1 + num_reasoning_experts  # Expert 0 is Scratchpad, 1..E are Reasoning
        self.num_layers = num_layers
        self.temperature = temperature

        mid_dim = router_hidden_dim or hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_layers * self.num_total_experts),
        )

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing logits and gate probabilities across all layers.

        Args:
            token_hidden_states: [batch_size, seq_len, hidden_dim] or [M, hidden_dim]
            query_embedding: [batch_size, hidden_dim] global pooled query representation

        Returns:
            gates: [..., 1 + E] if num_layers==1 else [..., num_layers, 1 + E]
            logits: [..., 1 + E] if num_layers==1 else [..., num_layers, 1 + E]
        """
        orig_shape = token_hidden_states.shape
        if token_hidden_states.dim() == 3:
            batch_size, seq_len, _ = orig_shape
            # Broadcast query embedding across sequence length
            query_expanded = query_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([token_hidden_states, query_expanded], dim=-1)
        elif token_hidden_states.dim() == 2:
            # Flattened token format [M, hidden_dim]
            # query_embedding assumed to match or broadcast
            if query_embedding.shape[0] != token_hidden_states.shape[0]:
                raise ValueError("When token_hidden_states is 2D, query_embedding must match outer dimension")
            combined = torch.cat([token_hidden_states, query_embedding], dim=-1)
        else:
            raise ValueError(f"Unsupported token_hidden_states dimension: {token_hidden_states.dim()}")

        # Ensure combined tensor matches router MLP weights dtype
        mlp_dtype = self.mlp[0].weight.dtype
        if combined.dtype != mlp_dtype:
            combined = combined.to(dtype=mlp_dtype)

        logits = self.mlp(combined)

        if self.num_layers > 1:
            lead_shape = logits.shape[:-1]
            logits = logits.view(*lead_shape, self.num_layers, self.num_total_experts)

        gates = F.softmax(logits / self.temperature, dim=-1)
        if gates.dtype != token_hidden_states.dtype:
            gates = gates.to(dtype=token_hidden_states.dtype)
        return gates, logits


class AsymmetricBalancingLoss(nn.Module):
    """Asymmetric Load Balancing Loss over Reasoning Experts R = {1..E}.

    Scratchpad C = {0} is excluded from the entropy penalty, leaving its utilization
    strictly driven by task loss gradients. Supports both global and layer-level logits.
    """

    def __init__(
        self,
        num_reasoning_experts: int = 4,
        num_layers: int = 1,
        lambda_bal: float = 0.01,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_reasoning_experts = num_reasoning_experts
        self.num_total_experts = 1 + num_reasoning_experts
        self.num_layers = num_layers
        self.lambda_bal = lambda_bal
        self.temperature = temperature
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate asymmetric balancing loss from pre-softmax router logits.

        Args:
            logits: [*batch_dims, 1 + E] or [*batch_dims, num_layers, 1 + E]

        Returns:
            Scalar tensor: lambda_bal * L_balance_reason
        """
        has_layers = (logits.dim() == 4) or (
            self.num_layers > 1 and logits.dim() == 3 and logits.shape[-2] == self.num_layers
        )

        # Case A: Layer-level logits [..., num_layers, 1 + E]
        if has_layers:
            num_l = logits.shape[-2]
            flat_logits = logits.reshape(-1, num_l, self.num_total_experts)  # [M, L, 1 + E]
            reasoning_logits = flat_logits[:, :, 1:]  # [M, L, E]
            p_reason = F.softmax(reasoning_logits / self.temperature, dim=-1)  # [M, L, E]
            q_reason = torch.mean(p_reason, dim=0)  # [L, E]
            loss_balance_reason = -torch.sum(torch.log(q_reason + self.eps), dim=-1).mean()
            return self.lambda_bal * loss_balance_reason

        # Case B: Standard global logits [*batch_dims, 1 + E]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        reasoning_logits = flat_logits[:, 1:]  # [M, E]

        p_reason = F.softmax(reasoning_logits / self.temperature, dim=-1)  # [M, E]
        q_reason = torch.mean(p_reason, dim=0)  # [E]
        loss_balance_reason = -torch.sum(torch.log(q_reason + self.eps))

        return self.lambda_bal * loss_balance_reason
