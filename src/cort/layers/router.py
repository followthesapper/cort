"""
CoRT Router: Phase coherence-based token routing.

The router determines which tokens need full attention versus lightweight
mixing based on their phase coherence (R-bar) scores.

Routing Logic:
    - Low coherence tokens → Full attention (need complex relationships)
    - High coherence tokens → Mixing (already aligned, cheap to process)

This is the key insight that enables CoRT's efficiency: most tokens in
natural language are contextually coherent and don't need expensive
attention computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from ..core.coherence import (
    compute_phase_coherence,
    compute_local_coherence,
    compute_cross_coherence,
)


@dataclass
class RoutingConfig:
    """Configuration for coherent routing."""

    route_frac: float = 0.15
    r_bar_threshold: float = 0.5
    entropy_weight: float = 0.4
    coherence_weight: float = 0.4
    learned_weight: float = 0.2
    coherence_mode: str = "combined"


class CoherentRouter(nn.Module):
    """
    Routes tokens based on phase coherence metrics.

    Computes a routing score for each token combining:
        1. Entropy-based importance (from hidden states)
        2. Phase coherence (R-bar)
        3. Learned routing weights

    Tokens with low routing scores go to attention, high scores go to mixing.

    Args:
        config: Routing configuration
        d_model: Hidden dimension size

    Example:
        >>> router = CoherentRouter(RoutingConfig(), d_model=512)
        >>> hidden = torch.randn(2, 128, 512)
        >>> attn_mask, mix_mask, scores = router(hidden)
    """

    def __init__(self, config: RoutingConfig, d_model: int):
        super().__init__()

        self.config = config
        self.d_model = d_model

        # Learned routing projection
        self.route_proj = nn.Linear(d_model, 1)

        # Coherence window for local computation
        self.coherence_window = 8

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing masks based on coherence.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional [batch, seq_len] mask

        Returns:
            attn_indices: Indices of tokens for attention
            mix_indices: Indices of tokens for mixing
            routing_scores: [batch, seq_len] routing scores
        """
        B, L, D = hidden_states.shape

        # Compute routing components
        scores = self._compute_routing_scores(hidden_states)

        # Determine number of tokens for attention
        n_attn = max(1, int(L * self.config.route_frac))

        # Select tokens with lowest scores for attention (need most processing)
        _, attn_indices = torch.topk(scores, n_attn, dim=-1, largest=False)

        # Create masks
        attn_mask = torch.zeros(B, L, device=hidden_states.device, dtype=torch.bool)
        attn_mask.scatter_(1, attn_indices, True)
        mix_mask = ~attn_mask

        return attn_mask, mix_mask, scores

    def _compute_routing_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute combined routing scores."""
        B, L, D = hidden_states.shape
        device = hidden_states.device

        scores = torch.zeros(B, L, device=device)

        # 1. Entropy-based component
        if self.config.entropy_weight > 0:
            # Use variance as proxy for information content
            variance = hidden_states.var(dim=-1)
            entropy_score = torch.sigmoid(variance)
            scores += self.config.entropy_weight * entropy_score

        # 2. Coherence component
        if self.config.coherence_weight > 0:
            coherence = self._compute_coherence(hidden_states)
            scores += self.config.coherence_weight * coherence

        # 3. Learned component
        if self.config.learned_weight > 0:
            learned = torch.sigmoid(self.route_proj(hidden_states).squeeze(-1))
            scores += self.config.learned_weight * learned

        return scores

    def _compute_coherence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute coherence based on mode."""
        mode = self.config.coherence_mode

        if mode == "token":
            return compute_phase_coherence(hidden_states, mode="token")
        elif mode == "local":
            return compute_local_coherence(hidden_states, window=self.coherence_window)
        elif mode == "cross":
            return compute_cross_coherence(hidden_states)
        elif mode == "combined":
            token_coh = compute_phase_coherence(hidden_states, mode="token")
            local_coh = compute_local_coherence(hidden_states, window=self.coherence_window)
            return 0.5 * token_coh + 0.5 * local_coh
        else:
            raise ValueError(f"Unknown coherence mode: {mode}")


class AdaptiveCoherentRouter(CoherentRouter):
    """
    Router with PID-tuned adaptive routing fraction.

    Dynamically adjusts the routing fraction based on observed coherence
    levels, allowing the model to adapt to different input characteristics.

    The PID controller targets a desired coherence level and adjusts
    route_frac to achieve it.

    Args:
        config: Routing configuration
        d_model: Hidden dimension size
        target_coherence: Target coherence level (default: 0.5)
        kp, ki, kd: PID controller gains
    """

    def __init__(
        self,
        config: RoutingConfig,
        d_model: int,
        target_coherence: float = 0.5,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
    ):
        super().__init__(config, d_model)

        self.target_coherence = target_coherence
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID state (not saved in state_dict)
        self.register_buffer("integral", torch.tensor(0.0))
        self.register_buffer("prev_error", torch.tensor(0.0))

        # Adaptive route fraction bounds
        self.min_route_frac = 0.05
        self.max_route_frac = 0.50

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with adaptive routing fraction."""
        B, L, D = hidden_states.shape

        # Compute current coherence
        coherence = self._compute_coherence(hidden_states)
        mean_coherence = coherence.mean()

        # PID update
        error = self.target_coherence - mean_coherence
        self.integral = self.integral + error
        derivative = error - self.prev_error
        self.prev_error = error

        # Compute adjustment
        adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Update route fraction
        route_frac = self.config.route_frac + adjustment.item()
        route_frac = max(self.min_route_frac, min(self.max_route_frac, route_frac))

        # Compute routing scores
        scores = self._compute_routing_scores(hidden_states)

        # Select tokens for attention
        n_attn = max(1, int(L * route_frac))
        _, attn_indices = torch.topk(scores, n_attn, dim=-1, largest=False)

        # Create masks
        attn_mask = torch.zeros(B, L, device=hidden_states.device, dtype=torch.bool)
        attn_mask.scatter_(1, attn_indices, True)
        mix_mask = ~attn_mask

        return attn_mask, mix_mask, scores
