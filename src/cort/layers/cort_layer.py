"""
CoRT Layer: Complete transformer layer with coherence-based routing.

This is the main building block of CoRT, combining:
    1. CoherentRouter: Decides which tokens need attention vs mixing
    2. MultiHeadAttention: Full attention for low-coherence tokens
    3. CoherenceMixer: Lightweight mixing for high-coherence tokens
    4. Feedforward: Standard FFN for all tokens

Architecture:
    Input → LayerNorm → Router → [Attention OR Mixer] → Merge → LayerNorm → FFN → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .router import CoherentRouter, AdaptiveCoherentRouter, RoutingConfig
from .mixer import CoherenceMixer
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Standard feedforward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoRTLayer(nn.Module):
    """
    Single CoRT transformer layer with coherence-based routing.

    Routes tokens between full attention and lightweight mixing based
    on their phase coherence scores. Low-coherence tokens (complex
    relationships) get attention; high-coherence tokens (aligned)
    get efficient mixing.

    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout: Dropout probability
        route_frac: Fraction of tokens for attention
        adaptive_routing: Use PID-tuned adaptive routing
        layer_norm_eps: LayerNorm epsilon

    Example:
        >>> layer = CoRTLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> hidden = torch.randn(2, 128, 512)
        >>> output, stats = layer(hidden)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        route_frac: float = 0.15,
        adaptive_routing: bool = True,
        coherence_mode: str = "combined",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Router
        router_config = RoutingConfig(
            route_frac=route_frac,
            coherence_mode=coherence_mode,
        )

        if adaptive_routing:
            self.router = AdaptiveCoherentRouter(router_config, d_model)
        else:
            self.router = CoherentRouter(router_config, d_model)

        # Attention for low-coherence tokens
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Mixer for high-coherence tokens
        self.mixer = CoherenceMixer(d_model)

        # Feedforward
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with coherence-based routing.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            return_stats: Return routing statistics

        Returns:
            output: [batch, seq_len, d_model]
            stats: Optional dict with routing statistics
        """
        B, L, D = hidden_states.shape

        # Pre-norm
        normed = self.norm1(hidden_states)

        # Route tokens
        attn_mask, mix_mask, routing_scores = self.router(normed, attention_mask)

        # Process attention path
        attn_output = self._attention_path(normed, attn_mask, attention_mask)

        # Process mixing path
        mix_output = self._mixing_path(normed, mix_mask, routing_scores)

        # Merge paths
        merged = torch.where(
            attn_mask.unsqueeze(-1).expand_as(hidden_states),
            attn_output,
            mix_output,
        )

        # Residual connection
        hidden_states = hidden_states + self.dropout(merged)

        # FFN with pre-norm and residual
        hidden_states = hidden_states + self.dropout(self.ffn(self.norm2(hidden_states)))

        # Collect stats if requested
        stats = None
        if return_stats:
            stats = {
                "attn_ratio": attn_mask.float().mean().item(),
                "mix_ratio": mix_mask.float().mean().item(),
                "mean_coherence": routing_scores.mean().item(),
            }

        return hidden_states, stats

    def _attention_path(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply attention to selected tokens."""
        # For simplicity, apply attention to all and let the merge select
        # In production, you'd want sparse attention for efficiency
        output, _ = self.attention(hidden_states, attention_mask)
        return output

    def _mixing_path(
        self,
        hidden_states: torch.Tensor,
        mix_mask: torch.Tensor,
        coherence_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mixing to selected tokens."""
        # Apply mixer with coherence weighting
        output = self.mixer(hidden_states, coherence_scores)
        return output
