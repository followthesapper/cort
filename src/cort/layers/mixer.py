"""
CoRT Mixer: Lightweight mixing for high-coherence tokens.

When tokens have high phase coherence (R-bar), they're already aligned
and don't need expensive attention. Instead, we use a simple learned
mixing operation that's much cheaper but still effective.

The mixer uses phase-aware combinations of neighboring tokens,
respecting the coherence structure rather than disrupting it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CoherenceMixer(nn.Module):
    """
    Phase-aware token mixing for high-coherence tokens.

    Instead of full attention O(n²), uses O(n) mixing operations
    that preserve phase relationships between aligned tokens.

    The mixing combines:
        1. Learned linear projections
        2. Phase-weighted gating
        3. Residual connections

    Args:
        d_model: Hidden dimension size
        mix_strength: Base mixing strength (0-1)
        learnable: Use learnable mixing weights

    Example:
        >>> mixer = CoherenceMixer(d_model=512)
        >>> hidden = torch.randn(2, 128, 512)
        >>> mixed = mixer(hidden)
    """

    def __init__(
        self,
        d_model: int,
        mix_strength: float = 0.5,
        learnable: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.mix_strength = mix_strength

        if learnable:
            # Learned mixing projections
            self.W1 = nn.Linear(d_model, d_model, bias=False)
            self.W2 = nn.Linear(d_model, d_model, bias=False)
            self.gate = nn.Linear(2 * d_model, 1)
        else:
            self.W1 = None
            self.W2 = None
            self.gate = None

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        coherence_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply phase-aware mixing to hidden states.

        Args:
            hidden_states: [batch, seq_len, d_model]
            coherence_scores: Optional [batch, seq_len] coherence weights

        Returns:
            Mixed hidden states [batch, seq_len, d_model]
        """
        B, L, D = hidden_states.shape

        if L < 2:
            return hidden_states

        # Compute phase-weighted mixing using vectorized operations
        mixed = self._vectorized_mix(hidden_states)

        # Apply coherence weighting if provided
        if coherence_scores is not None:
            # Higher coherence = more mixing
            weight = coherence_scores.unsqueeze(-1)
            mixed = weight * mixed + (1 - weight) * hidden_states

        # Output projection with residual
        output = self.out_proj(mixed)

        return output

    def _vectorized_mix(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Efficient vectorized mixing of token pairs."""
        B, L, D = hidden_states.shape

        # Ensure even length for pair processing
        L_even = L - (L % 2)

        if L_even < 2:
            return hidden_states

        # Extract even and odd position tokens
        h_even = hidden_states[:, 0:L_even:2, :]  # [B, L//2, D]
        h_odd = hidden_states[:, 1:L_even:2, :]  # [B, L//2, D]

        # Compute phase for gating
        phase_even = 2 * np.pi * torch.sigmoid(h_even)
        phase_odd = 2 * np.pi * torch.sigmoid(h_odd)
        phase_diff = phase_even - phase_odd

        # Phase coherence between pairs
        cos_diff = torch.cos(phase_diff).mean(dim=-1, keepdim=True)
        pair_coherence = torch.abs(cos_diff)

        if self.W1 is not None:
            # Learned mixing
            h1 = self.W1(h_even)
            h2 = self.W2(h_odd)

            # Gated combination
            combined = torch.cat([h_even, h_odd], dim=-1)
            gate = torch.sigmoid(self.gate(combined))

            mixed_even = gate * h1 + (1 - gate) * h_even
            mixed_odd = gate * h2 + (1 - gate) * h_odd
        else:
            # Simple averaging weighted by coherence
            mixed_even = pair_coherence * (h_even + h_odd) / 2 + (1 - pair_coherence) * h_even
            mixed_odd = pair_coherence * (h_even + h_odd) / 2 + (1 - pair_coherence) * h_odd

        # Interleave back
        mixed = hidden_states.clone()
        mixed[:, 0:L_even:2, :] = mixed_even
        mixed[:, 1:L_even:2, :] = mixed_odd

        return mixed


class SimpleMixer(nn.Module):
    """
    Simplified mixer using only linear projections.

    A lighter alternative to CoherenceMixer when maximum speed
    is needed and phase-awareness is less critical.

    Args:
        d_model: Hidden dimension size
        expansion: Expansion factor for hidden layer
    """

    def __init__(self, d_model: int, expansion: float = 2.0):
        super().__init__()

        hidden_dim = int(d_model * expansion)

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply simple MLP mixing."""
        return self.net(hidden_states)
