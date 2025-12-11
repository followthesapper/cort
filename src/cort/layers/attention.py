"""
CoRT Attention: Standard multi-head attention for low-coherence tokens.

Tokens with low phase coherence need full attention to resolve their
complex relationships. This module provides standard scaled dot-product
attention, used only for the subset of tokens selected by the router.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Used for tokens that the router determines need full attention
    (low coherence tokens with complex relationships).

    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        dropout: Attention dropout probability
        bias: Use bias in projections

    Example:
        >>> attn = MultiHeadAttention(d_model=512, n_heads=8)
        >>> hidden = torch.randn(2, 128, 512)
        >>> output, weights = attn(hidden)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi-head self-attention.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional [batch, seq_len, seq_len] mask
            is_causal: Apply causal (autoregressive) masking
            return_weights: Return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            weights: Optional [batch, n_heads, seq_len, seq_len]
        """
        B, L, D = hidden_states.shape

        # Project to Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=hidden_states.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply additional attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention to values
        output = torch.matmul(weights, V)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(B, L, D)

        # Output projection
        output = self.out_proj(output)

        if return_weights:
            return output, weights
        return output, None


class SparseAttention(nn.Module):
    """
    Sparse attention that only attends to selected token positions.

    Used when the router selects a subset of tokens for attention,
    avoiding computation on high-coherence tokens.

    Args:
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        dropout: Attention dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_indices: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Apply attention only to selected tokens.

        Args:
            hidden_states: [batch, seq_len, d_model]
            selected_indices: [batch, n_selected] indices of tokens to attend
            is_causal: Apply causal masking

        Returns:
            output: [batch, seq_len, d_model] with attention applied to selected
        """
        B, L, D = hidden_states.shape
        n_selected = selected_indices.shape[1]

        # Gather selected tokens
        batch_indices = torch.arange(B, device=hidden_states.device)[:, None]
        selected = hidden_states[batch_indices, selected_indices]  # [B, n_selected, D]

        # Apply attention to selected tokens
        attended, _ = self.attention(selected, is_causal=is_causal)

        # Scatter back to original positions
        output = hidden_states.clone()
        output[batch_indices, selected_indices] = attended

        return output
