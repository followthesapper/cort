"""
CoRT Coherence: Phase coherence metrics for token routing.

The core insight of CoRT is that tokens with high phase coherence (aligned
representations) can be processed with lightweight mixing, while tokens with
low coherence need full attention to resolve their relationships.

Key Metrics:
    - R-bar (R̄): Mean Resultant Length from circular statistics
    - Range: [0, 1] where 1 = perfect alignment, 0 = uniform distribution
    - Computed as: R̄ = |mean(exp(i * phases))|

The phase coherence determines routing:
    - High R̄ (> threshold): Token representations are aligned → use mixing
    - Low R̄ (< threshold): Token representations are scattered → use attention
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def r_bar_from_phases(phases: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Mean Resultant Length (R-bar) from phase angles.

    R-bar is a measure of phase concentration from circular statistics.
    Values close to 1 indicate tight clustering, values close to 0 indicate
    uniform distribution around the circle.

    Args:
        phases: Phase angles in radians, shape [..., N]
        dim: Dimension along which to compute R-bar

    Returns:
        R-bar values, shape [...] (dim reduced)

    Example:
        >>> phases = torch.tensor([0.1, 0.1, 0.1, 0.1])  # Aligned
        >>> r_bar_from_phases(phases)
        tensor(0.9950)  # Close to 1

        >>> phases = torch.tensor([0, np.pi/2, np.pi, 3*np.pi/2])  # Uniform
        >>> r_bar_from_phases(phases)
        tensor(0.0)  # Close to 0
    """
    cos_sum = torch.cos(phases).mean(dim=dim)
    sin_sum = torch.sin(phases).mean(dim=dim)
    r_bar = torch.sqrt(cos_sum**2 + sin_sum**2)
    return r_bar


def compute_phase_coherence(
    hidden_states: torch.Tensor,
    mode: str = "token",
) -> torch.Tensor:
    """
    Compute phase coherence from hidden states.

    Converts hidden state magnitudes to phases and computes R-bar.
    This measures how aligned the token representations are.

    Args:
        hidden_states: Token embeddings, shape [batch, seq_len, d_model]
        mode: Coherence computation mode
            - "token": Per-token coherence across embedding dimensions
            - "sequence": Per-sequence coherence across tokens

    Returns:
        Coherence scores (R-bar values)
            - "token" mode: shape [batch, seq_len]
            - "sequence" mode: shape [batch]

    Example:
        >>> hidden = torch.randn(2, 128, 512)
        >>> coherence = compute_phase_coherence(hidden, mode="token")
        >>> coherence.shape
        torch.Size([2, 128])
    """
    # Convert hidden states to phases via sigmoid → [0, 2π]
    phases = 2 * np.pi * torch.sigmoid(hidden_states)

    if mode == "token":
        # Coherence per token across embedding dimensions
        return r_bar_from_phases(phases, dim=-1)
    elif mode == "sequence":
        # Coherence per sequence across all tokens and dimensions
        B, L, D = phases.shape
        phases_flat = phases.view(B, -1)
        return r_bar_from_phases(phases_flat, dim=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'token' or 'sequence'.")


def compute_local_coherence(
    hidden_states: torch.Tensor,
    window: int = 8,
) -> torch.Tensor:
    """
    Compute local phase coherence using sliding windows.

    For each token, computes R-bar over a local window of neighboring
    tokens. This captures local coherence patterns in the sequence.

    Uses vectorized avg_pool1d for efficiency instead of Python loops.

    Args:
        hidden_states: Token embeddings, shape [batch, seq_len, d_model]
        window: Size of the local window (centered on each token)

    Returns:
        Local coherence scores, shape [batch, seq_len]

    Example:
        >>> hidden = torch.randn(2, 128, 512)
        >>> local_coh = compute_local_coherence(hidden, window=8)
        >>> local_coh.shape
        torch.Size([2, 128])
    """
    B, L, D = hidden_states.shape

    # Convert to phases
    phases = 2 * np.pi * torch.sigmoid(hidden_states)

    # Compute sin and cos
    cos_phases = torch.cos(phases)
    sin_phases = torch.sin(phases)

    # Pad for sliding window
    pad = window // 2
    cos_padded = F.pad(cos_phases, (0, 0, pad, pad), mode="replicate")
    sin_padded = F.pad(sin_phases, (0, 0, pad, pad), mode="replicate")

    # Use avg_pool1d for efficient sliding window mean
    # Transpose to [B, D, L+2*pad] for pooling
    cos_t = cos_padded.transpose(1, 2)
    sin_t = sin_padded.transpose(1, 2)

    # Pool across sequence dimension
    mean_cos = F.avg_pool1d(cos_t, kernel_size=window, stride=1)
    mean_sin = F.avg_pool1d(sin_t, kernel_size=window, stride=1)

    # Compute R-bar: sqrt(mean_cos^2 + mean_sin^2), averaged over dimensions
    r_bar_per_dim = torch.sqrt(mean_cos**2 + mean_sin**2)
    local_coherence = r_bar_per_dim.mean(dim=1)  # Average over d_model

    return local_coherence


def compute_cross_coherence(
    hidden_states: torch.Tensor,
    n_pairs: int = 4,
) -> torch.Tensor:
    """
    Compute cross-token phase coherence between neighboring pairs.

    Measures how aligned consecutive token pairs are, which indicates
    whether they can be processed together efficiently.

    Args:
        hidden_states: Token embeddings, shape [batch, seq_len, d_model]
        n_pairs: Number of consecutive pairs to consider

    Returns:
        Cross-coherence scores, shape [batch, seq_len]
    """
    B, L, D = hidden_states.shape

    # Convert to phases
    phases = 2 * np.pi * torch.sigmoid(hidden_states)

    cross_coherence = torch.zeros(B, L, device=hidden_states.device)

    for offset in range(1, n_pairs + 1):
        if offset >= L:
            break

        # Compare token i with token i+offset
        phases_i = phases[:, :-offset, :]
        phases_j = phases[:, offset:, :]

        # Phase difference
        phase_diff = phases_i - phases_j
        cos_diff = torch.cos(phase_diff).mean(dim=-1)
        sin_diff = torch.sin(phase_diff).mean(dim=-1)
        r_bar_pair = torch.sqrt(cos_diff**2 + sin_diff**2)

        # Accumulate (pad to match original length)
        cross_coherence[:, :-offset] += r_bar_pair

    # Normalize by number of pairs considered
    cross_coherence = cross_coherence / n_pairs

    return cross_coherence
