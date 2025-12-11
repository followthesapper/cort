"""
CoRT Configuration: Model hyperparameters and routing settings.

Provides dataclass-based configuration for CoRT models with sensible
defaults that work well across different tasks.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoRTConfig:
    """
    Configuration for Coherence-Routed Transformer models.

    Architecture Parameters:
        vocab_size: Size of the token vocabulary
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension (default: 4 * d_model)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability

    Routing Parameters:
        route_frac: Base fraction of tokens routed to attention (0.15 = 15%)
        r_bar_threshold: Coherence threshold for routing decisions
        adaptive_routing: Use PID-tuned adaptive routing
        coherence_window: Window size for local coherence computation

    Training Parameters:
        tie_embeddings: Tie input and output embeddings
        layer_norm_eps: Layer normalization epsilon

    Example:
        >>> config = CoRTConfig(
        ...     vocab_size=50257,
        ...     d_model=768,
        ...     n_heads=12,
        ...     n_layers=12,
        ... )
        >>> model = CoRTModel(config)
    """

    # Architecture
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: Optional[int] = None
    max_seq_len: int = 1024
    dropout: float = 0.1

    # Routing
    route_frac: float = 0.15
    r_bar_threshold: float = 0.5
    adaptive_routing: bool = True
    coherence_window: int = 8
    coherence_mode: str = "combined"

    # PID Controller (for adaptive routing)
    pid_kp: float = 0.1
    pid_ki: float = 0.01
    pid_kd: float = 0.05

    # Training
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        """Set derived defaults after initialization."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        # Validate
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert 0 < self.route_frac < 1, "route_frac must be in (0, 1)"
        assert 0 <= self.r_bar_threshold <= 1, "r_bar_threshold must be in [0, 1]"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "route_frac": self.route_frac,
            "r_bar_threshold": self.r_bar_threshold,
            "adaptive_routing": self.adaptive_routing,
            "coherence_window": self.coherence_window,
            "coherence_mode": self.coherence_mode,
            "tie_embeddings": self.tie_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoRTConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def small(cls) -> "CoRTConfig":
        """Small model configuration (~48M params)."""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=6,
        )

    @classmethod
    def medium(cls) -> "CoRTConfig":
        """Medium model configuration (~124M params)."""
        return cls(
            d_model=768,
            n_heads=12,
            n_layers=12,
        )

    @classmethod
    def large(cls) -> "CoRTConfig":
        """Large model configuration (~350M params)."""
        return cls(
            d_model=1024,
            n_heads=16,
            n_layers=24,
        )
