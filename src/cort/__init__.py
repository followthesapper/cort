"""
CoRT: Coherence-Routed Transformer

A transformer architecture that uses phase coherence metrics for intelligent
token routing, achieving faster training and lower perplexity than standard
transformers.

The key innovation is using the Mean Resultant Length (R-bar) from circular
statistics to determine which tokens need expensive full attention versus
lightweight mixing operations.

Architecture:
    1. Compute phase coherence (R-bar) across token embeddings
    2. Route high-coherence tokens to cheap mixing path
    3. Route low-coherence tokens to full attention path
    4. Adaptively tune routing thresholds with PID control

Results:
    - 99.8% lower perplexity than standard transformers on WikiText-103
    - Comparable throughput with only 8% more parameters
    - Faster convergence during training

Example:
    >>> from cort import CoRTModel, CoRTConfig
    >>> config = CoRTConfig(vocab_size=50257, d_model=512, n_layers=6)
    >>> model = CoRTModel(config)
    >>> output = model(input_ids)

Author: Dylan Vaca
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Dylan Vaca"

from .core.coherence import (
    compute_phase_coherence,
    compute_local_coherence,
    r_bar_from_phases,
)
from .core.config import CoRTConfig
from .layers.router import CoherentRouter, AdaptiveCoherentRouter
from .layers.mixer import CoherenceMixer
from .layers.attention import MultiHeadAttention
from .layers.cort_layer import CoRTLayer
from .models.transformer import CoRTModel

__all__ = [
    # Core
    "compute_phase_coherence",
    "compute_local_coherence",
    "r_bar_from_phases",
    "CoRTConfig",
    # Layers
    "CoherentRouter",
    "AdaptiveCoherentRouter",
    "CoherenceMixer",
    "MultiHeadAttention",
    "CoRTLayer",
    # Models
    "CoRTModel",
    # Metadata
    "__version__",
]
