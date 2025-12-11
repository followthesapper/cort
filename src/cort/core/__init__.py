"""
CoRT Core: Phase coherence metrics and configuration.

This module provides the mathematical foundation for coherence-routed
attention, including R-bar computation and routing configuration.
"""

from .coherence import (
    compute_phase_coherence,
    compute_local_coherence,
    r_bar_from_phases,
)
from .config import CoRTConfig

__all__ = [
    "compute_phase_coherence",
    "compute_local_coherence",
    "r_bar_from_phases",
    "CoRTConfig",
]
