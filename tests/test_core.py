"""Tests for cort.core module."""

import torch
import numpy as np
import pytest

from cort.core.coherence import (
    r_bar_from_phases,
    compute_phase_coherence,
    compute_local_coherence,
)
from cort.core.config import CoRTConfig


class TestRBarFromPhases:
    """Test R-bar computation."""

    def test_aligned_phases(self):
        """Identical phases should give R-bar ≈ 1."""
        phases = torch.tensor([0.5, 0.5, 0.5, 0.5])
        r_bar = r_bar_from_phases(phases)
        assert abs(r_bar.item() - 1.0) < 1e-6

    def test_uniform_phases(self):
        """Uniformly distributed phases should give R-bar ≈ 0."""
        phases = torch.tensor([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        r_bar = r_bar_from_phases(phases)
        assert abs(r_bar.item()) < 1e-6

    def test_range(self):
        """R-bar should always be in [0, 1]."""
        for _ in range(10):
            phases = torch.rand(100) * 2 * np.pi
            r_bar = r_bar_from_phases(phases)
            assert 0 <= r_bar.item() <= 1

    def test_batched(self):
        """Test batched computation."""
        phases = torch.rand(4, 32, 64)
        r_bar = r_bar_from_phases(phases, dim=-1)
        assert r_bar.shape == (4, 32)


class TestPhaseCoherence:
    """Test phase coherence computation."""

    def test_token_mode_shape(self):
        """Token mode should return [batch, seq_len]."""
        hidden = torch.randn(2, 128, 512)
        coherence = compute_phase_coherence(hidden, mode="token")
        assert coherence.shape == (2, 128)

    def test_sequence_mode_shape(self):
        """Sequence mode should return [batch]."""
        hidden = torch.randn(2, 128, 512)
        coherence = compute_phase_coherence(hidden, mode="sequence")
        assert coherence.shape == (2,)

    def test_range(self):
        """Coherence should be in [0, 1]."""
        hidden = torch.randn(4, 64, 256)
        coherence = compute_phase_coherence(hidden, mode="token")
        assert (coherence >= 0).all()
        assert (coherence <= 1).all()


class TestLocalCoherence:
    """Test local coherence computation."""

    def test_output_shape(self):
        """Output should match input spatial dimensions."""
        hidden = torch.randn(2, 128, 512)
        local_coh = compute_local_coherence(hidden, window=8)
        assert local_coh.shape == (2, 128)

    def test_different_windows(self):
        """Should work with different window sizes."""
        hidden = torch.randn(2, 64, 256)

        for window in [4, 8, 16]:
            local_coh = compute_local_coherence(hidden, window=window)
            assert local_coh.shape == (2, 64)

    def test_range(self):
        """Local coherence should be in [0, 1]."""
        hidden = torch.randn(4, 64, 256)
        local_coh = compute_local_coherence(hidden, window=8)
        assert (local_coh >= 0).all()
        assert (local_coh <= 1).all()


class TestCoRTConfig:
    """Test configuration."""

    def test_defaults(self):
        """Default config should be valid."""
        config = CoRTConfig()
        assert config.d_model == 512
        assert config.n_layers == 6
        assert config.d_ff == 2048  # 4 * d_model

    def test_head_dim(self):
        """Head dimension should be computed correctly."""
        config = CoRTConfig(d_model=512, n_heads=8)
        assert config.head_dim == 64

    def test_to_dict_from_dict(self):
        """Config should round-trip through dict."""
        config = CoRTConfig(d_model=768, n_layers=12)
        d = config.to_dict()
        config2 = CoRTConfig.from_dict(d)
        assert config.d_model == config2.d_model
        assert config.n_layers == config2.n_layers

    def test_presets(self):
        """Preset configs should be valid."""
        small = CoRTConfig.small()
        medium = CoRTConfig.medium()
        large = CoRTConfig.large()

        assert small.d_model < medium.d_model < large.d_model

    def test_validation_head_dim(self):
        """Should fail if d_model not divisible by n_heads."""
        with pytest.raises(AssertionError):
            CoRTConfig(d_model=512, n_heads=7)

    def test_validation_route_frac(self):
        """Should fail if route_frac out of bounds."""
        with pytest.raises(AssertionError):
            CoRTConfig(route_frac=0)

        with pytest.raises(AssertionError):
            CoRTConfig(route_frac=1)
