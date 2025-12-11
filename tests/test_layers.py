"""Tests for cort.layers module."""

import torch
import pytest

from cort.layers.router import CoherentRouter, AdaptiveCoherentRouter, RoutingConfig
from cort.layers.mixer import CoherenceMixer
from cort.layers.attention import MultiHeadAttention
from cort.layers.cort_layer import CoRTLayer


class TestCoherentRouter:
    """Test routing functionality."""

    def test_output_shapes(self):
        """Router should return masks and scores with correct shapes."""
        config = RoutingConfig()
        router = CoherentRouter(config, d_model=512)

        hidden = torch.randn(2, 128, 512)
        attn_mask, mix_mask, scores = router(hidden)

        assert attn_mask.shape == (2, 128)
        assert mix_mask.shape == (2, 128)
        assert scores.shape == (2, 128)

    def test_masks_exclusive(self):
        """Attention and mix masks should be mutually exclusive."""
        config = RoutingConfig()
        router = CoherentRouter(config, d_model=512)

        hidden = torch.randn(2, 128, 512)
        attn_mask, mix_mask, _ = router(hidden)

        # XOR should be all True
        assert ((attn_mask ^ mix_mask) == True).all()

    def test_route_fraction(self):
        """Roughly route_frac tokens should go to attention."""
        config = RoutingConfig(route_frac=0.2)
        router = CoherentRouter(config, d_model=512)

        hidden = torch.randn(4, 100, 512)
        attn_mask, _, _ = router(hidden)

        # Allow some tolerance
        actual_frac = attn_mask.float().mean().item()
        assert 0.15 <= actual_frac <= 0.25


class TestAdaptiveRouter:
    """Test adaptive routing."""

    def test_adaptation(self):
        """Router should adapt routing fraction."""
        config = RoutingConfig(route_frac=0.15)
        router = AdaptiveCoherentRouter(config, d_model=512)

        # Multiple forward passes should update state
        hidden = torch.randn(2, 128, 512)
        for _ in range(5):
            router(hidden)

        # PID state should have been updated
        assert router.integral.item() != 0 or router.prev_error.item() != 0


class TestCoherenceMixer:
    """Test mixing layer."""

    def test_output_shape(self):
        """Output should match input shape."""
        mixer = CoherenceMixer(d_model=512)
        hidden = torch.randn(2, 128, 512)
        output = mixer(hidden)
        assert output.shape == hidden.shape

    def test_with_coherence_scores(self):
        """Should accept coherence scores."""
        mixer = CoherenceMixer(d_model=512)
        hidden = torch.randn(2, 128, 512)
        coherence = torch.rand(2, 128)
        output = mixer(hidden, coherence_scores=coherence)
        assert output.shape == hidden.shape

    def test_short_sequence(self):
        """Should handle very short sequences."""
        mixer = CoherenceMixer(d_model=512)
        hidden = torch.randn(2, 1, 512)
        output = mixer(hidden)
        assert output.shape == hidden.shape


class TestMultiHeadAttention:
    """Test attention layer."""

    def test_output_shape(self):
        """Output should match input shape."""
        attn = MultiHeadAttention(d_model=512, n_heads=8)
        hidden = torch.randn(2, 128, 512)
        output, _ = attn(hidden)
        assert output.shape == hidden.shape

    def test_return_weights(self):
        """Should return attention weights when requested."""
        attn = MultiHeadAttention(d_model=512, n_heads=8)
        hidden = torch.randn(2, 32, 512)
        output, weights = attn(hidden, return_weights=True)
        assert weights.shape == (2, 8, 32, 32)

    def test_causal_mask(self):
        """Causal mask should prevent attending to future tokens."""
        attn = MultiHeadAttention(d_model=512, n_heads=8)
        hidden = torch.randn(2, 16, 512)
        _, weights = attn(hidden, is_causal=True, return_weights=True)

        # Upper triangle should be zero (after softmax, -inf becomes 0)
        for i in range(16):
            for j in range(i + 1, 16):
                assert weights[0, 0, i, j].item() < 1e-6


class TestCoRTLayer:
    """Test complete CoRT layer."""

    def test_output_shape(self):
        """Output should match input shape."""
        layer = CoRTLayer(d_model=512, n_heads=8, d_ff=2048)
        hidden = torch.randn(2, 128, 512)
        output, _ = layer(hidden)
        assert output.shape == hidden.shape

    def test_return_stats(self):
        """Should return routing stats when requested."""
        layer = CoRTLayer(d_model=512, n_heads=8, d_ff=2048)
        hidden = torch.randn(2, 128, 512)
        output, stats = layer(hidden, return_stats=True)

        assert stats is not None
        assert "attn_ratio" in stats
        assert "mix_ratio" in stats
        assert "mean_coherence" in stats

    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        layer = CoRTLayer(d_model=512, n_heads=8, d_ff=2048)
        hidden = torch.randn(2, 64, 512, requires_grad=True)
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None
        assert not torch.isnan(hidden.grad).any()
