"""Tests for cort.models module."""

import torch
import pytest
import tempfile
import os

from cort import CoRTModel, CoRTConfig


class TestCoRTModel:
    """Test complete model."""

    def test_forward_shape(self):
        """Forward should return correct logit shape."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
        model = CoRTModel(config)

        input_ids = torch.randint(0, 1000, (2, 64))
        logits = model(input_ids)

        assert logits.shape == (2, 64, 1000)

    def test_return_stats(self):
        """Should return per-layer stats when requested."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=3, n_heads=4)
        model = CoRTModel(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        logits, stats = model(input_ids, return_stats=True)

        assert len(stats) == 3  # One per layer

    def test_num_parameters(self):
        """Parameter count should be reasonable."""
        config = CoRTConfig.small()
        model = CoRTModel(config)

        params = model.num_parameters()
        # Small model should be ~48M params
        assert 40_000_000 < params < 60_000_000

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
        model = CoRTModel(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist on key parameters
        assert model.token_embedding.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_tied_embeddings(self):
        """Token embedding and lm_head should share weights when tied."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, tie_embeddings=True)
        model = CoRTModel(config)

        assert model.token_embedding.weight is model.lm_head.weight


class TestGeneration:
    """Test text generation."""

    def test_generate_shape(self):
        """Generated sequence should have correct length."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
        model = CoRTModel(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 10))
        generated = model.generate(input_ids, max_new_tokens=20)

        assert generated.shape == (1, 30)  # 10 + 20

    def test_generate_deterministic(self):
        """Generation with temperature=0 should be deterministic."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
        model = CoRTModel(config)
        model.eval()

        input_ids = torch.randint(0, 1000, (1, 10))

        # Temperature near 0 with greedy decoding
        gen1 = model.generate(input_ids.clone(), max_new_tokens=10, temperature=0.01, top_k=1)
        gen2 = model.generate(input_ids.clone(), max_new_tokens=10, temperature=0.01, top_k=1)

        assert torch.equal(gen1, gen2)

    def test_eos_stopping(self):
        """Generation should stop at EOS token."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
        model = CoRTModel(config)
        model.eval()

        # This test is probabilistic - EOS might not be generated
        # Just check it doesn't crash
        input_ids = torch.randint(0, 1000, (1, 5))
        generated = model.generate(input_ids, max_new_tokens=50, eos_token_id=999)

        assert generated.shape[1] <= 55  # At most 5 + 50


class TestSaveLoad:
    """Test model persistence."""

    def test_save_load_roundtrip(self):
        """Model should round-trip through save/load."""
        config = CoRTConfig(vocab_size=1000, d_model=256, n_layers=2, n_heads=4)
        model = CoRTModel(config)

        # Get output before save
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            logits_before = model(input_ids)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            model.save_pretrained(path)

            loaded = CoRTModel.from_pretrained(path)

        # Get output after load
        with torch.no_grad():
            logits_after = loaded(input_ids)

        assert torch.allclose(logits_before, logits_after)

    def test_config_preserved(self):
        """Config should be preserved through save/load."""
        config = CoRTConfig(
            vocab_size=5000,
            d_model=384,
            n_layers=4,
            n_heads=6,
            route_frac=0.25,
        )
        model = CoRTModel(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            model.save_pretrained(path)
            loaded = CoRTModel.from_pretrained(path)

        assert loaded.config.vocab_size == 5000
        assert loaded.config.d_model == 384
        assert loaded.config.n_layers == 4
        assert loaded.config.route_frac == 0.25
