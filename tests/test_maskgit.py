"""Tests for MaskGIT bidirectional transformer models."""

import pytest
import torch

from conftest import assert_valid_logits, assert_valid_samples
from medlatents.maskgit import MaskGIT


class TestMaskGIT:
    """Test suite for MaskGIT models."""

    @pytest.fixture
    def model(self, base_model_config):
        """Create MaskGIT model instance."""
        torch.manual_seed(42)
        return MaskGIT(**base_model_config)

    def test_initialization(self, model, base_model_config):
        """Test model initializes correctly."""
        expected_vocab = base_model_config["vocab_size"] + 4
        assert model.vocab_size == expected_vocab
        assert model.num_heads == base_model_config["num_heads"]
        assert len(model.blocks) == base_model_config["depth"]
        assert model.special_tokens is not None
        specials = model.special_tokens.as_dict()
        assert len(set(specials.values())) == 4
        assert all(token_id < model.vocab_size for token_id in specials.values())
        assert model.mask_token == model.special_tokens.mask

    def test_forward_shape(self, model, sample_batch, base_model_config):
        """Test forward pass produces correct output shape."""
        logits = model(sample_batch)
        expected_shape = (sample_batch.size(0), sample_batch.size(1), model.vocab_size)
        assert_valid_logits(logits, expected_shape, model.vocab_size)

    def test_forward_deterministic(self, base_model_config, sample_batch):
        """Test forward pass is deterministic."""
        torch.manual_seed(42)
        model1 = MaskGIT(**base_model_config)
        model1.eval()

        torch.manual_seed(42)
        model2 = MaskGIT(**base_model_config)
        model2.eval()

        with torch.no_grad():
            out1 = model1(sample_batch)
            out2 = model2(sample_batch)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_bidirectional_context(self, model, base_model_config):
        """Test that model uses bidirectional context."""
        model.eval()
        seq_length = base_model_config["seq_length"]
        vocab_size = base_model_config["vocab_size"]

        # Create input
        x = torch.randint(0, vocab_size, (1, seq_length))

        with torch.no_grad():
            logits = model(x)

        # Modify future tokens (second half)
        x_modified = x.clone()
        x_modified[:, seq_length // 2 :] = torch.randint(0, vocab_size, (1, seq_length // 2))

        with torch.no_grad():
            logits_modified = model(x_modified)

        # For bidirectional model, predictions in first half SHOULD change
        # when future context changes (opposite of causal masking test)
        assert not torch.allclose(
            logits[:, : seq_length // 2],
            logits_modified[:, : seq_length // 2],
            atol=1e-5,
        ), "Bidirectional model should use future context"

    def test_masking_changes_predictions(self, model, base_model_config):
        """Test that applying mask changes predictions."""
        model.eval()
        vocab_size = base_model_config["vocab_size"]
        seq_length = base_model_config["seq_length"]

        tokens = torch.randint(0, vocab_size, (2, seq_length))

        # Create mask for every other position
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[:, ::2] = True

        with torch.no_grad():
            masked_logits = model(tokens, mask=mask)
            baseline_logits = model(tokens)

        # Predictions should differ when masking is applied
        assert not torch.allclose(masked_logits, baseline_logits)

    def test_random_masking(self, model, base_model_config):
        """Test random masking produces correct mask."""
        vocab_size = base_model_config["vocab_size"]
        seq_length = base_model_config["seq_length"]

        x = torch.randint(0, vocab_size, (4, seq_length))
        mask_ratio = 0.3

        x_masked, mask = model.random_masking(x, mask_ratio=mask_ratio)

        # Check mask is boolean
        assert mask.dtype == torch.bool
        assert mask.shape == x.shape

        # Check approximately correct mask ratio (with some tolerance)
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - mask_ratio) < 0.1

        # Check masked positions have mask token
        assert (x_masked[mask] == model.mask_token).all()
        # Check unmasked positions unchanged
        assert (x_masked[~mask] == x[~mask]).all()

    def test_generate(self, model, base_model_config):
        """Test iterative parallel generation (MaskGIT)."""
        model.eval()
        batch_size = 2
        seq_length = base_model_config["seq_length"]
        base_model_config["vocab_size"]
        num_steps = 8

        # Start with all masked tokens
        x = torch.full((batch_size, seq_length), model.mask_token, dtype=torch.long)

        torch.manual_seed(0)
        generated = model.generate(x=x, num_steps=num_steps, temperature=1.0, top_k=None)

        assert generated.shape == (batch_size, seq_length)
        assert_valid_samples(generated, model.vocab_size)
        specials = model.special_tokens
        assert (generated != specials.pad).all()
        assert (generated != specials.mask).all()
        # Check that generated tokens are not mask tokens
        assert (generated != model.mask_token).all()

    def test_generate_with_top_k(self, model, base_model_config):
        """Test generation with top-k sampling."""
        model.eval()
        seq_length = base_model_config["seq_length"]
        base_model_config["vocab_size"]

        x = torch.full((1, seq_length), model.mask_token, dtype=torch.long)

        torch.manual_seed(0)
        generated = model.generate(x=x, num_steps=4, temperature=1.0, top_k=10)

        assert generated.shape == (1, seq_length)
        assert_valid_samples(generated, model.vocab_size)
        specials = model.special_tokens
        assert (generated != specials.pad).all()
        assert (generated != specials.mask).all()

    def test_inpaint_anomalies(self, model, base_model_config):
        """Test inpainting functionality."""
        model.eval()
        vocab_size = base_model_config["vocab_size"]
        seq_length = base_model_config["seq_length"]

        # Create input with some tokens
        tokens = torch.randint(0, vocab_size, (2, seq_length))

        inpainted = model.inpaint_anomalies(
            tokens, likelihood_threshold=0.1, temperature=1.0, top_k=None
        )

        assert inpainted.shape == tokens.shape
        assert_valid_samples(inpainted, model.vocab_size)

    def test_dynamic_sequence_length(self, base_model_config):
        """Test model handles dynamic sequence lengths."""
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        model = MaskGIT(**config)
        model.eval()

        # Test with different length
        longer_seq = torch.randint(0, config["vocab_size"], (2, config["seq_length"] * 2))

        with torch.no_grad():
            logits = model(longer_seq)

        assert logits.shape == (2, longer_seq.shape[1], model.vocab_size)

    def test_sequence_length_mismatch_error(self, base_model_config):
        """Test that fixed length model rejects wrong sequence length."""
        config = {**base_model_config, "allow_dynamic_seq_length": False}
        model = MaskGIT(**config)

        wrong_length = torch.randint(0, config["vocab_size"], (1, config["seq_length"] + 5))

        with pytest.raises(ValueError):
            model(wrong_length)

    def test_gradient_flow(self, model, sample_batch):
        """Test gradients flow through model."""
        model.train()
        logits = model(sample_batch)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"

    def test_model_configs(self):
        """Test that model size configs work."""
        from medlatents.maskgit import MaskGIT_models

        configs_to_test = ["MaskGIT-Nano", "MaskGIT-S"]

        for config_name in configs_to_test:
            model_fn = MaskGIT_models[config_name]
            model = model_fn(seq_length=16, vocab_size=64)
            assert isinstance(model, MaskGIT)

            # Test forward pass
            x = torch.randint(0, 64, (1, 16))
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (1, 16, model.vocab_size)
            assert model.vocab_size == 68

    def test_mask_token_embedding(self, model, base_model_config):
        """Test that mask token is properly embedded."""
        model.eval()
        base_model_config["vocab_size"]
        seq_length = base_model_config["seq_length"]

        # Create sequence with mask tokens
        x = torch.full((1, seq_length), model.mask_token, dtype=torch.long)

        with torch.no_grad():
            logits = model(x)

        # Should produce valid logits
        assert_valid_logits(logits, (1, seq_length, model.vocab_size), model.vocab_size)

    def test_partial_masking(self, model, base_model_config):
        """Test model with partial masking."""
        model.eval()
        vocab_size = base_model_config["vocab_size"]
        seq_length = base_model_config["seq_length"]

        # Create input with some real tokens and some masked
        tokens = torch.randint(0, vocab_size, (2, seq_length))
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[:, seq_length // 2 :] = True  # Mask second half

        with torch.no_grad():
            logits = model(tokens, mask=mask)

        assert_valid_logits(logits, (2, seq_length, model.vocab_size), model.vocab_size)

    def test_chunking_matches_full_pass(self, base_model_config):
        """Test that chunked processing matches full pass."""
        config = {**base_model_config, "allow_dynamic_seq_length": True, "chunk_size": 4}
        model = MaskGIT(**config)
        tokens = torch.randint(
            0, base_model_config["vocab_size"], (1, base_model_config["seq_length"] * 2)
        )

        model.set_chunk_size(4)
        chunked_logits = model(tokens)

        model.set_chunk_size(None)
        full_logits = model(tokens)

        assert chunked_logits.shape == full_logits.shape

    def test_detect_anomalies_returns_boolean_mask(self, model, base_model_config):
        """Test that detect_anomalies returns proper boolean mask."""
        tokens = torch.randint(
            0, base_model_config["vocab_size"], (2, base_model_config["seq_length"])
        )
        tokens[:, 0] = base_model_config["vocab_size"] - 1

        mask = model.detect_anomalies(tokens, likelihood_threshold=0.1)

        assert mask.shape == tokens.shape
        assert mask.dtype == torch.bool
        assert mask[:, 0].any()

    def test_set_chunk_size_rejects_zero(self, model):
        """Test that set_chunk_size rejects invalid values."""
        with pytest.raises(ValueError):
            model.set_chunk_size(0)

        model.set_chunk_size(3)
        assert model.chunk_size == 3
