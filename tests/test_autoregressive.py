"""Tests for autoregressive transformer models."""

import pytest
import torch

from conftest import assert_valid_logits, assert_valid_samples
from medlatents.autoregressive import AutoregressiveTransformer


class TestAutoregressiveTransformer:
    """Test suite for autoregressive models."""

    @pytest.fixture
    def model(self, base_model_config):
        """Create autoregressive model instance."""
        torch.manual_seed(42)
        return AutoregressiveTransformer(**base_model_config)

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
        # Output projection should be properly initialised (not all zeros)
        proj_weight = model.final_layer[1].weight
        assert torch.any(proj_weight != 0), "Final projection weights should not be all zeros"

    def test_rope_supports_even_head_dim(self, base_model_config):
        """RoPE should accept even head dimensions and reject odd ones."""
        config = {
            **base_model_config,
            "hidden_size": base_model_config["num_heads"] * 6,
        }
        model = AutoregressiveTransformer(**config)
        x = torch.randint(0, base_model_config["vocab_size"], (2, config["seq_length"]))
        logits = model(x)
        assert logits.shape == (2, config["seq_length"], model.vocab_size)

        bad_config = {
            **base_model_config,
            "hidden_size": base_model_config["num_heads"] * 5,
        }
        with pytest.raises(ValueError):
            AutoregressiveTransformer(**bad_config)

    def test_forward_shape(self, model, sample_batch, base_model_config):
        """Test forward pass produces correct output shape."""
        logits = model(sample_batch)
        expected_shape = (sample_batch.size(0), sample_batch.size(1), model.vocab_size)
        assert_valid_logits(logits, expected_shape, model.vocab_size)

    def test_forward_deterministic(self, base_model_config, sample_batch):
        """Test forward pass is deterministic."""
        torch.manual_seed(42)
        model1 = AutoregressiveTransformer(**base_model_config)
        model1.eval()

        torch.manual_seed(42)
        model2 = AutoregressiveTransformer(**base_model_config)
        model2.eval()

        with torch.no_grad():
            out1 = model1(sample_batch)
            out2 = model2(sample_batch)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_input_sensitivity(self, base_model_config):
        """Test that model produces different outputs for different inputs."""
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        torch.manual_seed(42)
        model = AutoregressiveTransformer(**config)
        model.eval()

        seq_length = base_model_config["seq_length"]
        vocab_size = base_model_config["vocab_size"]

        # Create two completely different inputs
        torch.manual_seed(123)
        x1 = torch.randint(0, vocab_size, (1, seq_length))
        torch.manual_seed(456)
        x2 = torch.randint(0, vocab_size, (1, seq_length))

        # Ensure they're actually different
        assert not torch.equal(x1, x2)

        with torch.no_grad():
            logits1 = model(x1)
            logits2 = model(x2)

        # Different inputs should produce different outputs
        assert not torch.allclose(logits1, logits2, atol=1e-4), (
            "Model should produce different outputs for different inputs"
        )

    def test_causal_masking_invariant(self, base_model_config):
        """Test that causal masking prevents future tokens from affecting past logits.

        In a properly causal model, changing token at position i should:
        - NOT affect logits at positions 0 to i-1 (can't see future)
        - Affect logits at positions i and beyond (can see the change)
        """
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        torch.manual_seed(42)
        model = AutoregressiveTransformer(**config)
        model.eval()

        seq_length = base_model_config["seq_length"]
        vocab_size = base_model_config["vocab_size"]

        # Create base sequence
        torch.manual_seed(123)
        x1 = torch.randint(0, vocab_size, (1, seq_length))

        # Create modified sequence: change only position 5
        x2 = x1.clone()
        change_pos = 5
        x2[0, change_pos] = (x1[0, change_pos] + 1) % vocab_size

        with torch.no_grad():
            logits1 = model(x1)
            logits2 = model(x2)

        # Positions BEFORE the change should be IDENTICAL (causal: can't see future)
        past_logits1 = logits1[0, :change_pos]
        past_logits2 = logits2[0, :change_pos]
        assert torch.allclose(past_logits1, past_logits2, atol=1e-5), (
            f"Causal violation: logits before position {change_pos} should be identical, "
            f"but differ by max {(past_logits1 - past_logits2).abs().max().item():.2e}"
        )

        # Position AT and AFTER the change should be DIFFERENT (can see the change)
        future_logits1 = logits1[0, change_pos:]
        future_logits2 = logits2[0, change_pos:]
        assert not torch.allclose(future_logits1, future_logits2, atol=1e-4), (
            f"Logits at position {change_pos}+ should differ when input changes"
        )

    def test_generate(self, base_model_config):
        """Test autoregressive generation."""
        # Create model with dynamic seq length for generation
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        torch.manual_seed(42)
        model = AutoregressiveTransformer(**config)
        model.eval()

        batch_size = 2
        prompt_length = 8
        max_length = 16
        vocab_size = base_model_config["vocab_size"]

        prompt = torch.randint(0, vocab_size, (batch_size, prompt_length))

        generated = model.generate(
            prompt=prompt, max_length=max_length, temperature=1.0, top_k=None
        )

        assert generated.shape == (batch_size, max_length)
        assert_valid_samples(generated, model.vocab_size)
        specials = model.special_tokens
        assert not (generated == specials.pad).any()
        assert not (generated == specials.mask).any()
        # Check prompt is preserved
        assert torch.equal(generated[:, :prompt_length], prompt)

    def test_generate_with_top_k(self, base_model_config):
        """Test generation with top-k sampling."""
        # Create model with dynamic seq length for generation
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        torch.manual_seed(42)
        model = AutoregressiveTransformer(**config)
        model.eval()

        prompt = torch.randint(0, base_model_config["vocab_size"], (1, 4))

        generated = model.generate(prompt=prompt, max_length=12, temperature=1.0, top_k=10)

        assert generated.shape == (1, 12)
        assert_valid_samples(generated, model.vocab_size)
        specials = model.special_tokens
        assert not (generated == specials.pad).any()
        assert not (generated == specials.mask).any()

    def test_dynamic_sequence_length(self, base_model_config):
        """Test model handles dynamic sequence lengths."""
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        model = AutoregressiveTransformer(**config)
        model.eval()

        # Test with different length
        longer_seq = torch.randint(0, config["vocab_size"], (2, config["seq_length"] * 2))

        with torch.no_grad():
            logits = model(longer_seq)

        assert logits.shape == (2, longer_seq.shape[1], model.vocab_size)

    def test_sequence_length_mismatch_error(self, base_model_config):
        """Test that fixed length model rejects wrong sequence length."""
        config = {**base_model_config, "allow_dynamic_seq_length": False}
        model = AutoregressiveTransformer(**config)

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
        from medlatents.autoregressive import Autoreg_models

        configs_to_test = ["Autoreg-Nano", "Autoreg-S"]

        for config_name in configs_to_test:
            model_fn = Autoreg_models[config_name]
            model = model_fn(seq_length=16, vocab_size=64)
            assert isinstance(model, AutoregressiveTransformer)

            # Test forward pass
            x = torch.randint(0, 64, (1, 16))
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (1, 16, model.vocab_size)
            assert model.vocab_size == 68


class TestKVCache:
    """Tests for KV-cache implementation correctness."""

    @pytest.fixture
    def model(self, base_model_config):
        """Create autoregressive model instance."""
        torch.manual_seed(42)
        config = {**base_model_config, "allow_dynamic_seq_length": True}
        return AutoregressiveTransformer(**config)

    def test_forward_with_cache_shape(self, model, base_model_config):
        """Test forward_with_cache returns correct shapes."""
        batch_size = 2
        seq_len = base_model_config["seq_length"]
        vocab_size = base_model_config["vocab_size"]

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        model.eval()

        with torch.no_grad():
            logits, kv_cache = model.forward_with_cache(x)

        # Check logits shape
        assert logits.shape == (batch_size, seq_len, model.vocab_size)

        # Check cache structure
        assert len(kv_cache) == len(model.blocks)
        for k, v in kv_cache:
            assert k.shape == (batch_size, model.num_heads, seq_len, model.head_dim)
            assert v.shape == (batch_size, model.num_heads, seq_len, model.head_dim)

    def test_forward_with_cache_matches_forward(self, model, base_model_config):
        """Test that forward_with_cache produces same logits as forward."""
        batch_size = 2
        seq_len = base_model_config["seq_length"]
        vocab_size = base_model_config["vocab_size"]

        torch.manual_seed(123)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        model.eval()

        with torch.no_grad():
            logits_regular = model(x)
            logits_cached, _ = model.forward_with_cache(x)

        assert torch.allclose(logits_regular, logits_cached, atol=1e-5), (
            f"Cached forward should match regular forward. "
            f"Max diff: {(logits_regular - logits_cached).abs().max().item():.2e}"
        )

    def test_incremental_decoding_matches_full_forward(self, model, base_model_config):
        """Test that incremental decoding with cache matches full forward pass.

        This is the critical correctness test: processing tokens one-by-one with
        cache should produce the same logits as processing the full sequence.
        """
        batch_size = 2
        seq_len = 12
        vocab_size = base_model_config["vocab_size"]

        torch.manual_seed(456)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        model.eval()

        # Full forward pass for reference
        with torch.no_grad():
            logits_full = model(x)

        # Incremental decoding with cache
        with torch.no_grad():
            # Process first token to initialize cache
            logits_first, kv_cache = model.forward_with_cache(x[:, :1])

            # Process remaining tokens one at a time
            incremental_logits = [logits_first]
            for i in range(1, seq_len):
                logits_i, kv_cache = model.forward_with_cache(x[:, i : i + 1], kv_cache)
                incremental_logits.append(logits_i)

            # Concatenate all incremental logits
            logits_incremental = torch.cat(incremental_logits, dim=1)

        # Each position's logits should match
        assert torch.allclose(logits_full, logits_incremental, atol=1e-4), (
            f"Incremental decoding with cache should match full forward. "
            f"Max diff: {(logits_full - logits_incremental).abs().max().item():.2e}"
        )

    def test_cache_grows_correctly(self, model, base_model_config):
        """Test that cache grows correctly with each token."""
        batch_size = 2
        vocab_size = base_model_config["vocab_size"]

        torch.manual_seed(111)
        x = torch.randint(0, vocab_size, (batch_size, 8))
        model.eval()

        with torch.no_grad():
            # Start with 4 tokens
            _, cache = model.forward_with_cache(x[:, :4])
            for k, v in cache:
                assert k.shape[2] == 4  # past_len should be 4

            # Add 1 more token
            _, cache = model.forward_with_cache(x[:, 4:5], cache)
            for k, v in cache:
                assert k.shape[2] == 5  # past_len should be 5

            # Add 3 more tokens
            _, cache = model.forward_with_cache(x[:, 5:8], cache)
            for k, v in cache:
                assert k.shape[2] == 8  # past_len should be 8

    def test_cache_preserves_causal_property(self, model, base_model_config):
        """Test that cached decoding maintains causal property.

        Changing a future token should not affect logits for previous positions.
        """
        batch_size = 1
        seq_len = 10
        vocab_size = base_model_config["vocab_size"]

        torch.manual_seed(222)
        x1 = torch.randint(0, vocab_size, (batch_size, seq_len))
        x2 = x1.clone()
        # Change token at position 6
        x2[0, 6] = (x1[0, 6] + 1) % vocab_size

        model.eval()

        with torch.no_grad():
            # Process up to position 5 (before the change)
            logits1_prefix, cache1 = model.forward_with_cache(x1[:, :6])
            logits2_prefix, cache2 = model.forward_with_cache(x2[:, :6])

        # Logits should be identical for positions 0-5
        assert torch.allclose(logits1_prefix, logits2_prefix, atol=1e-5), (
            "Cached logits before changed position should be identical"
        )
