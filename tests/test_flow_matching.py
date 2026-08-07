"""Tests for discrete flow matching."""

import pytest
import torch

from conftest import assert_valid_samples
from medlatents.flow_matching import (
    compute_entropy,
    get_loss_function,
    get_path,
    get_source_distribution,
)
from medlatents.flow_matching.core import (
    MixtureDiscreteProbPath,
    MixturePathGeneralizedKL,
)
from medlatents.flow_matching.discrete import (
    MaskedSourceDistribution,
    UniformSourceDistribution,
)


class TestSourceDistributions:
    """Test suite for source distributions."""

    def test_masked_source_distribution_sample(self):
        """Test masked source distribution sampling."""
        mask_token = 100
        dist = MaskedSourceDistribution(mask_token=mask_token)

        batch_size, seq_len = 4, 16
        samples = dist.sample((batch_size, seq_len), device=torch.device("cpu"))

        assert samples.shape == (batch_size, seq_len)
        assert (samples == mask_token).all()
        assert samples.dtype == torch.long

    def test_masked_source_distribution_sample_like(self):
        """Test masked source distribution sample_like."""
        mask_token = 50
        dist = MaskedSourceDistribution(mask_token=mask_token)

        template = torch.randint(0, 10, (3, 8))
        samples = dist.sample_like(template)

        assert samples.shape == template.shape
        assert (samples == mask_token).all()
        assert samples.dtype == torch.long

    def test_masked_source_is_masked(self):
        """Test that masked source reports masked=True."""
        dist = MaskedSourceDistribution(mask_token=0)
        assert dist.masked is True

    def test_uniform_source_distribution_sample(self):
        """Test uniform source distribution sampling."""
        vocab_size = 128
        dist = UniformSourceDistribution(vocab_size=vocab_size)

        batch_size, seq_len = 4, 16
        samples = dist.sample((batch_size, seq_len), device=torch.device("cpu"))

        assert samples.shape == (batch_size, seq_len)
        assert_valid_samples(samples, vocab_size)

    def test_uniform_source_distribution_sample_like(self):
        """Test uniform source distribution sample_like."""
        vocab_size = 64
        dist = UniformSourceDistribution(vocab_size=vocab_size)

        template = torch.randint(0, vocab_size, (3, 8))
        samples = dist.sample_like(template)

        assert samples.shape == template.shape
        assert_valid_samples(samples, vocab_size)

    def test_uniform_source_is_not_masked(self):
        """Test that uniform source reports masked=False."""
        dist = UniformSourceDistribution(vocab_size=100)
        assert dist.masked is False

    def test_uniform_source_distribution_uniformity(self):
        """Test that uniform distribution is approximately uniform."""
        vocab_size = 10
        dist = UniformSourceDistribution(vocab_size=vocab_size)

        # Sample many times
        samples = dist.sample((1000, 100), device=torch.device("cpu"))

        # Count frequencies
        counts = torch.bincount(samples.flatten(), minlength=vocab_size)
        frequencies = counts.float() / counts.sum()

        # Each token should appear with ~10% frequency (with some tolerance)
        expected_freq = 1.0 / vocab_size
        assert torch.allclose(frequencies, torch.full_like(frequencies, expected_freq), atol=0.02)


class TestGetSourceDistribution:
    """Test source distribution factory function."""

    def test_get_mask_source_distribution(self):
        """Test getting masked source distribution."""
        vocab_size = 100
        dist = get_source_distribution("mask", vocab_size)

        assert isinstance(dist, MaskedSourceDistribution)
        assert dist.mask_token == vocab_size
        assert dist.masked is True

    def test_get_mask_source_distribution_custom_token(self):
        """Mask distribution respects explicit mask token id."""
        vocab_size = 100
        mask_id = 42
        dist = get_source_distribution("mask", vocab_size, mask_token=mask_id)

        assert isinstance(dist, MaskedSourceDistribution)
        assert dist.mask_token == mask_id

    def test_get_uniform_source_distribution(self):
        """Test getting uniform source distribution."""
        vocab_size = 128
        dist = get_source_distribution("uniform", vocab_size)

        assert isinstance(dist, UniformSourceDistribution)
        assert dist.vocab_size == vocab_size
        assert dist.masked is False

    def test_get_source_distribution_invalid(self):
        """Test that invalid source distribution raises error."""
        with pytest.raises(ValueError, match="not supported"):
            get_source_distribution("invalid", 100)


class TestGetPath:
    """Test path factory function."""

    def test_get_polynomial_path(self):
        """Test getting polynomial path."""
        path = get_path("polynomial", exponent=2.0)

        # Check it's the right type
        assert isinstance(path, MixtureDiscreteProbPath)

    def test_get_path_invalid(self):
        """Test that invalid scheduler type raises error."""
        with pytest.raises(ValueError, match="not supported"):
            get_path("invalid", exponent=1.0)


class TestGetLossFunction:
    """Test loss function factory."""

    def test_get_cross_entropy_loss(self):
        """Test getting cross entropy loss."""
        loss_fn = get_loss_function("cross_entropy")

        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_get_generalized_kl_loss(self):
        """Test getting generalized KL loss."""
        path = get_path("polynomial", exponent=2.0)
        loss_fn = get_loss_function("generalized_kl", path=path)

        assert isinstance(loss_fn, MixturePathGeneralizedKL)

    def test_get_generalized_kl_without_path_fails(self):
        """Test that generalized KL requires path."""
        with pytest.raises(ValueError, match="path argument"):
            get_loss_function("generalized_kl", path=None)

    def test_get_loss_function_invalid(self):
        """Test that invalid loss function raises error."""
        with pytest.raises(ValueError, match="not supported"):
            get_loss_function("invalid")


class TestComputeEntropy:
    """Test entropy computation."""

    def test_compute_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        # Create samples from uniform distribution
        vocab_size = 8
        samples = torch.randint(0, vocab_size, (100, 100))

        entropy = compute_entropy(samples)

        # Entropy of uniform distribution is log2(vocab_size)
        expected_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32))

        # Allow some tolerance for sampling variance
        assert torch.isclose(entropy, expected_entropy, atol=0.5)

    def test_compute_entropy_deterministic(self):
        """Test entropy for deterministic (single value) distribution."""
        # All samples are the same
        samples = torch.zeros((50, 50), dtype=torch.long)

        entropy = compute_entropy(samples)

        # Entropy of deterministic distribution is 0
        assert torch.isclose(entropy, torch.tensor(0.0), atol=1e-6)

    def test_compute_entropy_shape(self):
        """Test that entropy returns scalar."""
        samples = torch.randint(0, 10, (20, 30))

        entropy = compute_entropy(samples)

        assert entropy.ndim == 0  # Scalar
        assert entropy >= 0  # Entropy is non-negative

    def test_compute_entropy_device(self):
        """Test entropy computation preserves device."""
        device = torch.device("cpu")
        samples = torch.randint(0, 10, (10, 10), device=device)

        entropy = compute_entropy(samples)

        assert entropy.device == device


class TestPathIntegration:
    """Integration tests for path with source distributions."""

    def test_path_sample_with_masked_source(self):
        """Test path sampling with masked source."""
        vocab_size = 64
        batch_size, seq_len = 4, 16

        # Setup
        dist = get_source_distribution("mask", vocab_size)
        path = get_path("polynomial", exponent=2.0)

        # Sample from source
        x_0 = dist.sample((batch_size, seq_len), device=torch.device("cpu"))

        # Create target
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Sample along path at t=0.5
        t = torch.tensor([0.5] * batch_size)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        assert path_sample.x_t.shape == (batch_size, seq_len)
        assert_valid_samples(path_sample.x_t, vocab_size + 1)  # +1 for mask token

    def test_path_sample_with_uniform_source(self):
        """Test path sampling with uniform source."""
        vocab_size = 64
        batch_size, seq_len = 4, 16

        # Setup
        dist = get_source_distribution("uniform", vocab_size)
        path = get_path("polynomial", exponent=2.0)

        # Sample from source
        x_0 = dist.sample((batch_size, seq_len), device=torch.device("cpu"))

        # Create target
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Sample along path at t=0.5
        t = torch.tensor([0.5] * batch_size)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        assert path_sample.x_t.shape == (batch_size, seq_len)
        assert_valid_samples(path_sample.x_t, vocab_size)

    def test_path_sample_at_endpoints(self):
        """Test path sampling at t=0 and t=1."""
        vocab_size = 32
        batch_size, seq_len = 2, 8

        dist = get_source_distribution("uniform", vocab_size)
        path = get_path("polynomial", exponent=2.0)

        x_0 = dist.sample((batch_size, seq_len), device=torch.device("cpu"))
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_len))

        # At t=0, should be close to x_0
        sample_0 = path.sample(t=torch.tensor([0.0] * batch_size), x_0=x_0, x_1=x_1)
        # Allow for some stochasticity but most samples should match
        assert (sample_0.x_t == x_0).float().mean() > 0.8

        # At t=1, should be close to x_1
        sample_1 = path.sample(t=torch.tensor([1.0] * batch_size), x_0=x_0, x_1=x_1)
        assert (sample_1.x_t == x_1).float().mean() > 0.8


class TestLossFunctionIntegration:
    """Integration tests for loss functions."""

    def test_cross_entropy_loss_computation(self):
        """Test cross entropy loss with dummy logits."""
        batch_size, seq_len, vocab_size = 4, 8, 32

        loss_fn = get_loss_function("cross_entropy")

        # Create dummy logits and targets
        logits = torch.randn(batch_size, vocab_size, seq_len)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = loss_fn(logits, targets)

        assert isinstance(loss.item(), float)
        assert loss > 0
        assert torch.isfinite(loss)

    def test_generalized_kl_loss_computation(self):
        """Test generalized KL loss with path."""
        batch_size, seq_len, vocab_size = 4, 8, 32

        path = get_path("polynomial", exponent=2.0)
        loss_fn = get_loss_function("generalized_kl", path=path)

        # Create dummy data
        logits = torch.randn(batch_size, seq_len, vocab_size)
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_len))
        x_t = torch.randint(0, vocab_size, (batch_size, seq_len))
        t = torch.rand(1)

        loss = loss_fn(logits=logits, x_1=x_1, x_t=x_t, t=t)

        assert isinstance(loss.item(), float)
        assert torch.isfinite(loss)
