"""Tests for numerical stability utilities."""

import pytest
import torch

from medlatents.training.discrete import (
    NaNLossError,
    _check_gradients_finite,
    _check_loss_finite,
)
from medlatents.utils.numerics import (
    EPS_DIV,
    EPS_LOG,
    EPS_PROB,
    EPS_TIME,
    clamp_probs,
    clamp_timesteps,
    entropy,
    gumbel_max_sample,
    gumbel_noise,
    gumbel_softmax_sample,
    js_divergence,
    kl_divergence,
    safe_div,
    safe_log,
    safe_normalize,
)


class TestNumericsConstants:
    """Test that constants have sensible values.

    NOTE: Constants are set for fp16 safety (~1e-6).
    For float32-only code, smaller values like 1e-10 could be used.
    """

    def test_eps_log_is_small_positive(self):
        # Must be > fp16 min (~6e-8) to avoid underflow
        assert 0 < EPS_LOG <= 1e-5

    def test_eps_div_is_small_positive(self):
        # Must be > fp16 min for safe division
        assert 0 < EPS_DIV <= 1e-5

    def test_eps_prob_is_small_positive(self):
        # Must be > fp16 min for multinomial stability
        assert 0 < EPS_PROB <= 1e-5

    def test_eps_time_is_small_positive(self):
        assert 0 < EPS_TIME < 1e-3


class TestSafeLog:
    """Test safe_log function."""

    def test_safe_log_positive_values(self):
        """Safe log works on positive values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = safe_log(x)
        expected = torch.log(x + EPS_LOG)
        assert torch.allclose(result, expected)

    def test_safe_log_zero_no_nan(self):
        """Safe log handles zero without NaN."""
        x = torch.tensor([0.0, 0.0, 0.0])
        result = safe_log(x)
        assert torch.isfinite(result).all()

    def test_safe_log_very_small_values(self):
        """Safe log handles very small values."""
        x = torch.tensor([1e-20, 1e-30, 1e-40])
        result = safe_log(x)
        assert torch.isfinite(result).all()


class TestSafeDiv:
    """Test safe_div function."""

    def test_safe_div_normal_values(self):
        """Safe div works on normal values."""
        num = torch.tensor([1.0, 2.0, 3.0])
        denom = torch.tensor([2.0, 4.0, 6.0])
        result = safe_div(num, denom)
        expected = num / (denom + EPS_DIV)
        assert torch.allclose(result, expected)

    def test_safe_div_zero_denominator(self):
        """Safe div handles zero denominator."""
        num = torch.tensor([1.0, 2.0, 3.0])
        denom = torch.tensor([0.0, 0.0, 0.0])
        result = safe_div(num, denom)
        assert torch.isfinite(result).all()


class TestSafeNormalize:
    """Test safe_normalize function."""

    def test_safe_normalize_sums_to_one(self):
        """Safe normalize produces values that sum to ~1."""
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = safe_normalize(x, dim=-1)
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_safe_normalize_all_zeros(self):
        """Safe normalize handles all-zero input."""
        x = torch.zeros(2, 3)
        result = safe_normalize(x, dim=-1)
        assert torch.isfinite(result).all()


class TestGumbelNoise:
    """Test Gumbel noise generation."""

    def test_gumbel_noise_shape(self):
        """Gumbel noise has correct shape."""
        shape = (4, 10)
        device = torch.device("cpu")
        noise = gumbel_noise(shape, device)
        assert noise.shape == shape

    def test_gumbel_noise_finite(self):
        """Gumbel noise is always finite."""
        shape = (100, 100)
        device = torch.device("cpu")
        for _ in range(10):  # Multiple samples
            noise = gumbel_noise(shape, device)
            assert torch.isfinite(noise).all()

    def test_gumbel_noise_reasonable_range(self):
        """Gumbel noise is in reasonable range (Gumbel(0,1) typically in [-3, 10])."""
        shape = (1000,)
        device = torch.device("cpu")
        noise = gumbel_noise(shape, device)
        # Gumbel(0,1) has mean ~0.577 and std ~1.28
        assert noise.mean().abs() < 2.0
        assert noise.std() > 0.5


class TestGumbelMaxSample:
    """Test Gumbel-max sampling."""

    def test_gumbel_max_sample_shape(self):
        """Gumbel-max produces correct output shape."""
        logits = torch.randn(4, 10, 100)  # [batch, seq, vocab]
        samples = gumbel_max_sample(logits)
        assert samples.shape == (4, 10)

    def test_gumbel_max_sample_valid_indices(self):
        """Gumbel-max produces valid token indices."""
        vocab_size = 100
        logits = torch.randn(4, 10, vocab_size)
        samples = gumbel_max_sample(logits)
        assert samples.min() >= 0
        assert samples.max() < vocab_size

    def test_gumbel_max_sample_deterministic_argmax(self):
        """With very peaked logits, Gumbel-max approximates argmax."""
        # Very peaked distribution
        logits = torch.zeros(1, 5, 10)
        logits[0, :, 3] = 100.0  # Make index 3 dominant
        samples = gumbel_max_sample(logits)
        # Should almost always select index 3
        assert (samples == 3).sum() >= 4  # At least 4/5


class TestGumbelSoftmaxSample:
    """Test Gumbel-softmax sampling."""

    def test_gumbel_softmax_shape(self):
        """Gumbel-softmax produces correct output shape."""
        logits = torch.randn(4, 10, 100)
        samples = gumbel_softmax_sample(logits, temperature=1.0)
        assert samples.shape == logits.shape

    def test_gumbel_softmax_sums_to_one(self):
        """Gumbel-softmax outputs sum to 1."""
        logits = torch.randn(4, 10, 100)
        samples = gumbel_softmax_sample(logits, temperature=1.0)
        sums = samples.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gumbel_softmax_low_temp_peaked(self):
        """Low temperature produces more peaked distributions."""
        logits = torch.randn(4, 10, 100)
        high_temp = gumbel_softmax_sample(logits, temperature=2.0)
        low_temp = gumbel_softmax_sample(logits, temperature=0.5)

        # Low temp should have higher max values (more peaked)
        assert low_temp.max(dim=-1).values.mean() > high_temp.max(dim=-1).values.mean()


class TestClampFunctions:
    """Test clamping functions."""

    def test_clamp_probs_bounds(self):
        """Clamp probs keeps values in [eps, 1-eps]."""
        probs = torch.tensor([0.0, 0.5, 1.0])
        result = clamp_probs(probs)
        assert result.min() >= EPS_PROB
        assert result.max() <= 1.0 - EPS_PROB

    def test_clamp_timesteps_bounds(self):
        """Clamp timesteps keeps values in [eps, 1-eps]."""
        t = torch.tensor([0.0, 0.5, 1.0])
        result = clamp_timesteps(t)
        assert result.min() >= EPS_TIME
        assert result.max() <= 1.0 - EPS_TIME


class TestEntropy:
    """Test entropy computation."""

    def test_entropy_uniform_is_log_k(self):
        """Entropy of uniform distribution is log(k)."""
        k = 10
        probs = torch.ones(1, k) / k
        h = entropy(probs)
        expected = torch.log(torch.tensor(float(k)))
        assert torch.allclose(h, expected, atol=1e-5)

    def test_entropy_peaked_is_low(self):
        """Entropy of peaked distribution is low."""
        probs = torch.zeros(1, 10)
        probs[0, 0] = 1.0
        h = entropy(probs)
        assert h.item() < 0.1  # Close to 0

    def test_entropy_non_negative(self):
        """Entropy is always non-negative."""
        probs = torch.softmax(torch.randn(10, 100), dim=-1)
        h = entropy(probs)
        assert (h >= -1e-6).all()


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_same_distribution_is_zero(self):
        """KL(p || p) = 0."""
        probs = torch.softmax(torch.randn(4, 100), dim=-1)
        kl = kl_divergence(probs, probs)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_non_negative(self):
        """KL divergence is non-negative."""
        p = torch.softmax(torch.randn(10, 100), dim=-1)
        q = torch.softmax(torch.randn(10, 100), dim=-1)
        kl = kl_divergence(p, q)
        assert (kl >= -1e-6).all()


class TestJSDivergence:
    """Test JS divergence computation."""

    def test_js_same_distribution_is_zero(self):
        """JS(p || p) = 0."""
        probs = torch.softmax(torch.randn(4, 100), dim=-1)
        js = js_divergence(probs, probs)
        assert torch.allclose(js, torch.zeros_like(js), atol=1e-5)

    def test_js_symmetric(self):
        """JS is symmetric: JS(p || q) = JS(q || p)."""
        p = torch.softmax(torch.randn(10, 100), dim=-1)
        q = torch.softmax(torch.randn(10, 100), dim=-1)
        js_pq = js_divergence(p, q)
        js_qp = js_divergence(q, p)
        assert torch.allclose(js_pq, js_qp, atol=1e-5)

    def test_js_bounded(self):
        """JS divergence is bounded by log(2)."""
        p = torch.softmax(torch.randn(10, 100), dim=-1)
        q = torch.softmax(torch.randn(10, 100), dim=-1)
        js = js_divergence(p, q)
        assert (js <= torch.log(torch.tensor(2.0)) + 1e-5).all()


class TestNaNLossGuards:
    """Test NaN/Inf loss guards."""

    def test_check_loss_finite_valid_loss(self):
        """Valid loss passes check."""
        loss = torch.tensor(1.5)
        _check_loss_finite(loss, step=0)  # Should not raise

    def test_check_loss_finite_nan_raises(self):
        """NaN loss raises NaNLossError."""
        loss = torch.tensor(float("nan"))
        with pytest.raises(NaNLossError, match="Non-finite loss"):
            _check_loss_finite(loss, step=0)

    def test_check_loss_finite_inf_raises(self):
        """Inf loss raises NaNLossError."""
        loss = torch.tensor(float("inf"))
        with pytest.raises(NaNLossError, match="Non-finite loss"):
            _check_loss_finite(loss, step=0)

    def test_check_loss_finite_neg_inf_raises(self):
        """Negative Inf loss raises NaNLossError."""
        loss = torch.tensor(float("-inf"))
        with pytest.raises(NaNLossError, match="Non-finite loss"):
            _check_loss_finite(loss, step=0)


class TestGradientGuards:
    """Test gradient NaN/Inf guards."""

    def test_check_gradients_finite_valid(self):
        """Valid gradients pass check."""
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        result = _check_gradients_finite(model, step=0)
        assert "grad_norm/total" in result
        assert result["grad_norm/total"] > 0

    def test_check_gradients_finite_nan_raises(self):
        """NaN gradients raise NaNLossError."""
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        # Manually inject NaN
        model.weight.grad[0, 0] = float("nan")

        with pytest.raises(NaNLossError, match="Non-finite gradients"):
            _check_gradients_finite(model, step=0)

    def test_check_gradients_finite_inf_raises(self):
        """Inf gradients raise NaNLossError."""
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        y = model(x).sum()
        y.backward()

        # Manually inject Inf
        model.weight.grad[0, 0] = float("inf")

        with pytest.raises(NaNLossError, match="Non-finite gradients"):
            _check_gradients_finite(model, step=0)
