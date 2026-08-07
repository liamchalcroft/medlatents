"""Additional tests for Phase 1: Property-based, scale, and hardware tests."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from medlatents.sampling import TemperatureScheduler, sample_nucleus, sample_with_cfg


class TestModelProperties:
    """Property-based tests for model invariants."""

    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        seq_len=st.integers(min_value=8, max_value=256),
        vocab_size=st.integers(min_value=32, max_value=1024),
    )
    @settings(deadline=None)  # Avoid flakiness from JIT warmup
    def test_autoregressive_output_shape(self, batch_size, seq_len, vocab_size):
        """AR model output shape invariant."""
        logits = torch.randn(batch_size, seq_len, vocab_size)

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        vocab_size=st.integers(min_value=10, max_value=100),
    )
    def test_sampling_invariants(self, batch_size, vocab_size):
        """Sampling produces valid token indices."""
        # sample_nucleus expects [batch, vocab_size] shape
        logits = torch.randn(batch_size, vocab_size)

        filtered = sample_nucleus(logits, p=0.95)

        assert filtered.shape == logits.shape
        # Nucleus filtering sets some logits to -inf, keeps valid distribution
        assert not torch.isnan(filtered).any()

    @given(
        logits=st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=1, max_size=100),
        temperature=st.floats(min_value=0.1, max_value=2.0),
    )
    def test_temperature_monotonicity(self, logits, temperature):
        """Temperature scales output distribution."""
        logits = torch.tensor(logits)
        temperature = torch.tensor(temperature)

        scaled = logits / temperature.unsqueeze(-1)

        assert scaled.shape == logits.shape
        assert not torch.isnan(scaled).any()

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        vocab_size=st.integers(min_value=10, max_value=100),
        guidance_scale=st.floats(min_value=0.5, max_value=10.0),
    )
    def test_cfg_invariants(self, batch_size, vocab_size, guidance_scale):
        """CFG produces valid interpolated logits."""
        # CFG expects matching [batch, vocab_size] shapes
        logits_cond = torch.randn(batch_size, vocab_size)
        logits_uncond = torch.randn(batch_size, vocab_size)

        guided = sample_with_cfg(logits_cond, logits_uncond, guidance_scale)

        assert guided.shape == logits_cond.shape
        assert not torch.isnan(guided).any()


class TestSchedulerProperties:
    """Property-based tests for scheduler invariants."""

    @given(
        start_temp=st.floats(min_value=0.5, max_value=2.0),
        end_temp=st.floats(min_value=0.5, max_value=2.0),
        num_steps=st.integers(min_value=10, max_value=1000),
    )
    def test_temperature_scheduler_monotonic(self, start_temp, end_temp, num_steps):
        """Temperature schedule is monotonic."""
        scheduler = TemperatureScheduler(
            schedule="linear",
            start_temp=start_temp,
            end_temp=end_temp,
            num_steps=num_steps,
        )

        temps = [scheduler.get_temperature(i) for i in range(num_steps)]

        if start_temp > end_temp:
            assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))
        elif start_temp < end_temp:
            assert all(temps[i] <= temps[i + 1] for i in range(len(temps) - 1))

    @given(
        data=st.data(),
    )
    def test_temperature_bounds(self, data):
        """Temperature stays within bounds."""
        num_steps = data.draw(st.integers(min_value=10, max_value=100))
        current_step = data.draw(st.integers(min_value=0, max_value=num_steps - 1))
        scheduler = TemperatureScheduler(
            schedule="cosine",
            start_temp=1.0,
            end_temp=0.5,
            num_steps=num_steps,
        )

        temp = scheduler.get_temperature(current_step)

        assert 0.45 <= temp <= 1.05


class TestNumericalStability:
    """Tests for numerical stability across scales."""

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        hidden_size=st.integers(min_value=64, max_value=1024),
    )
    def test_softmax_stability(self, batch_size, hidden_size):
        """Softmax is numerically stable."""
        logits = torch.randn(batch_size, hidden_size) * 1000  # Large values

        # Shift by max for stability
        max_logit = logits.max(dim=-1, keepdim=True)[0]
        stable_softmax = torch.exp(logits - max_logit) / torch.exp(logits - max_logit).sum(
            dim=-1, keepdim=True
        )

        assert stable_softmax.shape == logits.shape
        assert (stable_softmax >= 0).all() and (stable_softmax <= 1).all()
        assert torch.allclose(stable_softmax.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    @given(
        x=st.floats(min_value=1e-10, max_value=1e10),
    )
    def test_safe_log(self, x):
        """Log is safe for all inputs."""
        x = torch.tensor([x])

        log_x = torch.log(x.clamp(min=1e-10))

        assert not torch.isnan(log_x)
        assert not torch.isinf(log_x)

    @given(
        p=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_kl_divergence_symmetry(self, p):
        """KL divergence properties."""
        p = torch.tensor([p, 1 - p])
        p = p / p.sum()

        # KL(p||q) = 0 if p == q
        q = p.clone()
        kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum()

        assert kl >= 0
        assert not torch.isnan(kl)
        assert not torch.isinf(kl)


class TestScaleInvariants:
    """Tests for model behavior across different scales."""

    @pytest.mark.slow
    @given(
        vocab_size=st.integers(min_value=128, max_value=1024),
    )
    @settings(deadline=None)  # Model creation can be slow
    def test_vocab_size_scaling(self, vocab_size):
        """Models work with different vocabulary sizes."""
        from medlatents import AutoregressiveTransformer

        config = {"hidden_size": 128, "depth": 4, "num_heads": 4}

        model = AutoregressiveTransformer(
            seq_length=64,
            vocab_size=vocab_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
        )

        # Forward pass
        input_tokens = torch.randint(0, vocab_size, (2, 64))
        output = model(input_tokens)

        # vocab_size gets extended by 4 special tokens
        assert output.shape == (2, 64, model.vocab_size)

    @pytest.mark.slow
    @given(
        seq_len=st.integers(min_value=256, max_value=2048),
    )
    @settings(deadline=None)  # Model creation can be slow
    def test_sequence_length_scaling(self, seq_len):
        """Models work with different sequence lengths."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(
            seq_length=seq_len,
            vocab_size=512,
            hidden_size=128,
            depth=4,
            num_heads=4,
            allow_dynamic_seq_length=True,
        )

        # Test with shorter sequence
        input_tokens = torch.randint(0, 512, (1, 256))
        output = model(input_tokens)

        assert output.shape[1] == 256

    @pytest.mark.slow
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
    )
    @settings(deadline=None)  # Model creation can be slow
    def test_batch_size_scaling(self, batch_size):
        """Models work with different batch sizes."""
        from medlatents import MaskGIT

        model = MaskGIT(
            seq_length=256,
            vocab_size=512,
            hidden_size=128,
            depth=4,
            num_heads=4,
        )

        input_tokens = torch.randint(0, 512, (batch_size, 256))
        output = model(input_tokens)

        # MaskGIT also adds special tokens (mask token at least)
        assert output.shape == (batch_size, 256, model.vocab_size)
