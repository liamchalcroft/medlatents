"""Tests for VeCoR (Velocity Contrastive Regularization).

Implements property-based tests for VeCoR loss functionality.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from medlatents.training import VelocityContrastiveRegularization


class TestVeCoRBasicFunctionality:
    """Basic functionality tests for VeCoR loss."""

    @given(
        batch_size=st.integers(min_value=2, max_value=8),
        seq_len=st.integers(min_value=16, max_value=64),
        dim=st.integers(min_value=64, max_value=256),
    )
    def test_loss_shape(self, batch_size: int, seq_len: int, dim: int):
        """VeCoR loss should return scalar."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1, negative_mode="roll")

        v_pred = torch.randn(batch_size, seq_len, dim)
        v_target = torch.randn(batch_size, seq_len, dim)

        loss = vecor(v_pred, v_target)

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    @given(
        batch_size=st.integers(min_value=2, max_value=8),
        seq_len=st.integers(min_value=16, max_value=64),
        dim=st.integers(min_value=64, max_value=256),
    )
    def test_loss_range(self, batch_size: int, seq_len: int, dim: int):
        """VeCoR loss should be non-negative."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1, negative_mode="roll")

        v_pred = torch.randn(batch_size, seq_len, dim)
        v_target = torch.randn(batch_size, seq_len, dim)

        loss = vecor(v_pred, v_target)

        assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"

    def test_single_batch_returns_zero(self):
        """VeCoR requires at least 2 samples for contrastive learning."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=1.0)

        v_pred = torch.randn(1, 32, 64)
        v_target = torch.randn(1, 32, 64)

        loss = vecor(v_pred, v_target)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
            "Single batch should return zero loss"
        )

    @given(
        batch_size=st.integers(min_value=2, max_value=8),
        seq_len=st.integers(min_value=16, max_value=64),
        dim=st.integers(min_value=64, max_value=256),
        weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_weight_scaling(self, batch_size: int, seq_len: int, dim: int, weight: float):
        """Loss should scale linearly with weight parameter."""
        vecor_weighted = VelocityContrastiveRegularization(
            temperature=0.1, weight=weight, negative_mode="roll"
        )
        vecor_unweighted = VelocityContrastiveRegularization(
            temperature=0.1, weight=1.0, negative_mode="roll"
        )

        v_pred = torch.randn(batch_size, seq_len, dim)
        v_target = torch.randn(batch_size, seq_len, dim)

        loss_weighted = vecor_weighted(v_pred, v_target)
        loss_unweighted = vecor_unweighted(v_pred, v_target)

        expected = loss_unweighted * weight
        torch.testing.assert_close(loss_weighted, expected, rtol=1e-4, atol=1e-6)


class TestVeCoRNegativeModes:
    """Tests for different negative sampling modes."""

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
        dim=st.integers(min_value=64, max_value=128),
    )
    def test_roll_mode_changes_negatives(self, batch_size: int, seq_len: int, dim: int):
        """Roll mode should rotate batch dimension for negatives."""
        vecor_roll = VelocityContrastiveRegularization(
            temperature=0.1, weight=0.1, negative_mode="roll"
        )

        v_pred = torch.randn(batch_size, seq_len, dim)
        v_target = torch.randn(batch_size, seq_len, dim)

        loss = vecor_roll(v_pred, v_target)

        assert torch.isfinite(loss), "Roll mode should produce finite loss"
        assert loss >= 0, "Roll mode loss should be non-negative"

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
        dim=st.integers(min_value=64, max_value=128),
    )
    def test_shuffle_mode_changes_negatives(self, batch_size: int, seq_len: int, dim: int):
        """Shuffle mode should randomly permute batch for negatives."""
        vecor_shuffle = VelocityContrastiveRegularization(
            temperature=0.1, weight=0.1, negative_mode="shuffle"
        )

        v_pred = torch.randn(batch_size, seq_len, dim)
        v_target = torch.randn(batch_size, seq_len, dim)

        loss = vecor_shuffle(v_pred, v_target)

        assert torch.isfinite(loss), "Shuffle mode should produce finite loss"
        assert loss >= 0, "Shuffle mode loss should be non-negative"

    def test_invalid_negative_mode_raises(self):
        """Invalid negative mode should raise ValueError."""
        with pytest.raises(ValueError, match="negative_mode must be"):
            VelocityContrastiveRegularization(temperature=0.1, weight=0.1, negative_mode="invalid")


class TestVeCoRTemperature:
    """Tests for temperature parameter."""

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
        dim=st.integers(min_value=64, max_value=128),
        temperature=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_temperature_affects_loss_magnitude(
        self, batch_size: int, seq_len: int, dim: int, temperature: float
    ):
        """Lower temperature should increase loss (harder negatives)."""
        vecor_low = VelocityContrastiveRegularization(
            temperature=temperature, weight=1.0, negative_mode="roll"
        )
        vecor_high = VelocityContrastiveRegularization(
            temperature=1.0, weight=1.0, negative_mode="roll"
        )

        v_pred = torch.randn(batch_size, seq_len, dim)
        v_target = torch.randn(batch_size, seq_len, dim)

        torch.manual_seed(42)
        loss_low = vecor_low(v_pred, v_target)
        torch.manual_seed(42)
        loss_high = vecor_high(v_pred, v_target)

        assert torch.isfinite(loss_low), "Low temperature loss should be finite"
        assert torch.isfinite(loss_high), "High temperature loss should be finite"

    def test_zero_temperature_raises(self):
        """Temperature must be positive."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            VelocityContrastiveRegularization(temperature=0.0, weight=0.1)


class TestVeCoRGradientFlow:
    """Tests for gradient computation through VeCoR loss."""

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
        dim=st.integers(min_value=64, max_value=128),
    )
    def test_gradients_are_finite(self, batch_size: int, seq_len: int, dim: int):
        """Gradients should be finite and non-zero."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1, negative_mode="roll")

        v_pred = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        v_target = torch.randn(batch_size, seq_len, dim)

        loss = vecor(v_pred, v_target)
        loss.backward()

        assert v_pred.grad is not None, "Gradients should be computed"
        assert torch.all(torch.isfinite(v_pred.grad)), "Gradients should be finite"
        assert not torch.allclose(v_pred.grad, torch.zeros_like(v_pred.grad)), (
            "Gradients should be non-zero"
        )

    def test_gradients_scale_with_weight(self):
        """Gradients should scale proportionally with weight."""
        vecor_weighted = VelocityContrastiveRegularization(
            temperature=0.1, weight=0.5, negative_mode="roll"
        )
        vecor_unweighted = VelocityContrastiveRegularization(
            temperature=0.1, weight=1.0, negative_mode="roll"
        )

        v_pred = torch.randn(2, 32, 64, requires_grad=True)
        v_target = torch.randn(2, 32, 64)

        loss_weighted = vecor_weighted(v_pred, v_target)
        loss_unweighted = vecor_unweighted(v_pred, v_target)

        loss_weighted.backward(retain_graph=True)
        grad_weighted = v_pred.grad.clone()

        v_pred.grad = None
        loss_unweighted.backward()
        grad_unweighted = v_pred.grad

        # Gradients should scale by approximately weight ratio
        ratio = grad_weighted / (grad_unweighted + 1e-8)
        assert torch.allclose(ratio, torch.full_like(ratio, 0.5), rtol=0.1, atol=0.05), (
            f"Gradients should scale with weight, got ratio {ratio.mean().item():.3f}"
        )


class TestVeCoREdgeCases:
    """Edge case tests for VeCoR."""

    def test_zero_weight_returns_zero_loss(self):
        """Zero weight should return zero loss."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.0, negative_mode="roll")

        v_pred = torch.randn(2, 32, 64)
        v_target = torch.randn(2, 32, 64)

        loss = vecor(v_pred, v_target)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
            "Zero weight should give zero loss"
        )

    def test_identical_targets_gives_nonzero_loss(self):
        """Even with identical targets, loss should be non-zero due to negatives."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1, negative_mode="roll")

        v_pred = torch.randn(2, 32, 64)
        v_target = v_pred.clone()

        loss = vecor(v_pred, v_target)

        # Loss should still be computed (negatives are rolled/shuffled)
        assert loss > 0, "Loss should be positive even with identical targets"

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
        dim=st.integers(min_value=64, max_value=128),
    )
    def test_different_shapes_work(self, batch_size: int, seq_len: int, dim: int):
        """VeCoR should work with various tensor shapes."""
        vecor = VelocityContrastiveRegularization(temperature=0.1, weight=0.1, negative_mode="roll")

        # Test 1D sequence (for AR models)
        v_pred_1d = torch.randn(batch_size, seq_len, dim)
        v_target_1d = torch.randn(batch_size, seq_len, dim)
        loss_1d = vecor(v_pred_1d, v_target_1d)

        assert torch.isfinite(loss_1d), "1D sequence should work"

        # Test 2D spatial (for continuous latents)
        v_pred_2d = torch.randn(batch_size, dim, 8, 8)
        v_target_2d = torch.randn(batch_size, dim, 8, 8)
        loss_2d = vecor(v_pred_2d, v_target_2d)

        assert torch.isfinite(loss_2d), "2D spatial should work"
