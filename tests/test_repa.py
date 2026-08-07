"""Tests for REPA (Representation Alignment for Generation).

Tests the actual REPALoss implementation which uses negative cosine similarity
(i.e., loss is negative when features are well-aligned).
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from medlatents.training import REPALoss


class TestREPALossBasics:
    """Basic tests for REPA alignment loss."""

    @given(
        batch_size=st.integers(min_value=2, max_value=8),
        seq_len=st.integers(min_value=16, max_value=64),
        hidden_dim=st.integers(min_value=128, max_value=256),
        target_dim=st.integers(min_value=128, max_value=256),
    )
    @settings(deadline=2000)  # Allow more time for property tests
    def test_loss_returns_scalar(
        self, batch_size: int, seq_len: int, hidden_dim: int, target_dim: int
    ):
        """REPA loss should return scalar."""
        repa_loss = REPALoss(
            hidden_dim=hidden_dim,
            target_dim=target_dim,
            weight=0.5,
            loss_type="cosine",
        )

        model_hidden = torch.randn(batch_size, seq_len, hidden_dim)
        encoder_features = torch.randn(batch_size, seq_len, target_dim)
        timesteps = torch.ones(batch_size) * 0.8  # Above default threshold

        loss = repa_loss(model_hidden, encoder_features, timesteps)

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_cosine_loss_is_finite(self):
        """Cosine loss should produce finite results."""
        repa_loss = REPALoss(hidden_dim=256, target_dim=768, loss_type="cosine", weight=1.0)

        model_hidden = torch.randn(4, 32, 256)
        encoder_features = torch.randn(4, 32, 768)
        timesteps = torch.ones(4) * 0.8

        loss = repa_loss(model_hidden, encoder_features, timesteps)

        assert torch.isfinite(loss), "Cosine loss should be finite"
        # REPA uses negative cosine similarity, so loss is typically negative
        # (good alignment = more negative, bad alignment = closer to 0 or slightly positive)

    def test_mse_loss_non_negative(self):
        """MSE loss should produce non-negative finite results."""
        repa_loss = REPALoss(hidden_dim=256, target_dim=768, loss_type="mse", weight=1.0)

        model_hidden = torch.randn(4, 32, 256)
        encoder_features = torch.randn(4, 32, 768)
        timesteps = torch.ones(4) * 0.8

        loss = repa_loss(model_hidden, encoder_features, timesteps)

        assert torch.isfinite(loss), "MSE loss should be finite"
        assert loss >= 0, "MSE loss should be non-negative"


class TestREPATimestepThresholding:
    """Tests for timestep-based REPA activation."""

    def test_above_threshold_computes_loss(self):
        """REPA should compute loss when t > threshold."""
        repa_loss = REPALoss(
            hidden_dim=256, target_dim=768, weight=0.5, loss_type="cosine", timestep_threshold=0.5
        )

        model_hidden = torch.randn(2, 16, 256)
        encoder_features = torch.randn(2, 16, 768)
        t = torch.tensor([0.8, 0.6])  # Both above threshold

        loss = repa_loss(model_hidden, encoder_features, t)

        assert torch.isfinite(loss), "Loss should be finite for t > threshold"
        # Cosine loss can be negative (negative similarity)
        assert loss != 0.0, "Loss should be non-zero when above threshold"

    def test_below_threshold_returns_zero(self):
        """REPA should return zero loss when t <= threshold."""
        repa_loss = REPALoss(
            hidden_dim=256, target_dim=768, weight=0.5, loss_type="cosine", timestep_threshold=0.5
        )

        model_hidden = torch.randn(2, 16, 256)
        encoder_features = torch.randn(2, 16, 768)
        t = torch.tensor([0.3, 0.1])  # Both below threshold

        loss = repa_loss(model_hidden, encoder_features, t)

        # Loss should be zero when t < threshold
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
            "Loss should be zero for t < threshold"
        )

    def test_mixed_timesteps_partial_activation(self):
        """Some samples above, some below threshold."""
        repa_loss = REPALoss(
            hidden_dim=256, target_dim=768, weight=0.5, loss_type="cosine", timestep_threshold=0.5
        )

        model_hidden = torch.randn(4, 16, 256)
        encoder_features = torch.randn(4, 16, 768)
        t = torch.tensor([0.8, 0.2, 0.6, 0.1])

        loss = repa_loss(model_hidden, encoder_features, t)

        assert torch.isfinite(loss), "Loss should be finite for mixed timesteps"
        # Only 2 of 4 samples contribute, loss should be non-zero but not as large


class TestREPAWeighting:
    """Tests for REPA loss weight parameter."""

    @given(
        weight=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=2000)
    def test_weight_scales_loss(self, weight: float):
        """REPA weight should linearly scale the loss."""
        # Use same projection for both - test weight scaling only
        torch.manual_seed(42)

        model_hidden = torch.randn(2, 16, 256)
        encoder_features = torch.randn(2, 16, 768)
        timesteps = torch.ones(2) * 0.8

        # Create single REPALoss and test weight effect
        repa_loss = REPALoss(hidden_dim=256, target_dim=768, weight=1.0, loss_type="cosine")

        loss_base = repa_loss(model_hidden, encoder_features, timesteps)

        # Manually compute weighted loss
        expected_weighted = loss_base * weight

        # Verify loss is finite and weight scaling is mathematically correct
        assert torch.isfinite(loss_base), "Base loss should be finite"
        assert torch.isfinite(expected_weighted), "Weighted loss should be finite"

        # Now test actual weighted loss module
        repa_weighted = REPALoss(hidden_dim=256, target_dim=768, weight=weight, loss_type="cosine")
        # Copy the projection weights to ensure same projection
        repa_weighted.projection.load_state_dict(repa_loss.projection.state_dict())

        loss_weighted = repa_weighted(model_hidden, encoder_features, timesteps)
        torch.testing.assert_close(loss_weighted, expected_weighted, rtol=1e-4, atol=1e-6)

    def test_zero_weight_returns_zero(self):
        """Zero weight should return zero loss."""
        repa_loss = REPALoss(hidden_dim=256, target_dim=768, weight=0.0, loss_type="cosine")

        model_hidden = torch.randn(2, 16, 256)
        encoder_features = torch.randn(2, 16, 768)
        timesteps = torch.ones(2) * 0.8

        loss = repa_loss(model_hidden, encoder_features, timesteps)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
            "Zero weight should give zero loss"
        )


class TestREPAGradientFlow:
    """Tests for gradient computation through REPA loss."""

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
    )
    @settings(deadline=2000)
    def test_gradients_are_computed(self, batch_size: int, seq_len: int):
        """Gradients should be computed and finite."""
        repa_loss = REPALoss(hidden_dim=256, target_dim=768, weight=0.5, loss_type="cosine")

        model_hidden = torch.randn(batch_size, seq_len, 256, requires_grad=True)
        encoder_features = torch.randn(batch_size, seq_len, 768)
        timesteps = torch.ones(batch_size) * 0.8

        loss = repa_loss(model_hidden, encoder_features, timesteps)
        loss.backward()

        assert model_hidden.grad is not None, "Gradients should be computed for model_hidden"
        assert torch.all(torch.isfinite(model_hidden.grad)), "Gradients should be finite"

    def test_gradients_scale_with_weight(self):
        """Gradients should scale proportionally with weight."""
        torch.manual_seed(42)

        encoder_features = torch.randn(2, 16, 768)
        timesteps = torch.ones(2) * 0.8

        # Create base loss module
        repa_base = REPALoss(hidden_dim=256, target_dim=768, weight=1.0, loss_type="cosine")

        # Test with weight=1.0
        model_hidden_1 = torch.randn(2, 16, 256, requires_grad=True)
        loss_1 = repa_base(model_hidden_1, encoder_features, timesteps)
        loss_1.backward()
        assert model_hidden_1.grad is not None
        grad_base = model_hidden_1.grad.clone()

        # Test with weight=0.3 (same projection)
        repa_weighted = REPALoss(hidden_dim=256, target_dim=768, weight=0.3, loss_type="cosine")
        repa_weighted.projection.load_state_dict(repa_base.projection.state_dict())

        model_hidden_2 = torch.randn(2, 16, 256, requires_grad=True)
        # Use same input for fair comparison
        model_hidden_2.data.copy_(model_hidden_1.data.detach())

        loss_2 = repa_weighted(model_hidden_2, encoder_features, timesteps)
        loss_2.backward()
        assert model_hidden_2.grad is not None
        grad_weighted = model_hidden_2.grad

        # Gradients should scale by weight ratio
        ratio = grad_weighted / (grad_base + 1e-8)
        mean_ratio = ratio.mean().item()
        assert abs(mean_ratio - 0.3) < 0.1, (
            f"Gradients should scale with weight, got ratio {mean_ratio:.3f}"
        )


class TestREPAEdgeCases:
    """Edge case tests for REPA."""

    def test_aligned_features_have_more_negative_cosine_loss(self):
        """When features are aligned, cosine loss should be more negative."""
        repa_loss = REPALoss(hidden_dim=256, target_dim=256, loss_type="cosine", weight=1.0)

        # Create aligned features (same projection)
        shared = torch.randn(2, 16, 256)

        loss_aligned = repa_loss(shared, shared.clone(), torch.ones(2) * 0.8)
        loss_random = repa_loss(shared, torch.randn(2, 16, 256), torch.ones(2) * 0.8)

        # Aligned features should give more negative loss (better similarity)
        assert loss_aligned < loss_random, (
            f"Aligned features should give more negative loss: aligned={loss_aligned.item()}, random={loss_random.item()}"
        )

    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        seq_len=st.integers(min_value=16, max_value=32),
    )
    @settings(deadline=2000)
    def test_different_loss_types_finite(self, batch_size: int, seq_len: int):
        """All loss types should produce finite results."""
        model_hidden = torch.randn(batch_size, seq_len, 256)
        encoder_features = torch.randn(batch_size, seq_len, 768)
        timesteps = torch.ones(batch_size) * 0.8

        for loss_type in ("cosine", "mse"):
            repa_loss = REPALoss(hidden_dim=256, target_dim=768, loss_type=loss_type, weight=1.0)  # type: ignore[arg-type]
            loss = repa_loss(model_hidden, encoder_features, timesteps)

            assert torch.isfinite(loss), f"{loss_type} loss should be finite"

    def test_mse_loss_is_non_negative(self):
        """MSE loss should always be non-negative."""
        repa_loss = REPALoss(hidden_dim=256, target_dim=768, loss_type="mse", weight=1.0)

        model_hidden = torch.randn(2, 16, 256)
        encoder_features = torch.randn(2, 16, 768)
        timesteps = torch.ones(2) * 0.8

        loss = repa_loss(model_hidden, encoder_features, timesteps)
        assert loss >= 0, f"MSE loss should be non-negative, got {loss.item()}"


class TestREPAProjection:
    """Tests for REPAProjection module."""

    def test_projection_output_shape(self):
        """Projection should output correct shape."""
        from medlatents.training import REPAProjection

        proj = REPAProjection(hidden_dim=256, target_dim=768, proj_dim=128)

        hidden = torch.randn(2, 16, 256)
        target = torch.randn(2, 16, 768)

        proj_hidden = proj(hidden)
        proj_target = proj.project_target(target)

        assert proj_hidden.shape == (
            2,
            16,
            128,
        ), f"Hidden projection shape wrong: {proj_hidden.shape}"
        assert proj_target.shape == (
            2,
            16,
            128,
        ), f"Target projection shape wrong: {proj_target.shape}"

    def test_projection_gradient_flow(self):
        """Gradients should flow through projection."""
        from medlatents.training import REPAProjection

        proj = REPAProjection(hidden_dim=256, target_dim=768, proj_dim=128)

        hidden = torch.randn(2, 16, 256, requires_grad=True)

        proj_hidden = proj(hidden)
        loss = proj_hidden.sum()
        loss.backward()

        assert hidden.grad is not None, "Gradients should flow to input"
        assert torch.all(torch.isfinite(hidden.grad)), "Gradients should be finite"
