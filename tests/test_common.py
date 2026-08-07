"""Tests for common utilities in medlatents.common."""

import pytest
import torch

from medlatents.common.schedules import (
    get_cosine_schedule_with_warmup,
)
from medlatents.common.time import (
    get_timestep_sampler,
    sample_timesteps_logit_normal,
    sample_timesteps_u_shaped,
    sample_timesteps_uniform,
    sample_timesteps_uniform_shifted,
)

# =============================================================================
# SCHEDULE TESTS
# =============================================================================


class TestCosineScheduleWithWarmup:
    """Tests for get_cosine_schedule_with_warmup function."""

    def test_warmup_linear_increase(self):
        """During warmup, LR should increase linearly from 0 to max."""
        max_lr = 1e-4
        warmup_steps = 100
        total_steps = 1000

        # At step 0
        lr_0 = get_cosine_schedule_with_warmup(0, warmup_steps, total_steps, max_lr, 0)
        assert lr_0 == 0.0

        # At half warmup
        lr_half = get_cosine_schedule_with_warmup(50, warmup_steps, total_steps, max_lr, 0)
        assert abs(lr_half - max_lr * 0.5) < 1e-10

        # At end of warmup
        lr_end = get_cosine_schedule_with_warmup(warmup_steps, warmup_steps, total_steps, max_lr, 0)
        assert abs(lr_end - max_lr) < 1e-10

    def test_cosine_decay_after_warmup(self):
        """After warmup, LR should decay following cosine schedule."""
        max_lr = 1e-4
        min_lr = 1e-6
        warmup_steps = 100
        total_steps = 1000

        # At end of warmup should be max_lr
        lr_warmup_end = get_cosine_schedule_with_warmup(
            warmup_steps, warmup_steps, total_steps, max_lr, min_lr
        )
        assert abs(lr_warmup_end - max_lr) < 1e-10

        # At end of training should be close to min_lr
        lr_final = get_cosine_schedule_with_warmup(
            total_steps - 1, warmup_steps, total_steps, max_lr, min_lr
        )
        # Allow some tolerance for cosine
        assert abs(lr_final - min_lr) < 1e-6

    def test_warmup_equals_total_returns_max(self):
        """When warmup_steps == total_steps, should return max_lr."""
        max_lr = 1e-4
        lr = get_cosine_schedule_with_warmup(50, 100, 100, max_lr, 1e-6)
        assert lr == max_lr


# =============================================================================
# TIME/TIMESTEP TESTS
# =============================================================================


class TestTimestepSampling:
    """Tests for timestep sampling functions."""

    def test_uniform_timesteps_in_range(self):
        """Uniform timesteps should be in [0, 1]."""
        t = sample_timesteps_uniform(100, device=None)
        assert torch.all(t >= 0) and torch.all(t <= 1)
        assert t.shape == (100,)

    def test_uniform_shifted_timesteps_avoid_boundaries(self):
        """Uniform shifted timesteps should avoid exact 0 and 1."""
        t = sample_timesteps_uniform_shifted(100, eps=1e-3, device=None)
        assert torch.all(t > 1e-3) and torch.all(t < 1 - 1e-3)

    def test_u_shaped_timesteps_in_range(self):
        """U-shaped timesteps should be in [0, 1]."""
        t = sample_timesteps_u_shaped(100, concentration=2.0, device=None)
        assert torch.all(t >= 0) and torch.all(t <= 1)

    def test_logit_normal_timesteps_in_range(self):
        """Logit-normal timesteps should be in [0, 1]."""
        t = sample_timesteps_logit_normal(100, mean=0.0, std=1.0, device=None)
        assert torch.all(t >= 0) and torch.all(t <= 1)

    def test_get_timestep_sampler(self):
        """Factory function should return correct sampler."""
        sampler = get_timestep_sampler("uniform")
        t = sampler(10, None)
        assert t.shape == (10,)

        with pytest.raises(ValueError):
            get_timestep_sampler("unknown")
