"""Tests for new flow matching features: timestep sampling, OT coupling, CFG-Zero."""

import torch

from medlatents.flow_matching import (
    DiscreteFlowTrainer,
    MixtureDiscreteProbPath,
    PolynomialConvexScheduler,
    compute_log_snr,
    compute_ot_coupling,
    create_time_grid,
    get_timestep_sampler,
    ot_flow_sample_path,
    sample_from_coupling,
    sample_timesteps_logit_normal,
    sample_timesteps_u_shaped,
    sample_timesteps_uniform,
)
from medlatents.sampling import cfg_zero_guidance, get_guidance_schedule


class TestTimestepSampling:
    """Tests for timestep sampling strategies."""

    def test_uniform_timesteps_in_range(self):
        """Uniform samples should be in (eps, 1-eps)."""
        t = sample_timesteps_uniform(100, torch.device("cpu"), eps=1e-5)
        assert t.shape == (100,)
        assert t.min() >= 1e-5
        assert t.max() <= 1 - 1e-5

    def test_u_shaped_timesteps_in_range(self):
        """U-shaped samples should be in (0, 1)."""
        t = sample_timesteps_u_shaped(100, torch.device("cpu"))
        assert t.shape == (100,)
        assert t.min() >= 0
        assert t.max() <= 1

    def test_u_shaped_concentrates_at_boundaries(self):
        """U-shaped distribution should have more samples near 0 and 1."""
        t = sample_timesteps_u_shaped(10000, torch.device("cpu"), concentration=0.5)
        # Count samples in boundary regions
        near_0 = (t < 0.2).sum().item()
        near_1 = (t > 0.8).sum().item()
        middle = ((t >= 0.4) & (t <= 0.6)).sum().item()
        # Boundaries should have more samples than middle for U-shaped
        assert near_0 + near_1 > middle

    def test_logit_normal_timesteps_in_range(self):
        """Logit-normal samples should be in (0, 1)."""
        t = sample_timesteps_logit_normal(100, torch.device("cpu"))
        assert t.shape == (100,)
        assert t.min() > 0
        assert t.max() < 1

    def test_logit_normal_concentrates_in_middle(self):
        """Logit-normal with loc=0 should concentrate around 0.5."""
        t = sample_timesteps_logit_normal(10000, torch.device("cpu"), loc=0.0, scale=1.0)
        middle = ((t >= 0.3) & (t <= 0.7)).sum().item()
        # Should have more samples in middle
        assert middle > 5000

    def test_get_timestep_sampler(self):
        """Test that get_timestep_sampler returns callable."""
        for strategy in ["uniform", "u_shaped", "logit_normal"]:
            sampler = get_timestep_sampler(strategy)
            t = sampler(10, torch.device("cpu"))
            assert t.shape == (10,)

    def test_compute_log_snr(self):
        """Test log-SNR computation."""
        t = torch.tensor([0.1, 0.5, 0.9])
        log_snr = compute_log_snr(t)
        assert log_snr.shape == t.shape
        # At t=0.5, alpha=sigma=0.5, so log(1) = 0
        assert torch.abs(log_snr[1]) < 0.1
        # log_snr should increase with t (more signal)
        assert log_snr[0] < log_snr[1] < log_snr[2]


class TestOptimalTransport:
    """Tests for optimal transport coupling."""

    def test_coupling_is_doubly_stochastic(self):
        """OT coupling should be doubly stochastic."""
        noise = torch.randn(8, 4)
        data = torch.randn(8, 4)
        coupling = compute_ot_coupling(noise, data)
        assert coupling.shape == (8, 8)
        # Rows and columns should sum to 1/n
        row_sums = coupling.sum(dim=1)
        col_sums = coupling.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(8) / 8, atol=1e-4)
        assert torch.allclose(col_sums, torch.ones(8) / 8, atol=1e-4)

    def test_sample_from_coupling_returns_valid_indices(self):
        """sample_from_coupling should return valid reordering."""
        noise = torch.randn(8, 4)
        data = torch.randn(8, 4)
        coupling = compute_ot_coupling(noise, data)
        reordered = sample_from_coupling(coupling, data)
        assert reordered.shape == data.shape

    def test_sample_from_coupling_with_indices(self):
        """sample_from_coupling with return_indices should return indices."""
        noise = torch.randn(8, 4)
        data = torch.randn(8, 4)
        coupling = compute_ot_coupling(noise, data)
        reordered, indices = sample_from_coupling(coupling, data, return_indices=True)
        assert indices.shape == (8,)
        assert (indices >= 0).all()
        assert (indices < 8).all()
        # Check that reordering matches indices
        assert torch.allclose(reordered, data[indices])

    def test_ot_flow_sample_path(self):
        """ot_flow_sample_path should return valid interpolation and velocity."""
        x0 = torch.randn(4, 3)
        x1 = torch.randn(4, 3)
        t = torch.rand(4)
        x_t, target_v = ot_flow_sample_path(x0, x1, t)
        assert x_t.shape == x0.shape
        assert target_v.shape == x0.shape

    def test_ot_flow_sample_path_with_coupling(self):
        """ot_flow_sample_path should optionally return coupling."""
        x0 = torch.randn(4, 3)
        x1 = torch.randn(4, 3)
        t = torch.rand(4)
        x_t, target_v, coupling = ot_flow_sample_path(x0, x1, t, return_coupling=True)
        assert coupling.shape == (4, 4)


class TestCFGZero:
    """Tests for CFG-Zero guidance schedule."""

    def test_cfg_zero_at_boundaries(self):
        """CFG-Zero should return 1.0 at t=0 and t=1."""
        assert cfg_zero_guidance(0.0, 5.0) == 1.0
        assert cfg_zero_guidance(1.0, 5.0) == 1.0

    def test_cfg_zero_peaks_at_middle(self):
        """CFG-Zero should peak at t=0.5."""
        mid_scale = cfg_zero_guidance(0.5, 5.0)
        low_scale = cfg_zero_guidance(0.1, 5.0)
        high_scale = cfg_zero_guidance(0.9, 5.0)
        assert mid_scale > low_scale
        assert mid_scale > high_scale
        # At t=0.5, weight = 4*0.5*0.5 = 1.0, so scale = 1 + (5-1)*1 = 5
        assert abs(mid_scale - 5.0) < 1e-6

    def test_cfg_zero_symmetric(self):
        """CFG-Zero should be symmetric around t=0.5."""
        assert abs(cfg_zero_guidance(0.2, 5.0) - cfg_zero_guidance(0.8, 5.0)) < 1e-6
        assert abs(cfg_zero_guidance(0.3, 5.0) - cfg_zero_guidance(0.7, 5.0)) < 1e-6

    def test_cfg_zero_in_registry(self):
        """CFG-Zero should be accessible via get_guidance_schedule."""
        schedule = get_guidance_schedule("cfg_zero")
        result = schedule(0.5, 5.0)
        assert abs(result - 5.0) < 1e-6


class TestDiscreteFlowEnhancements:
    """Tests for discrete flow matching enhancements."""

    def test_create_time_grid_uniform(self):
        """Uniform time grid should have equal spacing."""
        grid = create_time_grid(10, torch.device("cpu"), "uniform")
        assert grid.shape == (11,)
        assert grid[0] == 0.0
        assert grid[-1] == 1.0
        # Check uniform spacing
        diffs = grid[1:] - grid[:-1]
        assert torch.allclose(diffs, diffs[0] * torch.ones_like(diffs))

    def test_create_time_grid_quadratic(self):
        """Quadratic time grid should have more steps near t=0."""
        grid = create_time_grid(10, torch.device("cpu"), "quadratic")
        assert grid.shape == (11,)
        assert grid[0] == 0.0
        assert grid[-1] == 1.0
        # Early steps should be smaller than later steps
        early_step = grid[1] - grid[0]
        late_step = grid[-1] - grid[-2]
        assert early_step < late_step

    def test_create_time_grid_cosine(self):
        """Cosine time grid should be S-shaped."""
        grid = create_time_grid(10, torch.device("cpu"), "cosine")
        assert grid.shape == (11,)
        assert grid[0] == 0.0
        assert torch.abs(grid[-1] - 1.0) < 1e-6

    def test_discrete_flow_trainer_with_custom_sampler(self):
        """DiscreteFlowTrainer should use custom timestep sampler."""
        path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))

        # Custom sampler that always returns 0.5
        def constant_sampler(batch_size, device):
            return torch.full((batch_size,), 0.5, device=device)

        trainer = DiscreteFlowTrainer(path, timestep_sampler=constant_sampler)
        t = trainer.sample_timesteps(10, torch.device("cpu"))
        assert torch.allclose(t, torch.full((10,), 0.5))

    def test_discrete_flow_trainer_default_sampler(self):
        """DiscreteFlowTrainer default should use uniform sampling."""
        path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        trainer = DiscreteFlowTrainer(path)
        t = trainer.sample_timesteps(100, torch.device("cpu"))
        assert t.shape == (100,)
        assert t.min() >= 0
        assert t.max() <= 1

    def test_discrete_flow_trainer_with_u_shaped(self):
        """DiscreteFlowTrainer should work with u-shaped timestep sampler."""
        path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))

        trainer = DiscreteFlowTrainer(
            path,
            timestep_sampler=lambda bs, dev: sample_timesteps_u_shaped(bs, dev),
        )
        t = trainer.sample_timesteps(1000, torch.device("cpu"))
        # U-shaped should have more samples near boundaries
        near_0 = (t < 0.2).sum().item()
        near_1 = (t > 0.8).sum().item()
        middle = ((t >= 0.4) & (t <= 0.6)).sum().item()
        assert near_0 + near_1 > middle
