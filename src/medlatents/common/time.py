"""Unified time/timestep management for diffusion and flow models.

This module consolidates time-related utilities that were previously
duplicated across multiple modules.

Note: Some modules (diffusion/schedules.py, flow_matching/schedulers.py)
have domain-specific implementations of similar functions. These are
intentionally different - the implementations here are canonical/generic
versions that can be used across model types.

For domain-specific implementations, use:
- medlatents.diffusion.schedules for diffusion-specific SNR functions
- medlatents.flow_matching.schedulers for flow-matching timestep sampling
"""

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

TimestepSampler = Literal[
    "uniform",
    "uniform_shifted",  # Uniform with eps padding
    "u_shaped",  # Beta-distributed for boundary focus
    "logit_normal",  # Logit-normal for middle focus
    "edm",  # EDM (Karras et al.) lognormal in sigma-space
]

# ============================================================================
# TIMESTEP DISTRIBUTIONS
# ============================================================================


def sample_timesteps_uniform(batch_size: int, device: torch.device | None = None) -> Tensor:
    """Sample uniform timesteps in [0, 1]."""
    return torch.rand(batch_size, device=device)


def sample_timesteps_uniform_shifted(
    batch_size: int, eps: float = 1e-3, device: torch.device | None = None
) -> Tensor:
    """Sample uniform timesteps in [eps, 1-eps]."""
    t = torch.rand(batch_size, device=device)
    return t * (1 - 2 * eps) + eps


def sample_timesteps_u_shaped(
    batch_size: int, concentration: float = 2.0, device: torch.device | None = None
) -> Tensor:
    """
    Sample timesteps using Beta distribution for U-shaped sampling.

    Focuses on endpoints (early and late timesteps).

    Args:
        batch_size: Number of samples
        concentration: Beta concentration (>1 for U-shape)
        device: Torch device
    """
    from torch.distributions import Beta

    beta_dist = Beta(concentration, concentration)
    t = beta_dist.sample((batch_size,)).to(device)
    return t


def sample_timesteps_logit_normal(
    batch_size: int,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device | None = None,
) -> Tensor:
    """
    Sample timesteps using logit-normal distribution.

    Focuses on middle timesteps.

    Args:
        batch_size: Number of samples
        mean: Logit-normal mean
        std: Logit-normal std
        device: Torch device
    """
    from torch.distributions import Normal

    normal_dist = Normal(mean, std)
    logit_t = normal_dist.sample((batch_size,)).to(device)
    return torch.sigmoid(logit_t)


def sample_timesteps_edm(
    batch_size: int,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    device: torch.device | None = None,
) -> Tensor:
    """
    Sample timesteps using EDM's lognormal distribution in sigma-space.

    From "Elucidating the Design Space of Diffusion-Based Generative Models"
    (Karras et al., NeurIPS 2022). This sampling distribution concentrates
    more samples at intermediate noise levels where training signal is strongest.

    The noise level sigma is sampled from a lognormal distribution, then
    converted to a timestep t in [0, 1] via: t = sigma / (sigma + 1).

    Args:
        batch_size: Number of samples
        P_mean: Mean of the log-normal distribution (default -1.2 from EDM paper)
        P_std: Std of the log-normal distribution (default 1.2 from EDM paper)
        sigma_min: Minimum sigma value for clamping
        sigma_max: Maximum sigma value for clamping
        device: Torch device

    Returns:
        Timesteps in [0, 1] sampled according to EDM distribution

    Reference:
        https://arxiv.org/abs/2206.00364 (Section 5, Training)
    """
    # Sample log(sigma) from normal distribution
    log_sigma = torch.randn(batch_size, device=device) * P_std + P_mean
    sigma = log_sigma.exp()

    # Clamp sigma to valid range
    sigma = sigma.clamp(sigma_min, sigma_max)

    # Convert sigma to timestep t in [0, 1]
    # Using the relation: t = sigma / (sigma + 1)
    # At sigma=0: t=0, at sigma=inf: t=1
    t = sigma / (sigma + 1.0)

    return t


def get_timestep_sampler(
    sampler_type: TimestepSampler = "uniform", **kwargs
) -> Callable[[int, torch.device | None], Tensor]:
    """
    Get a timestep sampler function.

    Args:
        sampler_type: Type of sampler
        **kwargs: Sampler-specific parameters

    Returns:
        Function that takes (batch_size, device) and returns timesteps
    """
    samplers = {
        "uniform": lambda bs, dev=None: sample_timesteps_uniform(bs, dev),
        "uniform_shifted": lambda bs, dev=None: sample_timesteps_uniform_shifted(
            bs, kwargs.get("eps", 1e-3), dev
        ),
        "u_shaped": lambda bs, dev=None: sample_timesteps_u_shaped(
            bs, kwargs.get("concentration", 2.0), dev
        ),
        "logit_normal": lambda bs, dev=None: sample_timesteps_logit_normal(
            bs, kwargs.get("mean", 0.0), kwargs.get("std", 1.0), dev
        ),
        "edm": lambda bs, dev=None: sample_timesteps_edm(
            bs,
            kwargs.get("P_mean", -1.2),
            kwargs.get("P_std", 1.2),
            kwargs.get("sigma_min", 0.002),
            kwargs.get("sigma_max", 80.0),
            dev,
        ),
    }

    if sampler_type not in samplers:
        raise ValueError(
            f"Unknown sampler_type: {sampler_type}. Available: {list(samplers.keys())}"
        )

    return samplers[sampler_type]


# ============================================================================
# SNR / NOISE SCHEDULE UTILITIES
# ============================================================================


def apply_timestep_shift(t: Tensor, shift: float) -> Tensor:
    """
    Apply shift to timesteps.

    Args:
        t: Timestep values in [0, 1]
        shift: Shift amount

    Returns:
        Shifted timesteps clamped to [0, 1]
    """
    shifted = t + shift
    return torch.clamp(shifted, 0.0, 1.0)


# ============================================================================
# FLOW-MATCHING TIMESTEP SAMPLING (shared with flow_matching/schedulers.py)
# ============================================================================


def sample_timesteps_u_shaped_flow(
    batch_size: int,
    device: torch.device | None = None,
    concentration: float = 0.5,
) -> Tensor:
    """Sample timesteps with U-shaped distribution (more at t=0 and t=1).

    Uses Beta(concentration, concentration) distribution where concentration < 1
    gives U-shape. Default concentration=0.5 gives the arcsine distribution.

    For concentration=0.5 (arcsine), uses exact formula for numerical stability.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        concentration: Beta distribution parameter. Lower = more U-shaped.
            0.5 = arcsine (strong U), 0.75 = moderate U, 1.0 = uniform

    Returns:
        Timesteps of shape [batch_size]
    """
    import math

    if concentration == 0.5:
        u = torch.rand(batch_size, device=device)
        t = torch.sin(u * math.pi / 2).pow(2)
    else:
        from torch.distributions import Beta

        beta_dist = Beta(concentration, concentration)
        t = beta_dist.sample((batch_size,)).to(device)

    return t.clamp(1e-5, 1 - 1e-5)


def sample_timesteps_logit_normal_flow(
    batch_size: int,
    device: torch.device | None = None,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    """Sample timesteps from logit-normal distribution.

    Samples concentrate in the middle, which works well empirically for images.
    The logit-normal is obtained by passing normal samples through sigmoid.

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        loc: Mean of the underlying normal distribution
        scale: Standard deviation of the underlying normal

    Returns:
        Timesteps of shape [batch_size]
    """
    normal_samples = torch.randn(batch_size, device=device) * scale + loc
    return torch.sigmoid(normal_samples)


def get_timestep_sampler_flow(
    strategy: str,
    **kwargs,
):
    """Get a flow-matching timestep sampler function by name.

    Args:
        strategy: Sampling strategy ('uniform', 'u_shaped', 'logit_normal')
        **kwargs: Additional arguments for the sampler

    Returns:
        Function that takes (batch_size, device) and returns timesteps
    """
    samplers = {
        "uniform": sample_timesteps_uniform_shifted,
        "u_shaped": sample_timesteps_u_shaped_flow,
        "logit_normal": sample_timesteps_logit_normal_flow,
    }
    if strategy not in samplers:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(samplers.keys())}")

    sampler = samplers[strategy]

    def configured_sampler(batch_size: int, device: torch.device | None = None) -> Tensor:
        return sampler(batch_size, device=device, **kwargs)

    return configured_sampler


__all__ = [
    # Timestep sampling
    "sample_timesteps_uniform",
    "sample_timesteps_uniform_shifted",
    "sample_timesteps_u_shaped",
    "sample_timesteps_logit_normal",
    "sample_timesteps_edm",
    "get_timestep_sampler",
    "TimestepSampler",
    # SNR utilities
    "apply_timestep_shift",
    # Flow-matching specific (shared with flow_matching/schedulers.py)
    "sample_timesteps_u_shaped_flow",
    "sample_timesteps_logit_normal_flow",
    "get_timestep_sampler_flow",
]
