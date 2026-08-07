"""Timestep sampling strategies and conditioning for flow matching.

Provides different strategies for sampling training timesteps, which can
significantly affect learning dynamics and sample quality.

Reference:
    - SD3: Stable Diffusion 3 (timestep shifting)
    - "Flow Matching for Generative Modeling" (Lipman et al., 2022)

Note: u_shaped, logit_normal samplers and get_timestep_sampler are imported
from common/time.py to avoid duplication.
"""

from __future__ import annotations

import torch
from torch import Tensor

from medlatents.common.time import (
    get_timestep_sampler_flow,
    sample_timesteps_logit_normal_flow,
    sample_timesteps_u_shaped_flow,
)

__all__ = [
    "sample_timesteps_uniform",
    "sample_timesteps_u_shaped",
    "sample_timesteps_logit_normal",
    "compute_log_snr",
    "get_timestep_sampler",
    "apply_timestep_shift",
]


def sample_timesteps_uniform(
    batch_size: int,
    device: torch.device,
    eps: float = 1e-5,
) -> Tensor:
    """Sample timesteps uniformly from [eps, 1-eps].

    Clamping avoids boundary issues:
    - t=0: pure noise with no signal about target
    - t=1: trivially small velocity to predict

    Args:
        batch_size: Number of timesteps to sample
        device: Device to create tensor on
        eps: Boundary epsilon to avoid exact 0 or 1

    Returns:
        Timesteps of shape [batch_size]
    """
    t = torch.rand(batch_size, device=device)
    return t * (1 - 2 * eps) + eps


sample_timesteps_u_shaped = sample_timesteps_u_shaped_flow
sample_timesteps_logit_normal = sample_timesteps_logit_normal_flow
get_timestep_sampler = get_timestep_sampler_flow


def compute_log_snr(
    t: Tensor,
    alpha_fn=None,
    sigma_fn=None,
) -> Tensor:
    """Compute log signal-to-noise ratio: log(α²/σ²).

    For standard rectified flow with x_t = (1-t)*x_0 + t*x_1:
    - α(t) = t (signal coefficient)
    - σ(t) = 1-t (noise coefficient)

    Log-SNR conditioning can improve sample quality by providing a more
    informative time representation to the network.

    Args:
        t: Timesteps [batch] or [batch, 1]
        alpha_fn: Optional custom alpha schedule
        sigma_fn: Optional custom sigma schedule (required if alpha_fn provided)

    Returns:
        Log-SNR values with same shape as t
    """
    if alpha_fn is None:
        alpha_t = t
        sigma_t = 1 - t
    else:
        if sigma_fn is None:
            raise ValueError("sigma_fn must be provided when alpha_fn is provided")
        alpha_t = alpha_fn(t)
        sigma_t = sigma_fn(t)

    alpha_t = alpha_t.clamp(min=1e-8)
    sigma_t = sigma_t.clamp(min=1e-8)

    return torch.log(alpha_t.pow(2) / sigma_t.pow(2))


def apply_timestep_shift(t: Tensor, shift: float) -> Tensor:
    """Apply timestep shift from SD3.

    Shifts timesteps to focus training on later parts of the flow.
    When shift > 1, more samples are drawn near t=1 (close to data).

    Args:
        t: Timesteps [batch]
        shift: Shift factor. 1.0 = no shift, >1 = more near t=1

    Returns:
        Shifted timesteps
    """
    if shift == 1.0:
        return t
    return t / (1 + (shift - 1) * (1 - t))
