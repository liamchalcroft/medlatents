"""Noise/accuracy schedules for Bayesian Flow Networks.

BFN uses accuracy schedules (alpha, beta) that control how information
flows from data to the prior distribution. These schedules determine
the rate at which the model receives information during training and
how quickly samples converge during generation.

References:
- Graves et al. "Bayesian Flow Networks" (2023)
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from jaxtyping import Float


def exponential_schedule(
    t: Float[torch.Tensor, "batch"],
    beta_final: float = 1.0,
) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """
    Exponential accuracy schedule for discrete BFN.

    alpha(t) = 2^t * beta_final
    beta(t) = t^2 * beta_final

    This schedule provides smooth accuracy growth that works well
    for discrete token generation.

    Args:
        t: Time in [0, 1]
        beta_final: Final accuracy (default 1.0 for discrete)

    Returns:
        alpha: Instantaneous accuracy rate
        beta: Cumulative accuracy
    """
    t_clamped = t.clamp(0.0, 1.0)
    alpha = (2**t_clamped) * beta_final
    beta = (t_clamped**2) * beta_final
    return alpha, beta


def linear_entropy_schedule(
    t: Float[torch.Tensor, "batch"],
    sigma_final: float = 0.0141,
) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """
    Continuous schedule where entropy decreases linearly with time.

    Designed such that the entropy of the posterior decreases
    linearly from maximum (uniform) at t=0 to near-zero at t=1.
    sigma_final = 1/sqrt(5001) ≈ 0.0141 is a good default.

    Args:
        t: Time in [0, 1]
        sigma_final: Final noise level

    Returns:
        alpha: Instantaneous accuracy rate
        beta: Cumulative accuracy
    """
    t_clamped = t.clamp(0.0, 1.0)

    alpha = -2 * math.log(sigma_final) * (sigma_final**2) * t_clamped
    beta = (sigma_final**-2) ** t_clamped - 1

    return alpha, beta


def cosine_schedule(
    t: Float[torch.Tensor, "batch"],
    beta_max: float = 1.0,
    s: float = 0.008,
) -> tuple[Float[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
    """
    Cosine accuracy schedule (adapted from diffusion models).

    Provides smoother transitions at the boundaries compared to
    linear schedules.

    Args:
        t: Time in [0, 1]
        beta_max: Maximum cumulative accuracy
        s: Small offset to prevent singularity at t=0

    Returns:
        alpha: Instantaneous accuracy rate
        beta: Cumulative accuracy
    """
    t_clamped = t.clamp(0.0, 1.0)

    # Cosine schedule
    f_t = torch.cos((t_clamped + s) / (1 + s) * math.pi / 2) ** 2
    f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    beta = beta_max * (1 - f_t / f_0)

    # Compute alpha from beta derivative
    # alpha = d(beta)/dt
    alpha = (
        beta_max * math.pi / (2 * (1 + s)) * torch.sin((t_clamped + s) / (1 + s) * math.pi) / f_0
    )

    return alpha, beta


def get_bfn_schedule(
    name: str,
) -> Callable:
    """
    Get a BFN accuracy schedule by name.

    Args:
        name: Schedule name ('exponential', 'linear_entropy', 'cosine')

    Returns:
        Schedule function that takes ``(t, **kwargs)`` and returns ``(alpha, beta)``

    Raises:
        ValueError: If schedule name is unknown
    """
    schedules = {
        "exponential": exponential_schedule,
        "linear_entropy": linear_entropy_schedule,
        "cosine": cosine_schedule,
    }

    if name not in schedules:
        raise ValueError(f"Unknown BFN schedule: {name}. Available: {list(schedules.keys())}")

    return schedules[name]


__all__ = [
    "exponential_schedule",
    "linear_entropy_schedule",
    "cosine_schedule",
    "get_bfn_schedule",
]
