"""Noise schedules for discrete diffusion models.

Includes improvements from 2024-2025 literature:
- Zero-Terminal SNR correction (WACV 2024)
- Min-SNR loss weighting
- Log-SNR importance sampling

References:
- "Common Diffusion Noise Schedules and Sample Steps are Flawed" (WACV 2024)
- "Efficient Diffusion Training via Min-SNR Weighting Strategy"

Note: Shared SNR utilities (compute_snr, compute_log_snr, min_snr_weighting, enforce_zero_terminal_snr)
are imported from common/time.py to avoid duplication.
"""

import math
from typing import Literal

import torch


def get_beta_schedule(
    schedule_type: Literal["linear", "cosine", "quadratic", "sigmoid"],
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """
    Get noise schedule (beta) for diffusion process.

    Args:
        schedule_type: Type of schedule
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value (for linear/sigmoid)
        beta_end: Ending beta value (for linear/sigmoid)
        cosine_s: Offset for cosine schedule

    Returns:
        betas: [num_timesteps] beta values
    """
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)

    elif schedule_type == "cosine":
        # Improved cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / num_timesteps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    elif schedule_type == "quadratic":
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        return betas

    elif schedule_type == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def get_absorbing_transition_mat(
    num_classes: int, beta: float, absorbing_state: int | None = None
) -> torch.Tensor:
    """
    Get transition matrix for absorbing diffusion (D3PM variant).

    The absorbing state is a special "mask" state that tokens transition to.

    Args:
        num_classes: Number of classes (vocabulary size)
        beta: Noise level at this timestep
        absorbing_state: Index of absorbing state (default: num_classes)

    Returns:
        Q_t: [num_classes+1, num_classes+1] transition matrix
    """
    if absorbing_state is None:
        absorbing_state = num_classes

    # Create transition matrix
    Q_t = torch.zeros(num_classes + 1, num_classes + 1)

    # Diagonal: probability of staying in current state
    Q_t[:num_classes, :num_classes] = torch.eye(num_classes) * (1 - beta)

    # Transition to absorbing state with probability beta
    Q_t[:num_classes, absorbing_state] = beta

    # Absorbing state stays absorbing
    Q_t[absorbing_state, absorbing_state] = 1.0

    return Q_t


def get_uniform_transition_mat(num_classes: int, beta: float) -> torch.Tensor:
    """
    Get transition matrix for uniform diffusion (original D3PM).

    Tokens transition uniformly to all other classes.

    Args:
        num_classes: Number of classes
        beta: Noise level at this timestep

    Returns:
        Q_t: [num_classes, num_classes] transition matrix
    """
    # Diagonal: probability of staying in current state
    Q_t = torch.eye(num_classes) * (1 - beta)

    # Off-diagonal: uniform probability to all other states
    Q_t += beta / num_classes

    return Q_t


def get_discretized_gaussian_transition_mat(
    num_classes: int, beta: float, sigma: float = 1.0
) -> torch.Tensor:
    """
    Get transition matrix based on discretized Gaussian kernel.

    This creates a "blurring" effect similar to continuous diffusion.

    Args:
        num_classes: Number of classes
        beta: Noise level at this timestep
        sigma: Gaussian standard deviation

    Returns:
        Q_t: [num_classes, num_classes] transition matrix
    """
    # Create Gaussian kernel
    indices = torch.arange(num_classes).float()
    distances = (indices.unsqueeze(0) - indices.unsqueeze(1)) ** 2
    gaussian = torch.exp(-distances / (2 * sigma**2))

    # Normalize rows
    gaussian = gaussian / gaussian.sum(dim=1, keepdim=True)

    # Interpolate between identity and Gaussian based on beta
    Q_t = (1 - beta) * torch.eye(num_classes) + beta * gaussian

    return Q_t


def compute_transition_matrices(
    betas: torch.Tensor,
    num_classes: int,
    transition_type: Literal["absorbing", "uniform", "gaussian"] = "absorbing",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute transition matrices for all timesteps.

    Args:
        betas: [num_timesteps] beta schedule
        num_classes: Number of classes
        transition_type: Type of transition matrix

    Returns:
        Q_t: [num_timesteps, num_classes, num_classes] single-step transitions
        Q_bar_t: [num_timesteps, num_classes, num_classes] cumulative transitions
        Q_bar_t_minus_1: [num_timesteps, num_classes, num_classes] cumulative for t-1
    """
    num_timesteps = len(betas)

    # Choose transition matrix function
    if transition_type == "absorbing":

        def get_Q_fn(beta):
            return get_absorbing_transition_mat(num_classes, beta)

        mat_size = num_classes + 1
    elif transition_type == "uniform":

        def get_Q_fn(beta):
            return get_uniform_transition_mat(num_classes, beta)

        mat_size = num_classes
    elif transition_type == "gaussian":

        def get_Q_fn(beta):
            return get_discretized_gaussian_transition_mat(num_classes, beta)

        mat_size = num_classes
    else:
        raise ValueError(f"Unknown transition type: {transition_type}")

    # Compute single-step transitions
    Q_t = torch.stack([get_Q_fn(beta.item()) for beta in betas])

    # Compute cumulative transitions Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t
    Q_bar_t = torch.zeros(num_timesteps, mat_size, mat_size)
    Q_bar_t[0] = Q_t[0]
    for t in range(1, num_timesteps):
        Q_bar_t[t] = Q_bar_t[t - 1] @ Q_t[t]

    # Compute Q_bar_{t-1} (shifted)
    Q_bar_t_minus_1 = torch.zeros(num_timesteps, mat_size, mat_size)
    Q_bar_t_minus_1[0] = torch.eye(mat_size)  # Q_bar_0 = I
    Q_bar_t_minus_1[1:] = Q_bar_t[:-1]

    return Q_t, Q_bar_t, Q_bar_t_minus_1


# =============================================================================
# Zero-Terminal SNR and SNR Utilities (2024 improvements)
# =============================================================================
# Note: These are diffusion-specific implementations. For general SNR utilities,
# see medlatents.common.time (compute_snr, compute_log_snr, min_snr_weighting).
# The diffusion-specific versions here use slightly different formulations
# optimized for discrete diffusion models.


def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Enforce zero terminal SNR by rescaling alphas_cumprod.

    Standard schedules have non-zero SNR at the final timestep,
    causing train/inference mismatch. This correction ensures
    the schedule ends at pure noise (SNR=0).

    Reference: "Common Diffusion Noise Schedules and Sample Steps are Flawed" (WACV 2024)

    Args:
        betas: Original beta schedule [timesteps]

    Returns:
        Corrected betas with zero terminal SNR
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Rescale sqrt(alpha_cumprod) to have zero terminal value
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)

    # Shift so terminal is zero
    alphas_cumprod_sqrt_shifted = alphas_cumprod_sqrt - alphas_cumprod_sqrt[-1]
    alphas_cumprod_sqrt_shifted = alphas_cumprod_sqrt_shifted / alphas_cumprod_sqrt_shifted[0]

    # Convert back to alphas_cumprod and betas
    alphas_cumprod_new = alphas_cumprod_sqrt_shifted**2
    alphas_cumprod_new[-1] = 0.0  # Exactly zero terminal

    # Reconstruct alphas from alphas_cumprod
    alphas_new = alphas_cumprod_new.clone()
    alphas_new[1:] = alphas_cumprod_new[1:] / alphas_cumprod_new[:-1].clamp(min=1e-8)
    betas_new = 1.0 - alphas_new

    return betas_new.clamp(1e-5, 0.9999)


def compute_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio from alphas_cumprod.

    SNR(t) = alpha_bar_t / (1 - alpha_bar_t)

    Args:
        alphas_cumprod: Cumulative product of alphas [timesteps]

    Returns:
        SNR values [timesteps]
    """
    return alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)


def compute_log_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Compute log Signal-to-Noise Ratio.

    log_SNR(t) = log(alpha_bar_t) - log(1 - alpha_bar_t)

    Useful for importance sampling (sample more at log_SNR ≈ 0).
    """
    return torch.log(alphas_cumprod.clamp(min=1e-8)) - torch.log(
        (1.0 - alphas_cumprod).clamp(min=1e-8)
    )


def min_snr_weighting(snr: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """Min-SNR weighting for loss.

    Reweights the loss to emphasize timesteps with lower SNR,
    which are harder to denoise.

    Reference: "Efficient Diffusion Training via Min-SNR Weighting Strategy"

    Args:
        snr: SNR values at each timestep
        gamma: Min-SNR gamma value (higher = more uniform weighting)

    Returns:
        Weight for each timestep
    """
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr


def get_beta_schedule_with_zero_snr(
    schedule_type: Literal["linear", "cosine", "quadratic", "sigmoid"],
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    cosine_s: float = 0.008,
) -> torch.Tensor:
    """Get noise schedule with Zero-Terminal SNR correction.

    This ensures the final timestep has SNR=0 (pure noise),
    fixing the train/inference mismatch in standard schedules.

    Args:
        schedule_type: Type of schedule
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value (for linear/sigmoid)
        beta_end: Ending beta value (for linear/sigmoid)
        cosine_s: Offset for cosine schedule

    Returns:
        betas: [num_timesteps] corrected beta values
    """
    # Get base schedule
    betas = get_beta_schedule(schedule_type, num_timesteps, beta_start, beta_end, cosine_s)
    # Apply zero terminal SNR correction
    return enforce_zero_terminal_snr(betas)


def sample_timesteps_log_snr_importance(
    batch_size: int,
    device: torch.device,
    log_snr: torch.Tensor,
    concentration: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps with importance sampling around log_SNR = 0.

    The log_SNR = 0 point (where signal and noise are equal) is often
    the most important for learning. This samples more timesteps there.

    Args:
        batch_size: Number of timesteps to sample
        device: Device for output tensor
        log_snr: Pre-computed log-SNR values [num_timesteps]
        concentration: Higher = more concentration around log_SNR=0

    Returns:
        Sampled timestep indices [batch_size]
    """
    # Importance weights: higher near log_SNR = 0
    weights = torch.exp(-concentration * log_snr.abs())
    weights = weights / weights.sum()

    # Sample according to weights
    indices = torch.multinomial(weights.to(device), batch_size, replacement=True)

    return indices
