"""D3PM: Discrete Denoising Diffusion Probabilistic Models.

D3PM provides the diffusion process (forward/reverse transitions) for discrete data.
For the neural network backbone, use DiscreteDiT from medlatents.networks:

    from medlatents.networks import DiscreteDiT_models
    from medlatents.diffusion import D3PM

    # Create network and diffusion process
    dit = DiscreteDiT_models['DiscreteDiT-S'](vocab_size=1024, ...)
    d3pm = D3PM(num_classes=1024, num_timesteps=1000, ...)

For continuous latents, pair ContinuousGaussianDiffusion with ContinuousDiT.
See medlatents.sampling.diffusion for sampling schedulers (DDIM, DPM-Solver, etc.).
"""

from .continuous import ContinuousGaussianDiffusion
from .d3pm import D3PM, D3PMKLASS
from .schedules import (
    compute_log_snr,
    compute_snr,
    compute_transition_matrices,
    enforce_zero_terminal_snr,
    get_absorbing_transition_mat,
    get_beta_schedule,
    get_beta_schedule_with_zero_snr,
    get_discretized_gaussian_transition_mat,
    get_uniform_transition_mat,
    min_snr_weighting,
    sample_timesteps_log_snr_importance,
)
from .sedd import (
    ContinuousTimeMDLM,
    MDLMLoss,
    ScoreEntropyLoss,
    SEDDLoss,
)

__all__ = [
    "D3PM",
    "D3PMKLASS",
    "ContinuousGaussianDiffusion",
    # Schedules
    "get_beta_schedule",
    "get_absorbing_transition_mat",
    "get_uniform_transition_mat",
    "get_discretized_gaussian_transition_mat",
    "compute_transition_matrices",
    # Zero-Terminal SNR improvements (WACV 2024)
    "enforce_zero_terminal_snr",
    "compute_snr",
    "compute_log_snr",
    "min_snr_weighting",
    "get_beta_schedule_with_zero_snr",
    "sample_timesteps_log_snr_importance",
    # SEDD and MDLM losses (2024)
    "SEDDLoss",
    "MDLMLoss",
    "ContinuousTimeMDLM",
    "ScoreEntropyLoss",
]
