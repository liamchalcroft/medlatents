"""Bayesian Flow Networks for discrete latent modelling.

This module provides:
- BayesianFlowTransformer: Core transformer architecture
- Entropy encoding utilities for uncertainty-aware generation
- Conditional sampling methods (score-guided)
- Accuracy schedules for controlling information flow
- ODE/SDE solvers for generation

References:
- Graves et al. "Bayesian Flow Networks" (2023)
- Xue et al. "Unifying Bayesian Flow Networks and Diffusion Models through SDEs" (2024)
"""

from .entropy import (
    compute_entropy,
    encode_with_entropy,
)
from .guided_samplers import ScoreGuidedSampler
from .loss import (
    ResidualLossWrapper,
)
from .schedules import (
    cosine_schedule,
    exponential_schedule,
    get_bfn_schedule,
    linear_entropy_schedule,
)
from .solvers import (
    BaseBFNSolver,
    DPMSolver2,
    DPMSolver3,
    EulerSolver,
    ExponentialIntegrator,
    HeunSolver,
    StochasticHeun,
    get_bfn_solver,
)
from .transformer import (
    BayesianFlowTransformer,
    BFN_models,
)

__all__ = [
    # Core model
    "BayesianFlowTransformer",
    "BFN_models",
    # Entropy encoding
    "compute_entropy",
    "encode_with_entropy",
    # Conditional sampling
    "ScoreGuidedSampler",
    # Accuracy schedules
    "exponential_schedule",
    "linear_entropy_schedule",
    "cosine_schedule",
    "get_bfn_schedule",
    # Training utilities
    "ResidualLossWrapper",
    # ODE/SDE solvers
    "BaseBFNSolver",
    "EulerSolver",
    "HeunSolver",
    "DPMSolver2",
    "DPMSolver3",
    "ExponentialIntegrator",
    "StochasticHeun",
    "get_bfn_solver",
]
