"""Flow matching utilities for discrete and continuous generative models.

Core components for flow matching:
- Path construction (MixtureDiscreteProbPath)
- Source distributions (masked, uniform)
- Loss functions (cross-entropy, generalized KL)
- Evaluation utilities (entropy, likelihood)
- Generation utilities (sampling with solvers)
- Timestep sampling strategies (uniform, u-shaped, logit-normal)
- Optimal transport coupling for straighter flow paths

For classifier-free guidance, use medlatents.sampling.sample_with_cfg_flow
"""

from .continuous import RectifiedFlow, RectifiedFlowPP
from .core import (
    DiscreteFlowTrainer,
    MixtureDiscreteEulerSolver,
    MixtureDiscreteProbPath,
    MixturePathGeneralizedKL,
    PolynomialConvexScheduler,
    create_time_grid,
)
from .discrete import (
    MaskedSourceDistribution,
    SourceDistribution,
    UniformSourceDistribution,
    get_loss_function,
    get_path,
    get_source_distribution,
)
from .evaluate import compute_entropy, estimate_likelihood
from .generate import WrappedModel, generate_counterfactual, generate_samples
from .optimal_transport import (
    compute_ot_coupling,
    ot_flow_sample_path,
    sample_from_coupling,
)
from .schedulers import (
    compute_log_snr,
    get_timestep_sampler,
    sample_timesteps_logit_normal,
    sample_timesteps_u_shaped,
    sample_timesteps_uniform,
)
from .shortcut import (
    AdaptiveStepSampler,
    ShortcutFlowMatchingLoss,
    ShortcutFlowMatchingModel,
    StepInvariantModel,
    TimeInvariantVectorField,
)

__all__ = [
    # Evaluation
    "compute_entropy",
    "estimate_likelihood",
    # Flow setup
    "get_path",
    "get_source_distribution",
    "get_loss_function",
    "SourceDistribution",
    "MaskedSourceDistribution",
    "UniformSourceDistribution",
    # Generation
    "generate_samples",
    "generate_counterfactual",
    "WrappedModel",
    "RectifiedFlow",
    "RectifiedFlowPP",
    # Discrete flow core
    "MixtureDiscreteProbPath",
    "MixtureDiscreteEulerSolver",
    "MixturePathGeneralizedKL",
    "PolynomialConvexScheduler",
    "create_time_grid",
    "DiscreteFlowTrainer",
    # Timestep sampling
    "sample_timesteps_uniform",
    "sample_timesteps_u_shaped",
    "sample_timesteps_logit_normal",
    "compute_log_snr",
    "get_timestep_sampler",
    # Optimal transport
    "compute_ot_coupling",
    "sample_from_coupling",
    "ot_flow_sample_path",
    # Shortcut models for flexible step count (2024)
    "ShortcutFlowMatchingModel",
    "TimeInvariantVectorField",
    "StepInvariantModel",
    "ShortcutFlowMatchingLoss",
    "AdaptiveStepSampler",
]
