"""Advanced sampling techniques for discrete generative models."""

from .autoregressive import (
    TemperatureScheduler,
    sample_autoregressive,
    sample_min_p,
    sample_nucleus,
    sample_with_cfg,
)
from .bayesian_flow import (
    BayesianFlowSampler,
    get_confidence,
    sample_bayesian_flow,
    sample_with_early_stopping,
)
from .compile import (
    CompiledSampler,
    compile_function,
    compile_model,
    is_compile_available,
)
from .diffusion import (
    AncestralSamplingScheduler,
    DDIMScheduler,
    DPMSolverScheduler,
    EulerDiscreteScheduler,
)
from .discretization import DecoupledSTGumbelSoftmax, ReinMax
from .flow_diffusion import (
    CFGDiffuser,
    CFGFlowMatcher,
    generate_with_cfg,
    rescale_cfg,
    sample_with_cfg_diffusion,
    sample_with_cfg_flow,
    sample_with_dynamic_cfg,
)
from .guidance import GuidedSampler
from .maskgit import (
    KLASSGenerator,
    MaskGITScheduler,
    RCRGenerator,
    RunningConfidenceRemasker,
    RunningConfidenceScheduler,
    adaptive_masking,
    confidence_based_schedule,
    gumbel_max_sampling,
)
from .schedules import (
    GuidanceSchedule,
    cfg_zero_guidance,
    cfg_zero_star_guidance,
    constant_guidance,
    cosine_decay_guidance,
    cosine_guidance,
    dynamic_guidance,
    get_guidance_schedule,
    linear_guidance,
    triangular_guidance,
)

__all__ = [
    # Autoregressive sampling
    "sample_nucleus",
    "sample_min_p",
    "sample_autoregressive",
    "sample_with_cfg",
    "TemperatureScheduler",
    # MaskGIT sampling
    "confidence_based_schedule",
    "adaptive_masking",
    "gumbel_max_sampling",
    "MaskGITScheduler",
    # Running Confidence Remasking (2024)
    "RunningConfidenceRemasker",
    "RunningConfidenceScheduler",
    "RCRGenerator",
    # KLASS early stopping
    "KLASSGenerator",
    # Diffusion sampling
    "DDIMScheduler",
    "DPMSolverScheduler",
    "EulerDiscreteScheduler",
    "AncestralSamplingScheduler",
    # Flow/Diffusion CFG
    "sample_with_cfg_flow",
    "sample_with_cfg_diffusion",
    "CFGFlowMatcher",
    "CFGDiffuser",
    "generate_with_cfg",
    "rescale_cfg",
    "sample_with_dynamic_cfg",
    # Bayesian Flow sampling
    "sample_bayesian_flow",
    "sample_with_early_stopping",
    "get_confidence",
    "BayesianFlowSampler",
    # Discretization / Straight-Through Estimators
    "DecoupledSTGumbelSoftmax",
    "ReinMax",
    # Guidance schedules
    "GuidedSampler",
    "GuidanceSchedule",
    "get_guidance_schedule",
    "constant_guidance",
    "linear_guidance",
    "cosine_guidance",
    "cosine_decay_guidance",
    "triangular_guidance",
    "cfg_zero_guidance",
    # CFG-Zero* (2025) and Dynamic Guidance
    "cfg_zero_star_guidance",
    "dynamic_guidance",
    # torch.compile utilities
    "is_compile_available",
    "compile_model",
    "compile_function",
    "CompiledSampler",
]
