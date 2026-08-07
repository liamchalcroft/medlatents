"""MaskGIT sampling subpackage.

This module provides advanced sampling techniques for MaskGIT-style
iterative parallel decoding.

Submodules:
- compile: Compile-friendly pure kernels for torch.compile(fullgraph=True)
- schedules: Halton sequences and confidence-based masking schedules
- sampling: Gumbel sampling and MaskGITScheduler
- klass: KLASS early stopping generator
- remasking: Running confidence remasking strategies
- critic: Token critic for quality-guided generation

Main exports (for backwards compatibility):
"""

from .compile import (
    CompileCache,
    get_confidences,
    sample_with_gumbel,
    step_final_kernel,
    step_normal_kernel,
)
from .critic import (
    CriticGuidedGenerator,
    TokenCritic,
    train_critic_step,
)
from .klass import (
    KLASSGenerator,
    compute_js_divergence,
    compute_kl_divergence,
)
from .remasking import (
    RCRGenerator,
    RunningConfidenceRemasker,
    RunningConfidenceScheduler,
)
from .sampling import (
    MaskGITScheduler,
    gumbel_max_sampling,
    sample_with_reranking,
)
from .schedules import (
    adaptive_masking,
    confidence_based_schedule,
    halton_schedule_1d,
    halton_schedule_2d,
    halton_sequence,
)

__all__ = [
    # Compile kernels
    "sample_with_gumbel",
    "get_confidences",
    "step_normal_kernel",
    "step_final_kernel",
    "CompileCache",
    # Schedules
    "halton_sequence",
    "halton_schedule_1d",
    "halton_schedule_2d",
    "confidence_based_schedule",
    "adaptive_masking",
    # Sampling
    "gumbel_max_sampling",
    "sample_with_reranking",
    "MaskGITScheduler",
    # KLASS
    "KLASSGenerator",
    "compute_kl_divergence",
    "compute_js_divergence",
    # Remasking
    "RunningConfidenceRemasker",
    "RunningConfidenceScheduler",
    "RCRGenerator",
    # Critic
    "TokenCritic",
    "CriticGuidedGenerator",
    "train_critic_step",
]
