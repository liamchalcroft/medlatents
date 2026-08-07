"""Discrete and continuous latent generative models for medical imagery."""

import os
import sys

if sys.version_info < (3, 10):  # noqa: UP036
    raise RuntimeError(
        "medlatents requires Python 3.10+. "
        f"Detected {sys.version_info.major}.{sys.version_info.minor}."
    )

# Keep cuBLAS workspace bounded before first CUDA context initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

from . import (
    autoregressive,
    bayesian_flow,
    configs,
    data,
    diffusion,
    flow_matching,
    generation,
    inference,
    maskgit,
    networks,
    rasterization,
    sampling,
    training,
)

# Import pydantic configs (optional - requires pydantic to be installed)
PYDANTIC_AVAILABLE = False
try:
    from .config_models import PYDANTIC_AVAILABLE as _PA

    PYDANTIC_AVAILABLE = _PA
    if PYDANTIC_AVAILABLE:
        from .config_models import (  # noqa: F401
            BASE_CONFIG,
            NANO_CONFIG,
            SMALL_CONFIG,
            ExperimentConfig,
            ModelConfig,
            TrainingConfig,
        )
except (ImportError, ModuleNotFoundError):
    pass

# Top-level imports
from .autoregressive import Autoreg_models, AutoregressiveTransformer
from .generation import ContinuousLatentGenerator, DiscreteLatentGenerator
from .maskgit import MaskGIT, MaskGIT_models
from .networks import (
    ContinuousDiT,
    ContinuousDiT_models,
    DiscreteDiT,
    DiscreteDiT_models,
)
from .training import ContinuousLatentTrainer, DiscreteLatentTrainer

__version__ = "0.1.0"


def _warmup_cuda_rng_state() -> None:
    """Initialize CUDA RNG state once so memory baselines are stable across runs."""
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    try:
        sample = torch.randint(0, 2, (1,), device="cuda")
        del sample
    except Exception:
        # Never fail import because of optional CUDA warmup.
        return


_warmup_cuda_rng_state()

__all__ = [
    "configs",
    "data",
    "training",
    "generation",
    "inference",
    "networks",
    "autoregressive",
    "maskgit",
    "diffusion",
    "bayesian_flow",
    "flow_matching",
    "sampling",
    "rasterization",
    "AutoregressiveTransformer",
    "MaskGIT",
    "DiscreteDiT",
    "ContinuousDiT",
    "Autoreg_models",
    "MaskGIT_models",
    "DiscreteDiT_models",
    "ContinuousDiT_models",
    "DiscreteLatentTrainer",
    "ContinuousLatentTrainer",
    "DiscreteLatentGenerator",
    "ContinuousLatentGenerator",
]
