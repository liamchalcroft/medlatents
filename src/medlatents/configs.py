"""Shared configuration constants for medlatents models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ============================================================================
# Model Size Configurations
# ============================================================================


@dataclass(frozen=True)
class ModelSize:
    """Standard transformer model size configuration."""

    depth: int
    hidden_size: int
    num_heads: int


# Single source of truth for model sizes. Edit dimensions HERE only.
# Lowercase names (nano/small/base/large/xl) are the canonical keys used by the
# training scripts; MODEL_SIZES exposes the same specs as dataclasses under the
# conventional capitalized names (Nano/S/B/L/XL) for create_model_variants.
_SIZE_SPECS: dict[str, dict[str, int]] = {
    "nano": {"hidden_size": 192, "depth": 8, "num_heads": 3},
    "small": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "base": {"hidden_size": 768, "depth": 16, "num_heads": 12},
    "large": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "xl": {"hidden_size": 1536, "depth": 32, "num_heads": 24},
}

MODEL_CONFIGS: dict[str, dict[str, int]] = {name: dict(spec) for name, spec in _SIZE_SPECS.items()}

_CAPITALIZED_NAMES = {"nano": "Nano", "small": "S", "base": "B", "large": "L", "xl": "XL"}
MODEL_SIZES: dict[str, ModelSize] = {
    _CAPITALIZED_NAMES[name]: ModelSize(**spec) for name, spec in _SIZE_SPECS.items()
}

MODEL_TYPES = ["autoreg", "maskgit", "flow", "d3pm", "bayesian_flow"]


def create_model_variants(
    model_cls: type,
    base_name: str,
    sizes: dict[str, ModelSize] | None = None,
    **default_kwargs,
) -> dict[str, Callable[..., Any]]:
    """Generate model size variants (Nano/S/B/L/XL) for a model class.

    Args:
        model_cls: The model class to create variants for
        base_name: Base name for the model (e.g., "Autoreg", "MaskGIT")
        sizes: Custom size configs, defaults to MODEL_SIZES
        **default_kwargs: Default kwargs passed to all variants

    Returns:
        Dictionary mapping variant names to factory functions

    Example:
        >>> variants = create_model_variants(AutoregressiveTransformer, "Autoreg")
        >>> model = variants["Autoreg-S"](seq_length=1024, vocab_size=512)
    """
    if sizes is None:
        sizes = MODEL_SIZES

    variants = {}
    for size_name, config in sizes.items():

        def factory(cfg: ModelSize = config, **kwargs):
            merged = {**default_kwargs, **kwargs}
            return model_cls(
                depth=cfg.depth,
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                **merged,
            )

        variants[f"{base_name}-{size_name}"] = factory

    return variants


DEFAULT_ROPE_THETA = 10000.0
DEFAULT_MLP_RATIO = 4.0

# Diffusion hyperparameters
DIFFUSION_CONFIGS = {
    "num_timesteps": 1000,
    "schedule_type": "cosine",  # 'linear', 'cosine', 'quadratic', 'sigmoid'
    "transition_type": "absorbing",  # 'absorbing', 'uniform', 'gaussian'
    "hybrid_loss_coeff": 0.001,
    "loss_type": "hybrid",  # 'vb', 'hybrid', 'cross_entropy'
}

BAYESIAN_FLOW_CONFIGS = {
    "num_steps": 1000,
    "sigma_schedule": "cosine",  # 'linear', 'cosine', 'log'
    "sigma_min": 1e-4,
    "sigma_max": 1.0,
}

# Training enhancements
EMA_DECAY = 0.9999
GRAD_CLIP_DEFAULT = 1.0
WEIGHT_DECAY_DEFAULT = 0.01
