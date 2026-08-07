"""Utility helpers for medlatents."""

from .numerics import (
    EPS_CLAMP,
    EPS_DIV,
    EPS_LAYERNORM,
    EPS_LOG,
    EPS_LOSS,
    EPS_PROB,
    EPS_TIME,
    clamp_probs,
    clamp_timesteps,
    entropy,
    gumbel_max_sample,
    gumbel_noise,
    gumbel_softmax_sample,
    js_divergence,
    kl_divergence,
    safe_div,
    safe_log,
    safe_log_softmax,
    safe_multinomial,
    safe_normalize,
    validate_finite,
)
from .special_tokens import (
    SpecialTokenIds,
    derive_special_tokens,
    resolve_special_tokens,
)
from .state_dict import load_state_dict_compat

__all__ = [
    # Special tokens
    "SpecialTokenIds",
    "derive_special_tokens",
    "resolve_special_tokens",
    # State dict
    "load_state_dict_compat",
    # Numerics - constants
    "EPS_LOG",
    "EPS_DIV",
    "EPS_CLAMP",
    "EPS_PROB",
    "EPS_TIME",
    "EPS_LAYERNORM",
    "EPS_LOSS",
    # Numerics - functions
    "safe_log",
    "safe_log_softmax",
    "safe_div",
    "safe_normalize",
    "gumbel_noise",
    "gumbel_softmax_sample",
    "gumbel_max_sample",
    "clamp_probs",
    "clamp_timesteps",
    "entropy",
    "kl_divergence",
    "js_divergence",
    "validate_finite",
    "safe_multinomial",
]
