"""Numerical stability constants and utilities.

This module centralizes all numerical stability constants used throughout medlatents
to ensure consistency and make tuning easier.

Guidelines:
- EPS_LOG: Use for safe log operations (log(x + EPS_LOG))
- EPS_DIV: Use for safe division (x / (y + EPS_DIV))
- EPS_CLAMP: Use for clamping denominators (x.clamp(min=EPS_CLAMP))
- EPS_PROB: Use for probability clamping (probs.clamp(min=EPS_PROB))
- EPS_TIME: Use for timestep boundary epsilon
"""

import torch
from torch import Tensor

# Core epsilon constants
# NOTE: All values chosen to be safe for float16 (smallest positive ~6e-8)
# For float32-only code paths, smaller values like 1e-10 are safe.
EPS_LOG = 1e-6  # For log(x + eps) - fp16-safe
EPS_DIV = 1e-6  # For safe division x / (y + eps)
EPS_CLAMP = 1e-6  # For clamping denominators
EPS_PROB = 1e-6  # For probability clamping (critical for multinomial)
EPS_TIME = 1e-5  # For timestep boundaries [eps, 1-eps]
EPS_LAYERNORM = 1e-6  # Standard LayerNorm epsilon (PyTorch default)

# Loss computation
EPS_LOSS = 1e-8  # For loss stability


def safe_log(x: Tensor, eps: float = EPS_LOG) -> Tensor:
    """Compute log(x) safely, avoiding log(0).

    Args:
        x: Input tensor (should be non-negative)
        eps: Small constant added before log

    Returns:
        log(x + eps)
    """
    return torch.log(x + eps)


def safe_log_softmax(logits: Tensor, dim: int = -1, eps: float = EPS_LOG) -> Tensor:
    """Compute log_softmax safely.

    Uses PyTorch's numerically stable implementation.

    Args:
        logits: Input logits
        dim: Dimension to compute over
        eps: Not used (kept for API compatibility)

    Returns:
        log_softmax(logits)
    """
    return torch.nn.functional.log_softmax(logits, dim=dim)


def safe_div(numerator: Tensor, denominator: Tensor, eps: float = EPS_DIV) -> Tensor:
    """Safe division avoiding divide by zero.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small constant added to denominator

    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def safe_normalize(x: Tensor, dim: int = -1, eps: float = EPS_DIV) -> Tensor:
    """Safely normalize tensor to sum to 1 along given dimension.

    Args:
        x: Input tensor (should be non-negative for probabilities)
        dim: Dimension to normalize
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    return x / (x.sum(dim=dim, keepdim=True) + eps)


def gumbel_noise(shape: tuple, device: torch.device, eps: float = EPS_LOG) -> Tensor:
    """Generate Gumbel noise for Gumbel-Softmax / Gumbel-max.

    Gumbel(0, 1) = -log(-log(U)) where U ~ Uniform(0, 1)

    Args:
        shape: Output shape
        device: Device for the tensor
        eps: Small constant for numerical stability in log

    Returns:
        Gumbel noise tensor
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits: Tensor, temperature: float = 1.0, eps: float = EPS_LOG) -> Tensor:
    """Sample from Gumbel-Softmax distribution.

    Args:
        logits: Input logits [batch, ..., vocab]
        temperature: Temperature for softmax
        eps: Epsilon for Gumbel noise

    Returns:
        Soft samples from Gumbel-Softmax
    """
    gumbel = gumbel_noise(logits.shape, logits.device, eps)
    return torch.softmax((logits + gumbel) / temperature, dim=-1)


def gumbel_max_sample(logits: Tensor, eps: float = EPS_LOG) -> Tensor:
    """Sample using Gumbel-max trick (categorical sampling).

    Args:
        logits: Input logits [batch, ..., vocab]
        eps: Epsilon for Gumbel noise

    Returns:
        Sampled indices
    """
    gumbel = gumbel_noise(logits.shape, logits.device, eps)
    return torch.argmax(logits + gumbel, dim=-1)


def clamp_probs(probs: Tensor, eps: float = EPS_PROB) -> Tensor:
    """Clamp probabilities to valid range [eps, 1-eps].

    Args:
        probs: Probability tensor
        eps: Minimum probability

    Returns:
        Clamped probabilities (still sums to ~1)
    """
    return probs.clamp(min=eps, max=1.0 - eps)


def clamp_timesteps(t: Tensor, eps: float = EPS_TIME) -> Tensor:
    """Clamp timesteps to valid range [eps, 1-eps].

    Args:
        t: Timestep tensor in [0, 1]
        eps: Boundary epsilon

    Returns:
        Clamped timesteps
    """
    return t.clamp(min=eps, max=1.0 - eps)


def entropy(probs: Tensor, dim: int = -1, eps: float = EPS_LOG) -> Tensor:
    """Compute entropy of probability distribution.

    H(p) = -sum(p * log(p))

    Args:
        probs: Probability tensor
        dim: Dimension to compute over
        eps: Epsilon for log stability

    Returns:
        Entropy values
    """
    return -(probs * safe_log(probs, eps)).sum(dim=dim)


def kl_divergence(p: Tensor, q: Tensor, dim: int = -1, eps: float = EPS_LOG) -> Tensor:
    """Compute KL divergence D_KL(p || q).

    KL(p || q) = sum(p * log(p / q))

    Args:
        p: First probability distribution (reference)
        q: Second probability distribution
        dim: Dimension to compute over
        eps: Epsilon for log stability

    Returns:
        KL divergence values
    """
    return (p * (safe_log(p, eps) - safe_log(q, eps))).sum(dim=dim)


def js_divergence(p: Tensor, q: Tensor, dim: int = -1, eps: float = EPS_LOG) -> Tensor:
    """Compute Jensen-Shannon divergence between p and q.

    JS(p || q) = 0.5 * (KL(p || m) + KL(q || m)) where m = (p + q) / 2

    Args:
        p: First probability distribution
        q: Second probability distribution
        dim: Dimension to compute over
        eps: Epsilon for log stability

    Returns:
        JS divergence values (symmetric, always finite)
    """
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m, dim, eps) + kl_divergence(q, m, dim, eps))


def validate_finite(tensor: Tensor, name: str = "tensor", raise_error: bool = True) -> bool:
    """Check if tensor contains NaN or Inf values.

    Args:
        tensor: Input tensor to validate
        name: Name for error messages
        raise_error: If True, raise ValueError on non-finite values

    Returns:
        True if tensor is finite, False otherwise (only if raise_error=False)

    Raises:
        ValueError: If tensor contains NaN or Inf and raise_error=True
    """
    is_finite = torch.isfinite(tensor).all()
    if not is_finite:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        msg = (
            f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf. "
            f"Shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}"
        )
        if raise_error:
            raise ValueError(msg)
        return False
    return True


def safe_multinomial(
    probs: Tensor,
    num_samples: int = 1,
    eps: float = EPS_PROB,
    validate: bool = True,
) -> Tensor:
    """Multinomial sampling with probability validation and clamping.

    This function ensures probabilities are valid before calling torch.multinomial,
    which can produce undefined behavior or crashes with invalid inputs.

    Args:
        probs: Probability tensor [..., vocab_size]
        num_samples: Number of samples per distribution
        eps: Minimum probability value
        validate: If True, check for NaN/Inf before sampling

    Returns:
        Sampled indices [..., num_samples]

    Raises:
        ValueError: If probs contains NaN/Inf and validate=True
    """
    if validate:
        validate_finite(probs, "probs for multinomial")

    # Clamp to avoid numerical issues
    probs = probs.clamp(min=eps)

    # Renormalize to ensure they sum to 1
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=eps)

    # Flatten for multinomial if needed, then reshape back
    original_shape = probs.shape[:-1]
    vocab_size = probs.shape[-1]
    flat_probs = probs.reshape(-1, vocab_size)

    samples = torch.multinomial(flat_probs, num_samples, replacement=(num_samples > 1))

    if num_samples == 1:
        return samples.reshape(original_shape)
    return samples.reshape(*original_shape, num_samples)


__all__ = [
    # Constants
    "EPS_LOG",
    "EPS_DIV",
    "EPS_CLAMP",
    "EPS_PROB",
    "EPS_TIME",
    "EPS_LAYERNORM",
    "EPS_LOSS",
    # Functions
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
