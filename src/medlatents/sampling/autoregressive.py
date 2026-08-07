"""Sampling techniques for autoregressive discrete generative models.

Autoregressive models work for discrete data by modeling sequences:
    p(x_1, x_2, ..., x_n) = ∏ p(x_i | x_1, ..., x_{i-1})

For spatial data (images, volumes), use consistent rasterization order:
- 2D images: raster scan, Hilbert curve, Z-order
- 3D volumes: slice-by-slice raster, 3D Hilbert curve

Available sampling techniques:
- Min-p sampling: Scales with model confidence (recommended)
- Nucleus (top-p): Dynamic vocabulary filtering
- Temperature scheduling: Start high (exploration), end low (refinement)
- Classifier-Free Guidance (CFG): Strengthen conditioning signal

See rasterization module for converting spatial data to sequences.
"""

import math

import torch
import torch.nn.functional as F


def _ensure_finite_positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def _ensure_probability(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    lower_ok = value >= 0 if allow_zero else value > 0
    if not (lower_ok and value <= 1.0):
        lower_bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {lower_bound} and <= 1.0, got {value}")


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k sampling: keep only top k logits.

    Args:
        logits: [batch, vocab_size] unnormalized logits
        k: number of top logits to keep

    Returns:
        filtered_logits: [batch, vocab_size] with low-probability logits set to -inf
    """
    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k)}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k <= 0:
        return logits

    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, torch.full_like(logits, -float("Inf")), logits)


def sample_nucleus(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep smallest set of tokens with cumulative probability >= p.

    More dynamic than top-k: adapts to the confidence of the model.
    Useful for discrete sampling in any domain including medical imaging.

    Args:
        logits: [batch, vocab_size] unnormalized logits
        p: cumulative probability threshold (0 < p <= 1)

    Returns:
        filtered_logits: [batch, vocab_size] with low-probability logits set to -inf
    """
    _ensure_probability(p, "p")
    if p >= 1.0:
        return logits

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[:, 0] = False

    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )

    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = -float("Inf")
    return filtered_logits


def sample_min_p(
    logits: torch.Tensor, min_p: float = 0.05, base_top_p: float = 1.0
) -> torch.Tensor:
    """
    Min-p sampling: Remove tokens with probability < min_p * max_probability.

    Better than top-p: scales with model confidence instead of using fixed threshold.
    When model is confident, it's more selective. When uncertain, it's more diverse.

    Works well for discrete latent codes across domains.

    Reference: https://github.com/ggerganov/llama.cpp/pull/3841

    Args:
        logits: [batch, vocab_size] unnormalized logits
        min_p: minimum probability threshold (relative to max)
        base_top_p: optional top-p to apply first

    Returns:
        filtered_logits: [batch, vocab_size] with low-probability logits set to -inf
    """
    _ensure_probability(min_p, "min_p", allow_zero=True)
    _ensure_probability(base_top_p, "base_top_p")
    # Optional: apply top-p first
    if base_top_p < 1.0:
        logits = sample_nucleus(logits, base_top_p)

    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1, keepdim=True)[0]
    threshold = min_p * max_probs

    filtered_logits = logits.clone()
    filtered_logits[probs < threshold] = -float("Inf")
    return filtered_logits


def sample_with_cfg(
    conditional_logits: torch.Tensor,
    unconditional_logits: torch.Tensor,
    guidance_scale: float = 1.5,
) -> torch.Tensor:
    """
    Classifier-Free Guidance: Interpolate between conditional and unconditional predictions.

    Useful for conditional generation tasks such as:
    - Generating specific modalities or conditions
    - Conditioning on metadata
    - Guided inpainting/reconstruction

    Formula: logits_cfg = unconditional + guidance_scale * (conditional - unconditional)

    Args:
        conditional_logits: [batch, vocab_size] logits with conditioning
        unconditional_logits: [batch, vocab_size] logits without conditioning
        guidance_scale: strength of guidance (1.0 = no guidance, >1.0 = stronger conditioning)

    Returns:
        guided_logits: [batch, vocab_size]
    """
    return unconditional_logits + guidance_scale * (conditional_logits - unconditional_logits)


class TemperatureScheduler:
    """
    Dynamic temperature scheduling during generation.

    Start with high temperature (exploration), end with low (refinement):
    - High temperature early: Explore different structures
    - Low temperature late: Refine details with high confidence

    Supports multiple schedules:
    - constant: fixed temperature
    - linear: linearly decrease temperature
    - cosine: smooth annealing (recommended)
    - exponential: exponential decay
    - adaptive: adjust based on entropy (experimental)
    """

    def __init__(
        self,
        schedule: str = "constant",
        start_temp: float = 1.0,
        end_temp: float = 0.7,
        num_steps: int | None = None,
    ):
        """
        Args:
            schedule: 'constant', 'linear', 'cosine', 'exponential', or 'adaptive'
            start_temp: initial temperature (higher = more exploration)
            end_temp: final temperature (lower = more confident)
            num_steps: number of generation steps (required for non-constant schedules)
        """
        self.schedule = schedule
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.num_steps = num_steps

    def get_temperature(self, step: int, logits: torch.Tensor | None = None) -> float:
        """Get temperature for current step."""
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if self.schedule == "constant":
            return self.start_temp

        if self.num_steps is None:
            raise ValueError(
                f"num_steps is required for '{self.schedule}' schedule. "
                "Example: TemperatureScheduler(schedule='linear', start_temp=1.0, "
                "end_temp=0.5, num_steps=100)"
            )
        if step >= self.num_steps:
            raise ValueError(f"step must be < num_steps ({self.num_steps}), got {step}")

        progress = step / max(self.num_steps - 1, 1)

        if self.schedule == "linear":
            return self.start_temp + (self.end_temp - self.start_temp) * progress

        elif self.schedule == "cosine":
            # Smooth annealing (recommended for images)
            return self.end_temp + (self.start_temp - self.end_temp) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        elif self.schedule == "exponential":
            return self.start_temp * (self.end_temp / self.start_temp) ** progress

        elif self.schedule == "adaptive":
            # Adjust temperature based on entropy (experimental)
            if logits is None:
                return self.start_temp
            probs = F.softmax(logits / self.start_temp, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            # Higher entropy -> higher temperature (more uncertainty)
            vocab_size = logits.size(-1)
            target_entropy = math.log(vocab_size) / 2
            temp_adjust = (entropy / target_entropy).item()
            return self.start_temp * temp_adjust

        else:
            raise ValueError(
                f"Unknown schedule: '{self.schedule}'. "
                "Valid options: 'constant', 'linear', 'cosine', 'exponential', 'adaptive'."
            )


def sample_autoregressive(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> torch.Tensor:
    """
    Sample from autoregressive model with temperature and filtering.

    Order of operations:
    1. Apply temperature
    2. Apply filtering (top-k, top-p, or min-p)
    3. Sample from filtered distribution

    Args:
        logits: [batch, vocab_size] unnormalized logits
        temperature: sampling temperature (higher = more random)
        top_k: top-k filtering (None = disabled)
        top_p: nucleus sampling threshold (None = disabled)
        min_p: min-p sampling threshold (None = disabled, recommended: 0.05)

    Returns:
        sampled_tokens: [batch] sampled token indices
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [batch, vocab], got {tuple(logits.shape)}")
    _ensure_finite_positive(temperature, "temperature")
    if top_k is not None:
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an int, got {type(top_k)}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
    if top_p is not None:
        _ensure_probability(top_p, "top_p")
    if min_p is not None:
        _ensure_probability(min_p, "min_p", allow_zero=True)

    # 1. Temperature
    logits = logits / temperature

    # 2. Filtering (in order of preference)
    if min_p is not None:
        logits = sample_min_p(logits, min_p=min_p)
    elif top_p is not None:
        logits = sample_nucleus(logits, p=top_p)
    elif top_k is not None:
        logits = sample_top_k(logits, k=top_k)

    # 3. Sample
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled
