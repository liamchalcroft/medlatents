"""MaskGIT scheduling strategies: Halton sequences and confidence-based masking.

This module provides:
- Halton sequence generation for spatially-dispersed unmasking
- Confidence-based adaptive masking schedules
- Temperature scheduling for iterative decoding
"""

import math

import torch
import torch.nn.functional as F


def halton_sequence(index: int, base: int) -> float:
    """
    Generate the nth value of the Halton sequence for a given base.

    The Halton sequence is a low-discrepancy sequence that provides
    better spatial coverage than random sampling.

    Args:
        index: The index in the sequence (0-indexed)
        base: The base (prime number, typically 2, 3, 5, 7, etc.)

    Returns:
        value in [0, 1)
    """
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f = f / base
    return result


def halton_schedule_1d(step: int, total_steps: int, base: int = 2) -> float:
    """
    Halton-based masking schedule for 1D sequences.

    Uses the Halton sequence to ensure spatially dispersed unmasking.
    This is particularly useful for spatial data (e.g., image patches arranged
    via space-filling curves) where structure should emerge coherently.

    Reference: "Halton Scheduler for Masked Generative Image Transformer" (arXiv:2503.17076)

    Args:
        step: current step
        total_steps: total number of steps
        base: Halton sequence base (default: 2)

    Returns:
        mask_ratio: fraction of tokens to keep masked
    """
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step}")
    if not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError(f"total_steps must be a positive int, got {total_steps}")
    if not isinstance(base, int) or base < 2:
        raise ValueError(f"base must be an int >= 2, got {base}")
    if step >= total_steps:
        return 0.0

    halton_value = halton_sequence(step, base)
    progress = (step + 1) / total_steps
    cosine_component = math.cos(progress * math.pi / 2)

    return cosine_component * (1.0 - halton_value * 0.3)


def halton_schedule_2d(
    step: int,
    total_steps: int,
    width: int,
    height: int,
    base_x: int = 2,
    base_y: int = 3,
) -> list[tuple[int, int]]:
    """
    2D Halton schedule for spatial data.

    Returns the unmasking order for a 2D grid using 2D Halton sequence.
    This ensures spatially dispersed unmasking across both dimensions.

    Args:
        step: current step
        total_steps: total number of steps
        width: grid width
        height: grid height
        base_x: Halton base for x dimension (default: 2)
        base_y: Halton base for y dimension (default: 3)

    Returns:
        List of (x, y) coordinates to unmask at this step
    """
    num_tokens = width * height
    tokens_to_unmask = int(num_tokens * (1.0 - halton_schedule_1d(step, total_steps)))

    halton_points = []
    for i in range(num_tokens):
        h_x = halton_sequence(i, base_x)
        h_y = halton_sequence(i, base_y)
        x = int(h_x * width)
        y = int(h_y * height)
        halton_points.append((h_x * h_y, x, y))

    halton_points.sort(key=lambda p: p[0])
    return [(p[1], p[2]) for p in halton_points[:tokens_to_unmask]]


def confidence_based_schedule(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    mask_token: int,
    target_mask_ratio: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Adaptive masking based on model confidence.

    Instead of using a fixed schedule, mask tokens where the model
    is least confident. This allows the model to focus on difficult regions.

    Args:
        logits: [batch, seq_len, vocab_size] model predictions
        current_tokens: [batch, seq_len] current token assignments
        mask_token: index of mask token
        target_mask_ratio: desired fraction of tokens to mask
        temperature: temperature for confidence scoring

    Returns:
        mask: [batch, seq_len] boolean mask (True = keep masked)
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must have shape [batch, seq_len, vocab], got {tuple(logits.shape)}"
        )
    if current_tokens.dim() != 2:
        raise ValueError(
            f"current_tokens must have shape [batch, seq_len], got {tuple(current_tokens.shape)}"
        )
    if not (0.0 <= target_mask_ratio <= 1.0):
        raise ValueError(f"target_mask_ratio must be in [0, 1], got {target_mask_ratio}")
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(f"temperature must be finite and > 0, got {temperature}")

    batch_size, seq_len, vocab_size = logits.shape

    probs = F.softmax(logits / temperature, dim=-1)

    confidence = torch.gather(probs, dim=-1, index=current_tokens.unsqueeze(-1)).squeeze(-1)

    is_masked = current_tokens == mask_token
    confidence[is_masked] = float("inf")

    num_to_mask = int(seq_len * target_mask_ratio)

    if num_to_mask > 0:
        threshold = torch.kthvalue(confidence, num_to_mask, dim=-1, keepdim=True)[0]
        new_mask = confidence <= threshold
    else:
        new_mask = torch.zeros_like(current_tokens, dtype=torch.bool)

    mask = is_masked | new_mask

    return mask


def adaptive_masking(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    mask_token: int,
    step: int,
    total_steps: int,
    base_schedule: str = "cosine",
    confidence_weight: float = 0.5,
    temperature: float = 1.0,
    schedule_power: float = 2.0,
    halton_base: int = 2,
) -> torch.Tensor:
    """
    Hybrid masking: combine fixed schedule with confidence-based adjustment.

    Args:
        logits: [batch, seq_len, vocab_size] model predictions
        current_tokens: [batch, seq_len] current token assignments
        mask_token: index of mask token
        step: current iteration
        total_steps: total iterations
        base_schedule: base schedule type ('cosine', 'linear', 'sqrt', 'power', 'halton')
        confidence_weight: weight for confidence-based adjustment (0-1)
        temperature: temperature for confidence scoring
        schedule_power: power for power schedule
        halton_base: base for halton schedule

    Returns:
        mask: [batch, seq_len] boolean mask
    """
    if not (0.0 <= confidence_weight <= 1.0):
        raise ValueError(f"confidence_weight must be in [0, 1], got {confidence_weight}")
    if schedule_power <= 0:
        raise ValueError(f"schedule_power must be > 0, got {schedule_power}")
    if halton_base < 2:
        raise ValueError(f"halton_base must be >= 2, got {halton_base}")

    # Use (total_steps - 1) as denominator so final step has progress=1.0
    progress = step / max(total_steps - 1, 1)

    if base_schedule == "cosine":
        # Cosine decay from 1.0 to 0.0
        base_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif base_schedule == "linear":
        # Linear decay from 1.0 to 0.0
        base_ratio = 1.0 - progress
    elif base_schedule == "sqrt":
        # Sqrt decay from 1.0 to 0.0
        base_ratio = 1.0 - math.sqrt(progress)
    elif base_schedule == "power":
        # Power decay from 1.0 to 0.0
        base_ratio = 1.0 - (progress**schedule_power)
    elif base_schedule == "halton":
        # Halton uses its own (step+1) convention
        base_ratio = halton_schedule_1d(step, total_steps, base=halton_base)
    else:
        raise ValueError(f"Unknown schedule: {base_schedule}")

    return confidence_based_schedule(logits, current_tokens, mask_token, base_ratio, temperature)


__all__ = [
    "halton_sequence",
    "halton_schedule_1d",
    "halton_schedule_2d",
    "confidence_based_schedule",
    "adaptive_masking",
]
