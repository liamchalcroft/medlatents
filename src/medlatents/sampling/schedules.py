"""Guidance schedules for classifier-free guidance.

Provides time-dependent guidance scale schedules for classifier-free guidance (CFG).
Every schedule is a callable ``(progress, base_scale) -> float`` that returns the
guidance weight ``w`` used as ``uncond + w * (cond - uncond)``; ``w = 1`` means no
guidance and ``w = base_scale`` means full guidance.

References:
- "Theory-Informed Improvements to Classifier-Free Guidance" (arXiv:2507.08965)
- "What Does Guidance Do in Masked Discrete Diffusion?" (arXiv:2506.10971)
- "CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models" (arXiv:2503.18886)
"""

import math
from collections.abc import Callable

GuidanceSchedule = Callable[[float, float], float]


# ============================================================================
# SAMPLING-SPECIFIC GUIDANCE SCHEDULES
# ============================================================================


def _validate_progress(progress: float) -> None:
    if not isinstance(progress, (int, float)) or not (0.0 <= progress <= 1.0):
        raise ValueError(f"progress must be in [0, 1], got {progress}")


def constant_guidance(progress: float, base_scale: float) -> float:
    """Constant guidance weight ``base_scale`` for the whole trajectory."""
    _validate_progress(progress)
    return base_scale


def linear_guidance(progress: float, base_scale: float) -> float:
    """Linear ramp of the guidance weight from 1.0 to ``base_scale``."""
    _validate_progress(progress)
    return 1.0 + (base_scale - 1.0) * progress


def cosine_guidance(progress: float, base_scale: float) -> float:
    """Cosine ramp of the guidance weight from 1.0 (at t=0) to ``base_scale`` (at t=1)."""
    _validate_progress(progress)
    return 1.0 + (base_scale - 1.0) * (1.0 - math.cos(math.pi * progress)) / 2.0


def cosine_decay_guidance(progress: float, base_scale: float) -> float:
    """Cosine decay of the guidance weight from ``base_scale`` (at t=0) to 1.0 (at t=1)."""
    _validate_progress(progress)
    return 1.0 + (base_scale - 1.0) * (1.0 + math.cos(math.pi * progress)) / 2.0


def triangular_guidance(progress: float, base_scale: float) -> float:
    """Triangular schedule peaking at ``base_scale`` at the midpoint (t=0.5)."""
    _validate_progress(progress)
    return 1.0 + (base_scale - 1.0) * (1.0 - abs(2.0 * progress - 1.0))


def cfg_zero_guidance(
    progress: float,
    base_scale: float,
) -> float:
    """
    CFG-Zero time-dependent guidance weighting.

    Uses parabolic weighting: w(t) = 4*t*(1-t) which peaks at t=0.5
    and is zero at the boundaries. This prevents over-guidance at
    the start and end of sampling.

    Reference:
        "Classifier-Free Guidance is a Predictor-Corrector" and
        related work on time-dependent CFG weighting.

    Args:
        progress: Progress from 0 to 1 (corresponds to time t)
        base_scale: Base guidance scale

    Returns:
        Time-weighted guidance scale: 1 + (base_scale - 1) * 4*t*(1-t)
    """
    _validate_progress(progress)
    t_weight = 4 * progress * (1 - progress)
    return 1.0 + (base_scale - 1.0) * t_weight


def cfg_zero_star_guidance(
    progress: float,
    base_scale: float,
    zero_init_steps: float = 0.1,
    scale_factor: float = 1.0,
) -> float:
    """
    CFG-Zero* improved guidance (2025).

    Adds two improvements over CFG-Zero:
    1. Zero-init: Zero out guidance for the first few steps (pure noise region)
    2. Optimized scale: Apply a learned/tuned scale factor

    Reference: "CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models"
               (arXiv:2503.18886)

    Args:
        progress: Progress from 0 to 1 (corresponds to time t)
        base_scale: Base guidance scale
        zero_init_steps: Fraction of steps at start to zero out guidance (default 0.1 = 10%)
        scale_factor: Scale factor for guidance (tune per model, default 1.0)

    Returns:
        Optimized guidance scale
    """
    _validate_progress(progress)
    if not (0.0 <= zero_init_steps <= 1.0):
        raise ValueError(f"zero_init_steps must be in [0, 1], got {zero_init_steps}")

    # Zero-init: no guidance in the pure noise region
    if progress < zero_init_steps:
        return 1.0

    # Parabolic weighting (same as CFG-Zero)
    t_weight = 4 * progress * (1 - progress)

    # Apply optimized scale factor
    effective_scale = 1.0 + (base_scale - 1.0) * t_weight * scale_factor

    return effective_scale


def dynamic_guidance(
    progress: float,
    base_scale: float,
    warmup: float = 0.2,
    cooldown: float = 0.1,
) -> float:
    """
    Dynamic guidance schedule with warmup and cooldown.

    Provides weak guidance at start (noise region) and end (fine details),
    with strong guidance in the middle where structure forms.

    Based on findings from discrete diffusion guidance research (2024-2025).

    Args:
        progress: Progress from 0 to 1
        base_scale: Maximum guidance scale
        warmup: Fraction of steps for warmup (weak guidance)
        cooldown: Fraction of steps for cooldown (weak guidance)

    Returns:
        Guidance scale at this progress
    """
    _validate_progress(progress)
    if not (0.0 <= warmup <= 1.0):
        raise ValueError(f"warmup must be in [0, 1], got {warmup}")
    if not (0.0 <= cooldown <= 1.0):
        raise ValueError(f"cooldown must be in [0, 1], got {cooldown}")
    if warmup + cooldown > 1.0:
        raise ValueError(f"warmup + cooldown must be <= 1.0, got {warmup + cooldown}")

    if progress < warmup:
        # Linear warmup from 1.0 to base_scale
        return 1.0 + (base_scale - 1.0) * (progress / warmup)
    elif progress > (1.0 - cooldown):
        # Linear cooldown from base_scale to 1.0
        remaining = 1.0 - progress
        return 1.0 + (base_scale - 1.0) * (remaining / cooldown)
    else:
        # Full guidance in the middle
        return base_scale


_GUIDANCE_SCHEDULES: dict[str, GuidanceSchedule] = {
    # Standard ramps
    "constant": constant_guidance,
    "linear": linear_guidance,
    "cosine": cosine_guidance,
    "cosine_decay": cosine_decay_guidance,
    "triangular": triangular_guidance,
    # Sampling-specific (time-dependent) schedules
    "cfg_zero": cfg_zero_guidance,
    "cfg_zero_star": cfg_zero_star_guidance,
    "dynamic": dynamic_guidance,
}


def get_guidance_schedule(schedule: str) -> GuidanceSchedule:
    """
    Get a guidance schedule function by name.

    Args:
        schedule: Schedule type name. One of: ``constant``, ``linear``, ``cosine``,
            ``cosine_decay``, ``triangular``, ``cfg_zero``, ``cfg_zero_star``,
            ``dynamic``.

    Returns:
        Function that takes ``(progress, base_scale)`` and returns a guidance scale.

    Examples:
        >>> schedule = get_guidance_schedule("cosine")
        >>> scale = schedule(0.5, 5.0)  # guidance scale at 50% progress
    """
    if schedule in _GUIDANCE_SCHEDULES:
        return _GUIDANCE_SCHEDULES[schedule]

    available = list(_GUIDANCE_SCHEDULES.keys())
    raise ValueError(f"Unknown guidance schedule: {schedule}. Available: {available}")


__all__ = [
    "GuidanceSchedule",
    "constant_guidance",
    "linear_guidance",
    "cosine_guidance",
    "cosine_decay_guidance",
    "triangular_guidance",
    "cfg_zero_guidance",
    "cfg_zero_star_guidance",
    "dynamic_guidance",
    "get_guidance_schedule",
]
