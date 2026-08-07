"""Learning-rate schedule utilities.

Provides the cosine-with-warmup learning-rate schedule used across training.
"""

import math


def get_cosine_schedule_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float = 0.0,
) -> float:
    """
    Cosine schedule with linear warmup for learning rates.

    Standard learning rate schedule used in transformers and diffusion models.

    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate

    Returns:
        Learning rate at current step
    """
    # Edge case: if warmup_steps equals total_steps, always return max_lr
    if warmup_steps >= total_steps:
        return max_lr

    if step < warmup_steps:
        return max_lr * (step / max(warmup_steps, 1))

    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


__all__ = [
    "get_cosine_schedule_with_warmup",
]
