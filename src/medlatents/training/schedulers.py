"""MaskGIT-specific scheduler for training.

This module provides the get_mask_ratio function which is specific to
MaskGIT training schedules (cosine, linear, square decay of mask ratio).

For general-purpose schedule functions (cosine_schedule, linear_schedule,
get_cosine_schedule_with_warmup), use medlatents.common.schedules.
"""

from typing import Literal


def get_mask_ratio(
    progress: float, schedule: Literal["cosine", "linear", "square"] = "cosine"
) -> float:
    """
    Get mask ratio based on training progress (0 to 1) and schedule type.

    MaskGIT training uses a decreasing mask ratio schedule, starting with
    high masking (e.g., 90% masked) and ending with low masking (e.g., 10%).

    Args:
        progress: Training progress from 0.0 (start) to 1.0 (end)
        schedule: Schedule type ('cosine', 'linear', 'square')

    Returns:
        Mask ratio at this progress (higher = more tokens masked)

    Examples:
        >>> get_mask_ratio(0.0, "cosine")  # Start: max masking
        1.0
        >>> get_mask_ratio(1.0, "cosine")  # End: min masking
        0.0
        >>> get_mask_ratio(0.5, "cosine")  # Mid: half masking
        0.5
    """
    if schedule == "cosine":
        return 0.5 * (1 + __import__("math").cos(progress * __import__("math").pi))
    elif schedule == "linear":
        return 1 - progress
    elif schedule == "square":
        return (1 - progress) ** 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


__all__ = ["get_mask_ratio"]
