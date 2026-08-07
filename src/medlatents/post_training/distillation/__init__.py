"""Distillation methods for faster inference.

This module implements distillation techniques to enable faster sampling:

- **Consistency Training/Distillation**: Learn to map any noise level directly to data
- **Reflow**: Iteratively straighten flow trajectories for fewer steps

These methods allow sampling with 1-4 steps instead of 50-1000 steps.

References:
- Consistency Models: "Consistency Models" (Song et al., 2023)
- Reflow: "Flow Straight and Fast" (Liu et al., 2022)
"""

from .consistency import (
    ConsistencyTrainer,
    get_discretization_schedule,
    get_ema_decay_schedule,
    pseudo_huber_loss,
)
from .reflow import (
    ReflowPairGenerator,
    ReflowTrainer,
)

__all__ = [
    # Consistency
    "ConsistencyTrainer",
    "pseudo_huber_loss",
    "get_discretization_schedule",
    "get_ema_decay_schedule",
    # Reflow
    "ReflowTrainer",
    "ReflowPairGenerator",
]
