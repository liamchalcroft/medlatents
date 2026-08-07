"""Generation utilities for medlatents.

High-level generation API for discrete and continuous latent models.
For low-level sampling techniques, see medlatents.sampling module.
"""

from .generator import ContinuousLatentGenerator, DiscreteLatentGenerator

__all__ = [
    "DiscreteLatentGenerator",
    "ContinuousLatentGenerator",
]
