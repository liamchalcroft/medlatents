"""Self-Play methods for iterative self-improvement.

This module implements self-play techniques that allow models to improve
through training on their own outputs:

- **SPIN**: Self-Play Fine-Tuning that uses model's own generations as negative examples
- **RFT**: Rejection Fine-Tuning that selects and fine-tunes on best samples

These methods create a curriculum where models learn to distinguish
their current outputs from higher-quality targets.

References:
- SPIN: "Self-Play Fine-Tuning Converts Weak LMs to Strong LMs" (Chen et al., 2024)
- RFT: "Rejection Sampling Fine-Tuning for LLMs" (Yuan et al., 2023)
"""

from .iterative import SPINTrainer
from .rejection import RFTTrainer

__all__ = [
    # SPIN
    "SPINTrainer",
    # RFT
    "RFTTrainer",
]
