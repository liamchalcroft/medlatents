"""Reinforcement Learning methods for post-training.

This module implements RL-based fine-tuning methods for generative models:

- **DDPO**: Denoising Diffusion Policy Optimization - PPO-style training for diffusion
- **GRPO**: Group Relative Policy Optimization - memory-efficient without value network
- **GARDO**: Gradient-Aware Reward Distance Optimization - adaptive KL penalty

All methods support trajectory-based optimization where the generative process
is treated as a multi-step MDP.

References:
- DDPO: "Training Diffusion Models with RL" (Black et al., 2023)
- GRPO: "DeepSeekMath" (Shao et al., 2024)
"""

from .base import (
    BaseRLTrainer,
    RLBatch,
    Trajectory,
    TrajectoryBuffer,
    compute_advantages_gae,
    compute_advantages_monte_carlo,
    normalize_advantages,
)
from .ddpo import DDPODiscreteFlowTrainer, DDPOTrainer
from .gardo import GARDOTrainer
from .grpo import GRPOTrainer

__all__ = [
    # Base infrastructure
    "BaseRLTrainer",
    "Trajectory",
    "RLBatch",
    "TrajectoryBuffer",
    "compute_advantages_gae",
    "compute_advantages_monte_carlo",
    "normalize_advantages",
    # DDPO
    "DDPOTrainer",
    "DDPODiscreteFlowTrainer",
    # GRPO
    "GRPOTrainer",
    # GARDO
    "GARDOTrainer",
]
