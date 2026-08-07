"""Post-training methods for generative models.

This module provides implementations of state-of-the-art post-training techniques
for improving and aligning generative models after initial training. Includes:

- **Preference Learning**: DPO, SPO, and related methods for aligning models to preferences
- **Reinforcement Learning**: DDPO, GRPO, GARDO for reward-based fine-tuning (coming soon)
- **Distillation**: Consistency models, reflow for faster inference (coming soon)
- **Self-Play**: Iterative self-improvement methods (coming soon)

All methods support FSDP for distributed training and are designed to work with
the full range of model types in medlatents (autoregressive, MaskGIT, D3PM,
flow matching, Bayesian flow).

Example usage:
    >>> from medlatents.post_training import PostTrainingConfig, DPOLoss
    >>> from medlatents.post_training.dpo import DiscreteDPOTrainer
    >>> from medlatents.post_training.preference import PreferenceDataset
    >>>
    >>> # Create preference data from generations
    >>> dataset = PreferenceDataset.from_generations(
    ...     generator=model,
    ...     prompts=prompts,
    ...     reward_fn=reward_model,
    ...     num_samples_per_prompt=4,
    ... )
    >>>
    >>> # Configure and run DPO
    >>> config = PostTrainingConfig(method="dpo", beta=0.1, lr=1e-5)
    >>> trainer = DiscreteDPOTrainer(model, ref_model, config)
    >>> trainer.train(dataset)
"""

from .configs import (
    DDPO_STANDARD,
    DPO_LARGE,
    DPO_SMALL,
    DPO_STANDARD,
    GRPO_EFFICIENT,
    SPO_STANDARD,
    PostTrainingConfig,
)

# Import DPO trainers and utilities
from .dpo import (
    # Model-specific trainers
    AutoregressiveDPOTrainer,
    # Base classes
    BaseDPOTrainer,
    D3PMDPOTrainer,
    DiffusionDPOTrainer,
    DiscreteDPOTrainer,
    DiscreteFlowDPOTrainer,
    DPOLoss,
    DPOOutput,
    FlowDPOTrainer,
    MaskGITDPOTrainer,
    SPOStepOutput,
    # SPO
    SPOTrainer,
    StepPreferenceModel,
)

# Import preference data utilities
from .preference import (
    PreferenceDataset,
    PreferencePair,
    RankedSamples,
    RewardModel,
    RewardModelTrainer,
    StepwisePreference,
    collate_preference_pairs,
    create_reward_model,
)

__all__ = [
    # Configuration
    "PostTrainingConfig",
    "DPO_SMALL",
    "DPO_STANDARD",
    "DPO_LARGE",
    "SPO_STANDARD",
    "DDPO_STANDARD",
    "GRPO_EFFICIENT",
    # Preference Data
    "PreferencePair",
    "PreferenceDataset",
    "RankedSamples",
    "StepwisePreference",
    "collate_preference_pairs",
    "RewardModel",
    "RewardModelTrainer",
    "create_reward_model",
    # DPO/SPO Trainers and Outputs
    "AutoregressiveDPOTrainer",
    "BaseDPOTrainer",
    "D3PMDPOTrainer",
    "DiffusionDPOTrainer",
    "DiscreteDPOTrainer",
    "DiscreteFlowDPOTrainer",
    "DPOLoss",
    "DPOOutput",
    "FlowDPOTrainer",
    "MaskGITDPOTrainer",
    "SPOStepOutput",
    "SPOTrainer",
    "StepPreferenceModel",
]
