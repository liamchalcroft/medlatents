"""Preference data and reward modeling for post-training.

This module provides infrastructure for:
1. Preference data collection and storage
2. Reward model training from preferences
3. AI-based preference scoring (synthetic feedback)
"""

from .ai_feedback import (
    AIPreferenceScorer,
    DiscriminatorScorer,
    StepAwarePreferenceModel,
)
from .data import (
    PreferenceDataset,
    PreferencePair,
    RankedSamples,
    StepwisePreference,
    collate_preference_pairs,
)
from .reward_model import (
    RewardModel,
    RewardModelTrainer,
    create_reward_model,
)

__all__ = [
    # Data structures
    "PreferencePair",
    "RankedSamples",
    "StepwisePreference",
    "PreferenceDataset",
    "collate_preference_pairs",
    # Reward models
    "RewardModel",
    "RewardModelTrainer",
    "create_reward_model",
    # AI feedback
    "AIPreferenceScorer",
    "StepAwarePreferenceModel",
    "DiscriminatorScorer",
]
