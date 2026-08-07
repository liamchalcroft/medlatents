"""Unified conditioning infrastructure for generative models.

This module provides:
- ConditioningBundle: Standardized container for all conditioning inputs
- ConditioningConfig: Configuration for model conditioning capabilities
- Encoders: Pretrained encoder wrappers (DINOv2, SigLIP, medical models)
- Collate functions: For conditional training data pipelines
"""

from .bundle import ConditioningBundle, ConditioningConfig
from .collate import (
    ConditionalBatchConfig,
    collate_conditional_batch,
)
from .encoders import (
    FrozenDINOv2,
    FrozenEncoder,
    FrozenMedSigLIP,
    FrozenNeuroVFM,
    FrozenSigLIP,
    SliceWiseEncoder,
    create_encoder,
)

__all__ = [
    # Core
    "ConditioningBundle",
    "ConditioningConfig",
    # Encoders
    "FrozenEncoder",
    "FrozenDINOv2",
    "FrozenSigLIP",
    "FrozenNeuroVFM",
    "FrozenMedSigLIP",
    "SliceWiseEncoder",
    "create_encoder",
    # Data pipeline
    "ConditionalBatchConfig",
    "collate_conditional_batch",
]
