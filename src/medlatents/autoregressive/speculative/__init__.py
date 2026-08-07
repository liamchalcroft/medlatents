"""Speculative decoding subpackage.

Modules:
- decoder: SpeculativeDecoder, MultiDraftSpeculativeDecoder
- medusa: MedusaHead, MedusaModel, MultiTokenPredictionLoss, MedusaTrainer
- eagle: EAGLE speculative decoder with feature-level autoregression
"""

from .decoder import (
    MultiDraftSpeculativeDecoder,
    SpeculativeDecoder,
)
from .eagle import (
    EAGLEConfig,
    EAGLEDecoder,
    EAGLEHead,
    EAGLETrainer,
    create_eagle_from_base,
)
from .medusa import (
    MedusaHead,
    MedusaModel,
    MedusaTrainer,
    MultiTokenPredictionLoss,
)

__all__ = [
    # Core speculative decoding
    "SpeculativeDecoder",
    "MultiDraftSpeculativeDecoder",
    # Medusa multi-head decoding
    "MedusaHead",
    "MedusaModel",
    "MultiTokenPredictionLoss",
    "MedusaTrainer",
    # EAGLE speculative decoding
    "EAGLEConfig",
    "EAGLEHead",
    "EAGLEDecoder",
    "EAGLETrainer",
    "create_eagle_from_base",
]
