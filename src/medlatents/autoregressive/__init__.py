"""Autoregressive transformers for discrete latent sequence modeling."""

from .speculative import (
    MedusaHead,
    MedusaModel,
    MedusaTrainer,
    MultiDraftSpeculativeDecoder,
    MultiTokenPredictionLoss,
    SpeculativeDecoder,
)
from .transformer import Autoreg_models, AutoregressiveTransformer

__all__ = [
    "AutoregressiveTransformer",
    "Autoreg_models",
    "SpeculativeDecoder",
    "MultiDraftSpeculativeDecoder",
    # Medusa: Multi-head speculative decoding (2024)
    "MedusaModel",
    "MedusaHead",
    "MedusaTrainer",
    "MultiTokenPredictionLoss",
]
