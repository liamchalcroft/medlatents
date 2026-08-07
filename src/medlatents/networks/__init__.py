"""Shared network architectures for discrete and continuous latent models."""

from .diffusion_transformer import (
    ContinuousDiT,
    ContinuousDiT_models,
    ContinuousSequenceEmbedder,
    DiscreteDiT,
    DiscreteDiT_models,
    DiscreteSequenceEmbedder,
    DiTAttention,
    DiTBlock,
    DiTBlockWithCrossAttention,
    DiTCrossAttention,
    FinalLayer,
    LabelEmbedder,
    MMDiTBlock,
    TimestepEmbedder,
)
from .sprint import (
    SPRINTConfig,
    SPRINTDiT,
    SPRINTScheduler,
    TokenRestorer,
    TokenSelector,
)
from .transformer import (
    Attention,
    AttentionWithValueResidual,
    ContinuousTransformer,
    DiscreteTransformer,
    RMSNorm,
    RMSNormZero,
    TransformerBlock,
    apply_rotary_emb,
    init_weights,
    precompute_freqs_cis,
)

__all__ = [
    # Base transformer infrastructure
    "DiscreteTransformer",
    "ContinuousTransformer",
    "Attention",
    "AttentionWithValueResidual",
    "TransformerBlock",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "init_weights",
    "RMSNorm",
    "RMSNormZero",
    # DiT (Diffusion Transformer)
    "DiscreteDiT",
    "ContinuousDiT",
    "DiscreteDiT_models",
    "ContinuousDiT_models",
    "DiTBlock",
    "DiTBlockWithCrossAttention",
    "DiTCrossAttention",
    "MMDiTBlock",
    "DiTAttention",
    "FinalLayer",
    "LabelEmbedder",
    "DiscreteSequenceEmbedder",
    "ContinuousSequenceEmbedder",
    "TimestepEmbedder",
    # SPRINT (token dropping)
    "SPRINTConfig",
    "SPRINTDiT",
    "SPRINTScheduler",
    "TokenSelector",
    "TokenRestorer",
]
