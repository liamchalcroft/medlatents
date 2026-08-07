"""Conditional inference utilities for discrete latent models."""

# Base utilities
from .base import (
    apply_token_mask,
    confidence_mask,
    create_spatial_mask,
    decode_large_volume,
    encode_large_volume,
    interpolate_spatial_upsampling,
    reconstruct_large_volume,
    reslice_volume,
    spatial_to_sequence_mask,
)

# Inpainting
from .inpainting import (
    inpaint_autoregressive,
    inpaint_bayesian_flow,
    inpaint_diffusion_repaint,
    inpaint_flow_matching,
    inpaint_maskgit,
    inpaint_volume,
)

# KV-cache utilities
from .kv_cache import (
    PagedKVCache,
    QuantizedKVCache,
)
from .kv_cache import (
    QuantizationConfig as KVQuantizationConfig,  # Renamed to avoid conflict
)

# Model quantization and pruning
from .quantization import (
    ModelPruner,
    ModelQuantizer,
    PrunedLinear,
    PruningConfig,
    PruningMethod,
    QuantizationConfig,
    QuantizationMethod,
    QuantizedLinear,
    compute_head_importance,
    compute_importance_scores,
    compute_scale_zero_point,
    create_pruning_mask,
    dequantize_tensor,
    estimate_model_size,
    export_to_onnx,
    prune_attention_heads,
    quantize_tensor,
)

# Super-resolution
from .super_resolution import (
    anisotropic_super_resolution,
    compare_with_interpolation,
    progressive_super_resolution,
    super_resolve_slices,
)

__all__ = [
    # Base utilities
    "create_spatial_mask",
    "spatial_to_sequence_mask",
    "apply_token_mask",
    "confidence_mask",
    "interpolate_spatial_upsampling",
    "reslice_volume",
    # Large volume processing
    "encode_large_volume",
    "decode_large_volume",
    "reconstruct_large_volume",
    # Inpainting (model-specific)
    "inpaint_autoregressive",
    "inpaint_maskgit",
    "inpaint_flow_matching",
    "inpaint_diffusion_repaint",
    "inpaint_bayesian_flow",
    # Inpainting (end-to-end)
    "inpaint_volume",
    # Super-resolution
    "super_resolve_slices",
    "anisotropic_super_resolution",
    "progressive_super_resolution",
    "compare_with_interpolation",
    # KV-cache
    "KVQuantizationConfig",
    "QuantizedKVCache",
    "PagedKVCache",
    # Model quantization
    "QuantizationMethod",
    "QuantizationConfig",
    "ModelQuantizer",
    "QuantizedLinear",
    "compute_scale_zero_point",
    "quantize_tensor",
    "dequantize_tensor",
    # Model pruning
    "PruningMethod",
    "PruningConfig",
    "ModelPruner",
    "PrunedLinear",
    "compute_importance_scores",
    "create_pruning_mask",
    # Attention head pruning
    "prune_attention_heads",
    "compute_head_importance",
    # Export utilities
    "export_to_onnx",
    "estimate_model_size",
]
