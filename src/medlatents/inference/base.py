"""Base utilities for conditional inference with discrete latent models."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def create_spatial_mask(
    shape: tuple[int, ...],
    mask_type: Literal["random", "block", "slice", "checkerboard"],
    mask_ratio: float = 0.5,
    **mask_kwargs,
) -> torch.Tensor:
    """Create spatial masks for inpainting/super-resolution. True = keep, False = mask."""
    mask = torch.ones(shape, dtype=torch.bool)

    if mask_type == "random":
        num_mask = int(mask.numel() * mask_ratio)
        flat_mask = mask.flatten()
        mask_indices = torch.randperm(flat_mask.numel())[:num_mask]
        flat_mask[mask_indices] = False
        mask = flat_mask.view(shape)

    elif mask_type == "block":
        # Block/region masking for inpainting
        start = mask_kwargs.get("start")
        end = mask_kwargs.get("end")

        if start is None or end is None:
            raise ValueError(
                "Block masking requires 'start' and 'end' tuples. "
                "Example: create_spatial_mask(shape, 'block', start=(10, 10), end=(50, 50))"
            )

        # Create slice for each dimension
        slices = tuple(slice(s, e) for s, e in zip(start, end))
        mask[slices] = False

    elif mask_type == "slice":
        # Slice-based masking for super-resolution
        axis = mask_kwargs.get("axis", 0)
        stride = mask_kwargs.get("stride", 2)
        start_offset = mask_kwargs.get("start_offset", 1)

        # Mask every N-th slice
        if axis == 0:
            mask[start_offset::stride, ...] = False
        elif axis == 1:
            mask[:, start_offset::stride, ...] = False
        elif axis == 2:
            mask[:, :, start_offset::stride] = False
        else:
            raise ValueError(
                f"Invalid axis: {axis}. Valid options: 0 (depth), 1 (height), 2 (width)."
            )

    elif mask_type == "checkerboard":
        # Checkerboard pattern for 2D/3D
        if len(shape) == 2:
            H, W = shape
            mask = (torch.arange(H)[:, None] + torch.arange(W)[None, :]) % 2 == 0
        elif len(shape) == 3:
            D, H, W = shape
            mask = (
                torch.arange(D)[:, None, None]
                + torch.arange(H)[None, :, None]
                + torch.arange(W)[None, None, :]
            ) % 2 == 0
        else:
            raise ValueError(
                f"Checkerboard mask only supports 2D (H, W) or 3D (D, H, W) shapes, "
                f"got {len(shape)}D shape: {shape}"
            )

    else:
        raise ValueError(
            f"Unknown mask_type: '{mask_type}'. "
            "Valid options: 'random', 'block', 'slice', 'checkerboard'."
        )

    return mask


def spatial_to_sequence_mask(
    spatial_mask: torch.Tensor,
    rasterization_method: Literal["raster", "hilbert", "zorder"] = "hilbert",
    **raster_kwargs,
) -> torch.Tensor:
    """Convert spatial mask to sequence mask using rasterization."""
    from ..rasterization import HilbertCurve, RasterScan, ZOrderCurve

    # Add batch and channel dims if needed
    if spatial_mask.ndim == 2:
        spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif spatial_mask.ndim == 3:
        spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    # Convert to float for rasterization
    mask_float = spatial_mask.float()

    # Rasterize
    if rasterization_method == "hilbert":
        if mask_float.ndim == 4:
            sequence, _ = HilbertCurve.spatial_to_sequence_2d(mask_float)
        else:
            sequence, _ = HilbertCurve.spatial_to_sequence_3d(mask_float)
    elif rasterization_method == "zorder":
        if mask_float.ndim == 4:
            sequence, _ = ZOrderCurve.spatial_to_sequence_2d(mask_float)
        else:
            sequence, _ = ZOrderCurve.spatial_to_sequence_3d(mask_float)
    elif rasterization_method == "raster":
        if mask_float.ndim == 4:
            sequence = RasterScan.spatial_to_sequence_2d(mask_float)
        else:
            sequence = RasterScan.spatial_to_sequence_3d(mask_float)
    else:
        raise ValueError(
            f"Unknown rasterization_method: '{rasterization_method}'. "
            "Valid options: 'hilbert', 'zorder', 'raster'."
        )

    # Convert back to bool and remove batch/channel dims
    sequence_mask = sequence.squeeze(0).squeeze(0) > 0.5

    return sequence_mask


def apply_token_mask(
    tokens: Int[torch.Tensor, "batch seq"],
    mask: torch.Tensor,
    mask_value: int,
) -> Int[torch.Tensor, "batch seq"]:
    """Apply mask to token sequence. True = keep, False = mask.

    Accepts ``mask`` as either a 1-D ``(seq,)`` tensor (broadcast across batch)
    or a 2-D ``(batch, seq)`` tensor (per-sample mask). Both are common in
    practice — a single shared mask is natural for ``centre-mask`` inpainting
    of a fixed region, and per-sample masks are needed for anomaly-detection
    pipelines where each image has the anomaly at a different location.
    """
    masked_tokens = tokens.clone()
    if mask.dim() == 1:
        masked_tokens[:, ~mask] = mask_value
    elif mask.dim() == 2:
        if mask.shape != tokens.shape:
            raise ValueError(f"2-D mask shape {mask.shape} must match tokens shape {tokens.shape}")
        masked_tokens[~mask] = mask_value
    else:
        raise ValueError(f"mask must be 1-D (seq,) or 2-D (batch, seq); got shape {mask.shape}")
    return masked_tokens


def confidence_mask(
    logits: Float[torch.Tensor, "batch seq vocab"],
    threshold: float,
) -> torch.Tensor:
    """Create mask based on prediction confidence. True = high confidence, False = low."""
    probs = F.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return max_probs >= threshold


def interpolate_spatial_upsampling(
    volume: torch.Tensor,
    target_shape: tuple[int, ...],
    mode: str = "trilinear",
) -> torch.Tensor:
    """Upsample volume using interpolation.

    Args:
        volume: 4D (batch, channels, H, W) or 5D (batch, channels, D, H, W) tensor
        target_shape: Target spatial dimensions
        mode: Interpolation mode ('bilinear' for 2D, 'trilinear' for 3D)
    """
    if volume.ndim not in (4, 5):
        raise ValueError(
            f"Invalid volume dimensions: {volume.ndim}. "
            "Expected 4D (batch, channels, H, W) or 5D (batch, channels, D, H, W)."
        )
    return F.interpolate(volume, size=target_shape, mode=mode, align_corners=False)


def reslice_volume(
    volume: torch.Tensor,
    axis: int,
    target_slices: int,
    mode: str = "trilinear",
) -> torch.Tensor:
    """Reslice volume along specific axis to target resolution."""
    if volume.ndim != 5:
        raise ValueError(
            f"reslice_volume only supports 5D volumes [batch, channels, D, H, W], "
            f"got {volume.ndim}D tensor with shape {tuple(volume.shape)}."
        )

    current_shape = list(volume.shape[2:])  # [D, H, W]
    target_shape = current_shape.copy()
    target_shape[axis] = target_slices

    return F.interpolate(volume, size=target_shape, mode=mode, align_corners=False)


def encode_large_volume(
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
) -> torch.Tensor:
    """Encode large volume. The tokenizer handles sliding window inference automatically."""
    return tokenizer.tokenize(volume)


def decode_large_volume(
    tokenizer,  # DiscreteTokenizer
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Decode large token volume. The tokenizer handles sliding window inference automatically."""
    return tokenizer.detokenize(tokens)


def reconstruct_large_volume(
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    roi_size: tuple[int, ...] = (64, 64, 64),
    overlap: float = 0.5,
) -> torch.Tensor:
    """Reconstruct large volume using sliding window inference.

    The tokenizer's reconstruct() method handles sliding window inference
    internally for large volumes.
    """
    return tokenizer.reconstruct(volume, roi_size=roi_size, overlap=overlap)
