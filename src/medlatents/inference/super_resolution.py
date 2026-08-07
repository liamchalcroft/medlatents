"""Super-resolution utilities for discrete latent models."""

import logging
from typing import Literal

import torch
import torch.nn as nn

from .base import (
    create_spatial_mask,
    interpolate_spatial_upsampling,
    reslice_volume,
    spatial_to_sequence_mask,
)
from .inpainting import (
    inpaint_autoregressive,
    inpaint_bayesian_flow,
    inpaint_diffusion_repaint,
    inpaint_flow_matching,
    inpaint_maskgit,
)

logger = logging.getLogger(__name__)


def super_resolve_slices(
    model: nn.Module,
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    target_slices: int,
    axis: int = 0,
    model_type: Literal["autoreg", "maskgit", "flow", "diffusion", "bayesian_flow"] = "maskgit",
    interpolation_mode: str = "trilinear",
    rasterization_method: str = "hilbert",
    **inpaint_kwargs,
) -> torch.Tensor:
    """Super-resolve volume slices: upsample + inpaint interpolated slices."""
    batch, channels, *spatial_dims = volume.shape

    if len(spatial_dims) != 3:
        raise ValueError("super_resolve_slices requires 3D volume [batch, channels, D, H, W]")

    original_slices = spatial_dims[axis]
    upsampling_factor = target_slices / original_slices

    if target_slices <= original_slices:
        raise ValueError(f"target_slices ({target_slices}) must be > original ({original_slices})")

    # Step 1: Interpolate to target resolution
    interpolated = reslice_volume(volume, axis, target_slices, mode=interpolation_mode)

    # Step 2: Create mask for original vs interpolated slices
    # Original slices are at indices: 0, step, 2*step, ...
    # where step = upsampling_factor
    step = int(upsampling_factor)
    start_offset = step // 2  # Center the original slices

    # Create spatial mask
    mask_shape = list(interpolated.shape[2:])
    mask = create_spatial_mask(
        tuple(mask_shape),
        mask_type="slice",
        axis=axis,
        stride=step,
        start_offset=start_offset,
    )

    # Invert mask: True = original (keep), False = interpolated (inpaint)
    # But we want to inpaint the interpolated slices, so invert
    original_slice_mask = torch.ones_like(mask, dtype=torch.bool)
    if axis == 0:
        original_slice_mask[start_offset::step, ...] = False
    elif axis == 1:
        original_slice_mask[:, start_offset::step, ...] = False
    elif axis == 2:
        original_slice_mask[:, :, start_offset::step] = False

    # Note: original_slice_mask is True for original slices, False for interpolated
    # For inpainting, we need the opposite
    inpaint_mask = ~original_slice_mask

    # Step 3: Inpaint interpolated slices
    super_resolved = _inpaint_volume_with_mask(
        model=model,
        tokenizer=tokenizer,
        volume=interpolated,
        spatial_mask=inpaint_mask,
        model_type=model_type,
        rasterization_method=rasterization_method,
        **inpaint_kwargs,
    )

    return super_resolved


def anisotropic_super_resolution(
    model: nn.Module,
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    target_shape: tuple[int, int, int],
    model_type: Literal["autoreg", "maskgit", "flow", "diffusion", "bayesian_flow"] = "maskgit",
    interpolation_mode: str = "trilinear",
    rasterization_method: str = "hilbert",
    **inpaint_kwargs,
) -> torch.Tensor:
    """Super-resolve volume with anisotropic resolution."""
    batch, channels, *spatial_dims = volume.shape

    if len(spatial_dims) != 3:
        raise ValueError("anisotropic_super_resolution requires 3D volume")

    # Compute upsampling factors per axis
    upsampling_factors = [target_shape[i] / spatial_dims[i] for i in range(3)]

    # Interpolate to target shape
    interpolated = interpolate_spatial_upsampling(volume, target_shape, mode=interpolation_mode)

    # Build a mask over interpolated voxels along the most-upsampled axis.
    max_factor = max(upsampling_factors)
    axis_with_max = upsampling_factors.index(max_factor)

    # Use slice masking for axis with highest upsampling
    step = int(max_factor)
    mask = create_spatial_mask(
        target_shape, mask_type="slice", axis=axis_with_max, stride=step, start_offset=1
    )

    # Inpaint interpolated regions
    super_resolved = _inpaint_volume_with_mask(
        model=model,
        tokenizer=tokenizer,
        volume=interpolated,
        spatial_mask=mask,
        model_type=model_type,
        rasterization_method=rasterization_method,
        **inpaint_kwargs,
    )

    return super_resolved


def progressive_super_resolution(
    model: nn.Module,
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    target_slices: int,
    axis: int = 0,
    num_stages: int = 2,
    model_type: Literal["autoreg", "maskgit", "flow", "diffusion", "bayesian_flow"] = "maskgit",
    **inpaint_kwargs,
) -> torch.Tensor:
    """Progressive super-resolution: upsample in multiple stages."""
    batch, channels, *spatial_dims = volume.shape
    original_slices = spatial_dims[axis]

    # Compute intermediate slice counts
    total_factor = target_slices / original_slices
    stage_factor = total_factor ** (1 / num_stages)

    current_volume = volume
    current_slices = original_slices

    for stage in range(num_stages):
        # Compute target for this stage
        if stage == num_stages - 1:
            # Final stage: exact target
            stage_target = target_slices
        else:
            # Intermediate stage
            stage_target = int(current_slices * stage_factor)

        logger.info(f"Stage {stage + 1}/{num_stages}: {current_slices} → {stage_target} slices")

        # Super-resolve this stage
        current_volume = super_resolve_slices(
            model=model,
            tokenizer=tokenizer,
            volume=current_volume,
            target_slices=stage_target,
            axis=axis,
            model_type=model_type,
            **inpaint_kwargs,
        )

        current_slices = stage_target

    return current_volume


def _inpaint_volume_with_mask(
    model: nn.Module,
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    spatial_mask: torch.Tensor,
    model_type: str,
    rasterization_method: str,
    **inpaint_kwargs,
) -> torch.Tensor:
    """Internal: Inpaint volume with spatial mask."""
    device = volume.device

    # Tokenize
    with torch.no_grad():
        tokens = tokenizer.encode(volume)

    # Convert spatial mask to sequence mask
    seq_mask = spatial_to_sequence_mask(spatial_mask, rasterization_method)
    seq_mask = seq_mask.to(device)

    # Select inpainting method
    if model_type == "autoreg":
        inpainted_tokens = inpaint_autoregressive(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "maskgit":
        inpainted_tokens = inpaint_maskgit(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "flow":
        inpainted_tokens = inpaint_flow_matching(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "diffusion":
        inpainted_tokens = inpaint_diffusion_repaint(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "bayesian_flow":
        inpainted_tokens = inpaint_bayesian_flow(model, tokens, seq_mask, **inpaint_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Decode
    with torch.no_grad():
        inpainted_volume = tokenizer.detokenize(inpainted_tokens)

    return inpainted_volume


def compare_with_interpolation(
    model: nn.Module,
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    target_slices: int,
    axis: int = 0,
    model_type: str = "maskgit",
    **inpaint_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare model-based super-resolution with interpolation baseline."""
    # Baseline: interpolation only
    interpolated = reslice_volume(volume, axis, target_slices, mode="trilinear")

    # Model-based enhancement
    enhanced = super_resolve_slices(
        model=model,
        tokenizer=tokenizer,
        volume=volume,
        target_slices=target_slices,
        axis=axis,
        model_type=model_type,
        **inpaint_kwargs,
    )

    return interpolated, enhanced
