"""Z-order (Morton) curve for spatial-to-sequence conversion."""

import math
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from .hilbert_curve import HilbertCurve
from .raster_scan import RasterScan
from .s_curve import SCurve


class ZOrderCurve:
    """
    Z-order (Morton) curve for hierarchical locality-preserving rasterization.

    The Z-order curve interleaves the bits of x and y coordinates, creating
    a hierarchical quad-tree traversal with fast computation via bit operations.

    Properties:
    - Hierarchical locality preservation
    - Fast computation via bit interleaving
    - Natural quad-tree structure
    - Requires power-of-2 dimensions

    Example 4x4:
        [[0,  1,  4,  5],
         [2,  3,  6,  7],
         [8,  9,  12, 13],
         [10, 11, 14, 15]]
    """

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_zorder_indices_2d(order: int) -> np.ndarray:
        """Get precomputed reorder indices for 2D Z-order curve. Cached for performance."""
        curve = ZOrderCurve._z_order_curve_2d(order)
        return curve.flatten().argsort()

    @staticmethod
    def _part_by_1_vectorized(n: np.ndarray) -> np.ndarray:
        """Vectorized bit spreading by factor of 2 for 2D Morton encoding."""
        n = n.astype(np.int64)
        n &= 0x0000FFFF
        n = (n ^ (n << 8)) & 0x00FF00FF
        n = (n ^ (n << 4)) & 0x0F0F0F0F
        n = (n ^ (n << 2)) & 0x33333333
        n = (n ^ (n << 1)) & 0x55555555
        return n

    @staticmethod
    @lru_cache(maxsize=32)
    def _z_order_curve_2d(order: int) -> np.ndarray:
        """Generate 2D Z-order curve using vectorized operations. Cached for performance."""
        n = 2**order
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:n, 0:n]
        # Vectorized Morton encoding: interleave bits of x and y
        curve = ZOrderCurve._part_by_1_vectorized(x_coords) | (
            ZOrderCurve._part_by_1_vectorized(y_coords) << 1
        )
        return curve.astype(np.int64)

    @staticmethod
    def spatial_to_sequence_2d(spatial: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Convert 2D spatial to sequence using Z-order curve."""
        original_shape = spatial.shape
        h, w = spatial.shape[-2:]

        # Pad to power of 2
        size = max(h, w)
        order = math.ceil(math.log2(size))
        n = 2**order

        # Get cached curve and precomputed indices
        curve = ZOrderCurve._z_order_curve_2d(order)
        curve = torch.from_numpy(curve).to(spatial.device)
        reorder_indices = torch.from_numpy(ZOrderCurve._get_zorder_indices_2d(order)).to(
            spatial.device
        )

        # Pad if needed
        pad_h, pad_w = n - h, n - w
        if pad_h > 0 or pad_w > 0:
            spatial = torch.nn.functional.pad(spatial, (0, pad_w, 0, pad_h))

        # Create mask for valid (non-padded) positions
        valid_mask = torch.ones((n, n), dtype=torch.bool, device=spatial.device)
        if pad_h > 0 or pad_w > 0:
            valid_mask[h:, :] = False
            valid_mask[:, w:] = False

        # Flatten and reorder using precomputed indices
        if spatial.dim() == 4:
            flat = spatial.flatten(start_dim=-2)
            sequence = flat[:, :, reorder_indices]
            # Extract only valid positions (in Z-order)
            valid_mask_reordered = valid_mask.flatten()[reorder_indices]
            sequence = sequence[:, :, valid_mask_reordered]
        else:
            flat = spatial.flatten(start_dim=-2)
            sequence = flat[:, reorder_indices]
            # Extract only valid positions (in Z-order)
            valid_mask_reordered = valid_mask.flatten()[reorder_indices]
            sequence = sequence[:, valid_mask_reordered]

        metadata = {
            "curve": curve,
            "original_shape": original_shape,
            "padding": (pad_h, pad_w),
            "order": order,
            "valid_mask": valid_mask,
        }

        return sequence, metadata

    @staticmethod
    def sequence_to_spatial_2d(sequence: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Convert sequence back to spatial using Z-order curve."""
        curve = metadata["curve"]
        original_shape = metadata["original_shape"]
        pad_h, pad_w = metadata["padding"]
        order = metadata["order"]
        valid_mask = metadata["valid_mask"]
        n = 2**order
        h, w = original_shape[-2:]

        # Reorder indices for Z-order curve
        reorder_indices = curve.flatten().argsort()
        valid_mask_reordered = valid_mask.flatten()[reorder_indices]

        # Create full padded sequence (fill invalid positions with zeros)
        padded_len = n * n
        if sequence.dim() == 3:
            batch, channels, _ = sequence.shape
            full_sequence = torch.zeros(
                batch,
                channels,
                padded_len,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            full_sequence[:, :, valid_mask_reordered] = sequence
        else:
            batch, _ = sequence.shape
            full_sequence = torch.zeros(
                batch, padded_len, device=sequence.device, dtype=sequence.dtype
            )
            full_sequence[:, valid_mask_reordered] = sequence

        # Reverse reordering
        inverse_indices = torch.zeros_like(reorder_indices)
        inverse_indices[reorder_indices] = torch.arange(
            len(reorder_indices), device=reorder_indices.device
        )

        if sequence.dim() == 3:
            reordered = full_sequence[:, :, inverse_indices]
            spatial = reordered.view(batch, channels, n, n)
        else:
            reordered = full_sequence[:, inverse_indices]
            spatial = reordered.view(batch, n, n)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            spatial = spatial[..., :h, :w]

        return spatial

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_zorder_indices_3d(order: int) -> np.ndarray:
        """Get precomputed reorder indices for 3D Z-order curve. Cached for performance."""
        curve = ZOrderCurve._z_order_curve_3d(order)
        return curve.flatten().argsort()

    @staticmethod
    def _part_by_2_vectorized(n: np.ndarray) -> np.ndarray:
        """Vectorized bit spreading by factor of 3 for 3D Morton encoding."""
        n = n.astype(np.int64)
        n &= 0x000003FF  # Keep only lowest 10 bits
        n = (n ^ (n << 16)) & 0xFF0000FF
        n = (n ^ (n << 8)) & 0x0300F00F
        n = (n ^ (n << 4)) & 0x030C30C3
        n = (n ^ (n << 2)) & 0x09249249
        return n

    @staticmethod
    @lru_cache(maxsize=32)
    def _z_order_curve_3d(order: int) -> np.ndarray:
        """Generate 3D Z-order curve using vectorized operations. Cached for performance."""
        n = 2**order
        # Create 3D coordinate grids
        z_coords, y_coords, x_coords = np.mgrid[0:n, 0:n, 0:n]
        # Vectorized 3D Morton encoding: interleave bits of x, y, z
        curve = (
            ZOrderCurve._part_by_2_vectorized(x_coords)
            | (ZOrderCurve._part_by_2_vectorized(y_coords) << 1)
            | (ZOrderCurve._part_by_2_vectorized(z_coords) << 2)
        )
        return curve.astype(np.int64)

    @staticmethod
    def spatial_to_sequence_3d(spatial: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Convert 3D spatial data to sequence using Z-order curve.

        Args:
            spatial: [batch, channels, D, H, W] or [batch, D, H, W]

        Returns:
            sequence: [batch, channels, D*H*W] or [batch, D*H*W]
            metadata: dict with curve, padding, original_shape, order
        """
        original_shape = spatial.shape
        d, h, w = spatial.shape[-3:]

        # Pad to cubic power of 2
        size = max(d, h, w)
        order = math.ceil(math.log2(size))
        n = 2**order

        # Get cached curve and precomputed indices
        curve = ZOrderCurve._z_order_curve_3d(order)
        curve = torch.from_numpy(curve).to(spatial.device)
        reorder_indices = torch.from_numpy(ZOrderCurve._get_zorder_indices_3d(order)).to(
            spatial.device
        )

        # Pad if needed
        pad_d, pad_h, pad_w = n - d, n - h, n - w
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            if spatial.dim() == 5:
                spatial = torch.nn.functional.pad(spatial, (0, pad_w, 0, pad_h, 0, pad_d))
            else:
                spatial = torch.nn.functional.pad(spatial, (0, pad_w, 0, pad_h, 0, pad_d))

        # Create mask for valid (non-padded) voxels
        valid_mask = torch.ones((n, n, n), dtype=torch.bool, device=spatial.device)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            valid_mask[d:, :, :] = False
            valid_mask[:, h:, :] = False
            valid_mask[:, :, w:] = False

        # Flatten and reorder using precomputed indices
        if spatial.dim() == 5:
            batch, channels, _, _, _ = spatial.shape
            flat = spatial.flatten(start_dim=-3)
            sequence = flat[:, :, reorder_indices]
            # Extract only valid voxels
            valid_mask_reordered = valid_mask.flatten()[reorder_indices]
            sequence = sequence[:, :, valid_mask_reordered]
        else:
            batch, _, _, _ = spatial.shape
            flat = spatial.flatten(start_dim=-3)
            sequence = flat[:, reorder_indices]
            # Extract only valid voxels
            valid_mask_reordered = valid_mask.flatten()[reorder_indices]
            sequence = sequence[:, valid_mask_reordered]

        metadata = {
            "curve": curve,
            "original_shape": original_shape,
            "padding": (pad_d, pad_h, pad_w),
            "order": order,
            "valid_mask": valid_mask,
        }

        return sequence, metadata

    @staticmethod
    def sequence_to_spatial_3d(
        sequence: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        """
        Convert sequence back to 3D spatial using Z-order curve.

        Args:
            sequence: [batch, channels, seq_len] or [batch, seq_len]
            metadata: dict from spatial_to_sequence_3d

        Returns:
            spatial: [batch, channels, D, H, W] or [batch, D, H, W]
        """
        curve = metadata["curve"]
        original_shape = metadata["original_shape"]
        pad_d, pad_h, pad_w = metadata["padding"]
        order = metadata["order"]
        valid_mask = metadata["valid_mask"]
        n = 2**order

        depth, height, width = original_shape[-3:]

        # Reorder indices for Z-order curve
        reorder_indices = curve.flatten().argsort()
        valid_mask_reordered = valid_mask.flatten()[reorder_indices]

        # Create full padded sequence (fill invalid positions with zeros)
        padded_len = n * n * n
        if sequence.dim() == 3:
            batch, channels, _ = sequence.shape
            full_sequence = torch.zeros(
                batch,
                channels,
                padded_len,
                device=sequence.device,
                dtype=sequence.dtype,
            )
            full_sequence[:, :, valid_mask_reordered] = sequence
        else:
            batch, _ = sequence.shape
            full_sequence = torch.zeros(
                batch, padded_len, device=sequence.device, dtype=sequence.dtype
            )
            full_sequence[:, valid_mask_reordered] = sequence

        # Reverse reordering
        inverse_indices = torch.zeros_like(reorder_indices)
        inverse_indices[reorder_indices] = torch.arange(
            len(reorder_indices), device=reorder_indices.device
        )

        if sequence.dim() == 3:
            reordered = full_sequence[:, :, inverse_indices]
            spatial = reordered.view(batch, channels, n, n, n)
        else:
            reordered = full_sequence[:, inverse_indices]
            spatial = reordered.view(batch, n, n, n)

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            spatial = spatial[..., :depth, :height, :width]

        return spatial


# Convenience functions
def rasterize_2d(
    spatial: torch.Tensor,
    method: Literal["raster", "scurve", "hilbert", "zorder"] = "raster",
    **kwargs,
) -> tuple[torch.Tensor, dict | None]:
    """
    Convert 2D spatial data to 1D sequence.

    Args:
        spatial: [batch, channels, H, W] or [batch, H, W]
        method: 'raster' (row-by-row), 'scurve' (serpentine),
                'hilbert' (space-filling curve), 'zorder' (Morton/quad-tree)
        **kwargs: method-specific arguments (e.g., order='row'/'col')

    Returns:
        sequence: [batch, channels, H*W] or [batch, H*W]
        metadata: None for raster/scurve, dict for hilbert/zorder (needed for reconstruction)
    """
    if method == "raster":
        return RasterScan.spatial_to_sequence_2d(spatial, **kwargs), None
    elif method == "scurve":
        return SCurve.spatial_to_sequence_2d(spatial, **kwargs), None
    elif method == "hilbert":
        return HilbertCurve.spatial_to_sequence_2d(spatial)
    elif method == "zorder":
        return ZOrderCurve.spatial_to_sequence_2d(spatial)
    else:
        raise ValueError(f"Unknown method: {method}")


def unrasterize_2d(
    sequence: torch.Tensor,
    height: int,
    width: int,
    method: Literal["raster", "scurve", "hilbert", "zorder"] = "raster",
    metadata: dict | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Convert 1D sequence back to 2D spatial data.

    Args:
        sequence: [batch, channels, H*W] or [batch, H*W]
        height, width: spatial dimensions
        method: must match the rasterization method used
        metadata: required for hilbert/zorder, None for raster/scurve
        **kwargs: method-specific arguments (e.g., order='row'/'col')

    Returns:
        spatial: [batch, channels, H, W] or [batch, H, W]
    """
    if method == "raster":
        return RasterScan.sequence_to_spatial_2d(sequence, height, width, **kwargs)
    elif method == "scurve":
        return SCurve.sequence_to_spatial_2d(sequence, height, width, **kwargs)
    elif method == "hilbert":
        if metadata is None:
            raise ValueError("metadata required for Hilbert curve reconstruction")
        return HilbertCurve.sequence_to_spatial_2d(sequence, metadata)
    elif method == "zorder":
        if metadata is None:
            raise ValueError("metadata required for Z-order reconstruction")
        return ZOrderCurve.sequence_to_spatial_2d(sequence, metadata)
    else:
        raise ValueError(f"Unknown method: {method}")
