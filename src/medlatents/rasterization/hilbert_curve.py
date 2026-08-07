"""Hilbert space-filling curve for spatial-to-sequence conversion."""

import math
from functools import lru_cache

import numpy as np
import torch


class HilbertCurve:
    """
    Hilbert space-filling curve for locality-preserving rasterization.

    The Hilbert curve traverses a 2D grid such that nearby points in the
    sequence are also nearby in 2D space, maintaining strong spatial correlations
    between consecutive positions.

    Properties:
    - Strong locality preservation (nearby points remain nearby)
    - Smooth, continuous traversal
    - Requires power-of-2 dimensions (pads if needed)

    Example 2x2:
        [[0, 1],
         [3, 2]]  (U-shape)

    Example 4x4:
        [[0,  1,  14, 15],
         [3,  2,  13, 12],
         [4,  7,  8,  11],
         [5,  6,  9,  10]]
    """

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_hilbert_indices_2d(order: int) -> np.ndarray:
        """
        Get precomputed reorder indices for 2D Hilbert curve.

        Cached for performance - indices are computed once per order.

        Returns:
            indices: flattened indices for reordering
        """
        curve = HilbertCurve._hilbert_curve_2d(order)
        return curve.flatten().argsort()

    @staticmethod
    @lru_cache(maxsize=32)
    def _hilbert_curve_2d(order: int) -> np.ndarray:
        """
        Generate 2D Hilbert curve of given order using vectorized operations.

        Cached for performance - curves are computed once per order.

        Args:
            order: curve order (size will be 2^order x 2^order)

        Returns:
            curve: [2^order, 2^order] array with traversal indices
        """
        n = 2**order
        total = n * n

        # Vectorized Hilbert d2xy conversion
        d = np.arange(total, dtype=np.int64)
        x = np.zeros(total, dtype=np.int64)
        y = np.zeros(total, dtype=np.int64)

        s = 1
        d_remaining = d.copy()
        while s < n:
            rx = (d_remaining >> 1) & 1
            ry = (d_remaining & 1) ^ rx

            # Vectorized rotation: where ry == 0
            mask_ry0 = ry == 0
            mask_rx1 = rx == 1

            # Where ry==0 and rx==1: flip both
            flip_mask = mask_ry0 & mask_rx1
            x[flip_mask] = s - 1 - x[flip_mask]
            y[flip_mask] = s - 1 - y[flip_mask]

            # Where ry==0: swap x, y
            x_temp = x[mask_ry0].copy()
            x[mask_ry0] = y[mask_ry0]
            y[mask_ry0] = x_temp

            x += s * rx
            y += s * ry
            d_remaining >>= 2
            s <<= 1

        # Build curve array
        curve = np.zeros((n, n), dtype=np.int64)
        curve[y, x] = d

        return curve

    @staticmethod
    def spatial_to_sequence_2d(spatial: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Convert 2D spatial data to sequence using Hilbert curve.

        Args:
            spatial: [batch, channels, height, width] or [batch, height, width]

        Returns:
            sequence: [batch, channels, height*width] or [batch, height*width]
            metadata: dict with 'curve', 'original_shape', 'padding' for reconstruction
        """
        original_shape = spatial.shape
        h, w = spatial.shape[-2:]

        # Pad to power of 2
        size = max(h, w)
        order = math.ceil(math.log2(size))
        n = 2**order

        # Get cached curve and precomputed indices
        curve = HilbertCurve._hilbert_curve_2d(order)
        curve = torch.from_numpy(curve).to(spatial.device)
        reorder_indices = torch.from_numpy(HilbertCurve._get_hilbert_indices_2d(order)).to(
            spatial.device
        )

        # Pad spatial data if needed
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
            batch, channels, _, _ = spatial.shape
            flat = spatial.flatten(start_dim=-2)  # [batch, channels, n*n]
            sequence = flat[:, :, reorder_indices]
            # Extract only valid positions (in Hilbert order)
            valid_mask_reordered = valid_mask.flatten()[reorder_indices]
            sequence = sequence[:, :, valid_mask_reordered]
        else:
            batch, _, _ = spatial.shape
            flat = spatial.flatten(start_dim=-2)  # [batch, n*n]
            sequence = flat[:, reorder_indices]
            # Extract only valid positions (in Hilbert order)
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
        """
        Convert sequence back to 2D spatial data using Hilbert curve.

        Args:
            sequence: [batch, channels, seq_len] or [batch, seq_len]
            metadata: returned from spatial_to_sequence_2d

        Returns:
            spatial: original shape
        """
        curve = metadata["curve"]
        original_shape = metadata["original_shape"]
        pad_h, pad_w = metadata["padding"]
        order = metadata["order"]
        valid_mask = metadata["valid_mask"]
        n = 2**order

        h, w = original_shape[-2:]

        # Reorder indices for Hilbert curve
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
    def _get_hilbert_indices_3d(order: int) -> np.ndarray:
        """
        Get precomputed reorder indices for 3D Hilbert curve.

        Cached for performance - indices are computed once per order.

        Returns:
            indices: flattened indices for reordering
        """
        curve = HilbertCurve._hilbert_curve_3d(order)
        return curve.flatten().argsort()

    @staticmethod
    @lru_cache(maxsize=32)
    def _hilbert_curve_3d(order: int) -> np.ndarray:
        """
        Generate 3D Hilbert curve of given order using vectorized operations.

        Cached for performance - curves are computed once per order.

        Args:
            order: curve order (size will be 2^order x 2^order x 2^order)

        Returns:
            curve: [2^order, 2^order, 2^order] array with traversal indices
        """
        n = 2**order
        total = n * n * n

        # Vectorized Hilbert d2xyz conversion for 3D
        d = np.arange(total, dtype=np.int64)
        x = np.zeros(total, dtype=np.int64)
        y = np.zeros(total, dtype=np.int64)
        z = np.zeros(total, dtype=np.int64)

        s = 1
        d_remaining = d.copy()
        while s < n:
            # Extract quadrant bits
            rx = (d_remaining >> 2) & 1
            ry = ((d_remaining >> 1) ^ rx) & 1
            rz = (d_remaining ^ (d_remaining >> 1)) & 1

            # Rotation for rz==0, ry==0 (swap x,z and conditionally flip)
            mask_rz0_ry0 = (rz == 0) & (ry == 0)
            mask_rz0_ry0_rx1 = mask_rz0_ry0 & (rx == 1)
            # Flip x and z where rz==0, ry==0, rx==1
            x[mask_rz0_ry0_rx1] = s - 1 - x[mask_rz0_ry0_rx1]
            z[mask_rz0_ry0_rx1] = s - 1 - z[mask_rz0_ry0_rx1]
            # Swap x, z where rz==0, ry==0
            x_temp = x[mask_rz0_ry0].copy()
            x[mask_rz0_ry0] = z[mask_rz0_ry0]
            z[mask_rz0_ry0] = x_temp

            # Rotation for rz==0, ry==1 (swap y,z and conditionally flip)
            mask_rz0_ry1 = (rz == 0) & (ry == 1)
            mask_rz0_ry1_rx0 = mask_rz0_ry1 & (rx == 0)
            # Flip y and z where rz==0, ry==1, rx==0
            y[mask_rz0_ry1_rx0] = s - 1 - y[mask_rz0_ry1_rx0]
            z[mask_rz0_ry1_rx0] = s - 1 - z[mask_rz0_ry1_rx0]
            # Swap y, z where rz==0, ry==1
            y_temp = y[mask_rz0_ry1].copy()
            y[mask_rz0_ry1] = z[mask_rz0_ry1]
            z[mask_rz0_ry1] = y_temp

            # Translate based on quadrant
            x += s * rx
            y += s * ry
            z += s * rz
            d_remaining >>= 3
            s <<= 1

        # Build curve array
        curve = np.zeros((n, n, n), dtype=np.int64)
        curve[z, y, x] = d

        return curve

    @staticmethod
    def spatial_to_sequence_3d(spatial: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Convert 3D spatial data to sequence using Hilbert curve.

        Args:
            spatial: [batch, channels, depth, height, width] or [batch, D, H, W]

        Returns:
            sequence: [batch, channels, D*H*W] or [batch, D*H*W]
            metadata: dict with 'curve', 'original_shape', 'padding' for reconstruction
        """
        original_shape = spatial.shape
        d, h, w = spatial.shape[-3:]

        # Pad to power of 2 (cubic)
        size = max(d, h, w)
        order = math.ceil(math.log2(size))
        n = 2**order

        # Get cached curve and precomputed indices
        curve = HilbertCurve._hilbert_curve_3d(order)
        curve = torch.from_numpy(curve).to(spatial.device)
        reorder_indices = torch.from_numpy(HilbertCurve._get_hilbert_indices_3d(order)).to(
            spatial.device
        )

        # Pad spatial data if needed
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
            flat = spatial.flatten(start_dim=-3)  # [batch, channels, n*n*n]
            sequence = flat[:, :, reorder_indices]
            # Extract only valid voxels
            valid_mask_reordered = valid_mask.flatten()[reorder_indices]
            sequence = sequence[:, :, valid_mask_reordered]
        else:
            batch, _, _, _ = spatial.shape
            flat = spatial.flatten(start_dim=-3)  # [batch, n*n*n]
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
    def sequence_to_spatial_3d(sequence: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Convert sequence back to 3D spatial data using Hilbert curve.

        Args:
            sequence: [batch, channels, seq_len] or [batch, seq_len]
            metadata: returned from spatial_to_sequence_3d

        Returns:
            spatial: original shape
        """
        curve = metadata["curve"]
        original_shape = metadata["original_shape"]
        pad_d, pad_h, pad_w = metadata["padding"]
        order = metadata["order"]
        valid_mask = metadata["valid_mask"]
        n = 2**order

        d, h, w = original_shape[-3:]

        # Reorder indices for Hilbert curve
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
            spatial = spatial[..., :d, :h, :w]

        return spatial
