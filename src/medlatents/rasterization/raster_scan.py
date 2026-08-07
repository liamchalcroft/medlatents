"""Standard raster scan for spatial-to-sequence conversion."""

from typing import Literal

import torch


class RasterScan:
    """
    Standard raster scan ordering: row-by-row (or slice-by-slice for 3D).

    Simple and fast. Consecutive tokens in the sequence may be spatially
    distant (e.g., end of row N and start of row N+1).

    Example 2x2 image:
        [[0, 1],
         [2, 3]]

    Sequence: [0, 1, 2, 3]
    """

    @staticmethod
    def spatial_to_sequence_2d(
        spatial: torch.Tensor, order: Literal["row", "col"] = "row"
    ) -> torch.Tensor:
        """
        Convert 2D spatial data to 1D sequence using raster scan.

        Args:
            spatial: [batch, channels, height, width] or [batch, height, width]
            order: 'row' (row-major) or 'col' (column-major)

        Returns:
            sequence: [batch, channels, height*width] or [batch, height*width]
        """
        if order == "row":
            # Flatten row-by-row (standard)
            return spatial.flatten(start_dim=-2)
        else:
            # Transpose then flatten for column-major
            return spatial.transpose(-2, -1).flatten(start_dim=-2)

    @staticmethod
    def sequence_to_spatial_2d(
        sequence: torch.Tensor,
        height: int,
        width: int,
        order: Literal["row", "col"] = "row",
    ) -> torch.Tensor:
        """
        Convert 1D sequence back to 2D spatial data.

        Args:
            sequence: [batch, channels, height*width] or [batch, height*width]
            height: spatial height
            width: spatial width
            order: 'row' or 'col'

        Returns:
            spatial: [batch, channels, height, width] or [batch, height, width]
        """
        if order == "col":
            # For col-major, data was stored as (width, height), so reshape accordingly
            if sequence.dim() == 3:
                batch, channels, seq_len = sequence.shape
                spatial = sequence.view(batch, channels, width, height)
            else:
                batch, seq_len = sequence.shape
                spatial = sequence.view(batch, width, height)
            return spatial.transpose(-2, -1)
        else:
            if sequence.dim() == 3:
                batch, channels, seq_len = sequence.shape
                spatial = sequence.view(batch, channels, height, width)
            else:
                batch, seq_len = sequence.shape
                spatial = sequence.view(batch, height, width)
            return spatial

    @staticmethod
    def spatial_to_sequence_3d(
        spatial: torch.Tensor, order: Literal["DHW", "WHD", "HWD"] = "DHW"
    ) -> torch.Tensor:
        """
        Convert 3D spatial data to 1D sequence.

        Args:
            spatial: [batch, channels, depth, height, width] or [batch, D, H, W]
            order: dimension ordering ('DHW' = slice-by-slice, etc.)

        Returns:
            sequence: [batch, channels, D*H*W] or [batch, D*H*W]
        """
        # Reorder dimensions if needed
        if order == "WHD":
            spatial = (
                spatial.permute(0, 1, 4, 3, 2)
                if spatial.dim() == 5
                else spatial.permute(0, 3, 2, 1)
            )
        elif order == "HWD":
            spatial = (
                spatial.permute(0, 1, 3, 4, 2)
                if spatial.dim() == 5
                else spatial.permute(0, 2, 3, 1)
            )

        # Flatten
        return spatial.flatten(start_dim=-3 if spatial.dim() == 5 else -3)

    @staticmethod
    def sequence_to_spatial_3d(
        sequence: torch.Tensor,
        depth: int,
        height: int,
        width: int,
        order: Literal["DHW", "WHD", "HWD"] = "DHW",
    ) -> torch.Tensor:
        """
        Convert 1D sequence back to 3D spatial data.

        Args:
            sequence: [batch, channels, D*H*W] or [batch, D*H*W]
            depth, height, width: spatial dimensions
            order: dimension ordering

        Returns:
            spatial: [batch, channels, D, H, W] or [batch, D, H, W]
        """
        if sequence.dim() == 3:
            batch, channels, seq_len = sequence.shape
            spatial = sequence.view(batch, channels, depth, height, width)
        else:
            batch, seq_len = sequence.shape
            spatial = sequence.view(batch, depth, height, width)

        # Reverse permutation
        if order == "WHD":
            spatial = (
                spatial.permute(0, 1, 4, 3, 2)
                if spatial.dim() == 5
                else spatial.permute(0, 3, 2, 1)
            )
        elif order == "HWD":
            spatial = (
                spatial.permute(0, 1, 4, 2, 3)
                if spatial.dim() == 5
                else spatial.permute(0, 3, 1, 2)
            )

        return spatial
