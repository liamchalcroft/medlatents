"""S-curve (serpentine) scan for spatial-to-sequence conversion."""

from typing import Literal

import torch


class SCurve:
    """
    S-curve (serpentine/boustrophedon) scan: alternating row direction.

    Traverses rows left-to-right then right-to-left alternately, creating
    a continuous snake-like path.

    Properties:
    - Simple to implement
    - All horizontal neighbors remain adjacent in sequence
    - Vertical neighbors have same distance as standard raster scan

    Example 4x4 image:
        Row 0: [0,  1,  2,  3 ] →
        Row 1: [7,  6,  5,  4 ] ←
        Row 2: [8,  9,  10, 11] →
        Row 3: [15, 14, 13, 12] ←

    Sequence: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    """

    @staticmethod
    def spatial_to_sequence_2d(
        spatial: torch.Tensor, order: Literal["row", "col"] = "row"
    ) -> torch.Tensor:
        """
        Convert 2D spatial data to 1D sequence using S-curve scan.

        Args:
            spatial: [batch, channels, height, width] or [batch, height, width]
            order: 'row' (horizontal serpentine) or 'col' (vertical serpentine)

        Returns:
            sequence: [batch, channels, height*width] or [batch, height*width]
        """
        if order == "row":
            # Horizontal serpentine
            h, w = spatial.shape[-2:]

            # Flip every other row
            result = spatial.clone()
            if result.dim() == 4:
                result[:, :, 1::2, :] = result[:, :, 1::2, :].flip(dims=[-1])
            else:
                result[:, 1::2, :] = result[:, 1::2, :].flip(dims=[-1])

            # Flatten
            return result.flatten(start_dim=-2)
        else:
            # Vertical serpentine: transpose, serpentine rows, flatten
            # Must clone since transpose returns a view
            transposed = spatial.transpose(-2, -1).clone()
            if transposed.dim() == 4:
                transposed[:, :, 1::2, :] = transposed[:, :, 1::2, :].flip(dims=[-1])
            else:
                transposed[:, 1::2, :] = transposed[:, 1::2, :].flip(dims=[-1])
            return transposed.flatten(start_dim=-2)

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
        # Reverse serpentine by flipping every other row
        if order == "row":
            # Reshape to spatial
            if sequence.dim() == 3:
                batch, channels, seq_len = sequence.shape
                spatial = sequence.view(batch, channels, height, width)
            else:
                batch, seq_len = sequence.shape
                spatial = sequence.view(batch, height, width)
            if spatial.dim() == 4:
                spatial[:, :, 1::2, :] = spatial[:, :, 1::2, :].flip(dims=[-1])
            else:
                spatial[:, 1::2, :] = spatial[:, 1::2, :].flip(dims=[-1])
        else:
            # For column order, reshape with swapped dimensions, then un-serpentine, then transpose
            if sequence.dim() == 3:
                batch, channels, seq_len = sequence.shape
                spatial = sequence.view(batch, channels, width, height)
            else:
                batch, seq_len = sequence.shape
                spatial = sequence.view(batch, width, height)
            if spatial.dim() == 4:
                spatial[:, :, 1::2, :] = spatial[:, :, 1::2, :].flip(dims=[-1])
            else:
                spatial[:, 1::2, :] = spatial[:, 1::2, :].flip(dims=[-1])
            spatial = spatial.transpose(-2, -1)

        return spatial

    @staticmethod
    def spatial_to_sequence_3d(
        spatial: torch.Tensor, order: Literal["D", "H", "W"] = "D"
    ) -> torch.Tensor:
        """
        Convert 3D spatial data to 1D sequence using S-curve scan.

        Args:
            spatial: [batch, channels, depth, height, width] or [batch, D, H, W]
            order: which dimension to serpentine ('D', 'H', or 'W')

        Returns:
            sequence: [batch, channels, D*H*W] or [batch, D*H*W]
        """
        if order == "D":
            # Serpentine along depth (flip every other slice)
            result = spatial.clone()
            if result.dim() == 5:
                result[:, :, 1::2, :, :] = result[:, :, 1::2, :, :].flip(dims=[-2, -1])
            else:
                result[:, 1::2, :, :] = result[:, 1::2, :, :].flip(dims=[-2, -1])
            return result.flatten(start_dim=-3 if result.dim() == 5 else -3)

        elif order == "H":
            # Serpentine along height
            result = spatial.clone()
            if result.dim() == 5:
                result[:, :, :, 1::2, :] = result[:, :, :, 1::2, :].flip(dims=[-1])
            else:
                result[:, :, 1::2, :] = result[:, :, 1::2, :].flip(dims=[-1])
            return result.flatten(start_dim=-3 if result.dim() == 5 else -3)

        else:  # order == "W"
            # Serpentine along width (standard row serpentine for each slice)
            result = spatial.clone()
            if result.dim() == 5:
                result[:, :, :, :, 1::2] = result[:, :, :, :, 1::2].flip(dims=[-2])
            else:
                result[:, :, :, 1::2] = result[:, :, :, 1::2].flip(dims=[-2])
            return result.flatten(start_dim=-3 if result.dim() == 5 else -3)

    @staticmethod
    def sequence_to_spatial_3d(
        sequence: torch.Tensor,
        depth: int,
        height: int,
        width: int,
        order: Literal["D", "H", "W"] = "D",
    ) -> torch.Tensor:
        """
        Convert 1D sequence back to 3D spatial data.

        Args:
            sequence: [batch, channels, D*H*W] or [batch, D*H*W]
            depth, height, width: spatial dimensions
            order: which dimension was serpentined

        Returns:
            spatial: [batch, channels, D, H, W] or [batch, D, H, W]
        """
        # Reshape
        if sequence.dim() == 3:
            batch, channels, seq_len = sequence.shape
            spatial = sequence.view(batch, channels, depth, height, width)
        else:
            batch, seq_len = sequence.shape
            spatial = sequence.view(batch, depth, height, width)

        # Reverse serpentine
        if order == "D":
            if spatial.dim() == 5:
                spatial[:, :, 1::2, :, :] = spatial[:, :, 1::2, :, :].flip(dims=[-2, -1])
            else:
                spatial[:, 1::2, :, :] = spatial[:, 1::2, :, :].flip(dims=[-2, -1])
        elif order == "H":
            if spatial.dim() == 5:
                spatial[:, :, :, 1::2, :] = spatial[:, :, :, 1::2, :].flip(dims=[-1])
            else:
                spatial[:, :, 1::2, :] = spatial[:, :, 1::2, :].flip(dims=[-1])
        else:  # order == "W"
            if spatial.dim() == 5:
                spatial[:, :, :, :, 1::2] = spatial[:, :, :, :, 1::2].flip(dims=[-2])
            else:
                spatial[:, :, :, 1::2] = spatial[:, :, :, 1::2].flip(dims=[-2])

        return spatial
