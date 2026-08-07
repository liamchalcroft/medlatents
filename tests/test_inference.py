"""Tests for inference utilities."""

import pytest
import torch

from medlatents.inference import (
    apply_token_mask,
    confidence_mask,
    create_spatial_mask,
    spatial_to_sequence_mask,
)


class TestCreateSpatialMask:
    """Tests for spatial mask creation."""

    def test_random_mask_ratio(self):
        """Random mask should mask approximately the correct ratio."""
        shape = (32, 32)
        mask_ratio = 0.3
        mask = create_spatial_mask(shape, mask_type="random", mask_ratio=mask_ratio)

        # True = keep, False = mask
        kept_ratio = mask.sum().item() / mask.numel()
        masked_ratio = 1 - kept_ratio

        # Allow 10% tolerance
        assert abs(masked_ratio - mask_ratio) < 0.1

    def test_block_mask(self):
        """Block mask should mask a rectangular region."""
        shape = (16, 16)
        mask = create_spatial_mask(shape, mask_type="block", start=(4, 4), end=(12, 12))

        # Inside block should be False (masked)
        assert not mask[6, 6]
        assert not mask[10, 10]

        # Outside block should be True (kept)
        assert mask[0, 0]
        assert mask[15, 15]
        assert mask[0, 15]

    def test_block_mask_requires_params(self):
        """Block mask should require start and end."""
        with pytest.raises(ValueError, match="start.*end"):
            create_spatial_mask((16, 16), mask_type="block")

    def test_slice_mask_2d(self):
        """Slice mask should mask every Nth row/column."""
        shape = (16, 16)
        mask = create_spatial_mask(shape, mask_type="slice", axis=0, stride=2, start_offset=1)

        # Rows 1, 3, 5, ... should be masked (False)
        assert not mask[1, 0]
        assert not mask[3, 5]
        assert not mask[5, 10]

        # Rows 0, 2, 4, ... should be kept (True)
        assert mask[0, 0]
        assert mask[2, 5]
        assert mask[4, 10]

    def test_slice_mask_3d(self):
        """Slice mask should work for 3D volumes."""
        shape = (8, 16, 16)
        mask = create_spatial_mask(shape, mask_type="slice", axis=0, stride=2, start_offset=1)

        # Slices 1, 3, 5, 7 should be masked
        assert not mask[1, 0, 0]
        assert not mask[3, 8, 8]

        # Slices 0, 2, 4, 6 should be kept
        assert mask[0, 0, 0]
        assert mask[2, 8, 8]

    def test_checkerboard_2d(self):
        """Checkerboard should create alternating pattern in 2D."""
        shape = (4, 4)
        mask = create_spatial_mask(shape, mask_type="checkerboard")

        # Check checkerboard pattern
        expected = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
            ]
        )
        assert torch.equal(mask, expected)

    def test_checkerboard_3d(self):
        """Checkerboard should create alternating pattern in 3D."""
        shape = (2, 2, 2)
        mask = create_spatial_mask(shape, mask_type="checkerboard")

        # (0,0,0) should be True (even sum)
        assert mask[0, 0, 0]
        # (1,1,1) should be False (odd sum)
        assert not mask[1, 1, 1]
        # (1,0,0) should be False (odd sum)
        assert not mask[1, 0, 0]
        # (1,1,0) should be True (even sum)
        assert mask[1, 1, 0]

    def test_unknown_mask_type_raises(self):
        """Unknown mask type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mask_type"):
            create_spatial_mask((16, 16), mask_type="invalid")

    def test_invalid_axis_raises(self):
        """Invalid axis for slice mask should raise."""
        with pytest.raises(ValueError, match="Invalid axis"):
            create_spatial_mask((16, 16), mask_type="slice", axis=5)


class TestSpatialToSequenceMask:
    """Tests for converting spatial masks to sequence masks."""

    @pytest.mark.parametrize("method", ["raster", "hilbert", "zorder"])
    def test_preserves_mask_count(self, method):
        """Conversion should preserve number of masked positions."""
        spatial_mask = torch.rand(8, 8) > 0.5

        sequence_mask = spatial_to_sequence_mask(spatial_mask, rasterization_method=method)

        assert spatial_mask.sum() == sequence_mask.sum()
        assert sequence_mask.shape == (64,)

    def test_works_with_3d(self):
        """Should work with 3D spatial masks."""
        spatial_mask = torch.rand(4, 4, 4) > 0.5

        sequence_mask = spatial_to_sequence_mask(spatial_mask, rasterization_method="raster")

        assert spatial_mask.sum() == sequence_mask.sum()
        assert sequence_mask.shape == (64,)

    def test_unknown_method_raises(self):
        """Unknown rasterization method should raise."""
        spatial_mask = torch.rand(8, 8) > 0.5
        with pytest.raises(ValueError, match="Unknown rasterization_method"):
            spatial_to_sequence_mask(spatial_mask, rasterization_method="invalid")


class TestApplyTokenMask:
    """Tests for applying masks to token sequences."""

    def test_applies_mask_correctly(self):
        """Should replace masked positions with mask_value."""
        tokens = torch.arange(10).unsqueeze(0)  # [1, 10]
        mask = torch.tensor([True, True, False, True, False, True, True, False, True, True])

        masked = apply_token_mask(tokens, mask, mask_value=999)

        # Positions 2, 4, 7 should be masked
        assert masked[0, 2] == 999
        assert masked[0, 4] == 999
        assert masked[0, 7] == 999

        # Other positions should be unchanged
        assert masked[0, 0] == 0
        assert masked[0, 1] == 1
        assert masked[0, 3] == 3

    def test_preserves_input(self):
        """Should not modify original tensor."""
        tokens = torch.arange(10).unsqueeze(0)
        mask = torch.tensor([True, True, False, True, True, True, True, True, True, True])
        original = tokens.clone()

        apply_token_mask(tokens, mask, mask_value=999)

        assert torch.equal(tokens, original)


class TestConfidenceMask:
    """Tests for confidence-based masking."""

    def test_high_confidence_passes(self):
        """High confidence predictions should pass the threshold."""
        # Create logits where first position is very confident
        logits = torch.zeros(1, 4, 10)
        logits[0, 0, 5] = 100.0  # Very confident at position 0
        logits[0, 1, :] = 1.0  # Uniform at position 1

        mask = confidence_mask(logits, threshold=0.5)

        # Position 0 should be True (high confidence)
        assert mask[0, 0]
        # Position 1 should be False (low confidence)
        assert not mask[0, 1]

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        # Create uniform logits
        logits = torch.zeros(1, 2, 4)  # 4 classes, uniform = 0.25 probability each

        mask_low = confidence_mask(logits, threshold=0.2)  # Should pass
        mask_high = confidence_mask(logits, threshold=0.3)  # Should fail

        assert mask_low[0, 0]  # 0.25 >= 0.2
        assert not mask_high[0, 0]  # 0.25 < 0.3
