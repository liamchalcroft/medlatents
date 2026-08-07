"""Tests for spatial-to-sequence rasterization methods."""

import numpy as np
import pytest
import torch

from medlatents.rasterization import (
    HilbertCurve,
    RasterScan,
    SCurve,
    ZOrderCurve,
    rasterize_2d,
    unrasterize_2d,
)


class TestRasterScan:
    """Tests for standard raster scan."""

    def test_2d_row_major_round_trip(self):
        """Test 2D raster scan with row-major order preserves data."""
        batch, h, w = 2, 4, 6
        spatial = torch.arange(batch * h * w).reshape(batch, h, w).float()

        sequence = RasterScan.spatial_to_sequence_2d(spatial, order="row")
        recovered = RasterScan.sequence_to_spatial_2d(sequence, h, w, order="row")

        assert sequence.shape == (batch, h * w)
        assert recovered.shape == spatial.shape
        assert torch.allclose(spatial, recovered)

    def test_2d_col_major_round_trip(self):
        """Test 2D raster scan with column-major order."""
        batch, h, w = 2, 4, 6
        spatial = torch.arange(batch * h * w).reshape(batch, h, w).float()

        sequence = RasterScan.spatial_to_sequence_2d(spatial, order="col")
        recovered = RasterScan.sequence_to_spatial_2d(sequence, h, w, order="col")

        assert torch.allclose(spatial, recovered)

    def test_2d_with_channels(self):
        """Test 2D raster scan with channel dimension."""
        batch, channels, h, w = 2, 3, 4, 5
        spatial = torch.randn(batch, channels, h, w)

        sequence = RasterScan.spatial_to_sequence_2d(spatial, order="row")
        recovered = RasterScan.sequence_to_spatial_2d(sequence, h, w, order="row")

        assert sequence.shape == (batch, channels, h * w)
        assert torch.allclose(spatial, recovered)

    def test_3d_round_trip(self):
        """Test 3D raster scan preserves data."""
        batch, d, h, w = 2, 3, 4, 5
        spatial = torch.randn(batch, d, h, w)

        sequence = RasterScan.spatial_to_sequence_3d(spatial, order="DHW")
        recovered = RasterScan.sequence_to_spatial_3d(sequence, d, h, w, order="DHW")

        assert sequence.shape == (batch, d * h * w)
        assert torch.allclose(spatial, recovered)

    def test_3d_with_channels(self):
        """Test 3D raster scan with channel dimension."""
        batch, channels, d, h, w = 2, 3, 4, 5, 6
        spatial = torch.randn(batch, channels, d, h, w)

        sequence = RasterScan.spatial_to_sequence_3d(spatial, order="DHW")
        recovered = RasterScan.sequence_to_spatial_3d(sequence, d, h, w, order="DHW")

        assert sequence.shape == (batch, channels, d * h * w)
        assert torch.allclose(spatial, recovered)


class TestSCurve:
    """Tests for serpentine/S-curve scan."""

    def test_2d_serpentine_pattern(self):
        """Test that S-curve creates serpentine pattern."""
        spatial = torch.arange(12).reshape(1, 3, 4).float()
        sequence = SCurve.spatial_to_sequence_2d(spatial, order="row")

        # Row 0: [0, 1, 2, 3] forward
        # Row 1: [7, 6, 5, 4] reversed
        # Row 2: [8, 9, 10, 11] forward
        expected = torch.tensor([[0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11]]).float()
        assert torch.allclose(sequence, expected)

    def test_2d_round_trip(self):
        """Test S-curve round trip preserves data."""
        batch, h, w = 2, 5, 6
        spatial = torch.randn(batch, h, w)

        sequence = SCurve.spatial_to_sequence_2d(spatial, order="row")
        recovered = SCurve.sequence_to_spatial_2d(sequence, h, w, order="row")

        assert torch.allclose(spatial, recovered)

    def test_2d_col_order_round_trip(self):
        """Test S-curve with column order."""
        batch, h, w = 2, 4, 5
        spatial = torch.randn(batch, h, w)

        sequence = SCurve.spatial_to_sequence_2d(spatial, order="col")
        recovered = SCurve.sequence_to_spatial_2d(sequence, h, w, order="col")

        assert torch.allclose(spatial, recovered)

    def test_2d_with_channels(self):
        """Test S-curve with channels."""
        batch, channels, h, w = 2, 3, 4, 5
        spatial = torch.randn(batch, channels, h, w)

        sequence = SCurve.spatial_to_sequence_2d(spatial, order="row")
        recovered = SCurve.sequence_to_spatial_2d(sequence, h, w, order="row")

        assert torch.allclose(spatial, recovered)

    def test_3d_round_trip(self):
        """Test 3D S-curve round trip."""
        batch, d, h, w = 2, 3, 4, 5
        spatial = torch.randn(batch, d, h, w)

        for order in ["D", "H", "W"]:
            sequence = SCurve.spatial_to_sequence_3d(spatial, order=order)
            recovered = SCurve.sequence_to_spatial_3d(sequence, d, h, w, order=order)
            assert torch.allclose(spatial, recovered), f"Failed for order={order}"


class TestHilbertCurve:
    """Tests for Hilbert space-filling curve."""

    def test_2d_curve_generation(self):
        """Test Hilbert curve generates valid indices."""
        curve = HilbertCurve._hilbert_curve_2d(2)

        assert curve.shape == (4, 4)
        # All indices from 0 to 15 should appear exactly once
        assert sorted(curve.flatten().tolist()) == list(range(16))

    def test_2d_round_trip_power_of_2(self):
        """Test Hilbert curve round trip with power-of-2 dimensions."""
        batch, h, w = 2, 4, 4
        spatial = torch.randn(batch, h, w)

        sequence, metadata = HilbertCurve.spatial_to_sequence_2d(spatial)
        recovered = HilbertCurve.sequence_to_spatial_2d(sequence, metadata)

        assert sequence.shape == (batch, h * w)
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_2d_round_trip_non_power_of_2(self):
        """Test Hilbert curve with non-power-of-2 dimensions (padding)."""
        batch, h, w = 2, 5, 6
        spatial = torch.randn(batch, h, w)

        sequence, metadata = HilbertCurve.spatial_to_sequence_2d(spatial)
        recovered = HilbertCurve.sequence_to_spatial_2d(sequence, metadata)

        assert sequence.shape == (batch, h * w)
        assert recovered.shape == spatial.shape
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_2d_with_channels(self):
        """Test Hilbert curve with channels."""
        batch, channels, h, w = 2, 3, 4, 4
        spatial = torch.randn(batch, channels, h, w)

        sequence, metadata = HilbertCurve.spatial_to_sequence_2d(spatial)
        recovered = HilbertCurve.sequence_to_spatial_2d(sequence, metadata)

        assert sequence.shape == (batch, channels, h * w)
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_3d_curve_generation(self):
        """Test 3D Hilbert curve generates valid indices."""
        curve = HilbertCurve._hilbert_curve_3d(2)

        assert curve.shape == (4, 4, 4)
        # All indices from 0 to 63 should appear exactly once
        assert sorted(curve.flatten().tolist()) == list(range(64))

    def test_3d_round_trip(self):
        """Test 3D Hilbert curve round trip."""
        batch, d, h, w = 2, 4, 4, 4
        spatial = torch.randn(batch, d, h, w)

        sequence, metadata = HilbertCurve.spatial_to_sequence_3d(spatial)
        recovered = HilbertCurve.sequence_to_spatial_3d(sequence, metadata)

        assert recovered.shape == spatial.shape
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_locality_preservation(self):
        """Test that Hilbert curve preserves spatial locality."""
        # In a Hilbert curve, adjacent sequence positions should be spatially close
        curve = HilbertCurve._hilbert_curve_2d(3)  # 8x8

        # Find coordinates of each sequence index
        coords = {}
        for y in range(8):
            for x in range(8):
                coords[curve[y, x]] = (y, x)

        # Check that consecutive indices are adjacent (Manhattan distance <= 1)
        for i in range(63):
            y1, x1 = coords[i]
            y2, x2 = coords[i + 1]
            manhattan = abs(y2 - y1) + abs(x2 - x1)
            assert manhattan == 1, f"Indices {i} and {i + 1} are not adjacent"


class TestZOrderCurve:
    """Tests for Z-order (Morton) curve."""

    def test_2d_curve_pattern(self):
        """Test Z-order curve produces expected Z pattern."""
        curve = ZOrderCurve._z_order_curve_2d(2)

        # Expected 4x4 Z-order pattern
        expected = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        assert np.array_equal(curve, expected)

    def test_2d_round_trip(self):
        """Test Z-order round trip preserves data."""
        batch, h, w = 2, 4, 4
        spatial = torch.randn(batch, h, w)

        sequence, metadata = ZOrderCurve.spatial_to_sequence_2d(spatial)
        recovered = ZOrderCurve.sequence_to_spatial_2d(sequence, metadata)

        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_2d_non_power_of_2(self):
        """Test Z-order with non-power-of-2 dimensions."""
        batch, h, w = 2, 5, 6
        spatial = torch.randn(batch, h, w)

        sequence, metadata = ZOrderCurve.spatial_to_sequence_2d(spatial)
        recovered = ZOrderCurve.sequence_to_spatial_2d(sequence, metadata)

        assert recovered.shape == spatial.shape
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_3d_round_trip(self):
        """Test 3D Z-order round trip."""
        batch, d, h, w = 2, 4, 4, 4
        spatial = torch.randn(batch, d, h, w)

        sequence, metadata = ZOrderCurve.spatial_to_sequence_3d(spatial)
        recovered = ZOrderCurve.sequence_to_spatial_3d(sequence, metadata)

        assert recovered.shape == spatial.shape
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_3d_with_channels(self):
        """Test 3D Z-order with channels."""
        batch, channels, d, h, w = 2, 3, 4, 4, 4
        spatial = torch.randn(batch, channels, d, h, w)

        sequence, metadata = ZOrderCurve.spatial_to_sequence_3d(spatial)
        recovered = ZOrderCurve.sequence_to_spatial_3d(sequence, metadata)

        assert torch.allclose(spatial, recovered, atol=1e-6)


class TestConvenienceFunctions:
    """Tests for rasterize_2d and unrasterize_2d convenience functions."""

    @pytest.mark.parametrize("method", ["raster", "scurve"])
    def test_simple_methods_round_trip(self, method):
        """Test raster and scurve methods (no metadata needed)."""
        batch, h, w = 2, 4, 5
        spatial = torch.randn(batch, h, w)

        sequence, metadata = rasterize_2d(spatial, method=method)
        assert metadata is None

        recovered = unrasterize_2d(sequence, h, w, method=method)
        assert torch.allclose(spatial, recovered)

    @pytest.mark.parametrize("method", ["hilbert", "zorder"])
    def test_curve_methods_round_trip(self, method):
        """Test hilbert and zorder methods (require metadata)."""
        batch, h, w = 2, 4, 4
        spatial = torch.randn(batch, h, w)

        sequence, metadata = rasterize_2d(spatial, method=method)
        assert metadata is not None

        recovered = unrasterize_2d(sequence, h, w, method=method, metadata=metadata)
        assert torch.allclose(spatial, recovered, atol=1e-6)

    def test_curve_methods_require_metadata(self):
        """Test that curve methods raise error without metadata."""
        sequence = torch.randn(2, 16)

        with pytest.raises(ValueError, match="metadata required"):
            unrasterize_2d(sequence, 4, 4, method="hilbert", metadata=None)

        with pytest.raises(ValueError, match="metadata required"):
            unrasterize_2d(sequence, 4, 4, method="zorder", metadata=None)

    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        spatial = torch.randn(2, 4, 4)

        with pytest.raises(ValueError, match="Unknown method"):
            rasterize_2d(spatial, method="invalid")


class TestCaching:
    """Test that curve generation is properly cached."""

    def test_hilbert_curve_cached(self):
        """Test Hilbert curve is cached and reused."""
        # Clear cache
        HilbertCurve._hilbert_curve_2d.cache_clear()

        # First call should compute
        curve1 = HilbertCurve._hilbert_curve_2d(3)
        info = HilbertCurve._hilbert_curve_2d.cache_info()
        assert info.misses == 1

        # Second call should hit cache
        curve2 = HilbertCurve._hilbert_curve_2d(3)
        info = HilbertCurve._hilbert_curve_2d.cache_info()
        assert info.hits == 1

        assert np.array_equal(curve1, curve2)

    def test_zorder_curve_cached(self):
        """Test Z-order curve is cached and reused."""
        ZOrderCurve._z_order_curve_2d.cache_clear()

        ZOrderCurve._z_order_curve_2d(3)
        info = ZOrderCurve._z_order_curve_2d.cache_info()
        assert info.misses == 1

        ZOrderCurve._z_order_curve_2d(3)
        info = ZOrderCurve._z_order_curve_2d.cache_info()
        assert info.hits == 1
