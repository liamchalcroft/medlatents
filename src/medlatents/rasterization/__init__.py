"""Spatial-to-sequence rasterization for autoregressive modeling.

Convert 2D/3D spatial data (images, volumes) to 1D sequences while preserving
spatial locality using space-filling curves and optimized scan patterns.

Available methods:
- RasterScan: Simple row-by-row or slice-by-slice scanning
- SCurve: Serpentine/boustrophedon scan pattern
- HilbertCurve: Space-filling curve maintaining strong spatial locality
- ZOrderCurve: Morton/Z-order quad-tree traversal
"""

from .hilbert_curve import HilbertCurve
from .raster_scan import RasterScan
from .s_curve import SCurve
from .zorder_curve import ZOrderCurve, rasterize_2d, unrasterize_2d

__all__ = [
    "RasterScan",
    "SCurve",
    "HilbertCurve",
    "ZOrderCurve",
    "rasterize_2d",
    "unrasterize_2d",
]
