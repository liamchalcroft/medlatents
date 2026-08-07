Rasterization API
=================

This module provides spatial-to-sequence rasterization for autoregressive modeling
of 2D/3D spatial data.

Rasterization Methods
---------------------

RasterScan
^^^^^^^^^^

Simple row-by-row or slice-by-slice scanning.

.. autoclass:: medlatents.rasterization.RasterScan
   :members:
   :undoc-members:
   :show-inheritance:

SCurve
^^^^^^

Serpentine/boustrophedon scan pattern with alternating row directions.

.. autoclass:: medlatents.rasterization.SCurve
   :members:
   :undoc-members:
   :show-inheritance:

HilbertCurve
^^^^^^^^^^^^

Space-filling curve with optimal locality preservation.

.. autoclass:: medlatents.rasterization.HilbertCurve
   :members:
   :undoc-members:
   :show-inheritance:

ZOrderCurve
^^^^^^^^^^^

Morton/Z-order curve using bit interleaving.

.. autoclass:: medlatents.rasterization.ZOrderCurve
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
---------------------

.. autofunction:: medlatents.rasterization.rasterize_2d

.. autofunction:: medlatents.rasterization.unrasterize_2d
