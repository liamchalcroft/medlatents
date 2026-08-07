Rasterization Guide
===================

This guide covers spatial-to-sequence rasterization in medlatents, enabling
autoregressive modeling of 2D/3D spatial data like medical images and volumes.

Overview
--------

Autoregressive models process sequences, but medical data is spatial (2D/3D).
Rasterization converts spatial data to sequences while preserving locality:

.. mermaid::

   flowchart LR
       subgraph Spatial["Spatial Data"]
           I[2D Image / 3D Volume]
       end

       subgraph Rasterize["Rasterization"]
           I --> R[Space-Filling Curve]
       end

       subgraph Sequence["Sequence"]
           R --> S[1D Token Sequence]
       end

       subgraph Generate["Generation"]
           S --> AR[Autoregressive Model]
           AR --> S2[Generated Sequence]
       end

       subgraph Unraster["Unrasterization"]
           S2 --> I2[Reconstructed Spatial]
       end

Why Locality Matters
^^^^^^^^^^^^^^^^^^^^

Adjacent tokens should represent nearby spatial locations:

- **Better context**: Autoregressive models use previous tokens as context
- **Faster learning**: Local patterns are easier to learn
- **Inpainting**: Masked regions should have nearby context

Available Methods
-----------------

RasterScan
^^^^^^^^^^

Simple row-by-row (or slice-by-slice) scanning.

.. code-block:: python

   from medlatents.rasterization import RasterScan

   # For 2D images
   rasterizer = RasterScan(
       height=64,
       width=64,
       order='row',  # Options: row, column
   )

   # For 3D volumes
   rasterizer = RasterScan(
       depth=32,
       height=64,
       width=64,
       order='slice',  # Options: slice, row, column
   )

   # Convert image to sequence
   image = torch.randn(1, 64, 64)  # [C, H, W]
   sequence = rasterizer.rasterize(image)  # [1, H*W]

   # Convert back
   reconstructed = rasterizer.unrasterize(sequence)  # [C, H, W]

**Properties**:

- [+] Simple and fast
- [+] No computation overhead
- [-] Poor locality at row/column boundaries

SCurve (Serpentine)
^^^^^^^^^^^^^^^^^^^

Boustrophedon pattern - alternating left-right, right-left rows.

.. code-block:: python

   from medlatents.rasterization import SCurve

   rasterizer = SCurve(
       height=64,
       width=64,
   )

   sequence = rasterizer.rasterize(image)

**Properties**:

- [+] Simple
- [+] Better than raster at row boundaries
- [-] Still has locality issues at row ends

HilbertCurve
^^^^^^^^^^^^

Space-filling curve with optimal locality preservation.

.. code-block:: python

   from medlatents.rasterization import HilbertCurve

   # Dimensions should be powers of 2
   rasterizer = HilbertCurve(
       height=64,  # Must be power of 2
       width=64,   # Must be power of 2
   )

   sequence = rasterizer.rasterize(image)

   # 3D Hilbert curve
   rasterizer_3d = HilbertCurve(
       depth=32,
       height=64,
       width=64,
   )

**Properties**:

- [+] Best locality (adjacent sequence = adjacent spatial)
- [+] Self-similar structure
- [!] Requires power-of-2 dimensions
- [!] More computation to generate curve

ZOrderCurve (Morton)
^^^^^^^^^^^^^^^^^^^^

Z-order (Morton) curve using bit interleaving.

.. code-block:: python

   from medlatents.rasterization import ZOrderCurve, rasterize_2d, unrasterize_2d

   rasterizer = ZOrderCurve(
       height=64,
       width=64,
   )

   sequence = rasterizer.rasterize(image)

   # Quick functions for common cases
   sequence = rasterize_2d(image, method='zorder')
   image = unrasterize_2d(sequence, height=64, width=64, method='zorder')

**Properties**:

- [+] Good locality (better than raster/S-curve)
- [+] Fast (bit operations)
- [+] Works with non-power-of-2 (with padding)
- [!] Slightly worse locality than Hilbert

Comparison
----------

Visual Comparison
^^^^^^^^^^^^^^^^^

.. code-block:: text

   Raster Scan:          S-Curve:             Hilbert:             Z-Order:
   → → → → ↓            → → → → ↓            ┌─┐ ┌─┐              ↘ ↓ ↘ ↓
   → → → → ↓            ← ← ← ← ↓            │ └─┘ │              → ↘ → ↘
   → → → → ↓            → → → → ↓            └─┐ ┌─┘              ↘ ↓ ↘ ↓
   → → → → ↓            ← ← ← ← ↓              └─┘                → ↘ → ↘

Locality Metrics
^^^^^^^^^^^^^^^^

+---------------+------------------+------------------+------------------+
| Method        | Avg Distance     | Max Distance     | Computation      |
+===============+==================+==================+==================+
| Raster        | O(W)             | O(W)             | O(1)             |
+---------------+------------------+------------------+------------------+
| S-Curve       | O(1) / O(W)      | O(W)             | O(1)             |
+---------------+------------------+------------------+------------------+
| Hilbert       | O(1)             | O(√N)            | O(N)             |
+---------------+------------------+------------------+------------------+
| Z-Order       | O(1) / O(log N)  | O(√N)            | O(N)             |
+---------------+------------------+------------------+------------------+

Practical Usage
---------------

With Autoregressive Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.rasterization import HilbertCurve
   from medlatents.autoregressive import AutoregressiveTransformer

   # Setup
   rasterizer = HilbertCurve(height=64, width=64)
   model = AutoregressiveTransformer(
       vocab_size=8192,
       max_seq_len=64 * 64,  # H * W
   )

   # Training
   for batch in dataloader:
       tokens = batch['tokens']  # [B, C, H, W] image tokens

       # Rasterize each channel
       sequences = []
       for c in range(tokens.shape[1]):
           seq = rasterizer.rasterize(tokens[:, c])  # [B, H*W]
           sequences.append(seq)
       sequence = torch.cat(sequences, dim=1)  # [B, C*H*W]

       # Train autoregressive model
       loss = model.compute_loss(sequence)

   # Generation
   generated_seq = model.generate(max_length=64 * 64)
   generated_image = rasterizer.unrasterize(generated_seq)

With Tokenizers
^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.data import VQTokenizer
   from medlatents.rasterization import ZOrderCurve

   # Tokenize image to discrete codes
   tokenizer = VQTokenizer.from_pretrained('path/to/vq')
   tokens = tokenizer.encode(image)  # [1, H//p, W//p] where p is patch size

   # Rasterize
   rasterizer = ZOrderCurve(
       height=tokens.shape[1],
       width=tokens.shape[2],
   )
   sequence = rasterizer.rasterize(tokens)

3D Medical Volumes
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.rasterization import HilbertCurve

   # 3D MRI volume
   rasterizer = HilbertCurve(
       depth=128,   # Slices
       height=256,  # Rows
       width=256,   # Columns
   )

   volume = torch.randn(1, 128, 256, 256)  # [C, D, H, W]
   sequence = rasterizer.rasterize(volume)  # [1, D*H*W]

   # Very long sequences! Consider:
   # 1. Downsampling first
   # 2. Using sliding windows
   # 3. Hierarchical generation

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: python

   # All methods support batched operations
   batch = torch.randn(32, 1, 64, 64)  # [B, C, H, W]

   rasterizer = HilbertCurve(64, 64)

   # Rasterize batch
   sequences = rasterizer.rasterize(batch)  # [B, 1, H*W]

   # Unrasterize batch
   images = rasterizer.unrasterize(sequences)  # [B, 1, H, W]

Choosing a Method
-----------------

Decision Guide
^^^^^^^^^^^^^^

.. code-block:: text

   Is dimension power of 2?
   ├── Yes
   │   └── Need best locality?
   │       ├── Yes → HilbertCurve
   │       └── No  → ZOrderCurve (faster)
   └── No
       └── Can pad to power of 2?
           ├── Yes → Hilbert/ZOrder with padding
           └── No  → SCurve (if 2D) or RasterScan

Recommendations by Use Case
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------+------------------+--------------------------------+
| Use Case             | Recommended      | Reason                         |
+======================+==================+================================+
| Medical images (2D)  | HilbertCurve     | Best locality for pathology    |
+----------------------+------------------+--------------------------------+
| Medical volumes (3D) | HilbertCurve     | Preserves 3D structure         |
+----------------------+------------------+--------------------------------+
| Fast prototyping     | RasterScan       | Simple, no overhead            |
+----------------------+------------------+--------------------------------+
| Non-power-of-2       | ZOrderCurve      | Handles any size with padding  |
+----------------------+------------------+--------------------------------+
| Sliding windows      | RasterScan       | Predictable boundaries         |
+----------------------+------------------+--------------------------------+

Handling Non-Power-of-2 Dimensions
----------------------------------

Padding Strategy
^^^^^^^^^^^^^^^^

.. code-block:: python

   import math

   def next_power_of_2(n):
       return 2 ** math.ceil(math.log2(n))

   # Original size
   H, W = 100, 150

   # Pad to power of 2
   H_padded = next_power_of_2(H)  # 128
   W_padded = next_power_of_2(W)  # 256

   # Pad image
   image_padded = F.pad(image, (0, W_padded - W, 0, H_padded - H))

   # Rasterize
   rasterizer = HilbertCurve(H_padded, W_padded)
   sequence = rasterizer.rasterize(image_padded)

   # After generation, crop back
   generated = rasterizer.unrasterize(generated_seq)
   generated = generated[:, :, :H, :W]

Tile-Based Approach
^^^^^^^^^^^^^^^^^^^

For very large images, process in tiles:

.. code-block:: python

   def process_tiles(image, tile_size=64):
       """Process image in tiles with overlap."""
       B, C, H, W = image.shape
       rasterizer = HilbertCurve(tile_size, tile_size)

       tiles = []
       for i in range(0, H, tile_size):
           for j in range(0, W, tile_size):
               tile = image[:, :, i:i+tile_size, j:j+tile_size]
               tile_seq = rasterizer.rasterize(tile)
               tiles.append(tile_seq)

       return torch.cat(tiles, dim=1)

Performance Tips
----------------

Caching Indices
^^^^^^^^^^^^^^^

For repeated use, cache the rasterization indices:

.. code-block:: python

   rasterizer = HilbertCurve(64, 64)

   # Pre-compute indices (done automatically on first call)
   _ = rasterizer.get_indices()

   # Now rasterization is just indexing (fast)
   for batch in dataloader:
       sequence = rasterizer.rasterize(batch['image'])

GPU Acceleration
^^^^^^^^^^^^^^^^

All operations are GPU-compatible:

.. code-block:: python

   rasterizer = HilbertCurve(64, 64)

   image = image.cuda()
   sequence = rasterizer.rasterize(image)  # Stays on GPU

Compiled Operations
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compile for speed (PyTorch 2.0+)
   rasterize_fn = torch.compile(rasterizer.rasterize)

   sequence = rasterize_fn(image)

API Reference
-------------

.. seealso::

   - :doc:`/api/rasterization` - Full API documentation
   - :doc:`/guides/autoregressive` - Autoregressive models
   - :doc:`/guides/maskgit` - MaskGIT (alternative to AR)
