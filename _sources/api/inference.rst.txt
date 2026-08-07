Inference API
=============

Conditional-inference utilities for discrete latent models: inpainting,
super-resolution, masking helpers, large-volume tiling, and deployment-oriented
optimizations (KV-cache, quantization, pruning, ONNX export).

Masking and Spatial Utilities
-----------------------------

.. autofunction:: medlatents.inference.create_spatial_mask

.. autofunction:: medlatents.inference.spatial_to_sequence_mask

.. autofunction:: medlatents.inference.apply_token_mask

.. autofunction:: medlatents.inference.confidence_mask

.. autofunction:: medlatents.inference.interpolate_spatial_upsampling

.. autofunction:: medlatents.inference.reslice_volume

Inpainting
----------

End-to-end inpainting plus per-architecture entry points.

.. autofunction:: medlatents.inference.inpaint_volume

.. autofunction:: medlatents.inference.inpaint_autoregressive

.. autofunction:: medlatents.inference.inpaint_maskgit

.. autofunction:: medlatents.inference.inpaint_flow_matching

.. autofunction:: medlatents.inference.inpaint_diffusion_repaint

.. autofunction:: medlatents.inference.inpaint_bayesian_flow

Super-Resolution
----------------

.. autofunction:: medlatents.inference.super_resolve_slices

.. autofunction:: medlatents.inference.anisotropic_super_resolution

.. autofunction:: medlatents.inference.progressive_super_resolution

.. autofunction:: medlatents.inference.compare_with_interpolation

Large-Volume Processing
-----------------------

.. autofunction:: medlatents.inference.encode_large_volume

.. autofunction:: medlatents.inference.decode_large_volume

.. autofunction:: medlatents.inference.reconstruct_large_volume

KV-Cache
--------

.. autoclass:: medlatents.inference.PagedKVCache
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.inference.QuantizedKVCache
   :members:
   :undoc-members:
   :show-inheritance:

Quantization and Pruning
------------------------

.. autoclass:: medlatents.inference.ModelQuantizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.inference.QuantizationConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.inference.ModelPruner
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.inference.PruningConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.inference.prune_attention_heads

.. autofunction:: medlatents.inference.export_to_onnx

.. autofunction:: medlatents.inference.estimate_model_size
