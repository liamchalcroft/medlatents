Tutorial 3: Inpainting
======================

Inpainting fills missing or corrupted regions of a medical image while keeping
the known region fixed. MedLatents supports inpainting for every discrete
architecture and provides both an end-to-end helper and per-architecture entry
points. The relevant API lives in :doc:`../api/inference`.

The workflow
------------

1. Encode the known image to latent tokens.
2. Build a mask marking which tokens are unknown.
3. Resample only the masked tokens, conditioned on the known ones.
4. Decode back to image space.

The :func:`~medlatents.inference.inpaint_volume` helper bundles all four steps;
the other :mod:`medlatents.inference` functions expose them individually for
token-level control.

Creating a mask
---------------

Use :func:`~medlatents.inference.create_spatial_mask` to mark the region to
fill. Masks are defined in image space and converted to token space internally.

.. code-block:: python

   from medlatents.inference import create_spatial_mask

   # Block region in a single 2D slice
   mask = create_spatial_mask(
       shape=(1, 256, 256),
       mask_type="block",
       start=(64, 64),
       end=(192, 192),
   )

End-to-end inpainting
---------------------

With a loaded generator, inpaint a masked region in one call using
:func:`~medlatents.inference.inpaint_volume`, which encodes, resamples the
masked tokens, and decodes for you:

.. code-block:: python

   from medlatents.generation import DiscreteLatentGenerator
   from medlatents.inference import inpaint_volume

   generator = DiscreteLatentGenerator.from_checkpoints(
       model_type="maskgit",
       model_path="checkpoints/model.pt",
       tokenizer_path="tokenizer.pt",
       device="cuda",
   )

   inpainted = inpaint_volume(
       model=generator.model,
       tokenizer=generator.tokenizer,
       volume=corrupted_image,
       mask=mask,
       model_type="maskgit",
       num_steps=12,
   )

Per-architecture entry points
-----------------------------

When you are working directly with a model and tokens, choose the function that
matches your architecture:

* :func:`~medlatents.inference.inpaint_maskgit` -- confidence-guided parallel
  refilling for MaskGIT.
* :func:`~medlatents.inference.inpaint_autoregressive` -- causal refilling for
  autoregressive models.
* :func:`~medlatents.inference.inpaint_flow_matching` -- flow-based inpainting.
* :func:`~medlatents.inference.inpaint_diffusion_repaint` -- RePaint-style
  resampling for D3PM.
* :func:`~medlatents.inference.inpaint_bayesian_flow` -- inpainting for Bayesian
  flow models.

For 3D data, :func:`~medlatents.inference.inpaint_volume` orchestrates masking
and decoding over a whole volume.

Super-resolution
----------------

The same conditional-inference machinery upscales low-resolution scans.
:func:`~medlatents.inference.super_resolve_slices` and
:func:`~medlatents.inference.anisotropic_super_resolution` are the main entry
points, with :func:`~medlatents.inference.compare_with_interpolation` for a
baseline comparison.

.. code-block:: python

   from medlatents.inference import anisotropic_super_resolution

   # target_shape is the absolute (D, H, W) to upscale to; here we 4x the
   # through-plane axis of a (1, 1, 64, 256, 256) volume and keep the rest.
   high_res = anisotropic_super_resolution(
       model=generator.model,
       tokenizer=generator.tokenizer,
       volume=low_res_volume,
       target_shape=(256, 256, 256),
   )

Working with large volumes
--------------------------

For volumes that do not fit in memory, tile encode/decode with
:func:`~medlatents.inference.encode_large_volume`,
:func:`~medlatents.inference.decode_large_volume`, and
:func:`~medlatents.inference.reconstruct_large_volume`.

Preserving metadata
-------------------

Inpainting and super-resolution operate on latents, but downstream
reconstruction fidelity depends on orientation and spacing. Carry the
``affine`` / ``spacing`` metadata attached at load time through your pipeline --
see :doc:`../token_interface` for the metadata contract.

Next steps
----------

* :doc:`04_evaluation` -- quantify reconstruction and generation quality.
