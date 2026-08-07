Token Interface
===============

MedLatents treats tokenization as a **strict interface**. Models consume the
output of a ``medtokenizers`` tokenizer, and any mismatch in dtype, shape, value
range, or finiteness is reported loudly rather than silently coerced. This page
documents the schema, layout, and failure policy so you can prepare data that
the models accept without surprises.

The upstream tokenizers are :class:`medlatents.data.DiscreteTokenizer` and
:class:`medlatents.data.ContinuousTokenizer`, which wrap
``medtokenizers.networks.discrete.DiscreteTokenizer`` and
``medtokenizers.networks.continuous.ContinuousTokenizer`` respectively.

Discrete tokens (codebook indices)
----------------------------------

Source and dtype
^^^^^^^^^^^^^^^^

Discrete tokens are codebook indices produced by a discrete tokenizer. They must
be ``torch.long`` (int64). There is no implicit casting.

Shapes
^^^^^^

The canonical shape for all discrete models is ``[B, L]`` (batch, sequence
length). A dataset may emit ``[L]`` for a single sample; the collate function is
responsible for batching to ``[B, L]``.

Value ranges and special tokens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base codebook indices lie in ``[0, codebook_size)``. Special tokens, when used,
are reserved *directly above* the base vocabulary. The behavior differs by model
family:

**Autoregressive and MaskGIT** reserve four special tokens:

.. code-block:: text

   BOS  = V        EOS = V + 1        PAD = V + 2        MASK = V + 3

where ``V = codebook_size``. These models therefore produce logits over
``V + 4`` classes.

**D3PM and discrete flow** use no BOS/EOS/PAD by default. For an absorbing
transition (``transition_type="absorbing"``) the mask token is ``V`` and the
model output vocabulary is ``V + 1``. Any token ``>= V + 1`` is invalid unless
you explicitly expand the vocabulary and update the model accordingly.

Layout and rasterization
^^^^^^^^^^^^^^^^^^^^^^^^

A tokenizer returns a sequence of codes. If you start from a spatial grid, the
flattening must be explicit. The default flattening is row-major:

.. code-block:: text

   2D:  [C, H, W]     ->  [H * W]
   3D:  [C, D, H, W]  ->  [D * H * W]

For locality-preserving alternatives, use :mod:`medlatents.rasterization`
(Hilbert or Z-order curves) and **record the method you used** so generation and
decoding stay consistent.

Model outputs
^^^^^^^^^^^^^

Discrete models output logits of shape ``[B, L, vocab]`` in a floating dtype,
with finite values only.

Continuous latents
------------------

Source and dtype
^^^^^^^^^^^^^^^^

Continuous latents come from a continuous tokenizer and may be ``float32``,
``float16``, or ``bfloat16``.

Shapes and layout
^^^^^^^^^^^^^^^^^

Continuous latents are channel-first:

.. code-block:: text

   2D:  [B, C, H, W]
   3D:  [B, C, D, H, W]

Sequence models (the continuous DiT) expect ``[B, L, C]``, and the flattening is
explicit:

.. code-block:: text

   2D:  [B, C, H, W]     ->  [B, H * W, C]
   3D:  [B, C, D, H, W]  ->  [B, D * H * W, C]

Scaling and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^

Latents are expected to be approximately ``N(0, 1)`` or to follow a documented
scale. If a tokenizer applies a scaling factor, record it in your experiment
config and keep it consistent across training, sampling, and decoding.

Metadata
--------

When loading NIfTI data through ``medrs``, datasets attach metadata: ``affine``,
``spacing``, ``shape``, and ``file_path``. Models do not use this metadata
directly, but downstream pipelines must preserve it whenever reconstruction
fidelity depends on orientation or spacing.

Batching and variable sizes
---------------------------

* **Discrete sequences** -- use
  :func:`medlatents.training.collate_variable_length` with the ``PAD`` token for
  autoregressive and MaskGIT models. Do not pad D3PM or flow inputs unless
  ``PAD`` is treated as a regular token and the model was trained that way.
* **Continuous latents** -- variable spatial sizes must be normalized upstream
  (crop, resize, or tile). MedLatents does not auto-pad continuous volumes.

Determinism
-----------

* **Training** -- set a seed and enable deterministic ops (the training scripts
  expose ``--seed`` and ``--deterministic``; the latter disables TF32 and cuDNN
  benchmarking).
* **Sampling** -- pass a :class:`torch.Generator` where supported; otherwise set
  ``torch.manual_seed`` and avoid data-dependent nondeterminism.
* **Dataloaders** -- pass ``seed`` to
  :func:`medlatents.data.get_tokenized_dataloaders` or
  :func:`medlatents.data.get_latent_dataloaders` for deterministic worker
  seeding.

Failure-mode policy
-------------------

Validate inputs before handing them to a model. The validation helpers raise a
``TypeError`` or ``ValueError`` with actionable context on any mismatch -- wrong
dtype, wrong shape, out-of-range values, or non-finite values. There is **no**
silent coercion and **no** auto-fixing of user inputs: the goal is to surface
data problems early rather than let them corrupt a long training run.
