Configs API
===========

Model-size presets and helpers shared across every architecture. Two registries
are provided:

* :data:`medlatents.configs.MODEL_CONFIGS` -- a dictionary keyed by lowercase
  size names (``"nano"``, ``"small"``, ``"base"``, ``"large"``, ``"xl"``) whose
  values are plain ``dict`` objects with ``hidden_size``, ``depth``, and
  ``num_heads``. This is the registry used throughout the examples and is the
  most convenient to splat into model constructors.
* :data:`medlatents.configs.MODEL_SIZES` -- a dictionary keyed by capitalized
  short names (``"Nano"``, ``"S"``, ``"B"``, ``"L"``, ``"XL"``) whose values are
  :class:`~medlatents.configs.ModelSize` dataclasses. The two registries are
  numerically consistent.

.. code-block:: python

   from medlatents import AutoregressiveTransformer
   from medlatents.configs import MODEL_CONFIGS

   cfg = MODEL_CONFIGS["nano"]
   model = AutoregressiveTransformer(
       seq_length=256,
       vocab_size=512,
       hidden_size=cfg["hidden_size"],
       depth=cfg["depth"],
       num_heads=cfg["num_heads"],
   )

Registries
----------

.. py:data:: medlatents.configs.MODEL_CONFIGS

   ``dict[str, dict[str, int]]`` keyed by ``nano``/``small``/``base``/``large``/``xl``;
   each value provides ``hidden_size``, ``depth`` and ``num_heads``.

.. py:data:: medlatents.configs.MODEL_SIZES

   ``dict[str, ModelSize]`` keyed by ``Nano``/``S``/``B``/``L``/``XL`` (dataclass view,
   numerically consistent with ``MODEL_CONFIGS``).

.. py:data:: medlatents.configs.MODEL_TYPES

   List of supported ``model_type`` strings.

Data Classes and Helpers
------------------------

.. autoclass:: medlatents.configs.ModelSize
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.configs.create_model_variants
