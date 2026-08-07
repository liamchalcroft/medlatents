Conditioning API
================

Unified conditioning infrastructure shared by every architecture. A
:class:`~medlatents.conditioning.ConditioningBundle` is the standardized
container for all conditioning inputs (class labels, text/image embeddings,
spatial conditions), and the frozen-encoder wrappers turn pretrained vision
backbones into conditioning feature extractors.

Conditioning Bundle
-------------------

.. autoclass:: medlatents.conditioning.ConditioningBundle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.conditioning.ConditioningConfig
   :members:
   :undoc-members:
   :show-inheritance:

Frozen Encoders
---------------

.. autoclass:: medlatents.conditioning.FrozenEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.conditioning.FrozenDINOv2
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.conditioning.FrozenSigLIP
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.conditioning.FrozenMedSigLIP
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.conditioning.FrozenNeuroVFM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.conditioning.SliceWiseEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.conditioning.create_encoder

Data Pipeline
-------------

.. autoclass:: medlatents.conditioning.ConditionalBatchConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.conditioning.collate_conditional_batch
