Generation API
==============

High-level, checkpoint-driven generation for discrete and continuous latent
models. These classes wrap a trained model together with its tokenizer so you
can go from a checkpoint to decoded images in a few lines. For low-level
sampling primitives (schedulers, guidance, early stopping), see
:doc:`sampling`.

Discrete Latent Generation
--------------------------

.. autoclass:: medlatents.generation.DiscreteLatentGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Continuous Latent Generation
----------------------------

.. autoclass:: medlatents.generation.ContinuousLatentGenerator
   :members:
   :undoc-members:
   :show-inheritance:
