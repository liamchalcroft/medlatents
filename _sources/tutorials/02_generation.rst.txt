Tutorial 2: Generating Samples
==============================

This tutorial covers generating new samples from a trained model, from the
quick high-level path to fine-grained control over the sampling process.

There are two layers of API:

* **High-level** (:doc:`../api/generation`):
  :class:`~medlatents.generation.DiscreteLatentGenerator` and
  :class:`~medlatents.generation.ContinuousLatentGenerator` wrap a checkpoint
  plus its tokenizer and return decoded images.
* **Low-level** (:doc:`../api/sampling`): schedulers, guidance, and early
  stopping that you compose yourself when you need full control.

High-level generation
---------------------

Load a trained model and its tokenizer together, then generate:

.. code-block:: python

   from medlatents.generation import DiscreteLatentGenerator

   generator = DiscreteLatentGenerator.from_checkpoints(
       model_type="maskgit",
       model_path="checkpoints/model.pt",
       tokenizer_path="tokenizer.pt",
       device="cuda",
   )

   images = generator.generate(num_samples=4)
   print(images.shape)

The generator decodes latents back to image space using the tokenizer, so the
output is ready to save or visualize.

Direct model sampling
---------------------

For autoregressive models you can sample straight from the model. Note that
:meth:`~medlatents.AutoregressiveTransformer.generate` is *prompt-driven*: pass a
starting context (for example a single beginning-of-sequence token) and the
target length.

.. code-block:: python

   import torch
   from medlatents import AutoregressiveTransformer

   model = AutoregressiveTransformer(seq_length=256, vocab_size=512,
                                     hidden_size=192, depth=8, num_heads=3)
   model.eval()

   prompt = torch.zeros((4, 1), dtype=torch.long)   # [batch, prompt_len]
   tokens = model.generate(prompt, max_length=256, temperature=1.0, top_k=50)
   print(tokens.shape)   # torch.Size([4, 256])

Sampling strategies
-------------------

The :mod:`medlatents.sampling` module provides token-level samplers:

.. code-block:: python

   from medlatents.sampling import sample_nucleus, sample_min_p

   # logits: [batch, vocab]
   tokens = sample_nucleus(logits, p=0.9)     # top-p / nucleus
   tokens = sample_min_p(logits, p=0.05)      # min-p sampling

A :class:`~medlatents.sampling.TemperatureScheduler` lets temperature vary over
the course of generation.

MaskGIT and KLASS early stopping
--------------------------------

MaskGIT decodes in parallel over a fixed number of steps. The
:class:`~medlatents.sampling.MaskGITScheduler` controls the unmasking schedule,
and :class:`~medlatents.sampling.KLASSGenerator` stops early once the predicted
distribution stabilizes (KL divergence below a threshold):

.. code-block:: python

   import torch
   from medlatents import MaskGIT
   from medlatents.sampling import MaskGITScheduler, KLASSGenerator

   model = MaskGIT(seq_length=256, vocab_size=512, hidden_size=192,
                   depth=8, num_heads=3)

   scheduler = MaskGITScheduler(num_steps=16)
   klass = KLASSGenerator(scheduler=scheduler, kl_threshold=0.01, min_steps=8)

   x = torch.full((1, 256), model.mask_token)
   generated, metrics = klass.generate(model, initial_tokens=x, mask_token=model.mask_token)
   print(f"Steps taken: {metrics['steps_taken']}")

Classifier-free guidance (CFG)
------------------------------

Conditional models support classifier-free guidance. The simplest path is the
string API on the high-level samplers; for time-dependent schedules use
:class:`~medlatents.sampling.GuidedSampler` with a schedule such as
:func:`~medlatents.sampling.cosine_guidance`:

.. code-block:: python

   from medlatents.sampling import GuidedSampler, cosine_guidance

   sampler = GuidedSampler(model=model, guidance_schedule=cosine_guidance, base_scale=5.0)
   guided_logits = sampler(x, t, conditioning={"class_id": labels}, progress=0.5)

For flow-matching and diffusion CFG specifically, see
:func:`~medlatents.sampling.sample_with_cfg_flow` and
:func:`~medlatents.sampling.sample_with_cfg_diffusion`.

Reproducibility
---------------

Pass a :class:`torch.Generator` to samplers that accept one, and seed it for
deterministic output. See :doc:`../token_interface` for the full determinism
policy.

Next steps
----------

* :doc:`03_inpainting` -- condition generation on existing image content.
* :doc:`04_evaluation` -- measure the quality of generated samples.
