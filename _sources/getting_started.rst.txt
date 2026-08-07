Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python >= 3.10
* PyTorch >= 2.0
* CUDA (optional, for GPU acceleration)

Install from PyPI
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install medlatents

This pulls in ``medtokenizers`` and ``medrs`` for tokenization and I/O.

Install from Source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/liamchalcroft/medlatents.git
   cd medlatents
   pip install -e .

Install with Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[dev]"

Install with Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[docs]"

Quick Start
-----------

Train Your First Model
~~~~~~~~~~~~~~~~~~~~~~

The snippet below trains a tiny model with a plain next-token loop -- a complete,
copy-pasteable starting point. For the full, runnable version see
``examples/quickstart.py``; for real workloads on tokenized medical imagery use
:class:`~medlatents.training.DiscreteLatentTrainer` with dataloaders (see
``scripts/train.py`` and :doc:`tutorials/01_training`).

.. code-block:: python

   import torch
   import torch.nn.functional as F
   from medlatents import AutoregressiveTransformer
   from medlatents.configs import MODEL_CONFIGS

   # Create a "nano" model from the shared size presets
   config = MODEL_CONFIGS["nano"]
   model = AutoregressiveTransformer(
       seq_length=256,
       vocab_size=512,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
   )

   # Synthetic token sequences (replace with your tokenized data)
   data = torch.randint(0, 512, (8, 256))
   optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

   model.train()
   for step in range(20):
       optimizer.zero_grad()
       logits = model(data[:, :-1])           # [B, L-1, vocab]
       loss = F.cross_entropy(
           logits.reshape(-1, logits.size(-1)),
           data[:, 1:].reshape(-1),
       )
       loss.backward()
       optimizer.step()

Generate Samples
~~~~~~~~~~~~~~~~

Autoregressive generation is prompt-driven: pass a starting context and the
target length.

.. code-block:: python

   model.eval()
   prompt = torch.zeros((4, 1), dtype=torch.long)   # [batch, prompt_len]
   samples = model.generate(prompt, max_length=256, temperature=1.0, top_k=50)

   print(f"Generated samples shape: {samples.shape}")
   # Output: Generated samples shape: torch.Size([4, 256])

MaskGIT Training
~~~~~~~~~~~~~~~~

.. code-block:: python

   from medlatents import MaskGIT
   from medlatents.sampling import MaskGITScheduler

   # Create MaskGIT model
   model = MaskGIT(
       seq_length=256,
       vocab_size=512,
       hidden_size=192,
       depth=8,
       num_heads=3,
   )

   # Create scheduler for masked training
   scheduler = MaskGITScheduler(num_steps=12)

   # Train with masked language modeling objective
   # (See tutorials/01_training for the full example)

Inpainting
~~~~~~~~~~

.. code-block:: python

   from medlatents.inference import create_spatial_mask, inpaint_volume
   from medlatents.generation import DiscreteLatentGenerator

   # Load generator (exposes the loaded .model and .tokenizer)
   generator = DiscreteLatentGenerator.from_checkpoints(
       model_type="maskgit",
       model_path="checkpoints/model.pt",
       tokenizer_path="tokenizer.pt",
       device="cuda",
   )

   # Create inpainting mask (block region); True keeps the known region,
   # False marks voxels to inpaint.
   mask = create_spatial_mask(
       shape=(1, 256, 256),
       mask_type="block",
       start=(64, 64),
       end=(192, 192),
   )

   # Inpaint the masked region end-to-end (volume -> tokens -> inpaint -> volume)
   inpainted = inpaint_volume(
       model=generator.model,
       tokenizer=generator.tokenizer,
       volume=corrupted_image,
       mask=mask,
       model_type="maskgit",
       num_steps=12,
   )

Next Steps
----------

* **Tutorials**: Learn by doing with our step-by-step guides/tutorials
* **User Guides**: Dive deeper into specific model types
* **API Reference**: Detailed documentation for all modules
* **Research**: Learn about the implemented methods

Examples
---------

See the ``scripts/`` directory for complete examples:

* ``scripts/train.py`` - Unified training script for all model types
* ``scripts/generate.py`` - Generate samples from trained models
* ``scripts/inpaint.py`` - Inpaint missing regions
* ``scripts/evaluate_model.py`` - Evaluate model quality

``scripts/train.py`` requires a discrete tokenizer checkpoint (``--tokenizer_path``) for on-the-fly tokenization.

Citation
---------

If you use MedLatents in your research, please cite:

.. code-block:: bibtex

   @software{medlatents2026,
     title={MedLatents: Discrete and Continuous Latent Generative Models for Medical Imagery},
     author={Chalcroft, Liam},
     year={2026},
     url={https://github.com/liamchalcroft/medlatents}
   }

Support
-------

* **GitHub Issues**: Report bugs and request features
* **Documentation**: https://liamchalcroft.github.io/medlatents
