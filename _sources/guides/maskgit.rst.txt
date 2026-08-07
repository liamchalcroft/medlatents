MaskGIT
=======

MaskGIT (Masked Generative Image Transformer) is a bidirectional masked transformer for discrete latent generation.

**Key Features:**

* Bidirectional attention (parallel decoding)
* Confidence-based masked prediction
* Halton scheduler for spatially-dispersed unmasking
* KLASS early stopping based on KL divergence
* Token critic for quality-guided generation

Architecture
------------

.. mermaid::

    graph LR
        A[Input: Masked Tokens] --> B[Token Embedding]
        B --> C[Positional Encoding]
        C --> D[Transformer Layers]
        D --> D1[Self-Attention Bidirectional]
        D --> D2[Feed-Forward Network]
        D --> D3[LayerNorm]
        D --> E[Output Layer]
        E --> F[Predictions for Unmasked]

        D1 --> style D1 fill:#ff9999
        D2 --> style D2 fill:#99ff99
        D3 --> style D3 fill:#9999ff

Basic Usage
-----------

Training
~~~~~~~~~

.. code-block:: python

    from medlatents import MaskGIT
    from medlatents.configs import MODEL_CONFIGS
    import torch

    config = MODEL_CONFIGS["nano"]
    model = MaskGIT(
        seq_length=256,
        vocab_size=512,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        gradient_checkpointing=True,
    )

    # Training uses masked language modeling
    # (See tutorials/01_training.ipynb)

Generation
~~~~~~~~~~

.. code-block:: python

    from medlatents.sampling import MaskGITScheduler

    # Initialize scheduler
    scheduler = MaskGITScheduler(num_steps=12, mask_schedule='cosine')

    # Start from fully masked input
    tokens = torch.full((1, 256), model.mask_token, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)

    # Iterative unmasking
    for step in range(scheduler.num_steps):
        logits = model(tokens)

        tokens, mask = scheduler.step(logits, tokens, mask, step)

    print(f"Generated tokens: {tokens.shape}")

Advanced Features
-----------------

Halton Scheduler (Spatially-Dispersed Unmasking)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-discrepancy sequence for better spatial locality:

.. code-block:: python

    from medlatents.sampling.maskgit import halton_schedule_1d

    # Use Halton sequence for spatially-dispersed unmasking
    scheduler = MaskGITScheduler(
        num_steps=16,
        mask_schedule="halton",  # Low-discrepancy spatial dispersion
    )

KLASS Early Stopping
~~~~~~~~~~~~~~~~~~~~~~

Adaptive stopping based on KL divergence:

.. code-block:: python

    from medlatents.sampling.maskgit import KLASSGenerator

    klass = KLASSGenerator(
        scheduler=scheduler,
        kl_threshold=0.01,  # Stop when KL falls below threshold
        min_steps=8,          # Minimum steps before early stopping
    )

    generated, metrics = klass.generate(model, initial_tokens=tokens, mask_token=model.mask_token)
    print(f"Steps taken: {metrics['steps_taken']}")

Token Critic
~~~~~~~~~~~~

Quality-guided generation with binary critic:

.. code-block:: python

    from medlatents.sampling.maskgit import TokenCritic, CriticGuidedGenerator

    critic = TokenCritic(vocab_size=512, hidden_dim=256)
    guided_gen = CriticGuidedGenerator(
        model=model,
        critic=critic,
        num_refinement=2,
    )

    generated = guided_gen.generate(
        batch_size=4,
        num_steps=12,
        use_best_of_n=True,  # Best-of-N sampling
    )

Masking Strategies
------------------

Confidence-Based Masking
~~~~~~~~~~~~~~~~~~~~~~~~

Unmask lowest-confidence tokens first:

.. code-block:: python

    from medlatents.sampling.maskgit import adaptive_masking

    # Compute token confidences
    confidences = torch.softmax(logits, dim=-1)

    # Mask lowest-confidence tokens
    tokens_to_mask = adaptive_masking(
        confidences=confidences,
        num_to_mask=64,
    )

Random Masking
~~~~~~~~~~~~~~

Random uniform masking:

.. code-block:: python

    scheduler = MaskGITScheduler(
        num_steps=12,
        mask_schedule='cosine',
    )

Curriculum Masking
~~~~~~~~~~~~~~~~~~

Gradually reduce mask ratio during training:

.. code-block:: python

    from medlatents.training import MaskingSchedule

    # Curriculum: 90% -> 50% -> 10% mask over training
    masking_schedule = MaskingSchedule(
        start_ratio=0.9,
        end_ratio=0.1,
        total_epochs=100,
    )

API Reference
-------------

Classes
~~~~~~~~~

.. autoclass:: medlatents.MaskGIT
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: medlatents.sampling.MaskGITScheduler
   :members:
   :no-index:

.. autoclass:: medlatents.sampling.maskgit.KLASSGenerator
   :members:
   :no-index:

.. autoclass:: medlatents.sampling.maskgit.TokenCritic
   :members:

Functions
~~~~~~~~~

.. autofunction:: medlatents.sampling.maskgit.halton_schedule_1d
.. autofunction:: medlatents.sampling.maskgit.adaptive_masking

See Also
---------

* `Autoregressive Guide <autoregressive.html>`__ - Unidirectional generation
* `Diffusion Guide <diffusion.html>`__ - Diffusion-based generation
* `Sampling Module <../api/sampling.html>`__ - Complete sampling utilities
