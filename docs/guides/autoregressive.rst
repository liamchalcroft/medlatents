Autoregressive Models
=====================

Overview
--------

The autoregressive (AR) module implements causal transformer models for sequential generation of discrete latent tokens.

**Key Features:**

* Causal attention (each token only depends on previous tokens)
* Rotary Position Embeddings (RoPE) for position awareness
* Speculative decoding for 1.5-3x speedup
* KV-cache for memory-efficient autoregressive generation
* Support for long sequences via chunking and gradient checkpointing

Architecture
------------

.. mermaid::

    graph LR
        A[Input Tokens] --> B[Token Embedding]
        B --> C[Positional Encoding]
        C --> D[Transformer Layers]
        D --> E[Output Layer]
        E --> F[Next Token Distribution]

        D --> D1[Self-Attention Causal]
        D --> D2[Feed-Forward Network]
        D --> D3[LayerNorm]

        style D1 fill:#ff9999
        style D2 fill:#99ff99
        style D3 fill:#9999ff

Basic Usage
-----------

Training
~~~~~~~~~

.. code-block:: python

   from medlatents import AutoregressiveTransformer
   from medlatents.configs import MODEL_CONFIGS
   import torch

   # Create model
   config = MODEL_CONFIGS["nano"]
   model = AutoregressiveTransformer(
       seq_length=256,
       vocab_size=512,
       hidden_size=config["hidden_size"],
       depth=config["depth"],
       num_heads=config["num_heads"],
       gradient_checkpointing=True,  # Enable for long sequences
   )

   # Forward pass (for training)
   logits = model(input_tokens)  # [batch, seq_len, vocab_size]
   loss = torch.nn.functional.cross_entropy(
       logits[:, :-1].flatten(),
       input_tokens[:, 1:].flatten()
   )

Generation
~~~~~~~~~~

.. code-block:: python

   # Generate new sequences
   samples = model.generate(
       batch_size=4,
       max_new_tokens=256,
       temperature=1.0,
       top_k=50,
       top_p=0.9,
   )

   # With KV-cache for efficiency
   samples = model.generate_with_cache(
       batch_size=4,
       max_new_tokens=256,
       temperature=1.0,
   )

Advanced Features
-----------------

Speculative Decoding
~~~~~~~~~~~~~~~~~~~~

Accelerate autoregressive generation using a smaller draft model:

.. code-block:: python

   from medlatents.autoregressive import SpeculativeDecoder

   # Create main model and draft model
   main_model = AutoregressiveTransformer(...)
   draft_model = AutoregressiveTransformer(...)  # Smaller model

   # Create speculative decoder
   decoder = SpeculativeDecoder(
       main_model=main_model,
       draft_model=draft_model,
       vocab_size=512,
       max_speculation=4,  # Draft 4 tokens ahead
   )

   # Generate with speedup
   tokens, metrics = decoder.generate(
       initial_tokens=context,
       max_new_tokens=256,
       eos_token=eos_id,
   )

   print(f"Speedup: {metrics['acceptance_rate']:.2%}")

Medusa Heads
~~~~~~~~~~~~~~~

Multi-head speculative decoding with parallel predictions:

.. code-block:: python

   from medlatents.autoregressive.speculative.medusa import Medusa, MedusaTrainer

   # Add Medusa heads to existing model
   medusa_model = Medusa(
       base_model=model,
       num_heads=4,  # Number of prediction heads
   )

   # Train Medusa heads (base model is frozen)
   trainer = MedusaTrainer(medusa_model)
   trainer.train(...)

   # Generate with Medusa
   samples = medusa_model.generate_medusa(...)

KV-Cache Quantization
~~~~~~~~~~~~~~~~~~~~~

Reduce memory usage by quantizing the KV-cache:

.. code-block:: python

   from medlatents.inference import QuantizedKVCache, QuantizationConfig

   # Configure quantization
   qconfig = QuantizationConfig(
       dtype=torch.uint8,  # INT8 = 4x memory reduction
       mode="per_token",
       symmetric=True,
   )

   # Generate with quantized cache
   cache = QuantizedKVCache(config=qconfig)
   samples = model.generate_with_cache(cache=cache, ...)

Sampling Strategies
-------------------

Nucleus Sampling (Top-p)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from medlatents.sampling import sample_nucleus

   logits = model(input_tokens)
   next_token = sample_nucleus(logits[:, -1], p=0.95)

Min-P Sampling
~~~~~~~~~~~~~~

Confidence-based filtering to remove low-probability tokens:

.. code-block:: python

   from medlatents.sampling import sample_min_p

   logits = model(input_tokens)
   next_token = sample_min_p(logits[:, -1], min_p=0.05)

Classifier-Free Guidance (CFG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Guide generation towards specific conditions:

.. code-block:: python

   from medlatents.sampling import sample_with_cfg

   logits_cond = model(input_tokens, condition=class_embedding)
   logits_uncond = model(input_tokens, condition=None)

   # Apply CFG
   guided_logits = sample_with_cfg(
       logits_cond=logits_cond,
       logits_uncond=logits_uncond,
       guidance_scale=2.0,
   )

Temperature Scheduling
~~~~~~~~~~~~~~~~~~~~~~

Vary sampling temperature during generation:

.. code-block:: python

   from medlatents.sampling import TemperatureScheduler

   scheduler = TemperatureScheduler(
       schedule='cosine',
       start_temp=1.0,
       end_temp=0.7,
       num_steps=256,
   )

   for step in range(256):
       temp = scheduler.get_temperature(step)
       next_token = sample(logits, temperature=temp)

Long Sequences
---------------

Chunking
~~~~~~~~~

For sequences longer than the model's context window:

.. code-block:: python

   model = AutoregressiveTransformer(
       seq_length=4096,
       chunk_size=1024,  # Process in 1024-token chunks
       ...
   )

   # Automatically handles long sequences
   outputs = model(very_long_sequence)  # Can be > seq_length

Gradient Checkpointing
~~~~~~~~~~~~~~~~~~~~~~

Trade compute for memory to enable longer sequences:

.. code-block:: python

   model = AutoregressiveTransformer(
       seq_length=16384,
       gradient_checkpointing=True,  # Enabled by default for large models
   )

   # Training with gradient checkpointing
   loss = model(input_tokens)
   loss.backward()  # Uses ~50% less memory

Rasterization
-------------

For medical images, use space-filling curves to preserve spatial locality:

.. code-block:: python

   from medlatents.rasterization import HilbertCurve

   # Convert 3D volume to sequence
   volume = torch.randn(1, 3, 32, 64, 48)
   sequence, metadata = HilbertCurve.spatial_to_sequence_3d(volume)

   # Train autoregressive model
   model = AutoregressiveTransformer(seq_length=sequence.shape[-1], ...)
   model.train(sequence[:, :-1], sequence[:, 1:])

   # Generate and reconstruct volume
   generated_seq = model.generate(...)
   generated_volume = HilbertCurve.sequence_to_spatial_3d(generated_seq, metadata)

API Reference
-------------

Classes
~~~~~~~~~~

.. autoclass:: medlatents.AutoregressiveTransformer
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: medlatents.autoregressive.SpeculativeDecoder
   :members:
   :no-index:

.. autoclass:: medlatents.autoregressive.speculative.medusa.MedusaModel
   :members:

.. autoclass:: medlatents.inference.kv_cache.QuantizedKVCache
   :members:

Functions
~~~~~~~~~

.. autofunction:: medlatents.sampling.sample_nucleus
   :no-index:
.. autofunction:: medlatents.sampling.sample_min_p
   :no-index:
.. autofunction:: medlatents.sampling.sample_with_cfg
   :no-index:
.. autofunction:: medlatents.sampling.TemperatureScheduler
   :no-index:

See Also
---------

* `MaskGIT Guide <maskgit.html>`__ - Bidirectional masked generation
* `Rasterization Guide <rasterization.html>`__ - Space-filling curves
* `Sampling Module <../api/sampling.html>`__ - Complete sampling utilities
