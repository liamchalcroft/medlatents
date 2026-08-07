Diffusion Models Guide
======================

This guide covers discrete and continuous diffusion models in medlatents,
including D3PM, SEDD/MDLM, and continuous Gaussian diffusion.

Overview
--------

Diffusion models learn to reverse a gradual corruption process:

1. **Forward process**: Progressively add noise to data
2. **Reverse process**: Learn to denoise step-by-step
3. **Generation**: Start from noise, iteratively denoise

medlatents provides both **discrete** diffusion (D3PM, SEDD, MDLM) for token-based
generation and **continuous** diffusion for latent space modeling.

.. mermaid::

   flowchart LR
       subgraph Forward["Forward Process"]
           X0[Clean Data] --> X1[Slightly Noisy]
           X1 --> X2[More Noisy]
           X2 --> XT[Pure Noise]
       end

       subgraph Reverse["Reverse Process (Learned)"]
           XT2[Pure Noise] --> X2r[Denoised]
           X2r --> X1r[More Denoised]
           X1r --> X0r[Clean Sample]
       end

       XT -.->|"Sample"| XT2

Discrete Diffusion (D3PM)
-------------------------

D3PM extends diffusion to discrete tokens using transition matrices.

Basic Setup
^^^^^^^^^^^

.. code-block:: python

   from medlatents.diffusion import D3PM
   from medlatents.networks import DiscreteDiT_models

   # Create the neural network backbone
   model = DiscreteDiT_models['DiscreteDiT-S'](
       vocab_size=8192,
       num_classes=0,  # Unconditional
       max_seq_len=4096,
   )

   # Create diffusion process
   d3pm = D3PM(
       num_classes=8192,
       num_timesteps=1000,
       transition_type='absorbing',  # Options: absorbing, uniform, gaussian
   )

Transition Types
^^^^^^^^^^^^^^^^

D3PM supports different corruption strategies:

**Absorbing State** (Recommended for text/tokens):

.. code-block:: python

   from medlatents.diffusion import get_absorbing_transition_mat

   # Tokens gradually transition to [MASK]
   Q = get_absorbing_transition_mat(vocab_size=8192, mask_token_id=8191)

   d3pm = D3PM(
       num_classes=8192,
       num_timesteps=1000,
       transition_type='absorbing',
   )

**Uniform Transition** (All tokens equally likely):

.. code-block:: python

   from medlatents.diffusion import get_uniform_transition_mat

   Q = get_uniform_transition_mat(vocab_size=8192)

   d3pm = D3PM(
       num_classes=8192,
       num_timesteps=1000,
       transition_type='uniform',
   )

**Discretized Gaussian** (Nearby tokens more likely):

.. code-block:: python

   from medlatents.diffusion import get_discretized_gaussian_transition_mat

   # Good for ordinal data (image tokens, audio)
   Q = get_discretized_gaussian_transition_mat(vocab_size=8192, beta=0.02)

   d3pm = D3PM(
       num_classes=8192,
       num_timesteps=1000,
       transition_type='gaussian',
   )

Training
^^^^^^^^

.. code-block:: python

   from medlatents.training import DiscreteTrainer

   trainer = DiscreteTrainer(
       model=model,
       diffusion=d3pm,
       optimizer=optimizer,
       gradient_accumulation_steps=4,
   )

   # Training loop
   for batch in dataloader:
       loss = trainer.train_step(batch['tokens'])

   # Or use the built-in training
   trainer.train(
       dataloader,
       num_epochs=100,
       checkpoint_dir='./checkpoints',
   )

KLASS Sampling
^^^^^^^^^^^^^^

D3PM supports KLASS (Know-your-Limits Adaptive Sampling Strategy) for
improved quality:

.. code-block:: python

   from medlatents.diffusion import D3PMKLASS

   # KLASS extends D3PM with adaptive sampling
   d3pm_klass = D3PMKLASS(
       num_classes=8192,
       num_timesteps=1000,
       transition_type='absorbing',
       klass_threshold=0.8,  # Confidence threshold for resampling
   )

   # Generate with KLASS
   samples = d3pm_klass.sample(
       model=model,
       shape=(4, 1024),  # batch_size, seq_len
       device='cuda',
       use_klass=True,
   )

Score Entropy Discrete Diffusion (SEDD)
---------------------------------------

SEDD uses score entropy for improved training, particularly effective for
language modeling.

Setup
^^^^^

.. code-block:: python

   from medlatents.diffusion import SEDDLoss, ContinuousTimeMDLM
   from medlatents.networks import DiscreteDiT_models

   model = DiscreteDiT_models['DiscreteDiT-B'](
       vocab_size=8192,
       max_seq_len=4096,
   )

   # SEDD loss for training
   sedd_loss = SEDDLoss(
       vocab_size=8192,
       mask_token_id=8191,
   )

   # Continuous-time formulation
   mdlm = ContinuousTimeMDLM(
       vocab_size=8192,
       mask_token_id=8191,
   )

Training with SEDD
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

   for batch in dataloader:
       x0 = batch['tokens']  # [B, L]

       # Sample timesteps (continuous time)
       t = torch.rand(x0.shape[0], device=x0.device)

       # Corrupt tokens
       xt = mdlm.corrupt(x0, t)

       # Get model predictions (score)
       logits = model(xt, t)

       # Compute SEDD loss
       loss = sedd_loss(logits, x0, xt, t)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

MDLM (Masked Diffusion Language Model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MDLM simplifies SEDD for masked language modeling:

.. code-block:: python

   from medlatents.diffusion import MDLMLoss

   mdlm_loss = MDLMLoss(
       vocab_size=8192,
       mask_token_id=8191,
       label_smoothing=0.1,  # Optional regularization
   )

   # Training is similar to SEDD
   loss = mdlm_loss(logits, x0, xt, t)

Continuous Gaussian Diffusion
-----------------------------

For continuous latent spaces (VAE embeddings, etc.):

.. code-block:: python

   from medlatents.diffusion import ContinuousGaussianDiffusion
   from medlatents.networks import ContinuousDiT

   # Model predicts continuous values
   model = ContinuousDiT(
       input_dim=256,
       hidden_dim=768,
       num_layers=12,
   )

   diffusion = ContinuousGaussianDiffusion(
       num_timesteps=1000,
       beta_schedule='cosine',  # Options: linear, cosine, sqrt
       prediction_type='v',  # Options: epsilon, v, x0
   )

Zero-Terminal SNR
^^^^^^^^^^^^^^^^^

Improved training with zero terminal SNR (WACV 2024):

.. code-block:: python

   from medlatents.diffusion import (
       get_beta_schedule_with_zero_snr,
       enforce_zero_terminal_snr,
       min_snr_weighting,
       compute_log_snr,
   )

   # Get betas with zero terminal SNR
   betas = get_beta_schedule_with_zero_snr(
       schedule='cosine',
       num_timesteps=1000,
   )

   # Or enforce on existing betas
   betas = enforce_zero_terminal_snr(betas)

   # Use Min-SNR weighting for loss
   alphas_cumprod = torch.cumprod(1 - betas, dim=0)
   log_snr = compute_log_snr(alphas_cumprod)
   weights = min_snr_weighting(log_snr, gamma=5.0)

Importance Sampling
^^^^^^^^^^^^^^^^^^^

Sample timesteps based on log-SNR for better training:

.. code-block:: python

   from medlatents.diffusion import sample_timesteps_log_snr_importance

   # Sample timesteps with importance weighting
   timesteps = sample_timesteps_log_snr_importance(
       batch_size=32,
       num_timesteps=1000,
       log_snr=log_snr,
       device='cuda',
   )

Sampling Strategies
-------------------

D3PM Sampling
^^^^^^^^^^^^^

.. code-block:: python

   # Standard ancestral sampling
   samples = d3pm.sample(
       model=model,
       shape=(batch_size, seq_len),
       device='cuda',
   )

   # With temperature
   samples = d3pm.sample(
       model=model,
       shape=(batch_size, seq_len),
       temperature=0.8,  # Lower = more deterministic
   )

   # Fewer steps (faster but lower quality)
   samples = d3pm.sample(
       model=model,
       shape=(batch_size, seq_len),
       num_steps=100,  # Default is num_timesteps
   )

DDIM-style Sampling
^^^^^^^^^^^^^^^^^^^

For continuous diffusion, use DDIM for deterministic sampling:

.. code-block:: python

   from medlatents.sampling import DDIMScheduler

   scheduler = DDIMScheduler(
       num_train_timesteps=1000,
       num_inference_steps=50,  # Fewer steps = faster
       eta=0.0,  # 0 = deterministic, 1 = DDPM
   )

   samples = diffusion.sample(
       model=model,
       shape=(batch_size, seq_len, dim),
       scheduler=scheduler,
   )

DPM-Solver
^^^^^^^^^^

Faster sampling with DPM-Solver:

.. code-block:: python

   from medlatents.sampling import DPMSolverScheduler

   scheduler = DPMSolverScheduler(
       num_train_timesteps=1000,
       num_inference_steps=20,  # Very few steps needed
       solver_order=2,  # 1, 2, or 3
   )

Classifier-Free Guidance
------------------------

Scale conditional generation with CFG:

.. code-block:: python

   from medlatents.sampling import sample_with_cfg

   # Model must support conditional and unconditional
   samples = sample_with_cfg(
       model=model,
       diffusion=d3pm,
       condition=class_labels,
       guidance_scale=7.5,  # Higher = stronger conditioning
       shape=(batch_size, seq_len),
   )

Best Practices
--------------

Choosing a Method
^^^^^^^^^^^^^^^^^

+-------------------+------------------------+------------------------+
| Method            | Best For               | Trade-offs             |
+===================+========================+========================+
| D3PM (Absorbing)  | Text, VQ tokens        | Simple, proven         |
+-------------------+------------------------+------------------------+
| D3PM (Gaussian)   | Ordinal tokens         | Better for image VQ    |
+-------------------+------------------------+------------------------+
| SEDD/MDLM         | Language modeling      | Continuous time        |
+-------------------+------------------------+------------------------+
| Continuous        | VAE latents            | Works with any latent  |
+-------------------+------------------------+------------------------+

Hyperparameters
^^^^^^^^^^^^^^^

.. code-block:: python

   # Recommended settings
   config = {
       'num_timesteps': 1000,  # Standard, can reduce for faster sampling
       'transition_type': 'absorbing',  # For discrete tokens
       'beta_schedule': 'cosine',  # For continuous
       'prediction_type': 'v',  # V-prediction often better
       'lr': 1e-4,
       'warmup_steps': 1000,
       'ema_decay': 0.9999,
   }

Memory Optimization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Gradient checkpointing for large models
   model.gradient_checkpointing_enable()

   # Mixed precision training
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       loss = compute_loss(...)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()

API Reference
-------------

.. seealso::

   - :doc:`/api/diffusion` - Full API documentation
   - :doc:`/guides/flow_matching` - Alternative: Flow Matching
   - :doc:`/guides/maskgit` - Alternative: MaskGIT
