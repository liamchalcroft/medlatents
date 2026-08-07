Flow Matching Guide
===================

This guide covers discrete and continuous flow matching in medlatents,
a simulation-free alternative to diffusion models with straighter generation paths.

Overview
--------

Flow matching learns a velocity field that transports noise to data:

1. **Define a path**: Interpolate between noise and data
2. **Learn velocities**: Predict the tangent vector at each point
3. **Generate**: Integrate the ODE from noise to data

Flow matching often provides:

- **Faster sampling** (straighter paths need fewer steps)
- **Simpler training** (no diffusion schedule tuning)
- **Better mode coverage** (optimal transport coupling)

.. mermaid::

   flowchart LR
       subgraph Flow["Flow Matching"]
           N[Noise Distribution] -->|"ODE Integration"| D[Data Distribution]
       end

       subgraph Velocity["Learned Velocity Field v(x, t)"]
           T0["t=0"] --> T05["t=0.5"] --> T1["t=1.0"]
       end

Discrete Flow Matching
----------------------

For token-based generation using mixture probability paths.

Basic Setup
^^^^^^^^^^^

.. code-block:: python

   from medlatents.flow_matching import (
       MixtureDiscreteProbPath,
       MaskedSourceDistribution,
       get_loss_function,
       DiscreteFlowTrainer,
   )
   from medlatents.networks import DiscreteDiT_models

   # Create model
   model = DiscreteDiT_models['DiscreteDiT-S'](
       vocab_size=8192,
       max_seq_len=4096,
   )

   # Source distribution (where generation starts)
   source = MaskedSourceDistribution(
       vocab_size=8192,
       mask_token_id=8191,
   )

   # Probability path interpolation
   path = MixtureDiscreteProbPath(
       source_distribution=source,
       scheduler='linear',  # Options: linear, cosine, polynomial
   )

   # Loss function
   loss_fn = get_loss_function('cross_entropy')

Source Distributions
^^^^^^^^^^^^^^^^^^^^

**Masked Source** (recommended for text/tokens):

.. code-block:: python

   from medlatents.flow_matching import MaskedSourceDistribution

   # All tokens start as [MASK]
   source = MaskedSourceDistribution(
       vocab_size=8192,
       mask_token_id=8191,
   )

**Uniform Source** (all tokens equally likely):

.. code-block:: python

   from medlatents.flow_matching import UniformSourceDistribution

   # Tokens sampled uniformly at t=0
   source = UniformSourceDistribution(vocab_size=8192)

Custom source distributions:

.. code-block:: python

   from medlatents.flow_matching import SourceDistribution

   class FrequencyWeightedSource(SourceDistribution):
       """Sample from token frequency distribution."""

       def __init__(self, frequencies: torch.Tensor):
           self.probs = frequencies / frequencies.sum()

       def sample(self, shape, device):
           return torch.multinomial(
               self.probs.expand(shape[0], -1),
               num_samples=shape[1],
           )

Training
^^^^^^^^

.. code-block:: python

   trainer = DiscreteFlowTrainer(
       model=model,
       path=path,
       loss_fn=loss_fn,
       optimizer=optimizer,
   )

   for batch in dataloader:
       x1 = batch['tokens']  # Target data

       # Sample source (noise)
       x0 = source.sample(x1.shape, x1.device)

       # Sample timestep
       t = torch.rand(x1.shape[0], device=x1.device)

       # Interpolate on path
       xt = path.sample(x0, x1, t)

       # Get model predictions
       logits = model(xt, t)

       # Compute loss
       loss = loss_fn(logits, x1, xt, t)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

Generalized KL Loss
^^^^^^^^^^^^^^^^^^^

For improved training with mixture paths:

.. code-block:: python

   from medlatents.flow_matching import MixturePathGeneralizedKL

   loss_fn = MixturePathGeneralizedKL(
       vocab_size=8192,
       path=path,
       label_smoothing=0.1,
   )

Timestep Sampling
^^^^^^^^^^^^^^^^^

Different strategies for sampling training timesteps:

.. code-block:: python

   from medlatents.flow_matching import (
       sample_timesteps_uniform,
       sample_timesteps_u_shaped,
       sample_timesteps_logit_normal,
       get_timestep_sampler,
   )

   # Uniform (standard)
   t = sample_timesteps_uniform(batch_size, device)

   # U-shaped (focus on t=0 and t=1)
   t = sample_timesteps_u_shaped(batch_size, device, alpha=0.5)

   # Logit-normal (focus on middle timesteps)
   t = sample_timesteps_logit_normal(batch_size, device, loc=0.0, scale=1.0)

   # Get sampler by name
   sampler = get_timestep_sampler('logit_normal')
   t = sampler(batch_size, device)

Generation
^^^^^^^^^^

.. code-block:: python

   from medlatents.flow_matching import (
       MixtureDiscreteEulerSolver,
       generate_samples,
       create_time_grid,
   )

   # Create solver
   solver = MixtureDiscreteEulerSolver(path=path)

   # Time grid (fewer steps = faster)
   time_grid = create_time_grid(num_steps=50)

   # Generate
   samples = generate_samples(
       model=model,
       solver=solver,
       source=source,
       time_grid=time_grid,
       shape=(batch_size, seq_len),
       device='cuda',
   )

Continuous Flow Matching
------------------------

For continuous latent spaces (Rectified Flow):

.. code-block:: python

   from medlatents.flow_matching import RectifiedFlow

   flow = RectifiedFlow(
       prediction_type='velocity',  # Options: velocity, x1, noise
   )

   # Training
   x1 = data  # Target [B, L, D]
   x0 = torch.randn_like(x1)  # Noise
   t = torch.rand(x1.shape[0], 1, 1, device=x1.device)

   # Linear interpolation
   xt = (1 - t) * x0 + t * x1

   # Target velocity
   v_target = x1 - x0

   # Model prediction
   v_pred = model(xt, t.squeeze())

   # MSE loss
   loss = F.mse_loss(v_pred, v_target)

Reflow (Straightening)
^^^^^^^^^^^^^^^^^^^^^^

Iteratively straighten flow paths for faster sampling:

.. code-block:: python

   # Generate pairs using current model
   x0 = torch.randn(batch_size, seq_len, dim)
   with torch.no_grad():
       x1 = flow.sample(model, x0, num_steps=100)

   # Retrain on these pairs (paths become straighter)
   # Repeat for 2-3 iterations

Optimal Transport Coupling
--------------------------

Use OT coupling for straighter flows (mini-batch OT):

.. code-block:: python

   from medlatents.flow_matching import (
       compute_ot_coupling,
       sample_from_coupling,
       ot_flow_sample_path,
   )

   # Compute OT coupling between source and target
   coupling = compute_ot_coupling(x0, x1)  # [B, B] coupling matrix

   # Sample matched pairs
   x0_matched, x1_matched = sample_from_coupling(x0, x1, coupling)

   # Or use the combined function
   xt, x0_matched, x1_matched = ot_flow_sample_path(
       x0, x1, t,
       use_ot=True,
   )

Shortcut Flow Matching
----------------------

Generate with flexible step counts using step-conditioned models (2024):

.. code-block:: python

   from medlatents.flow_matching import (
       ShortcutFlowMatchingModel,
       ShortcutFlowMatchingLoss,
       AdaptiveStepSampler,
   )

   # Wrap model to accept step conditioning
   shortcut_model = ShortcutFlowMatchingModel(
       base_model=model,
       max_steps=128,
   )

   # Training loss
   loss_fn = ShortcutFlowMatchingLoss()

   # During training, condition on random step counts
   for batch in dataloader:
       num_steps = torch.randint(1, 129, (batch_size,))
       loss = loss_fn(shortcut_model, x0, x1, t, num_steps)

Adaptive Step Sampler
^^^^^^^^^^^^^^^^^^^^^

Automatically choose step count based on sample difficulty:

.. code-block:: python

   sampler = AdaptiveStepSampler(
       model=shortcut_model,
       min_steps=4,
       max_steps=64,
       tolerance=0.01,
   )

   samples = sampler.sample(
       source_samples=x0,
       device='cuda',
   )

Time-Invariant Models
^^^^^^^^^^^^^^^^^^^^^

For models that don't need time conditioning:

.. code-block:: python

   from medlatents.flow_matching import TimeInvariantVectorField

   # Wrap time-dependent model to be time-invariant
   ti_model = TimeInvariantVectorField(model)

Evaluation
----------

Entropy Estimation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.flow_matching import compute_entropy

   # Estimate entropy of generated distribution
   entropy = compute_entropy(
       model=model,
       path=path,
       source=source,
       num_samples=1000,
   )

Likelihood Estimation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.flow_matching import estimate_likelihood

   # Estimate log-likelihood of data under model
   log_likelihood = estimate_likelihood(
       model=model,
       path=path,
       data=test_data,
       num_integration_steps=100,
   )

Counterfactual Generation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.flow_matching import generate_counterfactual

   # Generate counterfactual by partially integrating
   counterfactual = generate_counterfactual(
       model=model,
       source_data=x,
       target_time=0.7,  # How far to move toward generation
       num_steps=50,
   )

Polynomial Schedulers
---------------------

Control interpolation speed along the path:

.. code-block:: python

   from medlatents.flow_matching import PolynomialConvexScheduler

   # Polynomial schedule (slower at endpoints)
   scheduler = PolynomialConvexScheduler(
       degree=2.0,  # Higher = slower at endpoints
   )

   path = MixtureDiscreteProbPath(
       source_distribution=source,
       scheduler=scheduler,
   )

Best Practices
--------------

Choosing Settings
^^^^^^^^^^^^^^^^^

+---------------------------+---------------------------+
| Setting                   | Recommendation            |
+===========================+===========================+
| Source distribution       | Masked for tokens         |
+---------------------------+---------------------------+
| Timestep sampling         | Logit-normal or U-shaped  |
+---------------------------+---------------------------+
| Number of steps           | 50-100 for quality        |
+---------------------------+---------------------------+
| OT coupling               | Enable for continuous     |
+---------------------------+---------------------------+
| Loss function             | Generalized KL            |
+---------------------------+---------------------------+

Memory Optimization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use gradient checkpointing
   model.gradient_checkpointing_enable()

   # Smaller batch, more accumulation
   trainer = DiscreteFlowTrainer(
       model=model,
       gradient_accumulation_steps=8,
   )

Comparison with Diffusion
^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------+------------------+------------------+
| Aspect           | Flow Matching    | Diffusion        |
+==================+==================+==================+
| Training         | Simpler          | Schedule tuning  |
+------------------+------------------+------------------+
| Sampling steps   | Fewer (10-50)    | More (50-1000)   |
+------------------+------------------+------------------+
| Quality          | Comparable       | Proven           |
+------------------+------------------+------------------+
| Theory           | ODE-based        | SDE-based        |
+------------------+------------------+------------------+

API Reference
-------------

.. seealso::

   - :doc:`/api/flow_matching` - Full API documentation
   - :doc:`/guides/diffusion` - Alternative: Diffusion Models
   - :doc:`/guides/bayesian_flow` - Alternative: Bayesian Flow Networks
