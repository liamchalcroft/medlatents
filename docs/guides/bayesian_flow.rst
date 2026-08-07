Bayesian Flow Networks Guide
============================

This guide covers Bayesian Flow Networks (BFN) in medlatents, a principled
framework for discrete generative modeling based on Bayesian inference.

Overview
--------

Bayesian Flow Networks model generation as iterative Bayesian updating:

1. **Prior**: Start with uniform distribution over all tokens
2. **Bayesian Updates**: Progressively receive noisy observations
3. **Posterior**: Final distribution is the generated sample

Key advantages:

- **Principled uncertainty**: Full posterior over tokens at each step
- **Flexible generation**: Variable number of steps without retraining
- **Entropy control**: Explicit control over information flow

.. mermaid::

   flowchart LR
       subgraph BFN["Bayesian Flow"]
           P0["Prior (Uniform)"] -->|"Update 1"| P1["Posterior 1"]
           P1 -->|"Update 2"| P2["Posterior 2"]
           P2 -->|"..."| PN["Final Posterior"]
       end

       subgraph Info["Information Flow"]
           E0["High Entropy"] --> E1["..."] --> EN["Low Entropy"]
       end

BayesianFlowTransformer
-----------------------

The core model architecture:

.. code-block:: python

   from medlatents.bayesian_flow import (
       BayesianFlowTransformer,
       BFN_models,
   )

   # Use a preset configuration
   model = BFN_models['BFN-S'](
       vocab_size=8192,
       max_seq_len=4096,
   )

   # Or custom configuration
   model = BayesianFlowTransformer(
       vocab_size=8192,
       hidden_dim=768,
       num_layers=12,
       num_heads=12,
       max_seq_len=4096,
       dropout=0.1,
   )

Model Variants
^^^^^^^^^^^^^^

Available preset configurations:

+----------+-----------+--------+-------+---------+
| Name     | Hidden    | Layers | Heads | Params  |
+==========+===========+========+=======+=========+
| BFN-Ti   | 384       | 6      | 6     | ~15M    |
+----------+-----------+--------+-------+---------+
| BFN-S    | 768       | 12     | 12    | ~85M    |
+----------+-----------+--------+-------+---------+
| BFN-B    | 1024      | 24     | 16    | ~300M   |
+----------+-----------+--------+-------+---------+
| BFN-L    | 1536      | 24     | 24    | ~700M   |
+----------+-----------+--------+-------+---------+

Accuracy Schedules
------------------

Control how information flows during generation:

.. code-block:: python

   from medlatents.bayesian_flow import (
       exponential_schedule,
       linear_entropy_schedule,
       cosine_schedule,
       get_bfn_schedule,
   )

   # Exponential accuracy schedule (standard)
   beta = exponential_schedule(t, beta_1=1.0)  # Accuracy at time t

   # Linear entropy schedule
   beta = linear_entropy_schedule(t, min_entropy=0.1)

   # Cosine schedule (smoother)
   beta = cosine_schedule(t, s=0.008)

   # Get by name
   schedule_fn = get_bfn_schedule('exponential')

Schedule Comparison
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import torch

   t = torch.linspace(0, 1, 100)

   plt.figure(figsize=(10, 4))
   plt.plot(t, exponential_schedule(t), label='Exponential')
   plt.plot(t, cosine_schedule(t), label='Cosine')
   plt.plot(t, linear_entropy_schedule(t), label='Linear Entropy')
   plt.xlabel('Time t')
   plt.ylabel('Accuracy β(t)')
   plt.legend()
   plt.title('BFN Accuracy Schedules')

Training
--------

Basic Training Loop
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
   schedule_fn = get_bfn_schedule('exponential')

   for batch in dataloader:
       x = batch['tokens']  # [B, L]

       # Sample time
       t = torch.rand(x.shape[0], device=x.device)

       # Compute accuracy
       beta = schedule_fn(t)

       # Create noisy belief state
       # (model learns to predict true tokens from noisy observations)
       belief = model.create_belief_state(x, beta)

       # Forward pass
       logits = model(belief, t)

       # Cross-entropy loss
       loss = F.cross_entropy(
           logits.view(-1, logits.size(-1)),
           x.view(-1),
       )

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

Residual Loss Wrapper
^^^^^^^^^^^^^^^^^^^^^

For improved training stability:

.. code-block:: python

   from medlatents.bayesian_flow import ResidualLossWrapper

   loss_wrapper = ResidualLossWrapper(
       base_loss=F.cross_entropy,
       residual_weight=0.1,
   )

   loss = loss_wrapper(logits, x, belief)

Entropy Encoding
----------------

Entropy-aware training and generation:

.. code-block:: python

   from medlatents.bayesian_flow import (
       compute_entropy,
       encode_with_entropy,
   )

   # Compute entropy of current belief
   entropy = compute_entropy(belief)  # [B, L]

   # Encode with entropy information
   encoded = encode_with_entropy(
       tokens=x,
       entropy=entropy,
       temperature=1.0,
   )

Generation with Solvers
-----------------------

BFN supports multiple ODE/SDE solvers for generation:

Euler Solver (Simple)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.bayesian_flow import EulerSolver

   solver = EulerSolver(
       model=model,
       schedule_fn=schedule_fn,
       num_steps=100,
   )

   samples = solver.sample(
       shape=(batch_size, seq_len),
       device='cuda',
   )

Heun Solver (Higher Order)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.bayesian_flow import HeunSolver

   solver = HeunSolver(
       model=model,
       schedule_fn=schedule_fn,
       num_steps=50,  # Needs fewer steps than Euler
   )

   samples = solver.sample(shape=(batch_size, seq_len), device='cuda')

DPM-Solver (Fast)
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.bayesian_flow import DPMSolver2, DPMSolver3

   # Second-order DPM-Solver
   solver = DPMSolver2(
       model=model,
       schedule_fn=schedule_fn,
       num_steps=25,
   )

   # Third-order for even faster sampling
   solver = DPMSolver3(
       model=model,
       schedule_fn=schedule_fn,
       num_steps=15,
   )

Exponential Integrator
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.bayesian_flow import ExponentialIntegrator

   solver = ExponentialIntegrator(
       model=model,
       schedule_fn=schedule_fn,
       num_steps=30,
   )

Get Solver by Name
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.bayesian_flow import get_bfn_solver

   solver = get_bfn_solver(
       'dpm2',  # Options: euler, heun, dpm2, dpm3, exponential
       model=model,
       schedule_fn=schedule_fn,
       num_steps=25,
   )

Stochastic Sampling
-------------------

Add stochasticity for diversity:

.. code-block:: python

   from medlatents.bayesian_flow import StochasticHeun

   solver = StochasticHeun(
       model=model,
       schedule_fn=schedule_fn,
       num_steps=50,
       noise_scale=0.5,  # Controls stochasticity
   )

   # Multiple samples from same initial state
   samples_1 = solver.sample(shape=(1, seq_len), device='cuda')
   samples_2 = solver.sample(shape=(1, seq_len), device='cuda')
   # samples_1 != samples_2 due to stochasticity

Guided Sampling
---------------

Conditional generation with guidance.

Score-Guided Sampler
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from medlatents.bayesian_flow import ScoreGuidedSampler

   # Define a guidance function (e.g., classifier)
   def guidance_fn(x, t):
       # Returns gradient of log p(condition | x)
       logits = classifier(x)
       return torch.autograd.grad(logits[:, target_class].sum(), x)[0]

   sampler = ScoreGuidedSampler(
       model=model,
       schedule_fn=schedule_fn,
       guidance_fn=guidance_fn,
       guidance_scale=3.0,
   )

   guided_samples = sampler.sample(
       shape=(batch_size, seq_len),
       device='cuda',
   )

Advanced Features
-----------------

Variable Step Count
^^^^^^^^^^^^^^^^^^^

BFN can generate with any number of steps without retraining:

.. code-block:: python

   # Same model, different step counts
   for num_steps in [10, 25, 50, 100]:
       solver = EulerSolver(model, schedule_fn, num_steps=num_steps)
       samples = solver.sample(shape=(1, 1024), device='cuda')
       print(f"Steps: {num_steps}, Quality: {evaluate(samples)}")

Temperature Scaling
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   samples = solver.sample(
       shape=(batch_size, seq_len),
       device='cuda',
       temperature=0.8,  # Lower = more deterministic
   )

Top-k/Top-p Sampling
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   samples = solver.sample(
       shape=(batch_size, seq_len),
       device='cuda',
       top_k=50,  # Only consider top 50 tokens
       top_p=0.9,  # Or nucleus sampling
   )

Comparison with Other Methods
-----------------------------

+------------------+------------------+------------------+------------------+
| Aspect           | BFN              | Diffusion        | Flow Matching    |
+==================+==================+==================+==================+
| Uncertainty      | Full posterior   | Point estimate   | Point estimate   |
+------------------+------------------+------------------+------------------+
| Step flexibility | Any steps        | Fixed schedule   | Flexible         |
+------------------+------------------+------------------+------------------+
| Training         | Standard CE      | Schedule tuning  | Simple MSE/CE    |
+------------------+------------------+------------------+------------------+
| Theory           | Bayesian         | Score matching   | OT / ODE         |
+------------------+------------------+------------------+------------------+
| Sampling         | Multiple solvers | DDPM/DDIM/DPM    | Euler / ODE      |
+------------------+------------------+------------------+------------------+

Best Practices
--------------

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   # Recommended settings
   config = {
       'schedule': 'exponential',
       'solver': 'dpm2',
       'num_steps': 25,  # DPM2 is efficient
       'lr': 1e-4,
       'warmup_steps': 2000,
       'ema_decay': 0.9999,
   }

Training Tips
^^^^^^^^^^^^^

1. **Use EMA**: Essential for stable generation
2. **Warmup**: Learning rate warmup helps stability
3. **Schedule tuning**: Try different schedules for your data
4. **Residual loss**: Helps with long sequences

Sampling Tips
^^^^^^^^^^^^^

1. **Start with Euler**: Simple and reliable
2. **Move to DPM**: Once working, use DPM for speed
3. **Add stochasticity**: If samples lack diversity
4. **Temperature**: Tune for quality/diversity tradeoff

Memory Optimization
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Gradient checkpointing
   model.gradient_checkpointing_enable()

   # Compile for speed (PyTorch 2.0+)
   model = torch.compile(model)

   # Mixed precision
   with torch.autocast('cuda', dtype=torch.bfloat16):
       samples = solver.sample(...)

API Reference
-------------

.. seealso::

   - :doc:`/api/bayesian_flow` - Full API documentation
   - :doc:`/guides/diffusion` - Alternative: Diffusion Models
   - :doc:`/guides/flow_matching` - Alternative: Flow Matching
