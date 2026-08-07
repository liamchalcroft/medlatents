Post-Training Guide
===================

Post-training refines a *pretrained* generative model so its samples better
match human or automated preferences, optimize an explicit reward, or sample in
far fewer steps. MedLatents ships a unified post-training stack that works
across every architecture in the library -- autoregressive, MaskGIT, D3PM, flow
matching, and Bayesian flow -- through one shared configuration object.

There are five families of methods:

* **Preference optimization** -- DPO and its per-architecture variants, plus
  step-level SPO.
* **Reinforcement learning** -- DDPO, GRPO, and GARDO for reward-driven
  fine-tuning.
* **Distillation** -- reflow and consistency distillation for few-step sampling.
* **Reward modeling and feedback** -- trained reward models and AI/discriminator
  feedback used to generate or score preference data.
* **Self-play** -- SPIN and rejection fine-tuning (RFT) for iterative
  self-improvement.

The API reference for every class below is in :doc:`../api/post_training`.

Shared configuration
--------------------

Every trainer takes a :class:`~medlatents.post_training.PostTrainingConfig`,
which carries optimization, EMA, distributed (FSDP), logging, and
method-specific hyperparameters. Construct it directly or from a named preset.

.. code-block:: python

   from medlatents.post_training import PostTrainingConfig

   # Direct construction
   config = PostTrainingConfig(method="dpo", beta=0.1, lr=1e-5, max_steps=2000)

   # Or from a preset, overriding individual fields
   config = PostTrainingConfig.from_preset("dpo_standard", lr=5e-6)

Ready-made preset constants are also exported:
:data:`~medlatents.post_training.DPO_SMALL`,
:data:`~medlatents.post_training.DPO_STANDARD`,
:data:`~medlatents.post_training.DPO_LARGE`,
:data:`~medlatents.post_training.SPO_STANDARD`,
:data:`~medlatents.post_training.DDPO_STANDARD`, and
:data:`~medlatents.post_training.GRPO_EFFICIENT`.

.. note::

   ``PostTrainingConfig`` enables FSDP by default (``use_fsdp=True``). On a
   single device, pass ``use_fsdp=False`` or hand the trainer a pre-configured
   :class:`accelerate.Accelerator`.

Building preference data
------------------------

DPO, SPO, and reward-model training all consume *preference pairs*. The
recommended path is to generate samples from your current model and rank them
with a reward function via
:meth:`~medlatents.post_training.PreferenceDataset.from_generations`:

.. code-block:: python

   from medlatents.post_training import PreferenceDataset

   dataset = PreferenceDataset.from_generations(
       generator=model,                 # any model with a .generate() method
       prompts=prompts,                 # conditioning labels / prompts
       reward_fn=reward_fn,             # scores samples; higher is better
       num_samples_per_prompt=4,
       pair_strategy="best_worst",      # "all", "adjacent", or "best_worst"
       device="cuda",
   )

You can also assemble pairs manually with
:class:`~medlatents.post_training.PreferencePair` and
:meth:`PreferenceDataset.add_pair`, and batch them with
:func:`~medlatents.post_training.collate_preference_pairs`.

Direct Preference Optimization (DPO)
------------------------------------

DPO optimizes the policy directly against preference pairs, using a frozen
*reference* model to regularize the update (set ``ref_model=None`` for
reference-free training). Each architecture has a trainer that knows how to
compute log-probabilities for that model family.

Discrete models (autoregressive, MaskGIT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import copy
   from medlatents.post_training import PostTrainingConfig, AutoregressiveDPOTrainer

   ref_model = copy.deepcopy(model).eval()
   for p in ref_model.parameters():
       p.requires_grad_(False)

   config = PostTrainingConfig(method="dpo", beta=0.1, lr=1e-5, use_fsdp=False)
   trainer = AutoregressiveDPOTrainer(
       model=model,
       ref_model=ref_model,
       config=config,
       vocab_size=512,          # inferred from the model if omitted
   )
   history = trainer.train(dataset)

:class:`~medlatents.post_training.MaskGITDPOTrainer` follows the same signature
for bidirectional masked models, and
:class:`~medlatents.post_training.DiscreteDPOTrainer` is the shared base for
discrete-token DPO.

Diffusion models (D3PM)
^^^^^^^^^^^^^^^^^^^^^^^

Diffusion DPO scores pairs through the ELBO of the discrete diffusion process,
so the trainer needs the :class:`~medlatents.diffusion.D3PM` object:

.. code-block:: python

   from medlatents.post_training import D3PMDPOTrainer

   trainer = D3PMDPOTrainer(
       model=model,
       ref_model=ref_model,
       config=config,
       d3pm=d3pm,                       # the D3PM diffusion process
       num_timestep_samples=10,         # Monte-Carlo timesteps for the ELBO
       elbo_mode="monte_carlo",
   )

Flow-matching models
^^^^^^^^^^^^^^^^^^^^

Flow DPO estimates likelihood along the flow trajectory and needs the
probability path:

.. code-block:: python

   from medlatents.post_training import DiscreteFlowDPOTrainer

   trainer = DiscreteFlowDPOTrainer(
       model=model,
       ref_model=ref_model,
       config=config,
       path=path,                       # e.g. MixtureDiscreteProbPath
       likelihood_strategy="trajectory",
       num_timestep_samples=10,
   )

:class:`~medlatents.post_training.FlowDPOTrainer` provides the continuous
counterpart.

Loss variants
^^^^^^^^^^^^^

All DPO trainers share :class:`~medlatents.post_training.DPOLoss`, which
supports ``sigmoid`` (standard DPO), ``hinge``, ``ipo``, and ``kto`` objectives.
Select the variant with ``config.loss_type``.

Step-by-step Preference Optimization (SPO)
------------------------------------------

SPO optimizes preferences at the level of individual denoising/decoding steps
rather than only final samples. It generates ``num_candidates`` continuations
per step and scores intermediate states with a step preference model.

.. code-block:: python

   from medlatents.post_training import PostTrainingConfig, SPOTrainer

   config = PostTrainingConfig(method="spo", num_candidates=4, use_fsdp=False)
   trainer = SPOTrainer(
       model=model,
       config=config,
       diffusion=d3pm,                  # or flow_path=path for flow models
       num_candidates=4,
       num_steps=50,
   )
   history = trainer.train(seq_length=256)

The trainable step scorer is
:class:`~medlatents.post_training.StepPreferenceModel`.

Reinforcement learning
----------------------

RL trainers treat generation as a multi-step MDP and optimize a reward function
directly. They share :class:`~medlatents.post_training.rl.BaseRLTrainer`, which
takes a ``reward_fn`` and exposes ``generate_trajectories`` and
``train(seq_length=...)``.

* **DDPO** (:class:`~medlatents.post_training.rl.DDPOTrainer`) -- PPO-style
  policy optimization over the denoising trajectory, with GAE advantages
  (``config.gae_lambda``). :class:`~medlatents.post_training.rl.DDPODiscreteFlowTrainer`
  targets discrete flow models.
* **GRPO** (:class:`~medlatents.post_training.rl.GRPOTrainer`) -- group-relative
  policy optimization that drops the value network and normalizes rewards within
  a group of samples per prompt (memory-efficient).
* **GARDO** (:class:`~medlatents.post_training.rl.GARDOTrainer`) -- adds an
  adaptive KL/distance penalty (``config.reward_threshold``,
  ``config.distance_penalty``).

.. code-block:: python

   from medlatents.post_training import PostTrainingConfig
   from medlatents.post_training.rl import GRPOTrainer

   config = PostTrainingConfig(
       method="grpo",
       num_samples_per_prompt=8,
       clip_range=0.2,
       use_fsdp=False,
   )

   def reward_fn(samples):
       # Return a scalar reward per sample; higher is better.
       return scorer.score(samples)

   trainer = GRPOTrainer(
       model=model,
       reward_fn=reward_fn,
       config=config,
       ref_model=ref_model,             # optional KL anchor
   )
   history = trainer.train(seq_length=256)

Distillation
------------

Distillation compresses a many-step sampler into a few-step one.

Reflow
^^^^^^

Reflow iteratively straightens flow trajectories by retraining on
``(noise, data)`` pairs generated by the current model. Generate pairs with
:class:`~medlatents.post_training.distillation.ReflowPairGenerator`, then fit
them with :class:`~medlatents.post_training.distillation.ReflowTrainer`:

.. code-block:: python

   from medlatents.post_training.distillation import ReflowPairGenerator

   pair_gen = ReflowPairGenerator(
       model=model,
       path=path,
       vocab_size=512,
       device=device,
       num_inference_steps=50,
   )
   noise, data = pair_gen.generate_pairs(data_samples, batch_size=32)

Consistency distillation
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~medlatents.post_training.distillation.ConsistencyTrainer` learns to map
any noise level directly to data, enabling 1--4 step sampling. In
``"distillation"`` mode it requires a frozen teacher; in ``"training"`` mode it
trains a consistency model from scratch.

.. code-block:: python

   from medlatents.post_training import PostTrainingConfig
   from medlatents.post_training.distillation import ConsistencyTrainer

   config = PostTrainingConfig(method="consistency", teacher_path="teacher.pt", use_fsdp=False)
   trainer = ConsistencyTrainer(
       student=student,
       config=config,
       teacher=teacher,
       mode="distillation",
       s0=10, s1=1280,                  # discretization schedule endpoints
   )
   history = trainer.train()

The pseudo-Huber objective is available standalone as
:func:`~medlatents.post_training.distillation.pseudo_huber_loss`.

Reward modeling and feedback
----------------------------

Reward models and feedback scorers power both RL and synthetic preference data.

* :func:`~medlatents.post_training.create_reward_model` builds a sized
  :class:`~medlatents.post_training.RewardModel` (``nano``/``small``/``base``/
  ``large``; set ``timestep_aware=True`` for SPO-style step scoring).
* :class:`~medlatents.post_training.RewardModelTrainer` fits a reward model on a
  :class:`~medlatents.post_training.PreferenceDataset`.
* :class:`~medlatents.post_training.preference.AIPreferenceScorer` wraps any
  scoring callable (e.g. ``AIPreferenceScorer.from_reward_model(rm)`` or
  ``from_discriminator(...)``) for AI-feedback labeling.

.. code-block:: python

   from medlatents.post_training import create_reward_model, RewardModelTrainer
   from medlatents.post_training.preference import AIPreferenceScorer

   reward_model = create_reward_model(model_size="small", vocab_size=512)
   rm_trainer = RewardModelTrainer(reward_model, config)
   rm_trainer.train(dataset)

   # Turn the trained reward model into a standalone scorer
   scorer = AIPreferenceScorer.from_reward_model(reward_model)
   scores = scorer.score(samples)

Self-play
---------

Self-play methods improve a model by training on its own generations.

* **SPIN** (:class:`~medlatents.post_training.self_play.SPINTrainer`) treats the
  model's current samples as negatives against higher-quality targets, creating
  an implicit curriculum. It needs no external reward.
* **RFT** (:class:`~medlatents.post_training.self_play.RFTTrainer`) -- rejection
  fine-tuning: generate ``num_samples_per_prompt`` candidates, keep the top-``k``
  by reward, and fine-tune on those.

.. code-block:: python

   from medlatents.post_training import PostTrainingConfig
   from medlatents.post_training.self_play import RFTTrainer

   config = PostTrainingConfig(method="rft", use_fsdp=False)
   trainer = RFTTrainer(
       model=model,
       reward_fn=reward_fn,
       config=config,
       num_samples_per_prompt=8,
       top_k=4,
       model_type="maskgit",
   )
   history = trainer.train(seq_length=256)

Choosing a method
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Goal
     - Method
     - When to use
   * - Align to preferences, have pairs
     - DPO (per-architecture)
     - You can rank pairs of samples and want a stable, reward-free update.
   * - Align at the step level
     - SPO
     - Final-sample preferences are too coarse; you want per-step credit.
   * - Optimize an explicit reward
     - GRPO / DDPO / GARDO
     - You have a differentiable or black-box reward and a compute budget for
       rollouts. GRPO is the most memory-frugal.
   * - Sample in fewer steps
     - Reflow / consistency
     - Inference latency matters; you can afford a distillation phase.
   * - Improve without external labels
     - SPIN / RFT
     - No human labels available; you have a usable quality signal (SPIN) or a
       reward function (RFT).

See also
--------

* :doc:`../api/post_training` -- full API reference.
* :doc:`../research/implemented_methods` -- the methods and their source papers.
