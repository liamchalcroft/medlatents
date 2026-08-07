Implemented Methods
===================

MedLatents implements a broad set of generative-modeling, sampling, and
post-training methods, each grounded in the literature. This page gives a
high-level map of what is implemented and where; full citations are collected in
:doc:`references`.

Generative model families
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Family
     - Method
     - Module
   * - Autoregressive
     - Causal transformer over discrete tokens; speculative and Medusa
       multi-head decoding for acceleration.
     - :mod:`medlatents.autoregressive`
   * - Masked parallel
     - MaskGIT bidirectional masked transformer with confidence-based decoding.
     - :mod:`medlatents.maskgit`
   * - Discrete diffusion
     - D3PM (absorbing / uniform transitions), plus SEDD score-entropy and MDLM
       masked-diffusion objectives.
     - :mod:`medlatents.diffusion`
   * - Flow matching
     - Discrete mixture-path flow matching and continuous rectified flow, with
       optimal-transport coupling and shortcut models.
     - :mod:`medlatents.flow_matching`
   * - Bayesian flow
     - Bayesian Flow Networks for discrete data with entropy encoding,
       score-guided and particle sampling, and higher-order solvers.
     - :mod:`medlatents.bayesian_flow`
   * - Continuous latent diffusion
     - Continuous Gaussian diffusion paired with a continuous DiT backbone.
     - :mod:`medlatents.diffusion`, :mod:`medlatents.networks`

Sampling and inference techniques
---------------------------------

* **Halton scheduling** -- low-discrepancy, spatially dispersed unmasking for
  MaskGIT (:class:`medlatents.sampling.MaskGITScheduler`).
* **KLASS early stopping** -- KL-adaptive stability sampling that halts MaskGIT
  and D3PM once the prediction stabilizes
  (:class:`medlatents.sampling.KLASSGenerator`,
  :class:`medlatents.diffusion.D3PMKLASS`).
* **Running confidence remasking** -- confidence-driven re-masking for masked
  decoders (:class:`medlatents.sampling.RunningConfidenceRemasker`).
* **Classifier-free guidance** -- constant, linear, cosine, and triangular
  time-dependent schedules, plus CFG-Zero* rescaling
  (:class:`medlatents.sampling.GuidedSampler`,
  :func:`medlatents.sampling.cfg_zero_star_guidance`).
* **Decoupled straight-through estimators** -- decoupled ST-Gumbel-Softmax and
  ReinMax for low-variance discrete gradients
  (:class:`medlatents.sampling.DecoupledSTGumbelSoftmax`).
* **Higher-order ODE solvers** -- Euler / Heun / RK4 integration for rectified
  flow, and DPM-Solver / DDIM / Euler schedulers for diffusion
  (:mod:`medlatents.sampling`).
* **Speculative and Medusa decoding** -- draft-model and multi-head speculative
  decoding for autoregressive acceleration
  (:class:`medlatents.autoregressive.SpeculativeDecoder`,
  :class:`medlatents.autoregressive.MedusaModel`).

Training techniques
-------------------

* **Zero-terminal-SNR** schedules for diffusion
  (:func:`medlatents.diffusion.enforce_zero_terminal_snr`,
  :func:`medlatents.diffusion.min_snr_weighting`).
* **Representation alignment (REPA / REPA-E)** -- align internal features to
  pretrained encoders, including end-to-end VAE + DiT training
  (:class:`medlatents.training.REPALoss`,
  :class:`medlatents.training.REPAETrainer`).
* **Contrastive flow-matching** regularizers
  (:class:`medlatents.training.ContrastiveFlowMatchingLoss`).
* **Curriculum learning** over masking ratio, noise level, sequence length, and
  token difficulty (:mod:`medlatents.training`).
* **Unified conditioning** via :class:`medlatents.conditioning.ConditioningBundle`
  and frozen pretrained encoders (DINOv2, SigLIP, MedSigLIP, NeuroVFM).

Post-training and alignment
---------------------------

These methods are documented in depth in :doc:`../guides/post_training`.

* **Preference optimization** -- DPO and per-architecture variants
  (autoregressive, MaskGIT, D3PM, flow), with sigmoid/hinge/IPO/KTO losses, and
  step-level SPO.
* **Reinforcement learning** -- DDPO, GRPO, and GARDO for reward-driven
  fine-tuning.
* **Distillation** -- reflow and consistency distillation for few-step sampling.
* **Self-play** -- SPIN and rejection fine-tuning (RFT).

Spatial-to-sequence conversion
------------------------------

* **Space-filling curves** -- Hilbert and Z-order rasterization that preserve
  spatial locality when flattening grids to sequences
  (:mod:`medlatents.rasterization`). See :doc:`../token_interface` for the
  layout contract.

Research process
----------------

New methods are integrated through a deliberate triage / reproduce / ablate /
integrate gate rather than added speculatively. Open an issue to propose a
method before implementing it.
