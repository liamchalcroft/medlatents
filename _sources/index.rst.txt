MedLatents Documentation
========================

Discrete and continuous latent generative models for medical imagery.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/autoregressive
   guides/maskgit
   guides/diffusion
   guides/flow_matching
   guides/bayesian_flow
   guides/rasterization
   guides/post_training
   token_interface

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/autoregressive
   api/maskgit
   api/diffusion
   api/flow_matching
   api/bayesian_flow
   api/networks
   api/sampling
   api/training
   api/generation
   api/inference
   api/post_training
   api/evaluation
   api/conditioning
   api/rasterization
   api/data
   api/configs

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_training
   tutorials/02_generation
   tutorials/03_inpainting
   tutorials/04_evaluation

.. toctree::
   :maxdepth: 2
   :caption: Research

   research/implemented_methods
   research/references

.. toctree::
   :maxdepth: 1
   :caption: Project Info

   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Features
========

**Core Models**

* **Autoregressive Transformer**: Unidirectional generation with causal attention
* **MaskGIT**: Bidirectional masked transformer for parallel decoding
* **Discrete DiT**: Diffusion transformer for discrete latent spaces (D3PM)
* **Flow Matching**: Discrete and continuous flow-based generation
* **Bayesian Flow Networks**: Probabilistic flow models

**Advanced Features**

* **Halton Scheduler**: Spatially-dispersed unmasking for MaskGIT
* **KLASS Early Stopping**: Adaptive stopping based on KL divergence
* **Time-Dependent CFG**: Dynamic classifier-free guidance schedules
* **Speculative Decoding**: Accelerated autoregressive generation
* **Rectified Flow++**: Iterative reflow for straighter trajectories

**Medical Domain**

* **Rasterization**: Space-filling curves (Hilbert, Z-order) for spatial-to-sequence
* **Inpainting**: Fill missing regions in medical images
* **Super-Resolution**: Upscale low-resolution medical scans
* **Clinical Metrics**: Dice, IoU, PSNR, SSIM evaluation

Quick Links
===========

* `Installation <getting_started.html#installation>`__
* `Train a Model <tutorials/01_training.html>`__
* `Generate Samples <tutorials/02_generation.html>`__
* `API Reference <api/autoregressive.html>`__
* `Research Methods <research/implemented_methods.html>`__
