Diffusion API
=============

This module provides discrete and continuous diffusion models for generative modeling.

D3PM (Discrete Diffusion)
-------------------------

.. autoclass:: medlatents.diffusion.D3PM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.diffusion.D3PMKLASS
   :members:
   :undoc-members:
   :show-inheritance:

Continuous Diffusion
--------------------

.. autoclass:: medlatents.diffusion.ContinuousGaussianDiffusion
   :members:
   :undoc-members:
   :show-inheritance:

SEDD / MDLM
-----------

Score Entropy Discrete Diffusion and Masked Diffusion Language Models.

.. autoclass:: medlatents.diffusion.SEDDLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.diffusion.MDLMLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.diffusion.ContinuousTimeMDLM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.diffusion.ScoreEntropyLoss
   :members:
   :undoc-members:
   :show-inheritance:

Schedules
---------

Beta schedules and transition matrices for diffusion.

.. autofunction:: medlatents.diffusion.get_beta_schedule

.. autofunction:: medlatents.diffusion.get_absorbing_transition_mat

.. autofunction:: medlatents.diffusion.get_uniform_transition_mat

.. autofunction:: medlatents.diffusion.get_discretized_gaussian_transition_mat

.. autofunction:: medlatents.diffusion.compute_transition_matrices

Zero-Terminal SNR
-----------------

Improved training with zero terminal SNR (WACV 2024).

.. autofunction:: medlatents.diffusion.enforce_zero_terminal_snr

.. autofunction:: medlatents.diffusion.compute_snr

.. autofunction:: medlatents.diffusion.compute_log_snr

.. autofunction:: medlatents.diffusion.min_snr_weighting

.. autofunction:: medlatents.diffusion.get_beta_schedule_with_zero_snr

.. autofunction:: medlatents.diffusion.sample_timesteps_log_snr_importance
