Flow Matching API
=================

This module provides discrete and continuous flow matching for generative modeling.

Discrete Flow Matching
----------------------

Core Components
^^^^^^^^^^^^^^^

.. autoclass:: medlatents.flow_matching.MixtureDiscreteProbPath
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.MixtureDiscreteEulerSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.MixturePathGeneralizedKL
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.PolynomialConvexScheduler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.DiscreteFlowTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.flow_matching.create_time_grid

Source Distributions
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: medlatents.flow_matching.SourceDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.MaskedSourceDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.UniformSourceDistribution
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.flow_matching.get_source_distribution

.. autofunction:: medlatents.flow_matching.get_path

.. autofunction:: medlatents.flow_matching.get_loss_function

Continuous Flow Matching
------------------------

.. autoclass:: medlatents.flow_matching.RectifiedFlow
   :members:
   :undoc-members:
   :show-inheritance:

Generation
----------

.. autofunction:: medlatents.flow_matching.generate_samples

.. autofunction:: medlatents.flow_matching.generate_counterfactual

.. autoclass:: medlatents.flow_matching.WrappedModel
   :members:
   :undoc-members:
   :show-inheritance:

Timestep Sampling
-----------------

.. autofunction:: medlatents.flow_matching.sample_timesteps_uniform

.. autofunction:: medlatents.flow_matching.sample_timesteps_u_shaped

.. autofunction:: medlatents.flow_matching.sample_timesteps_logit_normal

.. autofunction:: medlatents.flow_matching.compute_log_snr

.. autofunction:: medlatents.flow_matching.get_timestep_sampler

Optimal Transport
-----------------

.. autofunction:: medlatents.flow_matching.compute_ot_coupling

.. autofunction:: medlatents.flow_matching.sample_from_coupling

.. autofunction:: medlatents.flow_matching.ot_flow_sample_path

Evaluation
----------

.. autofunction:: medlatents.flow_matching.compute_entropy

.. autofunction:: medlatents.flow_matching.estimate_likelihood

Shortcut Flow Matching
----------------------

Flexible step count generation (2024).

.. autoclass:: medlatents.flow_matching.ShortcutFlowMatchingModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.ShortcutFlowMatchingLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.AdaptiveStepSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.TimeInvariantVectorField
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.flow_matching.StepInvariantModel
   :members:
   :undoc-members:
   :show-inheritance:
