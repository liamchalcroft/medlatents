Bayesian Flow Networks API
==========================

This module provides Bayesian Flow Networks for discrete generative modeling.

Core Model
----------

.. autoclass:: medlatents.bayesian_flow.BayesianFlowTransformer
   :members:
   :undoc-members:
   :show-inheritance:

.. py:data:: medlatents.bayesian_flow.BFN_models

   Dict of preset BayesianFlowTransformer configurations (nano/small/base/large/xl).

Entropy Encoding
----------------

.. autofunction:: medlatents.bayesian_flow.compute_entropy

.. autofunction:: medlatents.bayesian_flow.encode_with_entropy

Guided Sampling
---------------

.. autoclass:: medlatents.bayesian_flow.ScoreGuidedSampler
   :members:
   :undoc-members:
   :show-inheritance:

Accuracy Schedules
------------------

.. autofunction:: medlatents.bayesian_flow.exponential_schedule

.. autofunction:: medlatents.bayesian_flow.linear_entropy_schedule

.. autofunction:: medlatents.bayesian_flow.cosine_schedule

.. autofunction:: medlatents.bayesian_flow.get_bfn_schedule

Training Utilities
------------------

.. autoclass:: medlatents.bayesian_flow.ResidualLossWrapper
   :members:
   :undoc-members:
   :show-inheritance:

ODE/SDE Solvers
---------------

Base Solver
^^^^^^^^^^^

.. autoclass:: medlatents.bayesian_flow.BaseBFNSolver
   :members:
   :undoc-members:
   :show-inheritance:

Deterministic Solvers
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: medlatents.bayesian_flow.EulerSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.bayesian_flow.HeunSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.bayesian_flow.DPMSolver2
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.bayesian_flow.DPMSolver3
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.bayesian_flow.ExponentialIntegrator
   :members:
   :undoc-members:
   :show-inheritance:

Stochastic Solvers
^^^^^^^^^^^^^^^^^^

.. autoclass:: medlatents.bayesian_flow.StochasticHeun
   :members:
   :undoc-members:
   :show-inheritance:

Solver Factory
^^^^^^^^^^^^^^

.. autofunction:: medlatents.bayesian_flow.get_bfn_solver
