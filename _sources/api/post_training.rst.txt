Post-Training API
=================

Post-training methods for improving and aligning generative models *after*
initial pretraining: preference optimization (DPO/SPO), reinforcement learning
(DDPO/GRPO/GARDO), distillation (reflow, consistency), reward modeling, and
self-play (SPIN, RFT). All trainers accept a shared
:class:`~medlatents.post_training.PostTrainingConfig` and integrate with
``accelerate`` (optionally FSDP) for distributed training.

For a task-oriented walkthrough, see :doc:`../guides/post_training`.

Configuration
-------------

.. autoclass:: medlatents.post_training.PostTrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autodata:: medlatents.post_training.DPO_SMALL
   :annotation: = PostTrainingConfig preset

.. autodata:: medlatents.post_training.DPO_STANDARD
   :annotation: = PostTrainingConfig preset

.. autodata:: medlatents.post_training.DPO_LARGE
   :annotation: = PostTrainingConfig preset

.. autodata:: medlatents.post_training.SPO_STANDARD
   :annotation: = PostTrainingConfig preset

.. autodata:: medlatents.post_training.DDPO_STANDARD
   :annotation: = PostTrainingConfig preset

.. autodata:: medlatents.post_training.GRPO_EFFICIENT
   :annotation: = PostTrainingConfig preset

Preference Data and Reward Models
---------------------------------

.. autoclass:: medlatents.post_training.PreferenceDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.PreferencePair
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.RankedSamples
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.StepwisePreference
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.post_training.collate_preference_pairs

.. autoclass:: medlatents.post_training.RewardModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.RewardModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.post_training.create_reward_model

AI Feedback
^^^^^^^^^^^

.. autoclass:: medlatents.post_training.preference.AIPreferenceScorer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.preference.DiscriminatorScorer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.preference.StepAwarePreferenceModel
   :members:
   :undoc-members:
   :show-inheritance:

Direct Preference Optimization (DPO)
------------------------------------

.. autoclass:: medlatents.post_training.BaseDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.DPOLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.DPOOutput
   :members:
   :undoc-members:
   :show-inheritance:

Discrete models (autoregressive, MaskGIT):

.. autoclass:: medlatents.post_training.AutoregressiveDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.MaskGITDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.DiscreteDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion models (D3PM):

.. autoclass:: medlatents.post_training.D3PMDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.DiffusionDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Flow-matching models:

.. autoclass:: medlatents.post_training.DiscreteFlowDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.FlowDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Step-by-step Preference Optimization (SPO):

.. autoclass:: medlatents.post_training.SPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.StepPreferenceModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.SPOStepOutput
   :members:
   :undoc-members:
   :show-inheritance:

Reinforcement Learning
----------------------

.. autoclass:: medlatents.post_training.rl.BaseRLTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.rl.DDPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.rl.DDPODiscreteFlowTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.rl.GRPOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.rl.GARDOTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.rl.Trajectory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.rl.TrajectoryBuffer
   :members:
   :undoc-members:
   :show-inheritance:

Distillation
------------

.. autoclass:: medlatents.post_training.distillation.ReflowTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.distillation.ReflowPairGenerator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.distillation.ConsistencyTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.post_training.distillation.pseudo_huber_loss

Self-Play
---------

.. autoclass:: medlatents.post_training.self_play.SPINTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.post_training.self_play.RFTTrainer
   :members:
   :undoc-members:
   :show-inheritance:
