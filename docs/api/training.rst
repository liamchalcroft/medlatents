Training API
============

Training loops, schedulers, curricula, distributed-training helpers, and
performance utilities for discrete and continuous latent models.

Trainers
--------

.. autoclass:: medlatents.training.DiscreteLatentTrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.ContinuousLatentTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Schedules
---------

.. autoclass:: medlatents.training.MaskingSchedule
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.NoiseSchedule
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.training.get_cosine_schedule_with_warmup

.. autofunction:: medlatents.training.get_mask_ratio

Curriculum Learning
-------------------

.. autoclass:: medlatents.training.CurriculumScheduler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.MaskingRatioCurriculum
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.NoiseLevelCurriculum
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.SequenceLengthCurriculum
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.training.create_curriculum_from_config

Conditional Training
--------------------

.. autoclass:: medlatents.training.ConditionalTrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.ConditionalLossWrapper
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.training.prepare_batch_with_conditioning

.. autofunction:: medlatents.training.compute_conditional_loss

Representation Alignment (REPA / REPA-E)
----------------------------------------

.. autoclass:: medlatents.training.REPALoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.REPAProjection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.REPAETrainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.REPAEConfig
   :members:
   :undoc-members:
   :show-inheritance:

Checkpointing
-------------

.. autofunction:: medlatents.training.save_checkpoint

.. autofunction:: medlatents.training.load_checkpoint

.. autofunction:: medlatents.training.resume_from_checkpoint

Distributed Training
--------------------

.. autoclass:: medlatents.training.DistributedConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.training.setup_distributed

.. autofunction:: medlatents.training.wrap_model_fsdp

.. autofunction:: medlatents.training.wrap_model_ddp

Data Loading
------------

.. autoclass:: medlatents.training.DataLoaderConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.training.create_optimized_dataloader

.. autofunction:: medlatents.training.collate_variable_length

Performance and Profiling
-------------------------

.. autofunction:: medlatents.training.enable_cuda_optimizations

.. autofunction:: medlatents.training.create_fused_adamw

.. autofunction:: medlatents.training.get_optimal_dtype

.. autoclass:: medlatents.training.TrainingProfiler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.training.MemoryProfiler
   :members:
   :undoc-members:
   :show-inheritance:
