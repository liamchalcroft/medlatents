Evaluation API
==============

Metrics for reconstruction quality, generative quality, memorization probing,
and bootstrap confidence intervals. The generative-quality entry points (FID,
precision/recall, NLL) are convenient one-call functions, with stateful
calculator classes for streaming features over many batches.

Generative Quality
------------------

.. autofunction:: medlatents.evaluation.calculate_fid

.. autofunction:: medlatents.evaluation.calculate_precision_recall

.. autofunction:: medlatents.evaluation.calculate_nll

.. autoclass:: medlatents.evaluation.FIDCalculator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: medlatents.evaluation.PrecisionRecallCalculator
   :members:
   :undoc-members:
   :show-inheritance:

FID Bootstrap and Feature Extraction
------------------------------------

.. autofunction:: medlatents.evaluation.fid_from_features

.. autofunction:: medlatents.evaluation.bootstrap_fid_real_vs_gen

.. autofunction:: medlatents.evaluation.bootstrap_fid_noise_floor

Reconstruction Metrics
----------------------

.. autofunction:: medlatents.evaluation.calculate_psnr

.. autofunction:: medlatents.evaluation.calculate_ssim

Memorization Probing
--------------------

.. autoclass:: medlatents.evaluation.InceptionPool3FeatureExtractor
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: medlatents.evaluation.memorization_metrics

.. autofunction:: medlatents.evaluation.memorization_pairs

.. autofunction:: medlatents.evaluation.nearest_neighbor_gallery_pairs

.. autofunction:: medlatents.evaluation.cosine_distance_matrix

Classifier Utility
------------------

.. autofunction:: medlatents.evaluation.build_grayscale_resnet18

.. autofunction:: medlatents.evaluation.train_classifier

.. autofunction:: medlatents.evaluation.evaluate_auc

.. autofunction:: medlatents.evaluation.classifier_fid
