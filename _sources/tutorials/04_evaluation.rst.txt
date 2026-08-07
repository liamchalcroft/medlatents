Tutorial 4: Evaluating Models
=============================

This tutorial covers measuring the quality of a trained model: reconstruction
fidelity, generative quality (FID, precision/recall, likelihood), and a
memorization probe to check that the model is not copying training data. The
relevant API is in :doc:`../api/evaluation`.

Reconstruction metrics
----------------------

For tasks where you have a ground-truth target (reconstruction, inpainting,
super-resolution), use PSNR and SSIM:

.. code-block:: python

   from medlatents.evaluation import calculate_psnr, calculate_ssim

   psnr = calculate_psnr(prediction, target)
   ssim = calculate_ssim(prediction, target)
   print(f"PSNR: {psnr:.2f} dB  SSIM: {ssim:.4f}")

Generative quality
------------------

For unconditional or class-conditional generation, compare the distribution of
generated images to real ones. The one-call helpers cover the common cases:

.. code-block:: python

   from medlatents.evaluation import (
       calculate_fid,
       calculate_precision_recall,
       calculate_nll,
   )

   # Frechet distance between real and generated feature sets
   fid = calculate_fid(real_features, gen_features)

   # Improved precision / recall (fidelity vs. coverage)
   precision, recall = calculate_precision_recall(real_features, gen_features)

   # Negative log-likelihood for likelihood-based models
   nll = calculate_nll(model, data)

When you need to stream features over many batches, use the stateful
calculators instead:

.. code-block:: python

   from medlatents.evaluation import FIDCalculator, PrecisionRecallCalculator

   fid_calc = FIDCalculator(extractor_type="radimagenet")
   pr_calc = PrecisionRecallCalculator(k=5)

.. note::

   For medical imagery, prefer a domain-appropriate feature extractor (for
   example RadImageNet) over natural-image Inception features. The
   ``extractor_type`` argument selects the backbone.

Confidence intervals
--------------------

FID is sensitive to sample size. Bootstrap a confidence interval and a
noise floor so improvements can be judged against measurement noise:

.. code-block:: python

   from medlatents.evaluation import (
       bootstrap_fid_real_vs_gen,
       bootstrap_fid_noise_floor,
   )

   mean, lo, hi = bootstrap_fid_real_vs_gen(real_features, gen_features, n_boot=1000)
   floor = bootstrap_fid_noise_floor(real_features, n_boot=1000)
   print(f"FID {mean:.2f} (95% CI [{lo:.2f}, {hi:.2f}]), noise floor {floor:.2f}")

Memorization probe
------------------

To verify the model generalizes rather than memorizes, compare generated samples
against their nearest training neighbors in a perceptual feature space:

.. code-block:: python

   from medlatents.evaluation import (
       InceptionPool3FeatureExtractor,
       memorization_metrics,
   )

   # memorization_metrics works on precomputed features, so extract them first.
   extractor = InceptionPool3FeatureExtractor()
   gen_features = extractor.extract(generated_images)     # (N_gen, 192)
   train_features = extractor.extract(training_images)    # (N_train, 192)

   metrics = memorization_metrics(gen_features, train_features)

Low nearest-neighbor distances concentrated near zero indicate copying; a
healthy model produces a distribution shifted away from the training set. Use
:func:`~medlatents.evaluation.nearest_neighbor_gallery_pairs` to build a visual
gallery of the closest matches for inspection.

Reproducible baselines
----------------------

See ``PERF_SCOREBOARD.md`` in the repository for measured speed baselines and
the exact benchmarking methodology used in this project.

Where to go next
----------------

* :doc:`../guides/post_training` -- improve a model after evaluating it.
* :doc:`../research/implemented_methods` -- the methods behind the metrics.
