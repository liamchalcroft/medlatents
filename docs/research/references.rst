References
==========

This page collects the primary literature behind the methods implemented in
MedLatents. For where each method lives in the codebase, see
:doc:`implemented_methods`.

Generative model families
-------------------------

* **D3PM** -- Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R.
  (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces.*
  NeurIPS.
* **MaskGIT** -- Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T.
  (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR.
* **SEDD** -- Lou, A., Meng, C., & Ermon, S. (2024). *Discrete Diffusion
  Modeling by Estimating the Ratios of the Data Distribution.* ICML.
* **MDLM** -- Sahoo, S. S., et al. (2024). *Simple and Effective Masked Diffusion
  Language Models.* NeurIPS.
* **Flow Matching** -- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., &
  Le, M. (2023). *Flow Matching for Generative Modeling.* ICLR.
* **Discrete Flow Matching** -- Gat, I., et al. (2024). *Discrete Flow
  Matching.* NeurIPS.
* **Rectified Flow** -- Liu, X., Gong, C., & Liu, Q. (2023). *Flow Straight and
  Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR.
* **Bayesian Flow Networks** -- Graves, A., Srivastava, R. K., Atkinson, T., &
  Gomez, F. (2023). *Bayesian Flow Networks.* arXiv:2308.07037.
* **Diffusion Transformer (DiT)** -- Peebles, W., & Xie, S. (2023). *Scalable
  Diffusion Models with Transformers.* ICCV.

Sampling and inference
----------------------

* **Zero-Terminal-SNR** -- Lin, S., Liu, B., Li, J., & Yang, X. (2024). *Common
  Diffusion Noise Schedules and Sample Steps are Flawed.* WACV.
* **DPM-Solver** -- Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J.
  (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model
  Sampling in Around 10 Steps.* NeurIPS.
* **DDIM** -- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion
  Implicit Models.* ICLR.
* **Classifier-Free Guidance** -- Ho, J., & Salimans, T. (2022).
  *Classifier-Free Diffusion Guidance.* arXiv:2207.12598.
* **CFG-Zero*** -- Fan, W., et al. (2025). *CFG-Zero*: Improved Classifier-Free
  Guidance for Flow-Matching Models.* arXiv:2503.18886.
* **ReinMax** -- Liu, L., et al. (2023). *Bridging Discrete and Backpropagation:
  Straight-Through and Beyond.* NeurIPS.
* **RePaint** -- Lugmayr, A., et al. (2022). *RePaint: Inpainting using
  Denoising Diffusion Probabilistic Models.* CVPR.
* **Speculative Decoding** -- Leviathan, Y., Kalman, M., & Matias, Y. (2023).
  *Fast Inference from Transformers via Speculative Decoding.* ICML.
* **Medusa** -- Cai, T., et al. (2024). *Medusa: Simple LLM Inference
  Acceleration Framework with Multiple Decoding Heads.* ICML.

Training and representation alignment
-------------------------------------

* **REPA** -- Yu, S., et al. (2025). *Representation Alignment for Generation:
  Training Diffusion Transformers Is Easier Than You Think.* ICLR.
* **REPA-E** -- Leng, X., et al. (2025). *REPA-E: Unlocking VAE for End-to-End
  Tuning with Latent Diffusion Transformers.* arXiv:2504.10483.

Preference optimization
-----------------------

* **DPO** -- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your
  Language Model is Secretly a Reward Model.* NeurIPS.
* **IPO** -- Azar, M. G., et al. (2023). *A General Theoretical Paradigm to
  Understand Learning from Human Preferences.* arXiv:2310.12036.
* **KTO** -- Ethayarajh, K., et al. (2024). *KTO: Model Alignment as Prospect
  Theoretic Optimization.* ICML.
* **Diffusion-DPO** -- Wallace, B., et al. (2023). *Diffusion Model Alignment
  Using Direct Preference Optimization.* arXiv:2311.12908.
* **SPO** -- Liang, Z., et al. (2024). *Step-by-Step Preference Optimization.*
  (CVPR 2025).

Reinforcement learning
----------------------

* **DDPO** -- Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023).
  *Training Diffusion Models with Reinforcement Learning.* arXiv:2305.13301.
* **GRPO** -- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.

Distillation
------------

* **Consistency Models** -- Song, Y., Dhariwal, P., Chen, M., & Sutskever, I.
  (2023). *Consistency Models.* ICML.
* **Reflow** -- Liu, X., Gong, C., & Liu, Q. (2022). *Flow Straight and Fast:
  Learning to Generate and Transfer Data with Rectified Flow.* arXiv:2209.03003.

Self-play
---------

* **SPIN** -- Chen, Z., et al. (2024). *Self-Play Fine-Tuning Converts Weak
  Language Models to Strong Language Models.* ICML.
* **RFT** -- Yuan, Z., et al. (2023). *Scaling Relationship on Learning
  Mathematical Reasoning with Large Language Models.* arXiv:2308.01825.

Citing MedLatents
-----------------

If you use MedLatents in your research, please cite the project. See the
:doc:`../getting_started` page and the ``CITATION.cff`` file at the repository
root for the current citation.
