# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-09-25

### Changed

- The README now points to the paper's reproduction repository,
  [tokenizer-generator-coupling](https://github.com/liamchalcroft/tokenizer-generator-coupling),
  which holds the experiments, result files and figure scripts.
- The paper is linked from the README (arXiv badge), from `pyproject.toml`
  (`Paper` project URL), and from `CITATION.cff`, whose preferred citation is
  now the NeurIPS 2026 conference paper (arXiv:2608.07713). The README BibTeX
  entry is updated to match.
- `scripts/tokenize_medmnist.py` imports the installed `medtokenizers` package
  instead of prepending a sibling source checkout (located via
  `MRI_DISCRETE_ROOT` / `MEDTOKENIZERS_ROOT`) to `sys.path`.

## [0.1.0] - 2026-08-07

Initial public release, accompanying the paper "Tokenizer-Generator Coupling in
Medical Image Generation".

### Added

- Generative model families for discrete and continuous medical-image latents:
  autoregressive transformer, MaskGIT, D3PM and SEDD discrete diffusion,
  discrete and continuous flow matching (incl. Rectified Flow and RF++), and
  Bayesian Flow Networks.
- Advanced sampling: Halton scheduler, KLASS early stopping, time-dependent
  classifier-free guidance schedules, decoupled ST-Gumbel-Softmax, and
  speculative decoding.
- Post-training: DPO (autoregressive / discrete / flow / diffusion, plus SPO),
  RL (GRPO / DDPO / GARDO), distillation (reflow, consistency),
  preference / reward modeling, and self-play.
- Rasterization utilities (Hilbert, Z-order, S-curve, raster scan).
- Evaluation: FID with bootstrap confidence intervals, precision/recall,
  reconstruction metrics, memorization probes, classifier utility, and
  factorial analysis.
- Inference helpers: inpainting, super-resolution, KV-cache, and quantization.
- Interoperability with `medtokenizers` tokenizers and `medrs` medical-image
  I/O, so a tokenizer trained upstream can be paired with any generator here.
- `py.typed` marker so downstream type checkers consume the library's type
  hints.
- Sphinx documentation, a CPU-only quickstart, and
  `examples/end_to_end_pipeline.py`, which runs tokenizer pre-training,
  tokenization, generator training, sampling, and FID-192 in a single file.
- Continuous integration (GitHub Actions) with CPU lint, test, and docs lanes
  across Python 3.10-3.12 on Linux and macOS, plus a `.pre-commit-config.yaml`.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CITATION.cff`, and an MIT `LICENSE`.

### Security

- All `torch.load` call sites default to `weights_only=True`, so loading a
  checkpoint does not deserialize arbitrary pickled objects.

[Unreleased]: https://github.com/liamchalcroft/medlatents/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/liamchalcroft/medlatents/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/liamchalcroft/medlatents/releases/tag/v0.1.0
