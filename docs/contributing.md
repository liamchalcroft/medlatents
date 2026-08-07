# MedLatents: Contributing Guide

Thank you for your interest in contributing to MedLatents!

> The authoritative contributor policy lives in `CONTRIBUTING.md` at the
> repository root (with the `CODE_OF_CONDUCT.md`). This page mirrors the
> day-to-day developer workflow for convenience.

## Getting Started

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/your-username/medlatents.git
cd medlatents
```````````````

2. Install development dependencies:
```bash
pip install -e ".[dev,docs]"
````````````````````````````

3. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
`````````````````````````````````````````````````````````````

### Running Tests

Run all tests:
```bash
pytest tests/ -v
````````````````

Run specific test module:
```bash
pytest tests/test_autoregressive.py -v
``````````````````````````````````````

Run with coverage:
```bash
pytest tests/ --cov=src/medlatents --cov-report=html
````````````````````````````````````````````````````

### Building Documentation

Build HTML documentation:
```bash
cd docs
make html
`````````

View documentation:
```bash
open _build/html/index.html
```````````````````````````

## Code Style

### Formatting

We use `ruff` for linting and formatting:

```bash
# Check formatting
ruff check src/ tests/

# Format code
ruff format src/ tests/
```````````````````````

### Type Hints

All public APIs must have type hints:

```python
from typing import Literal


def generate(
    model: torch.nn.Module,
    batch_size: int,
    temperature: float = 1.0,
) -> torch.Tensor: ...
```

Use modern Python 3.10+ type hints (e.g., `|` for unions).

### Docstrings

Use Google-style docstrings:

```python
def sample_nucleus(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Sample from top-p (nucleus) distribution.

    Args:
        logits: Logits of shape [batch, vocab_size].
        p: Nucleus threshold (0-1).

    Returns:
        Sampled token indices of shape [batch].

    Raises:
        ValueError: If p is not in (0, 1].
    """
    ...
```

## Project Structure

```
src/medlatents/
├── autoregressive/   # Autoregressive transformers + speculative/Medusa decoding
├── maskgit/          # MaskGIT bidirectional masked transformer
├── diffusion/        # D3PM, continuous Gaussian diffusion, SEDD/MDLM
├── flow_matching/    # Discrete and continuous flow matching
├── bayesian_flow/    # Bayesian Flow Networks
├── networks/         # Shared transformer/DiT building blocks
├── sampling/         # Sampling strategies, schedulers, guidance
├── training/         # Training loops, schedulers, curricula, distributed
├── post_training/    # DPO/SPO, RL, distillation, reward models, self-play
├── generation/       # High-level checkpoint-driven generation API
├── inference/        # Inpainting, super-resolution, quantization, KV-cache
├── conditioning/     # ConditioningBundle and frozen encoders
├── evaluation/       # FID, precision/recall, memorization probes
├── rasterization/    # Space-filling curves (Hilbert, Z-order)
├── data/             # Datasets, dataloaders, tokenizer wrappers
├── config_models/    # Optional pydantic experiment configs
├── configs.py        # Model-size presets (MODEL_CONFIGS, MODEL_SIZES)
├── common/           # Shared low-level utilities
└── utils/            # Special tokens and helpers
``````````````````````````````````````````````````

## Contribution Workflow

1. Create a feature branch:
```bash
git checkout -b feature/my-feature
``````````````````````````````````

2. Make your changes and commit:
```bash
git add .
git commit -m "Add my new feature"
``````````````````````````````````

3. Run tests and lint:
```bash
pytest tests/ -v
ruff check src/ tests/
``````````````````````

4. Push to your fork:
```bash
git push origin feature/my-feature
``````````````````````````````````

5. Open a pull request on GitHub

## Pull Request Guidelines

### Title Format

Use conventional commits:

* `feat: Add new feature`
* `fix: Fix bug in training loop`
* `docs: Update API documentation`
* `refactor: Improve performance`

### Description

Include:

* **What**: What does this PR do?
* **Why**: Why is this change needed?
* **How**: How does it work?
* **Testing**: What tests were added/modified?
* **Breaking changes**: Are there any breaking changes?

### Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No linting errors

## Adding New Features

### Adding a New Model

1. Create model file in appropriate directory
2. Inherit from base class (e.g., `DiscreteTransformer`)
3. Implement required methods (`forward`, `generate`)
4. Add tests in `tests/test_<model_type>.py`
5. Update API documentation
6. Add to `__init__.py` exports

### Adding a New Sampling Strategy

1. Implement in `src/medlatents/sampling/`
2. Add tests in `tests/test_sampling.py`
3. Update `sampling/README.md`
4. Add API docs

### Adding a New Scheduler

1. Implement in `src/medlatents/training/schedulers.py` or appropriate module
2. Add tests in `tests/test_training.py`
3. Add CLI flags if applicable
4. Update documentation

## Research Integration

We welcome research paper implementations. Open an issue to propose one
before starting, so the scope can be agreed up front.

### New Research Features

1. Open an issue describing the method and the evidence for it
2. Implement with comprehensive tests
3. Benchmark against baselines
4. Add documentation and examples
5. Update `docs/research/implemented_methods.rst`

## Issues

### Bug Reports

When reporting bugs, include:

* **OS and version**: e.g., Ubuntu 22.04
* **Python version**: e.g., Python 3.11
* **PyTorch version**: e.g., PyTorch 2.1.0
* **Code snippet**: Minimal reproduction
* **Error message**: Full traceback
* **Expected behavior**: What should happen
* **Actual behavior**: What actually happens

### Feature Requests

For feature requests, include:

* **Use case**: What problem are you trying to solve?
* **Proposed solution**: How should it work?
* **Alternatives**: What alternatives have you considered?
* **Additional context**: Any other relevant information

## Questions

* **GitHub Discussions**: Use for questions and ideas
* **GitHub Issues**: Use for bugs and feature requests
* **Discord/Slack**: Check if community chat exists

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Acknowledgments

Thank you for contributing to MedLatents!
