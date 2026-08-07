# Contributing to medlatents

Thank you for your interest in contributing to **medlatents**! This library
provides discrete and continuous latent generative models for medical imagery
(autoregressive transformers, MaskGIT, D3PM, SEDD, discrete/continuous flow
matching, and Bayesian Flow Networks), together with a unified post-training
stack (DPO/SPO, RL, distillation, preference modeling), evaluation utilities,
and rasterization tools. It interoperates with the
[`medtokenizers`](https://github.com/liamchalcroft/medtokenizers) and
[`medrs`](https://github.com/liamchalcroft/med-rs) packages.

Contributions of all kinds are welcome: bug fixes, new model and sampler
implementations, documentation, examples, tests, and performance work. This
guide explains how to set up a development environment and the standards we
ask contributors to follow.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

> Looking for the day-to-day workflow inside the rendered docs? The same
> material is mirrored at `docs/contributing.rst`. This file is the
> authoritative version.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Linting and Formatting](#linting-and-formatting)
- [Code Style](#code-style)
- [Adding Tests](#adding-tests)
- [Adding New Features](#adding-new-features)
- [Branch and Pull Request Workflow](#branch-and-pull-request-workflow)
- [Commit Guidance](#commit-guidance)
- [Reporting Bugs and Requesting Features](#reporting-bugs-and-requesting-features)
- [License](#license)

## Development Setup

`medlatents` targets **Python 3.10+** (modern type hints are used throughout).
We use [`uv`](https://github.com/astral-sh/uv) for environment and dependency
management.

1. Fork the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/medlatents.git
   cd medlatents
   ```

2. Create and activate a virtual environment with `uv`:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in editable mode with the development extras:

   ```bash
   uv pip install -e ".[dev]"
   ```

   To also build the documentation locally, install the docs extras too:

   ```bash
   uv pip install -e ".[dev,docs]"
   ```

`medlatents` depends on `medtokenizers` and `medrs`, which are installed
automatically from their Git sources by the commands above. PyTorch (>= 2.0)
is also required; if you need a specific CUDA build, install it before the
editable install so the correct wheel is selected.

4. Add the upstream remote so you can keep your fork up to date:

   ```bash
   git remote add upstream https://github.com/liamchalcroft/medlatents.git
   ```

## Running Tests

We use [`pytest`](https://docs.pytest.org/). Run the full suite from the
repository root:

```bash
pytest tests/
```

Some tests are marked `slow` (training loops, larger sampling runs). For a fast
inner-loop while developing, deselect them:

```bash
pytest tests/ -m "not slow"
```

Other useful invocations:

```bash
# Run a single test module
pytest tests/test_autoregressive.py -v

# Run a single test by node id
pytest tests/test_sampling.py::test_nucleus_sampling -v

# Run with coverage
pytest tests/ --cov=src/medlatents --cov-report=term-missing
```

Please make sure the test suite passes before opening a pull request. New code
should come with tests (see [Adding Tests](#adding-tests)).

## Linting and Formatting

We use [`ruff`](https://docs.astral.sh/ruff/) for both linting and formatting.
The configuration lives in `pyproject.toml` (line length **100**, double
quotes, target Python 3.10).

```bash
# Lint the codebase
ruff check .

# Auto-fix lint issues that can be fixed safely
ruff check . --fix

# Format the codebase
ruff format .
```

Both `ruff check .` and `ruff format --check .` must pass cleanly in CI, so run
them locally before pushing.

## Code Style

- **Python version**: write for Python 3.10+; use modern syntax such as `X | Y`
  unions and built-in generics (`list[int]`, `dict[str, Tensor]`).
- **Type hints**: all public functions, methods, and classes must be fully type
  annotated. We use [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping)
  for tensor shape and dtype annotations, checked at runtime via `beartype`
  where appropriate. Prefer descriptive shape names that match the surrounding
  code, for example:

  ```python
  from jaxtyping import Float, Int
  from torch import Tensor


  def cross_entropy(
      logits: Float[Tensor, "batch seq vocab"],
      targets: Int[Tensor, "batch seq"],
  ) -> Float[Tensor, ""]: ...
  ```

- **Quotes**: use **double quotes** for strings (enforced by `ruff format`).
- **Line length**: keep lines within **100** characters; let `ruff format`
  handle wrapping.
- **Docstrings**: use Google-style docstrings for public APIs, documenting
  `Args`, `Returns`, and `Raises`:

  ```python
  def sample_nucleus(logits: Float[Tensor, "batch vocab"], p: float) -> Int[Tensor, "batch"]:
      """Sample from the top-p (nucleus) distribution.

      Args:
          logits: Unnormalized logits over the vocabulary.
          p: Nucleus probability mass threshold in (0, 1].

      Returns:
          Sampled token indices, one per batch element.

      Raises:
          ValueError: If ``p`` is not in the half-open interval (0, 1].
      """
      ...
  ```

- **Imports**: `ruff`'s isort integration manages import ordering;
  `medlatents` is configured as first-party.

## Adding Tests

Tests live under `tests/`, mirroring the package layout (for example,
`tests/test_autoregressive.py`, `tests/test_sampling.py`,
`tests/test_training.py`).

- Add tests for every bug fix (a regression test that fails before your fix and
  passes after) and for all new functionality.
- Keep unit tests fast and deterministic. Seed any randomness
  (`torch.manual_seed(...)`) so tests are reproducible.
- Mark genuinely expensive tests with the `slow` marker so they can be skipped
  in the fast inner loop:

  ```python
  import pytest


  @pytest.mark.slow
  def test_full_training_loop(): ...
  ```

- Where it improves coverage of edge cases, property-based tests with
  [`hypothesis`](https://hypothesis.readthedocs.io/) (a dev dependency) are
  encouraged.
- Prefer tiny model presets (for example the `nano` size) and small tensors so
  the suite stays quick on CPU.

## Adding New Features

The package is organized by method family under `src/medlatents/` (for example
`autoregressive/`, `maskgit/`, `diffusion/`, `flow_matching/`, `bayesian_flow/`,
`sampling/`, `training/`, `post_training/`, `evaluation/`, `rasterization/`).
When adding functionality:

### Adding a new model

1. Place the implementation in the appropriate method directory.
2. Inherit from the relevant base class (for example a shared discrete
   transformer base) and implement the required interface methods.
3. Add tests under `tests/test_<method>.py`.
4. Export the public symbols from the relevant `__init__.py`.
5. Update the documentation and, if user-facing, the `README.md` examples.

### Adding a new sampling strategy

1. Implement it in `src/medlatents/sampling/`.
2. Add tests in `tests/test_sampling.py`.
3. Document the new strategy and its parameters.

When in doubt about scope or design, open an issue or a discussion first so we
can agree on the approach before you invest significant effort.

## Branch and Pull Request Workflow

1. Sync your fork with upstream `main`:

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   ```

2. Create a topic branch off `main`:

   ```bash
   git checkout -b feat/short-description
   ```

3. Make your changes, adding tests and documentation as needed.

4. Run the full local checklist:

   ```bash
   ruff check .
   ruff format --check .
   pytest tests/ -m "not slow"
   ```

5. Push your branch and open a pull request against `main`:

   ```bash
   git push origin feat/short-description
   ```

6. Fill in the pull request template, including the type of change and the
   checklist (tests pass, `ruff` clean, docs updated, `CHANGELOG.md` updated).

7. Address review feedback by pushing additional commits to the same branch.
   Keep the discussion focused and the diff scoped to a single logical change.

Update the `## [Unreleased]` section of [`CHANGELOG.md`](CHANGELOG.md) with a
short entry describing any user-visible change.

## Commit Guidance

We follow the [Conventional Commits](https://www.conventionalcommits.org/)
style for commit messages and pull request titles:

- `feat: add Halton scheduler for MaskGIT`
- `fix: correct KL term in D3PM loss`
- `docs: expand post-training guide`
- `refactor: deduplicate sampler step logic`
- `test: add regression test for nucleus sampling`
- `perf: cache rotary embeddings in attention`
- `chore: bump ruff to 0.5`

Guidance:

- Use the imperative mood ("add", not "added" or "adds").
- Keep the subject line concise (roughly 72 characters) and add a body when the
  change needs explanation of *what* and *why*.
- Make each commit a coherent, self-contained change; prefer several small
  commits over one large unstructured one.

## Reporting Bugs and Requesting Features

- **Bugs**: open an issue using the **Bug report** template and include your OS,
  Python version, PyTorch version, a minimal reproduction, and the full error
  traceback.
- **Features**: open an issue using the **Feature request** template and
  describe the problem, your proposed solution, and any alternatives you
  considered.
- **Security issues**: please do **not** open a public issue. See
  [`SECURITY.md`](SECURITY.md) for responsible disclosure instructions.

## License

By contributing to `medlatents`, you agree that your contributions will be
licensed under the [MIT License](LICENSE) that covers the project.
