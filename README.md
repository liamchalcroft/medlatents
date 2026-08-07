# medlatents

Discrete and continuous latent generative models for medical imagery, designed to work with pretrained tokenizers from `medtokenizers`.

## Ecosystem

`medlatents` is one of three packages that together form a toolkit for generative modeling of medical images:

- **[medrs](https://github.com/liamchalcroft/med-rs)** -- fast medical-image I/O (NIfTI/DICOM) and preprocessing primitives.
- **[medtokenizers](https://github.com/liamchalcroft/medtokenizers)** -- continuous and discrete tokenizers that compress 2D/3D scans into latents or token grids.
- **[medlatents](https://github.com/liamchalcroft/medlatents)** -- generative models (autoregressive, MaskGIT, diffusion, flow matching, Bayesian flow) that learn over those tokens *(this package)*.

Typical pipeline:

```text
images -> medrs (load) -> medtokenizers (tokenize) -> medlatents (generate) -> medtokenizers (detokenize) -> images
```

## Features

### Core Models
- **Autoregressive Transformer**: Unidirectional generation with causal attention
- **MaskGIT**: Bidirectional masked transformer for parallel decoding
- **Discrete DiT**: Diffusion transformer for discrete latent spaces (D3PM)
- **Flow Matching**: Discrete and continuous flow-based generation
- **Bayesian Flow Networks**: Probabilistic flow models
- **Rasterization**: Space-filling curves (Hilbert, Z-order) for locality-preserving spatial-to-sequence conversion

### Advanced Features (2024-2025)
- **Halton Scheduler**: Spatially-dispersed unmasking for MaskGIT using low-discrepancy sequences
- **KLASS (KL-Adaptive Stability Sampling)**: Early stopping based on KL divergence for MaskGIT and D3PM
- **Guidance Schedules**: Time-dependent classifier-free guidance (constant, linear, cosine, triangular)
- **Decoupled ST-Gumbel-Softmax**: Separate temperatures for forward/backward passes
- **RF++ (Rectified Flow++)**: Iterative reflow for straighter trajectories with higher-order ODE solvers
- **BFN Improvements**: Entropy encoding, score-based guidance, and particle-based sampling from AbBFN2
- **Speculative Decoding**: Accelerated autoregressive generation using draft models

## Installation

```bash
# Python 3.10+ is required (modern type hints)
python -m pip install -e .
python -m pip install -e ".[dev]"  # For development
```

Requires `medtokenizers` and `medrs` for tokenization and I/O operations.

## Quickstart

For a self-contained, CPU-only example that trains and samples a tiny model in
under a minute (no GPU, no medical data, no tokenizer required), run:

```bash
python examples/quickstart.py
```

It builds a `nano` autoregressive transformer, trains a few steps on random
integer tokens, samples a batch, and prints the loss and output shape. See
[`examples/quickstart.py`](examples/quickstart.py).

For the whole loop in one file (pre-train tokenizer, tokenize the dataset,
train a generator on the tokens, generate, compute FID-192), run

```bash
python examples/end_to_end_pipeline.py
```

See [`examples/end_to_end_pipeline.py`](examples/end_to_end_pipeline.py). It
runs on CPU; pass `--dataset chestmnist` to swap the toy images for real data.

## Usage

### Training

```bash
# Unified training script
python scripts/train.py \
    --model_type autoreg \
    --model_size nano \
    --train_pattern "data/train/*.npy" \
    --val_pattern "data/val/*.npy" \
    --tokenizer_path tokenizer.pt \
    --vocab_size 8192 \
    --seq_length 16384

# Add --deterministic for reproducible runs (disables TF32 and cuDNN benchmark)

# Model-specific scripts also available in scripts/autoreg/, scripts/maskgit/, scripts/flow/
```

The unified training script expects a discrete tokenizer checkpoint (`.pt`) for on-the-fly tokenization.
Valid `--model_type` values: `autoreg`, `maskgit`, `flow`, `d3pm`, `bayesian_flow`.

### Generation

```bash
python scripts/generate.py \
    --model_type autoreg \
    --model_path checkpoints/model.pt \
    --tokenizer_path tokenizer.pt \
    --output_dir samples/
```

### Benchmarking

```bash
# Benchmark all model types (forward pass + sampling speed)
python scripts/benchmark_models.py
```

See `PERF_SCOREBOARD.md` for measured baselines and methodology.

### Profiling

```bash
# Generate Chrome traces for hot path analysis
python scripts/profile_models.py

# Open in Chrome: chrome://tracing
# Load profile_autoreg.json, profile_maskgit.json, or profile_dit.json
```

## Project Structure

```
src/medlatents/
├── autoregressive/   # Autoregressive transformers + speculative/Medusa decoding
├── bayesian_flow/    # Bayesian Flow Networks (entropy encoding, guided sampling, solvers)
├── common/           # Shared low-level utilities (schedules, helpers)
├── conditioning/     # ConditioningBundle and frozen pretrained encoders
├── config_models/    # Optional pydantic experiment configuration models
├── data/             # Datasets, dataloaders, and tokenizer wrappers
├── diffusion/        # D3PM discrete diffusion, continuous Gaussian diffusion, SEDD/MDLM
├── evaluation/       # FID, precision/recall, NLL, memorization probes
├── flow_matching/    # Discrete and continuous flow matching (RF, RF++)
├── generation/       # High-level checkpoint-driven generation API
├── inference/        # Inpainting, super-resolution, quantization, pruning, KV-cache
├── maskgit/          # Bidirectional masked transformer (MaskGIT)
├── networks/         # Shared transformer / DiT building blocks
├── post_training/    # DPO/SPO, RL (DDPO/GRPO/GARDO), distillation, reward models, self-play
├── rasterization/    # Spatial-to-sequence conversion (Hilbert, Z-order curves)
├── sampling/         # Sampling strategies, schedulers, guidance, early stopping
├── training/         # Training loops, schedulers, curricula, distributed training
├── utils/            # Special tokens and utilities
└── configs.py        # Model-size presets (MODEL_CONFIGS, MODEL_SIZES)

examples/            # Runnable examples, including the one-file end-to-end pipeline
scripts/             # Training, generation, evaluation, and benchmarking entry points
tests/               # Test suite
```

## Advanced Usage

### Halton Scheduler for MaskGIT

```python
from medlatents.sampling.maskgit import MaskGITScheduler, halton_schedule_1d

# Use Halton sequence for spatially-dispersed unmasking
scheduler = MaskGITScheduler(
    num_steps=16,
    mask_schedule="halton",  # Low-discrepancy spatial dispersion
)
```

### KLASS Early Stopping

```python
from medlatents.sampling.maskgit import MaskGITScheduler, KLASSGenerator
from medlatents.maskgit import MaskGIT

# Set up KLASS for MaskGIT
scheduler = MaskGITScheduler(num_steps=16)
klass = KLASSGenerator(
    scheduler=scheduler,
    kl_threshold=0.01,  # Stop when KL divergence falls below threshold
    min_steps=8,  # Minimum steps before early stopping
)

# Generation with early stopping
model = MaskGIT(...)
x = torch.full((1, 256), model.mask_token)
generated, metrics = klass.generate(model, initial_tokens=x, mask_token=model.mask_token)
print(f"Steps taken: {metrics['steps_taken']}")
```

### Time-Dependent Classifier-Free Guidance

```python
from medlatents.sampling import GuidedSampler, cosine_guidance

# Apply time-dependent CFG with cosine schedule
sampler = GuidedSampler(
    model=model,
    guidance_schedule=cosine_guidance,
    base_scale=5.0,
)

guided_logits = sampler(x, t, conditioning={"class_id": labels}, progress=0.5)
```

### Higher-Order ODE Solvers for Rectified Flow

```python
from medlatents.flow_matching.continuous import RectifiedFlow

flow = RectifiedFlow(latent_shape=(latent_dim,))

# Sample with RK4 solver for better accuracy
samples = flow.sample(
    model=velocity_model,
    batch_size=4,
    num_steps=50,
    solver="rk4",  # Options: "euler", "heun", "rk4"
)
```

### Rectified Flow++ (Reflow)

```python
from medlatents.flow_matching.continuous import RectifiedFlowPP

# Initialize RF++ with 2 reflow iterations
rfpp = RectifiedFlowPP(
    num_reflow_iterations=2,
    latent_shape=(latent_dim,),
)

# First iteration: standard RF
for epoch in range(epochs):
    loss = rfpp.compute_reflow_loss(model, x1)
    loss.backward()

# Advance to next iteration
rfpp.advance_iteration()

# Subsequent iterations: train on learned trajectories
for epoch in range(epochs):
    loss = rfpp.compute_reflow_loss(model, x1)
    loss.backward()
```

### BFN with Entropy Encoding

```python
from medlatents.bayesian_flow import compute_entropy, exponential_schedule

# Compute normalized entropy for monitoring uncertainty
entropy = compute_entropy(logits, normalize=True)

# Use exponential accuracy schedule
t = torch.linspace(0, 1, num_steps)
alpha, beta = exponential_schedule(t)
```

### Speculative Decoding for Autoregressive Models

```python
from medlatents.autoregressive import SpeculativeDecoder

# Create speculative decoder with draft model
decoder = SpeculativeDecoder(
    main_model=large_model,  # Larger, accurate model
    draft_model=small_model,  # Smaller, faster model
    vocab_size=vocab_size,
    max_speculation=4,  # Speculate 4 tokens ahead
    temperature=1.0,
)

# Generate with acceleration
tokens, metrics = decoder.generate(
    initial_tokens=context,
    max_new_tokens=100,
    eos_token=eos_id,
)

print(f"Acceptance rate: {metrics['acceptance_rate']:.2%}")
# Higher acceptance rate = better speedup from draft model
```

### Post-Training (Alignment, RL, Distillation)

A unified post-training stack works across every architecture: preference
optimization (DPO/SPO), reinforcement learning (DDPO/GRPO/GARDO), distillation
(reflow, consistency), reward modeling, and self-play (SPIN/RFT).

```python
from medlatents.post_training import (
    PostTrainingConfig,
    PreferenceDataset,
    AutoregressiveDPOTrainer,
)

# Build preference pairs by ranking your model's own samples
dataset = PreferenceDataset.from_generations(
    generator=model,
    prompts=prompts,
    reward_fn=reward_fn,  # higher score = preferred
    num_samples_per_prompt=4,
)

# Align with DPO
config = PostTrainingConfig(method="dpo", beta=0.1, lr=1e-5, use_fsdp=False)
trainer = AutoregressiveDPOTrainer(model=model, ref_model=ref_model, config=config)
trainer.train(dataset)
```

See the [Post-Training guide](docs/guides/post_training.rst) for RL, distillation,
and self-play workflows.

## Testing

```bash
pytest tests/
```

## Documentation

Full documentation is available at `docs/` or online at https://liamchalcroft.github.io/medlatents/

Build docs locally:
```bash
cd docs
make html
open _build/html/index.html
```

## Dependencies

- PyTorch >= 2.0
- NumPy >= 1.23
- einops >= 0.6
- accelerate >= 0.20
- medtokenizers (for tokenization)
- medrs (for medical image I/O)

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for
the development setup, coding standards, and pull-request workflow.

## Citation

If you use MedLatents in your research, please cite both the accompanying
paper and the software. Citation metadata is provided in
[`CITATION.cff`](CITATION.cff):

```bibtex
@article{chalcroft2026coupling,
  author = {Chalcroft, Liam},
  title  = {Tokenizer--Generator Coupling in Medical Image Generation},
  year   = {2026},
  note   = {arXiv preprint, to appear}
}

@software{medlatents2026,
  title  = {MedLatents: Discrete and Continuous Latent Generative Models for Medical Imagery},
  author = {Chalcroft, Liam},
  year   = {2026},
  url    = {https://github.com/liamchalcroft/medlatents}
}
```

MedLatents accompanies the paper *Tokenizer-Generator Coupling in Medical
Image Generation*, a controlled factorial study of medical
image tokenizers, latent generators, and sampling budgets. The paper and its
companion library [medtokenizers](https://github.com/liamchalcroft/medtokenizers)
share the same citation.
