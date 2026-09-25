# MedMNIST Integration for medlatents

This module provides training, generation, and evaluation pipelines for MedMNIST datasets using tokenized representations from [medtokenizers](https://github.com/liamchalcroft/medtokenizers).

## Overview

```
MedMNIST Images → medtokenizers → Tokens → medlatents → Generated → Decode → Images

Discrete path:
  64x64 → 8x8 indices (512 vocab) → Transformer → 8x8 indices → Decoder → 64x64

Continuous path:
  64x64 → 4x8x8 latents → Diffusion → 4x8x8 latents → Decoder → 64x64
```

## Tokenizer Types

| Type | Output Shape | Generation Method | Use Case |
|------|--------------|-------------------|----------|
| **VQ** | (B, H', W') or (B, H', W', D') | Transformer, MaskGIT, Flow, D3PM, BFN | Maximum quality |
| **FSQ** | (B, H', W') or (B, H', W', D') | Transformer, MaskGIT, Flow, D3PM, BFN | Stable training |
| **LFQ** | (B, H', W') or (B, H', W', D') | Transformer, MaskGIT, Flow, D3PM, BFN | Large vocabulary |
| **RESFSQ** | (B, H', W') or (B, H', W', D') | Transformer, MaskGIT, Flow, D3PM, BFN | Hierarchical |
| **VAE** | (B, C, H', W') or (B, C, H', W', D') | Continuous Diffusion | Smooth interpolation |

## Dataset Summary

### 2D Datasets (12 datasets, 28-224px)

| Dataset | Samples | Channels | Description |
|---------|---------|----------|-------------|
| pathmnist | 89K | 3 | Colorectal histology |
| chestmnist | 78K | 1 | Chest X-ray disease |
| dermamnist | 7K | 3 | Skin lesion |
| octmnist | 97K | 1 | Retinal OCT |
| pneumoniamnist | 4.7K | 1 | Pneumonia |
| retinamnist | 1K | 3 | Diabetic retinopathy |
| breastmnist | 546 | 1 | Breast ultrasound |
| bloodmnist | 12K | 3 | Blood cell |
| tissuemnist | 165K | 1 | Kidney tissue |
| organamnist | 35K | 1 | Organ slices (axial) |
| organcmnist | 13K | 1 | Organ slices (coronal) |
| organsmnist | 14K | 1 | Organ slices (sagittal) |

### 3D Datasets (6 datasets, 64³px)

| Dataset | Samples | Description |
|---------|---------|-------------|
| organmnist3d | 971 | Abdominal organs |
| nodulemnist3d | 1.2K | Lung nodules |
| adrenalmnist3d | 98 | Adrenal glands |
| fracturemnist3d | 1K | Bone fractures |
| vesselmnist3d | 1.3K | Blood vessels |
| synapsemnist3d | 1.7K | Synapse instances |

## Quick Start

### 1. Tokenize MedMNIST Data

First, tokenize your MedMNIST data using medtokenizers:

```python
from medtokenizers import DiscreteTokenizer, ContinuousTokenizer
import numpy as np

# Discrete tokenizer
tokenizer = DiscreteTokenizer(quantizer="FSQ", levels=[8, 5, 5, 5])
indices = tokenizer.tokenize(images)  # (B, H', W')

# Save tokenized data
np.savez_compressed(
    "chestmnist_tokens.npz",
    train_tokens=train_indices,
    val_tokens=val_indices,
    test_tokens=test_indices,
)

# Continuous tokenizer
tokenizer = ContinuousTokenizer(formulation="VAE", latent_channels=4)
latents = tokenizer.tokenize(images)  # (B, C, H', W')

np.savez_compressed(
    "chestmnist_latents.npz",
    train_latents=train_latents,
    val_latents=val_latents,
    test_latents=test_latents,
)
```

### 2. Train Generation Model

**2D Discrete (Transformer):**
```bash
bash scripts/train_medmnist2d_tokens.sh \
    /path/to/chestmnist_tokens.npz \
    discrete \
    transformer \
    512 \
    64 \
    50 \
    32
```

**2D Continuous (Diffusion):**
```bash
bash scripts/train_medmnist2d_tokens.sh \
    /path/to/chestmnist_latents.npz \
    continuous \
    diffusion \
    4 \
    4_8_8 \
    200 \
    16
```

**3D Discrete:**
```bash
bash scripts/train_medmnist3d_tokens.sh \
    /path/to/organmnist3d_tokens.npz \
    discrete \
    transformer \
    512 \
    512 \
    "8 8 8" \
    100 \
    8
```

### 3. Generate Samples

```bash
bash scripts/generate_medmnist.sh \
    checkpoints/medmnist2d/checkpoint_best.pt \
    discrete \
    transformer \
    100 \
    16 \
    generated_tokens.npz
```

### 4. Evaluate

```python
python examples/evaluate_medmnist.py \
    --generated generated_tokens.npz \
    --ground_truth test_tokens.npz \
    --token_type discrete \
    --vocab_size 512
```

## File Structure

```
examples/
├── end_to_end_pipeline.py       # Tokenizer -> tokens -> generator -> samples -> FID in one file
├── train_medmnist2d_tokens.py   # Train 2D generation models (transformer, maskgit, diffusion)
├── train_medmnist3d_tokens.py   # Train 3D generation models
├── generate_medmnist.py         # Generate samples from trained models
├── evaluate_medmnist.py         # Evaluate generation quality
├── benchmark_generation.py      # Benchmark throughput
└── README_medmnist.md           # This documentation

scripts/
├── autoreg/                     # Autoregressive transformer
│   ├── train.py
│   └── generate.py
├── maskgit/                     # MaskGIT bidirectional transformer
│   ├── train.py
│   └── generate.py
├── flow/                        # Discrete flow matching
│   ├── train.py
│   └── generate.py
├── d3pm/                        # D3PM discrete diffusion
│   ├── train.py
│   └── generate.py
├── bayesian_flow/               # Bayesian Flow Networks
│   ├── train.py
│   └── generate.py
├── diffusion/                   # Continuous latent diffusion
│   ├── train.py
│   └── generate.py
├── tokenize_medmnist.py         # Encode a MedMNIST dataset to tokens
├── train_medmnist2d_tokens.sh   # Shell shortcut for 2D training
├── train_medmnist3d_tokens.sh   # Shell shortcut for 3D training
└── generate_medmnist.sh         # Shell shortcut for generation
```

The paper's factorial sweeps, sampling-hyperparameter sweep and figure scripts
live in the separate
[tokenizer-generator-coupling](https://github.com/liamchalcroft/tokenizer-generator-coupling)
repository.

## Model Architectures

### Discrete Models

All discrete models work with tokenized data (VQ, FSQ, LFQ, RESFSQ):

| Model | Script | Description | Best For |
|-------|--------|-------------|----------|
| **Autoregressive** | `scripts/autoreg/train.py` | Causal, left-to-right generation | High quality, sequential |
| **MaskGIT** | `scripts/maskgit/train.py` | Bidirectional, iterative refinement | Fast parallel generation |
| **Flow Matching** | `scripts/flow/train.py` | Continuous-time discrete flows | Flexible sampling |
| **D3PM** | `scripts/d3pm/train.py` | Discrete denoising diffusion | Absorbing/uniform diffusion |
| **Bayesian Flow** | `scripts/bayesian_flow/train.py` | Information-theoretic flows | Principled uncertainty |

```python
# Common discrete model config
{
    "vocab_size": 512,
    "seq_len": 64,  # 8x8 = 64 for 64x64 input with 8x compression
    "hidden_size": 512,
    "depth": 8,
    "num_heads": 8,
}
```

### Continuous Models

For VAE-style continuous latents, use `scripts/diffusion/train.py`:

```python
{
    "latent_channels": 4,
    "seq_length": 64,  # Flattened spatial dims (8x8)
    "hidden_size": 512,
    "depth": 8,
    "num_heads": 8,
    "num_timesteps": 1000,
}
```

## Loss Functions

### Discrete Models

| Model | Loss Function |
|-------|---------------|
| Autoregressive | Cross-entropy (next token prediction) |
| MaskGIT | Cross-entropy (masked token prediction) |
| Flow Matching | Cross-entropy or Generalized KL |
| D3PM | Hybrid (VB + cross-entropy), VB only, or cross-entropy |
| Bayesian Flow | Discrete-time KL or continuous-time KL |

### Continuous Models
- **MSE loss** between predicted and target noise (epsilon prediction)
- **v-prediction** and **x0-prediction** also supported

## Evaluation Metrics

### For Discrete
- **Vocabulary usage**: unique tokens / vocabulary size
- **Generation quality**: MSE, PSNR, SSIM (with decoder)
- **Throughput**: samples/second (autoregressive)

### For Continuous
- **Latent statistics**: mean, std of generated latents
- **Distribution matching**: Wasserstein distance to training latents
- **Throughput**: samples/second (diffusion steps)

## NPZ Data Format

```python
# Discrete tokens
np.savez_compressed(
    "tokens.npz",
    train_tokens=np.array([...]),  # (N_train, H', W') or (N_train, H', W', D')
    val_tokens=np.array([...]),
    test_tokens=np.array([...]),
)

# Continuous latents
np.savez_compressed(
    "latents.npz",
    train_latents=np.array([...]),  # (N_train, C, H', W') or (N_train, C, H', W', D')
    val_latents=np.array([...]),
    test_latents=np.array([...]),
)
```

## Advanced Usage

### Custom Model Configurations

```python
from examples.train_medmnist2d_tokens import create_discrete_model
from medlatents.utils import SpecialTokenIds

# Custom vocabulary with special tokens
special_tokens = SpecialTokenIds(
    pad=0,
    bos=1,
    eos=2,
    mask=3,
)

model = create_discrete_model(args, special_tokens=special_tokens)
```

### Mixed Precision Training

```bash
python examples/train_medmnist2d_tokens.py \
    --tokens_path tokens.npz \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

### Resume Training

```bash
python examples/train_medmnist2d_tokens.py \
    --tokens_path tokens.npz \
    --resume \
    --resume_best
```

## Integration with medtokenizers

The library expects tokenized data in the standard medtokenizers output format:

1. **Discrete**: Integer indices of shape (B, H', W') for 2D or (B, H', W', D') for 3D
2. **Continuous**: Float latents of shape (B, C, H', W') for 2D or (B, C, H', W', D') for 3D

To decode generated tokens back to images:

```python
from medtokenizers import DiscreteTokenizer, ContinuousTokenizer

tokenizer = DiscreteTokenizer.from_pretrained("path/to/tokenizer")
decoded = tokenizer.decode_code(generated_indices)
```

## Benchmarking

```python
python examples/benchmark_generation.py \
    --checkpoint checkpoints/model.pt \
    --token_type discrete \
    --num_samples 100 \
    --batch_size 16
```

Expected output:
```
Total time: 12.34s
Throughput: 8.10 samples/sec
Avg latency: 123.4ms/sample
Memory usage: 2.5 GB
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (`--mixed_precision fp16`)

### NaN Loss
- Reduce learning rate
- Check input data normalization
- Enable gradient clipping (`--grad_clip 1.0`)

### Poor Generation Quality
- Increase model size (depth, hidden_size)
- Train for more epochs
- Use curriculum learning (`--use_curriculum`)

## Training Examples

### D3PM (Discrete Diffusion)

```bash
python scripts/d3pm/train.py \
    --name medmnist_d3pm \
    --train_pattern "data/train/*.npy" \
    --val_pattern "data/val/*.npy" \
    --vocab_size 512 \
    --seq_length 64 \
    --transition_type absorbing \
    --loss_type hybrid \
    --epochs 100
```

### Bayesian Flow Networks

```bash
python scripts/bayesian_flow/train.py \
    --name medmnist_bfn \
    --train_pattern "data/train/*.npy" \
    --val_pattern "data/val/*.npy" \
    --vocab_size 512 \
    --seq_length 64 \
    --num_steps 1000 \
    --beta 1.0 \
    --epochs 100
```

### Continuous Diffusion

```bash
python scripts/diffusion/train.py \
    --name medmnist_diffusion \
    --data_format npz \
    --data_path /path/to/latents.npz \
    --schedule_type cosine \
    --prediction_type epsilon \
    --num_timesteps 1000 \
    --epochs 200
```

## References

- [MedMNIST: Biomedical Images for Machine Learning](https://medmnist.com/)
- [medtokenizers](https://github.com/liamchalcroft/medtokenizers)
- [MaskGIT: Masked Generative Image Transformer](https://github.com/google-research/maskgit)
- [DiT: Diffusion Transformer](https://github.com/facebookresearch/DiT)
- [D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
- [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037)
- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595)
