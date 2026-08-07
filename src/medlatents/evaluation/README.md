# Evaluation Metrics for Discrete Latent Models

Comprehensive evaluation utilities for discrete latent generative models, with specialized support for medical imaging.

## Overview

This module provides two categories of metrics:

1. **Reconstruction Metrics**: Evaluate tokenizer quality
   - LPIPS (Learned Perceptual Image Patch Similarity)
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)

2. **Generative Quality Metrics**: Evaluate generative model performance
   - FID (Fréchet Inception Distance)
   - Precision/Recall (quality vs diversity trade-off)
   - NLL (Negative Log-Likelihood)

## Quick Start

### Reconstruction Metrics

```python
from medlatents.evaluation import calculate_psnr, calculate_ssim

psnr = calculate_psnr(real_images, reconstructed_images)
ssim = calculate_ssim(real_images, reconstructed_images)
print(f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.3f}")
```

### FID (Fréchet Inception Distance)

```python
from medlatents.evaluation import FIDCalculator

# Initialize with medical domain features (recommended for MRI/CT)
fid_calc = FIDCalculator(extractor_type="radimagenet", device="cuda")

# Accumulate features from real images
for batch in real_dataloader:
    fid_calc.update_real(batch)

# Accumulate features from generated images
for batch in generated_dataloader:
    fid_calc.update_generated(batch)

# Compute FID
fid_score = fid_calc.compute()
print(f"FID: {fid_score:.2f}")  # Lower is better
```

### Precision and Recall

```python
from medlatents.evaluation import PrecisionRecallCalculator

pr_calc = PrecisionRecallCalculator(k=5, extractor_type="radimagenet")

# Accumulate features (same as FID)
for batch in real_dataloader:
    pr_calc.update_real(batch)

for batch in generated_dataloader:
    pr_calc.update_generated(batch)

# Compute metrics
precision, recall = pr_calc.compute()
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
```

### Negative Log-Likelihood

```python
from medlatents.evaluation import calculate_nll

# Evaluate on held-out token sequences
nll = calculate_nll(model, token_sequences, reduction="mean")
print(f"NLL: {nll:.4f}")  # Lower is better
```

## Feature Extractors

The module supports two feature extractors for FID/Precision-Recall:

### RadImageNet (Recommended for Medical Images)

Pretrained on medical imaging data. Provides more relevant features for MRI/CT evaluation.

```python
from medlatents.evaluation import get_feature_extractor

extractor = get_feature_extractor(
    "radimagenet",
    model_path="/path/to/radimagenet_weights.pth",  # Optional
    device="cuda",
)
```

**Note**: If RadImageNet weights are not available, automatically falls back to InceptionV3 with a warning.

### InceptionV3 (Standard FID)

ImageNet-pretrained features. Provided for compatibility but suboptimal for medical images.

```python
extractor = get_feature_extractor("inception", device="cuda")
```

## Complete Evaluation Pipeline

Example combining all metrics:

```python
import torch
from torch.utils.data import DataLoader
from medlatents.evaluation import (
    ReconstructionMetrics,
    FIDCalculator,
    PrecisionRecallCalculator,
)

# Load model and tokenizer
model = load_model("checkpoints/model.pt")
tokenizer = load_tokenizer("checkpoints/tokenizer.pt")

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32)

# 1. Reconstruction metrics
recon_metrics = ReconstructionMetrics(device="cuda")
for batch in dataloader:
    reconstructed = tokenizer.decode(tokenizer.encode(batch))
    metrics = recon_metrics.compute(batch, reconstructed)
    print(
        f"LPIPS: {metrics['lpips']:.4f}, PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
    )

# 2. Generative quality metrics
fid_calc = FIDCalculator(extractor_type="radimagenet")
pr_calc = PrecisionRecallCalculator(k=5)

# Extract features from real data
for batch in dataloader:
    fid_calc.update_real(batch)
    pr_calc.update_real(batch)

# Generate and extract features
for _ in range(num_batches):
    generated = model.generate(batch_size=32)
    generated_images = tokenizer.decode(generated)
    fid_calc.update_generated(generated_images)
    pr_calc.update_generated(generated_images)

# Compute
fid = fid_calc.compute()
precision, recall = pr_calc.compute()

print(f"FID: {fid:.2f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

## API Reference

### Reconstruction Metrics

#### `calculate_lpips(real, recon, net='alex', device=None)`

Calculate LPIPS distance.

**Args:**
- `real`: Real images in range [-1, 1] or [0, 1]
- `recon`: Reconstructed images (same range as real)
- `net`: Backbone network ('alex', 'vgg', 'squeeze')
- `device`: Device to run on

**Returns:** LPIPS distance (lower is better)

#### `calculate_psnr(real, recon, max_val=1.0)`

Calculate Peak Signal-to-Noise Ratio.

**Args:**
- `real`: Real images
- `recon`: Reconstructed images
- `max_val`: Maximum pixel value (1.0 for normalized images)

**Returns:** PSNR in dB (higher is better)

#### `calculate_ssim(real, recon, window_size=11, size_average=True)`

Calculate Structural Similarity Index.

**Args:**
- `real`: Real images
- `recon`: Reconstructed images
- `window_size`: Size of Gaussian window
- `size_average`: Whether to average over spatial dimensions

**Returns:** SSIM in [0, 1] (higher is better)

#### `ReconstructionMetrics`

Compute all reconstruction metrics in one pass.

```python
metrics = ReconstructionMetrics(device="cuda", lpips_net="alex")
results = metrics.compute(real, recon)
# Returns: {'lpips': ..., 'psnr': ..., 'ssim': ...}
```

### Generative Quality Metrics

#### `calculate_fid(real_features, generated_features, eps=1e-6)`

Calculate FID from precomputed features.

**Args:**
- `real_features`: Features from real images [N_real, D]
- `generated_features`: Features from generated images [N_gen, D]
- `eps`: Small constant for numerical stability

**Returns:** FID score (lower is better)

#### `calculate_precision_recall(real_features, generated_features, k=3)`

Calculate Precision and Recall from features.

**Args:**
- `real_features`: Features from real images [N_real, D]
- `generated_features`: Features from generated images [N_gen, D]
- `k`: Number of nearest neighbors

**Returns:** `(precision, recall)` tuple in [0, 1]

#### `calculate_nll(model, tokens, reduction='mean')`

Calculate Negative Log-Likelihood.

**Args:**
- `model`: Model with forward pass returning logits
- `tokens`: Token sequences [batch, seq_len]
- `reduction`: 'mean', 'sum', or 'none'

**Returns:** NLL value (lower is better)

#### `FIDCalculator`

Handles feature extraction and FID calculation.

```python
fid_calc = FIDCalculator(
    extractor_type="radimagenet",  # or 'inception'
    device="cuda",
    model_path="/path/to/weights.pth",  # Optional, for radimagenet
)

fid_calc.update_real(real_images)
fid_calc.update_generated(generated_images)
fid = fid_calc.compute()
```

#### `PrecisionRecallCalculator`

Handles feature extraction and Precision/Recall calculation.

```python
pr_calc = PrecisionRecallCalculator(
    k=5,  # Number of nearest neighbors
    extractor_type="radimagenet",
    device="cuda",
)

pr_calc.update_real(real_images)
pr_calc.update_generated(generated_images)
precision, recall = pr_calc.compute()
```

## Best Practices

### For Medical Imaging

1. **Use RadImageNet features** instead of InceptionV3 for FID/Precision-Recall
2. **Normalize images consistently** between training and evaluation
3. **Use sufficient samples**: At least 10,000 real and 10,000 generated images for FID
4. **Report multiple metrics**: FID alone can be misleading; include Precision/Recall

### For Natural Images

1. **Use InceptionV3 features** for comparability with literature
2. Follow standard FID protocols (50,000 samples when possible)

### General

- **LPIPS is more perceptually aligned** than PSNR/SSIM for reconstruction
- **Precision/Recall complement FID**: High FID could mean low precision OR low recall
- **NLL is model-specific**: Only comparable within the same model family

## Example Script

See `scripts/evaluate_model.py` for a complete example evaluation pipeline.

## Dependencies

- `torch >= 2.0`
- `torchvision`
- `lpips >= 0.1`
- `numpy`
- `scipy`
- `jaxtyping`

## References

- FID: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS 2017)
- Precision/Recall: Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models" (NeurIPS 2019)
- LPIPS: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)
- SSIM: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP 2004)
- RadImageNet: Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset" (Radiology 2022)
