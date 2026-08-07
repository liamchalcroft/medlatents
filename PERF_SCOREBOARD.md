# PERF_SCOREBOARD.md: Performance Baselines and Benchmarks

**Last updated**: 2026-01-09
**Repository**: medlatents

---

## 1) Reference Environment

The numbers in this document are indicative synthetic CPU micro-benchmarks. They
are intended to give a rough sense of relative cost across model families, not to
serve as absolute performance figures. They are reproducible with the scripts
referenced in the methodology section below. The reference environment used to
collect them was:

- **Python**: 3.11
- **Torch**: 2.8.0
- **Device**: CPU (no CUDA)
- **Platform**: macOS-26.1-arm64-arm-64bit

Results will vary with hardware, library versions, and configuration. Absolute
throughput on GPU is expected to differ substantially.

---

## 2) Methodology

Synthetic micro-benchmarks:
- **Continuous diffusion** using `ContinuousDiT`:
  - Model: hidden=64, depth=2, heads=4, no gradient checkpointing
  - Diffusion: 50 timesteps, cosine schedule
  - Training: 5 warmup + 20 timed steps (forward+back+AdamW)
  - Inference: 2 warmup + 5 timed runs, 20 inference steps
  - Latents flattened to `[B, L, C]` for `ContinuousDiT`
- **Discrete models** using `scripts/benchmark_models.py`:
  - Model: hidden=128, depth=4, heads=4, vocab=512
  - Forward: 5 warmup + 100 timed iterations
  - Sampling: AR uses KV-cache, MaskGIT uses 12 steps, DiscreteDiT uses 50 steps
- **Discrete sampling comparisons** (custom script):
  - AR: KV-cache vs no-cache
  - MaskGIT: baseline vs KLASS (avg steps recorded)
  - D3PM: baseline vs KLASS (avg steps recorded)
- **Continuous flow** using `RectifiedFlow` + `ContinuousDiT`:
  - Model: hidden=64, depth=2, heads=4
  - Training: 5 warmup + 20 timed steps (forward+back+AdamW)
  - Inference: 2 warmup + 5 timed runs, 20 steps

---

## 3) Phase 0 Baselines (Continuous Diffusion)

| Workload | Latent Shape (C, spatial) | Seq Len | Train steps/sec | Inference samples/sec | Batch | Steps |
|---|---|---:|---:|---:|---:|---:|
| 2D | (4, 16×16) | 256 | 116.668 | 33.377 | 2 | 20 |
| 3D | (4, 4×16×16) | 1024 | 21.213 | 7.597 | 2 | 20 |

---

## 4) Phase 0 Baselines (Discrete Inference)

| Model | Batch | Seq Len | Forward (ms) | Sampling (tokens/sec) | Peak Mem (MB) |
|---|---:|---:|---:|---:|---:|
| AutoregressiveTransformer | 4 | 256 | 31.47 | 242.2 | 0.0 |
| MaskGIT | 4 | 256 | 31.25 | 2203.2 | 0.0 |
| DiscreteDiT | 4 | 256 | 23.42 | 973.4 | 0.0 |

---

## 5) Phase 0 Baselines (Discrete Sampling Comparisons)

Config: batch=4, seq_len=256, vocab=512, hidden=128, depth=4, heads=4, CPU.

| Model | Tokens/sec | Avg Steps | Notes |
|---|---:|---:|---|
| AR (KV-cache) | 3006.6 | n/a | `model.generate` |
| AR (no cache) | 747.1 | n/a | Full forward each token |
| MaskGIT | 7745.9 | 12.0 | `num_steps=12` |
| MaskGIT + KLASS | 17734.9 | 5.0 | `kl_threshold=0.01` |
| D3PM | 202.5 | 50.0 | `num_timesteps=50` |
| D3PM + KLASS | 1472.1 | 7.0 | `kl_threshold=0.01` |

---

## 6) Phase 0 Baselines (Discrete Training)

Config: batch=4, seq_len=256, vocab=512, hidden=128, depth=4, heads=4, AdamW, CPU.

| Model | Steps/sec | Notes |
|---|---:|---|
| AutoregressiveTransformer | 7.856 | Next-token cross-entropy |
| MaskGIT | 11.273 | mask_ratio=0.5, full cross-entropy |
| DiscreteFlow | 9.363 | path=polynomial(2.0), loss=cross_entropy |
| D3PM | 7.084 | timesteps=50, loss=hybrid |
| BFN | 11.514 | loss=discrete |

---

## 7) Phase 0 Baselines (Continuous Flow)

| Workload | Latent Shape (C, spatial) | Seq Len | Train steps/sec | Inference samples/sec | Batch | Steps |
|---|---|---:|---:|---:|---:|---:|
| 2D | (4, 16×16) | 256 | 151.658 | 51.647 | 2 | 20 |
| 3D | (4, 4×16×16) | 1024 | 40.731 | 10.874 | 2 | 20 |

---

## 8) Gaps to Fill Next

- Discrete sampling quality vs KLASS (speed/quality tradeoff)
- GPU baselines once CUDA is available
