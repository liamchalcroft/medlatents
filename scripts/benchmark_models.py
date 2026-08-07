#!/usr/bin/env python3
"""Benchmark performance of medlatents models.

Measures:
- Forward pass latency
- Sampling speed (tokens/sec or samples/sec)
- Peak memory usage
"""

import gc
import time
from dataclasses import dataclass

import torch


@dataclass
class BenchmarkResults:
    model_name: str
    batch_size: int
    seq_length: int
    forward_latency_ms: float
    sampling_speed: float
    peak_memory_mb: float


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_peak_memory_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_autoregressive(
    batch_size: int = 4,
    seq_length: int = 256,
    vocab_size: int = 512,
    hidden_size: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    device: str = "cpu",
) -> BenchmarkResults:
    from medlatents.autoregressive import AutoregressiveTransformer

    clear_cache()
    device_obj = torch.device(device)
    model = AutoregressiveTransformer(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
    ).to(device_obj)

    model.eval()

    # Forward pass benchmark
    x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device_obj)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for i in range(5):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for i in range(100):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    forward_latency = (elapsed / 100) * 1000  # ms

    # Sampling benchmark
    prompt = torch.zeros((batch_size, 1), dtype=torch.long, device=device_obj)
    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model.generate(prompt, max_length=16, temperature=1.0)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(prompt, max_length=seq_length, temperature=1.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_generated = output.shape[1] - prompt.shape[1]
    sampling_speed = tokens_generated / elapsed  # tokens/sec
    peak_memory = get_peak_memory_mb()

    return BenchmarkResults(
        model_name="AutoregressiveTransformer",
        batch_size=batch_size,
        seq_length=seq_length,
        forward_latency_ms=forward_latency,
        sampling_speed=sampling_speed,
        peak_memory_mb=peak_memory,
    )


def benchmark_maskgit(
    batch_size: int = 4,
    seq_length: int = 256,
    vocab_size: int = 512,
    hidden_size: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    num_steps: int = 12,
    device: str = "cpu",
) -> BenchmarkResults:
    from medlatents.maskgit import MaskGIT

    clear_cache()
    device_obj = torch.device(device)
    model = MaskGIT(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
    ).to(device_obj)

    model.eval()

    # Forward pass benchmark
    x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device_obj)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for i in range(5):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for i in range(100):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    forward_latency = (elapsed / 100) * 1000  # ms

    # Sampling benchmark
    x_masked = torch.full(
        (batch_size, seq_length), model.mask_token, dtype=torch.long, device=device_obj
    )
    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model.generate(x_masked, num_steps=4, temperature=1.0)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(x_masked, num_steps=num_steps, temperature=1.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sampling_speed = output.numel() / elapsed  # tokens/sec
    peak_memory = get_peak_memory_mb()

    return BenchmarkResults(
        model_name="MaskGIT",
        batch_size=batch_size,
        seq_length=seq_length,
        forward_latency_ms=forward_latency,
        sampling_speed=sampling_speed,
        peak_memory_mb=peak_memory,
    )


def benchmark_discrete_dit(
    batch_size: int = 4,
    seq_length: int = 256,
    vocab_size: int = 512,
    hidden_size: int = 128,
    depth: int = 4,
    num_heads: int = 4,
    num_steps: int = 50,
    device: str = "cpu",
) -> BenchmarkResults:
    from medlatents.flow_matching.discrete import get_source_distribution
    from medlatents.networks import DiscreteDiT

    clear_cache()
    device_obj = torch.device(device)
    model = DiscreteDiT(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
    ).to(device_obj)

    model.eval()

    # Forward pass benchmark
    x = torch.randint(0, vocab_size, (batch_size, seq_length), device=device_obj)
    t = torch.rand(batch_size, device=device_obj)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for i in range(5):
        with torch.no_grad():
            _ = model(x, t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for i in range(100):
        with torch.no_grad():
            _ = model(x, t)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    forward_latency = (elapsed / 100) * 1000  # ms

    # Sampling benchmark using simple loop
    source_dist = get_source_distribution("uniform", vocab_size)
    model.eval()

    # Warmup
    with torch.no_grad():
        x_t = source_dist.sample((1, seq_length), device=device_obj)
        t_val = torch.rand(1, device=device_obj)
        _ = model(x_t, t_val)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        x_t = source_dist.sample((batch_size, seq_length), device=device_obj)
        for step_idx in range(num_steps):
            t_val = torch.full((batch_size,), step_idx / num_steps, device=device_obj)
            logits = model(x_t, t_val)
            probs = torch.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.reshape(-1, vocab_size), num_samples=1).view(
                batch_size, seq_length
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    sampling_speed = (batch_size * seq_length) / elapsed  # tokens/sec
    peak_memory = get_peak_memory_mb()

    return BenchmarkResults(
        model_name="DiscreteDiT",
        batch_size=batch_size,
        seq_length=seq_length,
        forward_latency_ms=forward_latency,
        sampling_speed=sampling_speed,
        peak_memory_mb=peak_memory,
    )


def print_results(results: BenchmarkResults):
    print(f"\n{'=' * 70}")
    print(f"Benchmark: {results.model_name}")
    print(f"{'=' * 70}")
    print(f"Batch size:      {results.batch_size}")
    print(f"Sequence length:  {results.seq_length}")
    print(f"Forward latency:  {results.forward_latency_ms:.2f} ms")
    print(f"Sampling speed:   {results.sampling_speed:.1f} tokens/sec")
    print(f"Peak memory:      {results.peak_memory_mb:.1f} MB")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device}")

    # Benchmark all model types
    results = [
        benchmark_autoregressive(device=device),
        benchmark_maskgit(device=device),
        benchmark_discrete_dit(device=device),
    ]

    for r in results:
        print_results(r)

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Forward (ms)':<15} {'Tokens/sec':<15} {'Memory (MB)':<15}")
    print("-" * 70)
    for r in results:
        print(
            f"{r.model_name:<25} {r.forward_latency_ms:<15.2f} {r.sampling_speed:<15.1f} {r.peak_memory_mb:<15.1f}"
        )
    print(f"{'=' * 70}\n")
