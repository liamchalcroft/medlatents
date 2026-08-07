#!/usr/bin/env python3
"""Profile medlatents models to identify bottlenecks."""

import torch
from torch.profiler import ProfilerActivity, profile


def profile_autoregressive():
    from medlatents.autoregressive import AutoregressiveTransformer

    print("Profiling AutoregressiveTransformer...")
    model = AutoregressiveTransformer(
        seq_length=256,
        vocab_size=512,
        hidden_size=128,
        depth=4,
        num_heads=4,
    )
    model.eval()
    x = torch.randint(0, 512, (4, 256))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

    prof.export_chrome_trace("profile_autoreg.json")
    print("  Trace saved to: profile_autoreg.json")


def profile_maskgit():
    from medlatents.maskgit import MaskGIT

    print("Profiling MaskGIT...")
    model = MaskGIT(
        seq_length=256,
        vocab_size=512,
        hidden_size=128,
        depth=4,
        num_heads=4,
    )
    model.eval()
    x = torch.randint(0, 512, (4, 256))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

    prof.export_chrome_trace("profile_maskgit.json")
    print("  Trace saved to: profile_maskgit.json")


def profile_discrete_dit():
    from medlatents.networks import DiscreteDiT

    print("Profiling DiscreteDiT...")
    model = DiscreteDiT(
        seq_length=256,
        vocab_size=512,
        hidden_size=128,
        depth=4,
        num_heads=4,
    )
    model.eval()
    x = torch.randint(0, 512, (4, 256))
    t = torch.rand(4)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            with torch.no_grad():
                _ = model(x, t)

    prof.export_chrome_trace("profile_dit.json")
    print("  Trace saved to: profile_dit.json")


if __name__ == "__main__":
    profile_autoregressive()
    profile_maskgit()
    profile_discrete_dit()
    print("\nProfiling complete. Open chrome://tracing and load .json files.")
