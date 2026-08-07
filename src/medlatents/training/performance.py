"""Performance optimization utilities for training.

Provides functions to enable hardware-specific optimizations:
- cuDNN benchmarking for consistent input sizes
- TF32 for Ampere+ GPUs (2-3x faster matmuls)
- Fused optimizer options
- torch.compile integration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.optim import Optimizer

__all__ = [
    "enable_cuda_optimizations",
    "create_fused_adamw",
    "compile_model",
    "get_optimal_dtype",
]


def enable_cuda_optimizations(
    benchmark: bool = True,
    tf32: bool = True,
    deterministic: bool = False,
) -> dict[str, bool]:
    """Enable CUDA performance optimizations.

    Should be called at the start of training before model creation.

    Args:
        benchmark: Enable cuDNN benchmarking. Best for consistent input sizes.
            Adds ~1-2min warmup but can give 10-30% speedup.
        tf32: Enable TensorFloat-32 for Ampere+ GPUs.
            2-3x faster matmuls with minimal precision loss.
        deterministic: Force deterministic operations.
            Disables some optimizations but ensures reproducibility.

    Returns:
        Dict of applied settings
    """
    settings = {
        "cuda_available": torch.cuda.is_available(),
        "benchmark": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "deterministic": deterministic,
    }

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic)

    if not torch.cuda.is_available():
        return settings

    # cuDNN benchmarking - finds optimal algorithms for consistent input sizes
    # Adds warmup time but speeds up subsequent runs
    if benchmark and not deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        settings["benchmark"] = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic

    # TF32 - uses 19-bit precision for 32-bit matmuls on Ampere+ (A100, 3090, 4090)
    # 2-3x faster with minimal accuracy impact for training
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings["tf32_matmul"] = True
        settings["tf32_cudnn"] = True

    return settings


def create_fused_adamw(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    foreach: bool | None = None,
) -> "Optimizer":
    """Create AdamW optimizer with fused kernels when available.

    Fused AdamW is 10-30% faster on CUDA by combining operations.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Epsilon for numerical stability
        foreach: Use foreach implementation (batched updates).
            None = auto-detect, True = force, False = disable.

    Returns:
        Configured AdamW optimizer
    """
    # Fused is fastest on CUDA (single kernel), foreach is good fallback
    use_fused = torch.cuda.is_available()

    # foreach batches parameter updates (faster than per-param on CPU)
    if foreach is None:
        foreach = not use_fused  # Use foreach if not using fused

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=use_fused,
        foreach=foreach if not use_fused else False,
    )


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str = "inductor",
) -> nn.Module:
    """Compile model with torch.compile for faster execution.

    Args:
        model: Model to compile
        mode: Compilation mode:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Minimize CUDA graph overhead (best for training)
            - "max-autotune": Maximum optimization (slower compile, faster run)
        fullgraph: Require full graph capture (fails if dynamic control flow)
        dynamic: Support dynamic shapes. None = auto, True = force, False = static.
        backend: Compiler backend ("inductor" is PyTorch's default)

    Returns:
        Compiled model (or original if compilation fails/unavailable)
    """
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "torch.compile is not available in this PyTorch build. "
            "Upgrade to PyTorch 2.x or disable compilation."
        )

    try:
        return torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
        )
    except Exception as e:
        raise RuntimeError(
            "torch.compile failed. Disable compilation or fix unsupported model ops."
        ) from e


def get_optimal_dtype(
    prefer_bf16: bool = True,
    check_cuda_capability: bool = True,
) -> torch.dtype:
    """Get optimal dtype for mixed precision training.

    Args:
        prefer_bf16: Prefer bfloat16 over float16 when available.
            bf16 has better numerical range, fp16 has better hardware support.
        check_cuda_capability: Check GPU capability for bf16 support.

    Returns:
        Recommended dtype for autocast
    """
    if not torch.cuda.is_available():
        return torch.float32

    if prefer_bf16:
        if check_cuda_capability:
            # bf16 requires compute capability >= 8.0 (Ampere+)
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                return torch.bfloat16
        else:
            # Trust the user
            return torch.bfloat16

    return torch.float16


class ZeroGradOptimizer:
    """Wrapper that uses set_to_none=True for zero_grad.

    set_to_none=True is faster than setting gradients to zero tensors
    because it avoids a memset operation.
    """

    def __init__(self, optimizer: "Optimizer"):
        self.optimizer = optimizer

    def zero_grad(self):
        """Zero gradients using set_to_none=True for performance."""
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        """Perform optimization step."""
        self.optimizer.step()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def optimize_dataloader(
    dataloader,
    pin_memory: bool | None = None,
    num_workers: int | None = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> dict:
    """Get optimal DataLoader settings for training.

    Returns a dict of recommended settings (does not modify the dataloader).

    Args:
        dataloader: Existing dataloader to analyze
        pin_memory: Pin memory for faster GPU transfer. None = auto.
        num_workers: Number of data loading workers. None = auto.
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs

    Returns:
        Dict of recommended DataLoader kwargs
    """
    import os

    settings = {}

    # Pin memory for faster CPU->GPU transfer
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    settings["pin_memory"] = pin_memory

    # Workers - use half of CPU cores, capped at 8
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8) // 2
    settings["num_workers"] = num_workers

    if num_workers > 0:
        settings["prefetch_factor"] = prefetch_factor
        settings["persistent_workers"] = persistent_workers

    return settings
