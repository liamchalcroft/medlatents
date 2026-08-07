"""torch.compile utilities for faster inference.

This module provides utilities for compiling models and sampling functions
using PyTorch 2.0+ torch.compile for significant speedups.

Usage:
    from medlatents.sampling.compile import compile_model, is_compile_available

    if is_compile_available():
        model = compile_model(model, mode="reduce-overhead")
        # Now model forward passes are compiled and faster
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    CompileMode = Literal["default", "reduce-overhead", "max-autotune"]


def is_compile_available() -> bool:
    """Check if torch.compile is available (PyTorch 2.0+)."""
    return hasattr(torch, "compile")


def compile_model(
    model: nn.Module,
    mode: CompileMode = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
    disable: bool = False,
) -> nn.Module:
    """
    Compile a model for faster inference using torch.compile.

    This is a no-op on PyTorch < 2.0.

    Args:
        model: Model to compile
        mode: Compilation mode:
            - "default": Good balance of compile time and runtime
            - "reduce-overhead": Minimize overhead, best for small models
            - "max-autotune": Maximum optimization, longer compile time
        fullgraph: If True, require entire function to be captured as single graph
        dynamic: If True, enable dynamic shape support (recommended)
        disable: If True, skip compilation (useful for debugging)

    Returns:
        Compiled model (or original if torch.compile unavailable)

    Example:
        >>> model = MaskGIT(vocab_size=8192, seq_length=1024)
        >>> model = compile_model(model, mode="reduce-overhead")
        >>> # Now generate() calls will be faster after warmup
        >>> samples = model.generate(initial_tokens)
    """
    if disable or not is_compile_available():
        return model

    return torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


def compile_function(
    fn: Callable,
    mode: CompileMode = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
    disable: bool = False,
) -> Callable:
    """
    Compile a function for faster execution using torch.compile.

    This is a no-op on PyTorch < 2.0.

    Args:
        fn: Function to compile
        mode: Compilation mode (see compile_model)
        fullgraph: If True, require entire function to be captured
        dynamic: If True, enable dynamic shape support
        disable: If True, skip compilation

    Returns:
        Compiled function (or original if torch.compile unavailable)

    Example:
        >>> @compile_function
        ... def sample_step(logits, mask):
        ...     probs = F.softmax(logits, dim=-1)
        ...     return torch.multinomial(probs, 1)
    """
    if disable or not is_compile_available():
        return fn

    return torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)


class CompiledSampler:
    """
    Wrapper that provides compiled sampling for any discrete generative model.

    This wrapper compiles the model's forward pass and provides optimized
    sampling methods. Use this for maximum inference throughput.

    Args:
        model: Model to wrap (MaskGIT, AR transformer, etc.)
        mode: Compilation mode
        warmup_steps: Number of warmup calls before measuring (compilation happens here)

    Example:
        >>> model = MaskGIT(vocab_size=8192, seq_length=1024)
        >>> sampler = CompiledSampler(model)
        >>> # First call triggers compilation (slow)
        >>> samples = sampler.generate(initial_tokens)
        >>> # Subsequent calls are fast
        >>> samples = sampler.generate(initial_tokens)
    """

    def __init__(
        self,
        model: nn.Module,
        mode: CompileMode = "reduce-overhead",
        warmup_steps: int = 3,
    ):
        self.model = model
        self.mode = mode
        self.warmup_steps = warmup_steps
        self._compiled = False
        self._compiled_forward: Callable | None = None

        # Compile the model if available
        if is_compile_available():
            self._compiled_forward = torch.compile(model, mode=mode, dynamic=True)
            self._compiled = True

    @property
    def is_compiled(self) -> bool:
        """Check if the model is compiled."""
        return self._compiled

    def forward(self, *args, **kwargs):
        """Forward pass through the (potentially compiled) model."""
        if self._compiled_forward is not None:
            return self._compiled_forward(*args, **kwargs)
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Call the model."""
        return self.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate samples using the model's generate method.

        Delegates to the underlying model's generate() method.
        """
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        raise AttributeError(f"{type(self.model).__name__} has no generate() method")

    def warmup(self, sample_input: torch.Tensor, num_steps: int | None = None) -> None:
        """
        Warmup the compiled model with sample inputs.

        This triggers compilation and optimizes the compute graph.

        Args:
            sample_input: Sample input tensor for warmup
            num_steps: Number of warmup steps (default: self.warmup_steps)
        """
        if not self._compiled:
            return

        steps = num_steps or self.warmup_steps
        with torch.no_grad():
            for _ in range(steps):
                _ = self.forward(sample_input)


__all__ = [
    "is_compile_available",
    "compile_model",
    "compile_function",
    "CompiledSampler",
]
