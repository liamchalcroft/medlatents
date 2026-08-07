"""Self-contained discrete flow-matching utilities.

This module implements lightweight equivalents of the classes we previously
depended on from the `flow_matching` reference implementation.  The goal is not
to be feature-complete, but to provide the minimal functionality required by
medlatents for text/sequence generation.

Includes:
- Polynomial convex scheduler for path interpolation
- Mixture discrete probability path (Bernoulli interpolation)
- Generalized KL loss for discrete flow matching
- Euler solver with configurable time scheduling
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class PolynomialConvexScheduler:
    """Convex polynomial schedule used by mixture paths.

    kappa(t) = t**n, with inverse kappa^{-1}(k) = k**(1/n).
    """

    def __init__(self, n: float = 1.0) -> None:
        if n <= 0:
            raise ValueError("PolynomialConvexScheduler requires n > 0")
        self.n = float(n)

    def kappa(self, t: Tensor) -> Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        return t.clamp(0.0, 1.0) ** self.n

    def kappa_inverse(self, k: Tensor) -> Tensor:
        k = torch.as_tensor(k, dtype=torch.float32)
        return k.clamp(0.0, 1.0) ** (1.0 / self.n)


@dataclass
class PathSample:
    x_t: Tensor
    mask: Tensor


class MixtureDiscreteProbPath:
    """Simple Bernoulli mixture path between source and target tokens."""

    def __init__(self, scheduler: PolynomialConvexScheduler) -> None:
        self.scheduler = scheduler

    def sample(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> PathSample:
        if x_0.shape != x_1.shape:
            raise ValueError("x_0 and x_1 must share the same shape")

        device = x_0.device
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
        if t.ndim == 0:
            t = t.expand(x_0.size(0))

        probs = self.scheduler.kappa(t).clamp(0.0, 1.0)
        while probs.ndim < x_0.ndim:
            probs = probs.unsqueeze(-1)

        mask = torch.bernoulli(probs).bool()
        x_t = torch.where(mask, x_1, x_0)
        return PathSample(x_t=x_t, mask=mask)


class ModelWrapper(nn.Module):
    """Wraps models to expose a flow-matching friendly signature."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x=x, t=t, **extras)


class MixturePathGeneralizedKL(nn.Module):
    """Simplified generalized KL divergence for mixture paths."""

    def __init__(self, path: MixtureDiscreteProbPath, reduction: str = "mean") -> None:
        super().__init__()
        self.path = path
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        x_1: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tensor:
        del x_t, t  # Unused in the simplified formulation
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_1.reshape(-1),
            reduction="none",
        ).view_as(x_1)

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


def create_time_grid(
    num_steps: int,
    device: torch.device,
    schedule: Literal["uniform", "quadratic", "cosine"] = "uniform",
) -> Tensor:
    """Create a time grid for discrete flow sampling.

    Different schedules can improve sample quality:
    - uniform: Equal spacing (standard)
    - quadratic: More steps near t=0 (helps early denoising)
    - cosine: Smooth acceleration (similar to diffusion schedules)

    Args:
        num_steps: Number of sampling steps
        device: Device for the tensor
        schedule: Time grid schedule type

    Returns:
        Time grid from 0 to 1 with num_steps+1 points
    """
    if schedule == "uniform":
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    # Create uniform grid in [0, 1], then transform
    u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    if schedule == "quadratic":
        # t = u^2: more steps near t=0
        return u.pow(2)

    if schedule == "cosine":
        # t = 0.5 * (1 - cos(π * u)): smooth S-curve
        import math

        return 0.5 * (1 - torch.cos(math.pi * u))

    raise ValueError(f"Unknown schedule: {schedule}")


class MixtureDiscreteEulerSolver:
    """Lightweight Euler sampler for discrete mixture flows.

    Iteratively refines token predictions by sampling from model logits
    at each timestep. Supports different time grid schedules.
    """

    def __init__(
        self, model: nn.Module, path: MixtureDiscreteProbPath, vocabulary_size: int
    ) -> None:
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        time_grid: Tensor | None = None,
        time_schedule: Literal["uniform", "quadratic", "cosine"] = "uniform",
        temperature: float = 1.0,
        top_k: int | None = None,
        dtype_categorical: torch.dtype = torch.float64,
        verbose: bool = False,
    ) -> Tensor:
        """Sample from the discrete flow model.

        Args:
            x_init: Initial tokens (e.g., from source distribution)
            step_size: Step size (used if time_grid is None)
            time_grid: Optional explicit time grid
            time_schedule: Schedule for automatic time grid
            temperature: Sampling temperature (lower = more confident)
            top_k: Optional top-k filtering
            dtype_categorical: Unused (kept for API compatibility)
            verbose: Print progress

        Returns:
            Generated token sequence
        """
        del dtype_categorical  # Not used in simplified solver

        device = x_init.device
        if time_grid is None:
            num_steps = max(int(1.0 / max(step_size, 1e-6)), 1)
            time_grid = create_time_grid(num_steps, device, time_schedule)
        else:
            time_grid = time_grid.to(device)

        x = x_init.clone()

        # Ensure model is in eval mode during sampling
        was_training = self.model.training
        self.model.eval()
        try:
            for idx, t in enumerate(time_grid[:-1]):  # Don't sample at t=1
                if verbose:
                    logger.info(
                        f"[FlowSolver] step {idx + 1}/{len(time_grid) - 1} | t={float(t):.3f}"
                    )

                t_batch = torch.full((x.size(0),), float(t), device=device, dtype=torch.float32)
                logits = self.model(x=x, t=t_batch)

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(-1, top_k_indices, top_k_logits)

                probs = torch.softmax(logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(x)
                x = samples
        finally:
            if was_training:
                self.model.train()

        return x


class DiscreteFlowTrainer:
    """Helper for discrete flow matching training with configurable timestep sampling.

    This wraps the training loop logic for discrete flow matching,
    providing easy integration of different timestep sampling strategies.
    """

    def __init__(
        self,
        path: MixtureDiscreteProbPath,
        timestep_sampler: Callable[[int, torch.device], Tensor] | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            path: The discrete probability path for interpolation
            timestep_sampler: Optional custom timestep sampler function.
                Should take (batch_size, device) and return timesteps.
                If None, uses uniform sampling.
        """
        self.path = path
        self.timestep_sampler = timestep_sampler

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample timesteps for training."""
        if self.timestep_sampler is not None:
            return self.timestep_sampler(batch_size, device)
        return torch.rand(batch_size, device=device)

    def compute_loss(
        self,
        model: nn.Module,
        x_0: Tensor,
        x_1: Tensor,
        loss_fn: nn.Module | None = None,
    ) -> Tensor:
        """Compute discrete flow matching loss.

        Args:
            model: Model that takes (x, t) and returns logits
            x_0: Source tokens (from source distribution)
            x_1: Target tokens (data)
            loss_fn: Loss function (default: cross-entropy)

        Returns:
            Training loss
        """
        batch_size = x_0.size(0)
        device = x_0.device

        # Sample timesteps
        t = self.sample_timesteps(batch_size, device)

        # Interpolate along path
        path_sample = self.path.sample(t, x_0, x_1)
        x_t = path_sample.x_t

        # Get model predictions
        logits = model(x=x_t, t=t)

        # Compute loss (predict x_1 from x_t)
        if loss_fn is not None:
            return loss_fn(logits, x_1, x_t, t)

        # Default: cross-entropy loss
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_1.reshape(-1),
        )


__all__ = [
    "PolynomialConvexScheduler",
    "MixtureDiscreteProbPath",
    "MixturePathGeneralizedKL",
    "MixtureDiscreteEulerSolver",
    "ModelWrapper",
    "PathSample",
    "create_time_grid",
    "DiscreteFlowTrainer",
]
