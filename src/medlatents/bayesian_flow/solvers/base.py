"""Base solver class for BFN ODE/SDE solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseBFNSolver(ABC):
    """Base class for BFN ODE/SDE solvers."""

    def __init__(self, model: nn.Module, num_steps: int = 50):
        self.model = model
        self.num_steps = num_steps

    @abstractmethod
    def step(
        self,
        params: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Take one solver step.

        Args:
            params: Current distribution parameters [batch, seq, vocab]
            t: Current time [batch]
            dt: Time step size [batch]
            temperature: Sampling temperature

        Returns:
            Updated parameters
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        temperature: float = 1.0,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Generate samples using the solver.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            temperature: Sampling temperature
            return_trajectory: Whether to return full trajectory

        Returns:
            samples: [batch, seq] generated samples
            metrics: Dict with generation statistics
        """
        device = self.model.device

        params = self.model.get_prior_params(batch_size, seq_len)

        trajectory = [params] if return_trajectory else []

        times = torch.linspace(0, 1, self.num_steps + 1, device=device)

        for i in range(self.num_steps):
            t = times[i].expand(batch_size)
            dt = times[i + 1] - times[i]
            dt = dt.expand(batch_size)

            params = self.step(params, t, dt, temperature)

            if return_trajectory:
                trajectory.append(params)

        final_probs = F.softmax(params / temperature, dim=-1)
        final_probs = self.model._sanitize_probs(final_probs)
        samples = torch.multinomial(
            final_probs.view(-1, self.model.num_classes),
            num_samples=1,
        ).view(batch_size, seq_len)

        metrics = {
            "steps_taken": self.num_steps,
            "solver": self.__class__.__name__,
        }

        if return_trajectory:
            metrics["trajectory"] = trajectory

        return samples, metrics


__all__ = ["BaseBFNSolver"]
