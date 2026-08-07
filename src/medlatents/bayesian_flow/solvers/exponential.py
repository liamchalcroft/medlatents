"""Exponential and stochastic integrators."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseBFNSolver


class ExponentialIntegrator(BaseBFNSolver):
    """Exponential integrator for BFN dynamics.

    Uses the matrix exponential for exact integration of the linear part,
    which is particularly effective for stiff ODEs.
    """

    def step(
        self,
        params: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        output = self.model(params, t, temperature=temperature)

        alpha_t, beta_t = self.model.get_accuracy(t, continuous_time=True)
        alpha_next, beta_next = self.model.get_accuracy(t + dt, continuous_time=True)

        K = params.size(-1)
        p_output = F.softmax(output, dim=-1)
        target = K * p_output - 1

        alpha_exp = alpha_t.view(-1, 1, 1)
        dt_exp = dt.view(-1, 1, 1)
        decay = torch.exp(-alpha_exp * dt_exp)

        params_new = decay * params + (1 - decay) * target

        return params_new


class StochasticHeun(BaseBFNSolver):
    """Stochastic Heun solver for SDE formulation.

    Adds controlled stochasticity during sampling, which can
    improve sample diversity and quality.
    """

    def __init__(
        self,
        model,
        num_steps: int = 50,
        noise_scale: float = 0.5,
    ):
        super().__init__(model, num_steps)
        self.noise_scale = noise_scale

    def step(
        self,
        params: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        dt_expanded = dt.view(-1, 1, 1)

        output_1 = self.model(params, t, temperature=temperature)
        velocity_1 = self._compute_velocity(params, output_1, t)

        noise = torch.randn_like(params) * self.noise_scale
        noise_term = torch.sqrt(dt_expanded) * noise

        params_pred = params + dt_expanded * velocity_1 + noise_term

        t_next = t + dt
        output_2 = self.model(params_pred, t_next, temperature=temperature)
        velocity_2 = self._compute_velocity(params_pred, output_2, t_next)

        velocity_avg = 0.5 * (velocity_1 + velocity_2)

        return params + dt_expanded * velocity_avg + noise_term

    def _compute_velocity(
        self,
        params: torch.Tensor,
        output_params: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        alpha, _ = self.model.get_accuracy(t, continuous_time=True)
        K = params.size(-1)
        p_output = F.softmax(output_params, dim=-1)
        alpha_exp = alpha.view(-1, 1, 1)
        return alpha_exp * (K * p_output - 1)


__all__ = ["ExponentialIntegrator", "StochasticHeun"]
