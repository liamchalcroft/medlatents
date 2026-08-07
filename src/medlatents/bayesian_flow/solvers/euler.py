"""First and second-order solvers: Euler and Heun."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseBFNSolver


class EulerSolver(BaseBFNSolver):
    """First-order Euler solver (baseline).

    Standard forward Euler integration:
        params_{t+dt} = params_t + dt * f(params_t, t)

    This is the simplest solver but requires many steps.
    """

    def step(
        self,
        params: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        output_params = self.model(params, t, temperature=temperature)

        velocity = self._compute_velocity(params, output_params, t)

        dt_expanded = dt.view(-1, 1, 1)
        return params + dt_expanded * velocity

    def _compute_velocity(
        self,
        params: torch.Tensor,
        output_params: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        alpha, beta = self.model.get_accuracy(t, continuous_time=True)

        K = params.size(-1)
        p_output = F.softmax(output_params, dim=-1)

        alpha_exp = alpha.view(-1, 1, 1)
        velocity = alpha_exp * (K * p_output - 1)

        return velocity


class HeunSolver(BaseBFNSolver):
    """Second-order Heun solver (predictor-corrector).

    Two-stage method that provides better accuracy than Euler:
        1. Predict: params_pred = params + dt * f(params, t)
        2. Correct: params_new = params + dt/2 * (f(params, t) + f(params_pred, t+dt))

    Typically requires 2x the compute per step but achieves same quality
    in ~half the steps.
    """

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
        params_pred = params + dt_expanded * velocity_1

        t_next = t + dt
        output_2 = self.model(params_pred, t_next, temperature=temperature)
        velocity_2 = self._compute_velocity(params_pred, output_2, t_next)

        velocity_avg = 0.5 * (velocity_1 + velocity_2)

        return params + dt_expanded * velocity_avg

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


__all__ = ["EulerSolver", "HeunSolver"]
