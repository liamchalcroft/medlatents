"""DPM-Solver variants: second and third-order multistep solvers.

Reference: Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic
Model Sampling" (NeurIPS 2022)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseBFNSolver


class DPMSolver2(BaseBFNSolver):
    """DPM-Solver-2: Second-order multistep solver.

    Uses information from previous steps for efficiency:
        params_{t+dt} = params_t + dt/2 * (3*f_t - f_{t-dt})

    Achieves high quality with very few steps (~10-20).
    """

    def __init__(self, model, num_steps: int = 20):
        super().__init__(model, num_steps)
        self.prev_velocity = None
        self.prev_t = None

    def step(
        self,
        params: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        dt_expanded = dt.view(-1, 1, 1)

        output = self.model(params, t, temperature=temperature)
        velocity = self._compute_velocity(params, output, t)

        if self.prev_velocity is None:
            params_new = params + dt_expanded * velocity
        else:
            params_new = params + dt_expanded * (1.5 * velocity - 0.5 * self.prev_velocity)

        self.prev_velocity = velocity.clone()
        self.prev_t = t.clone()

        return params_new

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

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        temperature: float = 1.0,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        self.prev_velocity = None
        self.prev_t = None
        return super().sample(batch_size, seq_len, temperature, return_trajectory)


class DPMSolver3(BaseBFNSolver):
    """DPM-Solver-3: Third-order multistep solver.

    Uses three previous evaluations for even higher accuracy.
    Can achieve excellent quality with just 10 steps.
    """

    def __init__(self, model, num_steps: int = 10):
        super().__init__(model, num_steps)
        self.velocity_history = []
        self.t_history = []
        self.max_history = 2

    def step(
        self,
        params: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        dt_expanded = dt.view(-1, 1, 1)

        output = self.model(params, t, temperature=temperature)
        velocity = self._compute_velocity(params, output, t)

        if len(self.velocity_history) == 0:
            params_new = params + dt_expanded * velocity

        elif len(self.velocity_history) == 1:
            params_new = params + dt_expanded * (1.5 * velocity - 0.5 * self.velocity_history[-1])

        else:
            v0 = velocity
            v1 = self.velocity_history[-1]
            v2 = self.velocity_history[-2]

            params_new = params + dt_expanded * (
                (23.0 / 12.0) * v0 - (16.0 / 12.0) * v1 + (5.0 / 12.0) * v2
            )

        self.velocity_history.append(velocity.clone())
        self.t_history.append(t.clone())

        if len(self.velocity_history) > self.max_history:
            self.velocity_history.pop(0)
            self.t_history.pop(0)

        return params_new

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

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        temperature: float = 1.0,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        self.velocity_history = []
        self.t_history = []
        return super().sample(batch_size, seq_len, temperature, return_trajectory)


__all__ = ["DPMSolver2", "DPMSolver3"]
