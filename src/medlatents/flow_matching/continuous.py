"""Continuous flow-matching utilities for latent spaces.

Implements:
- Rectified flow with straight-line trajectories
- RF++ (Reflow iteration for straighter paths)
- Higher-order ODE solvers (Euler, Heun, RK4)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..sampling.flow_diffusion import sample_with_cfg_flow

# ============================================================================
# ODE Solvers
# ============================================================================


def euler_step(
    model,
    x: torch.Tensor,
    t: float,
    dt: float,
    y=None,
    **model_kwargs,
) -> torch.Tensor:
    """
    Euler step for ODE integration.

    Args:
        model: Velocity field v(x, t)
        x: Current state
        t: Current time
        dt: Step size
        y: Conditioning

    Returns:
        Next state x + v(x, t) * dt
    """
    t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    velocity = _call_model(model, x, t_tensor, y=y, **model_kwargs)
    return x + velocity * dt


def heun_step(
    model,
    x: torch.Tensor,
    t: float,
    dt: float,
    y=None,
    **model_kwargs,
) -> torch.Tensor:
    """
    Heun's method (2nd order Runge-Kutta) for ODE integration.

    Predictor-corrector method with better accuracy than Euler.

    Args:
        model: Velocity field v(x, t)
        x: Current state
        t: Current time
        dt: Step size
        y: Conditioning

    Returns:
        Next state using Heun's method
    """
    t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    t_next_tensor = torch.full((x.shape[0],), t + dt, device=x.device, dtype=x.dtype)

    # Predictor (Euler step)
    v1 = _call_model(model, x, t_tensor, y=y, **model_kwargs)
    x_pred = x + v1 * dt

    # Corrector
    v2 = _call_model(model, x_pred, t_next_tensor, y=y, **model_kwargs)

    # Average of slopes
    return x + (v1 + v2) * dt / 2


def rk4_step(
    model,
    x: torch.Tensor,
    t: float,
    dt: float,
    y=None,
    **model_kwargs,
) -> torch.Tensor:
    """
    4th order Runge-Kutta method for ODE integration.

    Higher accuracy than Euler or Heun at cost of 4 model evaluations.

    Args:
        model: Velocity field v(x, t)
        x: Current state
        t: Current time
        dt: Step size
        y: Conditioning

    Returns:
        Next state using RK4
    """
    t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

    # k1
    k1 = _call_model(model, x, t_tensor, y=y, **model_kwargs)

    # k2
    t_half = torch.full((x.shape[0],), t + dt / 2, device=x.device, dtype=x.dtype)
    k2 = _call_model(model, x + k1 * dt / 2, t_half, y=y, **model_kwargs)

    # k3
    k3 = _call_model(model, x + k2 * dt / 2, t_half, y=y, **model_kwargs)

    # k4
    t_next = torch.full((x.shape[0],), t + dt, device=x.device, dtype=x.dtype)
    k4 = _call_model(model, x + k3 * dt, t_next, y=y, **model_kwargs)

    # Weighted average
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def _call_model(model, x: torch.Tensor, t: torch.Tensor, y=None, **kwargs):
    """Call model with standardized (x, t, y) signature."""
    return model(x=x, t=t, y=y, **kwargs)


class RectifiedFlow:
    """Rectified flow matching for continuous latent variables.

    Rectified flow learns straight-line trajectories between noise and data.
    Instead of curved diffusion paths, the flow follows
    ``x(t) = (1-t)·x_0 + t·x_1``,
    where x_0 is Gaussian noise and x_1 is the data distribution.
    The model learns to predict the constant velocity: ``v = x_1 - x_0``.

    Reference:
        Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer
        Data with Rectified Flow" https://arxiv.org/abs/2209.03003
    """

    def __init__(
        self,
        latent_shape: tuple[int, ...] | None = None,
        base_std: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.latent_shape = latent_shape
        self.base_std = base_std
        self.device = device or torch.device("cpu")

    def _ensure_shape(self, x: torch.Tensor) -> None:
        if self.latent_shape is None:
            self.latent_shape = tuple(x.shape[1:])

    def sample_base(self, batch_size: int) -> torch.Tensor:
        if self.latent_shape is None:
            raise ValueError(
                "latent_shape unknown. Call compute_loss at least once before sampling or set explicitly."
            )
        shape = (batch_size, *self.latent_shape)
        return torch.randn(shape, device=self.device) * self.base_std

    def sample_path(
        self,
        x0: Float[torch.Tensor, "batch ..."],
        x1: Float[torch.Tensor, "batch ..."],
        t: torch.Tensor,
    ) -> tuple[Float[torch.Tensor, "batch ..."], Float[torch.Tensor, "batch ..."]]:
        """Sample a point along the straight-line path from x_0 to x_1.

        The rectified flow follows a straight path: x(t) = (1-t)·x_0 + t·x_1
        with constant velocity: v = dx/dt = x_1 - x_0

        Args:
            x0: Starting point (noise) [batch, ...]
            x1: Ending point (data) [batch, ...]
            t: Time in [0, 1] [batch]

        Returns:
            x_t: Point on path at time t
            target_v: Constant velocity vector (x_1 - x_0)
        """
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)
        x_t = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0
        return x_t, target_v

    def compute_loss(
        self,
        model,
        x1: Float[torch.Tensor, "batch ..."],
        y=None,
        base_samples: Float[torch.Tensor, "batch ..."] | None = None,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Compute rectified flow training loss.

        The model learns to predict the velocity field v(x_t, t) = x_1 - x_0
        that pushes noise x_0 to data x_1 along straight paths.

        Loss: E_{t~U(0,1), x_0~p_0, x_1~p_data} ||v_θ(x_t, t) - (x_1 - x_0)||²

        Args:
            model: Velocity prediction network
            x1: Data samples [batch, ...]
            y: Optional conditioning (e.g., class labels)
            base_samples: Optional pre-sampled noise x_0. Must have same shape as x1.
            model_kwargs: Additional model arguments

        Returns:
            MSE loss between predicted and true velocity

        Raises:
            ValueError: If base_samples shape doesn't match x1 shape.
        """
        self._ensure_shape(x1)
        model_kwargs = model_kwargs or {}
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)

        if base_samples is not None:
            if base_samples.shape != x1.shape:
                raise ValueError(
                    f"base_samples shape {base_samples.shape} must match x1 shape {x1.shape}"
                )
            x0 = base_samples
        else:
            x0 = torch.randn_like(x1) * self.base_std

        x_t, target_v = self.sample_path(x0, x1, t)

        pred_v = _call_model(model, x_t, t, y=y, **model_kwargs)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(
        self,
        model,
        batch_size: int,
        num_steps: int = 100,
        solver: str = "euler",
        y=None,
        guidance_scale: float = 1.0,
        null_y=None,
        initial_state: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """
        Sample from the learned flow.

        Args:
            model: Velocity field v(x, t)
            batch_size: Number of samples
            num_steps: Number of integration steps
            solver: ODE solver ('euler', 'heun', 'rk4')
            y: Conditioning
            guidance_scale: CFG scale
            null_y: Null conditioning for CFG
            initial_state: Optional initial state
            model_kwargs: Additional model arguments

        Returns:
            Generated samples
        """
        model_kwargs = model_kwargs or {}
        if initial_state is None:
            x = self.sample_base(batch_size)
        else:
            x = initial_state.to(self.device)
            if self.latent_shape is None:
                self.latent_shape = tuple(x.shape[1:])

        time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        dt = time_grid[1] - time_grid[0]

        # Select solver
        if solver == "euler":
            step_fn = euler_step
        elif solver == "heun":
            step_fn = heun_step
        elif solver == "rk4":
            step_fn = rk4_step
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Create CFG model wrapper outside loop to avoid closure issues
        use_cfg = guidance_scale != 1.0 and y is not None and null_y is not None
        if use_cfg:
            # Capture current values explicitly to avoid closure variable capture issues
            _y = y
            _null_y = null_y
            _guidance_scale = guidance_scale

            def cfg_model(x, t, **kwargs):
                return sample_with_cfg_flow(
                    model,
                    x,
                    t,
                    condition=_y,
                    guidance_scale=_guidance_scale,
                    null_condition=_null_y,
                )

            effective_model = cfg_model
        else:
            effective_model = model

        for idx in range(num_steps):
            t_scalar = float(time_grid[idx])
            x = step_fn(effective_model, x, t_scalar, float(dt), y=y, **model_kwargs)

        return x


class RectifiedFlowPP:
    """
    Rectified Flow++ (RF++): Iterative reflow for straighter paths.

    RF++ improves rectified flow by iteratively straightening the trajectories:

    1. Train initial RF model
    2. Generate trajectories using the learned model
    3. Retrain on the generated trajectories (not straight lines)
    4. Repeat for further straightening

    Reference: "Improving the Training of Rectified Flows" (arXiv:2405.20320)

    Args:
        base_rf: Base RectifiedFlow instance
        num_reflow_iterations: Number of reflow iterations
    """

    def __init__(
        self,
        base_rf: RectifiedFlow | None = None,
        num_reflow_iterations: int = 2,
        latent_shape: tuple[int, ...] | None = None,
        base_std: float = 1.0,
        device: torch.device | None = None,
    ):
        self.num_reflow_iterations = num_reflow_iterations

        if base_rf is not None:
            self.rf = base_rf
            self.latent_shape = base_rf.latent_shape
            self.base_std = base_rf.base_std
            self.device = base_rf.device
        else:
            self.rf = RectifiedFlow(
                latent_shape=latent_shape,
                base_std=base_std,
                device=device,
            )
            self.latent_shape = latent_shape
            self.base_std = base_std
            self.device = device or torch.device("cpu")

        self.current_iteration = 0

    def _ensure_shape(self, x: torch.Tensor) -> None:
        if self.latent_shape is None:
            self.latent_shape = tuple(x.shape[1:])

    def sample_base(self, batch_size: int) -> torch.Tensor:
        return self.rf.sample_base(batch_size)

    @torch.no_grad()
    def generate_trajectory(
        self,
        model,
        x0: torch.Tensor,
        x1: torch.Tensor,
        num_steps: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a trajectory from x0 to x1 using the current model.

        This trajectory is used as the training target for reflow.

        Args:
            model: Current velocity model
            x0: Starting point (noise)
            x1: Target point (data) - used to get endpoint
            num_steps: Number of integration steps

        Returns:
            x0_trajectory: Re-generated starting point
            x1: Target (same as input)
        """
        # For reflow, we generate a trajectory from x0
        # The target is still x1, but we use the model's learned path
        x = x0.clone()
        time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=x1.device)
        dt = time_grid[1] - time_grid[0]

        for idx in range(num_steps):
            t_scalar = float(time_grid[idx])
            x = euler_step(model, x, t_scalar, dt)

        return x, x1

    def compute_reflow_loss(
        self,
        model,
        x1: torch.Tensor,
        y=None,
        num_trajectory_steps: int = 50,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """
        Compute reflow loss using trajectories from the previous iteration.

        In reflow, we train on model-generated trajectories instead of straight lines.
        This progressively straightens the paths.

        Args:
            model: Velocity model
            x1: Data samples
            y: Conditioning
            num_trajectory_steps: Steps for trajectory generation
            model_kwargs: Additional model arguments

        Returns:
            Reflow loss
        """
        self._ensure_shape(x1)
        model_kwargs = model_kwargs or {}
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)

        # Sample initial noise
        x0 = torch.randn_like(x1) * self.base_std

        # First iteration: standard rectified flow on straight noise->data lines.
        # Later iterations (reflow): integrate the current model from x0 to obtain
        # x1_hat, then learn the straight path x0 -> x1_hat. Training on the model's
        # own endpoints is what progressively straightens the trajectories.
        if self.current_iteration == 0:
            x_t, target_v = self.rf.sample_path(x0, x1, t)
        else:
            x1_hat, _ = self.generate_trajectory(model, x0, x1, num_steps=num_trajectory_steps)
            x_t, target_v = self.rf.sample_path(x0, x1_hat, t)

        pred_v = _call_model(model, x_t, t, y=y, **model_kwargs)
        return F.mse_loss(pred_v, target_v)

    def advance_iteration(self) -> None:
        """Advance to next reflow iteration."""
        self.current_iteration += 1

    @torch.no_grad()
    def sample(
        self,
        model,
        batch_size: int,
        num_steps: int = 100,
        solver: str = "euler",
        y=None,
        guidance_scale: float = 1.0,
        null_y=None,
        initial_state: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Sample using the reflow-trained model."""
        return self.rf.sample(
            model=model,
            batch_size=batch_size,
            num_steps=num_steps,
            solver=solver,
            y=y,
            guidance_scale=guidance_scale,
            null_y=null_y,
            initial_state=initial_state,
            model_kwargs=model_kwargs,
        )


__all__ = [
    "RectifiedFlow",
    "RectifiedFlowPP",
    "euler_step",
    "heun_step",
    "rk4_step",
]
