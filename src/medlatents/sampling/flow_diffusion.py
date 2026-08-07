"""Sampling utilities for flow matching and diffusion models.

Includes classifier-free guidance (CFG) for both flow matching and diffusion,
plus utilities for generating samples with various conditioning strategies.

Also includes Rectified Gradient Guidance (REG) from:
"REG: Rectified Gradient Guidance for Conditional Diffusion Models"
arXiv:2501.18865 (ICML'25)
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn


class GuidanceType(Enum):
    """Type of guidance to apply."""

    ORIGINAL = "original"  # Standard CFG
    REG = "reg"  # Rectified Gradient Guidance
    LINEAR = "linear"  # Linear schedule CFG
    COSINE = "cosine"  # Cosine schedule CFG
    REG_LINEAR = "reg_linear"  # REG with linear schedule
    REG_COSINE = "reg_cosine"  # REG with cosine schedule


@dataclass
class REGConfig:
    """Configuration for Rectified Gradient Guidance.

    REG modifies CFG by computing the gradient of the conditional output
    w.r.t. the input, which rectifies the guidance direction.

    Based on arXiv:2501.18865 (ICML'25).

    Attributes:
        guidance_scale: Base CFG scale (w-1 in paper notation).
        noise_schedule: Either 'ddpm' for discrete schedules (uses alpha_bar)
            or 'flow' for continuous flow matching (uses sigma/time).
        schedule_type: How to vary guidance over time:
            - 'constant': Use fixed guidance_scale
            - 'linear': Linear interpolation from guidance_scale to end_scale
            - 'cosine': Cosine schedule with power s
        end_scale: Final guidance scale for linear schedule.
        cosine_power: Power for cosine schedule (s parameter).
        interval: Optional (min, max) timestep interval for applying guidance.
            Outside this interval, only conditional prediction is used.
    """

    guidance_scale: float = 1.5
    noise_schedule: Literal["ddpm", "flow"] = "ddpm"
    schedule_type: Literal["constant", "linear", "cosine"] = "constant"
    end_scale: float = 0.0
    cosine_power: float = 1.0
    interval: tuple[float, float] | None = None


def _validate_guidance_scale(guidance_scale: float, name: str = "guidance_scale") -> None:
    if not isinstance(guidance_scale, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(guidance_scale)}")
    if not math.isfinite(guidance_scale):
        raise ValueError(f"{name} must be finite, got {guidance_scale}")


def sample_with_cfg_flow(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    null_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Classifier-Free Guidance for flow matching models.

    Args:
        model: Flow matching model
        x_t: Current state [batch, seq_len] or [batch, seq_len, dim]
        t: Time values [batch] or [batch, 1]
        condition: Conditioning tensor (class labels, text embeddings, etc.)
        guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
        null_condition: Null/unconditional token for CFG

    Returns:
        Guided model output (logits or probabilities)
    """
    _validate_guidance_scale(guidance_scale)
    if guidance_scale == 1.0 or condition is None:
        # No guidance, regular conditional generation
        return model(x_t=x_t, time=t, condition=condition)

    # Conditional prediction
    output_cond = model(x_t=x_t, time=t, condition=condition)

    # Unconditional prediction
    if null_condition is None:
        # Use None or zeros as unconditional
        output_uncond = model(x_t=x_t, time=t, condition=None)
    else:
        output_uncond = model(x_t=x_t, time=t, condition=null_condition)

    # Apply CFG: output = uncond + scale * (cond - uncond)
    return output_uncond + guidance_scale * (output_cond - output_uncond)


def sample_with_cfg_diffusion(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    null_condition: torch.Tensor | None = None,
    model_kwargs: dict | None = None,
) -> torch.Tensor:
    """
    Classifier-Free Guidance for diffusion models.

    Args:
        model: Diffusion model (predicts noise or x_0)
        x_t: Noisy sample [batch, channels, ...] or [batch, seq_len, dim]
        t: Timestep values [batch] or [batch, 1]
        condition: Conditioning tensor (class labels, text embeddings, etc.)
        guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
        null_condition: Null/unconditional token for CFG
        model_kwargs: Additional model arguments

    Returns:
        Guided model prediction (noise or x_0)
    """
    _validate_guidance_scale(guidance_scale)
    model_kwargs = model_kwargs or {}

    if guidance_scale == 1.0 or condition is None:
        # No guidance, regular conditional generation
        return model(x_t, t, condition=condition, **model_kwargs)

    # Conditional prediction
    pred_cond = model(x_t, t, condition=condition, **model_kwargs)

    # Unconditional prediction
    if null_condition is None:
        pred_uncond = model(x_t, t, condition=None, **model_kwargs)
    else:
        pred_uncond = model(x_t, t, condition=null_condition, **model_kwargs)

    # Apply CFG
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


class CFGFlowMatcher:
    """
    Wrapper for flow matching models with built-in CFG support.

    Simplifies CFG inference by handling conditional/unconditional splits.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 1.5,
        null_condition: torch.Tensor | None = None,
    ):
        _validate_guidance_scale(guidance_scale)
        self.model = model
        self.guidance_scale = guidance_scale
        self.null_condition = null_condition

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with CFG."""
        return sample_with_cfg_flow(
            self.model,
            x_t,
            t,
            condition=condition,
            guidance_scale=self.guidance_scale,
            null_condition=self.null_condition,
        )


class CFGDiffuser:
    """
    Wrapper for diffusion models with built-in CFG support.

    Simplifies CFG inference by handling conditional/unconditional splits.
    """

    def __init__(
        self,
        model: nn.Module,
        guidance_scale: float = 1.5,
        null_condition: torch.Tensor | None = None,
    ):
        _validate_guidance_scale(guidance_scale)
        self.model = model
        self.guidance_scale = guidance_scale
        self.null_condition = null_condition

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Forward pass with CFG."""
        return sample_with_cfg_diffusion(
            self.model,
            x_t,
            t,
            condition=condition,
            guidance_scale=self.guidance_scale,
            null_condition=self.null_condition,
            model_kwargs=model_kwargs,
        )


def generate_with_cfg(
    model: nn.Module,
    solver: Callable,
    x_init: torch.Tensor,
    condition: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    null_condition: torch.Tensor | None = None,
    model_type: str = "flow",
    **solver_kwargs,
) -> torch.Tensor:
    """
    Unified generation with CFG for flow matching or diffusion.

    Args:
        model: Model (flow matching or diffusion)
        solver: Solver function (e.g., MixtureDiscreteEulerSolver or scheduler.step)
        x_init: Initial noise/state
        condition: Conditioning information
        guidance_scale: CFG scale
        null_condition: Unconditional token
        model_type: 'flow' or 'diffusion'
        **solver_kwargs: Additional solver arguments

    Returns:
        Generated samples
    """
    guided_model: object  # type: ignore[assignment]
    if model_type == "flow":
        guided_model = CFGFlowMatcher(model, guidance_scale, null_condition)
    elif model_type == "diffusion":
        guided_model = CFGDiffuser(model, guidance_scale, null_condition)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'flow' or 'diffusion'.")

    # Use of solver with the guided model
    assert guided_model is not None
    return solver(guided_model, x_init, **solver_kwargs)


def rescale_cfg(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    rescale_factor: float = 0.7,
) -> torch.Tensor:
    """
    CFG with rescaling to prevent oversaturation.

    From "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    https://arxiv.org/abs/2305.08891

    Args:
        pred_cond: Conditional prediction
        pred_uncond: Unconditional prediction
        guidance_scale: CFG scale
        rescale_factor: Rescaling factor (0.7 is recommended)

    Returns:
        Rescaled CFG output
    """
    if not isinstance(guidance_scale, (int, float)):
        raise TypeError(f"guidance_scale must be a float, got {type(guidance_scale)}")
    if not math.isfinite(guidance_scale):
        raise ValueError(f"guidance_scale must be finite, got {guidance_scale}")
    if not isinstance(rescale_factor, (int, float)):
        raise TypeError(f"rescale_factor must be a float, got {type(rescale_factor)}")
    if not math.isfinite(rescale_factor):
        raise ValueError(f"rescale_factor must be finite, got {rescale_factor}")
    if not (0.0 <= rescale_factor <= 1.0):
        raise ValueError(f"rescale_factor must be between 0 and 1, got {rescale_factor}")

    # Standard CFG
    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    # Compute standard deviations
    std_cond = pred_cond.std(dim=tuple(range(1, pred_cond.ndim)), keepdim=True)
    std_guided = pred.std(dim=tuple(range(1, pred.ndim)), keepdim=True)
    if torch.any(std_guided == 0):
        raise ValueError("rescale_cfg requires non-constant guided predictions (std == 0)")

    # Rescale
    pred_rescaled = pred * (std_cond / std_guided)

    # Interpolate between original and rescaled
    return rescale_factor * pred_rescaled + (1 - rescale_factor) * pred


def sample_with_dynamic_cfg(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor | None = None,
    guidance_schedule: Callable[[float | torch.Tensor], float | torch.Tensor] = lambda t: 1.5,
    null_condition: torch.Tensor | None = None,
    model_type: str = "flow",
) -> torch.Tensor:
    """
    CFG with dynamic guidance scale based on timestep.

    Useful for adaptive guidance (e.g., stronger at early steps, weaker at late steps).

    Args:
        model: Model (flow matching or diffusion)
        x_t: Current state
        t: Time/timestep values [batch]
        condition: Conditioning tensor
        guidance_schedule: Function t -> guidance_scale (supports float or tensor input)
        null_condition: Unconditional token
        model_type: 'flow' or 'diffusion'

    Returns:
        Guided model output
    """
    t_scalar = t.mean()
    guidance_scale = guidance_schedule(t_scalar)
    if isinstance(guidance_scale, torch.Tensor):
        guidance_scale = guidance_scale.item()

    if model_type == "flow":
        return sample_with_cfg_flow(model, x_t, t, condition, guidance_scale, null_condition)
    elif model_type == "diffusion":
        return sample_with_cfg_diffusion(model, x_t, t, condition, guidance_scale, null_condition)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# Rectified Gradient Guidance (REG)
# arXiv:2501.18865 (ICML'25)
# =============================================================================


def _compute_reg_coefficient(
    x_t: torch.Tensor,
    pred_cond: torch.Tensor,
    t: torch.Tensor,
    alpha_bar: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    noise_schedule: str = "ddpm",
) -> torch.Tensor:
    """
    Compute the REG rectification coefficient.

    REG modifies the standard CFG formula:
        Standard CFG: guided = cond + w * (cond - uncond)
        REG:          guided = cond + w * (1 - noise_factor * grad) * (cond - uncond)

    Where grad = d(cond_output)/d(x_t) and noise_factor depends on the noise schedule.

    For DDPM: noise_factor = sqrt(1 - alpha_bar)
    For Flow: noise_factor = sigma (or 1 - t for linear interpolation)

    Args:
        x_t: Input tensor with gradients enabled
        pred_cond: Conditional model prediction
        t: Current timestep (normalized to [0, 1] for flow, discrete for DDPM)
        alpha_bar: Cumulative product of alphas for DDPM schedule [batch]
        sigma: Noise level for flow matching [batch]
        noise_schedule: 'ddpm' or 'flow'

    Returns:
        Rectification coefficient tensor, same shape as pred_cond
    """
    # Compute gradient of conditional prediction w.r.t. input
    # We sum over all dimensions to get a scalar for autograd
    grad = torch.autograd.grad(
        pred_cond.sum(),
        x_t,
        create_graph=False,
        retain_graph=False,
    )[0].detach()

    # Compute noise factor based on schedule
    shape = tuple([-1] + [1] * (pred_cond.ndim - 1))

    if noise_schedule == "ddpm":
        if alpha_bar is None:
            raise ValueError("alpha_bar required for DDPM noise schedule")
        # noise_factor = sqrt(1 - alpha_bar)
        noise_factor = torch.sqrt(1.0 - alpha_bar).view(*shape)
    elif noise_schedule == "flow":
        if sigma is not None:
            noise_factor = sigma.view(*shape)
        else:
            # For linear flow matching: sigma = 1 - t (noise at t=0, clean at t=1)
            noise_factor = (1.0 - t).view(*shape)
    else:
        raise ValueError(f"Unknown noise_schedule: {noise_schedule}")

    # REG coefficient: 1 - noise_factor * grad
    return 1.0 - noise_factor * grad


def _get_scheduled_scale(
    t: torch.Tensor,
    guidance_scale: float,
    schedule_type: str,
    end_scale: float = 0.0,
    cosine_power: float = 1.0,
    t_max: float = 1.0,
    output_ndim: int = 3,
) -> torch.Tensor:
    """
    Compute time-dependent guidance scale.

    Args:
        t: Current timestep [batch]
        guidance_scale: Base/start guidance scale
        schedule_type: 'constant', 'linear', or 'cosine'
        end_scale: End scale for linear schedule
        cosine_power: Power s for cosine schedule
        t_max: Maximum timestep value (1.0 for flow, 1000 for DDPM)
        output_ndim: Number of dimensions in output tensor (for proper broadcasting)

    Returns:
        Guidance scale tensor [batch, 1, 1, ...] with output_ndim dimensions
    """
    # Ensure t is 1D
    if t.ndim > 1:
        t = t.squeeze()

    if schedule_type == "constant":
        scale = torch.full_like(t, guidance_scale)

    elif schedule_type == "linear":
        # Linear from guidance_scale to end_scale as t goes from t_max to 0
        k = (end_scale - guidance_scale) / (0 - t_max)
        scale = t * k + guidance_scale

    elif schedule_type == "cosine":
        # Cosine schedule from MDT paper
        progress = 1.0 - t / t_max  # 0 to 1 as sampling progresses
        scale = (1.0 - torch.cos(math.pi * torch.pow(progress, cosine_power))) * 0.5
        scale = scale * guidance_scale

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    # Expand to match output dimensions: [batch] -> [batch, 1, 1, ...]
    for _ in range(output_ndim - 1):
        scale = scale.unsqueeze(-1)

    return scale


def sample_with_reg_diffusion(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor | None = None,
    alpha_bar: torch.Tensor | None = None,
    reg_config: REGConfig | None = None,
    null_condition: torch.Tensor | None = None,
    model_kwargs: dict | None = None,
) -> torch.Tensor:
    """
    Rectified Gradient Guidance (REG) for diffusion models.

    REG improves upon standard CFG by computing the gradient of the conditional
    output w.r.t. the input, which rectifies the guidance direction and provides
    a better approximation to the optimal guidance under the scaled joint
    distribution objective.

    Based on arXiv:2501.18865 (ICML'25).

    Args:
        model: Diffusion model (predicts noise or x_0)
        x_t: Noisy sample [batch, channels, ...] or [batch, seq_len, dim]
        t: Timestep values [batch] or [batch, 1]
        condition: Conditioning tensor (class labels, text embeddings, etc.)
        alpha_bar: Cumulative product of alphas [batch], required for DDPM
        reg_config: REG configuration (defaults to standard settings)
        null_condition: Null/unconditional token for CFG
        model_kwargs: Additional model arguments

    Returns:
        REG-guided model prediction (noise or x_0)

    Example:
        >>> config = REGConfig(guidance_scale=1.5, noise_schedule="ddpm")
        >>> # During sampling loop:
        >>> pred = sample_with_reg_diffusion(
        ...     model, x_t, t, condition,
        ...     alpha_bar=scheduler.alphas_cumprod[t],
        ...     reg_config=config
        ... )
    """
    reg_config = reg_config or REGConfig()
    model_kwargs = model_kwargs or {}

    _validate_guidance_scale(reg_config.guidance_scale)

    # No guidance case
    if reg_config.guidance_scale == 0.0 or condition is None:
        return model(x_t, t, condition=condition, **model_kwargs)

    # Check interval
    if reg_config.interval is not None:
        t_val = t.mean().item()
        if not (reg_config.interval[0] <= t_val <= reg_config.interval[1]):
            # Outside interval: only conditional prediction
            return model(x_t, t, condition=condition, **model_kwargs)

    # Enable gradients for input
    x_t_grad = x_t.detach().requires_grad_(True)

    # Conditional prediction (with grad tracking)
    pred_cond = model(x_t_grad, t, condition=condition, **model_kwargs)

    # Compute REG coefficient
    reg_coeff = _compute_reg_coefficient(
        x_t_grad,
        pred_cond,
        t,
        alpha_bar=alpha_bar,
        sigma=None,
        noise_schedule=reg_config.noise_schedule,
    )

    # Detach predictions after gradient computation
    pred_cond = pred_cond.detach()
    x_t_grad.requires_grad_(False)

    # Unconditional prediction (no grad needed)
    if null_condition is None:
        pred_uncond = model(x_t, t, condition=None, **model_kwargs)
    else:
        pred_uncond = model(x_t, t, condition=null_condition, **model_kwargs)

    # Get scheduled guidance scale
    t_max = 1.0 if reg_config.noise_schedule == "flow" else 1000.0
    scale = _get_scheduled_scale(
        t,
        reg_config.guidance_scale,
        reg_config.schedule_type,
        reg_config.end_scale,
        reg_config.cosine_power,
        t_max,
        output_ndim=pred_cond.ndim,
    )

    # Apply REG: cond + scale * reg_coeff * (cond - uncond)
    return pred_cond + scale * reg_coeff * (pred_cond - pred_uncond)


def sample_with_reg_flow(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    reg_config: REGConfig | None = None,
    null_condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Rectified Gradient Guidance (REG) for flow matching models.

    REG improves upon standard CFG by computing the gradient of the conditional
    output w.r.t. the input, which rectifies the guidance direction.

    For flow matching, the noise factor is typically sigma or (1-t) for linear
    interpolation schedules.

    Based on arXiv:2501.18865 (ICML'25).

    Args:
        model: Flow matching model
        x_t: Current state [batch, seq_len] or [batch, seq_len, dim]
        t: Time values [batch] or [batch, 1], normalized to [0, 1]
        condition: Conditioning tensor
        sigma: Noise level [batch], if None uses (1-t)
        reg_config: REG configuration
        null_condition: Null/unconditional token for CFG

    Returns:
        REG-guided velocity prediction

    Example:
        >>> config = REGConfig(guidance_scale=1.5, noise_schedule="flow")
        >>> velocity = sample_with_reg_flow(model, x_t, t, condition, reg_config=config)
    """
    reg_config = reg_config or REGConfig(noise_schedule="flow")
    _validate_guidance_scale(reg_config.guidance_scale)

    # No guidance case
    if reg_config.guidance_scale == 0.0 or condition is None:
        return model(x_t=x_t, time=t, condition=condition)

    # Check interval
    if reg_config.interval is not None:
        t_val = t.mean().item()
        if not (reg_config.interval[0] <= t_val <= reg_config.interval[1]):
            return model(x_t=x_t, time=t, condition=condition)

    # Enable gradients for input
    x_t_grad = x_t.detach().requires_grad_(True)

    # Conditional prediction (with grad tracking)
    output_cond = model(x_t=x_t_grad, time=t, condition=condition)

    # Compute REG coefficient
    reg_coeff = _compute_reg_coefficient(
        x_t_grad,
        output_cond,
        t,
        alpha_bar=None,
        sigma=sigma,
        noise_schedule="flow",
    )

    # Detach predictions
    output_cond = output_cond.detach()
    x_t_grad.requires_grad_(False)

    # Unconditional prediction
    if null_condition is None:
        output_uncond = model(x_t=x_t, time=t, condition=None)
    else:
        output_uncond = model(x_t=x_t, time=t, condition=null_condition)

    # Get scheduled guidance scale
    scale = _get_scheduled_scale(
        t,
        reg_config.guidance_scale,
        reg_config.schedule_type,
        reg_config.end_scale,
        reg_config.cosine_power,
        t_max=1.0,
        output_ndim=output_cond.ndim,
    )

    # Apply REG
    return output_cond + scale * reg_coeff * (output_cond - output_uncond)


class REGDiffuser:
    """
    Wrapper for diffusion models with Rectified Gradient Guidance (REG).

    REG provides improved guidance over standard CFG by computing gradient-based
    corrections that better approximate the optimal guidance under the scaled
    joint distribution objective.

    Based on arXiv:2501.18865 (ICML'25).

    Example:
        >>> config = REGConfig(guidance_scale=1.5)
        >>> guided_model = REGDiffuser(model, config)
        >>> # Use in sampling loop
        >>> pred = guided_model(x_t, t, condition, alpha_bar=alpha_bar_t)
    """

    def __init__(
        self,
        model: nn.Module,
        reg_config: REGConfig | None = None,
        null_condition: torch.Tensor | None = None,
    ):
        """
        Initialize REG wrapper.

        Args:
            model: Diffusion model
            reg_config: REG configuration
            null_condition: Unconditional token for CFG
        """
        self.model = model
        self.reg_config = reg_config or REGConfig()
        self.null_condition = null_condition
        _validate_guidance_scale(self.reg_config.guidance_scale)

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        alpha_bar: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Forward pass with REG."""
        return sample_with_reg_diffusion(
            self.model,
            x_t,
            t,
            condition=condition,
            alpha_bar=alpha_bar,
            reg_config=self.reg_config,
            null_condition=self.null_condition,
            model_kwargs=model_kwargs,
        )


class REGFlowMatcher:
    """
    Wrapper for flow matching models with Rectified Gradient Guidance (REG).

    Based on arXiv:2501.18865 (ICML'25).

    Example:
        >>> config = REGConfig(guidance_scale=1.5, noise_schedule="flow")
        >>> guided_model = REGFlowMatcher(model, config)
        >>> velocity = guided_model(x_t, t, condition)
    """

    def __init__(
        self,
        model: nn.Module,
        reg_config: REGConfig | None = None,
        null_condition: torch.Tensor | None = None,
    ):
        """
        Initialize REG wrapper.

        Args:
            model: Flow matching model
            reg_config: REG configuration
            null_condition: Unconditional token for CFG
        """
        self.model = model
        self.reg_config = reg_config or REGConfig(noise_schedule="flow")
        self.null_condition = null_condition
        _validate_guidance_scale(self.reg_config.guidance_scale)

    def __call__(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with REG."""
        return sample_with_reg_flow(
            self.model,
            x_t,
            t,
            condition=condition,
            sigma=sigma,
            reg_config=self.reg_config,
            null_condition=self.null_condition,
        )


def generate_with_reg(
    model: nn.Module,
    solver: Callable,
    x_init: torch.Tensor,
    condition: torch.Tensor | None = None,
    reg_config: REGConfig | None = None,
    null_condition: torch.Tensor | None = None,
    model_type: str = "flow",
    **solver_kwargs,
) -> torch.Tensor:
    """
    Unified generation with REG for flow matching or diffusion.

    Args:
        model: Model (flow matching or diffusion)
        solver: Solver function
        x_init: Initial noise/state
        condition: Conditioning information
        reg_config: REG configuration
        null_condition: Unconditional token
        model_type: 'flow' or 'diffusion'
        **solver_kwargs: Additional solver arguments

    Returns:
        Generated samples
    """
    guided_model: object  # type: ignore[assignment]
    if model_type == "flow":
        guided_model = REGFlowMatcher(model, reg_config, null_condition)
    elif model_type == "diffusion":
        guided_model = REGDiffuser(model, reg_config, null_condition)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'flow' or 'diffusion'.")

    assert guided_model is not None
    return solver(guided_model, x_init, **solver_kwargs)
