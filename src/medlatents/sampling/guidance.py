"""Classifier-free guidance with time-dependent schedules."""

import torch
import torch.nn as nn

from .schedules import GuidanceSchedule, get_guidance_schedule


class GuidedSampler:
    """
    Applies classifier-free guidance with time-dependent schedules.

    Reference:
    - "Theory-Informed Improvements to Classifier-Free Guidance" (arXiv:2507.08965)
    - "What Does Guidance Do in Masked Discrete Diffusion?" (arXiv:2506.10971)

    Args:
        model: Model that takes (x, t, conditioning) and returns logits
        uncond_model: Optional unconditioned model (if None, uses empty conditioning)
        guidance_schedule: Schedule function or name
        base_scale: Base guidance scale
    """

    def __init__(
        self,
        model: nn.Module,
        uncond_model: nn.Module | None = None,
        guidance_schedule: str | GuidanceSchedule = "constant",
        base_scale: float = 5.0,
    ):
        self.model = model
        self.uncond_model = uncond_model
        self.base_scale = base_scale

        if isinstance(guidance_schedule, str):
            self.guidance_schedule = get_guidance_schedule(guidance_schedule)
        else:
            self.guidance_schedule = guidance_schedule

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: dict[str, torch.Tensor] | None = None,
        progress: float = 0.5,
    ) -> torch.Tensor:
        """
        Get guided predictions.

        Args:
            x: Input tensor
            t: Time tensor
            conditioning: Conditioning dict (class labels, text, etc.)
            progress: Sampling progress (0 to 1) for guidance schedule

        Returns:
            Guided logits
        """
        # Conditional prediction
        cond_logits = self.model(x, t, **(conditioning or {}))

        # Unconditional prediction
        if self.uncond_model is not None:
            uncond_logits = self.uncond_model(x, t)
        else:
            uncond_logits = self.model(x, t)

        # Get guidance scale at this progress
        scale = self.guidance_schedule(progress, self.base_scale)

        # CFG: cond + scale * (cond - uncond)
        guided_logits = uncond_logits + scale * (cond_logits - uncond_logits)

        return guided_logits


__all__ = [
    "GuidedSampler",
    "GuidanceSchedule",
    "get_guidance_schedule",
]
