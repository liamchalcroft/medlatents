"""Enhanced conditional training support with ConditioningBundle integration.

This module provides utilities for training with rich conditioning information,
including support for REPA alignment, multiple conditioning modalities, and
CFG training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import torch
import torch.nn as nn

from ..conditioning.bundle import ConditioningBundle, ConditioningConfig

# Type variable for model outputs
T = TypeVar("T")


class ConditionalModel(Protocol):
    """Protocol for models that accept ConditioningBundle."""

    def forward(
        self,
        x: torch.Tensor,
        bundle: ConditioningBundle,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with conditioning bundle."""
        ...


@dataclass
class ConditionalTrainingConfig:
    """Configuration for conditional training.

    Attributes:
        conditioning_config: Model conditioning capabilities
        class_dropout_prob: CFG dropout probability for class labels
        text_dropout_prob: CFG dropout probability for text
        use_repa: Whether to use REPA alignment loss
        repa_weight: Weight for REPA loss
        repa_timestep_threshold: Only apply REPA at t > threshold
        use_vecor: Whether to use VeCoR contrastive loss
        vecor_weight: Weight for VeCoR loss
        use_haste: Whether to use HASTE termination scheduling
        haste_termination_step: Step to terminate alignment (for HASTE)
    """

    conditioning_config: ConditioningConfig | None = None

    # CFG dropout
    class_dropout_prob: float = 0.1
    text_dropout_prob: float = 0.1

    # REPA alignment
    use_repa: bool = False
    repa_weight: float = 0.5
    repa_timestep_threshold: float = 0.5
    repa_encoder_name: str = "dinov2_vitb14"

    # VeCoR contrastive loss
    use_vecor: bool = False
    vecor_weight: float = 0.1
    vecor_temperature: float = 0.1

    # HASTE termination
    use_haste: bool = False
    haste_termination_step: int | None = None
    haste_termination_type: Literal["hard", "linear", "cosine"] = "linear"


def prepare_batch_with_conditioning(
    batch: dict[str, torch.Tensor] | tuple | torch.Tensor,
    timesteps: torch.Tensor,
    config: ConditionalTrainingConfig,
    training: bool = True,
) -> tuple[torch.Tensor, ConditioningBundle]:
    """Prepare a batch for conditional training.

    Extracts data and conditioning from the batch, creates a ConditioningBundle,
    and optionally applies CFG dropout during training.

    Args:
        batch: Input batch (dict, tuple, or tensor)
        timesteps: Sampled timesteps for diffusion/flow
        config: Training configuration
        training: Whether in training mode (applies dropout)

    Returns:
        Tuple of (data_tensor, conditioning_bundle)
    """
    # Extract data from batch
    if isinstance(batch, dict):
        # Dict-style batch
        data = batch.get("data", batch.get("tokens", batch.get("x")))
        if data is None:
            raise ValueError(f"Could not find data in batch dict. Keys: {list(batch.keys())}")
        bundle = ConditioningBundle.from_batch(
            batch,
            timesteps=timesteps,
            config=config.conditioning_config,
        )
    elif isinstance(batch, (list, tuple)):
        # Tuple-style batch: (data, labels) or (data, labels, ...)
        data = batch[0]
        batch_dict = {"data": data}
        if len(batch) > 1:
            batch_dict["class_labels"] = batch[1]
        bundle = ConditioningBundle.from_batch(
            batch_dict,
            timesteps=timesteps,
            config=config.conditioning_config,
        )
    else:
        # Plain tensor batch
        data = batch
        bundle = ConditioningBundle(timesteps=timesteps)

    # Apply CFG dropout during training
    if training:
        bundle = bundle.apply_cfg_dropout(
            class_dropout_prob=config.class_dropout_prob,
            text_dropout_prob=config.text_dropout_prob,
        )

    return data, bundle


class ConditionalLossWrapper(nn.Module):
    """Wrapper that adds REPA and VeCoR losses to a base loss.

    This wrapper can be added around any diffusion/flow loss to incorporate
    representation alignment (REPA) and velocity contrastive regularization
    (VeCoR) losses.

    Example:
        >>> base_loss = MSELoss()
        >>> wrapper = ConditionalLossWrapper(
        ...     base_loss_fn=base_loss,
        ...     config=ConditionalTrainingConfig(use_repa=True),
        ...     repa_projection=REPAProjection(1024, 768),
        ...     frozen_encoder=encoder,
        ... )
        >>> loss = wrapper(pred, target, hidden_states=hidden, original=x, timesteps=t)
    """

    def __init__(
        self,
        base_loss_fn: nn.Module,
        config: ConditionalTrainingConfig,
        repa_projection: nn.Module | None = None,
        frozen_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.config = config

        # REPA components
        self.repa_projection = repa_projection
        self.frozen_encoder = frozen_encoder

        if config.use_repa:
            if repa_projection is None or frozen_encoder is None:
                raise ValueError("REPA requires repa_projection and frozen_encoder to be provided")

            # Import REPA loss
            from .repa import REPALoss

            self.repa_loss = REPALoss(
                hidden_dim=int(getattr(repa_projection, "hidden_dim", 1024)),
                target_dim=int(getattr(repa_projection, "target_dim", 768)),
                proj_dim=int(getattr(repa_projection, "proj_dim", 256)),
                timestep_threshold=config.repa_timestep_threshold,
                weight=config.repa_weight,
            )

        if config.use_vecor:
            # Import VeCoR loss
            from .losses import VelocityContrastiveRegularization

            self.vecor_loss = VelocityContrastiveRegularization(
                temperature=config.vecor_temperature,
                weight=config.vecor_weight,
            )

        # HASTE scheduler
        self.haste_scheduler = None
        if config.use_haste:
            from .repa import HASTEScheduler

            self.haste_scheduler = HASTEScheduler(
                termination_step=config.haste_termination_step or 100000,
                termination_mode=config.haste_termination_type,
            )

        self.current_step = 0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        original_data: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        **base_loss_kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss with optional REPA and VeCoR.

        Args:
            pred: Model prediction (velocity, noise, or x0)
            target: Target for denoising loss
            hidden_states: DiT hidden states for REPA (optional)
            original_data: Clean data for REPA encoder (optional)
            timesteps: Current timesteps for REPA threshold (optional)
            **base_loss_kwargs: Additional args for base loss

        Returns:
            Dict with 'total', 'base', and optional 'repa', 'vecor' losses
        """
        losses = {}

        # Base denoising loss
        base_loss = self.base_loss_fn(pred, target, **base_loss_kwargs)
        losses["base"] = base_loss
        total_loss = base_loss

        # REPA alignment loss
        if (
            self.config.use_repa
            and hidden_states is not None
            and original_data is not None
            and self.repa_projection is not None
        ):
            # Get HASTE weight (1.0 before termination, decays after)
            haste_weight = 1.0
            if self.haste_scheduler is not None:
                haste_weight = self.haste_scheduler.get_repa_weight(self.current_step)

            if haste_weight > 0 and self.frozen_encoder is not None:
                # Get encoder features
                with torch.no_grad():
                    encoder_features = self.frozen_encoder(original_data)  # type: ignore[misc]

                # Project hidden states
                proj_hidden = self.repa_projection(hidden_states)  # type: ignore[misc]
                proj_target = self.repa_projection.project_target(encoder_features)  # type: ignore[misc,union-attr,operator]

                # Compute REPA loss (only at high-noise timesteps)
                repa_loss = self.repa_loss(
                    proj_hidden,
                    proj_target,
                    timesteps=timesteps,
                )
                losses["repa"] = repa_loss
                total_loss = total_loss + haste_weight * repa_loss

        # VeCoR contrastive loss
        if self.config.use_vecor and timesteps is not None:
            vecor_loss = self.vecor_loss(
                pred,
                target,
                timesteps,
            )
            losses["vecor"] = vecor_loss
            total_loss = total_loss + vecor_loss

        losses["total"] = total_loss
        return losses

    def step(self) -> None:
        """Increment internal step counter (call after each optimizer step)."""
        self.current_step += 1


def compute_conditional_loss(
    model: nn.Module,
    data: torch.Tensor,
    bundle: ConditioningBundle,
    noise_process: nn.Module,
    loss_fn: nn.Module | ConditionalLossWrapper,
    return_hidden: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Compute loss for conditional diffusion/flow training.

    This is a high-level utility that handles:
    1. Sampling noisy data from the noise process
    2. Forward pass with conditioning
    3. Loss computation (optionally with REPA/VeCoR)

    Args:
        model: Denoising model
        data: Clean data tensor
        bundle: Conditioning bundle
        noise_process: Diffusion or flow matching process
        loss_fn: Loss function (optionally ConditionalLossWrapper)
        return_hidden: Whether to return hidden states (for REPA)

    Returns:
        Loss tensor, or (loss, info_dict) if return_hidden=True
    """
    # Sample noisy data
    t = bundle.timesteps
    noise_output = noise_process.add_noise(data, t)  # type: ignore[operator]

    # Forward pass
    model_kwargs: dict[str, torch.Tensor] = {}
    if hasattr(model, "forward_with_bundle"):
        # Model natively supports ConditioningBundle
        pred = model.forward_with_bundle(noise_output.x_t, bundle, **model_kwargs)  # type: ignore[operator]
    else:
        # Fall back to standard interface
        pred = model(
            noise_output.x_t,
            t,
            y=bundle.class_labels,
            **model_kwargs,
        )

    # Compute loss
    if isinstance(loss_fn, ConditionalLossWrapper):
        losses = loss_fn(
            pred=pred,
            target=noise_output.target,
            original_data=data,
            timesteps=t,
        )
        loss = losses["total"]
        if return_hidden:
            return loss, losses
    else:
        loss = loss_fn(pred, noise_output.target)
        if return_hidden:
            return loss, {"total": loss, "base": loss}

    return loss


__all__ = [
    "ConditionalTrainingConfig",
    "ConditionalLossWrapper",
    "ConditionalModel",
    "prepare_batch_with_conditioning",
    "compute_conditional_loss",
]
