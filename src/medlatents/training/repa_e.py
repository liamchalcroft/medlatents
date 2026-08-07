"""REPA-E: End-to-end VAE + DiT training with representation alignment.

Implements end-to-end training of VAE tokenizers alongside diffusion transformers
using REPA loss instead of standard diffusion loss for VAE gradient updates.

Key insight: Standard diffusion loss is ineffective for E2E VAE training, but
REPA alignment loss enables joint optimization, achieving 45x training speedup.

Reference:
    Leng et al., "REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers"
    arXiv:2504.10483

Usage:
    >>> trainer = REPAETrainer(
    ...     dit=dit_model,
    ...     vae=vae_model,
    ...     frozen_encoder=dino_encoder,
    ...     repa_projection=projection_head,
    ... )
    >>> loss = trainer.compute_loss(images)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .repa import REPALoss, REPAProjection


@dataclass
class REPAEConfig:
    """Configuration for REPA-E end-to-end training.

    Attributes:
        repa_weight: Weight for REPA alignment loss
        diffusion_weight: Weight for diffusion denoising loss
        vae_recon_weight: Weight for VAE reconstruction loss (optional)
        kl_weight: Weight for VAE KL divergence loss
        timestep_threshold: Only apply REPA at t > threshold
        vae_grad_scale: Scale factor for gradients flowing to VAE
        freeze_vae_encoder: Whether to freeze VAE encoder (only train decoder)
        freeze_vae_decoder: Whether to freeze VAE decoder (only train encoder)
        vae_warmup_steps: Steps to warmup VAE learning rate
        alignment_layers: Which DiT layers to use for alignment (list of indices)
    """

    repa_weight: float = 1.0
    diffusion_weight: float = 1.0
    vae_recon_weight: float = 0.0  # Optional reconstruction loss
    kl_weight: float = 0.0  # Optional KL loss for VAE

    timestep_threshold: float = 0.5
    vae_grad_scale: float = 1.0

    freeze_vae_encoder: bool = False
    freeze_vae_decoder: bool = False

    vae_warmup_steps: int = 0
    alignment_layers: list[int] | None = None  # None = use all layers


class VAEWrapper(nn.Module):
    """Wrapper for VAE models to provide a consistent interface.

    Handles both continuous VAEs (standard) and discrete tokenizers.
    """

    vae: nn.Module

    def __init__(
        self,
        vae: nn.Module,
        vae_type: Literal["continuous", "discrete"] = "continuous",
    ):
        super().__init__()
        self.vae = vae
        self.vae_type = vae_type

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode input to latent space.

        Returns:
            Dict with 'latent' and optionally 'mu', 'logvar' for continuous VAE
            or 'indices' for discrete tokenizer.
        """
        if self.vae_type == "continuous":
            # Standard VAE encoding
            if hasattr(self.vae, "encode"):
                enc_output = self.vae.encode(x)  # type: ignore[operator]
                if isinstance(enc_output, tuple):
                    mu, logvar = enc_output
                    # Reparameterization
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    latent = mu + eps * std
                    return {"latent": latent, "mu": mu, "logvar": logvar}
                else:
                    # Direct latent output
                    return {"latent": enc_output}
            else:
                raise ValueError("VAE must have encode method")

        elif self.vae_type == "discrete":
            # Discrete tokenizer
            if hasattr(self.vae, "encode"):
                indices = self.vae.encode(x)  # type: ignore[operator]
                return {"latent": indices, "indices": indices}
            elif hasattr(self.vae, "tokenize"):
                indices = self.vae.tokenize(x)  # type: ignore[operator]
                return {"latent": indices, "indices": indices}
            else:
                raise ValueError("Discrete VAE must have encode or tokenize method")

        else:
            raise ValueError(f"Unknown VAE type: {self.vae_type}")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space."""
        if hasattr(self.vae, "decode"):
            return self.vae.decode(latent)  # type: ignore[operator]
        elif hasattr(self.vae, "detokenize"):
            return self.vae.detokenize(latent)  # type: ignore[operator]
        else:
            raise ValueError("VAE must have decode or detokenize method")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full VAE forward pass."""
        enc_output = self.encode(x)
        recon = self.decode(enc_output["latent"])
        return {**enc_output, "reconstruction": recon}


class REPAELoss(nn.Module):
    """Combined loss for REPA-E end-to-end training.

    Combines:
    1. REPA alignment loss (for both DiT and VAE gradients)
    2. Diffusion denoising loss (for DiT gradients only)
    3. Optional VAE reconstruction loss
    4. Optional VAE KL divergence loss
    """

    def __init__(
        self,
        config: REPAEConfig,
        hidden_dim: int,
        target_dim: int,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.config = config

        # REPA loss for alignment
        self.repa_loss = REPALoss(
            hidden_dim=hidden_dim,
            target_dim=target_dim,
            proj_dim=proj_dim,
            timestep_threshold=config.timestep_threshold,
            weight=config.repa_weight,
        )

    def forward(
        self,
        # DiT outputs
        dit_hidden: torch.Tensor,
        dit_pred: torch.Tensor,
        # Targets
        encoder_features: torch.Tensor,
        diffusion_target: torch.Tensor,
        # VAE outputs (optional)
        vae_mu: torch.Tensor | None = None,
        vae_logvar: torch.Tensor | None = None,
        vae_recon: torch.Tensor | None = None,
        original_images: torch.Tensor | None = None,
        # Timesteps
        timesteps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined REPA-E loss.

        Args:
            dit_hidden: DiT hidden states for alignment [batch, seq, hidden]
            dit_pred: DiT prediction (noise/velocity) [batch, seq, channels]
            encoder_features: Frozen encoder features [batch, seq, target_dim]
            diffusion_target: Target for diffusion loss [batch, seq, channels]
            vae_mu: VAE encoder mean (for KL loss)
            vae_logvar: VAE encoder log variance (for KL loss)
            vae_recon: VAE reconstruction (for reconstruction loss)
            original_images: Original images (for reconstruction loss)
            timesteps: Diffusion timesteps [batch]

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # 1. REPA alignment loss (main loss for E2E training)
        # This provides gradients to both DiT AND VAE
        repa_loss = self.repa_loss(
            dit_hidden,
            encoder_features,
            timesteps=timesteps,
        )
        losses["repa"] = repa_loss

        # 2. Diffusion denoising loss (DiT only, stop grad to VAE)
        # We detach the target to prevent diffusion loss from updating VAE
        diffusion_loss = F.mse_loss(dit_pred, diffusion_target.detach())
        losses["diffusion"] = diffusion_loss

        # 3. Optional VAE reconstruction loss
        if (
            self.config.vae_recon_weight > 0
            and vae_recon is not None
            and original_images is not None
        ):
            recon_loss = F.mse_loss(vae_recon, original_images)
            losses["vae_recon"] = recon_loss
        else:
            losses["vae_recon"] = torch.tensor(0.0, device=dit_hidden.device)

        # 4. Optional VAE KL loss
        if self.config.kl_weight > 0 and vae_mu is not None and vae_logvar is not None:
            kl_loss = -0.5 * torch.mean(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp())
            losses["vae_kl"] = kl_loss
        else:
            losses["vae_kl"] = torch.tensor(0.0, device=dit_hidden.device)

        # Total loss
        total = (
            self.config.repa_weight * losses["repa"]
            + self.config.diffusion_weight * losses["diffusion"]
            + self.config.vae_recon_weight * losses["vae_recon"]
            + self.config.kl_weight * losses["vae_kl"]
        )
        losses["total"] = total

        return losses


class REPAETrainer:
    """End-to-end trainer for VAE + DiT using REPA alignment.

    This trainer enables joint optimization of VAE tokenizer and DiT by using
    REPA alignment loss instead of standard diffusion loss for VAE gradients.

    The key insight is that diffusion loss is ineffective for E2E VAE training,
    but REPA loss provides meaningful gradients for both components.

    Example:
        >>> trainer = REPAETrainer(
        ...     dit=dit_model,
        ...     vae=vae_model,
        ...     frozen_encoder=dino_encoder,
        ...     config=REPAEConfig(repa_weight=1.0),
        ... )
        >>>
        >>> for images in dataloader:
        ...     loss_dict = trainer.compute_loss(images, timesteps)
        ...     loss_dict["total"].backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        dit: nn.Module,
        vae: nn.Module,
        frozen_encoder: nn.Module,
        repa_projection: REPAProjection,
        config: REPAEConfig | None = None,
        vae_type: Literal["continuous", "discrete"] = "continuous",
        noise_scheduler: nn.Module | None = None,
    ):
        """
        Args:
            dit: Diffusion transformer model
            vae: VAE or discrete tokenizer
            frozen_encoder: Pretrained encoder (DINOv2, SigLIP, etc.)
            repa_projection: Projection head for alignment
            config: Training configuration
            vae_type: Type of VAE ('continuous' or 'discrete')
            noise_scheduler: Diffusion noise scheduler (for adding noise)
        """
        self.dit = dit
        self.vae = VAEWrapper(vae, vae_type)
        self.frozen_encoder = frozen_encoder
        self.repa_projection = repa_projection
        self.config = config or REPAEConfig()
        self.noise_scheduler = noise_scheduler

        # Freeze encoder
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False

        # Optionally freeze parts of VAE
        if self.config.freeze_vae_encoder:
            if hasattr(self.vae.vae, "encoder"):
                for param in self.vae.vae.encoder.parameters():  # type: ignore[union-attr]
                    param.requires_grad = False

        if self.config.freeze_vae_decoder:
            if hasattr(self.vae.vae, "decoder"):
                for param in self.vae.vae.decoder.parameters():  # type: ignore[union-attr]
                    param.requires_grad = False

        # Create loss module
        self.loss_fn = REPAELoss(
            config=self.config,
            hidden_dim=repa_projection.hidden_dim,
            target_dim=repa_projection.target_dim,
            proj_dim=repa_projection.proj_dim,
        )

        self.step = 0

    def _add_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latents for diffusion training.

        Returns:
            Tuple of (noisy_latents, noise/target)
        """
        if self.noise_scheduler is not None:
            # Use provided scheduler
            noise = torch.randn_like(latents)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore[operator]
            return noisy_latents, noise
        else:
            # Simple linear interpolation (flow matching style)
            noise = torch.randn_like(latents)
            t = timesteps.view(-1, *([1] * (latents.ndim - 1)))
            noisy_latents = t * latents + (1 - t) * noise
            target = latents - noise  # velocity
            return noisy_latents, target

    def compute_loss(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute REPA-E loss for a batch of images.

        Args:
            images: Input images [batch, channels, H, W]
            timesteps: Diffusion timesteps [batch]. If None, sampled uniformly.
            return_intermediates: Whether to return intermediate tensors

        Returns:
            Dict with loss components and optionally intermediates
        """
        batch_size = images.shape[0]
        device = images.device

        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=device)

        # 1. Encode images with VAE (with gradients for E2E training)
        vae_output = self.vae.encode(images)
        latents = vae_output["latent"]

        # Flatten spatial dims to sequence if needed
        if latents.ndim == 4:
            # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = latents.shape
            latents_seq = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            latents_seq = latents

        # 2. Add noise for diffusion
        noisy_latents, target = self._add_noise(latents_seq, timesteps)

        # 3. Get frozen encoder features (no gradients)
        with torch.no_grad():
            encoder_features = self.frozen_encoder(images)
            # Handle different encoder output formats
            if isinstance(encoder_features, dict):
                encoder_features = encoder_features.get(
                    "last_hidden_state", encoder_features.get("features")
                )

        # 4. DiT forward pass with hidden state extraction
        # We need hidden states from intermediate layers for REPA
        dit_output = self._forward_dit_with_hidden(noisy_latents, timesteps)
        dit_pred = dit_output["prediction"]
        dit_hidden = dit_output["hidden_states"]

        # 5. Project hidden states for alignment
        proj_hidden = self.repa_projection(dit_hidden)
        proj_target = self.repa_projection.project_target(cast(torch.Tensor, encoder_features))

        # 6. Compute combined loss
        losses = self.loss_fn(
            dit_hidden=proj_hidden,
            dit_pred=dit_pred,
            encoder_features=proj_target,
            diffusion_target=target,
            vae_mu=vae_output.get("mu"),
            vae_logvar=vae_output.get("logvar"),
            vae_recon=None,  # Skip reconstruction during main training
            original_images=images,
            timesteps=timesteps,
        )

        self.step += 1

        if return_intermediates:
            losses["intermediates"] = {
                "latents": latents_seq,
                "noisy_latents": noisy_latents,
                "dit_hidden": dit_hidden,
                "encoder_features": encoder_features,
            }

        return losses

    def _forward_dit_with_hidden(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the DiT, extracting intermediate hidden states.

        The DiT must expose ``forward_with_hidden(x, t)`` returning a dict with
        ``"prediction"`` and ``"hidden_states"``. REPA-E aligns *internal* features
        to the encoder, so substituting the network output as a stand-in would
        train a meaningless alignment; we fail loudly instead.
        """
        if hasattr(self.dit, "forward_with_hidden"):
            return self.dit.forward_with_hidden(x, t)  # type: ignore[operator]

        raise NotImplementedError(
            "REPA-E requires the DiT to expose intermediate features via "
            "`forward_with_hidden(x, t) -> {'prediction', 'hidden_states'}`. "
            "The configured DiT does not implement it."
        )

    def get_vae_parameters(self) -> list[nn.Parameter]:
        """Get VAE parameters for separate optimizer group."""
        params = []
        for name, param in self.vae.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def get_dit_parameters(self) -> list[nn.Parameter]:
        """Get DiT parameters for separate optimizer group."""
        params = list(self.dit.parameters())
        params.extend(self.repa_projection.parameters())
        return params

    def create_optimizer_groups(
        self,
        dit_lr: float = 1e-4,
        vae_lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> list[dict]:
        """Create optimizer parameter groups with different learning rates.

        VAE typically uses a lower learning rate to prevent catastrophic forgetting.
        """
        return [
            {
                "params": self.get_dit_parameters(),
                "lr": dit_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": self.get_vae_parameters(),
                "lr": vae_lr,
                "weight_decay": weight_decay,
            },
        ]


__all__ = [
    "REPAEConfig",
    "REPAELoss",
    "REPAETrainer",
    "VAEWrapper",
]
