"""REPA: Representation Alignment for faster diffusion training.

Implements:
- REPALoss: Alignment loss between DiT hidden states and pretrained encoder features
- REPAProjection: MLP projection head for feature alignment
- HASTEScheduler: Stage-wise termination of alignment loss (HASTE)
- AttentionAlignmentLoss: Attention map distillation for holistic alignment

Reference papers:
- REPA: "Representation Alignment for Generation" (arXiv:2410.06940, ICLR'25 Oral)
- HASTE: "REPA Works Until It Doesn't" (arXiv:2505.16792)

Key insight: Aligning DiT hidden states with frozen encoder (DINOv2, SigLIP)
features at high-noise timesteps dramatically accelerates training (17x speedup).
However, alignment helps early but can hurt later - HASTE terminates it optimally.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class REPAProjection(nn.Module):
    """MLP projection head for REPA feature alignment.

    Projects DiT hidden states to the same dimension as the target encoder
    for similarity computation. Applied after early-to-mid transformer blocks.

    Args:
        hidden_dim: DiT hidden dimension (input)
        target_dim: Target encoder dimension (e.g., 768 for DINOv2-B)
        proj_dim: Projection space dimension (for similarity computation)
        num_layers: Number of MLP layers
        activation: Activation function
        dropout: Dropout probability

    Usage:
        proj = REPAProjection(hidden_dim=1024, target_dim=768, proj_dim=256)

        # In forward pass, after block k:
        hidden = dit_block_k(x)
        proj_hidden = proj(hidden)  # [batch, seq, proj_dim]
    """

    projection: nn.Sequential
    target_proj: nn.Module

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int = 768,
        proj_dim: int = 256,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.proj_dim = proj_dim

        # Build MLP
        act_fn = nn.GELU() if activation == "gelu" else nn.SiLU()

        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for i in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, proj_dim),
                    act_fn,
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = proj_dim
        layers.append(nn.Linear(in_dim, proj_dim))

        self.projection = nn.Sequential(*layers)

        # Target projection (if target_dim != proj_dim)
        if target_dim != proj_dim:
            self.target_proj = nn.Linear(target_dim, proj_dim)
        else:
            self.target_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Project hidden states to alignment space.

        Args:
            hidden_states: DiT hidden states [batch, seq, hidden_dim]

        Returns:
            Projected features [batch, seq, proj_dim]
        """
        return self.projection(hidden_states)

    def project_target(
        self,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Project target encoder features to alignment space.

        Args:
            target_features: Encoder features [batch, seq, target_dim]

        Returns:
            Projected features [batch, seq, proj_dim]
        """
        return self.target_proj(target_features)


class REPALoss(nn.Module):
    """REPA: Representation Alignment Loss for accelerated diffusion training.

    Aligns DiT hidden states with frozen pretrained encoder features using
    cosine similarity. Applied only at high-noise timesteps where semantic
    alignment provides the most benefit.

    The key insight from REPA is that early in denoising (high t), the model
    benefits from semantic guidance from a pretrained encoder. As denoising
    progresses (low t), the model needs to focus on fine details.

    Reference:
        Yu et al., "Representation Alignment for Generation" (ICLR'25 Oral)
        https://arxiv.org/abs/2410.06940

    Args:
        hidden_dim: DiT hidden dimension
        target_dim: Target encoder dimension
        proj_dim: Projection dimension for alignment
        timestep_threshold: Only apply at t > threshold (default 0.5)
        weight: Loss weight relative to denoising loss
        normalize: Whether to L2-normalize before similarity
        loss_type: 'cosine' or 'mse'

    Usage:
        repa = REPALoss(hidden_dim=1024, target_dim=768)

        # In training loop:
        hidden_states = extract_hidden_from_dit(model, x_t, t)
        target_features = frozen_encoder(clean_images)

        repa_loss = repa(hidden_states, target_features, t)
        total_loss = denoising_loss + repa_loss
    """

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int = 768,
        proj_dim: int = 256,
        timestep_threshold: float = 0.5,
        weight: float = 0.5,
        normalize: bool = True,
        loss_type: Literal["cosine", "mse"] = "cosine",
        num_proj_layers: int = 2,
    ):
        super().__init__()

        if timestep_threshold < 0 or timestep_threshold > 1:
            raise ValueError(f"timestep_threshold must be in [0, 1], got {timestep_threshold}")
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")

        self.timestep_threshold = timestep_threshold
        self.weight = weight
        self.normalize = normalize
        self.loss_type = loss_type

        # Projection head
        self.projection = REPAProjection(
            hidden_dim=hidden_dim,
            target_dim=target_dim,
            proj_dim=proj_dim,
            num_layers=num_proj_layers,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_features: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute REPA alignment loss.

        Args:
            hidden_states: DiT hidden states [batch, seq, hidden_dim]
            target_features: Encoder features [batch, seq, target_dim]
            timesteps: Current timesteps [batch], in [0, 1]

        Returns:
            Weighted alignment loss (scalar)
        """
        # Only apply at high-noise timesteps
        mask = timesteps > self.timestep_threshold

        if not mask.any():
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # Select samples above threshold
        hidden_masked = hidden_states[mask]
        target_masked = target_features[mask]

        # Project to alignment space
        proj_hidden = self.projection(hidden_masked)
        proj_target = self.projection.project_target(target_masked)

        # Normalize if requested
        if self.normalize:
            proj_hidden = F.normalize(proj_hidden, dim=-1, p=2)
            proj_target = F.normalize(proj_target, dim=-1, p=2)

        # Compute loss
        if self.loss_type == "cosine":
            # Negative cosine similarity (minimize to maximize similarity)
            # Per-token similarity, averaged
            similarity = (proj_hidden * proj_target).sum(dim=-1)  # [masked_batch, seq]
            loss = -similarity.mean()
        else:  # mse
            loss = F.mse_loss(proj_hidden, proj_target)

        return self.weight * loss


class AttentionAlignmentLoss(nn.Module):
    """Attention map alignment loss for HASTE holistic alignment.

    Distills attention patterns from teacher encoder to DiT.
    Provides relational priors that complement feature alignment.

    Reference:
        Wang et al., "REPA Works Until It Doesn't" (arXiv:2505.16792)

    Args:
        temperature: Temperature for attention softmax
        weight: Loss weight

    Usage:
        attn_loss = AttentionAlignmentLoss()

        # Extract attention maps from both models
        dit_attn = extract_attention_maps(dit_model)
        teacher_attn = extract_attention_maps(teacher_encoder)

        loss = attn_loss(dit_attn, teacher_attn, timesteps)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        weight: float = 0.1,
        timestep_threshold: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.timestep_threshold = timestep_threshold

    def forward(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention alignment loss.

        Args:
            student_attn: DiT attention maps [batch, heads, seq, seq]
            teacher_attn: Teacher attention maps [batch, heads, seq, seq]
            timesteps: Current timesteps [batch]

        Returns:
            Attention distillation loss
        """
        mask = timesteps > self.timestep_threshold

        if not mask.any():
            return torch.tensor(0.0, device=student_attn.device, dtype=student_attn.dtype)

        student_masked = student_attn[mask]
        teacher_masked = teacher_attn[mask]

        # Ensure same shape (may need to interpolate if different resolutions)
        if student_masked.shape[-1] != teacher_masked.shape[-1]:
            # Interpolate teacher to match student
            teacher_masked = F.interpolate(
                teacher_masked,
                size=student_masked.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # KL divergence on attention distributions
        student_log_probs = F.log_softmax(student_masked / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_masked / self.temperature, dim=-1)

        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        return self.weight * loss


class HASTEScheduler:
    """HASTE: Holistic Alignment with Stage-wise Termination for Efficient training.

    Implements a two-phase training schedule:
    - Phase I: Apply full alignment (REPA + attention) for rapid convergence
    - Phase II: Terminate alignment to free generative capacity

    The key insight is that alignment helps early but hurts later due to
    capacity mismatch - the teacher's lower-dimensional embeddings become
    a straitjacket once the student starts modeling the full data distribution.

    Reference:
        Wang et al., "REPA Works Until It Doesn't" (arXiv:2505.16792)

    Args:
        termination_step: Step at which to terminate alignment
        termination_mode: How to terminate
            - 'hard': Immediate cutoff (recommended)
            - 'linear': Linear decay over transition_steps
            - 'cosine': Cosine decay over transition_steps
        transition_steps: Steps over which to decay (for 'linear'/'cosine')
        use_attention_alignment: Whether to use attention distillation in Phase I

    Usage:
        scheduler = HASTEScheduler(termination_step=50000)

        for step in range(total_steps):
            # Get current weights
            repa_weight = scheduler.get_repa_weight(step)
            attn_weight = scheduler.get_attention_weight(step)

            if repa_weight > 0:
                loss += repa_weight * repa_loss
            if attn_weight > 0:
                loss += attn_weight * attention_loss

            # Log phase
            if scheduler.phase == 1:
                log("Phase I: Alignment active")
            else:
                log("Phase II: Pure denoising")
    """

    def __init__(
        self,
        termination_step: int,
        termination_mode: Literal["hard", "linear", "cosine"] = "hard",
        transition_steps: int = 1000,
        use_attention_alignment: bool = True,
        initial_repa_weight: float = 0.5,
        initial_attention_weight: float = 0.1,
    ):
        if termination_step <= 0:
            raise ValueError(f"termination_step must be positive, got {termination_step}")

        self.termination_step = termination_step
        self.termination_mode = termination_mode
        self.transition_steps = transition_steps
        self.use_attention_alignment = use_attention_alignment
        self.initial_repa_weight = initial_repa_weight
        self.initial_attention_weight = initial_attention_weight

        self._phase = 1

    @property
    def phase(self) -> int:
        """Current training phase (1 = alignment, 2 = pure denoising)."""
        return self._phase

    def get_repa_weight(self, step: int) -> float:
        """Get REPA loss weight at current step.

        Args:
            step: Current training step

        Returns:
            Weight for REPA loss (0 if terminated)
        """
        if step >= self.termination_step + self.transition_steps:
            self._phase = 2
            return 0.0

        if step < self.termination_step:
            self._phase = 1
            return self.initial_repa_weight

        # In transition
        if self.termination_mode == "hard":
            self._phase = 2
            return 0.0

        progress = (step - self.termination_step) / self.transition_steps

        if self.termination_mode == "linear":
            return self.initial_repa_weight * (1 - progress)

        elif self.termination_mode == "cosine":
            return (
                self.initial_repa_weight
                * 0.5
                * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            )

        return 0.0

    def get_attention_weight(self, step: int) -> float:
        """Get attention alignment loss weight at current step."""
        if not self.use_attention_alignment:
            return 0.0

        # Same schedule as REPA
        repa_weight = self.get_repa_weight(step)
        if repa_weight == 0:
            return 0.0

        return self.initial_attention_weight * (repa_weight / self.initial_repa_weight)

    def should_compute_alignment(self, step: int) -> bool:
        """Check if alignment should be computed at this step.

        Useful for skipping expensive encoder forward passes.
        """
        return self.get_repa_weight(step) > 0

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "termination_step": self.termination_step,
            "termination_mode": self.termination_mode,
            "transition_steps": self.transition_steps,
            "use_attention_alignment": self.use_attention_alignment,
            "initial_repa_weight": self.initial_repa_weight,
            "initial_attention_weight": self.initial_attention_weight,
            "phase": self._phase,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self._phase = state.get("phase", 1)


class MultiEncoderREPA(nn.Module):
    """REPA with multiple encoder targets (as in original implementation).

    Supports aligning to multiple encoders simultaneously (e.g., DINOv2 + SigLIP)
    with separate projection heads for each.

    Args:
        hidden_dim: DiT hidden dimension
        encoder_dims: List of encoder dimensions
        proj_dim: Shared projection dimension
        timestep_threshold: Only apply at t > threshold
        weight: Total loss weight (distributed across encoders)

    Usage:
        repa = MultiEncoderREPA(
            hidden_dim=1024,
            encoder_dims=[768, 768],  # DINOv2-B + SigLIP-B
        )

        # Get features from multiple encoders
        dino_features = dino_encoder(images)
        siglip_features = siglip_encoder(images)

        loss = repa(hidden_states, [dino_features, siglip_features], timesteps)
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_dims: list[int],
        proj_dim: int = 256,
        timestep_threshold: float = 0.5,
        weight: float = 0.5,
    ):
        super().__init__()

        self.timestep_threshold = timestep_threshold
        self.weight = weight
        self.num_encoders = len(encoder_dims)

        # Separate projection for each encoder
        self.projections = nn.ModuleList(
            [
                REPAProjection(
                    hidden_dim=hidden_dim,
                    target_dim=enc_dim,
                    proj_dim=proj_dim,
                )
                for enc_dim in encoder_dims
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_features_list: list[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-encoder REPA loss.

        Args:
            hidden_states: DiT hidden states [batch, seq, hidden_dim]
            target_features_list: List of encoder features, each [batch, seq, enc_dim]
            timesteps: Current timesteps [batch]

        Returns:
            Combined alignment loss
        """
        if len(target_features_list) != self.num_encoders:
            raise ValueError(
                f"Expected {self.num_encoders} encoder features, got {len(target_features_list)}"
            )

        mask = timesteps > self.timestep_threshold

        if not mask.any():
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        hidden_masked = hidden_states[mask]

        total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        for projection, target_features in zip(self.projections, target_features_list):
            target_masked = target_features[mask]

            # Project both to alignment space
            proj_hidden = projection(hidden_masked)
            proj_target = projection.project_target(target_masked)  # type: ignore[operator]

            # Normalize
            proj_hidden = F.normalize(proj_hidden, dim=-1, p=2)
            proj_target = F.normalize(proj_target, dim=-1, p=2)

            # Cosine similarity loss
            similarity = (proj_hidden * proj_target).sum(dim=-1)
            loss = -similarity.mean()

            total_loss = total_loss + loss

        # Average over encoders
        total_loss = total_loss / self.num_encoders

        return self.weight * total_loss


__all__ = [
    "REPAProjection",
    "REPALoss",
    "AttentionAlignmentLoss",
    "HASTEScheduler",
    "MultiEncoderREPA",
]
