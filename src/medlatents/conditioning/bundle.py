"""Unified conditioning data structures for generative models.

Provides standardized containers for conditioning inputs during training and inference,
supporting multiple modalities (time, class, text, spatial/image) with CFG dropout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class ConditioningConfig:
    """Configuration for model conditioning capabilities.

    This config specifies what types of conditioning a model supports and how
    they should be processed. Used during model construction to set up the
    appropriate embedders and conditioning pathways.

    Attributes:
        use_timestep: Whether model uses timestep conditioning (required for diffusion/flow)
        timestep_embed_dim: Dimension of timestep embeddings
        use_class: Whether model uses class label conditioning
        num_classes: Number of classes (if class conditioning enabled)
        class_dropout_prob: Probability of dropping class labels for CFG training
        use_text: Whether model uses text conditioning
        text_embed_dim: Dimension of per-token text embeddings
        text_pooled_dim: Dimension of pooled text embeddings
        max_text_length: Maximum text sequence length
        use_spatial: Whether model uses spatial conditioning (inpainting, super-res, etc.)
        spatial_channels: Number of channels in spatial conditioning
        spatial_method: How to inject spatial conditioning
        block_conditioning: Conditioning method for transformer blocks
        fusion_method: How to fuse multiple conditioning types
    """

    # Timestep conditioning (required for diffusion/flow)
    use_timestep: bool = True
    timestep_embed_dim: int = 256

    # Class conditioning
    use_class: bool = False
    num_classes: int = 0
    class_dropout_prob: float = 0.1  # For CFG

    # Text conditioning
    use_text: bool = False
    text_embed_dim: int = 768
    text_pooled_dim: int = 768
    max_text_length: int = 77

    # Image/spatial conditioning (for inpainting, super-res, multi-contrast MRI, etc.)
    use_spatial: bool = False
    spatial_channels: int = 0
    spatial_method: Literal["concat", "cross_attention", "film", "add"] = "concat"

    # Segmentation mask conditioning (for anatomy-aware generation)
    use_segmentation: bool = False
    num_segmentation_classes: int = 0

    # Metadata conditioning (scanner parameters, acquisition settings, etc.)
    use_metadata: bool = False
    metadata_dim: int = 0

    # Block conditioning method
    block_conditioning: Literal[
        "adaln", "adaln_zero", "adain", "adain_zero", "film", "cross_attention"
    ] = "adaln_zero"

    # Multi-modal fusion
    fusion_method: Literal["add", "concat", "cross_attention", "gated"] = "add"

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.use_class and self.num_classes <= 0:
            raise ValueError("num_classes must be > 0 when use_class=True")
        if self.use_spatial and self.spatial_channels <= 0:
            raise ValueError("spatial_channels must be > 0 when use_spatial=True")
        if self.use_segmentation and self.num_segmentation_classes <= 0:
            raise ValueError("num_segmentation_classes must be > 0 when use_segmentation=True")
        if self.use_metadata and self.metadata_dim <= 0:
            raise ValueError("metadata_dim must be > 0 when use_metadata=True")


@dataclass
class ConditioningBundle:
    """Standardized container for all conditioning inputs.

    This dataclass provides a unified interface for passing conditioning information
    through models during training and inference. It supports multiple conditioning
    modalities and handles CFG dropout automatically.

    Usage:
        # Training
        bundle = ConditioningBundle.from_batch(batch_dict)
        bundle = bundle.apply_cfg_dropout(dropout_prob=0.1)
        output = model(x, bundle)

        # Inference with CFG
        bundle = ConditioningBundle(timesteps=t, class_labels=y)
        bundle_null = bundle.get_null_bundle()
        cond_out = model(x, bundle)
        uncond_out = model(x, bundle_null)
        output = uncond_out + scale * (cond_out - uncond_out)

    Attributes:
        timesteps: Diffusion/flow timesteps [batch]
        class_labels: Class label indices [batch]
        text_embeddings: Per-token text embeddings [batch, seq, dim]
        text_pooled: Pooled text embeddings [batch, dim]
        text_mask: Attention mask for text [batch, seq]
        spatial_condition: Spatial conditioning [batch, seq, channels] or [batch, C, H, W]
        spatial_mask: Mask for spatial conditioning (which positions are conditioned)
        segmentation_mask: Segmentation labels [batch, seq] or [batch, H, W]
        metadata: Scanner/acquisition metadata [batch, dim]
        null_class: Null class index for CFG
        null_text: Null text embedding for CFG
    """

    # Required
    timesteps: torch.Tensor

    # Optional conditioning (None = not used)
    class_labels: torch.Tensor | None = None
    text_embeddings: torch.Tensor | None = None
    text_pooled: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    spatial_condition: torch.Tensor | None = None
    spatial_mask: torch.Tensor | None = None
    segmentation_mask: torch.Tensor | None = None
    metadata: torch.Tensor | None = None

    # For CFG (null conditioning)
    null_class: int | None = None
    null_text: torch.Tensor | None = None

    # Internal flags
    _is_null: bool = field(default=False, repr=False)

    @property
    def batch_size(self) -> int:
        """Get batch size from timesteps."""
        return self.timesteps.shape[0]

    @property
    def device(self) -> torch.device:
        """Get device from timesteps."""
        return self.timesteps.device

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "ConditioningBundle":
        """Move all tensors to specified device with batched transfers."""

        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(device, non_blocking=non_blocking) if t is not None else None

        return ConditioningBundle(
            timesteps=self.timesteps.to(device, non_blocking=non_blocking),
            class_labels=_move(self.class_labels),
            text_embeddings=_move(self.text_embeddings),
            text_pooled=_move(self.text_pooled),
            text_mask=_move(self.text_mask),
            spatial_condition=_move(self.spatial_condition),
            spatial_mask=_move(self.spatial_mask),
            segmentation_mask=_move(self.segmentation_mask),
            metadata=_move(self.metadata),
            null_class=self.null_class,
            null_text=_move(self.null_text),
            _is_null=self._is_null,
        )

    def apply_cfg_dropout(
        self,
        class_dropout_prob: float = 0.0,
        text_dropout_prob: float = 0.0,
        force_drop_ids: torch.Tensor | None = None,
    ) -> "ConditioningBundle":
        """Apply CFG dropout to conditioning (for training).

        Randomly drops conditioning to enable classifier-free guidance at inference.

        Args:
            class_dropout_prob: Probability of dropping class labels
            text_dropout_prob: Probability of dropping text conditioning
            force_drop_ids: Optional tensor [batch] of 1s to force drop, 0s to keep

        Returns:
            New ConditioningBundle with dropout applied
        """
        new_class_labels = self.class_labels
        new_text_embeddings = self.text_embeddings
        new_text_pooled = self.text_pooled

        batch_size = self.batch_size
        device = self.device

        # Class dropout
        if self.class_labels is not None and class_dropout_prob > 0:
            if force_drop_ids is not None:
                drop_ids = force_drop_ids == 1
            else:
                drop_ids = torch.rand(batch_size, device=device) < class_dropout_prob

            if self.null_class is not None:
                new_class_labels = torch.where(
                    drop_ids,
                    torch.full_like(self.class_labels, self.null_class),
                    self.class_labels,
                )
            else:
                # Use num_classes as null token (common convention)
                null_val = int(self.class_labels.max().item()) + 1
                new_class_labels = torch.where(
                    drop_ids,
                    torch.full_like(self.class_labels, null_val),
                    self.class_labels,
                )

        # Text dropout
        if self.text_embeddings is not None and text_dropout_prob > 0:
            drop_ids = torch.rand(batch_size, device=device) < text_dropout_prob

            if self.null_text is not None:
                # Replace with null text embedding
                null_expanded = self.null_text.unsqueeze(0).expand(batch_size, -1, -1)
                new_text_embeddings = torch.where(
                    drop_ids.view(-1, 1, 1),
                    null_expanded,
                    self.text_embeddings,
                )
                if self.text_pooled is not None:
                    null_pooled = self.null_text.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
                    new_text_pooled = torch.where(
                        drop_ids.view(-1, 1),
                        null_pooled,
                        self.text_pooled,
                    )
            else:
                # Zero out text embeddings
                new_text_embeddings = torch.where(
                    drop_ids.view(-1, 1, 1),
                    torch.zeros_like(self.text_embeddings),
                    self.text_embeddings,
                )
                if self.text_pooled is not None:
                    new_text_pooled = torch.where(
                        drop_ids.view(-1, 1),
                        torch.zeros_like(self.text_pooled),
                        self.text_pooled,
                    )

        return ConditioningBundle(
            timesteps=self.timesteps,
            class_labels=new_class_labels,
            text_embeddings=new_text_embeddings,
            text_pooled=new_text_pooled,
            text_mask=self.text_mask,
            spatial_condition=self.spatial_condition,
            spatial_mask=self.spatial_mask,
            segmentation_mask=self.segmentation_mask,
            metadata=self.metadata,
            null_class=self.null_class,
            null_text=self.null_text,
            _is_null=False,
        )

    def get_null_bundle(self) -> "ConditioningBundle":
        """Get null (unconditional) version of this bundle for CFG inference.

        Returns:
            New ConditioningBundle with all conditioning replaced by null values
        """
        null_class_labels = None
        if self.class_labels is not None:
            if self.null_class is not None:
                null_class_labels = torch.full_like(self.class_labels, self.null_class)
            else:
                null_val = int(self.class_labels.max().item()) + 1
                null_class_labels = torch.full_like(self.class_labels, null_val)

        null_text_embeddings = None
        null_text_pooled = None
        if self.text_embeddings is not None:
            if self.null_text is not None:
                null_text_embeddings = self.null_text.unsqueeze(0).expand(self.batch_size, -1, -1)
                if self.text_pooled is not None:
                    null_text_pooled = (
                        self.null_text.mean(dim=0).unsqueeze(0).expand(self.batch_size, -1)
                    )
            else:
                null_text_embeddings = torch.zeros_like(self.text_embeddings)
                if self.text_pooled is not None:
                    null_text_pooled = torch.zeros_like(self.text_pooled)

        return ConditioningBundle(
            timesteps=self.timesteps,
            class_labels=null_class_labels,
            text_embeddings=null_text_embeddings,
            text_pooled=null_text_pooled,
            text_mask=self.text_mask,  # Keep mask (for attention)
            spatial_condition=None,  # Drop spatial conditioning
            spatial_mask=None,
            segmentation_mask=None,  # Drop segmentation
            metadata=None,  # Drop metadata
            null_class=self.null_class,
            null_text=self.null_text,
            _is_null=True,
        )

    @classmethod
    def from_batch(
        cls,
        batch: dict[str, torch.Tensor],
        timesteps: torch.Tensor | None = None,
        config: ConditioningConfig | None = None,
    ) -> "ConditioningBundle":
        """Create bundle from dictionary batch (for DataLoader compatibility).

        Args:
            batch: Dictionary with keys like 'tokens', 'class_labels', 'text_embeddings', etc.
            timesteps: Timestep tensor (required if not in batch)
            config: Optional config to determine null values

        Returns:
            ConditioningBundle populated from batch
        """
        # Get timesteps
        if timesteps is not None:
            t = timesteps
        elif "timesteps" in batch:
            t = batch["timesteps"]
        elif "t" in batch:
            t = batch["t"]
        else:
            raise ValueError("timesteps must be provided or present in batch")

        # Determine null class
        null_class = None
        if config is not None and config.use_class:
            null_class = config.num_classes  # Common convention

        def get_first_present(*keys: str) -> torch.Tensor | None:
            """Get first key that exists in batch."""
            for key in keys:
                if key in batch:
                    return batch[key]
            return None

        return cls(
            timesteps=t,
            class_labels=get_first_present("class_labels", "y", "labels"),
            text_embeddings=get_first_present("text_embeddings", "text_emb"),
            text_pooled=get_first_present("text_pooled", "pooled_text"),
            text_mask=get_first_present("text_mask", "attention_mask"),
            spatial_condition=get_first_present(
                "spatial_condition", "condition", "input_condition"
            ),
            spatial_mask=get_first_present("spatial_mask", "condition_mask"),
            segmentation_mask=get_first_present("segmentation_mask", "seg", "labels_seg"),
            metadata=get_first_present("metadata", "scanner_params"),
            null_class=null_class,
            null_text=batch.get("null_text"),
        )

    def get_combined_embedding(
        self,
        t_emb: torch.Tensor,
        y_emb: torch.Tensor | None = None,
        text_pooled_emb: torch.Tensor | None = None,
        metadata_emb: torch.Tensor | None = None,
        fusion: Literal["add", "concat"] = "add",
    ) -> torch.Tensor:
        """Combine multiple conditioning embeddings into single vector.

        This is a utility for models that need a single conditioning vector
        (e.g., for AdaLN modulation).

        Args:
            t_emb: Timestep embedding [batch, dim]
            y_emb: Optional class embedding [batch, dim]
            text_pooled_emb: Optional pooled text embedding [batch, dim]
            metadata_emb: Optional metadata embedding [batch, dim]
            fusion: How to combine ('add' or 'concat')

        Returns:
            Combined embedding [batch, dim] or [batch, combined_dim]
        """
        embeddings = [t_emb]

        if y_emb is not None:
            embeddings.append(y_emb)
        if text_pooled_emb is not None:
            embeddings.append(text_pooled_emb)
        if metadata_emb is not None:
            embeddings.append(metadata_emb)

        if fusion == "add":
            return sum(embeddings)  # type: ignore
        elif fusion == "concat":
            return torch.cat(embeddings, dim=-1)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")


__all__ = [
    "ConditioningConfig",
    "ConditioningBundle",
]
