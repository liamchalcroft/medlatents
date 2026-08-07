"""Collate functions for conditional training data pipelines.

Provides utilities for batching conditional training data with support for
multiple modalities (tokens, class labels, text, spatial conditioning, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ConditionalBatchConfig:
    """Configuration for conditional batch collation.

    Specifies the keys in the dataset dicts that correspond to each
    conditioning modality.

    Attributes:
        tokens_key: Key for token sequence (the main generation target)
        class_key: Key for class labels (None to disable)
        text_embeddings_key: Key for pre-computed text embeddings
        text_pooled_key: Key for pooled text embeddings
        text_mask_key: Key for text attention mask
        spatial_key: Key for spatial conditioning (e.g., input image for super-res)
        spatial_mask_key: Key for spatial conditioning mask
        segmentation_key: Key for segmentation masks
        metadata_key: Key for scanner/acquisition metadata
        raw_image_key: Key for raw images (used for REPA alignment)
    """

    tokens_key: str = "tokens"
    class_key: str | None = "class"
    text_embeddings_key: str | None = None
    text_pooled_key: str | None = None
    text_mask_key: str | None = None
    spatial_key: str | None = None
    spatial_mask_key: str | None = None
    segmentation_key: str | None = None
    metadata_key: str | None = None
    raw_image_key: str | None = None  # For REPA - need clean images

    # Padding configuration
    pad_token_id: int = 0
    max_seq_length: int | None = None


def collate_conditional_batch(
    batch: list[dict[str, Any]],
    config: ConditionalBatchConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Collate function for conditional training data.

    Handles variable-length sequences with padding and stacks all
    conditioning modalities into a single batch dict.

    Args:
        batch: List of sample dicts from dataset
        config: Collation configuration (uses defaults if None)

    Returns:
        Dict with batched tensors for all modalities
    """
    if config is None:
        config = ConditionalBatchConfig()

    result: dict[str, torch.Tensor] = {}

    # Stack tokens (with optional padding)
    if config.tokens_key in batch[0]:
        tokens = [b[config.tokens_key] for b in batch]
        result["tokens"] = _stack_or_pad(tokens, config.pad_token_id, config.max_seq_length)

    # Stack class labels if present
    if config.class_key and config.class_key in batch[0]:
        result["class_labels"] = torch.stack([b[config.class_key] for b in batch])

    # Stack text embeddings if present
    if config.text_embeddings_key and config.text_embeddings_key in batch[0]:
        result["text_embeddings"] = torch.stack([b[config.text_embeddings_key] for b in batch])

    if config.text_pooled_key and config.text_pooled_key in batch[0]:
        result["text_pooled"] = torch.stack([b[config.text_pooled_key] for b in batch])

    if config.text_mask_key and config.text_mask_key in batch[0]:
        result["text_mask"] = torch.stack([b[config.text_mask_key] for b in batch])

    # Stack spatial conditioning if present
    if config.spatial_key and config.spatial_key in batch[0]:
        result["spatial_condition"] = torch.stack([b[config.spatial_key] for b in batch])

    if config.spatial_mask_key and config.spatial_mask_key in batch[0]:
        result["spatial_mask"] = torch.stack([b[config.spatial_mask_key] for b in batch])

    # Stack segmentation masks if present
    if config.segmentation_key and config.segmentation_key in batch[0]:
        result["segmentation_mask"] = torch.stack([b[config.segmentation_key] for b in batch])

    # Stack metadata if present
    if config.metadata_key and config.metadata_key in batch[0]:
        result["metadata"] = torch.stack([b[config.metadata_key] for b in batch])

    # Stack raw images for REPA if present
    if config.raw_image_key and config.raw_image_key in batch[0]:
        result["raw_image"] = torch.stack([b[config.raw_image_key] for b in batch])

    return result


def _stack_or_pad(
    tensors: list[torch.Tensor],
    pad_value: int = 0,
    max_length: int | None = None,
) -> torch.Tensor:
    """Stack tensors with padding for variable lengths.

    Args:
        tensors: List of 1D tensors
        pad_value: Value to use for padding
        max_length: Optional maximum length (truncates if exceeded)

    Returns:
        Padded batch tensor [batch_size, max_seq_len]
    """
    # Check if all same length
    lengths = [len(t) for t in tensors]
    if len(set(lengths)) == 1:
        # All same length, just stack
        stacked = torch.stack(tensors)
        if max_length is not None and stacked.shape[1] > max_length:
            stacked = stacked[:, :max_length]
        return stacked

    # Variable lengths, need to pad
    target_length = max(lengths)
    if max_length is not None:
        target_length = min(target_length, max_length)

    batch_size = len(tensors)
    padded = torch.full(
        (batch_size, target_length),
        pad_value,
        dtype=tensors[0].dtype,
    )

    for i, t in enumerate(tensors):
        length = min(len(t), target_length)
        padded[i, :length] = t[:length]

    return padded


def create_conditional_collate_fn(
    config: ConditionalBatchConfig,
) -> Callable[[Any], dict[str, torch.Tensor]]:
    """Create a collate function with specific configuration.

    Usage:
        config = ConditionalBatchConfig(tokens_key='input_ids', class_key='label')
        collate_fn = create_conditional_collate_fn(config)
        dataloader = DataLoader(dataset, collate_fn=collate_fn)

    Args:
        config: Collation configuration

    Returns:
        Collate function for DataLoader
    """

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_conditional_batch(batch, config)

    return collate_fn


__all__ = [
    "ConditionalBatchConfig",
    "collate_conditional_batch",
    "create_conditional_collate_fn",
]
