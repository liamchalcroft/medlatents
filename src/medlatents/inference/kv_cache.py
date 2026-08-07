"""KV-cache management with optional quantization for memory-efficient generation.

Provides INT8 quantization for KV-cache to reduce memory usage by ~4x during
autoregressive generation of long sequences.

Reference:
- "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- Various KV-cache quantization papers (2023-2024)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class QuantizationConfig:
    """Configuration for KV-cache quantization.

    Attributes:
        enabled: Whether to use quantization
        dtype: Quantization dtype (int8 or fp16)
        per_token: Use per-token quantization (more accurate) vs per-tensor
        symmetric: Use symmetric quantization (faster) vs asymmetric (more accurate)
    """

    enabled: bool = True
    dtype: Literal["int8", "fp16"] = "int8"
    per_token: bool = True
    symmetric: bool = True


class QuantizedKVCache:
    """Memory-efficient KV-cache with INT8 quantization.

    Reduces memory usage by ~4x compared to FP32 or ~2x compared to FP16.
    Uses per-token symmetric quantization for a good accuracy/speed tradeoff.

    Usage::

        cache = QuantizedKVCache(num_layers=12, config=QuantizationConfig())

        # During generation
        for token in tokens:
            for layer_idx, block in enumerate(model.blocks):
                past_kv = cache.get(layer_idx)
                output, new_kv = block.forward_with_cache(x, freqs, past_kv)
                cache.update(layer_idx, new_kv)
    """

    def __init__(
        self,
        num_layers: int,
        config: QuantizationConfig | None = None,
        max_length: int | None = None,
    ):
        """Initialize quantized KV-cache.

        Args:
            num_layers: Number of transformer layers
            config: Quantization configuration
            max_length: Optional maximum sequence length (for pre-allocation)
        """
        self.num_layers = num_layers
        self.config = config or QuantizationConfig()
        self.max_length = max_length

        # Storage for quantized K/V and their scales
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers
        self._key_scales: list[torch.Tensor | None] = [None] * num_layers
        self._value_scales: list[torch.Tensor | None] = [None] * num_layers

        # Track current length
        self._current_length = 0

    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT8 with per-token scaling.

        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]

        Returns:
            quantized: INT8 tensor
            scales: Scale factors for dequantization
        """
        if not self.config.enabled:
            return x, torch.ones(1, device=x.device, dtype=x.dtype)

        if self.config.dtype == "fp16":
            return x.half(), torch.ones(1, device=x.device, dtype=torch.float16)

        # INT8 quantization
        if self.config.per_token:
            # Per-token: scale computed per [batch, heads, seq_pos]
            # x: [B, H, L, D] -> compute max over D
            abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            # Per-tensor: single scale for entire tensor
            abs_max = x.abs().amax().clamp(min=1e-8)

        # Scale to [-127, 127] range
        scale = abs_max / 127.0
        quantized = (x / scale).round().clamp(-127, 127).to(torch.int8)

        return quantized, scale

    def _dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize INT8 tensor back to float.

        Args:
            quantized: INT8 tensor
            scale: Scale factors
            dtype: Output dtype

        Returns:
            Dequantized tensor
        """
        if not self.config.enabled:
            return quantized.to(dtype)

        if self.config.dtype == "fp16":
            return quantized.to(dtype)

        return quantized.to(dtype) * scale.to(dtype)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get dequantized K/V for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            (key, value) tuple or None if cache is empty
        """
        k = self._keys[layer_idx]
        v = self._values[layer_idx]
        k_scale = self._key_scales[layer_idx]
        v_scale = self._value_scales[layer_idx]

        if k is None or v is None or k_scale is None or v_scale is None:
            return None

        key = self._dequantize(k, k_scale, dtype=torch.float32)
        value = self._dequantize(v, v_scale, dtype=torch.float32)

        return key, value

    def update(
        self,
        layer_idx: int,
        new_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Update cache with new K/V, quantizing the new values.

        Args:
            layer_idx: Layer index
            new_kv: (key, value) tuple from forward_with_cache
        """
        new_key, new_value = new_kv

        # Quantize new K/V
        q_key, key_scale = self._quantize(new_key)
        q_value, value_scale = self._quantize(new_value)

        # Store (overwrites previous - the new_kv from forward_with_cache
        # already includes concatenated past values)
        self._keys[layer_idx] = q_key
        self._values[layer_idx] = q_value
        self._key_scales[layer_idx] = key_scale
        self._value_scales[layer_idx] = value_scale

        self._current_length = new_key.size(2)

    def clear(self) -> None:
        """Clear the cache."""
        for i in range(self.num_layers):
            self._keys[i] = None
            self._values[i] = None
            self._key_scales[i] = None
            self._value_scales[i] = None
        self._current_length = 0

    @property
    def length(self) -> int:
        """Current cached sequence length."""
        return self._current_length

    def memory_usage(self) -> dict[str, float]:
        """Compute current memory usage in MB.

        Returns:
            Dict with memory breakdown
        """
        total_bytes = 0
        scale_bytes = 0

        for i in range(self.num_layers):
            k = self._keys[i]
            v = self._values[i]
            k_scale = self._key_scales[i]
            v_scale = self._value_scales[i]

            if k is not None and v is not None:
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()

                if k_scale is not None and v_scale is not None:
                    scale_bytes += k_scale.numel() * k_scale.element_size()
                    scale_bytes += v_scale.numel() * v_scale.element_size()

        return {
            "total_mb": (total_bytes + scale_bytes) / (1024 * 1024),
            "kv_mb": total_bytes / (1024 * 1024),
            "scales_mb": scale_bytes / (1024 * 1024),
            "compression_ratio": 4.0
            if self.config.enabled and self.config.dtype == "int8"
            else 1.0,
        }


class PagedKVCache:
    """Paged KV-cache for efficient memory management with variable-length sequences.

    Inspired by vLLM's PagedAttention. Allocates memory in fixed-size blocks
    to reduce fragmentation and enable efficient batched generation.

    This is a simplified implementation - full paged attention requires
    custom CUDA kernels for optimal performance.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ):
        """Initialize paged KV-cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            block_size: Tokens per block (default 16)
            num_blocks: Total blocks in the pool
            dtype: Storage dtype
            device: Device for storage
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = torch.device(device)

        # Pre-allocate block pool: [num_blocks, 2, num_layers, num_heads, block_size, head_dim]
        # 2 for key/value
        self.block_pool = torch.zeros(
            num_blocks,
            2,
            num_layers,
            num_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=self.device,
        )

        self.free_blocks: deque[int] = deque(range(num_blocks))

        # Per-sequence block tables: maps sequence -> list of block indices
        self.block_tables: dict[int, list[int]] = {}
        self.sequence_lengths: dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, initial_length: int = 0) -> None:
        """Allocate blocks for a new sequence.

        Args:
            seq_id: Unique sequence identifier
            initial_length: Initial sequence length (allocates blocks)
        """
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already allocated")

        num_blocks_needed = (initial_length + self.block_size - 1) // self.block_size

        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough blocks: need {num_blocks_needed}, have {len(self.free_blocks)}"
            )

        allocated = [self.free_blocks.popleft() for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated
        self.sequence_lengths[seq_id] = initial_length

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence.

        Args:
            seq_id: Sequence identifier
        """
        if seq_id not in self.block_tables:
            return

        self.free_blocks.extend(self.block_tables[seq_id])
        del self.block_tables[seq_id]
        del self.sequence_lengths[seq_id]

    def append(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append new K/V to a sequence's cache.

        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            key: New key tensor [1, num_heads, 1, head_dim] (single token)
            value: New value tensor [1, num_heads, 1, head_dim]
        """
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)

        current_len = self.sequence_lengths[seq_id]
        block_idx = current_len // self.block_size
        slot_idx = current_len % self.block_size

        # Allocate new block if needed
        if block_idx >= len(self.block_tables[seq_id]):
            if not self.free_blocks:
                raise RuntimeError("No free blocks available")
            self.block_tables[seq_id].append(self.free_blocks.pop())

        physical_block = self.block_tables[seq_id][block_idx]

        # Store K/V
        self.block_pool[physical_block, 0, layer_idx, :, slot_idx, :] = key.squeeze(0).squeeze(1)
        self.block_pool[physical_block, 1, layer_idx, :, slot_idx, :] = value.squeeze(0).squeeze(1)

        self.sequence_lengths[seq_id] = current_len + 1

    def get(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get K/V cache for a sequence and layer.

        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index

        Returns:
            (key, value) tensors or None if empty
        """
        if seq_id not in self.block_tables:
            return None

        seq_len = self.sequence_lengths[seq_id]
        if seq_len == 0:
            return None

        blocks = self.block_tables[seq_id]

        # Gather K/V from blocks
        keys = []
        values = []

        remaining = seq_len
        for block_id in blocks:
            tokens_in_block = min(remaining, self.block_size)
            keys.append(self.block_pool[block_id, 0, layer_idx, :, :tokens_in_block, :])
            values.append(self.block_pool[block_id, 1, layer_idx, :, :tokens_in_block, :])
            remaining -= tokens_in_block
            if remaining <= 0:
                break

        # Concatenate: [num_heads, total_len, head_dim] -> [1, num_heads, total_len, head_dim]
        key = torch.cat(keys, dim=1).unsqueeze(0)
        value = torch.cat(values, dim=1).unsqueeze(0)

        return key, value

    def memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics.

        Returns:
            Dict with memory info
        """
        total_bytes = self.block_pool.numel() * self.block_pool.element_size()
        used_blocks = self.num_blocks - len(self.free_blocks)

        return {
            "total_mb": total_bytes / (1024 * 1024),
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "utilization": used_blocks / self.num_blocks,
        }


__all__ = [
    "QuantizationConfig",
    "QuantizedKVCache",
    "PagedKVCache",
]
