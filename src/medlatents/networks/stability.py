"""Transformer stability improvements from recent literature (2024-2025).

This module provides stability enhancements for transformer training:
- QK LayerNorm: Prevents attention score explosions
- Peri-LN: Peripheral LayerNorm for input AND output normalization
- Embed LayerNorm: Normalize embeddings to prevent loss spikes
- RMSNorm: Efficient alternative to LayerNorm

References:
- QK-LN: "Spike No More" (arXiv:2312.16903)
- Peri-LN: "Peri-LN" (arXiv:2502.02732)
- DeepNorm: Microsoft Research (arXiv:2403.09635)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't compute mean.
    Used in LLaMA, Gemma, and other modern transformers.

    Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class QKNorm(nn.Module):
    """LayerNorm applied to Q and K projections before attention.

    Prevents attention score explosions and gradient spikes.
    Essential for stable training of deep transformers.

    Reference: "Spike No More" (arXiv:2312.16903)
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim, eps=eps)
        self.k_norm = nn.LayerNorm(head_dim, eps=eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Q and K tensors.

        Args:
            q: Query tensor [batch, heads, seq, head_dim]
            k: Key tensor [batch, heads, seq, head_dim]

        Returns:
            Normalized (q, k) tuple
        """
        # Apply LayerNorm across head_dim
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k


class EmbedLayerNorm(nn.Module):
    """LayerNorm applied after embedding layer.

    Prevents loss spikes in early training by normalizing
    embedding outputs before the transformer blocks.

    Reference: Le Scao et al. 2022, "Spike No More"
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SoftmaxCapping(nn.Module):
    """Soft capping for attention logits to prevent overflow.

    Applies tanh-based soft cap: cap * tanh(logits / cap)
    Used in Gemma 2 and other recent models.
    """

    def __init__(self, cap: float = 50.0):
        super().__init__()
        self.cap = cap

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.cap * torch.tanh(logits / self.cap)


class StableAttention(nn.Module):
    """Attention with QK LayerNorm and optional softmax capping.

    Combines multiple stability techniques:
    - QK LayerNorm before attention computation
    - Optional softmax capping to prevent overflow
    - Proper scaling for numerical stability
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        softmax_cap: float | None = None,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK LayerNorm for stability
        self.qk_norm = QKNorm(self.head_dim) if qk_norm else None

        # Softmax capping
        self.softmax_cap = SoftmaxCapping(softmax_cap) if softmax_cap else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        apply_rotary_fn=None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings if provided
        if freqs_cis is not None and apply_rotary_fn is not None:
            q = apply_rotary_fn(q, freqs_cis)
            k = apply_rotary_fn(k, freqs_cis)

        # Apply QK LayerNorm for stability
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Use Flash Attention when available
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=self.causal and attention_mask is None,
            dropout_p=0.0,
            scale=None,
        )

        x = x.transpose(1, 2).reshape(B, L, C)
        return self.proj(x)


class PeriLNBlock(nn.Module):
    """Transformer block with Peripheral LayerNorm (Peri-LN).

    Applies LayerNorm at both input AND output of sublayers.
    This eliminates early-stage training instability observed with Pre-LN.

    Architecture:
        x = x + norm_out(attn(norm_in(x)))
        x = x + norm_out(mlp(norm_in(x)))

    Reference: "Peri-LN" (arXiv:2502.02732)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        # Input norms (Pre-LN style)
        self.norm1_in = nn.LayerNorm(hidden_size)
        self.norm2_in = nn.LayerNorm(hidden_size)

        # Output norms (the Peri-LN addition)
        self.norm1_out = nn.LayerNorm(hidden_size)
        self.norm2_out = nn.LayerNorm(hidden_size)

        # Attention with QK-Norm
        self.attn = StableAttention(hidden_size, num_heads, qk_norm=qk_norm, causal=causal)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        apply_rotary_fn=None,
    ) -> torch.Tensor:
        # Attention with Peri-LN
        attn_out = self.attn(
            self.norm1_in(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            apply_rotary_fn=apply_rotary_fn,
        )
        x = x + self.norm1_out(attn_out)

        # MLP with Peri-LN
        mlp_out = self.mlp(self.norm2_in(x))
        x = x + self.norm2_out(mlp_out)

        return x


class DeepNormBlock(nn.Module):
    """Transformer block with DeepNorm for very deep transformers.

    DeepNorm modifies residual connections with scaling factors,
    enabling stable training of 200+ layer transformers.

    Reference: Microsoft Research (arXiv:2403.09635)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        qk_norm: bool = True,
        alpha: float = 1.0,  # Residual scale (computed from depth)
    ):
        super().__init__()
        self.alpha = alpha

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn = StableAttention(hidden_size, num_heads, qk_norm=qk_norm, causal=causal)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        apply_rotary_fn=None,
    ) -> torch.Tensor:
        # DeepNorm: x * alpha + sublayer(norm(x))
        x = x * self.alpha + self.attn(
            self.norm1(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            apply_rotary_fn=apply_rotary_fn,
        )
        x = x * self.alpha + self.mlp(self.norm2(x))
        return x


def compute_deepnorm_alpha(depth: int) -> float:
    """Compute DeepNorm alpha scaling factor based on model depth.

    Args:
        depth: Number of transformer layers

    Returns:
        Alpha scaling factor for residual connections
    """
    # From DeepNorm paper: alpha = (2 * depth) ** 0.25
    return (2 * depth) ** 0.25


def compute_deepnorm_beta(depth: int) -> float:
    """Compute DeepNorm beta for weight initialization.

    Args:
        depth: Number of transformer layers

    Returns:
        Beta factor for initializing sublayer weights
    """
    # From DeepNorm paper: beta = (8 * depth) ** -0.25
    return (8 * depth) ** -0.25


def initialize_deepnorm_weights(model: nn.Module, depth: int):
    """Initialize weights for DeepNorm training.

    Scales sublayer weights by beta factor for stable training.

    Args:
        model: Model containing DeepNormBlocks
        depth: Number of transformer layers
    """
    beta = compute_deepnorm_beta(depth)

    for module in model.modules():
        if isinstance(module, DeepNormBlock):
            # Scale attention output projection
            if hasattr(module.attn, "proj"):
                module.attn.proj.weight.data.mul_(beta)

            # Scale MLP output projection
            if isinstance(module.mlp, nn.Sequential):
                for layer in module.mlp:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.mul_(beta)
