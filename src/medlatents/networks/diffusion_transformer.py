"""Diffusion Transformer backbones for discrete and continuous latents."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..configs import create_model_variants
from .transformer import (
    ContinuousTransformer,
    DiscreteTransformer,
    apply_rotary_emb,
    init_weights,
)


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiscreteSequenceEmbedder(nn.Module):
    """Projects discrete token indices to embeddings."""

    def __init__(self, seq_length, vocab_size, hidden_size, allow_dynamic_seq_length=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.allow_dynamic_seq_length = allow_dynamic_seq_length

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        if not self.allow_dynamic_seq_length and x.size(-1) != self.seq_length:
            raise ValueError(f"Expected sequence length {self.seq_length}, got {x.size(-1)}")
        # Validate token values are within vocab range (fail-fast on invalid inputs)
        if x.min() < 0 or x.max() >= self.vocab_size:
            raise ValueError(
                f"Token values must be in [0, {self.vocab_size}), "
                f"got range [{x.min().item()}, {x.max().item()}]"
            )
        return self.token_embedding(x)


class ContinuousSequenceEmbedder(nn.Module):
    """Projects continuous latent codes to the transformer hidden size."""

    def __init__(self, seq_length, in_channels, hidden_size, allow_dynamic_seq_length=False):
        super().__init__()
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.allow_dynamic_seq_length = allow_dynamic_seq_length
        self.proj = nn.Linear(in_channels, hidden_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with shape [batch, seq_len, channels], got {tuple(x.shape)}"
            )

        if not self.allow_dynamic_seq_length and x.size(1) != self.seq_length:
            raise ValueError(f"Expected sequence length {self.seq_length}, got {x.size(1)}")

        if x.size(-1) != self.in_channels:
            raise ValueError(f"Expected feature dimension {self.in_channels}, got {x.size(-1)}")

        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels with support for classifier-free guidance."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drop labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTAttention(nn.Module):
    """Attention block with RoPE for DiT.

    Supports QK LayerNorm for training stability (arXiv:2312.16903).
    """

    def __init__(self, hidden_size, num_heads, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK LayerNorm for stability
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

    def forward(self, x, freqs_cis):
        B, L, C = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Apply QK LayerNorm for stability
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=None)

        x = x.transpose(1, 2).reshape(B, L, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Supports QK LayerNorm for training stability (arXiv:2312.16903).
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads=num_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, freqs_cis):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTCrossAttention(nn.Module):
    """Cross-attention module for DiT with optional QK-norm.

    Attends to conditioning context (e.g., text embeddings) from the main sequence.
    """

    def __init__(self, hidden_size, context_dim, num_heads, qkv_bias=True, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.context_dim = context_dim

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK LayerNorm for stability
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

    def forward(self, x, context, context_mask=None):
        """
        Args:
            x: Query tensor [batch, seq_len, hidden_size]
            context: Key/value context [batch, context_len, context_dim]
            context_mask: Optional mask [batch, context_len], True = attend, False = ignore

        Returns:
            Attended tensor [batch, seq_len, hidden_size]
        """
        B, L, _ = x.shape
        context_len = context.size(1)

        q = self.to_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK LayerNorm if enabled
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Prepare attention mask for scaled_dot_product_attention
        # PyTorch expects: True = mask out (don't attend), so we invert
        attn_mask = None
        if context_mask is not None:
            # context_mask: [B, context_len], True = attend
            # Need: [B, 1, 1, context_len] for broadcasting
            attn_mask = ~context_mask.unsqueeze(1).unsqueeze(2)  # True = don't attend

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, L, -1)
        return self.proj(x)


class DiTBlockWithCrossAttention(nn.Module):
    """DiT block with self-attention, cross-attention, and adaLN-Zero conditioning.

    This extends the standard DiTBlock with an optional cross-attention sublayer
    between self-attention and MLP, enabling conditioning on sequences like text
    embeddings, other image tokens, or multi-modal contexts.

    Architecture (per block):
        x -> adaLN -> self-attn -> + -> adaLN -> cross-attn -> + -> adaLN -> MLP -> +
             ↑                    |      ↑                    |      ↑            |
             c                    x      c                    x      c            x

    Based on architectures from:
    - PixArt-alpha (arXiv:2310.00426)
    - Stable Diffusion 3 (arXiv:2403.03206)
    - FLUX (Black Forest Labs)

    Args:
        hidden_size: Model hidden dimension
        context_dim: Dimension of cross-attention context (e.g., text embeddings)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension multiplier
        qk_norm: Whether to apply QK LayerNorm for stability
    """

    def __init__(
        self,
        hidden_size,
        context_dim,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = context_dim

        # Self-attention with adaLN
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads=num_heads, qk_norm=qk_norm)

        # Cross-attention with adaLN
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = DiTCrossAttention(
            hidden_size, context_dim, num_heads=num_heads, qk_norm=qk_norm
        )

        # MLP with adaLN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # adaLN modulation: 9 parameters (3 per sublayer: shift, scale, gate)
        # self-attn: shift_msa, scale_msa, gate_msa
        # cross-attn: shift_ca, scale_ca, gate_ca
        # MLP: shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, freqs_cis, context=None, context_mask=None):
        """
        Forward pass with optional cross-attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            c: Conditioning vector [batch, hidden_size] (timestep + class embedding)
            freqs_cis: RoPE frequencies
            context: Optional cross-attention context [batch, context_len, context_dim]
            context_mask: Optional mask for context [batch, context_len]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Get all modulation parameters
        modulation = self.adaLN_modulation(c)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(9, dim=1)

        # Self-attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis
        )

        # Cross-attention (only if context provided)
        if context is not None:
            x = x + gate_ca.unsqueeze(1) * self.cross_attn(
                modulate(self.norm_cross(x), shift_ca, scale_ca),
                context,
                context_mask,
            )

        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class MMDiTBlock(nn.Module):
    """Multi-Modal DiT block with joint attention over image and text tokens.

    Inspired by SD3/FLUX architecture where image and text tokens are concatenated
    and processed jointly through self-attention, enabling bidirectional attention
    between modalities.

    This is more computationally efficient than separate cross-attention when
    context lengths are similar to sequence lengths.

    Architecture:
        [image_tokens, text_tokens] -> joint self-attn -> split -> MLPs

    Based on:
    - Stable Diffusion 3 (arXiv:2403.03206)
    - FLUX (Black Forest Labs)

    Args:
        hidden_size: Model hidden dimension
        context_dim: Dimension of text/context tokens (projected to hidden_size)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension multiplier
        qk_norm: Whether to apply QK LayerNorm
    """

    def __init__(
        self,
        hidden_size,
        context_dim,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = context_dim

        # Project context to hidden_size if different
        if context_dim != hidden_size:
            self.context_proj = nn.Linear(context_dim, hidden_size)
        else:
            self.context_proj = nn.Identity()

        # Joint attention (processes concatenated image + text tokens)
        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiTAttention(hidden_size, num_heads=num_heads, qk_norm=qk_norm)

        # Separate MLPs for image and text
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # adaLN modulation for image stream (6 params: attn + mlp)
        self.adaLN_modulation_img = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # adaLN modulation for text stream (6 params: attn + mlp)
        self.adaLN_modulation_txt = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, freqs_cis, context=None, context_mask=None):
        """
        Forward pass with joint multi-modal attention.

        Args:
            x: Image tokens [batch, img_seq_len, hidden_size]
            c: Conditioning vector [batch, hidden_size]
            freqs_cis: RoPE frequencies (for image tokens only, or extended)
            context: Text tokens [batch, txt_seq_len, context_dim]
            context_mask: Optional mask for text [batch, txt_seq_len] (not used in joint attn)

        Returns:
            Output image tokens [batch, img_seq_len, hidden_size]
        """
        if context is None:
            # Fall back to standard DiT block behavior (self-attn only on image)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation_img(c).chunk(6, dim=1)
            )
            x = x + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1_img(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.mlp_img(
                modulate(self.norm2_img(x), shift_mlp, scale_mlp)
            )
            return x

        # Project context to hidden_size
        context = self.context_proj(context)

        img_len = x.size(1)
        txt_len = context.size(1)

        # Get modulation parameters
        shift_msa_img, scale_msa_img, gate_msa_img, shift_mlp_img, scale_mlp_img, gate_mlp_img = (
            self.adaLN_modulation_img(c).chunk(6, dim=1)
        )
        shift_msa_txt, scale_msa_txt, gate_msa_txt, shift_mlp_txt, scale_mlp_txt, gate_mlp_txt = (
            self.adaLN_modulation_txt(c).chunk(6, dim=1)
        )

        # Apply adaLN to both streams
        x_normed = modulate(self.norm1_img(x), shift_msa_img, scale_msa_img)
        ctx_normed = modulate(self.norm1_txt(context), shift_msa_txt, scale_msa_txt)

        # Concatenate for joint attention
        joint = torch.cat([x_normed, ctx_normed], dim=1)

        # Extend freqs_cis for joint sequence if needed
        # For simplicity, we use the same RoPE for text (could use separate or none)
        if freqs_cis.size(0) < img_len + txt_len:
            # Pad with last frequency (or zeros) for text tokens
            padding = freqs_cis[-1:].expand(txt_len, -1)
            freqs_cis_joint = torch.cat([freqs_cis[:img_len], padding], dim=0)
        else:
            freqs_cis_joint = freqs_cis[: img_len + txt_len]

        # Joint self-attention
        joint_out = self.attn(joint, freqs_cis_joint)

        # Split back
        x_attn, ctx_attn = joint_out[:, :img_len], joint_out[:, img_len:]

        # Residual connections with gates
        x = x + gate_msa_img.unsqueeze(1) * x_attn
        context = context + gate_msa_txt.unsqueeze(1) * ctx_attn

        # Separate MLPs
        x = x + gate_mlp_img.unsqueeze(1) * self.mlp_img(
            modulate(self.norm2_img(x), shift_mlp_img, scale_mlp_img)
        )
        # Note: We don't return modified context, but it's updated for potential future use
        # context = context + gate_mlp_txt.unsqueeze(1) * self.mlp_txt(
        #     modulate(self.norm2_txt(context), shift_mlp_txt, scale_mlp_txt)
        # )

        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiscreteDiT(DiscreteTransformer):
    """Diffusion Transformer with adaptive conditioning for discrete flows."""

    def __init__(
        self,
        seq_length=1024,
        vocab_size=100,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=0,
        rope_theta=10000.0,
        allow_dynamic_seq_length=True,
        chunk_size=None,
        masked=False,
        gradient_checkpointing=True,
        qk_norm=False,
    ):
        super().__init__(
            seq_length=seq_length,
            vocab_size=vocab_size + (1 if masked else 0),
            hidden_size=hidden_size,
            num_heads=num_heads,
            rope_theta=rope_theta,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
            chunk_size=chunk_size,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.out_channels = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.class_free_guidance = num_classes > 0 and class_dropout_prob > 0
        self.masked = masked

        # Embedders
        self.x_embedder = DiscreteSequenceEmbedder(
            seq_length,
            self.vocab_size,
            hidden_size,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = (
            LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            if self.class_free_guidance
            else None
        )

        # DiT blocks (with optional QK LayerNorm)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm)
                for _ in range(depth)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.vocab_size)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT paper with zero-init for adaLN and final layer."""
        self.apply(init_weights)

        # Zero-initialize adaLN modulation layers (ensures identity at initialization)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-initialize final layer (ensures identity at initialization)
        nn.init.zeros_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

    def forward(self, x, t, y=None):
        """Forward pass through DiT."""
        seq_len = x.size(-1)
        if seq_len != self.seq_len_cached:
            if not self.allow_dynamic_seq_length:
                raise ValueError(
                    f"Input sequence length {seq_len} does not match "
                    f"model sequence length {self.seq_len_cached}"
                )
            if seq_len > self.seq_len_cached:
                self.update_rotary_cache(seq_len)

        x = self.x_embedder(x)
        t = self.t_embedder(t)
        freqs_cis = self.freqs_cis[:seq_len]

        if self.class_free_guidance:
            if y is None:
                raise ValueError("Model requires class labels but none provided")
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(self.ckpt_wrapper(block), x, c, freqs_cis, use_reentrant=False)
            else:
                x = block(x, c, freqs_cis)

        return self.final_layer(x)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Forward pass with classifier-free guidance."""
        if not self.class_free_guidance:
            raise ValueError("Model was trained without classifier-free guidance")

        if x.size(0) % 2 != 0 or x.size(0) != 2 * y.size(0) or t.size(0) != x.size(0):
            raise ValueError("x and t must have twice the batch size of y")

        seq_len = x.size(-1)
        if seq_len != self.seq_len_cached:
            if not self.allow_dynamic_seq_length:
                raise ValueError(
                    f"Input sequence length {seq_len} does not match "
                    f"model sequence length {self.seq_len_cached}"
                )
            if seq_len > self.seq_len_cached:
                self.update_rotary_cache(seq_len)

        x = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        freqs_cis = self.freqs_cis[:seq_len]

        force_drop = torch.cat([torch.ones_like(y), torch.zeros_like(y)])
        y_emb = self.y_embedder(y.repeat(2), self.training, force_drop_ids=force_drop)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c, freqs_cis)

        x = self.final_layer(x)

        half = x.size(0) // 2
        uncond_out, cond_out = x[:half], x[half:]

        return uncond_out + cfg_scale * (cond_out - uncond_out)


# ---------------------------------------------------------------------------
# Factory helpers (continuous)


def _build_continuous_dit(
    defaults: dict, *, seq_length: int, in_channels: int, **kwargs
) -> ContinuousDiT:
    params = {**defaults, **kwargs}
    return ContinuousDiT(
        seq_length=seq_length,
        in_channels=in_channels,
        **params,
    )


def ContinuousDiT_XL(*, seq_length: int, in_channels: int, **kwargs):
    defaults = {"depth": 48, "hidden_size": 1536, "num_heads": 24}
    return _build_continuous_dit(defaults, seq_length=seq_length, in_channels=in_channels, **kwargs)


def ContinuousDiT_L(*, seq_length: int, in_channels: int, **kwargs):
    defaults = {"depth": 24, "hidden_size": 1024, "num_heads": 16}
    return _build_continuous_dit(defaults, seq_length=seq_length, in_channels=in_channels, **kwargs)


def ContinuousDiT_B(*, seq_length: int, in_channels: int, **kwargs):
    defaults = {"depth": 12, "hidden_size": 768, "num_heads": 12}
    return _build_continuous_dit(defaults, seq_length=seq_length, in_channels=in_channels, **kwargs)


def ContinuousDiT_S(*, seq_length: int, in_channels: int, **kwargs):
    defaults = {"depth": 6, "hidden_size": 384, "num_heads": 6}
    return _build_continuous_dit(defaults, seq_length=seq_length, in_channels=in_channels, **kwargs)


def ContinuousDiT_Nano(*, seq_length: int, in_channels: int, **kwargs):
    defaults = {"depth": 1, "hidden_size": 64, "num_heads": 2}
    return _build_continuous_dit(defaults, seq_length=seq_length, in_channels=in_channels, **kwargs)


ContinuousDiT_models = {
    "ContinuousDiT-XL": ContinuousDiT_XL,
    "ContinuousDiT-L": ContinuousDiT_L,
    "ContinuousDiT-B": ContinuousDiT_B,
    "ContinuousDiT-S": ContinuousDiT_S,
    "ContinuousDiT-Nano": ContinuousDiT_Nano,
}


# Model size configurations (discrete)
DiscreteDiT_models = create_model_variants(DiscreteDiT, "DiscreteDiT")


class ContinuousDiT(ContinuousTransformer):
    """Diffusion Transformer tailored for continuous latent vectors."""

    def __init__(
        self,
        seq_length: int,
        in_channels: int,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        num_classes: int = 0,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        gradient_checkpointing: bool = True,
        out_channels: int | None = None,
        qk_norm: bool = False,
    ) -> None:
        super().__init__(
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rope_theta=rope_theta,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
            chunk_size=chunk_size,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.class_free_guidance = num_classes > 0 and class_dropout_prob > 0

        self.x_embedder = ContinuousSequenceEmbedder(
            seq_length,
            in_channels,
            hidden_size,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = (
            LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            if self.class_free_guidance
            else None
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm)
                for _ in range(depth)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.out_channels),
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights following DiT paper with zero-init for adaLN and final layer."""
        self.apply(init_weights)

        # Zero-initialize adaLN modulation layers (ensures identity at initialization)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-initialize final layer (ensures identity at initialization)
        nn.init.zeros_(self.final_layer[-1].weight)
        nn.init.zeros_(self.final_layer[-1].bias)

    def _maybe_update_rope(self, x: torch.Tensor) -> None:
        seq_len = x.size(1)
        if seq_len != self.seq_len_cached:
            if not self.allow_dynamic_seq_length:
                raise ValueError(
                    f"Input sequence length {seq_len} does not match model sequence length {self.seq_len_cached}"
                )
            if seq_len > self.seq_len_cached:
                self.update_rotary_cache(seq_len)

    def forward(self, x, t, y=None):
        self._maybe_update_rope(x)
        seq_len = x.size(1)

        x = self.x_embedder(x)
        t = self.t_embedder(t)
        freqs_cis = self.freqs_cis[:seq_len]

        if self.class_free_guidance:
            if y is None:
                raise ValueError("Model requires class labels but none provided")
            y = self.y_embedder(y, self.training)
            c = t + y
        else:
            c = t

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    c,
                    freqs_cis,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, freqs_cis)

        return self.final_layer(x)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        if not self.class_free_guidance:
            raise ValueError("Model was trained without classifier-free guidance")

        if x.size(0) % 2 != 0 or x.size(0) != 2 * y.size(0) or t.size(0) != x.size(0):
            raise ValueError("x and t must have twice the batch size of y")

        self._maybe_update_rope(x)
        seq_len = x.size(1)

        x = self.x_embedder(x)
        t_emb = self.t_embedder(t)
        freqs_cis = self.freqs_cis[:seq_len]

        force_drop = torch.cat([torch.ones_like(y), torch.zeros_like(y)])
        y_emb = self.y_embedder(y.repeat(2), self.training, force_drop_ids=force_drop)
        c = t_emb + y_emb

        for block in self.blocks:
            x = block(x, c, freqs_cis)

        x = self.final_layer(x)

        half = x.size(0) // 2
        uncond_out, cond_out = x[:half], x[half:]

        return uncond_out + cfg_scale * (cond_out - uncond_out)
