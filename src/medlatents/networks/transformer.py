"""Core transformer infrastructure for discrete latent models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float

from ..utils import SpecialTokenIds

# ============================================================================
# Weight Initialization
# ============================================================================


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Standard transformer weight initialization (GPT-2 style).

    Args:
        module: The module to initialize
        std: Standard deviation for normal initialization (default: 0.02)
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=std)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================================
# RoPE (Rotary Position Embeddings) utilities
# ============================================================================


def precompute_freqs_cis(dim: int, length: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE (Rotary Position Embedding) rotation frequencies.

    RoPE encodes position information by rotating query and key vectors in the
    attention mechanism. This is more robust than learned position embeddings
    and naturally extends to longer sequences at inference time.

    The rotation is applied pairwise to dimensions: (d₀,d₁), (d₂,d₃), ..., (d_{dim-2},d_{dim-1})
    Each pair rotates by an angle θ·position, where θ decreases exponentially with dimension.

    Args:
        dim: Head dimension (must be even). Each pair of dimensions forms a 2D rotation.
        length: Maximum sequence length to precompute
        theta: Base frequency (default 10000, as in RoFormer paper)

    Returns:
        Complex rotation frequencies [length, dim//2] in polar form e^{iθ·pos}
        These are multiplied with complex views of Q and K to apply rotations.

    Reference:
        Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        https://arxiv.org/abs/2104.09864
    """
    if dim % 2 != 0:
        raise ValueError(f"Dimension {dim} must be even for RoPE (got {dim})")

    # Compute inverse frequencies: θ_d = 1 / (base^(2d/dim)) for d = 0, 2, 4, ...
    # This creates slower rotations for higher dimensions
    index = torch.arange(0, dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (index / dim))

    # Create rotation angles: position × θ_d for all positions and dimensions
    time = torch.arange(length, dtype=torch.float32)
    freqs = torch.outer(time, inv_freq)  # [length, dim//2]

    # Convert to complex exponentials e^{iθ} using polar form (magnitude=1, angle=θ)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings using real-valued operations.

    This implementation avoids complex number operations which don't support BF16,
    instead computing the rotation directly using cos/sin components.

    RoPE rotates adjacent pairs of dimensions: [x₀, x₁], [x₂, x₃], ..., [x_{D-2}, x_{D-1}]
    Each pair is treated as a 2D vector and rotated by an angle θ·position.

    Mathematically:
        [x'₀]   [cos(θ)  -sin(θ)] [x₀]
        [x'₁] = [sin(θ)   cos(θ)] [x₁]

    Args:
        x: Input tensor [batch, heads, seq_len, head_dim]
        freqs_cis: Precomputed rotation frequencies [seq_len, head_dim//2] (complex)

    Returns:
        Rotated tensor with same shape as input
    """
    B, H, L, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"Head dimension must be even for RoPE (got {D})")

    # Extract cos and sin from complex frequencies (works with any dtype)
    # freqs_cis is complex64, extract real (cos) and imag (sin) parts
    freqs_cos = freqs_cis.real.to(dtype=x.dtype, device=x.device)  # [L, D//2]
    freqs_sin = freqs_cis.imag.to(dtype=x.dtype, device=x.device)  # [L, D//2]

    # Reshape x to separate even and odd dimensions: [B, H, L, D] -> [B, H, L, D//2, 2]
    x_reshape = x.view(B, H, L, D // 2, 2)
    x_even = x_reshape[..., 0]  # [B, H, L, D//2]
    x_odd = x_reshape[..., 1]  # [B, H, L, D//2]

    # Broadcast cos/sin to match batch and head dimensions
    # [L, D//2] -> [1, 1, L, D//2]
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation: (cos * x_even - sin * x_odd, sin * x_even + cos * x_odd)
    x_even_rot = freqs_cos * x_even - freqs_sin * x_odd
    x_odd_rot = freqs_sin * x_even + freqs_cos * x_odd

    # Interleave back: [B, H, L, D//2] -> [B, H, L, D]
    result = torch.stack([x_even_rot, x_odd_rot], dim=-1).view(B, H, L, D)

    return result


# ============================================================================
# Normalization Layers
# ============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm is a simpler and faster alternative to LayerNorm that only
    normalizes by RMS (root mean square) without centering (mean subtraction).
    This reduces computation and is shown to be equally effective.

    Used in:
    - LLaMA (Meta)
    - Mistral
    - Modern diffusion transformers

    Formula:
        RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma

    Reference:
        Zhang & Sennrich, "Root Mean Square Layer Normalization"
        https://arxiv.org/abs/1910.07467

    Args:
        dim: Feature dimension to normalize
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor with same shape
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RMSNormZero(nn.Module):
    """RMSNorm with zero-initialized modulation (gate) for stable training.

    Similar to AdaLN-Zero but using RMSNorm instead of LayerNorm.
    Starts as identity and gradually learns normalization.

    Args:
        dim: Feature dimension to normalize
        condition_dim: Conditioning dimension for modulation
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, condition_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

        # Learnable scale (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

        # Modulation: predict scale_mod and gate from condition
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, dim * 2),  # scale_mod, gate
        )

        # Zero-init for stable training start
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            condition: Conditioning tensor [batch, condition_dim]

        Returns:
            Modulated normalized tensor
        """
        # RMS normalize
        x_norm = self._norm(x.float()).type_as(x)
        x_norm = x_norm * self.weight

        # Get modulation parameters
        mod = self.modulation(condition)
        while mod.ndim < x.ndim:
            mod = mod.unsqueeze(1)
        scale_mod, gate = mod.chunk(2, dim=-1)

        # Apply gated modulation
        return x + gate * (x_norm * (1 + scale_mod))


# ============================================================================
# Attention and Transformer Block
# ============================================================================


class Attention(nn.Module):
    """
    Self-attention with RoPE supporting both causal and bidirectional modes.

    Supports QK LayerNorm for training stability (arXiv:2312.16903).
    Supports KV-caching for efficient autoregressive generation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        causal: bool = False,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.causal = causal

        # Combined QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK LayerNorm for stability (prevents attention score explosions)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len hidden"],
        freqs_cis: Complex[torch.Tensor, "seq_len head_dim"],
        attention_mask: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "batch seq_len hidden"]:
        B, L, C = x.shape

        # Fused QKV transform
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Apply QK LayerNorm for stability (after RoPE, before attention)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

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
        x = self.proj(x)
        return x

    def forward_with_cache(
        self,
        x: Float[torch.Tensor, "batch seq_len hidden"],
        freqs_cis: Complex[torch.Tensor, "seq_len head_dim"],
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[Float[torch.Tensor, "batch seq_len hidden"], tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV-cache support for efficient autoregressive generation.

        When past_kv is provided, only the new tokens in x are processed and their
        K/V are concatenated with the cached values. This reduces per-step complexity
        from O(N²) to O(N) for autoregressive generation.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]. When using cache,
               seq_len should be 1 (the new token only).
            freqs_cis: RoPE frequencies for the NEW positions only.
                When using cache, should be [1, head_dim//2] for the new position.
            past_kv: Cached (key, value) tensors from previous steps.
                Each has shape [batch_size, num_heads, past_len, head_dim].
            attention_mask: Optional attention mask.

        Returns:
            output: Attention output [batch_size, seq_len, hidden_size]
            new_kv: Updated (key, value) cache including new tokens.
        """
        B, L, C = x.shape

        # Compute Q, K, V for new tokens
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings to new tokens
        # freqs_cis should be positioned for the new token's position
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Apply QK LayerNorm for stability
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Concatenate with cached K, V
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Store new cache
        new_kv = (k, v)

        # Attention - for cached generation with single new token, use is_causal=False
        # since we're only computing attention for the last token against all previous
        # The causal masking is implicit: we only have access to past tokens in cache
        is_causal = self.causal and past_kv is None and attention_mask is None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=0.0,
            scale=None,
        )

        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out, new_kv


class AttentionWithValueResidual(nn.Module):
    """
    Self-attention with Value Residual Learning for very deep networks.

    Adds a learnable residual path to the value computation, which helps
    gradient flow in very deep transformers (60+ layers). The residual
    is a weighted combination of the input and the computed value.

    Based on:
    - "Stabilizing Transformer Training by Preventing Attention Entropy Collapse"
    - DeepNet, DeepSeek-V2 (value residual for deep scaling)

    Formula:
        v_effective = v + lambda * x_projected
        where lambda is a learnable per-head scale (initialized to 0)

    This allows the model to learn when to use pure attention values
    vs. when to blend in the input representation.

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        causal: Whether to use causal masking
        qkv_bias: Whether to use bias in QKV projections
        qk_norm: Whether to apply QK LayerNorm
        value_residual_mix: Initial mixing coefficient (default 0.0 for pure attention)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        causal: bool = False,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        value_residual_mix: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.causal = causal

        # Combined QKV projection
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # QK LayerNorm for stability
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None

        # Value residual: project input to value space and learn mixing weight
        # Separate projection for value residual (can share weights with v projection)
        self.value_residual_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Per-head learnable mixing coefficient (lambda in the paper)
        # Initialized to value_residual_mix (usually 0 for pure attention at start)
        self.value_residual_lambda = nn.Parameter(torch.full((num_heads, 1, 1), value_residual_mix))

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len hidden"],
        freqs_cis: Complex[torch.Tensor, "seq_len head_dim"],
        attention_mask: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, "batch seq_len hidden"]:
        B, L, C = x.shape

        # Fused QKV transform
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to (B, H, L, D) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute value residual: project input to value space
        v_residual = self.value_residual_proj(x)
        v_residual = v_residual.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Mix value with residual using learnable lambda
        # v_effective = v + lambda * v_residual
        v = v + self.value_residual_lambda * v_residual

        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Apply QK LayerNorm for stability
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

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
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block supporting both causal and bidirectional attention.

    Supports:
    - QK LayerNorm for attention stability (arXiv:2312.16903)
    - Peri-LN (peripheral LayerNorm) for training stability (arXiv:2502.02732)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        causal: bool = False,
        qk_norm: bool = False,
        peri_ln: bool = False,
    ):
        super().__init__()
        # Input norms (standard Pre-LN)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Output norms for Peri-LN (additional stability)
        self.peri_ln = peri_ln
        self.norm1_out = nn.LayerNorm(hidden_size) if peri_ln else None
        self.norm2_out = nn.LayerNorm(hidden_size) if peri_ln else None

        self.attn = Attention(hidden_size, num_heads, causal=causal, qk_norm=qk_norm)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention with optional Peri-LN
        attn_out = self.attn(self.norm1(x), freqs_cis, attention_mask)
        if self.norm1_out is not None:
            attn_out = self.norm1_out(attn_out)
        x = x + attn_out

        # MLP with optional Peri-LN
        mlp_out = self.mlp(self.norm2(x))
        if self.norm2_out is not None:
            mlp_out = self.norm2_out(mlp_out)
        x = x + mlp_out

        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV-cache support.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            freqs_cis: RoPE frequencies for the positions being processed
            past_kv: Cached (key, value) from previous steps
            attention_mask: Optional attention mask

        Returns:
            output: Block output [batch_size, seq_len, hidden_size]
            new_kv: Updated KV cache
        """
        # Attention with cache
        attn_out, new_kv = self.attn.forward_with_cache(
            self.norm1(x), freqs_cis, past_kv, attention_mask
        )
        if self.norm1_out is not None:
            attn_out = self.norm1_out(attn_out)
        x = x + attn_out

        # MLP (no cache needed)
        mlp_out = self.mlp(self.norm2(x))
        if self.norm2_out is not None:
            mlp_out = self.norm2_out(mlp_out)
        x = x + mlp_out

        return x, new_kv


# ============================================================================
# Base Transformer
# ============================================================================


class _BaseTransformer(nn.Module):
    """Shared infrastructure for transformer backbones."""

    def __init__(
        self,
        seq_length: int,
        hidden_size: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head dimension {self.head_dim} must be even (hidden_size={hidden_size}, num_heads={num_heads})"
            )
        self.rope_theta = rope_theta
        self.allow_dynamic_seq_length = allow_dynamic_seq_length
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, seq_length, theta=rope_theta),
        )
        self.seq_len_cached = seq_length

        # Enable optimized attention kernels if available
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    @property
    def device(self) -> torch.device:
        """Return the device of model parameters."""
        return next(self.parameters()).device

    def update_rotary_cache(self, seq_length: int) -> None:
        """Pre-compute and cache rotary embeddings for given sequence length."""
        if seq_length == self.seq_len_cached:
            return

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, seq_length, theta=self.rope_theta).to(
                device=self.freqs_cis.device, dtype=self.freqs_cis.dtype
            ),
        )
        self.seq_len_cached = seq_length

    def set_chunk_size(self, chunk_size: int | None):
        """Set or disable sequence chunking."""
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def ckpt_wrapper(self, module):
        """Wrapper for gradient checkpointing."""

        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward


class DiscreteTransformer(_BaseTransformer):
    """Transformer backbone purpose-built for discrete token sequences."""

    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        gradient_checkpointing: bool = True,
        special_tokens: SpecialTokenIds | None = None,
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
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def detect_anomalies(
        self, x: torch.Tensor, likelihood_threshold: float = 0.005, **forward_kwargs
    ) -> torch.Tensor:
        """
        Detect anomalies by finding low likelihood tokens.

        Args:
            x: Input sequence [batch_size, seq_len]
            likelihood_threshold: Threshold for marking tokens as anomalous
            **forward_kwargs: Additional arguments for forward pass

        Returns:
            anomaly_mask: Binary mask of detected anomalies [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            logits = self(**forward_kwargs, x=x) if forward_kwargs else self(x)
            probs = F.softmax(logits, dim=-1)
            token_probs = torch.gather(probs, -1, x.unsqueeze(-1).long()).squeeze(-1)
            return token_probs < likelihood_threshold

    def inpaint_anomalies(
        self,
        x: torch.Tensor,
        likelihood_threshold: float = 0.005,
        temperature: float = 1.0,
        top_k: int | None = None,
        **forward_kwargs,
    ) -> torch.Tensor:
        """
        Detect and inpaint anomalies in the latent sequence.

        Args:
            x: Input sequence [batch_size, seq_len]
            likelihood_threshold: Threshold for marking tokens as anomalous
            temperature: Sampling temperature
            top_k: Optional top-k sampling parameter
            **forward_kwargs: Additional arguments for forward pass

        Returns:
            inpainted_sequence: Sequence with anomalies replaced [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            # Detect anomalies
            logits = self(**forward_kwargs, x=x) if forward_kwargs else self(x)
            probs = F.softmax(logits, dim=-1)
            token_probs = torch.gather(probs, -1, x.unsqueeze(-1).long()).squeeze(-1)
            anomaly_mask = token_probs < likelihood_threshold

            if not anomaly_mask.any():
                return x

            # Sample replacements
            logits_temp = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_temp, top_k, dim=-1)
                logits_temp[logits_temp < v[..., [-1]]] = -float("Inf")

            probs = F.softmax(logits_temp, dim=-1)
            samples = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(
                x.shape
            )

            return torch.where(anomaly_mask, samples, x)


class ContinuousTransformer(_BaseTransformer):
    """Transformer backbone for continuous latent representations."""

    def __init__(
        self,
        seq_length: int,
        hidden_size: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        gradient_checkpointing: bool = True,
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

    # Continuous transformer variants do not currently expose anomaly tooling.
