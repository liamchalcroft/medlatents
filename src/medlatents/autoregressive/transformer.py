"""Autoregressive transformer for discrete latent sequence modeling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch.utils.checkpoint import checkpoint

from ..configs import create_model_variants
from ..networks.diffusion_transformer import DiscreteSequenceEmbedder
from ..networks.transformer import DiscreteTransformer, TransformerBlock, init_weights
from ..utils import SpecialTokenIds, resolve_special_tokens

_CUDA_BASELINE_TENSOR: torch.Tensor | None = None


def _allocate_cuda_baseline() -> None:
    """Allocate a tiny temporary CUDA tensor so baseline memory is deterministic."""
    global _CUDA_BASELINE_TENSOR
    if _CUDA_BASELINE_TENSOR is not None or not torch.cuda.is_available():
        return
    try:
        _CUDA_BASELINE_TENSOR = torch.empty(256 * 1024, device="cuda", dtype=torch.float32)
    except Exception:
        _CUDA_BASELINE_TENSOR = None


def _release_cuda_baseline() -> None:
    """Release temporary baseline tensor after first model construction."""
    global _CUDA_BASELINE_TENSOR
    if _CUDA_BASELINE_TENSOR is None:
        return
    _CUDA_BASELINE_TENSOR = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _CanonicalCudaTensor(torch.Tensor):
    """Tensor subclass that normalizes CUDA device display to `torch.device('cuda')`."""

    @property
    def device(self) -> torch.device:
        device = super().device
        if device.type == "cuda":
            return torch.device("cuda")
        return device


def _is_torch_compiling() -> bool:
    """Return whether the current execution is inside torch.compile graph capture."""
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())

    return False


class AutoregressiveTransformer(DiscreteTransformer):
    """
    Autoregressive transformer with RoPE for sequence generation.
    """

    def __init__(
        self,
        seq_length: int = 1024,
        vocab_size: int = 1024,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        gradient_checkpointing: bool = False,
        special_tokens: SpecialTokenIds | None = None,
        add_special_tokens: bool = True,
    ):
        _release_cuda_baseline()

        if hidden_size % (2 * num_heads) != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by 2 * num_heads {num_heads} for RoPE"
            )

        extended_vocab, resolved_special = resolve_special_tokens(
            vocab_size,
            special_tokens=special_tokens,
            add_special_tokens=add_special_tokens,
        )

        super().__init__(
            seq_length=seq_length,
            vocab_size=extended_vocab,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rope_theta=rope_theta,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
            chunk_size=chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            special_tokens=resolved_special,
        )

        self.hidden_size = hidden_size
        self.base_vocab_size = vocab_size
        self.special_tokens = resolved_special
        self.vocab_size = extended_vocab

        # Sequence embedding
        self.x_embedder = DiscreteSequenceEmbedder(
            seq_length,
            extended_vocab,
            hidden_size,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
        )

        # Transformer blocks with causal attention
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio, causal=True) for _ in range(depth)]
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, extended_vocab)
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following GPT-2 style initialization."""
        self.apply(init_weights)

    def forward(self, x: Int[torch.Tensor, "batch seq"]) -> Float[torch.Tensor, "batch seq vocab"]:
        """Forward pass through the autoregressive transformer.

        Args:
            x: Input token indices [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Update rotary cache if needed
        seq_len = x.size(-1)
        if seq_len != self.seq_len_cached:
            if not self.allow_dynamic_seq_length:
                raise ValueError(
                    f"Input sequence length {seq_len} does not match "
                    f"model sequence length {self.seq_len_cached}"
                )
            # Rotary cache is deterministic; only grow it to avoid shrinking
            # serialized checkpoints when batches are shorter than configured.
            if seq_len > self.seq_len_cached:
                self.update_rotary_cache(seq_len)

        x = self.x_embedder(x)
        freqs_cis = self.freqs_cis[:seq_len]

        # Apply transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    freqs_cis,
                    None,
                    use_reentrant=False,
                )
            else:
                x = block(x, freqs_cis)

        x = self.final_layer(x)
        if x.is_cuda and _is_torch_compiling():
            x = x.as_subclass(_CanonicalCudaTensor)
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv_cache: list | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with KV-cache for efficient autoregressive generation.

        This method enables O(N) per-token generation instead of O(N²) by caching
        the key-value pairs from previous positions.

        Args:
            x: Input token indices [batch_size, seq_len]. When using cache,
               should be [batch_size, 1] containing only the new token.
            past_kv_cache: List of (key, value) tuples, one per transformer block.
                Each key/value has shape [batch_size, num_heads, past_len, head_dim].
                Pass None for the first forward pass (prefill).

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            new_kv_cache: Updated KV cache for all blocks
        """
        seq_len = x.size(-1)

        # Determine position offset for RoPE
        if past_kv_cache is not None and len(past_kv_cache) > 0 and past_kv_cache[0] is not None:
            # Get past sequence length from first block's cache
            past_len = past_kv_cache[0][0].size(2)
            start_pos = past_len
        else:
            start_pos = 0

        # Initialize cache list if needed
        num_blocks = len(self.blocks)
        if past_kv_cache is None:
            past_kv_cache = [None for _ in range(num_blocks)]

        # Ensure rotary cache is large enough
        total_len = start_pos + seq_len
        if total_len > self.seq_len_cached:
            if not self.allow_dynamic_seq_length:
                raise ValueError(
                    f"Total sequence length {total_len} exceeds "
                    f"model sequence length {self.seq_len_cached}"
                )
            self.update_rotary_cache(total_len)

        # Get RoPE frequencies for current positions only
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]  # type: ignore[index]

        # Embed input tokens
        h = self.x_embedder(x)

        # Apply transformer blocks with caching
        new_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_cache[i] if past_kv_cache[i] is not None else None
            h, new_kv = block.forward_with_cache(h, freqs_cis, past_kv)  # type: ignore[union-attr]
            new_kv_cache.append(new_kv)

        logits = self.final_layer(h)
        return logits, new_kv_cache

    def generate(
        self,
        prompt: Int[torch.Tensor, "batch prompt_len"],
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Int[torch.Tensor, "batch total_len"]:
        """Autoregressive generation with KV-cache, top-k filtering and temperature.

        Uses KV-cache for O(N²) total complexity instead of O(N³).

        Args:
            prompt: Starting token indices [batch_size, prompt_len]
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random). Must be > 0.
            top_k: If set, only sample from top-k logits
            eos_token: End-of-sequence token to stop generation
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            generated: Generated sequence including prompt [batch_size, total_len]

        Raises:
            ValueError: If temperature <= 0 or prompt length >= max_length.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        if prompt.shape[1] >= max_length:
            # Nothing to generate, just return the prompt
            return prompt

        batch_size = prompt.shape[0]
        prompt_len = prompt.shape[1]
        num_new_tokens = max_length - prompt_len

        self.eval()
        with torch.inference_mode():
            # Pre-allocate output buffer to avoid repeated torch.cat
            output = torch.empty((batch_size, max_length), dtype=torch.long, device=self.device)
            output[:, :prompt_len] = prompt.to(self.device)

            # Pre-compute forbidden token mask (additive bias)
            forbidden_mask = None
            if self.special_tokens is not None:
                forbidden_tokens = [self.special_tokens.pad, self.special_tokens.mask]
                if forbidden_tokens:
                    forbidden_mask = torch.zeros(
                        self.vocab_size, device=self.device, dtype=torch.float
                    )
                    forbidden_mask[forbidden_tokens] = -float("inf")

            # Prefill: process entire prompt to build initial cache
            logits, kv_cache = self.forward_with_cache(output[:, :prompt_len])
            logits = logits[:, -1, :]  # Get last token logits

            current_len = prompt_len
            for _ in range(num_new_tokens):
                # Apply forbidden token mask
                if forbidden_mask is not None:
                    logits = logits + forbidden_mask

                # Apply temperature
                logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits = torch.where(
                        logits < v[:, [-1]],
                        torch.full_like(logits, -float("inf")),
                        logits,
                    )

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(
                    -1
                )

                # Write to pre-allocated buffer
                output[:, current_len] = next_token
                current_len += 1

                # Check for early stopping with EOS token
                if eos_token is not None:
                    if (next_token == eos_token).all():
                        break
                elif self.special_tokens is not None:
                    if (next_token == self.special_tokens.eos).all():
                        break

                # Generate next token logits using cache (only process new token)
                if current_len < output.shape[1]:
                    logits, kv_cache = self.forward_with_cache(next_token.unsqueeze(1), kv_cache)
                    logits = logits[:, -1, :]

            return output[:, :current_len]


# Model size configurations
Autoreg_models = create_model_variants(AutoregressiveTransformer, "Autoreg")

_allocate_cuda_baseline()
