"""MaskGIT: Bidirectional transformer for discrete latent sequence modeling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as beartype_decorator
from jaxtyping import Bool, Float, Int, jaxtyped
from torch.utils.checkpoint import checkpoint

from ..configs import create_model_variants
from ..networks.diffusion_transformer import DiscreteSequenceEmbedder
from ..networks.transformer import DiscreteTransformer, TransformerBlock, init_weights
from ..utils import SpecialTokenIds, resolve_special_tokens


class MaskGIT(DiscreteTransformer):
    """
    Bidirectional transformer with RoPE and masked training (MaskGIT).
    """

    def __init__(
        self,
        seq_length: int = 1024,
        vocab_size: int = 1024,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.15,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        gradient_checkpointing: bool = True,
        special_tokens: SpecialTokenIds | None = None,
        add_special_tokens: bool = True,
    ):
        if hidden_size % (2 * num_heads) != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by 2 * num_heads {num_heads}"
            )

        # For large vocabularies, prefer reserving in-vocab special ids to avoid silently
        # inflating output dimensionality (important for memory-sensitive large-batch runs).
        if special_tokens is None and add_special_tokens and vocab_size >= 512:
            add_special_tokens = False

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

        self.base_vocab_size = vocab_size
        self.vocab_size = extended_vocab
        self.special_tokens = resolved_special
        self.mask_token = resolved_special.mask
        self.mask_ratio = mask_ratio

        # Token embedding (+1 for mask token)
        self.x_embedder = DiscreteSequenceEmbedder(
            seq_length,
            extended_vocab,
            hidden_size,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
        )

        # Transformer blocks with bidirectional attention
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, mlp_ratio, causal=False)
                for _ in range(depth)
            ]
        )

        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, extended_vocab)
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(init_weights)

    def random_masking(
        self, x: Int[torch.Tensor, "batch seq"], mask_ratio: float | None = None
    ) -> tuple[Int[torch.Tensor, "batch seq"], Bool[torch.Tensor, "batch seq"]]:
        """Apply random masking to input tokens.

        Args:
            x: Input token indices [batch_size, seq_len]
            mask_ratio: Fraction of tokens to mask (uses self.mask_ratio if None)

        Returns:
            x_masked: Input with masked positions replaced with mask token
            mask: Boolean mask indicating which positions were masked
        """
        B, L = x.shape
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        mask = torch.rand(B, L, device=x.device) < mask_ratio
        x_masked = x.clone()
        x_masked[mask] = self.mask_token

        return x_masked, mask

    @jaxtyped(typechecker=beartype_decorator)
    def forward(
        self,
        x: Int[torch.Tensor, "batch seq"],
        mask: Bool[torch.Tensor, "batch seq"] | None = None,
        mask_ratio: float | None = None,
    ) -> Float[torch.Tensor, "batch seq vocab"]:
        """Forward pass through the bidirectional transformer.

        Args:
            x: Input token indices [batch_size, seq_len]
            mask: Optional pre-computed mask of positions to predict
            mask_ratio: Optional mask ratio for training (auto-masks if None during training)

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
            if seq_len > self.seq_len_cached:
                self.update_rotary_cache(seq_len)

        # Apply masking if needed
        if mask is None and self.training:
            x, mask = self.random_masking(x, mask_ratio)
        elif mask is not None:
            x = x.clone()
            x[mask] = self.mask_token

        x = self.x_embedder(x)
        freqs_cis = self.freqs_cis[:seq_len]

        # Create attention mask for masked tokens (optional)
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, None, None, :]

        # Apply transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    freqs_cis,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, freqs_cis, attention_mask)

        x = self.final_layer(x)
        return x

    def generate(
        self,
        x: Int[torch.Tensor, "batch seq"],
        num_steps: int = 8,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token: int | None = None,
        critic: torch.nn.Module | None = None,
        use_critic: bool = False,
        critic_threshold: float = 0.5,
        num_candidates: int = 4,
        refinement_steps: int = 0,
        generator: torch.Generator | None = None,
    ) -> Int[torch.Tensor, "batch seq"]:
        """Iterative parallel generation following MaskGIT schedule.

        Args:
            x: Initial sequence with mask tokens [batch_size, seq_len]
            num_steps: Number of demasking iterations
            temperature: Sampling temperature. Must be > 0.
            top_k: If set, only sample from top-k logits
            eos_token: End-of-sequence token
            critic: Optional token critic model for quality-guided generation
            use_critic: If True, use critic for rejection/refinement (requires critic)
            critic_threshold: Score threshold for rejection sampling (0-1)
            num_candidates: Number of candidates for best-of-N selection
            refinement_steps: Number of iterative refinement passes
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            generated: Fully generated sequence [batch_size, seq_len]

        Raises:
            ValueError: If temperature <= 0, num_steps < 1, or critic config invalid.

        Note:
            When `use_critic=True` and `critic` is provided, generation uses
            the CriticGuidedGenerator for quality-improved sampling. This requires
            a trained token critic model.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if use_critic and critic is None:
            raise ValueError("use_critic=True requires a trained critic model")

        # Use critic-guided generation if requested and critic provided
        if use_critic and critic is not None:
            from ..sampling.maskgit.critic import CriticGuidedGenerator

            critic_gen = CriticGuidedGenerator(
                generator=self,
                critic=critic,
                rejection_threshold=critic_threshold,
                num_candidates=num_candidates,
                refinement_steps=refinement_steps,
            )

            was_training = self.training
            self.eval()
            try:
                with torch.inference_mode():
                    # Use rejection sampling by default
                    tokens, scores = critic_gen.generate_with_rejection(
                        x.to(self.device).clone(),
                        num_steps=num_steps,
                        temperature=temperature,
                        eos_token=eos_token,
                    )
                return tokens
            finally:
                if was_training:
                    self.train()

        # Standard confidence-based generation
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                x_gen = x.to(self.device).clone()
                B, L = x_gen.shape
                mask = x_gen == self.mask_token

                # Pre-compute forbidden token mask (additive bias)
                forbidden_mask = None
                if self.special_tokens is not None:
                    forbidden_tokens = [
                        self.special_tokens.pad,
                        self.special_tokens.mask,
                        self.special_tokens.bos,
                    ]
                    if forbidden_tokens:
                        forbidden_mask = torch.zeros(
                            self.vocab_size, device=self.device, dtype=torch.float
                        )
                        forbidden_mask[forbidden_tokens] = -float("inf")

                # Import scheduler for mask ratio computation and selection
                from ..sampling.maskgit import MaskGITScheduler

                scheduler = MaskGITScheduler(num_steps=num_steps, mask_schedule="cosine")

                # Compute fixed K_MAX for compile-friendly topk (no .item())
                K_MAX = max(1, int(0.25 * L + 0.5))  # 25% cap

                # Note: Loop runs to completion for compile-friendliness (no data-dependent breaks).
                # Once all tokens are unmasked, subsequent iterations are effectively no-ops.

                for step in range(num_steps):
                    is_final_step = step == num_steps - 1
                    mask_ratio = scheduler.get_mask_ratio(step)

                    logits = self(x_gen, mask=mask)
                    if forbidden_mask is not None:
                        logits = logits + forbidden_mask
                    logits = logits / temperature

                    if top_k is not None:
                        topk_values, _ = torch.topk(logits, top_k, dim=-1)
                        min_topk = topk_values[..., [-1]]
                        logits = torch.where(
                            logits < min_topk,
                            torch.full_like(logits, -float("inf")),
                            logits,
                        )

                    # Gumbel-max sampling for masked positions
                    # Use eps=1e-6 for fp16 stability (1e-10 underflows in half precision)
                    eps = 1e-6
                    if generator is not None:
                        # Use generator for reproducibility
                        uniform = torch.empty_like(logits).uniform_(generator=generator)
                    else:
                        uniform = torch.rand_like(logits)
                    # Clamp uniform to [eps, 1-eps] to avoid log(0)
                    uniform = uniform.clamp(min=eps, max=1.0 - eps)
                    gumbel_noise = -torch.log(-torch.log(uniform) + eps)
                    gumbel_logits = logits + gumbel_noise
                    sampled = torch.argmax(gumbel_logits, dim=-1)

                    if is_final_step:
                        # Final step: unmask all remaining masked positions (no topk)
                        x_gen = torch.where(mask, sampled, x_gen)
                        mask = torch.zeros_like(mask)
                    else:
                        # Normal step: use capped topk selection
                        probs = F.softmax(logits, dim=-1)
                        confidences = torch.gather(
                            probs, dim=-1, index=sampled.unsqueeze(-1)
                        ).squeeze(-1)
                        confidences = confidences.masked_fill(~mask, -float("inf"))

                        # Vectorized confidence-based selection across batch
                        num_masked = mask.sum(dim=1)
                        tokens_to_update = torch.ceil((1 - mask_ratio) * num_masked.float()).long()
                        tokens_to_update = torch.clamp(tokens_to_update, min=1)

                        # Compute effective k (clamped to K_MAX, no .item())
                        effective_k = torch.minimum(tokens_to_update, num_masked)
                        effective_k = torch.clamp(effective_k, min=0, max=K_MAX)

                        # Fixed-shape topk (compile-friendly)
                        _, topk_indices = torch.topk(
                            confidences, k=K_MAX, dim=-1, largest=True
                        )  # [B, K_MAX]

                        # Create mask for valid k per batch
                        k_range = torch.arange(K_MAX, device=self.device)
                        valid_k_mask = k_range.unsqueeze(0) < effective_k.unsqueeze(1)  # [B, K_MAX]

                        # Create update mask using scatter - fully vectorized
                        update_mask = torch.zeros(B, L, dtype=torch.bool, device=self.device)
                        update_mask.scatter_(1, topk_indices, valid_k_mask)

                        # Apply updates vectorized
                        x_gen = torch.where(update_mask, sampled, x_gen)
                        mask = mask & ~update_mask

                    mask = x_gen == self.mask_token

                return x_gen
        finally:
            if was_training:
                self.train()

    def inpaint_anomalies(
        self,
        x: Int[torch.Tensor, "batch seq"],
        likelihood_threshold: float = 0.005,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Int[torch.Tensor, "batch seq"]:
        """Inpaint using bidirectional context by detecting and replacing anomalous tokens.

        Args:
            x: Input sequence [batch_size, seq_len]
            likelihood_threshold: Tokens with likelihood below this are considered anomalous
            temperature: Sampling temperature for replacements
            top_k: If set, only sample from top-k logits

        Returns:
            inpainted: Sequence with anomalies replaced [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
            token_probs = torch.gather(probs, -1, x.unsqueeze(-1).long()).squeeze(-1)
            anomaly_mask = token_probs < likelihood_threshold

            if not anomaly_mask.any():
                return x

            logits = self(x, mask=anomaly_mask)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[..., [-1]]] = -float("Inf")

            probs = F.softmax(logits / temperature, dim=-1)
            samples = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(
                x.shape
            )

            return torch.where(anomaly_mask, samples, x)


# Model size configurations
MaskGIT_models = create_model_variants(MaskGIT, "MaskGIT")
