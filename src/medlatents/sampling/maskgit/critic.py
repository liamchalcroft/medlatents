"""Token Critic for MaskGIT quality improvement.

Implements a discriminator/critic model that scores generated tokens,
enabling rejection sampling and iterative refinement during generation.

References:
- "MUSE: Text-To-Image Generation via Masked Generative Transformers"
- "Reject Sampling for Discrete Diffusion Models" (arXiv:2305.17625)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCritic(nn.Module):
    """Binary classifier that predicts whether tokens are real or generated.

    Used during MaskGIT generation to score candidate tokens and enable:
    1. Rejection sampling: Reject low-confidence samples
    2. Iterative refinement: Re-mask and regenerate poorly-scored tokens
    3. Quality-guided selection: Pick best samples from multiple candidates

    The critic shares architecture with the generator but has a binary output head.

    Usage:
        critic = TokenCritic(hidden_size=512, num_heads=8, depth=6, vocab_size=1024)

        # During generation, score candidate tokens
        scores = critic(candidate_tokens)  # [batch, seq_len, 1]

        # Use scores to filter or refine
        low_quality_mask = scores.squeeze(-1) < threshold
        tokens[low_quality_mask] = mask_token  # Re-mask for another pass
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        depth: int = 6,
        vocab_size: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Initialize token critic.

        Args:
            hidden_size: Transformer hidden dimension
            num_heads: Number of attention heads
            depth: Number of transformer layers
            vocab_size: Vocabulary size for token embedding
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_size)

        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, hidden_size) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=int(hidden_size * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

        # Output head: per-token real/fake prediction
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small std for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Score tokens as real or generated.

        Args:
            tokens: Input tokens [batch, seq_len]
            return_hidden: Also return hidden states

        Returns:
            scores: Per-token scores [batch, seq_len, 1] (higher = more likely real)
            hidden: (optional) Hidden states [batch, seq_len, hidden_size]
        """
        seq_len = tokens.size(1)

        # Embed tokens
        h = self.embed(tokens)
        h = h + self.pos_embed[:, :seq_len]

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        # Normalize and predict
        h = self.norm(h)
        scores = self.head(h)

        if return_hidden:
            return scores, h
        return scores

    def compute_loss(
        self,
        real_tokens: torch.Tensor,
        fake_tokens: torch.Tensor,
        real_mask: torch.Tensor | None = None,
        fake_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute critic training loss.

        Args:
            real_tokens: Ground truth tokens [batch, seq_len]
            fake_tokens: Generated tokens [batch, seq_len]
            real_mask: Optional mask for real tokens (True = include in loss)
            fake_mask: Optional mask for fake tokens

        Returns:
            Dict with 'loss', 'real_acc', 'fake_acc'
        """
        # Score both (get just scores, not hidden states)
        real_out = self.forward(real_tokens, return_hidden=False)
        fake_out = self.forward(fake_tokens, return_hidden=False)

        # Ensure we have tensors (not tuples)
        real_scores = real_out if isinstance(real_out, torch.Tensor) else real_out[0]
        fake_scores = fake_out if isinstance(fake_out, torch.Tensor) else fake_out[0]

        # Binary cross-entropy (real=1, fake=0)
        real_labels = torch.ones_like(real_scores)
        fake_labels = torch.zeros_like(fake_scores)

        real_loss = F.binary_cross_entropy_with_logits(real_scores, real_labels, reduction="none")
        fake_loss = F.binary_cross_entropy_with_logits(fake_scores, fake_labels, reduction="none")

        # Apply masks if provided
        if real_mask is not None:
            real_loss = real_loss * real_mask.unsqueeze(-1)
            real_loss = real_loss.sum() / real_mask.sum().clamp(min=1)
        else:
            real_loss = real_loss.mean()

        if fake_mask is not None:
            fake_loss = fake_loss * fake_mask.unsqueeze(-1)
            fake_loss = fake_loss.sum() / fake_mask.sum().clamp(min=1)
        else:
            fake_loss = fake_loss.mean()

        total_loss = (real_loss + fake_loss) / 2

        # Compute accuracy
        with torch.no_grad():
            real_acc = (real_scores > 0).float().mean()
            fake_acc = (fake_scores <= 0).float().mean()

        return {
            "loss": total_loss,
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "real_acc": real_acc,
            "fake_acc": fake_acc,
        }


class CriticGuidedGenerator:
    """Generator wrapper that uses a critic for quality-guided sampling.

    Implements several critic-guided generation strategies:
    1. Rejection sampling: Sample multiple, keep best by critic score
    2. Iterative refinement: Re-mask low-scoring tokens
    3. Weighted sampling: Bias sampling toward critic-preferred tokens
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: TokenCritic,
        rejection_threshold: float = 0.0,
        num_candidates: int = 4,
        refinement_steps: int = 0,
    ):
        """Initialize critic-guided generator.

        Args:
            generator: MaskGIT or similar generator model
            critic: Trained token critic
            rejection_threshold: Score threshold for rejection (logit scale)
            num_candidates: Number of candidates for best-of-n sampling
            refinement_steps: Number of iterative refinement passes
        """
        if not isinstance(rejection_threshold, (int, float)):
            raise TypeError(f"rejection_threshold must be a float, got {type(rejection_threshold)}")
        if not isinstance(num_candidates, int) or num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        if not isinstance(refinement_steps, int) or refinement_steps < 0:
            raise ValueError(f"refinement_steps must be >= 0, got {refinement_steps}")

        self.generator = generator
        self.critic = critic
        self.rejection_threshold = rejection_threshold
        self.num_candidates = num_candidates
        self.refinement_steps = refinement_steps

    @torch.no_grad()
    def generate_with_rejection(
        self,
        x: torch.Tensor,
        num_steps: int = 8,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate with rejection sampling using critic scores.

        Args:
            x: Initial masked tokens [batch, seq_len]
            num_steps: Generation steps
            temperature: Sampling temperature
            **kwargs: Additional args for generator

        Returns:
            tokens: Generated tokens [batch, seq_len]
            scores: Per-token critic scores [batch, seq_len]
        """
        # Generate multiple candidates
        candidates = []
        all_scores = []

        for _ in range(self.num_candidates):
            generated = self.generator.generate(
                x.clone(), num_steps=num_steps, temperature=temperature, **kwargs
            )
            scores = self.critic(generated).squeeze(-1)

            candidates.append(generated)
            all_scores.append(scores.mean(dim=-1))  # Average score per sequence

        # Stack and select best
        candidates = torch.stack(candidates, dim=0)  # [n, batch, seq_len]
        all_scores = torch.stack(all_scores, dim=0)  # [n, batch]

        # Select best per batch item
        best_indices = all_scores.argmax(dim=0)  # [batch]
        batch_size = x.size(0)
        best_tokens = candidates[best_indices, torch.arange(batch_size)]
        best_scores = self.critic(best_tokens).squeeze(-1)

        return best_tokens, best_scores

    @torch.no_grad()
    def generate_with_refinement(
        self,
        x: torch.Tensor,
        num_steps: int = 8,
        temperature: float = 1.0,
        mask_token: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate with iterative critic-guided refinement.

        Low-scoring tokens are re-masked and regenerated.

        Args:
            x: Initial masked tokens [batch, seq_len]
            num_steps: Generation steps per iteration
            temperature: Sampling temperature
            mask_token: Token to use for masking (defaults to generator's mask_token)
            **kwargs: Additional args for generator

        Returns:
            tokens: Refined tokens [batch, seq_len]
            scores: Final critic scores [batch, seq_len]
        """
        mask_tok = mask_token or getattr(self.generator, "mask_token", None)
        if mask_tok is None:
            raise ValueError("mask_token required for refinement")

        # Initial generation
        tokens = self.generator.generate(
            x.clone(), num_steps=num_steps, temperature=temperature, **kwargs
        )

        # Iterative refinement
        for _ in range(self.refinement_steps):
            scores = self.critic(tokens).squeeze(-1)

            # Re-mask low-scoring tokens
            low_score_mask = scores < self.rejection_threshold
            if not low_score_mask.any():
                break

            tokens = tokens.clone()
            tokens[low_score_mask] = mask_tok

            # Regenerate (with fewer steps since most is already done)
            refinement_steps = max(2, num_steps // 2)
            tokens = self.generator.generate(
                tokens, num_steps=refinement_steps, temperature=temperature * 0.9, **kwargs
            )

        final_scores = self.critic(tokens).squeeze(-1)
        return tokens, final_scores

    @torch.no_grad()
    def score_samples(
        self,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score generated samples with the critic.

        Args:
            tokens: Generated tokens [batch, seq_len]

        Returns:
            Dict with various score statistics
        """
        scores = self.critic(tokens).squeeze(-1)
        probs = torch.sigmoid(scores)

        return {
            "raw_scores": scores,
            "probs": probs,
            "mean_score": scores.mean(dim=-1),
            "mean_prob": probs.mean(dim=-1),
            "min_score": scores.min(dim=-1).values,
            "low_quality_ratio": (scores < self.rejection_threshold).float().mean(dim=-1),
        }


def train_critic_step(
    critic: TokenCritic,
    generator: nn.Module,
    real_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mask_token: int,
    mask_ratio: float = 0.5,
) -> dict[str, float]:
    """Single training step for the token critic.

    Args:
        critic: Token critic model
        generator: Generator model (frozen during critic training)
        real_batch: Real token sequences [batch, seq_len]
        optimizer: Critic optimizer
        mask_token: Mask token ID
        mask_ratio: Ratio of tokens to mask for generation

    Returns:
        Dict with loss and accuracy metrics
    """
    critic.train()

    # Generate fake samples
    with torch.no_grad():
        # Create masked input
        mask = torch.rand_like(real_batch.float()) < mask_ratio
        masked = real_batch.clone()
        masked[mask] = mask_token

        # Generate with the generator
        if hasattr(generator, "generate"):
            fake_batch = generator.generate(masked, num_steps=8, temperature=1.0)
        else:
            # Fallback: use forward pass + sampling
            logits = generator(masked)
            fake_batch = logits.argmax(dim=-1)

    # Compute critic loss
    optimizer.zero_grad()
    loss_dict = critic.compute_loss(
        real_tokens=real_batch,
        fake_tokens=fake_batch,
        fake_mask=mask,  # Only score positions that were generated
    )

    loss_dict["loss"].backward()
    optimizer.step()

    return {k: v.item() for k, v in loss_dict.items()}


__all__ = [
    "TokenCritic",
    "CriticGuidedGenerator",
    "train_critic_step",
]
