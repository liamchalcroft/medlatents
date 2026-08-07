"""AI-based preference scoring for synthetic feedback.

This module provides methods for generating preference labels without human annotation:
1. DiscriminatorScorer: Uses a trained discriminator (real vs fake)
2. AIPreferenceScorer: Generic scoring interface
3. StepAwarePreferenceModel: Timestep-conditioned scoring for SPO

These enable scalable preference learning without expensive human labeling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .data import PreferencePair, RankedSamples
from .reward_model import RewardModel


class DiscriminatorScorer(nn.Module):
    """Score samples using a discriminator trained to distinguish real from generated.

    This is compatible with your existing TokenCritic and can also wrap
    any binary classifier.

    Example:
        >>> scorer = DiscriminatorScorer(critic_model)
        >>> scores = scorer(generated_samples)  # Higher = more "real"
    """

    def __init__(
        self,
        discriminator: nn.Module,
        score_transform: Literal["sigmoid", "logit", "raw"] = "sigmoid",
    ) -> None:
        """Initialize discriminator scorer.

        Args:
            discriminator: Model that outputs realness scores
            score_transform: How to transform raw outputs
                - "sigmoid": Apply sigmoid (if outputs are logits)
                - "logit": Keep as logits
                - "raw": No transformation
        """
        super().__init__()
        self.discriminator = discriminator
        self.score_transform = score_transform

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Score samples.

        Args:
            x: Input samples [batch, ...]
            **kwargs: Additional args for discriminator

        Returns:
            Scores [batch] where higher = more preferred
        """
        scores = self.discriminator(x, **kwargs)

        # Handle different output shapes
        if scores.dim() > 1:
            # If per-token scores, average over sequence
            scores = scores.mean(dim=tuple(range(1, scores.dim())))

        # Apply transformation
        if self.score_transform == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_transform == "logit":
            pass  # Keep as is

        return scores.squeeze(-1)

    @torch.no_grad()
    def rank_samples(self, samples: list[Tensor]) -> list[int]:
        """Rank samples from best to worst.

        Args:
            samples: List of samples to rank

        Returns:
            Indices sorted by score (best first)
        """
        stacked = torch.stack(samples)
        scores = self(stacked)
        return scores.argsort(descending=True).tolist()

    @torch.no_grad()
    def create_pairs(
        self,
        samples: list[Tensor],
        prompt: Any = None,
        strategy: Literal["best_worst", "adjacent", "all"] = "best_worst",
    ) -> list[PreferencePair]:
        """Create preference pairs from samples using discriminator scores.

        Args:
            samples: List of samples to compare
            prompt: Optional prompt/conditioning
            strategy: Pairing strategy

        Returns:
            List of PreferencePair objects
        """
        stacked = torch.stack(samples)
        scores = self(stacked).tolist()

        # Rank best-first so the position-based pairing in RankedSamples.to_pairs
        # reflects the discriminator scores rather than input order.
        order = sorted(range(len(samples)), key=lambda i: scores[i], reverse=True)
        ranked = RankedSamples(
            samples=[samples[i].cpu() for i in order],
            prompt=prompt,
            scores=[scores[i] for i in order],
        )
        return ranked.to_pairs(strategy)


class AIPreferenceScorer:
    """Generic AI-based preference scoring.

    Wraps various scoring backends (reward model, discriminator, external API)
    into a unified interface.

    Example:
        >>> # From reward model
        >>> scorer = AIPreferenceScorer.from_reward_model(rm)
        >>>
        >>> # Custom scoring function
        >>> scorer = AIPreferenceScorer(lambda x: my_score_fn(x))
        >>>
        >>> # Generate preference pairs
        >>> pairs = scorer.generate_pairs(samples, num_pairs=100)
    """

    def __init__(
        self,
        score_fn: Callable[[Tensor], Tensor],
        higher_is_better: bool = True,
    ) -> None:
        """Initialize with a scoring function.

        Args:
            score_fn: Function that takes samples and returns scores
            higher_is_better: If True, higher scores are preferred
        """
        self.score_fn = score_fn
        self.higher_is_better = higher_is_better

    @classmethod
    def from_reward_model(cls, model: RewardModel) -> "AIPreferenceScorer":
        """Create scorer from a RewardModel."""
        return cls(lambda x: model(x).squeeze(-1), higher_is_better=True)

    @classmethod
    def from_discriminator(cls, discriminator: nn.Module) -> "AIPreferenceScorer":
        """Create scorer from a discriminator."""
        scorer = DiscriminatorScorer(discriminator)
        return cls(scorer, higher_is_better=True)

    @torch.no_grad()
    def score(self, samples: Tensor) -> Tensor:
        """Score a batch of samples.

        Args:
            samples: Batch of samples [batch, ...]

        Returns:
            Scores [batch]
        """
        scores = self.score_fn(samples)
        if not self.higher_is_better:
            scores = -scores
        return scores

    @torch.no_grad()
    def rank(self, samples: Sequence[Tensor]) -> list[int]:
        """Get ranking of samples (best first).

        Args:
            samples: List of samples

        Returns:
            Indices sorted by preference (best first)
        """
        stacked = torch.stack(list(samples))
        scores = self.score(stacked)
        return scores.argsort(descending=True).tolist()

    @torch.no_grad()
    def create_pairs(
        self,
        samples: Sequence[Tensor],
        prompt: Any = None,
        strategy: Literal["best_worst", "adjacent", "all"] = "best_worst",
    ) -> list[PreferencePair]:
        """Create preference pairs from samples.

        Args:
            samples: Samples to compare
            prompt: Optional conditioning
            strategy: How to create pairs from ranking

        Returns:
            List of PreferencePair objects
        """
        sample_list = list(samples)
        stacked = torch.stack(sample_list)
        scores = self.score(stacked).tolist()

        # Rank best-first so the position-based pairing in RankedSamples.to_pairs
        # reflects the scores rather than input order.
        order = sorted(range(len(sample_list)), key=lambda i: scores[i], reverse=True)
        ranked = RankedSamples(
            samples=[
                sample_list[i].cpu()
                if isinstance(sample_list[i], Tensor)
                else torch.as_tensor(sample_list[i])
                for i in order
            ],
            prompt=prompt,
            scores=[scores[i] for i in order],
        )
        return ranked.to_pairs(strategy)

    @torch.no_grad()
    def generate_pairs_from_generator(
        self,
        generator: nn.Module,
        prompts: Sequence[Any],
        num_samples_per_prompt: int = 4,
        strategy: Literal["best_worst", "adjacent", "all"] = "best_worst",
        generation_kwargs: dict[str, Any] | None = None,
        device: torch.device | str = "cuda",
    ) -> list[PreferencePair]:
        """Generate preference pairs by sampling from a generator.

        Args:
            generator: Model with generate() method
            prompts: Conditioning prompts
            num_samples_per_prompt: Samples per prompt
            strategy: Pairing strategy
            generation_kwargs: Extra args for generation
            device: Device for generation

        Returns:
            List of PreferencePair objects
        """
        from tqdm import tqdm

        generation_kwargs = generation_kwargs or {}
        all_pairs = []

        generator.eval()
        for prompt in tqdm(prompts, desc="Generating pairs"):
            samples = []
            for _ in range(num_samples_per_prompt):
                with torch.no_grad():
                    if hasattr(generator, "generate"):
                        sample = generator.generate(prompt, **generation_kwargs)
                    else:
                        sample = generator(prompt).argmax(dim=-1)
                    samples.append(sample)

            pairs = self.create_pairs(samples, prompt=prompt, strategy=strategy)
            all_pairs.extend(pairs)

        return all_pairs


class StepAwarePreferenceModel(nn.Module):
    """Timestep-aware preference model for Step-by-step Preference Optimization (SPO).

    This model scores intermediate denoising states, allowing SPO to learn
    fine-grained preferences at each noise level.

    The key insight is that preferences differ at different timesteps:
    - Early steps (high noise): Layout/composition preferences
    - Late steps (low noise): Fine detail preferences

    Example:
        >>> step_model = StepAwarePreferenceModel(
        ...     base_model=reward_model,  # Must be timestep_aware=True
        ... )
        >>> scores = step_model(noisy_samples, t=timesteps)
    """

    def __init__(
        self,
        base_model: RewardModel | None = None,
        vocab_size: int | None = None,
        in_channels: int | None = None,
        seq_length: int = 256,
        hidden_size: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ) -> None:
        """Initialize step-aware preference model.

        Either provide a base_model (must have timestep_aware=True) or
        specify architecture parameters to create a new model.

        Args:
            base_model: Pre-existing timestep-aware RewardModel
            vocab_size: For creating new model (discrete)
            in_channels: For creating new model (continuous)
            seq_length: Sequence length
            hidden_size: Hidden dimension
            depth: Number of layers
            num_heads: Attention heads
        """
        super().__init__()

        if base_model is not None:
            if not base_model.timestep_aware:
                raise ValueError("base_model must have timestep_aware=True for SPO")
            self.model = base_model
        else:
            if vocab_size is None and in_channels is None:
                raise ValueError("Specify base_model or (vocab_size/in_channels)")

            self.model = RewardModel(
                vocab_size=vocab_size,
                in_channels=in_channels,
                seq_length=seq_length,
                hidden_size=hidden_size,
                depth=depth,
                num_heads=num_heads,
                timestep_aware=True,
            )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Score samples at given timesteps.

        Args:
            x: Noisy samples [batch, ...]
            t: Timesteps [batch] in [0, 1]

        Returns:
            Scores [batch]
        """
        return self.model(x, t).squeeze(-1)

    @torch.no_grad()
    def select_winner_loser(
        self,
        candidates: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select best and worst candidates for SPO training.

        Args:
            candidates: Candidate samples [num_candidates, batch, ...]
            t: Timestep [batch] or scalar

        Returns:
            (chosen, rejected, chosen_scores, rejected_scores)
        """
        num_candidates, batch_size = candidates.shape[:2]

        # Flatten for scoring
        flat_candidates = candidates.flatten(0, 1)  # [num_candidates * batch, ...]

        # Expand timesteps
        if t.dim() == 0:
            t = t.expand(batch_size)
        flat_t = t.repeat(num_candidates)

        # Score all candidates
        scores = self(flat_candidates, flat_t)
        scores = scores.view(num_candidates, batch_size)

        # Select best and worst per batch item
        best_idx = scores.argmax(dim=0)
        worst_idx = scores.argmin(dim=0)

        batch_indices = torch.arange(batch_size, device=candidates.device)

        chosen = candidates[best_idx, batch_indices]
        rejected = candidates[worst_idx, batch_indices]
        chosen_scores = scores[best_idx, batch_indices]
        rejected_scores = scores[worst_idx, batch_indices]

        return chosen, rejected, chosen_scores, rejected_scores

    @torch.no_grad()
    def sample_random_candidate(
        self,
        candidates: Tensor,
    ) -> Tensor:
        """Randomly select one candidate per batch item (for next SPO step).

        Args:
            candidates: [num_candidates, batch, ...]

        Returns:
            Selected samples [batch, ...]
        """
        num_candidates, batch_size = candidates.shape[:2]
        rand_idx = torch.randint(0, num_candidates, (batch_size,), device=candidates.device)
        batch_indices = torch.arange(batch_size, device=candidates.device)
        return candidates[rand_idx, batch_indices]


__all__ = [
    "DiscriminatorScorer",
    "AIPreferenceScorer",
    "StepAwarePreferenceModel",
]
