"""Preference data structures and datasets for post-training.

This module provides clean, flexible data structures for preference learning:
- PreferencePair: A single chosen/rejected comparison
- RankedSamples: Multiple samples with ranking
- StepwisePreference: Per-step preferences for SPO
- PreferenceDataset: PyTorch Dataset for training

The design prioritizes simplicity and compatibility with various data formats.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class PreferencePair:
    """A single preference comparison between two samples.

    Attributes:
        chosen: The preferred sample (tokens or continuous latents)
        rejected: The non-preferred sample
        prompt: Optional conditioning information (class label, text, etc.)
        margin: Preference strength in [0, 1]. Higher = stronger preference.
            Default 1.0 for binary preferences, can be soft for uncertain labels.
        metadata: Optional dictionary for additional information (source, annotator, etc.)

    Example:
        >>> pair = PreferencePair(
        ...     chosen=torch.tensor([1, 2, 3]),
        ...     rejected=torch.tensor([1, 2, 4]),
        ...     prompt="Generate a brain MRI",
        ...     margin=0.8,  # Slight preference
        ... )
    """

    chosen: Tensor
    rejected: Tensor
    prompt: Any = None
    margin: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate preference pair after initialization."""
        if not isinstance(self.chosen, Tensor):
            self.chosen = torch.as_tensor(self.chosen)
        if not isinstance(self.rejected, Tensor):
            self.rejected = torch.as_tensor(self.rejected)

        if self.chosen.shape != self.rejected.shape:
            raise ValueError(
                f"Chosen and rejected must have same shape. "
                f"Got {self.chosen.shape} vs {self.rejected.shape}"
            )

        if not 0 <= self.margin <= 1:
            raise ValueError(f"Margin must be in [0, 1], got {self.margin}")

    def to(self, device: torch.device) -> "PreferencePair":
        """Move tensors to device."""
        return PreferencePair(
            chosen=self.chosen.to(device),
            rejected=self.rejected.to(device),
            prompt=self.prompt,
            margin=self.margin,
            metadata=self.metadata,
        )


@dataclass
class RankedSamples:
    """Multiple samples with a preference ranking.

    Useful for generating multiple preference pairs from ranked data.
    Ranking is from best (index 0) to worst (index -1).

    Attributes:
        samples: List of samples ordered by preference (best first)
        prompt: Optional conditioning information
        scores: Optional scalar scores for each sample
        metadata: Optional dictionary for additional information

    Example:
        >>> ranked = RankedSamples(
        ...     samples=[best_sample, second_best, worst_sample],
        ...     scores=[0.9, 0.6, 0.2],
        ... )
        >>> pairs = ranked.to_pairs()  # Creates 3 preference pairs
    """

    samples: list[Tensor]
    prompt: Any = None
    scores: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ranked samples."""
        if len(self.samples) < 2:
            raise ValueError("Need at least 2 samples for ranking")

        if self.scores is not None and len(self.scores) != len(self.samples):
            raise ValueError("Scores must have same length as samples")

        # Convert to tensors if needed
        self.samples = [
            torch.as_tensor(s) if not isinstance(s, Tensor) else s for s in self.samples
        ]

    def to_pairs(
        self,
        strategy: Literal["all", "adjacent", "best_worst"] = "all",
    ) -> list[PreferencePair]:
        """Convert ranking to preference pairs.

        Args:
            strategy:
                - "all": All pairs where i < j (O(n^2) pairs)
                - "adjacent": Only adjacent pairs (O(n) pairs)
                - "best_worst": Only best vs worst (1 pair)

        Returns:
            List of PreferencePair objects
        """
        pairs = []
        n = len(self.samples)

        if strategy == "all":
            for i in range(n):
                for j in range(i + 1, n):
                    margin = self._compute_margin(i, j) if self.scores else 1.0
                    pairs.append(
                        PreferencePair(
                            chosen=self.samples[i],
                            rejected=self.samples[j],
                            prompt=self.prompt,
                            margin=margin,
                            metadata=self.metadata,
                        )
                    )

        elif strategy == "adjacent":
            for i in range(n - 1):
                margin = self._compute_margin(i, i + 1) if self.scores else 1.0
                pairs.append(
                    PreferencePair(
                        chosen=self.samples[i],
                        rejected=self.samples[i + 1],
                        prompt=self.prompt,
                        margin=margin,
                        metadata=self.metadata,
                    )
                )

        elif strategy == "best_worst":
            margin = self._compute_margin(0, -1) if self.scores else 1.0
            pairs.append(
                PreferencePair(
                    chosen=self.samples[0],
                    rejected=self.samples[-1],
                    prompt=self.prompt,
                    margin=margin,
                    metadata=self.metadata,
                )
            )

        return pairs

    def _compute_margin(self, i: int, j: int) -> float:
        """Compute margin from score difference."""
        if self.scores is None:
            return 1.0
        diff = self.scores[i] - self.scores[j]
        # Sigmoid to normalize to [0, 1]
        return 1.0 / (1.0 + (-diff).__abs__())


@dataclass
class StepwisePreference:
    """Preference at a specific denoising step for SPO.

    Used by Step-by-step Preference Optimization to capture fine-grained
    preferences at each noise level.

    Attributes:
        step_idx: Index of the denoising step
        timestep: Continuous timestep value in [0, 1]
        chosen_state: Preferred intermediate state
        rejected_state: Non-preferred intermediate state
        chosen_score: Score of chosen state
        rejected_score: Score of rejected state
    """

    step_idx: int
    timestep: float
    chosen_state: Tensor
    rejected_state: Tensor
    chosen_score: float
    rejected_score: float

    def to(self, device: torch.device) -> "StepwisePreference":
        """Move tensors to device."""
        return StepwisePreference(
            step_idx=self.step_idx,
            timestep=self.timestep,
            chosen_state=self.chosen_state.to(device),
            rejected_state=self.rejected_state.to(device),
            chosen_score=self.chosen_score,
            rejected_score=self.rejected_score,
        )


class PreferenceDataset(Dataset):
    """PyTorch Dataset for preference learning.

    Supports multiple input formats and provides convenient factory methods
    for common use cases like synthetic generation and loading annotations.

    Example:
        >>> # From explicit pairs
        >>> dataset = PreferenceDataset(pairs=[pair1, pair2, pair3])
        >>>
        >>> # From generations with reward model
        >>> dataset = PreferenceDataset.from_generations(
        ...     generator=model,
        ...     prompts=prompt_dataset,
        ...     reward_fn=reward_model,
        ... )
        >>>
        >>> # From saved file
        >>> dataset = PreferenceDataset.load("preferences.pt")
    """

    def __init__(
        self,
        pairs: Sequence[PreferencePair] | None = None,
        transform: Callable[[PreferencePair], PreferencePair] | None = None,
    ) -> None:
        """Initialize preference dataset.

        Args:
            pairs: Sequence of PreferencePair objects
            transform: Optional transform to apply to each pair
        """
        self.pairs = list(pairs) if pairs is not None else []
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        pair = self.pairs[idx]
        if self.transform is not None:
            pair = self.transform(pair)
        return pair

    def __iter__(self) -> Iterator[PreferencePair]:
        for i in range(len(self)):
            yield self[i]

    def add_pair(self, pair: PreferencePair) -> None:
        """Add a single preference pair."""
        self.pairs.append(pair)

    def add_ranked(
        self,
        ranked: RankedSamples,
        strategy: Literal["all", "adjacent", "best_worst"] = "best_worst",
    ) -> None:
        """Add pairs from ranked samples."""
        self.pairs.extend(ranked.to_pairs(strategy))

    def filter(self, predicate: Callable[[PreferencePair], bool]) -> "PreferenceDataset":
        """Return new dataset with filtered pairs."""
        filtered = [p for p in self.pairs if predicate(p)]
        return PreferenceDataset(pairs=filtered, transform=self.transform)

    def shuffle(self, seed: int | None = None) -> "PreferenceDataset":
        """Return new dataset with shuffled pairs."""
        import random

        pairs = list(self.pairs)
        if seed is not None:
            random.seed(seed)
        random.shuffle(pairs)
        return PreferenceDataset(pairs=pairs, transform=self.transform)

    def split(
        self,
        train_ratio: float = 0.9,
        seed: int | None = None,
    ) -> tuple["PreferenceDataset", "PreferenceDataset"]:
        """Split into train and validation sets."""
        shuffled = self.shuffle(seed)
        split_idx = int(len(shuffled) * train_ratio)
        return (
            PreferenceDataset(pairs=shuffled.pairs[:split_idx], transform=self.transform),
            PreferenceDataset(pairs=shuffled.pairs[split_idx:], transform=self.transform),
        )

    @classmethod
    def from_generations(
        cls,
        generator: torch.nn.Module,
        prompts: Sequence[Any],
        reward_fn: Callable[[Tensor], Tensor],
        num_samples_per_prompt: int = 4,
        pair_strategy: Literal["all", "adjacent", "best_worst"] = "best_worst",
        generation_kwargs: dict[str, Any] | None = None,
        device: torch.device | str = "cuda",
        show_progress: bool = True,
    ) -> "PreferenceDataset":
        """Create dataset by generating samples and ranking with reward function.

        This is the primary method for creating synthetic preference data.

        Args:
            generator: Model with a `generate` method
            prompts: Sequence of conditioning prompts/labels
            reward_fn: Function that scores samples (higher = better)
            num_samples_per_prompt: Number of samples to generate per prompt
            pair_strategy: How to create pairs from rankings
            generation_kwargs: Extra kwargs for generator.generate()
            device: Device for generation
            show_progress: Show progress bar

        Returns:
            PreferenceDataset with generated pairs
        """
        from tqdm import tqdm

        generation_kwargs = generation_kwargs or {}
        device = torch.device(device)
        dataset = cls()

        generator.eval()
        iterator = tqdm(prompts, desc="Generating preferences") if show_progress else prompts

        for prompt in iterator:
            # Generate multiple samples
            samples = []
            with torch.no_grad():
                for _ in range(num_samples_per_prompt):
                    if hasattr(generator, "generate"):
                        sample = generator.generate(prompt, **generation_kwargs)
                    else:
                        # Fallback: assume forward pass returns logits
                        logits = generator(prompt)
                        sample = logits.argmax(dim=-1)

                    if isinstance(sample, Tensor):
                        samples.append(sample.cpu())
                    else:
                        samples.append(torch.as_tensor(sample))

            # Score and rank
            stacked = torch.stack(samples).to(device)
            with torch.no_grad():
                scores = reward_fn(stacked)
                if scores.dim() > 1:
                    scores = scores.mean(dim=tuple(range(1, scores.dim())))
                scores = scores.cpu().tolist()

            # Sort by score (descending)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            sorted_samples = [samples[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]

            ranked = RankedSamples(
                samples=sorted_samples,
                prompt=prompt,
                scores=sorted_scores,
            )
            dataset.add_ranked(ranked, strategy=pair_strategy)

        return dataset

    @classmethod
    def from_json(cls, path: str | Path) -> "PreferenceDataset":
        """Load from JSON file.

        Expected format::

            [
                {"chosen": [...], "rejected": [...], "prompt": "...", "margin": 0.8},
                ...
            ]
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        pairs = []
        for item in data:
            pairs.append(
                PreferencePair(
                    chosen=torch.tensor(item["chosen"]),
                    rejected=torch.tensor(item["rejected"]),
                    prompt=item.get("prompt"),
                    margin=item.get("margin", 1.0),
                    metadata=item.get("metadata", {}),
                )
            )

        return cls(pairs=pairs)

    def save(self, path: str | Path) -> None:
        """Save dataset to file."""
        path = Path(path)
        data = {
            "pairs": [
                {
                    "chosen": p.chosen.tolist(),
                    "rejected": p.rejected.tolist(),
                    "prompt": p.prompt,
                    "margin": p.margin,
                    "metadata": p.metadata,
                }
                for p in self.pairs
            ]
        }

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data["pairs"], f)
        else:
            torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceDataset":
        """Load dataset from file."""
        path = Path(path)

        if path.suffix == ".json":
            return cls.from_json(path)

        data = torch.load(path, weights_only=False)
        pairs = [
            PreferencePair(
                chosen=torch.tensor(p["chosen"]),
                rejected=torch.tensor(p["rejected"]),
                prompt=p.get("prompt"),
                margin=p.get("margin", 1.0),
                metadata=p.get("metadata", {}),
            )
            for p in data["pairs"]
        ]
        return cls(pairs=pairs)


def collate_preference_pairs(
    batch: list[PreferencePair],
) -> dict[str, Tensor | list[Any]]:
    """Collate function for DataLoader with PreferenceDataset.

    Returns a dictionary with batched tensors.

    Example:
        >>> loader = DataLoader(dataset, collate_fn=collate_preference_pairs)
        >>> for batch in loader:
        ...     chosen = batch["chosen"]  # [batch_size, seq_len]
        ...     rejected = batch["rejected"]
        ...     margins = batch["margin"]
    """
    return {
        "chosen": torch.stack([p.chosen for p in batch]),
        "rejected": torch.stack([p.rejected for p in batch]),
        "margin": torch.tensor([p.margin for p in batch]),
        "prompt": [p.prompt for p in batch],
    }


__all__ = [
    "PreferencePair",
    "RankedSamples",
    "StepwisePreference",
    "PreferenceDataset",
    "collate_preference_pairs",
]
