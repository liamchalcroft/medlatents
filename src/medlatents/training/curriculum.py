"""Curriculum learning strategies for improved training convergence.

Curriculum learning progressively increases task difficulty during training,
leading to faster convergence and better generalization.

Strategies implemented:
1. Sequence length curriculum: Start with short sequences, increase over time
2. Masking ratio curriculum: Start with high masking, decrease (for MaskGIT)
3. Noise level curriculum: Start with high noise, decrease (for diffusion)
4. Token difficulty curriculum: Start with easy tokens, add harder ones
5. Multi-scale curriculum: Progressively increase resolution

References:
- Bengio et al., "Curriculum Learning" (ICML 2009)
- Soviany et al., "Curriculum Learning: A Survey" (2022)
- Li et al., "Curriculum Learning for Natural Language Understanding" (ACL 2020)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class CurriculumSchedule(Enum):
    """Schedule types for curriculum progression."""

    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"
    WARMUP_LINEAR = "warmup_linear"
    POLYNOMIAL = "polynomial"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Attributes:
        start_value: Initial curriculum value (e.g., min seq length)
        end_value: Final curriculum value (e.g., max seq length)
        warmup_steps: Steps before curriculum begins
        total_steps: Total steps for curriculum (None = use training steps)
        schedule: Schedule type for progression
        schedule_power: Power for polynomial schedule
    """

    start_value: float
    end_value: float
    warmup_steps: int = 0
    total_steps: int | None = None
    schedule: CurriculumSchedule = CurriculumSchedule.LINEAR
    schedule_power: float = 2.0


class CurriculumScheduler(ABC):
    """Base class for curriculum schedulers."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_step = 0

    def step(self) -> None:
        """Advance curriculum by one step."""
        self.current_step += 1

    def set_step(self, step: int) -> None:
        """Set curriculum to specific step."""
        self.current_step = step

    @abstractmethod
    def get_value(self) -> float:
        """Get current curriculum value."""
        raise NotImplementedError

    def _get_progress(self, total_steps: int) -> float:
        """Get progress fraction [0, 1]."""
        if self.current_step < self.config.warmup_steps:
            return 0.0

        effective_step = self.current_step - self.config.warmup_steps
        effective_total = total_steps - self.config.warmup_steps

        return min(1.0, effective_step / max(1, effective_total))

    def _interpolate(self, progress: float) -> float:
        """Interpolate between start and end values based on progress."""
        start = self.config.start_value
        end = self.config.end_value

        schedule = self.config.schedule

        if schedule == CurriculumSchedule.LINEAR:
            return start + (end - start) * progress

        elif schedule == CurriculumSchedule.COSINE:
            # Cosine annealing from start to end
            return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2

        elif schedule == CurriculumSchedule.EXPONENTIAL:
            # Exponential growth/decay
            if start == 0:
                start = 1e-6
            return start * (end / start) ** progress

        elif schedule == CurriculumSchedule.POLYNOMIAL:
            return start + (end - start) * (progress**self.config.schedule_power)

        elif schedule == CurriculumSchedule.WARMUP_LINEAR:
            # Linear warmup then hold
            return start + (end - start) * progress

        elif schedule == CurriculumSchedule.STEP:
            # Step function at midpoint
            return end if progress > 0.5 else start

        return start + (end - start) * progress


class SequenceLengthCurriculum(CurriculumScheduler):
    """Curriculum that progressively increases sequence length.

    Start training with short sequences for faster iteration,
    then increase to full length for learning long-range dependencies.

    Args:
        min_length: Starting sequence length
        max_length: Final sequence length
        total_steps: Total training steps
        schedule: Progression schedule
    """

    def __init__(
        self,
        min_length: int,
        max_length: int,
        total_steps: int,
        schedule: CurriculumSchedule = CurriculumSchedule.LINEAR,
        warmup_steps: int = 0,
    ):
        config = CurriculumConfig(
            start_value=float(min_length),
            end_value=float(max_length),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule=schedule,
        )
        super().__init__(config)
        self.min_length = min_length
        self.max_length = max_length

    def get_value(self) -> float:
        """Get current sequence length."""
        total = self.config.total_steps or 100000
        progress = self._get_progress(total)
        length = self._interpolate(progress)
        return int(round(length))

    def get_length(self) -> int:
        """Get current sequence length as integer."""
        return int(self.get_value())

    def truncate_batch(
        self,
        tokens: torch.Tensor,
        random_start: bool = True,
    ) -> torch.Tensor:
        """Truncate batch to current curriculum length.

        Args:
            tokens: [batch, seq_len] token tensor
            random_start: If True, sample random start position

        Returns:
            Truncated tokens [batch, curr_length]
        """
        curr_length = self.get_length()
        batch_size, seq_len = tokens.shape

        if seq_len <= curr_length:
            return tokens

        if random_start:
            max_start = seq_len - curr_length
            starts = torch.randint(0, max_start + 1, (batch_size,), device=tokens.device)

            # Gather sequences
            indices = starts.unsqueeze(1) + torch.arange(curr_length, device=tokens.device)
            return tokens.gather(1, indices)
        else:
            return tokens[:, :curr_length]


class MaskingRatioCurriculum(CurriculumScheduler):
    """Curriculum that decreases masking ratio over training.

    For MaskGIT-style training, start with high masking (easier task)
    and decrease to lower masking (harder, more context needed).

    Args:
        start_ratio: Initial masking ratio (e.g., 0.9)
        end_ratio: Final masking ratio (e.g., 0.1)
        total_steps: Total training steps
    """

    def __init__(
        self,
        start_ratio: float = 0.9,
        end_ratio: float = 0.1,
        total_steps: int = 100000,
        schedule: CurriculumSchedule = CurriculumSchedule.COSINE,
    ):
        config = CurriculumConfig(
            start_value=start_ratio,
            end_value=end_ratio,
            total_steps=total_steps,
            schedule=schedule,
        )
        super().__init__(config)

    def get_value(self) -> float:
        """Get current masking ratio."""
        total = self.config.total_steps or 100000
        progress = self._get_progress(total)
        return self._interpolate(progress)

    def get_mask_ratio(self) -> float:
        """Get current masking ratio."""
        return max(0.0, min(1.0, self.get_value()))

    def create_mask(
        self,
        shape: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Create mask with current curriculum ratio.

        Args:
            shape: (batch_size, seq_len)
            device: Device for tensor

        Returns:
            Boolean mask [batch, seq_len] where True = masked
        """
        ratio = self.get_mask_ratio()
        batch_size, seq_len = shape

        # Random mask with curriculum ratio
        mask = torch.rand(batch_size, seq_len, device=device) < ratio

        return mask


class NoiseLevelCurriculum(CurriculumScheduler):
    """Curriculum that adjusts noise level for diffusion training.

    For diffusion models, can start with higher noise levels (easier denoising)
    and progress to full noise schedule.

    Args:
        start_t_max: Initial maximum timestep (fraction of full schedule)
        end_t_max: Final maximum timestep (1.0 = full schedule)
        total_steps: Total training steps
    """

    def __init__(
        self,
        start_t_max: float = 0.5,
        end_t_max: float = 1.0,
        total_steps: int = 100000,
        schedule: CurriculumSchedule = CurriculumSchedule.LINEAR,
    ):
        config = CurriculumConfig(
            start_value=start_t_max,
            end_value=end_t_max,
            total_steps=total_steps,
            schedule=schedule,
        )
        super().__init__(config)

    def get_value(self) -> float:
        """Get current maximum timestep fraction."""
        total = self.config.total_steps or 100000
        progress = self._get_progress(total)
        return self._interpolate(progress)

    def sample_timesteps(
        self,
        batch_size: int,
        num_timesteps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample timesteps with curriculum-limited maximum.

        Args:
            batch_size: Number of samples
            num_timesteps: Total timesteps in diffusion schedule
            device: Device for tensor

        Returns:
            Timesteps [batch_size] in range [0, max_t]
        """
        t_max_frac = self.get_value()
        max_t = int(num_timesteps * t_max_frac)
        max_t = max(1, max_t)  # At least 1

        return torch.randint(0, max_t, (batch_size,), device=device)


class TokenDifficultyCurriculum(CurriculumScheduler):
    """Curriculum based on token difficulty/frequency.

    Train first on frequent/easy tokens, progressively include
    rare/difficult tokens.

    Args:
        token_frequencies: Frequency count for each token
        start_coverage: Initial vocabulary coverage (e.g., top 50%)
        end_coverage: Final vocabulary coverage (100%)
        total_steps: Total training steps
    """

    def __init__(
        self,
        token_frequencies: torch.Tensor,
        start_coverage: float = 0.5,
        end_coverage: float = 1.0,
        total_steps: int = 100000,
    ):
        config = CurriculumConfig(
            start_value=start_coverage,
            end_value=end_coverage,
            total_steps=total_steps,
            schedule=CurriculumSchedule.LINEAR,
        )
        super().__init__(config)

        # Sort tokens by frequency (descending)
        sorted_indices = torch.argsort(token_frequencies, descending=True)
        self.sorted_indices = sorted_indices
        self.vocab_size = len(token_frequencies)

        # Create mapping from sorted position to original index
        self.rank_to_token = sorted_indices

    def get_value(self) -> float:
        """Get current vocabulary coverage."""
        total = self.config.total_steps or 100000
        progress = self._get_progress(total)
        return self._interpolate(progress)

    def get_active_vocab_size(self) -> int:
        """Get current active vocabulary size."""
        coverage = self.get_value()
        return max(1, int(self.vocab_size * coverage))

    def get_active_tokens(self) -> torch.Tensor:
        """Get currently active token IDs."""
        n_active = self.get_active_vocab_size()
        return self.rank_to_token[:n_active]

    def filter_targets(
        self,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Filter targets to only include active tokens.

        Tokens not in active vocabulary are set to ignore_index.

        Args:
            targets: [batch, seq_len] target tokens
            ignore_index: Value for ignored positions

        Returns:
            Filtered targets
        """
        active_tokens = self.get_active_tokens()
        active_set = set(active_tokens.tolist())

        filtered = targets.clone()
        mask = torch.tensor(
            [[t.item() not in active_set for t in row] for row in targets],
            device=targets.device,
        )
        filtered[mask] = ignore_index

        return filtered


class MultiScaleCurriculum(CurriculumScheduler):
    """Curriculum for multi-scale/multi-resolution training.

    For image/volume models, start training at low resolution
    and progressively increase.

    Args:
        scales: List of scales (e.g., [0.25, 0.5, 1.0])
        steps_per_scale: Steps at each scale, or total_steps / len(scales)
        blend_steps: Steps to blend between scales
    """

    def __init__(
        self,
        scales: list[float],
        total_steps: int,
        steps_per_scale: int | None = None,
        blend_steps: int = 1000,
    ):
        self.scales = sorted(scales)
        self.total_steps = total_steps
        self.blend_steps = blend_steps

        if steps_per_scale is None:
            self.steps_per_scale = total_steps // len(scales)
        else:
            self.steps_per_scale = steps_per_scale

        # Use linear config for underlying progress
        config = CurriculumConfig(
            start_value=scales[0],
            end_value=scales[-1],
            total_steps=total_steps,
            schedule=CurriculumSchedule.STEP,
        )
        super().__init__(config)

    def get_value(self) -> float:
        """Get current scale."""
        return self.get_scale()

    def get_scale(self) -> float:
        """Get current training scale."""
        scale_idx = self.current_step // self.steps_per_scale
        scale_idx = min(scale_idx, len(self.scales) - 1)
        return self.scales[scale_idx]

    def get_scale_index(self) -> int:
        """Get current scale index."""
        scale_idx = self.current_step // self.steps_per_scale
        return min(scale_idx, len(self.scales) - 1)

    def should_increase_scale(self) -> bool:
        """Check if it's time to increase scale."""
        return (
            self.current_step > 0
            and self.current_step % self.steps_per_scale == 0
            and self.get_scale_index() < len(self.scales) - 1
        )


class CompositeCurriculum:
    """Combine multiple curriculum strategies.

    Manages multiple curricula that can be updated together.

    Args:
        curricula: Dict mapping names to curriculum schedulers
    """

    def __init__(self, curricula: dict[str, CurriculumScheduler]):
        self.curricula = curricula

    def step(self) -> None:
        """Advance all curricula by one step."""
        for curriculum in self.curricula.values():
            curriculum.step()

    def set_step(self, step: int) -> None:
        """Set all curricula to specific step."""
        for curriculum in self.curricula.values():
            curriculum.set_step(step)

    def get_values(self) -> dict[str, float]:
        """Get all current curriculum values."""
        return {name: c.get_value() for name, c in self.curricula.items()}

    def __getitem__(self, name: str) -> CurriculumScheduler:
        return self.curricula[name]

    def __contains__(self, name: str) -> bool:
        return name in self.curricula


class CurriculumDataLoader:
    """DataLoader wrapper that applies curriculum to batches.

    Args:
        dataloader: Base dataloader
        curriculum: Curriculum scheduler (e.g., SequenceLengthCurriculum)
        apply_fn: Function to apply curriculum to batch
    """

    def __init__(
        self,
        dataloader: Any,
        curriculum: CurriculumScheduler,
        apply_fn: Callable[[Any, CurriculumScheduler], Any] | None = None,
    ):
        self.dataloader = dataloader
        self.curriculum = curriculum
        self.apply_fn = apply_fn or self._default_apply

    def _default_apply(self, batch: Any, curriculum: CurriculumScheduler) -> Any:
        """Default application: truncate sequence length."""
        if isinstance(curriculum, SequenceLengthCurriculum):
            if isinstance(batch, dict) and "tokens" in batch:
                batch["tokens"] = curriculum.truncate_batch(batch["tokens"])
            elif isinstance(batch, torch.Tensor):
                batch = curriculum.truncate_batch(batch)
        return batch

    def __iter__(self) -> Iterator:
        for batch in self.dataloader:
            yield self.apply_fn(batch, self.curriculum)
            self.curriculum.step()

    def __len__(self) -> int:
        return len(self.dataloader)


def create_curriculum_from_config(config: dict[str, Any]) -> CurriculumScheduler:
    """Create curriculum from configuration dict.

    Args:
        config: Dict with curriculum type and parameters

    Returns:
        Configured curriculum scheduler
    """
    curriculum_type = config.get("type", "sequence_length")

    if curriculum_type == "sequence_length":
        return SequenceLengthCurriculum(
            min_length=config.get("min_length", 64),
            max_length=config.get("max_length", 1024),
            total_steps=config.get("total_steps", 100000),
            schedule=CurriculumSchedule(config.get("schedule", "linear")),
        )

    elif curriculum_type == "masking_ratio":
        return MaskingRatioCurriculum(
            start_ratio=config.get("start_ratio", 0.9),
            end_ratio=config.get("end_ratio", 0.1),
            total_steps=config.get("total_steps", 100000),
            schedule=CurriculumSchedule(config.get("schedule", "cosine")),
        )

    elif curriculum_type == "noise_level":
        return NoiseLevelCurriculum(
            start_t_max=config.get("start_t_max", 0.5),
            end_t_max=config.get("end_t_max", 1.0),
            total_steps=config.get("total_steps", 100000),
        )

    elif curriculum_type == "multi_scale":
        return MultiScaleCurriculum(
            scales=config.get("scales", [0.25, 0.5, 1.0]),
            total_steps=config.get("total_steps", 100000),
        )

    else:
        raise ValueError(f"Unknown curriculum type: {curriculum_type}")


__all__ = [
    "CurriculumSchedule",
    "CurriculumConfig",
    "CurriculumScheduler",
    "SequenceLengthCurriculum",
    "MaskingRatioCurriculum",
    "NoiseLevelCurriculum",
    "TokenDifficultyCurriculum",
    "MultiScaleCurriculum",
    "CompositeCurriculum",
    "CurriculumDataLoader",
    "create_curriculum_from_config",
]
