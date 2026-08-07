"""MaskGIT sampling utilities and scheduler.

Provides:
- Gumbel-based sampling strategies
- MaskGITScheduler for iterative parallel decoding
"""

import math

import torch
import torch.nn.functional as F

from .schedules import adaptive_masking, confidence_based_schedule, halton_schedule_1d


def _ensure_finite_positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def gumbel_max_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """
    Gumbel-max sampling for categorical distributions.

    Adds Gumbel noise and takes argmax (or softmax for soft samples).
    This is equivalent to sampling but differentiable when soft=True.

    Args:
        logits: [batch, seq_len, vocab_size] unnormalized logits
        temperature: sampling temperature
        hard: if True, return discrete samples, if False, return soft samples

    Returns:
        samples: [batch, seq_len] if hard, [batch, seq_len, vocab_size] if soft
    """
    _ensure_finite_positive(temperature, "temperature")
    eps = 1e-6
    u = torch.rand_like(logits).clamp(eps, 1 - eps)
    gumbel_noise = -torch.log(-torch.log(u) + eps)
    perturbed_logits = (logits + gumbel_noise) / temperature

    if hard:
        return torch.argmax(perturbed_logits, dim=-1)
    else:
        return F.softmax(perturbed_logits, dim=-1)


def sample_with_reranking(
    logits: torch.Tensor,
    num_samples: int = 4,
    rerank_by: str = "entropy",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample multiple completions and rerank them.

    Useful for final generation step to select best completion.

    Args:
        logits: [batch, seq_len, vocab_size] model predictions
        num_samples: number of samples to generate
        rerank_by: criterion for reranking ('entropy', 'confidence')
        temperature: sampling temperature

    Returns:
        best_sample: [batch, seq_len] best sample according to criterion
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must have shape [batch, seq_len, vocab], got {tuple(logits.shape)}"
        )
    if num_samples <= 0:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if rerank_by not in {"entropy", "confidence"}:
        raise ValueError(f"Unknown reranking criterion: {rerank_by}")
    _ensure_finite_positive(temperature, "temperature")

    batch_size, seq_len, vocab_size = logits.shape

    probs = F.softmax(logits / temperature, dim=-1)
    samples = []
    scores = []

    for _ in range(num_samples):
        sample = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(
            batch_size, seq_len
        )
        samples.append(sample)

        if rerank_by == "entropy":
            sample_probs = torch.gather(probs, dim=-1, index=sample.unsqueeze(-1)).squeeze(-1)
            entropy = -(sample_probs * torch.log(sample_probs.clamp(min=1e-6))).sum(dim=-1)
            score = -entropy
        elif rerank_by == "confidence":
            sample_probs = torch.gather(probs, dim=-1, index=sample.unsqueeze(-1)).squeeze(-1)
            score = sample_probs.mean(dim=-1)
        else:
            raise ValueError(f"Unknown reranking criterion: {rerank_by}")

        scores.append(score)

    samples = torch.stack(samples, dim=0)
    scores = torch.stack(scores, dim=0)

    best_indices = scores.argmax(dim=0)
    best_sample = samples[best_indices, torch.arange(batch_size)]

    return best_sample


class MaskGITScheduler:
    """
    Advanced scheduler for MaskGIT-style generation.

    Combines:
    - Flexible masking schedules (cosine, linear, sqrt, power, halton)
    - Temperature annealing
    - Confidence-based adaptation
    """

    def __init__(
        self,
        num_steps: int | None = None,
        mask_schedule: str | None = None,
        temp_schedule: str = "cosine",
        start_temp: float = 1.0,
        end_temp: float = 0.7,
        confidence_weight: float = 0.3,
        schedule_power: float = 2.0,
        halton_base: int = 2,
        max_unmask_per_step: float | None = 0.25,
        num_iterations: int | None = None,
        schedule: str | None = None,
    ):
        if num_steps is not None:
            resolved_steps = num_steps
        elif num_iterations is not None:
            resolved_steps = num_iterations
        else:
            resolved_steps = 12
        if resolved_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {resolved_steps}")
        if schedule_power <= 0:
            raise ValueError(f"schedule_power must be > 0, got {schedule_power}")
        if halton_base < 2:
            raise ValueError(f"halton_base must be >= 2, got {halton_base}")
        if not (0.0 <= confidence_weight <= 1.0):
            raise ValueError(f"confidence_weight must be between 0 and 1, got {confidence_weight}")
        if max_unmask_per_step is not None and not (0.0 < max_unmask_per_step <= 1.0):
            raise ValueError(
                f"max_unmask_per_step must be in (0, 1] or None, got {max_unmask_per_step}"
            )
        _ensure_finite_positive(start_temp, "start_temp")
        _ensure_finite_positive(end_temp, "end_temp")

        self.num_steps = resolved_steps
        self.mask_schedule = mask_schedule or schedule or "cosine"
        self.temp_schedule = temp_schedule
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.confidence_weight = confidence_weight
        self.schedule_power = schedule_power
        self.halton_base = halton_base
        self.max_unmask_per_step = max_unmask_per_step

    @property
    def num_iterations(self) -> int:
        return self.num_steps

    def step(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        step: int,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of MaskGIT generation.

        Uses a fixed K_MAX for topk to enable torch.compile(fullgraph=True).
        Final step uses a separate codepath that unmasks all remaining tokens
        without calling topk.

        Args:
            logits: [batch, seq_len, vocab_size] model output
            tokens: [batch, seq_len] current token sequence
            mask: [batch, seq_len] True for positions to update
            step: current step number
            temperature: optional override for temperature

        Returns:
            tokens: updated token sequence
            mask: updated mask (fewer positions still masked)
        """
        if step < 0 or step >= self.num_steps:
            raise ValueError(f"step must be in [0, {self.num_steps - 1}], got {step}")
        if logits.dim() != 3:
            raise ValueError(
                f"logits must have shape [batch, seq_len, vocab], got {tuple(logits.shape)}"
            )
        if tokens.dim() != 2 or mask.dim() != 2:
            raise ValueError(
                f"tokens and mask must have shape [batch, seq_len], got {tuple(tokens.shape)} and {tuple(mask.shape)}"
            )
        if tokens.shape != mask.shape:
            raise ValueError(
                f"tokens and mask must have matching shapes, got {tuple(tokens.shape)} and {tuple(mask.shape)}"
            )
        if logits.shape[:2] != tokens.shape:
            raise ValueError(
                f"logits batch/seq dims must match tokens, got {tuple(logits.shape[:2])} and {tuple(tokens.shape)}"
            )

        batch_size, seq_len = tokens.shape
        is_final_step = step == self.num_steps - 1

        # Dispatch to appropriate step kernel
        if is_final_step:
            return self._step_final(logits, tokens, mask, step, temperature)
        else:
            return self._step_normal(logits, tokens, mask, step, temperature)

    def _step_normal(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        step: int,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normal step: use capped topk for compile-friendly selection."""
        batch_size, seq_len = tokens.shape
        temp = temperature if temperature is not None else self.get_temperature(step)
        mask_ratio = self.get_mask_ratio(step)

        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)

        eps = 1e-6
        u = torch.rand_like(probs).clamp(eps, 1 - eps)
        gumbel = -torch.log(-torch.log(u) + eps)
        sampled = torch.argmax(probs + gumbel, dim=-1)

        confidences = torch.gather(probs, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
        confidences = confidences.masked_fill(~mask, -float("inf"))

        num_masked = mask.sum(dim=1)
        num_to_unmask = torch.ceil((1 - mask_ratio) * num_masked.float()).long()
        num_to_unmask = torch.clamp(num_to_unmask, min=1)

        # Compute K_MAX: fixed upper bound for topk (compile-friendly, no .item())
        if self.max_unmask_per_step is not None:
            K_MAX = max(1, int(self.max_unmask_per_step * seq_len + 0.5))
        else:
            K_MAX = seq_len

        # Clamp effective_k to K_MAX (all tensor ops, no sync)
        effective_k = torch.minimum(num_to_unmask, num_masked)
        effective_k = torch.clamp(effective_k, min=0, max=K_MAX)

        # Fixed-shape topk (no .item() for k)
        _, topk_indices = torch.topk(confidences, k=K_MAX, dim=-1, largest=True)

        # valid_k_mask filters to actual k per batch
        k_range = torch.arange(K_MAX, device=tokens.device)
        valid_k = k_range.unsqueeze(0) < effective_k.unsqueeze(1)

        update_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tokens.device)
        update_mask.scatter_(1, topk_indices, valid_k)

        new_tokens = torch.where(update_mask, sampled, tokens)
        new_mask = mask & ~update_mask

        return new_tokens, new_mask

    def _step_final(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        step: int,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Final step: unmask ALL remaining masked positions (no topk needed)."""
        temp = temperature if temperature is not None else self.get_temperature(step)

        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)

        eps = 1e-6
        u = torch.rand_like(probs).clamp(eps, 1 - eps)
        gumbel = -torch.log(-torch.log(u) + eps)
        sampled = torch.argmax(probs + gumbel, dim=-1)

        new_tokens = torch.where(mask, sampled, tokens)
        new_mask = torch.zeros_like(mask)

        return new_tokens, new_mask

    def get_mask_ratio(self, step: int) -> float:
        """
        Get mask ratio for a given step.

        CONTRACT:
        - mask_ratio is the fraction of tokens that should REMAIN masked
        - mask_ratio must decay from ~1.0 (step=0) to ~0.0 (step=T-1)
        - All schedules must return values in [0, 1]
        - progress is defined as step/(num_steps-1) so final step has progress=1.0

        The unmask_ratio = 1 - mask_ratio represents the fraction of
        currently-masked tokens to unmask at this step.

        Args:
            step: Current step (0-indexed, must be < num_steps)

        Returns:
            mask_ratio in [0, 1], decaying from ~1 at step=0 to ~0 at step=T-1
        """
        # Use (num_steps - 1) as denominator so final step has progress=1.0
        # This ensures mask_ratio reaches 0 at the final step
        progress = step / max(self.num_steps - 1, 1)

        if self.mask_schedule == "cosine":
            # Cosine decay from 1.0 to 0.0
            # At progress=0: cos(0) = 1, so (1 + 1)/2 = 1.0
            # At progress=1: cos(π) = -1, so (1 + (-1))/2 = 0.0
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.mask_schedule == "cosine_legacy":
            # Legacy cosine matching transformer.py inline formula
            # Uses (step+1)/num_steps instead of step/num_steps
            legacy_progress = (step + 1) / self.num_steps
            return max(0.0, math.cos(legacy_progress * math.pi / 2))
        elif self.mask_schedule == "linear":
            # Linear decay from 1.0 to 0.0
            return 1.0 - progress
        elif self.mask_schedule == "sqrt":
            # Sqrt decay from 1.0 to 0.0 (slower start, faster end)
            return 1.0 - math.sqrt(progress)
        elif self.mask_schedule == "power":
            # Power decay from 1.0 to 0.0
            return 1.0 - (progress**self.schedule_power)
        elif self.mask_schedule == "halton":
            # Halton schedule uses (step+1) convention - kept for compatibility
            return halton_schedule_1d(step, self.num_steps, base=self.halton_base)
        else:
            raise ValueError(f"Unknown mask schedule: {self.mask_schedule}")

    def get_temperature(self, step: int) -> float:
        progress = step / max(self.num_steps - 1, 1)

        if self.temp_schedule == "constant":
            return self.start_temp
        elif self.temp_schedule == "linear":
            return self.start_temp + (self.end_temp - self.start_temp) * progress
        elif self.temp_schedule == "cosine":
            return self.end_temp + (self.start_temp - self.end_temp) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(f"Unknown temp schedule: {self.temp_schedule}")

    def get_mask(
        self,
        logits: torch.Tensor,
        current_tokens: torch.Tensor,
        mask_token: int,
        step: int,
    ) -> torch.Tensor:
        target_ratio = self.get_mask_ratio(step)
        temperature = self.get_temperature(step)

        if self.confidence_weight > 0:
            return adaptive_masking(
                logits=logits,
                current_tokens=current_tokens,
                mask_token=mask_token,
                step=step,
                total_steps=self.num_steps,
                base_schedule=self.mask_schedule,
                confidence_weight=self.confidence_weight,
                temperature=temperature,
                halton_base=self.halton_base,
            )
        else:
            return confidence_based_schedule(
                logits=logits,
                current_tokens=current_tokens,
                mask_token=mask_token,
                target_mask_ratio=target_ratio,
                temperature=temperature,
            )


__all__ = [
    "gumbel_max_sampling",
    "sample_with_reranking",
    "MaskGITScheduler",
]
