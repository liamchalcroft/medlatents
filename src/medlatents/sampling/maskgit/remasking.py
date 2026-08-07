"""Running Confidence Remasking for MaskGIT.

Provides:
- RunningConfidenceRemasker: Tracks confidence across steps for stable unmasking
- RunningConfidenceScheduler: MaskGIT scheduler with running confidence remasking
- RCRGenerator: Full generation pipeline with running confidence remasking
"""

import torch
import torch.nn.functional as F

from .sampling import MaskGITScheduler


def _validate_probability(value: float, name: str, *, strict_upper: bool = True) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    upper_ok = value < 1.0 if strict_upper else value <= 1.0
    if not (value >= 0.0 and upper_ok):
        bound = "[0, 1)" if strict_upper else "[0, 1]"
        raise ValueError(f"{name} must be in {bound}, got {value}")


class RunningConfidenceRemasker:
    """Running Confidence Remasking for improved MaskGIT generation.

    Instead of using only the current step's confidence, this tracks
    confidence across multiple steps and uses running statistics to
    make more stable unmasking decisions.

    Reference: "Improving MaskGIT with Running Confidence" (2024)

    Args:
        decay: Exponential decay for running confidence (0.9 = 90% old, 10% new)
        remask_threshold: Tokens below this percentile of running confidence get remasked
        min_steps_before_remask: Don't remask until this many steps have passed
        max_remask_per_step: Maximum fraction of tokens to remask per step
    """

    def __init__(
        self,
        decay: float = 0.9,
        remask_threshold: float = 0.1,
        min_steps_before_remask: int = 2,
        max_remask_per_step: float = 0.1,
    ):
        if not (0.0 < decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        _validate_probability(remask_threshold, "remask_threshold", strict_upper=False)
        if not isinstance(min_steps_before_remask, int) or min_steps_before_remask < 0:
            raise ValueError(
                f"min_steps_before_remask must be non-negative int, got {min_steps_before_remask}"
            )
        _validate_probability(max_remask_per_step, "max_remask_per_step", strict_upper=False)

        self.decay = decay
        self.remask_threshold = remask_threshold
        self.min_steps_before_remask = min_steps_before_remask
        self.max_remask_per_step = max_remask_per_step

        self.running_confidence: torch.Tensor | None = None
        self.step_count = 0

    def reset(self):
        """Reset running statistics for new generation."""
        self.running_confidence = None
        self.step_count = 0

    def update(
        self,
        confidence: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Update running confidence and compute remasking decisions."""
        batch_size, seq_len = confidence.shape

        if self.running_confidence is None:
            self.running_confidence = confidence.clone()
            self.step_count = 1
            return torch.zeros_like(mask)

        unmasked = ~mask
        new_running = self.running_confidence.clone()
        new_running[unmasked] = (
            self.decay * self.running_confidence[unmasked] + (1 - self.decay) * confidence[unmasked]
        )
        self.running_confidence = new_running
        self.step_count += 1

        if self.step_count < self.min_steps_before_remask:
            return torch.zeros_like(mask)

        running_conf_masked = self.running_confidence.clone()
        running_conf_masked[mask] = float("inf")

        flat_conf = running_conf_masked.view(-1)
        valid_conf = flat_conf[flat_conf < float("inf")]

        if len(valid_conf) == 0:
            return torch.zeros_like(mask)

        threshold = torch.quantile(valid_conf, self.remask_threshold)

        should_remask = (running_conf_masked < threshold) & unmasked

        # Use fixed K_MAX for topk to enable torch.compile(fullgraph=True)
        # max_remask is the cap per-batch; K_MAX is the fixed topk size
        max_remask = int(seq_len * self.max_remask_per_step)
        remask_count = should_remask.sum(dim=1)

        # Compute excess per batch (how many above max_remask)
        excess_per_batch = (remask_count - max_remask).clamp(min=0)

        # Fixed K_MAX: use max_remask as upper bound (compile-friendly, no .item())
        K_MAX = max(1, max_remask)

        # Only process if there's any excess (tensor check avoids empty topk)
        has_excess = excess_per_batch.sum() > 0

        if has_excess:
            conf_for_limit = running_conf_masked.clone()
            conf_for_limit[~should_remask] = float("inf")

            # Fixed-shape topk (no .item() for k)
            _, topk_conf_indices = torch.topk(conf_for_limit, k=K_MAX, dim=-1, largest=True)

            # valid_excess filters to actual excess per batch
            excess_range = torch.arange(K_MAX, device=confidence.device)
            valid_excess = excess_range.unsqueeze(0) < excess_per_batch.unsqueeze(1)

            unremask_mask = torch.zeros_like(should_remask)
            unremask_mask.scatter_(1, topk_conf_indices, valid_excess)
            should_remask = should_remask & ~unremask_mask

        return should_remask


class RunningConfidenceScheduler(MaskGITScheduler):
    """MaskGIT scheduler with running confidence remasking."""

    def __init__(
        self,
        remasker: RunningConfidenceRemasker | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.remasker = remasker or RunningConfidenceRemasker()

    def step(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        step: int,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one step with running confidence remasking.

        Uses fixed K_MAX for topk to enable torch.compile(fullgraph=True).
        Final step unmasks all remaining positions without topk.
        """
        batch_size, seq_len = tokens.shape
        is_final_step = step == self.num_steps - 1

        temp = temperature if temperature is not None else self.get_temperature(step)
        mask_ratio = self.get_mask_ratio(step)

        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)

        # Gumbel noise (use eps=1e-6 for fp16 stability)
        eps = 1e-6
        u = torch.rand_like(probs).clamp(eps, 1 - eps)
        gumbel = -torch.log(-torch.log(u))
        sampled = torch.argmax(probs + gumbel, dim=-1)

        confidences = torch.gather(probs, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)

        remask = self.remasker.update(confidences, mask)
        new_mask = mask | remask

        # Final step: unmask all remaining masked positions (no topk needed)
        if is_final_step:
            new_tokens = torch.where(new_mask, sampled, tokens)
            final_mask = torch.zeros_like(new_mask)
            return new_tokens, final_mask

        # Normal step: use capped topk for compile-friendly selection
        num_masked = new_mask.sum(dim=1)
        num_to_unmask = torch.ceil((1 - mask_ratio) * num_masked.float()).long()
        num_to_unmask = torch.clamp(num_to_unmask, min=1)

        # Compute K_MAX: fixed upper bound for topk (compile-friendly, no .item())
        if self.max_unmask_per_step is not None:
            K_MAX = max(1, int(self.max_unmask_per_step * seq_len + 0.5))
        else:
            K_MAX = seq_len

        masked_conf = confidences.clone()
        masked_conf[~new_mask] = -float("inf")

        # Clamp effective_k to K_MAX (all tensor ops, no sync)
        effective_k = torch.minimum(num_to_unmask, num_masked)
        effective_k = torch.clamp(effective_k, min=0, max=K_MAX)

        # Fixed-shape topk (no .item() for k)
        _, topk_indices = torch.topk(masked_conf, k=K_MAX, dim=-1, largest=True)

        # valid_k_mask filters to actual k per batch
        k_range = torch.arange(K_MAX, device=tokens.device)
        valid_k = k_range.unsqueeze(0) < effective_k.unsqueeze(1)

        update_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tokens.device)
        update_mask.scatter_(1, topk_indices, valid_k)

        new_tokens = torch.where(update_mask, sampled, tokens)
        final_mask = new_mask & ~update_mask

        return new_tokens, final_mask

    def reset(self):
        """Reset remasker state for new generation."""
        self.remasker.reset()


class RCRGenerator:
    """Running Confidence Remasking Generator for MaskGIT."""

    def __init__(
        self,
        scheduler: RunningConfidenceScheduler | MaskGITScheduler,
        mask_token: int,
    ):
        self.scheduler = scheduler
        self.mask_token = mask_token

    def generate(
        self,
        model: torch.nn.Module,
        batch_size: int,
        seq_len: int,
        condition: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Generate tokens with running confidence remasking.

        Uses fixed-iteration loop (no Python breaks for compile-friendliness).
        """
        device = next(model.parameters()).device

        if hasattr(self.scheduler, "reset"):
            self.scheduler.reset()

        tokens = torch.full(
            (batch_size, seq_len),
            self.mask_token,
            dtype=torch.long,
            device=device,
        )
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        if condition is not None and condition_mask is not None:
            tokens[condition_mask] = condition[condition_mask]
            mask[condition_mask] = False

        metrics = {
            "steps_taken": 0,
            "remask_events": 0,
        }

        # Note: Loop runs to completion for compile-friendliness (no data-dependent breaks).

        # Track remask events for metrics (defer .item() to after loop)
        remask_count_tensor = torch.tensor(0, device=device)

        for step in range(self.scheduler.num_steps):
            mask_ratio = self.scheduler.get_mask_ratio(step)
            logits = model(tokens, mask_ratio=mask_ratio)

            mask_before = mask.clone()

            tokens, mask = self.scheduler.step(logits, tokens, mask, step)

            remasked = (~mask_before) & mask
            remask_count_tensor += remasked.sum()

        # Convert metrics after loop (single sync point)
        metrics["remask_events"] = remask_count_tensor.item()
        metrics["steps_taken"] = self.scheduler.num_steps

        return tokens, metrics


__all__ = [
    "RunningConfidenceRemasker",
    "RunningConfidenceScheduler",
    "RCRGenerator",
]
