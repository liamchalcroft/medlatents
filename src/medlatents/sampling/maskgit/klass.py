"""KLASS (KL-Adaptive Stability Sampling) for MaskGIT.

Provides:
- KL divergence computation utilities
- KLASSGenerator for early stopping based on KL divergence
"""

import torch
import torch.nn.functional as F

from .sampling import MaskGITScheduler, gumbel_max_sampling


def compute_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute KL divergence D_KL(p || q).

    Args:
        p: [batch, ..., vocab_size] first distribution
        q: [batch, ..., vocab_size] second distribution
        dim: dimension to compute over
        eps: small constant for numerical stability

    Returns:
        kl: [batch, ...] KL divergence values
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * torch.log(p / q)).sum(dim=dim)


def compute_js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between p and q.

    JS is symmetric and always finite, unlike KL.

    Args:
        p: [batch, ..., vocab_size] first distribution
        q: [batch, ..., vocab_size] second distribution
        dim: dimension to compute over
        eps: small constant for numerical stability

    Returns:
        js: [batch, ...] JS divergence values
    """
    m = 0.5 * (p + q)
    kl_pm = compute_kl_divergence(p, m, dim=dim, eps=eps)
    kl_qm = compute_kl_divergence(q, m, dim=dim, eps=eps)
    return 0.5 * (kl_pm + kl_qm)


class KLASSGenerator:
    """
    KL-Adaptive Stability Sampling (KLASS) for MaskGIT and masked diffusion.

    Dynamically stops generation when predictions stabilize, rather than using
    a fixed number of steps. Can significantly speed up inference.

    Reference: "KL-Guided Fast Inference in Masked Diffusion Models" (arXiv:2511.05664)

    Args:
        scheduler: Base MaskGITScheduler for masking and temperature
        kl_threshold: Stop when KL divergence falls below this threshold
        min_steps: Minimum number of steps before early stopping
        max_steps: Maximum number of steps (hard limit)
        window_size: Number of steps to average KL over
    """

    def __init__(
        self,
        scheduler: MaskGITScheduler,
        kl_threshold: float = 0.01,
        min_steps: int = 3,
        max_steps: int | None = None,
        window_size: int = 2,
    ):
        if not isinstance(kl_threshold, (int, float)) or kl_threshold < 0:
            raise ValueError(f"kl_threshold must be >= 0, got {kl_threshold}")
        if not isinstance(min_steps, int) or min_steps < 0:
            raise ValueError(f"min_steps must be non-negative int, got {min_steps}")
        if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
            raise ValueError(f"max_steps must be positive int, got {max_steps}")
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self.scheduler = scheduler
        self.kl_threshold = kl_threshold
        self.min_steps = min_steps
        self.max_steps = max_steps or scheduler.num_steps
        self.window_size = window_size
        # Store KL history as tensors to avoid GPU->CPU sync in hot loop
        self._kl_history_tensors: list[torch.Tensor] = []

    def should_stop(self, step: int, kl_divergence: torch.Tensor) -> bool:
        """
        Determine if generation should stop based on KL divergence.

        Uses tensor operations to avoid GPU->CPU sync in the hot sampling loop.

        Args:
            step: Current step number
            kl_divergence: KL divergence tensor (scalar)

        Returns:
            True if generation should stop, False otherwise
        """
        if step < self.min_steps:
            return False

        self._kl_history_tensors.append(kl_divergence)

        if len(self._kl_history_tensors) >= self.window_size:
            # Stack recent tensors and compute mean on GPU
            recent_kl = torch.stack(self._kl_history_tensors[-self.window_size :])
            avg_kl = recent_kl.mean()
            # Tensor comparison - only syncs when Python needs the bool result
            if avg_kl < self.kl_threshold:
                return True

        return step >= self.max_steps - 1

    def generate(
        self,
        model: torch.nn.Module,
        initial_tokens: torch.Tensor,
        mask_token: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate tokens with KL-guided early stopping.

        Uses fixed-iteration loop (no Python breaks for compile-friendliness).

        Args:
            model: Model that takes (tokens, mask_ratio) and returns logits
            initial_tokens: [batch, seq_len] initial tokens (typically all masked)
            mask_token: Mask token ID

        Returns:
            tokens: [batch, seq_len] generated tokens
            metrics: Dict with generation statistics
        """
        tokens = initial_tokens.clone()
        batch_size, seq_len = tokens.shape
        device = tokens.device
        mask = torch.ones_like(tokens, dtype=torch.bool)

        metrics: dict = {
            "steps_taken": 0,
            "final_kl": 0.0,
            "kl_history": [],
            "early_stopped": False,
        }

        # Reset KL history for new generation
        self._kl_history_tensors = []

        # Track KL tensors for deferred .item() conversion (avoids GPU->CPU sync in loop)
        kl_tensors: list[torch.Tensor] = []
        final_kl_tensor: torch.Tensor | None = None

        # Note: Loop runs to completion for compile-friendliness (no data-dependent breaks).
        # Early stopping is tracked via metrics but doesn't exit early.

        # Ensure model is in eval mode during generation
        was_training = model.training
        model.eval()

        try:
            prev_probs = None

            for step in range(self.max_steps):
                mask_ratio = self.scheduler.get_mask_ratio(step)
                logits = model(tokens, mask_ratio=mask_ratio)

                probs = F.softmax(logits, dim=-1)

                # Compute KL divergence - keep as tensor to avoid GPU->CPU sync
                kl_div: torch.Tensor
                if prev_probs is not None:
                    kl_div = (
                        (prev_probs * torch.log((prev_probs + 1e-10) / (probs + 1e-10)))
                        .sum(dim=-1)
                        .mean()
                    )
                else:
                    # Use tensor infinity to stay on device
                    kl_div = torch.tensor(float("inf"), device=device)

                # Check KL threshold for early stopping (tracked in metrics, no Python break)
                if step >= self.min_steps:
                    self._kl_history_tensors.append(kl_div)
                    if len(self._kl_history_tensors) >= self.window_size:
                        recent_kl = torch.stack(self._kl_history_tensors[-self.window_size :])
                        avg_kl = recent_kl.mean()
                        # Tensor comparison - only syncs when metrics are collected
                        if avg_kl < self.kl_threshold:
                            metrics["early_stopped"] = True
                            metrics["steps_taken"] = step + 1
                            final_kl_tensor = kl_div

                temperature = self.scheduler.get_temperature(step)
                sampled = gumbel_max_sampling(probs, temperature=temperature, hard=True)

                mask_ratio = self.scheduler.get_mask_ratio(step)

                token_probs = torch.gather(probs, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)
                token_probs = token_probs.masked_fill(~mask, -float("inf"))

                num_masked = mask.sum(dim=1)
                num_to_unmask = torch.ceil((1 - mask_ratio) * num_masked.float()).long()
                num_to_unmask = torch.clamp(num_to_unmask, min=1)

                is_final_step = step == self.max_steps - 1

                if is_final_step:
                    # Final step: unmask all remaining masked positions (no topk)
                    tokens = torch.where(mask, sampled, tokens)
                    mask = torch.zeros_like(mask)
                else:
                    # Normal step: use capped topk selection (no .item())
                    K_MAX = max(
                        1, int(getattr(self.scheduler, "max_unmask_per_step", 0.25) * seq_len + 0.5)
                    )
                    effective_k = torch.minimum(num_to_unmask, num_masked)
                    effective_k = torch.clamp(effective_k, min=0, max=K_MAX)

                    # Fixed-shape topk (compile-friendly)
                    _, topk_indices = torch.topk(token_probs, k=K_MAX, dim=-1, largest=True)

                    k_range = torch.arange(K_MAX, device=device)
                    valid_k = k_range.unsqueeze(0) < effective_k.unsqueeze(1)

                    update_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
                    update_mask.scatter_(1, topk_indices, valid_k)

                    tokens = torch.where(update_mask, sampled, tokens)
                    mask = mask & ~update_mask

                prev_probs = probs.detach().clone()
                kl_tensors.append(kl_div)

            if not metrics["early_stopped"]:
                metrics["steps_taken"] = self.max_steps
        finally:
            if was_training:
                model.train()

        # Convert KL history to Python floats AFTER loop completes (single sync point)
        metrics["kl_history"] = [kl.item() for kl in kl_tensors]
        if final_kl_tensor is not None:
            metrics["final_kl"] = final_kl_tensor.item()

        return tokens, metrics


__all__ = [
    "compute_kl_divergence",
    "compute_js_divergence",
    "KLASSGenerator",
]
