"""Compile-friendly kernels for MaskGIT sampling.

Provides pure functions that can be compiled with torch.compile(fullgraph=True).
These kernels extract the core sampling logic without Python-side state or breaks.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def sample_with_gumbel(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Gumbel-max sampling, pure function (compile-friendly).

    Args:
        logits: [batch, seq_len, vocab_size] unnormalized logits
        temperature: sampling temperature

    Returns:
        sampled: [batch, seq_len] sampled tokens
    """
    eps = 1e-6
    u = torch.rand_like(logits).clamp(eps, 1 - eps)
    gumbel = -torch.log(-torch.log(u))
    scaled_logits = logits / temperature
    perturbed = scaled_logits + gumbel
    return torch.argmax(perturbed, dim=-1)


def get_confidences(logits: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
    """Get confidence values for sampled tokens, pure function.

    Args:
        logits: [batch, seq_len, vocab_size] unnormalized logits
        sampled: [batch, seq_len] sampled token indices

    Returns:
        confidences: [batch, seq_len] confidence values
    """
    probs = F.softmax(logits, dim=-1)
    return torch.gather(probs, dim=-1, index=sampled.unsqueeze(-1)).squeeze(-1)


def step_normal_kernel(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_ratio: float,
    temperature: float,
    K_MAX: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal step kernel: capped topk for unmasking (compile-friendly).

    Pure function with no Python-side state, fixed-shape topk, no .item().

    Args:
        logits: [batch, seq_len, vocab_size] model output
        tokens: [batch, seq_len] current token sequence
        mask: [batch, seq_len] True for positions to update
        mask_ratio: fraction of tokens that should REMAIN masked
        temperature: sampling temperature
        K_MAX: fixed maximum k for topk (compile-friendly)

    Returns:
        tokens: [batch, seq_len] updated token sequence
        mask: [batch, seq_len] updated mask (fewer positions still masked)
    """
    batch_size, seq_len = tokens.shape

    # Sample with Gumbel noise
    sampled = sample_with_gumbel(logits, temperature)

    # Get confidences for masked positions
    confidences = get_confidences(logits, sampled)
    confidences = confidences.masked_fill(~mask, -float("inf"))

    # Compute number of tokens to unmask
    num_masked = mask.sum(dim=1)
    num_to_unmask = torch.ceil((1 - mask_ratio) * num_masked.float()).long()
    num_to_unmask = torch.clamp(num_to_unmask, min=1)

    # Clamp effective_k to K_MAX (all tensor ops, no sync)
    effective_k = torch.minimum(num_to_unmask, num_masked)
    effective_k = torch.clamp(effective_k, min=0, max=K_MAX)

    # Fixed-shape topk (compile-friendly)
    _, topk_indices = torch.topk(confidences, k=K_MAX, dim=-1, largest=True)

    # valid_k_mask filters to actual k per batch
    k_range = torch.arange(K_MAX, device=tokens.device)
    valid_k = k_range.unsqueeze(0) < effective_k.unsqueeze(1)

    update_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=tokens.device)
    update_mask.scatter_(1, topk_indices, valid_k)

    new_tokens = torch.where(update_mask, sampled, tokens)
    new_mask = mask & ~update_mask

    return new_tokens, new_mask


def step_final_kernel(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final step kernel: unmask all remaining positions (compile-friendly).

    Pure function with no topk, just direct unmasking.

    Args:
        logits: [batch, seq_len, vocab_size] model output
        tokens: [batch, seq_len] current token sequence
        mask: [batch, seq_len] True for positions to update
        temperature: sampling temperature

    Returns:
        tokens: [batch, seq_len] updated token sequence (all unmasked)
        mask: [batch, seq_len] all False (no masked positions)
    """
    sampled = sample_with_gumbel(logits, temperature)

    # Unmask all remaining masked positions
    new_tokens = torch.where(mask, sampled, tokens)
    new_mask = torch.zeros_like(mask)

    return new_tokens, new_mask


class CompileCache:
    """Cache for compiled step kernels.

    Maintains compiled versions of step_normal and step_final kernels.

    In strict mode (default), compilation failures raise errors to ensure
    torch.compile(fullgraph=True) invariants are maintained - no silent fallback.
    Set strict=False to allow fallback to uncompiled versions.
    """

    def __init__(self, use_compile: bool = True, strict: bool = True):
        """Initialize compile cache.

        Args:
            use_compile: If False, always return uncompiled kernels.
            strict: If True, raise on compile failure instead of fallback.
                   Default True to enforce fullgraph=True invariant.
        """
        self.use_compile = use_compile
        self.strict = strict
        self._compiled_step_normal: Callable | None = None
        self._compiled_step_final: Callable | None = None

    def get_step_normal(self) -> Callable:
        """Get compiled step_normal kernel.

        Raises:
            RuntimeError: If strict=True and compilation fails.
        """
        if self._compiled_step_normal is not None:
            return self._compiled_step_normal

        if not self.use_compile:
            return step_normal_kernel

        try:
            self._compiled_step_normal = torch.compile(
                step_normal_kernel,
                fullgraph=True,
                mode="reduce-overhead",
            )
            return self._compiled_step_normal
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"step_normal_kernel failed to compile with fullgraph=True: {e}\n"
                    "This indicates a graph break (e.g., .item() call). "
                    "Set strict=False to allow fallback to uncompiled version."
                ) from e
            return step_normal_kernel

    def get_step_final(self) -> Callable:
        """Get compiled step_final kernel.

        Raises:
            RuntimeError: If strict=True and compilation fails.
        """
        if self._compiled_step_final is not None:
            return self._compiled_step_final

        if not self.use_compile:
            return step_final_kernel

        try:
            self._compiled_step_final = torch.compile(
                step_final_kernel,
                fullgraph=True,
                mode="reduce-overhead",
            )
            return self._compiled_step_final
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"step_final_kernel failed to compile with fullgraph=True: {e}\n"
                    "This indicates a graph break (e.g., .item() call). "
                    "Set strict=False to allow fallback to uncompiled version."
                ) from e
            return step_final_kernel


__all__ = [
    "sample_with_gumbel",
    "get_confidences",
    "step_normal_kernel",
    "step_final_kernel",
    "CompileCache",
]
