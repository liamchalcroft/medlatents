"""DPO for discrete token models (MaskGIT, Autoregressive).

This module implements DPO trainers for discrete generative models that
directly output token probabilities:

1. AutoregressiveDPOTrainer: Standard causal log-prob computation
2. MaskGITDPOTrainer: Pseudo-likelihood via masked prediction

The key difference from language model DPO is that we're working with
image/volume latent tokens, which may have different sequence structure.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from ..configs import PostTrainingConfig
from .base import BaseDPOTrainer


class AutoregressiveDPOTrainer(BaseDPOTrainer):
    """DPO trainer for autoregressive discrete models.

    For autoregressive models, the log-probability is computed as:
        log p(x) = sum_{i=1}^L log p(x_i | x_{<i})

    This is the standard approach used in language model DPO.

    Example:
        >>> from medlatents.autoregressive import AutoregressiveTransformer
        >>> model = AutoregressiveTransformer(vocab_size=1024, ...)
        >>> ref_model = copy.deepcopy(model)
        >>> trainer = AutoregressiveDPOTrainer(model, ref_model, config)
        >>> trainer.train(preference_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
    ) -> None:
        """Initialize autoregressive DPO trainer.

        Args:
            model: Autoregressive model to train
            ref_model: Frozen reference model
            config: Training configuration
            accelerator: Optional pre-configured accelerator
            vocab_size: Vocabulary size (inferred from model if not provided)
        """
        super().__init__(model, ref_model, config, accelerator)

        # Get vocab size from model if not provided
        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            elif hasattr(model, "config") and hasattr(model.config, "vocab_size"):
                vocab_size = model.config.vocab_size
            else:
                raise ValueError(
                    "vocab_size must be provided or model must have vocab_size attribute"
                )

        self.vocab_size = vocab_size

    def compute_logprobs(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute autoregressive log-probabilities.

        For a sequence x = [x_1, x_2, ..., x_L], computes:
            log p(x) = sum_{i=1}^{L-1} log p(x_{i+1} | x_1, ..., x_i)

        Args:
            model: Autoregressive model
            samples: Token sequences [batch, seq_len]
            **kwargs: Additional model arguments

        Returns:
            Log-probabilities [batch]
        """
        batch_size, seq_len = samples.shape

        # Get logits for all positions (teacher-forced)
        # Model input: tokens[:-1], target: tokens[1:]
        inputs = samples[:, :-1]  # [batch, seq_len-1]
        targets = samples[:, 1:]  # [batch, seq_len-1]

        # Forward pass
        logits = model(inputs, **kwargs)  # [batch, seq_len-1, vocab_size]

        # Compute per-token log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len-1, vocab_size]

        # Gather log probs for actual tokens
        # Shape: [batch, seq_len-1]
        token_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Sum over sequence for total log probability
        total_log_probs = token_log_probs.sum(dim=-1)  # [batch]

        return total_log_probs


class MaskGITDPOTrainer(BaseDPOTrainer):
    """DPO trainer for MaskGIT bidirectional models.

    For bidirectional models, we use pseudo-likelihood:
        log p(x) ≈ sum_{i=1}^L log p(x_i | x_{\\i})

    where x_{\\i} denotes all tokens except position i.

    This is computed efficiently by masking each position and predicting it
    from the context. We can use different strategies:

    1. "full": Mask each position independently (L forward passes, expensive)
    2. "random": Random subset of positions (cheaper approximation)
    3. "parallel": Single forward with random masking (training-style)

    Example:
        >>> from medlatents.maskgit import MaskGIT
        >>> model = MaskGIT(vocab_size=1024, ...)
        >>> ref_model = copy.deepcopy(model)
        >>> trainer = MaskGITDPOTrainer(model, ref_model, config)
        >>> trainer.train(preference_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        likelihood_strategy: Literal["full", "random", "parallel"] = "parallel",
        num_mask_samples: int = 16,
    ) -> None:
        """Initialize MaskGIT DPO trainer.

        Args:
            model: MaskGIT model to train
            ref_model: Frozen reference model
            config: Training configuration
            accelerator: Optional pre-configured accelerator
            vocab_size: Vocabulary size (inferred from model if not provided)
            likelihood_strategy: How to compute pseudo-likelihood
            num_mask_samples: Number of positions to sample for "random" strategy
        """
        super().__init__(model, ref_model, config, accelerator)

        # Get vocab size from model
        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            else:
                raise ValueError("vocab_size must be provided")

        self.vocab_size = vocab_size
        self.likelihood_strategy = likelihood_strategy
        self.num_mask_samples = num_mask_samples

        # Get mask token from model
        if hasattr(model, "mask_token"):
            self.mask_token = model.mask_token
        elif hasattr(model, "special_tokens") and model.special_tokens is not None:
            self.mask_token = model.special_tokens.mask
        else:
            # Default: assume mask token is vocab_size (common convention)
            self.mask_token = vocab_size

    def compute_logprobs(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute pseudo-likelihood for MaskGIT.

        Uses the specified strategy to estimate log p(x).

        Args:
            model: MaskGIT model
            samples: Token sequences [batch, seq_len]
            **kwargs: Additional model arguments

        Returns:
            Log-probabilities [batch]
        """
        if self.likelihood_strategy == "full":
            return self._compute_full_pseudolikelihood(model, samples, **kwargs)
        elif self.likelihood_strategy == "random":
            return self._compute_random_pseudolikelihood(model, samples, **kwargs)
        else:  # parallel
            return self._compute_parallel_pseudolikelihood(model, samples, **kwargs)

    def _compute_full_pseudolikelihood(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute exact pseudo-likelihood (expensive).

        Masks each position independently and sums log probabilities.
        Requires L forward passes.
        """
        batch_size, seq_len = samples.shape
        device = samples.device

        total_log_probs = torch.zeros(batch_size, device=device)

        for pos in range(seq_len):
            # Create mask for position i
            masked = samples.clone()
            masked[:, pos] = self.mask_token

            # Create position mask
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            mask[:, pos] = True

            # Forward pass with mask
            logits = model(masked, mask=mask, **kwargs)

            # Get log prob at masked position
            log_probs = F.log_softmax(logits[:, pos], dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=samples[:, pos : pos + 1]
            ).squeeze(-1)

            total_log_probs += token_log_probs

        return total_log_probs

    def _compute_random_pseudolikelihood(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute pseudo-likelihood from random position subset.

        Samples K random positions and scales up the estimate.
        """
        batch_size, seq_len = samples.shape
        device = samples.device
        k = min(self.num_mask_samples, seq_len)

        total_log_probs = torch.zeros(batch_size, device=device)

        # Sample random positions
        positions = torch.randperm(seq_len, device=device)[:k]

        for pos in positions:
            pos = pos.item()
            masked = samples.clone()
            masked[:, pos] = self.mask_token

            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            mask[:, pos] = True

            logits = model(masked, mask=mask, **kwargs)
            log_probs = F.log_softmax(logits[:, pos], dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=samples[:, pos : pos + 1]
            ).squeeze(-1)

            total_log_probs += token_log_probs

        # Scale to full sequence estimate
        scale_factor = seq_len / k
        return total_log_probs * scale_factor

    def _compute_parallel_pseudolikelihood(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute pseudo-likelihood with parallel masking.

        Uses random masking similar to training, which allows batch
        processing but provides a noisier estimate.
        """
        batch_size, seq_len = samples.shape
        device = samples.device

        # Use a fixed mask ratio for consistency
        mask_ratio = 0.15

        # Create random mask
        mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
        # Ensure at least one token is masked per sample
        if not mask.any(dim=1).all():
            # Force at least one mask per sample
            random_pos = torch.randint(0, seq_len, (batch_size,), device=device)
            mask[torch.arange(batch_size, device=device), random_pos] = True

        # Apply mask
        masked = samples.clone()
        masked[mask] = self.mask_token

        # Forward pass
        logits = model(masked, mask=mask, **kwargs)

        # Compute log probs at masked positions only
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens at all positions
        token_log_probs = torch.gather(log_probs, dim=-1, index=samples.unsqueeze(-1)).squeeze(-1)

        # Only count masked positions, but scale to full sequence
        masked_log_probs = token_log_probs * mask.float()
        num_masked = mask.float().sum(dim=1).clamp(min=1)

        # Scale: (sum of masked log probs) * (seq_len / num_masked)
        total_log_probs = masked_log_probs.sum(dim=1) * (seq_len / num_masked)

        return total_log_probs


class DiscreteDPOTrainer(BaseDPOTrainer):
    """Unified DPO trainer that auto-detects discrete model type.

    This is a convenience class that automatically selects the appropriate
    training strategy based on the model type.

    Example:
        >>> # Works with any discrete model
        >>> trainer = DiscreteDPOTrainer(model, ref_model, config)
        >>> trainer.train(dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        accelerator: Accelerator | None = None,
        model_type: Literal["autoreg", "maskgit"] | None = None,
        vocab_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize discrete DPO trainer.

        Args:
            model: Discrete model to train
            ref_model: Frozen reference model
            config: Training configuration
            accelerator: Optional pre-configured accelerator
            model_type: Model type ("autoreg" or "maskgit"). Auto-detected if None.
            vocab_size: Vocabulary size
            **kwargs: Additional arguments passed to specific trainer
        """
        # Auto-detect model type
        if model_type is None:
            model_type = self._detect_model_type(model)

        self.model_type = model_type

        # Create appropriate trainer
        if model_type == "autoreg":
            self._trainer = AutoregressiveDPOTrainer(
                model, ref_model, config, accelerator, vocab_size
            )
        elif model_type == "maskgit":
            self._trainer = MaskGITDPOTrainer(
                model, ref_model, config, accelerator, vocab_size, **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Copy attributes from internal trainer
        self.model = self._trainer.model
        self.ref_model = self._trainer.ref_model
        self.config = self._trainer.config
        self.accelerator = self._trainer.accelerator
        self.optimizer = self._trainer.optimizer
        self.ema = self._trainer.ema
        self.loss_fn = self._trainer.loss_fn
        self.reference_free = self._trainer.reference_free
        self.global_step = self._trainer.global_step
        self.best_accuracy = self._trainer.best_accuracy

    def _detect_model_type(self, model: nn.Module) -> Literal["autoreg", "maskgit"]:
        """Detect model type from class name or attributes."""
        class_name = model.__class__.__name__.lower()

        if "maskgit" in class_name or "mask" in class_name:
            return "maskgit"
        if "autoreg" in class_name or "gpt" in class_name or "causal" in class_name:
            return "autoreg"

        # Check for mask token (MaskGIT models have this)
        if hasattr(model, "mask_token") or (
            hasattr(model, "special_tokens")
            and model.special_tokens is not None
            and hasattr(model.special_tokens, "mask")
        ):
            return "maskgit"

        # Check for causal attribute
        if hasattr(model, "causal") and model.causal:
            return "autoreg"

        # Default to autoregressive
        return "autoreg"

    def compute_logprobs(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Delegate to internal trainer."""
        return self._trainer.compute_logprobs(model, samples, **kwargs)

    def train(self, *args, **kwargs):
        """Delegate training to internal trainer."""
        return self._trainer.train(*args, **kwargs)

    def validate(self, *args, **kwargs):
        """Delegate validation to internal trainer."""
        return self._trainer.validate(*args, **kwargs)

    def save_checkpoint(self, *args, **kwargs):
        """Delegate checkpoint saving to internal trainer."""
        return self._trainer.save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        """Delegate checkpoint loading to internal trainer."""
        return self._trainer.load_checkpoint(*args, **kwargs)


__all__ = [
    "AutoregressiveDPOTrainer",
    "MaskGITDPOTrainer",
    "DiscreteDPOTrainer",
]
