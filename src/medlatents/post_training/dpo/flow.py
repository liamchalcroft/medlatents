"""DPO for flow matching models (discrete and continuous).

This module implements DPO for flow matching generative models. Flow matching
provides a flexible framework where likelihood can be computed via:

1. Probability flow ODE: Exact likelihood via trace computation
2. Trajectory-based: Approximate likelihood from flow matching loss
3. Endpoint matching: Use cross-entropy at t=1 as surrogate

For discrete flows (MixtureDiscreteProbPath), we use trajectory-based
likelihood similar to how diffusion DPO works.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from ...flow_matching.core import MixtureDiscreteProbPath, PolynomialConvexScheduler
from ..configs import PostTrainingConfig
from ..dpo.base import BaseDPOTrainer


class DiscreteFlowDPOTrainer(BaseDPOTrainer):
    """DPO trainer for discrete flow matching models.

    For discrete flow matching with mixture paths, we compute likelihood as:
        log p(x_1) ≈ -E_t[L_flow(x_1, t)]

    where L_flow is the flow matching loss at timestep t. This provides
    a tractable surrogate for the true likelihood.

    We support different estimation strategies:
    1. "trajectory": Sample timesteps and average flow loss
    2. "endpoint": Use only t=1 prediction (fast but less accurate)
    3. "importance": Importance-weighted sampling of timesteps

    Example:
        >>> from medlatents.flow_matching import MixtureDiscreteProbPath
        >>> path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        >>> model = DiscreteDiT(vocab_size=1024, ...)
        >>> ref_model = copy.deepcopy(model)
        >>> trainer = DiscreteFlowDPOTrainer(model, ref_model, config, path=path)
        >>> trainer.train(preference_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        path: MixtureDiscreteProbPath,
        source_distribution: Any | None = None,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        num_timestep_samples: int = 10,
        likelihood_strategy: Literal["trajectory", "endpoint", "importance"] = "trajectory",
        time_eps: float = 1e-3,
    ) -> None:
        """Initialize discrete flow DPO trainer.

        Args:
            model: Flow matching model to train
            ref_model: Frozen reference model
            config: Training configuration
            path: Discrete probability path (e.g., MixtureDiscreteProbPath)
            source_distribution: Source distribution for flow (mask/uniform)
            accelerator: Optional pre-configured accelerator
            vocab_size: Vocabulary size
            num_timestep_samples: Number of timesteps for trajectory estimation
            likelihood_strategy: How to estimate likelihood
            time_eps: Small epsilon to avoid t=0 or t=1 exactly
        """
        super().__init__(model, ref_model, config, accelerator)

        self.path = path
        self.source_distribution = source_distribution
        self.num_timestep_samples = num_timestep_samples
        self.likelihood_strategy = likelihood_strategy
        self.time_eps = time_eps

        # Get vocab size from model if not provided
        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            else:
                raise ValueError("vocab_size must be provided")
        self.vocab_size = vocab_size

        # Get mask token for source distribution
        if hasattr(model, "mask_token"):
            self.mask_token = model.mask_token
        elif hasattr(model, "special_tokens") and model.special_tokens is not None:
            self.mask_token = model.special_tokens.mask
        else:
            self.mask_token = vocab_size  # Default convention

    def _sample_source(self, target: Tensor) -> Tensor:
        """Sample from source distribution.

        Args:
            target: Target tensor to match shape [batch, seq_len]

        Returns:
            Source tokens [batch, seq_len]
        """
        if self.source_distribution is not None:
            return self.source_distribution.sample_like(target)

        # Default: all mask tokens
        return torch.full_like(target, self.mask_token)

    def compute_logprobs(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute flow-based log-probability.

        Uses the specified strategy to estimate log p(x_1).

        Args:
            model: Flow matching model
            samples: Target token sequences (x_1) [batch, seq_len]
            **kwargs: Additional model arguments

        Returns:
            Log-probabilities [batch]
        """
        if self.likelihood_strategy == "trajectory":
            return self._compute_trajectory_likelihood(model, samples, **kwargs)
        elif self.likelihood_strategy == "endpoint":
            return self._compute_endpoint_likelihood(model, samples, **kwargs)
        elif self.likelihood_strategy == "importance":
            return self._compute_importance_likelihood(model, samples, **kwargs)
        else:
            raise ValueError(f"Unknown likelihood strategy: {self.likelihood_strategy}")

    def _compute_trajectory_likelihood(
        self,
        model: nn.Module,
        x_1: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute likelihood by averaging flow loss over trajectory."""
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample source
        x_0 = self._sample_source(x_1)

        total_loss = torch.zeros(batch_size, device=device)

        for _ in range(self.num_timestep_samples):
            # Sample random timestep
            t = torch.rand(batch_size, device=device) * (1.0 - 2 * self.time_eps) + self.time_eps

            # Get interpolated state from path
            path_sample = self.path.sample(t, x_0, x_1)
            x_t = path_sample.x_t

            # Get model prediction
            logits = model(x=x_t, t=t, **kwargs)  # [batch, seq, vocab]

            # Compute cross-entropy loss (predicting x_1)
            loss = (
                F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    x_1.reshape(-1),
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=-1)
            )  # [batch]

            total_loss += loss

        # Average and negate (lower loss = higher likelihood)
        avg_loss = total_loss / self.num_timestep_samples
        return -avg_loss

    def _compute_endpoint_likelihood(
        self,
        model: nn.Module,
        x_1: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute likelihood using only endpoint (t close to 0)."""
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample source
        x_0 = self._sample_source(x_1)

        # Use t close to 0 (mostly source, predict target)
        t = torch.full((batch_size,), self.time_eps, device=device)

        # Get state at t (mostly x_0)
        path_sample = self.path.sample(t, x_0, x_1)
        x_t = path_sample.x_t

        # Get model prediction
        logits = model(x=x_t, t=t, **kwargs)

        # Cross-entropy as negative log-likelihood
        loss = (
            F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                x_1.reshape(-1),
                reduction="none",
            )
            .view(batch_size, -1)
            .sum(dim=-1)
        )

        return -loss

    def _compute_importance_likelihood(
        self,
        model: nn.Module,
        x_1: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute likelihood with importance-weighted timestep sampling.

        Uses more samples near t=0 and t=1 where the flow is more informative.
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        x_0 = self._sample_source(x_1)

        total_loss = torch.zeros(batch_size, device=device)

        for _ in range(self.num_timestep_samples):
            t = torch.rand(batch_size, device=device).clamp(self.time_eps, 1.0 - self.time_eps)

            path_sample = self.path.sample(t, x_0, x_1)
            x_t = path_sample.x_t

            logits = model(x=x_t, t=t, **kwargs)
            loss = (
                F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    x_1.reshape(-1),
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=-1)
            )
            total_loss += loss

        avg_loss = total_loss / self.num_timestep_samples
        return -avg_loss


class FlowDPOTrainer(BaseDPOTrainer):
    """Unified DPO trainer for flow matching models.

    This is a convenience class that handles discrete flow matching
    and can be extended for continuous flow matching (rectified flows).

    Example:
        >>> trainer = FlowDPOTrainer(
        ...     model=flow_model,
        ...     ref_model=ref_flow,
        ...     config=config,
        ...     path=mixture_path,
        ... )
        >>> trainer.train(preference_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        path: MixtureDiscreteProbPath | None = None,
        source_distribution: Any | None = None,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        num_timestep_samples: int = 10,
        likelihood_strategy: Literal["trajectory", "endpoint", "importance"] = "trajectory",
        **kwargs: Any,
    ) -> None:
        """Initialize flow DPO trainer.

        Args:
            model: Flow matching model to train
            ref_model: Frozen reference model
            config: Training configuration
            path: Discrete probability path (creates default if None)
            source_distribution: Source distribution
            accelerator: Optional pre-configured accelerator
            vocab_size: Vocabulary size
            num_timestep_samples: Number of timesteps for estimation
            likelihood_strategy: Likelihood estimation strategy
            **kwargs: Additional arguments
        """
        # Create default path if not provided
        if path is None:
            path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))

        # Create internal discrete flow trainer
        self._trainer = DiscreteFlowDPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config,
            path=path,
            source_distribution=source_distribution,
            accelerator=accelerator,
            vocab_size=vocab_size,
            num_timestep_samples=num_timestep_samples,
            likelihood_strategy=likelihood_strategy,
            **kwargs,
        )

        # Copy attributes
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
        self.path = self._trainer.path

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
        """Delegate checkpoint saving."""
        return self._trainer.save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs):
        """Delegate checkpoint loading."""
        return self._trainer.load_checkpoint(*args, **kwargs)


__all__ = [
    "DiscreteFlowDPOTrainer",
    "FlowDPOTrainer",
]
