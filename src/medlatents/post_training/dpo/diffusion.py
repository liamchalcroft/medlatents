"""DPO for diffusion models (D3PM and continuous diffusion).

This module implements DPO for diffusion-based generative models. The key challenge
is that diffusion models don't have explicit likelihoods - instead we use:

1. ELBO-based likelihood: -ELBO as a surrogate for negative log-likelihood
2. Denoising score matching: Implicit likelihood from score function
3. Timestep-weighted: Weight contributions by timestep importance

Reference:
- "Diffusion Model Alignment Using Direct Preference Optimization" (Wallace et al., 2023)
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from ...diffusion.d3pm import D3PM
from ..configs import PostTrainingConfig
from ..dpo.base import BaseDPOTrainer


class D3PMDPOTrainer(BaseDPOTrainer):
    """DPO trainer for D3PM discrete diffusion models.

    For D3PM models, we compute log-probability using the ELBO:
        log p(x_0) ≥ -ELBO = -E_t[KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))]

    The ELBO provides a lower bound on the log-likelihood, which we can use
    for DPO optimization.

    We support two modes:
    1. Monte Carlo ELBO: Sample timesteps and average
    2. Full ELBO: Compute over all timesteps (expensive but exact)

    Example:
        >>> from medlatents.diffusion import D3PM
        >>> d3pm = D3PM(num_classes=1024, num_timesteps=1000)
        >>> model = DiscreteDiT(vocab_size=1024, ...)
        >>> ref_model = copy.deepcopy(model)
        >>> trainer = D3PMDPOTrainer(model, ref_model, config, d3pm=d3pm)
        >>> trainer.train(preference_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        d3pm: D3PM,
        accelerator: Accelerator | None = None,
        num_timestep_samples: int = 10,
        elbo_mode: Literal["monte_carlo", "full"] = "monte_carlo",
    ) -> None:
        """Initialize D3PM DPO trainer.

        Args:
            model: D3PM backbone model to train
            ref_model: Frozen reference model
            config: Training configuration
            d3pm: D3PM diffusion process
            accelerator: Optional pre-configured accelerator
            num_timestep_samples: Number of timesteps to sample for Monte Carlo ELBO
            elbo_mode: How to compute ELBO ("monte_carlo" or "full")
        """
        super().__init__(model, ref_model, config, accelerator)

        self.d3pm = d3pm
        self.num_timestep_samples = num_timestep_samples
        self.elbo_mode = elbo_mode

        # Move D3PM tensors to device
        self.d3pm.device = self.accelerator.device
        self._move_d3pm_to_device()

    def _move_d3pm_to_device(self) -> None:
        """Move D3PM transition matrices to the accelerator device."""
        device = self.accelerator.device
        self.d3pm.betas = self.d3pm.betas.to(device)
        self.d3pm.alphas = self.d3pm.alphas.to(device)
        self.d3pm.alphas_cumprod = self.d3pm.alphas_cumprod.to(device)
        # Absorbing-state D3PM uses a matrix-free path where the transition
        # matrices are None; only move them to device when they exist.
        if self.d3pm.Q_t is not None:
            self.d3pm.Q_t = self.d3pm.Q_t.to(device)
        if self.d3pm.Q_bar_t is not None:
            self.d3pm.Q_bar_t = self.d3pm.Q_bar_t.to(device)
        if self.d3pm.Q_bar_t_minus_1 is not None:
            self.d3pm.Q_bar_t_minus_1 = self.d3pm.Q_bar_t_minus_1.to(device)

    def compute_logprobs(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute ELBO-based log-probability for D3PM.

        The ELBO is computed as:
            ELBO = E_t[KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))]

        We return -ELBO as a surrogate for log p(x).

        Args:
            model: D3PM backbone model
            samples: Clean token sequences [batch, seq_len]
            **kwargs: Additional model arguments

        Returns:
            Log-probabilities (negative ELBO) [batch]
        """
        if self.elbo_mode == "full":
            return self._compute_full_elbo(model, samples, **kwargs)
        else:
            return self._compute_monte_carlo_elbo(model, samples, **kwargs)

    def _compute_monte_carlo_elbo(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute ELBO via Monte Carlo sampling over timesteps."""
        batch_size = samples.shape[0]
        device = samples.device
        num_timesteps = self.d3pm.num_timesteps

        total_elbo = torch.zeros(batch_size, device=device)

        for _ in range(self.num_timestep_samples):
            # Sample random timestep for each sample
            t = torch.randint(1, num_timesteps, (batch_size,), device=device)

            # Compute single-timestep ELBO contribution
            elbo_t = self._compute_elbo_at_timestep(model, samples, t, **kwargs)
            total_elbo += elbo_t

        # Average over samples and scale by number of timesteps
        total_elbo = (total_elbo / self.num_timestep_samples) * num_timesteps

        # Return negative ELBO (higher is better log-prob)
        return -total_elbo

    def _compute_full_elbo(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute full ELBO over all timesteps (expensive)."""
        batch_size = samples.shape[0]
        device = samples.device
        num_timesteps = self.d3pm.num_timesteps

        total_elbo = torch.zeros(batch_size, device=device)

        for t_val in range(1, num_timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            elbo_t = self._compute_elbo_at_timestep(model, samples, t, **kwargs)
            total_elbo += elbo_t

        return -total_elbo

    def _compute_elbo_at_timestep(
        self,
        model: nn.Module,
        x_0: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute ELBO contribution at a single timestep.

        ELBO_t = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))

        Args:
            model: D3PM backbone model
            x_0: Clean data [batch, seq_len]
            t: Timesteps [batch]
            **kwargs: Additional model arguments

        Returns:
            ELBO contribution at timestep t [batch]
        """
        # Sample x_t from forward process
        x_t = self.d3pm.q_sample(x_0, t)

        # Get true posterior q(x_{t-1} | x_t, x_0)
        q_posterior = self.d3pm.q_posterior(x_0, x_t, t)  # [batch, seq, vocab]

        # Get model prediction p_θ(x_0 | x_t)
        logits = model(x=x_t, t=t, **kwargs)  # [batch, seq, vocab]
        log_probs = F.log_softmax(logits, dim=-1)

        # KL divergence: KL(q || p) = sum_x q(x) * (log q(x) - log p(x))
        # For numerical stability, add small epsilon to q_posterior
        eps = 1e-10
        q_posterior_safe = q_posterior.clamp(min=eps)

        # Per-position KL
        kl_per_position = (q_posterior_safe * (q_posterior_safe.log() - log_probs)).sum(
            dim=-1
        )  # [batch, seq]

        # Sum over sequence
        kl_total = kl_per_position.sum(dim=-1)  # [batch]

        return kl_total


class DiffusionDPOTrainer(BaseDPOTrainer):
    """Unified DPO trainer for diffusion models.

    This is a convenience class that handles both D3PM discrete diffusion
    and could be extended for continuous diffusion.

    For now, it primarily wraps D3PM but provides a cleaner interface.

    Example:
        >>> trainer = DiffusionDPOTrainer(
        ...     model=backbone_model,
        ...     ref_model=ref_backbone,
        ...     config=config,
        ...     diffusion=d3pm,
        ... )
        >>> trainer.train(preference_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        diffusion: D3PM,
        accelerator: Accelerator | None = None,
        num_timestep_samples: int = 10,
        elbo_mode: Literal["monte_carlo", "full"] = "monte_carlo",
    ) -> None:
        """Initialize diffusion DPO trainer.

        Args:
            model: Diffusion backbone model to train
            ref_model: Frozen reference model
            config: Training configuration
            diffusion: D3PM diffusion process
            accelerator: Optional pre-configured accelerator
            num_timestep_samples: Number of timesteps for Monte Carlo
            elbo_mode: ELBO computation mode
        """
        # Create internal D3PM trainer
        self._trainer = D3PMDPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config,
            d3pm=diffusion,
            accelerator=accelerator,
            num_timestep_samples=num_timestep_samples,
            elbo_mode=elbo_mode,
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
        self.d3pm = self._trainer.d3pm

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
    "D3PMDPOTrainer",
    "DiffusionDPOTrainer",
]
