"""Base DPO trainer and loss functions.

This module provides the foundation for Direct Preference Optimization across
all model types in medlatents. The key insight of DPO is that we can optimize
directly for preferences without training a separate reward model, by treating
the policy's log-probability ratio as an implicit reward.

References:
- DPO: "Direct Preference Optimization" (Rafailov et al., 2023)
- IPO: "A General Theoretical Paradigm" (Azar et al., 2023)
- KTO: "Kahneman-Tversky Optimization" (Ethayarajh et al., 2024)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..configs import PostTrainingConfig
from ..preference.data import PreferenceDataset, collate_preference_pairs


@dataclass
class DPOOutput:
    """Output from DPO loss computation.

    Attributes:
        loss: The total DPO loss
        chosen_rewards: Implicit rewards for chosen samples (log-prob ratio)
        rejected_rewards: Implicit rewards for rejected samples
        accuracy: Fraction where chosen_reward > rejected_reward
        reward_margin: Mean (chosen_reward - rejected_reward)
    """

    loss: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    accuracy: Tensor
    reward_margin: Tensor

    def to_dict(self) -> dict[str, float]:
        """Convert to logging dictionary."""
        return {
            "loss": self.loss.item(),
            "accuracy": self.accuracy.item(),
            "reward_margin": self.reward_margin.item(),
            "chosen_rewards": self.chosen_rewards.mean().item(),
            "rejected_rewards": self.rejected_rewards.mean().item(),
        }


class DPOLoss(nn.Module):
    """Flexible DPO loss with multiple variants.

    Supports:
    - sigmoid: Standard DPO with sigmoid loss (Rafailov et al.)
    - hinge: Margin-based loss for more aggressive optimization
    - ipo: Identity Preference Optimization (no sigmoid)
    - kto: Kahneman-Tversky Optimization (asymmetric)
    - disco: Distributionally robust DPO

    The DPO objective is:
        L = -log(sigmoid(β * (r_w - r_l)))

    where r_w and r_l are implicit rewards defined as:
        r(x) = β * log(π_θ(x) / π_ref(x))

    Example:
        >>> loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")
        >>> output = loss_fn(
        ...     chosen_logprobs=chosen_lp,
        ...     rejected_logprobs=rejected_lp,
        ...     ref_chosen_logprobs=ref_chosen_lp,
        ...     ref_rejected_logprobs=ref_rejected_lp,
        ... )
        >>> loss = output.loss
    """

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto", "disco"] = "sigmoid",
        label_smoothing: float = 0.0,
        reference_free: bool = False,
        kto_desirable_weight: float = 1.0,
        kto_undesirable_weight: float = 1.0,
    ) -> None:
        """Initialize DPO loss.

        Args:
            beta: KL penalty coefficient (controls deviation from reference)
            loss_type: Type of preference loss to use
            label_smoothing: Soft labels for robustness
            reference_free: If True, don't use reference model
            kto_desirable_weight: Weight for KTO desirable term
            kto_undesirable_weight: Weight for KTO undesirable term
        """
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free
        self.kto_desirable_weight = kto_desirable_weight
        self.kto_undesirable_weight = kto_undesirable_weight

    def forward(
        self,
        chosen_logprobs: Tensor,
        rejected_logprobs: Tensor,
        ref_chosen_logprobs: Tensor | None = None,
        ref_rejected_logprobs: Tensor | None = None,
        margin: Tensor | None = None,
    ) -> DPOOutput:
        """Compute DPO loss.

        Args:
            chosen_logprobs: Log probs of policy on chosen [batch]
            rejected_logprobs: Log probs of policy on rejected [batch]
            ref_chosen_logprobs: Log probs of reference on chosen [batch]
            ref_rejected_logprobs: Log probs of reference on rejected [batch]
            margin: Optional preference margin [batch]

        Returns:
            DPOOutput with loss and metrics
        """
        # Compute log-probability ratios (implicit rewards)
        if self.reference_free:
            chosen_rewards = self.beta * chosen_logprobs
            rejected_rewards = self.beta * rejected_logprobs
        else:
            if ref_chosen_logprobs is None or ref_rejected_logprobs is None:
                raise ValueError("Reference log probs required when reference_free=False")
            chosen_rewards = self.beta * (chosen_logprobs - ref_chosen_logprobs)
            rejected_rewards = self.beta * (rejected_logprobs - ref_rejected_logprobs)

        # Compute reward difference
        reward_diff = chosen_rewards - rejected_rewards

        # Apply margin if provided
        if margin is not None:
            reward_diff = margin * reward_diff

        # Compute loss based on type
        if self.loss_type == "sigmoid":
            # Standard DPO: -log(sigmoid(r_w - r_l))
            if self.label_smoothing > 0:
                # Soft labels: (1 - ε) * target + ε * (1 - target)
                targets = (1 - self.label_smoothing) * torch.ones_like(reward_diff)
                loss = F.binary_cross_entropy_with_logits(reward_diff, targets, reduction="none")
            else:
                loss = -F.logsigmoid(reward_diff)

        elif self.loss_type == "hinge":
            # Hinge loss: max(0, 1 - (r_w - r_l))
            loss = F.relu(1.0 - reward_diff)

        elif self.loss_type == "ipo":
            # IPO: (r_w - r_l - 1/(2β))^2
            # Avoids sigmoid saturation
            target = 0.5 / self.beta
            loss = (reward_diff - target).pow(2)

        elif self.loss_type == "kto":
            # KTO: Asymmetric loss based on prospect theory
            # Losses loom larger than gains
            # Reference point: use batch average as baseline
            ref_point = 0.5 * (chosen_rewards + rejected_rewards).mean()

            # Value function: v(x) = x if x >= ref_point, else λ * x
            chosen_value = torch.where(
                chosen_rewards >= ref_point,
                chosen_rewards - ref_point,
                2.0 * (chosen_rewards - ref_point),  # Loss aversion λ=2
            )
            rejected_value = torch.where(
                rejected_rewards >= ref_point,
                rejected_rewards - ref_point,
                2.0 * (rejected_rewards - ref_point),
            )

            # Weighted loss
            loss = self.kto_desirable_weight * -F.logsigmoid(
                chosen_value
            ) + self.kto_undesirable_weight * -F.logsigmoid(-rejected_value)

        elif self.loss_type == "disco":
            # Distributionally robust DPO
            # Uses a worst-case bound over noise distributions
            # L = log(1 + exp(-β * (r_w - r_l)))
            loss = torch.log1p(torch.exp(-reward_diff))

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute metrics
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()

        return DPOOutput(
            loss=loss.mean(),
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            accuracy=accuracy,
            reward_margin=reward_margin,
        )


class BaseDPOTrainer(ABC):
    """Abstract base class for DPO trainers.

    Provides common infrastructure for all model types:
    - Accelerator setup with FSDP support
    - EMA for stable training
    - Logging and checkpointing
    - Reference model management

    Subclasses must implement:
    - compute_logprobs: Model-specific log-probability computation
    - prepare_batch: Model-specific batch preprocessing

    Example:
        >>> class MyDPOTrainer(BaseDPOTrainer):
        ...     def compute_logprobs(self, model, batch):
        ...         # Model-specific implementation
        ...         return log_probs
        ...
        >>> trainer = MyDPOTrainer(model, ref_model, config)
        >>> trainer.train(dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module | None,
        config: PostTrainingConfig,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Initialize DPO trainer.

        Args:
            model: Model to train
            ref_model: Reference model (frozen). If None, uses reference_free mode.
            config: Training configuration
            accelerator: Optional pre-configured accelerator
        """
        self.model = model
        self.ref_model = ref_model
        self.config = config

        # Determine reference-free mode
        self.reference_free = ref_model is None or config.reference_free

        # Setup accelerator
        if accelerator is None:
            fsdp_plugin = None
            if config.use_fsdp:
                from accelerate import FullyShardedDataParallelPlugin
                from torch.distributed.fsdp.fully_sharded_data_parallel import (
                    FullOptimStateDictConfig,
                    FullStateDictConfig,
                )

                fsdp_plugin = FullyShardedDataParallelPlugin(
                    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                    optim_state_dict_config=FullOptimStateDictConfig(
                        offload_to_cpu=True, rank0_only=False
                    ),
                )

            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                log_with="wandb" if config.wandb_project else None,
                mixed_precision=config.mixed_precision,
                fsdp_plugin=fsdp_plugin,
            )
        else:
            self.accelerator = accelerator

        # Set seed
        if config.seed is not None:
            set_seed(config.seed)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Setup EMA
        self.ema = None
        if config.use_ema:
            self.ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)

        # Setup loss function
        self.loss_fn = DPOLoss(
            beta=config.beta,
            loss_type=config.loss_type,
            label_smoothing=config.label_smoothing,
            reference_free=self.reference_free,
        )

        # Freeze reference model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # Prepare with accelerator
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # Move EMA to device
        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0

    @abstractmethod
    def compute_logprobs(
        self,
        model: nn.Module,
        samples: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute log-probabilities for samples.

        This is model-type specific:
        - Autoregressive: Sum of log P(x_i | x_{<i})
        - MaskGIT: Pseudo-likelihood via masked prediction
        - Diffusion: Negative ELBO (Monte Carlo estimate)
        - Flow: Trajectory-based likelihood

        Args:
            model: Model to evaluate
            samples: Token sequences [batch, seq_len]
            **kwargs: Model-specific arguments

        Returns:
            Log-probabilities [batch]
        """
        raise NotImplementedError

    def prepare_batch(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """Prepare batch for DPO training.

        Args:
            batch: Collated batch from DataLoader

        Returns:
            Tuple of (chosen, rejected, margin, extra_kwargs)
        """
        chosen = batch["chosen"].to(self.accelerator.device)
        rejected = batch["rejected"].to(self.accelerator.device)
        margin = batch["margin"].to(self.accelerator.device)
        return chosen, rejected, margin, {}

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Collated preference batch

        Returns:
            Dictionary of metrics
        """
        chosen, rejected, margin, extra_kwargs = self.prepare_batch(batch)

        # Compute policy log-probs
        chosen_logprobs = self.compute_logprobs(self.model, chosen, **extra_kwargs)
        rejected_logprobs = self.compute_logprobs(self.model, rejected, **extra_kwargs)

        # Compute reference log-probs
        ref_chosen_logprobs = None
        ref_rejected_logprobs = None
        if not self.reference_free:
            with torch.no_grad():
                ref_chosen_logprobs = self.compute_logprobs(self.ref_model, chosen, **extra_kwargs)
                ref_rejected_logprobs = self.compute_logprobs(
                    self.ref_model, rejected, **extra_kwargs
                )

        # Compute DPO loss
        output = self.loss_fn(
            chosen_logprobs=chosen_logprobs,
            rejected_logprobs=rejected_logprobs,
            ref_chosen_logprobs=ref_chosen_logprobs,
            ref_rejected_logprobs=ref_rejected_logprobs,
            margin=margin,
        )

        # Backward pass
        self.accelerator.backward(output.loss)

        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        return output.to_dict()

    def train(
        self,
        dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
    ) -> dict[str, list[float]]:
        """Run full training loop.

        Args:
            dataset: Training preference dataset
            val_dataset: Optional validation dataset

        Returns:
            Dictionary of training history
        """
        # Create dataloaders
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_preference_pairs,
            drop_last=True,
        )
        train_loader = self.accelerator.prepare(train_loader)

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_preference_pairs,
            )
            val_loader = self.accelerator.prepare(val_loader)

        # Initialize wandb if configured
        if self.config.wandb_project:
            self.accelerator.init_trackers(
                project_name=self.config.wandb_project,
                config=self.config.to_dict(),
                init_kwargs={
                    "wandb": {
                        "entity": self.config.wandb_entity,
                        "name": self.config.run_name,
                    }
                },
            )

        # Create log directory
        os.makedirs(self.config.logdir, exist_ok=True)

        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # Learning rate schedule
        total_steps = self.config.max_steps
        warmup_steps = self.config.warmup_steps

        # Training loop
        self.model.train()
        pbar = tqdm(total=total_steps, desc="DPO Training")

        while self.global_step < total_steps:
            for batch in train_loader:
                if self.global_step >= total_steps:
                    break

                # Learning rate warmup
                if self.global_step < warmup_steps:
                    lr_scale = (self.global_step + 1) / warmup_steps
                else:
                    # Cosine decay
                    progress = (self.global_step - warmup_steps) / (total_steps - warmup_steps)
                    lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                    lr_scale = lr_scale.item()

                lr = self.config.lr * lr_scale
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                # Training step
                with self.accelerator.accumulate(self.model):
                    metrics = self.train_step(batch)

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every == 0:
                    metrics["lr"] = lr
                    if self.accelerator.is_main_process:
                        self.accelerator.log(metrics, step=self.global_step)
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        acc=f"{metrics['accuracy']:.4f}",
                    )
                    history["train_loss"].append(metrics["loss"])
                    history["train_accuracy"].append(metrics["accuracy"])

                # Validation
                if val_loader and self.global_step % self.config.eval_every == 0:
                    val_metrics = self.validate(val_loader)
                    history["val_accuracy"].append(val_metrics["accuracy"])

                    if val_metrics["accuracy"] > self.best_accuracy:
                        self.best_accuracy = val_metrics["accuracy"]
                        self.save_checkpoint("best")

                # Checkpointing
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint("latest")

                pbar.update(1)

        pbar.close()

        if self.config.wandb_project:
            self.accelerator.end_training()

        return history

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation.

        Args:
            val_loader: Validation dataloader

        Returns:
            Validation metrics
        """
        self.model.eval()

        # Use EMA weights if available
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        total_accuracy = 0.0
        total_margin = 0.0
        num_batches = 0

        for batch in val_loader:
            chosen, rejected, margin, extra_kwargs = self.prepare_batch(batch)

            chosen_logprobs = self.compute_logprobs(self.model, chosen, **extra_kwargs)
            rejected_logprobs = self.compute_logprobs(self.model, rejected, **extra_kwargs)

            # Compute rewards
            if self.reference_free:
                chosen_rewards = self.config.beta * chosen_logprobs
                rejected_rewards = self.config.beta * rejected_logprobs
            else:
                ref_chosen = self.compute_logprobs(self.ref_model, chosen, **extra_kwargs)
                ref_rejected = self.compute_logprobs(self.ref_model, rejected, **extra_kwargs)
                chosen_rewards = self.config.beta * (chosen_logprobs - ref_chosen)
                rejected_rewards = self.config.beta * (rejected_logprobs - ref_rejected)

            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin = (chosen_rewards - rejected_rewards).mean()

            total_accuracy += accuracy.item()
            total_margin += reward_margin.item()
            num_batches += 1

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        self.model.train()

        metrics = {
            "accuracy": total_accuracy / max(num_batches, 1),
            "reward_margin": total_margin / max(num_batches, 1),
        }

        if self.accelerator.is_main_process:
            self.accelerator.log(
                {f"val/{k}": v for k, v in metrics.items()},
                step=self.global_step,
            )

        return metrics

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save training checkpoint.

        Args:
            name: Checkpoint name (e.g., "latest", "best", "step_1000")
        """
        checkpoint_path = os.path.join(self.config.logdir, f"checkpoint_{name}.pt")

        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config.to_dict(),
            "best_accuracy": self.best_accuracy,
        }

        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

        self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        # save_checkpoint stores only state_dicts/ints/floats and config.to_dict()
        # (a primitives dict), so the checkpoint contains no arbitrary Python objects.
        checkpoint = torch.load(path, map_location=self.accelerator.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["step"]
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)

        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])


__all__ = [
    "DPOLoss",
    "DPOOutput",
    "BaseDPOTrainer",
]
