"""Consistency Model Training for fast sampling.

Consistency models learn to map any point on a diffusion trajectory directly
to the clean data, enabling single-step or few-step generation. This module
implements consistency training and consistency distillation.

Key concepts:
1. Consistency function: f(x_t, t) → x_0 for any t
2. Self-consistency: f(x_t, t) = f(x_{t'}, t') for t, t' on same trajectory
3. Boundary condition: f(x_0, 0) = x_0

References:
- "Consistency Models" (Song et al., 2023)
- "Improved Techniques for Consistency Training" (Song & Dhariwal, 2023)
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..configs import PostTrainingConfig


def pseudo_huber_loss(x: Tensor, y: Tensor, c: float = 0.00054) -> Tensor:
    """Pseudo-Huber loss for consistency training.

    More robust than MSE for high-dimensional data.
    L(x, y) = sqrt((x - y)^2 + c^2) - c

    Args:
        x: Predictions
        y: Targets
        c: Huber constant (controls transition from L2 to L1)

    Returns:
        Loss value
    """
    return (((x - y) ** 2 + c**2).sqrt() - c).mean()


def get_discretization_schedule(
    total_steps: int,
    s0: int = 10,
    s1: int = 1280,
    rho: float = 7.0,
) -> Callable[[int], int]:
    """Get the discretization schedule N(k) for consistency training.

    N(k) increases from s0 to s1 over training according to a schedule.

    Args:
        total_steps: Total training steps
        s0: Initial discretization steps
        s1: Final discretization steps
        rho: Schedule power parameter

    Returns:
        Function that maps training step to discretization N
    """

    def schedule(step: int) -> int:
        progress = step / max(total_steps, 1)
        n = s0 + (s1 - s0) * (progress**rho)
        return int(math.ceil(n))

    return schedule


def get_ema_decay_schedule(
    total_steps: int,
    s0: int = 10,
    s1: int = 1280,
    mu0: float = 0.95,
) -> Callable[[int], float]:
    """Get EMA decay schedule for consistency training.

    The EMA decay increases with N to maintain consistency.

    Args:
        total_steps: Total training steps
        s0: Initial discretization
        s1: Final discretization
        mu0: Initial EMA decay

    Returns:
        Function that maps training step to EMA decay
    """

    def schedule(step: int) -> float:
        progress = step / max(total_steps, 1)
        n = s0 + (s1 - s0) * progress
        # mu = exp(s0 * log(mu0) / n)
        return math.exp(s0 * math.log(mu0) / max(n, 1))

    return schedule


class ConsistencyTrainer:
    """Consistency training/distillation for diffusion models.

    Supports two modes:
    1. Consistency Training (CT): Train from scratch with self-consistency
    2. Consistency Distillation (CD): Distill from a pretrained teacher

    Example:
        >>> # Consistency Distillation
        >>> trainer = ConsistencyTrainer(
        ...     student=student_model,
        ...     teacher=teacher_model,
        ...     config=config,
        ...     mode="distillation",
        ... )
        >>> trainer.train(train_loader)
        >>>
        >>> # Consistency Training
        >>> trainer = ConsistencyTrainer(
        ...     student=model,
        ...     config=config,
        ...     mode="training",
        ... )
        >>> trainer.train(train_loader)
    """

    def __init__(
        self,
        student: nn.Module,
        config: PostTrainingConfig,
        teacher: nn.Module | None = None,
        diffusion: Any | None = None,
        accelerator: Accelerator | None = None,
        mode: Literal["training", "distillation"] = "distillation",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        s0: int = 10,
        s1: int = 1280,
        huber_c: float | None = None,
    ) -> None:
        """Initialize consistency trainer.

        Args:
            student: Model to train
            config: Training configuration
            teacher: Teacher model for distillation (required if mode="distillation")
            diffusion: D3PM for discrete models
            accelerator: Optional accelerator
            mode: "training" or "distillation"
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            sigma_data: Data standard deviation
            s0: Initial discretization steps
            s1: Final discretization steps
            huber_c: Pseudo-Huber constant
        """
        self.student = student
        self.teacher = teacher
        self.config = config
        self.diffusion = diffusion
        self.mode = mode

        if mode == "distillation" and teacher is None:
            raise ValueError("Teacher model required for distillation mode")

        # Noise schedule parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.s0 = s0
        self.s1 = s1
        self.huber_c = huber_c or config.huber_c

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

        if config.seed is not None:
            set_seed(config.seed)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Setup EMA for target network (in consistency training)
        self.ema = ExponentialMovingAverage(
            student.parameters(),
            decay=0.9999,  # Will be updated during training
        )

        # Freeze teacher
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        # Prepare with accelerator
        self.student, self.optimizer = self.accelerator.prepare(self.student, self.optimizer)
        if self.teacher is not None:
            self.teacher = self.accelerator.prepare(self.teacher)

        self.ema.to(self.accelerator.device)

        # Schedules
        self.discretization_schedule = get_discretization_schedule(config.max_steps, s0, s1)
        self.ema_schedule = get_ema_decay_schedule(config.max_steps, s0, s1)

        # State
        self.global_step = 0

    def _get_sigma_schedule(self, n: int) -> Tensor:
        """Get noise levels for discretization with N steps."""
        device = self.accelerator.device
        # Karras schedule: sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        rho = 7.0
        indices = torch.arange(n, device=device)
        sigmas = (
            self.sigma_max ** (1 / rho)
            + indices / (n - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho
        return sigmas

    def _c_skip(self, sigma: Tensor) -> Tensor:
        """Skip connection scaling."""
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def _c_out(self, sigma: Tensor) -> Tensor:
        """Output scaling."""
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

    def _c_in(self, sigma: Tensor) -> Tensor:
        """Input scaling."""
        return 1 / (sigma**2 + self.sigma_data**2).sqrt()

    def _add_noise(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Add Gaussian noise to continuous data."""
        noise = torch.randn_like(x.float()) * sigma.view(-1, *([1] * (x.dim() - 1)))
        return x.float() + noise

    def _consistency_function(
        self,
        model: nn.Module,
        x: Tensor,
        sigma: Tensor,
        t: Tensor | None = None,
    ) -> Tensor:
        """Apply consistency model with proper scaling.

        For discrete models, this returns token predictions.
        For continuous models, it returns denoised samples.
        """
        if self.diffusion is not None:
            # Discrete model: use timestep-based interface
            if t is None:
                # Convert sigma to timestep
                t = (
                    (sigma / self.sigma_max * self.diffusion.num_timesteps)
                    .long()
                    .clamp(0, self.diffusion.num_timesteps - 1)
                )
            logits = model(x=x, t=t)
            return logits.argmax(dim=-1)
        else:
            # Continuous model with preconditioning
            c_skip = self._c_skip(sigma)
            c_out = self._c_out(sigma)
            c_in = self._c_in(sigma)

            # Scale input and get model output
            if t is None:
                t = sigma
            model_out = model(x=x * c_in.view(-1, 1, 1), t=t)

            # Apply preconditioning
            return c_skip.view(-1, 1, 1) * x + c_out.view(-1, 1, 1) * model_out

    def train_step_distillation(
        self,
        batch: Tensor,
    ) -> dict[str, float]:
        """One step of consistency distillation.

        Uses teacher to generate targets for student.
        """
        device = self.accelerator.device
        batch_size = batch.shape[0]

        # Get current discretization
        n = self.discretization_schedule(self.global_step)
        sigmas = self._get_sigma_schedule(n)

        # Sample timestep index
        idx = torch.randint(1, n, (batch_size,), device=device)
        sigma = sigmas[idx]
        sigma_prev = sigmas[idx - 1]

        # Add noise
        if self.diffusion is not None:
            # Discrete: use diffusion forward process
            t = (
                (sigma / self.sigma_max * self.diffusion.num_timesteps)
                .long()
                .clamp(0, self.diffusion.num_timesteps - 1)
            )
            t_prev = (
                (sigma_prev / self.sigma_max * self.diffusion.num_timesteps)
                .long()
                .clamp(0, self.diffusion.num_timesteps - 1)
            )
            x_t = self.diffusion.q_sample(batch, t)
        else:
            x_t = self._add_noise(batch, sigma)
            t = sigma
            t_prev = sigma_prev

        # Teacher prediction (one step of DDIM/DPM)
        with torch.no_grad():
            if self.diffusion is not None:
                # For discrete, get teacher's denoised prediction
                teacher_logits = self.teacher(x=x_t, t=t)
                x_teacher = teacher_logits.argmax(dim=-1)
            else:
                # For continuous, use teacher for denoising step
                teacher_out = self.teacher(x=x_t, t=t)
                # Simple denoising step
                x_teacher = x_t - sigma.view(-1, 1, 1) * teacher_out

            # Target: student consistency function at (x_teacher, sigma_prev)
            with self.ema.average_parameters():
                target = self._consistency_function(self.student, x_teacher, sigma_prev, t_prev)

        # Student prediction
        pred = self._consistency_function(self.student, x_t, sigma, t)

        # Consistency loss
        if self.diffusion is not None:
            # Cross-entropy for discrete
            loss = F.cross_entropy(
                pred.view(-1, self.diffusion.effective_num_classes) if pred.dim() > 2 else pred,
                target.view(-1),
            )
        else:
            loss = pseudo_huber_loss(pred, target, c=self.huber_c)

        # Backward
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update EMA with scheduled decay
        ema_decay = self.ema_schedule(self.global_step)
        self.ema.decay = ema_decay
        self.ema.update()

        return {
            "loss": loss.item(),
            "n": n,
            "ema_decay": ema_decay,
        }

    def train_step_training(
        self,
        batch: Tensor,
    ) -> dict[str, float]:
        """One step of consistency training (no teacher).

        Uses self-consistency constraint.
        """
        device = self.accelerator.device
        batch_size = batch.shape[0]

        # Get current discretization
        n = self.discretization_schedule(self.global_step)
        sigmas = self._get_sigma_schedule(n)

        # Sample timestep index
        idx = torch.randint(1, n, (batch_size,), device=device)
        sigma = sigmas[idx]
        sigma_prev = sigmas[idx - 1]

        # Add noise
        if self.diffusion is not None:
            t = (
                (sigma / self.sigma_max * self.diffusion.num_timesteps)
                .long()
                .clamp(0, self.diffusion.num_timesteps - 1)
            )
            t_prev = (
                (sigma_prev / self.sigma_max * self.diffusion.num_timesteps)
                .long()
                .clamp(0, self.diffusion.num_timesteps - 1)
            )
            x_t = self.diffusion.q_sample(batch, t)
            x_t_prev = self.diffusion.q_sample(batch, t_prev)
        else:
            noise = torch.randn_like(batch.float())
            x_t = batch.float() + sigma.view(-1, 1, 1) * noise
            x_t_prev = batch.float() + sigma_prev.view(-1, 1, 1) * noise
            t = sigma
            t_prev = sigma_prev

        # Target: EMA model at (x_t_prev, sigma_prev)
        with torch.no_grad(), self.ema.average_parameters():
            target = self._consistency_function(self.student, x_t_prev, sigma_prev, t_prev)

        # Student prediction at (x_t, sigma)
        pred = self._consistency_function(self.student, x_t, sigma, t)

        # Self-consistency loss
        if self.diffusion is not None:
            loss = F.cross_entropy(
                pred.view(-1, self.diffusion.effective_num_classes) if pred.dim() > 2 else pred,
                target.view(-1),
            )
        else:
            loss = pseudo_huber_loss(pred, target, c=self.huber_c)

        # Backward
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update EMA
        ema_decay = self.ema_schedule(self.global_step)
        self.ema.decay = ema_decay
        self.ema.update()

        return {
            "loss": loss.item(),
            "n": n,
            "ema_decay": ema_decay,
        }

    def train(
        self,
        train_loader: Any,
    ) -> dict[str, list[float]]:
        """Run consistency training loop.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Training history
        """
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

        os.makedirs(self.config.logdir, exist_ok=True)

        history = {"loss": [], "n": []}

        train_loader = self.accelerator.prepare(train_loader)
        self.student.train()

        pbar = tqdm(range(self.config.max_steps), desc="Consistency Training")
        data_iter = iter(train_loader)

        for self.global_step in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            # Training step
            with self.accelerator.accumulate(self.student):
                if self.mode == "distillation":
                    metrics = self.train_step_distillation(batch)
                else:
                    metrics = self.train_step_training(batch)

            # Logging
            if self.global_step % self.config.log_every == 0:
                if self.accelerator.is_main_process:
                    self.accelerator.log(metrics, step=self.global_step)
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    n=metrics["n"],
                )
                history["loss"].append(metrics["loss"])
                history["n"].append(metrics["n"])

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint("latest")

        pbar.close()

        if self.config.wandb_project:
            self.accelerator.end_training()

        return history

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.config.logdir, f"consistency_checkpoint_{name}.pt")

        unwrapped = self.accelerator.unwrap_model(self.student)

        checkpoint = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "step": self.global_step,
            "config": self.config.to_dict(),
        }

        self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        # save_checkpoint stores only state_dicts/ints and config.to_dict() (a
        # primitives dict), so the checkpoint contains no arbitrary Python objects.
        checkpoint = torch.load(path, map_location=self.accelerator.device, weights_only=True)

        self.student.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.global_step = checkpoint["step"]


__all__ = [
    "ConsistencyTrainer",
    "pseudo_huber_loss",
    "get_discretization_schedule",
    "get_ema_decay_schedule",
]
