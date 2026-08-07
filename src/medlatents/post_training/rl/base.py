"""Base infrastructure for reinforcement learning post-training.

This module provides the foundation for RL-based fine-tuning methods
including trajectory storage, advantage estimation, and policy gradient
computation.

Key concepts:
- Trajectory: A sequence of (state, action, reward) tuples from generation
- Advantage: Relative value of an action compared to baseline
- Policy gradient: Gradient of expected reward w.r.t. policy parameters

References:
- DDPO: "Training Diffusion Models with Reinforcement Learning" (Black et al., 2023)
- GRPO: "DeepSeekMath: Pushing the Limits" (Shao et al., 2024)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..configs import PostTrainingConfig


@dataclass
class Trajectory:
    """A single generation trajectory with rewards.

    For diffusion/flow models, this represents the full denoising path.
    For autoregressive models, this is the sequence of token generations.

    Attributes:
        states: Intermediate states [num_steps, seq_len] or [num_steps, seq_len, dim]
        actions: Actions taken (tokens generated) [num_steps, seq_len]
        timesteps: Timestep at each step [num_steps]
        log_probs: Log probabilities of actions [num_steps]
        rewards: Rewards at each step [num_steps] (often only final is non-zero)
        final_sample: The final generated sample
        prompt: Optional conditioning information
    """

    states: Tensor
    actions: Tensor
    timesteps: Tensor
    log_probs: Tensor
    rewards: Tensor
    final_sample: Tensor
    prompt: Any = None
    advantages: Tensor | None = None
    returns: Tensor | None = None

    def to(self, device: torch.device) -> "Trajectory":
        """Move trajectory to device."""
        return Trajectory(
            states=self.states.to(device),
            actions=self.actions.to(device),
            timesteps=self.timesteps.to(device),
            log_probs=self.log_probs.to(device),
            rewards=self.rewards.to(device),
            final_sample=self.final_sample.to(device),
            prompt=self.prompt,
            advantages=self.advantages.to(device) if self.advantages is not None else None,
            returns=self.returns.to(device) if self.returns is not None else None,
        )


@dataclass
class RLBatch:
    """Batch of trajectories for RL training.

    Attributes:
        trajectories: List of Trajectory objects
        batch_rewards: Rewards for each trajectory [batch]
        batch_advantages: Normalized advantages [batch]
    """

    trajectories: list[Trajectory]
    batch_rewards: Tensor
    batch_advantages: Tensor | None = None

    def __len__(self) -> int:
        return len(self.trajectories)


class TrajectoryBuffer:
    """Buffer for storing and sampling trajectories.

    Implements experience replay for RL training with support for
    priority sampling and trajectory filtering.
    """

    def __init__(
        self,
        max_size: int = 10000,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize trajectory buffer.

        Args:
            max_size: Maximum number of trajectories to store
            device: Device for storing trajectories
        """
        self.max_size = max_size
        self.device = device
        self.buffer: list[Trajectory] = []
        self.priorities: list[float] = []

    def add(self, trajectory: Trajectory, priority: float = 1.0) -> None:
        """Add a trajectory to the buffer."""
        if len(self.buffer) >= self.max_size:
            # Remove lowest priority trajectory
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)

        self.buffer.append(trajectory)
        self.priorities.append(priority)

    def sample(self, batch_size: int, prioritized: bool = False) -> list[Trajectory]:
        """Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample
            prioritized: If True, sample proportional to priority

        Returns:
            List of sampled trajectories
        """
        if len(self.buffer) == 0:
            return []

        batch_size = min(batch_size, len(self.buffer))

        if prioritized:
            probs = torch.tensor(self.priorities)
            probs = probs / probs.sum()
            indices = torch.multinomial(probs, batch_size, replacement=False)
        else:
            indices = torch.randperm(len(self.buffer))[:batch_size]

        return [self.buffer[i] for i in indices]

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
        self.priorities = []

    def __len__(self) -> int:
        return len(self.buffer)


def compute_advantages_gae(
    rewards: Tensor,
    values: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a smooth tradeoff between bias and variance in
    advantage estimation.

    Args:
        rewards: Rewards at each step [T]
        values: Value estimates at each step [T]
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    last_gae = 0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def compute_advantages_monte_carlo(
    rewards: Tensor,
    gamma: float = 0.99,
) -> Tensor:
    """Compute Monte Carlo returns (cumulative discounted rewards).

    Simple but high variance advantage estimation.

    Args:
        rewards: Rewards at each step [T]
        gamma: Discount factor

    Returns:
        Returns at each step [T]
    """
    T = len(rewards)
    returns = torch.zeros_like(rewards)

    running_return = 0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def normalize_advantages(advantages: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize advantages to have zero mean and unit variance.

    This is a common technique to stabilize policy gradient training.

    Args:
        advantages: Raw advantages [batch] or [batch, T]
        eps: Small constant for numerical stability

    Returns:
        Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


class BaseRLTrainer(ABC):
    """Abstract base class for RL post-training methods.

    Provides common infrastructure for:
    - Trajectory generation and storage
    - Reward computation
    - Policy gradient estimation
    - Training loop with logging

    Subclasses must implement:
    - generate_trajectories: Model-specific generation
    - compute_policy_loss: Method-specific loss computation
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[Tensor], Tensor],
        config: PostTrainingConfig,
        ref_model: nn.Module | None = None,
        value_model: nn.Module | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Initialize RL trainer.

        Args:
            model: Policy model to train
            reward_fn: Function that computes rewards for samples
            config: Training configuration
            ref_model: Optional reference model for KL penalty
            value_model: Optional value function for advantage estimation
            accelerator: Optional pre-configured accelerator
        """
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.ref_model = ref_model
        self.value_model = value_model

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
        params = list(model.parameters())
        if value_model is not None:
            params += list(value_model.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Setup EMA
        self.ema = None
        if config.use_ema:
            self.ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)

        # Freeze reference model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # Prepare with accelerator
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare(self.ref_model)
        if self.value_model is not None:
            self.value_model = self.accelerator.prepare(self.value_model)

        # Move EMA to device
        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(device=self.accelerator.device)

        # Training state
        self.global_step = 0

    @abstractmethod
    def generate_trajectories(
        self,
        batch_size: int,
        seq_length: int,
        **kwargs: Any,
    ) -> list[Trajectory]:
        """Generate trajectories by sampling from the model.

        Args:
            batch_size: Number of trajectories to generate
            seq_length: Sequence length
            **kwargs: Additional generation arguments

        Returns:
            List of Trajectory objects
        """
        raise NotImplementedError

    @abstractmethod
    def compute_policy_loss(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute policy gradient loss.

        Args:
            trajectories: Batch of trajectories

        Returns:
            Tuple of (loss, metrics_dict)
        """
        raise NotImplementedError

    def compute_kl_penalty(
        self,
        log_probs: Tensor,
        ref_log_probs: Tensor,
    ) -> Tensor:
        """Compute KL divergence penalty.

        Args:
            log_probs: Log probs from current policy
            ref_log_probs: Log probs from reference policy

        Returns:
            KL divergence estimate
        """
        # Approximate KL: E[log p - log q]
        return (log_probs - ref_log_probs).mean()

    def train_step(self, seq_length: int) -> dict[str, float]:
        """Execute one training step.

        1. Generate trajectories
        2. Compute rewards
        3. Compute advantages
        4. Update policy

        Args:
            seq_length: Sequence length for generation

        Returns:
            Dictionary of metrics
        """
        batch_size = self.config.batch_size * self.config.num_samples_per_prompt

        # Generate trajectories
        trajectories = self.generate_trajectories(batch_size, seq_length)

        # Compute rewards for final samples
        with torch.no_grad():
            final_samples = torch.stack([t.final_sample for t in trajectories])
            rewards = self.reward_fn(final_samples)  # [batch]

            # Normalize rewards if configured
            if self.config.normalize_rewards:
                rewards = normalize_advantages(rewards)

        # Update trajectory rewards
        for i, traj in enumerate(trajectories):
            # Assign final reward to last step
            traj.rewards[-1] = rewards[i]

            # Compute advantages
            if self.value_model is not None:
                values = self.value_model(traj.states)
                advantages, returns = compute_advantages_gae(
                    traj.rewards, values, lam=self.config.gae_lambda
                )
            else:
                returns = compute_advantages_monte_carlo(traj.rewards)
                advantages = returns

            traj.advantages = normalize_advantages(advantages)
            traj.returns = returns

        # Compute policy loss
        loss, metrics = self.compute_policy_loss(trajectories)

        # Add reward metrics
        metrics["reward/mean"] = rewards.mean().item()
        metrics["reward/std"] = rewards.std().item()
        metrics["reward/max"] = rewards.max().item()
        metrics["reward/min"] = rewards.min().item()

        # Backward pass
        self.accelerator.backward(loss)

        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        return metrics

    def train(self, seq_length: int = 256) -> dict[str, list[float]]:
        """Run full RL training loop.

        Args:
            seq_length: Sequence length for generation

        Returns:
            Training history
        """
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

        os.makedirs(self.config.logdir, exist_ok=True)

        history: dict[str, list[float]] = {
            "loss": [],
            "reward_mean": [],
        }

        # Training loop
        self.model.train()
        pbar = tqdm(range(self.config.max_steps), desc="RL Training")

        for self.global_step in pbar:
            # Learning rate warmup
            if self.global_step < self.config.warmup_steps:
                lr_scale = (self.global_step + 1) / self.config.warmup_steps
            else:
                progress = (self.global_step - self.config.warmup_steps) / (
                    self.config.max_steps - self.config.warmup_steps
                )
                lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                lr_scale = lr_scale.item()

            lr = self.config.lr * lr_scale
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Training step
            with self.accelerator.accumulate(self.model):
                metrics = self.train_step(seq_length)

            # Logging
            if self.global_step % self.config.log_every == 0:
                metrics["lr"] = lr
                if self.accelerator.is_main_process:
                    self.accelerator.log(metrics, step=self.global_step)
                pbar.set_postfix(
                    loss=f"{metrics.get('loss', 0):.4f}",
                    reward=f"{metrics.get('reward/mean', 0):.4f}",
                )
                history["loss"].append(metrics.get("loss", 0))
                history["reward_mean"].append(metrics.get("reward/mean", 0))

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint("latest")

        pbar.close()

        if self.config.wandb_project:
            self.accelerator.end_training()

        return history

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.config.logdir, f"rl_checkpoint_{name}.pt")

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config.to_dict(),
        }

        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

        if self.value_model is not None:
            unwrapped_value = self.accelerator.unwrap_model(self.value_model)
            checkpoint["value_model"] = unwrapped_value.state_dict()

        self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        # save_checkpoint stores only state_dicts/ints and config.to_dict() (a
        # primitives dict), so the checkpoint contains no arbitrary Python objects.
        checkpoint = torch.load(path, map_location=self.accelerator.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["step"]

        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])

        if self.value_model is not None and "value_model" in checkpoint:
            self.value_model.load_state_dict(checkpoint["value_model"])


__all__ = [
    "Trajectory",
    "RLBatch",
    "TrajectoryBuffer",
    "BaseRLTrainer",
    "compute_advantages_gae",
    "compute_advantages_monte_carlo",
    "normalize_advantages",
]
