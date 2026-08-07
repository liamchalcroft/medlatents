"""Denoising Diffusion Policy Optimization (DDPO).

DDPO adapts PPO-style policy gradient methods to diffusion models by treating
the denoising process as a multi-step MDP. Each denoising step is an action,
and the reward is typically given only at the final step.

Key insights:
1. The diffusion model is a policy π(x_{t-1} | x_t)
2. Each trajectory is a full denoising path from noise to sample
3. We can compute per-step log-probs for policy gradient

Reference:
- "Training Diffusion Models with Reinforcement Learning" (Black et al., 2023)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from ...diffusion.d3pm import D3PM
from ..configs import PostTrainingConfig
from .base import BaseRLTrainer, Trajectory


class DDPOTrainer(BaseRLTrainer):
    """DDPO trainer for diffusion models.

    Implements PPO-style policy gradient optimization for diffusion models.
    The key idea is to:
    1. Sample full denoising trajectories
    2. Compute rewards for final samples
    3. Estimate advantages using GAE
    4. Update policy with clipped surrogate objective

    Example:
        >>> from medlatents.diffusion import D3PM
        >>> d3pm = D3PM(num_classes=1024, num_timesteps=1000)
        >>> trainer = DDPOTrainer(
        ...     model=backbone,
        ...     reward_fn=reward_model,
        ...     config=config,
        ...     diffusion=d3pm,
        ... )
        >>> trainer.train(seq_length=256)
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[Tensor], Tensor],
        config: PostTrainingConfig,
        diffusion: D3PM,
        ref_model: nn.Module | None = None,
        value_model: nn.Module | None = None,
        accelerator: Accelerator | None = None,
        num_inference_steps: int = 50,
        clip_range: float | None = None,
        kl_coeff: float | None = None,
    ) -> None:
        """Initialize DDPO trainer.

        Args:
            model: Diffusion backbone model
            reward_fn: Reward function for samples
            config: Training configuration
            diffusion: D3PM diffusion process
            ref_model: Reference model for KL penalty
            value_model: Value function for advantage estimation
            accelerator: Optional accelerator
            num_inference_steps: Number of denoising steps
            clip_range: PPO clipping range (uses config if None)
            kl_coeff: KL penalty coefficient (uses config if None)
        """
        super().__init__(
            model=model,
            reward_fn=reward_fn,
            config=config,
            ref_model=ref_model,
            value_model=value_model,
            accelerator=accelerator,
        )

        self.diffusion = diffusion
        self.num_inference_steps = num_inference_steps
        self.clip_range = clip_range or config.clip_range
        self.kl_coeff = kl_coeff or config.kl_coeff

        # Move diffusion to device
        self._move_diffusion_to_device()

    def _move_diffusion_to_device(self) -> None:
        """Move diffusion tensors to device with non_blocking transfers."""
        device = self.accelerator.device
        self.diffusion.device = device
        tensors = ["betas", "alphas", "alphas_cumprod", "Q_t", "Q_bar_t", "Q_bar_t_minus_1"]
        for name in tensors:
            tensor = getattr(self.diffusion, name)
            setattr(self.diffusion, name, tensor.to(device, non_blocking=True))

    def generate_trajectories(
        self,
        batch_size: int,
        seq_length: int,
        **kwargs: Any,
    ) -> list[Trajectory]:
        """Generate denoising trajectories.

        Samples full denoising paths while recording states, actions,
        and log probabilities at each step.

        Args:
            batch_size: Number of trajectories
            seq_length: Sequence length
            **kwargs: Additional arguments

        Returns:
            List of Trajectory objects
        """
        device = self.accelerator.device
        num_timesteps = self.diffusion.num_timesteps

        # Compute step size for inference
        step_size = num_timesteps // self.num_inference_steps
        timesteps = list(range(num_timesteps - 1, -1, -step_size))[: self.num_inference_steps]

        # Initialize from noise
        if self.diffusion.transition_type == "absorbing":
            x_t = torch.full(
                (batch_size, seq_length),
                self.diffusion.mask_token,
                dtype=torch.long,
                device=device,
            )
        else:
            x_t = torch.randint(
                0,
                self.diffusion.num_classes,
                (batch_size, seq_length),
                device=device,
            )

        # Storage for trajectories
        all_states = []
        all_actions = []
        all_timesteps = []
        all_log_probs = []

        self.model.eval()
        with torch.no_grad():
            for t_val in timesteps:
                t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)

                # Store current state
                all_states.append(x_t.clone())
                all_timesteps.append(t.clone())

                # Get model prediction
                logits = self.model(x=x_t, t=t)  # [batch, seq, vocab]
                probs = F.softmax(logits, dim=-1)

                # Sample next state (action)
                flat_probs = probs.view(-1, probs.size(-1))
                actions = torch.multinomial(flat_probs, num_samples=1)
                actions = actions.view(batch_size, seq_length)

                # Compute log probability of action
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = torch.gather(
                    log_probs, dim=-1, index=actions.unsqueeze(-1)
                ).squeeze(-1)  # [batch, seq]
                action_log_probs = action_log_probs.sum(dim=-1)  # [batch]

                all_actions.append(actions.clone())
                all_log_probs.append(action_log_probs.clone())

                # Denoise step
                if t_val > 0:
                    x_t = self.diffusion.p_sample(self.model, x_t, t, temperature=1.0)
                else:
                    x_t = actions

        self.model.train()

        # Create trajectory objects
        trajectories = []
        for b in range(batch_size):
            states = torch.stack([s[b] for s in all_states])
            actions = torch.stack([a[b] for a in all_actions])
            timesteps_tensor = torch.stack([t[b] for t in all_timesteps])
            log_probs = torch.stack([lp[b] for lp in all_log_probs])
            rewards = torch.zeros(len(all_states), device=device)

            trajectories.append(
                Trajectory(
                    states=states,
                    actions=actions,
                    timesteps=timesteps_tensor,
                    log_probs=log_probs,
                    rewards=rewards,
                    final_sample=x_t[b],
                )
            )

        return trajectories

    def compute_policy_loss(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute PPO-style clipped surrogate loss.

        The PPO objective is:
            L = -min(r*A, clip(r, 1-ε, 1+ε)*A)

        where r = π(a|s) / π_old(a|s) is the probability ratio.

        Args:
            trajectories: List of trajectories with advantages

        Returns:
            Tuple of (loss, metrics)
        """
        device = self.accelerator.device
        total_loss = torch.tensor(0.0, device=device)
        total_policy_loss = torch.tensor(0.0, device=device)
        total_value_loss = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)
        num_steps = 0

        for traj in trajectories:
            # Get current log probs for each step
            for step_idx in range(len(traj.states)):
                state = traj.states[step_idx].unsqueeze(0)
                action = traj.actions[step_idx].unsqueeze(0)
                t = traj.timesteps[step_idx].unsqueeze(0)
                old_log_prob = traj.log_probs[step_idx]
                advantage = traj.advantages[step_idx] if traj.advantages is not None else 0.0

                # Get current policy log prob
                logits = self.model(x=state, t=t)
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_prob = (
                    torch.gather(log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1).sum()
                )

                # Compute probability ratio
                ratio = torch.exp(new_log_prob - old_log_prob)

                # Clipped surrogate objective
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage
                policy_loss = -torch.min(surr1, surr2)

                total_policy_loss += policy_loss

                # KL penalty
                if self.ref_model is not None:
                    with torch.no_grad():
                        ref_logits = self.ref_model(x=state, t=t)
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_log_prob = (
                            torch.gather(ref_log_probs, dim=-1, index=action.unsqueeze(-1))
                            .squeeze(-1)
                            .sum()
                        )

                    kl = new_log_prob - ref_log_prob
                    total_kl += kl

                # Value loss
                if self.value_model is not None and traj.returns is not None:
                    value_pred = self.value_model(state, t)
                    target_return = traj.returns[step_idx]
                    value_loss = F.mse_loss(value_pred.squeeze(), target_return)
                    total_value_loss += value_loss

                num_steps += 1

        # Average losses
        if num_steps > 0:
            total_policy_loss = total_policy_loss / num_steps
            total_value_loss = total_value_loss / num_steps
            total_kl = total_kl / num_steps

        total_loss = total_policy_loss + 0.5 * total_value_loss + self.kl_coeff * total_kl

        metrics = {
            "loss": total_loss.item(),
            "policy_loss": total_policy_loss.item(),
            "value_loss": total_value_loss.item(),
            "kl": total_kl.item(),
        }

        return total_loss, metrics


class DDPODiscreteFlowTrainer(BaseRLTrainer):
    """DDPO trainer for discrete flow matching models.

    Adapts DDPO for discrete flow matching where the model predicts
    token distributions at each timestep along the flow.
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[Tensor], Tensor],
        config: PostTrainingConfig,
        path: Any,  # MixtureDiscreteProbPath
        source_distribution: Any | None = None,
        ref_model: nn.Module | None = None,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        num_inference_steps: int = 50,
    ) -> None:
        """Initialize discrete flow DDPO trainer."""
        super().__init__(
            model=model,
            reward_fn=reward_fn,
            config=config,
            ref_model=ref_model,
            accelerator=accelerator,
        )

        self.path = path
        self.source_distribution = source_distribution
        self.num_inference_steps = num_inference_steps

        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            else:
                raise ValueError("vocab_size must be provided")
        self.vocab_size = vocab_size

        # Get mask token
        if hasattr(model, "mask_token"):
            self.mask_token = model.mask_token
        else:
            self.mask_token = vocab_size

    def generate_trajectories(
        self,
        batch_size: int,
        seq_length: int,
        **kwargs: Any,
    ) -> list[Trajectory]:
        """Generate flow trajectories."""
        device = self.accelerator.device

        # Time grid
        time_grid = torch.linspace(0, 1, self.num_inference_steps + 1, device=device)

        # Initialize from source (mask tokens)
        if self.source_distribution is not None:
            x_init = self.source_distribution.sample((batch_size, seq_length))
        else:
            x_init = torch.full(
                (batch_size, seq_length),
                self.mask_token,
                dtype=torch.long,
                device=device,
            )

        x_t = x_init.clone()

        all_states = []
        all_actions = []
        all_timesteps = []
        all_log_probs = []

        self.model.eval()
        with torch.no_grad():
            for idx in range(len(time_grid) - 1):
                t = time_grid[idx]
                t_batch = torch.full((batch_size,), t.item(), device=device)

                all_states.append(x_t.clone())
                all_timesteps.append(t_batch.clone())

                # Get prediction
                logits = self.model(x=x_t, t=t_batch)
                probs = F.softmax(logits, dim=-1)

                # Sample
                flat_probs = probs.view(-1, self.vocab_size)
                actions = torch.multinomial(flat_probs, num_samples=1)
                actions = actions.view(batch_size, seq_length)

                # Log prob
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = (
                    torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1))
                    .squeeze(-1)
                    .sum(dim=-1)
                )

                all_actions.append(actions.clone())
                all_log_probs.append(action_log_probs.clone())

                x_t = actions

        self.model.train()

        # Build trajectories
        trajectories = []
        for b in range(batch_size):
            states = torch.stack([s[b] for s in all_states])
            actions = torch.stack([a[b] for a in all_actions])
            timesteps_tensor = torch.stack([t[b] for t in all_timesteps])
            log_probs = torch.stack([lp[b] for lp in all_log_probs])
            rewards = torch.zeros(len(all_states), device=device)

            trajectories.append(
                Trajectory(
                    states=states,
                    actions=actions,
                    timesteps=timesteps_tensor,
                    log_probs=log_probs,
                    rewards=rewards,
                    final_sample=x_t[b],
                )
            )

        return trajectories

    def compute_policy_loss(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute policy loss (same as DDPOTrainer)."""
        device = self.accelerator.device
        total_loss = torch.tensor(0.0, device=device)
        num_steps = 0

        for traj in trajectories:
            for step_idx in range(len(traj.states)):
                state = traj.states[step_idx].unsqueeze(0)
                action = traj.actions[step_idx].unsqueeze(0)
                t = traj.timesteps[step_idx].unsqueeze(0)
                old_log_prob = traj.log_probs[step_idx]
                advantage = traj.advantages[step_idx] if traj.advantages is not None else 0.0

                logits = self.model(x=state, t=t)
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_prob = (
                    torch.gather(log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1).sum()
                )

                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                    * advantage
                )
                policy_loss = -torch.min(surr1, surr2)

                total_loss += policy_loss
                num_steps += 1

        if num_steps > 0:
            total_loss = total_loss / num_steps

        return total_loss, {"loss": total_loss.item()}


__all__ = [
    "DDPOTrainer",
    "DDPODiscreteFlowTrainer",
]
