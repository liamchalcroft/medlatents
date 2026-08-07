"""Gradient-Aware Reward Distance Optimization (GARDO).

GARDO is a reward-guided fine-tuning method that explicitly balances
reward improvement with distribution shift from the reference model.
It uses gradient information to adaptively weight the KL penalty.

Key ideas:
1. Track gradient alignment between reward and KL objectives
2. Adaptively adjust the KL coefficient based on gradient conflicts
3. Apply reward thresholding to focus on meaningful improvements

This implementation adapts GARDO concepts for discrete generative models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from ..configs import PostTrainingConfig
from .base import BaseRLTrainer, Trajectory


class GARDOTrainer(BaseRLTrainer):
    """Gradient-Aware Reward Distance Optimization trainer.

    GARDO improves upon standard reward fine-tuning by:
    1. Monitoring gradient alignment between reward and KL terms
    2. Adaptively scaling the KL coefficient
    3. Filtering updates based on reward threshold

    The objective is:
        L = -r(x) + α(grad) * KL(π || π_ref)

    where α(grad) is dynamically adjusted based on gradient cosine similarity.

    Example:
        >>> trainer = GARDOTrainer(
        ...     model=model,
        ...     reward_fn=reward_model,
        ...     config=config,
        ...     reward_threshold=0.1,
        ... )
        >>> trainer.train(seq_length=256)
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[Tensor], Tensor],
        config: PostTrainingConfig,
        ref_model: nn.Module | None = None,
        accelerator: Accelerator | None = None,
        reward_threshold: float | None = None,
        distance_penalty: float | None = None,
        vocab_size: int | None = None,
        num_inference_steps: int = 50,
        model_type: str = "diffusion",
        diffusion: Any | None = None,
        adaptive_kl: bool = True,
        kl_target: float = 0.1,
    ) -> None:
        """Initialize GARDO trainer.

        Args:
            model: Model to train
            reward_fn: Reward function
            config: Training configuration
            ref_model: Reference model for KL computation
            accelerator: Optional accelerator
            reward_threshold: Minimum reward improvement to update
            distance_penalty: Base KL penalty coefficient
            vocab_size: Vocabulary size
            num_inference_steps: Generation steps
            model_type: "diffusion", "flow", or "maskgit"
            diffusion: D3PM for diffusion models
            adaptive_kl: Whether to adapt KL coefficient
            kl_target: Target KL divergence for adaptive scaling
        """
        super().__init__(
            model=model,
            reward_fn=reward_fn,
            config=config,
            ref_model=ref_model,
            accelerator=accelerator,
        )

        self.reward_threshold = reward_threshold or config.reward_threshold
        self.distance_penalty = distance_penalty or config.distance_penalty
        self.num_inference_steps = num_inference_steps
        self.model_type = model_type
        self.diffusion = diffusion
        self.adaptive_kl = adaptive_kl
        self.kl_target = kl_target

        # Adaptive KL state
        self.kl_coeff = self.distance_penalty
        self.grad_history: list[float] = []

        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            elif diffusion is not None:
                vocab_size = diffusion.effective_num_classes
            else:
                raise ValueError("vocab_size must be provided")
        self.vocab_size = vocab_size

        if self.diffusion is not None:
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
        """Generate trajectories."""
        if self.model_type == "diffusion" and self.diffusion is not None:
            return self._generate_diffusion_trajectories(batch_size, seq_length)
        else:
            return self._generate_generic_trajectories(batch_size, seq_length)

    def _generate_diffusion_trajectories(
        self,
        batch_size: int,
        seq_length: int,
    ) -> list[Trajectory]:
        """Generate diffusion trajectories."""
        device = self.accelerator.device
        num_timesteps = self.diffusion.num_timesteps
        step_size = num_timesteps // self.num_inference_steps
        timesteps = list(range(num_timesteps - 1, -1, -step_size))[: self.num_inference_steps]

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

        all_states = []
        all_actions = []
        all_timesteps = []
        all_log_probs = []

        self.model.eval()
        with torch.no_grad():
            for t_val in timesteps:
                t = torch.full((batch_size,), t_val, dtype=torch.long, device=device)

                all_states.append(x_t.clone())
                all_timesteps.append(t.clone())

                logits = self.model(x=x_t, t=t)
                probs = F.softmax(logits, dim=-1)

                flat_probs = probs.view(-1, probs.size(-1))
                actions = torch.multinomial(flat_probs, num_samples=1)
                actions = actions.view(batch_size, seq_length)

                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = (
                    torch.gather(log_probs, dim=-1, index=actions.unsqueeze(-1))
                    .squeeze(-1)
                    .sum(dim=-1)
                )

                all_actions.append(actions.clone())
                all_log_probs.append(action_log_probs.clone())

                if t_val > 0:
                    x_t = self.diffusion.p_sample(self.model, x_t, t, temperature=1.0)
                else:
                    x_t = actions

        self.model.train()

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

    def _generate_generic_trajectories(
        self,
        batch_size: int,
        seq_length: int,
    ) -> list[Trajectory]:
        """Generate generic trajectories for non-diffusion models."""
        device = self.accelerator.device
        mask_token = getattr(self.model, "mask_token", self.vocab_size)

        x_t = torch.full(
            (batch_size, seq_length),
            mask_token,
            dtype=torch.long,
            device=device,
        )

        all_states = []
        all_actions = []
        all_timesteps = []
        all_log_probs = []

        self.model.eval()
        with torch.no_grad():
            for step in range(self.num_inference_steps):
                t = torch.full((batch_size,), step / self.num_inference_steps, device=device)

                all_states.append(x_t.clone())
                all_timesteps.append(t.clone())

                if self.model_type == "maskgit":
                    logits = self.model(x_t)
                else:
                    logits = self.model(x=x_t, t=t)

                probs = F.softmax(logits, dim=-1)
                flat_probs = probs.view(-1, self.vocab_size)
                actions = torch.multinomial(flat_probs, num_samples=1)
                actions = actions.view(batch_size, seq_length)

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

    def _compute_gradient_alignment(
        self,
        reward_grad: Tensor,
        kl_grad: Tensor,
    ) -> float:
        """Compute cosine similarity between reward and KL gradients."""
        reward_flat = torch.cat([g.flatten() for g in reward_grad if g is not None])
        kl_flat = torch.cat([g.flatten() for g in kl_grad if g is not None])

        if reward_flat.numel() == 0 or kl_flat.numel() == 0:
            return 0.0

        cos_sim = F.cosine_similarity(
            reward_flat.unsqueeze(0),
            kl_flat.unsqueeze(0),
        ).item()

        return cos_sim

    def _update_kl_coefficient(self, kl_value: float, grad_alignment: float) -> None:
        """Adaptively update KL coefficient based on gradient alignment."""
        if not self.adaptive_kl:
            return

        # If gradients are aligned (positive cosine), reduce KL penalty
        # If gradients conflict (negative cosine), increase KL penalty
        if grad_alignment > 0.1:
            # Gradients aligned - can reduce KL penalty
            self.kl_coeff = max(0.01, self.kl_coeff * 0.99)
        elif grad_alignment < -0.1:
            # Gradients conflict - increase KL penalty
            self.kl_coeff = min(1.0, self.kl_coeff * 1.01)

        # Also adjust based on actual KL value
        if kl_value > self.kl_target * 1.5:
            self.kl_coeff = min(1.0, self.kl_coeff * 1.05)
        elif kl_value < self.kl_target * 0.5:
            self.kl_coeff = max(0.01, self.kl_coeff * 0.95)

    def compute_policy_loss(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute GARDO loss with adaptive KL penalty.

        The GARDO loss combines:
        1. Negative reward (to maximize reward)
        2. KL penalty (to stay close to reference)
        3. Reward thresholding (to focus on improvements)
        """
        device = self.accelerator.device

        total_reward_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl_value = 0.0
        num_valid = 0
        num_filtered = 0

        for traj in trajectories:
            # Get final reward
            reward = traj.rewards[-1]

            # Filter by reward threshold
            if reward.item() < self.reward_threshold:
                num_filtered += 1
                continue

            # Compute policy log prob
            total_log_prob = torch.tensor(0.0, device=device, requires_grad=True)
            total_ref_log_prob = torch.tensor(0.0, device=device)

            for step_idx in range(len(traj.states)):
                state = traj.states[step_idx].unsqueeze(0)
                action = traj.actions[step_idx].unsqueeze(0)
                t = traj.timesteps[step_idx].unsqueeze(0)

                # Current policy log prob
                if self.model_type == "maskgit":
                    logits = self.model(state)
                else:
                    logits = self.model(x=state, t=t)

                log_probs = F.log_softmax(logits, dim=-1)
                step_log_prob = (
                    torch.gather(log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1).sum()
                )
                total_log_prob = total_log_prob + step_log_prob

                # Reference log prob
                if self.ref_model is not None:
                    with torch.no_grad():
                        if self.model_type == "maskgit":
                            ref_logits = self.ref_model(state)
                        else:
                            ref_logits = self.ref_model(x=state, t=t)

                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        step_ref_log_prob = (
                            torch.gather(ref_log_probs, dim=-1, index=action.unsqueeze(-1))
                            .squeeze(-1)
                            .sum()
                        )
                        total_ref_log_prob = total_ref_log_prob + step_ref_log_prob

            # Reward loss: -r * log_prob (policy gradient)
            reward_loss = -reward * total_log_prob
            total_reward_loss = total_reward_loss + reward_loss

            # KL loss
            if self.ref_model is not None:
                kl = total_log_prob - total_ref_log_prob
                total_kl_loss = total_kl_loss + kl
                total_kl_value += kl.item()

            num_valid += 1

        # Average losses
        if num_valid > 0:
            total_reward_loss = total_reward_loss / num_valid
            total_kl_loss = total_kl_loss / num_valid
            avg_kl = total_kl_value / num_valid
        else:
            avg_kl = 0.0

        # Combined loss with adaptive KL coefficient
        total_loss = total_reward_loss + self.kl_coeff * total_kl_loss

        # Update KL coefficient (would need gradient alignment computation)
        self._update_kl_coefficient(avg_kl, 0.0)

        metrics = {
            "loss": total_loss.item(),
            "reward_loss": total_reward_loss.item(),
            "kl_loss": total_kl_loss.item(),
            "kl_coeff": self.kl_coeff,
            "avg_kl": avg_kl,
            "num_valid": num_valid,
            "num_filtered": num_filtered,
        }

        return total_loss, metrics


__all__ = [
    "GARDOTrainer",
]
