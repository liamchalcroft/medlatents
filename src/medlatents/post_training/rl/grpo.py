"""Group Relative Policy Optimization (GRPO).

GRPO is a memory-efficient alternative to PPO that eliminates the need for
a separate value network by using group-relative advantages. Instead of
estimating value functions, it normalizes rewards within a group of samples
from the same prompt.

Key advantages:
1. No value network needed (saves memory)
2. Natural baseline from group statistics
3. Works well with diverse reward distributions

Reference:
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (Shao et al., 2024)
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
from .base import BaseRLTrainer, Trajectory, normalize_advantages


class GRPOTrainer(BaseRLTrainer):
    """Group Relative Policy Optimization trainer.

    GRPO samples multiple outputs for each input and computes advantages
    relative to the group mean. This eliminates the need for a learned
    value function while providing low-variance gradients.

    The GRPO objective is:
        L = -E[A(x) * log π(x)] + β * KL(π || π_ref)

    where A(x) = (r(x) - mean(r)) / std(r) is the group-relative advantage.

    Example:
        >>> trainer = GRPOTrainer(
        ...     model=model,
        ...     reward_fn=reward_model,
        ...     config=config,
        ...     num_samples_per_prompt=8,
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
        num_samples_per_prompt: int | None = None,
        vocab_size: int | None = None,
        num_inference_steps: int = 50,
        model_type: str = "diffusion",
        diffusion: Any | None = None,
        flow_path: Any | None = None,
    ) -> None:
        """Initialize GRPO trainer.

        Args:
            model: Model to train
            reward_fn: Reward function
            config: Training configuration
            ref_model: Reference model for KL penalty
            accelerator: Optional accelerator
            num_samples_per_prompt: Samples per group (uses config if None)
            vocab_size: Vocabulary size
            num_inference_steps: Number of generation steps
            model_type: "diffusion", "flow", or "maskgit"
            diffusion: D3PM for diffusion models
            flow_path: Path for flow models
        """
        super().__init__(
            model=model,
            reward_fn=reward_fn,
            config=config,
            ref_model=ref_model,
            accelerator=accelerator,
        )

        self.num_samples = num_samples_per_prompt or config.num_samples_per_prompt
        self.num_inference_steps = num_inference_steps
        self.model_type = model_type
        self.diffusion = diffusion
        self.flow_path = flow_path

        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            elif diffusion is not None:
                vocab_size = diffusion.effective_num_classes
            else:
                raise ValueError("vocab_size must be provided")
        self.vocab_size = vocab_size

        # Move diffusion to device if present
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
        """Generate trajectories for GRPO.

        Generates multiple samples per "prompt" for group-relative advantages.
        """
        # For GRPO, batch_size should be divisible by num_samples
        # Each group shares the same initial state
        num_groups = max(1, batch_size // self.num_samples)
        actual_batch = num_groups * self.num_samples

        if self.model_type == "diffusion" and self.diffusion is not None:
            return self._generate_diffusion_trajectories(actual_batch, seq_length, num_groups)
        elif self.model_type == "flow":
            return self._generate_flow_trajectories(actual_batch, seq_length, num_groups)
        else:
            return self._generate_maskgit_trajectories(actual_batch, seq_length, num_groups)

    def _generate_diffusion_trajectories(
        self,
        batch_size: int,
        seq_length: int,
        num_groups: int,
    ) -> list[Trajectory]:
        """Generate diffusion trajectories."""
        device = self.accelerator.device
        num_timesteps = self.diffusion.num_timesteps
        step_size = num_timesteps // self.num_inference_steps
        timesteps = list(range(num_timesteps - 1, -1, -step_size))[: self.num_inference_steps]

        # Initialize - share initial state within groups
        if self.diffusion.transition_type == "absorbing":
            x_init = torch.full(
                (num_groups, seq_length),
                self.diffusion.mask_token,
                dtype=torch.long,
                device=device,
            )
        else:
            x_init = torch.randint(
                0,
                self.diffusion.num_classes,
                (num_groups, seq_length),
                device=device,
            )

        # Repeat for samples per group
        x_t = x_init.repeat_interleave(self.num_samples, dim=0)

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

        # Build trajectories with group information
        trajectories = []
        for b in range(batch_size):
            group_id = b // self.num_samples

            states = torch.stack([s[b] for s in all_states])
            actions = torch.stack([a[b] for a in all_actions])
            timesteps_tensor = torch.stack([t[b] for t in all_timesteps])
            log_probs = torch.stack([lp[b] for lp in all_log_probs])
            rewards = torch.zeros(len(all_states), device=device)

            traj = Trajectory(
                states=states,
                actions=actions,
                timesteps=timesteps_tensor,
                log_probs=log_probs,
                rewards=rewards,
                final_sample=x_t[b],
            )
            # Store group ID in prompt field for grouping
            traj.prompt = group_id
            trajectories.append(traj)

        return trajectories

    def _generate_flow_trajectories(
        self,
        batch_size: int,
        seq_length: int,
        num_groups: int,
    ) -> list[Trajectory]:
        """Generate flow trajectories."""
        device = self.accelerator.device
        time_grid = torch.linspace(0, 1, self.num_inference_steps + 1, device=device)

        # Initialize with shared state per group
        mask_token = getattr(self.model, "mask_token", self.vocab_size)
        x_init = torch.full(
            (num_groups, seq_length),
            mask_token,
            dtype=torch.long,
            device=device,
        )
        x_t = x_init.repeat_interleave(self.num_samples, dim=0)

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

                logits = self.model(x=x_t, t=t_batch)
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
            group_id = b // self.num_samples

            states = torch.stack([s[b] for s in all_states])
            actions = torch.stack([a[b] for a in all_actions])
            timesteps_tensor = torch.stack([t[b] for t in all_timesteps])
            log_probs = torch.stack([lp[b] for lp in all_log_probs])
            rewards = torch.zeros(len(all_states), device=device)

            traj = Trajectory(
                states=states,
                actions=actions,
                timesteps=timesteps_tensor,
                log_probs=log_probs,
                rewards=rewards,
                final_sample=x_t[b],
            )
            traj.prompt = group_id
            trajectories.append(traj)

        return trajectories

    def _generate_maskgit_trajectories(
        self,
        batch_size: int,
        seq_length: int,
        num_groups: int,
    ) -> list[Trajectory]:
        """Generate MaskGIT trajectories."""
        device = self.accelerator.device
        mask_token = getattr(self.model, "mask_token", self.vocab_size)

        # Initialize with shared mask state per group
        x_init = torch.full(
            (num_groups, seq_length),
            mask_token,
            dtype=torch.long,
            device=device,
        )
        x_t = x_init.repeat_interleave(self.num_samples, dim=0)

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

                # Get mask for positions that are still masked
                mask = x_t == mask_token
                if not mask.any():
                    break

                logits = self.model(x_t)
                probs = F.softmax(logits, dim=-1)

                flat_probs = probs.view(-1, self.vocab_size)
                actions = torch.multinomial(flat_probs, num_samples=1)
                actions = actions.view(batch_size, seq_length)

                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = torch.gather(
                    log_probs, dim=-1, index=actions.unsqueeze(-1)
                ).squeeze(-1)
                # Only count masked positions
                action_log_probs = (action_log_probs * mask.float()).sum(dim=-1)

                all_actions.append(actions.clone())
                all_log_probs.append(action_log_probs.clone())

                # Unmask some positions based on confidence
                confidences = probs.max(dim=-1).values
                confidences = confidences.masked_fill(~mask, -float("inf"))

                # Unmask top-k positions
                num_to_unmask = max(
                    1, int((1 - step / self.num_inference_steps) * mask.sum(dim=1).float().mean())
                )
                _, top_indices = confidences.topk(num_to_unmask, dim=-1)

                # Update x_t
                x_t = x_t.scatter(1, top_indices, actions.gather(1, top_indices))

        self.model.train()

        # Pad if we exited early
        while len(all_states) < self.num_inference_steps:
            all_states.append(x_t.clone())
            all_timesteps.append(torch.ones(batch_size, device=device))
            all_actions.append(x_t.clone())
            all_log_probs.append(torch.zeros(batch_size, device=device))

        trajectories = []
        for b in range(batch_size):
            group_id = b // self.num_samples

            states = torch.stack([s[b] for s in all_states])
            actions = torch.stack([a[b] for a in all_actions])
            timesteps_tensor = torch.stack([t[b] for t in all_timesteps])
            log_probs = torch.stack([lp[b] for lp in all_log_probs])
            rewards = torch.zeros(len(all_states), device=device)

            traj = Trajectory(
                states=states,
                actions=actions,
                timesteps=timesteps_tensor,
                log_probs=log_probs,
                rewards=rewards,
                final_sample=x_t[b],
            )
            traj.prompt = group_id
            trajectories.append(traj)

        return trajectories

    def compute_policy_loss(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute GRPO loss with group-relative advantages.

        Groups trajectories by prompt and computes advantages relative
        to the group mean reward.
        """
        device = self.accelerator.device

        # Group trajectories by prompt
        groups: dict[Any, list[Trajectory]] = {}
        for traj in trajectories:
            group_id = traj.prompt
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(traj)

        total_loss = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)
        num_samples = 0

        for group_id, group_trajs in groups.items():
            # Get rewards for this group
            group_rewards = torch.tensor(
                [t.rewards[-1].item() for t in group_trajs],
                device=device,
            )

            # Compute group-relative advantages
            if len(group_rewards) > 1:
                advantages = normalize_advantages(group_rewards)
            else:
                advantages = torch.zeros_like(group_rewards)

            # Update advantages in trajectories
            for i, traj in enumerate(group_trajs):
                traj.advantages = advantages[i].expand(len(traj.states))

            # Compute policy gradient for each trajectory
            for i, traj in enumerate(group_trajs):
                advantage = advantages[i]

                # Compute total log prob for trajectory
                total_log_prob = traj.log_probs.sum()

                # Policy gradient: -advantage * log_prob
                policy_loss = -advantage * total_log_prob
                total_loss += policy_loss

                # KL penalty
                if self.ref_model is not None:
                    with torch.no_grad():
                        ref_log_prob = torch.tensor(0.0, device=device)
                        for step_idx in range(len(traj.states)):
                            state = traj.states[step_idx].unsqueeze(0)
                            action = traj.actions[step_idx].unsqueeze(0)
                            t = traj.timesteps[step_idx].unsqueeze(0)

                            if self.model_type == "maskgit":
                                ref_logits = self.ref_model(state)
                            else:
                                ref_logits = self.ref_model(x=state, t=t)

                            ref_lp = F.log_softmax(ref_logits, dim=-1)
                            step_ref_lp = (
                                torch.gather(ref_lp, dim=-1, index=action.unsqueeze(-1))
                                .squeeze(-1)
                                .sum()
                            )
                            ref_log_prob += step_ref_lp

                    kl = total_log_prob - ref_log_prob
                    total_kl += kl
                    total_loss += self.config.kl_coeff * kl

                num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples
            total_kl = total_kl / num_samples

        metrics = {
            "loss": total_loss.item(),
            "kl": total_kl.item(),
            "num_groups": len(groups),
        }

        return total_loss, metrics


__all__ = [
    "GRPOTrainer",
]
