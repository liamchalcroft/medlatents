"""Step-by-step Preference Optimization (SPO) for iterative generative models.

SPO applies DPO at each denoising step, allowing fine-grained control over
the generation trajectory. This is particularly effective for diffusion and
flow matching models where each step contributes to the final quality.

The key insight is that preference learning at intermediate steps can guide
the model to stay on high-quality generation paths.

Reference:
- "Step-by-step Preference Optimization for Text-to-Image Generation" (Liang et al., 2024, CVPR 2025)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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
from .base import DPOLoss


@dataclass
class SPOStepOutput:
    """Output from a single SPO step.

    Attributes:
        step_idx: Index of the denoising step
        timestep: Continuous timestep value
        loss: DPO loss at this step
        chosen_state: Selected state to continue from
        accuracy: Whether chosen had higher reward
    """

    step_idx: int
    timestep: float
    loss: Tensor
    chosen_state: Tensor
    accuracy: float


class StepPreferenceModel(nn.Module):
    """Timestep-aware preference model for SPO.

    This model scores intermediate states at each denoising step,
    allowing the model to learn step-specific preferences.

    The architecture adds timestep conditioning to score states
    appropriately at different noise levels.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        vocab_size: int | None = None,
        seq_length: int = 256,
    ) -> None:
        """Initialize step preference model.

        Args:
            hidden_size: Hidden dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            vocab_size: Vocabulary size for discrete tokens
            seq_length: Maximum sequence length
        """
        super().__init__()

        # Use reward model architecture with timestep awareness
        from ..preference.reward_model import RewardModel

        self.reward_model = RewardModel(
            vocab_size=vocab_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            timestep_aware=True,  # Key for SPO
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Score a state at timestep t.

        Args:
            x: Token sequence [batch, seq_len]
            t: Timestep [batch]

        Returns:
            Scores [batch, 1]
        """
        return self.reward_model(x, t)


class SPOTrainer:
    """Step-by-step Preference Optimization trainer.

    SPO operates by:
    1. At each denoising step, generate K candidate states
    2. Score candidates with a step-aware preference model
    3. Apply DPO loss using best vs worst candidates
    4. Randomly select one candidate to continue generation

    This provides fine-grained supervision throughout the generation
    process, leading to better alignment than end-to-end DPO.

    Example:
        >>> # For D3PM diffusion
        >>> trainer = SPOTrainer(
        ...     model=backbone_model,
        ...     config=config,
        ...     diffusion=d3pm,
        ...     step_scorer=step_preference_model,
        ... )
        >>> trainer.train(prompt_dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        config: PostTrainingConfig,
        diffusion: Any | None = None,
        flow_path: Any | None = None,
        step_scorer: nn.Module | None = None,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        num_candidates: int = 4,
        num_steps: int = 50,
    ) -> None:
        """Initialize SPO trainer.

        Args:
            model: Generative model to train
            config: Training configuration
            diffusion: D3PM diffusion process (for diffusion models)
            flow_path: Flow matching path (for flow models)
            step_scorer: Model to score intermediate states
            accelerator: Optional pre-configured accelerator
            vocab_size: Vocabulary size
            num_candidates: Number of candidates per step (K)
            num_steps: Number of generation steps to optimize
        """
        self.model = model
        self.config = config
        self.diffusion = diffusion
        self.flow_path = flow_path
        self.num_candidates = num_candidates
        self.num_steps = num_steps

        # Determine model type
        self.model_type: Literal["diffusion", "flow", "maskgit"] = "diffusion"
        if flow_path is not None:
            self.model_type = "flow"
        elif hasattr(model, "mask_token"):
            self.model_type = "maskgit"

        # Get vocab size
        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
            elif diffusion is not None:
                vocab_size = diffusion.effective_num_classes
            else:
                raise ValueError("vocab_size must be provided")
        self.vocab_size = vocab_size

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

        # Setup step scorer
        if step_scorer is None:
            step_scorer = StepPreferenceModel(
                vocab_size=vocab_size,
                seq_length=getattr(model, "seq_length", 256),
            )
        self.step_scorer = step_scorer

        # Setup optimizer (train both model and scorer)
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(step_scorer.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Setup EMA
        self.ema = None
        if config.use_ema:
            self.ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay)

        # DPO loss
        self.dpo_loss = DPOLoss(
            beta=config.beta,
            loss_type=config.loss_type,
        )

        # Prepare with accelerator
        self.model, self.step_scorer, self.optimizer = self.accelerator.prepare(
            self.model, self.step_scorer, self.optimizer
        )

        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0

    def _get_initial_state(self, batch_size: int, seq_length: int) -> Tensor:
        """Get initial noisy state for generation."""
        device = self.accelerator.device

        if self.model_type == "diffusion" and self.diffusion is not None:
            if self.diffusion.transition_type == "absorbing":
                return torch.full(
                    (batch_size, seq_length),
                    self.diffusion.mask_token,
                    dtype=torch.long,
                    device=device,
                )
            else:
                return torch.randint(
                    0,
                    self.diffusion.num_classes,
                    (batch_size, seq_length),
                    device=device,
                )
        elif self.model_type == "maskgit":
            mask_token = getattr(self.model, "mask_token", self.vocab_size)
            return torch.full(
                (batch_size, seq_length),
                mask_token,
                dtype=torch.long,
                device=device,
            )
        else:
            # Flow: start from mask tokens
            return torch.full(
                (batch_size, seq_length),
                self.vocab_size,
                dtype=torch.long,
                device=device,
            )

    def _generate_candidates(
        self,
        x_t: Tensor,
        t: Tensor,
        num_candidates: int,
    ) -> Tensor:
        """Generate K candidate next states from current state.

        Args:
            x_t: Current state [batch, seq_len]
            t: Current timestep [batch]
            num_candidates: Number of candidates to generate

        Returns:
            Candidates [batch * num_candidates, seq_len]
        """
        batch_size, seq_len = x_t.shape

        # Expand x_t for all candidates
        x_t_expanded = x_t.unsqueeze(1).expand(-1, num_candidates, -1)
        x_t_expanded = x_t_expanded.reshape(batch_size * num_candidates, seq_len)

        t_expanded = t.unsqueeze(1).expand(-1, num_candidates)
        t_expanded = t_expanded.reshape(batch_size * num_candidates)

        # Generate candidates based on model type
        if self.model_type == "diffusion" and self.diffusion is not None:
            # D3PM: sample from reverse process
            candidates = self.diffusion.p_sample(
                self.model, x_t_expanded, t_expanded, temperature=1.0
            )
        elif self.model_type == "maskgit":
            # MaskGIT: sample from logits
            logits = self.model(x_t_expanded)
            probs = F.softmax(logits, dim=-1)
            candidates = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(
                batch_size * num_candidates, seq_len
            )
        else:
            # Flow: sample from model prediction
            logits = self.model(x=x_t_expanded, t=t_expanded)
            probs = F.softmax(logits, dim=-1)
            candidates = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(
                batch_size * num_candidates, seq_len
            )

        return candidates

    def _score_candidates(
        self,
        candidates: Tensor,
        t: Tensor,
        num_candidates: int,
    ) -> Tensor:
        """Score candidates with step-aware preference model.

        Args:
            candidates: Candidate states [batch * num_candidates, seq_len]
            t: Timestep [batch]
            num_candidates: Number of candidates per batch

        Returns:
            Scores [batch, num_candidates]
        """
        batch_size = t.shape[0]

        # Expand t for all candidates
        t_expanded = t.unsqueeze(1).expand(-1, num_candidates)
        t_expanded = t_expanded.reshape(batch_size * num_candidates)

        # Score all candidates
        with torch.no_grad():
            scores = self.step_scorer(candidates, t_expanded)  # [B*K, 1]

        # Reshape to [batch, num_candidates]
        scores = scores.view(batch_size, num_candidates)

        return scores

    def _compute_step_dpo_loss(
        self,
        candidates: Tensor,
        scores: Tensor,
        t: Tensor,
        num_candidates: int,
    ) -> tuple[Tensor, Tensor, float]:
        """Compute DPO loss for a single generation step.

        Args:
            candidates: Candidate states [batch * num_candidates, seq_len]
            scores: Candidate scores [batch, num_candidates]
            t: Timestep [batch]
            num_candidates: Number of candidates

        Returns:
            Tuple of (loss, chosen_state, accuracy)
        """
        batch_size = t.shape[0]
        seq_len = candidates.shape[1]

        # Reshape candidates
        candidates = candidates.view(batch_size, num_candidates, seq_len)

        # Get best and worst candidates
        best_idx = scores.argmax(dim=1)  # [batch]
        worst_idx = scores.argmin(dim=1)  # [batch]

        # Gather chosen and rejected
        batch_indices = torch.arange(batch_size, device=candidates.device)
        chosen = candidates[batch_indices, best_idx]  # [batch, seq_len]
        rejected = candidates[batch_indices, worst_idx]  # [batch, seq_len]

        # Compute log-probs for DPO
        # For SPO, we use the model's prediction probability at this timestep
        if self.model_type == "diffusion" and self.diffusion is not None:
            # For diffusion, use the model's denoising prediction
            # This is approximate - use logits at current state
            with torch.no_grad():
                x_t_chosen = self.diffusion.q_sample(chosen, t)
                x_t_rejected = self.diffusion.q_sample(rejected, t)

            logits_chosen = self.model(x=x_t_chosen, t=t)
            logits_rejected = self.model(x=x_t_rejected, t=t)

            # Log-prob of predicting the chosen/rejected from noisy state
            chosen_logprobs = (
                -F.cross_entropy(
                    logits_chosen.view(-1, self.vocab_size),
                    chosen.view(-1),
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=1)
            )

            rejected_logprobs = (
                -F.cross_entropy(
                    logits_rejected.view(-1, self.vocab_size),
                    rejected.view(-1),
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=1)
            )
        else:
            # For MaskGIT/Flow, use direct prediction
            logits_chosen = (
                self.model(x=chosen, t=t) if self.model_type == "flow" else self.model(chosen)
            )
            logits_rejected = (
                self.model(x=rejected, t=t) if self.model_type == "flow" else self.model(rejected)
            )

            chosen_logprobs = (
                -F.cross_entropy(
                    logits_chosen.view(-1, self.vocab_size),
                    chosen.view(-1),
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=1)
            )

            rejected_logprobs = (
                -F.cross_entropy(
                    logits_rejected.view(-1, self.vocab_size),
                    rejected.view(-1),
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=1)
            )

        # Compute DPO loss (reference-free for simplicity in SPO)
        dpo_output = self.dpo_loss(
            chosen_logprobs=chosen_logprobs,
            rejected_logprobs=rejected_logprobs,
            ref_chosen_logprobs=torch.zeros_like(chosen_logprobs),
            ref_rejected_logprobs=torch.zeros_like(rejected_logprobs),
        )

        # Randomly select one candidate to continue from (not necessarily best)
        random_idx = torch.randint(0, num_candidates, (batch_size,), device=candidates.device)
        continue_state = candidates[batch_indices, random_idx]

        return dpo_output.loss, continue_state, dpo_output.accuracy.item()

    def train_step(self, seq_length: int) -> dict[str, float]:
        """Execute one SPO training step.

        Generates a full trajectory with step-wise DPO optimization.

        Args:
            seq_length: Sequence length for generation

        Returns:
            Dictionary of metrics
        """
        batch_size = self.config.batch_size
        device = self.accelerator.device

        # Initialize state
        x_t = self._get_initial_state(batch_size, seq_length)

        total_loss = 0.0
        total_accuracy = 0.0
        num_steps_computed = 0

        # Generate timestep schedule
        if self.model_type == "diffusion" and self.diffusion is not None:
            timesteps = (
                torch.linspace(self.diffusion.num_timesteps - 1, 0, self.num_steps)
                .long()
                .to(device)
            )
        else:
            timesteps = torch.linspace(1.0, 0.0, self.num_steps).to(device)

        for step_idx, t_val in enumerate(timesteps):
            # Skip last step (t=0)
            if self.model_type == "diffusion":
                if t_val <= 0:
                    continue
                t = torch.full((batch_size,), t_val.item(), dtype=torch.long, device=device)
            else:
                if t_val <= 1e-3:
                    continue
                t = torch.full((batch_size,), t_val.item(), device=device)

            # Generate candidates
            candidates = self._generate_candidates(x_t, t, self.num_candidates)

            # Score candidates
            scores = self._score_candidates(candidates, t, self.num_candidates)

            # Compute DPO loss and get next state
            loss, x_t, accuracy = self._compute_step_dpo_loss(
                candidates, scores, t, self.num_candidates
            )

            total_loss += loss
            total_accuracy += accuracy
            num_steps_computed += 1

        # Average loss over steps
        if num_steps_computed > 0:
            avg_loss = total_loss / num_steps_computed
        else:
            avg_loss = torch.tensor(0.0, device=device)

        # Backward pass
        self.accelerator.backward(avg_loss)

        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        return {
            "loss": avg_loss.item(),
            "accuracy": total_accuracy / max(num_steps_computed, 1),
            "steps_computed": num_steps_computed,
        }

    def train(self, seq_length: int = 256) -> dict[str, list[float]]:
        """Run full SPO training loop.

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

        history = {
            "train_loss": [],
            "train_accuracy": [],
        }

        # Training loop
        self.model.train()
        self.step_scorer.train()
        pbar = tqdm(range(self.config.max_steps), desc="SPO Training")

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
                    loss=f"{metrics['loss']:.4f}",
                    acc=f"{metrics['accuracy']:.4f}",
                )
                history["train_loss"].append(metrics["loss"])
                history["train_accuracy"].append(metrics["accuracy"])

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint("latest")

        pbar.close()

        if self.config.wandb_project:
            self.accelerator.end_training()

        return history

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.config.logdir, f"spo_checkpoint_{name}.pt")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_scorer = self.accelerator.unwrap_model(self.step_scorer)

        checkpoint = {
            "model": unwrapped_model.state_dict(),
            "step_scorer": unwrapped_scorer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "config": self.config.to_dict(),
        }

        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

        self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        # save_checkpoint stores only state_dicts/ints and config.to_dict() (a
        # primitives dict), so the checkpoint contains no arbitrary Python objects.
        checkpoint = torch.load(path, map_location=self.accelerator.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model"])
        self.step_scorer.load_state_dict(checkpoint["step_scorer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["step"]

        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])


__all__ = [
    "SPOTrainer",
    "StepPreferenceModel",
    "SPOStepOutput",
]
