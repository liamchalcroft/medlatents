"""Self-Play Iterative Training (SPIN-style).

Self-play methods improve models by training on their own generations,
using the previous version as a negative example. This creates a
curriculum where the model learns to distinguish its outputs from
ground truth.

Key idea (SPIN):
1. Generate samples from current model
2. Train to prefer ground truth over model generations
3. Iterate with the new model

Reference:
- "Self-Play Fine-Tuning Converts Weak LMs to Strong LMs" (Chen et al., 2024)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

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
from ..dpo.base import DPOLoss

logger = logging.getLogger(__name__)


class SPINTrainer:
    """Self-Play Iterative Training (SPIN) for generative models.

    SPIN improves a model by creating a curriculum of increasingly
    difficult discrimination tasks:

    Iteration 1: Distinguish ground truth from weak model outputs
    Iteration 2: Distinguish ground truth from stronger model outputs
    ...

    This is implemented as iterative DPO where:
    - Chosen = ground truth data
    - Rejected = model's own generations

    Example:
        >>> trainer = SPINTrainer(
        ...     model=model,
        ...     config=config,
        ...     vocab_size=1024,
        ... )
        >>> trainer.train(data_loader, num_iterations=3)
    """

    def __init__(
        self,
        model: nn.Module,
        config: PostTrainingConfig,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        generate_fn: Callable[[nn.Module, int, int], Tensor] | None = None,
        num_generation_steps: int = 50,
        model_type: str = "maskgit",
    ) -> None:
        """Initialize SPIN trainer.

        Args:
            model: Model to train
            config: Training configuration
            accelerator: Optional accelerator
            vocab_size: Vocabulary size
            generate_fn: Custom generation function (model, batch_size, seq_len) -> samples
            num_generation_steps: Steps for generation
            model_type: Model type for default generation
        """
        self.model = model
        self.config = config
        self.model_type = model_type
        self.num_generation_steps = num_generation_steps
        self.generate_fn = generate_fn

        if vocab_size is None:
            if hasattr(model, "vocab_size"):
                vocab_size = model.vocab_size
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

        # DPO loss for preference learning
        self.dpo_loss = DPOLoss(
            beta=config.beta,
            loss_type=config.loss_type,
            reference_free=True,  # SPIN doesn't use reference model
        )

        # Prepare
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        # State
        self.global_step = 0
        self.iteration = 0

    @torch.no_grad()
    def _generate_samples(
        self,
        batch_size: int,
        seq_length: int,
    ) -> Tensor:
        """Generate samples from current model."""
        device = self.accelerator.device

        if self.generate_fn is not None:
            return self.generate_fn(
                self.accelerator.unwrap_model(self.model),
                batch_size,
                seq_length,
            )

        # Default generation based on model type
        self.model.eval()

        if self.model_type == "maskgit":
            mask_token = getattr(self.model, "mask_token", self.vocab_size)
            x = torch.full(
                (batch_size, seq_length),
                mask_token,
                dtype=torch.long,
                device=device,
            )

            if hasattr(self.model, "generate"):
                samples = self.model.generate(x, num_steps=self.num_generation_steps)
            else:
                # Simple iterative unmasking
                for _ in range(self.num_generation_steps):
                    mask = x == mask_token
                    if not mask.any():
                        break
                    logits = self.model(x)
                    probs = F.softmax(logits, dim=-1)
                    samples = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(
                        batch_size, seq_length
                    )

                    # Unmask highest confidence
                    confidences = probs.max(dim=-1).values
                    confidences = confidences.masked_fill(~mask, -float("inf"))
                    _, best_pos = confidences.max(dim=-1)
                    batch_idx = torch.arange(batch_size, device=device)
                    x[batch_idx, best_pos] = samples[batch_idx, best_pos]

                samples = x

        else:
            # Flow/diffusion style
            mask_token = getattr(self.model, "mask_token", self.vocab_size)
            x = torch.full(
                (batch_size, seq_length),
                mask_token,
                dtype=torch.long,
                device=device,
            )

            for step in range(self.num_generation_steps):
                t = torch.full(
                    (batch_size,),
                    step / self.num_generation_steps,
                    device=device,
                )
                logits = self.model(x=x, t=t)
                probs = F.softmax(logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(
                    batch_size, seq_length
                )

            samples = x

        self.model.train()
        return samples

    def _compute_log_probs(
        self,
        samples: Tensor,
    ) -> Tensor:
        """Compute log probabilities for samples."""
        batch_size, seq_length = samples.shape

        if self.model_type == "maskgit":
            # Pseudo-likelihood: mask each position and predict
            total_log_prob = torch.zeros(batch_size, device=samples.device)
            mask_token = getattr(self.model, "mask_token", self.vocab_size)

            # Use random subset for efficiency
            num_positions = min(16, seq_length)
            positions = torch.randperm(seq_length)[:num_positions]

            for pos in positions:
                masked = samples.clone()
                masked[:, pos] = mask_token
                mask = torch.zeros_like(samples, dtype=torch.bool)
                mask[:, pos] = True

                logits = self.model(masked, mask=mask)
                log_probs = F.log_softmax(logits[:, pos], dim=-1)
                token_log_probs = torch.gather(log_probs, -1, samples[:, pos : pos + 1]).squeeze(-1)
                total_log_prob += token_log_probs

            # Scale to full sequence
            total_log_prob = total_log_prob * (seq_length / num_positions)

        else:
            # Flow/diffusion: use t=0 prediction
            t = torch.zeros(batch_size, device=samples.device)
            logits = self.model(x=samples, t=t)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, -1, samples.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum(dim=-1)

        return total_log_prob

    def train_step(
        self,
        real_data: Tensor,
    ) -> dict[str, float]:
        """One SPIN training step.

        Args:
            real_data: Ground truth samples (chosen)

        Returns:
            Metrics dictionary
        """
        batch_size, seq_length = real_data.shape

        # Generate synthetic samples (rejected)
        synthetic = self._generate_samples(batch_size, seq_length)

        # Compute log probs
        real_log_probs = self._compute_log_probs(real_data)
        synthetic_log_probs = self._compute_log_probs(synthetic)

        # DPO loss: prefer real over synthetic
        output = self.dpo_loss(
            chosen_logprobs=real_log_probs,
            rejected_logprobs=synthetic_log_probs,
        )

        # Backward
        self.accelerator.backward(output.loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()

        self.global_step += 1

        return output.to_dict()

    def train_iteration(
        self,
        data_loader: DataLoader,
        num_steps: int | None = None,
    ) -> dict[str, list[float]]:
        """Run one SPIN iteration.

        Args:
            data_loader: DataLoader with ground truth data
            num_steps: Steps for this iteration

        Returns:
            Training history
        """
        num_steps = num_steps or (self.config.max_steps // self.config.num_self_play_iterations)

        data_loader = self.accelerator.prepare(data_loader)
        self.model.train()

        history = {"loss": [], "accuracy": []}

        pbar = tqdm(range(num_steps), desc=f"SPIN Iteration {self.iteration + 1}")
        data_iter = iter(data_loader)

        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            with self.accelerator.accumulate(self.model):
                metrics = self.train_step(batch)

            if step % self.config.log_every == 0:
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    acc=f"{metrics['accuracy']:.4f}",
                )
                history["loss"].append(metrics["loss"])
                history["accuracy"].append(metrics["accuracy"])

        self.iteration += 1
        return history

    def train(
        self,
        data_loader: DataLoader,
        num_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Run full SPIN training with multiple iterations.

        Args:
            data_loader: DataLoader with ground truth data
            num_iterations: Number of self-play iterations

        Returns:
            Complete training history
        """
        num_iterations = num_iterations or self.config.num_self_play_iterations

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

        all_history = {"iterations": []}

        for i in range(num_iterations):
            logger.info(f"\n=== SPIN Iteration {i + 1}/{num_iterations} ===")
            history = self.train_iteration(data_loader)
            all_history["iterations"].append(history)

            self.save_checkpoint(f"spin_iter_{i + 1}")

        if self.config.wandb_project:
            self.accelerator.end_training()

        return all_history

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.config.logdir, f"spin_checkpoint_{name}.pt")

        unwrapped = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "iteration": self.iteration,
            "config": self.config.to_dict(),
        }

        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

        self.accelerator.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        # save_checkpoint stores only state_dicts/ints and config.to_dict() (a
        # primitives dict), so the checkpoint contains no arbitrary Python objects.
        checkpoint = torch.load(path, map_location=self.accelerator.device, weights_only=True)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["step"]
        self.iteration = checkpoint.get("iteration", 0)

        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])


__all__ = [
    "SPINTrainer",
]
