"""Rejection Fine-Tuning (RFT) for quality-based selection.

RFT improves models by generating multiple samples, scoring them with
a reward model, and fine-tuning only on the best samples. This is
simpler than full RL but can still improve sample quality.

Key idea:
1. Generate K samples per prompt
2. Score with reward model
3. Keep top-k best samples
4. Fine-tune on selected samples

Reference:
- "Rejection Sampling Fine-Tuning for LLMs" (Yuan et al., 2023)
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..configs import PostTrainingConfig


class RFTTrainer:
    """Rejection Fine-Tuning trainer.

    RFT is a simple but effective method that:
    1. Generates multiple samples from the model
    2. Selects the best ones using a reward function
    3. Fine-tunes on the selected samples

    This provides quality improvement without the complexity of RL.

    Example:
        >>> trainer = RFTTrainer(
        ...     model=model,
        ...     reward_fn=reward_model,
        ...     config=config,
        ...     num_samples_per_prompt=8,
        ...     top_k=2,
        ... )
        >>> trainer.train(prompt_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[Tensor], Tensor],
        config: PostTrainingConfig,
        accelerator: Accelerator | None = None,
        vocab_size: int | None = None,
        num_samples_per_prompt: int = 8,
        top_k: int | None = None,
        generate_fn: Callable[[nn.Module, int, int], Tensor] | None = None,
        num_generation_steps: int = 50,
        model_type: str = "maskgit",
    ) -> None:
        """Initialize RFT trainer.

        Args:
            model: Model to train
            reward_fn: Function that scores samples (higher = better)
            config: Training configuration
            accelerator: Optional accelerator
            vocab_size: Vocabulary size
            num_samples_per_prompt: Number of samples to generate per prompt
            top_k: Number of top samples to keep (default: num_samples // 2)
            generate_fn: Custom generation function
            num_generation_steps: Steps for generation
            model_type: Model type for generation
        """
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.num_samples = num_samples_per_prompt
        self.top_k = top_k or max(1, num_samples_per_prompt // 2)
        self.generate_fn = generate_fn
        self.num_generation_steps = num_generation_steps
        self.model_type = model_type

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

        # Prepare
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        # State
        self.global_step = 0

    @torch.no_grad()
    def _generate_samples(
        self,
        batch_size: int,
        seq_length: int,
    ) -> Tensor:
        """Generate samples from model."""
        device = self.accelerator.device

        if self.generate_fn is not None:
            return self.generate_fn(
                self.accelerator.unwrap_model(self.model),
                batch_size,
                seq_length,
            )

        self.model.eval()

        # Initialize
        mask_token = getattr(self.model, "mask_token", self.vocab_size)
        x = torch.full(
            (batch_size, seq_length),
            mask_token,
            dtype=torch.long,
            device=device,
        )

        if self.model_type == "maskgit" and hasattr(self.model, "generate"):
            samples = self.model.generate(x, num_steps=self.num_generation_steps)
        else:
            # Generic iterative generation
            for step in range(self.num_generation_steps):
                if self.model_type == "maskgit":
                    logits = self.model(x)
                else:
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

    @torch.no_grad()
    def _generate_and_select(
        self,
        num_prompts: int,
        seq_length: int,
    ) -> Tensor:
        """Generate samples, score, and select top-k.

        Args:
            num_prompts: Number of "prompts" (parallel generations)
            seq_length: Sequence length

        Returns:
            Selected top samples
        """
        total_samples = num_prompts * self.num_samples

        # Generate all samples
        all_samples = self._generate_samples(total_samples, seq_length)

        # Reshape for per-prompt grouping
        all_samples = all_samples.view(num_prompts, self.num_samples, seq_length)

        # Score each sample
        scores = []
        for prompt_idx in range(num_prompts):
            prompt_samples = all_samples[prompt_idx]  # [num_samples, seq_len]
            prompt_scores = self.reward_fn(prompt_samples)  # [num_samples]
            scores.append(prompt_scores)

        scores = torch.stack(scores)  # [num_prompts, num_samples]

        # Select top-k per prompt
        _, top_indices = scores.topk(self.top_k, dim=1)  # [num_prompts, top_k]

        # Gather top samples
        selected = []
        for prompt_idx in range(num_prompts):
            for k in range(self.top_k):
                sample_idx = top_indices[prompt_idx, k]
                selected.append(all_samples[prompt_idx, sample_idx])

        return torch.stack(selected)  # [num_prompts * top_k, seq_len]

    def _compute_loss(
        self,
        samples: Tensor,
    ) -> Tensor:
        """Compute fine-tuning loss on selected samples.

        Uses negative log-likelihood to encourage generating
        high-quality samples.
        """
        batch_size, seq_length = samples.shape

        if self.model_type == "maskgit":
            # Train with random masking
            mask_token = getattr(self.model, "mask_token", self.vocab_size)
            mask_ratio = 0.15

            mask = torch.rand(batch_size, seq_length, device=samples.device) < mask_ratio
            if not mask.any():
                # Ensure at least one mask
                mask[:, 0] = True

            masked = samples.clone()
            masked[mask] = mask_token

            logits = self.model(masked, mask=mask)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                samples.view(-1),
            )

        else:
            # Flow-style: train at random timestep
            t = torch.rand(batch_size, device=samples.device)
            logits = self.model(x=samples, t=t)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                samples.view(-1),
            )

        return loss

    def train_step(
        self,
        seq_length: int,
        num_prompts: int,
    ) -> dict[str, float]:
        """One RFT training step.

        1. Generate samples
        2. Select top-k by reward
        3. Fine-tune on selected

        Args:
            seq_length: Sequence length
            num_prompts: Number of parallel generations

        Returns:
            Metrics dictionary
        """
        # Generate and select
        selected_samples = self._generate_and_select(num_prompts, seq_length)

        # Compute loss
        loss = self._compute_loss(selected_samples)

        # Backward
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.ema is not None:
            self.ema.update()

        self.global_step += 1

        return {
            "loss": loss.item(),
            "num_selected": len(selected_samples),
        }

    def train(
        self,
        seq_length: int = 256,
        num_prompts_per_step: int = 4,
    ) -> dict[str, list[float]]:
        """Run RFT training.

        Args:
            seq_length: Sequence length for generation
            num_prompts_per_step: Prompts per training step

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

        history = {"loss": []}

        self.model.train()
        pbar = tqdm(range(self.config.max_steps), desc="RFT Training")

        for step in pbar:
            with self.accelerator.accumulate(self.model):
                metrics = self.train_step(seq_length, num_prompts_per_step)

            if step % self.config.log_every == 0:
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
                history["loss"].append(metrics["loss"])

                if self.accelerator.is_main_process:
                    self.accelerator.log(metrics, step=step)

            if step % self.config.save_every == 0:
                self.save_checkpoint("latest")

        pbar.close()

        if self.config.wandb_project:
            self.accelerator.end_training()

        return history

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.config.logdir, f"rft_checkpoint_{name}.pt")

        unwrapped = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
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

        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])


__all__ = [
    "RFTTrainer",
]
