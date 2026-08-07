"""Rectified Flow Reflow for straighter trajectories.

Reflow is an iterative procedure that trains a flow model on pairs generated
by a previous flow model, producing straighter (more linear) trajectories
that can be sampled with fewer steps.

Key ideas:
1. Generate (noise, data) pairs by running the flow forward and backward
2. Train a new flow model on these pairs
3. Repeat to get progressively straighter trajectories

After reflow, few-step or even one-step generation becomes possible.

References:
- "Flow Straight and Fast" (Liu et al., 2022)
- "Rectified Flow++" (improved reflow techniques)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..configs import PostTrainingConfig

logger = logging.getLogger(__name__)


class ReflowPairGenerator:
    """Generate (noise, data) pairs for reflow training.

    Uses the current flow model to generate aligned pairs that represent
    straight-line trajectories in probability space.
    """

    def __init__(
        self,
        model: nn.Module,
        path: Any,  # MixtureDiscreteProbPath or similar
        vocab_size: int,
        device: torch.device,
        num_inference_steps: int = 50,
    ) -> None:
        """Initialize pair generator.

        Args:
            model: Current flow model
            path: Flow probability path
            vocab_size: Vocabulary size for discrete models
            device: Computation device
            num_inference_steps: Steps for generation
        """
        self.model = model
        self.path = path
        self.vocab_size = vocab_size
        self.device = device
        self.num_inference_steps = num_inference_steps

    @torch.no_grad()
    def generate_pairs(
        self,
        data_samples: Tensor,
        batch_size: int = 32,
    ) -> tuple[Tensor, Tensor]:
        """Generate (noise, data) pairs from real data.

        For each data sample x_1:
        1. Sample noise x_0 from source distribution
        2. Generate trajectory: x_0 -> x_1' (using model)
        3. Store pair (x_0, x_1)

        The key insight is that we use the REAL x_1 as target,
        not the model's generated x_1', creating straighter paths.

        Args:
            data_samples: Real data samples [N, seq_len]
            batch_size: Batch size for generation

        Returns:
            Tuple of (noise_samples, data_samples) for training
        """
        self.model.eval()
        all_noise = []
        all_data = []

        num_samples = data_samples.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating reflow pairs"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data_samples[start_idx:end_idx].to(self.device)
            current_batch_size = batch_data.shape[0]
            seq_len = batch_data.shape[1]

            # Sample source noise (mask tokens for discrete)
            mask_token = self.vocab_size  # Common convention
            x_0 = torch.full(
                (current_batch_size, seq_len),
                mask_token,
                dtype=torch.long,
                device=self.device,
            )

            all_noise.append(x_0.cpu())
            all_data.append(batch_data.cpu())

        return torch.cat(all_noise), torch.cat(all_data)


class ReflowTrainer:
    """Iterative reflow training for flow matching models.

    Implements the reflow procedure:
    1. Generate pairs from current model
    2. Train on pairs to get straighter trajectories
    3. Repeat

    Each iteration produces more linear trajectories, enabling
    faster sampling with fewer steps.

    Example:
        >>> trainer = ReflowTrainer(
        ...     model=flow_model,
        ...     config=config,
        ...     path=flow_path,
        ...     vocab_size=1024,
        ... )
        >>> # Run one reflow iteration
        >>> trainer.reflow_iteration(data_loader)
        >>>
        >>> # Run multiple iterations
        >>> for i in range(3):
        ...     trainer.reflow_iteration(data_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: PostTrainingConfig,
        path: Any,  # MixtureDiscreteProbPath
        vocab_size: int,
        accelerator: Accelerator | None = None,
        num_inference_steps: int = 50,
    ) -> None:
        """Initialize reflow trainer.

        Args:
            model: Flow model to train
            config: Training configuration
            path: Flow probability path
            vocab_size: Vocabulary size
            accelerator: Optional accelerator
            num_inference_steps: Inference steps for pair generation
        """
        self.model = model
        self.config = config
        self.path = path
        self.vocab_size = vocab_size
        self.num_inference_steps = num_inference_steps

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

        # Prepare with accelerator
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        if self.ema is not None:
            self.ema.to(self.accelerator.device)

        # Pair generator
        self.pair_generator = ReflowPairGenerator(
            model=self.accelerator.unwrap_model(self.model),
            path=path,
            vocab_size=vocab_size,
            device=self.accelerator.device,
            num_inference_steps=num_inference_steps,
        )

        # State
        self.global_step = 0
        self.reflow_iteration_count = 0

    def _collect_data_samples(
        self,
        data_loader: DataLoader,
        max_samples: int | None = None,
    ) -> Tensor:
        """Collect data samples from loader."""
        all_samples = []
        total = 0

        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            all_samples.append(batch.cpu())
            total += batch.shape[0]

            if max_samples is not None and total >= max_samples:
                break

        samples = torch.cat(all_samples)
        if max_samples is not None:
            samples = samples[:max_samples]

        return samples

    def train_on_pairs(
        self,
        noise_samples: Tensor,
        data_samples: Tensor,
        num_steps: int | None = None,
    ) -> dict[str, list[float]]:
        """Train on generated (noise, data) pairs.

        Args:
            noise_samples: Source noise [N, seq_len]
            data_samples: Target data [N, seq_len]
            num_steps: Training steps (uses config if None)

        Returns:
            Training history
        """
        num_steps = num_steps or self.config.max_steps // self.config.num_reflow_iterations

        # Create dataset and loader
        dataset = TensorDataset(noise_samples, data_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        loader = self.accelerator.prepare(loader)

        history = {"loss": []}

        self.model.train()
        pbar = tqdm(range(num_steps), desc=f"Reflow iteration {self.reflow_iteration_count}")
        data_iter = iter(loader)

        for step in pbar:
            # Get batch
            try:
                x_0, x_1 = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_0, x_1 = next(data_iter)

            # Sample timestep
            batch_size = x_0.shape[0]
            t = torch.rand(batch_size, device=x_0.device)

            # Interpolate along path
            path_sample = self.path.sample(t, x_0, x_1)
            x_t = path_sample.x_t

            # Forward pass
            logits = self.model(x=x_t, t=t)

            # Loss: cross-entropy to predict x_1
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                x_1.reshape(-1),
            )

            # Backward
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.ema is not None:
                self.ema.update()

            self.global_step += 1

            # Logging
            if step % self.config.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                history["loss"].append(loss.item())

        return history

    def reflow_iteration(
        self,
        data_loader: DataLoader,
        max_pairs: int | None = None,
        num_train_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run one complete reflow iteration.

        1. Collect data samples
        2. Generate (noise, data) pairs
        3. Train on pairs

        Args:
            data_loader: DataLoader with training data
            max_pairs: Maximum pairs to generate
            num_train_steps: Training steps for this iteration

        Returns:
            Dictionary with iteration metrics
        """
        max_pairs = max_pairs or self.config.pairs_per_iteration

        # 1. Collect data
        data_samples = self._collect_data_samples(data_loader, max_pairs)

        # 2. Generate pairs
        noise_samples, data_samples = self.pair_generator.generate_pairs(
            data_samples,
            batch_size=self.config.batch_size,
        )

        # 3. Train on pairs
        history = self.train_on_pairs(noise_samples, data_samples, num_train_steps)

        self.reflow_iteration_count += 1

        # Update pair generator model
        self.pair_generator.model = self.accelerator.unwrap_model(self.model)

        return {
            "iteration": self.reflow_iteration_count,
            "num_pairs": len(data_samples),
            "history": history,
        }

    def train(
        self,
        data_loader: DataLoader,
        num_iterations: int | None = None,
    ) -> dict[str, list[Any]]:
        """Run full reflow training with multiple iterations.

        Args:
            data_loader: DataLoader with training data
            num_iterations: Number of reflow iterations

        Returns:
            Training history across all iterations
        """
        num_iterations = num_iterations or self.config.num_reflow_iterations

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
            logger.info(f"\n=== Reflow Iteration {i + 1}/{num_iterations} ===")

            iteration_result = self.reflow_iteration(data_loader)
            all_history["iterations"].append(iteration_result)

            # Save checkpoint after each iteration
            self.save_checkpoint(f"reflow_iter_{i + 1}")

        if self.config.wandb_project:
            self.accelerator.end_training()

        return all_history

    def save_checkpoint(self, name: str = "latest") -> None:
        """Save checkpoint."""
        checkpoint_path = os.path.join(self.config.logdir, f"reflow_checkpoint_{name}.pt")

        unwrapped = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "model": unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.global_step,
            "reflow_iteration": self.reflow_iteration_count,
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
        self.reflow_iteration_count = checkpoint.get("reflow_iteration", 0)

        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])


__all__ = [
    "ReflowTrainer",
    "ReflowPairGenerator",
]
