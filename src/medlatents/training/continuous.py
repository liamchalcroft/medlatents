"""Trainer for continuous latent generative models (diffusion & flow)."""

from __future__ import annotations

from typing import Literal

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..common.schedules import get_cosine_schedule_with_warmup
from ..data.tokenized_datasets import validate_continuous_latents
from ..diffusion.continuous import ContinuousGaussianDiffusion
from ..flow_matching.continuous import RectifiedFlow
from ..utils import load_state_dict_compat
from .checkpointing import load_checkpoint
from .discrete import _check_gradients_finite, _check_loss_finite


class ContinuousLatentTrainer:
    """Unified trainer for continuous diffusion and flow-matching models."""

    def __init__(
        self,
        model_type: Literal["diffusion", "flow"],
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        args,
        accelerator: Accelerator | None = None,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        log_gradients: bool = False,
    ) -> None:
        self.model_type = model_type
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.log_gradients = log_gradients

        if accelerator is None:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                mixed_precision=getattr(args, "mixed_precision", "fp16"),
                log_with="wandb" if getattr(args, "use_wandb", False) else None,
            )
        else:
            self.accelerator = accelerator

        if hasattr(args, "seed") and args.seed is not None:
            set_seed(args.seed)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
            self.ema.to(self.accelerator.device)

        self.diffusion: ContinuousGaussianDiffusion | None = None
        self.flow: RectifiedFlow | None = None

        prepare_args = [self.model, self.optimizer, self.train_loader]
        if self.val_loader is not None:
            prepare_args.append(self.val_loader)

        prepared = self.accelerator.prepare(*prepare_args)

        if self.val_loader is not None:
            self.model, self.optimizer, self.train_loader, self.val_loader = prepared
        else:
            self.model, self.optimizer, self.train_loader = prepared

        self.global_step = 0

    def _ensure_process(self, batch: torch.Tensor) -> None:
        latent_shape = tuple(batch.shape[1:])
        device = batch.device

        if self.model_type == "diffusion" and self.diffusion is None:
            self.diffusion = ContinuousGaussianDiffusion(
                num_timesteps=getattr(self.args, "diffusion_steps", 1000),
                schedule_type=getattr(self.args, "diffusion_schedule", "cosine"),
                prediction_type=getattr(self.args, "prediction_type", "epsilon"),
                device=device,
            )
        elif self.model_type == "flow" and self.flow is None:
            self.flow = RectifiedFlow(
                latent_shape=latent_shape,
                base_std=getattr(self.args, "base_std", 1.0),
                device=device,
            )

    def _call_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def _log_gradients(self) -> dict:
        if not self.log_gradients:
            return {}
        total_norm = 0.0
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            if len(grad_stats) < 5:
                grad_stats[f"grad/{name}"] = param_norm
        grad_stats["grad/total_norm"] = total_norm**0.5
        return grad_stats

    def compute_loss(self, batch, condition=None):
        """Compute loss based on model type.

        Validates continuous latent contracts before computing loss.
        Expects floating-point tensors in [B, C, H, W] or [B, C, D, H, W] format.
        """
        batch = self._call_batch(batch)

        # Validate continuous latent contract at training boundary
        if batch.ndim >= 4:
            validate_continuous_latents(
                batch,
                expected_channels=None,  # Don't enforce specific channel count
                context=f"ContinuousLatentTrainer.compute_loss ({self.model_type})",
            )

        self._ensure_process(batch)

        if self.model_type == "diffusion":
            assert self.diffusion is not None
            return self.diffusion.compute_loss(self.model, batch, y=condition)
        if self.model_type == "flow":
            assert self.flow is not None
            return self.flow.compute_loss(self.model, batch, y=condition)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                loss = self.compute_loss(batch)

                # Fail-fast on NaN/Inf loss
                _check_loss_finite(loss, self.global_step, context="forward pass")

                self.accelerator.backward(loss)

                # Check gradients for NaN/Inf
                grad_stats = _check_gradients_finite(self.model, self.global_step)

                if self.accelerator.sync_gradients:
                    clip_value = getattr(self.args, "grad_clip", None)
                    if clip_value is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), clip_value)

                lr = get_cosine_schedule_with_warmup(
                    step=self.global_step,
                    warmup_steps=getattr(self.args, "lr_warmup_epochs", 1)
                    * len(self.train_loader)
                    // getattr(self.args, "gradient_accumulation_steps", 1),
                    total_steps=getattr(self.args, "epochs", 1)
                    * len(self.train_loader)
                    // getattr(self.args, "gradient_accumulation_steps", 1),
                    max_lr=self.args.lr,
                    min_lr=getattr(self.args, "min_lr", 0.0),
                )
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            log_items = {"train/loss": loss.item(), "train/lr": lr}
            log_items.update(grad_stats)
            self.accelerator.log(log_items, step=self.global_step)
            progress.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

    def validate(self, step: int | None = None) -> float | None:
        if self.val_loader is None:
            return None

        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._call_batch(batch)
                self._ensure_process(batch)
                if self.model_type == "diffusion":
                    assert self.diffusion is not None
                    loss = self.diffusion.compute_loss(self.model, batch)
                else:
                    assert self.flow is not None
                    loss = self.flow.compute_loss(self.model, batch)
                losses.append(loss.item())

        mean_loss = float(sum(losses) / len(losses)) if losses else None
        if mean_loss is not None:
            self.accelerator.log({"val/loss": mean_loss}, step=step or self.global_step)

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return mean_loss

    def save_checkpoint(
        self, output_dir: str, epoch: int, metric: float = 0.0, wandb_id: str = ""
    ) -> None:
        """Save training checkpoint to output_dir."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pt")

        state = {
            "net": self.accelerator.get_state_dict(self.model),
            "opt": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "wandb": wandb_id,
            "global_step": self.global_step,
        }
        if self.use_ema:
            state["ema"] = self.ema.state_dict()
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and optimizer state from a checkpoint file."""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_state_dict_compat(self.model, state.get("net", state.get("model", {})))
        self.optimizer.load_state_dict(state.get("opt", state.get("optimizer", {})))
        self.global_step = state.get("global_step", 0)
        if self.use_ema and "ema" in state:
            self.ema.load_state_dict(state["ema"])

    def resume_from_checkpoint(self, checkpoint_dir: str, name: str = "") -> int:
        """Resume training from the latest checkpoint in checkpoint_dir."""
        checkpoint = load_checkpoint(checkpoint_dir, name)
        if checkpoint is None:
            return 0

        load_state_dict_compat(self.model, checkpoint.get("net", {}))
        if "opt" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["opt"])
        self.global_step = checkpoint.get("global_step", 0)
        if self.use_ema and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        return checkpoint.get("epoch", 0)


__all__ = ["ContinuousLatentTrainer"]
