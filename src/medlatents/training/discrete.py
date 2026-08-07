"""Enhanced unified trainer for discrete latent generative models with EMA, logging, and diffusion support.

Features:
- Automatic mixed precision (AMP) with bf16/fp16 support
- Fused AdamW optimizer for CUDA
- EMA (Exponential Moving Average) for stable training
- Gradient monitoring and logging
- Curriculum learning
- Periodic sample generation
- Multi-GPU support with FSDP
"""

import logging
import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from ..common.schedules import get_cosine_schedule_with_warmup
from ..data.tokenized_datasets import validate_discrete_tokens
from ..diffusion import D3PM
from ..flow_matching.discrete import (
    get_loss_function,
    get_path,
    get_source_distribution,
)
from ..flow_matching.evaluate import compute_entropy
from .checkpointing import load_checkpoint, resume_from_checkpoint
from .profiling import TrainingProfiler
from .schedulers import get_mask_ratio

logger = logging.getLogger(__name__)


class NaNLossError(RuntimeError):
    """Raised when training produces NaN or Inf loss."""

    pass


def _check_loss_finite(loss: torch.Tensor, step: int, context: str = "training") -> None:
    """Check that loss is finite. Raises NaNLossError if not.

    Args:
        loss: Loss tensor
        step: Current training step
        context: Context string for error message

    Raises:
        NaNLossError: If loss is NaN or Inf
    """
    if not torch.isfinite(loss).all():
        loss_val = loss.item() if loss.numel() == 1 else loss.detach().cpu().numpy()
        raise NaNLossError(
            f"Non-finite loss detected during {context} at step {step}: {loss_val}. "
            "This typically indicates numerical instability. Check:\n"
            "  1. Learning rate may be too high\n"
            "  2. Gradient clipping value may be too high\n"
            "  3. Input data may contain NaN/Inf values\n"
            "  4. Model weights may have exploded"
        )


def _check_gradients_finite(model: nn.Module, step: int) -> dict:
    """Check gradients for NaN/Inf and return statistics.

    Args:
        model: Model to check
        step: Current training step

    Returns:
        dict with gradient statistics

    Raises:
        NaNLossError: If any gradients are NaN or Inf
    """
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    grad_count = len(grads)

    if grad_count == 0:
        return {"grad_norm/total": 0.0, "grad_count": 0}

    stacked_norms = torch.stack([g.norm(2) for g in grads])
    has_nan = torch.stack([torch.isnan(g).any() for g in grads]).any()
    has_inf = torch.stack([torch.isinf(g).any() for g in grads]).any()

    if has_nan or has_inf:
        nan_count = sum(torch.isnan(g).sum().item() for g in grads)
        inf_count = sum(torch.isinf(g).sum().item() for g in grads)
        raise NaNLossError(
            f"Non-finite gradients at step {step}: {nan_count} NaN, {inf_count} Inf. "
            "This indicates numerical instability in the backward pass."
        )

    total_norm = stacked_norms.norm(2).item()

    return {
        "grad_norm/total": total_norm,
        "grad_count": grad_count,
    }


class DiscreteLatentTrainer:
    """
    Enhanced unified trainer for discrete latent generative models.

    Supports:
    - autoreg: Autoregressive transformer
    - maskgit: Bidirectional transformer with masking (MaskGIT)
    - flow: Discrete flow matching with DiscreteDiT
    - d3pm: D3PM discrete diffusion (DiscreteDiT backbone)
    - bayesian_flow: Bayesian Flow Networks (DiscreteDiT backbone)

    Features:
    - EMA (Exponential Moving Average) for stable training
    - Gradient monitoring and logging
    - Curriculum learning
    - Periodic sample generation
    - Multi-GPU support with FSDP
    """

    def __init__(
        self,
        model_type: Literal["autoreg", "maskgit", "flow", "d3pm", "bayesian_flow"],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        args,
        accelerator: Accelerator | None = None,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        log_gradients: bool = True,
        generate_samples_every: int | None = None,
        enable_profiling: bool = False,
        profile_log_interval: int = 100,
        check_gradients_finite: bool = False,
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.log_gradients = log_gradients
        self.generate_samples_every = generate_samples_every
        self.profiler = TrainingProfiler(enabled=enable_profiling)
        self.profile_log_interval = profile_log_interval
        self.check_gradients_finite = check_gradients_finite

        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(model, mode=compile_mode)
        else:
            self.model = model

        # Setup accelerator with FSDP support if specified
        if accelerator is None:
            fsdp_plugin = None
            if hasattr(args, "use_fsdp") and args.use_fsdp:
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
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                log_with="wandb",
                mixed_precision=getattr(args, "mixed_precision", "fp16"),
                fsdp_plugin=fsdp_plugin,
            )
        else:
            self.accelerator = accelerator

        # Set seed if specified
        if hasattr(args, "seed") and args.seed is not None:
            set_seed(args.seed)

        # Setup optimizer with fused kernels when available (10-30% faster on CUDA)
        # Fused AdamW requires PyTorch 2.0+ and CUDA
        optimizer_kwargs = {
            "lr": args.lr,
            "weight_decay": getattr(args, "weight_decay", 0.01),
        }
        # Only add fused if PyTorch supports it (2.0+)
        use_fused = torch.cuda.is_available() and getattr(args, "fused_optimizer", True)
        if use_fused:
            # Check PyTorch version for fused support
            torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
            if len(torch_version) >= 2 and torch_version >= (2, 0):
                optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

        # Setup EMA
        self.use_ema = use_ema
        if self.use_ema:
            self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
            self.ema.to(self.accelerator.device)

        # Model-specific setup
        self._setup_model_specifics()

        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        # Tracking metrics
        self.global_step = 0

    def _setup_model_specifics(self):
        """Setup model-specific components."""
        if self.model_type == "flow":
            self.path = get_path(self.args.scheduler_type, self.args.scheduler_power)
            self.loss_fn = get_loss_function(self.args.loss_type, self.path)
            special_tokens = getattr(self.model, "special_tokens", None)
            mask_token = special_tokens.mask if special_tokens is not None else None
            self.source_dist = get_source_distribution(
                self.args.source_dist,
                self.args.vocab_size,
                mask_token=mask_token,
            )

        elif self.model_type == "d3pm":
            self.diffusion = D3PM(
                num_classes=self.args.vocab_size,
                num_timesteps=getattr(self.args, "diffusion_steps", 1000),
                schedule_type=getattr(self.args, "diffusion_schedule", "cosine"),
                transition_type=getattr(self.args, "diffusion_transition", "absorbing"),
                hybrid_loss_coeff=getattr(self.args, "diffusion_hybrid_coeff", 0.001),
                device=self.accelerator.device,
            )

        elif self.model_type == "bayesian_flow":
            if not hasattr(self.model, "compute_loss"):
                raise ValueError(
                    "Bayesian flow models must implement a compute_loss(tokens) method. "
                    "Your model class should define: def compute_loss(self, tokens: Tensor) -> Tensor"
                )

    def compute_loss(self, batch, step=None):
        """Compute loss based on model type.

        Validates token contracts before computing loss. All discrete models
        expect tokens in [0, vocab_size) with dtype torch.long.
        """
        # Validate discrete token contract at training boundary
        # For D3PM/flow with absorbing, allow special mask token at vocab_size
        allow_special = self.model_type in ("autoreg", "maskgit", "bayesian_flow")
        validate_discrete_tokens(
            batch,
            vocab_size=self.args.vocab_size,
            allow_special=allow_special,
            context=f"DiscreteLatentTrainer.compute_loss ({self.model_type})",
        )

        if self.model_type == "autoreg":
            return self._compute_autoreg_loss(batch)
        elif self.model_type == "maskgit":
            return self._compute_maskgit_loss(batch, step)
        elif self.model_type == "flow":
            return self._compute_flow_loss(batch)
        elif self.model_type == "d3pm":
            return self._compute_d3pm_loss(batch)
        elif self.model_type == "bayesian_flow":
            return self._compute_bayesian_flow_loss(batch)

    def _compute_autoreg_loss(self, batch):
        """Autoregressive cross-entropy loss."""
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = self.model(x)
        return F.cross_entropy(logits.reshape(-1, self.args.vocab_size), y.reshape(-1))

    def _compute_maskgit_loss(self, batch, step):
        """MaskGIT loss with curriculum learning for mask ratio."""
        # Curriculum learning: start with low mask ratio, increase over time
        if hasattr(self.args, "use_curriculum") and self.args.use_curriculum:
            max_steps = self.args.epochs * len(self.train_loader)
            progress = step / max_steps
            # Start at 0.15, end at 0.60
            curr_mask_ratio = 0.15 + 0.45 * progress
        else:
            progress = step / (self.args.epochs * len(self.train_loader))
            curr_mask_ratio = get_mask_ratio(progress, self.args.mask_schedule)

        logits = self.model(batch, mask_ratio=curr_mask_ratio)
        loss = F.cross_entropy(logits.reshape(-1, self.args.vocab_size), batch.reshape(-1))
        return loss, {"mask_ratio": curr_mask_ratio}

    def _compute_flow_loss(self, batch):
        """Discrete flow matching loss."""
        with torch.no_grad():
            x_0 = self.source_dist.sample_like(batch)
            t = torch.rand(batch.shape[0], device=batch.device) * (1.0 - self.args.time_eps)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=batch)

        logits = self.model(x=path_sample.x_t, t=t)

        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            loss = self.loss_fn(logits.flatten(0, 1), batch.flatten(0, 1))
        else:  # MixturePathGeneralizedKL
            loss = self.loss_fn(logits=logits, x_1=batch, x_t=path_sample.x_t, t=t)

        return loss.mean()

    def _compute_d3pm_loss(self, batch):
        """D3PM diffusion loss."""
        loss = self.diffusion.compute_loss(
            self.model,
            batch,
            loss_type=getattr(self.args, "diffusion_loss_type", "hybrid"),
        )
        return loss

    def _compute_bayesian_flow_loss(self, batch):
        """Bayesian Flow Networks loss."""
        tokens = batch
        if isinstance(batch, dict):
            tokens = batch.get("input_ids")
            if tokens is None:
                raise ValueError(
                    "Batch dict must contain 'input_ids' key for bayesian_flow model. "
                    "Got keys: " + str(list(batch.keys())) + ". "
                    "Ensure your dataset returns {'input_ids': tensor, ...}"
                )
        return self.model.compute_loss(tokens)

    def _log_gradients(self):
        """Log gradient norms for monitoring using vectorized computation."""
        if not self.log_gradients:
            return {}

        # Collect all gradients efficiently
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return {"grad_norm/total": 0.0}

        # Vectorized total norm computation (same as torch.nn.utils.clip_grad_norm_)
        total_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2).item()

        return {"grad_norm/total": total_norm}

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                # Compute loss (with optional profiling)
                with self.profiler.section("forward"):
                    loss_output = self.compute_loss(batch, step=self.global_step)
                if isinstance(loss_output, tuple):
                    loss, extra_metrics = loss_output
                else:
                    loss, extra_metrics = loss_output, {}

                # Check for NaN/Inf loss (fail-fast)
                _check_loss_finite(loss, self.global_step, context="forward pass")

                # Optimization (with optional profiling)
                with self.profiler.section("backward"):
                    self.accelerator.backward(loss)

                # Check gradients for NaN/Inf and log (optional, can be expensive)
                if self.check_gradients_finite:
                    grad_metrics = _check_gradients_finite(self.model, self.global_step)
                else:
                    grad_metrics = self._log_gradients()

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                # Update learning rate with temperature annealing support
                lr = get_cosine_schedule_with_warmup(
                    step=self.global_step,
                    warmup_steps=self.args.lr_warmup_epochs
                    * len(self.train_loader)
                    // self.args.gradient_accumulation_steps,
                    total_steps=self.args.epochs
                    * len(self.train_loader)
                    // self.args.gradient_accumulation_steps,
                    max_lr=self.args.lr,
                    min_lr=self.args.min_lr,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                with self.profiler.section("optimizer"):
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)  # Faster than zeroing

                # Update EMA
                if self.use_ema:
                    self.ema.update()

            # Logging
            log_dict = {"train/loss": loss.item(), "train/lr": lr}
            log_dict.update({f"train/{k}": v for k, v in extra_metrics.items()})
            log_dict.update(grad_metrics)

            # Log profiling metrics periodically
            if self.profiler.enabled and (step + 1) % self.profile_log_interval == 0:
                log_dict.update(self.profiler.get_wandb_metrics())

            self.accelerator.log(log_dict)

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

        # Generate samples periodically
        if (
            self.generate_samples_every is not None
            and (epoch + 1) % self.generate_samples_every == 0
        ):
            self._generate_and_log_samples(epoch)

    def _generate_and_log_samples(self, epoch):
        """Generate and log samples for visualization."""
        self.model.eval()
        with torch.no_grad():
            # Generate a few samples
            num_samples = 4
            seq_len = self.args.seq_length

            if self.use_ema:
                # Temporarily use EMA weights
                self.ema.store()
                self.ema.copy_to()

            # Generate based on model type
            if self.model_type == "autoreg":
                prompt = torch.zeros(
                    (num_samples, 1), dtype=torch.long, device=self.accelerator.device
                )
                samples = self.model.generate(prompt, max_length=seq_len, temperature=0.9)

            elif self.model_type == "maskgit":
                x = torch.full(
                    (num_samples, seq_len),
                    self.model.mask_token,
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                samples = self.model.generate(x, num_steps=12, temperature=0.9)

            elif self.model_type == "d3pm":
                samples = self.diffusion.sample(self.model, (num_samples, seq_len), temperature=0.9)

            elif self.model_type == "bayesian_flow":
                samples = self.model.sample((num_samples, seq_len), temperature=0.9)

            else:
                samples = None

            if self.use_ema:
                self.ema.restore()

            # Log samples (as histogram or other visualization)
            if samples is not None:
                # Log token distribution
                token_dist = torch.bincount(
                    samples.flatten(), minlength=self.args.vocab_size
                ).float()
                token_dist = token_dist / token_dist.sum()

                self.accelerator.log(
                    {
                        f"samples/epoch_{epoch}_token_dist": wandb.Histogram(
                            samples.cpu().numpy().flatten()
                        )
                    }
                )

        self.model.train()

    def validate(self):
        """Run validation."""
        # Use EMA weights for validation if available
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        val_entropy = 0.0 if self.model_type == "flow" else None
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if self.model_type == "autoreg":
                    x = batch[:, :-1]
                    y = batch[:, 1:]
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.reshape(-1, self.args.vocab_size), y.reshape(-1))

                elif self.model_type == "maskgit":
                    mask_ratio = getattr(self.model, "mask_ratio", 0.15)
                    logits = self.model(batch, mask_ratio=mask_ratio)
                    loss = F.cross_entropy(
                        logits.reshape(-1, self.args.vocab_size), batch.reshape(-1)
                    )

                elif self.model_type == "flow":
                    x_0 = self.source_dist.sample_like(batch)
                    t = torch.rand(batch.shape[0], device=batch.device) * (1.0 - self.args.time_eps)
                    path_sample = self.path.sample(t=t, x_0=x_0, x_1=batch)

                    logits = self.model(x=path_sample.x_t, t=t)
                    if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                        loss = self.loss_fn(logits.flatten(0, 1), batch.flatten(0, 1))
                    else:
                        loss = self.loss_fn(logits=logits, x_1=batch, x_t=path_sample.x_t, t=t)

                    val_entropy += compute_entropy(batch)

                elif self.model_type == "d3pm":
                    loss = self.diffusion.compute_loss(self.model, batch)

                elif self.model_type == "bayesian_flow":
                    loss = self.model.compute_loss(batch)

                val_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            self.accelerator.log({"val/loss": float("nan")})
            if self.use_ema:
                self.ema.restore()
            return float("nan")

        val_loss /= num_batches
        log_dict = {"val/loss": val_loss}

        if val_entropy is not None:
            val_entropy /= num_batches
            log_dict["val/entropy"] = val_entropy

        self.accelerator.log(log_dict)

        # Restore training weights if using EMA
        if self.use_ema:
            self.ema.restore()

        return val_loss

    def train(self, resume=False, resume_best=False, reset_wandb=False):
        """Main training loop."""
        start_epoch = 0
        metric_best = float("inf")
        wandb_id = None

        # Load checkpoint if resuming
        if resume or resume_best:
            checkpoint = load_checkpoint(
                self.args.logdir, self.args.name, best=resume_best, weights_only=False
            )
            if checkpoint is not None:
                start_epoch, metric_best, wandb_id = resume_from_checkpoint(
                    checkpoint, self.model, self.optimizer
                )
                # Restore EMA if present
                if self.use_ema and "ema" in checkpoint:
                    self.ema.load_state_dict(checkpoint["ema"])
                logger.info(f"\nResuming from epoch #{start_epoch} with WandB ID {wandb_id}")
            else:
                logger.info("\nNo checkpoints found. Beginning from epoch #0")

        # Initialize wandb
        self.accelerator.init_trackers(
            f"{self.model_type}-discrete-latents",
            config=vars(self.args),
            init_kwargs={
                "wandb": {
                    "entity": getattr(self.args, "wandb_entity", None),
                    "name": self.args.name,
                    "settings": wandb.Settings(start_method="fork"),
                    "resume": "must"
                    if (resume or resume_best) and not reset_wandb and wandb_id
                    else None,
                    "id": wandb_id if (resume or resume_best) and not reset_wandb else None,
                }
            },
        )

        os.makedirs(os.path.join(self.args.logdir, self.args.name), exist_ok=True)

        # Training loop
        for epoch in range(start_epoch, self.args.epochs):
            self.train_epoch(epoch)

            # Validation
            if (epoch + 1) % self.args.val_interval == 0:
                val_loss = self.validate()

                # Save checkpoint
                checkpoint_dict = {
                    "net": self.model.state_dict(),
                    "opt": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "metric": metric_best,
                    "wandb": wandb.run.id,
                    "hparams": getattr(self.args, "hparams", None),
                }

                if self.use_ema:
                    checkpoint_dict["ema"] = self.ema.state_dict()

                checkpoint_path = os.path.join(self.args.logdir, self.args.name, "checkpoint.pt")
                torch.save(checkpoint_dict, checkpoint_path)

                if val_loss < metric_best:
                    metric_best = val_loss
                    best_checkpoint_path = os.path.join(
                        self.args.logdir, self.args.name, "checkpoint_best.pt"
                    )
                    torch.save(checkpoint_dict, best_checkpoint_path)

        self.accelerator.end_training()
