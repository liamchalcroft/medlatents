"""Train generation models on 3D MedMNIST tokenized data.

Supports all model types with both discrete and continuous tokenizers:

Discrete tokenizers (VQ, FSQ, LFQ, RESFSQ):
- transformer: Autoregressive transformer (causal, left-to-right)
- maskgit: MaskGIT bidirectional transformer (parallel, iterative)
- flow: Discrete flow matching (continuous-time discrete flows)
- d3pm: D3PM discrete diffusion (absorbing/uniform transitions)
- bayesian_flow: Bayesian Flow Networks (information-theoretic)

Continuous tokenizers (VAE):
- diffusion: Continuous Gaussian diffusion

Usage:
    # Autoregressive transformer on 3D tokens
    python examples/train_medmnist3d_tokens.py \
        --tokens_path /path/to/tokenized_organmnist3d.npz \
        --model transformer \
        --vocab_size 512 --seq_len 512 --spatial_shape 8 8 8 --epochs 100

    # D3PM on 3D tokens
    python examples/train_medmnist3d_tokens.py \
        --tokens_path /path/to/tokenized_organmnist3d.npz \
        --model d3pm \
        --vocab_size 512 --seq_len 512 --epochs 100

    # Continuous diffusion on 3D VAE latents
    python examples/train_medmnist3d_tokens.py \
        --tokens_path /path/to/continuous_organmnist3d.npz \
        --model diffusion \
        --latent_channels 4 --latent_shape 4 8 8 8 --epochs 200
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from medlatents.autoregressive import AutoregressiveTransformer
from medlatents.bayesian_flow import BayesianFlowTransformer
from medlatents.common.schedules import get_cosine_schedule_with_warmup
from medlatents.diffusion.continuous import ContinuousGaussianDiffusion
from medlatents.diffusion.d3pm import D3PM
from medlatents.flow_matching.discrete import (
    get_loss_function,
    get_path,
    get_source_distribution,
)
from medlatents.maskgit import MaskGIT
from medlatents.networks.diffusion_transformer import ContinuousDiT, DiscreteDiT
from medlatents.training.discrete import _check_loss_finite
from medlatents.utils import SpecialTokenIds, load_state_dict_compat

# Model types that use discrete tokens
DISCRETE_MODELS = ["transformer", "maskgit", "flow", "d3pm", "bayesian_flow"]
# Model types that use continuous latents
CONTINUOUS_MODELS = ["diffusion"]


class MedMNIST3DTokenDataset(Dataset):
    """Dataset for loading pre-tokenized MedMNIST 3D data from NPZ files."""

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        token_type: Literal["discrete", "continuous"] = "discrete",
    ):
        self.split = split
        self.token_type = token_type
        self.max_token: int | None = None

        data = np.load(npz_path)
        if split == "train":
            self.tokens = torch.from_numpy(data["train_tokens"])
        elif split == "val":
            self.tokens = torch.from_numpy(data["val_tokens"])
        elif split == "test":
            self.tokens = torch.from_numpy(data["test_tokens"])
        else:
            raise ValueError(f"Unknown split: {split}")

        if token_type == "continuous":
            self.tokens = self.tokens.float()
        else:
            self.tokens = self.tokens.long()
            if self.tokens.numel() > 0:
                self.max_token = int(self.tokens.max().item())

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        token = self.tokens[idx]
        if self.token_type == "discrete" and token.ndim > 1:
            token = token.reshape(-1)
        return token


class MedMNIST3DContinuousDataset(Dataset):
    """Dataset for loading continuous latents from NPZ files."""

    def __init__(self, npz_path: str, split: str = "train"):
        self.split = split

        data = np.load(npz_path)
        if split == "train":
            self.latents = torch.from_numpy(data["train_latents"])
        elif split == "val":
            self.latents = torch.from_numpy(data["val_latents"])
        elif split == "test":
            self.latents = torch.from_numpy(data["test_latents"])
        else:
            raise ValueError(f"Unknown split: {split}")

        self.latents = self.latents.float()

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.latents[idx]


def sequence_to_indices_3d(seq: torch.Tensor, H: int, W: int, D: int) -> torch.Tensor:
    """Convert flattened sequence (B, H*W*D) back to 3D indices (B, H, W, D)."""
    B, _ = seq.shape
    return seq.view(B, H, W, D)


def _peek_sample(dataset: Dataset) -> torch.Tensor:
    """Get first sample tensor from Dataset or Subset."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    sample = dataset[0]
    if isinstance(sample, tuple):
        sample = sample[0]
    return sample


def _dataset_max_token(dataset: Dataset) -> int:
    base_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    max_token = getattr(base_dataset, "max_token", None)
    if max_token is not None:
        return int(max_token)
    sample = _peek_sample(dataset)
    return int(sample.max().item())


def _infer_discrete_data_contract(train_dataset: Dataset, val_dataset: Dataset) -> tuple[int, int]:
    """Infer sequence length and minimum required vocab size from datasets."""
    train_sample = _peek_sample(train_dataset)
    if train_sample.ndim != 1:
        raise ValueError(
            f"Expected flattened discrete tokens [seq_len], got shape {tuple(train_sample.shape)}"
        )
    seq_len = int(train_sample.numel())

    max_token = _dataset_max_token(train_dataset)
    val_sample = _peek_sample(val_dataset)
    if val_sample.ndim != 1:
        raise ValueError(
            f"Expected flattened discrete tokens [seq_len], got shape {tuple(val_sample.shape)}"
        )
    max_token = max(max_token, _dataset_max_token(val_dataset))
    inferred_vocab = max_token + 1
    return seq_len, inferred_vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Train generation models on 3D MedMNIST tokens")

    parser.add_argument(
        "--tokens_path", type=str, required=True, help="Path to NPZ file with tokenized data"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "maskgit", "flow", "d3pm", "bayesian_flow", "diffusion"],
        default="transformer",
        help="Model type to train",
    )

    # Discrete model parameters
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--spatial_shape", type=int, nargs="+", default=[8, 8, 8])

    # Continuous model parameters
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_shape", type=int, nargs="+", default=[4, 8, 8, 8])

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Validation and sampling
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--generate_samples_every", type=int, default=None)

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)

    # General settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Exponential Moving Average weights",
    )
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="medmnist3d-tokens")

    # Checkpointing
    parser.add_argument("--logdir", type=str, default="checkpoints")
    parser.add_argument("--name", type=str, default="medmnist3d")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_best", action="store_true")

    # Continuous diffusion parameters
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--diffusion_schedule", type=str, default="cosine")

    # MaskGIT parameters
    parser.add_argument(
        "--use_curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use progressive mask-ratio curriculum for MaskGIT training",
    )

    # Flow matching parameters
    parser.add_argument("--scheduler_type", type=str, default="polynomial")
    parser.add_argument("--scheduler_power", type=float, default=2.0)
    parser.add_argument("--source_dist", type=str, default="uniform", choices=["uniform", "mask"])
    parser.add_argument("--flow_loss_type", type=str, default="cross_entropy")
    parser.add_argument("--time_eps", type=float, default=1e-3)

    # D3PM parameters
    parser.add_argument("--d3pm_transition", type=str, default="absorbing")
    parser.add_argument("--d3pm_loss_type", type=str, default="hybrid")
    parser.add_argument("--d3pm_hybrid_coeff", type=float, default=0.001)

    # Bayesian Flow parameters
    parser.add_argument("--bfn_beta", type=float, default=1.0)
    parser.add_argument("--bfn_num_steps", type=int, default=1000)
    parser.add_argument("--bfn_loss_type", type=str, default="discrete")

    # System
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Enable pinned host memory for DataLoader",
    )

    return parser.parse_args()


def create_discrete_model(args, special_tokens: SpecialTokenIds | None = None) -> nn.Module:
    """Create discrete generation model based on model type."""
    if args.model == "transformer":
        return AutoregressiveTransformer(
            seq_length=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            special_tokens=special_tokens,
            gradient_checkpointing=True,
        )
    elif args.model == "maskgit":
        return MaskGIT(
            seq_length=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            special_tokens=special_tokens,
            gradient_checkpointing=True,
        )
    elif args.model == "flow":
        return DiscreteDiT(
            seq_length=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            masked=args.source_dist == "mask",
            gradient_checkpointing=True,
            allow_dynamic_seq_length=True,
        )
    elif args.model == "d3pm":
        return DiscreteDiT(
            seq_length=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            masked=args.d3pm_transition == "absorbing",
            gradient_checkpointing=True,
            allow_dynamic_seq_length=True,
        )
    elif args.model == "bayesian_flow":
        return BayesianFlowTransformer(
            seq_length=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            num_steps=args.bfn_num_steps,
            beta=args.bfn_beta,
            gradient_checkpointing=True,
            special_tokens=special_tokens,
        )
    else:
        raise ValueError(f"Unknown discrete model: {args.model}")


def create_continuous_model(args) -> nn.Module:
    """Create continuous generation model (diffusion)."""
    latent_dim = args.latent_channels
    spatial_shape = tuple(args.latent_shape[1:])
    seq_length = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

    return ContinuousDiT(
        seq_length=seq_length,
        in_channels=latent_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        gradient_checkpointing=True,
    )


class BaseTrainer:
    """Base trainer class with common functionality."""

    def __init__(self, model, train_loader, val_loader, args, accelerator):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.accelerator = accelerator
        self.global_step = 0

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.use_ema = args.use_ema
        if self.use_ema:
            from torch_ema import ExponentialMovingAverage

            self.ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    def _prepare(self):
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        if self.use_ema:
            self.ema.to(self.accelerator.device)

    def _get_lr(self) -> float:
        return get_cosine_schedule_with_warmup(
            step=self.global_step,
            warmup_steps=self.args.epochs * len(self.train_loader) // 20,
            total_steps=self.args.epochs * len(self.train_loader),
            max_lr=self.args.lr,
            min_lr=self.args.min_lr,
        )

    def _update_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def save_checkpoint(self, epoch: int, is_best: bool = False, metric: float = float("inf")):
        checkpoint = {
            "net": self.accelerator.get_state_dict(self.model),
            "opt": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric": metric,
            "global_step": self.global_step,
            "args": vars(self.args),
        }
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()

        output_dir = os.path.join(self.args.logdir, self.args.name)
        os.makedirs(output_dir, exist_ok=True)

        filename = "checkpoint_best.pt" if is_best else "checkpoint.pt"
        torch.save(checkpoint, os.path.join(output_dir, filename))

    def train_epoch(self, epoch: int) -> float:
        raise NotImplementedError

    def validate(self) -> float:
        raise NotImplementedError

    def generate_samples(self, epoch: int, num_samples: int = 2):
        raise NotImplementedError


class TransformerTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, args, accelerator):
        super().__init__(model, train_loader, val_loader, args, accelerator)
        self._prepare()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                x = batch[:, :-1]
                y = batch[:, 1:]
                logits = self.model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                _check_loss_finite(loss, self.global_step, "training")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                lr = self._get_lr()
                self._update_lr(lr)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            self.accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=self.global_step)
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / max(num_batches, 1)

    @torch.inference_mode()
    def validate(self) -> float:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            val_loss += loss.item()
            num_batches += 1

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return val_loss / max(num_batches, 1)

    @torch.inference_mode()
    def generate_samples(self, epoch: int, num_samples: int = 2) -> torch.Tensor:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        device = self.accelerator.device

        bos_token = (
            self.model.special_tokens.bos
            if getattr(self.model, "special_tokens", None) is not None
            else 0
        )
        prompt = torch.full((num_samples, 1), bos_token, dtype=torch.long, device=device)
        samples = self.model.generate(prompt, max_length=self.args.seq_len, temperature=0.9)

        H, W, D = self.args.spatial_shape
        indices_3d = sequence_to_indices_3d(samples, H, W, D)

        if self.use_ema:
            self.ema.restore()

        self.model.train()

        if self.accelerator.is_main_process and self.args.use_wandb:
            mid_slice = indices_3d[0, H // 2, :, :].cpu().numpy()
            wandb.log({f"samples/epoch_{epoch}": wandb.Image(mid_slice)})

        return indices_3d


class MaskGITTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, args, accelerator):
        super().__init__(model, train_loader, val_loader, args, accelerator)
        self._prepare()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                max_steps = self.args.epochs * len(self.train_loader)
                prog = self.global_step / max_steps
                mask_ratio = 0.15 + 0.45 * prog if self.args.use_curriculum else 0.15

                logits = self.model(batch, mask_ratio=mask_ratio)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.reshape(-1))

                _check_loss_finite(loss, self.global_step, "training")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                lr = self._get_lr()
                self._update_lr(lr)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            self.accelerator.log(
                {"train/loss": loss.item(), "train/lr": lr, "train/mask_ratio": mask_ratio},
                step=self.global_step,
            )
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / max(num_batches, 1)

    @torch.inference_mode()
    def validate(self) -> float:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            logits = self.model(batch, mask_ratio=0.15)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.reshape(-1))
            val_loss += loss.item()
            num_batches += 1

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return val_loss / max(num_batches, 1)

    @torch.inference_mode()
    def generate_samples(self, epoch: int, num_samples: int = 2) -> torch.Tensor:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        device = self.accelerator.device

        x = torch.full(
            (num_samples, self.args.seq_len),
            self.model.mask_token,
            dtype=torch.long,
            device=device,
        )
        samples = self.model.generate(x, num_steps=12, temperature=0.9)

        H, W, D = self.args.spatial_shape
        indices_3d = sequence_to_indices_3d(samples, H, W, D)

        if self.use_ema:
            self.ema.restore()

        self.model.train()

        if self.accelerator.is_main_process and self.args.use_wandb:
            mid_slice = indices_3d[0, H // 2, :, :].cpu().numpy()
            wandb.log({f"samples/epoch_{epoch}": wandb.Image(mid_slice)})

        return indices_3d


class FlowMatchingTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, args, accelerator):
        super().__init__(model, train_loader, val_loader, args, accelerator)
        self.path = get_path(args.scheduler_type, args.scheduler_power)
        self.loss_fn = get_loss_function(args.flow_loss_type, self.path)
        self.source_dist = get_source_distribution(args.source_dist, args.vocab_size)
        self._prepare()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                with torch.no_grad():
                    x_0 = self.source_dist.sample_like(batch)
                    t = torch.rand(batch.shape[0], device=batch.device) * (1.0 - self.args.time_eps)
                    path_sample = self.path.sample(t=t, x_0=x_0, x_1=batch)

                logits = self.model(x=path_sample.x_t, t=t)

                if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                    loss = self.loss_fn(logits.flatten(0, 1), batch.flatten(0, 1))
                else:
                    loss = self.loss_fn(logits=logits, x_1=batch, x_t=path_sample.x_t, t=t)
                loss = loss.mean()

                _check_loss_finite(loss, self.global_step, "training")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                lr = self._get_lr()
                self._update_lr(lr)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            self.accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=self.global_step)
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / max(num_batches, 1)

    @torch.inference_mode()
    def validate(self) -> float:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x_0 = self.source_dist.sample_like(batch)
            t = torch.rand(batch.shape[0], device=batch.device) * (1.0 - self.args.time_eps)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=batch)

            logits = self.model(x=path_sample.x_t, t=t)
            if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                loss = self.loss_fn(logits.flatten(0, 1), batch.flatten(0, 1))
            else:
                loss = self.loss_fn(logits=logits, x_1=batch, x_t=path_sample.x_t, t=t)
            val_loss += loss.mean().item()
            num_batches += 1

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return val_loss / max(num_batches, 1)

    @torch.inference_mode()
    def generate_samples(self, epoch: int, num_samples: int = 2) -> torch.Tensor:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        device = self.accelerator.device

        from medlatents.flow_matching.core import MixtureDiscreteEulerSolver, ModelWrapper

        class WrappedModel(ModelWrapper):
            @torch.no_grad()
            def forward(self, x, t, **extras):
                return torch.softmax(self.model(x=x, t=t, **extras).float(), dim=-1)

        wrapped_model = WrappedModel(model=self.model)
        add_token = 1 if getattr(self.source_dist, "masked", False) else 0
        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model,
            path=self.path,
            vocabulary_size=self.args.vocab_size + add_token,
        )

        x_init = self.source_dist.sample((num_samples, self.args.seq_len), device)
        samples = solver.sample(
            x_init=x_init,
            step_size=1 / 100,
            verbose=False,
            time_grid=torch.tensor([0.0, 1.0 - self.args.time_eps], device=device),
        )

        H, W, D = self.args.spatial_shape
        indices_3d = sequence_to_indices_3d(samples, H, W, D)

        if self.use_ema:
            self.ema.restore()

        self.model.train()

        if self.accelerator.is_main_process and self.args.use_wandb:
            mid_slice = indices_3d[0, H // 2, :, :].cpu().numpy()
            wandb.log({f"samples/epoch_{epoch}": wandb.Image(mid_slice)})

        return indices_3d


class D3PMTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, args, accelerator):
        super().__init__(model, train_loader, val_loader, args, accelerator)
        self.d3pm = D3PM(
            num_classes=args.vocab_size,
            num_timesteps=args.diffusion_steps,
            schedule_type=args.diffusion_schedule,
            transition_type=args.d3pm_transition,
            hybrid_loss_coeff=args.d3pm_hybrid_coeff,
            device=accelerator.device,
        )
        self._prepare()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                loss = self.d3pm.compute_loss(self.model, batch, loss_type=self.args.d3pm_loss_type)

                _check_loss_finite(loss, self.global_step, "training")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                lr = self._get_lr()
                self._update_lr(lr)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            self.accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=self.global_step)
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / max(num_batches, 1)

    @torch.inference_mode()
    def validate(self) -> float:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss = self.d3pm.compute_loss(self.model, batch, loss_type=self.args.d3pm_loss_type)
            val_loss += loss.item()
            num_batches += 1

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return val_loss / max(num_batches, 1)

    @torch.inference_mode()
    def generate_samples(self, epoch: int, num_samples: int = 2) -> torch.Tensor:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()

        samples = self.d3pm.sample(self.model, (num_samples, self.args.seq_len), temperature=0.9)

        H, W, D = self.args.spatial_shape
        indices_3d = sequence_to_indices_3d(samples, H, W, D)

        if self.use_ema:
            self.ema.restore()

        self.model.train()

        if self.accelerator.is_main_process and self.args.use_wandb:
            mid_slice = indices_3d[0, H // 2, :, :].cpu().numpy()
            wandb.log({f"samples/epoch_{epoch}": wandb.Image(mid_slice)})

        return indices_3d


class BayesianFlowTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, args, accelerator):
        super().__init__(model, train_loader, val_loader, args, accelerator)
        self._prepare()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                loss = self.model.compute_loss(batch, loss_type=self.args.bfn_loss_type)

                _check_loss_finite(loss, self.global_step, "training")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                lr = self._get_lr()
                self._update_lr(lr)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            self.accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=self.global_step)
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / max(num_batches, 1)

    @torch.inference_mode()
    def validate(self) -> float:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            loss = self.model.compute_loss(batch, loss_type=self.args.bfn_loss_type)
            val_loss += loss.item()
            num_batches += 1

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return val_loss / max(num_batches, 1)

    @torch.inference_mode()
    def generate_samples(self, epoch: int, num_samples: int = 2) -> torch.Tensor:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()

        samples = self.model.sample(
            (num_samples, self.args.seq_len), num_steps=100, temperature=0.9
        )

        H, W, D = self.args.spatial_shape
        indices_3d = sequence_to_indices_3d(samples, H, W, D)

        if self.use_ema:
            self.ema.restore()

        self.model.train()

        if self.accelerator.is_main_process and self.args.use_wandb:
            mid_slice = indices_3d[0, H // 2, :, :].cpu().numpy()
            wandb.log({f"samples/epoch_{epoch}": wandb.Image(mid_slice)})

        return indices_3d


class ContinuousDiffusionTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, args, accelerator):
        super().__init__(model, train_loader, val_loader, args, accelerator)
        self.diffusion = ContinuousGaussianDiffusion(
            num_timesteps=args.diffusion_steps,
            schedule_type=args.diffusion_schedule,
            device=accelerator.device,
        )
        self._prepare()

    def _prepare_batch(self, batch):
        if batch.ndim == 5:
            B, C, H, W, D = batch.shape
            return batch.permute(0, 2, 3, 4, 1).reshape(B, H * W * D, C)
        return batch

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        progress = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            disable=not self.accelerator.is_local_main_process,
        )
        progress.set_description(f"Epoch {epoch}")

        epoch_loss = 0.0
        num_batches = 0

        for step, batch in progress:
            self.global_step = epoch * len(self.train_loader) + step

            with self.accelerator.accumulate(self.model):
                batch = self._prepare_batch(batch.to(self.accelerator.device))
                loss = self.diffusion.compute_loss(self.model, batch)

                _check_loss_finite(loss, self.global_step, "training")
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                lr = self._get_lr()
                self._update_lr(lr)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.use_ema:
                    self.ema.update()

            self.accelerator.log({"train/loss": loss.item(), "train/lr": lr}, step=self.global_step)
            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / max(num_batches, 1)

    @torch.inference_mode()
    def validate(self) -> float:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = self._prepare_batch(batch.to(self.accelerator.device))
            loss = self.diffusion.compute_loss(self.model, batch)
            val_loss += loss.item()
            num_batches += 1

        if self.use_ema:
            self.ema.restore()

        self.model.train()
        return val_loss / max(num_batches, 1)

    @torch.inference_mode()
    def generate_samples(self, epoch: int, num_samples: int = 2) -> torch.Tensor:
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

        self.model.eval()
        C = self.args.latent_channels
        H, W, D = self.args.latent_shape[1], self.args.latent_shape[2], self.args.latent_shape[3]
        shape = (num_samples, H * W * D, C)
        samples = self.diffusion.sample(self.model, shape, temperature=0.9)

        if self.use_ema:
            self.ema.restore()

        self.model.train()

        if self.accelerator.is_main_process and self.args.use_wandb:
            wandb.log(
                {
                    f"samples/epoch_{epoch}_mean": samples.mean().item(),
                    f"samples/epoch_{epoch}_std": samples.std().item(),
                }
            )

        return samples


def get_trainer(model, train_loader, val_loader, args, accelerator) -> BaseTrainer:
    if args.model == "transformer":
        return TransformerTrainer(model, train_loader, val_loader, args, accelerator)
    elif args.model == "maskgit":
        return MaskGITTrainer(model, train_loader, val_loader, args, accelerator)
    elif args.model == "flow":
        return FlowMatchingTrainer(model, train_loader, val_loader, args, accelerator)
    elif args.model == "d3pm":
        return D3PMTrainer(model, train_loader, val_loader, args, accelerator)
    elif args.model == "bayesian_flow":
        return BayesianFlowTrainer(model, train_loader, val_loader, args, accelerator)
    elif args.model == "diffusion":
        return ContinuousDiffusionTrainer(model, train_loader, val_loader, args, accelerator)
    else:
        raise ValueError(f"Unknown model type: {args.model}")


def load_checkpoint(checkpoint_dir: str, best: bool = False) -> dict | None:
    filename = "checkpoint_best.pt" if best else "checkpoint.pt"
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "none" else None,
        log_with="wandb" if args.use_wandb else None,
    )

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    is_discrete = args.model in DISCRETE_MODELS

    if is_discrete:
        train_dataset = MedMNIST3DTokenDataset(
            args.tokens_path, split="train", token_type="discrete"
        )
        val_dataset = MedMNIST3DTokenDataset(args.tokens_path, split="val", token_type="discrete")

        if args.train_samples:
            train_dataset = torch.utils.data.Subset(
                train_dataset, range(min(args.train_samples, len(train_dataset)))
            )
        if args.val_samples:
            val_dataset = torch.utils.data.Subset(
                val_dataset, range(min(args.val_samples, len(val_dataset)))
            )

        inferred_seq_len, inferred_vocab_size = _infer_discrete_data_contract(
            train_dataset, val_dataset
        )
        if inferred_seq_len != args.seq_len:
            print(
                f"[info] Overriding --seq_len from {args.seq_len} to inferred dataset length {inferred_seq_len}"
            )
            args.seq_len = inferred_seq_len
        if args.vocab_size < inferred_vocab_size:
            print(
                f"[info] Increasing --vocab_size from {args.vocab_size} to inferred minimum {inferred_vocab_size}"
            )
            args.vocab_size = inferred_vocab_size

        base_train = (
            train_dataset.dataset
            if isinstance(train_dataset, torch.utils.data.Subset)
            else train_dataset
        )
        token_shape = getattr(base_train, "tokens", None)
        if token_shape is not None and base_train.tokens.ndim >= 4:
            inferred_spatial_shape = list(base_train.tokens.shape[1:])
            if len(inferred_spatial_shape) == 3 and args.spatial_shape != inferred_spatial_shape:
                print(
                    f"[info] Overriding --spatial_shape from {args.spatial_shape} "
                    f"to inferred dataset shape {inferred_spatial_shape}"
                )
                args.spatial_shape = inferred_spatial_shape

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        model = create_discrete_model(args)
    else:
        train_dataset = MedMNIST3DContinuousDataset(args.tokens_path, split="train")
        val_dataset = MedMNIST3DContinuousDataset(args.tokens_path, split="val")

        if args.train_samples:
            train_dataset = torch.utils.data.Subset(
                train_dataset, range(min(args.train_samples, len(train_dataset)))
            )
        if args.val_samples:
            val_dataset = torch.utils.data.Subset(
                val_dataset, range(min(args.val_samples, len(val_dataset)))
            )

        sample_latent = _peek_sample(train_dataset)
        if sample_latent.ndim == 4:
            inferred_latent_shape = list(sample_latent.shape)
            if args.latent_shape != inferred_latent_shape:
                print(
                    f"[info] Overriding --latent_shape from {args.latent_shape} "
                    f"to inferred dataset shape {inferred_latent_shape}"
                )
                args.latent_shape = inferred_latent_shape
            args.latent_channels = inferred_latent_shape[0]

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        model = create_continuous_model(args)

    trainer = get_trainer(model, train_loader, val_loader, args, accelerator)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume or args.resume_best:
        checkpoint = load_checkpoint(os.path.join(args.logdir, args.name), best=args.resume_best)
        if checkpoint is not None:
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("metric", best_val_loss)
            load_state_dict_compat(trainer.model, checkpoint["net"])
            trainer.optimizer.load_state_dict(checkpoint["opt"])
            trainer.global_step = checkpoint.get("global_step", 0)
            if args.use_ema and "ema" in checkpoint:
                trainer.ema.load_state_dict(checkpoint["ema"])
            print(f"Resumed from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")

    if args.use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            args.project_name,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "entity": args.wandb_entity,
                    "name": args.name,
                    "settings": wandb.Settings(start_method="fork"),
                }
            },
        )

    for epoch in range(start_epoch, args.epochs):
        train_loss = trainer.train_epoch(epoch)

        if (epoch + 1) % args.val_interval == 0:
            val_loss = trainer.validate()

            if accelerator.is_main_process:
                accelerator.log(
                    {"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss},
                    step=trainer.global_step,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trainer.save_checkpoint(epoch, is_best=True, metric=val_loss)
                    print(f"New best validation loss: {val_loss:.4f}")

            trainer.save_checkpoint(epoch, is_best=False, metric=val_loss)

        if args.generate_samples_every and (epoch + 1) % args.generate_samples_every == 0:
            trainer.generate_samples(epoch)

    if accelerator.is_main_process:
        accelerator.end_training()
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
