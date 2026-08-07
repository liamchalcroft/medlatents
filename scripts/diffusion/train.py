"""Training script for continuous diffusion models on latent spaces."""

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medlatents.diffusion.continuous import ContinuousGaussianDiffusion  # noqa: E402
from medlatents.networks.diffusion_transformer import ContinuousDiT  # noqa: E402
from training_utils import (  # noqa: E402
    MODEL_CONFIGS,
    Epoch,
    Metric,
    WandBID,
    get_lr,
    get_wandb_init_kwargs,
)


class ContinuousLatentDataset(Dataset):
    """Dataset for loading continuous latent vectors from numpy files."""

    def __init__(self, file_pattern: str):
        self.files = glob.glob(file_pattern)
        if len(self.files) == 0:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        self.data = np.stack([np.load(f) for f in self.files], axis=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx]).float()


class NpzLatentDataset(Dataset):
    """Dataset for loading continuous latents from NPZ files (MedMNIST format)."""

    def __init__(self, npz_path: str, split: str = "train"):
        data = np.load(npz_path)
        key = f"{split}_latents"
        if key not in data:
            raise ValueError(f"Key '{key}' not found in {npz_path}")
        self.latents = torch.from_numpy(data[key]).float()

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.latents[idx]


def flatten_latents(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim == 4:
        b, c, h, w = batch.shape
        return batch.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if batch.ndim == 5:
        b, c, d, h, w = batch.shape
        return batch.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)
    return batch


def get_loaders(args) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    if args.data_format == "npz":
        train_dataset = NpzLatentDataset(args.data_path, split="train")
        val_dataset = NpzLatentDataset(args.data_path, split="val")
    else:
        train_dataset = ContinuousLatentDataset(args.train_pattern)
        val_dataset = ContinuousLatentDataset(args.val_pattern)

    generator = None
    worker_init_fn = None
    if args.seed is not None:
        if not isinstance(args.seed, int):
            raise TypeError(f"seed must be an int, got {type(args.seed)}")
        generator = torch.Generator()
        generator.manual_seed(args.seed)

        def _seed_worker(worker_id: int) -> None:
            worker_seed = (args.seed + worker_id) % 2**32
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader


def run_model(args, train_loader, val_loader):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        mixed_precision="fp16",
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Infer input dimensions from data
    sample_batch = next(iter(train_loader))
    if sample_batch.ndim == 2:
        # Shape: (batch, features) -> seq_length=1, in_channels=features
        seq_length = 1
        in_channels = sample_batch.shape[1]
    elif sample_batch.ndim == 3:
        # Shape: (batch, seq, channels)
        seq_length = sample_batch.shape[1]
        in_channels = sample_batch.shape[2]
    elif sample_batch.ndim == 4:
        # Shape: (batch, C, H, W) -> flatten spatial dims
        in_channels = sample_batch.shape[1]
        seq_length = sample_batch.shape[2] * sample_batch.shape[3]
    elif sample_batch.ndim == 5:
        # Shape: (batch, C, D, H, W) -> flatten spatial dims
        in_channels = sample_batch.shape[1]
        seq_length = sample_batch.shape[2] * sample_batch.shape[3] * sample_batch.shape[4]
    else:
        raise ValueError(f"Unsupported data shape: {sample_batch.shape}")

    # Override with args if provided
    if args.seq_length is not None:
        seq_length = args.seq_length
    if args.in_channels is not None:
        in_channels = args.in_channels

    # Model setup
    config = MODEL_CONFIGS[args.model_size]
    model = ContinuousDiT(
        seq_length=seq_length,
        in_channels=in_channels,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        allow_dynamic_seq_length=True,
        gradient_checkpointing=True,
    )

    model_kwargs = {
        "model_size": args.model_size,
        "seq_length": seq_length,
        "in_channels": in_channels,
        "allow_dynamic_seq_length": True,
    }

    # Initialize diffusion process
    diffusion = ContinuousGaussianDiffusion(
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        prediction_type=args.prediction_type,
        device=accelerator.device,
    )

    # Resume from checkpoint if needed
    checkpoint = None
    start_epoch = 0
    metric_best = float("inf")

    if args.resume or args.resume_best:
        ckpt_name = "checkpoint.pt" if args.resume else "checkpoint_best.pt"
        ckpt_path = os.path.join(args.logdir, args.name, ckpt_name)
        ckpts = glob.glob(ckpt_path)
        if len(ckpts) == 0:
            args.resume = False
            args.resume_best = False
            print("\nNo checkpoints found. Beginning from epoch #0")
        else:
            checkpoint = torch.load(ckpts[0], map_location="cpu", weights_only=True)
            print(
                f"\nResuming from epoch #{checkpoint['epoch']} with WandB ID {checkpoint['wandb']}"
            )
            model.load_state_dict(checkpoint["net"])
            start_epoch = checkpoint["epoch"] + 1
            metric_best = checkpoint["metric"]

    # Initialize optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if checkpoint is not None:
        opt.load_state_dict(checkpoint["opt"])

    # Prepare with accelerator
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    # Initialize wandb
    wandb_kwargs = get_wandb_init_kwargs(args, checkpoint)
    accelerator.init_trackers(
        "continuous-diffusion",
        config=vars(args),
        init_kwargs={"wandb": wandb_kwargs},
    )

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            global_step = epoch * len(train_loader) + step

            with accelerator.accumulate(model):
                batch = flatten_latents(batch)

                # Compute diffusion loss
                loss = diffusion.compute_loss(model, batch)

                # Optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                # Update learning rate
                lr = get_lr(global_step, args, len(train_loader))
                for param_group in opt.param_groups:
                    param_group["lr"] = lr

                opt.step()
                opt.zero_grad()

            accelerator.log({"train/loss": loss.item(), "train/lr": lr})
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = flatten_latents(batch)
                    loss = diffusion.compute_loss(model, batch)
                    val_loss += loss.item()
                    val_steps += 1

            val_loss /= val_steps
            accelerator.log({"val/loss": val_loss})

            # Save checkpoints
            ckpt_data = {
                "net": model.state_dict(),
                "opt": opt.state_dict(),
                "wandb": WandBID(wandb.run.id).state_dict(),
                "epoch": Epoch(epoch).state_dict(),
                "metric": Metric(metric_best).state_dict(),
                "hparams": model_kwargs,
            }

            if val_loss < metric_best:
                metric_best = val_loss
                ckpt_data["metric"] = Metric(metric_best).state_dict()
                torch.save(
                    ckpt_data,
                    os.path.join(args.logdir, args.name, "checkpoint_best.pt"),
                )

            torch.save(
                ckpt_data,
                os.path.join(args.logdir, args.name, "checkpoint.pt"),
            )

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train continuous diffusion model")

    # Required arguments
    parser.add_argument("--name", type=str, required=True, help="Name of the run")

    # Data arguments (mutually exclusive formats)
    parser.add_argument(
        "--data_format",
        type=str,
        default="npz",
        choices=["npz", "npy"],
        help="Data format: npz (single file) or npy (glob patterns)",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to NPZ file (for npz format)"
    )
    parser.add_argument(
        "--train_pattern",
        type=str,
        default=None,
        help="Glob pattern for training data (for npy format)",
    )
    parser.add_argument(
        "--val_pattern",
        type=str,
        default=None,
        help="Glob pattern for validation data (for npy format)",
    )

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "small", "base", "large", "xl"],
        help="Model size configuration",
    )
    parser.add_argument("--seq_length", type=int, default=None, help="Override sequence length")
    parser.add_argument("--in_channels", type=int, default=None, help="Override input channels")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Optimization parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--lr_warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Diffusion specific parameters
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument(
        "--schedule_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Beta schedule type",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "x0", "v"],
        help="What the model predicts",
    )

    # System parameters
    parser.add_argument("--logdir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")

    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--reset_wandb", action="store_true", help="Reset WandB run ID")

    args = parser.parse_args()

    # Validation
    if args.min_lr > args.lr:
        raise ValueError("min_lr should be less than lr")
    if args.lr_warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs should be non-negative")

    if args.data_format == "npz":
        if args.data_path is None:
            raise ValueError("--data_path required for npz format")
    else:
        if args.train_pattern is None or args.val_pattern is None:
            raise ValueError("--train_pattern and --val_pattern required for npy format")

    train_loader, val_loader = get_loaders(args)

    run_model(args, train_loader, val_loader)


if __name__ == "__main__":
    main()
