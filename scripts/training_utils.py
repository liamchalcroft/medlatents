"""Shared utilities for training scripts."""

import glob
import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Dataset

_MmapMode = Literal["r+", "r", "w+", "c"] | None


class NpyDataset(Dataset):
    """Dataset for loading numpy arrays from files matching a glob pattern."""

    def __init__(self, file_pattern: str, mmap_mode: _MmapMode = "r"):
        self.files = sorted(glob.glob(file_pattern))
        if len(self.files) == 0:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        self.mmap_mode: _MmapMode = mmap_mode
        self._cache: np.ndarray | None = None

    def _load_data(self) -> np.ndarray:
        if self._cache is None:
            self._cache = np.stack(
                [np.load(f, mmap_mode=self.mmap_mode) for f in self.files], axis=0
            )
        return self._cache

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self._load_data()
        return torch.from_numpy(np.ascontiguousarray(data[idx])).long().flatten()


def get_loaders(
    train_pattern: str,
    val_pattern: str,
    batch_size: int,
    num_workers: int = 4,
    seed: int | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        seed: Optional base seed for deterministic shuffling and worker seeding.
        persistent_workers: Keep workers alive between epochs (reduces overhead).
        prefetch_factor: Number of batches to prefetch per worker.
    """
    generator = None
    worker_init_fn = None
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed)}")
        generator = torch.Generator()
        generator.manual_seed(seed)

        def _seed_worker(worker_id: int) -> None:
            worker_seed = (seed + worker_id) % 2**32
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        worker_init_fn = _seed_worker

    use_persistent = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        NpyDataset(train_pattern),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        NpyDataset(val_pattern),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return train_loader, val_loader


class WandBID:
    """Wrapper for WandB run ID to make it serializable in checkpoints."""

    def __init__(self, wandb_id: str):
        self.wandb_id = wandb_id

    def state_dict(self) -> str:
        return self.wandb_id


class Epoch:
    """Wrapper for epoch number to make it serializable in checkpoints."""

    def __init__(self, epoch: int):
        self.epoch = epoch

    def state_dict(self) -> int:
        return self.epoch


class Metric:
    """Wrapper for metric value to make it serializable in checkpoints."""

    def __init__(self, metric: float):
        self.metric = metric

    def state_dict(self) -> float:
        return self.metric


def get_lr(step: int, args, steps_per_epoch: int) -> float:
    """Get learning rate based on current step with warmup and cosine decay.

    Args:
        step: Current global step
        args: Namespace with lr, min_lr, lr_warmup_epochs, epochs, gradient_accumulation_steps
        steps_per_epoch: Number of steps per epoch

    Returns:
        Learning rate for the current step
    """
    warmup_steps = args.lr_warmup_epochs * steps_per_epoch // args.gradient_accumulation_steps
    total_steps = args.epochs * steps_per_epoch // args.gradient_accumulation_steps

    if total_steps == warmup_steps:
        return args.lr

    if step < warmup_steps:
        # Linear warmup
        return args.lr * (step / warmup_steps)
    else:
        # Cosine decay from lr to min_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def validate_file_pattern(pattern: str, description: str = "pattern") -> None:
    """Validate that a file pattern matches at least one file.

    Args:
        pattern: Glob pattern to validate
        description: Description for error messages

    Raises:
        ValueError: If no files match the pattern
    """
    files = glob.glob(pattern)
    if len(files) == 0:
        raise ValueError(f"No files found matching {description}: {pattern}")


def validate_checkpoint_path(path: str) -> None:
    """Validate that a checkpoint path exists.

    Args:
        path: Path to checkpoint file

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")


def add_common_training_args(parser) -> None:
    """Add common training arguments to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    # Required arguments
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument(
        "--train_pattern",
        type=str,
        required=True,
        help="Glob pattern for training data",
    )
    parser.add_argument(
        "--val_pattern",
        type=str,
        required=True,
        help="Glob pattern for validation data",
    )
    parser.add_argument("--vocab_size", type=int, required=True, help="Size of vocabulary")
    parser.add_argument("--seq_length", type=int, required=True, help="Length of sequence")

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "small", "base", "large", "xl"],
        help="Model size configuration",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for sequence processing (None for no chunking)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Optimization parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for learning rate warmup",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # System parameters
    parser.add_argument("--logdir", type=str, default="./checkpoints")
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (team or username). If not set, uses default.",
    )

    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume from best checkpoint")
    parser.add_argument(
        "--reset_wandb", action="store_true", help="Reset WandB run ID when resuming"
    )


def validate_training_args(args) -> None:
    """Validate common training arguments.

    Args:
        args: Parsed arguments namespace

    Raises:
        ValueError: If arguments are invalid
    """
    if args.min_lr > args.lr:
        raise ValueError("min_lr should be less than lr")
    if args.lr_warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs should be non-negative")

    # Validate file patterns
    validate_file_pattern(args.train_pattern, "training data")
    validate_file_pattern(args.val_pattern, "validation data")


def get_wandb_init_kwargs(args, checkpoint=None) -> dict:
    """Get WandB initialization kwargs.

    Args:
        args: Parsed arguments with wandb_entity, resume, resume_best, reset_wandb, name
        checkpoint: Optional checkpoint dict with 'wandb' key

    Returns:
        Dict of kwargs for wandb.init or accelerator.init_trackers
    """
    should_resume = (args.resume or args.resume_best) and not args.reset_wandb

    kwargs = {
        "name": args.name,
        "settings": wandb.Settings(start_method="fork"),
        "resume": "must" if should_resume else None,
        "id": checkpoint["wandb"] if should_resume and checkpoint else None,
    }

    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity

    return kwargs


def get_mask_ratio(ratio: float, schedule: str = "cosine") -> float:
    """Get mask ratio based on training progress and schedule type.

    Args:
        ratio: Training progress (0 to 1)
        schedule: Schedule type ('cosine', 'linear', or 'square')

    Returns:
        Mask ratio for current training step
    """
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(ratio * math.pi))
    elif schedule == "linear":
        return 1 - ratio
    elif schedule == "square":
        return (1 - ratio) ** 2
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    use_fused = (
        torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        fused=use_fused,
    )


def enable_cuda_optimizations(deterministic: bool = False) -> dict:
    settings = {
        "cuda_available": torch.cuda.is_available(),
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "benchmark": False,
        "deterministic": deterministic,
    }
    if not torch.cuda.is_available():
        return settings

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        settings["tf32_matmul"] = True
        settings["tf32_cudnn"] = True
        settings["benchmark"] = True

    return settings


# Re-export MODEL_CONFIGS from configs module for convenience
from medlatents.configs import MODEL_CONFIGS  # noqa: E402, F401
