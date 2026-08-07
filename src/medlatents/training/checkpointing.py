"""Checkpoint saving and loading utilities."""

import glob
import os

import torch
import torch.nn as nn

from ..utils import load_state_dict_compat


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    wandb_id: str,
    hparams: dict | None = None,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "net": model.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "metric": metric,
        "wandb": wandb_id,
    }
    if hparams is not None:
        checkpoint["hparams"] = hparams
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_dir: str, name: str, best: bool = False, weights_only: bool = True
) -> dict | None:
    """Load a training checkpoint if it exists."""
    filename = "checkpoint_best.pt" if best else "checkpoint.pt"
    pattern = os.path.join(checkpoint_dir, name, filename)
    ckpts = glob.glob(pattern)

    if len(ckpts) == 0:
        return None

    return torch.load(ckpts[0], map_location="cpu", weights_only=weights_only)


def resume_from_checkpoint(
    checkpoint: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float, str]:
    """Resume training from a checkpoint.

    Returns:
        start_epoch: Epoch to resume from
        metric_best: Best metric value
        wandb_id: W&B run ID for resuming
    """
    load_state_dict_compat(model, checkpoint["net"])
    if optimizer is not None and "opt" in checkpoint:
        optimizer.load_state_dict(checkpoint["opt"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    metric_best = checkpoint.get("metric", float("inf"))
    wandb_id = checkpoint.get("wandb")

    return start_epoch, metric_best, wandb_id
