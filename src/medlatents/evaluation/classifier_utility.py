"""Classifier-utility evaluation: TSTR + domain-FID feature extraction.

A common downstream check for generative models is whether their samples
preserve the *labels* of the training distribution. This module provides
a domain-agnostic utility:

  * :func:`build_grayscale_resnet18` --- ResNet-18 with a single-channel
    input stem and a configurable multi-label head.
  * :func:`train_classifier` --- training loop with per-epoch validation,
    returns best-val state dict + final test metrics.
  * :func:`evaluate_auc` --- per-class and mean-AUC evaluation.
  * :func:`extract_features` --- penultimate-layer feature extractor (any
    classifier with an ``fc`` attribute), suitable for use as a domain-
    relevant FID backbone alternative to InceptionPool3.

Dataset-specific glue (loading raw images, pathology names, output paths)
stays in the calling script. This separation makes the same utility usable
for ChestMNIST, PneumoniaMNIST, or any other multi-label medical dataset
the user wires up.

Typical usage::

    from medlatents.evaluation.classifier_utility import (
        build_grayscale_resnet18, train_classifier, evaluate_auc, extract_features,
    )

    model = build_grayscale_resnet18(num_classes=14).to(device)
    history = train_classifier(model, train_loader, val_loader, epochs=10, device=device)
    test_metrics = evaluate_auc(model, test_loader, device=device)
    # use the trained backbone as a domain-FID feature extractor:
    feats = extract_features(model, test_imgs, device=device)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def build_grayscale_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 with single-channel conv1 + multi-label classification head."""
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _maybe_normalise(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.uint8:
        return x.float() / 255.0
    return x.float()


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str | torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    log_fn: Callable[[str], None] = print,
) -> dict:
    """Train ``model`` with BCE-with-logits, AdamW, per-epoch val AUC.

    Returns ``{"best_val_mean_auc", "best_state_dict", "history": [...]}``.
    The model is mutated in place; restore best state via
    ``model.load_state_dict(returned["best_state_dict"])`` before evaluation.
    """
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state: dict | None = None
    history: list[dict] = []

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x = _maybe_normalise(x).to(device)
            y = y.to(device).float()
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total / max(n, 1)

        val = evaluate_auc(model, val_loader, device)
        epoch_time = time.time() - t0
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_mean_auc": val["mean_auc"],
                "time_s": epoch_time,
            }
        )
        log_fn(
            f"  ep {ep + 1:02d}: loss={train_loss:.4f}  val_mean_AUC={val['mean_auc']:.4f}  ({epoch_time:.0f}s)"
        )

        if val["mean_auc"] > best_val:
            best_val = val["mean_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "best_val_mean_auc": best_val,
        "best_state_dict": best_state,
        "history": history,
    }


@torch.no_grad()
def evaluate_auc(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Per-class + mean ROC-AUC over a DataLoader (multi-label).

    Classes for which the loader contains all-positive or all-negative
    labels yield NaN AUC and are excluded from the mean.
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = _maybe_normalise(x).to(device)
        logits = model(x)
        preds.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    per_class: list[float] = []
    for c in range(ys.shape[1]):
        if ys[:, c].sum() == 0 or ys[:, c].sum() == len(ys):
            per_class.append(float("nan"))
        else:
            per_class.append(float(roc_auc_score(ys[:, c], preds[:, c])))
    valid = [a for a in per_class if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float("nan")
    if class_names is not None:
        named = dict(zip(class_names, per_class))
    else:
        named = {f"class_{i}": v for i, v in enumerate(per_class)}
    return {"mean_auc": mean_auc, "per_class_auc": named}


@torch.no_grad()
def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: str | torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract penultimate-layer features by temporarily swapping ``model.fc``.

    Designed for the ResNet-family backbones built by
    :func:`build_grayscale_resnet18`; works on any classifier whose final
    classifier is exposed as ``model.fc``. Output shape is
    ``(N, model.fc.in_features)``.

    Use as a domain-relevant alternative to InceptionPool3 for FID --- e.g.
    swap ``InceptionPool3FeatureExtractor.extract`` for this in the same FID
    pipeline to obtain a "ChestMNIST-FID" rather than a generic
    "ImageNet-FID".
    """
    if not hasattr(model, "fc"):
        raise AttributeError("model must expose .fc; pass a torchvision-style classifier")
    model.eval()
    fc = model.fc
    model.fc = nn.Identity()
    feats: list[torch.Tensor] = []
    try:
        for start in range(0, len(images), batch_size):
            batch = _maybe_normalise(images[start : start + batch_size]).to(device)
            f = model(batch)
            feats.append(f.cpu())
    finally:
        model.fc = fc
    return torch.cat(feats, dim=0)


def extract_classifier_features(
    model: nn.Module,
    images: torch.Tensor,
    device: str | torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Alias for penultimate-layer classifier features."""
    return extract_features(model, images, device=device, batch_size=batch_size)


def classifier_fid(
    model: nn.Module,
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    device: str | torch.device,
    batch_size: int = 256,
    ridge: float = 1e-6,
) -> float:
    """FID in a trained classifier's penultimate feature space."""
    from .fid import fid_from_features

    real_feats = extract_classifier_features(
        model, real_images, device=device, batch_size=batch_size
    )
    gen_feats = extract_classifier_features(model, gen_images, device=device, batch_size=batch_size)
    return fid_from_features(real_feats.numpy(), gen_feats.numpy(), ridge=ridge)


__all__ = [
    "build_grayscale_resnet18",
    "train_classifier",
    "evaluate_auc",
    "extract_features",
    "extract_classifier_features",
    "classifier_fid",
]
