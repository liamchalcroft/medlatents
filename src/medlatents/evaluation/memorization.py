"""Memorization probes for generative models.

Detects whether a generator's outputs are systematically closer to specific
training images than the training set is to itself --- the classical
nearest-neighbor signature of memorization (Carlini et al. 2023; Stein et
al. 2024).

Two scalar metrics are exposed:

  * **memorization_ratio** ``R = median(d_g→t) / median(d_t→t)``
    where ``d_g→t`` is the cosine distance from each generated sample to its
    nearest training-set neighbor and ``d_t→t`` is the same quantity
    computed within the training set (excluding self-matches).
    ``R ≈ 1``: generations as far from training as training is from itself.
    ``R < 1``: generations *closer* to training than the training set is to
    itself --- a memorization signal.
    ``R > 1``: generations more diverse than training (mode underfitting).

  * **auth_pct** : fraction of generated samples whose nearest training
    neighbor is closer than the *second*-nearest training neighbor; this
    flags samples that look like a single specific training exemplar
    rather than a regional average. Inspired by the AuthPct metric of
    Stein et al. (2024).

Features default to InceptionV3 pool3 (192-dim, the same backbone used by
the FID-192 metric in this codebase), so memorization scores are
comparable to the FID values reported in the paper without introducing a
new feature space.

Typical usage::

    from medlatents.evaluation.memorization import (
        InceptionPool3FeatureExtractor, memorization_metrics, memorization_pairs,
    )

    extractor = InceptionPool3FeatureExtractor(device="cuda")
    train_feats = extractor.extract(train_images_uint8)   # (N_train, 192)
    gen_feats   = extractor.extract(generated_images_uint8)  # (N_gen, 192)

    metrics = memorization_metrics(gen_feats, train_feats)
    # {'memorization_ratio': 0.94, 'auth_pct': 0.07,
    #  'median_d_gen_to_train': 0.18, 'median_d_train_to_train': 0.19, ...}

    # For the qualitative side-by-side gallery:
    pairs = memorization_pairs(gen_feats, train_feats, top_k=8)
    # [(gen_idx, train_idx, distance), ...] sorted by distance
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class InceptionPool3FeatureExtractor:
    """Extract 192-d Inception V3 pool3 features (matches FID-192).

    The backbone is the same one used by ``torchmetrics.image.fid`` with
    ``feature=192``; sharing the feature space ensures the memorization
    metric and the reported FID values live on the same Inception manifold.
    """

    def __init__(self, device: str | torch.device = "cpu", batch_size: int = 64):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.device = torch.device(device)
        self.batch_size = batch_size
        # We instantiate FID solely to lift its inception backbone; we never
        # actually use update()/compute().
        self._fid = FrechetInceptionDistance(feature=192, normalize=True).to(self.device)
        self._fid.eval()

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(N, 192)`` float features for the given images.

        ``images`` may be ``(N, 1, H, W)`` or ``(N, 3, H, W)``, dtype uint8 or
        float in [0,1]. Grayscale inputs are tiled to RGB. The inception
        backbone in ``torchmetrics.image.fid`` expects uint8 inputs (the
        normalisation is handled internally), so floats are rescaled to
        ``[0, 255]`` and cast.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be (N, C, H, W); got shape {images.shape}")
        if images.dtype != torch.uint8:
            images = (images.float().clamp(0, 1) * 255).to(torch.uint8)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        feats: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size].to(self.device)
            f = self._fid.inception(batch)
            feats.append(f.detach().cpu())
        return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------


def _l2_normalise(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1)


def cosine_distance_matrix(
    query: torch.Tensor,
    reference: torch.Tensor,
    device: str | torch.device = "cpu",
    block: int = 1024,
) -> torch.Tensor:
    """Compute ``(N_q, N_r)`` cosine-distance matrix in blocks.

    Returns a CPU tensor; switches to GPU for the matmul if ``device`` is
    cuda. Block size keeps memory bounded for large reference sets.
    """
    q = _l2_normalise(query).to(device)
    r = _l2_normalise(reference).to(device)
    out = torch.empty((q.shape[0], r.shape[0]), dtype=torch.float32, device="cpu")
    for start in range(0, q.shape[0], block):
        end = min(start + block, q.shape[0])
        sim = q[start:end] @ r.T  # (block, N_r) cosine similarity
        out[start:end] = (1.0 - sim).cpu()
    return out


# ---------------------------------------------------------------------------
# Memorization scalars
# ---------------------------------------------------------------------------


def memorization_metrics(
    gen_features: torch.Tensor,
    train_features: torch.Tensor,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Compute the memorization-ratio and AuthPct scalars.

    Returns a dict with keys:

    - ``memorization_ratio``: median(d_g->t) / median(d_t->t)
    - ``auth_pct``: fraction of generated samples for which
      d(gen, NN_train) < d(NN_train, NN2_train)
    - ``median_d_gen_to_train``: median nearest-neighbor distance from generated to training
    - ``median_d_train_to_train``: median nearest-neighbor distance within training (excluding self)
    - ``mean_d_gen_to_train``: mean version of the above
    - ``mean_d_train_to_train``: ditto
    - ``n_gen``, ``n_train``: sample counts
    """
    # Distance from generated to training (find nearest training neighbor)
    d_gt = cosine_distance_matrix(gen_features, train_features, device=device)
    nn_dist_gt, nn_idx_gt = d_gt.min(dim=1)  # (N_gen,)

    # Distance from training to training (excluding self)
    d_tt = cosine_distance_matrix(train_features, train_features, device=device)
    d_tt.fill_diagonal_(float("inf"))
    nn_dist_tt, _ = d_tt.min(dim=1)  # (N_train,)

    # AuthPct: for each generated sample, is its NN training neighbor closer
    # to it than that training neighbor's own nearest training neighbor?
    # i.e. is the gen "authentic" (further from any training sample than
    # training samples are from each other)?
    nn_train_distances = nn_dist_tt[nn_idx_gt]  # (N_gen,) — d(NN_train, NN2_train)
    is_close_to_specific_train = (nn_dist_gt < nn_train_distances).float()
    auth_pct = float(is_close_to_specific_train.mean().item())

    median_gt = float(nn_dist_gt.median().item())
    median_tt = float(nn_dist_tt.median().item())

    return {
        "memorization_ratio": median_gt / median_tt if median_tt > 0 else float("nan"),
        "auth_pct": auth_pct,
        "median_d_gen_to_train": median_gt,
        "median_d_train_to_train": median_tt,
        "mean_d_gen_to_train": float(nn_dist_gt.mean().item()),
        "mean_d_train_to_train": float(nn_dist_tt.mean().item()),
        "n_gen": int(gen_features.shape[0]),
        "n_train": int(train_features.shape[0]),
    }


def memorization_pairs(
    gen_features: torch.Tensor,
    train_features: torch.Tensor,
    top_k: int = 8,
    device: str | torch.device = "cpu",
) -> list[tuple[int, int, float]]:
    """Return the ``top_k`` (gen_idx, train_idx, cosine_distance) pairs with
    the smallest gen→train distances --- the most likely memorisation
    suspects to display in a side-by-side gallery.
    """
    d = cosine_distance_matrix(gen_features, train_features, device=device)
    nn_dist, nn_idx = d.min(dim=1)  # (N_gen,)
    order = torch.argsort(nn_dist)[:top_k]
    return [
        (int(gen_idx.item()), int(nn_idx[gen_idx].item()), float(nn_dist[gen_idx].item()))
        for gen_idx in order
    ]


def nearest_neighbor_gallery_pairs(
    generated_images: torch.Tensor,
    train_images: torch.Tensor,
    pairs: Sequence[tuple[int, int, float] | dict[str, float | int]],
    max_pairs: int | None = None,
) -> list[dict[str, torch.Tensor | float | int]]:
    """Select image tensors for generated/training nearest-neighbor galleries."""

    if generated_images.ndim != 4 or train_images.ndim != 4:
        raise ValueError("images must be (N, C, H, W)")
    selected = list(pairs if max_pairs is None else pairs[:max_pairs])
    rows: list[dict[str, torch.Tensor | float | int]] = []
    for pair in selected:
        if isinstance(pair, dict):
            gen_idx = int(pair["gen_idx"])
            train_idx = int(pair["train_idx"])
            distance = float(pair["distance"])
        else:
            gen_idx, train_idx, distance = pair
        rows.append(
            {
                "gen_idx": gen_idx,
                "train_idx": train_idx,
                "distance": float(distance),
                "generated": generated_images[gen_idx].detach().cpu(),
                "train": train_images[train_idx].detach().cpu(),
            }
        )
    return rows


__all__ = [
    "InceptionPool3FeatureExtractor",
    "cosine_distance_matrix",
    "memorization_metrics",
    "memorization_pairs",
    "nearest_neighbor_gallery_pairs",
]
