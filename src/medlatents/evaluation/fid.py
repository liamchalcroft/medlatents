"""FID-192 utilities with bootstrap CI estimation.

Two reasons this lives in ``medlatents.evaluation`` rather than as a one-off
script:

  * The fast ``fid_from_features`` (eigendecomposition-based, ~50× faster
    than ``scipy.linalg.sqrtm`` on 192-dim covariances) is broadly useful.

  * Bootstrap CI estimation for FID --- both the "real-vs-real noise floor"
    (split a held-out set in two, repeat) and "real-vs-gen estimator CI"
    (resample (real_idx, gen_idx) pairs, repeat) --- shows whether a gap
    between two FID numbers is larger than the estimator's sampling noise.
    Exposing it as a library function lets
    users of medlatents reproduce the same uncertainty bound on their own
    generators with three lines of code.

Typical usage::

    from medlatents.evaluation import (
        InceptionPool3FeatureExtractor,
        fid_from_features,
        bootstrap_fid_noise_floor,
        bootstrap_fid_real_vs_gen,
    )

    extractor = InceptionPool3FeatureExtractor(device="cuda")
    real_feats = extractor.extract(real_test_uint8).numpy()
    gen_feats  = extractor.extract(generated_uint8).numpy()

    # Single FID:
    fid = fid_from_features(real_feats, gen_feats)

    # 95% CI on the FID estimator at a given sample size:
    ci = bootstrap_fid_real_vs_gen(real_feats, gen_feats, n=10_000, B=200)
    # {'mean': ..., 'p2.5': ..., 'p97.5': ..., 'ci_width_95': ...}

    # Noise floor of FID at this domain at varying sample sizes:
    noise = bootstrap_fid_noise_floor(real_feats, n_per_half=[1000, 5000, 10000])
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def fid_from_features(real_feats: np.ndarray, gen_feats: np.ndarray, ridge: float = 1e-6) -> float:
    """Closed-form Fréchet Inception Distance on already-extracted features.

    Uses the symmetric form
    ``tr( (sigma_r sigma_g)^{1/2} ) = sum sqrt eig( sigma_r^{1/2} sigma_g sigma_r^{1/2} )``
    computed via two ``eigh`` calls. ~50× faster than ``scipy.linalg.sqrtm``
    on the 192-dim covariances used for FID-192 at low resolution, and
    numerically more stable (eigenvalues are non-negative by construction).
    """
    if real_feats.ndim != 2 or gen_feats.ndim != 2:
        raise ValueError(
            f"features must be (N, D); got real {real_feats.shape}, gen {gen_feats.shape}"
        )
    if real_feats.shape[1] != gen_feats.shape[1]:
        raise ValueError(f"feature dim mismatch: {real_feats.shape[1]} vs {gen_feats.shape[1]}")

    mu_r = real_feats.mean(axis=0)
    mu_g = gen_feats.mean(axis=0)
    eye = np.eye(real_feats.shape[1])
    sigma_r = np.cov(real_feats, rowvar=False) + ridge * eye
    sigma_g = np.cov(gen_feats, rowvar=False) + ridge * eye

    diff = mu_r - mu_g

    eig_r, V_r = np.linalg.eigh(sigma_r)
    eig_r = np.clip(eig_r, 0, None)
    sigma_r_half = (V_r * np.sqrt(eig_r)) @ V_r.T

    inner = sigma_r_half @ sigma_g @ sigma_r_half
    eigs = np.linalg.eigvalsh(inner)
    eigs = np.clip(eigs, 0, None)
    tr_covmean = float(np.sqrt(eigs).sum())

    return float(diff @ diff + np.trace(sigma_r) + np.trace(sigma_g) - 2 * tr_covmean)


def bootstrap_fid_noise_floor(
    features: np.ndarray,
    n_per_half: Sequence[int] = (1000, 2500, 5000, 10000),
    B: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> dict[str, dict]:
    """Estimate FID's *estimator-variance* noise floor via random splits.

    For each ``n`` in ``n_per_half``, draw ``B`` random splits of
    ``features`` into two disjoint halves of size ``n`` each, compute
    FID(half_A, half_B), and return summary statistics. Because both halves
    come from the same underlying distribution, the resulting FID
    distribution characterises the *minimum detectable* FID gap at sample
    size ``n`` --- gaps below this floor are within metric noise.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    N = features.shape[0]
    for n in n_per_half:
        if 2 * n > N:
            if verbose:
                logger.info(f"  skip n={n} (need {2 * n} disjoint, have {N})")
            continue
        fids = []
        t0 = time.time()
        for _ in range(B):
            idx = rng.permutation(N)[: 2 * n]
            fids.append(fid_from_features(features[idx[:n]], features[idx[n:]]))
        fids = np.array(fids)
        out[str(n)] = {
            "n_per_half": int(n),
            "B": int(B),
            "mean": float(fids.mean()),
            "std": float(fids.std()),
            "p2.5": float(np.percentile(fids, 2.5)),
            "p97.5": float(np.percentile(fids, 97.5)),
            "ci_width_95": float(np.percentile(fids, 97.5) - np.percentile(fids, 2.5)),
            "time_s": time.time() - t0,
        }
        if verbose:
            r = out[str(n)]
            logger.info(
                f"  n={n}: FID(real,real) mean={r['mean']:.4f} 95%CI=[{r['p2.5']:.4f},{r['p97.5']:.4f}] ({r['time_s']:.0f}s)"
            )
    return out


def bootstrap_fid_real_vs_gen(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    n: int | None = None,
    B: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """95% CI for FID(real, gen) via paired bootstrap of feature indices.

    For ``B`` reps, sample ``n`` indices with replacement from each of
    ``real_feats`` and ``gen_feats``, recompute FID. The resulting
    distribution captures the *estimator* uncertainty at the given sample
    size, given a fixed underlying generator. It does *not* capture
    train-time stochasticity (which requires re-sampling the model itself).
    Defaults ``n`` to ``min(len(real_feats), len(gen_feats))``.
    """
    rng = np.random.default_rng(seed)
    if n is None:
        n = min(real_feats.shape[0], gen_feats.shape[0])
    fids = []
    t0 = time.time()
    for _ in range(B):
        r_idx = rng.choice(real_feats.shape[0], size=n, replace=True)
        g_idx = rng.choice(gen_feats.shape[0], size=n, replace=True)
        fids.append(fid_from_features(real_feats[r_idx], gen_feats[g_idx]))
    fids = np.array(fids)
    out = {
        "n": int(n),
        "B": int(B),
        "mean": float(fids.mean()),
        "std": float(fids.std()),
        "p2.5": float(np.percentile(fids, 2.5)),
        "p97.5": float(np.percentile(fids, 97.5)),
        "ci_width_95": float(np.percentile(fids, 97.5) - np.percentile(fids, 2.5)),
        "time_s": time.time() - t0,
    }
    if verbose:
        logger.info(
            f"  FID(real,gen) mean={out['mean']:.3f} 95%CI=[{out['p2.5']:.3f},{out['p97.5']:.3f}] ({out['time_s']:.0f}s)"
        )
    return out


__all__ = [
    "fid_from_features",
    "bootstrap_fid_noise_floor",
    "bootstrap_fid_real_vs_gen",
]
