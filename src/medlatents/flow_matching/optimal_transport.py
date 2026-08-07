"""Optimal transport coupling for flow matching.

Implements OT algorithms for pairing noise samples with data samples,
creating "straighter" interpolation paths compared to random pairing.

The key insight: instead of randomly pairing noise[i] with data[j], we find
the pairing that minimizes total "effort" (squared Euclidean distance).

Uses entropic regularization (Sinkhorn algorithm) for efficient, differentiable
coupling computation.

Reference:
    Tong et al. (2023) "Improving and generalizing flow-based generative
    models with minibatch optimal transport" - OT-CFM paper
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "compute_ot_coupling",
    "sample_from_coupling",
]


def compute_ot_coupling(
    noise: Tensor,
    data: Tensor,
    reg: float = 0.05,
    num_iter: int = 50,
) -> Tensor:
    """Compute optimal transport coupling between noise and data.

    Uses entropic regularization (Sinkhorn) to compute a smooth,
    differentiable coupling matrix.

    Args:
        noise: Source samples (typically Gaussian noise) [batch, ...]
        data: Target samples [batch, ...]
        reg: Entropic regularization strength. Lower = closer to exact OT.
            Sweet spot: 0.01-0.1.
        num_iter: Sinkhorn iterations. 50 usually suffices.

    Returns:
        Doubly stochastic coupling matrix P [batch, batch] where rows and
        columns sum to 1/n.
    """
    batch_size = noise.shape[0]

    # Flatten for distance computation
    noise_flat = noise.reshape(batch_size, -1)
    data_flat = data.reshape(batch_size, -1)

    # Squared Euclidean cost matrix
    cost = torch.cdist(noise_flat, data_flat, p=2).pow(2)

    return _sinkhorn_log_domain(cost, reg=reg, num_iter=num_iter)


def _sinkhorn_log_domain(
    cost: Tensor,
    reg: float,
    num_iter: int,
) -> Tensor:
    """Log-domain Sinkhorn-Knopp for numerically stable entropic OT.

    The standard Sinkhorn algorithm can overflow when cost/reg is large.
    The log-domain version works with log(u), log(v), log(K) directly
    and is numerically stable regardless of cost magnitude.

    Args:
        cost: Cost matrix [n, m]
        reg: Entropic regularization
        num_iter: Number of Sinkhorn iterations

    Returns:
        Coupling matrix P [n, m]

    Raises:
        ValueError: If cost matrix has invalid dimensions (n <= 0 or m <= 0)
    """
    n, m = cost.shape
    if n <= 0 or m <= 0:
        raise ValueError(f"Cost matrix must have positive dimensions, got shape ({n}, {m})")
    dtype, device = cost.dtype, cost.device

    log_a = torch.full((n,), -math.log(n), dtype=dtype, device=device)
    log_b = torch.full((m,), -math.log(m), dtype=dtype, device=device)

    # Normalize cost to prevent extreme values
    cost_normalized = cost / cost.max().clamp(min=1e-8)
    log_K = -cost_normalized / reg

    # Initialize dual variables
    log_u = torch.zeros(n, dtype=dtype, device=device)
    log_v = torch.zeros(m, dtype=dtype, device=device)

    # Sinkhorn iterations with early stopping
    convergence_threshold = 1e-6
    for i in range(num_iter):
        # Row normalization
        log_u_new = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        # Column normalization
        log_v_new = log_b - torch.logsumexp(log_K.T + log_u_new.unsqueeze(0), dim=1)

        # Check convergence every 5 iterations
        if i % 5 == 4:
            log_coupling = log_u_new.unsqueeze(1) + log_K + log_v_new.unsqueeze(0)
            coupling = torch.exp(log_coupling)
            row_err = (coupling.sum(dim=1) - 1.0 / n).abs().max()
            col_err = (coupling.sum(dim=0) - 1.0 / m).abs().max()
            if row_err < convergence_threshold and col_err < convergence_threshold:
                log_u, log_v = log_u_new, log_v_new
                break

        log_u, log_v = log_u_new, log_v_new

    # Reconstruct coupling: P = diag(u) @ K @ diag(v)
    log_coupling = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    return torch.exp(log_coupling)


def sample_from_coupling(
    coupling: Tensor,
    data: Tensor,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Reshuffle data according to OT coupling.

    Given the transport plan P, sample a permutation of data indices
    where each noise sample i is paired with data sample j according to
    the coupling probabilities P[i, :].

    Args:
        coupling: Transport plan from compute_ot_coupling [batch, batch]
        data: Data samples to reshuffle [batch, ...]
        return_indices: If True, also return the sampled indices

    Returns:
        If return_indices=False: Reshuffled data
        If return_indices=True: Tuple of (reshuffled_data, indices)
    """
    # Sample indices from coupling rows (each row is a distribution over data)
    indices = torch.multinomial(coupling, num_samples=1).squeeze(-1)

    if return_indices:
        return data[indices], indices
    return data[indices]


def ot_flow_sample_path(
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    reg: float = 0.05,
    num_iter: int = 50,
    return_coupling: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Sample a point along OT-coupled flow path.

    Computes OT coupling between x0 (noise) and x1 (data), then samples
    the interpolated point and velocity.

    Args:
        x0: Noise samples [batch, ...]
        x1: Data samples [batch, ...]
        t: Timesteps [batch]
        reg: OT regularization
        num_iter: Sinkhorn iterations
        return_coupling: If True, also return the coupling matrix

    Returns:
        x_t: Interpolated points [batch, ...]
        target_v: Target velocities [batch, ...]
        coupling (optional): Coupling matrix [batch, batch]
    """
    # Compute OT coupling
    coupling = compute_ot_coupling(x0, x1, reg=reg, num_iter=num_iter)

    # Reshuffle x1 according to coupling
    x1_reordered, indices = sample_from_coupling(coupling, x1, return_indices=True)

    # Standard linear interpolation with reordered data
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)

    x_t = (1.0 - t) * x0 + t * x1_reordered
    target_v = x1_reordered - x0

    if return_coupling:
        return x_t, target_v, coupling
    return x_t, target_v
