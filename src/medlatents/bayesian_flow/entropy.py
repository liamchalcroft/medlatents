"""Entropy encoding utilities for Bayesian Flow Networks.

Entropy encoding helps address the mismatch between training and sampling
entropy levels for discrete variables. By explicitly encoding the uncertainty
level, the network can better handle the transition from noisy to clean states.

References:
- Graves et al. "Bayesian Flow Networks" (2023)
- Xue et al. "Unifying Bayesian Flow Networks and Diffusion Models through SDEs" (2024)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


def compute_entropy(
    logits: Float[torch.Tensor, "batch seq classes"],
    normalize: bool = True,
) -> Float[torch.Tensor, "batch seq"]:
    """
    Compute normalized entropy of categorical distribution.

    Explicitly encoding entropy helps the model be aware of the uncertainty
    level at each position, addressing the train/sample entropy mismatch.

    Args:
        logits: Unnormalized logits [batch, seq, vocab_size]
        normalize: If True, normalize by log(K) to get values in [0, 1]

    Returns:
        entropy: Entropy values [batch, seq], normalized if requested
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-10))
    entropy = -(probs * log_probs).sum(dim=-1)

    if normalize:
        K = logits.size(-1)
        entropy = entropy / math.log(K)

    return entropy


def encode_with_entropy(
    logits: Float[torch.Tensor, "batch seq classes"],
    hidden_dim: int,
    entropy_projection: nn.Linear | None = None,
    prob_projection: nn.Linear | None = None,
) -> Float[torch.Tensor, "batch seq hidden*2"]:
    """
    Encode distribution parameters with entropy information.

    Concatenates a projection of the probability distribution with a
    projection of its entropy. This gives the network explicit access
    to uncertainty information.

    Args:
        logits: Input distribution parameters [batch, seq, vocab_size]
        hidden_dim: Dimension for each projection
        entropy_projection: Optional linear layer for entropy projection
        prob_projection: Optional linear layer for probability projection

    Returns:
        Encoding: Concatenated projections [batch, seq, hidden_dim * 2]
    """
    probs = F.softmax(logits, dim=-1)
    batch_size, seq_len, vocab_size = probs.shape

    # Compute normalized entropy
    H_tilde = compute_entropy(logits, normalize=True)

    # Project entropy [batch, seq, 1] -> [batch, seq, hidden_dim]
    if entropy_projection is None:
        entropy_projection = nn.Linear(1, hidden_dim, device=logits.device)
    h_ent = entropy_projection(H_tilde.unsqueeze(-1))  # [batch, seq, hidden_dim]

    # Project probabilities [batch, seq, vocab_size] -> [batch, seq, hidden_dim]
    if prob_projection is None:
        prob_projection = nn.Linear(vocab_size, hidden_dim, device=logits.device)
    h_prob = prob_projection(probs)  # Vectorized: [batch, seq, hidden_dim]

    return torch.cat([h_prob, h_ent], dim=-1)


__all__ = [
    "compute_entropy",
    "encode_with_entropy",
]
