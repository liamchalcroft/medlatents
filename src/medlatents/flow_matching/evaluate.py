"""Evaluation utilities for discrete flow matching models.

Includes entropy computation and likelihood estimation for assessing
model quality and sample diversity.
"""

import math
from collections import Counter

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .core import (
    MixtureDiscreteProbPath,
    MixturePathGeneralizedKL,
    ModelWrapper,
    PolynomialConvexScheduler,
)
from .discrete import SourceDistribution


def _sample_entropy(sample: list) -> float:
    histogram = Counter(sample)
    total = sum(histogram.values())
    entropy = 0

    for count in histogram.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def compute_entropy(samples: Tensor) -> Tensor:
    entropies = [_sample_entropy(sample.tolist()) for sample in samples]
    entropy = sum(entropies) / len(entropies)

    return torch.tensor(entropy, device=samples.device)


@torch.no_grad()
def estimate_likelihood(
    model: nn.Module,
    dataloader: DataLoader,
    source_distribution: SourceDistribution,
    path: MixtureDiscreteProbPath,
    n_discretization: int,
    device: torch.device,
    batch_size: int = 32,
    epsilon: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Estimate the likelihood (ELBO) of the model on a dataset.

    Args:
        model: The flow matching model.
        dataloader: DataLoader yielding batches with 'input_ids' key.
        source_distribution: Source distribution for sampling x_0.
        path: Discrete probability path.
        n_discretization: Number of discretization steps.
        device: Device to run computations on.
        batch_size: Batch size for discretization grid.
        epsilon: Small offset from t=1 to avoid numerical issues.

    Returns:
        Tuple of (elbo, n_elements) where:
        - elbo: Sum of generalized KL losses (Tensor of shape (1,))
        - n_elements: Total number of elements processed (Tensor of shape (1,))
    """
    wrapped = ModelWrapper(model)

    linear_scheduler = PolynomialConvexScheduler(n=1.0)
    linear_path = MixtureDiscreteProbPath(scheduler=linear_scheduler)
    generalized_kl_fn = MixturePathGeneralizedKL(path=linear_path, reduction="none")

    discretization = (
        torch.linspace(0, 1, n_discretization + 1, device=device)[:-1]
        .view(-1, 1)
        .repeat(1, batch_size)
    )

    elbo = torch.zeros((1,), device=device)
    n_elements = torch.zeros((1,), device=device)

    for x_1 in tqdm(dataloader, total=len(dataloader)):
        x_1 = x_1["input_ids"].to(device)

        discretization = (discretization + torch.rand((1, batch_size), device=device)) % 1
        discretization = discretization * (1 - epsilon)

        for k in discretization[:, : x_1.shape[0]]:
            x_0 = source_distribution.sample_like(x_1)
            x_t = linear_path.sample(t=k, x_0=x_0, x_1=x_1).x_t

            t = path.scheduler.kappa_inverse(k)

            logits = wrapped(x=x_t, t=t)

            generalized_kl = generalized_kl_fn(logits=logits, x_1=x_1, x_t=x_t, t=k)
            n_elements += generalized_kl.numel()

            elbo += generalized_kl.sum()

    return elbo, n_elements
