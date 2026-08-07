"""Flow matching for discrete latent sequences.

Implements mixture discrete probability paths for generative modeling of
discrete data using continuous-time flows.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .core import (
    MixtureDiscreteProbPath,
    MixturePathGeneralizedKL,
    PolynomialConvexScheduler,
)


class SourceDistribution(ABC):
    @abstractmethod
    def sample(self, tensor_size: tuple[int, ...], device: torch.device) -> Tensor:
        """Sample from the source distribution."""
        ...

    @abstractmethod
    def sample_like(self, tensor_like: Tensor) -> Tensor:
        """Sample with same shape as the given tensor."""
        ...


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


class UniformSourceDistribution(SourceDistribution):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    @property
    def masked(self) -> bool:
        return False

    def sample(self, tensor_size: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.randint_like(tensor_like, high=self.vocab_size)


def get_path(scheduler_type: str, exponent: float | None = None) -> MixtureDiscreteProbPath:
    """Get a discrete probability path with the specified scheduler.

    Args:
        scheduler_type: Type of scheduler. Currently supports "polynomial".
        exponent: Required for polynomial scheduler. Must be > 0.

    Returns:
        MixtureDiscreteProbPath with the specified scheduler.

    Raises:
        ValueError: If scheduler_type is not supported or exponent is missing/invalid.
    """
    if scheduler_type == "polynomial":
        if exponent is None:
            raise ValueError(
                "exponent is required for polynomial scheduler. "
                "Example: get_path('polynomial', exponent=2.0)"
            )
        scheduler = PolynomialConvexScheduler(n=exponent)
    else:
        raise ValueError(f"{scheduler_type} is not supported")

    return MixtureDiscreteProbPath(scheduler=scheduler)


def get_source_distribution(
    source_distribution: str,
    vocab_size: int,
    mask_token: int | None = None,
) -> SourceDistribution:
    if source_distribution == "mask":
        mask_id = mask_token if mask_token is not None else vocab_size
        return MaskedSourceDistribution(mask_token=mask_id)
    elif source_distribution == "uniform":
        return UniformSourceDistribution(vocab_size=vocab_size)
    else:
        raise ValueError(f"{source_distribution} is not supported")


def get_loss_function(loss_function: str, path: MixtureDiscreteProbPath | None = None) -> _Loss:
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_function == "generalized_kl":
        if path is None:
            raise ValueError("generalized_kl loss requires a path argument")
        return MixturePathGeneralizedKL(path=path)
    else:
        raise ValueError(f"{loss_function} is not supported")
