"""Generative model quality metrics."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from scipy import linalg

from .features import get_feature_extractor

logger = logging.getLogger(__name__)


def calculate_fid(
    real_features: Float[torch.Tensor, "n_real features"],
    generated_features: Float[torch.Tensor, "n_gen features"],
    eps: float = 1e-6,
) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and generated features.

    FID = ||mu_real - mu_gen||^2 + Tr(Sigma_real + Sigma_gen - 2*sqrt(Sigma_real @ Sigma_gen))

    Args:
        real_features: Features from real images [N_real, D]
        generated_features: Features from generated images [N_gen, D]
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better)
    """
    # Convert to numpy for scipy
    real_features = real_features.cpu().numpy()
    generated_features = generated_features.cpu().numpy()

    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_gen

    # Product might be negative due to numerical error
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_gen, disp=False)

    if not np.isfinite(covmean).all():
        logger.warning(f"FID calculation: adding {eps} to diagonal of covariance estimates")
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_gen + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m} too large in FID calculation")
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fid)


def calculate_precision_recall(
    real_features: Float[torch.Tensor, "n_real features"],
    generated_features: Float[torch.Tensor, "n_gen features"],
    k: int = 3,
) -> tuple[float, float]:
    """
    Calculate Precision and Recall for generative models.

    Precision: What fraction of generated samples are realistic?
    Recall: What fraction of real samples are covered by the generator?

    Based on "Improved Precision and Recall Metric for Assessing Generative Models"
    (Kynkäänniemi et al., 2019)

    Args:
        real_features: Features from real images [N_real, D]
        generated_features: Features from generated images [N_gen, D]
        k: Number of nearest neighbors

    Returns:
        (precision, recall) tuple, both in [0, 1]
    """
    # Move to CPU for distance calculations
    real_features = real_features.cpu()
    generated_features = generated_features.cpu()

    # Calculate pairwise distances
    # Real to real
    real_to_real_dists = _pairwise_distances(real_features, real_features)
    # Generated to real
    gen_to_real_dists = _pairwise_distances(generated_features, real_features)
    # Real to generated
    real_to_gen_dists = _pairwise_distances(real_features, generated_features)

    # Get k-th nearest neighbor distances
    # For real manifold
    real_kth_dist = torch.kthvalue(real_to_real_dists, k + 1, dim=1)[0]

    # Precision: fraction of generated in real manifold
    gen_in_real_manifold = (gen_to_real_dists <= real_kth_dist.unsqueeze(0)).any(dim=1)
    precision = float(gen_in_real_manifold.float().mean().item())

    # For generated manifold
    gen_kth_dist = torch.kthvalue(
        _pairwise_distances(generated_features, generated_features), k + 1, dim=1
    )[0]

    # Recall: fraction of real covered by generated manifold
    real_in_gen_manifold = (real_to_gen_dists <= gen_kth_dist.unsqueeze(0)).any(dim=1)
    recall = float(real_in_gen_manifold.float().mean().item())

    return precision, recall


def _pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise Euclidean distances between rows of x and y."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x@y.T
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True)
    dists = x_norm + y_norm.T - 2 * x @ y.T
    dists = torch.clamp(dists, min=0.0)  # Numerical stability
    return torch.sqrt(dists)


def calculate_nll(
    model: torch.nn.Module,
    tokens: Int[torch.Tensor, "batch seq"],
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> float | torch.Tensor:
    """
    Calculate Negative Log-Likelihood for discrete latent models.

    Args:
        model: Model with forward pass returning logits
        tokens: Token sequences [batch, seq_len]
        reduction: How to reduce across batch/sequence

    Returns:
        NLL value (lower is better)
    """
    model.eval()

    with torch.no_grad():
        logits = model(tokens)  # [batch, seq_len, vocab_size]

        # Calculate cross-entropy (NLL for categorical)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens.reshape(-1),
            reduction=reduction,
        )

    if reduction == "none":
        return nll.reshape(tokens.shape)
    else:
        return float(nll.item())


class FIDCalculator:
    """
    FID calculator with feature extraction.

    Handles feature extraction and FID calculation in one class.

    Example:
        >>> calculator = FIDCalculator(extractor_type='radimagenet')
        >>> # Extract features from real images
        >>> for batch in real_loader:
        ...     calculator.update_real(batch)
        >>> # Extract features from generated images
        >>> for batch in gen_loader:
        ...     calculator.update_generated(batch)
        >>> fid = calculator.compute()
    """

    def __init__(
        self,
        extractor_type: Literal["inception", "radimagenet"] = "radimagenet",
        device: str = "cuda",
        **extractor_kwargs,
    ):
        """
        Initialize FID calculator.

        Args:
            extractor_type: Type of feature extractor
            device: Device to run on
            **extractor_kwargs: Additional kwargs for feature extractor
        """
        self.device = device
        self.extractor = get_feature_extractor(extractor_type, device=device, **extractor_kwargs)
        self.real_features = []
        self.generated_features = []

    def update_real(self, images: Float[torch.Tensor, "batch channel height width"]) -> None:
        """
        Extract and store features from real images.

        Args:
            images: Real images [B, C, H, W]
        """
        images = images.to(self.device)
        features = self.extractor(images)
        self.real_features.append(features.cpu())

    def update_generated(self, images: Float[torch.Tensor, "batch channel height width"]) -> None:
        """
        Extract and store features from generated images.

        Args:
            images: Generated images [B, C, H, W]
        """
        images = images.to(self.device)
        features = self.extractor(images)
        self.generated_features.append(features.cpu())

    def compute(self) -> float:
        """
        Compute FID from accumulated features.

        Returns:
            FID score
        """
        if not self.real_features or not self.generated_features:
            raise ValueError("Must accumulate features before computing FID")

        real_features = torch.cat(self.real_features, dim=0)
        gen_features = torch.cat(self.generated_features, dim=0)

        return calculate_fid(real_features, gen_features)

    def reset(self) -> None:
        """Clear accumulated features."""
        self.real_features = []
        self.generated_features = []


class PrecisionRecallCalculator:
    """
    Precision/Recall calculator with feature extraction.

    Example:
        >>> calculator = PrecisionRecallCalculator(k=5)
        >>> for batch in real_loader:
        ...     calculator.update_real(batch)
        >>> for batch in gen_loader:
        ...     calculator.update_generated(batch)
        >>> precision, recall = calculator.compute()
    """

    def __init__(
        self,
        k: int = 3,
        extractor_type: Literal["inception", "radimagenet"] = "radimagenet",
        device: str = "cuda",
        **extractor_kwargs,
    ):
        """
        Initialize Precision/Recall calculator.

        Args:
            k: Number of nearest neighbors
            extractor_type: Type of feature extractor
            device: Device to run on
            **extractor_kwargs: Additional kwargs for feature extractor
        """
        self.k = k
        self.device = device
        self.extractor = get_feature_extractor(extractor_type, device=device, **extractor_kwargs)
        self.real_features = []
        self.generated_features = []

    def update_real(self, images: Float[torch.Tensor, "batch channel height width"]) -> None:
        """Extract and store features from real images."""
        images = images.to(self.device)
        features = self.extractor(images)
        self.real_features.append(features.cpu())

    def update_generated(self, images: Float[torch.Tensor, "batch channel height width"]) -> None:
        """Extract and store features from generated images."""
        images = images.to(self.device)
        features = self.extractor(images)
        self.generated_features.append(features.cpu())

    def compute(self) -> tuple[float, float]:
        """
        Compute Precision and Recall from accumulated features.

        Returns:
            (precision, recall) tuple
        """
        if not self.real_features or not self.generated_features:
            raise ValueError("Must accumulate features before computing P/R")

        real_features = torch.cat(self.real_features, dim=0)
        gen_features = torch.cat(self.generated_features, dim=0)

        return calculate_precision_recall(real_features, gen_features, k=self.k)

    def reset(self) -> None:
        """Clear accumulated features."""
        self.real_features = []
        self.generated_features = []
