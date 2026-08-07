"""Loss computation utilities for Bayesian Flow Networks.

Provides training wrappers that modify loss computation strategies
for improved training dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualLossWrapper(nn.Module):
    """
    Training wrapper with residual/reconstruction loss weighting.

    Instead of 50/50 residual/reconstruction loss, uses a configurable
    split (default 95/5) which recovers the same objective while focusing
    compute on the more informative residual loss.

    Args:
        model: BayesianFlowTransformer
        residual_ratio: Fraction of samples using residual loss (default 0.95)
    """

    def __init__(
        self,
        model: nn.Module,
        residual_ratio: float = 0.95,
    ):
        super().__init__()
        self.model = model
        self.residual_ratio = residual_ratio

    def forward(
        self,
        x_0: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute weighted loss with residual/reconstruction split.

        Args:
            x_0: Input data
            **kwargs: Additional arguments passed to model loss methods

        Returns:
            loss: Reweighted loss value
        """
        # Randomly choose loss type
        use_residual = torch.rand(1, device=x_0.device).item() < self.residual_ratio

        if use_residual:
            # Residual loss (discrete time)
            loss = self.model.discrete_time_loss(x_0, **kwargs)
        else:
            # Reconstruction loss at t=1
            loss = self.model.continuous_time_loss(x_0, **kwargs)

        # Reweight to recover original objective
        if use_residual:
            loss = loss / self.residual_ratio
        else:
            loss = loss / (1 - self.residual_ratio)

        return loss


__all__ = [
    "ResidualLossWrapper",
]
