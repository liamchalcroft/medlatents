"""Straight-through estimators for discrete latent variables.

Provides improved gradient estimators for sampling discrete tokens:
- Decoupled ST-Gumbel-Softmax: Different temperatures for forward/backward passes
- ReinMax: Second-order accurate straight-through estimator

References:
- "Improving Discrete Optimisation Via Decoupled Straight-Through Gumbel-Softmax" (arXiv:2410.13331)
- "Bridging Discrete and Backpropagation: Straight-Through Estimators" (NeurIPS 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledSTGumbelSoftmax(nn.Module):
    """
    Decoupled Straight-Through Gumbel-Softmax estimator.

    Uses different temperatures for forward (sampling) and backward (gradient) passes.
    This provides better gradient estimates while maintaining sharp samples.

    Reference: "Improving Discrete Optimisation Via Decoupled Straight-Through
                Gumbel-Softmax" (arXiv:2410.13331)

    Args:
        forward_temp: Temperature for forward pass (sampling)
        backward_temp: Temperature for backward pass (gradients)
        hard: If True, use straight-through estimator
        dim: Dimension to softmax over
    """

    def __init__(
        self,
        forward_temp: float = 1.0,
        backward_temp: float = 0.5,
        hard: bool = True,
        dim: int = -1,
    ):
        super().__init__()
        self.forward_temp = forward_temp
        self.backward_temp = backward_temp
        self.hard = hard
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply decoupled ST-Gumbel-Softmax.

        Args:
            logits: [batch, seq_len, vocab_size] unnormalized logits

        Returns:
            samples: [batch, seq_len, vocab_size] one-hot if hard=True, else soft samples
        """
        # Sample Gumbel noise (use eps=1e-6 for fp16 stability)
        eps = 1e-6
        u = torch.rand_like(logits).clamp(eps, 1 - eps)
        gumbel_noise = -torch.log(-torch.log(u))

        if self.training:
            # During training: use backward_temp for gradients
            soft_samples = F.softmax((logits + gumbel_noise) / self.backward_temp, dim=self.dim)

            if self.hard:
                # Straight-through: forward path uses hard samples, backward uses soft
                with torch.no_grad():
                    # Hard samples using forward_temp
                    hard_samples = F.softmax(
                        (logits + gumbel_noise) / self.forward_temp, dim=self.dim
                    )
                    index = hard_samples.max(dim=self.dim, keepdim=True)[1]
                    hard_one_hot = torch.zeros_like(hard_samples)
                    hard_one_hot.scatter_(self.dim, index, 1.0)

                # Straight-through: hard_one_hot in forward, soft_samples in backward
                return hard_one_hot - soft_samples.detach() + soft_samples
            else:
                return soft_samples
        else:
            # During inference: just use forward_temp
            return F.softmax((logits + gumbel_noise) / self.forward_temp, dim=self.dim)


class ReinMax(nn.Module):
    """
    ReinMax: Second-order accurate straight-through estimator.

    Provides better gradient estimates than standard ST-Gumbel.

    Reference: "Bridging Discrete and Backpropagation: Straight-Through...=" (NeurIPS 2023)

    Args:
        temperature: Sampling temperature
        dim: Dimension to softmax over
    """

    def __init__(self, temperature: float = 1.0, dim: int = -1):
        super().__init__()
        self.temperature = temperature
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply ReinMax estimator.

        Args:
            logits: [batch, seq_len, vocab_size] unnormalized logits

        Returns:
            samples: [batch, seq_len, vocab_size]
        """
        # Sample Gumbel noise (use eps=1e-6 for fp16 stability)
        eps = 1e-6
        u = torch.rand_like(logits).clamp(eps, 1 - eps)
        gumbel_noise = -torch.log(-torch.log(u))

        # Perturbed logits
        perturbed = (logits + gumbel_noise) / self.temperature

        # Soft probabilities
        soft_probs = F.softmax(perturbed, dim=self.dim)

        # Hard samples
        index = soft_probs.max(dim=self.dim, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(soft_probs)
        hard_one_hot.scatter_(self.dim, index, 1.0)

        if self.training:
            # ReinMax: second-order correction
            max_val = perturbed.max(dim=self.dim, keepdim=True)[0]
            log_sum_exp = torch.logsumexp(perturbed, dim=self.dim, keepdim=True)
            variance = torch.exp(log_sum_exp - max_val) - 1.0
            correction = 0.5 * variance * (1.0 - hard_one_hot)
            return hard_one_hot - (soft_probs + correction).detach() + (soft_probs + correction)
        else:
            return hard_one_hot


__all__ = [
    "DecoupledSTGumbelSoftmax",
    "ReinMax",
]
