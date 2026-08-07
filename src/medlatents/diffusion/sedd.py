"""Score Entropy Discrete Diffusion (SEDD) and MDLM objectives.

SEDD models the probability ratio p(y|x)/p(x) instead of the full distribution,
achieving 25-75% perplexity reduction while being theoretically principled.

MDLM provides a Rao-Blackwellized objective for masked diffusion that eliminates
variance from the discrete sampling process.

References:
- "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (ICML 2024 Best Paper)
- "Simple and Effective Masked Diffusion Language Models" (arXiv:2406.07524)
- "Score Entropy Discrete Diffusion" (arXiv:2310.16834)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEDDLoss(nn.Module):
    """Score Entropy Discrete Diffusion loss.

    SEDD models the ratio score: s(x) = log(p(y)/p(x)) for all y ≠ x,
    instead of modeling the full distribution. This is more efficient
    because we only need to predict K-1 values instead of K.

    The key insight is that for discrete diffusion, we can decompose:
        L_SEDD = E_t,x [sum_{y≠x} q(y|x,t) * (s(y,x) - log(q(x|y,t)/q(y|x,t)))^2]

    Where s(y,x) = log(p(y)/p(x)) is the ratio score we're learning.

    Reference:
        Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios
        of the Data Distribution" (ICML 2024 Best Paper)

    Args:
        vocab_size: Number of discrete classes
        noise_schedule: Type of noise schedule ('uniform', 'absorbing')
        hybrid_coeff: Weight for auxiliary cross-entropy loss (default 0.001)
    """

    def __init__(
        self,
        vocab_size: int,
        noise_schedule: str = "uniform",
        hybrid_coeff: float = 0.001,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.noise_schedule = noise_schedule
        self.hybrid_coeff = hybrid_coeff

    def compute_transition_rates(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute transition rates q(y|x,t) for the forward process.

        For uniform noise: q(y|x,t) = beta(t)/K for y ≠ x
        For absorbing: q(y|x,t) = beta(t) * (y == mask)

        Args:
            t: Time values [batch]
            x_t: Current states [batch, seq_len]

        Returns:
            rates: [batch, seq_len, vocab_size] transition rates
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device

        # Time-dependent rate
        # beta(t) increases from 0 to infinity as t goes from 0 to 1
        # Using log-linear schedule: beta(t) = exp(log_beta_min + t * (log_beta_max - log_beta_min))
        log_beta_min = -5.0
        log_beta_max = 5.0
        beta_t = torch.exp(log_beta_min + t * (log_beta_max - log_beta_min)).view(-1, 1, 1)

        if self.noise_schedule == "uniform":
            # Uniform: equal probability to all other states
            # Note: beta_t is [batch, 1, 1], we expand it to the full rate tensor
            rates = beta_t.expand(batch_size, seq_len, self.vocab_size) / self.vocab_size
            rates = rates.clone()  # Make writeable
            # Zero out rate to stay in current state
            x_t_expanded = x_t.unsqueeze(-1)
            rates = rates.scatter(-1, x_t_expanded, 0.0)

        elif self.noise_schedule == "absorbing":
            # Absorbing: only transition to mask state
            rates = torch.zeros((batch_size, seq_len, self.vocab_size), device=device)
            # Mask token is last token
            mask_idx = self.vocab_size - 1
            rates[..., mask_idx] = beta_t.squeeze(-1).expand(batch_size, seq_len)
            # Zero out if already at mask
            is_mask = x_t == mask_idx
            rates[is_mask, mask_idx] = 0.0

        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

        return rates

    def forward(
        self,
        score_logits: torch.Tensor,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SEDD loss.

        Args:
            score_logits: [batch, seq_len, vocab_size] predicted ratio scores
                          where score_logits[..., y] = log(p(y)/p(x)) for y ≠ x
            x_0: [batch, seq_len] original clean data
            x_t: [batch, seq_len] noisy data at time t
            t: [batch] time values in [0, 1]

        Returns:
            loss: Scalar SEDD loss
        """
        batch_size, seq_len = x_t.shape

        # Get transition rates
        rates = self.compute_transition_rates(t, x_t)

        # Compute target ratio scores from true data distribution
        # For denoising score matching: target = log(p(x_0)/p(x_t))
        # We approximate this using the posterior q(x_0|x_t)

        # Target scores: should point from x_t toward x_0
        # log(p(y)/p(x_t)) where y = x_0
        target_scores = torch.zeros_like(score_logits)
        # Set high score for the correct target
        target_scores.scatter_(-1, x_0.unsqueeze(-1), 10.0)
        # Set low score for current state
        target_scores.scatter_(-1, x_t.unsqueeze(-1), 0.0)

        # Compute score matching loss
        # Weighted by transition rates
        diff = score_logits - target_scores
        weighted_diff = rates * (diff**2)

        # Sum over vocabulary, mean over batch and sequence
        sedd_loss = weighted_diff.sum(dim=-1).mean()

        # Optional: hybrid loss with cross-entropy for stability
        if self.hybrid_coeff > 0:
            ce_loss = F.cross_entropy(
                score_logits.view(-1, self.vocab_size),
                x_0.view(-1),
            )
            sedd_loss = sedd_loss + self.hybrid_coeff * ce_loss

        return sedd_loss


class MDLMLoss(nn.Module):
    """Masked Diffusion Language Model (MDLM) Rao-Blackwellized objective.

    MDLM provides a variance-reduced objective for masked diffusion by
    analytically marginalizing out the discrete sampling in the ELBO.

    Key insight: Instead of sampling x_t and computing the loss,
    we can directly compute the expected loss over all possible x_t values,
    weighted by their probabilities under the forward process.

    This Rao-Blackwellization eliminates variance from discrete sampling,
    leading to faster and more stable training.

    Reference:
        Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
        (arXiv:2406.07524)

    Args:
        vocab_size: Number of discrete classes
        mask_token: Index of mask token (default: vocab_size - 1)
        time_reweighting: Type of time reweighting ('uniform', 'importance', 'snr')
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token: int | None = None,
        time_reweighting: str = "uniform",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token = mask_token if mask_token is not None else vocab_size - 1
        self.time_reweighting = time_reweighting

    def get_time_weights(self, t: torch.Tensor) -> torch.Tensor:
        """Get importance weights for different time values.

        Args:
            t: Time values [batch] in [0, 1]

        Returns:
            weights: [batch] importance weights
        """
        if self.time_reweighting == "uniform":
            return torch.ones_like(t)

        elif self.time_reweighting == "importance":
            # Weight more at intermediate times where learning is harder
            # U-shaped: high at t=0 and t=1, low in middle
            return 1.0 + 4 * t * (1 - t)

        elif self.time_reweighting == "snr":
            # Weight by signal-to-noise ratio proxy
            # Higher weight at high noise (low t) where signal is weak
            eps = 1e-5
            return 1.0 / (t + eps)

        else:
            raise ValueError(f"Unknown time reweighting: {self.time_reweighting}")

    def forward(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MDLM Rao-Blackwellized loss.

        Args:
            logits: [batch, seq_len, vocab_size] predicted logits
            x_0: [batch, seq_len] clean data
            t: [batch] time values in [0, 1]

        Returns:
            loss: Scalar MDLM loss
        """
        batch_size, seq_len = x_0.shape

        # Compute mask probability at time t
        # P(masked at time t) = 1 - exp(-integral_0^t beta(s) ds)
        # For linear schedule: mask_prob = t
        mask_prob = t.view(-1, 1)  # [batch, 1]

        # Create per-token mask probabilities
        # Each token is independently masked with probability mask_prob
        per_token_mask_prob = mask_prob.expand(batch_size, seq_len)

        # Rao-Blackwellized loss:
        # E_{x_t ~ q(x_t|x_0)} [log p(x_0|x_t)]
        # = sum_i mask_prob * log p(x_0[i] | masked) + (1 - mask_prob) * log p(x_0[i] | x_0[i])

        # For masked positions: standard cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)

        # For unmasked positions: should predict the same token (high confidence)
        # This is automatically handled since input = output for unmasked

        # Weighted combination
        loss_per_token = -target_log_probs  # Negative log likelihood

        # Weight by mask probability
        weighted_loss = per_token_mask_prob * loss_per_token

        # Apply time weights
        time_weights = self.get_time_weights(t).view(-1, 1)
        weighted_loss = weighted_loss * time_weights

        # Mean over batch and sequence
        return weighted_loss.mean()


class ContinuousTimeMDLM(nn.Module):
    """Continuous-time MDLM with proper ELBO derivation.

    Uses continuous-time formulation for tighter bounds and
    proper handling of the diffusion process.

    Reference: Sahoo et al., "Simple and Effective Masked Diffusion Language Models"
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token: int | None = None,
        schedule_type: str = "linear",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token = mask_token if mask_token is not None else vocab_size - 1
        self.schedule_type = schedule_type

    def alpha_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha(t) = P(not masked at time t).

        Args:
            t: Time values [batch] in [0, 1]

        Returns:
            alpha: [batch] probability of being unmasked
        """
        if self.schedule_type == "linear":
            return 1.0 - t

        elif self.schedule_type == "cosine":
            return torch.cos(t * math.pi / 2)

        elif self.schedule_type == "sqrt":
            return torch.sqrt(1.0 - t)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def forward(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute continuous-time MDLM ELBO.

        Args:
            logits: [batch, seq_len, vocab_size] predicted logits
            x_0: [batch, seq_len] clean data
            t: [batch] time values in [0, 1]

        Returns:
            loss: Scalar ELBO loss
        """
        batch_size, seq_len = x_0.shape

        # Get mask probability
        alpha_t = self.alpha_schedule(t).view(-1, 1)  # P(unmasked)
        mask_prob = 1.0 - alpha_t

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Target log probability
        target_log_prob = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)

        # ELBO loss: weighted by masking probability
        # Only masked tokens contribute to the loss
        loss = -mask_prob * target_log_prob

        # Normalize by expected number of masked tokens
        expected_masked = mask_prob.sum()
        if expected_masked > 0:
            loss = loss.sum() / expected_masked
        else:
            loss = loss.mean()

        return loss


class ScoreEntropyLoss(nn.Module):
    """Pure score entropy loss for discrete diffusion.

    Directly minimizes the entropy of the predicted score distribution,
    encouraging the model to be confident in its predictions.

    This is a simpler alternative to full SEDD that still captures
    the key benefits of score-based modeling.

    Args:
        vocab_size: Number of discrete classes
        entropy_weight: Weight for entropy regularization
        temperature: Temperature for softmax
    """

    def __init__(
        self,
        vocab_size: int,
        entropy_weight: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.entropy_weight = entropy_weight
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute score entropy loss.

        Args:
            logits: [batch, seq_len, vocab_size] predicted logits
            targets: [batch, seq_len] target tokens
            t: [batch] optional time values for weighting

        Returns:
            loss: Scalar loss
            metrics: Dict with loss components
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
        )

        # Entropy of predictions (encourage confidence)
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        mean_entropy = entropy.mean()

        # Cross-entropy plus an entropy penalty that encourages confident
        # (low-entropy) predictions, weighted by ``entropy_weight``.
        loss = ce_loss + self.entropy_weight * mean_entropy

        metrics = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "entropy": mean_entropy.item(),
            "perplexity": torch.exp(ce_loss).item(),
        }

        return loss, metrics


__all__ = [
    "SEDDLoss",
    "MDLMLoss",
    "ContinuousTimeMDLM",
    "ScoreEntropyLoss",
]
