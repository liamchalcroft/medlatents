"""Conditional sampling methods for Bayesian Flow Networks.

Provides score-guided and particle-based samplers for conditional generation
tasks like inpainting, where some positions are fixed and others generated.

References:
- Xue et al. "Unifying Bayesian Flow Networks and Diffusion Models through SDEs" (2024)
- Wu et al. "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models" (NeurIPS 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


class ScoreGuidedSampler:
    """
    Score-based conditional generation for BFN using SDE formulation.

    BFN can be expressed as an SDE, allowing for classifier-free guidance
    through score functions. The key insight is ``s = grad_xc log p_o(x_c)``,
    where p_o is the output distribution and x_c are conditioning variables.

    This enables conditional generation tasks like inpainting where some
    positions are known and others need to be generated.

    Args:
        model: BayesianFlowTransformer
        max_score: Maximum score magnitude for clipping (stability)
        score_scale: Scaling factor for guidance strength
    """

    def __init__(
        self,
        model: nn.Module,
        max_score: float = 1.0,
        score_scale: float = 1.0,
    ):
        self.model = model
        self.max_score = max_score
        self.score_scale = score_scale

    def compute_score_function(
        self,
        params: Float[torch.Tensor, "batch seq classes"],
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> Float[torch.Tensor, "batch seq classes"]:
        """
        Compute score function: s = grad_xc log p_o(x_c)

        The gradient of log probability of conditioning data with respect
        to the distribution parameters.

        Args:
            params: Current distribution parameters (logits)
            condition: Conditioning data (token indices)
            condition_mask: Mask indicating which positions are conditioned

        Returns:
            score: Score function values
        """
        # Get log probs for conditioned positions
        log_probs = F.log_softmax(params, dim=-1)

        # Get probability of conditioning data
        condition_scores = log_probs.gather(-1, condition.unsqueeze(-1)).squeeze(-1)

        # Build score tensor
        score = torch.zeros_like(params)
        score = score.scatter(-1, condition.unsqueeze(-1), condition_scores.unsqueeze(-1))

        # Apply mask
        mask_expanded = condition_mask.unsqueeze(-1).expand_as(score)
        score = score * mask_expanded

        return score

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        condition: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Sample using SDE formulation with score-based guidance.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_steps: Number of sampling steps
            condition: Optional conditioning data [batch, seq]
            condition_mask: Mask for conditioned positions [batch, seq]
            temperature: Sampling temperature

        Returns:
            samples: Generated samples [batch, seq]
            metrics: Dict with sampling statistics
        """
        device = self.model.device

        # Initialize prior
        params = self.model.get_prior_params(batch_size, seq_len)
        s = None

        if condition is None:
            condition = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            condition_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for i in range(num_steps):
            t = torch.full((batch_size,), float(i), device=device)

            # Get accuracy parameters
            alpha = self.model.get_accuracy(t, continuous_time=False)

            # Compute output distribution
            output_params = self.model(params, t, temperature=temperature)

            # Compute score function
            s_new = self.compute_score_function(output_params, condition, condition_mask)
            s_new = torch.clamp(s_new, -self.max_score, self.max_score) * self.score_scale
            s = s_new

            # Sample from receiver distribution
            probs = F.softmax(output_params / temperature, dim=-1)
            probs = self.model._sanitize_probs(probs)
            x_current = torch.multinomial(
                probs.view(-1, self.model.num_classes),
                num_samples=1,
            ).view(batch_size, seq_len)

            # Sample receiver with guidance
            y_receiver = self.model.sample_receiver_distribution(
                x_current, t, alpha, temperature=temperature
            )

            # Apply score-guided update
            params = self._update_with_guidance(params, y_receiver, alpha, condition_mask, s)

        # Final sampling
        final_probs = F.softmax(params / temperature, dim=-1)
        final_probs = self.model._sanitize_probs(final_probs)
        samples = torch.multinomial(
            final_probs.view(-1, self.model.num_classes),
            num_samples=1,
        ).view(batch_size, seq_len)

        return samples, {"steps_taken": num_steps}

    def _update_with_guidance(
        self,
        params: torch.Tensor,
        y: torch.Tensor,
        alpha: torch.Tensor,
        condition_mask: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Update parameters with score guidance."""
        # Standard Bayesian update
        z = params + y

        # Apply score guidance where conditioned
        if condition_mask is not None and score is not None:
            K = params.size(-1)
            alpha_exp = alpha.view(-1, 1, 1)
            guidance_term = alpha_exp * K * score
            mask_expanded = condition_mask.unsqueeze(-1).expand_as(guidance_term)
            z = z + torch.where(mask_expanded, guidance_term, 0)

        return z


__all__ = [
    "ScoreGuidedSampler",
]
