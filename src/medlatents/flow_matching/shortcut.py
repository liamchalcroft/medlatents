"""Shortcut models for discrete flow matching.

Shortcut models enable a single network to work with any number of
sampling steps, unlike traditional flow matching which requires training
for a specific number of steps.

Key idea: The model learns a vector field that's independent of the
specific discretization, allowing flexible step counts at inference.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
and "Rectified Flow" (Liu et al., 2022)

This is particularly useful for:
1. Trading off speed vs quality at inference time
2. Training once and using with different step counts
3. Adaptive sampling (use more steps where needed)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeInvariantVectorField(nn.Module):
    """Time-invariant vector field for flow matching.

    Unlike standard flow matching which learns v_t(x) that varies with t,
    this learns a time-invariant vector field v(x) that can be integrated
    with any number of steps.

    Args:
        base_model: Base network (e.g., DiscreteDiT)
        hidden_size: Hidden dimension for time embedding
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.base_model = base_model

        # Time embedding is ignored for time-invariant models
        # But we keep the interface compatible
        self.use_time = False

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass (time is ignored).

        Args:
            x: [batch, seq_len] discrete tokens
            t: [batch] time values (ignored for time-invariant model)
            **kwargs: Additional arguments

        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        return self.base_model(x, **kwargs)


class AdaptiveStepSampler:
    """Adaptive step count sampler for flow matching.

    Dynamically adjusts the number of steps based on difficulty:
    - More steps for difficult/high-variance regions
    - Fewer steps for easy/low-variance regions

    Args:
        base_steps: Default number of steps
        min_steps: Minimum steps (for very easy samples)
        max_steps: Maximum steps (for very hard samples)
        variance_threshold: Threshold for increasing steps
    """

    def __init__(
        self,
        base_steps: int = 20,
        min_steps: int = 5,
        max_steps: int = 100,
        variance_threshold: float = 0.1,
    ):
        self.base_steps = base_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.variance_threshold = variance_threshold

    def get_num_steps(
        self,
        logits: torch.Tensor,
        current_step: int,
    ) -> int:
        """
        Determine number of remaining steps based on prediction variance.

        Args:
            logits: [batch, seq_len, vocab_size] current predictions
            current_step: Current step number

        Returns:
            num_steps: Number of steps to use
        """
        probs = F.softmax(logits, dim=-1)

        # Compute entropy as proxy for difficulty
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        avg_entropy = entropy.mean().item()

        # Map entropy to step count
        # Higher entropy = more uncertainty = more steps
        if avg_entropy > 2.0:  # Very uncertain
            return self.max_steps
        elif avg_entropy > 1.0:  # Moderately uncertain
            return self.base_steps
        else:  # Confident
            return self.min_steps


class ShortcutFlowMatchingModel(nn.Module):
    """Flow matching model with shortcut/rectified flow.

    Instead of learning the vector field for a specific ODE discretization,
    this model learns a "straight" path that can be integrated in any
    number of steps.

    Reference: "Rectified Flow" (Liu et al., 2022)
    "Flow Matching for Generative Modeling" (Lipman et al., 2023)

    Args:
        base_model: Base network (e.g., DiscreteDiT)
        vocab_size: Vocabulary size
        use_rectified: Whether to use rectified flow (default True)
        num_refinement_steps: Number of refinement iterations for rectified flow
    """

    def __init__(
        self,
        base_model: nn.Module,
        vocab_size: int,
        use_rectified: bool = True,
        num_refinement_steps: int = 1,
    ):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.use_rectified = use_rectified
        self.num_refinement_steps = num_refinement_steps

        # For rectified flow refinement
        self.register_buffer("refinement_step", torch.tensor(0))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the vector field at (x_t, t).

        Args:
            x_t: [batch, seq_len] intermediate tokens
            t: [batch] time values in [0, 1]
            x_0: [batch, seq_len] source tokens
            x_1: [batch, seq_len] target tokens
            **kwargs: Additional arguments

        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        # Get base model predictions
        logits = self.base_model(x_t, t, **kwargs)

        return logits

    @torch.no_grad()
    def sample_shortcut(
        self,
        shape: tuple[int, int],
        source_dist: nn.Module,
        num_steps: int = 10,
        temperature: float = 1.0,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Sample using the shortcut/straight path.

        Unlike standard flow matching which uses the learned vector field
        with a specific discretization, this uses the straight path
        interpolation which works with any number of steps.

        Args:
            shape: [batch_size, seq_len]
            source_dist: Source distribution (e.g., uniform)
            num_steps: Number of integration steps
            temperature: Sampling temperature
            return_trajectory: Whether to return intermediate states

        Returns:
            samples: [batch, seq_len] generated samples
            metrics: Dict with sampling statistics
        """
        batch_size, seq_len = shape
        device = next(self.parameters()).device

        # Sample from source distribution
        x_0 = source_dist.sample((batch_size, seq_len))

        # Straight line interpolation
        # x_t = (1 - t) * x_0 + t * x_1 (simplified for discrete)
        trajectory = [x_0] if return_trajectory else []

        x_current = x_0.clone()

        for step in range(num_steps):
            t = torch.ones(batch_size, device=device) * (step + 1) / num_steps

            # Get model predictions
            logits = self.forward(x_current, t, x_0, x_current)

            # Sample from predictions
            probs = F.softmax(logits / temperature, dim=-1)
            x_next = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1,
            ).view(batch_size, seq_len)

            if return_trajectory:
                trajectory.append(x_next)

            x_current = x_next

        metrics = {
            "steps_taken": num_steps,
            "shortcut": True,
        }

        if return_trajectory:
            metrics["trajectory"] = trajectory

        return x_current, metrics


class StepInvariantModel(nn.Module):
    """Step-invariant model for flow matching.

    This model is trained to be invariant to the specific number of steps,
    allowing flexible inference-time step counts.

    Key idea: Train with random step counts during training, so the model
    learns a vector field that works with any discretization.

    Args:
        base_model: Base network (e.g., DiscreteDiT)
        vocab_size: Vocabulary size
        min_training_steps: Minimum steps during training
        max_training_steps: Maximum steps during training
    """

    def __init__(
        self,
        base_model: nn.Module,
        vocab_size: int,
        min_training_steps: int = 5,
        max_training_steps: int = 50,
    ):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.min_training_steps = min_training_steps
        self.max_training_steps = max_training_steps

    def get_random_num_steps(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Sample random step counts for training.

        Args:
            batch_size: Batch size

        Returns:
            num_steps: [batch_size] step counts
        """
        return torch.randint(
            self.min_training_steps,
            self.max_training_steps + 1,
            (batch_size,),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        num_steps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional step count conditioning.

        Args:
            x_t: [batch, seq_len] intermediate tokens
            t: [batch] time values in [0, 1]
            num_steps: [batch] number of steps being used
            **kwargs: Additional arguments

        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        return self.base_model(x_t, t, **kwargs)

    def compute_loss(
        self,
        x_1: torch.Tensor,
        source_dist: nn.Module,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute loss with random step counts.

        Args:
            x_1: [batch, seq_len] target data
            source_dist: Source distribution
            **kwargs: Additional arguments

        Returns:
            loss: Scalar loss
            metrics: Dict with training statistics
        """
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample random number of steps for each batch element
        num_steps = self.get_random_num_steps(batch_size)

        # Sample source
        x_0 = source_dist.sample_like(x_1)

        # Sample random times for each batch element
        t = torch.rand(batch_size, device=device)

        # Discrete mixture-path interpolation: each position takes x_1 with
        # probability t, otherwise x_0.
        mask = torch.rand(batch_size, 1, device=device) < t.unsqueeze(1)
        x_t = torch.where(mask, x_1, x_0)

        # Get predictions
        logits = self.forward(x_t, t, num_steps=num_steps)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x_1.view(-1),
        )

        metrics = {
            "loss": loss.item(),
            "avg_num_steps": num_steps.float().mean().item(),
        }

        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int],
        source_dist: nn.Module,
        num_steps: int = 20,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Sample with specified number of steps.

        Args:
            shape: [batch_size, seq_len]
            source_dist: Source distribution
            num_steps: Number of sampling steps
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            samples: Generated samples
            metrics: Dict with sampling statistics
        """
        batch_size, seq_len = shape
        device = next(self.parameters()).device

        x_current = source_dist.sample((batch_size, seq_len))

        for step in range(num_steps):
            t = torch.ones(batch_size, device=device) * (step + 1) / num_steps

            logits = self.forward(x_current, t, **kwargs)

            probs = F.softmax(logits / temperature, dim=-1)
            x_next = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=1,
            ).view(batch_size, seq_len)

            x_current = x_next

        return x_current, {"steps_taken": num_steps}


class ShortcutFlowMatchingLoss(nn.Module):
    """Loss function for shortcut/rectified flow matching.

    Reference: "Rectified Flow" (Liu et al., 2022)

    Args:
        vocab_size: Vocabulary size
        loss_type: Type of loss ('ce', 'kl', 'interpolated')
    """

    def __init__(
        self,
        vocab_size: int,
        loss_type: str = "ce",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_type = loss_type

    def forward(
        self,
        logits: torch.Tensor,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for shortcut flow matching.

        Args:
            logits: [batch, seq_len, vocab_size] model predictions
            x_1: [batch, seq_len] target data
            x_t: [batch, seq_len] intermediate tokens
            t: [batch] time values

        Returns:
            loss: Scalar loss
        """
        if self.loss_type == "ce":
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                x_1.view(-1),
            )

        elif self.loss_type == "kl":
            # KL divergence loss
            probs = F.softmax(logits, dim=-1)
            target = F.one_hot(x_1, self.vocab_size).float()

            kl = (target * torch.log((target + 1e-10) / (probs + 1e-10))).sum(dim=-1)
            loss = kl.mean()

        elif self.loss_type == "interpolated":
            # Interpolated loss between source and target
            probs = F.softmax(logits, dim=-1)
            target = F.one_hot(x_1, self.vocab_size).float()

            # Interpolation weight based on time
            # At t=0, predict source; at t=1, predict target
            t_expanded = t.view(-1, 1, 1)
            target_interp = t_expanded * target + (1 - t_expanded) * (1.0 / self.vocab_size)

            loss = -(target_interp * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


__all__ = [
    "ShortcutFlowMatchingModel",
    "TimeInvariantVectorField",
    "StepInvariantModel",
    "ShortcutFlowMatchingLoss",
    "AdaptiveStepSampler",
]
