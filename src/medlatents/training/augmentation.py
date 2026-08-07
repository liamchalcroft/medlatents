"""Training-time sampling and augmentation techniques.

Shared utilities for improving training across all model types.
"""

import math
from collections.abc import Callable

import torch


class CurriculumSampler:
    """
    Curriculum learning: gradually increase task difficulty.

    Start with easier examples (shorter sequences, common tokens)
    and progressively include harder ones.
    """

    def __init__(
        self,
        mode: str = "length",
        start_difficulty: float = 0.3,
        end_difficulty: float = 1.0,
        num_steps: int = 100000,
        schedule: str = "linear",
    ):
        """
        Args:
            mode: 'length', 'rarity', or 'custom'
            start_difficulty: initial difficulty (0-1)
            end_difficulty: final difficulty (0-1)
            num_steps: steps to reach end_difficulty
            schedule: 'linear', 'exp', or 'cosine'
        """
        self.mode = mode
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.num_steps = num_steps
        self.schedule = schedule
        self.current_step = 0

    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        progress = min(1.0, self.current_step / self.num_steps)

        if self.schedule == "linear":
            difficulty = (
                self.start_difficulty + (self.end_difficulty - self.start_difficulty) * progress
            )
        elif self.schedule == "exp":
            # Exponential increase
            difficulty = (
                self.start_difficulty * (self.end_difficulty / self.start_difficulty) ** progress
            )
        elif self.schedule == "cosine":
            # Smooth increase
            difficulty = self.end_difficulty - (
                self.end_difficulty - self.start_difficulty
            ) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return difficulty

    def filter_batch(
        self,
        batch: dict[str, torch.Tensor],
        difficulty_fn: Callable | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Filter batch based on difficulty.

        Args:
            batch: dictionary with 'input_ids' and other keys
            difficulty_fn: function that computes difficulty of each example

        Returns:
            filtered_batch: subset of batch matching current difficulty
        """
        difficulty = self.get_difficulty()

        if self.mode == "length":
            # Filter by sequence length
            lengths = (batch["input_ids"] != 0).sum(dim=1)  # Assuming 0 is pad
            max_length = int(lengths.max().item() * difficulty)
            mask = lengths <= max_length
        elif self.mode == "rarity":
            # Filter by token rarity (need token frequencies)
            if difficulty_fn is None:
                raise ValueError("Need difficulty_fn for rarity mode")
            difficulties = torch.tensor([difficulty_fn(seq) for seq in batch["input_ids"]])
            threshold = difficulties.quantile(difficulty)
            mask = difficulties <= threshold
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Filter batch
        filtered_batch = {k: v[mask] for k, v in batch.items()}

        self.current_step += 1
        return filtered_batch

    def step(self):
        """Increment curriculum step."""
        self.current_step += 1


class ScheduledSampling:
    """
    Scheduled sampling for autoregressive models.

    During training, sometimes use predicted tokens instead of ground truth.
    Reduces exposure bias.

    **WARNING**: This technique is designed for autoregressive TEXT generation.
    For MRI imaging, most models use MaskGIT (bidirectional) or diffusion, NOT
    sequential autoregressive generation. Only use if you're specifically training
    an autoregressive model with raster-order generation.

    Reference: "Scheduled Sampling for Sequence Prediction with RNNs" (Bengio et al., 2015)
    """

    def __init__(
        self,
        mode: str = "linear",
        start_sampling_prob: float = 0.0,
        end_sampling_prob: float = 1.0,
        num_steps: int = 50000,
    ):
        """
        Args:
            mode: 'linear', 'exp', 'inverse_sigmoid'
            start_sampling_prob: initial prob of using predictions
            end_sampling_prob: final prob of using predictions
            num_steps: steps to reach end_sampling_prob
        """
        self.mode = mode
        self.start_prob = start_sampling_prob
        self.end_prob = end_sampling_prob
        self.num_steps = num_steps
        self.current_step = 0

    def get_sampling_prob(self) -> float:
        """Get current sampling probability."""
        progress = min(1.0, self.current_step / self.num_steps)

        if self.mode == "linear":
            prob = self.start_prob + (self.end_prob - self.start_prob) * progress
        elif self.mode == "exp":
            # Handle edge case where start_prob is 0
            if self.start_prob == 0:
                prob = self.end_prob * progress
            else:
                prob = self.start_prob * (self.end_prob / self.start_prob) ** progress
        elif self.mode == "inverse_sigmoid":
            # Handle edge case where start_prob or end_prob is 0
            if self.start_prob == 0 or self.end_prob == 0:
                prob = self.start_prob + (self.end_prob - self.start_prob) * progress
            else:
                k = 10  # Steepness
                prob = self.end_prob / (
                    self.end_prob + (self.start_prob / self.end_prob) * math.exp(-k * progress)
                )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return prob

    def sample_tokens(
        self,
        ground_truth: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix ground truth and predictions based on schedule.

        Args:
            ground_truth: [batch, seq_len] true tokens
            predictions: [batch, seq_len] predicted tokens

        Returns:
            mixed: [batch, seq_len] mixed tokens
        """
        sampling_prob = self.get_sampling_prob()

        # Sample mask: True = use prediction, False = use ground truth
        mask = torch.rand_like(ground_truth, dtype=torch.float) < sampling_prob

        mixed = torch.where(mask, predictions, ground_truth)
        self.current_step += 1

        return mixed


class NoiseSchedule:
    """
    Noise scheduling for training with noisy inputs.

    Gradually reduce noise during training for better convergence.
    For discrete tokens, applies random replacement or dropout.
    """

    def __init__(
        self,
        noise_type: str = "uniform",
        start_noise: float = 0.1,
        end_noise: float = 0.01,
        num_steps: int = 100000,
        schedule: str = "linear",
    ):
        """
        Args:
            noise_type: 'uniform' (random replacement) or 'dropout' (set to 0)
            start_noise: initial noise level (probability of noising each token)
            end_noise: final noise level
            num_steps: steps to reach end_noise
            schedule: 'linear', 'exp', 'cosine'
        """
        self.noise_type = noise_type
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.num_steps = num_steps
        self.schedule = schedule
        self.current_step = 0

    def get_noise_level(self) -> float:
        """Get current noise level."""
        progress = min(1.0, self.current_step / self.num_steps)

        if self.schedule == "linear":
            noise = self.start_noise - (self.start_noise - self.end_noise) * progress
        elif self.schedule == "exp":
            noise = self.start_noise * (self.end_noise / self.start_noise) ** progress
        elif self.schedule == "cosine":
            noise = self.end_noise + (self.start_noise - self.end_noise) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return noise

    def add_noise(
        self,
        tokens: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Add noise to tokens.

        Args:
            tokens: [batch, seq_len]
            vocab_size: vocabulary size

        Returns:
            noisy_tokens: [batch, seq_len]
        """
        noise_level = self.get_noise_level()

        if self.noise_type == "uniform":
            # Replace tokens with uniform random
            mask = torch.rand_like(tokens, dtype=torch.float) < noise_level
            random_tokens = torch.randint_like(tokens, high=vocab_size)
            noisy_tokens = torch.where(mask, random_tokens, tokens)
        elif self.noise_type == "dropout":
            # Drop tokens (set to 0 or mask token)
            mask = torch.rand_like(tokens, dtype=torch.float) < noise_level
            noisy_tokens = tokens.clone()
            noisy_tokens[mask] = 0
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        self.current_step += 1
        return noisy_tokens


class MaskingSchedule:
    """
    Dynamic masking schedule for MaskGIT-style training.

    Adaptively adjust mask ratio during training.
    """

    def __init__(
        self,
        start_mask_ratio: float = 0.5,
        end_mask_ratio: float = 0.15,
        num_steps: int = 100000,
        schedule: str = "cosine",
        min_ratio: float = 0.10,
    ):
        """
        Args:
            start_mask_ratio: initial masking ratio
            end_mask_ratio: final masking ratio
            num_steps: steps to reach end_mask_ratio
            schedule: 'linear', 'cosine', 'constant'
            min_ratio: minimum mask ratio (safety)
        """
        self.start_ratio = start_mask_ratio
        self.end_ratio = end_mask_ratio
        self.num_steps = num_steps
        self.schedule = schedule
        self.min_ratio = min_ratio
        self.current_step = 0

    def get_mask_ratio(self) -> float:
        """Get current mask ratio."""
        if self.schedule == "constant":
            return self.start_ratio

        progress = min(1.0, self.current_step / self.num_steps)

        if self.schedule == "linear":
            ratio = self.start_ratio - (self.start_ratio - self.end_ratio) * progress
        elif self.schedule == "cosine":
            ratio = self.end_ratio + (self.start_ratio - self.end_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return max(ratio, self.min_ratio)

    def step(self):
        """Increment step."""
        self.current_step += 1


class GradientNoiseInjection:
    """
    Add noise to gradients for better generalization.

    Annealed Langevin dynamics - add noise that decreases over time.

    Reference: "Adding Gradient Noise Improves Learning for Very Deep Networks"
    """

    def __init__(
        self,
        eta: float = 0.01,
        gamma: float = 0.55,
    ):
        """
        Args:
            eta: noise scale
            gamma: annealing rate (noise ~ 1/t^gamma)
        """
        self.eta = eta
        self.gamma = gamma
        self.t = 0

    def add_noise(self, model: torch.nn.Module):
        """Add noise to gradients."""
        if not model.training:
            return

        self.t += 1
        std = self.eta / (1 + self.t) ** self.gamma

        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std
                param.grad.add_(noise)


class AdaptiveLossWeighting:
    """
    Automatically balance multiple loss terms.

    Uses uncertainty weighting or gradient magnitude balancing.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(
        self,
        num_losses: int,
        mode: str = "uncertainty",
    ):
        """
        Args:
            num_losses: number of loss terms to balance
            mode: 'uncertainty' or 'grad_norm'
        """
        self.num_losses = num_losses
        self.mode = mode

        if mode == "uncertainty":
            # Learnable log variance parameters
            self.log_vars = torch.nn.Parameter(torch.zeros(num_losses))
        elif mode == "grad_norm":
            # Track gradient norms
            self.grad_norms = [1.0] * num_losses
            self.momentum = 0.9

    def compute_weighted_loss(
        self,
        losses: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weighted sum of losses.

        Args:
            losses: list of scalar losses

        Returns:
            weighted_loss: scalar
        """
        if self.mode == "uncertainty":
            # Uncertainty weighting
            weighted = sum(
                torch.exp(-log_var) * loss + log_var for loss, log_var in zip(losses, self.log_vars)
            )
            return weighted

        elif self.mode == "grad_norm":
            # Use inverse gradient norm as weight
            weights = [1.0 / (norm + 1e-8) for norm in self.grad_norms]
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

            return sum(w * loss for w, loss in zip(weights, losses))

    def update_grad_norms(self, losses: list[torch.Tensor], model: torch.nn.Module):
        """Update gradient norm tracking."""
        if self.mode != "grad_norm":
            return

        for i, loss in enumerate(losses):
            # Compute gradient of this loss
            grads = torch.autograd.grad(
                loss, model.parameters(), retain_graph=True, allow_unused=True
            )
            grad_norm = sum(g.norm().item() for g in grads if g is not None)

            # Update moving average
            self.grad_norms[i] = (
                self.momentum * self.grad_norms[i] + (1 - self.momentum) * grad_norm
            )
