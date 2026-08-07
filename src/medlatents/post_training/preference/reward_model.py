"""Reward models for preference learning.

Provides trainable reward models that learn from preference data.
Designed to reuse existing transformer blocks from medlatents.networks.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...networks.diffusion_transformer import (
    ContinuousSequenceEmbedder,
    DiscreteSequenceEmbedder,
    DiTBlock,
    TimestepEmbedder,
)
from ...networks.transformer import init_weights, precompute_freqs_cis
from .data import PreferenceDataset, collate_preference_pairs

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Trainable reward model that scores samples.

    Uses the same transformer architecture as the generative models,
    but with a scalar output head. This ensures architectural compatibility
    and enables transfer learning from pretrained generators.

    The model can operate in two modes:
    1. Discrete: Input is token indices (for MaskGIT, autoreg, D3PM, discrete flow)
    2. Continuous: Input is latent vectors (for continuous diffusion/flow)

    It optionally accepts timestep conditioning for step-aware scoring (SPO).

    Example:
        >>> # Create reward model for discrete tokens
        >>> model = RewardModel(
        ...     vocab_size=1024,
        ...     seq_length=256,
        ...     hidden_size=512,
        ...     depth=6,
        ...     num_heads=8,
        ... )
        >>>
        >>> # Score samples
        >>> scores = model(tokens)  # [batch, 1]
        >>>
        >>> # Train from preferences
        >>> trainer = RewardModelTrainer(model, dataset)
        >>> trainer.train(epochs=3)
    """

    def __init__(
        self,
        # Input configuration
        vocab_size: int | None = None,  # For discrete mode
        in_channels: int | None = None,  # For continuous mode
        seq_length: int = 256,
        # Architecture
        hidden_size: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        # Options
        timestep_aware: bool = False,  # For SPO
        pooling: Literal["mean", "cls", "last"] = "mean",
        qk_norm: bool = False,
        rope_theta: float = 10000.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize reward model.

        Args:
            vocab_size: Vocabulary size for discrete mode (mutually exclusive with in_channels)
            in_channels: Input channels for continuous mode
            seq_length: Maximum sequence length
            hidden_size: Transformer hidden dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            timestep_aware: If True, accept timestep conditioning
            pooling: How to aggregate sequence for scalar output
            qk_norm: Use QK LayerNorm for stability
            rope_theta: RoPE base frequency
            dropout: Dropout rate
        """
        super().__init__()

        # Validate input mode
        if vocab_size is None and in_channels is None:
            raise ValueError(
                "Must specify either vocab_size (discrete) or in_channels (continuous)"
            )
        if vocab_size is not None and in_channels is not None:
            raise ValueError("Specify only one of vocab_size or in_channels")

        self.is_discrete = vocab_size is not None
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.timestep_aware = timestep_aware
        self.pooling = pooling

        # Input embedding
        if self.is_discrete:
            self.x_embedder = DiscreteSequenceEmbedder(
                seq_length=seq_length,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                allow_dynamic_seq_length=True,
            )
        else:
            self.x_embedder = ContinuousSequenceEmbedder(
                seq_length=seq_length,
                in_channels=in_channels,
                hidden_size=hidden_size,
                allow_dynamic_seq_length=True,
            )

        # Optional timestep embedding
        if timestep_aware:
            self.t_embedder = TimestepEmbedder(hidden_size)
        else:
            self.t_embedder = None

        # CLS token for cls pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        else:
            self.cls_token = None

        # Transformer blocks (reuse DiTBlock with identity conditioning if no timestep)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm)
                for _ in range(depth)
            ]
        )

        # Rotary position embeddings
        self.freqs_cis = precompute_freqs_cis(
            hidden_size // num_heads,
            seq_length + 1,  # +1 for potential CLS token
            theta=rope_theta,
        )
        self.seq_len_cached = seq_length + 1

        # Output head
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        # Initialize
        self.apply(init_weights)
        # Zero-init adaLN for stable start
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        # Zero-init output head
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _update_rope_cache(self, seq_len: int) -> None:
        """Update RoPE cache if sequence length changed."""
        if seq_len + 1 > self.seq_len_cached:
            self.freqs_cis = precompute_freqs_cis(
                self.hidden_size // self.blocks[0].attn.num_heads,
                seq_len + 1,
            ).to(self.freqs_cis.device)
            self.seq_len_cached = seq_len + 1

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
    ) -> Tensor:
        """Compute reward score for samples.

        Args:
            x: Input samples
                - Discrete: [batch, seq_len] token indices
                - Continuous: [batch, seq_len, channels] latent vectors
            t: Optional timestep for step-aware scoring [batch]

        Returns:
            Scalar rewards [batch, 1]
        """
        # Get sequence length and update RoPE if needed
        seq_len = x.size(1) if x.dim() == 3 else x.size(-1)
        self._update_rope_cache(seq_len)
        freqs = self.freqs_cis[: seq_len + (1 if self.cls_token is not None else 0)]
        freqs = freqs.to(x.device)

        # Embed input
        h = self.x_embedder(x)

        # Add CLS token if using cls pooling
        if self.cls_token is not None:
            batch_size = h.size(0)
            cls = self.cls_token.expand(batch_size, -1, -1)
            h = torch.cat([cls, h], dim=1)

        # Get conditioning (timestep or zeros for unconditional)
        if self.timestep_aware and t is not None:
            c = self.t_embedder(t)
        else:
            # Use zeros for unconditional - adaLN will be identity-ish
            c = torch.zeros(h.size(0), self.hidden_size, device=h.device, dtype=h.dtype)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, c, freqs)

        # Pool to single vector
        h = self.norm(h)
        if self.pooling == "cls":
            h = h[:, 0]  # Take CLS token
        elif self.pooling == "last":
            h = h[:, -1]  # Take last token
        else:  # mean
            h = h.mean(dim=1)

        # Output scalar reward
        return self.head(h)

    def compute_preference_loss(
        self,
        chosen: Tensor,
        rejected: Tensor,
        margin: Tensor | None = None,
        t: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute Bradley-Terry preference loss.

        Args:
            chosen: Preferred samples [batch, ...]
            rejected: Non-preferred samples [batch, ...]
            margin: Optional preference strength [batch]
            t: Optional timestep for step-aware training

        Returns:
            Dict with loss and metrics
        """
        r_chosen = self(chosen, t)
        r_rejected = self(rejected, t)

        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        diff = r_chosen - r_rejected
        if margin is not None:
            diff = margin.view(-1, 1) * diff

        loss = -F.logsigmoid(diff).mean()

        # Compute accuracy
        with torch.no_grad():
            accuracy = (r_chosen > r_rejected).float().mean()
            reward_margin = (r_chosen - r_rejected).mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "reward_margin": reward_margin,
            "chosen_reward": r_chosen.mean(),
            "rejected_reward": r_rejected.mean(),
        }


class RewardModelTrainer:
    """Trainer for reward models from preference data.

    Handles the training loop with proper logging, checkpointing, and
    distributed training support.

    Example:
        >>> trainer = RewardModelTrainer(
        ...     model=reward_model,
        ...     dataset=preference_dataset,
        ...     lr=1e-4,
        ... )
        >>> trainer.train(epochs=3)
    """

    def __init__(
        self,
        model: RewardModel,
        dataset: PreferenceDataset,
        val_dataset: PreferenceDataset | None = None,
        lr: float = 1e-4,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        grad_clip: float = 1.0,
        device: torch.device | str = "cuda",
    ) -> None:
        """Initialize trainer.

        Args:
            model: RewardModel to train
            dataset: Training preference data
            val_dataset: Optional validation data
            lr: Learning rate
            batch_size: Batch size
            weight_decay: Weight decay for AdamW
            warmup_ratio: Fraction of steps for warmup
            grad_clip: Gradient clipping norm
            device: Training device
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_preference_pairs,
            drop_last=True,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_preference_pairs,
            )
        else:
            self.val_loader = None

    def train(
        self,
        epochs: int = 3,
        log_every: int = 10,
        eval_every: int = 100,
    ) -> dict[str, list[float]]:
        """Train the reward model.

        Args:
            epochs: Number of training epochs
            log_every: Log metrics every N steps
            eval_every: Run validation every N steps

        Returns:
            Dictionary of training metrics over time
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        global_step = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                # Move to device
                chosen = batch["chosen"].to(self.device)
                rejected = batch["rejected"].to(self.device)
                margin = batch["margin"].to(self.device)

                # Learning rate warmup
                if global_step < warmup_steps:
                    lr = self.lr * (global_step + 1) / warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                # Forward pass
                self.optimizer.zero_grad()
                metrics = self.model.compute_preference_loss(chosen, rejected, margin)
                loss = metrics["loss"]

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                epoch_acc += metrics["accuracy"].item()
                global_step += 1

                # Logging
                if global_step % log_every == 0:
                    history["train_loss"].append(loss.item())
                    history["train_accuracy"].append(metrics["accuracy"].item())
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc": f"{metrics['accuracy'].item():.4f}",
                        }
                    )

                # Validation
                if self.val_loader is not None and global_step % eval_every == 0:
                    val_metrics = self.validate()
                    history["val_loss"].append(val_metrics["loss"])
                    history["val_accuracy"].append(val_metrics["accuracy"])
                    self.model.train()

            # End of epoch stats
            avg_loss = epoch_loss / len(self.train_loader)
            avg_acc = epoch_acc / len(self.train_loader)
            logger.info(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

        return history

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and return metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in self.val_loader:
            chosen = batch["chosen"].to(self.device)
            rejected = batch["rejected"].to(self.device)
            margin = batch["margin"].to(self.device)

            metrics = self.model.compute_preference_loss(chosen, rejected, margin)
            total_loss += metrics["loss"].item()
            total_acc += metrics["accuracy"].item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_acc / num_batches,
        }


def create_reward_model(
    model_size: Literal["nano", "small", "base", "large"] = "small",
    vocab_size: int | None = None,
    in_channels: int | None = None,
    seq_length: int = 256,
    timestep_aware: bool = False,
    **kwargs: Any,
) -> RewardModel:
    """Factory function to create reward models with standard configurations.

    Args:
        model_size: Size preset (nano, small, base, large)
        vocab_size: For discrete mode
        in_channels: For continuous mode
        seq_length: Maximum sequence length
        timestep_aware: Enable timestep conditioning for SPO
        **kwargs: Additional arguments to RewardModel

    Returns:
        Configured RewardModel
    """
    configs = {
        "nano": {"hidden_size": 128, "depth": 2, "num_heads": 4},
        "small": {"hidden_size": 384, "depth": 6, "num_heads": 6},
        "base": {"hidden_size": 768, "depth": 12, "num_heads": 12},
        "large": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model_size '{model_size}'. Available: {list(configs.keys())}")

    config = configs[model_size]
    config.update(kwargs)

    return RewardModel(
        vocab_size=vocab_size,
        in_channels=in_channels,
        seq_length=seq_length,
        timestep_aware=timestep_aware,
        **config,
    )


__all__ = [
    "RewardModel",
    "RewardModelTrainer",
    "create_reward_model",
]
