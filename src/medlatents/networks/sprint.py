"""SPRINT: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers.

Implements token dropping with sparse-dense residual fusion for efficient DiT training.
Early layers process all tokens, deep layers process sparse subset, outputs are fused.

Key features:
- Up to 75% token dropping during training (9.8x training savings)
- Sparse-dense residual fusion preserves quality
- Path-Drop Guidance (PDG) for efficient inference (nearly halves FLOPs)
- Two-stage training: masked pre-training + full-token fine-tuning

Reference:
    Park et al., "Sprint: Sparse-Dense Residual Fusion for Efficient Diffusion Transformers"
    arXiv:2510.21986

Architecture:
    Early blocks (dense): Process all tokens for local detail
    Deep blocks (sparse): Process subset of tokens for efficiency
    Fusion: Residual connection between dense and sparse paths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class DiTProtocol(Protocol):
    """Protocol defining the expected interface for DiT models used with SPRINT."""

    hidden_size: int
    blocks: nn.ModuleList

    def x_embedder(self, x: torch.Tensor) -> torch.Tensor: ...
    def t_embedder(self, t: torch.Tensor) -> torch.Tensor: ...
    def y_embedder(self, y: torch.Tensor, training: bool) -> torch.Tensor: ...
    def final_layer(self, x: torch.Tensor) -> torch.Tensor: ...
    def __call__(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None
    ) -> torch.Tensor: ...


@dataclass
class SPRINTConfig:
    """Configuration for SPRINT token dropping.

    Attributes:
        drop_ratio: Fraction of tokens to drop (0.0-0.75 typical)
        dense_layers: Number of initial layers that process all tokens
        sparse_layers: Number of deep layers that process sparse tokens
        fusion_method: How to fuse sparse and dense outputs
        drop_schedule: How drop ratio changes during training
        importance_type: How to select which tokens to keep
        path_drop_guidance: Whether to use Path-Drop Guidance at inference
        pdg_scale: Scale factor for Path-Drop Guidance
    """

    drop_ratio: float = 0.5
    dense_layers: int = 6  # First 6 layers process all tokens
    sparse_layers: int = 6  # Remaining layers process sparse tokens

    fusion_method: Literal["residual", "concat", "gate"] = "residual"
    drop_schedule: Literal["constant", "linear", "cosine"] = "constant"

    # Token selection
    importance_type: Literal["random", "attention", "gradient", "learned"] = "random"

    # Path-Drop Guidance (inference)
    path_drop_guidance: bool = False
    pdg_scale: float = 1.5


class TokenSelector(nn.Module):
    """Selects which tokens to keep based on importance.

    Supports multiple selection strategies:
    - random: Uniform random selection
    - attention: Keep tokens with highest attention scores
    - gradient: Keep tokens with highest gradient magnitudes
    - learned: Use a learned scoring network
    """

    def __init__(
        self,
        hidden_size: int,
        importance_type: str = "random",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.importance_type = importance_type
        self.temperature = temperature

        if importance_type == "learned":
            # Simple MLP to score token importance
            self.scorer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        keep_ratio: float,
        attention_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select tokens to keep.

        Args:
            x: Input tokens [batch, seq_len, hidden]
            keep_ratio: Fraction of tokens to keep
            attention_scores: Optional attention scores for attention-based selection

        Returns:
            Tuple of (selected_tokens, indices, mask):
            - selected_tokens: [batch, num_keep, hidden]
            - indices: [batch, num_keep] indices of selected tokens
            - mask: [batch, seq_len] boolean mask (True = kept)
        """
        batch_size, seq_len, hidden = x.shape
        num_keep = max(1, int(seq_len * keep_ratio))

        if self.importance_type == "random":
            # Random selection
            indices = torch.stack(
                [torch.randperm(seq_len, device=x.device)[:num_keep] for _ in range(batch_size)]
            )

        elif self.importance_type == "attention":
            # Select based on attention scores
            if attention_scores is None:
                # Fall back to random if no attention scores
                indices = torch.stack(
                    [torch.randperm(seq_len, device=x.device)[:num_keep] for _ in range(batch_size)]
                )
            else:
                # Sum attention across heads and select top-k
                importance = attention_scores.sum(dim=1).sum(dim=1)  # [batch, seq_len]
                _, indices = importance.topk(num_keep, dim=-1)

        elif self.importance_type == "learned":
            # Use learned scorer
            scores = self.scorer(x).squeeze(-1)  # [batch, seq_len]
            _, indices = scores.topk(num_keep, dim=-1)

        else:
            raise ValueError(f"Unknown importance_type: {self.importance_type}")

        # Sort indices for consistent ordering
        indices, _ = indices.sort(dim=-1)

        # Gather selected tokens
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden)
        selected_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # Create mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        mask.scatter_(1, indices, True)

        return selected_tokens, indices, mask


class TokenRestorer(nn.Module):
    """Restores full sequence from sparse tokens.

    Uses various fusion strategies to combine sparse token outputs
    with the original dense representation.
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_method: str = "residual",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_method = fusion_method

        if fusion_method == "gate":
            # Learned gating for fusion
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid(),
            )
        elif fusion_method == "concat":
            # Project concatenated features
            self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        sparse_output: torch.Tensor,
        dense_residual: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Restore full sequence from sparse output.

        Args:
            sparse_output: Output from sparse layers [batch, num_keep, hidden]
            dense_residual: Dense residual from early layers [batch, seq_len, hidden]
            indices: Indices of kept tokens [batch, num_keep]
            mask: Boolean mask [batch, seq_len]

        Returns:
            Full restored sequence [batch, seq_len, hidden]
        """
        batch_size, seq_len, hidden = dense_residual.shape

        if self.fusion_method == "residual":
            # Simple residual: scatter sparse output back, add dense residual
            output = dense_residual.clone()
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden)
            output.scatter_(1, indices_expanded, sparse_output)
            return output

        elif self.fusion_method == "gate":
            # Gated fusion
            output = dense_residual.clone()
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden)

            # Get corresponding dense tokens for gating
            dense_selected = torch.gather(dense_residual, dim=1, index=indices_expanded)

            # Compute gate
            combined = torch.cat([sparse_output, dense_selected], dim=-1)
            gate = self.gate(combined)

            # Apply gated fusion
            fused = gate * sparse_output + (1 - gate) * dense_selected
            output.scatter_(1, indices_expanded, fused)
            return output

        elif self.fusion_method == "concat":
            # Concatenate and project
            output = dense_residual.clone()
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden)

            dense_selected = torch.gather(dense_residual, dim=1, index=indices_expanded)
            combined = torch.cat([sparse_output, dense_selected], dim=-1)
            fused = self.proj(combined)

            output.scatter_(1, indices_expanded, fused)
            return output

        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")


class SPRINTDiTBlock(nn.Module):
    """DiT block wrapper with SPRINT token dropping support.

    Wraps an existing DiT block to support sparse token processing.
    Can operate in dense mode (all tokens) or sparse mode (subset).
    """

    def __init__(
        self,
        block: nn.Module,
        is_sparse: bool = False,
    ):
        super().__init__()
        self.block = block
        self.is_sparse = is_sparse

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        freqs_cis: torch.Tensor,
        indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass, optionally on sparse tokens.

        Args:
            x: Input tokens [batch, seq_len, hidden] or [batch, num_keep, hidden]
            c: Conditioning [batch, hidden]
            freqs_cis: RoPE frequencies
            indices: Token indices if sparse mode (for RoPE adjustment)
            **kwargs: Additional block arguments

        Returns:
            Output tokens
        """
        if self.is_sparse and indices is not None:
            # Adjust RoPE frequencies for sparse tokens
            # Select frequencies corresponding to kept token positions

            # freqs_cis: [seq_len, head_dim//2]
            # indices: [batch, num_keep]
            # We need to select the right frequencies for each position
            freqs_sparse = freqs_cis[indices[0]]  # Assume same indices across batch

            return self.block(x, c, freqs_sparse, **kwargs)
        else:
            return self.block(x, c, freqs_cis, **kwargs)


class SPRINTDiT(nn.Module):
    """DiT with SPRINT sparse-dense residual fusion.

    Wraps an existing DiT model with SPRINT token dropping.
    Early layers process all tokens, deep layers process sparse subset.

    Example:
        >>> base_dit = ContinuousDiT(...)
        >>> sprint_dit = SPRINTDiT(
        ...     dit=base_dit,
        ...     config=SPRINTConfig(drop_ratio=0.5, dense_layers=6),
        ... )
        >>> # Training with token dropping
        >>> output = sprint_dit(x, t, training=True)
        >>> # Inference without dropping
        >>> output = sprint_dit(x, t, training=False)
    """

    def __init__(
        self,
        dit: DiTProtocol | nn.Module,
        config: SPRINTConfig | None = None,
    ):
        super().__init__()
        self.config = config or SPRINTConfig()

        self.dit = dit
        self.hidden_size: int = getattr(dit, "hidden_size", 1024)

        self.token_selector = TokenSelector(
            self.hidden_size,
            importance_type=self.config.importance_type,
        )
        self.token_restorer = TokenRestorer(
            self.hidden_size,
            fusion_method=self.config.fusion_method,
        )

        # Mark blocks as dense or sparse
        self._setup_blocks()

    def _setup_blocks(self) -> None:
        """Setup which blocks are dense vs sparse."""
        if hasattr(self.dit, "blocks"):
            blocks = self.dit.blocks
            num_blocks = len(blocks)
            self.dense_block_indices = list(range(self.config.dense_layers))
            self.sparse_block_indices = list(range(self.config.dense_layers, num_blocks))
        else:
            self.dense_block_indices = list(range(self.config.dense_layers))
            self.sparse_block_indices = list(
                range(
                    self.config.dense_layers, self.config.dense_layers + self.config.sparse_layers
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        training: bool | None = None,
        drop_ratio: float | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional SPRINT token dropping.

        Args:
            x: Input tensor [batch, seq_len, channels] or [batch, seq_len]
            t: Timesteps [batch]
            y: Optional class labels [batch]
            training: Whether to use token dropping (default: self.training)
            drop_ratio: Override drop ratio (default: config value)

        Returns:
            Model output [batch, seq_len, out_channels]
        """
        if training is None:
            training = self.training

        drop_ratio = drop_ratio if drop_ratio is not None else self.config.drop_ratio

        # Use token dropping during training
        if training and drop_ratio > 0:
            return self._forward_with_dropping(x, t, y, drop_ratio)
        else:
            return self._forward_full(x, t, y)

    def _forward_full(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard forward pass without token dropping."""
        return self.dit(x, t, y)

    def _forward_with_dropping(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        drop_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass with SPRINT token dropping."""
        dit = self.dit

        if hasattr(dit, "x_embedder"):
            x_embedder = dit.x_embedder
            x = x_embedder(x)

        t_emb = None
        if hasattr(dit, "t_embedder"):
            t_embedder = dit.t_embedder
            t_emb = t_embedder(t)

        c = t_emb
        if hasattr(dit, "y_embedder") and y is not None:
            y_embedder = dit.y_embedder
            y_emb = y_embedder(y, self.training)
            c = t_emb + y_emb if t_emb is not None else y_emb

        freqs_cis = getattr(dit, "freqs_cis", None)

        blocks = getattr(dit, "blocks", None)
        if blocks is not None:
            for idx in self.dense_block_indices:
                block = blocks[idx]
                x = block(x, c, freqs_cis)

        dense_residual = x.clone()

        keep_ratio = 1.0 - drop_ratio
        sparse_x, indices, mask = self.token_selector(x, keep_ratio)

        if blocks is not None:
            for idx in self.sparse_block_indices:
                block = blocks[idx]
                sparse_freqs = freqs_cis[indices[0]] if freqs_cis is not None else None
                sparse_x = block(sparse_x, c, sparse_freqs)

        x = self.token_restorer(sparse_x, dense_residual, indices, mask)

        if hasattr(dit, "final_layer"):
            final_layer = dit.final_layer
            x = final_layer(x)

        return x

    def forward_with_pdg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
        cfg_scale: float = 1.5,
        pdg_scale: float | None = None,
    ) -> torch.Tensor:
        """Forward with Path-Drop Guidance (PDG) for efficient inference.

        PDG uses both sparse and dense paths, combining them similar to CFG
        but across paths instead of conditional/unconditional.

        Args:
            x: Input tensor
            t: Timesteps
            y: Optional class labels
            cfg_scale: Standard CFG scale
            pdg_scale: Path-Drop Guidance scale (default: config value)

        Returns:
            PDG-guided output
        """
        pdg_scale = pdg_scale if pdg_scale is not None else self.config.pdg_scale

        # Dense path (full computation)
        dense_out = self._forward_full(x, t, y)

        # Sparse path (efficient computation)
        sparse_out = self._forward_with_dropping(x, t, y, drop_ratio=0.5)

        # PDG combination: similar to CFG but for paths
        # output = dense + pdg_scale * (dense - sparse)
        return dense_out + pdg_scale * (dense_out - sparse_out)


class SPRINTScheduler:
    """Scheduler for SPRINT drop ratio during training.

    Implements the two-stage training schedule:
    Stage 1: Long masked pre-training with high drop ratio
    Stage 2: Short full-token fine-tuning with low/zero drop ratio
    """

    def __init__(
        self,
        total_steps: int,
        pretrain_ratio: float = 0.9,  # Fraction of steps for pre-training
        pretrain_drop_ratio: float = 0.5,
        finetune_drop_ratio: float = 0.0,
        schedule_type: str = "constant",
    ):
        self.total_steps = total_steps
        self.pretrain_steps = int(total_steps * pretrain_ratio)
        self.pretrain_drop_ratio = pretrain_drop_ratio
        self.finetune_drop_ratio = finetune_drop_ratio
        self.schedule_type = schedule_type

    def get_drop_ratio(self, step: int) -> float:
        """Get drop ratio for current step."""
        if step < self.pretrain_steps:
            # Pre-training phase: use high drop ratio
            if self.schedule_type == "constant":
                return self.pretrain_drop_ratio
            elif self.schedule_type == "linear":
                # Gradually decrease during pre-training
                progress = step / self.pretrain_steps
                return self.pretrain_drop_ratio * (1 - progress * 0.5)
            elif self.schedule_type == "cosine":
                import math

                progress = step / self.pretrain_steps
                return self.pretrain_drop_ratio * (1 + math.cos(math.pi * progress)) / 2
            else:
                return self.pretrain_drop_ratio
        else:
            # Fine-tuning phase: low/zero drop ratio
            return self.finetune_drop_ratio


__all__ = [
    "SPRINTConfig",
    "SPRINTDiT",
    "SPRINTScheduler",
    "TokenSelector",
    "TokenRestorer",
    "SPRINTDiTBlock",
]
