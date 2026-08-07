"""Medusa: Multi-Head Speculative Decoding.

Reference:
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with
  Multiple Decoding Heads" (https://arxiv.org/abs/2401.10774)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedusaHead(nn.Module):
    """A single Medusa prediction head.

    Each head predicts tokens at a specific offset ahead of the current position.
    The base model predicts the next token, while Medusa heads predict
    further ahead (e.g., +2, +4, +8 tokens).

    Args:
        hidden_size: Hidden dimension of the base model
        vocab_size: Size of the vocabulary
        offset: How many tokens ahead this head predicts (1 = next token)
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        offset: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.offset = offset
        self.vocab_size = vocab_size

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] from base model

        Returns:
            logits: [batch, seq_len, vocab_size] predictions
        """
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MedusaModel(nn.Module):
    """Autoregressive model with Medusa speculative decoding heads.

    Wraps a base autoregressive model with multiple prediction heads
    for parallel speculative decoding.

    Args:
        base_model: Base autoregressive transformer
        num_medusa_heads: Number of Medusa heads (default 4)
        medusa_hidden_size: Hidden size for Medusa heads (default same as base)
        medusa_dropout: Dropout for Medusa heads
        medusa_offsets: Custom offsets for each head (default powers of 2)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_medusa_heads: int = 4,
        medusa_hidden_size: int | None = None,
        medusa_dropout: float = 0.0,
        medusa_offsets: list[int] | None = None,
    ):
        super().__init__()
        self.base_model = base_model

        if hasattr(base_model, "vocab_size"):
            vocab_size = base_model.vocab_size
        else:
            vocab_size = base_model.lm_head.weight.shape[0]

        if hasattr(base_model, "hidden_size"):
            hidden_size = base_model.hidden_size
        else:
            hidden_size = base_model.lm_head.weight.shape[1]

        self.vocab_size = vocab_size
        self.medusa_hidden_size = medusa_hidden_size or hidden_size

        if medusa_offsets is None:
            medusa_offsets = [2**i for i in range(num_medusa_heads)]
        else:
            medusa_offsets = medusa_offsets[:num_medusa_heads]

        self.medusa_offsets = medusa_offsets
        self.num_medusa_heads = len(medusa_offsets)

        self.medusa_heads = nn.ModuleList(
            [
                MedusaHead(
                    hidden_size=self.medusa_hidden_size,
                    vocab_size=vocab_size,
                    offset=offset,
                    dropout=medusa_dropout,
                )
                for offset in medusa_offsets
            ]
        )

        if self.medusa_hidden_size != hidden_size:
            self.hidden_proj = nn.Linear(hidden_size, self.medusa_hidden_size)
        else:
            self.hidden_proj = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with both base model and Medusa predictions.

        Args:
            input_ids: [batch, seq_len] input token IDs
            **kwargs: Additional arguments for base model

        Returns:
            Dict with:
                base_logits: [batch, seq_len, vocab_size] base model predictions
                medusa_logits: List of [batch, seq_len, vocab_size] for each head
        """
        base_output = self.base_model(input_ids, **kwargs)

        if isinstance(base_output, dict):
            hidden_states = base_output.get("hidden_states")
            base_logits = base_output.get("logits")
        elif hasattr(base_output, "last_hidden_state"):
            hidden_states = base_output.last_hidden_state
            base_logits = base_output.logits if hasattr(base_output, "logits") else None
        elif isinstance(base_output, tuple):
            hidden_states = base_output[0]
            base_logits = base_output[1] if len(base_output) > 1 else None
        else:
            hidden_states = None
            base_logits = base_output

        if hidden_states is not None:
            medusa_hidden = self.hidden_proj(hidden_states)
        else:
            medusa_hidden = None

        medusa_logits = []
        if medusa_hidden is not None:
            for head in self.medusa_heads:
                medusa_logits.append(head(medusa_hidden))

        return {
            "base_logits": base_logits,
            "medusa_logits": medusa_logits,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        max_medusa_tokens: int = 4,
    ) -> tuple[torch.Tensor, dict]:
        """Generate using speculative decoding with Medusa heads."""
        batch_size, seq_len = input_ids.shape

        generated = input_ids.clone()
        metrics = {
            "base_model_calls": 0,
            "medusa_accepts": 0,
            "medusa_rejects": 0,
            "total_tokens": 0,
        }

        while generated.shape[1] < max_length:
            outputs = self.forward(generated)
            base_logits = outputs["base_logits"]
            medusa_logits = outputs["medusa_logits"]

            metrics["base_model_calls"] += 1

            base_probs = F.softmax(base_logits[:, -1] / temperature, dim=-1)

            if top_k is not None:
                v, _ = torch.topk(base_probs, top_k)
                base_probs[base_probs < v[:, [-1]]] = 0
                base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(base_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                base_probs[indices_to_remove] = 0
                base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(base_probs, num_samples=1)

            medusa_tokens = []
            medusa_probs = []

            for head_logits in medusa_logits:
                head_probs = F.softmax(head_logits[:, -1] / temperature, dim=-1)
                head_token = torch.multinomial(head_probs, num_samples=1)
                medusa_tokens.append(head_token)
                medusa_probs.append(head_probs)

            speculated_tokens = [next_token]
            for i in range(min(len(medusa_tokens), max_medusa_tokens)):
                speculated_tokens.append(medusa_tokens[i])

            verify_ids = torch.cat([generated, torch.stack(speculated_tokens, dim=-1)], dim=1)
            verify_outputs = self.base_model(verify_ids)
            verify_logits = (
                verify_outputs
                if isinstance(verify_outputs, torch.Tensor)
                else verify_outputs.logits
            )

            n_accepted = 0
            for i in range(len(speculated_tokens)):
                expected_logits = verify_logits[:, generated.shape[1] + i - 1]
                expected_token = expected_logits.argmax(dim=-1, keepdim=True)

                if torch.all(speculated_tokens[i] == expected_token):
                    n_accepted += 1
                    metrics["medusa_accepts"] += 1
                else:
                    break

            metrics["medusa_rejects"] += len(speculated_tokens) - n_accepted

            if n_accepted > 0:
                accepted = torch.cat(speculated_tokens[:n_accepted], dim=-1)
                generated = torch.cat([generated, accepted], dim=1)
                metrics["total_tokens"] += accepted.shape[1]
            else:
                generated = torch.cat([generated, next_token], dim=1)
                metrics["total_tokens"] += 1

            if generated.shape[1] >= max_length:
                break

        return generated, metrics

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_speculative: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens (with or without speculative decoding)."""
        if use_speculative:
            generated, _ = self.generate_speculative(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
            return generated
        else:
            return self.base_model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )


class MultiTokenPredictionLoss(nn.Module):
    """Multi-token prediction auxiliary loss for training.

    Encourages the model to predict multiple future tokens simultaneously.

    Reference: "Lookahead Decoding: Fast Language Model Inference with
    Parallel Decoding" (Fu et al., 2023)

    Args:
        vocab_size: Vocabulary size
        num_future_tokens: Number of future tokens to predict (default 3)
        loss_weight: Weight for auxiliary loss (default 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        num_future_tokens: int = 3,
        loss_weight: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_future_tokens = num_future_tokens
        self.loss_weight = loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute multi-token prediction loss.

        Args:
            logits: [batch, seq_len, vocab_size] model predictions
            targets: [batch, seq_len] target tokens

        Returns:
            loss: Combined loss (base CE + multi-token auxiliary)
            metrics: Dict with loss components
        """
        batch_size, seq_len, vocab_size = logits.shape

        base_ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            targets[:, 1:].reshape(-1),
        )

        aux_losses = []
        for i in range(1, self.num_future_tokens + 1):
            if seq_len > i:
                pred_logits = logits[:, : -(i + 1)]
                future_targets = targets[:, (i + 1) :]

                if pred_logits.shape[1] > 0 and future_targets.shape[1] > 0:
                    min_len = min(pred_logits.shape[1], future_targets.shape[1])
                    pred_logits = pred_logits[:, :min_len]
                    future_targets = future_targets[:, :min_len]

                    aux_loss = F.cross_entropy(
                        pred_logits.reshape(-1, vocab_size),
                        future_targets.reshape(-1),
                    )
                    aux_losses.append(aux_loss)

        if aux_losses:
            aux_loss = sum(aux_losses) / len(aux_losses)
        else:
            aux_loss = torch.tensor(0.0, device=logits.device)

        total_loss = base_ce + self.loss_weight * aux_loss

        metrics = {
            "base_ce": base_ce.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics


class MedusaTrainer(nn.Module):
    """Training wrapper for Medusa models.

    Handles the joint training of base model and Medusa heads.

    Args:
        model: MedusaModel to train
        medusa_loss_weight: Weight for Medusa head losses
        multi_token_loss: Optional multi-token prediction loss
    """

    def __init__(
        self,
        model: MedusaModel,
        medusa_loss_weight: float = 1.0,
        multi_token_loss: MultiTokenPredictionLoss | None = None,
    ):
        super().__init__()
        self.model = model
        self.medusa_loss_weight = medusa_loss_weight
        self.multi_token_loss = multi_token_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute training loss.

        Args:
            input_ids: [batch, seq_len] input tokens
            targets: [batch, seq_len] target tokens (default: input_ids shifted)

        Returns:
            Dict with loss and metrics
        """
        if targets is None:
            targets = input_ids.clone()

        outputs = self.model(input_ids)
        base_logits = outputs["base_logits"]
        medusa_logits = outputs["medusa_logits"]

        base_ce = F.cross_entropy(
            base_logits[:, :-1].reshape(-1, self.model.vocab_size),
            targets[:, 1:].reshape(-1),
        )

        medusa_losses = []
        for i, head_logits in enumerate(medusa_logits):
            offset = self.model.medusa_offsets[i]
            if targets.shape[1] > offset:
                pred_logits = head_logits[:, :-(offset)]
                future_targets = targets[:, offset:]

                min_len = min(pred_logits.shape[1], future_targets.shape[1])
                if min_len > 0:
                    pred_logits = pred_logits[:, :min_len]
                    future_targets = future_targets[:, :min_len]

                    medusa_losses.append(
                        F.cross_entropy(
                            pred_logits.reshape(-1, self.model.vocab_size),
                            future_targets.reshape(-1),
                        )
                    )

        if medusa_losses:
            medusa_loss = sum(medusa_losses) / len(medusa_losses)
        else:
            medusa_loss = torch.tensor(0.0, device=base_logits.device)

        if self.multi_token_loss is not None:
            aux_ce, aux_metrics = self.multi_token_loss(base_logits, targets)
        else:
            aux_ce = torch.tensor(0.0, device=base_logits.device)
            aux_metrics = {}

        total_loss = base_ce + self.medusa_loss_weight * medusa_loss + aux_ce

        return {
            "loss": total_loss,
            "base_ce": base_ce,
            "medusa_loss": medusa_loss,
            "aux_loss": aux_ce,
            **aux_metrics,
        }


__all__ = [
    "MedusaHead",
    "MedusaModel",
    "MultiTokenPredictionLoss",
    "MedusaTrainer",
]
