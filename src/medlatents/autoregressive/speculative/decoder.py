"""Speculative decoding core: single and multi-draft decoders.

Reference:
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"
  (https://arxiv.org/abs/2302.01318)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable


class SpeculativeDecoder:
    """
    Speculative decoding wrapper for autoregressive models.

    Uses a smaller draft model to predict multiple tokens ahead,
    then verifies them with the main model in parallel.

    Args:
        main_model: The larger, more accurate model
        draft_model: The smaller, faster draft model (can be same as main)
        vocab_size: Vocabulary size
        max_speculation: Maximum number of tokens to speculate (K)
        temperature: Sampling temperature
        top_k: Optional top-k sampling
        top_p: Optional nucleus sampling threshold
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_model: nn.Module,
        vocab_size: int,
        max_speculation: int = 4,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.vocab_size = vocab_size
        self.max_speculation = max_speculation
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    @torch.no_grad()
    def _sample_token(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a token from logits with optional filtering.

        Args:
            logits: [batch, vocab_size] unnormalized logits

        Returns:
            token: [batch] sampled token indices
        """
        logits = logits / self.temperature

        if self.top_k is not None:
            logits = self._apply_top_k(logits, self.top_k)

        if self.top_p is not None:
            logits = self._apply_top_p(logits, self.top_p)

        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return tokens

    def _apply_top_k(
        self,
        logits: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Keep only top k logits."""
        top_k = min(k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, [-1]]
        return torch.where(
            logits < min_values,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """Keep smallest set of tokens with cumulative prob >= p."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        return torch.where(
            indices_to_remove,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    @torch.no_grad()
    def _verify_speculation(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
        main_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Verify draft tokens against main model predictions.

        Args:
            context: [batch, seq_len] context tokens
            draft_tokens: [batch, K] draft predicted tokens
            main_logits: [batch, K+1, vocab_size] main model logits

        Returns:
            accepted_mask: [batch, K] boolean mask of accepted tokens
            num_accepted: Number of accepted tokens (first rejection point)
        """
        batch_size, K = draft_tokens.shape

        main_probs = F.softmax(main_logits / self.temperature, dim=-1)

        draft_probs = torch.zeros(batch_size, K, device=context.device)
        for i in range(K):
            draft_probs[:, i] = (
                main_probs[:, i]
                .gather(
                    dim=-1,
                    index=draft_tokens[:, i].unsqueeze(-1),
                )
                .squeeze(-1)
            )

        uniform_samples = torch.rand(batch_size, K, device=context.device)

        acceptance_ratio = draft_probs / (draft_probs + 1e-10)
        accepted_mask = uniform_samples < acceptance_ratio

        num_accepted = 0
        for i in range(K):
            if accepted_mask[:, i].all():
                num_accepted += 1
            else:
                break

        return accepted_mask, num_accepted

    @torch.no_grad()
    def generate(
        self,
        initial_tokens: torch.Tensor,
        max_new_tokens: int,
        eos_token: int | None = None,
        callback: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate tokens using speculative decoding.

        Args:
            initial_tokens: [batch, seq_len] initial context
            max_new_tokens: Maximum number of tokens to generate
            eos_token: Optional end-of-sequence token
            callback: Optional callback called after each step

        Returns:
            tokens: [batch, seq_len + generated] generated sequence
            metrics: Dict with generation statistics
        """
        batch_size = initial_tokens.shape[0]

        tokens = initial_tokens.clone()
        total_generated = 0
        total_draft_tokens = 0
        total_accepted = 0

        while total_generated < max_new_tokens:
            if eos_token is not None:
                has_eos = (tokens == eos_token).any(dim=-1)
                if has_eos.all():
                    break

            remaining = max_new_tokens - total_generated
            K = min(self.max_speculation, remaining)

            draft_tokens = self._draft_generate(tokens, K)

            context_len = tokens.shape[1]
            verification_input = torch.cat([tokens, draft_tokens], dim=-1)

            main_logits = self.main_model(verification_input)
            main_logits = main_logits[:, context_len - 1 : context_len + K]

            accepted_mask, num_accepted = self._verify_speculation(
                tokens,
                draft_tokens,
                main_logits,
            )

            if num_accepted > 0:
                tokens_to_append = draft_tokens[:, :num_accepted]
                tokens = torch.cat([tokens, tokens_to_append], dim=-1)
                total_generated += num_accepted
                total_accepted += num_accepted * batch_size

            if num_accepted < K:
                reject_logits = main_logits[:, num_accepted]
                reject_token = self._sample_token(reject_logits)
                tokens = torch.cat([tokens, reject_token.unsqueeze(-1)], dim=-1)
                total_generated += 1

            total_draft_tokens += K * batch_size

            if callback is not None:
                callback(tokens)

        metrics = {
            "total_tokens": total_generated * batch_size,
            "draft_tokens": total_draft_tokens,
            "accepted_tokens": total_accepted,
            "acceptance_rate": total_accepted / total_draft_tokens if total_draft_tokens > 0 else 0,
        }

        return tokens, metrics

    @torch.no_grad()
    def _draft_generate(
        self,
        context: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Generate K draft tokens using the draft model."""
        batch_size = context.shape[0]
        draft_tokens = torch.zeros(batch_size, K, device=context.device, dtype=torch.long)
        current = context.clone()

        for i in range(K):
            logits = self.draft_model(current)[:, -1]
            token = self._sample_token(logits)
            draft_tokens[:, i] = token
            current = torch.cat([current, token.unsqueeze(-1)], dim=-1)

        return draft_tokens


class MultiDraftSpeculativeDecoder(SpeculativeDecoder):
    """
    Speculative decoding with multiple independent draft models.

    Uses multiple draft models in parallel and selects the best predictions
    for verification, improving the acceptance rate.

    Args:
        main_model: The main model for verification
        draft_models: List of draft models
        vocab_size: Vocabulary size
        max_speculation: Maximum tokens to speculate
        selection_strategy: How to select from multiple drafts
            - "ensemble": Average draft logits
            - "voting": Majority vote
            - "best": Select draft with lowest perplexity
    """

    def __init__(
        self,
        main_model: nn.Module,
        draft_models: list[nn.Module],
        vocab_size: int,
        max_speculation: int = 4,
        selection_strategy: str = "ensemble",
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        super().__init__(
            main_model=main_model,
            draft_model=draft_models[0],
            vocab_size=vocab_size,
            max_speculation=max_speculation,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        self.draft_models = draft_models
        self.selection_strategy = selection_strategy

    @torch.no_grad()
    def _draft_generate(
        self,
        context: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Generate K draft tokens using multiple draft models."""
        batch_size = context.shape[0]
        device = context.device

        all_draft_tokens = []
        all_draft_logits = []

        for draft_model in self.draft_models:
            draft_tokens = torch.zeros(batch_size, K, device=device, dtype=torch.long)
            draft_logits_list = []
            current = context.clone()

            for i in range(K):
                logits = draft_model(current)[:, -1]
                draft_logits_list.append(logits)
                token = self._sample_token(logits)
                draft_tokens[:, i] = token
                current = torch.cat([current, token.unsqueeze(-1)], dim=-1)

            all_draft_tokens.append(draft_tokens)
            all_draft_logits.append(torch.stack(draft_logits_list, dim=1))

        return self._select_draft(all_draft_tokens, all_draft_logits)

    def _select_draft(
        self,
        all_draft_tokens: list[torch.Tensor],
        all_draft_logits: list[torch.Tensor],
    ) -> torch.Tensor:
        """Select the best draft from multiple models."""
        if self.selection_strategy == "ensemble":
            avg_logits = torch.stack(all_draft_logits, dim=0).mean(dim=0)
            batch_size, K, vocab_size = avg_logits.shape
            selected_tokens = torch.zeros(batch_size, K, device=avg_logits.device, dtype=torch.long)

            for i in range(K):
                selected_tokens[:, i] = self._sample_token(avg_logits[:, i])

            return selected_tokens

        elif self.selection_strategy == "voting":
            stacked = torch.stack(all_draft_tokens, dim=0)
            selected_tokens, _ = torch.mode(stacked, dim=0)
            return selected_tokens.squeeze(0)

        elif self.selection_strategy == "best":
            best_idx = 0
            best_perplexity = float("inf")

            for i, logits in enumerate(all_draft_logits):
                probs = F.softmax(logits, dim=-1)
                token_probs = torch.gather(probs, -1, all_draft_tokens[i].unsqueeze(-1)).squeeze(-1)
                ppl = (-torch.log(token_probs.clamp_min(1e-10))).mean()
                if ppl < best_perplexity:
                    best_perplexity = ppl
                    best_idx = i

            return all_draft_tokens[best_idx]

        else:
            return all_draft_tokens[0]


__all__ = [
    "SpeculativeDecoder",
    "MultiDraftSpeculativeDecoder",
]
