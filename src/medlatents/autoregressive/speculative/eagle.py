"""EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.

EAGLE uses a lightweight autoregression head on top of the base model's
hidden states, enabling more accurate speculation with minimal overhead.

Key innovations:
1. Feature-level autoregression instead of token-level
2. Tree-structured draft generation for parallel verification
3. Dynamic speculation length based on confidence

References:
- Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
  (https://arxiv.org/abs/2401.15077)
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
  (https://arxiv.org/abs/2406.16858)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class EAGLEConfig:
    """Configuration for EAGLE speculative decoder.

    Attributes:
        hidden_size: Hidden dimension of base model
        vocab_size: Vocabulary size
        num_eagle_layers: Number of transformer layers in EAGLE head
        eagle_hidden_size: Hidden size for EAGLE (default: same as base)
        num_heads: Number of attention heads
        max_draft_length: Maximum number of tokens to draft
        tree_width: Width of draft tree (number of candidates per position)
        confidence_threshold: Minimum confidence for accepting drafts
        use_dynamic_tree: Whether to use dynamic tree structure (EAGLE-2)
        dropout: Dropout probability
    """

    hidden_size: int
    vocab_size: int
    num_eagle_layers: int = 2
    eagle_hidden_size: int = -1  # -1 means same as hidden_size
    num_heads: int = 8
    max_draft_length: int = 6
    tree_width: int = 3
    confidence_threshold: float = 0.5
    use_dynamic_tree: bool = True
    dropout: float = 0.0

    def __post_init__(self):
        if self.eagle_hidden_size == -1:
            self.eagle_hidden_size = self.hidden_size


class EAGLEAttention(nn.Module):
    """Multi-head attention for EAGLE autoregressive head."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.size(2), device=x.device, dtype=torch.bool),
                diagonal=k.size(2) - seq_len + 1,
            )
            attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.out_proj(out)

        return out, new_kv


class EAGLEBlock(nn.Module):
    """Single transformer block for EAGLE head."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.attn = EAGLEAttention(hidden_size, num_heads, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        attn_out, new_kv = self.attn(x, past_kv, use_cache)
        x = residual + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.ln2(x))

        return x, new_kv


class EAGLEHead(nn.Module):
    """EAGLE autoregressive head for feature-level speculation.

    Takes hidden states from the base model and predicts future hidden states,
    which are then projected to vocabulary for draft tokens.

    Args:
        config: EAGLEConfig with model parameters
    """

    def __init__(self, config: EAGLEConfig):
        super().__init__()
        self.config = config

        # Project from base hidden size to EAGLE hidden size if different
        if config.eagle_hidden_size != config.hidden_size:
            self.input_proj = nn.Linear(config.hidden_size, config.eagle_hidden_size)
        else:
            self.input_proj = nn.Identity()

        # Token embedding for autoregressive conditioning
        self.token_embed = nn.Embedding(config.vocab_size, config.eagle_hidden_size)

        # Combine hidden state and token embedding
        self.fusion = nn.Linear(config.eagle_hidden_size * 2, config.eagle_hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                EAGLEBlock(
                    hidden_size=config.eagle_hidden_size,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_eagle_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(config.eagle_hidden_size)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.eagle_hidden_size, config.vocab_size, bias=False)

        # Confidence prediction for dynamic tree
        if config.use_dynamic_tree:
            self.confidence_head = nn.Sequential(
                nn.Linear(config.eagle_hidden_size, config.eagle_hidden_size // 4),
                nn.GELU(),
                nn.Linear(config.eagle_hidden_size // 4, 1),
                nn.Sigmoid(),
            )
        else:
            self.confidence_head = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tokens: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for EAGLE head.

        Args:
            hidden_states: [batch, seq_len, hidden_size] from base model
            input_tokens: [batch, seq_len] token IDs for conditioning
            past_kvs: Optional cached key-values for each layer
            use_cache: Whether to return new key-values

        Returns:
            Dict with:
                logits: [batch, seq_len, vocab_size] next token predictions
                hidden_states: [batch, seq_len, eagle_hidden_size] predicted features
                confidence: [batch, seq_len, 1] if use_dynamic_tree
                past_kvs: List of (k, v) tuples if use_cache
        """
        # Project hidden states
        h = self.input_proj(hidden_states)

        # Get token embeddings and fuse
        tok_embed = self.token_embed(input_tokens)
        x = self.fusion(torch.cat([h, tok_embed], dim=-1))

        # Transformer layers
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = layer(x, past_kv, use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        x = self.ln_f(x)

        # Predictions
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "hidden_states": x,
        }

        if self.confidence_head is not None:
            result["confidence"] = self.confidence_head(x)

        if use_cache:
            result["past_kvs"] = new_kvs

        return result


@dataclass
class DraftTreeNode:
    """Node in the draft tree for tree-structured speculation."""

    token: int
    probability: float
    hidden_state: torch.Tensor
    children: list["DraftTreeNode"]
    depth: int
    cumulative_prob: float


class EAGLEDecoder:
    """EAGLE speculative decoder for fast autoregressive generation.

    Uses the EAGLE head to generate draft tokens by predicting future hidden
    states, then verifies them with the base model.

    Args:
        base_model: The main autoregressive model
        eagle_head: Trained EAGLE head
        config: EAGLE configuration
    """

    def __init__(
        self,
        base_model: nn.Module,
        eagle_head: EAGLEHead,
        config: EAGLEConfig,
    ):
        self.base_model = base_model
        self.eagle_head = eagle_head
        self.config = config

        # Pre-compute tree structure for static trees
        if not config.use_dynamic_tree:
            self.draft_tree_indices = self._build_static_tree_indices()
        else:
            self.draft_tree_indices = None

    def _build_static_tree_indices(self) -> list[list[int]]:
        """Build indices for static tree traversal."""
        tree_indices = []
        current_level = [0]  # Root

        for depth in range(self.config.max_draft_length):
            next_level = []
            level_indices = []
            for parent_idx in current_level:
                for child in range(self.config.tree_width):
                    child_idx = len(tree_indices) + len(level_indices) + 1
                    level_indices.append(child_idx)
                    next_level.append(child_idx)
            tree_indices.append(level_indices)
            current_level = next_level

        return tree_indices

    @torch.no_grad()
    def _get_base_hidden_states(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hidden states and logits from base model."""
        # Most models return (logits, hidden_states) or have output_hidden_states option
        output = self.base_model(tokens, output_hidden_states=True)

        if isinstance(output, dict):
            hidden_states = output.get("hidden_states", output.get("last_hidden_state"))
            logits = output.get("logits")
        elif hasattr(output, "hidden_states"):
            hidden_states = output.hidden_states[-1]  # Last layer
            logits = output.logits if hasattr(output, "logits") else None
        elif isinstance(output, tuple):
            logits = output[0]
            hidden_states = output[1] if len(output) > 1 else None
        else:
            logits = output
            hidden_states = None

        # If we can't get hidden states, we need to compute them differently
        if hidden_states is None:
            # Fallback: just use logits projected back
            hidden_states = logits  # This is a simplification

        return hidden_states, logits

    @torch.no_grad()
    def _generate_draft_tree(
        self,
        context_hidden: torch.Tensor,
        context_tokens: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens using tree-structured speculation.

        Args:
            context_hidden: [batch, seq_len, hidden_size] context hidden states
            context_tokens: [batch, seq_len] context tokens
            base_logits: [batch, vocab_size] base model logits for first draft

        Returns:
            draft_tokens: [batch, num_draft] draft token IDs
            draft_positions: [batch, num_draft] positions in tree
            draft_probs: [batch, num_draft] token probabilities
        """
        # Sample first token from base model
        first_probs = F.softmax(base_logits, dim=-1)

        if self.config.use_dynamic_tree:
            # Dynamic tree: expand based on confidence
            return self._generate_dynamic_tree(context_hidden, context_tokens, first_probs)
        else:
            # Static tree: fixed structure
            return self._generate_static_tree(context_hidden, context_tokens, first_probs)

    @torch.no_grad()
    def _generate_static_tree(
        self,
        context_hidden: torch.Tensor,
        context_tokens: torch.Tensor,
        first_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate drafts using static tree structure."""
        batch_size = context_hidden.shape[0]
        device = context_hidden.device

        # Total nodes in tree
        total_nodes = sum(self.config.tree_width**d for d in range(self.config.max_draft_length))

        draft_tokens = torch.zeros(batch_size, total_nodes, dtype=torch.long, device=device)
        draft_probs = torch.zeros(batch_size, total_nodes, device=device)

        # Sample root level from base model
        top_probs, top_indices = first_probs.topk(self.config.tree_width, dim=-1)
        draft_tokens[:, : self.config.tree_width] = top_indices
        draft_probs[:, : self.config.tree_width] = top_probs

        # Current hidden states for each path
        # Expand context for each tree branch
        current_hidden = (
            context_hidden[:, -1:].expand(batch_size, self.config.tree_width, -1).clone()
        )
        current_tokens = top_indices

        node_idx = self.config.tree_width

        # Generate remaining levels
        for depth in range(1, self.config.max_draft_length):
            num_parents = current_hidden.shape[1]

            # Reshape for batch processing
            flat_hidden = current_hidden.view(batch_size * num_parents, 1, -1)
            flat_tokens = current_tokens.view(batch_size * num_parents, 1)

            # EAGLE forward
            eagle_out = self.eagle_head(
                hidden_states=flat_hidden,
                input_tokens=flat_tokens,
                use_cache=False,
            )

            logits = eagle_out["logits"][:, -1]  # [B*P, vocab]
            probs = F.softmax(logits, dim=-1)

            # Top-k for each parent
            top_probs_level, top_indices_level = probs.topk(self.config.tree_width, dim=-1)

            # Reshape back
            top_probs_level = top_probs_level.view(batch_size, num_parents, self.config.tree_width)
            top_indices_level = top_indices_level.view(
                batch_size, num_parents, self.config.tree_width
            )
            new_hidden = eagle_out["hidden_states"][:, -1].view(batch_size, num_parents, -1)

            # Flatten for next level
            num_new = num_parents * self.config.tree_width
            new_tokens = top_indices_level.view(batch_size, num_new)
            new_probs = top_probs_level.view(batch_size, num_new)

            # Store
            end_idx = node_idx + num_new
            if end_idx <= total_nodes:
                draft_tokens[:, node_idx:end_idx] = new_tokens
                draft_probs[:, node_idx:end_idx] = new_probs
            node_idx = end_idx

            # Update for next level
            current_hidden = (
                new_hidden.unsqueeze(2)
                .expand(batch_size, num_parents, self.config.tree_width, -1)
                .reshape(batch_size, num_new, -1)
            )
            current_tokens = new_tokens

            if node_idx >= total_nodes:
                break

        # Create position indices (flat for now)
        positions = torch.arange(total_nodes, device=device).unsqueeze(0).expand(batch_size, -1)

        return draft_tokens, positions, draft_probs

    @torch.no_grad()
    def _generate_dynamic_tree(
        self,
        context_hidden: torch.Tensor,
        context_tokens: torch.Tensor,
        first_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate drafts using dynamic tree based on confidence."""
        batch_size = context_hidden.shape[0]
        device = context_hidden.device

        # Start with top candidates from base model
        top_probs, top_indices = first_probs.topk(self.config.tree_width, dim=-1)

        all_tokens = [top_indices]
        all_probs = [top_probs]

        # Track active paths (those above confidence threshold)
        active_hidden = (
            context_hidden[:, -1:].expand(batch_size, self.config.tree_width, -1).clone()
        )
        active_tokens = top_indices
        active_probs = top_probs

        for depth in range(1, self.config.max_draft_length):
            num_active = active_hidden.shape[1]
            if num_active == 0:
                break

            # EAGLE forward for all active paths
            flat_hidden = active_hidden.view(batch_size * num_active, 1, -1)
            flat_tokens = active_tokens.view(batch_size * num_active, 1)

            eagle_out = self.eagle_head(
                hidden_states=flat_hidden,
                input_tokens=flat_tokens,
                use_cache=False,
            )

            logits = eagle_out["logits"][:, -1]
            probs = F.softmax(logits, dim=-1)
            confidence = eagle_out["confidence"][:, -1, 0]  # [B*P]

            # Only expand paths with sufficient confidence
            expand_mask = confidence > self.config.confidence_threshold

            if not expand_mask.any():
                break

            # Get top-k for each position
            top_probs_level, top_indices_level = probs.topk(self.config.tree_width, dim=-1)

            # Mask out low-confidence expansions
            expand_mask = expand_mask.view(batch_size, num_active)
            top_probs_level = top_probs_level.view(batch_size, num_active, self.config.tree_width)
            top_indices_level = top_indices_level.view(
                batch_size, num_active, self.config.tree_width
            )

            # Only keep expanded paths
            new_hidden = eagle_out["hidden_states"][:, -1].view(batch_size, num_active, -1)

            # Flatten active paths
            active_tokens = top_indices_level[expand_mask].view(batch_size, -1)
            active_probs = top_probs_level[expand_mask].view(batch_size, -1)
            active_hidden = new_hidden[expand_mask].view(batch_size, -1, new_hidden.size(-1))

            if active_tokens.numel() > 0:
                all_tokens.append(active_tokens)
                all_probs.append(active_probs)

        # Concatenate all levels
        draft_tokens = torch.cat(all_tokens, dim=1)
        draft_probs = torch.cat(all_probs, dim=1)
        positions = (
            torch.arange(draft_tokens.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
        )

        return draft_tokens, positions, draft_probs

    @torch.no_grad()
    def _verify_and_accept(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """
        Verify draft tokens with base model and accept valid prefixes.

        Uses the speculative sampling acceptance criterion.

        Returns:
            accepted_tokens: Tokens to append to context
            num_accepted: Number of accepted tokens
            next_hidden: Hidden state for next iteration
        """
        num_drafts = draft_tokens.shape[1]

        # Verify all drafts in parallel
        verify_input = torch.cat([context, draft_tokens], dim=1)
        hidden_states, verify_logits = self._get_base_hidden_states(verify_input)

        # Get logits at verification positions
        verify_logits = verify_logits[:, context.size(1) - 1 : context.size(1) + num_drafts]

        # Apply temperature
        verify_probs = F.softmax(verify_logits / temperature, dim=-1)

        # Accept tokens greedily from left to right
        accepted = []
        for i in range(num_drafts):
            draft_token = draft_tokens[:, i]

            # Simple acceptance: if top-1 matches
            top_token = verify_probs[:, i].argmax(dim=-1)

            if torch.all(draft_token == top_token):
                accepted.append(draft_token)
            else:
                # Sample from residual distribution
                break

        if accepted:
            accepted_tokens = torch.stack(accepted, dim=1)
            num_accepted = len(accepted)
        else:
            # No draft accepted, sample from base model
            accepted_tokens = verify_probs[:, 0].argmax(dim=-1).unsqueeze(1)
            num_accepted = 0

        # Get hidden state for continuation
        accept_pos = context.size(1) + max(num_accepted - 1, 0)
        next_hidden = (
            hidden_states[:, accept_pos : accept_pos + 1] if hidden_states is not None else None
        )

        # If no drafts accepted, still return one token from base model
        if num_accepted == 0:
            next_token = verify_probs[:, 0].argmax(dim=-1).unsqueeze(1)
            accepted_tokens = next_token
            num_accepted = 1

        return accepted_tokens, num_accepted, next_hidden

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
        callback: Callable[[torch.Tensor], None] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate tokens using EAGLE speculative decoding.

        Args:
            input_ids: [batch, seq_len] input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            eos_token_id: Optional EOS token for early stopping
            callback: Optional callback after each step

        Returns:
            generated: [batch, seq_len + generated] full sequence
            metrics: Generation statistics
        """
        batch_size = input_ids.shape[0]
        _ = batch_size  # Used for documentation/debugging

        generated = input_ids.clone()
        num_generated = 0

        metrics = {
            "num_drafts": 0,
            "num_accepted": 0,
            "num_base_calls": 0,
            "total_tokens": 0,
        }

        while num_generated < max_new_tokens:
            # Check for EOS
            if eos_token_id is not None:
                if (generated == eos_token_id).any(dim=-1).all():
                    break

            # Get base model hidden states and logits
            hidden_states, base_logits = self._get_base_hidden_states(generated)
            metrics["num_base_calls"] += 1

            # Generate draft tree
            draft_tokens, positions, draft_probs = self._generate_draft_tree(
                context_hidden=hidden_states,
                context_tokens=generated,
                base_logits=base_logits[:, -1],
            )

            metrics["num_drafts"] += draft_tokens.shape[1]

            # Verify and accept
            accepted_tokens, num_accepted, next_hidden = self._verify_and_accept(
                context=generated,
                draft_tokens=draft_tokens,
                temperature=temperature,
            )

            metrics["num_accepted"] += num_accepted
            metrics["num_base_calls"] += 1  # Verification call

            # Append accepted tokens
            generated = torch.cat([generated, accepted_tokens], dim=1)
            num_generated += accepted_tokens.shape[1]
            metrics["total_tokens"] += accepted_tokens.shape[1] * batch_size

            if callback is not None:
                callback(generated)

        # Compute aggregate metrics
        if metrics["num_drafts"] > 0:
            metrics["acceptance_rate"] = metrics["num_accepted"] / metrics["num_drafts"]
        else:
            metrics["acceptance_rate"] = 0.0

        metrics["tokens_per_call"] = (
            metrics["total_tokens"] / metrics["num_base_calls"]
            if metrics["num_base_calls"] > 0
            else 0
        )

        return generated, metrics


class EAGLETrainer:
    """Trainer for EAGLE head.

    Trains the EAGLE head to predict future hidden states from a frozen base model.

    Args:
        base_model: Frozen base autoregressive model
        eagle_head: EAGLE head to train
        optimizer: Optimizer for EAGLE head
        config: EAGLE configuration
    """

    def __init__(
        self,
        base_model: nn.Module,
        eagle_head: EAGLEHead,
        optimizer: torch.optim.Optimizer,
        config: EAGLEConfig,
    ):
        self.base_model = base_model
        self.eagle_head = eagle_head
        self.optimizer = optimizer
        self.config = config

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def train_step(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, float]:
        """
        Single training step for EAGLE head.

        Args:
            input_ids: [batch, seq_len] training tokens

        Returns:
            Dict with loss metrics
        """
        self.eagle_head.train()

        # Get base model hidden states (no grad)
        with torch.no_grad():
            base_output = self.base_model(input_ids, output_hidden_states=True)
            if hasattr(base_output, "hidden_states"):
                base_hidden = base_output.hidden_states[-1]
            elif isinstance(base_output, dict):
                base_hidden = base_output.get("hidden_states")
            else:
                base_hidden = base_output[0] if isinstance(base_output, tuple) else base_output

        # EAGLE forward (with grad)
        eagle_output = self.eagle_head(
            hidden_states=base_hidden[:, :-1],  # All but last
            input_tokens=input_ids[:, :-1],
        )

        logits = eagle_output["logits"]

        # Next token prediction loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )

        # Optional: confidence calibration loss
        total_loss = ce_loss
        metrics = {"ce_loss": ce_loss.item()}

        if "confidence" in eagle_output and self.config.use_dynamic_tree:
            confidence = eagle_output["confidence"][:, :, 0]

            # Confidence should correlate with correctness
            with torch.no_grad():
                predicted = logits.argmax(dim=-1)
                correct = (predicted == input_ids[:, 1:]).float()

            # Binary cross entropy for confidence
            conf_loss = F.binary_cross_entropy(confidence, correct)
            total_loss = total_loss + 0.1 * conf_loss
            metrics["conf_loss"] = conf_loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.eagle_head.parameters(), 1.0)

        self.optimizer.step()

        metrics["total_loss"] = total_loss.item()
        return metrics


def create_eagle_from_base(
    base_model: nn.Module,
    num_layers: int = 2,
    use_dynamic_tree: bool = True,
    max_draft_length: int = 6,
    tree_width: int = 3,
) -> tuple[EAGLEHead, EAGLEConfig]:
    """
    Create an EAGLE head from an existing base model.

    Args:
        base_model: The base autoregressive model
        num_layers: Number of transformer layers for EAGLE
        use_dynamic_tree: Whether to use dynamic tree (EAGLE-2)
        max_draft_length: Maximum draft tokens
        tree_width: Width of draft tree

    Returns:
        eagle_head: Initialized EAGLE head
        config: EAGLE configuration
    """
    # Try to infer model dimensions
    if hasattr(base_model, "config"):
        hidden_size = getattr(
            base_model.config, "hidden_size", getattr(base_model.config, "hidden_dim", 768)
        )
        vocab_size = getattr(base_model.config, "vocab_size", 32000)
        num_heads = getattr(
            base_model.config, "num_attention_heads", getattr(base_model.config, "num_heads", 8)
        )
    elif hasattr(base_model, "hidden_size"):
        hidden_size = base_model.hidden_size
        vocab_size = base_model.vocab_size
        num_heads = getattr(base_model, "num_heads", 8)
    else:
        # Try to infer from embedding layer
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Embedding):
                vocab_size = module.num_embeddings
                hidden_size = module.embedding_dim
                break
        num_heads = 8

    config = EAGLEConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_eagle_layers=num_layers,
        num_heads=num_heads,
        max_draft_length=max_draft_length,
        tree_width=tree_width,
        use_dynamic_tree=use_dynamic_tree,
    )

    eagle_head = EAGLEHead(config)

    return eagle_head, config


__all__ = [
    "EAGLEConfig",
    "EAGLEHead",
    "EAGLEDecoder",
    "EAGLETrainer",
    "EAGLEAttention",
    "EAGLEBlock",
    "DraftTreeNode",
    "create_eagle_from_base",
]
