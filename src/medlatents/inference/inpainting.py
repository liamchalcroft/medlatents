"""Inpainting utilities for all discrete latent model families.

Supports:
- Autoregressive: Iterative inpainting with confidence masking
- MaskGIT: Natural inpainting with bidirectional context
- Flow Matching: Conditional flow from masked state
- D3PM/Diffusion: Repaint algorithm
- Bayesian Flow: Conditional refinement
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Int

from .base import apply_token_mask, confidence_mask


@torch.no_grad()
def inpaint_autoregressive(
    model: nn.Module,
    tokens: Int[torch.Tensor, "batch seq"],
    mask: torch.Tensor,
    mask_token: int,
    num_iterations: int = 5,
    temperature: float = 1.0,
    top_p: float | None = 0.95,
    confidence_threshold: float = 0.1,
) -> Int[torch.Tensor, "batch seq"]:
    """Inpaint using autoregressive model with iterative refinement."""
    from ..sampling import sample_nucleus

    result = tokens.clone()

    # Apply initial mask
    result = apply_token_mask(result, mask, mask_token)

    for _iteration in range(num_iterations):
        # Forward pass
        logits = model(result)

        # Apply temperature and get logits for masked positions
        scaled_logits = logits / temperature

        # Apply nucleus filtering if top_p is set
        if top_p is not None and top_p < 1.0:
            # Reshape for batch filtering: [batch * seq_len, vocab_size]
            batch_size, seq_len, vocab_size = scaled_logits.shape
            flat_logits = scaled_logits.view(-1, vocab_size)
            filtered_flat = sample_nucleus(flat_logits, p=top_p)
            scaled_logits = filtered_flat.view(batch_size, seq_len, vocab_size)

        # Sample all positions at once
        probs = F.softmax(scaled_logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
        sampled = sampled.view(result.shape)

        # Only update masked positions (where mask is False)
        result = torch.where(mask.unsqueeze(0).expand_as(result), result, sampled)

        # Check confidence
        final_logits = model(result)
        conf_mask = confidence_mask(final_logits, confidence_threshold)

        # Update mask to include high-confidence predictions
        # Only re-generate positions that are still low confidence
        mask = mask | conf_mask.all(dim=0)  # Keep if all samples are confident

        # Early stopping if all positions are confident
        if mask.all():
            break

    return result


@torch.no_grad()
def inpaint_maskgit(
    model: nn.Module,
    tokens: Int[torch.Tensor, "batch seq"],
    mask: torch.Tensor,
    mask_token: int,
    num_steps: int = 12,
    temperature: float = 1.0,
    schedule: Literal["cosine", "linear", "sqrt"] = "cosine",
) -> Int[torch.Tensor, "batch seq"]:
    """Inpaint using MaskGIT with confidence-based scheduling.

    ``mask`` is True at positions to keep (visible) and False at positions to
    inpaint. Accepts either a 1-D ``(seq,)`` mask (shared across the batch,
    natural for centre-mask demos) or a 2-D ``(batch, seq)`` mask (per-sample,
    natural for anomaly-detection pipelines where each image's anomaly is at a
    different location).
    """
    from ..sampling import MaskGITScheduler

    if mask.dim() not in (1, 2):
        raise ValueError(f"mask must be 1-D or 2-D; got shape {mask.shape}")
    # Promote a 1-D mask to (batch, seq) for downstream batched ops.
    mask_b = mask if mask.dim() == 2 else mask.unsqueeze(0).expand_as(tokens).contiguous()
    if mask_b.shape != tokens.shape:
        raise ValueError(f"mask shape {mask.shape} incompatible with tokens shape {tokens.shape}")

    result = tokens.clone()

    # Apply initial mask (apply_token_mask handles both 1-D and 2-D masks).
    result = apply_token_mask(result, mask, mask_token)

    # Create scheduler
    scheduler = MaskGITScheduler(num_iterations=num_steps, schedule=schedule)

    # Iterative refinement — scheduler.step takes the (batch, seq) "is-masked"
    # mask, which is the negation of our visible mask.
    for step in range(num_steps):
        logits = model(result)
        result, _current_mask = scheduler.step(
            logits, result, ~mask_b, step, temperature=temperature
        )
        # Preserve known (visible) tokens after each step.
        result = torch.where(mask_b, tokens, result)

    return result


@torch.no_grad()
def inpaint_flow_matching(
    model: nn.Module,
    tokens: Int[torch.Tensor, "batch seq"],
    mask: torch.Tensor,
    mask_token: int,
    num_steps: int = 100,
    temperature: float = 1.0,
    path_type: str = "polynomial",
    source_dist_type: str = "uniform",
    vocab_size: int = 1024,
) -> Int[torch.Tensor, "batch seq"]:
    """Inpaint using flow matching with conditional generation."""
    from ..flow_matching import get_path, get_source_distribution

    device = tokens.device
    batch_size, seq_len = tokens.shape

    # Initialize result with known and unknown regions
    result = tokens.clone()

    # Initialize masked regions from source distribution
    mask_token = getattr(model, "special_tokens", None)
    mask_id = mask_token.mask if mask_token is not None else None
    source_dist = get_source_distribution(source_dist_type, vocab_size, mask_token=mask_id)
    masked_init = source_dist.sample((batch_size, (~mask).sum().item())).to(device)

    # Place initialized masked tokens
    result[:, ~mask] = masked_init

    # Setup path
    get_path(path_type, power=2.0)

    # Flow from t=0 to t=1
    time_steps = torch.linspace(0.0, 1.0, num_steps, device=device)
    dt = time_steps[1] - time_steps[0]

    for t in time_steps[:-1]:
        t_batch = t.expand(batch_size)

        # Get velocity
        v = model(x_t=result, time=t_batch)

        # Apply temperature
        if temperature != 1.0:
            v = v / temperature

        # Update only masked positions
        result_next = result + v * dt

        # Convert to tokens
        if source_dist_type == "mask":
            result_next = torch.argmax(result_next, dim=-1)

        # Preserve known tokens
        result_next[:, mask] = tokens[:, mask]

        result = result_next

    return result


@torch.no_grad()
def inpaint_diffusion_repaint(
    model: nn.Module,
    diffusion,  # D3PM instance
    tokens: Int[torch.Tensor, "batch seq"],
    mask: torch.Tensor,
    num_steps: int = 50,
    jump_length: int = 10,
    jump_n_sample: int = 10,
) -> Int[torch.Tensor, "batch seq"]:
    """Inpaint using RePaint algorithm for discrete diffusion (D3PM)."""
    device = tokens.device
    batch_size, seq_len = tokens.shape

    # Start from noise
    x_t = torch.randint(0, diffusion.num_classes, (batch_size, seq_len), device=device)

    # Timesteps for reverse process
    timesteps = torch.linspace(diffusion.num_timesteps - 1, 0, num_steps, device=device).long()

    for i, t in enumerate(timesteps):
        t_batch = t.expand(batch_size)

        # Reverse diffusion step (denoise)
        logits = model(x=x_t, t=t_batch)
        probs = F.softmax(logits, dim=-1)

        # Sample from predicted distribution
        x_0_pred = torch.multinomial(probs.view(-1, diffusion.num_classes), num_samples=1).view(
            batch_size, seq_len
        )

        # Apply mask: keep known regions
        x_0_pred[:, mask] = tokens[:, mask]

        # Forward diffusion on known regions (if not last step and doing jumps)
        if i < len(timesteps) - 1 and (i + 1) % jump_length == 0 and jump_n_sample > 0:
            # Jump back: add noise to known regions
            jump_t = min(t + jump_length, diffusion.num_timesteps - 1)
            jump_t_batch = torch.full((batch_size,), jump_t, device=device)

            # Forward diffuse known regions
            x_t_known = diffusion.q_sample(tokens[:, mask].unsqueeze(0), jump_t_batch)
            x_0_pred[:, mask] = x_t_known.squeeze(0)

        x_t = x_0_pred

    return x_t


@torch.no_grad()
def inpaint_bayesian_flow(
    model: nn.Module,
    tokens: Int[torch.Tensor, "batch seq"],
    mask: torch.Tensor,
    num_steps: int = 100,
    temperature: float = 1.0,
) -> Int[torch.Tensor, "batch seq"]:
    """Inpaint using Bayesian Flow Network with conditional refinement."""
    device = tokens.device
    batch_size, seq_len = tokens.shape

    # Initialize: uniform prior for masked, one-hot for known
    num_classes = getattr(model, "num_classes", model.vocab_size)
    params = torch.zeros(batch_size, seq_len, num_classes, device=device)

    # Set known tokens to high confidence (one-hot-like logits)
    known_tokens = tokens[:, mask]
    params[:, mask, :] = F.one_hot(known_tokens, num_classes).float() * 10.0

    # Iteratively refine masked regions
    for i in range(num_steps):
        # Get accuracy
        alpha = model.get_accuracy(
            torch.full((batch_size,), float(i), device=device), continuous_time=False
        )

        # Sample from current distribution
        probs = F.softmax(params / temperature, dim=-1)
        x_current = torch.multinomial(probs.view(-1, num_classes), num_samples=1).view(
            batch_size, seq_len
        )

        # Get network prediction
        t = torch.full((batch_size,), i / num_steps, device=device)
        y_receiver = model.sample_receiver_distribution(
            x_current,
            t,
            alpha,
            temperature=temperature,
        )

        # Bayesian update only for masked regions
        params_masked = model.update_params_bayesian(params[:, ~mask, :], y_receiver[:, ~mask, :])
        params[:, ~mask, :] = params_masked

        # Keep known regions fixed
        params[:, mask, :] = F.one_hot(tokens[:, mask], num_classes).float() * 10.0

    # Final sample
    final_probs = F.softmax(params / temperature, dim=-1)
    result = torch.multinomial(final_probs.view(-1, num_classes), num_samples=1).view(
        batch_size, seq_len
    )

    # Ensure known regions are preserved
    result[:, mask] = tokens[:, mask]

    return result


@torch.no_grad()
def inpaint_volume(
    model: nn.Module,
    tokenizer,  # DiscreteTokenizer
    volume: torch.Tensor,
    mask: torch.Tensor,
    model_type: Literal["autoreg", "maskgit", "flow", "diffusion", "bayesian_flow"],
    rasterization_method: str = "hilbert",
    **inpaint_kwargs,
) -> torch.Tensor:
    """End-to-end volume inpainting: volume → tokens → inpaint → volume."""
    from .base import spatial_to_sequence_mask

    device = volume.device

    # Tokenize volume
    tokens = tokenizer.encode(volume)

    # Convert spatial mask to sequence mask
    seq_mask = spatial_to_sequence_mask(mask, rasterization_method)
    seq_mask = seq_mask.to(device)

    # Inpaint in token space
    if model_type == "autoreg":
        inpainted_tokens = inpaint_autoregressive(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "maskgit":
        inpainted_tokens = inpaint_maskgit(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "flow":
        inpainted_tokens = inpaint_flow_matching(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "diffusion":
        inpainted_tokens = inpaint_diffusion_repaint(model, tokens, seq_mask, **inpaint_kwargs)
    elif model_type == "bayesian_flow":
        inpainted_tokens = inpaint_bayesian_flow(model, tokens, seq_mask, **inpaint_kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Decode back to volume
    inpainted_volume = tokenizer.detokenize(inpainted_tokens)

    return inpainted_volume
