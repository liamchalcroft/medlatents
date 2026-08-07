#!/usr/bin/env python3
"""Example: Using Token Critic with MaskGIT

This script demonstrates how to use the existing CriticGuidedGenerator
to improve MaskGIT generation quality through rejection sampling.
"""

import torch

from medlatents.maskgit import MaskGIT
from medlatents.sampling.maskgit import CriticGuidedGenerator, TokenCritic


def example_standard_generation():
    """Standard MaskGIT generation without critic."""
    model = MaskGIT(
        seq_length=256,
        vocab_size=512,
        hidden_size=128,
        depth=4,
        num_heads=4,
    )
    model.eval()

    # Start with all masked tokens
    x_masked = torch.full((1, 256), model.mask_token, dtype=torch.long)

    # Standard generation
    with torch.no_grad():
        generated = model.generate(
            x_masked,
            num_steps=12,
            temperature=1.0,
        )

    print(f"Standard generation: {generated.shape}")
    return generated


def example_critic_guided_generation():
    """Critic-guided MaskGIT generation with rejection sampling.

    Note: This requires a trained TokenCritic model.
    """
    model = MaskGIT(
        seq_length=256,
        vocab_size=512,
        hidden_size=128,
        depth=4,
        num_heads=4,
    )
    model.eval()

    # TokenCritic - must be trained on real vs generated tokens
    critic = TokenCritic(
        hidden_size=128,
        num_heads=4,
        depth=4,
        vocab_size=model.vocab_size,
    )
    critic.eval()

    # Wrap generator with critic for quality-guided sampling
    critic_gen = CriticGuidedGenerator(
        generator=model,
        critic=critic,
        rejection_threshold=0.0,  # Accept all samples
        num_candidates=4,  # Generate 4 candidates, pick best
        refinement_steps=0,  # No iterative refinement
    )

    # Start with all masked tokens
    x_masked = torch.full((1, 256), model.mask_token, dtype=torch.long)

    # Critic-guided generation (rejection sampling)
    with torch.no_grad():
        generated, scores = critic_gen.generate_with_rejection(
            x_masked,
            num_steps=12,
            temperature=1.0,
            eos_token=model.special_tokens.eos if model.special_tokens else None,
        )

    print(f"Critic-guided generation: {generated.shape}")
    print(f"Critic scores (batch average): {scores.mean().item():.3f}")
    return generated, scores


def example_refinement_generation():
    """Critic-guided MaskGIT generation with iterative refinement.

    Note: This requires a trained TokenCritic model.
    """
    model = MaskGIT(
        seq_length=256,
        vocab_size=512,
        hidden_size=128,
        depth=4,
        num_heads=4,
    )
    model.eval()

    # TokenCritic
    critic = TokenCritic(
        hidden_size=128,
        num_heads=4,
        depth=4,
        vocab_size=model.vocab_size,
    )
    critic.eval()

    # Wrap with refinement
    critic_gen = CriticGuidedGenerator(
        generator=model,
        critic=critic,
        rejection_threshold=0.0,
        num_candidates=1,  # Single candidate
        refinement_steps=2,  # 2 refinement iterations
    )

    # Start with all masked tokens
    x_masked = torch.full((1, 256), model.mask_token, dtype=torch.long)

    # Refinement-based generation
    with torch.no_grad():
        generated, scores = critic_gen.generate_with_refinement(
            x_masked,
            num_steps=12,
            temperature=1.0,
            mask_token=model.mask_token,
            eos_token=model.special_tokens.eos if model.special_tokens else None,
        )

    print(f"Refinement generation: {generated.shape}")
    print(f"Final critic scores: {scores.mean().item():.3f}")
    return generated, scores


if __name__ == "__main__":
    print("=== Standard Generation ===")
    example_standard_generation()

    print("\n=== Critic-Guided Generation (requires trained critic) ===")
    print("Note: This will use an untrained critic for demonstration.")
    print("For real use, load a trained TokenCritic model.")
    example_critic_guided_generation()

    print("\n=== Refinement Generation (requires trained critic) ===")
    example_refinement_generation()

    print("\n=== Summary ===")
    print("To use critic-guided generation:")
    print("1. Train a TokenCritic model on real vs generated tokens")
    print("2. Load trained critic model")
    print("3. Use CriticGuidedGenerator wrapper with MaskGIT")
    print("4. Choose strategy:")
    print("   - Rejection sampling (generate_with_rejection)")
    print("   - Iterative refinement (generate_with_refinement)")
