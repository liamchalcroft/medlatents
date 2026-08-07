"""Generation utilities for discrete flow matching models.

Includes sample generation with the Euler solver and counterfactual
generation for anomaly detection and inpainting.
"""

from pathlib import Path

import torch
from medtokenizers.networks.discrete import DiscreteTokenizer
from torch import Tensor, nn

from .core import MixtureDiscreteEulerSolver, MixtureDiscreteProbPath, ModelWrapper
from .discrete import SourceDistribution


class WrappedModel(ModelWrapper):
    @torch.no_grad()
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return torch.softmax(self.model(x_t=x, time=t, **extras).float(), dim=-1)


@torch.no_grad()
def generate_counterfactual(
    model: nn.Module,
    x_input: Tensor,
    vocab_size: int,
    device: torch.device,
    path: MixtureDiscreteProbPath,
    sampling_steps: int,
    likelihood_threshold: float = 0.1,  # Threshold for identifying anomalies
    time_epsilon: float = 0.0,
    dtype_categorical: torch.dtype = torch.float64,
) -> Tensor:
    """Generate healthy counterfactuals by masking and regenerating anomalous tokens."""
    wrapped_probability_denoiser = WrappedModel(model=model)

    # Setup solver with masking enabled
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + 1,  # +1 for mask token
    )

    # First pass: Identify anomalies
    # Get token probabilities at t=1.0 (final distribution)
    probs = wrapped_probability_denoiser(
        x_input, torch.ones(x_input.size(0), device=device, dtype=torch.float32)
    )  # [B, L, V]

    # Get probability of each actual token
    token_probs = torch.gather(probs, dim=-1, index=x_input.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # Identify low probability tokens as anomalies
    anomaly_mask = token_probs < likelihood_threshold

    # Create masked version of input
    x_masked = x_input.clone()
    x_masked[anomaly_mask] = vocab_size  # Use mask token

    # Second pass: Generate with masked anomalies
    sample = solver.sample(
        x_init=x_masked,
        step_size=1 / sampling_steps,
        verbose=True,
        dtype_categorical=dtype_categorical,
        time_grid=torch.tensor([0.0, 1.0 - time_epsilon], device=device),
    )

    return sample, anomaly_mask


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: DiscreteTokenizer,
    rank: int,
    device: torch.device,
    path: MixtureDiscreteProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    generate_counterfactuals: bool = False,
    likelihood_threshold: float = 0.1,
    time_epsilon: float = 0.0,
    sample_dir: Path | None = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> Tensor:
    """Extended generate_samples with counterfactual generation option."""
    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    x_init = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    )

    sample = solver.sample(
        x_init=x_init,
        step_size=1 / sampling_steps,
        verbose=True,
        dtype_categorical=dtype_categorical,
        time_grid=torch.tensor([0.0, 1.0 - time_epsilon], device=device),
    )

    if generate_counterfactuals:
        counterfactual_sample, anomaly_mask = generate_counterfactual(
            model=model,
            x_input=sample,
            vocab_size=vocab_size,
            device=device,
            path=path,
            sampling_steps=sampling_steps,
            likelihood_threshold=likelihood_threshold,
            time_epsilon=time_epsilon,
            dtype_categorical=dtype_categorical,
        )

        # Save both original and counterfactual samples
        if sample_dir is not None:
            file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
            file_name.parents[0].mkdir(exist_ok=True, parents=True)

            original_sentences = tokenizer.batch_decode(sample)
            counterfactual_sentences = tokenizer.batch_decode(counterfactual_sample)

            with open(file_name, "w") as file:
                for orig, counter in zip(original_sentences, counterfactual_sentences):
                    file.write(f"Original: {orig}\n")
                    file.write(f"Counterfactual: {counter}\n")
                    file.write(f"{'=' * 20} New sample {'=' * 20}\n")

        return sample, counterfactual_sample, anomaly_mask

    # Original behavior when not generating counterfactuals
    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        sentences = tokenizer.batch_decode(sample)
        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"{sentence}\n{'=' * 20} New sample {'=' * 20}\n")

    return sample
