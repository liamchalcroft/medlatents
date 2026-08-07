"""Generation script for continuous diffusion models."""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medlatents.diffusion.continuous import ContinuousGaussianDiffusion  # noqa: E402
from medlatents.networks.diffusion_transformer import ContinuousDiT  # noqa: E402
from training_utils import MODEL_CONFIGS  # noqa: E402


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    diffusion: ContinuousGaussianDiffusion,
    num_samples: int,
    sample_shape: tuple[int, ...],
    batch_size: int = 16,
    num_inference_steps: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate samples using diffusion reverse process."""
    model.eval()
    all_samples = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating"):
        current_batch = min(batch_size, num_samples - i)
        current_shape = (current_batch,) + sample_shape

        samples = diffusion.sample(
            model,
            shape=current_shape,
            num_inference_steps=num_inference_steps,
            temperature=temperature,
        )
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from diffusion model")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument(
        "--num_inference_steps", type=int, default=100, help="Number of inference steps"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--output", type=str, default="generated_latents.npy", help="Output file path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Diffusion parameters (should match training)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="cosine")
    parser.add_argument("--prediction_type", type=str, default="epsilon")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hparams", {})

    # Get model configuration
    model_size = hparams.get("model_size", "nano")
    config = MODEL_CONFIGS[model_size]

    seq_length = hparams.get("seq_length", 64)
    in_channels = hparams.get("in_channels", 4)

    # Create model
    model = ContinuousDiT(
        seq_length=seq_length,
        in_channels=in_channels,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        allow_dynamic_seq_length=True,
        gradient_checkpointing=False,
    )

    # Load weights
    model.load_state_dict(checkpoint["net"])
    model = model.to(device)
    model.eval()

    # Create diffusion process
    diffusion = ContinuousGaussianDiffusion(
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        prediction_type=args.prediction_type,
        device=device,
    )

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    sample_shape = (seq_length, in_channels)
    samples = generate_samples(
        model=model,
        diffusion=diffusion,
        num_samples=args.num_samples,
        sample_shape=sample_shape,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
    )

    # Save samples
    np.save(args.output, samples.numpy())
    print(f"Saved {len(samples)} samples to {args.output}")

    # Print statistics
    print("\nSample statistics:")
    print(f"  Shape: {samples.shape}")
    print(f"  Mean: {samples.mean().item():.4f}")
    print(f"  Std: {samples.std().item():.4f}")
    print(f"  Min: {samples.min().item():.4f}")
    print(f"  Max: {samples.max().item():.4f}")


if __name__ == "__main__":
    main()
