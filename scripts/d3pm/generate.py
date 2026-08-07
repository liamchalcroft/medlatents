"""Generation script for D3PM (discrete diffusion) models."""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medlatents.diffusion.d3pm import D3PM  # noqa: E402
from medlatents.networks import DiscreteDiT  # noqa: E402
from training_utils import MODEL_CONFIGS  # noqa: E402


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    d3pm: D3PM,
    num_samples: int,
    seq_length: int,
    batch_size: int = 16,
    temperature: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate samples using D3PM reverse diffusion."""
    model.eval()
    all_samples = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating"):
        current_batch = min(batch_size, num_samples - i)
        samples = d3pm.sample(
            model,
            shape=(current_batch, seq_length),
            temperature=temperature,
        )
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from D3PM model")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--output", type=str, default="generated_samples.npy", help="Output file path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # D3PM parameters (should match training)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="cosine")
    parser.add_argument("--transition_type", type=str, default="absorbing")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hparams", {})

    # Get model configuration
    model_size = hparams.get("model_size", "nano")
    config = MODEL_CONFIGS[model_size]

    seq_length = hparams.get("seq_length", 64)
    vocab_size = hparams.get("vocab_size", 512)

    # Create model
    model = DiscreteDiT(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        chunk_size=hparams.get("chunk_size"),
        allow_dynamic_seq_length=True,
    )

    # Load weights
    model.load_state_dict(checkpoint["net"])
    model = model.to(device)
    model.eval()

    # Create D3PM
    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        transition_type=args.transition_type,
        device=device,
    )

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = generate_samples(
        model=model,
        d3pm=d3pm,
        num_samples=args.num_samples,
        seq_length=seq_length,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=device,
    )

    # Save samples
    np.save(args.output, samples.numpy())
    print(f"Saved {len(samples)} samples to {args.output}")

    # Print statistics
    print("\nSample statistics:")
    print(f"  Shape: {samples.shape}")
    print(f"  Unique tokens: {len(torch.unique(samples))}")
    print(f"  Min token: {samples.min().item()}")
    print(f"  Max token: {samples.max().item()}")


if __name__ == "__main__":
    main()
