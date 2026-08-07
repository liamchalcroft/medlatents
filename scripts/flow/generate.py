import argparse
from pathlib import Path

import medrs
import numpy as np
import torch
from medtokenizers.networks.discrete import DiscreteTokenizer
from tqdm import tqdm

from medlatents.flow_matching.discrete import get_source_distribution
from medlatents.networks import DiscreteDiT_models


def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--dit_path", type=str, required=True, help="Path to DiT model checkpoint")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer weights"
    )
    parser.add_argument("--seq_length", type=int, required=True, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of sampling steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--scheduler_type", type=str, default="polynomial", help="Scheduler type")
    parser.add_argument("--scheduler_power", type=float, default=2.0, help="Scheduler power")
    parser.add_argument(
        "--source_dist",
        type=str,
        default="uniform",
        choices=["uniform", "mask"],
        help="Source distribution",
    )

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Device parameters
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained tokenizer
    tokenizer_weights = torch.load(args.tokenizer_path, map_location=device, weights_only=True)
    tokenizer = DiscreteTokenizer(tokenizer_weights["hparams"]).to(device)
    tokenizer.load_state_dict(tokenizer_weights["net"])
    tokenizer.eval()

    # Load pre-trained generative model
    dit_weights = torch.load(args.dit_path, map_location=device, weights_only=True)
    dit_hparams = dit_weights["hparams"]
    dit_size = dit_hparams.pop("model_size")
    dit = DiscreteDiT_models[f"DiscreteDiT-{dit_size.upper()}"](**dit_hparams).to(device)
    dit.load_state_dict(dit_weights["net"])
    dit.eval()

    # Initialize source distribution
    source_dist = get_source_distribution(args.source_dist, args.vocab_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    volumes_dir = output_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        for i in tqdm(range(args.num_samples)):
            # Sample from source distribution
            x = source_dist.sample((1, args.seq_length)).to(device)

            # Generate sequence using Euler solver
            time_steps = torch.linspace(1.0, 0.0, args.num_steps, device=device)
            dt = time_steps[0] - time_steps[1]

            for t in time_steps:
                # Get velocity field from model
                t_batch = t.expand(1)
                v = dit(x, t_batch)

                # Apply temperature
                if args.temperature != 1.0:
                    v = v / args.temperature

                # Euler step
                if t > time_steps[-1]:  # Don't update on last step
                    x = x + v * dt
                    # Optionally project back to valid discrete tokens
                    if args.source_dist == "mask":
                        x = torch.argmax(x, dim=-1)

            # Decode to volume
            volume = tokenizer.detokenize(x)

            # Save volume as NIfTI using medrs
            volume_nii = medrs.NiftiImage(volume[0].cpu().numpy(), np.eye(4))
            volume_path = volumes_dir / f"volume_{i:04d}.nii.gz"
            volume_nii.save(str(volume_path))


if __name__ == "__main__":
    main()
