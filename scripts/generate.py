"""Unified generation script for all discrete latent generative models."""

import argparse

import torch

from medlatents.generation import DiscreteLatentGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate samples from discrete latent models")

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["autoreg", "maskgit", "flow", "diffusion"],
        help="Type of generative model",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer weights"
    )
    parser.add_argument("--seq_length", type=int, required=True, help="Sequence length")

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of generation steps (maskgit/flow/diffusion)",
    )

    # Flow-specific parameters
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="polynomial",
        help="Scheduler type (flow only)",
    )
    parser.add_argument(
        "--scheduler_power", type=float, default=2.0, help="Scheduler power (flow only)"
    )
    parser.add_argument(
        "--source_dist",
        type=str,
        default="uniform",
        choices=["uniform", "mask"],
        help="Source distribution (flow only)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocabulary size (flow only, inferred from model if not specified)",
    )

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    # Set default num_steps based on model type
    if args.num_steps is None:
        if args.model_type == "maskgit":
            args.num_steps = 12
        elif args.model_type == "flow":
            args.num_steps = 100
        elif args.model_type == "diffusion":
            args.num_steps = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    print(f"Loading {args.model_type} model...")
    generator = DiscreteLatentGenerator.from_checkpoints(
        model_type=args.model_type,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )

    # Generate and save volumes
    flow_kwargs = {}
    if args.model_type == "flow":
        flow_kwargs = {
            "scheduler_type": args.scheduler_type,
            "scheduler_power": args.scheduler_power,
            "source_dist": args.source_dist,
            "vocab_size": args.vocab_size,
        }

    generator.generate_volumes(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_k=args.top_k,
        num_steps=args.num_steps,
        **flow_kwargs,
    )

    print(f"Generation complete! Volumes saved to {args.output_dir}/volumes/")


if __name__ == "__main__":
    main()
