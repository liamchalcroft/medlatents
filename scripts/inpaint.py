"""Unified inpainting script for all discrete latent generative models."""

import argparse

import torch

from medlatents.generation import DiscreteLatentGenerator


def main():
    parser = argparse.ArgumentParser(description="Inpaint anomalies using discrete latent models")

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

    # Input/Output parameters
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help="Glob pattern for input volumes",
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # Inpainting parameters
    parser.add_argument(
        "--likelihood_threshold",
        type=float,
        default=0.005,
        help="Threshold for anomaly detection",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load generator
    print(f"Loading {args.model_type} model...")
    generator = DiscreteLatentGenerator.from_checkpoints(
        model_type=args.model_type,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )

    # Inpaint volumes
    generator.inpaint_volumes(
        input_pattern=args.input_pattern,
        output_dir=args.output_dir,
        likelihood_threshold=args.likelihood_threshold,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(f"Inpainting complete! Volumes saved to {args.output_dir}/inpainted/")


if __name__ == "__main__":
    main()
