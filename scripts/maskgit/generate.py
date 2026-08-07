import argparse
from pathlib import Path

import medrs
import numpy as np
import torch
from medtokenizers.networks.discrete import DiscreteTokenizer
from tqdm import tqdm

from medlatents.configs import MODEL_CONFIGS
from medlatents.maskgit import MaskGIT


def load_model(checkpoint_path, model_size="base", seq_length=1024, vocab_size=16384):
    """Load a trained bidirectional (MaskGIT) model"""
    model = MaskGIT(
        **MODEL_CONFIGS[model_size],
        seq_length=seq_length,
        vocab_size=vocab_size,
        allow_dynamic_seq_length=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["net"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--maskgit_path",
        type=str,
        required=True,
        help="Path to MaskGIT model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer weights"
    )
    parser.add_argument("--seq_length", type=int, required=True, help="Sequence length")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=12, help="Number of generation steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained tokenizer
    tokenizer_weights = torch.load(args.tokenizer_path, map_location=device, weights_only=True)
    tokenizer = DiscreteTokenizer(tokenizer_weights["hparams"]).to(device)
    tokenizer.load_state_dict(tokenizer_weights["net"])
    tokenizer.eval()

    # Load pre-trained generative model
    maskgit_weights = torch.load(args.maskgit_path, map_location=device, weights_only=True)
    maskgit_hparams = maskgit_weights["hparams"]
    maskgit_size = maskgit_hparams.pop("model_size")
    maskgit = MaskGIT(**MODEL_CONFIGS[maskgit_size], **maskgit_hparams).to(device)
    maskgit.load_state_dict(maskgit_weights["net"])
    maskgit.eval()

    # Create output directories
    output_dir = Path(args.output_dir)
    volumes_dir = output_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        for i in tqdm(range(args.num_samples)):
            # Start with all mask tokens
            x = torch.full(
                (1, args.seq_length),
                maskgit.mask_token,
                dtype=torch.long,
                device=device,
            )

            # Generate sequence
            x = maskgit.generate(
                x=x,
                num_steps=args.num_steps,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # Decode to volume
            volume = tokenizer.detokenize(x)

            # Save volume as NIfTI using medrs
            volume_nii = medrs.NiftiImage(volume[0].cpu().numpy(), np.eye(4))
            volume_path = volumes_dir / f"volume_{i:04d}.nii.gz"
            volume_nii.save(str(volume_path))


if __name__ == "__main__":
    main()
