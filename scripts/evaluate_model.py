"""Example script for evaluating discrete latent generative models.

Shows how to use reconstruction and generative quality metrics.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medlatents.evaluation import (
    FIDCalculator,
    PrecisionRecallCalculator,
    calculate_nll,
    calculate_psnr,
    calculate_ssim,
)


def evaluate_reconstruction(
    model,
    tokenizer,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict:
    """
    Evaluate reconstruction quality of tokenizer.

    Args:
        model: Generative model (not used for reconstruction)
        tokenizer: Discrete tokenizer with encode/decode
        dataloader: DataLoader yielding images
        device: Device to run on

    Returns:
        Dictionary with reconstruction metrics
    """
    all_metrics = {"psnr": [], "ssim": []}

    model.eval()
    tokenizer.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing reconstruction metrics"):
            images = batch.to(device)

            # Encode and decode
            codes = tokenizer.encode(images)
            reconstructed = tokenizer.decode(codes)

            # Calculate metrics
            all_metrics["psnr"].append(calculate_psnr(reconstructed, images))
            all_metrics["ssim"].append(calculate_ssim(reconstructed, images))

    # Average across batches
    return {key: sum(values) / len(values) for key, values in all_metrics.items()}


def evaluate_generative_quality(
    model,
    tokenizer,
    real_dataloader: DataLoader,
    num_generated: int = 10000,
    device: str = "cuda",
    extractor_type: str = "radimagenet",
) -> dict:
    """
    Evaluate generative model quality with FID and Precision/Recall.

    Args:
        model: Generative model
        tokenizer: Discrete tokenizer with decode
        real_dataloader: DataLoader yielding real images
        num_generated: Number of samples to generate
        device: Device to run on
        extractor_type: Feature extractor ('radimagenet' or 'inception')

    Returns:
        Dictionary with FID, precision, recall
    """
    model.eval()
    tokenizer.eval()

    # Initialize calculators
    fid_calc = FIDCalculator(extractor_type=extractor_type, device=device)
    pr_calc = PrecisionRecallCalculator(k=5, extractor_type=extractor_type, device=device)

    # Extract features from real images
    print("Extracting features from real images...")
    with torch.no_grad():
        for batch in tqdm(real_dataloader):
            images = batch.to(device)
            # Decode to pixel space if needed
            if images.shape[1] != 3 and images.shape[1] != 1:
                # Assume these are codes
                images = tokenizer.decode(images)

            fid_calc.update_real(images)
            pr_calc.update_real(images)

    # Generate samples and extract features
    print(f"Generating {num_generated} samples...")
    generated_count = 0
    batch_size = real_dataloader.batch_size

    with torch.no_grad():
        while generated_count < num_generated:
            current_batch_size = min(batch_size, num_generated - generated_count)

            # Generate codes
            codes = model.generate(batch_size=current_batch_size)

            # Decode to images
            images = tokenizer.decode(codes)

            fid_calc.update_generated(images)
            pr_calc.update_generated(images)

            generated_count += current_batch_size

    # Compute metrics
    print("Computing FID...")
    fid = fid_calc.compute()

    print("Computing Precision/Recall...")
    precision, recall = pr_calc.compute()

    return {
        "fid": fid,
        "precision": precision,
        "recall": recall,
    }


def evaluate_nll(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Evaluate Negative Log-Likelihood on held-out data.

    Args:
        model: Generative model
        dataloader: DataLoader yielding token sequences
        device: Device to run on

    Returns:
        Average NLL
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing NLL"):
            tokens = batch.to(device)

            nll = calculate_nll(model, tokens, reduction="sum")
            total_nll += nll
            total_tokens += tokens.numel()

    return total_nll / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-generated", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--extractor",
        type=str,
        default="radimagenet",
        choices=["radimagenet", "inception"],
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["reconstruction", "fid", "nll"],
        choices=["reconstruction", "fid", "precision_recall", "nll"],
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    # Provide a project-specific model loader here.
    # model = load_model(args.model_path, device=args.device)

    print(f"Loading tokenizer from {args.tokenizer_path}...")
    # Provide a project-specific tokenizer loader here.
    # tokenizer = load_tokenizer(args.tokenizer_path, device=args.device)

    print(f"Loading data from {args.data_path}...")
    # Provide a project-specific dataloader here.
    # dataloader = create_dataloader(args.data_path, batch_size=args.batch_size)

    results = {}

    # Reconstruction metrics
    if "reconstruction" in args.metrics:
        print("\n=== Reconstruction Metrics ===")
        recon_metrics = evaluate_reconstruction(model, tokenizer, dataloader, device=args.device)
        results.update(recon_metrics)
        for key, value in recon_metrics.items():
            print(f"{key.upper()}: {value:.4f}")

    # FID and Precision/Recall
    if "fid" in args.metrics or "precision_recall" in args.metrics:
        print("\n=== Generative Quality Metrics ===")
        gen_metrics = evaluate_generative_quality(
            model,
            tokenizer,
            dataloader,
            num_generated=args.num_generated,
            device=args.device,
            extractor_type=args.extractor,
        )
        results.update(gen_metrics)
        print(f"FID: {gen_metrics['fid']:.2f}")
        print(f"Precision: {gen_metrics['precision']:.4f}")
        print(f"Recall: {gen_metrics['recall']:.4f}")

    # NLL
    if "nll" in args.metrics:
        print("\n=== Negative Log-Likelihood ===")
        # Provide a dataloader that yields token sequences.
        # nll = evaluate_nll(model, token_dataloader, device=args.device)
        # results['nll'] = nll
        # print(f"NLL: {nll:.4f}")
        print("NLL evaluation requires token dataloader (not implemented in example)")

    print("\n=== Summary ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
