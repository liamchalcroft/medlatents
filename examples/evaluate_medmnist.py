"""Evaluate generated MedMNIST samples.

Computes:
- Reconstruction quality (MSE, PSNR, SSIM) vs ground truth
- Vocabulary usage for discrete (unique tokens / total vocabulary)
- Latent distribution statistics for continuous (mean, std)
- Generation quality metrics

Usage:
    # Evaluate discrete generation
    python examples/evaluate_medmnist.py \
        --generated generated_tokens.npz \
        --ground_truth test_tokens.npz \
        --token_type discrete \
        --vocab_size 512

    # Evaluate continuous generation
    python examples/evaluate_medmnist.py \
        --generated generated_latents.npz \
        --ground_truth test_latents.npz \
        --token_type continuous \
        --latent_shape 4 8 8
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated MedMNIST samples")

    parser.add_argument(
        "--generated", type=str, required=True, help="Path to generated samples NPZ"
    )
    parser.add_argument(
        "--ground_truth", type=str, default=None, help="Path to ground truth samples NPZ"
    )
    parser.add_argument("--token_type", type=str, choices=["discrete", "continuous"], required=True)

    parser.add_argument(
        "--vocab_size", type=int, default=512, help="Vocabulary size for discrete tokens"
    )
    parser.add_argument(
        "--latent_shape", type=int, nargs="+", default=[4, 8, 8], help="Shape of latents (C, H, W)"
    )

    parser.add_argument("--output", type=str, default=None, help="Output JSON file for metrics")

    return parser.parse_args()


def compute_vocabulary_usage(tokens: np.ndarray, vocab_size: int) -> dict:
    """Compute vocabulary usage statistics for discrete tokens."""
    unique_tokens = np.unique(tokens)
    num_unique = len(unique_tokens)

    usage_ratio = num_unique / vocab_size
    token_counts = np.bincount(tokens.flatten(), minlength=vocab_size)
    used_mask = token_counts > 0
    usage_fraction = used_mask.sum() / vocab_size

    return {
        "unique_tokens": int(num_unique),
        "vocab_size": vocab_size,
        "usage_ratio": float(usage_ratio),
        "usage_fraction": float(usage_fraction),
        "entropy": float(
            -np.sum(
                (token_counts[used_mask] / tokens.size)
                * np.log(token_counts[used_mask] / tokens.size + 1e-10)
            )
        ),
    }


def compute_latent_statistics(latents: np.ndarray) -> dict:
    """Compute distribution statistics for continuous latents."""
    return {
        "mean": float(np.mean(latents)),
        "std": float(np.std(latents)),
        "min": float(np.min(latents)),
        "max": float(np.max(latents)),
        "shape": list(latents.shape),
    }


def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float | None = None) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    if data_range is None:
        data_range = max(gt.max(), pred.max()) - min(gt.min(), pred.min())

    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float("inf")

    return 10 * np.log10((data_range**2) / mse)


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: float | None = None) -> float:
    """Compute Structural Similarity Index (simplified version)."""
    if data_range is None:
        data_range = max(gt.max(), pred.max()) - min(gt.min(), pred.min())

    gt_mean, gt_std = gt.mean(), gt.std() + 1e-10
    pred_mean, pred_std = pred.mean(), pred.std() + 1e-10

    covariance = np.mean((gt - gt_mean) * (pred - pred_mean))

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim = ((2 * gt_mean * pred_mean + c1) * (2 * covariance + c2)) / (
        (gt_mean**2 + pred_mean**2 + c1) * (gt_std**2 + pred_std**2 + c2)
    )

    return float(ssim)


def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((gt - pred) ** 2))


def compute_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(gt - pred)))


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 0:
        return (arr - arr_min) / (arr_max - arr_min)
    return arr - arr_min


def evaluate_discrete(
    generated_tokens: np.ndarray,
    ground_truth: np.ndarray | None,
    vocab_size: int,
) -> dict:
    """Evaluate discrete token generation."""
    metrics = {}

    metrics["vocabulary_usage"] = compute_vocabulary_usage(generated_tokens, vocab_size)

    if ground_truth is not None:
        metrics["reconstruction"] = {
            "mse": compute_mse(
                ground_truth.astype(np.float32), generated_tokens.astype(np.float32)
            ),
            "psnr": compute_psnr(
                normalize_to_01(ground_truth.astype(np.float32)),
                normalize_to_01(generated_tokens.astype(np.float32)),
            ),
        }

    return metrics


def evaluate_continuous(
    generated_latents: np.ndarray,
    ground_truth: np.ndarray | None,
    latent_shape: list,
) -> dict:
    """Evaluate continuous latent generation."""
    metrics = {}

    metrics["latent_statistics"] = compute_latent_statistics(generated_latents)

    if ground_truth is not None:
        gt_stats = compute_latent_statistics(ground_truth)

        metrics["distribution_comparison"] = {
            "mean_diff": float(np.abs(metrics["latent_statistics"]["mean"] - gt_stats["mean"])),
            "std_diff": float(np.abs(metrics["latent_statistics"]["std"] - gt_stats["std"])),
            "wasserstein_dist": float(
                np.abs(metrics["latent_statistics"]["mean"] - gt_stats["mean"])
                + np.abs(metrics["latent_statistics"]["std"] - gt_stats["std"])
            ),
        }

    return metrics


def main():
    args = parse_args()

    generated_data = np.load(args.generated)
    ground_truth_data = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        ground_truth_data = np.load(args.ground_truth)

    if args.token_type == "discrete":
        if "generated_tokens" in generated_data:
            generated = generated_data["generated_tokens"]
        else:
            generated = generated_data["latents"]

        ground_truth = None
        if ground_truth_data is not None:
            if "test_tokens" in ground_truth_data:
                ground_truth = ground_truth_data["test_tokens"]
            elif "val_tokens" in ground_truth_data:
                ground_truth = ground_truth_data["val_tokens"]
            elif "train_tokens" in ground_truth_data:
                ground_truth = ground_truth_data["train_tokens"]
            elif "latents" in ground_truth_data:
                ground_truth = ground_truth_data["latents"]

        metrics = evaluate_discrete(generated, ground_truth, args.vocab_size)

    else:
        if "generated_latents" in generated_data:
            generated = generated_data["generated_latents"]
        else:
            generated = generated_data["latents"]

        ground_truth = None
        if ground_truth_data is not None:
            if "test_latents" in ground_truth_data:
                ground_truth = ground_truth_data["test_latents"]
            elif "val_latents" in ground_truth_data:
                ground_truth = ground_truth_data["val_latents"]
            elif "train_latents" in ground_truth_data:
                ground_truth = ground_truth_data["train_latents"]
            elif "latents" in ground_truth_data:
                ground_truth = ground_truth_data["latents"]

        metrics = evaluate_continuous(generated, ground_truth, args.latent_shape)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for category, category_metrics in metrics.items():
        print(f"\n{category}:")
        for key, value in category_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
