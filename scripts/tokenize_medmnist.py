"""Tokenize MedMNIST datasets using trained medtokenizer checkpoints.

Loads a trained tokenizer from an Accelerate-format checkpoint (model.safetensors),
downloads MedMNIST data via the ``medmnist`` package, tokenizes all splits
(train/val/test), and saves the result as a single NPZ file consumable by
``examples/train_medmnist2d_tokens.py`` or ``examples/train_medmnist3d_tokens.py``.

Usage:
    # Discrete tokenizer (VQ, FSQ, RESFSQ)
    python scripts/tokenize_medmnist.py \
        --dataset chestmnist \
        --quantizer FSQ \
        --checkpoint_dir /path/to/medtokenizers/checkpoints/medmnist/chestmnist_FSQ_final \
        --output_dir ./data/tokenized \
        --device cuda

    # Continuous tokenizer (VAE, AE)
    python scripts/tokenize_medmnist.py \
        --dataset chestmnist \
        --quantizer VAE \
        --checkpoint_dir /path/to/medtokenizers/checkpoints/medmnist/chestmnist_VAE_final \
        --output_dir ./data/tokenized \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import medmnist
import numpy as np
import torch
from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer
from safetensors.torch import load_file
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registries (mirrored from medtokenizers training scripts)
# ---------------------------------------------------------------------------

DATASETS_2D = {
    "pathmnist": {"class": medmnist.PathMNIST, "channels": 3},
    "chestmnist": {"class": medmnist.ChestMNIST, "channels": 1},
    "dermamnist": {"class": medmnist.DermaMNIST, "channels": 3},
    "octmnist": {"class": medmnist.OCTMNIST, "channels": 1},
    "pneumoniamnist": {"class": medmnist.PneumoniaMNIST, "channels": 1},
    "retinamnist": {"class": medmnist.RetinaMNIST, "channels": 3},
    "breastmnist": {"class": medmnist.BreastMNIST, "channels": 1},
    "bloodmnist": {"class": medmnist.BloodMNIST, "channels": 3},
    "tissuemnist": {"class": medmnist.TissueMNIST, "channels": 1},
    "organamnist": {"class": medmnist.OrganAMNIST, "channels": 1},
    "organcmnist": {"class": medmnist.OrganCMNIST, "channels": 1},
    "organsmnist": {"class": medmnist.OrganSMNIST, "channels": 1},
}

DATASETS_3D = {
    "organmnist3d": {"class": medmnist.OrganMNIST3D, "channels": 1},
    "nodulemnist3d": {"class": medmnist.NoduleMNIST3D, "channels": 1},
    "adrenalmnist3d": {"class": medmnist.AdrenalMNIST3D, "channels": 1},
    "fracturemnist3d": {"class": medmnist.FractureMNIST3D, "channels": 1},
    "vesselmnist3d": {"class": medmnist.VesselMNIST3D, "channels": 1},
    "synapsemnist3d": {"class": medmnist.SynapseMNIST3D, "channels": 1},
}

DISCRETE_QUANTIZERS = {"VQ", "FSQ", "RESFSQ", "LFQ"}
CONTINUOUS_QUANTIZERS = {"VAE", "AE"}

# ---------------------------------------------------------------------------
# Dataset loading (reused patterns from medtokenizers/examples/train_medmnist*.py)
# ---------------------------------------------------------------------------


def normalize_to_01(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    mn, mx = data.min(), data.max()
    if mx - mn > 1e-8:
        data = (data - mn) / (mx - mn)
    return data


def load_split(
    dataset_name: str,
    split: str,
    data_dir: str,
    size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a single split of a MedMNIST dataset, returning (images, labels)."""
    is_3d = dataset_name in DATASETS_3D
    registry = DATASETS_3D if is_3d else DATASETS_2D
    info = registry[dataset_name]
    dataset_class = info["class"]

    ds_data_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(ds_data_dir, exist_ok=True)

    # Try cached NPZ first
    npz_path = os.path.join(ds_data_dir, f"{dataset_name}_{size}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        images = data[f"{split}_images"]
        labels = data.get(f"{split}_labels")
        # Squeeze trailing channel dim if singleton
        if (
            is_3d
            and images.ndim == 5
            and images.shape[-1] == 1
            or not is_3d
            and images.ndim == 4
            and images.shape[-1] == 1
        ):
            images = images.squeeze(-1)
        return normalize_to_01(images), labels

    dataset = dataset_class(root=ds_data_dir, split=split, download=True, size=size)
    if hasattr(dataset, "imgs"):
        images = dataset.imgs
        labels = dataset.labels.flatten() if hasattr(dataset, "labels") else None
    elif hasattr(dataset, "data"):
        images = dataset.data
        labels = dataset.labels.flatten() if hasattr(dataset, "labels") else None
    else:
        raise ValueError("Cannot access dataset images/labels")

    if (
        is_3d
        and images.ndim == 5
        and images.shape[-1] == 1
        or not is_3d
        and images.ndim == 4
        and images.shape[-1] == 1
    ):
        images = images.squeeze(-1)

    return normalize_to_01(images), labels


def images_to_tensor(images: np.ndarray, is_3d: bool) -> torch.Tensor:
    """Convert numpy images to PyTorch (N, C, ...) format."""
    if is_3d:
        # 3D volumes are always single-channel: (N, D, H, W) -> (N, 1, D, H, W)
        return torch.from_numpy(images).unsqueeze(1)
    else:
        if images.ndim == 4:  # (N, H, W, C) -> (N, C, H, W)
            return torch.from_numpy(images).permute(0, 3, 1, 2)
        else:  # (N, H, W) -> (N, 1, H, W)
            return torch.from_numpy(images).unsqueeze(1)


# ---------------------------------------------------------------------------
# Tokenizer construction & checkpoint loading
# ---------------------------------------------------------------------------


def build_tokenizer(
    quantizer: str,
    in_channels: int,
    dim: int,
    resolution: int,
    spatial_compression: int = 8,
    num_embeddings: int | None = None,
    codebook_size: int | None = None,
    codebook_dim: int | None = None,
    levels: list[int] | None = None,
    num_codebooks: int | None = None,
    embedding_dim: int = 16,
    latent_channels: int = 4,
) -> DiscreteTokenizer | ContinuousTokenizer:
    """Instantiate a tokenizer with the standardised MedMNIST config.

    Vocabulary kwargs (num_embeddings/codebook_size/levels) must match training.
    Defaults below correspond to the ChestMNIST paper's vocab-1024 configs."""
    channels = 48 if dim == 3 else 64

    common = {
        "dim": dim,
        "in_channels": in_channels,
        "out_channels": in_channels,
        "z_channels": 64,
        "channels": channels,
        "channels_mult": (1, 2, 4),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.0,
        "resolution": resolution,
        "spatial_compression": spatial_compression,
    }

    if quantizer in DISCRETE_QUANTIZERS:
        q_kwargs: dict = {}
        if quantizer == "VQ":
            q_kwargs["num_embeddings"] = num_embeddings or 1024
            q_kwargs["use_ema"] = True
            q_kwargs["use_norm"] = True
        elif quantizer == "FSQ":
            q_kwargs["levels"] = levels or [4, 4, 4, 4, 4]
        elif quantizer == "RESFSQ":
            q_kwargs["num_codebooks"] = num_codebooks or 2
            q_kwargs["levels"] = levels or [8, 8, 8]
        elif quantizer == "LFQ":
            q_kwargs["codebook_size"] = codebook_size or 1024
            q_kwargs["codebook_dim"] = codebook_dim or 10
            q_kwargs["num_codebooks"] = num_codebooks or 1

        return DiscreteTokenizer(
            **common,
            embedding_dim=embedding_dim,
            quantizer=quantizer,
            **q_kwargs,
        )
    else:  # VAE / AE
        return ContinuousTokenizer(
            **common,
            latent_channels=latent_channels,
            formulation=quantizer,
        )


def load_checkpoint(
    model: DiscreteTokenizer | ContinuousTokenizer,
    checkpoint_dir: str | Path,
) -> None:
    """Load Accelerate-format checkpoint (model.safetensors) into *model*."""
    ckpt_path = Path(checkpoint_dir) / "model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = load_file(str(ckpt_path))
    model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint from {ckpt_path}")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _get_vocab_size(model: DiscreteTokenizer) -> int:
    """Extract configured vocabulary size from a discrete tokenizer's quantizer."""
    q = model.quantizer
    cls_name = type(q).__name__
    if cls_name == "VectorQuantizer":
        return int(q.n_e)
    elif cls_name == "FSQuantizer":
        return int(q.codebook_size)
    elif cls_name == "ResidualFSQuantizer":
        return int(q.layers[0].codebook_size)
    elif cls_name == "LFQuantizer":
        return int(2**q.codebook_dim)
    elif cls_name == "MultiCodebookQuantizer":
        return int(q.codebook_size)
    else:
        raise ValueError(f"Unknown quantizer type: {cls_name}")


def _get_num_codebooks(model: DiscreteTokenizer) -> int:
    """Extract number of codebooks from a discrete tokenizer's quantizer."""
    q = model.quantizer
    cls_name = type(q).__name__
    if cls_name == "ResidualFSQuantizer":
        return len(q.layers)
    elif cls_name in ("LFQuantizer", "MultiCodebookQuantizer"):
        return int(q.num_codebooks)
    return 1


@torch.no_grad()
def tokenize_discrete(
    model: DiscreteTokenizer,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode images into discrete token indices."""
    all_indices = []
    for start in tqdm(range(0, len(images), batch_size), desc="Tokenizing"):
        batch = images[start : start + batch_size].to(device)
        indices, _codes, _q_loss = model.encode(batch)
        all_indices.append(indices.cpu())
    indices_t = torch.cat(all_indices, dim=0)
    arr = indices_t.numpy()
    # Use int16 if possible (vocab <= 32767), else int32
    if arr.max() <= np.iinfo(np.int16).max:
        return arr.astype(np.int16)
    return arr.astype(np.int32)


@torch.no_grad()
def tokenize_continuous(
    model: ContinuousTokenizer,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode images into continuous latent representations."""
    all_latents = []
    for start in tqdm(range(0, len(images), batch_size), desc="Tokenizing"):
        batch = images[start : start + batch_size].to(device)
        latents, _dist_output = model.encode(batch)
        all_latents.append(latents.cpu())
    latents_t = torch.cat(all_latents, dim=0)
    return latents_t.numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize MedMNIST datasets using trained tokenizer checkpoints"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(list(DATASETS_2D.keys()) + list(DATASETS_3D.keys())),
        help="MedMNIST dataset name",
    )
    parser.add_argument(
        "--quantizer",
        type=str,
        required=True,
        choices=sorted(DISCRETE_QUANTIZERS | CONTINUOUS_QUANTIZERS),
        help="Quantizer / formulation type (VQ, FSQ, RESFSQ, VAE, AE)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to Accelerate checkpoint directory (containing model.safetensors)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tokenized",
        help="Directory to save tokenized NPZ files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for raw MedMNIST downloads",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="MedMNIST image resolution (default: 64)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for tokenization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    # Vocabulary / config knobs (must match training)
    parser.add_argument("--num_embeddings", type=int, default=None, help="VQ codebook size")
    parser.add_argument("--codebook_size", type=int, default=None, help="LFQ codebook size")
    parser.add_argument(
        "--codebook_dim",
        type=int,
        default=None,
        help="LFQ codebook dim (log2 of size for power-of-2)",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="FSQ/RESFSQ levels")
    parser.add_argument("--num_codebooks", type=int, default=None, help="LFQ/RESFSQ num codebooks")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--latent_channels", type=int, default=4, help="VAE/AE latent channels")

    args = parser.parse_args()

    is_3d = args.dataset in DATASETS_3D
    registry = DATASETS_3D if is_3d else DATASETS_2D
    in_channels = registry[args.dataset]["channels"]
    dim = 3 if is_3d else 2
    is_discrete = args.quantizer in DISCRETE_QUANTIZERS

    logger.info("=" * 60)
    logger.info("MedMNIST Tokenization")
    logger.info("=" * 60)
    logger.info(f"Dataset:    {args.dataset} ({'3D' if is_3d else '2D'})")
    logger.info(f"Quantizer:  {args.quantizer} ({'discrete' if is_discrete else 'continuous'})")
    logger.info(f"Channels:   {in_channels}")
    logger.info(f"Resolution: {args.size}")
    logger.info(f"Device:     {args.device}")
    logger.info("=" * 60)

    device = torch.device(args.device)

    # Build tokenizer and load checkpoint
    model = build_tokenizer(
        quantizer=args.quantizer,
        in_channels=in_channels,
        dim=dim,
        resolution=args.size,
        num_embeddings=args.num_embeddings,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        levels=args.levels,
        num_codebooks=args.num_codebooks,
        embedding_dim=args.embedding_dim,
        latent_channels=args.latent_channels,
    )
    load_checkpoint(model, args.checkpoint_dir)
    model.to(device)
    model.eval()

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.quantizer}.npz")

    arrays: dict[str, np.ndarray] = {}

    for split in ("train", "val", "test"):
        logger.info(f"\nProcessing split: {split}")
        images, labels = load_split(args.dataset, split, args.data_dir, args.size)
        logger.info(f"  Raw images shape: {images.shape}")
        tensor = images_to_tensor(images, is_3d)
        logger.info(f"  Tensor shape:     {tensor.shape}")

        if is_discrete:
            tokens = tokenize_discrete(model, tensor, args.batch_size, device)
            logger.info(f"  Tokens shape:     {tokens.shape}  dtype={tokens.dtype}")
            arrays[f"{split}_tokens"] = tokens
        else:
            latents = tokenize_continuous(model, tensor, args.batch_size, device)
            logger.info(f"  Latents shape:    {latents.shape}  dtype={latents.dtype}")
            arrays[f"{split}_latents"] = latents

        if labels is not None:
            arrays[f"{split}_labels"] = labels

    logger.info(f"\nSaving to {out_path}")
    np.savez_compressed(out_path, **arrays)

    # Write metadata.json alongside the NPZ for downstream consumers
    if is_discrete:
        token_arrays = [v for k, v in arrays.items() if k.endswith("_tokens")]
        sample_shape = token_arrays[0].shape[1:]  # per-image shape, e.g. (8, 8) or (K, 8, 8)
        num_codebooks = _get_num_codebooks(model)
        vocab_size = _get_vocab_size(model)
        # Spatial shape is the last 2 (2D) or 3 (3D) dims after codebook axis
        spatial_ndim = 3 if is_3d else 2
        spatial_shape = list(sample_shape[-spatial_ndim:])
        seq_len = num_codebooks * math.prod(spatial_shape)
        observed_max = max(int(arr.max()) for arr in token_arrays)
        if observed_max >= vocab_size:
            logger.warning(
                f"Observed max token {observed_max} >= configured vocab_size {vocab_size}! "
                "This indicates a tokenizer/metadata mismatch."
            )
        metadata = {
            "token_type": "discrete",
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "num_codebooks": num_codebooks,
            "spatial_shape": spatial_shape,
            "quantizer": args.quantizer,
        }
    else:
        latent_arrays = [v for k, v in arrays.items() if k.endswith("_latents")]
        latent_shape = list(latent_arrays[0].shape[1:])  # e.g. (4, 8, 8)
        metadata = {
            "token_type": "continuous",
            "latent_channels": latent_shape[0],
            "latent_shape": latent_shape,
        }

    meta_path = os.path.splitext(out_path)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata written to {meta_path}")
    logger.info(f"  {metadata}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
