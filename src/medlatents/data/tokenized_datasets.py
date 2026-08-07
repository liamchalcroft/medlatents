"""Datasets with tokenization support for medical imaging."""

from __future__ import annotations

import glob
import hashlib
import logging
import random
from collections.abc import Callable
from pathlib import Path

import medrs
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizers import ContinuousTokenizer, DiscreteTokenizer

logger = logging.getLogger(__name__)

# =============================================================================
# Token Contract Validation
# =============================================================================


def validate_discrete_tokens(
    tokens: torch.Tensor,
    vocab_size: int,
    allow_special: bool = True,
    context: str = "tokens",
) -> None:
    """Validate discrete token tensor against contract.

    Args:
        tokens: Token tensor to validate
        vocab_size: Base vocabulary size (excluding special tokens)
        allow_special: If True, allow special tokens [vocab_size, vocab_size+4)
        context: Context string for error messages

    Raises:
        TypeError: If dtype is not torch.long
        ValueError: If shape is wrong or values are out of range
    """
    if tokens.dtype != torch.long:
        raise TypeError(
            f"{context}: Expected dtype torch.long, got {tokens.dtype}. "
            "Convert with tokens.long() before use."
        )

    if tokens.ndim not in (1, 2):
        raise ValueError(
            f"{context}: Expected 1D [seq_len] or 2D [batch, seq_len], "
            f"got {tokens.ndim}D with shape {tuple(tokens.shape)}"
        )

    max_valid = vocab_size + 4 if allow_special else vocab_size

    if tokens.numel() > 0:
        min_val = tokens.min().item()
        max_val = tokens.max().item()

        if min_val < 0:
            raise ValueError(
                f"{context}: Contains negative values (min={min_val}). "
                "Token indices must be non-negative."
            )

        if max_val >= max_valid:
            raise ValueError(
                f"{context}: Contains out-of-range values (max={max_val}, "
                f"allowed=[0, {max_valid})). Check vocab_size={vocab_size} "
                f"or set allow_special={'True' if not allow_special else 'False'}."
            )


def validate_continuous_latents(
    latents: torch.Tensor,
    expected_channels: int | None = None,
    context: str = "latents",
) -> None:
    """Validate continuous latent tensor against contract.

    Args:
        latents: Latent tensor to validate
        expected_channels: Expected number of channels (optional)
        context: Context string for error messages

    Raises:
        TypeError: If dtype is not floating point
        ValueError: If shape is wrong or contains non-finite values
    """
    if latents.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            f"{context}: Expected floating dtype (float32/float16/bfloat16), "
            f"got {latents.dtype}. Convert with latents.float() before use."
        )

    if latents.ndim not in (4, 5):
        raise ValueError(
            f"{context}: Expected 4D [B,C,H,W] or 5D [B,C,D,H,W], "
            f"got {latents.ndim}D with shape {tuple(latents.shape)}"
        )

    if expected_channels is not None and latents.shape[1] != expected_channels:
        raise ValueError(
            f"{context}: Expected {expected_channels} channels, "
            f"got {latents.shape[1]}. Shape is {tuple(latents.shape)}."
        )

    if not torch.isfinite(latents).all():
        nan_count = torch.isnan(latents).sum().item()
        inf_count = torch.isinf(latents).sum().item()
        raise ValueError(
            f"{context}: Contains non-finite values ({nan_count} NaN, {inf_count} Inf). "
            "Check encoder output or input data for corruption."
        )


def validate_image_for_tokenization(
    data: torch.Tensor,
    context: str = "image",
) -> None:
    """Validate image tensor before tokenization.

    Args:
        data: Image tensor [C, H, W] or [C, D, H, W]
        context: Context string for error messages

    Raises:
        TypeError: If dtype is not floating point
        ValueError: If shape is wrong or contains non-finite values
    """
    if data.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            f"{context}: Expected floating dtype for tokenization, "
            f"got {data.dtype}. Convert with data.float() before tokenizing."
        )

    if data.ndim not in (3, 4):
        raise ValueError(
            f"{context}: Expected 3D [C,H,W] or 4D [C,D,H,W] for tokenization, "
            f"got {data.ndim}D with shape {tuple(data.shape)}. "
            "Add batch dimension with data.unsqueeze(0) if needed."
        )

    if not torch.isfinite(data).all():
        nan_count = torch.isnan(data).sum().item()
        inf_count = torch.isinf(data).sum().item()
        raise ValueError(
            f"{context}: Contains non-finite values ({nan_count} NaN, {inf_count} Inf). "
            "Check input data for corruption or normalize before tokenizing."
        )


def _build_dataloader_seed(seed: int | None) -> tuple[torch.Generator | None, Callable | None]:
    if seed is None:
        return None, None
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}")
    generator = torch.Generator()
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return generator, _seed_worker


def _load_array(file_path: str) -> tuple[torch.Tensor, dict]:
    """Load medical imaging data from disk using medrs for NIfTI files."""

    if file_path.endswith((".nii", ".nii.gz")):
        img = medrs.load(file_path)
        data = medrs.load_to_torch(file_path, dtype=torch.float32)
        # Handle channel dimension - medrs returns (C, H, W, D) or (H, W, D)
        if data.ndim == 3:
            data = data.unsqueeze(0)  # Add channel dim
        metadata = {
            "file_path": file_path,
            "affine": img.affine,
            "spacing": img.spacing,
            "shape": tuple(data.shape),
        }
    elif file_path.endswith(".npy"):
        data = torch.from_numpy(np.load(file_path)).float()
        metadata = {"file_path": file_path, "shape": data.shape}
    elif file_path.endswith(".npz"):
        npz = np.load(file_path)
        data = torch.from_numpy(npz["data"]).float()
        metadata = {
            "file_path": file_path,
            "shape": data.shape,
            "keys": list(npz.keys()),
        }
    else:
        raise ValueError(
            f"Unsupported file format: '{file_path}'. "
            "Supported formats: .nii, .nii.gz (NIfTI), .npy, .npz (NumPy)"
        )

    if data.ndim == 3:
        data = data.unsqueeze(0)
    elif data.ndim == 2:
        data = data.unsqueeze(0).unsqueeze(0)

    return data, metadata


class TokenizedImageDataset(Dataset):
    """On-the-fly discrete tokenization from NIfTI/NumPy with optional caching."""

    def __init__(
        self,
        file_pattern: str,
        tokenizer: DiscreteTokenizer | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_metadata: bool = False,
        normalize: bool = True,
        flatten: bool = True,
        **tokenizer_kwargs,
    ) -> None:
        self.file_pattern = file_pattern
        self.files = sorted(glob.glob(file_pattern))

        if not self.files:
            raise ValueError(
                f"No files found matching pattern: '{file_pattern}'. "
                "Check that the path exists and uses glob syntax (e.g., 'data/*.nii.gz')"
            )

        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and cache_dir is not None
        self.transform = transform
        self.return_metadata = return_metadata
        self.normalize = normalize
        self.flatten = flatten
        self.tokenizer_kwargs = tokenizer_kwargs

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, file_path: str) -> Path:
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.pt"

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        file_path = self.files[idx]

        if self.use_cache:
            cache_path = self._get_cache_path(file_path)
            if cache_path.exists():
                cached = torch.load(cache_path, weights_only=True)
                tokens = cached["tokens"]
                metadata = cached.get("metadata", {"file_path": file_path})
                return (tokens, metadata) if self.return_metadata else tokens

        data, metadata = _load_array(file_path)

        if self.normalize:
            data = (data - data.mean()) / (data.std() + 1e-8)

        if self.transform:
            data = self.transform(data)

        if not self.tokenizer:
            raise ValueError(
                "Tokenizer is required to tokenize data. "
                "Pass a DiscreteTokenizer instance to TokenizedImageDataset constructor."
            )

        # Validate input data before tokenization
        validate_image_for_tokenization(data, context=f"TokenizedImageDataset[{file_path}]")

        with torch.no_grad():
            # Ensure batch dimension for tokenizer
            if data.ndim == 3:
                data = data.unsqueeze(0)
            tokens = self.tokenizer.tokenize(data, **self.tokenizer_kwargs)
            # Remove batch dimension if added
            if tokens.ndim == 2 and tokens.shape[0] == 1:
                tokens = tokens.squeeze(0)
            if self.flatten:
                tokens = tokens.flatten()

        if self.use_cache:
            torch.save(
                {"tokens": tokens, "metadata": metadata},
                self._get_cache_path(file_path),
            )

        return (tokens, metadata) if self.return_metadata else tokens


class ContinuousLatentDataset(Dataset):
    """Dataset that produces continuous latent codes using a tokenizer/encoder."""

    def __init__(
        self,
        file_pattern: str,
        tokenizer: ContinuousTokenizer | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_metadata: bool = False,
        normalize: bool = True,
        flatten: bool = False,
        channel_first: bool = True,
        **tokenizer_kwargs,
    ) -> None:
        self.file_pattern = file_pattern
        self.files = sorted(glob.glob(file_pattern))
        if not self.files:
            raise ValueError(
                f"No files found matching pattern: '{file_pattern}'. "
                "Check that the path exists and uses glob syntax (e.g., 'data/*.nii.gz')"
            )

        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache and cache_dir is not None
        self.transform = transform
        self.return_metadata = return_metadata
        self.normalize = normalize
        self.flatten = flatten
        self.channel_first = channel_first
        self.tokenizer_kwargs = tokenizer_kwargs

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, file_path: str) -> Path:
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.pt"

    def __len__(self) -> int:
        return len(self.files)

    def _encode_latents(self, data: torch.Tensor) -> torch.Tensor:
        if not self.tokenizer:
            raise ValueError(
                "Tokenizer is required to encode latents. "
                "Pass a ContinuousTokenizer instance to ContinuousLatentDataset constructor."
            )

        # Prioritized method resolution for encoding:
        # 1. tokenize: Returns just latents (preferred for ContinuousTokenizer)
        # 2. encode_latents: Custom method name some tokenizers might use
        # 3. encode: Returns (latents, extras) tuple - we extract latents
        # Note: forward/__call__ are intentionally excluded as they typically
        # perform full reconstruction, not encoding.
        with torch.no_grad():
            if hasattr(self.tokenizer, "tokenize"):
                latents = self.tokenizer.tokenize(data, **self.tokenizer_kwargs)
            elif hasattr(self.tokenizer, "encode_latents"):
                latents = self.tokenizer.encode_latents(data, **self.tokenizer_kwargs)
            elif hasattr(self.tokenizer, "encode"):
                result = self.tokenizer.encode(data, **self.tokenizer_kwargs)
                # encode() returns (latents, extras) tuple - extract latents
                if isinstance(result, tuple):
                    latents = result[0]
                else:
                    latents = result
            else:
                raise AttributeError(
                    f"Tokenizer {type(self.tokenizer).__name__} does not expose "
                    "tokenize/encode_latents/encode method. "
                    "Ensure you're using a ContinuousTokenizer from medtokenizers."
                )

        latents = torch.as_tensor(latents)

        if self.flatten:
            latents = latents.flatten(start_dim=1)

        if self.channel_first and latents.ndim > 2 and latents.shape[-1] != latents.shape[1]:
            # Assume shape (batch, H, W, C) -> (batch, C, H, W)
            latents = latents.movedim(-1, 1)

        return latents

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        file_path = self.files[idx]

        if self.use_cache:
            cache_path = self._get_cache_path(file_path)
            if cache_path.exists():
                cached = torch.load(cache_path, weights_only=True)
                latents = cached["latents"]
                metadata = cached.get("metadata", {"file_path": file_path})
                return (latents, metadata) if self.return_metadata else latents

        data, metadata = _load_array(file_path)

        if self.normalize:
            data = (data - data.mean()) / (data.std() + 1e-8)

        if self.transform:
            data = self.transform(data)

        # Validate input data before encoding
        validate_image_for_tokenization(data, context=f"ContinuousLatentDataset[{file_path}]")

        latents = self._encode_latents(data)

        # Validate output latents
        if latents.ndim >= 4:
            validate_continuous_latents(
                latents, context=f"ContinuousLatentDataset output[{file_path}]"
            )

        if latents.shape[0] == 1:
            latents = latents.squeeze(0)

        if self.use_cache:
            torch.save(
                {"latents": latents, "metadata": metadata},
                self._get_cache_path(file_path),
            )

        return (latents, metadata) if self.return_metadata else latents


class PreTokenizedDataset(Dataset):
    """Fast loading of pre-tokenized sequences from .pt files."""

    def __init__(
        self,
        file_pattern: str,
        consolidated: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.file_pattern = file_pattern
        self.consolidated = consolidated
        self.return_metadata = return_metadata

        if consolidated:
            data = torch.load(file_pattern, weights_only=True)
            if isinstance(data, dict):
                self.tokens = data["tokens"]
                self.metadata = data.get("metadata")
            else:
                self.tokens = data
                self.metadata = None
        else:
            self.files = sorted(glob.glob(file_pattern))
            if not self.files:
                raise ValueError(
                    f"No files found matching pattern: '{file_pattern}'. "
                    "Check that the path exists and uses glob syntax (e.g., 'data/*.pt')"
                )
            self.tokens = None
            self.metadata = None

    def __len__(self) -> int:
        return len(self.tokens) if self.consolidated else len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if self.consolidated:
            tokens = self.tokens[idx]
            if self.return_metadata and self.metadata:
                return tokens, self.metadata[idx]
            return tokens

        data = torch.load(self.files[idx], weights_only=True)
        if isinstance(data, dict):
            tokens = data["tokens"]
            metadata = data.get("metadata", {"file_path": self.files[idx]})
        else:
            tokens = data
            metadata = {"file_path": self.files[idx]}

        return (tokens, metadata) if self.return_metadata else tokens


class PairedDataset(Dataset):
    """Paired data for conditional generation (super-resolution, inpainting)."""

    def __init__(
        self,
        input_pattern: str,
        target_pattern: str | None = None,
        tokenizer: DiscreteTokenizer | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        pair_mode: str = "matched",
        mask_ratio: float | None = None,
        mask_token: int | None = None,
        normalize: bool = True,
        **tokenizer_kwargs,
    ) -> None:
        self.input_dataset = TokenizedImageDataset(
            input_pattern,
            tokenizer=tokenizer,
            cache_dir=f"{cache_dir}/inputs" if cache_dir else None,
            use_cache=use_cache,
            return_metadata=True,
            normalize=normalize,
            **tokenizer_kwargs,
        )

        if target_pattern is None:
            self.target_dataset = self.input_dataset
            self.same_data = True
        else:
            self.target_dataset = TokenizedImageDataset(
                target_pattern,
                tokenizer=tokenizer,
                cache_dir=f"{cache_dir}/targets" if cache_dir else None,
                use_cache=use_cache,
                return_metadata=True,
                normalize=normalize,
                **tokenizer_kwargs,
            )
            self.same_data = False

        self.pair_mode = pair_mode
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

        if mask_ratio and not mask_token:
            raise ValueError(
                "mask_token is required when using mask_ratio. "
                "Example: PairedDataset(..., mask_ratio=0.15, mask_token=tokenizer.mask_token)"
            )

        if pair_mode == "matched" and not self.same_data:
            if len(self.input_dataset) != len(self.target_dataset):
                raise ValueError(
                    f"Input/target dataset length mismatch in 'matched' mode: "
                    f"{len(self.input_dataset)} inputs vs {len(self.target_dataset)} targets. "
                    "Use pair_mode='indexed' or 'random' for mismatched lengths."
                )

    def __len__(self) -> int:
        return len(self.input_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        input_tokens, input_meta = self.input_dataset[idx]

        if self.same_data:
            target_tokens = input_tokens.clone()
            target_meta = input_meta.copy()
        elif self.pair_mode == "matched":
            target_tokens, target_meta = self.target_dataset[idx]
        elif self.pair_mode == "indexed":
            target_idx = idx % len(self.target_dataset)
            target_tokens, target_meta = self.target_dataset[target_idx]
        elif self.pair_mode == "random":
            target_idx = torch.randint(0, len(self.target_dataset), (1,)).item()
            target_tokens, target_meta = self.target_dataset[target_idx]
        else:
            raise ValueError(
                f"Unknown pair_mode: '{self.pair_mode}'. "
                "Valid options: 'matched' (1:1 pairing), 'indexed' (wrap-around), 'random'."
            )

        if self.mask_ratio:
            input_tokens = input_tokens.clone()
            num_mask = int(len(input_tokens) * self.mask_ratio)
            mask_indices = torch.randperm(len(input_tokens))[:num_mask]
            input_tokens[mask_indices] = self.mask_token

        metadata = {
            "input": input_meta,
            "target": target_meta,
            "masked": self.mask_ratio is not None,
        }

        return input_tokens, target_tokens, metadata


def pretokenize_dataset(
    input_pattern: str,
    output_dir: str,
    tokenizer: DiscreteTokenizer,
    consolidate: bool = False,
    num_workers: int = 4,
    batch_size: int = 8,
    **dataset_kwargs,
) -> None:
    """Pre-tokenize dataset and save to disk."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = TokenizedImageDataset(
        input_pattern,
        tokenizer=tokenizer,
        cache_dir=None,
        use_cache=False,
        return_metadata=True,
        **dataset_kwargs,
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if consolidate:
        all_tokens = []
        all_metadata = []

        logger.info(f"Tokenizing {len(dataset)} files...")
        for tokens, metadata in loader:
            all_tokens.append(tokens)
            all_metadata.extend(metadata)

        all_tokens = torch.cat(all_tokens, dim=0)
        output_file = output_path / "tokens.pt"
        torch.save({"tokens": all_tokens, "metadata": all_metadata}, output_file)
        logger.info(f"Saved to {output_file} ({all_tokens.shape})")
    else:
        logger.info(f"Tokenizing {len(dataset)} files...")
        for i, (tokens, metadata) in enumerate(loader):
            for j in range(len(tokens)):
                output_file = output_path / f"tokens_{i * batch_size + j:06d}.pt"
                torch.save({"tokens": tokens[j], "metadata": metadata[j]}, output_file)
        logger.info(f"Saved {len(dataset)} files to {output_dir}")


def get_tokenized_dataloaders(
    train_pattern: str,
    val_pattern: str,
    tokenizer: DiscreteTokenizer | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: str | None = None,
    use_cache: bool = True,
    pin_memory: bool = True,
    consolidated: bool = False,
    seed: int | None = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with discrete tokenization support.

    Args:
        seed: Optional base seed for deterministic shuffling and worker seeding.
    """

    if tokenizer is None and not consolidated:
        raise ValueError(
            "tokenizer is required unless using pre-tokenized data. "
            "Pass a DiscreteTokenizer or set consolidated=True with .pt tokens."
        )

    if consolidated:
        train_dataset = PreTokenizedDataset(train_pattern, consolidated=True)
        val_dataset = PreTokenizedDataset(val_pattern, consolidated=True)
    else:
        train_dataset = TokenizedImageDataset(
            train_pattern,
            tokenizer=tokenizer,
            cache_dir=f"{cache_dir}/train" if cache_dir else None,
            use_cache=use_cache,
            **dataset_kwargs,
        )
        val_dataset = TokenizedImageDataset(
            val_pattern,
            tokenizer=tokenizer,
            cache_dir=f"{cache_dir}/val" if cache_dir else None,
            use_cache=use_cache,
            **dataset_kwargs,
        )

    generator, worker_init_fn = _build_dataloader_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_latent_dataloaders(
    train_pattern: str,
    val_pattern: str,
    tokenizer: ContinuousTokenizer | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: str | None = None,
    use_cache: bool = True,
    pin_memory: bool = True,
    seed: int | None = None,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for continuous latent codes.

    Args:
        seed: Optional base seed for deterministic shuffling and worker seeding.
    """

    if tokenizer is None:
        raise ValueError(
            "tokenizer is required for continuous latents. Pass a ContinuousTokenizer instance."
        )

    train_dataset = ContinuousLatentDataset(
        train_pattern,
        tokenizer=tokenizer,
        cache_dir=f"{cache_dir}/train" if cache_dir else None,
        use_cache=use_cache,
        **dataset_kwargs,
    )
    val_dataset = ContinuousLatentDataset(
        val_pattern,
        tokenizer=tokenizer,
        cache_dir=f"{cache_dir}/val" if cache_dir else None,
        use_cache=use_cache,
        **dataset_kwargs,
    )

    generator, worker_init_fn = _build_dataloader_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


__all__ = [
    "TokenizedImageDataset",
    "ContinuousLatentDataset",
    "PreTokenizedDataset",
    "PairedDataset",
    "pretokenize_dataset",
    "get_tokenized_dataloaders",
    "get_latent_dataloaders",
    # Validation functions
    "validate_discrete_tokens",
    "validate_continuous_latents",
    "validate_image_for_tokenization",
]
