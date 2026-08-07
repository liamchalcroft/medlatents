"""High-performance data loading utilities for training.

Provides optimized data loading with:
- Async prefetching to GPU
- Multi-worker data loading
- Memory-mapped dataset support
- Automatic batching optimizations
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class DataLoaderConfig:
    """Configuration for high-performance data loading.

    Attributes:
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        drop_last: Drop incomplete last batch (important for DDP)
        shuffle: Shuffle data
    """

    batch_size: int = 32
    num_workers: int | None = None  # Auto-detect
    pin_memory: bool | None = None  # Auto-detect
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = True
    shuffle: bool = True

    def __post_init__(self) -> None:
        # Auto-detect optimal settings
        if self.num_workers is None:
            cpu_count = os.cpu_count() or 4
            self.num_workers = min(cpu_count // 2, 8)

        if self.pin_memory is None:
            self.pin_memory = torch.cuda.is_available()


class AsyncPrefetcher:
    """Async prefetcher that moves batches to GPU in background thread.

    Overlaps data transfer with computation for better GPU utilization.

    Usage:
        loader = DataLoader(dataset, ...)
        prefetcher = AsyncPrefetcher(loader, device='cuda')

        for batch in prefetcher:
            # batch is already on GPU
            loss = model(batch)
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device | str = "cuda",
        queue_size: int = 2,
    ):
        """Initialize async prefetcher.

        Args:
            loader: Base DataLoader
            device: Target device for batches
            queue_size: Number of batches to prefetch
        """
        self.loader = loader
        self.device = torch.device(device)
        self.queue_size = queue_size

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _prefetch_worker(self) -> None:
        """Background thread that prefetches batches to GPU."""
        stream = torch.cuda.Stream() if self.device.type == "cuda" else None

        for batch in self.loader:
            if self._stop_event.is_set():
                break

            # Move to device in background stream
            if stream is not None:
                with torch.cuda.stream(stream):
                    batch = self._to_device(batch)
                stream.synchronize()
            else:
                batch = self._to_device(batch)

            # Put in queue (blocks if full)
            try:
                self._queue.put(batch, timeout=60)
            except queue.Full:
                if not self._stop_event.is_set():
                    raise
                break

        # Signal end
        self._queue.put(None)

    def _to_device(self, batch: Any) -> Any:
        """Recursively move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            moved = [self._to_device(x) for x in batch]
            return type(batch)(moved)
        else:
            return batch

    def __iter__(self) -> Iterator[Any]:
        """Iterate over prefetched batches."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()

        try:
            while True:
                batch = self._queue.get(timeout=120)
                if batch is None:
                    break
                yield batch
        finally:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=5)

    def __len__(self) -> int:
        return len(self.loader)


class CUDAPrefetcher:
    """Simple CUDA prefetcher using CUDA streams.

    Lighter weight than AsyncPrefetcher, uses CUDA streams instead of threads.
    Best for GPU-bound workloads where data loading is not the bottleneck.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device | str = "cuda",
    ):
        """Initialize CUDA prefetcher.

        Args:
            loader: Base DataLoader
            device: Target CUDA device
        """
        self.loader = loader
        self.device = torch.device(device)
        self._stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def __iter__(self) -> Iterator[Any]:
        """Iterate with CUDA stream prefetching."""
        loader_iter = iter(self.loader)

        # Load first batch
        try:
            batch = next(loader_iter)
        except StopIteration:
            return

        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                batch = self._to_device(batch)
        else:
            batch = self._to_device(batch)

        while True:
            # Wait for current batch to be ready
            if self._stream is not None:
                torch.cuda.current_stream().wait_stream(self._stream)

            current_batch = batch

            # Start loading next batch
            try:
                next_batch = next(loader_iter)
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        batch = self._to_device(next_batch)
                else:
                    batch = self._to_device(next_batch)
            except StopIteration:
                yield current_batch
                break

            yield current_batch

    def _to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            moved = [self._to_device(x) for x in batch]
            return type(batch)(moved)
        else:
            return batch

    def __len__(self) -> int:
        return len(self.loader)


class MemoryMappedDataset(Dataset):
    """Dataset backed by memory-mapped numpy array for large datasets.

    Avoids loading entire dataset into RAM, enabling training on
    datasets larger than available memory.
    """

    def __init__(
        self,
        path: str,
        dtype: torch.dtype = torch.long,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize memory-mapped dataset.

        Args:
            path: Path to .npy file (created with np.save)
            dtype: Torch dtype for output tensors
            transform: Optional transform to apply
        """
        import numpy as np

        # Memory-map the file
        self.data = np.load(path, mmap_mode="r")
        self.dtype = dtype
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load single item (fast due to mmap)
        item = torch.from_numpy(self.data[idx].copy()).to(self.dtype)

        if self.transform is not None:
            item = self.transform(item)

        return item


class ShardedIterableDataset(IterableDataset):
    """Iterable dataset that shards across distributed workers.

    Useful for very large datasets that don't fit in memory and
    need to be streamed. Automatically handles sharding for DDP.
    """

    def __init__(
        self,
        file_paths: list[str],
        load_fn: Callable[[str], Iterator[torch.Tensor]],
        shuffle_files: bool = True,
        world_size: int | None = None,
        rank: int | None = None,
    ):
        """Initialize sharded iterable dataset.

        Args:
            file_paths: List of data file paths
            load_fn: Function to load and yield samples from a file
            shuffle_files: Whether to shuffle files each epoch
            world_size: Total number of distributed processes (auto-detect if None)
            rank: Rank of current process (auto-detect if None)
        """
        self.file_paths = file_paths
        self.load_fn = load_fn
        self.shuffle_files = shuffle_files

        # Auto-detect distributed settings
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        if rank is None:
            rank = int(os.environ.get("RANK", 0))

        self.world_size = world_size
        self.rank = rank

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Shard files across workers
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Multi-worker: shard by worker
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            # Combine DDP rank and worker ID for sharding
            global_worker_id = self.rank * num_workers + worker_id
            total_workers = self.world_size * num_workers
        else:
            # Single worker: just use DDP rank
            global_worker_id = self.rank
            total_workers = self.world_size

        # Get files for this worker
        files = self.file_paths[global_worker_id::total_workers]

        if self.shuffle_files:
            import random

            files = files.copy()
            random.shuffle(files)

        for file_path in files:
            yield from self.load_fn(file_path)


def create_optimized_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig | None = None,
    distributed: bool = False,
    prefetch: bool = True,
    device: torch.device | str | None = None,
) -> DataLoader | AsyncPrefetcher | CUDAPrefetcher:
    """Create an optimized DataLoader with optional prefetching.

    Args:
        dataset: PyTorch Dataset
        config: DataLoader configuration (uses defaults if None)
        distributed: Whether to use DistributedSampler
        prefetch: Whether to wrap with async prefetcher
        device: Target device for prefetching

    Returns:
        DataLoader or prefetcher wrapper
    """
    if config is None:
        config = DataLoaderConfig()

    # Create sampler for distributed training
    sampler = None
    shuffle = config.shuffle

    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=config.shuffle)
        shuffle = False  # Sampler handles shuffling

    # Create base DataLoader
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last,
    }

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False

    if config.num_workers is not None and config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loader_kwargs["persistent_workers"] = config.persistent_workers

    loader = DataLoader(dataset, **loader_kwargs)

    # Wrap with prefetcher if requested
    if prefetch and device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            return CUDAPrefetcher(loader, device)

    return loader


def collate_variable_length(
    batch: list[torch.Tensor],
    pad_value: int = 0,
    max_length: int | None = None,
) -> torch.Tensor:
    """Collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch.

    Args:
        batch: List of 1D tensors
        pad_value: Value to use for padding
        max_length: Optional maximum length (truncates if exceeded)

    Returns:
        Padded batch tensor [batch_size, max_seq_len]
    """
    lengths = [len(x) for x in batch]
    target_length = max(lengths)

    if max_length is not None:
        target_length = min(target_length, max_length)

    # Create padded tensor
    padded = torch.full(
        (len(batch), target_length),
        pad_value,
        dtype=batch[0].dtype,
    )

    for i, x in enumerate(batch):
        length = min(len(x), target_length)
        padded[i, :length] = x[:length]

    return padded


__all__ = [
    "DataLoaderConfig",
    "AsyncPrefetcher",
    "CUDAPrefetcher",
    "MemoryMappedDataset",
    "ShardedIterableDataset",
    "create_optimized_dataloader",
    "collate_variable_length",
]
