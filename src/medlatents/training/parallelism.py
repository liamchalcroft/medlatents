"""Distributed training utilities with DDP and FSDP support.

Provides easy-to-use wrappers for:
- DistributedDataParallel (DDP) for multi-GPU training
- FullyShardedDataParallel (FSDP) for memory-efficient large model training
- Automatic mixed precision integration
- Gradient accumulation helpers
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        world_size: Total number of processes
        rank: Rank of current process
        local_rank: Local rank on current node
        master_addr: Master node address
        master_port: Master node port
        use_fsdp: Use FSDP instead of DDP
        fsdp_sharding_strategy: FSDP sharding strategy
        mixed_precision: Mixed precision dtype ('fp16', 'bf16', or None)
    """

    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "full"  # 'full', 'shard_grad_op', 'no_shard'
    mixed_precision: str | None = "bf16"
    gradient_checkpointing: bool = True

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Create config from environment variables (set by torchrun)."""
        return cls(
            backend=os.environ.get("DIST_BACKEND", "nccl"),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=os.environ.get("MASTER_PORT", "29500"),
        )

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is enabled."""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0


def setup_distributed(config: DistributedConfig | None = None) -> DistributedConfig:
    """Initialize distributed training environment.

    Args:
        config: Distributed configuration (auto-detects from env if None)

    Returns:
        Initialized configuration
    """
    if config is None:
        config = DistributedConfig.from_env()

    if not config.is_distributed:
        return config

    # Set environment variables
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            world_size=config.world_size,
            rank=config.rank,
        )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)

    return config


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    config: DistributedConfig,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        config: Distributed configuration
        find_unused_parameters: Enable for models with unused params
        gradient_as_bucket_view: Memory optimization

    Returns:
        DDP-wrapped model
    """
    if not config.is_distributed:
        return model

    device_id = config.local_rank if torch.cuda.is_available() else None

    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id] if device_id is not None else None,
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )


def wrap_model_fsdp(
    model: nn.Module,
    config: DistributedConfig,
    auto_wrap_policy: Any | None = None,
    cpu_offload: bool = False,
) -> nn.Module:
    """Wrap model with FullyShardedDataParallel.

    FSDP shards model parameters across GPUs for memory efficiency.

    Args:
        model: Model to wrap
        config: Distributed configuration
        auto_wrap_policy: Policy for automatic module wrapping
        cpu_offload: Offload parameters to CPU (slower but lower GPU memory)

    Returns:
        FSDP-wrapped model
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    # Determine sharding strategy
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = strategy_map.get(config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Set up mixed precision
    mp_policy = None
    if config.mixed_precision is not None:
        dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        mp_policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    # Set up CPU offload
    cpu_offload_config = None
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload

        cpu_offload_config = CPUOffload(offload_params=True)

    # Auto wrap policy - use size-based policy if none provided
    if auto_wrap_policy is None:
        from functools import partial

        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,  # Wrap modules with > 1M params
        )

    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_config,
        auto_wrap_policy=auto_wrap_policy,
        device_id=config.local_rank,
    )


def wrap_model(
    model: nn.Module,
    config: DistributedConfig,
    **kwargs: Any,
) -> nn.Module:
    """Wrap model with appropriate distributed strategy.

    Chooses DDP or FSDP based on configuration.

    Args:
        model: Model to wrap
        config: Distributed configuration
        **kwargs: Additional arguments for DDP/FSDP

    Returns:
        Wrapped model
    """
    if not config.is_distributed:
        return model

    if config.use_fsdp:
        return wrap_model_fsdp(model, config, **kwargs)
    else:
        return wrap_model_ddp(model, config, **kwargs)


@contextmanager
def sync_gradients(model: nn.Module, sync: bool = True) -> Iterator[None]:
    """Context manager to control gradient synchronization in DDP.

    Use with gradient accumulation to skip synchronization on intermediate steps.

    Usage:
        for i, batch in enumerate(loader):
            sync = (i + 1) % accumulation_steps == 0
            with sync_gradients(model, sync):
                loss = compute_loss(batch)
                loss.backward()

            if sync:
                optimizer.step()
                optimizer.zero_grad()
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        with model.no_sync() if not sync else contextmanager(lambda: (yield))():
            yield
    else:
        yield


class GradientAccumulator:
    """Helper for gradient accumulation with DDP sync control.

    Automatically handles gradient synchronization, accumulation,
    and optimizer stepping.

    Usage:
        accumulator = GradientAccumulator(model, optimizer, steps=4)

        for batch in loader:
            with accumulator.accumulate():
                loss = model(batch)
                loss.backward()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        max_grad_norm: float | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ):
        """Initialize gradient accumulator.

        Args:
            model: Model (may be DDP/FSDP wrapped)
            optimizer: Optimizer
            steps: Number of accumulation steps
            max_grad_norm: Gradient clipping threshold
            scaler: AMP grad scaler (for mixed precision)
        """
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler

        self._step_count = 0

    @contextmanager
    def accumulate(self) -> Iterator[bool]:
        """Context manager for a single accumulation step.

        Yields:
            bool: Whether this step should trigger an optimizer update
        """
        self._step_count += 1
        should_sync = self._step_count >= self.steps

        with sync_gradients(self.model, should_sync):
            yield should_sync

        if should_sync:
            self._do_step()
            self._step_count = 0

    def _do_step(self) -> None:
        """Perform optimizer step with optional gradient clipping."""
        if self.scaler is not None:
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

    @property
    def is_sync_step(self) -> bool:
        """Whether current step will trigger synchronization."""
        return (self._step_count + 1) >= self.steps


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce tensor and compute mean across processes.

    Useful for aggregating loss values across distributed processes.

    Args:
        tensor: Input tensor (will be modified in-place)

    Returns:
        Reduced tensor (averaged across all processes)
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast Python object from source rank to all processes.

    Useful for sharing configuration or random seeds.

    Args:
        obj: Object to broadcast (only used on source rank)
        src: Source rank

    Returns:
        Broadcast object
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier() -> None:
    """Synchronization barrier across all processes."""
    if dist.is_initialized():
        dist.barrier()


__all__ = [
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_ddp",
    "wrap_model_fsdp",
    "wrap_model",
    "sync_gradients",
    "GradientAccumulator",
    "all_reduce_mean",
    "broadcast_object",
    "barrier",
]
