"""Profiling utilities for training bottleneck detection."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed section."""

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.count, 1)

    def update(self, elapsed: float) -> None:
        self.count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)


@dataclass
class TrainingProfiler:
    """Profiler for training loop bottleneck detection.

    Usage:
        profiler = TrainingProfiler(enabled=True)

        for batch in dataloader:
            with profiler.section("data_load"):
                data = preprocess(batch)

            with profiler.section("forward"):
                loss = model(data)

            with profiler.section("backward"):
                loss.backward()

            with profiler.section("optimizer"):
                optimizer.step()

        profiler.print_report()
    """

    enabled: bool = True
    sync_cuda: bool = True  # Synchronize CUDA for accurate GPU timing
    _stats: dict[str, TimingStats] = field(default_factory=lambda: defaultdict(TimingStats))
    _active_section: str | None = field(default=None, init=False)
    _section_start: float = field(default=0.0, init=False)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a section of code.

        Args:
            name: Name of the section (e.g., "forward", "backward", "data_load")
        """
        if not self.enabled:
            yield
            return

        # Sync CUDA before starting
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            # Sync CUDA before measuring end time
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            self._stats[name].update(elapsed)

    def reset(self) -> None:
        """Reset all timing statistics."""
        self._stats.clear()

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get timing statistics for all sections.

        Returns:
            Dict mapping section names to their timing statistics.
        """
        return {
            name: {
                "count": stats.count,
                "total_ms": stats.total_time * 1000,
                "avg_ms": stats.avg_time * 1000,
                "min_ms": stats.min_time * 1000 if stats.min_time != float("inf") else 0,
                "max_ms": stats.max_time * 1000,
            }
            for name, stats in self._stats.items()
        }

    def get_percentages(self) -> dict[str, float]:
        """Get percentage of time spent in each section.

        Returns:
            Dict mapping section names to their percentage of total time.
        """
        total = sum(s.total_time for s in self._stats.values())
        if total == 0:
            return {}
        return {name: (stats.total_time / total) * 100 for name, stats in self._stats.items()}

    def print_report(self, top_n: int | None = None) -> None:
        """Print a formatted profiling report.

        Args:
            top_n: Only show top N sections by time (None = show all)
        """
        if not self._stats:
            logger.info("No profiling data collected.")
            return

        stats = self.get_stats()
        percentages = self.get_percentages()

        # Sort by total time
        sorted_sections = sorted(stats.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        if top_n:
            sorted_sections = sorted_sections[:top_n]

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PROFILER REPORT")
        logger.info("=" * 70)
        logger.info(f"{'Section':<20} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'%':>8}")
        logger.info("-" * 70)

        for name, s in sorted_sections:
            pct = percentages.get(name, 0)
            logger.info(
                f"{name:<20} {s['count']:>8} {s['total_ms']:>12.2f} {s['avg_ms']:>10.3f} {pct:>7.1f}%"
            )

        logger.info("-" * 70)
        total_ms = sum(s["total_ms"] for s in stats.values())
        logger.info(f"{'TOTAL':<20} {'':<8} {total_ms:>12.2f}")
        logger.info("=" * 70 + "\n")

    def get_wandb_metrics(self, prefix: str = "profiler/") -> dict[str, float]:
        """Get metrics formatted for wandb logging.

        Args:
            prefix: Prefix for metric names

        Returns:
            Dict of metrics suitable for wandb.log()
        """
        metrics = {}
        for name, stats in self.get_stats().items():
            metrics[f"{prefix}{name}_avg_ms"] = stats["avg_ms"]
            metrics[f"{prefix}{name}_total_ms"] = stats["total_ms"]
        return metrics


class MemoryProfiler:
    """Track GPU memory usage during training."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self._peak_memory: dict[str, int] = {}
        self._current_memory: dict[str, int] = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        """Track memory usage for a section of code."""
        if not self.enabled:
            yield
            return

        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        try:
            yield
        finally:
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()

            self._current_memory[name] = end_mem - start_mem
            self._peak_memory[name] = peak_mem

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get memory statistics in MB."""
        return {
            name: {
                "delta_mb": self._current_memory.get(name, 0) / (1024 * 1024),
                "peak_mb": self._peak_memory.get(name, 0) / (1024 * 1024),
            }
            for name in set(self._current_memory) | set(self._peak_memory)
        }

    def print_report(self) -> None:
        """Print memory usage report."""
        if not self._peak_memory:
            logger.info("No memory profiling data collected.")
            return

        stats = self.get_stats()

        logger.info("\n" + "=" * 50)
        logger.info("MEMORY PROFILER REPORT")
        logger.info("=" * 50)
        logger.info(f"{'Section':<20} {'Delta(MB)':>12} {'Peak(MB)':>12}")
        logger.info("-" * 50)

        for name, s in sorted(stats.items(), key=lambda x: x[1]["peak_mb"], reverse=True):
            logger.info(f"{name:<20} {s['delta_mb']:>12.2f} {s['peak_mb']:>12.2f}")

        logger.info("=" * 50 + "\n")


def profile_training_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    profiler: TrainingProfiler | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Profile a single training step.

    Convenience function that profiles forward, backward, and optimizer steps.

    Args:
        model: The model to train
        batch: Input batch
        optimizer: The optimizer
        loss_fn: Loss function that takes (model, batch) and returns loss
        profiler: Optional profiler (creates one if None)

    Returns:
        Tuple of (loss, timing_metrics)

    Example:
        loss, metrics = profile_training_step(
            model, batch, optimizer,
            loss_fn=lambda m, b: F.cross_entropy(m(b), targets)
        )
        wandb.log(metrics)
    """
    if profiler is None:
        profiler = TrainingProfiler(enabled=True)

    with profiler.section("forward"):
        loss = loss_fn(model, batch)

    with profiler.section("backward"):
        loss.backward()

    with profiler.section("optimizer"):
        optimizer.step()
        optimizer.zero_grad()

    return loss, profiler.get_wandb_metrics()
