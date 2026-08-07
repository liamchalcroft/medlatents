"""Shared test fixtures and utilities for all tests."""

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Ensure src/ is on the path for direct test invocation
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def device() -> torch.device:
    """Get test device (CPU for deterministic tests)."""
    return torch.device("cpu")


@pytest.fixture
def base_model_config() -> dict[str, Any]:
    """Standard small model config for fast testing."""
    return {
        "seq_length": 32,
        "vocab_size": 128,
        "hidden_size": 64,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "allow_dynamic_seq_length": False,
        "gradient_checkpointing": False,
    }


@pytest.fixture
def nano_model_config() -> dict[str, Any]:
    """Tiny model config for very fast tests."""
    return {
        "seq_length": 16,
        "vocab_size": 64,
        "hidden_size": 32,
        "depth": 1,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "allow_dynamic_seq_length": False,
        "gradient_checkpointing": False,
    }


@pytest.fixture
def sample_batch(base_model_config: dict[str, Any]) -> torch.Tensor:
    """Create a sample batch of tokens."""
    torch.manual_seed(42)
    batch_size = 4
    seq_length = base_model_config["seq_length"]
    vocab_size = base_model_config["vocab_size"]
    return torch.randint(0, vocab_size, (batch_size, seq_length))


@pytest.fixture
def sample_small_batch(nano_model_config: dict[str, Any]) -> torch.Tensor:
    """Create a small sample batch for fast tests."""
    torch.manual_seed(42)
    batch_size = 2
    seq_length = nano_model_config["seq_length"]
    vocab_size = nano_model_config["vocab_size"]
    return torch.randint(0, vocab_size, (batch_size, seq_length))


def assert_valid_logits(logits: torch.Tensor, expected_shape: tuple, vocab_size: int) -> None:
    """Assert logits have correct shape and are finite."""
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    assert logits.size(-1) == vocab_size, f"Last dim should be vocab_size {vocab_size}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


def assert_valid_samples(
    samples: torch.Tensor, vocab_size: int, expected_shape: tuple = None
) -> None:
    """Assert samples are valid token indices."""
    if expected_shape is not None:
        assert samples.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {samples.shape}"
        )
    assert samples.min() >= 0, "Samples contain negative indices"
    assert samples.max() < vocab_size, f"Samples exceed vocab_size {vocab_size}"
    assert samples.dtype in [torch.long, torch.int], f"Expected integer dtype, got {samples.dtype}"
