"""State-dict compatibility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn


def load_state_dict_compat(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    ignored_suffixes: tuple[str, ...] = ("freqs_cis",),
) -> None:
    """Load a state dict while ignoring deterministic cache buffers.

    Some models serialize rotary cache buffers (for example ``freqs_cis``) whose
    shapes can vary across runs depending on the most recent sequence length.
    Those buffers are deterministic and can be safely ignored when loading.
    """

    filtered = {
        key: value for key, value in state_dict.items() if not key.endswith(ignored_suffixes)
    }
    incompat = module.load_state_dict(filtered, strict=False)
    missing = [key for key in incompat.missing_keys if not key.endswith(ignored_suffixes)]
    unexpected = [key for key in incompat.unexpected_keys if not key.endswith(ignored_suffixes)]
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible state_dict load. Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
