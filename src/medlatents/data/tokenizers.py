"""Tokenizer loading and management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from medtokenizers.networks.continuous import ContinuousTokenizer
from medtokenizers.networks.discrete import DiscreteTokenizer

if TYPE_CHECKING:
    from medtokenizers.networks.continuous import ContinuousTokenizer as _ContinuousTokenizer
    from medtokenizers.networks.discrete import DiscreteTokenizer as _DiscreteTokenizer
else:
    _ContinuousTokenizer = ContinuousTokenizer
    _DiscreteTokenizer = DiscreteTokenizer

Tokenizer = DiscreteTokenizer | ContinuousTokenizer


def get_tokenizer_info(tokenizer: Tokenizer) -> dict:
    """Extract key information from a tokenizer instance."""
    info = {
        "quantizer_type": getattr(tokenizer, "quantizer_type", None),
        "spatial_compression": getattr(tokenizer, "spatial_compression", None),
        "embedding_dim": getattr(tokenizer, "embedding_dim", None),
        "dim": getattr(tokenizer, "dim", None),
    }

    quantizer = getattr(tokenizer, "quantizer", None)
    if quantizer is not None:
        for attr in (
            "codebook_size",
            "num_embeddings",
            "levels",
            "latent_dim",
            "channels",
        ):
            if hasattr(quantizer, attr):
                info[attr] = getattr(quantizer, attr)

    for attr in ("latent_dim", "latent_channels", "latent_shape"):
        if hasattr(tokenizer, attr):
            info[attr] = getattr(tokenizer, attr)

    info["is_discrete"] = isinstance(tokenizer, DiscreteTokenizer)
    info["is_continuous"] = isinstance(tokenizer, ContinuousTokenizer)

    return info


def get_tokenizer_module_name(tokenizer: Tokenizer) -> str:
    """Get the module name of a tokenizer for better error messages.

    Args:
        tokenizer: Tokenizer instance from medtokenizers

    Returns:
        Module name string (e.g., 'ContinuousTokenizer', 'DiscreteTokenizer')
    """
    return type(tokenizer).__name__


__all__ = [
    "Tokenizer",
    "get_tokenizer_info",
    "get_tokenizer_module_name",
    "DiscreteTokenizer",
    "ContinuousTokenizer",
]
