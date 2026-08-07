"""Shared utilities for managing BOS/EOS/PAD/MASK token ids."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokenIds:
    """Container for special token identifiers."""

    bos: int
    eos: int
    pad: int
    mask: int

    def as_dict(self) -> dict:
        return {"bos": self.bos, "eos": self.eos, "pad": self.pad, "mask": self.mask}


def derive_special_tokens(
    base_vocab_size: int,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    mask_token_id: int | None = None,
) -> tuple[int, SpecialTokenIds]:
    """Allocate special tokens just above the base vocab.

    Args:
        base_vocab_size: tokenizer codebook size (no specials).
        *_token_id: optionally pre-defined ids; otherwise we assign sequentially.

    Returns:
        extended_vocab_size, SpecialTokenIds.
    """

    next_id = base_vocab_size

    def _assign(requested: int | None) -> int:
        nonlocal next_id
        if requested is not None:
            return requested
        assigned = next_id
        next_id += 1
        return assigned

    bos = _assign(bos_token_id)
    eos = _assign(eos_token_id)
    pad = _assign(pad_token_id)
    mask = _assign(mask_token_id)

    special = SpecialTokenIds(bos=bos, eos=eos, pad=pad, mask=mask)
    extended_vocab = max(special.bos, special.eos, special.pad, special.mask) + 1
    return extended_vocab, special


def resolve_special_tokens(
    base_vocab_size: int,
    special_tokens: SpecialTokenIds | None = None,
    add_special_tokens: bool = True,
) -> tuple[int, SpecialTokenIds]:
    """Resolve final vocab size and special tokens configuration."""

    if not add_special_tokens:
        if special_tokens is None:
            if base_vocab_size < 4:
                raise ValueError(
                    "base_vocab_size must be >= 4 when reusing in-vocab special token ids"
                )
            special_tokens = SpecialTokenIds(
                bos=base_vocab_size - 4,
                eos=base_vocab_size - 3,
                pad=base_vocab_size - 2,
                mask=base_vocab_size - 1,
            )
        extended_vocab = max(special_tokens.as_dict().values()) + 1
        return extended_vocab, special_tokens

    return derive_special_tokens(
        base_vocab_size,
        bos_token_id=getattr(special_tokens, "bos", None) if special_tokens else None,
        eos_token_id=getattr(special_tokens, "eos", None) if special_tokens else None,
        pad_token_id=getattr(special_tokens, "pad", None) if special_tokens else None,
        mask_token_id=getattr(special_tokens, "mask", None) if special_tokens else None,
    )


__all__ = ["SpecialTokenIds", "derive_special_tokens", "resolve_special_tokens"]
