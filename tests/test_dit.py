import pytest
import torch

from medlatents.networks import DiscreteDiT


@pytest.fixture
def base_kwargs() -> dict:
    return {
        "seq_length": 16,
        "vocab_size": 64,
        "hidden_size": 128,
        "depth": 2,
        "num_heads": 4,
        "allow_dynamic_seq_length": False,
        "gradient_checkpointing": False,
    }


def test_unconditional_forward_shape(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, num_classes=0)
    batch_size = 2
    x = torch.randint(0, base_kwargs["vocab_size"], (batch_size, 1, base_kwargs["seq_length"]))
    t = torch.randint(0, 1000, (batch_size,))

    logits = model(x, t)

    # DiscreteDiT doesn't add special tokens (only BayesianFlowTransformer does)
    assert logits.shape == (
        batch_size,
        base_kwargs["seq_length"],
        base_kwargs["vocab_size"],
    )


def test_conditional_requires_labels(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, num_classes=4, class_dropout_prob=0.1)
    batch_size = 2
    x = torch.randint(0, base_kwargs["vocab_size"], (batch_size, 1, base_kwargs["seq_length"]))
    t = torch.randint(0, 1000, (batch_size,))

    with pytest.raises(ValueError):
        model(x, t)


def test_forward_with_cfg_returns_logits(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, num_classes=4, class_dropout_prob=0.2)
    batch_size = 2
    x = torch.randint(0, base_kwargs["vocab_size"], (batch_size * 2, 1, base_kwargs["seq_length"]))
    t = torch.randint(0, 1000, (batch_size * 2,))
    y = torch.randint(0, 4, (batch_size,))

    guided_logits = model.forward_with_cfg(x, t, y, cfg_scale=3.5)

    # DiscreteDiT doesn't add special tokens (only BayesianFlowTransformer does)
    assert guided_logits.shape == (
        batch_size,
        base_kwargs["seq_length"],
        base_kwargs["vocab_size"],
    )


def test_forward_with_cfg_rejects_unconditional_model(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, num_classes=0)
    x = torch.randint(0, base_kwargs["vocab_size"], (4, 1, base_kwargs["seq_length"]))
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 4, (2,))

    with pytest.raises(ValueError):
        model.forward_with_cfg(x, t, y, cfg_scale=2.0)


def test_sequence_length_validation(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, num_classes=0)
    batch_size = 1
    x = torch.randint(0, base_kwargs["vocab_size"], (batch_size, 1, base_kwargs["seq_length"] + 4))
    t = torch.randint(0, 1000, (batch_size,))

    with pytest.raises(ValueError):
        model(x, t)


def test_dynamic_sequence_length_updates_cache(base_kwargs: dict) -> None:
    model = DiscreteDiT(**{**base_kwargs, "allow_dynamic_seq_length": True}, num_classes=0)
    batch_size = 1
    new_length = base_kwargs["seq_length"] + 8
    x = torch.randint(0, base_kwargs["vocab_size"], (batch_size, 1, new_length))
    t = torch.randint(0, 1000, (batch_size,))

    logits = model(x, t)

    assert logits.shape[1] == new_length
    assert model.seq_len_cached == new_length


def test_vocab_validation_raises(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, num_classes=0)
    batch_size = 1
    # Use a token ID that's out of range (valid is 0 to vocab-1)
    x = torch.full(
        (batch_size, 1, base_kwargs["seq_length"]),
        base_kwargs["vocab_size"],  # This is out of range
        dtype=torch.long,
    )
    t = torch.randint(0, 1000, (batch_size,))

    with pytest.raises(ValueError, match="Token values must be in"):
        model(x, t)


def test_masked_model_expands_vocab(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, masked=True, num_classes=0)
    batch_size = 1
    # With masked=True, vocab is base + 1 (for mask token)
    x = torch.randint(0, base_kwargs["vocab_size"] + 1, (batch_size, 1, base_kwargs["seq_length"]))
    t = torch.randint(0, 1000, (batch_size,))

    logits = model(x, t)

    # With masked=True, output vocab is base + 1
    assert logits.shape[-1] == base_kwargs["vocab_size"] + 1


def test_masked_model_rejects_out_of_range_tokens(base_kwargs: dict) -> None:
    model = DiscreteDiT(**base_kwargs, masked=True, num_classes=0)
    batch_size = 1
    # Token ID that's out of range (valid is 0 to vocab, inclusive with masked=True)
    x = torch.full(
        (batch_size, 1, base_kwargs["seq_length"]),
        base_kwargs["vocab_size"] + 1,  # Out of range
        dtype=torch.long,
    )
    t = torch.randint(0, 1000, (batch_size,))

    with pytest.raises(ValueError, match="Token values must be in"):
        model(x, t)
