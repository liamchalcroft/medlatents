"""Tests for high-level generation helpers."""

import torch

from medlatents.generation import DiscreteLatentGenerator


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("DummyModel should not be called in these tests")


class TokenizerWithTokenize:
    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def tokenize(self, x):
        self.calls.append("tokenize")
        return torch.full((1, 3), 7, dtype=torch.long)

    def encode(self, x):
        self.calls.append("encode")
        return torch.full((1, 3), 9, dtype=torch.long)


class TokenizerWithEncodeTuple:
    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, x):
        return (torch.full((1, 2), 5, dtype=torch.long), {"meta": True})


class TokenizerWithEncodeTensor:
    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, x):
        return torch.full((1, 4), 3, dtype=torch.long)


class TokenizerMissingEncode:
    def to(self, device):
        return self

    def eval(self):
        return self


def _make_generator(tokenizer):
    model = DummyModel()
    return DiscreteLatentGenerator(
        model_type="autoreg",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )


def test_encode_tokens_prefers_tokenize():
    tokenizer = TokenizerWithTokenize()
    generator = _make_generator(tokenizer)

    tokens = generator._encode_tokens(torch.zeros(1, 1, 1))

    assert tokenizer.calls == ["tokenize"]
    assert tokens.shape == (1, 3)
    assert tokens.dtype == torch.long
    assert torch.all(tokens == 7)


def test_encode_tokens_tuple_from_encode():
    tokenizer = TokenizerWithEncodeTuple()
    generator = _make_generator(tokenizer)

    tokens = generator._encode_tokens(torch.zeros(1, 1, 1))

    assert tokens.shape == (1, 2)
    assert tokens.dtype == torch.long
    assert torch.all(tokens == 5)


def test_encode_tokens_tensor_from_encode():
    tokenizer = TokenizerWithEncodeTensor()
    generator = _make_generator(tokenizer)

    tokens = generator._encode_tokens(torch.zeros(1, 1, 1))

    assert tokens.shape == (1, 4)
    assert tokens.dtype == torch.long
    assert torch.all(tokens == 3)


def test_encode_tokens_missing_methods_raises():
    tokenizer = TokenizerMissingEncode()
    generator = _make_generator(tokenizer)

    try:
        generator._encode_tokens(torch.zeros(1, 1, 1))
    except AttributeError as exc:
        assert "tokenize" in str(exc) or "encode" in str(exc)
    else:
        raise AssertionError("Expected AttributeError for tokenizer without encode/tokenize")
