"""Tests for data loader validation behavior."""

import pytest
import torch

from medlatents.data.tokenized_datasets import (
    get_latent_dataloaders,
    get_tokenized_dataloaders,
    validate_continuous_latents,
    validate_discrete_tokens,
    validate_image_for_tokenization,
)


def test_get_tokenized_dataloaders_requires_tokenizer():
    with pytest.raises(ValueError, match="tokenizer is required"):
        get_tokenized_dataloaders("data/train/*.nii.gz", "data/val/*.nii.gz", tokenizer=None)


def test_get_latent_dataloaders_requires_tokenizer():
    with pytest.raises(ValueError, match="tokenizer is required"):
        get_latent_dataloaders("data/train/*.nii.gz", "data/val/*.nii.gz", tokenizer=None)


class TestValidateDiscreteTokens:
    """Tests for discrete token validation."""

    def test_valid_tokens_pass(self):
        """Valid tokens should pass without error."""
        tokens = torch.randint(0, 100, (4, 32), dtype=torch.long)
        validate_discrete_tokens(tokens, vocab_size=100)
        assert tokens.shape == (4, 32)

    def test_valid_tokens_with_special(self):
        """Tokens including special tokens should pass when allowed."""
        tokens = torch.randint(0, 104, (4, 32), dtype=torch.long)
        validate_discrete_tokens(tokens, vocab_size=100, allow_special=True)
        assert tokens.dtype == torch.long

    def test_wrong_dtype_raises(self):
        """Non-long dtype should raise TypeError."""
        tokens = torch.randint(0, 100, (4, 32), dtype=torch.int32)
        with pytest.raises(TypeError, match="Expected dtype torch.long"):
            validate_discrete_tokens(tokens, vocab_size=100)

    def test_float_dtype_raises(self):
        """Float dtype should raise TypeError."""
        tokens = torch.randn(4, 32)
        with pytest.raises(TypeError, match="Expected dtype torch.long"):
            validate_discrete_tokens(tokens, vocab_size=100)

    def test_wrong_ndim_raises(self):
        """3D tensor should raise ValueError."""
        tokens = torch.randint(0, 100, (4, 32, 8), dtype=torch.long)
        with pytest.raises(ValueError, match="Expected 1D.*or 2D"):
            validate_discrete_tokens(tokens, vocab_size=100)

    def test_negative_values_raise(self):
        """Negative token values should raise ValueError."""
        tokens = torch.randint(-10, 100, (4, 32), dtype=torch.long)
        with pytest.raises(ValueError, match="Contains negative values"):
            validate_discrete_tokens(tokens, vocab_size=100)

    def test_out_of_range_raises(self):
        """Token values >= vocab_size should raise ValueError."""
        tokens = torch.randint(0, 150, (4, 32), dtype=torch.long)
        with pytest.raises(ValueError, match="Contains out-of-range values"):
            validate_discrete_tokens(tokens, vocab_size=100, allow_special=False)

    def test_special_tokens_rejected_when_disallowed(self):
        """Special token values should raise when allow_special=False."""
        tokens = torch.tensor([[100, 101, 102]], dtype=torch.long)  # special tokens
        with pytest.raises(ValueError, match="Contains out-of-range values"):
            validate_discrete_tokens(tokens, vocab_size=100, allow_special=False)

    def test_1d_tokens_valid(self):
        """1D tensor should be valid."""
        tokens = torch.randint(0, 100, (32,), dtype=torch.long)
        validate_discrete_tokens(tokens, vocab_size=100)
        assert tokens.ndim == 1

    def test_empty_tokens_valid(self):
        """Empty tensor should pass (edge case)."""
        tokens = torch.empty(0, dtype=torch.long)
        validate_discrete_tokens(tokens, vocab_size=100)
        assert len(tokens) == 0


class TestValidateContinuousLatents:
    """Tests for continuous latent validation."""

    def test_valid_4d_latents_pass(self):
        """Valid 4D latents should pass."""
        latents = torch.randn(4, 8, 16, 16)
        validate_continuous_latents(latents)

    def test_valid_5d_latents_pass(self):
        """Valid 5D latents should pass."""
        latents = torch.randn(4, 8, 8, 16, 16)
        validate_continuous_latents(latents)

    def test_wrong_dtype_raises(self):
        """Non-floating dtype should raise TypeError."""
        latents = torch.randint(0, 100, (4, 8, 16, 16))
        with pytest.raises(TypeError, match="Expected floating dtype"):
            validate_continuous_latents(latents)

    def test_wrong_ndim_raises(self):
        """3D tensor should raise ValueError."""
        latents = torch.randn(4, 8, 16)
        with pytest.raises(ValueError, match="Expected 4D.*or 5D"):
            validate_continuous_latents(latents)

    def test_nan_raises(self):
        """Tensor with NaN should raise ValueError."""
        latents = torch.randn(4, 8, 16, 16)
        latents[0, 0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="Contains non-finite values"):
            validate_continuous_latents(latents)

    def test_inf_raises(self):
        """Tensor with Inf should raise ValueError."""
        latents = torch.randn(4, 8, 16, 16)
        latents[0, 0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="Contains non-finite values"):
            validate_continuous_latents(latents)

    def test_wrong_channels_raises(self):
        """Wrong channel count should raise ValueError."""
        latents = torch.randn(4, 8, 16, 16)
        with pytest.raises(ValueError, match="Expected 16 channels"):
            validate_continuous_latents(latents, expected_channels=16)

    def test_fp16_valid(self):
        """float16 should be valid."""
        latents = torch.randn(4, 8, 16, 16, dtype=torch.float16)
        validate_continuous_latents(latents)

    def test_bf16_valid(self):
        """bfloat16 should be valid."""
        latents = torch.randn(4, 8, 16, 16, dtype=torch.bfloat16)
        validate_continuous_latents(latents)


class TestValidateImageForTokenization:
    """Tests for image validation before tokenization."""

    def test_valid_3d_image_pass(self):
        """Valid 3D image should pass."""
        image = torch.randn(1, 64, 64)
        validate_image_for_tokenization(image)

    def test_valid_4d_image_pass(self):
        """Valid 4D image should pass."""
        image = torch.randn(1, 32, 64, 64)
        validate_image_for_tokenization(image)

    def test_wrong_dtype_raises(self):
        """Non-floating dtype should raise TypeError."""
        image = torch.randint(0, 255, (1, 64, 64))
        with pytest.raises(TypeError, match="Expected floating dtype"):
            validate_image_for_tokenization(image)

    def test_wrong_ndim_raises(self):
        """2D tensor should raise ValueError."""
        image = torch.randn(64, 64)
        with pytest.raises(ValueError, match="Expected 3D.*or 4D"):
            validate_image_for_tokenization(image)

    def test_5d_raises(self):
        """5D tensor should raise ValueError (needs batch dim added differently)."""
        image = torch.randn(1, 1, 32, 64, 64)
        with pytest.raises(ValueError, match="Expected 3D.*or 4D"):
            validate_image_for_tokenization(image)

    def test_nan_raises(self):
        """Image with NaN should raise ValueError."""
        image = torch.randn(1, 64, 64)
        image[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="Contains non-finite values"):
            validate_image_for_tokenization(image)


class TestEncodeLatentsFallback:
    """Tests for _encode_latents method resolution."""

    def test_tokenize_method_preferred(self):
        """tokenize() method should be called first if available."""
        from medlatents.data.tokenized_datasets import ContinuousLatentDataset

        class MockTokenizer:
            def tokenize(self, x, **kwargs):
                return x * 2  # Simple transform to verify this method was called

            def encode(self, x, **kwargs):
                return (x * 3, None)  # Different transform

        tokenizer = MockTokenizer()
        dataset = ContinuousLatentDataset.__new__(ContinuousLatentDataset)
        dataset.tokenizer = tokenizer
        dataset.tokenizer_kwargs = {}
        dataset.flatten = False
        dataset.channel_first = False

        data = torch.randn(1, 4, 8, 8)
        result = dataset._encode_latents(data)

        # Should use tokenize (multiply by 2), not encode (multiply by 3)
        expected = data * 2
        assert torch.allclose(result, expected), "tokenize() should be preferred over encode()"

    def test_encode_tuple_extraction(self):
        """encode() returning tuple should extract first element."""
        from medlatents.data.tokenized_datasets import ContinuousLatentDataset

        class MockTokenizer:
            def encode(self, x, **kwargs):
                latents = x * 2
                extras = {"some": "metadata"}
                return (latents, extras)

        tokenizer = MockTokenizer()
        dataset = ContinuousLatentDataset.__new__(ContinuousLatentDataset)
        dataset.tokenizer = tokenizer
        dataset.tokenizer_kwargs = {}
        dataset.flatten = False
        dataset.channel_first = False

        data = torch.randn(1, 4, 8, 8)
        result = dataset._encode_latents(data)

        expected = data * 2
        assert torch.allclose(result, expected), "encode() tuple should extract latents"

    def test_encode_non_tuple_passthrough(self):
        """encode() returning non-tuple should pass through."""
        from medlatents.data.tokenized_datasets import ContinuousLatentDataset

        class MockTokenizer:
            def encode(self, x, **kwargs):
                return x * 2  # Returns tensor directly, not tuple

        tokenizer = MockTokenizer()
        dataset = ContinuousLatentDataset.__new__(ContinuousLatentDataset)
        dataset.tokenizer = tokenizer
        dataset.tokenizer_kwargs = {}
        dataset.flatten = False
        dataset.channel_first = False

        data = torch.randn(1, 4, 8, 8)
        result = dataset._encode_latents(data)

        expected = data * 2
        assert torch.allclose(result, expected), "encode() non-tuple should pass through"

    def test_encode_latents_method(self):
        """encode_latents() method should be used if tokenize not available."""
        from medlatents.data.tokenized_datasets import ContinuousLatentDataset

        class MockTokenizer:
            def encode_latents(self, x, **kwargs):
                return x * 4

            def encode(self, x, **kwargs):
                return (x * 3, None)

        tokenizer = MockTokenizer()
        dataset = ContinuousLatentDataset.__new__(ContinuousLatentDataset)
        dataset.tokenizer = tokenizer
        dataset.tokenizer_kwargs = {}
        dataset.flatten = False
        dataset.channel_first = False

        data = torch.randn(1, 4, 8, 8)
        result = dataset._encode_latents(data)

        # Should use encode_latents (multiply by 4), not encode (multiply by 3)
        expected = data * 4
        assert torch.allclose(result, expected), "encode_latents() should be used before encode()"

    def test_no_method_raises_attribute_error(self):
        """Missing all encode methods should raise AttributeError."""
        from medlatents.data.tokenized_datasets import ContinuousLatentDataset

        class MockTokenizer:
            def forward(self, x, **kwargs):  # forward should NOT be used
                return x

        tokenizer = MockTokenizer()
        dataset = ContinuousLatentDataset.__new__(ContinuousLatentDataset)
        dataset.tokenizer = tokenizer
        dataset.tokenizer_kwargs = {}
        dataset.flatten = False
        dataset.channel_first = False

        data = torch.randn(1, 4, 8, 8)
        with pytest.raises(AttributeError, match="does not expose"):
            dataset._encode_latents(data)

    def test_forward_not_used_as_fallback(self):
        """forward() should NOT be used as encoding fallback."""
        from medlatents.data.tokenized_datasets import ContinuousLatentDataset

        class MockTokenizer:
            def forward(self, x, **kwargs):
                return x * 5  # Should not be called

            def __call__(self, x, **kwargs):
                return x * 6  # Should not be called

        tokenizer = MockTokenizer()
        dataset = ContinuousLatentDataset.__new__(ContinuousLatentDataset)
        dataset.tokenizer = tokenizer
        dataset.tokenizer_kwargs = {}
        dataset.flatten = False
        dataset.channel_first = False

        data = torch.randn(1, 4, 8, 8)
        # Should raise because tokenize/encode_latents/encode are all missing
        with pytest.raises(AttributeError, match="does not expose"):
            dataset._encode_latents(data)
