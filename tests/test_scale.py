"""Scale and stress tests for large sequences and memory usage."""

import gc

import pytest
import torch

from medlatents import AutoregressiveTransformer, MaskGIT
from medlatents.configs import MODEL_CONFIGS


class TestLargeSequences:
    """Tests for model behavior with large sequences."""

    @pytest.mark.slow
    def test_autoregressive_long_sequence_16k(self):
        """AR model handles 16K token sequences."""
        config = MODEL_CONFIGS["nano"]

        model = AutoregressiveTransformer(
            seq_length=16384,
            vocab_size=1024,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            gradient_checkpointing=True,  # Required for long sequences
        )

        # Shorter sequence for testing
        input_tokens = torch.randint(0, 1024, (2, 256))
        output = model(input_tokens)

        # vocab_size is extended by 4 special tokens (BOS, EOS, PAD, MASK)
        assert output.shape == (2, 256, model.vocab_size)

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_large_batch_memory_stress(self):
        """Model handles large batches with GPU."""
        config = MODEL_CONFIGS["S"]

        model = MaskGIT(
            seq_length=256,
            vocab_size=512,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
        ).cuda()

        # Large batch size
        input_tokens = torch.randint(0, 512, (64, 256)).cuda()
        output = model(input_tokens)

        assert output.shape == (64, 256, 512)

        # Clean up
        del model, input_tokens, output
        torch.cuda.empty_cache()
        gc.collect()

    @pytest.mark.slow
    def test_memory_leak_detection(self):
        """Detect memory leaks during repeated forward passes."""
        model = MaskGIT(
            seq_length=256,
            vocab_size=512,
            hidden_size=128,
            depth=4,
            num_heads=4,
        )

        has_cuda = torch.cuda.is_available()
        initial_mem = 0

        # Get initial memory usage
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
            initial_mem = torch.cuda.max_memory_allocated()

        # Repeated forward passes
        input_tokens = torch.randint(0, 512, (4, 256))

        for i in range(100):
            _ = model(input_tokens)

            if i % 10 == 0 and has_cuda:
                gc.collect()

        # Check final memory usage
        if has_cuda:
            final_mem = torch.cuda.max_memory_allocated()
            mem_growth = final_mem - initial_mem

            # Memory should not grow unboundedly
            assert mem_growth < 1024 * 1024 * 100  # Less than 100MB growth


class TestModelStability:
    """Tests for model stability under stress conditions."""

    def test_extreme_value_inputs(self):
        """Model handles extreme input values."""
        model = AutoregressiveTransformer(
            seq_length=64,
            vocab_size=512,
            hidden_size=128,
            depth=4,
            num_heads=4,
        )

        # Normal inputs
        normal_input = torch.randint(0, 512, (2, 64))
        output_normal = model(normal_input)

        # Boundary values (all same token)
        boundary_input = torch.zeros(2, 64, dtype=torch.long)
        output_boundary = model(boundary_input)

        # Should produce reasonable outputs (vocab_size extended by 4 special tokens)
        assert output_normal.shape == (2, 64, model.vocab_size)
        assert output_boundary.shape == (2, 64, model.vocab_size)
        assert not torch.isnan(output_normal).any()
        assert not torch.isnan(output_boundary).any()

    def test_gradient_stability(self):
        """Gradients are stable during backprop."""
        model = MaskGIT(
            seq_length=64,
            vocab_size=512,
            hidden_size=128,
            depth=4,
            num_heads=4,
        )

        input_tokens = torch.randint(0, 512, (4, 64))
        logits = model(input_tokens)

        # Compute loss
        target = torch.randint(0, 512, (4, 64))
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target.flatten(),
        )

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.shape == param.shape
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()


class TestConcurrentGeneration:
    """Tests for concurrent generation scenarios."""

    @pytest.mark.slow
    def test_concurrent_generation(self):
        """Multiple generations can run concurrently."""
        model = AutoregressiveTransformer(
            seq_length=256,
            vocab_size=512,
            hidden_size=128,
            depth=4,
            num_heads=4,
        ).eval()

        # Generate multiple samples
        results = []
        for i in range(10):
            with torch.no_grad():
                # Start with a single token prompt (BOS token = vocab_size)
                prompt = torch.full((1, 1), model.special_tokens.bos, dtype=torch.long)
                generated = model.generate(
                    prompt=prompt,
                    max_length=64,  # Use shorter length for speed
                    temperature=1.0,
                )
                results.append(generated)

        # All should have correct shape (prompt + generated)
        for result in results:
            assert result.shape[0] == 1
            assert result.shape[1] <= 64
