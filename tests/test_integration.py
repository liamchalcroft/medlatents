"""Integration tests for end-to-end workflows.

These tests verify that components work together correctly in realistic pipelines.
"""

import math

import pytest
import torch

from medlatents.autoregressive.transformer import AutoregressiveTransformer
from medlatents.diffusion.d3pm import D3PM
from medlatents.flow_matching.core import MixtureDiscreteProbPath, PolynomialConvexScheduler
from medlatents.flow_matching.discrete import MaskedSourceDistribution
from medlatents.maskgit.transformer import MaskGIT
from medlatents.rasterization.hilbert_curve import HilbertCurve
from medlatents.rasterization.raster_scan import RasterScan
from medlatents.rasterization.zorder_curve import ZOrderCurve
from medlatents.sampling.autoregressive import (
    TemperatureScheduler,
    sample_autoregressive,
)


class TestAutoregressiveEndToEnd:
    """Test autoregressive model training and generation pipeline."""

    @pytest.fixture
    def ar_model(self):
        """Create small AR model for testing."""
        return AutoregressiveTransformer(
            seq_length=32,
            vocab_size=64,
            hidden_size=32,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
        )

    def test_train_and_generate_cycle(self, ar_model):
        """Test that we can train and then generate with an AR model."""
        batch_size = 2
        seq_length = 32
        vocab_size = 64

        # Training step
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        logits = ar_model(x)

        # AR model expands vocab for special tokens
        output_vocab_size = logits.size(-1)
        assert logits.shape == (batch_size, seq_length, output_vocab_size)
        assert torch.isfinite(logits).all()

        # Compute loss (next token prediction)
        targets = x[:, 1:]
        preds = logits[:, :-1]
        loss = torch.nn.functional.cross_entropy(
            preds.reshape(-1, output_vocab_size), targets.reshape(-1)
        )
        assert loss.item() > 0
        assert torch.isfinite(loss)

        # Generate from prompt
        prompt = torch.randint(0, vocab_size, (1, 4))
        generated = [prompt]

        ar_model.eval()
        with torch.no_grad():
            for _ in range(10):
                current_seq = torch.cat(generated, dim=1)
                logits = ar_model(current_seq)
                next_logits = logits[:, -1, :]
                next_token = sample_autoregressive(next_logits, temperature=0.8)
                generated.append(next_token.unsqueeze(1))

        final_seq = torch.cat(generated, dim=1)
        assert final_seq.shape == (1, 14)  # 4 prompt + 10 generated
        assert final_seq.min() >= 0

    def test_temperature_scheduling(self, ar_model):
        """Test generation with temperature scheduling."""
        vocab_size = 64
        num_steps = 10

        scheduler = TemperatureScheduler(
            schedule="cosine", start_temp=1.2, end_temp=0.5, num_steps=num_steps
        )

        prompt = torch.randint(0, vocab_size, (1, 2))
        generated = [prompt]

        ar_model.eval()
        with torch.no_grad():
            for step in range(num_steps):
                current_seq = torch.cat(generated, dim=1)
                logits = ar_model(current_seq)
                next_logits = logits[:, -1, :]

                temp = scheduler.get_temperature(step)
                assert 0.5 <= temp <= 1.2

                next_token = sample_autoregressive(next_logits, temperature=temp)
                generated.append(next_token.unsqueeze(1))

        final_seq = torch.cat(generated, dim=1)
        assert final_seq.shape == (1, 2 + num_steps)


class TestMaskGITEndToEnd:
    """Test MaskGIT model training and generation pipeline."""

    @pytest.fixture
    def maskgit_model(self):
        """Create small MaskGIT model for testing."""
        return MaskGIT(
            seq_length=32,
            vocab_size=64,
            hidden_size=32,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
        )

    def test_train_and_generate_cycle(self, maskgit_model):
        """Test MaskGIT training and iterative decoding."""
        batch_size = 2
        seq_length = 32
        vocab_size = 64
        mask_token_id = maskgit_model.special_tokens.mask

        # Training: mask some tokens and predict them
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        mask_ratio = 0.5
        mask = torch.rand(batch_size, seq_length) < mask_ratio
        x_masked = x.clone()
        x_masked[mask] = mask_token_id

        logits = maskgit_model(x_masked)
        # MaskGIT outputs vocab_size + special tokens
        output_vocab_size = logits.size(-1)
        assert logits.shape == (batch_size, seq_length, output_vocab_size)

        # Loss only on masked positions
        loss = torch.nn.functional.cross_entropy(logits[mask], x[mask])
        assert torch.isfinite(loss)

        # Generation: iterative decoding
        maskgit_model.eval()
        num_steps = 8

        # Start fully masked
        x_gen = torch.full((1, seq_length), mask_token_id, dtype=torch.long)

        with torch.no_grad():
            for step in range(num_steps):
                logits = maskgit_model(x_gen)

                # Find masked positions
                is_masked = x_gen == mask_token_id
                if not is_masked.any():
                    break

                # Sample for masked positions
                probs = torch.softmax(logits / 0.9, dim=-1)
                samples = torch.multinomial(probs.view(-1, output_vocab_size), num_samples=1).view(
                    1, seq_length
                )

                # Confidence scores
                confidence = probs.max(dim=-1).values

                # Unmask top-k confident predictions
                progress = step / max(num_steps, 1)
                # Cosine-decay schedule from 1.0 -> 0.0 over progress in [0, 1]
                ratio = 0.5 * (1 + math.cos(math.pi * progress))
                num_to_unmask = max(1, int((1 - ratio) * is_masked.sum().item()))

                masked_confidence = confidence.clone()
                masked_confidence[~is_masked] = -float("inf")
                _, top_indices = masked_confidence.view(-1).topk(num_to_unmask)

                x_gen.view(-1)[top_indices] = samples.view(-1)[top_indices]

        assert x_gen.min() >= 0


class TestD3PMEndToEnd:
    """Test D3PM diffusion training and sampling pipeline."""

    @pytest.fixture
    def d3pm_setup(self):
        """Create D3PM setup with small model."""
        num_classes = 64
        num_timesteps = 10

        model = MaskGIT(
            seq_length=32,
            vocab_size=num_classes,
            hidden_size=32,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
        )

        d3pm = D3PM(
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            schedule_type="cosine",
        )

        return model, d3pm, num_classes, num_timesteps

    def test_train_step(self, d3pm_setup):
        """Test D3PM forward noising and training loss computation."""
        model, d3pm, num_classes, num_timesteps = d3pm_setup
        batch_size = 2
        seq_length = 32

        # Training step: forward noising
        x_0 = torch.randint(0, num_classes, (batch_size, seq_length))
        t = torch.randint(1, num_timesteps, (batch_size,))

        x_t = d3pm.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
        assert x_t.min() >= 0
        assert x_t.max() < d3pm.effective_num_classes

        # Model prediction
        logits = model(x_t)
        output_vocab_size = logits.size(-1)

        # Compute cross-entropy loss (D3PM.compute_loss expects model with (x, t) interface)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, output_vocab_size), x_0.reshape(-1)
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_posterior_computation(self, d3pm_setup):
        """Test D3PM posterior distribution computation."""
        model, d3pm, num_classes, num_timesteps = d3pm_setup
        batch_size = 2
        seq_length = 32

        x_0 = torch.randint(0, num_classes, (batch_size, seq_length))
        t = torch.randint(2, num_timesteps, (batch_size,))  # t >= 2 for valid posterior

        x_t = d3pm.q_sample(x_0, t)

        # Compute posterior q(x_{t-1} | x_t, x_0)
        q_posterior = d3pm.q_posterior(x_0, x_t, t)

        # Posterior should be valid probability distributions
        assert q_posterior.shape == (batch_size, seq_length, d3pm.effective_num_classes)
        assert torch.isfinite(q_posterior).all()
        assert (q_posterior >= 0).all()
        # Each position should sum to ~1 (allowing small numerical error)
        assert torch.allclose(
            q_posterior.sum(dim=-1), torch.ones(batch_size, seq_length), atol=1e-5
        )


class TestFlowMatchingEndToEnd:
    """Test discrete flow matching training pipeline."""

    @pytest.fixture
    def flow_setup(self):
        """Create flow matching setup."""
        vocab_size = 64
        source = MaskedSourceDistribution(mask_token=vocab_size - 1)
        scheduler = PolynomialConvexScheduler(n=1.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        return source, path, vocab_size

    def test_training_step(self, flow_setup):
        """Test flow matching training step."""
        source, path, vocab_size = flow_setup
        batch_size = 2
        seq_length = 32

        # Sample source and target
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_length))
        x_0 = source.sample((batch_size, seq_length), x_1.device)

        # Sample time and interpolate
        t = torch.rand(batch_size)
        sample = path.sample(t, x_0, x_1)
        x_t = sample.x_t
        mask = sample.mask

        assert x_t.shape == x_1.shape
        # mask shape matches spatial dimensions, may be broadcast
        assert mask.numel() == x_1.numel() or mask.shape[0] == batch_size

        # At t=0, should be source; at t=1, should be target
        t_zero = torch.zeros(batch_size)
        sample_0 = path.sample(t_zero, x_0, x_1)
        assert (sample_0.x_t == x_0).all()

        t_one = torch.ones(batch_size)
        sample_1 = path.sample(t_one, x_0, x_1)
        assert (sample_1.x_t == x_1).all()


class TestRasterizationPipeline:
    """Test rasterization for spatial-to-sequence conversion."""

    @pytest.mark.parametrize(
        "curve_class",
        [HilbertCurve, ZOrderCurve],
    )
    def test_2d_round_trip_pipeline(self, curve_class):
        """Test 2D image to sequence and back (Hilbert/ZOrder with metadata)."""
        batch_size = 2
        height, width = 16, 16
        vocab_size = 64

        # Create "image" as 2D token grid with batch dim
        image_tokens = torch.randint(0, vocab_size, (batch_size, height, width))

        # Convert to sequence
        sequence, metadata = curve_class.spatial_to_sequence_2d(image_tokens)
        assert sequence.dim() == 2  # [batch, seq_len]
        assert sequence.size(0) == batch_size

        # Convert back to spatial
        reconstructed = curve_class.sequence_to_spatial_2d(sequence, metadata)
        assert reconstructed.shape == (batch_size, height, width)
        assert (reconstructed == image_tokens).all()

    @pytest.mark.parametrize(
        "curve_class",
        [HilbertCurve, ZOrderCurve],
    )
    def test_3d_round_trip_pipeline(self, curve_class):
        """Test 3D volume to sequence and back (Hilbert/ZOrder with metadata)."""
        batch_size = 2
        depth, height, width = 8, 8, 8
        vocab_size = 64

        # Create "volume" as 3D token grid with batch dim
        volume_tokens = torch.randint(0, vocab_size, (batch_size, depth, height, width))

        # Convert to sequence
        sequence, metadata = curve_class.spatial_to_sequence_3d(volume_tokens)
        assert sequence.dim() == 2  # [batch, seq_len]
        assert sequence.size(0) == batch_size

        # Convert back to spatial
        reconstructed = curve_class.sequence_to_spatial_3d(sequence, metadata)
        assert reconstructed.shape == (batch_size, depth, height, width)
        assert (reconstructed == volume_tokens).all()

    def test_raster_scan_2d_round_trip(self):
        """Test RasterScan 2D round-trip (simpler API without metadata)."""
        batch_size = 2
        height, width = 16, 16
        vocab_size = 64

        image_tokens = torch.randint(0, vocab_size, (batch_size, height, width))

        # RasterScan returns tensor directly, no metadata
        sequence = RasterScan.spatial_to_sequence_2d(image_tokens)
        assert sequence.dim() == 2  # [batch, seq_len]
        assert sequence.size(0) == batch_size

        # Reconstruct with explicit dims
        reconstructed = RasterScan.sequence_to_spatial_2d(sequence, height, width)
        assert reconstructed.shape == (batch_size, height, width)
        assert (reconstructed == image_tokens).all()

    def test_raster_scan_3d_round_trip(self):
        """Test RasterScan 3D round-trip (simpler API without metadata)."""
        batch_size = 2
        depth, height, width = 8, 8, 8
        vocab_size = 64

        volume_tokens = torch.randint(0, vocab_size, (batch_size, depth, height, width))

        # RasterScan returns tensor directly
        sequence = RasterScan.spatial_to_sequence_3d(volume_tokens)
        assert sequence.dim() == 2  # [batch, seq_len]
        assert sequence.size(0) == batch_size

        # Reconstruct with explicit dims
        reconstructed = RasterScan.sequence_to_spatial_3d(sequence, depth, height, width)
        assert reconstructed.shape == (batch_size, depth, height, width)
        assert (reconstructed == volume_tokens).all()

    def test_batch_rasterization_with_channels(self):
        """Test rasterization with batched data and channels."""
        batch_size = 4
        channels = 3
        height, width = 8, 8

        # Batch of images with channels
        images = torch.randn(batch_size, channels, height, width)

        # Convert to sequence
        sequence, metadata = HilbertCurve.spatial_to_sequence_2d(images)
        assert sequence.dim() == 3  # [batch, channels, seq_len]
        assert sequence.size(0) == batch_size
        assert sequence.size(1) == channels

        # Reconstruct
        reconstructed = HilbertCurve.sequence_to_spatial_2d(sequence, metadata)
        assert reconstructed.shape == images.shape
        assert torch.allclose(reconstructed, images)


class TestCrossComponentIntegration:
    """Test integration across different components."""

    def test_ar_with_rasterization(self):
        """Test AR model with rasterized spatial data."""
        batch_size = 2
        height, width = 8, 8
        vocab_size = 64
        seq_length = height * width

        # Create AR model
        model = AutoregressiveTransformer(
            seq_length=seq_length,
            vocab_size=vocab_size,
            hidden_size=32,
            depth=2,
            num_heads=2,
        )

        # Create spatial data and rasterize
        image = torch.randint(0, vocab_size, (batch_size, height, width))
        sequence, metadata = HilbertCurve.spatial_to_sequence_2d(image)

        # Forward pass
        logits = model(sequence)
        output_vocab_size = logits.size(-1)
        assert logits.shape == (batch_size, sequence.size(1), output_vocab_size)

        # Reconstruct predictions to spatial
        predicted_tokens = logits.argmax(dim=-1)
        predicted_image = HilbertCurve.sequence_to_spatial_2d(predicted_tokens, metadata)
        assert predicted_image.shape == (batch_size, height, width)

    def test_maskgit_with_rasterization(self):
        """Test MaskGIT model with rasterized spatial data."""
        batch_size = 2
        height, width = 8, 8
        vocab_size = 64
        seq_length = height * width

        model = MaskGIT(
            seq_length=seq_length,
            vocab_size=vocab_size,
            hidden_size=32,
            depth=2,
            num_heads=2,
        )

        mask_token_id = model.special_tokens.mask

        # Create and rasterize image
        image = torch.randint(0, vocab_size, (batch_size, height, width))

        # Mask center region in spatial domain
        masked_image = image.clone()
        masked_image[:, 2:6, 2:6] = mask_token_id
        masked_sequence, metadata = ZOrderCurve.spatial_to_sequence_2d(masked_image)

        # Forward pass
        logits = model(masked_sequence)
        output_vocab_size = logits.size(-1)
        assert logits.shape == (batch_size, masked_sequence.size(1), output_vocab_size)

        # Verify we can inpaint
        predicted_sequence = logits.argmax(dim=-1)
        inpainted_image = ZOrderCurve.sequence_to_spatial_2d(predicted_sequence, metadata)
        assert inpainted_image.shape == (batch_size, height, width)
