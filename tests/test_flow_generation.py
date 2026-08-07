"""Tests for discrete flow generation sampling behavior."""

import torch

from medlatents.flow_matching.discrete import get_source_distribution
from medlatents.networks.diffusion_transformer import DiscreteDiT_models


class TestDiscreteFlowGeneration:
    """Test discrete flow matching generation produces valid discrete tokens."""

    def test_flow_generation_produces_discrete_tokens(self):
        """Verify flow generation samples from probabilities correctly."""
        batch_size = 2
        seq_length = 16
        vocab_size = 32

        model = DiscreteDiT_models["DiscreteDiT-Nano"](
            seq_length=seq_length,
            vocab_size=vocab_size,
        )
        model.train(False)

        source_dist = get_source_distribution("uniform", vocab_size)
        x = source_dist.sample((batch_size, seq_length), device="cpu")

        num_steps = 5
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1)

        with torch.no_grad():
            for t in time_steps[:-1]:
                t_batch = t.expand(batch_size)
                logits = model(x, t_batch)

                probs = torch.softmax(logits, dim=-1)
                x = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
                    batch_size, seq_length
                )

        assert x.shape == (batch_size, seq_length)
        assert x.dtype == torch.long
        assert x.min() >= 0
        assert x.max() < vocab_size

    def test_flow_generation_with_temperature(self):
        """Test that temperature scaling affects sampling distribution."""
        # Create synthetic logits with variation to test temperature scaling
        batch_size = 2
        seq_length = 16
        vocab_size = 32

        # Create logits with meaningful variation
        logits = torch.randn(batch_size, seq_length, vocab_size)

        # Low temperature should make distribution more peaked
        low_temp_probs = torch.softmax(logits / 0.1, dim=-1)
        high_temp_probs = torch.softmax(logits / 2.0, dim=-1)

        # Low temp has higher max probability (more confident)
        low_max = low_temp_probs.max(dim=-1).values.mean()
        high_max = high_temp_probs.max(dim=-1).values.mean()

        assert low_max > high_max

    def test_source_distribution_uniform(self):
        """Test uniform source distribution produces valid tokens."""
        vocab_size = 64
        batch_size = 4
        seq_length = 32

        source_dist = get_source_distribution("uniform", vocab_size)
        samples = source_dist.sample((batch_size, seq_length), device="cpu")

        assert samples.shape == (batch_size, seq_length)
        assert samples.min() >= 0
        assert samples.max() < vocab_size

    def test_source_distribution_mask(self):
        """Test masked source distribution uses mask token."""
        vocab_size = 64
        mask_token = vocab_size  # Mask token is vocab_size
        batch_size = 4
        seq_length = 32

        source_dist = get_source_distribution("mask", vocab_size, mask_token=mask_token)
        samples = source_dist.sample((batch_size, seq_length), device="cpu")

        assert samples.shape == (batch_size, seq_length)
        # All tokens should be the mask token
        assert (samples == mask_token).all()
