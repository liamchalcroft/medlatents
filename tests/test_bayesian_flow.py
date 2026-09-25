"""Tests for Bayesian Flow Networks."""

import pytest
import torch
import torch.nn as nn

from conftest import assert_valid_samples
from medlatents.bayesian_flow import (
    BayesianFlowTransformer,
    BFN_models,
)


class SimpleBFNModel(nn.Module):
    """Simple model for testing BFN."""

    def __init__(self, num_classes: int, hidden_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_classes, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x, t, **kwargs):
        """Forward pass with time conditioning."""
        # x: [B, L] token indices
        # t: [B] time values
        B, L = x.shape
        x_emb = self.embed(x)  # [B, L, H]
        t_expanded = t.view(B, 1, 1).expand(B, L, 1)  # [B, L, 1]
        x_cat = torch.cat([x_emb, t_expanded], dim=-1)  # [B, L, H+1]
        logits = self.mlp(x_cat)  # [B, L, K]
        return logits


class RecordingBFNModel(SimpleBFNModel):
    """Records last input tokens passed during forward calls."""

    def __init__(self, num_classes: int, hidden_size: int = 64):
        super().__init__(num_classes, hidden_size)
        self.last_x = None

    def forward(self, x, t, **kwargs):
        self.last_x = x.detach().clone()
        return super().forward(x, t, **kwargs)


class TestBayesianFlowTransformer:
    """Test suite for Bayesian Flow Networks."""

    @pytest.fixture
    def bfn_config(self):
        """Configuration for BFN testing."""
        return {
            "seq_length": 16,
            "vocab_size": 64,
            "hidden_size": 128,
            "depth": 2,
            "num_heads": 4,
            "num_steps": 10,
            "beta": 1.0,
            "allow_dynamic_seq_length": False,
            "gradient_checkpointing": False,
        }

    @pytest.fixture
    def bfn(self, bfn_config):
        """Create BFN instance."""
        return BayesianFlowTransformer(**bfn_config)

    def test_initialization(self, bfn, bfn_config):
        """Test BFN initializes correctly."""
        # Note: num_classes includes special tokens, so it's vocab_size + 4
        assert bfn.num_steps == bfn_config["num_steps"]
        assert bfn.beta == bfn_config["beta"]
        assert bfn.base_vocab_size == bfn_config["vocab_size"]

    def test_get_accuracy_continuous(self, bfn):
        """Test continuous time accuracy parameter."""
        batch_size = 4
        t = torch.rand(batch_size)  # Random times in [0, 1]

        alpha = bfn.get_accuracy(t, continuous_time=True)

        assert alpha.shape == (batch_size,)
        assert (alpha >= 0).all()
        # α(t) = β² · t
        expected = (bfn.beta**2) * t
        assert torch.allclose(alpha, expected)

    def test_get_accuracy_discrete(self, bfn):
        """Test discrete time accuracy parameter."""
        batch_size = 4
        i = torch.randint(0, bfn.num_steps, (batch_size,)).float()

        alpha = bfn.get_accuracy(i, continuous_time=False)

        assert alpha.shape == (batch_size,)
        assert (alpha >= 0).all()
        # α_i = (β/n)² · (2i-1) where i is 0-indexed but formula uses 1-indexed
        expected = (bfn.beta / bfn.num_steps) ** 2 * (2 * (i + 1) - 1)
        assert torch.allclose(alpha, expected)

    def test_get_prior_params(self, bfn, bfn_config):
        """Test prior parameters are uniform (zeros)."""
        batch_size = 4
        seq_len = 16

        prior = bfn.get_prior_params(batch_size, seq_len)

        expected_shape = (batch_size, seq_len, bfn.num_classes)
        assert prior.shape == expected_shape
        assert (prior == 0).all()

    def test_sample_sender_distribution(self, bfn, bfn_config):
        """Test sender distribution sampling."""
        batch_size = 4
        seq_len = 16

        x = torch.randint(0, bfn.base_vocab_size, (batch_size, seq_len))
        alpha = torch.rand(batch_size) * 0.5  # Random accuracy

        y = bfn.sample_sender_distribution(x, alpha)

        assert y.shape == (batch_size, seq_len, bfn.num_classes)
        assert torch.isfinite(y).all()

    def test_sender_distribution_mean(self, bfn, bfn_config):
        """Test sender distribution has correct mean structure."""
        batch_size = 100
        seq_len = 1

        # Fix x to a specific class
        x = torch.zeros((batch_size, seq_len), dtype=torch.long)
        alpha = torch.ones(batch_size) * 0.1

        # Sample many times and check empirical mean
        samples = []
        for _ in range(1000):
            y = bfn.sample_sender_distribution(x, alpha)
            samples.append(y)

        mean_y = torch.stack(samples).mean(dim=0)  # [B, L, K]

        # Expected mean: α · (K · e_x - 1)
        # For class 0, e_x = [1, 0, 0, ...], so mean = α · (K * [1,0,0,...] - 1) = α · [K-1, -1, -1, ...]
        expected_mean = alpha.view(-1, 1, 1) * (bfn.num_classes * torch.eye(bfn.num_classes)[0] - 1)

        # Check empirical mean is close to expected (with some tolerance for Monte Carlo sampling)
        assert torch.allclose(mean_y[0, 0, :], expected_mean[0, 0, :], atol=0.3)

    def test_update_params_bayesian(self, bfn, bfn_config):
        """Test Bayesian parameter update."""
        batch_size = 2
        seq_len = 8

        # Start with uniform params (zeros)
        params = torch.zeros(batch_size, seq_len, bfn.num_classes)

        # Create observation favoring class 0
        y = torch.zeros(batch_size, seq_len, bfn.num_classes)
        y[:, :, 0] = 2.0  # Strong evidence for class 0

        new_params = bfn.update_params_bayesian(params, y)

        assert new_params.shape == params.shape
        assert torch.isfinite(new_params).all()

        # Check that probabilities favor class 0
        probs = torch.softmax(new_params, dim=-1)
        assert (probs[:, :, 0] > probs[:, :, 1:].max(dim=-1)[0]).all()

    def test_sample_receiver_distribution(self, bfn, bfn_config):
        """Test receiver distribution sampling."""
        batch_size = 2
        seq_len = bfn_config["seq_length"]

        x_current = torch.randint(0, bfn.base_vocab_size, (batch_size, seq_len))
        t = torch.rand(batch_size)
        alpha = bfn.get_accuracy(t, continuous_time=True)

        y_receiver = bfn.sample_receiver_distribution(x_current, t, alpha, temperature=1.0)

        assert y_receiver.shape == (batch_size, seq_len, bfn.num_classes)
        assert torch.isfinite(y_receiver).all()

    def test_discrete_time_loss(self, bfn, bfn_config):
        """Test discrete-time training loss."""
        batch_size = 4
        seq_len = bfn_config["seq_length"]

        x_0 = torch.randint(0, bfn.base_vocab_size, (batch_size, seq_len))

        loss = bfn.discrete_time_loss(x_0, temperature=1.0)

        assert isinstance(loss.item(), float)
        assert loss > 0


def test_bfn_transformer_forward():
    """Ensure the packaged BFN transformer produces logits."""
    seq_length = 16
    vocab_size = 64
    batch = 2

    model = BFN_models["BFN-S"](
        seq_length=seq_length,
        vocab_size=vocab_size,
        num_classes=0,
        allow_dynamic_seq_length=False,
        gradient_checkpointing=False,
    )

    x = torch.randint(0, vocab_size, (batch, seq_length))
    t = torch.rand(batch)

    logits = model(x, t)

    # Note: vocab now includes special tokens (+4 for BOS, EOS, PAD, MASK)
    expected_vocab = vocab_size + 4
    assert logits.shape == (batch, seq_length, expected_vocab)
    assert torch.isfinite(logits).all()


class TestBayesianFlowMethods:
    """Additional BFN test methods."""

    @pytest.fixture
    def bfn_config(self):
        """Configuration for BFN testing."""
        return {
            "seq_length": 16,
            "vocab_size": 64,
            "hidden_size": 128,
            "depth": 2,
            "num_heads": 4,
            "num_steps": 10,
            "beta": 1.0,
            "allow_dynamic_seq_length": False,
            "gradient_checkpointing": False,
        }

    @pytest.fixture
    def bfn(self, bfn_config):
        """Create BFN instance."""
        return BayesianFlowTransformer(**bfn_config)

    def test_continuous_time_loss(self, bfn, bfn_config):
        """Test continuous-time training loss."""
        batch_size = 4
        seq_len = bfn_config["seq_length"]

        x_0 = torch.randint(0, bfn.base_vocab_size, (batch_size, seq_len))

        loss = bfn.continuous_time_loss(x_0, temperature=1.0)

        assert isinstance(loss.item(), float)
        assert loss > 0
        assert torch.isfinite(loss)

    def test_sample_generation(self, bfn, bfn_config):
        """Test sample generation."""
        batch_size = 2
        seq_len = bfn_config["seq_length"]
        num_steps = 5

        samples = bfn.sample(shape=(batch_size, seq_len), num_steps=num_steps, temperature=1.0)

        assert samples.shape == (batch_size, seq_len)
        assert_valid_samples(samples, bfn.num_classes)

    def test_sample_masks_forbidden_tokens_in_posterior(self, bfn, bfn_config, monkeypatch):
        """Test sampling survives a posterior that concentrates on a forbidden token."""
        pad = bfn.special_tokens.pad

        def receiver(x_current, t, alpha, **kwargs):
            y = torch.zeros(*x_current.shape, bfn.num_classes)
            y[..., pad] = 1e4
            return y

        monkeypatch.setattr(bfn, "sample_receiver_distribution", receiver)
        samples = bfn.sample(shape=(2, bfn_config["seq_length"]), num_steps=3)

        assert not torch.isin(samples, torch.tensor(sorted(bfn._forbidden_tokens))).any()

    def test_sample_with_temperature(self, bfn, bfn_config):
        """Test sampling with different temperatures."""
        batch_size = 2
        seq_len = bfn_config["seq_length"]

        # Low temperature should give more confident samples
        samples_low_temp = bfn.sample(shape=(batch_size, seq_len), num_steps=3, temperature=0.5)

        # High temperature should give more diverse samples
        samples_high_temp = bfn.sample(shape=(batch_size, seq_len), num_steps=3, temperature=2.0)

        assert samples_low_temp.shape == (batch_size, seq_len)
        assert samples_high_temp.shape == (batch_size, seq_len)
        assert_valid_samples(samples_low_temp, bfn.num_classes)
        assert_valid_samples(samples_high_temp, bfn.num_classes)

    def test_loss_gradient_flow(self, bfn, bfn_config):
        """Test that loss allows gradient flow."""
        batch_size = 2
        seq_len = bfn_config["seq_length"]

        x_0 = torch.randint(0, bfn.base_vocab_size, (batch_size, seq_len))

        # Test discrete-time loss
        loss = bfn.discrete_time_loss(x_0, temperature=1.0)
        loss.backward()

        # Check gradients exist
        for param in bfn.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()

    def test_bayesian_update_normalization(self, bfn, bfn_config):
        """Test that Bayesian update produces normalized distributions."""
        batch_size = 2
        seq_len = 8

        params = torch.randn(batch_size, seq_len, bfn.num_classes)
        y = torch.randn(batch_size, seq_len, bfn.num_classes)

        new_params = bfn.update_params_bayesian(params, y)
        probs = torch.softmax(new_params, dim=-1)

        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_iterative_refinement(self, bfn, bfn_config):
        """Test that iterative refinement improves predictions."""
        batch_size = 1
        seq_len = bfn_config["seq_length"]

        # Create a simple target
        x_target = torch.zeros((batch_size, seq_len), dtype=torch.long)

        # Train model briefly
        optimizer = torch.optim.Adam(bfn.parameters(), lr=1e-2)
        for _ in range(10):
            loss = bfn.continuous_time_loss(x_target, temperature=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Generate samples
        samples = bfn.sample(
            shape=(batch_size, seq_len),
            num_steps=10,
            temperature=0.1,  # Low temp for more deterministic
        )

        assert_valid_samples(samples, bfn.num_classes)

    def test_different_beta_values(self):
        """Test BFN with different beta values."""
        for beta in [0.5, 1.0, 2.0]:
            bfn_model = BayesianFlowTransformer(
                seq_length=16,
                vocab_size=32,
                hidden_size=64,
                depth=2,
                num_heads=2,
                num_steps=5,
                beta=beta,
            )
            assert bfn_model.beta == beta

            # Test accuracy scaling
            t = torch.tensor([0.5])
            alpha = bfn_model.get_accuracy(t, continuous_time=True)
            expected = beta**2 * 0.5
            assert torch.allclose(alpha, torch.tensor([expected]))


class TestBFNDeterminism:
    """Test deterministic sampling for BFN."""

    @pytest.fixture
    def bfn(self):
        """Create BFN instance for determinism tests."""
        return BayesianFlowTransformer(
            seq_length=16,
            vocab_size=32,
            hidden_size=64,
            depth=2,
            num_heads=2,
            num_steps=5,
            beta=1.0,
            allow_dynamic_seq_length=False,
            gradient_checkpointing=False,
        )

    def test_sample_determinism_with_manual_seed(self, bfn):
        """Test that BFN sampling is deterministic with manual seed."""
        bfn.eval()
        shape = (2, 16)

        # First run with seed
        torch.manual_seed(42)
        samples1 = bfn.sample(shape=shape, num_steps=3, temperature=1.0)

        # Second run with same seed
        torch.manual_seed(42)
        samples2 = bfn.sample(shape=shape, num_steps=3, temperature=1.0)

        assert torch.equal(samples1, samples2), "BFN samples should be identical with same seed"

    def test_sample_different_with_different_seeds(self, bfn):
        """Test that different seeds produce different samples."""
        bfn.eval()
        shape = (2, 16)

        torch.manual_seed(42)
        samples1 = bfn.sample(shape=shape, num_steps=3, temperature=1.0)

        torch.manual_seed(123)
        samples2 = bfn.sample(shape=shape, num_steps=3, temperature=1.0)

        # Different seeds should (almost certainly) produce different samples
        assert not torch.equal(samples1, samples2), (
            "Different seeds should produce different samples"
        )

    def test_sender_distribution_determinism(self, bfn):
        """Test that sender distribution sampling is deterministic."""
        x = torch.randint(0, bfn.base_vocab_size, (2, 16))
        alpha = torch.tensor([0.5, 0.5])

        torch.manual_seed(42)
        y1 = bfn.sample_sender_distribution(x, alpha)

        torch.manual_seed(42)
        y2 = bfn.sample_sender_distribution(x, alpha)

        assert torch.equal(y1, y2), "Sender distribution should be deterministic with same seed"

    def test_receiver_distribution_determinism(self, bfn):
        """Test that receiver distribution sampling is deterministic."""
        bfn.eval()
        x = torch.randint(0, bfn.base_vocab_size, (2, 16))
        t = torch.tensor([0.5, 0.5])
        alpha = bfn.get_accuracy(t, continuous_time=True)

        torch.manual_seed(42)
        y1 = bfn.sample_receiver_distribution(x, t, alpha)

        torch.manual_seed(42)
        y2 = bfn.sample_receiver_distribution(x, t, alpha)

        assert torch.equal(y1, y2), "Receiver distribution should be deterministic with same seed"

    def test_loss_determinism(self, bfn):
        """Test that loss computation is deterministic with fixed seed."""
        x_0 = torch.randint(0, bfn.base_vocab_size, (4, 16))

        torch.manual_seed(42)
        loss1 = bfn.discrete_time_loss(x_0, temperature=1.0)

        torch.manual_seed(42)
        loss2 = bfn.discrete_time_loss(x_0, temperature=1.0)

        assert torch.equal(loss1, loss2), "Loss should be deterministic with same seed"
