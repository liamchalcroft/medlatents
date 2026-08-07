"""End-to-end tests for D3PM."""

import pytest
import torch
import torch.nn.functional as F

from medlatents.configs import MODEL_CONFIGS
from medlatents.diffusion.d3pm import D3PM
from medlatents.networks import DiscreteDiT


class DeltaDenoiser(torch.nn.Module):
    """Always predicts a single clean token class."""

    def __init__(self, num_classes: int, target_token: int = 0):
        super().__init__()
        self.num_classes = num_classes
        self.target_token = target_token

    def forward(self, x, t, **kwargs):
        logits = torch.zeros(*x.shape, self.num_classes, device=x.device)
        logits[..., self.target_token] = 25.0
        return logits


def test_d3pm_forward_reverse():
    """Test D3PM forward and reverse diffusion process."""
    batch_size = 4
    seq_length = 64
    vocab_size = 512
    num_timesteps = 100

    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=num_timesteps,
        schedule_type="cosine",
        transition_type="absorbing",
    )

    x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Test forward diffusion at different timesteps
    timesteps = [0, 25, 50, 75, 99]
    for t in timesteps:
        t_tensor = torch.tensor([t] * batch_size)
        x_t = d3pm.q_sample(x_0, t_tensor)

        assert x_t.shape == x_0.shape, f"Shape mismatch at t={t}"
        assert x_t.min() >= 0 and x_t.max() < vocab_size + 1, f"Invalid values at t={t}"

    # Test reverse diffusion
    config = MODEL_CONFIGS["nano"]
    model = DiscreteDiT(
        seq_length=seq_length,
        vocab_size=vocab_size + 1,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
    )
    model.eval()

    with torch.no_grad():
        t = torch.tensor([99] * batch_size)
        x_t = d3pm.q_sample(x_0, t)
        t_tensor = torch.tensor([99] * batch_size)
        x_t_minus_1 = d3pm.p_sample(model, x_t, t_tensor)

        assert x_t_minus_1.shape == x_0.shape


def test_d3pm_reverse_uses_forward_kernel():
    """Reverse sampling should use the q(x_{t-1}|x_t, x0) kernel."""
    torch.manual_seed(0)
    num_classes = 4
    seq_len = 2
    batch_size = 4

    d3pm = D3PM(
        num_classes=num_classes,
        num_timesteps=10,
        schedule_type="linear",
        transition_type="uniform",
    )

    model = DeltaDenoiser(num_classes)

    x_0 = torch.zeros(batch_size, seq_len, dtype=torch.long)
    t = torch.full((batch_size,), 5, dtype=torch.long)
    x_t = d3pm.q_sample(x_0, t)

    with torch.no_grad():
        logits = model(x=x_t, t=t)
        p_x0 = F.softmax(logits, dim=-1)

        # Manually compute marginalized distribution
        p_tm1_expected = torch.zeros(
            batch_size, seq_len, d3pm.effective_num_classes, device=x_t.device
        )
        for x0_val in range(d3pm.effective_num_classes):
            x0_candidate = torch.full_like(x_t, x0_val)
            q_posterior = d3pm.q_posterior(x0_candidate, x_t, t)
            p_tm1_expected += p_x0[..., x0_val : x0_val + 1] * q_posterior

        # Since the model predicts x_0=0 with very high probability,
        # the marginalized distribution should match q(x_{t-1} | x_t, x_0=0)
        posterior = d3pm.q_posterior(torch.zeros_like(x_0), x_t, t)

        assert torch.allclose(p_tm1_expected, posterior, atol=1e-6), (
            "Marginalized distribution should match posterior when model is certain"
        )


def test_d3pm_loss():
    """Test D3PM loss computation."""
    batch_size = 4
    seq_length = 64
    vocab_size = 512

    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=100,
        schedule_type="cosine",
        transition_type="absorbing",
    )

    config = MODEL_CONFIGS["nano"]
    model = DiscreteDiT(
        seq_length=seq_length,
        vocab_size=vocab_size + 1,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
    )

    x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))

    loss_types = ["vb", "hybrid", "cross_entropy"]
    for loss_type in loss_types:
        loss = d3pm.compute_loss(model, x_0, loss_type=loss_type)

        assert loss.dim() == 0, f"Loss should be scalar for {loss_type}"
        assert loss.item() > 0, f"Loss should be positive for {loss_type}"
        assert not torch.isnan(loss), f"Loss is NaN for {loss_type}"


def test_d3pm_model_vocab_mismatch_raises():
    """D3PM should fail fast when model vocab doesn't include the mask token."""
    vocab_size = 32
    seq_length = 16

    config = MODEL_CONFIGS["nano"]
    model = DiscreteDiT(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=config["hidden_size"],
        depth=2,
        num_heads=2,
    )
    d3pm = D3PM(num_classes=vocab_size, num_timesteps=10, transition_type="absorbing")

    x_0 = torch.randint(0, vocab_size, (2, seq_length))

    with pytest.raises(ValueError, match="model.vocab_size"):
        _ = d3pm.compute_loss(model, x_0)


def test_diffusion_transition_types():
    """Test different transition matrix types."""
    batch_size = 4
    seq_length = 64
    vocab_size = 512

    x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))

    transition_types = ["absorbing", "uniform", "gaussian"]
    for trans_type in transition_types:
        d3pm = D3PM(
            num_classes=vocab_size,
            num_timesteps=100,
            schedule_type="cosine",
            transition_type=trans_type,
        )

        t = torch.tensor([50] * batch_size)
        x_t = d3pm.q_sample(x_0, t)

        assert x_t.shape == x_0.shape
        assert x_t.min() >= 0


def test_diffusion_schedules():
    """Test different noise schedules."""
    batch_size = 4
    seq_length = 64
    vocab_size = 512

    x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))

    schedules = ["linear", "cosine", "quadratic", "sigmoid"]
    for schedule in schedules:
        d3pm = D3PM(
            num_classes=vocab_size,
            num_timesteps=100,
            schedule_type=schedule,
            transition_type="absorbing",
        )

        t = torch.tensor([50] * batch_size)
        x_t = d3pm.q_sample(x_0, t)

        assert x_t.shape == x_0.shape


def test_sampling_integration():
    """Test diffusion sampling loop."""
    batch_size = 2
    seq_length = 64
    vocab_size = 512
    num_steps = 20

    config = MODEL_CONFIGS["nano"]
    model = DiscreteDiT(
        seq_length=seq_length,
        vocab_size=vocab_size + 1,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
    )
    model.eval()

    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=num_steps,
        schedule_type="cosine",
        transition_type="absorbing",
    )

    x = torch.randint(0, vocab_size + 1, (batch_size, seq_length))

    with torch.no_grad():
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t] * batch_size)
            x = d3pm.p_sample(model, x, t_tensor)

    assert x.min() >= 0 and x.max() < vocab_size


class TestD3PMDeterminism:
    """Test deterministic sampling for D3PM."""

    @pytest.fixture
    def d3pm(self):
        """Create D3PM instance for determinism tests."""
        return D3PM(
            num_classes=64,
            num_timesteps=10,
            schedule_type="cosine",
            transition_type="absorbing",
        )

    @pytest.fixture
    def model(self):
        """Create model for D3PM tests."""
        config = MODEL_CONFIGS["nano"]
        model = DiscreteDiT(
            seq_length=16,
            vocab_size=65,  # 64 + 1 for mask token
            hidden_size=config["hidden_size"],
            depth=2,
            num_heads=2,
        )
        model.eval()
        return model

    def test_q_sample_determinism(self, d3pm):
        """Test that forward diffusion is deterministic with fixed seed."""
        x_0 = torch.randint(0, 64, (4, 16))
        t = torch.tensor([5, 5, 5, 5])

        torch.manual_seed(42)
        x_t1 = d3pm.q_sample(x_0, t)

        torch.manual_seed(42)
        x_t2 = d3pm.q_sample(x_0, t)

        assert torch.equal(x_t1, x_t2), "Forward diffusion should be deterministic with same seed"

    def test_p_sample_determinism(self, d3pm, model):
        """Test that reverse sampling is deterministic with fixed seed."""
        x_t = torch.randint(0, 65, (2, 16))
        t = torch.tensor([5, 5])

        torch.manual_seed(42)
        x_tm1_1 = d3pm.p_sample(model, x_t, t)

        torch.manual_seed(42)
        x_tm1_2 = d3pm.p_sample(model, x_t, t)

        assert torch.equal(x_tm1_1, x_tm1_2), (
            "Reverse sampling should be deterministic with same seed"
        )

    def test_full_sample_determinism(self, d3pm, model):
        """Test that full sampling loop is deterministic with fixed seed."""
        shape = (2, 16)

        torch.manual_seed(42)
        samples1 = d3pm.sample(model, shape, temperature=1.0)

        torch.manual_seed(42)
        samples2 = d3pm.sample(model, shape, temperature=1.0)

        assert torch.equal(samples1, samples2), (
            "Full sampling should be deterministic with same seed"
        )

    def test_sample_different_with_different_seeds(self, d3pm, model):
        """Test that different seeds produce different samples."""
        shape = (2, 16)

        torch.manual_seed(42)
        samples1 = d3pm.sample(model, shape, temperature=1.0)

        torch.manual_seed(123)
        samples2 = d3pm.sample(model, shape, temperature=1.0)

        # Different seeds should (almost certainly) produce different samples
        assert not torch.equal(samples1, samples2), (
            "Different seeds should produce different samples"
        )

    def test_loss_determinism(self, d3pm, model):
        """Test that loss computation is deterministic with fixed seed."""
        x_0 = torch.randint(0, 64, (4, 16))

        torch.manual_seed(42)
        loss1 = d3pm.compute_loss(model, x_0, loss_type="hybrid")

        torch.manual_seed(42)
        loss2 = d3pm.compute_loss(model, x_0, loss_type="hybrid")

        assert torch.equal(loss1, loss2), "Loss should be deterministic with same seed"

    def test_uniform_transition_determinism(self):
        """Test determinism with uniform transition type."""
        d3pm_uniform = D3PM(
            num_classes=64,
            num_timesteps=10,
            schedule_type="cosine",
            transition_type="uniform",
        )

        x_0 = torch.randint(0, 64, (4, 16))
        t = torch.tensor([5, 5, 5, 5])

        torch.manual_seed(42)
        x_t1 = d3pm_uniform.q_sample(x_0, t)

        torch.manual_seed(42)
        x_t2 = d3pm_uniform.q_sample(x_0, t)

        assert torch.equal(x_t1, x_t2), "Uniform transition should be deterministic"

    def test_gaussian_transition_determinism(self):
        """Test determinism with Gaussian transition type."""
        d3pm_gaussian = D3PM(
            num_classes=64,
            num_timesteps=10,
            schedule_type="cosine",
            transition_type="gaussian",
        )

        x_0 = torch.randint(0, 64, (4, 16))
        t = torch.tensor([5, 5, 5, 5])

        torch.manual_seed(42)
        x_t1 = d3pm_gaussian.q_sample(x_0, t)

        torch.manual_seed(42)
        x_t2 = d3pm_gaussian.q_sample(x_0, t)

        assert torch.equal(x_t1, x_t2), "Gaussian transition should be deterministic"
