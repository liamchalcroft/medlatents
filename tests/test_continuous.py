"""Tests for continuous latent diffusion and flow support."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medlatents.data.tokenized_datasets import ContinuousLatentDataset
from medlatents.diffusion.continuous import ContinuousGaussianDiffusion
from medlatents.flow_matching.continuous import RectifiedFlow
from medlatents.generation import ContinuousLatentGenerator
from medlatents.networks import ContinuousDiT
from medlatents.training import ContinuousLatentTrainer


class DummyContinuousTokenizer(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_shape = (latent_dim,)

    def encode_latents(self, x, **_):
        batch = x.shape[0]
        return torch.ones((batch, self.latent_dim))

    def decode_latents(self, latents, **_):
        return latents


class DummyDiffusionModel(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x, t, y=None):
        if x.dim() > 2:
            x_flat = x.flatten(start_dim=1)
        else:
            x_flat = x
        out = self.net(x_flat)
        return out.view_as(x)


class DummyFlowModel(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x, t, y=None):
        if x.dim() > 2:
            x_flat = x.flatten(start_dim=1)
        else:
            x_flat = x
        out = self.net(x_flat)
        return out.view_as(x)


def test_continuous_latent_dataset(tmp_path):
    data = np.random.randn(4, 4, 4)
    file_path = tmp_path / "volume.npy"
    np.save(file_path, data)

    tokenizer = DummyContinuousTokenizer(latent_dim=6)
    dataset = ContinuousLatentDataset(
        str(tmp_path / "*.npy"),
        tokenizer=tokenizer,
        use_cache=False,
        normalize=False,
        flatten=True,
    )

    latents = dataset[0]
    assert latents.shape == (6,)


def test_continuous_diffusion_loss():
    diffusion = ContinuousGaussianDiffusion(num_timesteps=10, schedule_type="linear")
    model = DummyDiffusionModel(latent_dim=8)
    batch = torch.randn(2, 8)

    loss = diffusion.compute_loss(model, batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_continuous_flow_loss():
    flow = RectifiedFlow(latent_shape=(8,))
    model = DummyFlowModel(latent_dim=8)
    batch = torch.randn(2, 8)

    loss = flow.compute_loss(model, batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_continuous_trainer_single_epoch():
    latent_dim = 8
    data = torch.randn(8, latent_dim)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=4)

    model = DummyDiffusionModel(latent_dim)
    args = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        use_wandb=False,
        seed=42,
        diffusion_steps=10,
        diffusion_schedule="linear",
        prediction_type="epsilon",
        epochs=1,
        lr_warmup_epochs=1,
        min_lr=0.0,
        grad_clip=1.0,
    )

    trainer = ContinuousLatentTrainer(
        model_type="diffusion",
        model=model,
        train_loader=loader,
        val_loader=None,
        args=args,
        use_ema=False,
    )

    trainer.train_epoch(0)


def test_continuous_generator_shapes():
    latent_dim = 4
    model = DummyDiffusionModel(latent_dim)
    tokenizer = DummyContinuousTokenizer(latent_dim=latent_dim)

    diffusion = ContinuousGaussianDiffusion(num_timesteps=5)
    generator = ContinuousLatentGenerator(
        model_type="diffusion",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        diffusion=diffusion,
    )

    latents = generator.generate(num_samples=2, num_steps=3)
    assert latents.shape == (2, latent_dim)


def test_continuous_generator_flow():
    latent_dim = 4
    model = DummyFlowModel(latent_dim)
    flow = RectifiedFlow(latent_shape=(latent_dim,), device=torch.device("cpu"))

    generator = ContinuousLatentGenerator(
        model_type="flow",
        model=model,
        tokenizer=None,
        device=torch.device("cpu"),
        flow=flow,
    )

    latents = generator.generate(num_samples=2, latent_shape=(latent_dim,), num_steps=4)
    assert latents.shape == (2, latent_dim)


def test_continuous_dit_forward():
    seq_len, channels = 6, 3
    model = ContinuousDiT(
        seq_length=seq_len,
        in_channels=channels,
        hidden_size=48,
        depth=2,
        num_heads=6,
        class_dropout_prob=0.0,
        num_classes=0,
        gradient_checkpointing=False,
    )

    x = torch.randn(2, seq_len, channels)
    t = torch.randint(0, 10, (2,))

    output = model(x, t)
    assert output.shape == (2, seq_len, channels)


def test_continuous_dit_forward_with_cfg():
    seq_len, channels = 4, 2
    model = ContinuousDiT(
        seq_length=seq_len,
        in_channels=channels,
        hidden_size=32,
        depth=2,
        num_heads=4,
        class_dropout_prob=0.5,
        num_classes=3,
        gradient_checkpointing=False,
    )

    batch = 2
    x = torch.randn(batch * 2, seq_len, channels)
    t = torch.randint(0, 10, (batch * 2,))
    y = torch.randint(0, 3, (batch,))

    guided = model.forward_with_cfg(x, t, y, cfg_scale=1.2)
    assert guided.shape == (batch, seq_len, channels)


class TestContinuousDiffusionDeterminism:
    """Test deterministic sampling for continuous Gaussian diffusion."""

    def test_q_sample_determinism(self):
        """Forward diffusion should be deterministic with fixed seed."""
        diffusion = ContinuousGaussianDiffusion(num_timesteps=10, schedule_type="linear")
        x_start = torch.randn(2, 8)
        t = torch.tensor([5, 5])

        torch.manual_seed(42)
        x_t1 = diffusion.q_sample(x_start, t)

        torch.manual_seed(42)
        x_t2 = diffusion.q_sample(x_start, t)

        assert torch.equal(x_t1, x_t2), "Forward diffusion should be deterministic"

    def test_sample_determinism(self):
        """Full sampling loop should be deterministic with fixed seed."""
        diffusion = ContinuousGaussianDiffusion(num_timesteps=10, schedule_type="linear")
        model = DummyDiffusionModel(latent_dim=8)
        model.eval()
        shape = (2, 8)

        torch.manual_seed(42)
        samples1 = diffusion.sample(model, shape, num_inference_steps=5)

        torch.manual_seed(42)
        samples2 = diffusion.sample(model, shape, num_inference_steps=5)

        assert torch.equal(samples1, samples2), "Sampling should be deterministic"

    def test_sample_with_initial_noise_determinism(self):
        """Sampling with explicit initial noise should be deterministic."""
        diffusion = ContinuousGaussianDiffusion(num_timesteps=10, schedule_type="linear")
        model = DummyDiffusionModel(latent_dim=8)
        model.eval()
        shape = (2, 8)

        # Same initial noise should give same results
        initial_noise = torch.randn(shape)

        torch.manual_seed(42)
        samples1 = diffusion.sample(
            model, shape, num_inference_steps=5, initial_noise=initial_noise
        )

        torch.manual_seed(42)
        samples2 = diffusion.sample(
            model, shape, num_inference_steps=5, initial_noise=initial_noise
        )

        assert torch.equal(samples1, samples2), "Sampling with same initial noise should match"

    def test_loss_determinism(self):
        """Loss computation should be deterministic with fixed seed."""
        diffusion = ContinuousGaussianDiffusion(num_timesteps=10, schedule_type="linear")
        model = DummyDiffusionModel(latent_dim=8)
        x_start = torch.randn(4, 8)

        torch.manual_seed(42)
        loss1 = diffusion.compute_loss(model, x_start)

        torch.manual_seed(42)
        loss2 = diffusion.compute_loss(model, x_start)

        assert torch.equal(loss1, loss2), "Loss should be deterministic"

    def test_different_seeds_different_samples(self):
        """Different seeds should produce different samples."""
        diffusion = ContinuousGaussianDiffusion(num_timesteps=10, schedule_type="linear")
        model = DummyDiffusionModel(latent_dim=8)
        model.eval()
        shape = (2, 8)

        torch.manual_seed(42)
        samples1 = diffusion.sample(model, shape, num_inference_steps=5)

        torch.manual_seed(123)
        samples2 = diffusion.sample(model, shape, num_inference_steps=5)

        assert not torch.equal(samples1, samples2), "Different seeds should give different samples"


class TestRectifiedFlowDeterminism:
    """Test deterministic sampling for rectified flow."""

    def test_sample_base_determinism(self):
        """Base distribution sampling should be deterministic."""
        flow = RectifiedFlow(latent_shape=(8,))

        torch.manual_seed(42)
        x1 = flow.sample_base(4)

        torch.manual_seed(42)
        x2 = flow.sample_base(4)

        assert torch.equal(x1, x2), "Base sampling should be deterministic"

    def test_sample_path_determinism(self):
        """Path sampling should be deterministic (no randomness involved)."""
        flow = RectifiedFlow(latent_shape=(8,))
        x0 = torch.randn(2, 8)
        x1 = torch.randn(2, 8)
        t = torch.tensor([0.3, 0.7])

        x_t1, v1 = flow.sample_path(x0, x1, t)
        x_t2, v2 = flow.sample_path(x0, x1, t)

        assert torch.equal(x_t1, x_t2), "Path interpolation should be deterministic"
        assert torch.equal(v1, v2), "Velocity should be deterministic"

    def test_sample_euler_determinism(self):
        """Euler sampling should be deterministic with fixed seed."""
        flow = RectifiedFlow(latent_shape=(8,), device=torch.device("cpu"))
        model = DummyFlowModel(latent_dim=8)
        model.eval()

        torch.manual_seed(42)
        samples1 = flow.sample(model, batch_size=2, num_steps=5, solver="euler")

        torch.manual_seed(42)
        samples2 = flow.sample(model, batch_size=2, num_steps=5, solver="euler")

        assert torch.equal(samples1, samples2), "Euler sampling should be deterministic"

    def test_sample_heun_determinism(self):
        """Heun sampling should be deterministic with fixed seed."""
        flow = RectifiedFlow(latent_shape=(8,), device=torch.device("cpu"))
        model = DummyFlowModel(latent_dim=8)
        model.eval()

        torch.manual_seed(42)
        samples1 = flow.sample(model, batch_size=2, num_steps=5, solver="heun")

        torch.manual_seed(42)
        samples2 = flow.sample(model, batch_size=2, num_steps=5, solver="heun")

        assert torch.equal(samples1, samples2), "Heun sampling should be deterministic"

    def test_sample_rk4_determinism(self):
        """RK4 sampling should be deterministic with fixed seed."""
        flow = RectifiedFlow(latent_shape=(8,), device=torch.device("cpu"))
        model = DummyFlowModel(latent_dim=8)
        model.eval()

        torch.manual_seed(42)
        samples1 = flow.sample(model, batch_size=2, num_steps=5, solver="rk4")

        torch.manual_seed(42)
        samples2 = flow.sample(model, batch_size=2, num_steps=5, solver="rk4")

        assert torch.equal(samples1, samples2), "RK4 sampling should be deterministic"

    def test_sample_with_initial_state_determinism(self):
        """Sampling with explicit initial state should be deterministic."""
        flow = RectifiedFlow(latent_shape=(8,), device=torch.device("cpu"))
        model = DummyFlowModel(latent_dim=8)
        model.eval()

        initial_state = torch.randn(2, 8)

        samples1 = flow.sample(model, batch_size=2, num_steps=5, initial_state=initial_state)
        samples2 = flow.sample(model, batch_size=2, num_steps=5, initial_state=initial_state)

        # No randomness when initial_state is provided, should be identical
        assert torch.equal(samples1, samples2), "Sampling with same initial state should match"

    def test_loss_determinism(self):
        """Loss computation should be deterministic with fixed seed."""
        flow = RectifiedFlow(latent_shape=(8,))
        model = DummyFlowModel(latent_dim=8)
        x1 = torch.randn(4, 8)

        torch.manual_seed(42)
        loss1 = flow.compute_loss(model, x1)

        torch.manual_seed(42)
        loss2 = flow.compute_loss(model, x1)

        assert torch.equal(loss1, loss2), "Loss should be deterministic"

    def test_different_seeds_different_samples(self):
        """Different seeds should produce different samples."""
        flow = RectifiedFlow(latent_shape=(8,), device=torch.device("cpu"))
        model = DummyFlowModel(latent_dim=8)
        model.eval()

        torch.manual_seed(42)
        samples1 = flow.sample(model, batch_size=2, num_steps=5)

        torch.manual_seed(123)
        samples2 = flow.sample(model, batch_size=2, num_steps=5)

        assert not torch.equal(samples1, samples2), "Different seeds should give different samples"
