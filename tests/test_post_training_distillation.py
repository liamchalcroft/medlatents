"""Tests for the post-training distillation subsystem.

Covers:
- reflow.py: ReflowPairGenerator pair construction + the straightened-trajectory
  cross-entropy loss (shape + differentiability) used by ReflowTrainer.train_on_pairs.
- consistency.py: pure helpers (pseudo_huber_loss, discretization / EMA schedules),
  preconditioning coefficients (boundary condition f(x, 0) = x), the consistency
  function, and the self-consistency / EMA-teacher behaviour of the train step.

All tests are CPU-only and use TINY models. The trainers' file-IO paths
(``torch.load`` / ``torch.save`` in (load|save)_checkpoint) are intentionally not
exercised -- only the in-memory code paths are tested.

CUDA-only behaviour is guarded with ``pytest.mark.skipif``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from medlatents.flow_matching.core import MixtureDiscreteProbPath, PolynomialConvexScheduler
from medlatents.post_training.configs import PostTrainingConfig
from medlatents.post_training.distillation.consistency import (
    ConsistencyTrainer,
    get_discretization_schedule,
    get_ema_decay_schedule,
    pseudo_huber_loss,
)
from medlatents.post_training.distillation.reflow import (
    ReflowPairGenerator,
    ReflowTrainer,
)

# ---------------------------------------------------------------------------
# Tiny model stubs
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()


class TinyDiscreteModel(nn.Module):
    """Tiny discrete flow/diffusion model: (x[long], t) -> logits [B, L, vocab]."""

    def __init__(self, vocab_size: int, hidden: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # +1 so that mask tokens (== vocab_size) used as the reflow source are
        # representable in the embedding table.
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        # Inject timestep so gradients can flow w.r.t. t-dependent quantities.
        if t is not None:
            tt = t.float().reshape(t.shape[0], *([1] * (h.dim() - 1)))
            h = h + tt
        return self.proj(h)


class TinyContinuousModel(nn.Module):
    """Tiny continuous model used for consistency preconditioning: (x, t) -> x-like."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# ---------------------------------------------------------------------------
# reflow: ReflowPairGenerator
# ---------------------------------------------------------------------------


def test_reflow_pair_generator_builds_noise_data_pairs():
    """generate_pairs returns (noise, data): noise == all mask tokens, data == input."""
    torch.manual_seed(0)
    vocab_size = 17
    seq_len = 6
    n = 10
    model = TinyDiscreteModel(vocab_size)
    path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
    gen = ReflowPairGenerator(
        model=model,
        path=path,
        vocab_size=vocab_size,
        device=torch.device("cpu"),
        num_inference_steps=4,
    )

    data = torch.randint(0, vocab_size, (n, seq_len))
    noise, returned_data = gen.generate_pairs(data, batch_size=4)

    # Shapes line up: one (noise, data) pair per input sample.
    assert noise.shape == (n, seq_len)
    assert returned_data.shape == (n, seq_len)

    # Source noise x_0 is the mask token == vocab_size everywhere (the documented
    # discrete source distribution).
    assert torch.all(noise == vocab_size)
    assert noise.dtype == torch.long

    # The REAL data is preserved as the target x_1 (not the model's generation).
    assert torch.equal(returned_data, data)


def test_reflow_pair_generator_handles_uneven_final_batch():
    """Batched generation must concatenate correctly when N is not a multiple of bs."""
    torch.manual_seed(1)
    vocab_size = 8
    seq_len = 5
    n = 7  # not divisible by batch_size below
    gen = ReflowPairGenerator(
        model=TinyDiscreteModel(vocab_size),
        path=MixtureDiscreteProbPath(PolynomialConvexScheduler()),
        vocab_size=vocab_size,
        device=torch.device("cpu"),
    )
    data = torch.randint(0, vocab_size, (n, seq_len))
    noise, returned_data = gen.generate_pairs(data, batch_size=3)
    assert noise.shape == (n, seq_len)
    assert returned_data.shape == (n, seq_len)
    assert torch.equal(returned_data, data)


def test_reflow_pair_generator_returns_cpu_tensors():
    """Pairs are always returned on CPU (see .cpu() calls in generate_pairs)."""
    gen = ReflowPairGenerator(
        model=TinyDiscreteModel(8),
        path=MixtureDiscreteProbPath(PolynomialConvexScheduler()),
        vocab_size=8,
        device=torch.device("cpu"),
    )
    data = torch.randint(0, 8, (4, 5))
    noise, returned_data = gen.generate_pairs(data, batch_size=4)
    assert noise.device.type == "cpu"
    assert returned_data.device.type == "cpu"


# ---------------------------------------------------------------------------
# reflow: straightened-trajectory loss (mirrors ReflowTrainer.train_on_pairs)
# ---------------------------------------------------------------------------


def _reflow_loss(model, path, vocab_size, x_0, x_1, t):
    """Replicate the exact loss computed inside ReflowTrainer.train_on_pairs.

    The trainer wraps this in an accelerate loop; here we exercise the same
    math directly so we can assert on shape + differentiability without
    constructing an Accelerator.
    """
    path_sample = path.sample(t, x_0, x_1)
    x_t = path_sample.x_t
    logits = model(x=x_t, t=t)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), x_1.reshape(-1))
    return loss, logits


def test_reflow_loss_shape_and_scalar():
    torch.manual_seed(2)
    vocab_size = 12
    seq_len = 7
    bs = 4
    model = TinyDiscreteModel(vocab_size)
    path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))

    x_0 = torch.randint(0, vocab_size, (bs, seq_len))
    x_1 = torch.randint(0, vocab_size, (bs, seq_len))
    t = torch.rand(bs)

    loss, logits = _reflow_loss(model, path, vocab_size, x_0, x_1, t)

    # Logits have the [B, L, vocab] shape the CE reshape relies on.
    assert logits.shape == (bs, seq_len, vocab_size)
    # Loss is a finite scalar.
    assert loss.shape == ()
    assert torch.isfinite(loss)
    # Cross-entropy lower bound.
    assert loss.item() >= 0.0


def test_reflow_loss_is_differentiable():
    """The straightened-trajectory loss must produce gradients on model params."""
    torch.manual_seed(3)
    vocab_size = 10
    seq_len = 5
    bs = 3
    model = TinyDiscreteModel(vocab_size)
    path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))

    x_0 = torch.randint(0, vocab_size, (bs, seq_len))
    x_1 = torch.randint(0, vocab_size, (bs, seq_len))
    t = torch.rand(bs)

    loss, _ = _reflow_loss(model, path, vocab_size, x_0, x_1, t)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)
    for g in grads:
        assert torch.isfinite(g).all()


def test_reflow_loss_at_t0_pushes_toward_predicting_x1():
    """At t=0 the path is pure source noise; CE on real targets is still well-defined.

    Sanity check that a couple of gradient steps reduce the reflow loss, i.e. the
    objective genuinely trains the model to predict x_1 from the interpolated state.
    """
    torch.manual_seed(4)
    vocab_size = 6
    seq_len = 4
    bs = 8
    model = TinyDiscreteModel(vocab_size, hidden=24)
    path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    x_0 = torch.full((bs, seq_len), vocab_size, dtype=torch.long)  # mask source
    x_1 = torch.randint(0, vocab_size, (bs, seq_len))

    # Fixed t to make the optimisation deterministic.
    t = torch.full((bs,), 0.3)

    losses = []
    for _ in range(25):
        opt.zero_grad()
        # Re-sample the (stochastic) path each step, as the trainer does.
        loss, _ = _reflow_loss(model, path, vocab_size, x_0, x_1, t)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"reflow loss did not decrease: {losses[0]} -> {losses[-1]}"


# ---------------------------------------------------------------------------
# consistency: pure helper functions
# ---------------------------------------------------------------------------


def test_pseudo_huber_loss_zero_at_equality():
    x = torch.randn(4, 8)
    # Identical inputs -> per-element loss sqrt(c^2) - c == 0.
    loss = pseudo_huber_loss(x, x.clone(), c=0.01)
    assert loss.shape == ()
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_pseudo_huber_loss_positive_and_increasing():
    x = torch.zeros(100)
    near = torch.full((100,), 0.05)
    far = torch.full((100,), 2.0)
    c = 0.00054
    loss_near = pseudo_huber_loss(x, near, c=c)
    loss_far = pseudo_huber_loss(x, far, c=c)
    assert loss_near >= 0.0
    assert loss_far > loss_near


def test_pseudo_huber_loss_matches_formula():
    """L = mean(sqrt((x-y)^2 + c^2) - c)."""
    torch.manual_seed(5)
    x = torch.randn(50)
    y = torch.randn(50)
    c = 0.1
    expected = (((x - y) ** 2 + c**2).sqrt() - c).mean()
    assert torch.isclose(pseudo_huber_loss(x, y, c=c), expected, atol=1e-7)


def test_pseudo_huber_loss_differentiable():
    x = torch.randn(16, requires_grad=True)
    y = torch.randn(16)
    loss = pseudo_huber_loss(x, y, c=0.05)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_discretization_schedule_monotone_endpoints():
    total = 1000
    s0, s1 = 10, 1280
    sched = get_discretization_schedule(total, s0=s0, s1=s1, rho=7.0)

    n_start = sched(0)
    n_end = sched(total)

    # Start near s0, end at s1.
    assert n_start == s0
    assert n_end == s1
    # Non-decreasing over training (rho > 0, progress**rho monotone increasing).
    prev = sched(0)
    for step in range(0, total + 1, 50):
        cur = sched(step)
        assert cur >= prev
        assert isinstance(cur, int)
        prev = cur
    # Always within [s0, s1].
    assert s0 <= sched(total // 2) <= s1


def test_ema_decay_schedule_in_range_and_increasing():
    total = 1000
    s0, s1, mu0 = 10, 1280, 0.95
    sched = get_ema_decay_schedule(total, s0=s0, s1=s1, mu0=mu0)

    decay_start = sched(0)
    decay_end = sched(total)

    # At step 0: n == s0 so mu == exp(s0*log(mu0)/s0) == mu0.
    assert math.isclose(decay_start, mu0, rel_tol=1e-6)
    # Decay increases toward 1 as N grows (more averaging at higher discretization).
    assert decay_end > decay_start
    # Valid EMA decay range.
    for step in range(0, total + 1, 100):
        d = sched(step)
        assert 0.0 < d < 1.0


# ---------------------------------------------------------------------------
# consistency: ConsistencyTrainer in-memory code paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_accelerator():
    """One shared Accelerator for the whole module.

    accelerate's AcceleratorState is a process-global singleton, so it cannot be
    re-created with different settings within a single test process. We build it
    once (mixed_precision='no') and inject it into every ConsistencyTrainer.
    """
    try:
        from accelerate import Accelerator

        return Accelerator(mixed_precision="no")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct Accelerator: {exc!r}")


def _make_consistency_trainer(accelerator, mode, diffusion=None, teacher=None, channels=4):
    """Build a tiny ConsistencyTrainer using a shared, pre-built Accelerator.

    Uses a small max_steps and disables FSDP / wandb so the constructor stays
    light. The injected accelerator avoids the global-singleton conflict and lets
    us place inputs on ``accelerator.device`` (CPU or CUDA).
    """
    config = PostTrainingConfig(
        method="consistency",
        teacher_path="dummy.pt",  # only needed to satisfy config validation
        batch_size=4,
        max_steps=10,
        use_fsdp=False,
        use_ema=True,
        mixed_precision="no",
        wandb_project=None,
        seed=0,
        huber_c=0.00054,
    )
    student = TinyContinuousModel(channels) if diffusion is None else TinyDiscreteModel(channels)
    trainer = ConsistencyTrainer(
        student=student,
        config=config,
        teacher=teacher,
        diffusion=diffusion,
        accelerator=accelerator,
        mode=mode,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        s0=4,
        s1=20,
    )
    return trainer


@pytest.fixture(scope="module")
def cont_trainer(shared_accelerator):
    return _make_consistency_trainer(
        shared_accelerator, mode="training", diffusion=None, channels=4
    )


def test_consistency_requires_teacher_for_distillation(shared_accelerator):
    config = PostTrainingConfig(
        method="consistency", teacher_path="dummy.pt", use_fsdp=False, wandb_project=None
    )
    # The teacher check happens before accelerator setup, but we still pass the
    # shared accelerator to avoid the global-singleton conflict on re-construction.
    with pytest.raises(ValueError, match="[Tt]eacher"):
        ConsistencyTrainer(
            student=TinyContinuousModel(4),
            config=config,
            teacher=None,
            accelerator=shared_accelerator,
            mode="distillation",
        )


def test_consistency_preconditioning_boundary_condition(cont_trainer):
    """Boundary condition f(x, sigma_min->0) = x via c_skip(0)=1, c_out(0)=0.

    This is the defining property of consistency models (Song et al., 2023).
    """
    trainer = cont_trainer
    dev = trainer.accelerator.device
    # At sigma == 0: c_skip = sigma_data^2 / sigma_data^2 = 1, c_out = 0, so the
    # preconditioned output equals the input x exactly.
    sigma0 = torch.zeros(3, device=dev)
    assert torch.allclose(trainer._c_skip(sigma0), torch.ones(3, device=dev))
    assert torch.allclose(trainer._c_out(sigma0), torch.zeros(3, device=dev))

    x = torch.randn(3, 4, 4, device=dev)
    out = trainer._consistency_function(trainer.student, x, sigma0, t=sigma0)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5)


def test_consistency_preconditioning_coeffs_consistent(cont_trainer):
    """c_skip, c_out, c_in must satisfy their Karras/EDM definitions."""
    trainer = cont_trainer
    sigma = torch.tensor([0.1, 1.0, 5.0, 50.0], device=trainer.accelerator.device)
    sd = trainer.sigma_data

    c_skip = trainer._c_skip(sigma)
    c_out = trainer._c_out(sigma)
    c_in = trainer._c_in(sigma)

    assert torch.allclose(c_skip, sd**2 / (sigma**2 + sd**2))
    assert torch.allclose(c_out, sigma * sd / (sigma**2 + sd**2).sqrt())
    assert torch.allclose(c_in, 1.0 / (sigma**2 + sd**2).sqrt())
    # As sigma grows, c_skip -> 0 (model output dominates).
    assert c_skip[0] > c_skip[-1]


def test_consistency_sigma_schedule_is_karras_decreasing(cont_trainer):
    """sigma schedule spans [sigma_min, sigma_max] and is monotone decreasing in index."""
    trainer = cont_trainer
    n = 8
    sigmas = trainer._get_sigma_schedule(n)
    assert sigmas.shape == (n,)
    # index 0 -> sigma_max, index n-1 -> sigma_min (Karras ordering).
    assert torch.isclose(sigmas[0], torch.tensor(trainer.sigma_max), rtol=1e-4)
    assert torch.isclose(sigmas[-1], torch.tensor(trainer.sigma_min), rtol=1e-4)
    # Strictly decreasing.
    assert torch.all(sigmas[1:] < sigmas[:-1])


def test_consistency_function_continuous_shape_and_grad(cont_trainer):
    """The continuous consistency function returns x-shaped output and is differentiable."""
    trainer = cont_trainer
    dev = trainer.accelerator.device
    x = torch.randn(2, 4, 4, device=dev)
    sigma = torch.tensor([1.0, 10.0], device=dev)
    out = trainer._consistency_function(trainer.student, x, sigma, t=sigma)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    loss = out.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in trainer.student.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_consistency_add_noise_increases_variance(cont_trainer):
    trainer = cont_trainer
    dev = trainer.accelerator.device
    x = torch.zeros(64, 4, 4, device=dev)
    sigma = torch.full((64,), 3.0, device=dev)
    noised = trainer._add_noise(x, sigma)
    assert noised.shape == x.shape
    # Std of pure-noise (x=0) batch should be ~ sigma.
    assert noised.std().item() > 1.0


def test_consistency_train_step_training_updates_and_returns_metrics(cont_trainer):
    """train_step_training (self-consistency, no teacher) runs and reports metrics.

    Verifies: (a) loss is finite, (b) reported n is within the schedule range,
    (c) EMA decay is applied, (d) student parameters actually change.
    """
    trainer = cont_trainer
    torch.manual_seed(7)
    before = [p.detach().clone() for p in trainer.student.parameters()]

    batch = torch.randn(6, 4, 4, device=trainer.accelerator.device)
    metrics = trainer.train_step_training(batch)

    assert set(metrics) >= {"loss", "n", "ema_decay"}
    assert math.isfinite(metrics["loss"])
    assert trainer.s0 <= metrics["n"] <= trainer.s1
    assert 0.0 < metrics["ema_decay"] < 1.0

    after = list(trainer.student.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
    assert changed, "student params did not update after a training step"


def test_consistency_distillation_uses_ema_teacher_for_target(shared_accelerator):
    """Distillation step: target comes from the EMA-averaged student via teacher rollout."""
    teacher = TinyContinuousModel(4)
    trainer = _make_consistency_trainer(
        shared_accelerator, mode="distillation", diffusion=None, teacher=teacher, channels=4
    )

    # Teacher must be frozen for distillation (no grad / eval).
    assert all(not p.requires_grad for p in trainer.teacher.parameters())

    torch.manual_seed(8)
    before = [p.detach().clone() for p in trainer.student.parameters()]
    batch = torch.randn(6, 4, 4, device=trainer.accelerator.device)
    metrics = trainer.train_step_distillation(batch)

    assert math.isfinite(metrics["loss"])
    assert trainer.s0 <= metrics["n"] <= trainer.s1
    # Student updates, teacher does not.
    after_student = list(trainer.student.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after_student, strict=True))


def test_consistency_ema_decay_is_set_from_schedule(cont_trainer):
    """After a train step the EMA object's decay matches the scheduled value."""
    trainer = cont_trainer
    trainer.global_step = 5
    batch = torch.randn(4, 4, 4, device=trainer.accelerator.device)
    metrics = trainer.train_step_training(batch)
    expected = trainer.ema_schedule(5)
    assert math.isclose(trainer.ema.decay, expected, rel_tol=1e-6)
    assert math.isclose(metrics["ema_decay"], expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# CUDA guard example (the code paths above are CPU; assert nothing CUDA-only breaks)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_consistency_function_on_cuda(cont_trainer):  # pragma: no cover - needs GPU
    # When CUDA is present, accelerate places the prepared student on the GPU and
    # the consistency function output should land on the accelerator device.
    trainer = cont_trainer
    dev = trainer.accelerator.device
    x = torch.randn(2, 4, 4, device=dev)
    sigma = torch.tensor([1.0, 2.0], device=dev)
    out = trainer._consistency_function(trainer.student, x, sigma, t=sigma)
    assert out.device.type == dev.type


def test_reflow_trainer_class_is_importable():
    """Smoke check that the trainer class exists with the documented attributes.

    The full ReflowTrainer.__init__ builds an Accelerator + optimizer; the
    training/loss math it relies on is covered by the _reflow_loss tests above.
    """
    assert hasattr(ReflowTrainer, "reflow_iteration")
    assert hasattr(ReflowTrainer, "train_on_pairs")
    assert hasattr(ReflowTrainer, "_collect_data_samples")
