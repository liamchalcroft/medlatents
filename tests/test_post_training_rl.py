"""Tests for the post_training RL subsystem (medlatents.post_training.rl).

Scope: advantage/return/reward math, GRPO group-relative normalization,
policy-gradient / importance-ratio loss shape + differentiability, KL-penalty
behavior, and PPO-style clipping (DDPO / GARDO).

All tests are CPU-only and use tiny tensors / models with fixed seeds.

The trainer ``compute_policy_loss`` methods are exercised by constructing the
trainer object via ``__new__`` and attaching only the minimal attributes the
loss method touches. This deliberately avoids the heavyweight ``__init__``
(which builds an ``accelerate.Accelerator``, optimizer, EMA, etc.) so the loss
*math* can be unit-tested in isolation on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from medlatents.post_training.configs import PostTrainingConfig
from medlatents.post_training.rl.base import (
    BaseRLTrainer,
    RLBatch,
    Trajectory,
    TrajectoryBuffer,
    compute_advantages_gae,
    compute_advantages_monte_carlo,
    normalize_advantages,
)
from medlatents.post_training.rl.ddpo import DDPOTrainer
from medlatents.post_training.rl.gardo import GARDOTrainer
from medlatents.post_training.rl.grpo import GRPOTrainer

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _StubAccelerator:
    """Minimal stand-in for accelerate.Accelerator used by loss methods."""

    def __init__(self, device: torch.device = CPU) -> None:
        self.device = device


class TinyTokenModel(nn.Module):
    """Tiny token model: maps (x, t) -> logits [B, L, vocab].

    Accepts the calling conventions used across the RL trainers:
    - diffusion/flow: model(x=tokens, t=timesteps)
    - maskgit:        model(tokens) / model(tokens, mask=...)
    """

    def __init__(self, vocab_size: int, hidden: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, hidden)  # +1 for mask token
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x=None, t=None, mask=None):  # noqa: D401, ANN001
        tokens = x
        tokens = tokens.clamp(0, self.vocab_size)  # mask token == vocab_size
        h = self.embed(tokens)
        if t is not None:
            # Inject a tiny, differentiable time dependence.
            t_scalar = t.float().reshape(t.shape[0], *([1] * (h.dim() - 1)))
            h = h + 0.01 * t_scalar
        return self.proj(h)


def _make_trajectory(
    n_steps: int,
    seq_len: int,
    vocab_size: int,
    final_reward: float,
    *,
    requires_grad: bool = False,
    group_id=None,
    seed: int = 0,
) -> Trajectory:
    """Build a single Trajectory with a known final reward."""
    g = torch.Generator().manual_seed(seed)
    states = torch.randint(0, vocab_size, (n_steps, seq_len), generator=g)
    actions = torch.randint(0, vocab_size, (n_steps, seq_len), generator=g)
    timesteps = torch.linspace(0.0, 1.0, n_steps)
    log_probs = torch.randn(n_steps, generator=g)
    if requires_grad:
        log_probs = log_probs.clone().requires_grad_(True)
    rewards = torch.zeros(n_steps)
    rewards[-1] = final_reward
    traj = Trajectory(
        states=states,
        actions=actions,
        timesteps=timesteps,
        log_probs=log_probs,
        rewards=rewards,
        final_sample=actions[-1].clone(),
    )
    traj.prompt = group_id
    return traj


# --------------------------------------------------------------------------- #
# Monte Carlo returns
# --------------------------------------------------------------------------- #
class TestMonteCarloReturns:
    def test_shape_matches_input(self):
        rewards = torch.tensor([0.0, 0.0, 1.0, 0.0])
        returns = compute_advantages_monte_carlo(rewards, gamma=0.9)
        assert returns.shape == rewards.shape

    def test_undiscounted_terminal_reward_propagates_unchanged(self):
        # gamma=1.0: every step's return equals the sum of future rewards.
        rewards = torch.tensor([0.0, 0.0, 0.0, 5.0])
        returns = compute_advantages_monte_carlo(rewards, gamma=1.0)
        torch.testing.assert_close(returns, torch.tensor([5.0, 5.0, 5.0, 5.0]))

    def test_discounted_terminal_reward_handchecked(self):
        # Only the final step has reward r=2. With gamma=0.5, the return at
        # step t is r * gamma^(T-1-t):  [0.25, 0.5, 1.0, 2.0].
        gamma = 0.5
        rewards = torch.tensor([0.0, 0.0, 0.0, 2.0])
        returns = compute_advantages_monte_carlo(rewards, gamma=gamma)
        expected = torch.tensor([2.0 * gamma**3, 2.0 * gamma**2, 2.0 * gamma, 2.0])
        torch.testing.assert_close(returns, expected)

    def test_dense_rewards_handchecked(self):
        # G_t = r_t + gamma * G_{t+1}
        gamma = 0.9
        rewards = torch.tensor([1.0, 2.0, 3.0])
        g2 = 3.0
        g1 = 2.0 + gamma * g2
        g0 = 1.0 + gamma * g1
        returns = compute_advantages_monte_carlo(rewards, gamma=gamma)
        torch.testing.assert_close(returns, torch.tensor([g0, g1, g2]))


# --------------------------------------------------------------------------- #
# GAE advantages / returns
# --------------------------------------------------------------------------- #
class TestGAE:
    def test_shapes(self):
        rewards = torch.zeros(5)
        values = torch.zeros(5)
        adv, ret = compute_advantages_gae(rewards, values)
        assert adv.shape == rewards.shape
        assert ret.shape == rewards.shape

    def test_returns_equal_advantages_plus_values(self):
        # By construction returns[t] = advantages[t] + values[t].
        torch.manual_seed(0)
        rewards = torch.randn(6)
        values = torch.randn(6)
        adv, ret = compute_advantages_gae(rewards, values, gamma=0.99, lam=0.95)
        torch.testing.assert_close(ret, adv + values)

    def test_zero_values_lambda_one_reduces_to_monte_carlo(self):
        # With values == 0 and lam == 1, GAE advantage reduces to the
        # discounted return sum_{k>=t} gamma^{k-t} r_k.
        gamma = 0.9
        rewards = torch.tensor([1.0, 0.0, 2.0, 0.0])
        values = torch.zeros(4)
        adv, _ = compute_advantages_gae(rewards, values, gamma=gamma, lam=1.0)
        mc = compute_advantages_monte_carlo(rewards, gamma=gamma)
        torch.testing.assert_close(adv, mc)

    def test_single_step_advantage_handchecked(self):
        # Last step: next_value forced to 0, so delta = r - V.
        rewards = torch.tensor([3.0])
        values = torch.tensor([1.0])
        adv, ret = compute_advantages_gae(rewards, values, gamma=0.99, lam=0.95)
        torch.testing.assert_close(adv, torch.tensor([2.0]))  # 3 - 1
        torch.testing.assert_close(ret, torch.tensor([3.0]))  # adv + V


# --------------------------------------------------------------------------- #
# normalize_advantages
# --------------------------------------------------------------------------- #
class TestNormalizeAdvantages:
    def test_zero_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out = normalize_advantages(x)
        assert abs(out.mean().item()) < 1e-5

    def test_unit_ish_std(self):
        # std should be ~1 (slightly below 1 due to the +eps in denominator).
        x = torch.tensor([10.0, -3.0, 4.0, 7.0, -1.0, 2.0])
        out = normalize_advantages(x, eps=1e-6)
        assert out.std().item() == pytest.approx(1.0, abs=1e-3)

    def test_matches_manual_formula(self):
        x = torch.tensor([2.0, 4.0, 6.0])
        eps = 1e-6
        expected = (x - x.mean()) / (x.std() + eps)
        torch.testing.assert_close(normalize_advantages(x, eps=eps), expected)

    def test_constant_input_does_not_nan(self):
        # std == 0; eps keeps it finite. Result should be all-zeros (no NaN).
        x = torch.full((4,), 3.0)
        out = normalize_advantages(x)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_preserves_shape_2d(self):
        x = torch.randn(3, 4)
        out = normalize_advantages(x)
        assert out.shape == x.shape


# --------------------------------------------------------------------------- #
# Trajectory dataclass
# --------------------------------------------------------------------------- #
class TestTrajectory:
    def test_to_device_cpu_roundtrip(self):
        traj = _make_trajectory(3, 5, 8, final_reward=1.0, group_id=7)
        moved = traj.to(CPU)
        assert moved.states.device == CPU
        assert moved.prompt == 7  # non-tensor field preserved
        assert moved.advantages is None
        assert moved.returns is None

    def test_to_device_moves_optional_fields(self):
        traj = _make_trajectory(3, 5, 8, final_reward=1.0)
        traj.advantages = torch.zeros(3)
        traj.returns = torch.ones(3)
        moved = traj.to(CPU)
        assert moved.advantages is not None and moved.advantages.device == CPU
        assert moved.returns is not None and moved.returns.device == CPU


# --------------------------------------------------------------------------- #
# RLBatch dataclass
# --------------------------------------------------------------------------- #
class TestRLBatch:
    def test_len(self):
        trajs = [_make_trajectory(2, 4, 8, final_reward=float(i)) for i in range(3)]
        batch = RLBatch(trajectories=trajs, batch_rewards=torch.tensor([0.0, 1.0, 2.0]))
        assert len(batch) == 3
        assert batch.batch_advantages is None


# --------------------------------------------------------------------------- #
# TrajectoryBuffer
# --------------------------------------------------------------------------- #
class TestTrajectoryBuffer:
    def test_add_and_len(self):
        buf = TrajectoryBuffer(max_size=10)
        assert len(buf) == 0
        buf.add(_make_trajectory(2, 4, 8, final_reward=1.0))
        assert len(buf) == 1

    def test_eviction_drops_lowest_priority(self):
        buf = TrajectoryBuffer(max_size=2)
        a = _make_trajectory(2, 4, 8, final_reward=1.0, group_id="a")
        b = _make_trajectory(2, 4, 8, final_reward=2.0, group_id="b")
        c = _make_trajectory(2, 4, 8, final_reward=3.0, group_id="c")
        buf.add(a, priority=0.1)  # lowest -> should be evicted when full
        buf.add(b, priority=5.0)
        buf.add(c, priority=5.0)
        assert len(buf) == 2
        remaining = {t.prompt for t in buf.buffer}
        assert "a" not in remaining
        assert remaining == {"b", "c"}

    def test_sample_empty_returns_empty(self):
        buf = TrajectoryBuffer()
        assert buf.sample(4) == []

    def test_sample_caps_at_buffer_size(self):
        buf = TrajectoryBuffer()
        for i in range(3):
            buf.add(_make_trajectory(2, 4, 8, final_reward=float(i)))
        out = buf.sample(10)
        assert len(out) == 3

    def test_sample_prioritized_returns_requested_count(self):
        torch.manual_seed(0)
        buf = TrajectoryBuffer()
        for i in range(5):
            buf.add(_make_trajectory(2, 4, 8, final_reward=float(i)), priority=float(i + 1))
        out = buf.sample(3, prioritized=True)
        assert len(out) == 3

    def test_clear(self):
        buf = TrajectoryBuffer()
        buf.add(_make_trajectory(2, 4, 8, final_reward=1.0))
        buf.clear()
        assert len(buf) == 0
        assert buf.priorities == []


# --------------------------------------------------------------------------- #
# BaseRLTrainer.compute_kl_penalty (static behavior)
# --------------------------------------------------------------------------- #
class _ConcreteRLTrainer(BaseRLTrainer):
    """Minimal concrete subclass so the (non-abstract) helper methods on
    BaseRLTrainer can be exercised without the heavyweight __init__."""

    def generate_trajectories(self, batch_size, seq_length, **kwargs):  # noqa: ANN001
        return []

    def compute_policy_loss(self, trajectories):  # noqa: ANN001
        return torch.tensor(0.0), {}


class TestKLPenalty:
    def _trainer(self):
        # BaseRLTrainer is abstract; instantiate a concrete subclass via __new__
        # to bypass the Accelerator-building __init__.
        return _ConcreteRLTrainer.__new__(_ConcreteRLTrainer)

    def test_identical_logprobs_give_zero(self):
        trainer = self._trainer()
        lp = torch.tensor([-1.0, -2.0, -3.0])
        kl = trainer.compute_kl_penalty(lp, lp.clone())
        assert kl.item() == pytest.approx(0.0, abs=1e-6)

    def test_handchecked_mean_difference(self):
        trainer = self._trainer()
        lp = torch.tensor([0.0, 0.0])
        ref = torch.tensor([-1.0, -3.0])
        # mean(lp - ref) = mean([1, 3]) = 2.0
        kl = trainer.compute_kl_penalty(lp, ref)
        assert kl.item() == pytest.approx(2.0, abs=1e-6)

    def test_sign_when_policy_more_confident(self):
        # If policy assigns higher log-prob than ref, KL estimate is positive.
        trainer = self._trainer()
        lp = torch.tensor([-0.5, -0.5])
        ref = torch.tensor([-2.0, -2.0])
        assert trainer.compute_kl_penalty(lp, ref).item() > 0


# --------------------------------------------------------------------------- #
# GRPO policy loss: group-relative normalization + PG correctness
# --------------------------------------------------------------------------- #
def _make_grpo_trainer(num_samples=4, kl_coeff=0.1, ref_model=None, model_type="maskgit"):
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.accelerator = _StubAccelerator()
    # Config validation requires num_samples_per_prompt >= 2 for RL methods;
    # the compute_policy_loss path itself groups by traj.prompt and never reads
    # self.num_samples, so we keep the config valid and set num_samples freely.
    trainer.config = PostTrainingConfig(
        method="grpo", num_samples_per_prompt=max(2, num_samples), kl_coeff=kl_coeff
    )
    trainer.ref_model = ref_model
    trainer.model = None
    trainer.model_type = model_type
    trainer.num_samples = num_samples
    return trainer


class TestGRPOLoss:
    def test_group_advantages_zero_mean_unit_std(self):
        trainer = _make_grpo_trainer(num_samples=4)
        rewards = [1.0, 2.0, 3.0, 4.0]
        trajs = [
            _make_trajectory(3, 5, 8, final_reward=r, group_id=0, seed=i)
            for i, r in enumerate(rewards)
        ]
        trainer.compute_policy_loss(trajs)
        advs = torch.stack([t.advantages[0] for t in trajs])
        assert advs.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert advs.std().item() == pytest.approx(1.0, abs=1e-3)

    def test_advantage_broadcast_over_trajectory_length(self):
        trainer = _make_grpo_trainer(num_samples=2)
        trajs = [
            _make_trajectory(4, 5, 8, final_reward=1.0, group_id=0, seed=0),
            _make_trajectory(4, 5, 8, final_reward=3.0, group_id=0, seed=1),
        ]
        trainer.compute_policy_loss(trajs)
        for t in trajs:
            assert t.advantages.shape == (len(t.states),)

    def test_loss_is_scalar_and_differentiable(self):
        trainer = _make_grpo_trainer(num_samples=4)
        trajs = [
            _make_trajectory(3, 5, 8, final_reward=r, group_id=0, requires_grad=True, seed=i)
            for i, r in enumerate([1.0, 2.0, 3.0, 4.0])
        ]
        loss, metrics = trainer.compute_policy_loss(trajs)
        assert loss.ndim == 0
        assert loss.requires_grad
        loss.backward()
        # At least one trajectory's log_probs receives a gradient.
        assert any(t.log_probs.grad is not None for t in trajs)
        assert "loss" in metrics and "kl" in metrics
        assert metrics["num_groups"] == 1

    def test_policy_gradient_value_handchecked(self):
        # With one group of 4 rewards [1,2,3,4]:
        #   advantages = normalize([1,2,3,4]) (zero-mean, ~unit-std).
        # Per-traj loss = -adv * sum(log_probs); averaged over 4 samples.
        # No ref_model => no KL term.
        trainer = _make_grpo_trainer(num_samples=4, ref_model=None)
        log_prob_sums = [0.5, -1.0, 2.0, -0.5]
        rewards = [1.0, 2.0, 3.0, 4.0]
        trajs = []
        for i, (r, lps) in enumerate(zip(rewards, log_prob_sums)):
            t = _make_trajectory(2, 5, 8, final_reward=r, group_id=0, seed=i)
            # Overwrite log_probs so the sum is exactly known.
            t.log_probs = torch.tensor([lps, 0.0])
            trajs.append(t)
        loss, _ = trainer.compute_policy_loss(trajs)

        adv = normalize_advantages(torch.tensor(rewards))
        expected = sum(-adv[i] * torch.tensor(log_prob_sums[i]) for i in range(4)) / 4.0
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_two_groups_counted_separately(self):
        trainer = _make_grpo_trainer(num_samples=2)
        trajs = [
            _make_trajectory(2, 5, 8, final_reward=1.0, group_id=0, seed=0),
            _make_trajectory(2, 5, 8, final_reward=2.0, group_id=0, seed=1),
            _make_trajectory(2, 5, 8, final_reward=5.0, group_id=1, seed=2),
            _make_trajectory(2, 5, 8, final_reward=6.0, group_id=1, seed=3),
        ]
        _, metrics = trainer.compute_policy_loss(trajs)
        assert metrics["num_groups"] == 2

    def test_singleton_group_has_zero_advantage(self):
        # A group with a single trajectory cannot be normalized -> advantage 0,
        # so its policy-gradient contribution is exactly 0.
        trainer = _make_grpo_trainer(num_samples=1)
        trajs = [_make_trajectory(3, 5, 8, final_reward=9.0, group_id=0, seed=0)]
        loss, _ = trainer.compute_policy_loss(trajs)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        torch.testing.assert_close(trajs[0].advantages, torch.zeros(3))

    def test_kl_term_active_with_ref_model(self):
        # With a ref model, the loss includes kl_coeff * KL and metrics["kl"]
        # becomes non-zero (the two models differ in parameters).
        torch.manual_seed(0)
        vocab = 8
        ref = TinyTokenModel(vocab)
        trainer = _make_grpo_trainer(num_samples=2, kl_coeff=0.5, ref_model=ref)
        trajs = [
            _make_trajectory(2, 5, vocab, final_reward=1.0, group_id=0, seed=0),
            _make_trajectory(2, 5, vocab, final_reward=3.0, group_id=0, seed=1),
        ]
        _, metrics = trainer.compute_policy_loss(trajs)
        assert "kl" in metrics
        assert metrics["kl"] != 0.0


# --------------------------------------------------------------------------- #
# DDPO policy loss: PPO clipped surrogate + importance ratio
# --------------------------------------------------------------------------- #
def _make_ddpo_trainer(clip_range=0.2, kl_coeff=0.1, ref_model=None, value_model=None, model=None):
    trainer = DDPOTrainer.__new__(DDPOTrainer)
    trainer.accelerator = _StubAccelerator()
    trainer.config = PostTrainingConfig(method="ddpo", clip_range=clip_range, kl_coeff=kl_coeff)
    trainer.ref_model = ref_model
    trainer.value_model = value_model
    trainer.model = model
    trainer.clip_range = clip_range
    trainer.kl_coeff = kl_coeff
    return trainer


class TestDDPOLoss:
    def test_loss_shape_and_differentiable(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_ddpo_trainer(model=model)
        traj = _make_trajectory(3, 5, vocab, final_reward=1.0, seed=1)
        traj.advantages = torch.tensor([0.5, -0.5, 1.0])
        loss, metrics = trainer.compute_policy_loss([traj])
        assert loss.ndim == 0
        assert loss.requires_grad
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert {"loss", "policy_loss", "value_loss", "kl"} <= set(metrics)

    def test_zero_advantage_gives_zero_policy_loss(self):
        # surr1 = surr2 = ratio*0 = 0 -> min == 0 -> policy_loss == 0.
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_ddpo_trainer(model=model)
        traj = _make_trajectory(3, 5, vocab, final_reward=1.0, seed=2)
        traj.advantages = torch.zeros(3)
        loss, metrics = trainer.compute_policy_loss([traj])
        assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-6)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_clipping_caps_loss_for_positive_advantage(self):
        # PPO surrogate for A>0 is bounded below (loss bounded above) by the
        # clipped branch. We compare a tiny clip_range vs a huge one with the
        # SAME model+trajectory; the heavily-clipped run must give a loss that
        # is >= the unclipped-ish run is NOT generally true, so instead assert
        # the documented identity directly: loss == -min(surr1, surr2).
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        clip = 0.2
        trainer = _make_ddpo_trainer(clip_range=clip, model=model)
        traj = _make_trajectory(1, 5, vocab, final_reward=1.0, seed=3)
        advantage = 2.0
        traj.advantages = torch.tensor([advantage])

        # Recompute the expected per-step loss exactly as the method does.
        state = traj.states[0].unsqueeze(0)
        action = traj.actions[0].unsqueeze(0)
        t = traj.timesteps[0].unsqueeze(0)
        old_log_prob = traj.log_probs[0]
        with torch.no_grad():
            logits = model(x=state, t=t)
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_prob = (
                torch.gather(log_probs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1).sum()
            )
            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
            expected = (-torch.min(surr1, surr2)).item()

        loss, metrics = trainer.compute_policy_loss([traj])
        assert metrics["policy_loss"] == pytest.approx(expected, abs=1e-5)
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_ratio_is_one_when_logprob_unchanged(self):
        # If we set old_log_prob to exactly the model's current log-prob, the
        # ratio is 1, so loss == -advantage (for A within the clip band).
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_ddpo_trainer(clip_range=0.5, model=model)
        traj = _make_trajectory(1, 5, vocab, final_reward=1.0, seed=4)
        state = traj.states[0].unsqueeze(0)
        action = traj.actions[0].unsqueeze(0)
        t = traj.timesteps[0].unsqueeze(0)
        with torch.no_grad():
            logits = model(x=state, t=t)
            lp = F.log_softmax(logits, dim=-1)
            cur = torch.gather(lp, -1, action.unsqueeze(-1)).squeeze(-1).sum()
        traj.log_probs = cur.detach().reshape(1).clone()
        advantage = 0.3  # within +/-0.5 clip band
        traj.advantages = torch.tensor([advantage])
        _, metrics = trainer.compute_policy_loss([traj])
        assert metrics["policy_loss"] == pytest.approx(-advantage, abs=1e-4)

    def test_kl_metric_reported_with_ref_model(self):
        # With a ref model the KL divergence is computed and surfaced in metrics.
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        ref = TinyTokenModel(vocab)
        trainer = _make_ddpo_trainer(kl_coeff=0.5, ref_model=ref, model=model)
        traj = _make_trajectory(2, 5, vocab, final_reward=1.0, seed=5)
        traj.advantages = torch.zeros(2)
        _, metrics = trainer.compute_policy_loss([traj])
        assert metrics["kl"] != 0.0

    def test_kl_penalty_enters_optimized_loss(self):
        # Regression: the KL penalty must contribute to the optimized loss. With
        # zero advantage and no value model, the policy and value losses are 0, so
        # the returned loss equals kl_coeff * mean_kl (non-zero).
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        ref = TinyTokenModel(vocab)
        trainer = _make_ddpo_trainer(kl_coeff=0.5, ref_model=ref, model=model)
        traj = _make_trajectory(2, 5, vocab, final_reward=1.0, seed=5)
        traj.advantages = torch.zeros(2)  # zero out policy loss
        loss, metrics = trainer.compute_policy_loss([traj])
        assert metrics["policy_loss"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["kl"] != 0.0  # KL is real and non-zero
        # ...and the optimized loss now reflects it: loss == kl_coeff * mean_kl
        assert loss.item() == pytest.approx(0.5 * metrics["kl"], abs=1e-5)
        assert loss.item() != pytest.approx(0.0, abs=1e-6)

    def test_value_loss_contributes_when_value_model_present(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)

        class ValueModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab + 1, 4)
                self.head = nn.Linear(4, 1)

            def forward(self, x, t):  # noqa: ANN001
                return self.head(self.embed(x.clamp(0, vocab)).mean(dim=1))

        vm = ValueModel()
        trainer = _make_ddpo_trainer(model=model, value_model=vm)
        traj = _make_trajectory(2, 5, vocab, final_reward=1.0, seed=6)
        traj.advantages = torch.zeros(2)
        traj.returns = torch.tensor([1.0, 2.0])
        _, metrics = trainer.compute_policy_loss([traj])
        assert metrics["value_loss"] > 0.0


# --------------------------------------------------------------------------- #
# GARDO policy loss: reward thresholding + adaptive KL
# --------------------------------------------------------------------------- #
def _make_gardo_trainer(
    reward_threshold=0.1,
    distance_penalty=0.01,
    ref_model=None,
    model=None,
    model_type="maskgit",
    adaptive_kl=False,
    kl_target=0.1,
):
    trainer = GARDOTrainer.__new__(GARDOTrainer)
    trainer.accelerator = _StubAccelerator()
    trainer.config = PostTrainingConfig(
        method="gardo",
        reward_threshold=reward_threshold,
        distance_penalty=distance_penalty,
    )
    trainer.ref_model = ref_model
    trainer.model = model
    trainer.model_type = model_type
    trainer.reward_threshold = reward_threshold
    trainer.distance_penalty = distance_penalty
    trainer.adaptive_kl = adaptive_kl
    trainer.kl_target = kl_target
    trainer.kl_coeff = distance_penalty
    trainer.grad_history = []
    return trainer


class TestGARDOLoss:
    def test_filters_below_threshold(self):
        # All rewards below threshold => everything filtered, nothing valid.
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_gardo_trainer(reward_threshold=10.0, model=model)
        trajs = [
            _make_trajectory(2, 5, vocab, final_reward=1.0, seed=0),
            _make_trajectory(2, 5, vocab, final_reward=2.0, seed=1),
        ]
        loss, metrics = trainer.compute_policy_loss(trajs)
        assert metrics["num_filtered"] == 2
        assert metrics["num_valid"] == 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_keeps_above_threshold(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_gardo_trainer(reward_threshold=1.5, model=model)
        trajs = [
            _make_trajectory(2, 5, vocab, final_reward=1.0, seed=0),  # filtered
            _make_trajectory(2, 5, vocab, final_reward=2.0, seed=1),  # kept
            _make_trajectory(2, 5, vocab, final_reward=3.0, seed=2),  # kept
        ]
        _, metrics = trainer.compute_policy_loss(trajs)
        assert metrics["num_filtered"] == 1
        assert metrics["num_valid"] == 2

    def test_loss_differentiable_through_model(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_gardo_trainer(reward_threshold=0.0, model=model)
        trajs = [_make_trajectory(2, 5, vocab, final_reward=2.0, seed=3)]
        loss, _ = trainer.compute_policy_loss(trajs)
        assert loss.requires_grad
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_reward_loss_handchecked_no_ref(self):
        # GARDO reward loss per kept trajectory = -reward * sum(policy_logprob).
        # With a single kept trajectory and no ref model, total_loss equals that
        # (averaged over num_valid == 1). We recompute the policy log-prob.
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_gardo_trainer(reward_threshold=0.0, ref_model=None, model=model)
        reward = 2.5
        traj = _make_trajectory(2, 5, vocab, final_reward=reward, seed=7)

        with torch.no_grad():
            total_lp = torch.tensor(0.0)
            for step_idx in range(len(traj.states)):
                state = traj.states[step_idx].unsqueeze(0)
                action = traj.actions[step_idx].unsqueeze(0)
                logits = model(state)
                lp = F.log_softmax(logits, dim=-1)
                total_lp = total_lp + torch.gather(lp, -1, action.unsqueeze(-1)).squeeze(-1).sum()
            expected = (-reward * total_lp).item()

        loss, metrics = trainer.compute_policy_loss([traj])
        assert metrics["reward_loss"] == pytest.approx(expected, abs=1e-4)
        assert loss.item() == pytest.approx(expected, abs=1e-4)

    def test_kl_loss_present_with_ref_and_in_total(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        ref = TinyTokenModel(vocab)
        kl_coeff = 0.3
        trainer = _make_gardo_trainer(
            reward_threshold=0.0, distance_penalty=kl_coeff, ref_model=ref, model=model
        )
        traj = _make_trajectory(2, 5, vocab, final_reward=2.0, seed=8)
        loss, metrics = trainer.compute_policy_loss([traj])
        # total_loss = reward_loss + kl_coeff * kl_loss
        expected = metrics["reward_loss"] + trainer.kl_coeff * metrics["kl_loss"]
        assert loss.item() == pytest.approx(expected, abs=1e-4)
        assert metrics["kl_loss"] != 0.0

    def test_adaptive_kl_increases_when_kl_far_above_target(self):
        # _update_kl_coefficient: kl_value > 1.5 * kl_target raises kl_coeff.
        trainer = _make_gardo_trainer(distance_penalty=0.1, adaptive_kl=True, kl_target=0.1)
        before = trainer.kl_coeff
        trainer._update_kl_coefficient(kl_value=10.0, grad_alignment=0.0)
        assert trainer.kl_coeff > before

    def test_adaptive_kl_decreases_when_kl_far_below_target(self):
        trainer = _make_gardo_trainer(distance_penalty=0.5, adaptive_kl=True, kl_target=0.1)
        before = trainer.kl_coeff
        trainer._update_kl_coefficient(kl_value=0.0, grad_alignment=0.0)
        assert trainer.kl_coeff < before

    def test_adaptive_kl_noop_when_disabled(self):
        trainer = _make_gardo_trainer(distance_penalty=0.2, adaptive_kl=False)
        before = trainer.kl_coeff
        trainer._update_kl_coefficient(kl_value=100.0, grad_alignment=-1.0)
        assert trainer.kl_coeff == before

    def test_adaptive_kl_clamped_to_max_one(self):
        trainer = _make_gardo_trainer(distance_penalty=0.99, adaptive_kl=True, kl_target=0.1)
        trainer.kl_coeff = 0.99
        for _ in range(50):
            trainer._update_kl_coefficient(kl_value=100.0, grad_alignment=-1.0)
        assert trainer.kl_coeff <= 1.0

    def test_gradient_alignment_cosine_handchecked(self):
        # Identical gradient lists -> cosine similarity == 1.
        trainer = _make_gardo_trainer()
        g = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
        cos = trainer._compute_gradient_alignment(g, [t.clone() for t in g])
        assert cos == pytest.approx(1.0, abs=1e-5)

    def test_gradient_alignment_opposite_is_minus_one(self):
        trainer = _make_gardo_trainer()
        g = [torch.tensor([1.0, 2.0, 3.0])]
        neg = [-t for t in g]
        cos = trainer._compute_gradient_alignment(g, neg)
        assert cos == pytest.approx(-1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Config validation relevant to RL methods
# --------------------------------------------------------------------------- #
class TestRLConfigValidation:
    def test_rl_requires_two_samples(self):
        with pytest.raises(ValueError):
            PostTrainingConfig(method="grpo", num_samples_per_prompt=1)

    def test_rl_ok_with_two_samples(self):
        cfg = PostTrainingConfig(method="ddpo", num_samples_per_prompt=2)
        assert cfg.num_samples_per_prompt == 2

    def test_ddpo_preset_has_small_clip_range(self):
        cfg = PostTrainingConfig.from_preset("ddpo_standard")
        assert cfg.method == "ddpo"
        assert cfg.clip_range == pytest.approx(1e-4)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
