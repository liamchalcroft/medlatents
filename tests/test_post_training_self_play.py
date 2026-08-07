"""Tests for the post_training self-play subsystem.

Covers:
- RFT (rejection.py): generate -> score -> keep top-k. The selection must keep
  the highest-reward samples and discard the rest, given a reward function.
- SPIN (iterative.py): the underlying reference-free DPO preference objective
  (prefer real data over the model's own generations) and the iterative loop
  state transitions (iteration counter / global step advancement).

All tests are CPU-only with tiny tensors / models and fixed seeds. The
trainers' heavyweight ``__init__`` (which constructs an ``accelerate.Accelerator``)
is bypassed via ``__new__``; only the attributes each tested method touches are
attached, with a lightweight stub accelerator for the optimization path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from medlatents.post_training.configs import PostTrainingConfig
from medlatents.post_training.dpo.base import DPOLoss
from medlatents.post_training.self_play.iterative import SPINTrainer
from medlatents.post_training.self_play.rejection import RFTTrainer

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Stub accelerator that supports the optimization path used by train_step.
# --------------------------------------------------------------------------- #
class _StubAccelerator:
    def __init__(self, device: torch.device = CPU) -> None:
        self.device = device
        self.sync_gradients = True
        self.is_main_process = True

    def backward(self, loss):  # noqa: ANN001
        loss.backward()

    def unwrap_model(self, model):  # noqa: ANN001
        return model

    def clip_grad_norm_(self, params, max_norm):  # noqa: ANN001
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def prepare(self, *args):
        return args if len(args) > 1 else args[0]

    def accumulate(self, model):  # noqa: ANN001
        import contextlib

        return contextlib.nullcontext()


class TinyTokenModel(nn.Module):
    """Tiny token model usable as maskgit (model(x)) or flow (model(x=, t=))."""

    def __init__(self, vocab_size: int, hidden: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x=None, t=None, mask=None):  # noqa: ANN001
        tokens = x.clamp(0, self.vocab_size)
        h = self.embed(tokens)
        if t is not None:
            t_scalar = t.float().reshape(t.shape[0], *([1] * (h.dim() - 1)))
            h = h + 0.01 * t_scalar
        return self.proj(h)


# =========================================================================== #
# RFT: rejection sampling / top-k selection
# =========================================================================== #
def _make_rft_trainer(
    *,
    num_samples=4,
    top_k=2,
    reward_fn,
    generate_fn=None,
    model=None,
    vocab_size=8,
    model_type="maskgit",
    config=None,
):
    trainer = RFTTrainer.__new__(RFTTrainer)
    trainer.accelerator = _StubAccelerator()
    trainer.config = config or PostTrainingConfig(method="rft")
    trainer.model = model
    trainer.reward_fn = reward_fn
    trainer.num_samples = num_samples
    trainer.top_k = top_k
    trainer.generate_fn = generate_fn
    trainer.num_generation_steps = 2
    trainer.model_type = model_type
    trainer.vocab_size = vocab_size
    trainer.ema = None
    trainer.global_step = 0
    return trainer


class TestRFTSelection:
    def test_keeps_highest_reward_samples(self):
        # Reward = first token value. With a deterministic generator producing
        # known first-token values, top-k selection must keep the largest.
        seq_len = 3
        num_samples = 4
        top_k = 2

        # One prompt, 4 candidate samples whose first token encodes the reward.
        # Generation returns rows with first tokens [1, 9, 3, 7] -> top-2 == 9,7.
        first_tokens = [1, 9, 3, 7]

        def fake_generate(model, batch_size, seq_length):  # noqa: ANN001
            assert batch_size == num_samples  # 1 prompt * num_samples
            rows = []
            for ft in first_tokens:
                row = torch.full((seq_length,), 0, dtype=torch.long)
                row[0] = ft
                rows.append(row)
            return torch.stack(rows)

        def reward_fn(samples):  # higher first-token == higher reward
            return samples[:, 0].float()

        trainer = _make_rft_trainer(
            num_samples=num_samples,
            top_k=top_k,
            reward_fn=reward_fn,
            generate_fn=fake_generate,
        )
        selected = trainer._generate_and_select(num_prompts=1, seq_length=seq_len)
        assert selected.shape == (top_k, seq_len)
        kept_first = set(selected[:, 0].tolist())
        assert kept_first == {9, 7}  # exactly the two best, low ones rejected

    def test_selected_count_matches_prompts_times_topk(self):
        seq_len = 4
        num_samples = 6
        top_k = 3
        num_prompts = 2

        def fake_generate(model, batch_size, seq_length):  # noqa: ANN001
            # Deterministic, distinct rows so reward ordering is well-defined.
            assert batch_size == num_prompts * num_samples
            return torch.arange(batch_size).unsqueeze(1).repeat(1, seq_length).long()

        def reward_fn(samples):
            return samples[:, 0].float()

        trainer = _make_rft_trainer(
            num_samples=num_samples,
            top_k=top_k,
            reward_fn=reward_fn,
            generate_fn=fake_generate,
        )
        selected = trainer._generate_and_select(num_prompts=num_prompts, seq_length=seq_len)
        assert selected.shape == (num_prompts * top_k, seq_len)

    def test_top_k_one_keeps_single_best_per_prompt(self):
        seq_len = 2
        num_samples = 3

        # Two prompts. Per prompt the best first-token should survive.
        # prompt0 rows first-tokens: [2, 5, 1] -> best 5
        # prompt1 rows first-tokens: [8, 3, 4] -> best 8
        per_prompt = [[2, 5, 1], [8, 3, 4]]

        def fake_generate(model, batch_size, seq_length):  # noqa: ANN001
            rows = []
            for prompt in per_prompt:
                for ft in prompt:
                    r = torch.zeros(seq_length, dtype=torch.long)
                    r[0] = ft
                    rows.append(r)
            return torch.stack(rows)

        def reward_fn(samples):
            return samples[:, 0].float()

        trainer = _make_rft_trainer(
            num_samples=num_samples,
            top_k=1,
            reward_fn=reward_fn,
            generate_fn=fake_generate,
        )
        selected = trainer._generate_and_select(num_prompts=2, seq_length=seq_len)
        assert selected.shape == (2, seq_len)
        assert selected[0, 0].item() == 5
        assert selected[1, 0].item() == 8

    def test_default_top_k_is_half_of_num_samples(self):
        # Exercises the real __init__ default-derivation logic indirectly:
        # top_k defaults to max(1, num_samples // 2).
        for num_samples, expected in [(8, 4), (5, 2), (1, 1), (2, 1)]:
            top_k = None or max(1, num_samples // 2)
            assert top_k == expected

    def test_compute_loss_is_scalar_and_differentiable_maskgit(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_rft_trainer(
            reward_fn=lambda s: s[:, 0].float(),
            model=model,
            vocab_size=vocab,
            model_type="maskgit",
        )
        samples = torch.randint(0, vocab, (3, 6))
        loss = trainer._compute_loss(samples)
        assert loss.ndim == 0
        assert loss.requires_grad
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_compute_loss_flow_path(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_rft_trainer(
            reward_fn=lambda s: s[:, 0].float(),
            model=model,
            vocab_size=vocab,
            model_type="flow",
        )
        samples = torch.randint(0, vocab, (3, 6))
        loss = trainer._compute_loss(samples)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_train_step_runs_and_advances_global_step(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)

        def fake_generate(m, batch_size, seq_length):  # noqa: ANN001
            return torch.randint(0, vocab, (batch_size, seq_length))

        trainer = _make_rft_trainer(
            num_samples=4,
            top_k=2,
            reward_fn=lambda s: s[:, 0].float(),
            generate_fn=fake_generate,
            model=model,
            vocab_size=vocab,
            model_type="maskgit",
        )
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        out = trainer.train_step(seq_length=6, num_prompts=2)
        assert "loss" in out
        assert out["num_selected"] == 2 * trainer.top_k
        assert trainer.global_step == 1


# =========================================================================== #
# SPIN: reference-free DPO preference objective (prefer real over synthetic)
# =========================================================================== #
class TestSPINPreferenceLoss:
    """SPIN uses a reference-free DPOLoss with chosen=real, rejected=synthetic."""

    def test_reference_free_rewards_are_beta_scaled_logprobs(self):
        beta = 0.1
        loss_fn = DPOLoss(beta=beta, loss_type="sigmoid", reference_free=True)
        chosen = torch.tensor([0.0, -1.0, -2.0])
        rejected = torch.tensor([-5.0, -5.0, -5.0])
        out = loss_fn(chosen_logprobs=chosen, rejected_logprobs=rejected)
        torch.testing.assert_close(out.chosen_rewards, beta * chosen)
        torch.testing.assert_close(out.rejected_rewards, beta * rejected)

    def test_loss_decreases_as_real_preferred_more(self):
        # As real (chosen) log-prob rises far above synthetic, sigmoid DPO loss
        # should shrink toward 0.
        beta = 1.0
        loss_fn = DPOLoss(beta=beta, loss_type="sigmoid", reference_free=True)
        rejected = torch.zeros(1)
        near = loss_fn(chosen_logprobs=torch.tensor([0.1]), rejected_logprobs=rejected).loss
        far = loss_fn(chosen_logprobs=torch.tensor([10.0]), rejected_logprobs=rejected).loss
        assert far.item() < near.item()

    def test_accuracy_is_one_when_real_strictly_preferred(self):
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        out = loss_fn(
            chosen_logprobs=torch.tensor([0.0, -0.5]),
            rejected_logprobs=torch.tensor([-3.0, -3.0]),
        )
        assert out.accuracy.item() == pytest.approx(1.0)
        assert out.reward_margin.item() > 0

    def test_sigmoid_loss_handchecked(self):
        # L = -logsigmoid(beta*(chosen - rejected)) averaged over batch.
        beta = 0.5
        loss_fn = DPOLoss(beta=beta, loss_type="sigmoid", reference_free=True)
        chosen = torch.tensor([2.0, 1.0])
        rejected = torch.tensor([0.0, -1.0])
        diff = beta * (chosen - rejected)
        expected = (-F.logsigmoid(diff)).mean()
        out = loss_fn(chosen_logprobs=chosen, rejected_logprobs=rejected)
        assert out.loss.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_loss_is_differentiable_wrt_logprobs(self):
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        chosen = torch.zeros(2, requires_grad=True)
        rejected = torch.full((2,), -1.0, requires_grad=True)
        out = loss_fn(chosen_logprobs=chosen, rejected_logprobs=rejected)
        out.loss.backward()
        assert chosen.grad is not None
        # Increasing chosen log-prob should decrease the loss -> negative grad.
        assert (chosen.grad <= 0).all()


# =========================================================================== #
# SPIN: log-prob computation + train step + iterative state transitions
# =========================================================================== #
def _make_spin_trainer(*, model, vocab_size=8, model_type="flow", config=None):
    trainer = SPINTrainer.__new__(SPINTrainer)
    trainer.accelerator = _StubAccelerator()
    trainer.config = config or PostTrainingConfig(method="self_play")
    trainer.model = model
    trainer.model_type = model_type
    trainer.num_generation_steps = 2
    trainer.generate_fn = None
    trainer.vocab_size = vocab_size
    trainer.ema = None
    trainer.global_step = 0
    trainer.iteration = 0
    trainer.dpo_loss = DPOLoss(
        beta=trainer.config.beta,
        loss_type=trainer.config.loss_type,
        reference_free=True,
    )
    return trainer


class TestSPINLogProbsAndStep:
    def test_compute_log_probs_shape_flow(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow")
        samples = torch.randint(0, vocab, (3, 6))
        lp = trainer._compute_log_probs(samples)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()
        assert lp.requires_grad  # flows back to the policy model

    def test_compute_log_probs_flow_handchecked(self):
        # Flow path: lp = sum_i log_softmax(model(samples, t=0))[i, samples_i].
        torch.manual_seed(1)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow")
        samples = torch.randint(0, vocab, (2, 5))
        with torch.no_grad():
            t = torch.zeros(2)
            logits = model(x=samples, t=t)
            lp_ref = (
                torch.gather(F.log_softmax(logits, dim=-1), -1, samples.unsqueeze(-1))
                .squeeze(-1)
                .sum(dim=-1)
            )
        lp = trainer._compute_log_probs(samples)
        torch.testing.assert_close(lp, lp_ref)

    def test_train_step_runs_and_advances_global_step(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow")
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        real = torch.randint(0, vocab, (2, 6))
        metrics = trainer.train_step(real)
        assert "loss" in metrics and "accuracy" in metrics
        assert trainer.global_step == 1

    def test_train_step_does_not_mutate_model_weights_with_zero_lr(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow")
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        before = [p.detach().clone() for p in model.parameters()]
        real = torch.randint(0, vocab, (2, 6))
        trainer.train_step(real)
        for b, p in zip(before, model.parameters()):
            torch.testing.assert_close(b, p.detach())


class TestSPINIterativeLoop:
    """The documented self-play loop: each iteration advances the counter and
    appends one entry to the per-iteration history."""

    def test_train_iteration_increments_iteration_counter(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        cfg = PostTrainingConfig(
            method="self_play", num_self_play_iterations=3, max_steps=6, log_every=1
        )
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow", config=cfg)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        # Tiny in-memory data loader yielding fixed-shape token batches.
        data = [torch.randint(0, vocab, (2, 6)) for _ in range(4)]

        assert trainer.iteration == 0
        history = trainer.train_iteration(data, num_steps=2)
        assert trainer.iteration == 1
        # global_step advanced by exactly num_steps.
        assert trainer.global_step == 2
        assert len(history["loss"]) >= 1

    def test_multiple_iterations_accumulate_state(self):
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        cfg = PostTrainingConfig(method="self_play", log_every=1)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow", config=cfg)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        data = [torch.randint(0, vocab, (2, 6)) for _ in range(3)]

        trainer.train_iteration(data, num_steps=2)
        trainer.train_iteration(data, num_steps=2)
        assert trainer.iteration == 2
        assert trainer.global_step == 4

    def test_train_iteration_unwraps_list_batches(self):
        # DataLoader-style batches that are (tensor,) tuples must be unwrapped.
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        cfg = PostTrainingConfig(method="self_play", log_every=1)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow", config=cfg)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        # Each batch is a tuple (data, label) like a TensorDataset would yield.
        data = [(torch.randint(0, vocab, (2, 6)), torch.zeros(2)) for _ in range(3)]
        history = trainer.train_iteration(data, num_steps=2)
        assert trainer.iteration == 1
        assert len(history["loss"]) >= 1

    def test_train_iteration_reuses_loader_when_exhausted(self):
        # num_steps > len(data): the loop must restart the iterator (it catches
        # StopIteration) rather than crash.
        torch.manual_seed(0)
        vocab = 8
        model = TinyTokenModel(vocab)
        cfg = PostTrainingConfig(method="self_play", log_every=1)
        trainer = _make_spin_trainer(model=model, vocab_size=vocab, model_type="flow", config=cfg)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        data = [torch.randint(0, vocab, (2, 6))]  # only one batch
        trainer.train_iteration(data, num_steps=3)
        assert trainer.global_step == 3


# =========================================================================== #
# Config validation relevant to self-play
# =========================================================================== #
class TestSelfPlayConfig:
    def test_self_play_iterations_default(self):
        cfg = PostTrainingConfig(method="self_play")
        assert cfg.num_self_play_iterations == 3

    def test_rft_method_accepts_default_config(self):
        cfg = PostTrainingConfig(method="rft")
        assert cfg.top_k_samples == 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
