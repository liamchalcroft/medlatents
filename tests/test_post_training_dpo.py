"""Tests for the post_training DPO subsystem.

Covers:
- DPOLoss math: shape, differentiability, monotonicity, beta scaling, reference role.
- Trainer construction + a real train_step for each DPO variant
  (autoregressive / discrete / flow / diffusion + SPO).
- Preference-pair handling (chosen vs rejected) and length/shape validation.
- Documented edge cases / input validation (pytest.raises).

All models/configs are tiny and run on CPU. We force ``use_fsdp=False`` and
``mixed_precision="no"`` so the accelerate ``Accelerator`` stays on the CPU
single-process path (no distributed init, no autocast surprises).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from medlatents.diffusion.d3pm import D3PM
from medlatents.flow_matching.core import (
    MixtureDiscreteProbPath,
    PolynomialConvexScheduler,
)
from medlatents.post_training import (
    AutoregressiveDPOTrainer,
    D3PMDPOTrainer,
    DiffusionDPOTrainer,
    DiscreteDPOTrainer,
    DiscreteFlowDPOTrainer,
    DPOLoss,
    DPOOutput,
    FlowDPOTrainer,
    PostTrainingConfig,
    PreferenceDataset,
    PreferencePair,
    SPOTrainer,
    StepPreferenceModel,
    collate_preference_pairs,
)

VOCAB = 16
SEQ = 6
BATCH = 2
HIDDEN = 16


@pytest.fixture(scope="module")
def cpu_accel():
    """A single CPU-pinned Accelerator shared by all trainers in this module.

    accelerate keeps process-global state, and this sandbox may expose a GPU.
    Forcing ``cpu=True`` keeps every model + buffer on CPU so our hand-built
    CPU input tensors line up with the trainer's parameters.
    """
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, PartialState

    # AcceleratorState is a process-global singleton; another test module may
    # have initialized it (on GPU in some environments), which would make
    # Accelerator(cpu=True) raise "already initialized". Reset it so this module
    # runs on CPU, and reset again on teardown so later modules start clean.
    AcceleratorState._reset_state()
    PartialState._reset_state()
    accel = Accelerator(cpu=True)
    yield accel
    AcceleratorState._reset_state()
    PartialState._reset_state()


# --------------------------------------------------------------------------- #
# Tiny stand-in models matching each trainer's call convention.
# --------------------------------------------------------------------------- #
class TinyAutoregModel(nn.Module):
    """Causal-style model: forward(tokens) -> [B, T, vocab].

    The class name contains "autoreg" so DiscreteDPOTrainer auto-detects it.
    """

    def __init__(self, vocab_size: int = VOCAB, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.causal = True
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(self.embed(tokens))


class TinyTXModel(nn.Module):
    """Model that takes x=, t= keyword args and predicts ``out_vocab`` logits.

    Used for flow (out_vocab == vocab, inputs may include the mask token =
    vocab, so the embedding is sized vocab+1) and diffusion (out_vocab ==
    effective_num_classes).
    """

    def __init__(self, vocab_size: int, out_vocab: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # +1 so a mask token at index ``vocab_size`` is always embeddable.
        self.embed = nn.Embedding(out_vocab + 1, hidden)
        self.proj = nn.Linear(hidden, out_vocab)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        del t  # timestep unused by this stub
        return self.proj(self.embed(x))


def make_cpu_config(**overrides) -> PostTrainingConfig:
    """A DPO config that stays on the CPU single-process path."""
    kwargs = {
        "method": "dpo",
        "beta": 0.1,
        "use_fsdp": False,
        "mixed_precision": "no",
        "use_ema": False,
        "gradient_accumulation_steps": 1,
        "batch_size": BATCH,
        "seed": 0,
    }
    kwargs.update(overrides)
    return PostTrainingConfig(**kwargs)


def make_pref_dataset(n_pairs: int = 4, seq: int = SEQ, vocab: int = VOCAB) -> PreferenceDataset:
    torch.manual_seed(123)
    pairs = []
    for _ in range(n_pairs):
        chosen = torch.randint(0, vocab, (seq,))
        rejected = torch.randint(0, vocab, (seq,))
        pairs.append(PreferencePair(chosen=chosen, rejected=rejected, margin=1.0))
    return PreferenceDataset(pairs=pairs)


def make_batch(seq: int = SEQ, vocab: int = VOCAB, batch: int = BATCH) -> dict:
    """A collated batch in the shape the trainers' prepare_batch expects."""
    torch.manual_seed(7)
    pairs = [
        PreferencePair(
            chosen=torch.randint(0, vocab, (seq,)),
            rejected=torch.randint(0, vocab, (seq,)),
            margin=1.0,
        )
        for _ in range(batch)
    ]
    return collate_preference_pairs(pairs)


# ===========================================================================
# 1. DPOLoss math
# ===========================================================================
class TestDPOLossMath:
    def test_sigmoid_loss_is_scalar_and_finite(self):
        torch.manual_seed(0)
        loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")
        chosen = torch.randn(BATCH)
        rejected = torch.randn(BATCH)
        ref_c = torch.randn(BATCH)
        ref_r = torch.randn(BATCH)
        out = loss_fn(chosen, rejected, ref_c, ref_r)
        assert isinstance(out, DPOOutput)
        assert out.loss.ndim == 0  # scalar
        assert torch.isfinite(out.loss)
        # per-sample reward tensors keep the batch dimension
        assert out.chosen_rewards.shape == (BATCH,)
        assert out.rejected_rewards.shape == (BATCH,)

    def test_loss_is_differentiable_wrt_policy_logprobs(self):
        loss_fn = DPOLoss(beta=0.2, loss_type="sigmoid")
        chosen = torch.randn(BATCH, requires_grad=True)
        rejected = torch.randn(BATCH, requires_grad=True)
        ref_c = torch.randn(BATCH)
        ref_r = torch.randn(BATCH)
        out = loss_fn(chosen, rejected, ref_c, ref_r)
        out.loss.backward()
        assert chosen.grad is not None and torch.isfinite(chosen.grad).all()
        assert rejected.grad is not None and torch.isfinite(rejected.grad).all()
        # Increasing chosen logprob should decrease the loss -> negative grad.
        assert (chosen.grad <= 0).all()
        # Increasing rejected logprob should increase the loss -> positive grad.
        assert (rejected.grad >= 0).all()

    def test_higher_reward_gap_gives_lower_loss(self):
        """Preferred completions with a larger reward gap -> lower DPO loss."""
        loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")
        ref_c = torch.zeros(BATCH)
        ref_r = torch.zeros(BATCH)
        # Strongly-preferred chosen.
        good = loss_fn(
            chosen_logprobs=torch.full((BATCH,), 5.0),
            rejected_logprobs=torch.full((BATCH,), -5.0),
            ref_chosen_logprobs=ref_c,
            ref_rejected_logprobs=ref_r,
        )
        # Reversed preference (chosen worse than rejected).
        bad = loss_fn(
            chosen_logprobs=torch.full((BATCH,), -5.0),
            rejected_logprobs=torch.full((BATCH,), 5.0),
            ref_chosen_logprobs=ref_c,
            ref_rejected_logprobs=ref_r,
        )
        assert good.loss.item() < bad.loss.item()
        # Accuracy / margin metrics should reflect the ordering.
        assert good.accuracy.item() == 1.0
        assert bad.accuracy.item() == 0.0
        assert good.reward_margin.item() > 0
        assert bad.reward_margin.item() < 0

    def test_beta_scales_the_logit_gap(self):
        """The implicit reward gap is beta * (logprob gap)."""
        chosen = torch.tensor([2.0, 2.0])
        rejected = torch.tensor([0.0, 0.0])
        ref = torch.zeros(BATCH)
        out_small = DPOLoss(beta=0.1)(chosen, rejected, ref, ref)
        out_large = DPOLoss(beta=0.5)(chosen, rejected, ref, ref)
        gap_small = out_small.chosen_rewards - out_small.rejected_rewards
        gap_large = out_large.chosen_rewards - out_large.rejected_rewards
        # beta=0.1 -> gap 0.2 ; beta=0.5 -> gap 1.0
        assert torch.allclose(gap_small, torch.full((BATCH,), 0.2), atol=1e-6)
        assert torch.allclose(gap_large, torch.full((BATCH,), 1.0), atol=1e-6)
        # Larger gap (same sign) -> strictly smaller sigmoid loss.
        assert out_large.loss.item() < out_small.loss.item()

    def test_reference_model_changes_rewards(self):
        """The reference subtracts off its log-probs (KL anchor)."""
        chosen = torch.tensor([3.0, 3.0])
        rejected = torch.tensor([1.0, 1.0])
        ref_c = torch.tensor([2.0, 2.0])
        ref_r = torch.tensor([0.5, 0.5])
        loss_fn = DPOLoss(beta=1.0)
        out = loss_fn(chosen, rejected, ref_c, ref_r)
        # r = beta * (pi - pi_ref)
        assert torch.allclose(out.chosen_rewards, chosen - ref_c, atol=1e-6)
        assert torch.allclose(out.rejected_rewards, rejected - ref_r, atol=1e-6)

    def test_reference_free_uses_raw_logprobs(self):
        chosen = torch.tensor([3.0, 3.0])
        rejected = torch.tensor([1.0, 1.0])
        loss_fn = DPOLoss(beta=1.0, reference_free=True)
        out = loss_fn(chosen, rejected)  # no reference needed
        assert torch.allclose(out.chosen_rewards, chosen, atol=1e-6)
        assert torch.allclose(out.rejected_rewards, rejected, atol=1e-6)

    def test_missing_reference_raises(self):
        loss_fn = DPOLoss(beta=0.1, reference_free=False)
        with pytest.raises(ValueError, match="Reference log probs required"):
            loss_fn(torch.randn(BATCH), torch.randn(BATCH))

    def test_unknown_loss_type_raises(self):
        # Bypass the Literal typing to feed an invalid loss type at runtime.
        loss_fn = DPOLoss(beta=0.1)
        loss_fn.loss_type = "not_a_real_loss"
        with pytest.raises(ValueError, match="Unknown loss type"):
            loss_fn(
                torch.randn(BATCH),
                torch.randn(BATCH),
                torch.zeros(BATCH),
                torch.zeros(BATCH),
            )

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo", "kto", "disco"])
    def test_all_loss_types_scalar_and_differentiable(self, loss_type):
        loss_fn = DPOLoss(beta=0.1, loss_type=loss_type)
        chosen = torch.randn(BATCH, requires_grad=True)
        rejected = torch.randn(BATCH, requires_grad=True)
        ref_c = torch.randn(BATCH)
        ref_r = torch.randn(BATCH)
        out = loss_fn(chosen, rejected, ref_c, ref_r)
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss)
        out.loss.backward()
        assert chosen.grad is not None
        assert torch.isfinite(chosen.grad).all()

    def test_hinge_loss_saturates_to_zero(self):
        """Hinge loss is exactly 0 once the (beta-scaled) gap exceeds 1."""
        loss_fn = DPOLoss(beta=1.0, loss_type="hinge")
        # gap = beta * (10 - 0) = 10 > 1 -> relu(1 - 10) == 0
        out = loss_fn(
            torch.full((BATCH,), 10.0),
            torch.zeros(BATCH),
            torch.zeros(BATCH),
            torch.zeros(BATCH),
        )
        assert out.loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_label_smoothing_path_runs(self):
        loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid", label_smoothing=0.1)
        out = loss_fn(
            torch.randn(BATCH, requires_grad=True),
            torch.randn(BATCH),
            torch.zeros(BATCH),
            torch.zeros(BATCH),
        )
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss)

    def test_margin_scales_reward_diff(self):
        """A per-sample margin multiplies the reward difference inside the loss.

        With margin 0 the effective reward_diff is 0 -> sigmoid loss == log 2.
        """
        loss_fn = DPOLoss(beta=1.0, loss_type="sigmoid")
        out = loss_fn(
            chosen_logprobs=torch.full((BATCH,), 5.0),
            rejected_logprobs=torch.full((BATCH,), -5.0),
            ref_chosen_logprobs=torch.zeros(BATCH),
            ref_rejected_logprobs=torch.zeros(BATCH),
            margin=torch.zeros(BATCH),
        )
        import math

        assert out.loss.item() == pytest.approx(math.log(2.0), abs=1e-5)


# ===========================================================================
# 2. DPOOutput
# ===========================================================================
def test_dpo_output_to_dict():
    out = DPOOutput(
        loss=torch.tensor(0.5),
        chosen_rewards=torch.tensor([1.0, 2.0]),
        rejected_rewards=torch.tensor([0.0, 1.0]),
        accuracy=torch.tensor(1.0),
        reward_margin=torch.tensor(1.0),
    )
    d = out.to_dict()
    assert set(d) == {"loss", "accuracy", "reward_margin", "chosen_rewards", "rejected_rewards"}
    assert d["loss"] == pytest.approx(0.5)
    assert d["chosen_rewards"] == pytest.approx(1.5)  # mean of [1, 2]
    assert all(isinstance(v, float) for v in d.values())


# ===========================================================================
# 3. Preference data / collation
# ===========================================================================
class TestPreferenceData:
    def test_pair_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            PreferencePair(chosen=torch.zeros(SEQ), rejected=torch.zeros(SEQ + 1))

    def test_pair_margin_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Margin must be in"):
            PreferencePair(chosen=torch.zeros(SEQ), rejected=torch.zeros(SEQ), margin=2.0)

    def test_collate_stacks_chosen_rejected_and_margin(self):
        batch = make_batch()
        assert batch["chosen"].shape == (BATCH, SEQ)
        assert batch["rejected"].shape == (BATCH, SEQ)
        assert batch["margin"].shape == (BATCH,)
        assert len(batch["prompt"]) == BATCH

    def test_collate_preserves_chosen_vs_rejected_identity(self):
        """The collate must not swap chosen/rejected ordering."""
        c0 = torch.tensor([1, 2, 3, 4])
        r0 = torch.tensor([5, 6, 7, 8])
        c1 = torch.tensor([9, 10, 11, 12])
        r1 = torch.tensor([13, 14, 15, 0])
        pairs = [
            PreferencePair(chosen=c0, rejected=r0),
            PreferencePair(chosen=c1, rejected=r1),
        ]
        batch = collate_preference_pairs(pairs)
        assert torch.equal(batch["chosen"][0], c0)
        assert torch.equal(batch["rejected"][0], r0)
        assert torch.equal(batch["chosen"][1], c1)
        assert torch.equal(batch["rejected"][1], r1)


# ===========================================================================
# 4. Autoregressive DPO trainer
# ===========================================================================
class TestAutoregressiveDPOTrainer:
    def test_compute_logprobs_shape_and_sign(self, cpu_accel):
        model = TinyAutoregModel()
        cfg = make_cpu_config()
        ref = copy.deepcopy(model)
        trainer = AutoregressiveDPOTrainer(model, ref, cfg, accelerator=cpu_accel, vocab_size=VOCAB)
        samples = torch.randint(0, VOCAB, (BATCH, SEQ))
        lp = trainer.compute_logprobs(trainer.model, samples)
        assert lp.shape == (BATCH,)
        # Sum of (SEQ-1) log-probs -> strictly negative & finite.
        assert torch.isfinite(lp).all()
        assert (lp < 0).all()

    def test_compute_logprobs_matches_manual_autoregressive(self, cpu_accel):
        torch.manual_seed(1)
        model = TinyAutoregModel()
        cfg = make_cpu_config()
        trainer = AutoregressiveDPOTrainer(
            model, None, cfg, accelerator=cpu_accel, vocab_size=VOCAB
        )
        samples = torch.randint(0, VOCAB, (BATCH, SEQ))
        lp = trainer.compute_logprobs(trainer.model, samples)

        inputs, targets = samples[:, :-1], samples[:, 1:]
        logits = trainer.model(inputs)
        manual = (
            torch.gather(F.log_softmax(logits, dim=-1), -1, targets.unsqueeze(-1))
            .squeeze(-1)
            .sum(dim=-1)
        )
        assert torch.allclose(lp, manual, atol=1e-5)

    def test_train_step_runs_and_returns_metrics(self, cpu_accel):
        model = TinyAutoregModel()
        ref = copy.deepcopy(model)
        cfg = make_cpu_config()
        trainer = AutoregressiveDPOTrainer(model, ref, cfg, accelerator=cpu_accel, vocab_size=VOCAB)
        metrics = trainer.train_step(make_batch())
        assert {"loss", "accuracy", "reward_margin"}.issubset(metrics)
        assert all(isinstance(v, float) for v in metrics.values())

    def test_train_step_updates_policy_not_reference(self, cpu_accel):
        torch.manual_seed(2)
        model = TinyAutoregModel()
        ref = copy.deepcopy(model)
        cfg = make_cpu_config(lr=1e-1)  # large LR so the change is visible
        trainer = AutoregressiveDPOTrainer(model, ref, cfg, accelerator=cpu_accel, vocab_size=VOCAB)

        policy_before = trainer.model.proj.weight.detach().clone()
        ref_before = trainer.ref_model.proj.weight.detach().clone()
        trainer.train_step(make_batch())
        # Policy moved; reference is frozen.
        assert not torch.allclose(policy_before, trainer.model.proj.weight)
        assert torch.allclose(ref_before, trainer.ref_model.proj.weight)

    def test_reference_params_are_frozen(self, cpu_accel):
        model = TinyAutoregModel()
        ref = copy.deepcopy(model)
        trainer = AutoregressiveDPOTrainer(
            model, ref, make_cpu_config(), accelerator=cpu_accel, vocab_size=VOCAB
        )
        assert all(not p.requires_grad for p in trainer.ref_model.parameters())

    def test_reference_free_when_no_ref_model(self, cpu_accel):
        model = TinyAutoregModel()
        trainer = AutoregressiveDPOTrainer(
            model, None, make_cpu_config(), accelerator=cpu_accel, vocab_size=VOCAB
        )
        assert trainer.reference_free is True
        # train_step still works without a reference model.
        metrics = trainer.train_step(make_batch())
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_missing_vocab_size_raises(self, cpu_accel):
        # Has params (so optimizer construction succeeds) but no vocab_size attr.
        class NoVocab(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

            def forward(self, x, **kw):
                return x

        with pytest.raises(ValueError, match="vocab_size"):
            AutoregressiveDPOTrainer(NoVocab(), None, make_cpu_config(), accelerator=cpu_accel)

    def test_full_train_loop_short(self, tmp_path, cpu_accel):
        model = TinyAutoregModel()
        ref = copy.deepcopy(model)
        cfg = make_cpu_config(max_steps=2, warmup_steps=1, log_every=1, logdir=str(tmp_path))
        trainer = AutoregressiveDPOTrainer(model, ref, cfg, accelerator=cpu_accel, vocab_size=VOCAB)
        history = trainer.train(make_pref_dataset(n_pairs=BATCH * 2))
        assert trainer.global_step == 2
        assert len(history["train_loss"]) >= 1


# ===========================================================================
# 5. DiscreteDPOTrainer (auto-detect)
# ===========================================================================
class TestDiscreteDPOTrainer:
    def test_autodetect_autoreg(self, cpu_accel):
        model = TinyAutoregModel()  # class name contains "autoreg"
        trainer = DiscreteDPOTrainer(
            model, None, make_cpu_config(), accelerator=cpu_accel, vocab_size=VOCAB
        )
        assert trainer.model_type == "autoreg"

    def test_explicit_model_type(self, cpu_accel):
        model = TinyAutoregModel()
        trainer = DiscreteDPOTrainer(
            model,
            None,
            make_cpu_config(),
            accelerator=cpu_accel,
            model_type="autoreg",
            vocab_size=VOCAB,
        )
        assert trainer.model_type == "autoreg"

    def test_unknown_model_type_raises(self, cpu_accel):
        model = TinyAutoregModel()
        with pytest.raises(ValueError, match="Unknown model type"):
            DiscreteDPOTrainer(
                model,
                None,
                make_cpu_config(),
                accelerator=cpu_accel,
                model_type="bogus",
                vocab_size=VOCAB,
            )

    def test_delegated_compute_logprobs(self, cpu_accel):
        model = TinyAutoregModel()
        trainer = DiscreteDPOTrainer(
            model, None, make_cpu_config(), accelerator=cpu_accel, vocab_size=VOCAB
        )
        lp = trainer.compute_logprobs(trainer.model, torch.randint(0, VOCAB, (BATCH, SEQ)))
        assert lp.shape == (BATCH,)


# ===========================================================================
# 6. Flow DPO trainers
# ===========================================================================
def _flow_path() -> MixtureDiscreteProbPath:
    return MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))


class TestFlowDPOTrainer:
    def test_discrete_flow_compute_logprobs_shape(self, cpu_accel):
        torch.manual_seed(3)
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        ref = copy.deepcopy(model)
        cfg = make_cpu_config()
        trainer = DiscreteFlowDPOTrainer(
            model,
            ref,
            cfg,
            path=_flow_path(),
            accelerator=cpu_accel,
            vocab_size=VOCAB,
            num_timestep_samples=2,
        )
        samples = torch.randint(0, VOCAB, (BATCH, SEQ))
        lp = trainer.compute_logprobs(trainer.model, samples)
        assert lp.shape == (BATCH,)
        # logprob = -average CE over sampled timesteps -> negative.
        assert (lp <= 0).all()
        assert torch.isfinite(lp).all()

    def test_default_mask_token_is_vocab_size(self, cpu_accel):
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        trainer = DiscreteFlowDPOTrainer(
            model,
            None,
            make_cpu_config(),
            path=_flow_path(),
            accelerator=cpu_accel,
            vocab_size=VOCAB,
        )
        # model has no mask_token/special_tokens -> default convention.
        assert trainer.mask_token == VOCAB

    @pytest.mark.parametrize("strategy", ["trajectory", "endpoint", "importance"])
    def test_likelihood_strategies(self, strategy, cpu_accel):
        torch.manual_seed(4)
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        trainer = DiscreteFlowDPOTrainer(
            model,
            None,
            make_cpu_config(),
            path=_flow_path(),
            accelerator=cpu_accel,
            vocab_size=VOCAB,
            num_timestep_samples=2,
            likelihood_strategy=strategy,
        )
        lp = trainer.compute_logprobs(trainer.model, torch.randint(0, VOCAB, (BATCH, SEQ)))
        assert lp.shape == (BATCH,)
        assert torch.isfinite(lp).all()

    def test_unknown_strategy_raises(self, cpu_accel):
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        trainer = DiscreteFlowDPOTrainer(
            model,
            None,
            make_cpu_config(),
            path=_flow_path(),
            accelerator=cpu_accel,
            vocab_size=VOCAB,
        )
        trainer.likelihood_strategy = "nope"
        with pytest.raises(ValueError, match="Unknown likelihood strategy"):
            trainer.compute_logprobs(trainer.model, torch.randint(0, VOCAB, (BATCH, SEQ)))

    def test_missing_vocab_size_raises(self, cpu_accel):
        # Has params (so optimizer construction succeeds) but no vocab_size attr.
        class NoVocab(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

            def forward(self, x, t, **kw):
                return x

        with pytest.raises(ValueError, match="vocab_size must be provided"):
            DiscreteFlowDPOTrainer(
                NoVocab(), None, make_cpu_config(), path=_flow_path(), accelerator=cpu_accel
            )

    def test_train_step_differentiates_policy(self, cpu_accel):
        torch.manual_seed(5)
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        ref = copy.deepcopy(model)
        cfg = make_cpu_config(lr=1e-1)
        trainer = DiscreteFlowDPOTrainer(
            model,
            ref,
            cfg,
            path=_flow_path(),
            accelerator=cpu_accel,
            vocab_size=VOCAB,
            num_timestep_samples=2,
        )
        before = trainer.model.proj.weight.detach().clone()
        metrics = trainer.train_step(make_batch())
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert not torch.allclose(before, trainer.model.proj.weight)

    def test_flow_wrapper_creates_default_path(self, cpu_accel):
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        trainer = FlowDPOTrainer(
            model,
            None,
            make_cpu_config(),
            path=None,
            accelerator=cpu_accel,
            vocab_size=VOCAB,
        )
        assert isinstance(trainer.path, MixtureDiscreteProbPath)
        lp = trainer.compute_logprobs(trainer.model, torch.randint(0, VOCAB, (BATCH, SEQ)))
        assert lp.shape == (BATCH,)


# ===========================================================================
# 7. Diffusion (D3PM) DPO trainers
# ===========================================================================
def _make_d3pm() -> D3PM:
    # NOTE: We use ``uniform`` transitions here on purpose. The ``absorbing``
    # (default) transition type drives D3PM down its matrix-free path where
    # Q_t / Q_bar_t / Q_bar_t_minus_1 are None, and
    # D3PMDPOTrainer._move_d3pm_to_device() unconditionally calls
    # ``self.d3pm.Q_t.to(device)`` -> AttributeError. See
    # ``test_absorbing_d3pm_is_unsupported_by_dpo_trainer`` which pins that.
    # For uniform, effective_num_classes == num_classes == VOCAB.
    return D3PM(num_classes=VOCAB, num_timesteps=5, transition_type="uniform")


class TestDiffusionDPOTrainer:
    def test_d3pm_compute_logprobs_shape(self, cpu_accel):
        torch.manual_seed(6)
        d3pm = _make_d3pm()
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=d3pm.effective_num_classes)
        ref = copy.deepcopy(model)
        trainer = D3PMDPOTrainer(
            model,
            ref,
            make_cpu_config(),
            d3pm=d3pm,
            accelerator=cpu_accel,
            num_timestep_samples=2,
        )
        samples = torch.randint(0, VOCAB, (BATCH, SEQ))
        lp = trainer.compute_logprobs(trainer.model, samples)
        assert lp.shape == (BATCH,)
        assert torch.isfinite(lp).all()
        # log-prob is -ELBO; ELBO (a KL sum) is non-negative -> logprob <= 0.
        assert (lp <= 1e-4).all()

    def test_d3pm_moves_schedule_tensors(self, cpu_accel):
        d3pm = _make_d3pm()
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=d3pm.effective_num_classes)
        trainer = D3PMDPOTrainer(model, None, make_cpu_config(), d3pm=d3pm, accelerator=cpu_accel)
        # The trainer copies the accelerator device onto the d3pm object.
        assert trainer.d3pm.device == trainer.accelerator.device
        assert trainer.d3pm.alphas_cumprod.device.type == trainer.accelerator.device.type

    def test_absorbing_d3pm_dpo_trainer_constructs(self, cpu_accel):
        # Regression: absorbing-state D3PM uses a matrix-free path where the
        # transition matrices (Q_t, Q_bar_t, Q_bar_t_minus_1) are None. The DPO
        # trainer must construct without trying to .to() those None matrices.
        d3pm = D3PM(num_classes=VOCAB, num_timesteps=5, transition_type="absorbing")
        assert d3pm.Q_t is None  # confirms the matrix-free path
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=d3pm.effective_num_classes)
        trainer = D3PMDPOTrainer(model, None, make_cpu_config(), d3pm=d3pm, accelerator=cpu_accel)
        assert trainer.d3pm.Q_t is None  # still matrix-free after the device move

    def test_full_elbo_mode_runs(self, cpu_accel):
        torch.manual_seed(8)
        d3pm = _make_d3pm()
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=d3pm.effective_num_classes)
        trainer = D3PMDPOTrainer(
            model, None, make_cpu_config(), d3pm=d3pm, accelerator=cpu_accel, elbo_mode="full"
        )
        lp = trainer.compute_logprobs(trainer.model, torch.randint(0, VOCAB, (BATCH, SEQ)))
        assert lp.shape == (BATCH,)
        assert torch.isfinite(lp).all()

    def test_diffusion_wrapper_train_step(self, cpu_accel):
        torch.manual_seed(9)
        d3pm = _make_d3pm()
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=d3pm.effective_num_classes)
        ref = copy.deepcopy(model)
        cfg = make_cpu_config(lr=1e-1)
        trainer = DiffusionDPOTrainer(
            model, ref, cfg, diffusion=d3pm, accelerator=cpu_accel, num_timestep_samples=2
        )
        before = trainer.model.proj.weight.detach().clone()
        metrics = trainer.train_step(make_batch())
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert not torch.allclose(before, trainer.model.proj.weight)


# ===========================================================================
# 8. SPO trainer + StepPreferenceModel
# ===========================================================================
class TestSPO:
    def test_step_preference_model_scores_shape(self):
        torch.manual_seed(10)
        scorer = StepPreferenceModel(
            hidden_size=HIDDEN, depth=1, num_heads=2, vocab_size=VOCAB, seq_length=SEQ
        )
        x = torch.randint(0, VOCAB, (BATCH, SEQ))
        t = torch.rand(BATCH)
        scores = scorer(x, t)
        assert scores.shape == (BATCH, 1)
        assert torch.isfinite(scores).all()

    def _make_spo_trainer(self, cpu_accel) -> SPOTrainer:
        torch.manual_seed(11)
        model = TinyTXModel(vocab_size=VOCAB, out_vocab=VOCAB)
        scorer = StepPreferenceModel(
            hidden_size=HIDDEN, depth=1, num_heads=2, vocab_size=VOCAB, seq_length=SEQ
        )
        cfg = make_cpu_config(method="spo", num_candidates=3)
        return SPOTrainer(
            model,
            cfg,
            flow_path=_flow_path(),
            step_scorer=scorer,
            accelerator=cpu_accel,
            vocab_size=VOCAB,
            num_candidates=3,
            num_steps=3,
        )

    def test_spo_construction_detects_flow(self, cpu_accel):
        trainer = self._make_spo_trainer(cpu_accel)
        assert trainer.model_type == "flow"
        assert trainer.num_candidates == 3

    def test_spo_step_dpo_loss_shapes(self, cpu_accel):
        trainer = self._make_spo_trainer(cpu_accel)
        k = trainer.num_candidates
        candidates = torch.randint(0, VOCAB, (BATCH * k, SEQ))
        # Make candidate 0 best, candidate k-1 worst, deterministically.
        scores = torch.zeros(BATCH, k)
        scores[:, 0] = 10.0
        scores[:, -1] = -10.0
        t = torch.rand(BATCH)
        loss, continue_state, acc = trainer._compute_step_dpo_loss(candidates, scores, t, k)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert continue_state.shape == (BATCH, SEQ)
        assert 0.0 <= acc <= 1.0

    def test_spo_train_step_runs(self, cpu_accel):
        trainer = self._make_spo_trainer(cpu_accel)
        metrics = trainer.train_step(seq_length=SEQ)
        assert {"loss", "accuracy", "steps_computed"}.issubset(metrics)
        assert metrics["steps_computed"] >= 1
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    def test_spo_config_requires_two_candidates(self):
        with pytest.raises(ValueError, match="at least 2 candidates"):
            PostTrainingConfig(method="spo", num_candidates=1, use_fsdp=False)


# ===========================================================================
# 9. Config validation
# ===========================================================================
class TestConfig:
    def test_rl_methods_require_two_samples(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            PostTrainingConfig(method="grpo", num_samples_per_prompt=1)

    def test_consistency_requires_teacher(self):
        with pytest.raises(ValueError, match="requires a teacher"):
            PostTrainingConfig(method="consistency", teacher_path=None)

    def test_from_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            PostTrainingConfig.from_preset("does_not_exist")

    def test_roundtrip_to_from_dict(self):
        cfg = make_cpu_config(beta=0.33)
        cfg2 = PostTrainingConfig.from_dict(cfg.to_dict())
        assert cfg2.beta == pytest.approx(0.33)
        assert cfg2.method == "dpo"
