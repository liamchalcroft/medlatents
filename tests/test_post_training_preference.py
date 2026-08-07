"""Tests for the post-training preference subsystem.

Covers:
- reward_model.py: RewardModel forward pass (discrete + continuous), scoring
  shape, Bradley-Terry preference loss + accuracy, timestep-aware path,
  create_reward_model factory, and a short RewardModelTrainer learning run.
- data.py: PreferencePair validation, RankedSamples.to_pairs strategies,
  PreferenceDataset construction / indexing / iteration / split,
  and collate_preference_pairs batch shapes.
- ai_feedback.py: DiscriminatorScorer + AIPreferenceScorer + StepAwarePreferenceModel
  scoring / ranking / pairing APIs, mocking external models with tiny stub
  nn.Modules (NO network / remote calls).

CPU-only, TINY models, seeded. CUDA-only paths guarded with skipif.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from medlatents.post_training.preference.ai_feedback import (
    AIPreferenceScorer,
    DiscriminatorScorer,
    StepAwarePreferenceModel,
)
from medlatents.post_training.preference.data import (
    PreferenceDataset,
    PreferencePair,
    RankedSamples,
    StepwisePreference,
    collate_preference_pairs,
)
from medlatents.post_training.preference.reward_model import (
    RewardModel,
    RewardModelTrainer,
    create_reward_model,
)

CUDA_AVAILABLE = torch.cuda.is_available()

# Small shared dims for tiny reward models.
VOCAB = 32
SEQ = 12
HIDDEN = 32
HEADS = 4
DEPTH = 2


def _tiny_discrete_reward_model(**kw):
    defaults = {
        "vocab_size": VOCAB,
        "seq_length": SEQ,
        "hidden_size": HIDDEN,
        "depth": DEPTH,
        "num_heads": HEADS,
        "dropout": 0.0,
    }
    defaults.update(kw)
    return RewardModel(**defaults)


# ---------------------------------------------------------------------------
# RewardModel: construction validation
# ---------------------------------------------------------------------------


def test_reward_model_requires_exactly_one_input_mode():
    with pytest.raises(ValueError, match="either vocab_size"):
        RewardModel(vocab_size=None, in_channels=None)
    with pytest.raises(ValueError, match="only one"):
        RewardModel(vocab_size=VOCAB, in_channels=4)


# ---------------------------------------------------------------------------
# RewardModel: forward / scoring shape + range
# ---------------------------------------------------------------------------


def test_reward_model_discrete_forward_shape():
    torch.manual_seed(0)
    model = _tiny_discrete_reward_model()
    model.eval()
    x = torch.randint(0, VOCAB, (5, SEQ))
    scores = model(x)
    assert scores.shape == (5, 1)
    assert torch.isfinite(scores).all()


def test_reward_model_continuous_forward_shape():
    torch.manual_seed(0)
    model = RewardModel(
        in_channels=6, seq_length=SEQ, hidden_size=HIDDEN, depth=DEPTH, num_heads=HEADS, dropout=0.0
    )
    model.eval()
    x = torch.randn(4, SEQ, 6)
    scores = model(x)
    assert scores.shape == (4, 1)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("pooling", ["mean", "cls", "last"])
def test_reward_model_pooling_variants(pooling):
    torch.manual_seed(1)
    model = _tiny_discrete_reward_model(pooling=pooling)
    model.eval()
    x = torch.randint(0, VOCAB, (3, SEQ))
    scores = model(x)
    assert scores.shape == (3, 1)
    assert torch.isfinite(scores).all()
    if pooling == "cls":
        assert model.cls_token is not None
    else:
        assert model.cls_token is None


def test_reward_model_zero_init_head_gives_zero_scores():
    """Output head is zero-initialised, so an untrained model scores exactly 0."""
    torch.manual_seed(2)
    model = _tiny_discrete_reward_model()
    model.eval()
    x = torch.randint(0, VOCAB, (4, SEQ))
    scores = model(x)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-6)


def test_reward_model_is_differentiable():
    torch.manual_seed(3)
    model = _tiny_discrete_reward_model()
    model.train()
    x = torch.randint(0, VOCAB, (4, SEQ))
    scores = model(x)
    loss = scores.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g != 0) for g in grads)


def test_reward_model_timestep_aware_forward():
    torch.manual_seed(4)
    model = _tiny_discrete_reward_model(timestep_aware=True)
    model.eval()
    assert model.t_embedder is not None
    x = torch.randint(0, VOCAB, (3, SEQ))
    t = torch.rand(3)
    scores = model(x, t)
    assert scores.shape == (3, 1)
    assert torch.isfinite(scores).all()


def test_reward_model_dynamic_seq_length_updates_rope_cache():
    """Sequences longer than the configured seq_length should trigger RoPE recompute.

    The embedders are built with allow_dynamic_seq_length=True so a longer
    sequence is accepted; this exercises _update_rope_cache.
    """
    torch.manual_seed(5)
    model = _tiny_discrete_reward_model(seq_length=SEQ)
    model.eval()
    longer = SEQ + 8
    x = torch.randint(0, VOCAB, (2, longer))
    scores = model(x)
    assert scores.shape == (2, 1)
    assert model.seq_len_cached >= longer + 1


# ---------------------------------------------------------------------------
# RewardModel: Bradley-Terry preference loss
# ---------------------------------------------------------------------------


def test_preference_loss_keys_and_shapes():
    torch.manual_seed(6)
    model = _tiny_discrete_reward_model()
    chosen = torch.randint(0, VOCAB, (4, SEQ))
    rejected = torch.randint(0, VOCAB, (4, SEQ))
    out = model.compute_preference_loss(chosen, rejected)
    for k in ["loss", "accuracy", "reward_margin", "chosen_reward", "rejected_reward"]:
        assert k in out
        assert out[k].shape == ()
    assert torch.isfinite(out["loss"])
    # At zero-init both rewards are 0 => diff 0 => loss = -log(sigmoid(0)) = log 2.
    assert torch.isclose(out["loss"], torch.tensor(0.6931472), atol=1e-4)
    assert torch.isclose(out["reward_margin"], torch.tensor(0.0), atol=1e-6)


def test_preference_loss_accuracy_in_unit_interval():
    torch.manual_seed(7)
    model = _tiny_discrete_reward_model()
    # Perturb head so rewards are non-trivial.
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.1)
        model.head[-1].bias.normal_(std=0.1)
    chosen = torch.randint(0, VOCAB, (8, SEQ))
    rejected = torch.randint(0, VOCAB, (8, SEQ))
    out = model.compute_preference_loss(chosen, rejected)
    acc = out["accuracy"].item()
    assert 0.0 <= acc <= 1.0


def test_preference_loss_margin_scales_difference():
    """Passing a per-sample margin scales the BT logit (margin * (r_c - r_r))."""
    torch.manual_seed(8)
    model = _tiny_discrete_reward_model()
    with torch.no_grad():
        model.head[-1].weight.normal_(std=0.2)
        model.head[-1].bias.normal_(std=0.2)
    chosen = torch.randint(0, VOCAB, (5, SEQ))
    rejected = torch.randint(0, VOCAB, (5, SEQ))
    no_margin = model.compute_preference_loss(chosen, rejected)
    with_margin = model.compute_preference_loss(chosen, rejected, margin=torch.full((5,), 0.5))
    # Different margins must yield a different loss value in general.
    assert not torch.isclose(no_margin["loss"], with_margin["loss"], atol=1e-6)


def test_preference_loss_is_differentiable():
    torch.manual_seed(9)
    model = _tiny_discrete_reward_model()
    chosen = torch.randint(0, VOCAB, (4, SEQ))
    rejected = torch.randint(0, VOCAB, (4, SEQ))
    out = model.compute_preference_loss(chosen, rejected)
    out["loss"].backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_reward_model_learns_to_prefer_chosen():
    """A few gradient steps on fixed (chosen, rejected) should raise BT accuracy/margin."""
    torch.manual_seed(10)
    model = _tiny_discrete_reward_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    chosen = torch.randint(0, VOCAB, (16, SEQ))
    rejected = torch.randint(0, VOCAB, (16, SEQ))

    first = model.compute_preference_loss(chosen, rejected)["loss"].item()
    for _ in range(40):
        opt.zero_grad()
        out = model.compute_preference_loss(chosen, rejected)
        out["loss"].backward()
        opt.step()
    final = model.compute_preference_loss(chosen, rejected)
    assert final["loss"].item() < first
    # With enough capacity on a tiny fixed set it should separate the pairs.
    assert final["accuracy"].item() > 0.5


# ---------------------------------------------------------------------------
# create_reward_model factory
# ---------------------------------------------------------------------------


def test_create_reward_model_presets():
    model = create_reward_model("nano", vocab_size=VOCAB, seq_length=SEQ)
    assert isinstance(model, RewardModel)
    assert model.hidden_size == 128
    assert len(model.blocks) == 2


def test_create_reward_model_unknown_size_raises():
    with pytest.raises(ValueError, match="Unknown model_size"):
        create_reward_model("gigantic", vocab_size=VOCAB)


def test_create_reward_model_kwargs_override():
    model = create_reward_model("nano", vocab_size=VOCAB, seq_length=SEQ, timestep_aware=True)
    assert model.timestep_aware is True
    assert model.t_embedder is not None


# ---------------------------------------------------------------------------
# RewardModelTrainer (CPU)
# ---------------------------------------------------------------------------


def _make_pref_dataset(n=24, seq=SEQ, vocab=VOCAB, seed=0):
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(n):
        chosen = torch.randint(0, vocab, (seq,), generator=g)
        rejected = torch.randint(0, vocab, (seq,), generator=g)
        pairs.append(PreferencePair(chosen=chosen, rejected=rejected))
    return PreferenceDataset(pairs=pairs)


def test_reward_model_trainer_runs_on_cpu():
    torch.manual_seed(11)
    model = _tiny_discrete_reward_model()
    dataset = _make_pref_dataset(n=16)
    trainer = RewardModelTrainer(model=model, dataset=dataset, lr=1e-3, batch_size=4, device="cpu")
    history = trainer.train(epochs=1, log_every=1, eval_every=1000)
    assert "train_loss" in history
    assert len(history["train_loss"]) > 0
    assert all(v == v for v in history["train_loss"])  # no NaN


def test_reward_model_trainer_validation():
    torch.manual_seed(12)
    model = _tiny_discrete_reward_model()
    train_ds = _make_pref_dataset(n=16, seed=1)
    val_ds = _make_pref_dataset(n=8, seed=2)
    trainer = RewardModelTrainer(
        model=model,
        dataset=train_ds,
        val_dataset=val_ds,
        lr=1e-3,
        batch_size=4,
        device="cpu",
    )
    metrics = trainer.validate()
    assert set(metrics) == {"loss", "accuracy"}
    assert metrics["loss"] == metrics["loss"]
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# data.py: PreferencePair
# ---------------------------------------------------------------------------


def test_preference_pair_validates_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        PreferencePair(chosen=torch.zeros(4), rejected=torch.zeros(5))


def test_preference_pair_validates_margin_bounds():
    with pytest.raises(ValueError, match="Margin"):
        PreferencePair(chosen=torch.zeros(3), rejected=torch.ones(3), margin=1.5)
    with pytest.raises(ValueError, match="Margin"):
        PreferencePair(chosen=torch.zeros(3), rejected=torch.ones(3), margin=-0.1)


def test_preference_pair_coerces_non_tensor_inputs():
    pair = PreferencePair(chosen=[1, 2, 3], rejected=[4, 5, 6])
    assert isinstance(pair.chosen, torch.Tensor)
    assert isinstance(pair.rejected, torch.Tensor)
    assert pair.chosen.shape == (3,)


def test_preference_pair_to_device():
    pair = PreferencePair(chosen=torch.zeros(3), rejected=torch.ones(3), margin=0.7)
    moved = pair.to(torch.device("cpu"))
    assert moved.chosen.device.type == "cpu"
    assert moved.margin == 0.7


# ---------------------------------------------------------------------------
# data.py: RankedSamples + to_pairs strategies
# ---------------------------------------------------------------------------


def test_ranked_samples_requires_two():
    with pytest.raises(ValueError, match="at least 2"):
        RankedSamples(samples=[torch.zeros(3)])


def test_ranked_samples_scores_length_check():
    with pytest.raises(ValueError, match="same length"):
        RankedSamples(samples=[torch.zeros(2), torch.ones(2)], scores=[1.0])


def test_ranked_to_pairs_best_worst():
    samples = [torch.full((3,), float(i)) for i in range(4)]  # best -> worst
    ranked = RankedSamples(samples=samples)
    pairs = ranked.to_pairs("best_worst")
    assert len(pairs) == 1
    assert torch.equal(pairs[0].chosen, samples[0])
    assert torch.equal(pairs[0].rejected, samples[-1])


def test_ranked_to_pairs_adjacent():
    samples = [torch.full((2,), float(i)) for i in range(4)]
    ranked = RankedSamples(samples=samples)
    pairs = ranked.to_pairs("adjacent")
    assert len(pairs) == 3  # n-1
    for i, p in enumerate(pairs):
        assert torch.equal(p.chosen, samples[i])
        assert torch.equal(p.rejected, samples[i + 1])


def test_ranked_to_pairs_all():
    samples = [torch.full((2,), float(i)) for i in range(4)]
    ranked = RankedSamples(samples=samples)
    pairs = ranked.to_pairs("all")
    assert len(pairs) == 6  # n*(n-1)/2
    # Chosen always ranked above rejected (i < j).
    for p in pairs:
        assert p.chosen[0] < p.rejected[0]


def test_ranked_to_pairs_margin_in_unit_interval():
    """_compute_margin must produce a value the PreferencePair accepts ([0,1])."""
    samples = [torch.zeros(2), torch.ones(2), torch.full((2,), 2.0)]
    scores = [3.0, 1.0, -2.0]
    ranked = RankedSamples(samples=samples, scores=scores)
    pairs = ranked.to_pairs("all")
    for p in pairs:
        assert 0.0 <= p.margin <= 1.0


# ---------------------------------------------------------------------------
# data.py: PreferenceDataset
# ---------------------------------------------------------------------------


def test_preference_dataset_len_index_iter():
    pairs = [
        PreferencePair(chosen=torch.zeros(3) + i, rejected=torch.ones(3) + i) for i in range(5)
    ]
    ds = PreferenceDataset(pairs=pairs)
    assert len(ds) == 5
    assert isinstance(ds[0], PreferencePair)
    collected = list(iter(ds))
    assert len(collected) == 5
    assert torch.equal(collected[2].chosen, pairs[2].chosen)


def test_preference_dataset_empty_default():
    ds = PreferenceDataset()
    assert len(ds) == 0
    ds.add_pair(PreferencePair(chosen=torch.zeros(2), rejected=torch.ones(2)))
    assert len(ds) == 1


def test_preference_dataset_transform_applied():
    flag = {"called": 0}

    def transform(pair):
        flag["called"] += 1
        return pair

    ds = PreferenceDataset(
        pairs=[PreferencePair(chosen=torch.zeros(2), rejected=torch.ones(2))],
        transform=transform,
    )
    _ = ds[0]
    assert flag["called"] == 1


def test_preference_dataset_add_ranked():
    ds = PreferenceDataset()
    samples = [torch.full((2,), float(i)) for i in range(3)]
    ds.add_ranked(RankedSamples(samples=samples), strategy="adjacent")
    assert len(ds) == 2


def test_preference_dataset_filter_and_shuffle():
    pairs = [
        PreferencePair(chosen=torch.zeros(2), rejected=torch.ones(2), margin=m)
        for m in [0.2, 0.4, 0.8, 1.0]
    ]
    ds = PreferenceDataset(pairs=pairs)
    high = ds.filter(lambda p: p.margin >= 0.5)
    assert len(high) == 2
    shuffled = ds.shuffle(seed=123)
    assert len(shuffled) == len(ds)
    # Shuffle returns a new dataset; original is untouched.
    assert shuffled is not ds


def test_preference_dataset_split():
    pairs = [PreferencePair(chosen=torch.zeros(2), rejected=torch.ones(2)) for _ in range(10)]
    ds = PreferenceDataset(pairs=pairs)
    train, val = ds.split(train_ratio=0.8, seed=0)
    assert len(train) == 8
    assert len(val) == 2
    assert len(train) + len(val) == len(ds)


# ---------------------------------------------------------------------------
# data.py: collate_preference_pairs (batch shapes)
# ---------------------------------------------------------------------------


def test_collate_preference_pairs_shapes():
    pairs = [
        PreferencePair(
            chosen=torch.randint(0, VOCAB, (SEQ,)),
            rejected=torch.randint(0, VOCAB, (SEQ,)),
            margin=0.5,
            prompt=f"p{i}",
        )
        for i in range(4)
    ]
    batch = collate_preference_pairs(pairs)
    assert batch["chosen"].shape == (4, SEQ)
    assert batch["rejected"].shape == (4, SEQ)
    assert batch["margin"].shape == (4,)
    assert torch.allclose(batch["margin"], torch.full((4,), 0.5))
    assert batch["prompt"] == ["p0", "p1", "p2", "p3"]


def test_collate_via_dataloader():
    from torch.utils.data import DataLoader

    ds = _make_pref_dataset(n=8)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_preference_pairs)
    seen = 0
    for batch in loader:
        assert batch["chosen"].shape == (4, SEQ)
        assert batch["rejected"].shape == (4, SEQ)
        assert batch["margin"].shape == (4,)
        seen += 1
    assert seen == 2


def test_stepwise_preference_to_device():
    sp = StepwisePreference(
        step_idx=3,
        timestep=0.5,
        chosen_state=torch.zeros(2, 4),
        rejected_state=torch.ones(2, 4),
        chosen_score=0.9,
        rejected_score=0.1,
    )
    moved = sp.to(torch.device("cpu"))
    assert moved.step_idx == 3
    assert moved.chosen_state.device.type == "cpu"
    assert moved.chosen_score == 0.9


# ---------------------------------------------------------------------------
# ai_feedback.py: DiscriminatorScorer
# ---------------------------------------------------------------------------


class StubDiscriminatorLogits(nn.Module):
    """Tiny stub: maps token mean to a per-sample logit [batch, 1]. No remote calls."""

    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        # x: [batch, seq] long tokens -> mean -> [batch, 1] logit
        return x.float().mean(dim=1, keepdim=True)


class StubDiscriminatorPerToken(nn.Module):
    """Stub returning per-token scores [batch, seq] to test sequence averaging."""

    def forward(self, x, **kwargs):
        return x.float()


def test_discriminator_scorer_sigmoid_range():
    scorer = DiscriminatorScorer(StubDiscriminatorLogits(), score_transform="sigmoid")
    x = torch.randint(0, 10, (5, SEQ))
    scores = scorer(x)
    assert scores.shape == (5,)
    assert torch.all((scores >= 0) & (scores <= 1))


def test_discriminator_scorer_raw_and_logit():
    raw = DiscriminatorScorer(StubDiscriminatorLogits(), score_transform="raw")
    logit = DiscriminatorScorer(StubDiscriminatorLogits(), score_transform="logit")
    x = torch.randint(0, 10, (4, SEQ))
    # raw / logit do not pass through sigmoid, so they can exceed [0, 1].
    rs = raw(x)
    ls = logit(x)
    assert rs.shape == (4,)
    assert ls.shape == (4,)
    assert torch.allclose(rs, ls)  # both are untransformed here


def test_discriminator_scorer_averages_per_token_outputs():
    scorer = DiscriminatorScorer(StubDiscriminatorPerToken(), score_transform="raw")
    x = torch.randint(0, 10, (3, SEQ))
    scores = scorer(x)
    assert scores.shape == (3,)
    # Mean over the sequence dim equals the per-token stub's row means.
    assert torch.allclose(scores, x.float().mean(dim=1))


def test_discriminator_scorer_rank_samples_orders_by_score():
    scorer = DiscriminatorScorer(StubDiscriminatorLogits(), score_transform="raw")
    # Higher token values -> higher mean -> higher score.
    low = torch.zeros(SEQ, dtype=torch.long)
    mid = torch.full((SEQ,), 3, dtype=torch.long)
    high = torch.full((SEQ,), 7, dtype=torch.long)
    order = scorer.rank_samples([low, high, mid])
    assert order == [1, 2, 0]  # high, mid, low


def test_discriminator_scorer_create_pairs_respects_presorted_order():
    """When samples are already passed best-first, create_pairs builds the right pair.

    RankedSamples treats index 0 as 'best', so passing high (better) first yields
    chosen=high, rejected=low.
    """
    scorer = DiscriminatorScorer(StubDiscriminatorLogits(), score_transform="raw")
    low = torch.zeros(SEQ, dtype=torch.long)
    high = torch.full((SEQ,), 9, dtype=torch.long)
    pairs = scorer.create_pairs([high, low], strategy="best_worst")
    assert len(pairs) == 1
    assert torch.equal(pairs[0].chosen, high)
    assert torch.equal(pairs[0].rejected, low)


def test_discriminator_scorer_create_pairs_ranks_by_score():
    # Regression: create_pairs sorts by score (best first) before pairing, so the
    # higher-scoring sample is chosen regardless of input order.
    scorer = DiscriminatorScorer(StubDiscriminatorLogits(), score_transform="raw")
    low = torch.zeros(SEQ, dtype=torch.long)
    high = torch.full((SEQ,), 9, dtype=torch.long)
    pairs = scorer.create_pairs([low, high], strategy="best_worst")
    assert len(pairs) == 1
    assert torch.equal(pairs[0].chosen, high)
    assert torch.equal(pairs[0].rejected, low)


# ---------------------------------------------------------------------------
# ai_feedback.py: AIPreferenceScorer
# ---------------------------------------------------------------------------


def test_ai_preference_scorer_from_callable_score_and_rank():
    # Score by sum of tokens.
    scorer = AIPreferenceScorer(lambda x: x.float().sum(dim=1), higher_is_better=True)
    a = torch.ones(SEQ, dtype=torch.long)
    b = torch.full((SEQ,), 5, dtype=torch.long)
    c = torch.full((SEQ,), 3, dtype=torch.long)
    stacked = torch.stack([a, b, c])
    scores = scorer.score(stacked)
    assert scores.shape == (3,)
    order = scorer.rank([a, b, c])
    assert order == [1, 2, 0]  # b > c > a


def test_ai_preference_scorer_higher_is_better_false_inverts():
    base = AIPreferenceScorer(lambda x: x.float().sum(dim=1), higher_is_better=True)
    inv = AIPreferenceScorer(lambda x: x.float().sum(dim=1), higher_is_better=False)
    a = torch.ones(SEQ, dtype=torch.long)
    b = torch.full((SEQ,), 5, dtype=torch.long)
    # With higher_is_better=False, the smaller-sum sample ranks first.
    assert base.rank([a, b]) == [1, 0]
    assert inv.rank([a, b]) == [0, 1]


def test_ai_preference_scorer_from_reward_model():
    torch.manual_seed(20)
    rm = _tiny_discrete_reward_model()
    # Make the reward model produce non-degenerate scores.
    with torch.no_grad():
        rm.head[-1].weight.normal_(std=0.3)
        rm.head[-1].bias.normal_(std=0.3)
    scorer = AIPreferenceScorer.from_reward_model(rm)
    samples = [torch.randint(0, VOCAB, (SEQ,)) for _ in range(4)]
    order = scorer.rank(samples)
    assert sorted(order) == [0, 1, 2, 3]  # a valid permutation
    pairs = scorer.create_pairs(samples, strategy="best_worst")
    assert len(pairs) == 1
    assert pairs[0].chosen.shape == (SEQ,)


def test_ai_preference_scorer_from_discriminator():
    scorer = AIPreferenceScorer.from_discriminator(StubDiscriminatorLogits())
    samples = [torch.full((SEQ,), v, dtype=torch.long) for v in (0, 4, 8)]
    order = scorer.rank(samples)
    assert order == [2, 1, 0]


def test_ai_preference_scorer_create_pairs_all_strategy():
    scorer = AIPreferenceScorer(lambda x: x.float().sum(dim=1))
    samples = [torch.full((SEQ,), v, dtype=torch.long) for v in (1, 2, 3)]
    pairs = scorer.create_pairs(samples, strategy="all")
    assert len(pairs) == 3  # n*(n-1)/2 over the 3 ranked samples


def test_ai_preference_scorer_create_pairs_ranks_by_score():
    # Regression: create_pairs sorts by score before pairing; the higher-scoring
    # sample is chosen regardless of input order.
    scorer = AIPreferenceScorer(lambda x: x.float().sum(dim=1), higher_is_better=True)
    low = torch.zeros(SEQ, dtype=torch.long)
    high = torch.full((SEQ,), 9, dtype=torch.long)
    pairs = scorer.create_pairs([low, high], strategy="best_worst")
    assert torch.equal(pairs[0].chosen, high)
    assert torch.equal(pairs[0].rejected, low)


def test_ai_preference_scorer_generate_pairs_from_generator():
    """generate_pairs_from_generator with a stub generator exposing .generate()."""

    class StubGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self._call = 0

        def generate(self, prompt, **kw):
            # Deterministic-ish samples that depend on call index so ranking varies.
            self._call += 1
            return torch.full((SEQ,), self._call % 7, dtype=torch.long)

    gen = StubGenerator()
    scorer = AIPreferenceScorer(lambda x: x.float().sum(dim=1))
    pairs = scorer.generate_pairs_from_generator(
        generator=gen,
        prompts=[0, 1],
        num_samples_per_prompt=3,
        strategy="best_worst",
        device="cpu",
    )
    # One best_worst pair per prompt.
    assert len(pairs) == 2
    for p in pairs:
        assert p.chosen.shape == (SEQ,)
        assert p.rejected.shape == (SEQ,)


# ---------------------------------------------------------------------------
# ai_feedback.py: StepAwarePreferenceModel
# ---------------------------------------------------------------------------


def test_step_aware_requires_timestep_aware_base():
    base = _tiny_discrete_reward_model(timestep_aware=False)
    with pytest.raises(ValueError, match="timestep_aware"):
        StepAwarePreferenceModel(base_model=base)


def test_step_aware_requires_inputs_when_no_base():
    with pytest.raises(ValueError, match="base_model"):
        StepAwarePreferenceModel(base_model=None, vocab_size=None, in_channels=None)


def test_step_aware_builds_own_model_and_scores():
    torch.manual_seed(21)
    model = StepAwarePreferenceModel(
        vocab_size=VOCAB, seq_length=SEQ, hidden_size=HIDDEN, depth=DEPTH, num_heads=HEADS
    )
    assert model.model.timestep_aware is True
    x = torch.randint(0, VOCAB, (4, SEQ))
    t = torch.rand(4)
    scores = model(x, t)
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_step_aware_wraps_existing_base_model():
    base = _tiny_discrete_reward_model(timestep_aware=True)
    model = StepAwarePreferenceModel(base_model=base)
    assert model.model is base


def test_step_aware_select_winner_loser():
    torch.manual_seed(22)
    model = StepAwarePreferenceModel(
        vocab_size=VOCAB, seq_length=SEQ, hidden_size=HIDDEN, depth=DEPTH, num_heads=HEADS
    )
    # Perturb head so scores differ across candidates.
    with torch.no_grad():
        model.model.head[-1].weight.normal_(std=0.3)
        model.model.head[-1].bias.normal_(std=0.3)

    num_candidates, batch = 3, 4
    candidates = torch.randint(0, VOCAB, (num_candidates, batch, SEQ))
    t = torch.rand(batch)

    chosen, rejected, chosen_scores, rejected_scores = model.select_winner_loser(candidates, t)

    assert chosen.shape == (batch, SEQ)
    assert rejected.shape == (batch, SEQ)
    assert chosen_scores.shape == (batch,)
    assert rejected_scores.shape == (batch,)
    # Winner score must be >= loser score for every batch element (argmax vs argmin).
    assert torch.all(chosen_scores >= rejected_scores)


def test_step_aware_select_winner_loser_scalar_timestep():
    torch.manual_seed(23)
    model = StepAwarePreferenceModel(
        vocab_size=VOCAB, seq_length=SEQ, hidden_size=HIDDEN, depth=DEPTH, num_heads=HEADS
    )
    with torch.no_grad():
        model.model.head[-1].weight.normal_(std=0.3)
    candidates = torch.randint(0, VOCAB, (3, 2, SEQ))
    t = torch.tensor(0.5)  # scalar timestep, must be broadcast to batch
    chosen, rejected, cs, rs = model.select_winner_loser(candidates, t)
    assert chosen.shape == (2, SEQ)
    assert torch.all(cs >= rs)


def test_step_aware_sample_random_candidate_shape():
    torch.manual_seed(24)
    model = StepAwarePreferenceModel(
        vocab_size=VOCAB, seq_length=SEQ, hidden_size=HIDDEN, depth=DEPTH, num_heads=HEADS
    )
    candidates = torch.randint(0, VOCAB, (5, 4, SEQ))
    picked = model.sample_random_candidate(candidates)
    assert picked.shape == (4, SEQ)
    # Each picked row must equal one of the candidate rows for that batch item.
    for b in range(4):
        assert any(torch.equal(picked[b], candidates[c, b]) for c in range(5))


# ---------------------------------------------------------------------------
# CUDA guard
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_reward_model_forward_cuda():  # pragma: no cover - needs GPU
    model = _tiny_discrete_reward_model().cuda()
    x = torch.randint(0, VOCAB, (3, SEQ), device="cuda")
    scores = model(x)
    assert scores.is_cuda
    assert scores.shape == (3, 1)
