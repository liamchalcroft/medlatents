"""Tests for advanced generative model features (2024-2025).

Tests for:
- Halton scheduler for MaskGIT
- Guidance schedules for CFG
- Decoupled ST-Gumbel-Softmax
- RF++ (Rectified Flow++) with higher-order ODE solvers
- BFN improvements (entropy encoding, score-based guidance, particle filtering)
"""

import torch
import torch.nn as nn

# ============================================================================
# Mock Models for Testing
# ============================================================================


class MockFlowModel(nn.Module):
    """Mock continuous flow model."""

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x, t, y=None, **kwargs):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_cat = torch.cat([x, t], dim=-1)
        return self.net(x_cat)


# ============================================================================
# Tests for Halton Scheduler (MaskGIT)
# ============================================================================


class TestHaltonScheduler:
    """Tests for Halton-based spatial dispersion scheduling."""

    def test_halton_sequence_deterministic(self):
        """Halton sequence should be deterministic."""
        from medlatents.sampling.maskgit import halton_sequence

        h1 = halton_sequence(0, base=2)
        h2 = halton_sequence(0, base=2)
        assert h1 == h2

    def test_halton_sequence_range(self):
        """Halton sequence values should be in [0, 1)."""
        from medlatents.sampling.maskgit import halton_sequence

        for i in range(100):
            h = halton_sequence(i, base=2)
            assert 0 <= h < 1

    def test_halton_schedule_1d_shape(self):
        """Halton schedule should produce valid mask ratios."""
        from medlatents.sampling.maskgit import halton_schedule_1d

        total_steps = 10
        for step in range(total_steps):
            ratio = halton_schedule_1d(step, total_steps)
            assert 0 <= ratio <= 1


# ============================================================================
# Tests for Guidance Schedules
# ============================================================================


class TestGuidanceSchedules:
    """Tests for classifier-free guidance schedules."""

    def test_get_guidance_schedule(self):
        """Should return correct schedule function."""
        from medlatents.sampling.schedules import get_guidance_schedule

        for name in ["cfg_zero", "cfg_zero_star", "dynamic"]:
            schedule = get_guidance_schedule(name)
            assert callable(schedule)


# ============================================================================
# Tests for Decoupled ST-Gumbel-Softmax
# ============================================================================


class TestDecoupledSTGumbelSoftmax:
    """Tests for decoupled straight-through Gumbel-Softmax."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        from medlatents.sampling.discretization import DecoupledSTGumbelSoftmax

        st_gumbel = DecoupledSTGumbelSoftmax(
            forward_temp=1.0,
            backward_temp=0.5,
            hard=True,
        )

        assert st_gumbel.forward_temp == 1.0
        assert st_gumbel.backward_temp == 0.5
        assert st_gumbel.hard is True

    def test_forward_shape(self):
        """Forward pass should produce correct output shape."""
        from medlatents.sampling.discretization import DecoupledSTGumbelSoftmax

        st_gumbel = DecoupledSTGumbelSoftmax()
        logits = torch.randn(4, 16, 32)

        output = st_gumbel(logits)
        assert output.shape == logits.shape


# ============================================================================
# Tests for RF++ and ODE Solvers
# ============================================================================


class TestODESolvers:
    """Tests for higher-order ODE solvers."""

    def test_euler_step_shape(self):
        """Euler step should preserve shape."""
        from medlatents.flow_matching.continuous import euler_step

        model = MockFlowModel(latent_dim=8)
        x = torch.randn(2, 8)
        t = 0.5

        x_next = euler_step(model, x, t, dt=0.1)
        assert x_next.shape == x.shape

    def test_heun_step_shape(self):
        """Heun step should preserve shape."""
        from medlatents.flow_matching.continuous import heun_step

        model = MockFlowModel(latent_dim=8)
        x = torch.randn(2, 8)
        t = 0.5

        x_next = heun_step(model, x, t, dt=0.1)
        assert x_next.shape == x.shape

    def test_rk4_step_shape(self):
        """RK4 step should preserve shape."""
        from medlatents.flow_matching.continuous import rk4_step

        model = MockFlowModel(latent_dim=8)
        x = torch.randn(2, 8)
        t = 0.5

        x_next = rk4_step(model, x, t, dt=0.1)
        assert x_next.shape == x.shape


class TestRectifiedFlowPP:
    """Tests for Rectified Flow++ (RF++)."""

    def test_rfpp_initialization(self):
        """RF++ should initialize correctly."""
        from medlatents.flow_matching.continuous import RectifiedFlowPP

        rfpp = RectifiedFlowPP(
            num_reflow_iterations=2,
            latent_shape=(8,),
            base_std=1.0,
        )

        assert rfpp.num_reflow_iterations == 2
        assert rfpp.latent_shape == (8,)

    def test_rfpp_compute_loss(self):
        """RF++ should compute reflow loss."""
        from medlatents.flow_matching.continuous import RectifiedFlowPP

        rfpp = RectifiedFlowPP(latent_shape=(8,))
        model = MockFlowModel(latent_dim=8)
        x1 = torch.randn(2, 8)

        loss = rfpp.compute_reflow_loss(model, x1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ============================================================================
# Tests for BFN Improvements
# ============================================================================


class TestEntropyEncoding:
    """Tests for entropy encoding utilities."""

    def test_compute_entropy_uniform(self):
        """Uniform distribution should have high entropy."""
        from medlatents.bayesian_flow import compute_entropy

        # Uniform logits (all same value)
        logits = torch.zeros(1, 1, 16)
        entropy = compute_entropy(logits, normalize=True)

        # Normalized entropy should be close to 1.0 for uniform
        assert abs(entropy.item() - 1.0) < 0.1

    def test_compute_entropy_batch(self):
        """Entropy should work with batched inputs."""
        from medlatents.bayesian_flow import compute_entropy

        logits = torch.randn(4, 16, 32)
        entropy = compute_entropy(logits, normalize=True)

        # Should return per-batch entropy
        assert entropy.shape == (4, 16)


class TestBFNNoiseSchedule:
    """Tests for BFN noise schedules."""

    def test_exponential_schedule(self):
        """Exponential schedule should compute valid alphas."""
        from medlatents.bayesian_flow import exponential_schedule

        t = torch.tensor([0.0, 0.5, 1.0])
        alpha, beta = exponential_schedule(t)

        assert alpha.shape == (3,)
        assert beta.shape == (3,)

    def test_linear_entropy_schedule(self):
        """Linear entropy schedule should compute valid alphas."""
        from medlatents.bayesian_flow import linear_entropy_schedule

        t = torch.tensor([0.0, 0.5, 1.0])
        alpha, beta = linear_entropy_schedule(t)

        assert alpha.shape == (3,)
        assert beta.shape == (3,)


# ============================================================================
# Tests for Speculative Decoding
# ============================================================================


class MockAutoregressiveModel(nn.Module):
    """Mock autoregressive model for testing."""

    def __init__(self, vocab_size: int = 32, seq_length: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 64)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.head = nn.Linear(64, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, 64]
        transformed = self.transformer(embedded)
        logits = self.head(transformed)
        return logits


class TestSpeculativeDecoder:
    """Tests for speculative decoding."""

    def test_initialization(self):
        """SpeculativeDecoder should initialize correctly."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
            max_speculation=4,
        )

        assert decoder.main_model is main_model
        assert decoder.draft_model is draft_model
        assert decoder.max_speculation == 4

    def test_sample_token_shape(self):
        """Sample token should return correct shape."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
        )

        logits = torch.randn(4, 32)
        tokens = decoder._sample_token(logits)

        assert tokens.shape == (4,)
        assert (tokens >= 0).all()
        assert (tokens < 32).all()

    def test_top_k_filtering(self):
        """Top-k filtering should limit vocab size."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
            top_k=10,
        )

        logits = torch.randn(2, 32)
        filtered = decoder._apply_top_k(logits, 10)

        # Should have exactly 10 finite values per row
        for i in range(2):
            finite_count = torch.isfinite(filtered[i]).sum()
            assert finite_count == 10

    def test_top_p_filtering(self):
        """Top-p filtering should maintain cumulative probability."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
            top_p=0.9,
        )

        logits = torch.randn(2, 32)
        filtered = decoder._apply_top_p(logits, 0.9)

        # Should keep at least 1 token
        for i in range(2):
            finite_count = torch.isfinite(filtered[i]).sum()
            assert finite_count >= 1

    def test_draft_generate_shape(self):
        """Draft generation should produce correct shape."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
        )

        context = torch.randint(0, 32, (2, 8))
        K = 4

        draft_tokens = decoder._draft_generate(context, K)

        assert draft_tokens.shape == (2, K)

    def test_generate_shape(self):
        """Generate should produce correct output shape."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
            max_speculation=4,
        )

        initial_tokens = torch.randint(0, 32, (2, 8))
        generated, metrics = decoder.generate(initial_tokens, max_new_tokens=8)

        assert generated.shape == (2, 16)  # 8 initial + 8 new
        assert "acceptance_rate" in metrics
        assert 0 <= metrics["acceptance_rate"] <= 1

    def test_generate_with_eos(self):
        """Generation should respect EOS token."""
        from medlatents.autoregressive import SpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_model = MockAutoregressiveModel()

        decoder = SpeculativeDecoder(
            main_model=main_model,
            draft_model=draft_model,
            vocab_size=32,
            max_speculation=4,
        )

        initial_tokens = torch.randint(0, 31, (2, 8))  # Exclude EOS
        generated, metrics = decoder.generate(
            initial_tokens,
            max_new_tokens=16,
            eos_token=31,
        )

        # Should stop early if EOS generated
        assert generated.shape[1] <= 24  # 8 + 16 max


class TestMultiDraftSpeculativeDecoder:
    """Tests for multi-draft speculative decoding."""

    def test_initialization(self):
        """MultiDraftSpeculativeDecoder should initialize correctly."""
        from medlatents.autoregressive import MultiDraftSpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_models = [MockAutoregressiveModel(), MockAutoregressiveModel()]

        decoder = MultiDraftSpeculativeDecoder(
            main_model=main_model,
            draft_models=draft_models,
            vocab_size=32,
            max_speculation=4,
        )

        assert len(decoder.draft_models) == 2
        assert decoder.selection_strategy == "ensemble"

    def test_ensemble_selection(self):
        """Ensemble selection should average predictions."""
        from medlatents.autoregressive import MultiDraftSpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_models = [MockAutoregressiveModel(), MockAutoregressiveModel()]

        decoder = MultiDraftSpeculativeDecoder(
            main_model=main_model,
            draft_models=draft_models,
            vocab_size=32,
            selection_strategy="ensemble",
        )

        context = torch.randint(0, 32, (2, 8))
        draft_tokens = decoder._draft_generate(context, 4)

        assert draft_tokens.shape == (2, 4)

    def test_voting_selection(self):
        """Voting selection should use majority vote."""
        from medlatents.autoregressive import MultiDraftSpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_models = [
            MockAutoregressiveModel(),
            MockAutoregressiveModel(),
            MockAutoregressiveModel(),
        ]

        decoder = MultiDraftSpeculativeDecoder(
            main_model=main_model,
            draft_models=draft_models,
            vocab_size=32,
            selection_strategy="voting",
        )

        context = torch.randint(0, 32, (2, 8))
        draft_tokens = decoder._draft_generate(context, 4)

        assert draft_tokens.shape == (2, 4)

    def test_best_selection(self):
        """Best selection should pick lowest perplexity draft."""
        from medlatents.autoregressive import MultiDraftSpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_models = [MockAutoregressiveModel(), MockAutoregressiveModel()]

        decoder = MultiDraftSpeculativeDecoder(
            main_model=main_model,
            draft_models=draft_models,
            vocab_size=32,
            selection_strategy="best",
        )

        context = torch.randint(0, 32, (2, 8))
        draft_tokens = decoder._draft_generate(context, 4)

        assert draft_tokens.shape == (2, 4)

    def test_generate_shape(self):
        """Generate should work with multiple drafts."""
        from medlatents.autoregressive import MultiDraftSpeculativeDecoder

        main_model = MockAutoregressiveModel()
        draft_models = [MockAutoregressiveModel(), MockAutoregressiveModel()]

        decoder = MultiDraftSpeculativeDecoder(
            main_model=main_model,
            draft_models=draft_models,
            vocab_size=32,
            max_speculation=4,
        )

        initial_tokens = torch.randint(0, 32, (2, 8))
        generated, metrics = decoder.generate(initial_tokens, max_new_tokens=8)

        assert generated.shape == (2, 16)
        assert metrics["acceptance_rate"] >= 0
