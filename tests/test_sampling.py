"""Tests for sampling techniques and loss functions."""

import pytest
import torch
import torch.nn.functional as F

from medlatents.sampling import (
    # Loss functions now in medlatents.training
    DDIMScheduler,
    MaskGITScheduler,
    TemperatureScheduler,
    sample_autoregressive,
    sample_min_p,
    sample_nucleus,
    sample_with_cfg,
)
from medlatents.sampling.autoregressive import sample_top_k

# Import loss functions from new location
from medlatents.training.losses import (
    AdaptiveTemperatureScaling,
    DistillationLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    TokenDropout,
    compute_perplexity,
    compute_token_accuracy,
)


class TestTopKSampling:
    """Tests for top-k sampling."""

    def test_top_k_filters_correctly(self):
        """Top-k should keep only the top k logits."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = sample_top_k(logits, k=3)

        # Top 3 are indices 2, 3, 4 with values 3, 4, 5
        assert torch.isfinite(filtered[0, 2])
        assert torch.isfinite(filtered[0, 3])
        assert torch.isfinite(filtered[0, 4])
        # Bottom 2 should be -inf
        assert filtered[0, 0] == float("-inf")
        assert filtered[0, 1] == float("-inf")

    def test_top_k_zero_returns_original(self):
        """Top-k with k=0 should return original logits."""
        logits = torch.randn(2, 10)
        filtered = sample_top_k(logits, k=0)
        assert torch.allclose(logits, filtered)

    def test_top_k_exceeds_vocab(self):
        """Top-k with k > vocab_size should work correctly."""
        logits = torch.randn(2, 10)
        filtered = sample_top_k(logits, k=100)
        assert torch.allclose(logits, filtered)


class TestNucleusSampling:
    """Tests for nucleus (top-p) sampling."""

    def test_nucleus_preserves_top_tokens(self):
        """Nucleus should keep tokens until cumulative prob > p."""
        # Create logits where one token dominates
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        filtered = sample_nucleus(logits, p=0.5)

        # The dominant token should be kept
        assert torch.isfinite(filtered[0, 0])
        # Probs are ~1.0 for first token, so others are likely -inf
        # (depends on exact threshold)

    def test_nucleus_p_one_returns_original(self):
        """Nucleus with p=1.0 should return original logits."""
        logits = torch.randn(2, 10)
        filtered = sample_nucleus(logits, p=1.0)
        assert torch.allclose(logits, filtered)

    def test_nucleus_always_keeps_one(self):
        """Nucleus should always keep at least one token."""
        logits = torch.randn(2, 10)
        filtered = sample_nucleus(logits, p=0.001)  # Very restrictive

        # Check each sample has at least one finite logit
        for i in range(2):
            assert torch.any(torch.isfinite(filtered[i]))

    def test_nucleus_invalid_p_raises(self):
        """Invalid p should raise."""
        logits = torch.randn(2, 10)
        with pytest.raises(ValueError):
            sample_nucleus(logits, p=0.0)
        with pytest.raises(ValueError):
            sample_nucleus(logits, p=1.5)


class TestMinPSampling:
    """Tests for min-p sampling."""

    def test_min_p_scales_with_confidence(self):
        """Min-p should be more selective when model is confident."""
        # Confident model (one dominant logit)
        confident_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        confident_filtered = sample_min_p(confident_logits, min_p=0.1)

        # Uncertain model (uniform logits)
        uncertain_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        uncertain_filtered = sample_min_p(uncertain_logits, min_p=0.1)

        # Confident should filter more (fewer finite values)
        confident_finite = torch.isfinite(confident_filtered).sum()
        uncertain_finite = torch.isfinite(uncertain_filtered).sum()

        # For confident model, only top token should remain
        assert confident_finite <= uncertain_finite

    def test_min_p_with_base_top_p(self):
        """Min-p with base_top_p should apply both filters."""
        logits = torch.randn(2, 100)
        filtered = sample_min_p(logits, min_p=0.05, base_top_p=0.9)

        # Should have some tokens filtered out
        assert torch.any(filtered == float("-inf"))

    def test_min_p_invalid_args_raises(self):
        """Invalid min_p or base_top_p should raise."""
        logits = torch.randn(2, 10)
        with pytest.raises(ValueError):
            sample_min_p(logits, min_p=-0.1)
        with pytest.raises(ValueError):
            sample_min_p(logits, min_p=1.5)
        with pytest.raises(ValueError):
            sample_min_p(logits, min_p=0.1, base_top_p=0.0)


class TestCFGSampling:
    """Tests for classifier-free guidance."""

    def test_cfg_scale_one_returns_conditional(self):
        """CFG with scale=1 should return conditional logits."""
        cond = torch.randn(2, 10)
        uncond = torch.randn(2, 10)
        result = sample_with_cfg(cond, uncond, guidance_scale=1.0)
        assert torch.allclose(result, cond)

    def test_cfg_amplifies_difference(self):
        """CFG with scale>1 should amplify conditional-unconditional difference."""
        cond = torch.tensor([[1.0, 0.0]])
        uncond = torch.tensor([[0.0, 0.0]])

        result = sample_with_cfg(cond, uncond, guidance_scale=2.0)
        # Result = uncond + 2 * (cond - uncond) = 0 + 2 * 1 = 2 for first
        expected = torch.tensor([[2.0, 0.0]])
        assert torch.allclose(result, expected)


class TestTemperatureScheduler:
    """Tests for temperature scheduling."""

    def test_constant_schedule(self):
        """Constant schedule should return same temperature."""
        scheduler = TemperatureScheduler(schedule="constant", start_temp=0.7)
        for step in range(10):
            assert scheduler.get_temperature(step) == 0.7

    def test_linear_schedule(self):
        """Linear schedule should decrease linearly."""
        scheduler = TemperatureScheduler(
            schedule="linear", start_temp=1.0, end_temp=0.5, num_steps=10
        )
        # At step 0, should be start_temp
        assert scheduler.get_temperature(0) == 1.0
        # At last step, should be end_temp
        assert scheduler.get_temperature(9) == 0.5
        # At step 4: progress=4/9≈0.444, temp=1.0-0.5*0.444≈0.778
        assert abs(scheduler.get_temperature(4) - 0.778) < 0.01

    def test_cosine_schedule(self):
        """Cosine schedule should smoothly anneal."""
        scheduler = TemperatureScheduler(
            schedule="cosine", start_temp=1.0, end_temp=0.5, num_steps=10
        )
        # At step 0, should be start_temp
        assert abs(scheduler.get_temperature(0) - 1.0) < 1e-6
        # At last step, should be end_temp
        assert abs(scheduler.get_temperature(9) - 0.5) < 1e-6

    def test_exponential_schedule(self):
        """Exponential schedule should decay exponentially."""
        scheduler = TemperatureScheduler(
            schedule="exponential", start_temp=1.0, end_temp=0.1, num_steps=10
        )
        temps = [scheduler.get_temperature(i) for i in range(10)]
        # Should be monotonically decreasing
        for i in range(1, len(temps)):
            assert temps[i] < temps[i - 1]

    def test_schedule_without_num_steps_raises(self):
        """Non-constant schedule without num_steps should raise."""
        scheduler = TemperatureScheduler(schedule="linear", start_temp=1.0)
        with pytest.raises(ValueError):
            scheduler.get_temperature(5)


class TestSampleAutoregressive:
    """Tests for the combined autoregressive sampling function."""

    def test_sample_autoregressive_returns_valid_indices(self):
        """Sample should return valid token indices."""
        logits = torch.randn(4, 100)
        samples = sample_autoregressive(logits, temperature=1.0)

        assert samples.shape == (4,)
        assert samples.dtype == torch.long
        assert torch.all(samples >= 0)
        assert torch.all(samples < 100)

    def test_temperature_affects_distribution(self):
        """Higher temperature should produce more uniform sampling."""
        torch.manual_seed(42)
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]] * 100)

        # Low temperature - should almost always pick max
        low_temp_samples = sample_autoregressive(logits, temperature=0.1)
        low_temp_zeros = (low_temp_samples == 0).sum()

        # High temperature - should be more spread out
        torch.manual_seed(42)
        high_temp_samples = sample_autoregressive(logits, temperature=5.0)
        high_temp_zeros = (high_temp_samples == 0).sum()

        assert low_temp_zeros > high_temp_zeros

    def test_invalid_temperature_raises(self):
        """Temperature <= 0 should raise."""
        logits = torch.randn(2, 10)
        with pytest.raises(ValueError):
            sample_autoregressive(logits, temperature=0.0)

    def test_invalid_top_k_raises(self):
        """Negative top_k should raise."""
        logits = torch.randn(2, 10)
        with pytest.raises(ValueError):
            sample_autoregressive(logits, temperature=1.0, top_k=-1)


class TestMaskGITScheduler:
    """Tests for MaskGIT scheduler validation."""

    def test_invalid_num_steps_raises(self):
        with pytest.raises(ValueError):
            MaskGITScheduler(num_steps=0)

    def test_step_out_of_range_raises(self):
        scheduler = MaskGITScheduler(num_steps=4)
        logits = torch.randn(2, 5, 7)
        tokens = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.ones(2, 5, dtype=torch.bool)
        with pytest.raises(ValueError):
            scheduler.step(logits, tokens, mask, step=4)


class TestMaskGITScheduleContract:
    """Tests for MaskGIT schedule contract compliance.

    Contract:
    - mask_ratio is the fraction of tokens that should REMAIN masked
    - mask_ratio must decay from ~1.0 (step=0) to ~0.0 (step=T-1)
    - All schedules must return values in [0, 1]
    - Cosine/linear/sqrt/power must be monotonically non-increasing
    - Halton may not be strictly monotonic but must start high and end low
    """

    @pytest.mark.parametrize("schedule", ["cosine", "linear", "sqrt", "power"])
    def test_monotonic_schedules_start_high(self, schedule):
        """Monotonic schedules should start with mask_ratio >= 0.9 at step 0."""
        scheduler = MaskGITScheduler(num_steps=12, mask_schedule=schedule)
        ratio = scheduler.get_mask_ratio(0)
        assert ratio >= 0.9, f"{schedule} schedule: expected start >= 0.9, got {ratio}"

    @pytest.mark.parametrize("schedule", ["cosine", "linear", "sqrt", "power"])
    def test_monotonic_schedules_end_low(self, schedule):
        """Monotonic schedules should end with mask_ratio <= 0.1 at final step."""
        scheduler = MaskGITScheduler(num_steps=12, mask_schedule=schedule)
        ratio = scheduler.get_mask_ratio(11)
        assert ratio <= 0.1, f"{schedule} schedule: expected end <= 0.1, got {ratio}"

    @pytest.mark.parametrize("schedule", ["cosine", "linear", "sqrt", "power"])
    def test_monotonic_schedules_decrease(self, schedule):
        """Monotonic schedules should be monotonically non-increasing."""
        scheduler = MaskGITScheduler(num_steps=12, mask_schedule=schedule)
        ratios = [scheduler.get_mask_ratio(s) for s in range(12)]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1], (
                f"{schedule} not monotonic at step {i}: {ratios[i]} < {ratios[i + 1]}"
            )

    @pytest.mark.parametrize("schedule", ["cosine", "linear", "sqrt", "power", "halton"])
    def test_all_schedules_in_range(self, schedule):
        """All schedules should return mask_ratio in [0, 1]."""
        scheduler = MaskGITScheduler(num_steps=12, mask_schedule=schedule)
        for step in range(12):
            ratio = scheduler.get_mask_ratio(step)
            assert 0.0 <= ratio <= 1.0, f"{schedule} out of range at step {step}: {ratio}"

    def test_halton_starts_high(self):
        """Halton schedule should start with mask_ratio >= 0.5."""
        scheduler = MaskGITScheduler(num_steps=12, mask_schedule="halton")
        ratio = scheduler.get_mask_ratio(0)
        # Halton uses (step+1) so first step may be lower than other schedules
        assert ratio >= 0.5, f"halton schedule: expected start >= 0.5, got {ratio}"

    def test_halton_ends_low(self):
        """Halton schedule should end with mask_ratio <= 0.1."""
        scheduler = MaskGITScheduler(num_steps=12, mask_schedule="halton")
        ratio = scheduler.get_mask_ratio(11)
        assert ratio <= 0.1, f"halton schedule: expected end <= 0.1, got {ratio}"

    @pytest.mark.parametrize("num_steps", [4, 12, 24, 50])
    def test_schedule_works_at_different_lengths(self, num_steps):
        """Schedules should work correctly at different step counts."""
        scheduler = MaskGITScheduler(num_steps=num_steps, mask_schedule="cosine")
        # Should not raise
        ratios = [scheduler.get_mask_ratio(s) for s in range(num_steps)]
        # First should be high, last should be low
        assert ratios[0] >= 0.9
        assert ratios[-1] <= 0.1
        # All in range
        assert all(0.0 <= r <= 1.0 for r in ratios)


class TestDiffusionSchedulers:
    """Tests for diffusion scheduler validation."""

    def test_step_requires_set_timesteps(self):
        scheduler = DDIMScheduler(num_train_steps=10, num_inference_steps=5)
        probs = torch.softmax(torch.randn(2, 4, 8), dim=-1)
        sample = torch.zeros(2, 4, dtype=torch.long)
        with pytest.raises(RuntimeError, match="set_timesteps"):
            scheduler.step(probs, timestep=0, sample=sample)

    def test_invalid_set_timesteps_raises(self):
        scheduler = DDIMScheduler(num_train_steps=10, num_inference_steps=5)
        with pytest.raises(ValueError):
            scheduler.set_timesteps(0)
        with pytest.raises(ValueError):
            scheduler.set_timesteps(20)

    def test_timestep_not_in_schedule_raises(self):
        scheduler = DDIMScheduler(num_train_steps=10, num_inference_steps=5)
        scheduler.set_timesteps(5)
        probs = torch.softmax(torch.randn(2, 4, 8), dim=-1)
        sample = torch.zeros(2, 4, dtype=torch.long)
        with pytest.raises(ValueError, match="not in scheduler.timesteps"):
            scheduler.step(probs, timestep=7, sample=sample)


class TestLabelSmoothingCrossEntropy:
    """Tests for label smoothing loss."""

    def test_label_smoothing_reduces_confidence(self):
        """Label smoothing should reduce confidence in predictions."""
        logits = torch.randn(4, 16, 100)
        targets = torch.randint(0, 100, (4, 16))

        no_smooth = LabelSmoothingCrossEntropy(smoothing=0.0)
        with_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)

        loss_no_smooth = no_smooth(logits, targets)
        loss_with_smooth = with_smooth(logits, targets)

        # Loss should change with smoothing
        assert not torch.isclose(loss_no_smooth, loss_with_smooth)

    def test_label_smoothing_zero_equals_ce(self):
        """Label smoothing=0 should equal cross entropy."""
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))

        smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, targets)
        ce_loss = F.cross_entropy(logits, targets)

        assert torch.allclose(smooth_loss, ce_loss, atol=1e-5)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        logits = torch.randn(4, 16, 100)
        targets = torch.randint(0, 100, (4, 16))

        mean_loss = LabelSmoothingCrossEntropy(reduction="mean")(logits, targets)
        sum_loss = LabelSmoothingCrossEntropy(reduction="sum")(logits, targets)
        none_loss = LabelSmoothingCrossEntropy(reduction="none")(logits, targets)

        assert mean_loss.ndim == 0
        assert sum_loss.ndim == 0
        assert none_loss.shape == (4, 16)


class TestFocalLoss:
    """Tests for focal loss."""

    def test_focal_gamma_zero_equals_ce(self):
        """Focal loss with gamma=0 should equal cross entropy."""
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))

        focal = FocalLoss(gamma=0.0)(logits, targets)
        ce = F.cross_entropy(logits, targets)

        assert torch.allclose(focal, ce, atol=1e-5)

    def test_focal_downweights_easy(self):
        """Focal loss should downweight easy examples."""
        # Create "easy" example (high probability for correct class)
        easy_logits = torch.tensor([[10.0, 0.0, 0.0]])
        easy_target = torch.tensor([0])

        # Create "hard" example (low probability for correct class)
        hard_logits = torch.tensor([[0.0, 10.0, 10.0]])
        hard_target = torch.tensor([0])

        focal = FocalLoss(gamma=2.0, reduction="none")

        easy_loss = focal(easy_logits, easy_target)
        hard_loss = focal(hard_logits, hard_target)

        # Hard example should have higher loss
        assert hard_loss > easy_loss


class TestAdaptiveTemperatureScaling:
    """Tests for adaptive temperature scaling."""

    def test_global_mode(self):
        """Global mode should scale all logits equally."""
        scaler = AdaptiveTemperatureScaling(mode="global", init_temp=1.0)
        logits = torch.randn(2, 16, 100)

        scaled = scaler(logits)
        assert scaled.shape == logits.shape

    def test_per_position_mode(self):
        """Per-position mode should have different temp per position."""
        scaler = AdaptiveTemperatureScaling(mode="per_position", seq_length=16)
        logits = torch.randn(2, 16, 100)

        scaled = scaler(logits)
        assert scaled.shape == logits.shape
        assert scaler.temperature.shape == (16,)

    def test_per_class_mode(self):
        """Per-class mode should have different temp per class."""
        scaler = AdaptiveTemperatureScaling(mode="per_class", vocab_size=100)
        logits = torch.randn(2, 16, 100)

        scaled = scaler(logits)
        assert scaled.shape == logits.shape
        assert scaler.temperature.shape == (100,)


class TestTokenDropout:
    """Tests for token dropout."""

    def test_no_dropout_in_eval(self):
        """Dropout should not apply in eval mode."""
        dropout = TokenDropout(p=0.5, mask_token=0, mode="mask")
        dropout.eval()

        tokens = torch.randint(1, 100, (4, 16))
        output = dropout(tokens)

        assert torch.equal(tokens, output)

    def test_dropout_applies_in_train(self):
        """Dropout should apply in train mode."""
        torch.manual_seed(42)
        dropout = TokenDropout(p=0.5, mask_token=0, mode="mask")
        dropout.train()

        tokens = torch.randint(1, 100, (4, 16))
        output = dropout(tokens)

        # Some tokens should be masked
        assert not torch.equal(tokens, output)
        assert torch.any(output == 0)

    def test_dropout_random_mode(self):
        """Random mode should replace with random tokens."""
        torch.manual_seed(42)
        dropout = TokenDropout(p=0.5, vocab_size=100, mode="random")
        dropout.train()

        tokens = torch.zeros(4, 16).long()  # All zeros
        output = dropout(tokens)

        # Some tokens should be changed to non-zero
        assert torch.any(output != 0)


class TestDistillationLoss:
    """Tests for distillation loss."""

    def test_distillation_combines_losses(self):
        """Distillation should combine soft and hard losses."""
        student = torch.randn(2, 16, 100)
        teacher = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))

        loss = DistillationLoss(temperature=3.0, alpha=0.5)
        result = loss(student, teacher, targets)

        assert result.ndim == 0
        assert torch.isfinite(result)

    def test_alpha_one_ignores_hard_loss(self):
        """Alpha=1 should only use distillation loss."""
        student = torch.randn(2, 16, 100)
        teacher = student.clone()  # Same as teacher
        targets = torch.randint(0, 100, (2, 16))

        # With alpha=1, loss should be ~0 when student matches teacher
        loss = DistillationLoss(temperature=1.0, alpha=1.0)
        result = loss(student, teacher, targets)

        # KL divergence of identical distributions should be ~0
        assert result < 0.1


class TestMetrics:
    """Tests for perplexity and accuracy metrics."""

    def test_perplexity_perfect_prediction(self):
        """Perfect prediction should have low perplexity."""
        vocab_size = 100
        targets = torch.randint(0, vocab_size, (2, 16))

        # Create logits that exactly predict targets
        logits = torch.zeros(2, 16, vocab_size)
        for i in range(2):
            for j in range(16):
                logits[i, j, targets[i, j]] = 100.0

        perplexity = compute_perplexity(logits, targets)
        assert perplexity < 2.0  # Should be close to 1

    def test_accuracy_perfect_prediction(self):
        """Perfect prediction should have accuracy 1.0."""
        vocab_size = 100
        targets = torch.randint(0, vocab_size, (2, 16))

        logits = torch.zeros(2, 16, vocab_size)
        for i in range(2):
            for j in range(16):
                logits[i, j, targets[i, j]] = 100.0

        accuracy = compute_token_accuracy(logits, targets)
        assert accuracy == 1.0

    def test_accuracy_with_ignore_index(self):
        """Accuracy should ignore specified index."""
        logits = torch.randn(2, 16, 100)
        targets = torch.randint(0, 100, (2, 16))
        targets[:, :8] = -100  # Ignore first half

        accuracy = compute_token_accuracy(logits, targets, ignore_index=-100)
        assert 0 <= accuracy <= 1.0
