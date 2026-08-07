"""Tests for training utilities and schedulers."""

import time

import pytest
import torch
import torch.nn as nn

from medlatents.training import (
    MemoryProfiler,
    TrainingProfiler,
    get_cosine_schedule_with_warmup,
    get_mask_ratio,
)
from medlatents.training.augmentation import (
    AdaptiveLossWeighting,
    CurriculumSampler,
    GradientNoiseInjection,
    MaskingSchedule,
    NoiseSchedule,
    ScheduledSampling,
)


class TestCosineScheduleWithWarmup:
    """Tests for learning rate scheduling with warmup."""

    def test_warmup_linear_increase(self):
        """During warmup, LR should increase linearly from 0 to max."""
        max_lr = 1e-4
        warmup_steps = 100
        total_steps = 1000

        # At step 0
        lr_0 = get_cosine_schedule_with_warmup(0, warmup_steps, total_steps, max_lr, 0)
        assert lr_0 == 0.0

        # At half warmup
        lr_half = get_cosine_schedule_with_warmup(50, warmup_steps, total_steps, max_lr, 0)
        assert abs(lr_half - max_lr * 0.5) < 1e-10

        # At end of warmup
        lr_end = get_cosine_schedule_with_warmup(warmup_steps, warmup_steps, total_steps, max_lr, 0)
        assert abs(lr_end - max_lr) < 1e-10

    def test_cosine_decay_after_warmup(self):
        """After warmup, LR should decay following cosine schedule."""
        max_lr = 1e-4
        min_lr = 1e-6
        warmup_steps = 100
        total_steps = 1000

        # At end of warmup should be max_lr
        lr_warmup_end = get_cosine_schedule_with_warmup(
            warmup_steps, warmup_steps, total_steps, max_lr, min_lr
        )
        assert abs(lr_warmup_end - max_lr) < 1e-10

        # At end of training should be min_lr
        lr_final = get_cosine_schedule_with_warmup(
            total_steps - 1, warmup_steps, total_steps, max_lr, min_lr
        )
        # Allow some tolerance for cosine
        assert abs(lr_final - min_lr) < 1e-6

        # Should decrease monotonically
        prev_lr = max_lr
        for step in range(warmup_steps, total_steps, 50):
            lr = get_cosine_schedule_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr)
            assert lr <= prev_lr
            prev_lr = lr

    def test_warmup_equals_total_returns_max(self):
        """When warmup_steps == total_steps, should return max_lr."""
        max_lr = 1e-4
        lr = get_cosine_schedule_with_warmup(50, 100, 100, max_lr, 1e-6)
        assert lr == max_lr

    def test_various_step_values(self):
        """Test schedule at various points."""
        max_lr = 1e-4
        min_lr = 1e-6
        warmup_steps = 1000
        total_steps = 10000

        test_points = [
            (0, 0.0),  # Start
            (500, max_lr * 0.5),  # Half warmup
            (1000, max_lr),  # End warmup
        ]

        for step, expected in test_points:
            lr = get_cosine_schedule_with_warmup(step, warmup_steps, total_steps, max_lr, min_lr)
            assert abs(lr - expected) < 1e-10, f"Step {step}: expected {expected}, got {lr}"


class TestMaskRatioSchedule:
    """Tests for mask ratio scheduling during MaskGIT training."""

    def test_cosine_schedule_boundaries(self):
        """Cosine schedule should start at 1 and end at 0."""
        # At progress=0, should be 1.0
        ratio_start = get_mask_ratio(0.0, schedule="cosine")
        assert abs(ratio_start - 1.0) < 1e-10

        # At progress=1, should be 0.0
        ratio_end = get_mask_ratio(1.0, schedule="cosine")
        assert abs(ratio_end - 0.0) < 1e-10

        # At progress=0.5, should be 0.5
        ratio_mid = get_mask_ratio(0.5, schedule="cosine")
        assert abs(ratio_mid - 0.5) < 1e-10

    def test_linear_schedule(self):
        """Linear schedule should decrease linearly."""
        # At progress=0, should be 1.0
        assert get_mask_ratio(0.0, schedule="linear") == 1.0

        # At progress=1, should be 0.0
        assert get_mask_ratio(1.0, schedule="linear") == 0.0

        # At progress=0.5, should be 0.5
        assert get_mask_ratio(0.5, schedule="linear") == 0.5

        # At progress=0.25, should be 0.75
        assert get_mask_ratio(0.25, schedule="linear") == 0.75

    def test_square_schedule(self):
        """Square schedule should decrease quadratically."""
        # At progress=0, should be 1.0
        assert get_mask_ratio(0.0, schedule="square") == 1.0

        # At progress=1, should be 0.0
        assert get_mask_ratio(1.0, schedule="square") == 0.0

        # At progress=0.5, should be 0.25
        assert get_mask_ratio(0.5, schedule="square") == 0.25

    def test_schedules_monotonic_decrease(self):
        """All schedules should monotonically decrease."""
        for schedule in ["cosine", "linear", "square"]:
            prev_ratio = 1.0
            for progress in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                ratio = get_mask_ratio(progress, schedule=schedule)
                assert ratio <= prev_ratio, f"{schedule} not monotonic at {progress}"
                prev_ratio = ratio

    def test_unknown_schedule_raises(self):
        """Unknown schedule should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_mask_ratio(0.5, schedule="unknown")


class TestTrainingProfiler:
    """Tests for the TrainingProfiler utility."""

    def test_section_timing_basic(self):
        """Basic section timing should record elapsed time."""
        profiler = TrainingProfiler(enabled=True, sync_cuda=False)

        with profiler.section("test_section"):
            time.sleep(0.01)  # 10ms

        stats = profiler.get_stats()
        assert "test_section" in stats
        assert stats["test_section"]["count"] == 1
        assert stats["test_section"]["avg_ms"] >= 5  # At least 5ms (lenient for CI)

    def test_disabled_profiler_no_overhead(self):
        """Disabled profiler should not collect stats."""
        profiler = TrainingProfiler(enabled=False)

        with profiler.section("test_section"):
            pass

        stats = profiler.get_stats()
        assert len(stats) == 0

    def test_multiple_sections(self):
        """Multiple sections should be tracked separately."""
        profiler = TrainingProfiler(enabled=True, sync_cuda=False)

        with profiler.section("forward"):
            time.sleep(0.005)

        with profiler.section("backward"):
            time.sleep(0.005)

        with profiler.section("forward"):
            time.sleep(0.005)

        stats = profiler.get_stats()
        assert stats["forward"]["count"] == 2
        assert stats["backward"]["count"] == 1

    def test_get_percentages(self):
        """Percentages should sum to 100."""
        profiler = TrainingProfiler(enabled=True, sync_cuda=False)

        with profiler.section("a"):
            time.sleep(0.01)

        with profiler.section("b"):
            time.sleep(0.01)

        percentages = profiler.get_percentages()
        total = sum(percentages.values())
        assert abs(total - 100.0) < 1.0  # Allow some floating point error

    def test_wandb_metrics_format(self):
        """Wandb metrics should have correct format."""
        profiler = TrainingProfiler(enabled=True, sync_cuda=False)

        with profiler.section("forward"):
            pass

        metrics = profiler.get_wandb_metrics(prefix="train/")
        assert "train/forward_avg_ms" in metrics
        assert "train/forward_total_ms" in metrics

    def test_reset_clears_stats(self):
        """Reset should clear all statistics."""
        profiler = TrainingProfiler(enabled=True, sync_cuda=False)

        with profiler.section("test"):
            pass

        assert len(profiler.get_stats()) > 0
        profiler.reset()
        assert len(profiler.get_stats()) == 0


class TestMemoryProfiler:
    """Tests for the MemoryProfiler utility."""

    def test_memory_profiler_disabled_without_cuda(self):
        """Memory profiler should be disabled when CUDA unavailable."""
        profiler = MemoryProfiler(enabled=True)

        # If no CUDA, profiler should be disabled
        if not torch.cuda.is_available():
            assert not profiler.enabled

    def test_memory_tracking_no_error(self):
        """Memory tracking should not raise errors."""
        profiler = MemoryProfiler(enabled=True)

        # This should work regardless of CUDA availability
        with profiler.track("test"):
            _ = torch.zeros(100, 100)

        # Should not raise
        _ = profiler.get_stats()


class TestCurriculumSampler:
    """Tests for CurriculumSampler."""

    def test_difficulty_starts_at_start_value(self):
        """Difficulty should start at start_difficulty."""
        sampler = CurriculumSampler(start_difficulty=0.3, end_difficulty=1.0)
        assert sampler.get_difficulty() == 0.3

    def test_difficulty_increases_with_steps(self):
        """Difficulty should increase as steps progress."""
        sampler = CurriculumSampler(
            start_difficulty=0.3, end_difficulty=1.0, num_steps=100, schedule="linear"
        )

        d1 = sampler.get_difficulty()
        sampler.step()
        sampler.step()
        sampler.step()
        d2 = sampler.get_difficulty()

        assert d2 > d1

    def test_difficulty_reaches_end_value(self):
        """Difficulty should reach end_difficulty after num_steps."""
        sampler = CurriculumSampler(
            start_difficulty=0.3, end_difficulty=1.0, num_steps=10, schedule="linear"
        )

        for _ in range(20):  # More than num_steps
            sampler.step()

        assert abs(sampler.get_difficulty() - 1.0) < 1e-6

    def test_different_schedules(self):
        """Test different schedule types."""
        for schedule in ["linear", "exp", "cosine"]:
            sampler = CurriculumSampler(
                start_difficulty=0.3, end_difficulty=1.0, num_steps=100, schedule=schedule
            )
            # Should not raise
            _ = sampler.get_difficulty()


class TestScheduledSampling:
    """Tests for ScheduledSampling."""

    def test_sampling_prob_starts_at_start_value(self):
        """Sampling probability should start at start_sampling_prob."""
        ss = ScheduledSampling(start_sampling_prob=0.0, end_sampling_prob=1.0)
        assert ss.get_sampling_prob() == 0.0

    def test_sample_tokens_mixes_correctly(self):
        """sample_tokens should mix ground truth and predictions."""
        ss = ScheduledSampling(start_sampling_prob=0.5, end_sampling_prob=0.5, num_steps=1)

        gt = torch.zeros(10, 20, dtype=torch.long)
        pred = torch.ones(10, 20, dtype=torch.long)

        mixed = ss.sample_tokens(gt, pred)

        # With p=0.5, should have roughly half of each
        ones_ratio = mixed.float().mean().item()
        assert 0.2 < ones_ratio < 0.8  # Allow some variance

    def test_different_modes(self):
        """Test different schedule modes."""
        for mode in ["linear", "exp", "inverse_sigmoid"]:
            ss = ScheduledSampling(mode=mode, num_steps=100)
            # Should not raise
            _ = ss.get_sampling_prob()


class TestNoiseSchedule:
    """Tests for NoiseSchedule."""

    def test_noise_decreases_over_time(self):
        """Noise level should decrease with steps."""
        ns = NoiseSchedule(start_noise=0.5, end_noise=0.01, num_steps=100, schedule="linear")

        n1 = ns.get_noise_level()
        assert n1 == 0.5

        # Advance steps by calling add_noise
        tokens = torch.zeros(4, 10, dtype=torch.long)
        for _ in range(100):
            ns.add_noise(tokens, vocab_size=256)

        n2 = ns.get_noise_level()
        assert n2 < n1

    def test_add_noise_produces_noisy_tokens(self):
        """add_noise should modify some tokens."""
        ns = NoiseSchedule(noise_type="uniform", start_noise=0.5, end_noise=0.5)
        tokens = torch.zeros(10, 50, dtype=torch.long)
        noisy = ns.add_noise(tokens, vocab_size=256)

        # Should have some non-zero tokens
        assert noisy.sum() > 0


class TestMaskingSchedule:
    """Tests for MaskingSchedule."""

    def test_mask_ratio_decreases(self):
        """Mask ratio should decrease over time."""
        ms = MaskingSchedule(
            start_mask_ratio=0.8, end_mask_ratio=0.2, num_steps=100, schedule="linear"
        )

        r1 = ms.get_mask_ratio()
        assert r1 == 0.8

        for _ in range(100):
            ms.step()

        r2 = ms.get_mask_ratio()
        assert r2 < r1

    def test_constant_schedule(self):
        """Constant schedule should not change."""
        ms = MaskingSchedule(start_mask_ratio=0.5, end_mask_ratio=0.2, schedule="constant")

        r1 = ms.get_mask_ratio()
        for _ in range(50):
            ms.step()
        r2 = ms.get_mask_ratio()

        assert r1 == r2 == 0.5

    def test_min_ratio_enforced(self):
        """Min ratio should be enforced."""
        ms = MaskingSchedule(start_mask_ratio=0.5, end_mask_ratio=0.0, min_ratio=0.1, num_steps=10)

        for _ in range(100):
            ms.step()

        assert ms.get_mask_ratio() >= 0.1


class TestGradientNoiseInjection:
    """Tests for GradientNoiseInjection."""

    def test_no_noise_when_not_training(self):
        """Should not add noise when model is not in training mode."""
        gni = GradientNoiseInjection(eta=0.1)
        model = nn.Linear(10, 10)
        model.training = False  # Set to inference mode

        # Create fake gradients
        x = torch.randn(5, 10)
        y = model(x).sum()
        y.backward()

        grad_before = model.weight.grad.clone()
        gni.add_noise(model)
        grad_after = model.weight.grad

        # Gradients should be unchanged
        assert torch.allclose(grad_before, grad_after)

    def test_noise_added_in_train_mode(self):
        """Should add noise when model is in train mode."""
        gni = GradientNoiseInjection(eta=1.0)  # Large eta for visible effect
        model = nn.Linear(10, 10)
        model.train()

        # Create fake gradients
        x = torch.randn(5, 10)
        y = model(x).sum()
        y.backward()

        grad_before = model.weight.grad.clone()
        gni.add_noise(model)
        grad_after = model.weight.grad

        # Gradients should be changed
        assert not torch.allclose(grad_before, grad_after)

    def test_noise_decreases_over_time(self):
        """Noise should decrease with annealing."""
        gni = GradientNoiseInjection(eta=1.0, gamma=0.55)

        model = nn.Linear(10, 10)
        model.train()

        diffs = []
        for _ in range(10):
            x = torch.randn(5, 10)
            y = model(x).sum()
            y.backward()
            grad_before = model.weight.grad.clone()
            gni.add_noise(model)
            diff = (model.weight.grad - grad_before).abs().mean().item()
            diffs.append(diff)
            model.zero_grad()

        # Later diffs should generally be smaller (noise decreases)
        # Just check first is larger than last
        assert diffs[0] > diffs[-1]


class TestAdaptiveLossWeighting:
    """Tests for AdaptiveLossWeighting."""

    def test_uncertainty_mode_creates_parameters(self):
        """Uncertainty mode should create learnable parameters."""
        alw = AdaptiveLossWeighting(num_losses=3, mode="uncertainty")
        assert hasattr(alw, "log_vars")
        assert alw.log_vars.requires_grad

    def test_grad_norm_mode_tracks_norms(self):
        """Grad norm mode should track gradient norms."""
        alw = AdaptiveLossWeighting(num_losses=2, mode="grad_norm")
        assert hasattr(alw, "grad_norms")
        assert len(alw.grad_norms) == 2

    def test_weighted_loss_combines_losses(self):
        """compute_weighted_loss should combine losses."""
        alw = AdaptiveLossWeighting(num_losses=2, mode="uncertainty")
        losses = [torch.tensor(1.0), torch.tensor(2.0)]
        weighted = alw.compute_weighted_loss(losses)
        assert weighted.item() > 0
