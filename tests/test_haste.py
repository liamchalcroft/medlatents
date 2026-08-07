"""Tests for HASTEScheduler.

Tests the actual HASTEScheduler implementation for phase-based training control
with termination modes (hard/linear/cosine).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from medlatents.training import HASTEScheduler


class TestHASTEBasics:
    """Basic functionality tests for HASTEScheduler."""

    @given(
        termination_step=st.integers(min_value=100, max_value=10000),
        transition_steps=st.integers(min_value=100, max_value=1000),
        initial_repa_weight=st.floats(
            min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_scheduler_initialization(
        self, termination_step: int, transition_steps: int, initial_repa_weight: float
    ):
        """HASTEScheduler should initialize correctly."""
        scheduler = HASTEScheduler(
            termination_step=termination_step,
            termination_mode="hard",
            transition_steps=transition_steps,
            initial_repa_weight=initial_repa_weight,
        )

        assert scheduler.termination_step == termination_step
        assert scheduler.transition_steps == transition_steps
        assert scheduler.initial_repa_weight == initial_repa_weight
        assert scheduler.phase == 1  # Initial phase

    def test_invalid_termination_step_raises(self):
        """Negative termination_step should raise ValueError."""
        with pytest.raises(ValueError, match="termination_step must be positive"):
            HASTEScheduler(termination_step=-10)

        with pytest.raises(ValueError, match="termination_step must be positive"):
            HASTEScheduler(termination_step=0)


class TestHASTEPhaseTransitions:
    """Tests for phase transition behavior."""

    def test_phase1_before_termination(self):
        """Before termination_step, should be in phase 1 with full weight."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="hard",
            initial_repa_weight=0.5,
        )

        for step in [0, 100, 500, 999]:
            weight = scheduler.get_repa_weight(step)
            assert weight == 0.5, f"Step {step}: weight should be 0.5, got {weight}"
            assert scheduler.phase == 1

    def test_hard_termination(self):
        """Hard mode should immediately cut to zero at termination_step."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="hard",
            initial_repa_weight=0.5,
        )

        # Before termination
        assert scheduler.get_repa_weight(999) == 0.5
        assert scheduler.phase == 1

        # At termination
        assert scheduler.get_repa_weight(1000) == 0.0
        assert scheduler.phase == 2

        # After termination
        assert scheduler.get_repa_weight(2000) == 0.0
        assert scheduler.phase == 2

    def test_linear_termination(self):
        """Linear mode should decay weight over transition_steps."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="linear",
            transition_steps=100,
            initial_repa_weight=0.5,
        )

        # Before termination
        assert scheduler.get_repa_weight(999) == 0.5

        # At termination (start of transition)
        weight_at_term = scheduler.get_repa_weight(1000)
        assert weight_at_term == 0.5  # Still full at start

        # Midway through transition
        weight_mid = scheduler.get_repa_weight(1050)
        assert 0.0 < weight_mid < 0.5, (
            f"Mid-transition weight should be between 0 and 0.5, got {weight_mid}"
        )

        # After transition
        weight_after = scheduler.get_repa_weight(1100)
        assert weight_after == 0.0

    def test_cosine_termination(self):
        """Cosine mode should decay weight smoothly."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="cosine",
            transition_steps=100,
            initial_repa_weight=0.5,
        )

        # Before termination
        assert scheduler.get_repa_weight(999) == 0.5

        # During transition - should decay smoothly
        weights = [scheduler.get_repa_weight(1000 + i) for i in range(0, 101, 10)]

        # Weights should be monotonically non-increasing
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], f"Weights should decrease: {weights}"

        # After transition
        assert scheduler.get_repa_weight(1100) == 0.0


class TestHASTEAttentionWeight:
    """Tests for attention alignment weight."""

    def test_attention_weight_follows_repa(self):
        """Attention weight should scale with REPA weight."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="hard",
            initial_repa_weight=0.5,
            initial_attention_weight=0.1,
            use_attention_alignment=True,
        )

        # Before termination
        repa_weight = scheduler.get_repa_weight(500)
        attn_weight = scheduler.get_attention_weight(500)
        assert repa_weight == 0.5
        assert attn_weight == 0.1

        # After termination
        repa_weight = scheduler.get_repa_weight(1000)
        attn_weight = scheduler.get_attention_weight(1000)
        assert repa_weight == 0.0
        assert attn_weight == 0.0

    def test_attention_disabled(self):
        """Attention weight should be zero when disabled."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            use_attention_alignment=False,
            initial_attention_weight=0.1,
        )

        assert scheduler.get_attention_weight(500) == 0.0
        assert scheduler.get_attention_weight(1500) == 0.0


class TestHASTEHelperMethods:
    """Tests for helper methods."""

    def test_should_compute_alignment(self):
        """should_compute_alignment should return True only in phase 1."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="hard",
        )

        assert scheduler.should_compute_alignment(500) is True
        assert scheduler.should_compute_alignment(1000) is False
        assert scheduler.should_compute_alignment(1500) is False

    def test_state_dict_roundtrip(self):
        """State dict should preserve scheduler state."""
        scheduler = HASTEScheduler(
            termination_step=1000,
            termination_mode="linear",
            transition_steps=100,
            initial_repa_weight=0.5,
        )

        # Advance to phase 2
        scheduler.get_repa_weight(1200)
        assert scheduler.phase == 2

        # Save and load state
        state = scheduler.state_dict()

        new_scheduler = HASTEScheduler(termination_step=1000)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.phase == 2


class TestHASTEEdgeCases:
    """Edge case tests."""

    def test_single_transition_step(self):
        """Single transition step should work."""
        scheduler = HASTEScheduler(
            termination_step=100,
            termination_mode="linear",
            transition_steps=1,
            initial_repa_weight=1.0,
        )

        # At termination
        assert scheduler.get_repa_weight(100) == 1.0
        # After single step
        assert scheduler.get_repa_weight(101) == 0.0

    @given(
        termination_step=st.integers(min_value=1, max_value=10000),
    )
    def test_weight_always_finite(self, termination_step: int):
        """Weight should always be finite and in valid range."""
        scheduler = HASTEScheduler(
            termination_step=termination_step,
            termination_mode="cosine",
            transition_steps=100,
            initial_repa_weight=0.5,
        )

        for step in [
            0,
            termination_step // 2,
            termination_step,
            termination_step + 50,
            termination_step + 200,
        ]:
            weight = scheduler.get_repa_weight(step)
            assert isinstance(weight, float), f"Weight should be float, got {type(weight)}"
            assert 0.0 <= weight <= 0.5, f"Weight should be in [0, 0.5], got {weight}"
