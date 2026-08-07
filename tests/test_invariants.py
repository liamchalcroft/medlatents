"""Property-based tests for key invariants.

Uses hypothesis to test invariants across a wide range of inputs.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from medlatents.flow_matching.core import MixtureDiscreteProbPath, PolynomialConvexScheduler
from medlatents.rasterization.hilbert_curve import HilbertCurve
from medlatents.rasterization.raster_scan import RasterScan
from medlatents.rasterization.zorder_curve import ZOrderCurve
from medlatents.sampling.autoregressive import (
    TemperatureScheduler,
    sample_min_p,
    sample_nucleus,
    sample_top_k,
    sample_with_cfg,
)


class TestRasterizationInvariants:
    """Property-based tests for rasterization invariants."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.integers(min_value=2, max_value=16),
        width=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_hilbert_2d_round_trip_invariant(self, batch_size, height, width):
        """Hilbert curve 2D: unflatten(flatten(x)) == x for any valid input."""
        data = torch.randn(batch_size, height, width)
        sequence, metadata = HilbertCurve.spatial_to_sequence_2d(data)
        reconstructed = HilbertCurve.sequence_to_spatial_2d(sequence, metadata)
        assert torch.allclose(reconstructed, data)

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.integers(min_value=2, max_value=16),
        width=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_zorder_2d_round_trip_invariant(self, batch_size, height, width):
        """ZOrder curve 2D: unflatten(flatten(x)) == x for any valid input."""
        data = torch.randn(batch_size, height, width)
        sequence, metadata = ZOrderCurve.spatial_to_sequence_2d(data)
        reconstructed = ZOrderCurve.sequence_to_spatial_2d(sequence, metadata)
        assert torch.allclose(reconstructed, data)

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.integers(min_value=2, max_value=16),
        width=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_raster_scan_2d_round_trip_invariant(self, batch_size, height, width):
        """RasterScan 2D: unflatten(flatten(x)) == x for any valid input."""
        data = torch.randn(batch_size, height, width)
        sequence = RasterScan.spatial_to_sequence_2d(data)
        reconstructed = RasterScan.sequence_to_spatial_2d(sequence, height, width)
        assert torch.allclose(reconstructed, data)

    @given(
        batch_size=st.integers(min_value=1, max_value=2),
        depth=st.integers(min_value=2, max_value=8),
        height=st.integers(min_value=2, max_value=8),
        width=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=10, deadline=None)
    def test_hilbert_3d_round_trip_invariant(self, batch_size, depth, height, width):
        """Hilbert curve 3D: unflatten(flatten(x)) == x for any valid input."""
        data = torch.randn(batch_size, depth, height, width)
        sequence, metadata = HilbertCurve.spatial_to_sequence_3d(data)
        reconstructed = HilbertCurve.sequence_to_spatial_3d(sequence, metadata)
        assert torch.allclose(reconstructed, data)

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        height=st.integers(min_value=2, max_value=16),
        width=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=20, deadline=None)
    def test_sequence_length_preserved(self, batch_size, height, width):
        """Sequence length equals product of spatial dimensions."""
        data = torch.randn(batch_size, height, width)

        seq_hilbert, _ = HilbertCurve.spatial_to_sequence_2d(data)
        seq_zorder, _ = ZOrderCurve.spatial_to_sequence_2d(data)
        seq_raster = RasterScan.spatial_to_sequence_2d(data)

        expected_len = height * width
        assert seq_hilbert.size(-1) == expected_len
        assert seq_zorder.size(-1) == expected_len
        assert seq_raster.size(-1) == expected_len


class TestSamplingInvariants:
    """Property-based tests for sampling invariants."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        vocab_size=st.integers(min_value=10, max_value=100),
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30, deadline=None)
    def test_top_k_keeps_at_most_k(self, batch_size, vocab_size, k):
        """Top-k sampling keeps at most k non-inf values per row."""
        logits = torch.randn(batch_size, vocab_size)
        filtered = sample_top_k(logits, k=min(k, vocab_size))

        for i in range(batch_size):
            non_inf = (filtered[i] != float("-inf")).sum()
            assert non_inf <= min(k, vocab_size)

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        vocab_size=st.integers(min_value=10, max_value=100),
        p=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_nucleus_keeps_at_least_one(self, batch_size, vocab_size, p):
        """Nucleus sampling always keeps at least one token."""
        logits = torch.randn(batch_size, vocab_size)
        filtered = sample_nucleus(logits, p=p)

        for i in range(batch_size):
            non_inf = (filtered[i] != float("-inf")).sum()
            assert non_inf >= 1

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        vocab_size=st.integers(min_value=10, max_value=100),
        min_p=st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=30, deadline=None)
    def test_min_p_filters_relative_to_max(self, batch_size, vocab_size, min_p):
        """Min-p filters tokens relative to max probability."""
        logits = torch.randn(batch_size, vocab_size)
        filtered = sample_min_p(logits, min_p=min_p)

        # At least the max should always be kept
        for i in range(batch_size):
            non_inf = (filtered[i] != float("-inf")).sum()
            assert non_inf >= 1

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        vocab_size=st.integers(min_value=10, max_value=100),
        guidance_scale=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_cfg_interpolation(self, batch_size, vocab_size, guidance_scale):
        """CFG correctly interpolates between conditional and unconditional."""
        cond = torch.randn(batch_size, vocab_size)
        uncond = torch.randn(batch_size, vocab_size)

        result = sample_with_cfg(cond, uncond, guidance_scale=guidance_scale)

        # Verify formula: result = uncond + scale * (cond - uncond)
        expected = uncond + guidance_scale * (cond - uncond)
        assert torch.allclose(result, expected)


class TestTemperatureSchedulerInvariants:
    """Property-based tests for temperature scheduler invariants."""

    @given(
        start_temp=st.floats(min_value=0.5, max_value=2.0),
        end_temp=st.floats(min_value=0.1, max_value=1.5),
        num_steps=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_linear_schedule_monotonic(self, start_temp, end_temp, num_steps):
        """Linear schedule is monotonic between start and end."""
        scheduler = TemperatureScheduler(
            schedule="linear",
            start_temp=start_temp,
            end_temp=end_temp,
            num_steps=num_steps,
        )

        temps = [scheduler.get_temperature(i) for i in range(num_steps)]

        # Check monotonicity
        if start_temp >= end_temp:
            for i in range(len(temps) - 1):
                assert temps[i] >= temps[i + 1] - 1e-6
        else:
            for i in range(len(temps) - 1):
                assert temps[i] <= temps[i + 1] + 1e-6

    @given(
        start_temp=st.floats(min_value=0.5, max_value=2.0),
        end_temp=st.floats(min_value=0.1, max_value=1.5),
        num_steps=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_schedule_boundaries(self, start_temp, end_temp, num_steps):
        """All schedule types start at start_temp and end at end_temp."""
        for schedule in ["linear", "cosine", "exponential"]:
            scheduler = TemperatureScheduler(
                schedule=schedule,
                start_temp=start_temp,
                end_temp=end_temp,
                num_steps=num_steps,
            )

            first = scheduler.get_temperature(0)
            last = scheduler.get_temperature(num_steps - 1)

            assert abs(first - start_temp) < 1e-5
            assert abs(last - end_temp) < 1e-5


class TestFlowMatchingInvariants:
    """Property-based tests for flow matching invariants."""

    @given(
        n=st.floats(min_value=0.5, max_value=3.0),
        t=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_kappa_inverse_is_inverse(self, n, t):
        """kappa_inverse(kappa(t)) == t for polynomial scheduler."""
        scheduler = PolynomialConvexScheduler(n=n)
        t_tensor = torch.tensor(t)

        kappa_t = scheduler.kappa(t_tensor)
        recovered = scheduler.kappa_inverse(kappa_t)

        assert torch.allclose(recovered, t_tensor, atol=1e-5)

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_length=st.integers(min_value=4, max_value=32),
        vocab_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_path_sample_t0_equals_source(self, batch_size, seq_length, vocab_size):
        """At t=0, path.sample returns source distribution."""
        scheduler = PolynomialConvexScheduler(n=1.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)

        x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_length))
        t = torch.zeros(batch_size)

        sample = path.sample(t, x_0, x_1)
        assert (sample.x_t == x_0).all()

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_length=st.integers(min_value=4, max_value=32),
        vocab_size=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_path_sample_t1_equals_target(self, batch_size, seq_length, vocab_size):
        """At t=1, path.sample returns target distribution."""
        scheduler = PolynomialConvexScheduler(n=1.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)

        x_0 = torch.randint(0, vocab_size, (batch_size, seq_length))
        x_1 = torch.randint(0, vocab_size, (batch_size, seq_length))
        t = torch.ones(batch_size)

        sample = path.sample(t, x_0, x_1)
        assert (sample.x_t == x_1).all()


class TestCausalMaskingInvariant:
    """Test that autoregressive models respect causal masking."""

    @given(
        seq_length=st.integers(min_value=4, max_value=32),
        vocab_size=st.integers(min_value=10, max_value=64),
    )
    @settings(max_examples=10, deadline=None)
    def test_ar_output_independent_of_future(self, seq_length, vocab_size):
        """AR model output at position i is independent of tokens at positions > i."""
        from medlatents.autoregressive.transformer import AutoregressiveTransformer

        model = AutoregressiveTransformer(
            seq_length=seq_length,
            vocab_size=vocab_size,
            hidden_size=32,
            depth=2,
            num_heads=2,
        )
        model.eval()

        # Create two sequences that differ only in later positions
        seq1 = torch.randint(0, vocab_size, (1, seq_length))
        seq2 = seq1.clone()
        # Modify positions 5 and onwards
        change_pos = min(5, seq_length - 1)
        seq2[0, change_pos:] = torch.randint(0, vocab_size, (seq_length - change_pos,))

        with torch.no_grad():
            out1 = model(seq1)
            out2 = model(seq2)

        # Outputs at positions < change_pos should be identical
        # (causal masking means future tokens can't influence past predictions)
        assert torch.allclose(out1[:, :change_pos], out2[:, :change_pos], atol=1e-5)
