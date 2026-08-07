"""Tests for torch.compile compatibility of MaskGIT sampling kernels.

These tests verify that MaskGIT step kernels can be compiled with
torch.compile(fullgraph=True), which requires no graph breaks from
.item() calls, data-dependent control flow, or Python side effects.
"""

import pytest
import torch

# Skip all tests if torch.compile is not available (older PyTorch versions)
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile not available in this PyTorch version",
)


class TestMaskGITCompileHealth:
    """Tests for MaskGIT step kernel compile compatibility."""

    @pytest.fixture
    def scheduler(self):
        """Create a MaskGITScheduler for testing."""
        from medlatents.sampling import MaskGITScheduler

        return MaskGITScheduler(num_steps=12, mask_schedule="cosine")

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for step kernel testing."""
        batch_size, seq_len, vocab_size = 2, 64, 512
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        return logits, tokens, mask, device

    def test_scheduler_step_runs(self, scheduler, sample_inputs):
        """Basic test that scheduler.step() runs without error."""
        logits, tokens, mask, device = sample_inputs
        new_tokens, new_mask = scheduler.step(logits, tokens, mask, step=0)

        assert new_tokens.shape == tokens.shape
        assert new_mask.shape == mask.shape
        assert new_tokens.device.type == device
        assert new_mask.device.type == device

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for fullgraph compile test",
    )
    def test_scheduler_step_compiles_with_fullgraph(self, scheduler, sample_inputs):
        """MaskGIT step should compile with fullgraph=True.

        This test will FAIL if there are any graph breaks, such as:
        - .item() calls to sync tensor values to Python
        - Data-dependent control flow (if statements on tensor values)
        - Python side effects (list.append, etc.)

        The step kernels use fixed K_MAX for topk to avoid .item() calls,
        enabling fullgraph compilation.
        """
        logits, tokens, mask, device = sample_inputs

        def step_fn(logits, tokens, mask):
            return scheduler.step(logits, tokens, mask, step=0)

        compiled_step = torch.compile(step_fn, fullgraph=True)
        new_tokens, new_mask = compiled_step(logits, tokens, mask)

        assert new_tokens.shape == tokens.shape
        assert new_mask.shape == mask.shape

    def test_schedule_computation_is_pure(self, scheduler):
        """get_mask_ratio should be a pure function with no side effects."""
        # Call multiple times with same input
        ratio1 = scheduler.get_mask_ratio(5)
        ratio2 = scheduler.get_mask_ratio(5)

        assert ratio1 == ratio2, "get_mask_ratio should be deterministic"

        # Verify it doesn't modify scheduler state
        ratios_before = [scheduler.get_mask_ratio(s) for s in range(12)]
        _ = scheduler.get_mask_ratio(5)
        ratios_after = [scheduler.get_mask_ratio(s) for s in range(12)]

        assert ratios_before == ratios_after, "get_mask_ratio should not modify state"


class TestMaskGITInvariants:
    """Tests for MaskGIT correctness invariants."""

    @pytest.fixture
    def scheduler(self):
        """Create a MaskGITScheduler for testing."""
        from medlatents.sampling import MaskGITScheduler

        return MaskGITScheduler(num_steps=12, mask_schedule="cosine")

    def test_unmasked_positions_unchanged(self, scheduler):
        """Unmasked positions should not change after a step."""
        batch_size, seq_len, vocab_size = 2, 64, 512

        logits = torch.randn(batch_size, seq_len, vocab_size)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Random mask: some positions masked, some not
        mask = torch.rand(batch_size, seq_len) > 0.5

        tokens_before = tokens.clone()

        new_tokens, _ = scheduler.step(logits, tokens, mask, step=0)

        # Unmasked positions should be unchanged
        unmasked_unchanged = (new_tokens[~mask] == tokens_before[~mask]).all()
        assert unmasked_unchanged, "Unmasked positions should not change"

    def test_mask_only_shrinks_in_base_scheduler(self, scheduler):
        """Base scheduler mask should only shrink (no remasking)."""
        batch_size, seq_len, vocab_size = 2, 64, 512

        logits = torch.randn(batch_size, seq_len, vocab_size)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        for step in range(12):
            logits = torch.randn(batch_size, seq_len, vocab_size)
            new_tokens, new_mask = scheduler.step(logits, tokens, mask, step)

            # New mask should be subset of (or equal to) old mask
            # i.e., new_mask implies old mask was True
            assert (new_mask <= mask).all(), (
                f"Mask grew at step {step}: positions became masked that weren't"
            )

            mask = new_mask
            tokens = new_tokens


class TestCompileKernels:
    """Tests for compile-friendly pure step kernels."""

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for kernel testing."""
        batch_size, seq_len, vocab_size = 2, 64, 512
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        return logits, tokens, mask, device

    def test_sample_with_gumbel(self, sample_inputs):
        """Gumbel sampling kernel should work correctly."""
        from medlatents.sampling.maskgit import sample_with_gumbel

        logits, _, _, device = sample_inputs

        sampled = sample_with_gumbel(logits, 1.0)

        assert sampled.shape == logits.shape[:2]
        assert sampled.device.type == device
        assert (sampled >= 0).all()

    def test_step_normal_kernel(self, sample_inputs):
        """Normal step kernel should work correctly."""
        from medlatents.sampling.maskgit import step_normal_kernel

        logits, tokens, mask, device = sample_inputs
        mask_ratio = 0.8
        temperature = 1.0
        K_MAX = 16

        new_tokens, new_mask = step_normal_kernel(
            logits, tokens, mask, mask_ratio, temperature, K_MAX
        )

        assert new_tokens.shape == tokens.shape
        assert new_mask.shape == mask.shape
        assert new_tokens.device.type == device
        assert new_mask.device.type == device
        # Unmasked positions should be unchanged
        unmasked_unchanged = (new_tokens[~mask] == tokens[~mask]).all()
        assert unmasked_unchanged

    def test_step_final_kernel(self, sample_inputs):
        """Final step kernel should unmask all positions."""
        from medlatents.sampling.maskgit import step_final_kernel

        logits, tokens, mask, device = sample_inputs
        temperature = 1.0

        new_tokens, new_mask = step_final_kernel(logits, tokens, mask, temperature)

        assert new_tokens.shape == tokens.shape
        assert new_mask.shape == mask.shape
        assert new_tokens.device.type == device
        assert new_mask.device.type == device
        # Final step should unmask all positions
        assert not new_mask.any(), "Final step should have no masked positions"
        # Unmasked positions should be unchanged
        unmasked_unchanged = (new_tokens[~mask] == tokens[~mask]).all()
        assert unmasked_unchanged

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for compile test",
    )
    def test_step_normal_compiles(self, sample_inputs):
        """Normal step kernel should compile with fullgraph=True."""
        from medlatents.sampling.maskgit import step_normal_kernel

        logits, tokens, mask, device = sample_inputs
        mask_ratio = 0.8
        temperature = 1.0
        K_MAX = 16

        compiled_kernel = torch.compile(
            lambda lo, to, ma: step_normal_kernel(lo, to, ma, mask_ratio, temperature, K_MAX),
            fullgraph=True,
        )

        new_tokens, new_mask = compiled_kernel(logits, tokens, mask)

        assert new_tokens.shape == tokens.shape
        assert new_mask.shape == mask.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for compile test",
    )
    def test_step_final_compiles(self, sample_inputs):
        """Final step kernel should compile with fullgraph=True."""
        from medlatents.sampling.maskgit import step_final_kernel

        logits, tokens, mask, device = sample_inputs
        temperature = 1.0

        compiled_kernel = torch.compile(
            lambda lo, to, ma: step_final_kernel(lo, to, ma, temperature),
            fullgraph=True,
        )

        new_tokens, new_mask = compiled_kernel(logits, tokens, mask)

        assert new_tokens.shape == tokens.shape
        assert new_mask.shape == mask.shape

    def test_compile_cache(self, sample_inputs):
        """CompileCache should return cached compiled kernels."""
        from medlatents.sampling.maskgit import CompileCache

        cache = CompileCache(use_compile=True)

        step_normal = cache.get_step_normal()
        step_final = cache.get_step_final()

        assert step_normal is not None
        assert step_final is not None

        # Subsequent calls should return same compiled function
        assert cache.get_step_normal() is step_normal
        assert cache.get_step_final() is step_final
