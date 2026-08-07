"""Tests for performance optimization utilities."""

import torch
from torch import nn

from medlatents.training import (
    ZeroGradOptimizer,
    compile_model,
    create_fused_adamw,
    enable_cuda_optimizations,
    get_optimal_dtype,
    optimize_dataloader,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class TestCUDAOptimizations:
    """Tests for CUDA optimization utilities."""

    def test_enable_cuda_optimizations_returns_dict(self):
        """enable_cuda_optimizations should return settings dict."""
        settings = enable_cuda_optimizations()
        assert isinstance(settings, dict)
        assert "cuda_available" in settings
        assert "benchmark" in settings
        assert "tf32_matmul" in settings
        assert "deterministic" in settings

    def test_enable_cuda_optimizations_deterministic_mode(self):
        """Deterministic mode should disable benchmark."""
        settings = enable_cuda_optimizations(deterministic=True)
        assert settings["deterministic"] is True
        # Benchmark should be False when deterministic is True
        assert settings["benchmark"] is False

    def test_enable_cuda_optimizations_no_cuda(self):
        """Should handle no CUDA gracefully."""
        # This test works even without CUDA
        settings = enable_cuda_optimizations()
        assert "cuda_available" in settings


class TestFusedAdamW:
    """Tests for fused AdamW optimizer creation."""

    def test_create_fused_adamw_returns_optimizer(self):
        """create_fused_adamw should return AdamW optimizer."""
        model = SimpleModel()
        optimizer = create_fused_adamw(model, lr=1e-4)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_fused_adamw_with_custom_params(self):
        """create_fused_adamw should respect custom parameters."""
        model = SimpleModel()
        optimizer = create_fused_adamw(model, lr=5e-4, weight_decay=0.05, betas=(0.95, 0.999))
        assert optimizer.defaults["lr"] == 5e-4
        assert optimizer.defaults["weight_decay"] == 0.05
        assert optimizer.defaults["betas"] == (0.95, 0.999)


class TestCompileModel:
    """Tests for torch.compile wrapper."""

    def test_compile_model_returns_module(self):
        """compile_model should return a module."""
        model = SimpleModel()
        compiled = compile_model(model)
        # Should return some form of callable (compiled or original)
        assert callable(compiled)

    def test_compile_model_with_different_modes(self):
        """compile_model should work with different modes."""
        model = SimpleModel()
        for mode in ["default", "reduce-overhead"]:
            compiled = compile_model(model, mode=mode)
            assert callable(compiled)


class TestOptimalDtype:
    """Tests for optimal dtype selection."""

    def test_get_optimal_dtype_returns_dtype(self):
        """get_optimal_dtype should return a torch dtype."""
        dtype = get_optimal_dtype()
        assert dtype in [torch.float16, torch.bfloat16, torch.float32]

    def test_get_optimal_dtype_no_cuda_returns_float32(self):
        """Without CUDA, should return float32."""
        if not torch.cuda.is_available():
            dtype = get_optimal_dtype()
            assert dtype == torch.float32


class TestZeroGradOptimizer:
    """Tests for ZeroGradOptimizer wrapper."""

    def test_zero_grad_optimizer_wraps_optimizer(self):
        """ZeroGradOptimizer should wrap an optimizer."""
        model = SimpleModel()
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        wrapped = ZeroGradOptimizer(base_optimizer)
        assert hasattr(wrapped, "zero_grad")
        assert hasattr(wrapped, "step")

    def test_zero_grad_optimizer_zero_grad(self):
        """ZeroGradOptimizer.zero_grad should work."""
        model = SimpleModel()
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        wrapped = ZeroGradOptimizer(base_optimizer)

        # Create some gradients
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()

        # Zero them
        wrapped.zero_grad()

        # Check gradients are None (set_to_none=True)
        for param in model.parameters():
            assert param.grad is None

    def test_zero_grad_optimizer_step(self):
        """ZeroGradOptimizer.step should update parameters."""
        model = SimpleModel()
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        wrapped = ZeroGradOptimizer(base_optimizer)

        # Store original weights
        original_weight = model.fc.weight.clone()

        # Forward and backward
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        wrapped.step()

        # Weights should have changed
        assert not torch.allclose(model.fc.weight, original_weight)

    def test_zero_grad_optimizer_getattr_delegation(self):
        """ZeroGradOptimizer should delegate unknown attributes."""
        model = SimpleModel()
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        wrapped = ZeroGradOptimizer(base_optimizer)

        # Should delegate to wrapped optimizer
        assert wrapped.defaults == base_optimizer.defaults


class TestOptimizeDataloader:
    """Tests for DataLoader optimization settings."""

    def test_optimize_dataloader_returns_dict(self):
        """optimize_dataloader should return settings dict."""
        # Create a dummy dataloader (not actually used)
        settings = optimize_dataloader(None)
        assert isinstance(settings, dict)
        assert "pin_memory" in settings
        assert "num_workers" in settings

    def test_optimize_dataloader_pin_memory(self):
        """pin_memory should be True when CUDA available."""
        settings = optimize_dataloader(None)
        if torch.cuda.is_available():
            assert settings["pin_memory"] is True
        else:
            assert settings["pin_memory"] is False

    def test_optimize_dataloader_explicit_settings(self):
        """Should respect explicit settings."""
        settings = optimize_dataloader(None, pin_memory=False, num_workers=2)
        assert settings["pin_memory"] is False
        assert settings["num_workers"] == 2

    def test_optimize_dataloader_prefetch_with_workers(self):
        """Should include prefetch settings when workers > 0."""
        settings = optimize_dataloader(None, num_workers=4)
        assert settings["num_workers"] == 4
        assert "prefetch_factor" in settings
        assert "persistent_workers" in settings
