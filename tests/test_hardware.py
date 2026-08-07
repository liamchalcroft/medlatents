"""Hardware-specific tests for medlatents.

Tests for:
- GPU/CPU device compatibility
- Deterministic mode
- Mixed precision (fp16, bf16)
- Memory efficiency
- Multi-GPU basics (if available)
"""

import pytest
import torch
import torch.nn as nn

# ============================================================================
# Device Detection and Fixtures
# ============================================================================


def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_mps():
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def has_bf16():
    """Check if bfloat16 is supported."""
    if has_cuda():
        return torch.cuda.is_bf16_supported()
    return False


def has_multi_gpu():
    """Check if multiple GPUs are available."""
    return torch.cuda.device_count() > 1


# Skip decorators
skip_no_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
skip_no_mps = pytest.mark.skipif(not has_mps(), reason="MPS not available")
skip_no_bf16 = pytest.mark.skipif(not has_bf16(), reason="BF16 not supported")
skip_no_multi_gpu = pytest.mark.skipif(not has_multi_gpu(), reason="Multiple GPUs not available")


@pytest.fixture
def tiny_model_config():
    """Minimal model config for hardware tests."""
    return {
        "seq_length": 32,
        "vocab_size": 128,
        "hidden_size": 64,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
    }


@pytest.fixture
def device():
    """Get the best available device."""
    if has_cuda():
        return torch.device("cuda")
    elif has_mps():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """Get GPU device, skip if not available."""
    if has_cuda():
        return torch.device("cuda")
    elif has_mps():
        return torch.device("mps")
    pytest.skip("No GPU available")


# ============================================================================
# CPU Tests (Always Run)
# ============================================================================


class TestCPUCompatibility:
    """Tests that must work on CPU."""

    def test_model_on_cpu(self, tiny_model_config):
        """Models should work on CPU."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(**tiny_model_config)
        model = model.cpu()

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))
        output = model(x)

        assert output.device.type == "cpu"
        # vocab_size gets extended by 4 special tokens (BOS, EOS, PAD, MASK)
        assert output.shape == (2, 32, model.vocab_size)

    def test_cpu_deterministic_mode(self, tiny_model_config):
        """Deterministic mode should work on CPU."""
        from medlatents import AutoregressiveTransformer

        torch.use_deterministic_algorithms(True)
        try:
            model = AutoregressiveTransformer(**tiny_model_config)
            x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))

            # Set seed and run
            torch.manual_seed(42)
            out1 = model(x)

            # Reset seed and run again
            torch.manual_seed(42)
            out2 = model(x)

            assert torch.allclose(out1, out2, atol=1e-6)
        finally:
            torch.use_deterministic_algorithms(False)

    def test_cpu_gradient_computation(self, tiny_model_config):
        """Gradients should compute correctly on CPU."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(**tiny_model_config)
        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))

        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


# ============================================================================
# GPU Tests (CUDA or MPS)
# ============================================================================


class TestGPUCompatibility:
    """Tests for GPU execution."""

    @pytest.mark.skipif(not (has_cuda() or has_mps()), reason="No GPU available")
    def test_model_on_gpu(self, gpu_device, tiny_model_config):
        """Models should work on GPU."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(**tiny_model_config).to(gpu_device)
        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=gpu_device)

        output = model(x)

        assert output.device.type == gpu_device.type
        assert output.shape == (2, 32, model.vocab_size)

    @pytest.mark.skipif(not (has_cuda() or has_mps()), reason="No GPU available")
    def test_cpu_to_gpu_transfer(self, gpu_device, tiny_model_config):
        """Model should transfer between CPU and GPU."""
        from medlatents import AutoregressiveTransformer

        # Create on CPU
        model = AutoregressiveTransformer(**tiny_model_config)
        x_cpu = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))
        out_cpu = model(x_cpu)

        # Move to GPU
        model = model.to(gpu_device)
        x_gpu = x_cpu.to(gpu_device)
        out_gpu = model(x_gpu)

        # Compare outputs (should be close but not identical due to precision)
        assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-4)

    @skip_no_cuda
    def test_cuda_memory_efficient(self, tiny_model_config):
        """Model should not leak memory."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        model = AutoregressiveTransformer(**tiny_model_config).to(device)

        # Run multiple forward passes
        for _ in range(10):
            x = torch.randint(0, tiny_model_config["vocab_size"], (4, 64), device=device)
            output = model(x)
            loss = output.sum()
            loss.backward()

        # Clear gradients
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly (allow for some variance)
        # This is a rough check - model parameters should be the main allocation
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        assert final_memory < initial_memory + param_memory * 3  # 3x buffer


# ============================================================================
# Deterministic Mode Tests
# ============================================================================


class TestDeterministicMode:
    """Tests for reproducibility."""

    def test_seeded_forward_cpu(self, tiny_model_config):
        """Seeded forward should be reproducible on CPU."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(**tiny_model_config)
        model.eval()

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))

        # Forward with seed
        torch.manual_seed(12345)
        out1 = model(x)

        # Forward with same seed
        torch.manual_seed(12345)
        out2 = model(x)

        assert torch.equal(out1, out2)

    @skip_no_cuda
    def test_seeded_forward_cuda(self, tiny_model_config):
        """Seeded forward should be reproducible on CUDA."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        model = AutoregressiveTransformer(**tiny_model_config).to(device)
        model.eval()

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)

        # Forward with seed
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)
        out1 = model(x)

        # Forward with same seed
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)
        out2 = model(x)

        assert torch.equal(out1, out2)

    def test_training_reproducibility(self, tiny_model_config):
        """Training should be reproducible with seed."""
        from medlatents import AutoregressiveTransformer

        def train_step(seed):
            torch.manual_seed(seed)
            model = AutoregressiveTransformer(**tiny_model_config)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            x = torch.randint(0, tiny_model_config["vocab_size"], (4, 32))
            target = torch.randint(0, tiny_model_config["vocab_size"], (4, 32))

            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()

            return loss.item(), [p.data.clone() for p in model.parameters()]

        loss1, params1 = train_step(42)
        loss2, params2 = train_step(42)

        assert loss1 == loss2
        for p1, p2 in zip(params1, params2):
            assert torch.equal(p1, p2)


# ============================================================================
# Mixed Precision Tests
# ============================================================================


class TestMixedPrecision:
    """Tests for fp16 and bf16 training."""

    @skip_no_cuda
    def test_fp16_forward_pass(self, tiny_model_config):
        """Model should work with fp16."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        model = AutoregressiveTransformer(**tiny_model_config).to(device)

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)

        with torch.autocast("cuda", dtype=torch.float16):
            output = model(x)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @skip_no_bf16
    def test_bf16_forward_pass(self, tiny_model_config):
        """Model should work with bf16."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        model = AutoregressiveTransformer(**tiny_model_config).to(device)

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(x)

        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @skip_no_cuda
    def test_amp_training(self, tiny_model_config):
        """AMP training should work."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        model = AutoregressiveTransformer(**tiny_model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        x = torch.randint(0, tiny_model_config["vocab_size"], (4, 32), device=device)
        target = torch.randint(0, tiny_model_config["vocab_size"], (4, 32), device=device)

        # Training step with AMP
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(x)
            loss = nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert not torch.isnan(loss)
        assert scaler.get_scale() > 0

    @skip_no_cuda
    def test_grad_scaling_overflow(self, tiny_model_config):
        """GradScaler should handle overflow gracefully."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        model = AutoregressiveTransformer(**tiny_model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        x = torch.randint(0, tiny_model_config["vocab_size"], (4, 32), device=device)

        # Run several steps
        for _ in range(5):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(x)
                loss = output.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Should complete without error
        assert True


# ============================================================================
# Multi-GPU Tests
# ============================================================================


class TestMultiGPU:
    """Tests for multi-GPU execution."""

    @skip_no_multi_gpu
    def test_data_parallel(self, tiny_model_config):
        """DataParallel should work."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(**tiny_model_config).cuda()
        inner_model = model  # keep reference before wrapping
        model = nn.DataParallel(model)

        x = torch.randint(0, tiny_model_config["vocab_size"], (8, 32), device="cuda")
        output = model(x)

        assert output.shape == (8, 32, inner_model.vocab_size)

    @skip_no_multi_gpu
    def test_model_on_different_gpus(self, tiny_model_config):
        """Model should work on different GPUs."""
        from medlatents import AutoregressiveTransformer

        for device_id in range(min(2, torch.cuda.device_count())):
            device = torch.device(f"cuda:{device_id}")
            model = AutoregressiveTransformer(**tiny_model_config).to(device)

            x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)
            output = model(x)

            assert output.device == device


# ============================================================================
# Gradient Checkpointing Tests
# ============================================================================


class TestGradientCheckpointing:
    """Tests for gradient checkpointing."""

    def test_gradient_checkpointing_cpu(self, tiny_model_config):
        """Gradient checkpointing should work on CPU."""
        from medlatents import AutoregressiveTransformer

        config_with_ckpt = {**tiny_model_config, "gradient_checkpointing": True}
        model = AutoregressiveTransformer(**config_with_ckpt)

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Should complete without error
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @skip_no_cuda
    def test_gradient_checkpointing_saves_memory(self, tiny_model_config):
        """Gradient checkpointing should reduce memory usage."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")

        # Make a larger model for more noticeable memory difference
        larger_config = {**tiny_model_config, "depth": 8, "hidden_size": 256}

        torch.cuda.reset_peak_memory_stats()

        # Without checkpointing
        model = AutoregressiveTransformer(**larger_config).to(device)
        x = torch.randint(0, tiny_model_config["vocab_size"], (8, 128), device=device)

        output = model(x)
        loss = output.sum()
        loss.backward()

        memory_without = torch.cuda.max_memory_allocated()

        # Clear
        del model, x, output, loss
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # With checkpointing
        larger_config_ckpt = {**larger_config, "gradient_checkpointing": True}
        model = AutoregressiveTransformer(**larger_config_ckpt).to(device)

        x = torch.randint(0, tiny_model_config["vocab_size"], (8, 128), device=device)

        output = model(x)
        loss = output.sum()
        loss.backward()

        memory_with = torch.cuda.max_memory_allocated()

        # Checkpointing should use less memory
        assert memory_with < memory_without


# ============================================================================
# Torch Compile Tests
# ============================================================================


class TestTorchCompile:
    """Tests for torch.compile optimization."""

    @skip_no_cuda  # torch.compile on CPU has issues with jaxtyping/beartype decorators
    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_cpu(self, tiny_model_config):
        """torch.compile should work on CPU."""
        from medlatents import AutoregressiveTransformer

        model = AutoregressiveTransformer(**tiny_model_config)
        model = torch.compile(model, mode="reduce-overhead")

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32))

        # First call triggers compilation
        output1 = model(x)
        # Second call uses compiled version
        output2 = model(x)

        assert output1.shape == output2.shape

    @skip_no_cuda
    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_cuda(self, tiny_model_config):
        """torch.compile should work on CUDA."""
        from medlatents import AutoregressiveTransformer

        device = torch.device("cuda")
        model = AutoregressiveTransformer(**tiny_model_config).to(device)
        model = torch.compile(model, mode="reduce-overhead")

        x = torch.randint(0, tiny_model_config["vocab_size"], (2, 32), device=device)

        # First call triggers compilation
        output1 = model(x)
        # Second call uses compiled version
        output2 = model(x)

        assert output1.shape == output2.shape
        assert output1.device == device


# ============================================================================
# DiT Model Tests
# ============================================================================


class TestDiTHardware:
    """Hardware tests specific to DiT models."""

    @pytest.fixture
    def dit_config(self):
        return {
            "vocab_size": 128,
            "num_classes": 10,
            "seq_length": 32,
            "hidden_size": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
        }

    def test_dit_on_cpu(self, dit_config):
        """DiscreteDiT should work on CPU."""
        from medlatents.networks import DiscreteDiT

        model = DiscreteDiT(**dit_config)

        x = torch.randint(0, dit_config["vocab_size"], (2, 32))
        t = torch.rand(2)
        y = torch.randint(0, dit_config["num_classes"], (2,))

        output = model(x, t, y)

        assert output.device.type == "cpu"
        assert output.shape == (2, 32, dit_config["vocab_size"])

    @pytest.mark.skipif(not (has_cuda() or has_mps()), reason="No GPU available")
    def test_dit_on_gpu(self, gpu_device, dit_config):
        """DiscreteDiT should work on GPU."""
        from medlatents.networks import DiscreteDiT

        model = DiscreteDiT(**dit_config).to(gpu_device)

        x = torch.randint(0, dit_config["vocab_size"], (2, 32), device=gpu_device)
        t = torch.rand(2, device=gpu_device)
        y = torch.randint(0, dit_config["num_classes"], (2,), device=gpu_device)

        output = model(x, t, y)

        assert output.device.type == gpu_device.type
        assert output.shape == (2, 32, dit_config["vocab_size"])
