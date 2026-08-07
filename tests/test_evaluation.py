"""Tests for evaluation metrics."""

import torch


class TestPSNR:
    """Tests for PSNR calculation."""

    def test_psnr_identical_images(self):
        """Identical images should have infinite PSNR."""
        from medlatents.evaluation import calculate_psnr

        img = torch.rand(2, 1, 32, 32)
        psnr = calculate_psnr(img, img)
        assert psnr == float("inf")

    def test_psnr_increases_with_similarity(self):
        """More similar images should have higher PSNR."""
        from medlatents.evaluation import calculate_psnr

        real = torch.rand(2, 1, 32, 32)
        small_noise = real + 0.01 * torch.randn_like(real)
        large_noise = real + 0.1 * torch.randn_like(real)

        psnr_small = calculate_psnr(real, small_noise)
        psnr_large = calculate_psnr(real, large_noise)

        assert psnr_small > psnr_large

    def test_psnr_reasonable_range(self):
        """PSNR should be in reasonable range for noisy images."""
        from medlatents.evaluation import calculate_psnr

        real = torch.rand(2, 1, 32, 32)
        noisy = real + 0.05 * torch.randn_like(real)

        psnr = calculate_psnr(real, noisy.clamp(0, 1))

        # For 5% noise, PSNR typically 20-40 dB
        assert 15 < psnr < 50


class TestSSIM:
    """Tests for SSIM calculation."""

    def test_ssim_identical_images(self):
        """Identical images should have SSIM close to 1."""
        from medlatents.evaluation import calculate_ssim

        img = torch.rand(2, 1, 32, 32)
        ssim = calculate_ssim(img, img)

        assert abs(ssim - 1.0) < 0.01

    def test_ssim_increases_with_similarity(self):
        """More similar images should have higher SSIM."""
        from medlatents.evaluation import calculate_ssim

        real = torch.rand(2, 1, 32, 32)
        small_noise = real + 0.01 * torch.randn_like(real)
        large_noise = real + 0.2 * torch.randn_like(real)

        ssim_small = calculate_ssim(real, small_noise.clamp(0, 1))
        ssim_large = calculate_ssim(real, large_noise.clamp(0, 1))

        assert ssim_small > ssim_large

    def test_ssim_range(self):
        """SSIM should be in [0, 1] range."""
        from medlatents.evaluation import calculate_ssim

        real = torch.rand(2, 1, 32, 32)
        recon = torch.rand(2, 1, 32, 32)

        ssim = calculate_ssim(real, recon)

        assert 0 <= ssim <= 1

    def test_ssim_multichannel(self):
        """SSIM should work with multiple channels."""
        from medlatents.evaluation import calculate_ssim

        real = torch.rand(2, 3, 32, 32)
        recon = real + 0.05 * torch.randn_like(real)

        ssim = calculate_ssim(real, recon.clamp(0, 1))

        assert 0 < ssim < 1


class TestWindowCreation:
    """Tests for SSIM window helper."""

    def test_window_shape(self):
        """Window should have correct shape."""
        from medlatents.evaluation.reconstruction import _create_window

        window = _create_window(window_size=11, channel=3)

        assert window.shape == (3, 1, 11, 11)

    def test_window_normalized(self):
        """Window should be normalized (sum to 1 per channel)."""
        from medlatents.evaluation.reconstruction import _create_window

        window = _create_window(window_size=11, channel=1)

        # Each channel's window should sum to approximately 1
        assert abs(window[0, 0].sum().item() - 1.0) < 0.01
