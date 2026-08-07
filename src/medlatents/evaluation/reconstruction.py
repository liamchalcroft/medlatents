"""Reconstruction quality metrics (PSNR/SSIM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_image_batch(x: torch.Tensor) -> torch.Tensor:
    """Normalize input to [batch, channel, height, width] format."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(
        f"Expected tensor with 2D, 3D, or 4D shape, got {x.ndim}D with shape {tuple(x.shape)}"
    )


def _validate_inputs(
    reference: torch.Tensor, prediction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.shape != prediction.shape:
        raise ValueError(
            f"reference and prediction must have identical shapes, got "
            f"{tuple(reference.shape)} vs {tuple(prediction.shape)}"
        )
    ref = _as_image_batch(reference).float()
    pred = _as_image_batch(prediction).float()
    if ref.shape != pred.shape:
        raise ValueError(
            f"reference and prediction must have identical image shapes, got "
            f"{tuple(ref.shape)} vs {tuple(pred.shape)}"
        )
    return ref, pred


def _infer_data_range(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    min_value = torch.minimum(reference.min(), prediction.min())
    max_value = torch.maximum(reference.max(), prediction.max())
    data_range = (max_value - min_value).item()
    return data_range if data_range > 0 else 1.0


def calculate_psnr(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    data_range: float | None = None,
) -> float:
    """Compute peak signal-to-noise ratio in dB."""
    ref, pred = _validate_inputs(reference, prediction)

    mse = F.mse_loss(pred, ref, reduction="mean").item()
    if mse == 0.0:
        return float("inf")

    if data_range is None:
        data_range = _infer_data_range(ref, pred)
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    psnr = 10.0 * torch.log10(torch.tensor((data_range * data_range) / mse)).item()
    return float(psnr)


def _gaussian(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return gauss / gauss.sum()


def _create_window(
    window_size: int,
    channel: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a normalized 2D Gaussian window for SSIM."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if channel <= 0:
        raise ValueError(f"channel must be positive, got {channel}")

    gauss_1d = _gaussian(window_size)
    window_2d = torch.outer(gauss_1d, gauss_1d)
    window_2d = window_2d / window_2d.sum()
    window = window_2d.view(1, 1, window_size, window_size)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window.to(dtype=dtype, device=device)


def calculate_ssim(
    reference: torch.Tensor,
    prediction: torch.Tensor,
    *,
    window_size: int = 11,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute structural similarity index (SSIM)."""
    ref, pred = _validate_inputs(reference, prediction)

    if ref.shape[-1] < window_size or ref.shape[-2] < window_size:
        window_size = max(1, min(ref.shape[-1], ref.shape[-2], window_size))
    if window_size % 2 == 0 and window_size > 1:
        window_size -= 1

    channels = ref.shape[1]
    window = _create_window(window_size, channels, dtype=ref.dtype, device=ref.device)
    padding = window_size // 2

    mu_ref = F.conv2d(ref, window, padding=padding, groups=channels)
    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)

    mu_ref_sq = mu_ref.pow(2)
    mu_pred_sq = mu_pred.pow(2)
    mu_ref_pred = mu_ref * mu_pred

    sigma_ref_sq = F.conv2d(ref * ref, window, padding=padding, groups=channels) - mu_ref_sq
    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_ref_pred = F.conv2d(ref * pred, window, padding=padding, groups=channels) - mu_ref_pred

    if data_range is None:
        data_range = _infer_data_range(ref, pred)
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_ref_pred + c1) * (2.0 * sigma_ref_pred + c2)
    denominator = (mu_ref_sq + mu_pred_sq + c1) * (sigma_ref_sq + sigma_pred_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)

    ssim = ssim_map.mean().item()
    return float(max(0.0, min(1.0, ssim)))


__all__ = ["calculate_psnr", "calculate_ssim", "_create_window"]
