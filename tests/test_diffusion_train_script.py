"""Tests for diffusion training script helpers."""

import importlib.util
from pathlib import Path

import torch


def _load_train_module():
    train_path = Path(__file__).parent.parent / "scripts" / "diffusion" / "train.py"
    spec = importlib.util.spec_from_file_location("diffusion_train", train_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Could not load diffusion train module")
    spec.loader.exec_module(module)
    return module


def test_flatten_latents_2d():
    module = _load_train_module()
    flatten_latents = module.flatten_latents

    batch = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    out = flatten_latents(batch)

    expected = batch.permute(0, 2, 3, 1).reshape(2, 4 * 5, 3)
    assert out.shape == (2, 20, 3)
    assert torch.equal(out, expected)


def test_flatten_latents_3d():
    module = _load_train_module()
    flatten_latents = module.flatten_latents

    batch = torch.arange(2 * 3 * 2 * 4 * 5, dtype=torch.float32).reshape(2, 3, 2, 4, 5)
    out = flatten_latents(batch)

    expected = batch.permute(0, 2, 3, 4, 1).reshape(2, 2 * 4 * 5, 3)
    assert out.shape == (2, 40, 3)
    assert torch.equal(out, expected)


def test_flatten_latents_passthrough():
    module = _load_train_module()
    flatten_latents = module.flatten_latents

    batch = torch.randn(2, 7, 3)
    out = flatten_latents(batch)

    assert out.shape == batch.shape
    assert torch.equal(out, batch)
