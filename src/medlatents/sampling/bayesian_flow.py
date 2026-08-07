"""Sampling utilities for the integrated Bayesian Flow Transformer."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from ..bayesian_flow import BayesianFlowTransformer


def _ensure_positive_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value)}")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def _ensure_finite_positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def _ensure_probability(value: float, name: str, *, strict: bool = True) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    lo, hi = (0.0, 1.0) if strict else (-1e-6, 1.0 + 1e-6)
    if not (lo < value <= hi):
        raise ValueError(f"{name} must be in (0, 1], got {value}")


@torch.no_grad()
def sample_bayesian_flow(
    model: BayesianFlowTransformer,
    batch_size: int,
    seq_len: int,
    num_steps: int | None = None,
    temperature: float = 1.0,
    temperature_schedule: Callable[[int, int], float] | None = None,
    condition: torch.Tensor | None = None,
    guidance_scale: float | None = None,
    null_condition: torch.Tensor | None = None,
    **model_kwargs,
) -> Int[torch.Tensor, "batch seq"]:
    _ensure_positive_int(batch_size, "batch_size")
    _ensure_positive_int(seq_len, "seq_len")
    if num_steps is not None:
        _ensure_positive_int(num_steps, "num_steps")
    _ensure_finite_positive(temperature, "temperature")
    if guidance_scale is not None:
        if not isinstance(guidance_scale, (int, float)) or not math.isfinite(guidance_scale):
            raise ValueError(f"guidance_scale must be finite, got {guidance_scale}")

    if temperature_schedule is not None:
        return _sample_with_temperature_schedule(
            model,
            (batch_size, seq_len),
            num_steps or model.num_steps,
            temperature_schedule,
            condition=condition,
            guidance_scale=guidance_scale,
            null_condition=null_condition,
            **model_kwargs,
        )

    return model.sample(
        (batch_size, seq_len),
        num_steps=num_steps,
        temperature=temperature,
        condition=condition,
        guidance_scale=guidance_scale,
        null_condition=null_condition,
        **model_kwargs,
    )


@torch.no_grad()
def _sample_with_temperature_schedule(
    model: BayesianFlowTransformer,
    shape: tuple[int, int],
    num_steps: int,
    temperature_schedule: Callable[[int, int], float],
    *,
    condition: torch.Tensor | None = None,
    guidance_scale: float | None = None,
    null_condition: torch.Tensor | None = None,
    **model_kwargs,
) -> Int[torch.Tensor, "batch seq"]:
    batch_size, seq_len = shape
    device = model.device

    params = model.get_prior_params(batch_size, seq_len)

    for i in range(num_steps):
        temp = temperature_schedule(i, num_steps)
        alpha = model.get_accuracy(
            torch.full((batch_size,), float(i), device=device),
            continuous_time=False,
        )

        probs = F.softmax(params / temp, dim=-1)
        if hasattr(model, "_sanitize_probs"):
            probs = model._sanitize_probs(probs)
        x_current = torch.multinomial(
            probs.view(-1, model.num_classes),
            num_samples=1,
        ).view(batch_size, seq_len)

        t = torch.full((batch_size,), i / num_steps, device=device)
        y_receiver = model.sample_receiver_distribution(
            x_current,
            t,
            alpha,
            temperature=temp,
            condition=condition,
            guidance_scale=guidance_scale,
            null_condition=null_condition,
            **model_kwargs,
        )

        params = model.update_params_bayesian(params, y_receiver)

    final_temp = temperature_schedule(num_steps - 1, num_steps)
    final_probs = F.softmax(params / final_temp, dim=-1)
    if hasattr(model, "_sanitize_probs"):
        final_probs = model._sanitize_probs(final_probs)
    samples = torch.multinomial(
        final_probs.view(-1, model.num_classes),
        num_samples=1,
    ).view(batch_size, seq_len)

    return samples


def get_confidence(
    params: Float[torch.Tensor, "batch seq classes"],
) -> Float[torch.Tensor, "batch seq"]:
    probs = F.softmax(params, dim=-1)
    return probs.max(dim=-1).values


@torch.no_grad()
def sample_with_early_stopping(
    model: BayesianFlowTransformer,
    batch_size: int,
    seq_len: int,
    confidence_threshold: float = 0.95,
    min_steps: int = 10,
    max_steps: int | None = None,
    temperature: float = 1.0,
    **model_kwargs,
) -> tuple[Int[torch.Tensor, "batch seq"], int]:
    _ensure_positive_int(batch_size, "batch_size")
    _ensure_positive_int(seq_len, "seq_len")
    _ensure_probability(confidence_threshold, "confidence_threshold")
    _ensure_positive_int(min_steps, "min_steps", allow_zero=True)
    if max_steps is not None:
        _ensure_positive_int(max_steps, "max_steps")
    _ensure_finite_positive(temperature, "temperature")

    steps = max_steps or model.num_steps
    device = model.device

    params = model.get_prior_params(batch_size, seq_len)
    steps_taken = 0

    for i in range(steps):
        steps_taken = i + 1
        alpha = model.get_accuracy(
            torch.full((batch_size,), float(i), device=device),
            continuous_time=False,
        )

        probs = F.softmax(params / temperature, dim=-1)
        if hasattr(model, "_sanitize_probs"):
            probs = model._sanitize_probs(probs)
        x_current = torch.multinomial(
            probs.view(-1, model.num_classes),
            num_samples=1,
        ).view(batch_size, seq_len)

        t = torch.full((batch_size,), i / steps, device=device)
        y_receiver = model.sample_receiver_distribution(
            x_current,
            t,
            alpha,
            temperature=temperature,
            **model_kwargs,
        )

        params = model.update_params_bayesian(params, y_receiver)

        if i >= min_steps:
            confidence = get_confidence(params)
            if confidence.mean().item() >= confidence_threshold:
                break

    final_probs = F.softmax(params / temperature, dim=-1)
    if hasattr(model, "_sanitize_probs"):
        final_probs = model._sanitize_probs(final_probs)
    samples = torch.multinomial(
        final_probs.view(-1, model.num_classes),
        num_samples=1,
    ).view(batch_size, seq_len)

    return samples, steps_taken


class BayesianFlowSampler:
    """Convenience wrapper for Bayesian Flow sampling strategies."""

    def __init__(
        self,
        model: BayesianFlowTransformer,
        default_temperature: float = 1.0,
        default_num_steps: int | None = None,
    ):
        _ensure_finite_positive(default_temperature, "default_temperature")
        if default_num_steps is not None:
            _ensure_positive_int(default_num_steps, "default_num_steps")
        self.model = model
        self.default_temperature = default_temperature
        self.default_num_steps = default_num_steps or model.num_steps

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int | None = None,
        temperature: float | None = None,
        **model_kwargs,
    ) -> Int[torch.Tensor, "batch seq"]:
        return sample_bayesian_flow(
            self.model,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps or self.default_num_steps,
            temperature=temperature or self.default_temperature,
            **model_kwargs,
        )

    @torch.no_grad()
    def sample_with_cfg(
        self,
        batch_size: int,
        seq_len: int,
        condition: torch.Tensor,
        guidance_scale: float = 1.5,
        null_condition: torch.Tensor | None = None,
        num_steps: int | None = None,
        temperature: float | None = None,
        **model_kwargs,
    ) -> Int[torch.Tensor, "batch seq"]:
        return sample_bayesian_flow(
            self.model,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps or self.default_num_steps,
            temperature=temperature or self.default_temperature,
            condition=condition,
            guidance_scale=guidance_scale,
            null_condition=null_condition,
            **model_kwargs,
        )

    @torch.no_grad()
    def sample_with_schedule(
        self,
        batch_size: int,
        seq_len: int,
        temperature_schedule: Callable[[int, int], float],
        **model_kwargs,
    ) -> Int[torch.Tensor, "batch seq"]:
        return sample_bayesian_flow(
            self.model,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=self.default_num_steps,
            temperature_schedule=temperature_schedule,
            **model_kwargs,
        )

    @torch.no_grad()
    def sample_with_early_stopping(
        self,
        batch_size: int,
        seq_len: int,
        confidence_threshold: float = 0.95,
        min_steps: int = 10,
        max_steps: int | None = None,
        temperature: float | None = None,
        **model_kwargs,
    ) -> tuple[Int[torch.Tensor, "batch seq"], int]:
        return sample_with_early_stopping(
            self.model,
            batch_size=batch_size,
            seq_len=seq_len,
            confidence_threshold=confidence_threshold,
            min_steps=min_steps,
            max_steps=max_steps or self.default_num_steps,
            temperature=temperature or self.default_temperature,
            **model_kwargs,
        )


__all__ = [
    "sample_bayesian_flow",
    "sample_with_early_stopping",
    "BayesianFlowSampler",
    "get_confidence",
]
