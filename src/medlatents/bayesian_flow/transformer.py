"""Integrated Bayesian Flow Transformer."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from ..configs import create_model_variants
from ..networks.diffusion_transformer import DiscreteDiT
from ..utils import SpecialTokenIds, resolve_special_tokens


class BayesianFlowTransformer(DiscreteDiT):
    """Transformer backbone with built-in Bayesian Flow process."""

    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.0,
        num_classes: int = 0,
        rope_theta: float = 10000.0,
        allow_dynamic_seq_length: bool = True,
        chunk_size: int | None = None,
        masked: bool = False,
        gradient_checkpointing: bool = True,
        num_steps: int = 1000,
        beta: float = 1.0,
        sigma_schedule: str | None = None,
        special_tokens: SpecialTokenIds | None = None,
        add_special_tokens: bool = True,
    ) -> None:
        extended_vocab, resolved_special = resolve_special_tokens(
            vocab_size,
            special_tokens=special_tokens,
            add_special_tokens=add_special_tokens,
        )

        super().__init__(
            seq_length=seq_length,
            vocab_size=extended_vocab,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            rope_theta=rope_theta,
            allow_dynamic_seq_length=allow_dynamic_seq_length,
            chunk_size=chunk_size,
            masked=masked,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.base_vocab_size = vocab_size
        self.special_tokens = resolved_special
        self.num_steps = num_steps
        self.beta = beta
        self.sigma_schedule = sigma_schedule or "cosine"
        self.vocab_size = self.final_layer[-1].out_features
        self.num_classes = self.vocab_size

        forbidden: set[int] = set()
        if self.special_tokens is not None:
            forbidden.update({self.special_tokens.pad, self.special_tokens.mask})
        self._forbidden_tokens = forbidden

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Flow utilities

    def get_accuracy(
        self,
        t: Float[torch.Tensor, "batch"],
        continuous_time: bool = True,
    ) -> Float[torch.Tensor, "batch"]:
        if continuous_time:
            return (self.beta**2) * t
        i = t.long() + 1
        return (self.beta / self.num_steps) ** 2 * (2 * i - 1)

    def get_prior_params(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Float[torch.Tensor, "batch seq classes"]:
        return torch.zeros(
            batch_size,
            seq_len,
            self.num_classes,
            device=self.device,
        )

    def sample_sender_distribution(
        self,
        x: Int[torch.Tensor, "batch seq"],
        alpha: Float[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch seq classes"]:
        batch_size, seq_len = x.shape
        K = self.num_classes

        x_onehot = F.one_hot(x, num_classes=K).float()
        alpha = alpha.view(-1, 1, 1)
        mean = alpha * (K * x_onehot - 1)
        std = torch.sqrt(alpha * K)
        noise = torch.randn_like(mean)
        return mean + std * noise

    def update_params_bayesian(
        self,
        params: Float[torch.Tensor, "batch seq classes"],
        y: Float[torch.Tensor, "batch seq classes"],
    ) -> Float[torch.Tensor, "batch seq classes"]:
        # Work in log-space to avoid exp() overflow at large vocabs/many steps:
        # log(new_probs) = log_softmax(params) + y - log_sum_exp(log_softmax(params) + y)
        log_probs = F.log_softmax(params, dim=-1)
        log_new_unnorm = log_probs + y
        return log_new_unnorm - torch.logsumexp(log_new_unnorm, dim=-1, keepdim=True)

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._forbidden_tokens:
            return logits
        logits = logits.clone()
        indices = torch.tensor(
            sorted(self._forbidden_tokens), device=logits.device, dtype=torch.long
        )
        logits.index_fill_(-1, indices, -float("inf"))
        return logits

    def _sanitize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if not self._forbidden_tokens:
            return probs
        indices = torch.tensor(
            sorted(self._forbidden_tokens), device=probs.device, dtype=torch.long
        )
        probs = probs.clone()
        probs.index_fill_(-1, indices, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs

    def _compute_logits(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        temperature: float,
        condition: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        null_condition: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        if guidance_scale is not None and condition is not None:
            logits_cond = self(x, t, y=condition, **model_kwargs)
            logits_uncond = self(x, t, y=null_condition, **model_kwargs)
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        else:
            logits = self(x, t, y=condition, **model_kwargs)

        logits = logits / temperature
        return self._mask_logits(logits)

    def sample_receiver_distribution(
        self,
        x_current: Int[torch.Tensor, "batch seq"],
        t: Float[torch.Tensor, "batch"],
        alpha: Float[torch.Tensor, "batch"],
        *,
        temperature: float = 1.0,
        condition: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        null_condition: torch.Tensor | None = None,
        **model_kwargs,
    ) -> Float[torch.Tensor, "batch seq classes"]:
        logits = self._compute_logits(
            x_current,
            t,
            temperature=temperature,
            condition=condition,
            guidance_scale=guidance_scale,
            null_condition=null_condition,
            **model_kwargs,
        )

        pred_probs = F.softmax(logits, dim=-1)
        pred_probs = self._sanitize_probs(pred_probs)

        alpha_exp = alpha.view(-1, 1, 1)
        mean = alpha_exp * (self.num_classes * pred_probs - 1)
        std = torch.sqrt(alpha_exp * self.num_classes)
        noise = torch.randn_like(mean)
        return mean + std * noise

    def discrete_time_loss(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        *,
        temperature: float = 1.0,
        **model_kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = x_0.shape
        device = x_0.device

        steps = torch.randint(0, self.num_steps, (batch_size,), device=device)
        t = steps.float() / self.num_steps
        alpha = self.get_accuracy(steps.float(), continuous_time=False)

        y_sender = self.sample_sender_distribution(x_0, alpha)
        sender_probs = F.softmax(y_sender, dim=-1)
        sender_probs = self._sanitize_probs(sender_probs)
        x_noisy = torch.multinomial(
            sender_probs.view(-1, self.num_classes),
            num_samples=1,
        ).view(batch_size, seq_len)

        y_receiver = self.sample_receiver_distribution(
            x_noisy,
            t,
            alpha,
            temperature=temperature,
            **model_kwargs,
        )

        variance = alpha.view(-1, 1, 1) * self.num_classes
        kl_div = (y_sender - y_receiver).pow(2).sum(dim=-1) / (2 * variance)
        loss = self.num_steps * kl_div.mean()
        return loss

    def continuous_time_loss(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        *,
        temperature: float = 1.0,
        **model_kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len = x_0.shape
        device = x_0.device

        t = torch.rand(batch_size, device=device)
        alpha = self.get_accuracy(t, continuous_time=True)

        logits = self._compute_logits(
            x_0,
            t,
            temperature=temperature,
            **model_kwargs,
        )
        pred_probs = F.softmax(logits, dim=-1)
        pred_probs = self._sanitize_probs(pred_probs)

        kl_div = -torch.log(pred_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1) + 1e-10)
        weight = alpha.view(-1, 1)
        return (weight * kl_div).mean()

    def compute_loss(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        *,
        loss_type: str = "discrete",
        temperature: float = 1.0,
        **model_kwargs,
    ) -> torch.Tensor:
        if loss_type == "discrete":
            return self.discrete_time_loss(x_0, temperature=temperature, **model_kwargs)
        if loss_type == "continuous":
            return self.continuous_time_loss(x_0, temperature=temperature, **model_kwargs)
        raise ValueError(f"Unknown loss_type: {loss_type}")

    def sample(
        self,
        shape: tuple[int, int],
        *,
        num_steps: int | None = None,
        temperature: float = 1.0,
        temperature_schedule: Callable[[int, int], float] | None = None,
        condition: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        null_condition: torch.Tensor | None = None,
        **model_kwargs,
    ) -> Int[torch.Tensor, "batch seq"]:
        batch_size, seq_len = shape
        device = self.device
        steps = num_steps or self.num_steps

        params = self.get_prior_params(batch_size, seq_len)

        for i in range(steps):
            temp = (
                temperature_schedule(i, steps) if temperature_schedule is not None else temperature
            )

            alpha = self.get_accuracy(
                torch.full((batch_size,), float(i), device=device),
                continuous_time=False,
            )

            probs = F.softmax(params / temp, dim=-1)
            probs = self._sanitize_probs(probs)
            x_current = torch.multinomial(
                probs.view(-1, self.num_classes),
                num_samples=1,
            ).view(batch_size, seq_len)

            t = torch.full((batch_size,), i / steps, device=device)

            y_receiver = self.sample_receiver_distribution(
                x_current,
                t,
                alpha,
                temperature=temp,
                condition=condition,
                guidance_scale=guidance_scale,
                null_condition=null_condition,
                **model_kwargs,
            )

            params = self.update_params_bayesian(params, y_receiver)

        final_temp = (
            temperature_schedule(steps - 1, steps)
            if temperature_schedule is not None
            else temperature
        )
        final_probs = F.softmax(params / final_temp, dim=-1)
        final_probs = self._sanitize_probs(final_probs)
        samples = torch.multinomial(
            final_probs.view(-1, self.num_classes),
            num_samples=1,
        ).view(batch_size, seq_len)

        return samples


#: Preset BayesianFlowTransformer model configurations (nano/small/base/large/xl).
BFN_models = create_model_variants(BayesianFlowTransformer, "BFN")


__all__ = [
    "BayesianFlowTransformer",
    "BFN_models",
]
