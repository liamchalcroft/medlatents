"""Continuous diffusion utilities for latent generative modelling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float

from ..sampling.flow_diffusion import sample_with_cfg_diffusion


def _linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class ContinuousGaussianDiffusion:
    """Gaussian diffusion process for continuous latent spaces."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        prediction_type: str = "epsilon",
        device: torch.device | None = None,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.device = device or torch.device("cpu")

        if schedule_type == "linear":
            betas = _linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = _cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.register_schedule(betas.to(self.device))

    def register_schedule(self, betas: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=betas.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        while out.ndim < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def q_sample(
        self,
        x_start: Float[torch.Tensor, "batch ..."],
        t: torch.Tensor,
        noise: Float[torch.Tensor, "batch ..."] | None = None,
    ) -> Float[torch.Tensor, "batch ..."]:
        """Sample from q(x_t | x_0) using the reparameterization trick.

        The forward diffusion process gradually adds Gaussian noise to the data:
        q(x_t | x_0) = 𝒩(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)

        where ᾱ_t = ∏ᵢ₌₁ᵗ αᵢ is the cumulative product of noise schedule.

        This is efficiently sampled using the reparameterization trick:
        x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε, where ε ~ 𝒩(0, I)

        Args:
            x_start: Clean data x_0
            t: Timestep indices [batch]
            noise: Optional pre-sampled noise ε

        Returns:
            Noisy data x_t at timestep t
        """
        if noise is None:
            noise = self.sample_noise_like(x_start)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def _call_model(self, model, x: torch.Tensor, t: torch.Tensor, y=None, **model_kwargs):
        """Call model with standardized (x, t, y) signature."""
        return model(x=x, t=t, y=y, **model_kwargs)

    def compute_loss(
        self,
        model,
        x_start: Float[torch.Tensor, "batch ..."],
        y=None,
        noise: Float[torch.Tensor, "batch ..."] | None = None,
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Compute diffusion training loss.

        The model learns to predict one of three targets:
        - "epsilon": The noise ε that was added (original DDPM)
        - "x0": The clean data x_0 directly
        - "v": The velocity v_t = √ᾱ_t · ε - √(1 - ᾱ_t) · x_0 (progressive distillation)

        Args:
            model: Denoising network
            x_start: Clean data x_0
            y: Optional conditioning (e.g., class labels)
            noise: Optional pre-sampled noise
            model_kwargs: Additional model arguments

        Returns:
            MSE loss between model prediction and target
        """
        model_kwargs = model_kwargs or {}
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device)
        noise = noise if noise is not None else self.sample_noise_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        model_output = self._call_model(model, x_t, t, y=y, **model_kwargs)

        # Determine target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise  # Predict the noise that was added
        elif self.prediction_type == "x0":
            target = x_start  # Predict the clean data directly
        elif self.prediction_type == "v":
            # Predict the velocity (progressive distillation formulation)
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        return F.mse_loss(model_output, target)

    def predict_x0(
        self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha_cumprod
        if self.prediction_type == "x0":
            return model_output
        if self.prediction_type == "v":
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
        raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y=None,
        guidance_scale: float = 1.0,
        null_y=None,
        model_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_kwargs = model_kwargs or {}

        if guidance_scale != 1.0 and y is not None and null_y is not None:
            model_output = sample_with_cfg_diffusion(
                model,
                x_t,
                t,
                condition=y,
                guidance_scale=guidance_scale,
                null_condition=null_y,
                model_kwargs=model_kwargs,
            )
        else:
            model_output = self._call_model(model, x_t, t, y=y, **model_kwargs)

        pred_x0 = self.predict_x0(model_output, x_t, t)

        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple[int, ...],
        y=None,
        guidance_scale: float = 1.0,
        null_y=None,
        num_inference_steps: int | None = None,
        temperature: float = 1.0,
        model_kwargs: dict | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_kwargs = model_kwargs or {}
        steps = num_inference_steps or self.num_timesteps

        x = initial_noise if initial_noise is not None else torch.randn(shape, device=self.device)
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, steps, device=self.device, dtype=torch.long
        )

        for idx, timestep in enumerate(timesteps):
            t = timestep.repeat(shape[0]).to(self.device, dtype=torch.long)
            mean, variance, log_variance = self.p_sample(
                model,
                x,
                t,
                y=y,
                guidance_scale=guidance_scale,
                null_y=null_y,
                model_kwargs=model_kwargs,
            )
            if idx < steps - 1:
                noise = torch.randn_like(x) * temperature
                x = mean + torch.exp(0.5 * log_variance) * noise
            else:
                x = mean

        return x


__all__ = ["ContinuousGaussianDiffusion"]
