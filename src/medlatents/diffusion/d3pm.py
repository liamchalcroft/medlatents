"""D3PM: Discrete Denoising Diffusion Probabilistic Models.

Based on "Structured Denoising Diffusion Models in Discrete State-Spaces"
https://arxiv.org/abs/2107.03006
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from ..utils.numerics import EPS_PROB
from .schedules import compute_transition_matrices, get_beta_schedule


def _get_torch_compile() -> Callable | None:
    """Get torch.compile if available (PyTorch 2.0+), else return None."""
    if hasattr(torch, "compile"):
        return torch.compile
    return None


class D3PM:
    """
    Discrete Denoising Diffusion Probabilistic Model.

    Supports multiple transition types:
    - absorbing: Transitions to a mask state (like masked language modeling)
    - uniform: Uniform transitions to all classes (original D3PM)
    - gaussian: Discretized Gaussian transitions (smooth blurring)
    """

    def __init__(
        self,
        num_classes: int,
        num_timesteps: int = 1000,
        schedule_type: Literal["linear", "cosine", "quadratic", "sigmoid"] = "cosine",
        transition_type: Literal["absorbing", "uniform", "gaussian"] = "absorbing",
        hybrid_loss_coeff: float = 0.001,
        device: torch.device | None = None,
    ):
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.transition_type = transition_type
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.device = device or torch.device("cpu")

        betas = get_beta_schedule(schedule_type, num_timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # For absorbing diffusion, track the mask token
        if transition_type == "absorbing":
            self.mask_token = num_classes
            self.effective_num_classes = num_classes + 1
        else:
            self.mask_token = None
            self.effective_num_classes = num_classes

        # Use matrix-free path for absorbing transitions to avoid O(K^2) memory.
        # Absorbing transitions have closed-form structure that only needs the
        # scalar beta/alpha schedules, not full (K+1)x(K+1) matrices.
        self._matrix_free = transition_type == "absorbing"

        if self._matrix_free:
            alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
            for name, t in [
                ("betas", betas),
                ("alphas", alphas),
                ("alphas_cumprod", alphas_cumprod),
                ("alphas_cumprod_prev", alphas_cumprod_prev),
            ]:
                setattr(self, name, t.to(device, non_blocking=True))
            self.Q_t = self.Q_bar_t = self.Q_bar_t_minus_1 = None
        else:
            Q_t, Q_bar_t, Q_bar_t_minus_1 = compute_transition_matrices(
                betas, num_classes, transition_type
            )
            schedule_tensors = [betas, alphas, alphas_cumprod, Q_t, Q_bar_t, Q_bar_t_minus_1]
            moved = [t.to(device, non_blocking=True) for t in schedule_tensors]
            (
                self.betas,
                self.alphas,
                self.alphas_cumprod,
                self.Q_t,
                self.Q_bar_t,
                self.Q_bar_t_minus_1,
            ) = moved

        # Compiled sampling step (set via compile_sampling())
        self._compiled_p_sample_step: Callable | None = None

    def compile_sampling(
        self,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> bool:
        """
        Compile the sampling step using torch.compile for faster inference.

        Requires PyTorch 2.0+. On older versions, this is a no-op.

        Args:
            mode: Compilation mode. Options:
                - "default": Good balance of compile time and runtime
                - "reduce-overhead": Minimize overhead, best for small models
                - "max-autotune": Maximum optimization, longer compile time
            fullgraph: If True, require the entire function to be captured as a single graph
            dynamic: If True, enable dynamic shape support (recommended for variable batch/seq)

        Returns:
            True if compilation was successful, False if torch.compile unavailable
        """
        torch_compile = _get_torch_compile()
        if torch_compile is None:
            return False

        # Create a compiled version of the inner sampling computation
        @torch_compile(mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        def _compiled_step(
            logits: torch.Tensor,
            x_t: torch.Tensor,
            Q_t: torch.Tensor,
            Q_bar_t_minus_1: torch.Tensor,
            temperature: float,
            effective_num_classes: int,
        ) -> torch.Tensor:
            """Compiled inner loop of p_sample - the heavy computation."""
            batch_size, seq_len = x_t.shape

            # Predict clean tokens distribution
            logits = logits / temperature
            p_x0 = F.softmax(logits, dim=-1)  # [B, L, C]

            # Vectorized marginalization over all x_0 values
            Q_t_expanded = Q_t.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, C, C]
            x_t_indices = x_t.view(batch_size, seq_len, 1, 1).expand(
                -1, -1, effective_num_classes, 1
            )
            q_t_given_tm1 = torch.gather(Q_t_expanded, dim=3, index=x_t_indices).squeeze(-1)

            # Compute unnormalized posterior
            unnorm_posterior = q_t_given_tm1.unsqueeze(2) * Q_bar_t_minus_1.unsqueeze(1)

            # Normalize over x_{t-1} for each x_0
            Z = unnorm_posterior.sum(dim=-1, keepdim=True).clamp_min(EPS_PROB)
            q_posterior_all = unnorm_posterior / Z

            # Marginalize
            p_tm1 = torch.einsum("bli,blij->blj", p_x0, q_posterior_all)

            # Apply temperature scaling if needed
            if temperature != 1.0:
                p_tm1 = p_tm1.clamp_min(EPS_PROB).pow(1.0 / temperature)
                p_tm1 = p_tm1 / p_tm1.sum(dim=-1, keepdim=True).clamp_min(EPS_PROB)

            # Sample from the marginalized distribution
            p_tm1_flat = p_tm1.reshape(-1, effective_num_classes)
            x_tm1 = torch.multinomial(p_tm1_flat.clamp_min(EPS_PROB), num_samples=1).squeeze(-1)
            return x_tm1.view(batch_size, seq_len)

        self._compiled_p_sample_step = _compiled_step
        return True

    def _validate_model_vocab(self, model: nn.Module) -> None:
        model_vocab = getattr(model, "vocab_size", None)
        if model_vocab is None:
            return
        if model_vocab != self.effective_num_classes:
            if self.transition_type == "absorbing":
                hint = (
                    "For absorbing transitions, set the model output vocab to "
                    "num_classes + 1 (mask token). For DiscreteDiT, use masked=True "
                    "with base vocab_size."
                )
            else:
                hint = "Set the model output vocab to match num_classes."
            raise ValueError(
                "D3PM model vocab mismatch: "
                f"model.vocab_size={model_vocab}, "
                f"effective_num_classes={self.effective_num_classes}. "
                f"{hint}"
            )

    def q_sample(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        t: Int[torch.Tensor, "batch"],
        noise: torch.Tensor | None = None,
    ) -> Int[torch.Tensor, "batch seq"]:
        """
        Sample from q(x_t | x_0) - the forward diffusion process.

        Args:
            x_0: Clean data [batch, seq_len]
            t: Timesteps [batch]
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy data at timestep t
        """
        if self._matrix_free:
            return self._q_sample_absorbing(x_0, t)

        batch_size, seq_len = x_0.shape

        # Convert to one-hot
        x_0_onehot = F.one_hot(x_0, num_classes=self.effective_num_classes).float()  # [B, L, C]

        # Get Q_bar_t for each sample in batch
        Q_bar = self.Q_bar_t[t]  # [B, C, C]

        # Apply transition: x_t_onehot = x_0_onehot @ Q_bar^T
        x_t_probs = torch.einsum("blc,bcd->bld", x_0_onehot, Q_bar)  # [B, L, C]

        # Sample from categorical distribution
        x_t_probs = x_t_probs.reshape(-1, self.effective_num_classes)  # [B*L, C]
        x_t = torch.multinomial(x_t_probs.clamp_min(EPS_PROB), num_samples=1).squeeze(-1)  # [B*L]
        x_t = x_t.reshape(batch_size, seq_len)  # [B, L]

        return x_t

    def _q_sample_absorbing(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        t: Int[torch.Tensor, "batch"],
    ) -> Int[torch.Tensor, "batch seq"]:
        """Forward diffusion for absorbing transitions (matrix-free).

        Each token independently stays with prob alpha_bar_t or absorbs to
        mask with prob 1 - alpha_bar_t.
        """
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)  # [B, 1]
        keep = torch.rand_like(x_0.float()) < alpha_bar_t
        return torch.where(keep, x_0, self.mask_token)

    def q_posterior(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        x_t: Int[torch.Tensor, "batch seq"],
        t: Int[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch seq num_classes"]:
        """Compute the posterior distribution q(x_{t-1} | x_t, x_0).

        This computes the true posterior for the reverse process, which is used
        for training. The posterior is computed using Bayes' rule:

        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)

        Where:
        - q(x_t | x_{t-1}) comes from the transition matrix Q_t
        - q(x_{t-1} | x_0) comes from the cumulative transition Q̄_{t-1}

        Args:
            x_0: Clean data [batch, seq_len]
            x_t: Noisy data at timestep t [batch, seq_len]
            t: Timesteps [batch]

        Returns:
            Normalized posterior probabilities [batch, seq_len, num_classes]
        """
        if self._matrix_free:
            return self._q_posterior_absorbing(x_0, x_t, t)

        batch_size, seq_len = x_0.shape

        # Convert discrete tokens to one-hot vectors for matrix operations
        x_0_onehot = F.one_hot(x_0, num_classes=self.effective_num_classes).float()  # [B, L, C]

        # Get transition matrices for this timestep
        Q_t = self.Q_t[t]  # Single-step transitions [B, C, C]
        Q_bar_t_minus_1 = self.Q_bar_t_minus_1[t]  # Cumulative transitions [B, C, C]

        # Step 1: Compute q(x_{t-1} | x_0) for all possible x_{t-1}
        # This gives us the prior distribution over x_{t-1} given the clean data
        q_tm1_given_0 = torch.einsum("blc,bcd->bld", x_0_onehot, Q_bar_t_minus_1)  # [B, L, C]

        # Step 2: Compute q(x_t | x_{t-1}) for all possible x_{t-1}
        # We need to extract the column corresponding to the observed x_t
        # Q_t[b, i, j] = P(x_t=j | x_{t-1}=i), so we want Q_t[b, :, x_t[b,l]]
        Q_t_expanded = Q_t.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, C, C]
        x_t_indices = x_t.view(batch_size, seq_len, 1, 1).expand(
            -1, -1, self.effective_num_classes, 1
        )
        q_t_given_tm1 = torch.gather(Q_t_expanded, dim=3, index=x_t_indices).squeeze(
            -1
        )  # [B, L, C]

        # Step 3: Apply Bayes' rule (multiply and normalize)
        # q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)
        posterior = q_t_given_tm1 * q_tm1_given_0
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(EPS_PROB)

        return posterior

    def _q_posterior_absorbing(
        self,
        x_0: Int[torch.Tensor, "batch seq"],
        x_t: Int[torch.Tensor, "batch seq"],
        t: Int[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch seq num_classes"]:
        """Posterior q(x_{t-1} | x_t, x_0) for absorbing transitions (matrix-free).

        For absorbing diffusion the posterior has closed form:
        - If x_t != MASK: x_{t-1} = x_t with certainty (token survived).
        - If x_t == MASK: x_{t-1} = x_0 with prob ∝ β_t·ᾱ_{t-1},
                          x_{t-1} = MASK with prob ∝ 1 - ᾱ_{t-1}.
        """
        B, L = x_0.shape
        beta_t = self.betas[t][:, None]  # [B, 1]
        alpha_bar_tm1 = self.alphas_cumprod_prev[t][:, None]  # [B, 1]

        is_masked = (x_t == self.mask_token).float()  # [B, L]

        x_t_oh = F.one_hot(x_t, self.effective_num_classes).float()  # [B, L, C]
        x_0_oh = F.one_hot(x_0, self.effective_num_classes).float()  # [B, L, C]

        # Posterior for masked positions: support on {x_0, MASK}
        p_unmask = beta_t * alpha_bar_tm1  # [B, 1]
        p_stay = 1 - alpha_bar_tm1  # [B, 1]
        Z = (p_unmask + p_stay).clamp_min(EPS_PROB)

        posterior_masked = (p_unmask / Z).unsqueeze(-1) * x_0_oh  # [B, L, C]
        posterior_masked[:, :, self.mask_token] += (p_stay / Z).expand(B, L)

        # Combine: unmasked → deterministic x_t, masked → posterior_masked
        posterior = (1 - is_masked.unsqueeze(-1)) * x_t_oh + is_masked.unsqueeze(
            -1
        ) * posterior_masked
        return posterior

    def p_sample(
        self,
        model: nn.Module,
        x_t: Int[torch.Tensor, "batch seq"],
        t: Int[torch.Tensor, "batch"],
        temperature: float = 1.0,
        **model_kwargs,
    ) -> Int[torch.Tensor, "batch seq"]:
        """
        Sample from p(x_{t-1} | x_t) - the learned reverse process.

        Uses vectorized marginalization over all x_0 values simultaneously.
        If compile_sampling() was called, uses the compiled computation graph.

        Args:
            model: Denoising model that predicts p(x_0 | x_t) or p(x_{t-1} | x_t)
            x_t: Noisy data at timestep t
            t: Timesteps
            temperature: Sampling temperature
            **model_kwargs: Additional arguments for model

        Returns:
            x_{t-1}: Sample at timestep t-1
        """
        self._validate_model_vocab(model)

        # Get model logits
        logits = model(x=x_t, t=t, **model_kwargs)  # [B, L, C]

        if self._matrix_free:
            return self._p_sample_step_absorbing(logits, x_t, t, temperature)

        # Get transition matrices for this timestep
        Q_t = self.Q_t[t]  # [B, C, C] - single step transitions
        Q_bar_t_minus_1 = self.Q_bar_t_minus_1[t]  # [B, C, C] - cumulative transitions

        # Use compiled step if available
        if self._compiled_p_sample_step is not None:
            return self._compiled_p_sample_step(
                logits, x_t, Q_t, Q_bar_t_minus_1, temperature, self.effective_num_classes
            )

        # Fallback to uncompiled implementation
        return self._p_sample_step(logits, x_t, Q_t, Q_bar_t_minus_1, temperature)

    def _p_sample_step(
        self,
        logits: torch.Tensor,
        x_t: torch.Tensor,
        Q_t: torch.Tensor,
        Q_bar_t_minus_1: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Inner computation for p_sample - can be compiled via compile_sampling()."""
        batch_size, seq_len = x_t.shape

        # Predict clean tokens distribution
        logits = logits / temperature
        p_x0 = F.softmax(logits, dim=-1)  # [B, L, C]

        # Vectorized marginalization over all x_0 values
        # p(x_{t-1}=j | x_t) = sum_i p(x_0=i | x_t) * q(x_{t-1}=j | x_t, x_0=i)
        # where q(x_{t-1}=j | x_t, x_0=i) ∝ q(x_t | x_{t-1}=j) * q(x_{t-1}=j | x_0=i)

        # q(x_t | x_{t-1}=j) for observed x_t: extract column x_t from Q_t
        # Q_t[b, j, k] = P(x_t=k | x_{t-1}=j), we want Q_t[b, :, x_t[b,l]] for each l
        Q_t_expanded = Q_t.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, C, C]
        x_t_indices = x_t.view(batch_size, seq_len, 1, 1).expand(
            -1, -1, self.effective_num_classes, 1
        )
        q_t_given_tm1 = torch.gather(Q_t_expanded, dim=3, index=x_t_indices).squeeze(
            -1
        )  # [B, L, C]

        # Compute unnormalized posterior for all x_0 and x_{t-1} combinations:
        # unnorm[i, j] = q(x_t | x_{t-1}=j) * q(x_{t-1}=j | x_0=i)
        # Shape: [B, L, C_x0, C_tm1]
        unnorm_posterior = q_t_given_tm1.unsqueeze(2) * Q_bar_t_minus_1.unsqueeze(1)  # [B, L, C, C]

        # Normalize over x_{t-1} for each x_0
        Z = unnorm_posterior.sum(dim=-1, keepdim=True).clamp_min(EPS_PROB)  # [B, L, C, 1]
        q_posterior_all = unnorm_posterior / Z  # [B, L, C_x0, C_tm1]

        # Marginalize: p(x_{t-1}=j) = sum_i p(x_0=i) * q(x_{t-1}=j | x_t, x_0=i)
        # = einsum('bli,blij->blj', p_x0, q_posterior_all)
        p_tm1 = torch.einsum("bli,blij->blj", p_x0, q_posterior_all)

        # Sample from the marginalized distribution
        # (temperature already applied to logits before softmax — do NOT apply again)
        p_tm1_flat = p_tm1.reshape(-1, self.effective_num_classes)
        x_tm1 = torch.multinomial(p_tm1_flat.clamp_min(EPS_PROB), num_samples=1).squeeze(-1)
        return x_tm1.view(batch_size, seq_len)

    def _p_sample_step_absorbing(
        self,
        logits: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Reverse sampling step for absorbing transitions (matrix-free).

        For unmasked positions x_{t-1} = x_t deterministically.
        For masked positions:
          p(x_{t-1}=j | x_t=MASK) = p(x_0=j) · β_t·ᾱ_{t-1} / (1-ᾱ_t)  for j < K
          p(x_{t-1}=MASK | x_t=MASK) = (1-ᾱ_{t-1}) / (1-ᾱ_t)

        Memory: O(B·L·K) instead of O(B·L·K²).
        """
        B, L = x_t.shape

        beta_t = self.betas[t][:, None]  # [B, 1]
        alpha_bar_t = self.alphas_cumprod[t][:, None]  # [B, 1]
        alpha_bar_tm1 = self.alphas_cumprod_prev[t][:, None]  # [B, 1]

        is_masked = x_t == self.mask_token  # [B, L]

        # p(x_0 | x_t) from model
        p_x0 = F.softmax(logits / temperature, dim=-1)  # [B, L, C]

        # Build p(x_{t-1} | x_t=MASK): scale non-mask probs, set mask prob
        denom = (1 - alpha_bar_t).clamp_min(EPS_PROB)  # [B, 1]
        coeff = beta_t * alpha_bar_tm1 / denom  # [B, 1]
        p_tm1 = p_x0 * coeff.unsqueeze(-1)  # [B, L, C]
        p_tm1[:, :, self.mask_token] = ((1 - alpha_bar_tm1) / denom).expand(B, L)

        # Normalize
        # (temperature already applied to logits before softmax — do NOT apply again)
        p_tm1 = p_tm1 / p_tm1.sum(dim=-1, keepdim=True).clamp_min(EPS_PROB)

        # Sample
        x_tm1_sampled = (
            torch.multinomial(
                p_tm1.reshape(-1, self.effective_num_classes).clamp_min(EPS_PROB),
                num_samples=1,
            )
            .squeeze(-1)
            .view(B, L)
        )

        # Unmasked positions keep their value
        return torch.where(is_masked, x_tm1_sampled, x_t)

    def compute_loss(
        self,
        model: nn.Module,
        x_0: Int[torch.Tensor, "batch seq"],
        t: Int[torch.Tensor, "batch"] | None = None,
        loss_type: Literal["vb", "hybrid", "cross_entropy"] = "hybrid",
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            model: Denoising model
            x_0: Clean data
            t: Timesteps (randomly sampled if None)
            loss_type: Type of loss to compute
            **model_kwargs: Additional model arguments

        Returns:
            loss: Scalar loss
        """
        self._validate_model_vocab(model)
        batch_size = x_0.shape[0]

        # Sample timesteps
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)

        # Sample x_t
        x_t = self.q_sample(x_0, t)

        # Get model prediction
        logits = model(x=x_t, t=t, **model_kwargs)  # [B, L, C]

        if loss_type == "cross_entropy":
            # Simple cross-entropy loss predicting x_0
            loss = F.cross_entropy(logits.reshape(-1, self.effective_num_classes), x_0.reshape(-1))

        elif loss_type == "vb":
            # Variational bound: KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            q_posterior = self.q_posterior(x_0, x_t, t)  # [B, L, C]
            loss = F.kl_div(F.log_softmax(logits, dim=-1), q_posterior, reduction="batchmean")

        elif loss_type == "hybrid":
            # Hybrid loss: weighted combination of VB and simple prediction
            # Cross-entropy component
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.effective_num_classes), x_0.reshape(-1)
            )

            # VB component (simplified)
            q_posterior = self.q_posterior(x_0, x_t, t)
            vb_loss = F.kl_div(F.log_softmax(logits, dim=-1), q_posterior, reduction="batchmean")

            loss = ce_loss + self.hybrid_loss_coeff * vb_loss

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss

    @torch.no_grad()
    def sample(
        self, model: nn.Module, shape: tuple, temperature: float = 1.0, **model_kwargs
    ) -> Int[torch.Tensor, "batch seq"]:
        """
        Generate samples using the reverse diffusion process.

        Args:
            model: Trained denoising model
            shape: Shape of samples (batch_size, seq_len)
            temperature: Sampling temperature
            **model_kwargs: Additional model arguments

        Returns:
            x_0: Generated samples
        """
        batch_size, seq_len = shape
        device = self.device

        # Ensure model is in eval mode during sampling
        was_training = model.training
        model.eval()

        try:
            # Start from pure noise
            if self.transition_type == "absorbing":
                # Start from all mask tokens (mask_token is guaranteed set when transition_type == absorbing)
                mask_token: int = self.mask_token  # type: ignore[assignment]
                x_t = torch.full((batch_size, seq_len), mask_token, dtype=torch.long, device=device)
            else:
                # Start from uniform random
                x_t = torch.randint(0, self.num_classes, (batch_size, seq_len), device=device)

            # Reverse diffusion
            for t_idx in reversed(range(self.num_timesteps)):
                t = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)
                x_t = self.p_sample(model, x_t, t, temperature=temperature, **model_kwargs)
        finally:
            if was_training:
                model.train()

        return x_t


class D3PMKLASS:
    """
    KL-Adaptive Stability Sampling (KLASS) for D3PM/discrete diffusion.

    Dynamically stops sampling when predictions stabilize, rather than using
    all timesteps. Can significantly speed up inference.

    Reference: "KL-Guided Fast Inference in Masked Diffusion Models" (arXiv:2511.05664)

    Args:
        d3pm: Base D3PM instance
        kl_threshold: Stop when KL divergence falls below this threshold
        min_steps: Minimum number of steps before early stopping
        window_size: Number of steps to average KL over
    """

    def __init__(
        self,
        d3pm: D3PM,
        kl_threshold: float = 0.01,
        min_steps: int = 50,
        window_size: int = 5,
    ):
        self.d3pm = d3pm
        self.kl_threshold = kl_threshold
        self.min_steps = min_steps
        self.window_size = window_size
        # Store KL history as tensors to avoid GPU->CPU sync in hot loop
        self._kl_history_tensors: list[torch.Tensor] = []

    def reset(self) -> None:
        """Reset state for a new sampling run."""
        self._kl_history_tensors = []

    def should_stop(self, step: int, max_steps: int, kl_divergence: torch.Tensor) -> bool:
        """
        Determine if generation should stop based on KL divergence.

        Uses tensor operations to avoid GPU->CPU sync in the hot sampling loop.

        Args:
            step: Current step number (from 0, counting down)
            max_steps: Maximum number of steps
            kl_divergence: KL divergence tensor (scalar) between consecutive predictions

        Returns:
            True if generation should stop, False otherwise
        """
        steps_taken = max_steps - step
        if steps_taken < self.min_steps:
            return False

        self._kl_history_tensors.append(kl_divergence)

        # Trim history to avoid memory growth (keep only what's needed for window)
        if len(self._kl_history_tensors) > self.window_size * 2:
            self._kl_history_tensors = self._kl_history_tensors[-self.window_size :]

        # Average over window using tensor operations (no .item() sync)
        if len(self._kl_history_tensors) >= self.window_size:
            # Stack recent tensors and compute mean on GPU
            recent_kl = torch.stack(self._kl_history_tensors[-self.window_size :])
            avg_kl = recent_kl.mean()
            # Tensor comparison - only syncs when Python needs the bool result
            if avg_kl < self.kl_threshold:
                return True

        return step == 0

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        temperature: float = 1.0,
        **model_kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generate samples with KL-guided early stopping.

        Args:
            model: Trained denoising model
            shape: Shape of samples (batch_size, seq_len)
            temperature: Sampling temperature
            **model_kwargs: Additional model arguments

        Returns:
            x_0: Generated samples
            metrics: Dict with generation statistics
        """
        # Reset state for new sampling run
        self.reset()

        batch_size, seq_len = shape
        device = self.d3pm.device
        max_steps = self.d3pm.num_timesteps

        metrics: dict = {
            "steps_taken": 0,
            "final_kl": 0.0,
            "kl_history": [],
            "early_stopped": False,
        }

        # Ensure model is in eval mode during sampling
        was_training = model.training
        model.eval()

        # Track KL tensors for deferred .item() conversion (avoids GPU->CPU sync in loop)
        kl_tensors: list[torch.Tensor] = []
        final_kl_tensor: torch.Tensor | None = None

        try:
            # Start from pure noise
            if self.d3pm.transition_type == "absorbing":
                # mask_token is guaranteed set when transition_type == absorbing
                mask_token: int = self.d3pm.mask_token  # type: ignore[assignment]
                x_t = torch.full((batch_size, seq_len), mask_token, dtype=torch.long, device=device)
            else:
                x_t = torch.randint(0, self.d3pm.num_classes, (batch_size, seq_len), device=device)

            # Previous predictions for KL computation
            prev_logits = None

            # Reverse diffusion with early stopping
            for t_idx in reversed(range(max_steps)):
                t = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)

                # Get current predictions
                logits = model(x=x_t, t=t, **model_kwargs)

                # Compute KL divergence if we have previous predictions
                # Keep as tensor to avoid GPU->CPU sync in hot loop
                kl_div: torch.Tensor
                if prev_logits is not None:
                    # KL(prev || curr) - measures how much predictions changed
                    kl_div = F.kl_div(
                        F.log_softmax(prev_logits, dim=-1),
                        F.softmax(logits, dim=-1),
                        reduction="batchmean",
                    )
                else:
                    # Use tensor infinity to stay on device
                    kl_div = torch.tensor(float("inf"), device=device)

                # Check if we should stop (uses tensor comparison internally)
                if self.should_stop(t_idx, max_steps, kl_div):
                    metrics["early_stopped"] = True
                    metrics["steps_taken"] = max_steps - t_idx
                    final_kl_tensor = kl_div
                    break

                # Sample x_{t-1}
                x_t = self.d3pm.p_sample(model, x_t, t, temperature=temperature, **model_kwargs)

                # Store logits for next iteration
                prev_logits = logits.detach().clone()
                kl_tensors.append(kl_div)

            if not metrics["early_stopped"]:
                metrics["steps_taken"] = max_steps
        finally:
            if was_training:
                model.train()

        # Convert KL history to Python floats AFTER loop completes (single sync point)
        metrics["kl_history"] = [kl.item() for kl in kl_tensors]
        if final_kl_tensor is not None:
            metrics["final_kl"] = final_kl_tensor.item()

        return x_t, metrics
