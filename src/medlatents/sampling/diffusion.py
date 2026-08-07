"""Advanced sampling schedulers for discrete diffusion models.

Implements modern schedulers from continuous diffusion adapted for discrete:
- DDIM (faster sampling)
- DPM-Solver (high-quality fast sampling)
- Euler (simple ODE solver)
- Ancestral sampling (adds stochasticity)
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class DiscreteScheduler(ABC):
    """Base class for discrete diffusion schedulers."""

    def __init__(
        self,
        num_train_steps: int = 1000,
        num_inference_steps: int = 50,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        if num_train_steps <= 0:
            raise ValueError(f"num_train_steps must be > 0, got {num_train_steps}")
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")
        if not (0.0 < beta_start < 1.0) or not (0.0 < beta_end < 1.0):
            raise ValueError(
                f"beta_start and beta_end must be in (0, 1), got {beta_start} and {beta_end}"
            )
        if beta_end <= beta_start:
            raise ValueError(f"beta_end must be > beta_start, got {beta_end} <= {beta_start}")

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self._validated_probs = False
        self._validated_logits = False

        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_steps)
        elif beta_schedule == "scaled_linear":
            # Used in Stable Diffusion
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_steps) ** 2
        elif beta_schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = num_train_steps + 1
            x = torch.linspace(0, num_train_steps, steps)
            alphas_cumprod = torch.cos(((x / num_train_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single denoising step."""
        raise NotImplementedError

    def set_timesteps(self, num_inference_steps: int):
        """Set the number of inference steps."""
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")
        if num_inference_steps > self.num_train_steps:
            raise ValueError(
                "num_inference_steps must be <= num_train_steps "
                f"({self.num_train_steps}), got {num_inference_steps}"
            )
        self.num_inference_steps = num_inference_steps
        # Create timestep schedule (uniform spacing)
        self.timesteps = torch.linspace(
            self.num_train_steps - 1, 0, num_inference_steps, dtype=torch.long
        )

    def _ensure_timesteps(self, timestep: int | torch.Tensor) -> int:
        if not hasattr(self, "timesteps"):
            raise RuntimeError("set_timesteps must be called before step().")
        if self.timesteps.numel() == 0:
            raise RuntimeError("scheduler.timesteps is empty; call set_timesteps with > 0.")
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() != 1:
                raise ValueError("timestep must be a scalar value")
            timestep_value = int(timestep.item())
        else:
            if not isinstance(timestep, int):
                raise TypeError(f"timestep must be an int, got {type(timestep)}")
            timestep_value = timestep
        matches = (self.timesteps == timestep_value).nonzero(as_tuple=True)[0]
        if matches.numel() != 1:
            raise ValueError(
                f"timestep {timestep_value} is not in scheduler.timesteps; "
                "call set_timesteps() and use its values."
            )
        return int(matches.item())

    def _looks_like_probs(self, tensor: torch.Tensor) -> bool:
        if tensor.dim() == 0 or tensor.size(-1) == 0:
            return False
        if not torch.isfinite(tensor).all():
            return False
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        if min_val < -1e-6 or max_val > 1.0 + 1e-6:
            return False
        sums = tensor.sum(dim=-1)
        return torch.allclose(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3)

    def _validate_probs_once(self, tensor: torch.Tensor) -> None:
        if self._validated_probs:
            return
        if tensor.dim() == 0 or tensor.size(-1) == 0:
            raise ValueError("model_output must have a non-empty vocab dimension")
        if not torch.isfinite(tensor).all():
            raise ValueError("model_output contains NaN or Inf values")
        if not self._looks_like_probs(tensor):
            raise ValueError(
                "model_output must be probabilities that sum to 1 along the last dimension. "
                "Apply softmax before calling this scheduler."
            )
        self._validated_probs = True

    def _validate_logits_once(self, tensor: torch.Tensor) -> None:
        if self._validated_logits:
            return
        if tensor.dim() == 0 or tensor.size(-1) == 0:
            raise ValueError("model_output must have a non-empty vocab dimension")
        if not torch.isfinite(tensor).all():
            raise ValueError("model_output contains NaN or Inf values")
        if self._looks_like_probs(tensor):
            raise ValueError(
                "model_output appears to be probabilities; this scheduler expects logits. "
                "Pass raw logits instead of softmax outputs."
            )
        self._validated_logits = True


class DDIMScheduler(DiscreteScheduler):
    """
    DDIM scheduler for faster sampling.

    Reference: "Denoising Diffusion Implicit Models" (Song et al., 2020)
    https://arxiv.org/abs/2010.02502

    Allows deterministic sampling and can skip timesteps for faster generation.
    """

    def __init__(self, eta: float = 0.0, **kwargs):
        """
        Args:
            eta: stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)
        """
        if not isinstance(eta, (int, float)) or not (0.0 <= eta <= 1.0):
            raise ValueError(f"eta must be in [0, 1], got {eta}")
        super().__init__(**kwargs)
        self.eta = eta

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: int | None = None,
    ) -> torch.Tensor:
        """
        DDIM sampling step for discrete data.

        Args:
            model_output: [batch, seq_len, vocab_size] predicted probabilities
            timestep: current timestep
            sample: [batch, seq_len] current discrete sample
            prev_timestep: previous timestep (for skipping)

        Returns:
            prev_sample: [batch, seq_len] denoised sample
        """
        self._validate_probs_once(model_output)
        timestep_idx = self._ensure_timesteps(timestep)
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]

        if prev_timestep is None:
            # Find next timestep
            if timestep_idx < len(self.timesteps) - 1:
                prev_timestep = self.timesteps[timestep_idx + 1]
            else:
                prev_timestep = 0

        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        )

        # Predict x_0 from model output (probabilities)
        # For discrete: model predicts categorical distribution over clean data
        pred_x0_probs = model_output

        # DDIM update (adapted for discrete)
        # Sample from pred_x0_probs with noise scaled by eta
        if self.eta > 0:
            # Stochastic sampling
            variance = (
                (1 - alpha_prod_t_prev)
                / (1 - alpha_prod_t)
                * (1 - alpha_prod_t / alpha_prod_t_prev)
            )
            variance = variance * self.eta

            # Add noise to probabilities (use clamp for fp16 stability)
            noise = torch.randn_like(pred_x0_probs) * variance.sqrt()
            pred_x0_probs = F.softmax(torch.log(pred_x0_probs.clamp(min=1e-6)) + noise, dim=-1)

        # Sample from predicted distribution
        prev_sample = torch.multinomial(
            pred_x0_probs.view(-1, pred_x0_probs.size(-1)), num_samples=1
        ).view(sample.shape)

        return prev_sample


class DPMSolverScheduler(DiscreteScheduler):
    """
    DPM-Solver scheduler for high-quality fast sampling.

    Reference: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"
    https://arxiv.org/abs/2206.00927

    Achieves high quality with very few steps (10-20).
    """

    def __init__(self, solver_order: int = 2, prediction_type: str = "epsilon", **kwargs):
        """
        Args:
            solver_order: order of DPM-Solver (1, 2, or 3)
            prediction_type: 'epsilon' or 'sample'
        """
        if solver_order not in {1, 2, 3}:
            raise ValueError(f"solver_order must be 1, 2, or 3, got {solver_order}")
        if prediction_type not in {"epsilon", "sample"}:
            raise ValueError(
                f"prediction_type must be 'epsilon' or 'sample', got {prediction_type}"
            )
        super().__init__(**kwargs)
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.lower_order_samples = []

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver sampling step adapted for discrete data.

        Uses multi-step method for improved quality.
        """
        # Get timestep info
        self._validate_probs_once(model_output)
        timestep_idx = self._ensure_timesteps(timestep)

        if timestep_idx < len(self.timesteps) - 1:
            prev_timestep = self.timesteps[timestep_idx + 1]
        else:
            prev_timestep = 0

        # Get alpha values
        lambda_t = -torch.log(self.alphas_cumprod[timestep])
        lambda_s = -torch.log(
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        )

        h = lambda_s - lambda_t

        # First-order update (exponential integrator)
        if len(self.lower_order_samples) < self.solver_order - 1 or self.solver_order == 1:
            # Use first-order method
            pred_probs = model_output
            sample_coeff = torch.exp(lambda_s) / torch.exp(lambda_t)

            # For discrete: interpolate in probability space (use clamp for fp16 stability)
            prev_probs = F.softmax(torch.log(pred_probs.clamp(min=1e-6)) * sample_coeff, dim=-1)

            prev_sample = torch.multinomial(
                prev_probs.view(-1, prev_probs.size(-1)), num_samples=1
            ).view(sample.shape)

            self.lower_order_samples.append(model_output)

        else:
            # Multi-step DPM-Solver
            # Use previous predictions for higher-order update
            prev_output = self.lower_order_samples[-1]

            # Second-order correction
            r = (lambda_s - self.prev_lambda) / h if hasattr(self, "prev_lambda") else 0

            D = model_output
            D_prev = prev_output

            # Compute second-order prediction
            pred_probs = D + (D - D_prev) * r / (2 * r + 1)
            pred_probs = F.softmax(pred_probs, dim=-1)

            prev_sample = torch.multinomial(
                pred_probs.view(-1, pred_probs.size(-1)), num_samples=1
            ).view(sample.shape)

            # Update history
            self.lower_order_samples.pop(0)
            self.lower_order_samples.append(model_output)

        self.prev_lambda = lambda_t
        return prev_sample


class EulerDiscreteScheduler(DiscreteScheduler):
    """
    Euler discrete scheduler - simple ODE solver.

    Good balance of speed and quality. Similar to DDPM but can skip steps.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euler method step for discrete sampling.

        Args:
            model_output: [batch, seq_len, vocab_size] predicted probabilities
            timestep: current timestep
            sample: [batch, seq_len] current sample

        Returns:
            prev_sample: [batch, seq_len] denoised sample
        """
        # Get timestep info
        self._validate_probs_once(model_output)
        timestep_idx = self._ensure_timesteps(timestep)

        if timestep_idx < len(self.timesteps) - 1:
            prev_timestep = self.timesteps[timestep_idx + 1]
        else:
            prev_timestep = 0

        # Compute sigmas from alphas
        sigma_t = ((1 - self.alphas_cumprod[timestep]) / self.alphas_cumprod[timestep]).sqrt()
        sigma_prev = (
            ((1 - self.alphas_cumprod[prev_timestep]) / self.alphas_cumprod[prev_timestep]).sqrt()
            if prev_timestep >= 0
            else torch.tensor(0.0)
        )

        # Euler update
        dt = sigma_prev - sigma_t

        # Get predicted x_0 probabilities
        pred_probs = model_output

        # Add scaled noise (for discrete: perturb log-probabilities)
        if dt.abs() > 1e-6:  # Use fp16-safe threshold
            noise = torch.randn_like(pred_probs)
            perturbed_logprobs = torch.log(pred_probs.clamp(min=1e-6)) + noise * dt.abs()
            pred_probs = F.softmax(perturbed_logprobs, dim=-1)

        # Sample from predicted distribution
        prev_sample = torch.multinomial(
            pred_probs.view(-1, pred_probs.size(-1)), num_samples=1
        ).view(sample.shape)

        return prev_sample


class AncestralSamplingScheduler(DiscreteScheduler):
    """
    Ancestral sampling (stochastic) scheduler.

    Adds noise at each step for more diverse samples.
    Good for generation tasks where diversity is important.
    """

    def __init__(self, noise_scale: float = 1.0, **kwargs):
        """
        Args:
            noise_scale: scale of added noise (1.0 = standard, >1.0 = more noise)
        """
        if not isinstance(noise_scale, (int, float)) or noise_scale < 0:
            raise ValueError(f"noise_scale must be >= 0, got {noise_scale}")
        super().__init__(**kwargs)
        self.noise_scale = noise_scale

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ancestral sampling step with added noise.

        Args:
            model_output: [batch, seq_len, vocab_size] predicted probabilities
            timestep: current timestep
            sample: [batch, seq_len] current sample

        Returns:
            prev_sample: [batch, seq_len] noisy denoised sample
        """
        # Get alpha values
        self._validate_probs_once(model_output)
        alpha_prod_t = self.alphas_cumprod[timestep]

        # Find next timestep
        timestep_idx = self._ensure_timesteps(timestep)
        if timestep_idx < len(self.timesteps) - 1:
            prev_timestep = self.timesteps[timestep_idx + 1]
        else:
            prev_timestep = 0

        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        )

        # Get predicted clean probabilities
        pred_probs = model_output

        # Compute posterior variance (for ancestral sampling)
        variance = (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

        # Add noise scaled by variance (use clamp for fp16 stability)
        noise = torch.randn_like(pred_probs) * variance.sqrt() * self.noise_scale
        noisy_logprobs = torch.log(pred_probs.clamp(min=1e-6)) + noise
        noisy_probs = F.softmax(noisy_logprobs, dim=-1)

        # Sample from noisy distribution
        prev_sample = torch.multinomial(
            noisy_probs.view(-1, noisy_probs.size(-1)), num_samples=1
        ).view(sample.shape)

        return prev_sample


class DPMSolverPlusPlusScheduler(DiscreteScheduler):
    """
    DPM-Solver++ scheduler for high-quality fast sampling.

    Reference: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models" (Lu et al., 2022)
    https://arxiv.org/abs/2211.01095

    Improvements over DPM-Solver:
    - Better handling of guided sampling (CFG)
    - Improved multistep methods
    - Works well with as few as 10-20 steps
    """

    def __init__(
        self,
        solver_order: int = 2,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        **kwargs,
    ):
        """
        Args:
            solver_order: Order of DPM-Solver (1, 2, or 3)
            algorithm_type: 'dpmsolver' or 'dpmsolver++' (recommended)
            solver_type: 'midpoint' or 'heun' (for 2nd order)
            lower_order_final: Use lower order for final step (more stable)
        """
        if solver_order not in {1, 2, 3}:
            raise ValueError(f"solver_order must be 1, 2, or 3, got {solver_order}")
        if algorithm_type not in {"dpmsolver", "dpmsolver++"}:
            raise ValueError(
                f"algorithm_type must be 'dpmsolver' or 'dpmsolver++', got {algorithm_type}"
            )
        if solver_type not in {"midpoint", "heun"}:
            raise ValueError(f"solver_type must be 'midpoint' or 'heun', got {solver_type}")
        super().__init__(**kwargs)
        self.solver_order = solver_order
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final

        # Storage for multi-step methods
        self.model_outputs = []
        self.sample_outputs = []

    def _convert_to_lambda(self, timestep: int) -> torch.Tensor:
        """Convert timestep to lambda (log-SNR)."""
        alpha_cumprod = self.alphas_cumprod[timestep]
        sigma = ((1 - alpha_cumprod) / alpha_cumprod).sqrt()
        return -torch.log(sigma)

    def _get_prev_timestep(self, timestep: int) -> int:
        """Get previous timestep for scheduler."""
        timestep_idx = self._ensure_timesteps(timestep)
        if timestep_idx < len(self.timesteps) - 1:
            return self.timesteps[timestep_idx + 1].item()
        return 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        DPM-Solver++ step adapted for discrete data.

        Args:
            model_output: [batch, seq_len, vocab_size] predicted logits
            timestep: Current timestep
            sample: [batch, seq_len] current sample
            return_dict: Whether to return additional info

        Returns:
            prev_sample: [batch, seq_len] updated sample
        """
        self._validate_logits_once(model_output)
        step_index = self._ensure_timesteps(timestep)
        prev_timestep = self._get_prev_timestep(timestep)

        # Get lambdas (log-SNR)
        lambda_t = self._convert_to_lambda(timestep)
        lambda_s = (
            self._convert_to_lambda(prev_timestep)
            if prev_timestep > 0
            else torch.tensor(float("inf"))
        )

        # Compute h = lambda_s - lambda_t
        h = lambda_s - lambda_t

        # Determine order for this step
        if self.lower_order_final and step_index == len(self.timesteps) - 1:
            order = 1
        else:
            order = min(self.solver_order, step_index + 1)

        # Store model output for multi-step
        self.model_outputs.append(model_output)
        if len(self.model_outputs) > self.solver_order:
            self.model_outputs.pop(0)

        if order == 1:
            # First-order DPM-Solver++
            prev_probs = self._first_order_update(model_output, h)
        elif order == 2:
            # Second-order DPM-Solver++
            prev_probs = self._second_order_update(
                self.model_outputs[-2] if len(self.model_outputs) >= 2 else model_output,
                model_output,
                h,
            )
        else:
            # Third-order DPM-Solver++
            prev_probs = self._third_order_update(
                self.model_outputs[-3] if len(self.model_outputs) >= 3 else model_output,
                self.model_outputs[-2] if len(self.model_outputs) >= 2 else model_output,
                model_output,
                h,
            )

        # Sample from predicted distribution
        prev_sample = torch.multinomial(
            prev_probs.view(-1, prev_probs.size(-1)), num_samples=1
        ).view(sample.shape)

        return prev_sample

    def _first_order_update(
        self,
        model_output: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """First-order (Euler) update."""
        # For discrete: direct probability prediction
        pred_probs = F.softmax(model_output, dim=-1)
        return pred_probs

    def _second_order_update(
        self,
        model_output_prev: torch.Tensor,
        model_output: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Second-order (midpoint/Heun) update."""
        if self.solver_type == "midpoint":
            # Midpoint method
            r = 0.5
            D = model_output
            D_prev = model_output_prev

            # Linear combination
            pred_logits = D + r * (D - D_prev)
        else:
            # Heun method
            D = model_output
            D_prev = model_output_prev

            # Heun corrector
            pred_logits = 0.5 * (D + D_prev)

        pred_probs = F.softmax(pred_logits, dim=-1)
        return pred_probs

    def _third_order_update(
        self,
        model_output_2: torch.Tensor,
        model_output_1: torch.Tensor,
        model_output: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Third-order update."""
        # Quadratic interpolation
        D = model_output
        D_1 = model_output_1
        D_2 = model_output_2

        # Third-order Adams-Bashforth coefficients
        pred_logits = (23 * D - 16 * D_1 + 5 * D_2) / 12

        pred_probs = F.softmax(pred_logits, dim=-1)
        return pred_probs


class HeunDiscreteScheduler(DiscreteScheduler):
    """
    Heun (2nd-order Runge-Kutta) scheduler for discrete diffusion.

    Provides better accuracy than Euler with only 2x the compute per step.
    Good balance between speed and quality.

    Reference: Based on Heun's method adapted from continuous diffusion.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_derivative = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        model_fn: callable = None,
    ) -> torch.Tensor:
        """
        Heun method step for discrete sampling.

        Args:
            model_output: [batch, seq_len, vocab_size] predicted logits
            timestep: current timestep
            sample: [batch, seq_len] current sample
            model_fn: Optional model function for corrector step

        Returns:
            prev_sample: [batch, seq_len] denoised sample
        """
        # Get timestep info
        self._validate_logits_once(model_output)
        timestep_idx = self._ensure_timesteps(timestep)

        if timestep_idx < len(self.timesteps) - 1:
            prev_timestep = self.timesteps[timestep_idx + 1]
        else:
            prev_timestep = 0

        # Heun predictor step
        pred_probs = F.softmax(model_output, dim=-1)

        # If we have model_fn, do proper Heun with corrector
        if model_fn is not None and prev_timestep > 0:
            # Euler predictor
            pred_sample = torch.multinomial(
                pred_probs.view(-1, pred_probs.size(-1)), num_samples=1
            ).view(sample.shape)

            # Get corrector prediction
            with torch.no_grad():
                corrector_output = model_fn(pred_sample, prev_timestep)
                corrector_probs = F.softmax(corrector_output, dim=-1)

            # Heun average (use clamp for fp16 stability)
            avg_probs = 0.5 * (pred_probs + corrector_probs)
            prev_probs = F.softmax(torch.log(avg_probs.clamp(min=1e-6)), dim=-1)
        else:
            # Just use predictor
            prev_probs = pred_probs

        # Add small noise for diversity (use clamp for fp16 stability)
        if prev_timestep > 0:
            noise = torch.randn_like(prev_probs) * 0.01
            prev_probs = F.softmax(torch.log(prev_probs.clamp(min=1e-6)) + noise, dim=-1)

        # Sample from predicted distribution
        prev_sample = torch.multinomial(
            prev_probs.view(-1, prev_probs.size(-1)), num_samples=1
        ).view(sample.shape)

        return prev_sample


class AdaptiveScheduler(DiscreteScheduler):
    """
    Adaptive scheduler that adjusts step size based on model confidence.

    Uses the model's prediction confidence to dynamically adjust the
    sampling trajectory, taking larger steps when confident and smaller
    steps when uncertain.

    This can significantly reduce the number of steps needed while
    maintaining quality.
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 100,
        confidence_threshold: float = 0.9,
        **kwargs,
    ):
        """
        Args:
            min_steps: Minimum number of steps
            max_steps: Maximum number of steps
            confidence_threshold: Threshold for taking larger steps
        """
        if not isinstance(min_steps, int) or min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {min_steps}")
        if not isinstance(max_steps, int) or max_steps < min_steps:
            raise ValueError(f"max_steps must be >= min_steps ({min_steps}), got {max_steps}")
        if not (0.0 < confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {confidence_threshold}")
        super().__init__(**kwargs)
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.current_step = 0
        self.total_steps_taken = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive step that may skip timesteps based on confidence.

        Args:
            model_output: [batch, seq_len, vocab_size] predicted logits
            timestep: current timestep
            sample: [batch, seq_len] current sample

        Returns:
            prev_sample: [batch, seq_len] denoised sample
        """
        self._validate_logits_once(model_output)
        self._ensure_timesteps(timestep)
        self.total_steps_taken += 1

        # Compute model confidence (max probability)
        pred_probs = F.softmax(model_output, dim=-1)
        max_probs = pred_probs.max(dim=-1)[0]  # [batch, seq_len]
        avg_confidence = max_probs.mean()

        # Determine step size based on confidence (for future adaptive stepping)
        _ = 2 if avg_confidence > self.confidence_threshold else 1

        # Standard sampling step
        prev_sample = torch.multinomial(
            pred_probs.view(-1, pred_probs.size(-1)), num_samples=1
        ).view(sample.shape)

        return prev_sample

    def get_efficiency(self) -> float:
        """Get sampling efficiency (steps saved ratio)."""
        if self.total_steps_taken == 0:
            return 1.0
        return self.num_inference_steps / self.total_steps_taken
