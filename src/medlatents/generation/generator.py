"""High-level generators for discrete and continuous latent models."""

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from ..autoregressive import Autoreg_models
from ..data.tokenizers import ContinuousTokenizer, DiscreteTokenizer
from ..diffusion import D3PM
from ..diffusion.continuous import ContinuousGaussianDiffusion
from ..flow_matching.continuous import RectifiedFlow
from ..flow_matching.discrete import get_source_distribution
from ..maskgit import MaskGIT_models
from ..networks.diffusion_transformer import DiscreteDiT_models

logger = logging.getLogger(__name__)


def _get_nibabel():
    """Lazy import for nibabel to avoid loading it at module import time."""
    import nibabel

    return nibabel


def _ensure_finite_positive(value: float, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value)}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def _ensure_int_ge(value: int, name: str, minimum: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value)}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


class DiscreteLatentGenerator:
    """
    Unified generator for discrete latent generative models.

    Supports:
    - autoreg: Autoregressive generation
    - maskgit: Iterative parallel generation (MaskGIT)
    - flow: Discrete flow matching generation
    - diffusion: D3PM discrete diffusion generation
    """

    def __init__(
        self,
        model_type: Literal["autoreg", "maskgit", "flow", "diffusion"],
        model: torch.nn.Module,
        tokenizer: DiscreteTokenizer,
        device: torch.device,
    ):
        self.model_type = model_type
        self.model = model.to(device)
        self.tokenizer = tokenizer.to(device)
        self.device = device
        self.special_tokens = getattr(model, "special_tokens", None)

        # Initialize D3PM if using diffusion
        if model_type == "diffusion":
            vocab_size = getattr(model, "out_channels", getattr(model, "vocab_size", None))
            if vocab_size is None:
                raise ValueError(
                    "Could not determine vocab_size from model. "
                    "Ensure your model has an 'out_channels' or 'vocab_size' attribute. "
                    "For DiscreteDiT models, set out_channels in the constructor."
                )
            self.diffusion = D3PM(num_classes=vocab_size, device=device)
        else:
            self.diffusion = None

        self.model.eval()
        self.tokenizer.eval()

    def _encode_tokens(self, volume: torch.Tensor) -> torch.Tensor:
        if hasattr(self.tokenizer, "tokenize"):
            tokens = self.tokenizer.tokenize(volume)
        elif hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(volume)
            tokens = result[0] if isinstance(result, tuple) else result
        else:
            from ..data.tokenizers import get_tokenizer_module_name

            raise AttributeError(
                f"Tokenizer {get_tokenizer_module_name(self.tokenizer)} (from {get_tokenizer_module_name(self.tokenizer.__class__.__module__)}) "
                f"must expose tokenize() or encode() method"
            )
        return torch.as_tensor(tokens)

    @classmethod
    def from_checkpoints(
        cls,
        model_type: Literal["autoreg", "maskgit", "flow", "diffusion"],
        model_path: str,
        tokenizer_path: str,
        device: torch.device | None = None,
        weights_only: bool = True,
    ):
        """Load generator from model and tokenizer checkpoints."""
        if device is None:
            device = torch.device("cpu")

        # Load tokenizer
        tokenizer_weights = torch.load(
            tokenizer_path, map_location=device, weights_only=weights_only
        )
        tokenizer = DiscreteTokenizer(tokenizer_weights["hparams"]).to(device)
        tokenizer.load_state_dict(tokenizer_weights["net"])

        # Load model
        model_weights = torch.load(model_path, map_location=device, weights_only=weights_only)
        hparams = model_weights["hparams"]
        model_size = hparams.pop("model_size")

        # Select model class based on type
        if model_type == "autoreg":
            model = Autoreg_models[f"Autoreg-{model_size.upper()}"](**hparams)
        elif model_type == "maskgit":
            model = MaskGIT_models[f"MaskGIT-{model_size.upper()}"](**hparams)
        elif model_type == "flow" or model_type == "diffusion":
            model = DiscreteDiT_models[f"DiscreteDiT-{model_size.upper()}"](**hparams)
        else:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Valid options are: 'autoreg', 'maskgit', 'flow', 'diffusion'."
            )

        model.load_state_dict(model_weights["net"])

        return cls(model_type, model, tokenizer, device)

    def generate(
        self,
        num_samples: int,
        seq_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples based on model type.

        Args:
            num_samples: Number of samples to generate
            seq_length: Length of sequences
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            num_steps: Number of iterative steps (maskgit/flow/diffusion)
            seed: Random seed for reproducibility (None for random)
            **kwargs: Additional model-specific parameters

        Returns:
            Generated token sequences [num_samples, seq_length]
        """
        _ensure_int_ge(num_samples, "num_samples", 1)
        _ensure_int_ge(seq_length, "seq_length", 1)
        _ensure_finite_positive(temperature, "temperature")
        if top_k is not None:
            _ensure_int_ge(top_k, "top_k", 0)
        if num_steps is not None:
            _ensure_int_ge(num_steps, "num_steps", 1)
        if seed is not None and not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed)}")

        if seed is not None:
            torch.manual_seed(seed)

        with torch.inference_mode():
            if self.model_type == "autoreg":
                return self._generate_autoreg(num_samples, seq_length, temperature, top_k)
            elif self.model_type == "maskgit":
                return self._generate_maskgit(
                    num_samples, seq_length, temperature, top_k, num_steps or 12
                )
            elif self.model_type == "flow":
                return self._generate_flow(
                    num_samples, seq_length, temperature, num_steps or 100, **kwargs
                )
            elif self.model_type == "diffusion":
                return self._generate_diffusion(num_samples, seq_length, temperature)

    def _generate_autoreg(
        self,
        num_samples: int,
        seq_length: int,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        """Generate using autoregressive sampling."""
        if self.special_tokens is not None:
            prompt_token = self.special_tokens.bos
            eos_token = self.special_tokens.eos
        else:
            prompt_token = 0
            eos_token = None
        prompt = torch.full((num_samples, 1), prompt_token, dtype=torch.long, device=self.device)
        return self.model.generate(
            prompt=prompt,
            max_length=seq_length,
            temperature=temperature,
            top_k=top_k,
            eos_token=eos_token,
        )

    def _generate_maskgit(
        self,
        num_samples: int,
        seq_length: int,
        temperature: float,
        top_k: int | None,
        num_steps: int,
    ) -> torch.Tensor:
        """Generate using iterative parallel decoding (MaskGIT)."""
        x = torch.full(
            (num_samples, seq_length),
            self.model.mask_token,
            dtype=torch.long,
            device=self.device,
        )
        return self.model.generate(x=x, num_steps=num_steps, temperature=temperature, top_k=top_k)

    def _generate_flow(
        self,
        num_samples: int,
        seq_length: int,
        temperature: float,
        num_steps: int,
        scheduler_type: str = "polynomial",
        scheduler_power: float = 2.0,
        source_dist: str = "uniform",
        vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Generate using discrete flow matching.

        Uses iterative sampling: at each step, the model predicts logits
        which are converted to probabilities and sampled from.
        """
        _ensure_int_ge(num_steps, "num_steps", 1)
        if vocab_size is None:
            vocab_size = self.model.out_channels

        mask_token = self.special_tokens.mask if self.special_tokens is not None else None
        source_distribution = get_source_distribution(
            source_dist, vocab_size, mask_token=mask_token
        )
        x = source_distribution.sample((num_samples, seq_length)).to(self.device)

        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)

        for t in time_steps[:-1]:
            t_batch = t.expand(num_samples)
            logits = self.model(x, t_batch)

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            x = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
                num_samples, seq_length
            )

        return x

    def _generate_diffusion(
        self, num_samples: int, seq_length: int, temperature: float
    ) -> torch.Tensor:
        """Generate using D3PM discrete diffusion."""
        return self.diffusion.sample(
            self.model, shape=(num_samples, seq_length), temperature=temperature
        )

    def generate_volumes(
        self,
        num_samples: int,
        seq_length: int,
        output_dir: str,
        temperature: float = 1.0,
        top_k: int | None = None,
        num_steps: int | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Generate samples and save as NIfTI volumes."""
        output_path = Path(output_dir)
        volumes_dir = output_path / "volumes"
        volumes_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_samples} samples...")
        for i in tqdm(range(num_samples)):
            # Generate single sample
            seed_i = None if seed is None else seed + i
            x = self.generate(
                num_samples=1,
                seq_length=seq_length,
                temperature=temperature,
                top_k=top_k,
                num_steps=num_steps,
                seed=seed_i,
                **kwargs,
            )

            # Decode to volume
            volume = self.tokenizer.detokenize(x)

            # Save as NIfTI
            volume_nii = _get_nibabel().Nifti1Image(volume[0].cpu().numpy(), np.eye(4))
            volume_path = volumes_dir / f"volume_{i:04d}.nii.gz"
            _get_nibabel().save(volume_nii, volume_path)

    def inpaint_volumes(
        self,
        input_pattern: str,
        output_dir: str,
        likelihood_threshold: float = 0.005,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """Inpaint anomalies in volumes."""
        import glob

        output_path = Path(output_dir)
        volumes_dir = output_path / "inpainted"
        volumes_dir.mkdir(parents=True, exist_ok=True)

        input_files = glob.glob(input_pattern)
        logger.info(f"Inpainting {len(input_files)} volumes...")

        for i, file_path in enumerate(tqdm(input_files)):
            # Load volume
            volume_nii = _get_nibabel().load(file_path)
            volume = torch.from_numpy(volume_nii.get_fdata()).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                tokens = self._encode_tokens(volume)

            inpainted_tokens = self.model.inpaint_anomalies(
                tokens,
                likelihood_threshold=likelihood_threshold,
                temperature=temperature,
                top_k=top_k,
            )

            with torch.inference_mode():
                inpainted_volume = self.tokenizer.detokenize(inpainted_tokens)

            # Save
            inpainted_nii = _get_nibabel().Nifti1Image(
                inpainted_volume[0].cpu().numpy(), volume_nii.affine
            )
            output_path = volumes_dir / f"inpainted_{i:04d}.nii.gz"
            _get_nibabel().save(inpainted_nii, output_path)


class ContinuousLatentGenerator:
    """Generator for continuous latent diffusion and flow models."""

    def __init__(
        self,
        model_type: Literal["diffusion", "flow"],
        model: torch.nn.Module,
        tokenizer: ContinuousTokenizer | None,
        device: torch.device,
        diffusion: ContinuousGaussianDiffusion | None = None,
        flow: RectifiedFlow | None = None,
    ) -> None:
        if model_type not in {"diffusion", "flow"}:
            raise ValueError(
                f"Unsupported model_type '{model_type}' for ContinuousLatentGenerator. "
                f"Valid options are: 'diffusion', 'flow'. "
                f"For discrete models, use DiscreteLatentGenerator instead."
            )

        self.model_type = model_type
        self.model = model.to(device)
        self.tokenizer = (
            tokenizer.to(device) if tokenizer and hasattr(tokenizer, "to") else tokenizer
        )
        self.device = device

        self.diffusion = diffusion
        self.flow = flow

        self.model.eval()
        if hasattr(self.tokenizer, "eval"):
            self.tokenizer.eval()

    def _infer_latent_shape(self) -> tuple[int, ...]:
        """Infer latent shape from tokenizer configuration."""
        if self.tokenizer is None:
            raise ValueError(
                "Cannot infer latent shape: no tokenizer provided. "
                "Either pass a tokenizer to the constructor, or provide latent_shape "
                "explicitly to the generate() method."
            )

        if hasattr(self.tokenizer, "latent_shape"):
            return tuple(self.tokenizer.latent_shape)

        raise ValueError(
            f"Tokenizer {type(self.tokenizer).__name__} must have 'latent_shape' attribute. "
            "This attribute should define the spatial dimensions of the latent representation."
        )

    def _ensure_diffusion(self, latent_shape: tuple[int, ...]) -> ContinuousGaussianDiffusion:
        if self.diffusion is None:
            self.diffusion = ContinuousGaussianDiffusion(device=self.device)
        return self.diffusion

    def _ensure_flow(self, latent_shape: tuple[int, ...]) -> RectifiedFlow:
        if self.flow is None:
            self.flow = RectifiedFlow(latent_shape=latent_shape, device=self.device)
        elif self.flow.latent_shape is None:
            self.flow.latent_shape = latent_shape
        return self.flow

    def generate(
        self,
        num_samples: int,
        latent_shape: tuple[int, ...] | None = None,
        guidance_scale: float = 1.0,
        condition=None,
        null_condition=None,
        num_steps: int | None = None,
        temperature: float = 1.0,
        initial_noise: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        model_kwargs = model_kwargs or {}
        latent_shape = latent_shape or self._infer_latent_shape()
        shape = (num_samples, *latent_shape)

        if seed is not None:
            torch.manual_seed(seed)

        with torch.inference_mode():
            if self.model_type == "diffusion":
                diffusion = self._ensure_diffusion(latent_shape)
                return diffusion.sample(
                    self.model,
                    shape,
                    y=condition,
                    guidance_scale=guidance_scale,
                    null_y=null_condition,
                    num_inference_steps=num_steps,
                    temperature=temperature,
                    model_kwargs=model_kwargs,
                    initial_noise=initial_noise,
                )

            flow = self._ensure_flow(latent_shape)
            return flow.sample(
                self.model,
                batch_size=num_samples,
                num_steps=num_steps or 100,
                y=condition,
                guidance_scale=guidance_scale,
                null_y=null_condition,
                initial_state=initial_noise,
                model_kwargs=model_kwargs,
            )

    def decode_latents(self, latents: torch.Tensor, **decode_kwargs) -> torch.Tensor:
        """Decode latents to reconstruction using tokenizer."""
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer required to decode latents. "
                "Pass a tokenizer to the constructor or use generate_volumes() instead."
            )

        if hasattr(self.tokenizer, "decode"):
            return torch.as_tensor(self.tokenizer.decode(latents, **decode_kwargs))

        raise AttributeError(
            f"Tokenizer {type(self.tokenizer).__name__} must have 'decode' method. "
            "Ensure you're using a ContinuousTokenizer or DiscreteTokenizer from medtokenizers."
        )

    def generate_volumes(
        self,
        num_samples: int,
        output_dir: str,
        latent_shape: tuple[int, ...] | None = None,
        guidance_scale: float = 1.0,
        condition=None,
        null_condition=None,
        num_steps: int | None = None,
        temperature: float = 1.0,
        initial_noise: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
        affine: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        latents = self.generate(
            num_samples=num_samples,
            latent_shape=latent_shape,
            guidance_scale=guidance_scale,
            condition=condition,
            null_condition=null_condition,
            num_steps=num_steps,
            temperature=temperature,
            initial_noise=initial_noise,
            model_kwargs=model_kwargs,
            seed=seed,
        )

        reconstruction = self.decode_latents(latents)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        affine_matrix = affine if affine is not None else np.eye(4)
        for i in range(num_samples):
            volume = reconstruction[i].cpu().numpy()
            nii = _get_nibabel().Nifti1Image(volume, affine_matrix)
            _get_nibabel().save(nii, output_path / f"sample_{i:04d}.nii.gz")
