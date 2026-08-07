"""Generate samples from trained MedMNIST checkpoints."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from medlatents.autoregressive import AutoregressiveTransformer
from medlatents.bayesian_flow import BayesianFlowTransformer
from medlatents.diffusion.continuous import ContinuousGaussianDiffusion
from medlatents.diffusion.d3pm import D3PM
from medlatents.flow_matching.core import MixtureDiscreteEulerSolver, ModelWrapper
from medlatents.flow_matching.discrete import get_path, get_source_distribution
from medlatents.maskgit import MaskGIT
from medlatents.networks import ContinuousDiT, DiscreteDiT
from medlatents.utils import load_state_dict_compat

DISCRETE_MODEL_TYPES = {"transformer", "maskgit", "flow", "d3pm", "bayesian_flow"}
CONTINUOUS_MODEL_TYPES = {"diffusion"}
ALL_MODEL_TYPES = sorted(DISCRETE_MODEL_TYPES | CONTINUOUS_MODEL_TYPES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples from trained MedMNIST checkpoints"
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--token_type",
        type=str,
        choices=["discrete", "continuous"],
        default=None,
        help="Override token type (inferred from model_type when omitted)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=ALL_MODEL_TYPES,
        default=None,
        help="Model architecture (inferred from checkpoint when omitted)",
    )

    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Sampling batch size")
    parser.add_argument("--output", type=str, required=True, help="Output NPZ file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")

    # Optional overrides. When omitted, values are loaded from checkpoint args if available.
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=None)
    parser.add_argument("--spatial_shape", type=int, nargs="+", default=None)
    parser.add_argument("--latent_channels", type=int, default=None)
    parser.add_argument("--latent_shape", type=int, nargs="+", default=None)

    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_steps", type=int, default=None, help="Model-specific sampling steps")
    parser.add_argument(
        "--diffusion_steps", type=int, default=None, help="Diffusion timestep count"
    )
    parser.add_argument(
        "--diffusion_schedule", type=str, default=None, choices=["linear", "cosine"]
    )

    parser.add_argument("--scheduler_type", type=str, default=None, help="Flow scheduler type")
    parser.add_argument("--scheduler_power", type=float, default=None, help="Flow scheduler power")
    parser.add_argument("--source_dist", type=str, default=None, choices=["uniform", "mask"])
    parser.add_argument("--time_eps", type=float, default=None, help="Flow max-time epsilon")
    parser.add_argument(
        "--d3pm_transition", type=str, default=None, choices=["absorbing", "uniform", "gaussian"]
    )
    parser.add_argument("--d3pm_hybrid_coeff", type=float, default=None)
    parser.add_argument("--bfn_beta", type=float, default=None)
    parser.add_argument("--bfn_num_steps", type=int, default=None)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _resolve_value(
    cli_value: Any,
    ckpt_args: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    if key in ckpt_args and ckpt_args[key] is not None:
        return ckpt_args[key]
    return default


def _resolve_model_type(args: argparse.Namespace, ckpt_args: Mapping[str, Any]) -> str:
    if args.model_type is not None:
        return args.model_type
    for key in ("model_type", "model"):
        value = ckpt_args.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("Could not infer model_type from checkpoint; pass --model_type explicitly")


def _resolve_token_type(args: argparse.Namespace, model_type: str) -> str:
    if args.token_type is not None:
        return args.token_type
    return "continuous" if model_type in CONTINUOUS_MODEL_TYPES else "discrete"


def _load_checkpoint(path: str, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format in {path}; expected dict with 'net' key")

    raw_args = checkpoint.get("args", {})
    ckpt_args = raw_args if isinstance(raw_args, dict) else {}
    return checkpoint, ckpt_args


def _infer_default_spatial_shape(seq_len: int) -> list[int]:
    side = int(seq_len**0.5)
    if side * side == seq_len:
        return [side, side]
    return [seq_len]


def _reshape_discrete_samples(
    samples: torch.Tensor, spatial_shape: list[int] | None
) -> torch.Tensor:
    if spatial_shape is None:
        return samples
    if len(spatial_shape) == 1:
        seq_len = spatial_shape[0]
        if seq_len != samples.shape[1]:
            raise ValueError(
                f"spatial_shape {spatial_shape} is incompatible with seq_len={samples.shape[1]}"
            )
        return samples
    if len(spatial_shape) == 2:
        h, w = spatial_shape
        if h * w != samples.shape[1]:
            raise ValueError(
                f"spatial_shape {spatial_shape} is incompatible with seq_len={samples.shape[1]}"
            )
        return samples.view(samples.shape[0], h, w)
    if len(spatial_shape) == 3:
        h, w, d = spatial_shape
        if h * w * d != samples.shape[1]:
            raise ValueError(
                f"spatial_shape {spatial_shape} is incompatible with seq_len={samples.shape[1]}"
            )
        return samples.view(samples.shape[0], h, w, d)
    raise ValueError(f"spatial_shape must have length 1, 2, or 3, got {spatial_shape}")


def _reshape_continuous_samples(
    samples: torch.Tensor, latent_shape: list[int] | None
) -> torch.Tensor:
    if latent_shape is None:
        return samples
    if len(latent_shape) not in (3, 4):
        raise ValueError(f"latent_shape must be [C H W] or [C H W D], got {latent_shape}")

    channels = latent_shape[0]
    spatial = latent_shape[1:]
    seq_len = math.prod(spatial)
    if samples.shape[1] != seq_len or samples.shape[2] != channels:
        raise ValueError(
            f"latent_shape {latent_shape} is incompatible with sampled shape {tuple(samples.shape)}"
        )

    if len(spatial) == 2:
        h, w = spatial
        return samples.view(samples.shape[0], h, w, channels).permute(0, 3, 1, 2).contiguous()

    h, w, d = spatial
    return samples.view(samples.shape[0], h, w, d, channels).permute(0, 4, 1, 2, 3).contiguous()


def _build_discrete_model(
    model_type: str,
    ckpt_args: Mapping[str, Any],
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    seq_len = int(_resolve_value(args.seq_len, ckpt_args, "seq_len", 64))
    vocab_size = int(_resolve_value(args.vocab_size, ckpt_args, "vocab_size", 512))
    hidden_size = int(_resolve_value(args.hidden_size, ckpt_args, "hidden_size", 512))
    depth = int(_resolve_value(args.depth, ckpt_args, "depth", 8))
    num_heads = int(_resolve_value(args.num_heads, ckpt_args, "num_heads", 8))
    mlp_ratio = float(_resolve_value(args.mlp_ratio, ckpt_args, "mlp_ratio", 4.0))
    source_dist_name = str(_resolve_value(args.source_dist, ckpt_args, "source_dist", "uniform"))
    d3pm_transition = str(
        _resolve_value(args.d3pm_transition, ckpt_args, "d3pm_transition", "absorbing")
    )
    bfn_num_steps = int(_resolve_value(args.bfn_num_steps, ckpt_args, "bfn_num_steps", 1000))
    bfn_beta = float(_resolve_value(args.bfn_beta, ckpt_args, "bfn_beta", 1.0))

    if model_type == "transformer":
        model = AutoregressiveTransformer(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gradient_checkpointing=False,
        )
    elif model_type == "maskgit":
        model = MaskGIT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gradient_checkpointing=False,
        )
    elif model_type == "flow":
        model = DiscreteDiT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            masked=source_dist_name == "mask",
            gradient_checkpointing=False,
            allow_dynamic_seq_length=True,
        )
    elif model_type == "d3pm":
        model = DiscreteDiT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            masked=d3pm_transition == "absorbing",
            gradient_checkpointing=False,
            allow_dynamic_seq_length=True,
        )
    elif model_type == "bayesian_flow":
        model = BayesianFlowTransformer(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_steps=bfn_num_steps,
            beta=bfn_beta,
            gradient_checkpointing=False,
        )
    else:
        raise ValueError(f"Unsupported discrete model_type: {model_type}")

    load_state_dict_compat(model, checkpoint["net"])
    model.to(device)
    model.eval()

    sampling_cfg = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "source_dist": source_dist_name,
        "d3pm_transition": d3pm_transition,
        "d3pm_hybrid_coeff": float(
            _resolve_value(args.d3pm_hybrid_coeff, ckpt_args, "d3pm_hybrid_coeff", 0.001)
        ),
        "scheduler_type": str(
            _resolve_value(args.scheduler_type, ckpt_args, "scheduler_type", "polynomial")
        ),
        "scheduler_power": float(
            _resolve_value(args.scheduler_power, ckpt_args, "scheduler_power", 2.0)
        ),
        "time_eps": float(_resolve_value(args.time_eps, ckpt_args, "time_eps", 1e-3)),
        "diffusion_steps": int(
            _resolve_value(args.diffusion_steps, ckpt_args, "diffusion_steps", 1000)
        ),
        "diffusion_schedule": str(
            _resolve_value(args.diffusion_schedule, ckpt_args, "diffusion_schedule", "cosine")
        ),
        "bfn_num_steps": bfn_num_steps,
    }
    return model, sampling_cfg


def _build_continuous_model(
    ckpt_args: Mapping[str, Any],
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ContinuousDiT, dict[str, Any]]:
    latent_shape = _resolve_value(args.latent_shape, ckpt_args, "latent_shape", [4, 8, 8])
    if not isinstance(latent_shape, (list, tuple)) or len(latent_shape) not in (3, 4):
        raise ValueError(f"latent_shape must be [C H W] or [C H W D], got {latent_shape}")
    latent_shape = [int(x) for x in latent_shape]

    latent_channels = int(
        _resolve_value(args.latent_channels, ckpt_args, "latent_channels", latent_shape[0])
    )
    if latent_channels != latent_shape[0]:
        latent_shape[0] = latent_channels

    seq_len = math.prod(latent_shape[1:])
    hidden_size = int(_resolve_value(args.hidden_size, ckpt_args, "hidden_size", 512))
    depth = int(_resolve_value(args.depth, ckpt_args, "depth", 8))
    num_heads = int(_resolve_value(args.num_heads, ckpt_args, "num_heads", 8))
    mlp_ratio = float(_resolve_value(args.mlp_ratio, ckpt_args, "mlp_ratio", 4.0))

    model = ContinuousDiT(
        seq_length=seq_len,
        in_channels=latent_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        gradient_checkpointing=False,
        allow_dynamic_seq_length=True,
    )
    load_state_dict_compat(model, checkpoint["net"])
    model.to(device)
    model.eval()

    sampling_cfg = {
        "latent_shape": latent_shape,
        "diffusion_steps": int(
            _resolve_value(args.diffusion_steps, ckpt_args, "diffusion_steps", 1000)
        ),
        "diffusion_schedule": str(
            _resolve_value(args.diffusion_schedule, ckpt_args, "diffusion_schedule", "cosine")
        ),
        "latent_mean": float(ckpt_args.get("latent_mean", 0.0)),
        "latent_std": float(ckpt_args.get("latent_std", 1.0)),
    }
    return model, sampling_cfg


@torch.inference_mode()
def _generate_discrete_batch(
    model: torch.nn.Module,
    model_type: str,
    batch_size: int,
    temperature: float,
    num_steps: int,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    seq_len = int(cfg["seq_len"])
    vocab_size = int(cfg["vocab_size"])

    if model_type == "transformer":
        bos_token = (
            model.special_tokens.bos if getattr(model, "special_tokens", None) is not None else 0
        )
        prompt = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)
        return model.generate(prompt, max_length=seq_len, temperature=temperature)

    if model_type == "maskgit":
        x = torch.full((batch_size, seq_len), model.mask_token, dtype=torch.long, device=device)
        return model.generate(x, num_steps=num_steps, temperature=temperature)

    if model_type == "flow":
        path = get_path(str(cfg["scheduler_type"]), float(cfg["scheduler_power"]))
        source_dist = get_source_distribution(str(cfg["source_dist"]), vocab_size)

        class WrappedModel(ModelWrapper):
            @torch.no_grad()
            def forward(self, x, t, **extras):
                return torch.softmax(self.model(x=x, t=t, **extras).float(), dim=-1)

        wrapped_model = WrappedModel(model=model)
        add_token = 1 if getattr(source_dist, "masked", False) else 0
        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model,
            path=path,
            vocabulary_size=vocab_size + add_token,
        )
        x_init = source_dist.sample((batch_size, seq_len), device=device)
        return solver.sample(
            x_init=x_init,
            step_size=1 / 100,
            verbose=False,
            time_grid=torch.tensor([0.0, 1.0 - float(cfg["time_eps"])], device=device),
        )

    if model_type == "d3pm":
        d3pm = D3PM(
            num_classes=vocab_size,
            num_timesteps=int(cfg["diffusion_steps"]),
            schedule_type=str(cfg["diffusion_schedule"]),
            transition_type=str(cfg["d3pm_transition"]),
            hybrid_loss_coeff=float(cfg["d3pm_hybrid_coeff"]),
            device=device,
        )
        return d3pm.sample(model, (batch_size, seq_len), temperature=temperature)

    if model_type == "bayesian_flow":
        return model.sample((batch_size, seq_len), num_steps=num_steps, temperature=temperature)

    raise ValueError(f"Unsupported discrete model_type: {model_type}")


@torch.inference_mode()
def generate_discrete_samples(
    model: torch.nn.Module,
    model_type: str,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    default_steps = int(cfg["bfn_num_steps"]) if model_type == "bayesian_flow" else 12
    num_steps = int(args.num_steps) if args.num_steps is not None else default_steps
    num_steps = max(1, num_steps)

    all_samples = []
    for start in tqdm(
        range(0, args.num_samples, args.batch_size), desc="Generating discrete samples"
    ):
        batch_size = min(args.batch_size, args.num_samples - start)
        batch = _generate_discrete_batch(
            model,
            model_type=model_type,
            batch_size=batch_size,
            temperature=args.temperature,
            num_steps=num_steps,
            cfg=cfg,
            device=device,
        )
        all_samples.append(batch.cpu())

    samples = torch.cat(all_samples, dim=0)
    spatial_shape = args.spatial_shape
    if spatial_shape is None:
        spatial_shape = _infer_default_spatial_shape(int(cfg["seq_len"]))
    return _reshape_discrete_samples(samples, [int(x) for x in spatial_shape])


@torch.inference_mode()
def generate_continuous_samples(
    model: ContinuousDiT,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    latent_shape = [int(x) for x in cfg["latent_shape"]]
    seq_len = math.prod(latent_shape[1:])
    channels = latent_shape[0]
    diffusion_steps = int(cfg["diffusion_steps"])
    diffusion_schedule = str(cfg["diffusion_schedule"])

    diffusion = ContinuousGaussianDiffusion(
        num_timesteps=diffusion_steps,
        schedule_type=diffusion_schedule,
        device=device,
    )

    all_samples = []
    for start in tqdm(
        range(0, args.num_samples, args.batch_size), desc="Generating continuous samples"
    ):
        batch_size = min(args.batch_size, args.num_samples - start)
        batch = diffusion.sample(
            model,
            shape=(batch_size, seq_len, channels),
            num_inference_steps=diffusion_steps,
            temperature=args.temperature,
        )
        all_samples.append(batch.cpu())

    samples = torch.cat(all_samples, dim=0)

    # Un-normalize if model was trained with normalized latents
    latent_mean = float(cfg.get("latent_mean", 0.0))
    latent_std = float(cfg.get("latent_std", 1.0))
    if latent_std != 1.0 or latent_mean != 0.0:
        samples = samples * latent_std + latent_mean

    return _reshape_continuous_samples(samples, latent_shape)


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"Output file exists: {args.output}. Pass --overwrite to replace it.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    checkpoint, ckpt_args = _load_checkpoint(args.checkpoint, device)
    model_type = _resolve_model_type(args, ckpt_args)
    token_type = _resolve_token_type(args, model_type)

    if token_type == "discrete" and model_type not in DISCRETE_MODEL_TYPES:
        raise ValueError(f"Model type {model_type} is not discrete")
    if token_type == "continuous" and model_type not in CONTINUOUS_MODEL_TYPES:
        raise ValueError(f"Model type {model_type} is not continuous")

    if token_type == "discrete":
        model, cfg = _build_discrete_model(model_type, ckpt_args, checkpoint, args, device)
        samples = generate_discrete_samples(model, model_type, cfg, args, device)
        np.savez_compressed(
            args.output,
            generated_tokens=samples.numpy(),
            metadata={
                "token_type": token_type,
                "model_type": model_type,
                "num_samples": args.num_samples,
                "seq_len": int(cfg["seq_len"]),
                "vocab_size": int(cfg["vocab_size"]),
                "spatial_shape": list(samples.shape[1:]),
            },
        )
    else:
        model, cfg = _build_continuous_model(ckpt_args, checkpoint, args, device)
        samples = generate_continuous_samples(model, cfg, args, device)
        np.savez_compressed(
            args.output,
            generated_latents=samples.numpy(),
            metadata={
                "token_type": token_type,
                "model_type": model_type,
                "num_samples": args.num_samples,
                "latent_shape": list(samples.shape[1:]),
            },
        )

    print(f"Generated {args.num_samples} samples -> {args.output}")
    print(f"Output shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
