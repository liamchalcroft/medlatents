"""Benchmark MedMNIST generation throughput across model types."""

from __future__ import annotations

import argparse
import gc
import math
import time
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
    parser = argparse.ArgumentParser(description="Benchmark MedMNIST generation throughput")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--model_type", type=str, choices=ALL_MODEL_TYPES, default=None)
    parser.add_argument(
        "--token_type",
        type=str,
        choices=["discrete", "continuous"],
        default=None,
        help="Override token type (inferred from model_type when omitted)",
    )

    parser.add_argument("--num_warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num_samples", type=int, default=100, help="Total samples to benchmark")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_steps", type=int, default=None, help="Model-specific sampling steps")

    # Optional architecture/data overrides
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=None)
    parser.add_argument("--latent_channels", type=int, default=None)
    parser.add_argument("--latent_shape", type=int, nargs="+", default=None)

    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument(
        "--diffusion_schedule", type=str, default=None, choices=["linear", "cosine"]
    )
    parser.add_argument("--scheduler_type", type=str, default=None)
    parser.add_argument("--scheduler_power", type=float, default=None)
    parser.add_argument("--source_dist", type=str, default=None, choices=["uniform", "mask"])
    parser.add_argument("--time_eps", type=float, default=None)
    parser.add_argument(
        "--d3pm_transition", type=str, default=None, choices=["absorbing", "uniform", "gaussian"]
    )
    parser.add_argument("--d3pm_hybrid_coeff", type=float, default=None)
    parser.add_argument("--bfn_beta", type=float, default=None)
    parser.add_argument("--bfn_num_steps", type=int, default=None)

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_value(cli_value: Any, ckpt_args: Mapping[str, Any], key: str, default: Any) -> Any:
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
    raise ValueError("Could not infer model_type from checkpoint; pass --model_type")


def _resolve_token_type(args: argparse.Namespace, model_type: str) -> str:
    if args.token_type is not None:
        return args.token_type
    return "continuous" if model_type in CONTINUOUS_MODEL_TYPES else "discrete"


def _load_checkpoint(path: str, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "net" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format in {path}; expected dict with 'net'")
    ckpt_args = checkpoint.get("args", {})
    return checkpoint, ckpt_args if isinstance(ckpt_args, dict) else {}


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

    cfg = {
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
    return model, cfg


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

    cfg = {
        "seq_len": seq_len,
        "channels": latent_channels,
        "diffusion_steps": int(
            _resolve_value(args.diffusion_steps, ckpt_args, "diffusion_steps", 1000)
        ),
        "diffusion_schedule": str(
            _resolve_value(args.diffusion_schedule, ckpt_args, "diffusion_schedule", "cosine")
        ),
    }
    return model, cfg


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
def _generate_continuous_batch(
    model: ContinuousDiT,
    batch_size: int,
    temperature: float,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    diffusion = ContinuousGaussianDiffusion(
        num_timesteps=int(cfg["diffusion_steps"]),
        schedule_type=str(cfg["diffusion_schedule"]),
        device=device,
    )
    return diffusion.sample(
        model,
        shape=(batch_size, int(cfg["seq_len"]), int(cfg["channels"])),
        num_inference_steps=int(cfg["diffusion_steps"]),
        temperature=temperature,
    )


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _memory_usage(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    _cuda_sync(device)
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


def _summarize_latencies(latencies: list[float], total_samples: int) -> dict[str, float]:
    total_time = float(sum(latencies))
    throughput = total_samples / total_time if total_time > 0 else 0.0
    return {
        "total_time_sec": total_time,
        "samples_per_second": throughput,
        "avg_latency_ms": (total_time / total_samples) * 1000 if total_samples > 0 else 0.0,
        "min_batch_time_ms": min(latencies) * 1000 if latencies else 0.0,
        "max_batch_time_ms": max(latencies) * 1000 if latencies else 0.0,
        "std_batch_time_ms": float(np.std(latencies) * 1000) if latencies else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    checkpoint, ckpt_args = _load_checkpoint(args.checkpoint, device)
    model_type = _resolve_model_type(args, ckpt_args)
    token_type = _resolve_token_type(args, model_type)

    if token_type == "discrete" and model_type not in DISCRETE_MODEL_TYPES:
        raise ValueError(f"Model type {model_type} is not discrete")
    if token_type == "continuous" and model_type not in CONTINUOUS_MODEL_TYPES:
        raise ValueError(f"Model type {model_type} is not continuous")

    print("=" * 60)
    print("GENERATION BENCHMARK")
    print("=" * 60)
    print(f"Device:      {device}")
    print(f"Token type:  {token_type}")
    print(f"Model type:  {model_type}")
    print(f"Samples:     {args.num_samples}")
    print(f"Batch size:  {args.batch_size}")

    if token_type == "discrete":
        model, cfg = _build_discrete_model(model_type, ckpt_args, checkpoint, args, device)
        default_steps = int(cfg["bfn_num_steps"]) if model_type == "bayesian_flow" else 12
        num_steps = int(args.num_steps) if args.num_steps is not None else default_steps
        num_steps = max(1, num_steps)

        print("Warmup...")
        for _ in range(args.num_warmup):
            _ = _generate_discrete_batch(
                model,
                model_type=model_type,
                batch_size=min(args.batch_size, args.num_samples),
                temperature=args.temperature,
                num_steps=num_steps,
                cfg=cfg,
                device=device,
            )
        _cuda_sync(device)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = _memory_usage(device)
        latencies: list[float] = []
        generated = 0
        for _ in tqdm(range(0, args.num_samples, args.batch_size), desc="Benchmarking"):
            batch_size = min(args.batch_size, args.num_samples - generated)
            start = time.perf_counter()
            _ = _generate_discrete_batch(
                model,
                model_type=model_type,
                batch_size=batch_size,
                temperature=args.temperature,
                num_steps=num_steps,
                cfg=cfg,
                device=device,
            )
            _cuda_sync(device)
            latencies.append(time.perf_counter() - start)
            generated += batch_size

    else:
        model, cfg = _build_continuous_model(ckpt_args, checkpoint, args, device)

        print("Warmup...")
        for _ in range(args.num_warmup):
            _ = _generate_continuous_batch(
                model,
                batch_size=min(args.batch_size, args.num_samples),
                temperature=args.temperature,
                cfg=cfg,
                device=device,
            )
        _cuda_sync(device)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        mem_before = _memory_usage(device)
        latencies = []
        generated = 0
        for _ in tqdm(range(0, args.num_samples, args.batch_size), desc="Benchmarking"):
            batch_size = min(args.batch_size, args.num_samples - generated)
            start = time.perf_counter()
            _ = _generate_continuous_batch(
                model,
                batch_size=batch_size,
                temperature=args.temperature,
                cfg=cfg,
                device=device,
            )
            _cuda_sync(device)
            latencies.append(time.perf_counter() - start)
            generated += batch_size

    mem_after = _memory_usage(device)
    metrics = _summarize_latencies(latencies, total_samples=args.num_samples)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total time:      {metrics['total_time_sec']:.3f}s")
    print(f"Throughput:      {metrics['samples_per_second']:.2f} samples/sec")
    print(f"Avg latency:     {metrics['avg_latency_ms']:.2f} ms/sample")
    print(f"Min batch time:  {metrics['min_batch_time_ms']:.2f} ms")
    print(f"Max batch time:  {metrics['max_batch_time_ms']:.2f} ms")
    print(f"Std batch time:  {metrics['std_batch_time_ms']:.2f} ms")
    print(f"\nMemory before:   {mem_before['allocated_gb']:.3f} GB allocated")
    print(f"Memory after:    {mem_after['allocated_gb']:.3f} GB allocated")
    print(f"Peak memory:     {mem_after['max_allocated_gb']:.3f} GB allocated")


if __name__ == "__main__":
    main()
