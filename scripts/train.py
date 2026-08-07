"""Unified training script for all discrete latent generative models."""

import argparse

import torch

from medlatents.autoregressive import AutoregressiveTransformer
from medlatents.bayesian_flow import BayesianFlowTransformer
from medlatents.configs import MODEL_CONFIGS
from medlatents.data import get_tokenized_dataloaders
from medlatents.maskgit import MaskGIT
from medlatents.networks import DiscreteDiT
from medlatents.training import compile_model, enable_cuda_optimizations
from medlatents.training.discrete import DiscreteLatentTrainer


def get_optimal_mixed_precision() -> str:
    """Auto-detect optimal mixed precision setting for current GPU.

    Returns:
        Mixed precision mode: 'bf16', 'fp16', or 'no'
    """
    if not torch.cuda.is_available():
        return "no"

    # Check compute capability for bf16 support (Ampere+ = 8.0+)
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:
        return "bf16"  # A100, A6000, RTX 30xx/40xx
    elif capability[0] >= 7:
        return "fp16"  # V100, RTX 20xx
    else:
        return "no"


def load_tokenizer(tokenizer_path: str):
    try:
        from medtokenizers.networks.discrete import DiscreteTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "medtokenizers is required for tokenized training. "
            "Install it to load discrete tokenizer weights."
        ) from exc

    weights = torch.load(tokenizer_path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or "hparams" not in weights or "net" not in weights:
        raise ValueError(
            "Tokenizer checkpoint must contain 'hparams' and 'net' keys. "
            f"Got keys: {sorted(weights.keys()) if isinstance(weights, dict) else type(weights)}"
        )

    tokenizer = DiscreteTokenizer(weights["hparams"]).to("cpu")
    tokenizer.load_state_dict(weights["net"])
    tokenizer.eval()
    return tokenizer


def create_model(args):
    """Create model based on type and size."""
    config = MODEL_CONFIGS[args.model_size]

    if args.model_type == "autoreg":
        return AutoregressiveTransformer(
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            chunk_size=args.chunk_size,
            allow_dynamic_seq_length=True,
        )
    elif args.model_type == "maskgit":
        return MaskGIT(
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            chunk_size=args.chunk_size,
            allow_dynamic_seq_length=True,
        )
    elif args.model_type == "flow":
        return DiscreteDiT(
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            chunk_size=args.chunk_size,
            allow_dynamic_seq_length=True,
            masked=False,
        )
    elif args.model_type == "d3pm":
        return DiscreteDiT(
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            chunk_size=args.chunk_size,
            allow_dynamic_seq_length=True,
            masked=True,
        )
    elif args.model_type == "bayesian_flow":
        return BayesianFlowTransformer(
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            chunk_size=args.chunk_size,
            allow_dynamic_seq_length=True,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train discrete latent generative models")

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["autoreg", "maskgit", "flow", "d3pm", "bayesian_flow"],
        help="Type of generative model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "small", "base", "large", "xl"],
        help="Model size configuration",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for sequence processing",
    )

    # Data configuration
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument(
        "--train_pattern",
        type=str,
        required=True,
        help="Glob pattern for training data",
    )
    parser.add_argument(
        "--val_pattern",
        type=str,
        required=True,
        help="Glob pattern for validation data",
    )
    parser.add_argument("--vocab_size", type=int, required=True, help="Size of vocabulary")
    parser.add_argument("--seq_length", type=int, required=True, help="Length of sequence")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to tokenizer weights (.pt)",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Optimization parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for learning rate warmup",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # MaskGIT specific arguments
    parser.add_argument(
        "--mask_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "square"],
        help="Schedule for mask ratio (maskgit only)",
    )

    # Flow matching parameters
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="polynomial",
        help="Flow scheduler type (flow only)",
    )
    parser.add_argument(
        "--scheduler_power",
        type=float,
        default=2.0,
        help="Flow scheduler power (flow only)",
    )
    parser.add_argument(
        "--source_dist",
        type=str,
        default="uniform",
        choices=["uniform", "mask"],
        help="Source distribution (flow only)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "generalized_kl"],
        help="Loss function (flow only)",
    )
    parser.add_argument("--time_eps", type=float, default=1e-3, help="Time epsilon (flow only)")

    # System parameters
    parser.add_argument("--logdir", type=str, default="./checkpoints")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic ops (disables cuDNN benchmark and TF32).",
    )

    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume from best checkpoint")
    parser.add_argument(
        "--reset_wandb", action="store_true", help="Reset WandB run ID when resuming"
    )

    # Performance flags
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "no"],
        help="Mixed precision mode. 'auto' detects optimal setting for GPU.",
    )
    parser.add_argument(
        "--fused_optimizer",
        action="store_true",
        default=True,
        help="Use fused AdamW optimizer (faster on CUDA)",
    )
    parser.add_argument(
        "--no_fused_optimizer",
        action="store_false",
        dest="fused_optimizer",
        help="Disable fused AdamW optimizer",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.min_lr > args.lr:
        raise ValueError("min_lr should be less than lr")
    if args.lr_warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs should be non-negative")

    # Auto-detect mixed precision if set to auto
    if args.mixed_precision == "auto":
        args.mixed_precision = get_optimal_mixed_precision()
        print(f"Auto-detected mixed precision: {args.mixed_precision}")

    # Enable CUDA optimizations if available
    cuda_settings = enable_cuda_optimizations(
        benchmark=not args.deterministic,
        tf32=not args.deterministic,
        deterministic=args.deterministic,
    )
    if cuda_settings["cuda_available"]:
        print(
            f"CUDA optimizations: TF32={cuda_settings['tf32_matmul']}, benchmark={cuda_settings['benchmark']}"
        )
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Get dataloaders
    tokenizer = load_tokenizer(args.tokenizer_path)
    train_loader, val_loader = get_tokenized_dataloaders(
        args.train_pattern,
        args.val_pattern,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Create model
    model = create_model(args)

    # Optionally compile model for faster training
    if args.compile:
        print(f"Compiling model with mode={args.compile_mode}...")
        model = compile_model(model, mode=args.compile_mode)

    # Store hparams for checkpoint saving
    args.hparams = {
        "model_size": args.model_size,
        "chunk_size": args.chunk_size,
        "seq_length": args.seq_length,
        "vocab_size": args.vocab_size,
        "allow_dynamic_seq_length": True,
    }
    if args.model_type in {"flow", "d3pm"}:
        args.hparams["masked"] = args.model_type == "d3pm"

    # Create trainer and train
    trainer = DiscreteLatentTrainer(
        model_type=args.model_type,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
    )

    trainer.train(resume=args.resume, resume_best=args.resume_best, reset_wandb=args.reset_wandb)


if __name__ == "__main__":
    main()
