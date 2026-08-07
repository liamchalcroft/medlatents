"""Training script for autoregressive transformer models."""

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medlatents.autoregressive import AutoregressiveTransformer  # noqa: E402
from training_utils import (  # noqa: E402
    MODEL_CONFIGS,
    Epoch,
    Metric,
    WandBID,
    create_optimizer,
    enable_cuda_optimizations,
    get_loaders,
    get_lr,
    get_wandb_init_kwargs,
)


def run_model(args, train_loader, val_loader):
    enable_cuda_optimizations(deterministic=args.deterministic)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        mixed_precision="fp16",
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Model setup
    config = MODEL_CONFIGS[args.model_size]
    model = AutoregressiveTransformer(
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        chunk_size=args.chunk_size,
        allow_dynamic_seq_length=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Resume from checkpoint if needed
    checkpoint = None
    start_epoch = 0
    metric_best = float("inf")

    if args.resume or args.resume_best:
        ckpt_name = "checkpoint.pt" if args.resume else "checkpoint_best.pt"
        ckpt_path = os.path.join(args.logdir, args.name, ckpt_name)
        ckpts = glob.glob(ckpt_path)
        if len(ckpts) == 0:
            args.resume = False
            args.resume_best = False
            print("\nNo checkpoints found. Beginning from epoch #0")
        else:
            checkpoint = torch.load(ckpts[0], map_location="cpu", weights_only=True)
            print(
                f"\nResuming from epoch #{checkpoint['epoch']} with WandB ID {checkpoint['wandb']}"
            )
            model.load_state_dict(checkpoint["net"])
            start_epoch = checkpoint["epoch"] + 1
            metric_best = checkpoint["metric"]

    # Initialize optimizer
    opt = create_optimizer(model, lr=args.lr)
    if checkpoint is not None:
        opt.load_state_dict(checkpoint["opt"])

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    # Prepare with accelerator
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    # Initialize wandb
    wandb_kwargs = get_wandb_init_kwargs(args, checkpoint)
    accelerator.init_trackers(
        "autoreg-transformer",
        config=vars(args),
        init_kwargs={"wandb": wandb_kwargs},
    )

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            global_step = epoch * len(train_loader) + step

            with accelerator.accumulate(model):
                # Prepare input and target for autoregressive training
                x = batch[:, :-1]
                y = batch[:, 1:]

                # Forward pass
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

                # Optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                # Update learning rate
                lr = get_lr(global_step, args, len(train_loader))
                for param_group in opt.param_groups:
                    param_group["lr"] = lr

                opt.step()
                opt.zero_grad(set_to_none=True)

            accelerator.log({"train/loss": loss.item(), "train/lr": lr})
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[:, :-1]
                    y = batch[:, 1:]
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
                    val_loss += loss.item()
                    val_steps += 1

            val_loss /= val_steps
            accelerator.log({"val/loss": val_loss})

            # Save checkpoints
            ckpt_data = {
                "net": model.state_dict(),
                "opt": opt.state_dict(),
                "wandb": WandBID(wandb.run.id).state_dict(),
                "epoch": Epoch(epoch).state_dict(),
                "metric": Metric(metric_best).state_dict(),
            }

            if val_loss < metric_best:
                metric_best = val_loss
                ckpt_data["metric"] = Metric(metric_best).state_dict()
                torch.save(
                    ckpt_data,
                    os.path.join(args.logdir, args.name, "checkpoint_best.pt"),
                )

            torch.save(
                ckpt_data,
                os.path.join(args.logdir, args.name, "checkpoint.pt"),
            )

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train autoregressive transformer")

    # Required arguments
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument(
        "--train_pattern", type=str, required=True, help="Glob pattern for training data"
    )
    parser.add_argument(
        "--val_pattern", type=str, required=True, help="Glob pattern for validation data"
    )
    parser.add_argument("--vocab_size", type=int, required=True, help="Size of vocabulary")
    parser.add_argument("--seq_length", type=int, required=True, help="Length of sequence")

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "small", "base", "large", "xl"],
        help="Model size configuration",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=None, help="Chunk size for sequence processing"
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
    parser.add_argument("--lr_warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # System parameters
    parser.add_argument("--logdir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")

    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--reset_wandb", action="store_true", help="Reset WandB run ID")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic ops")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing",
    )

    args = parser.parse_args()

    # Validation
    if args.min_lr > args.lr:
        raise ValueError("min_lr should be less than lr")
    if args.lr_warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs should be non-negative")

    train_loader, val_loader = get_loaders(
        args.train_pattern, args.val_pattern, args.batch_size, seed=args.seed
    )

    run_model(args, train_loader, val_loader)


if __name__ == "__main__":
    main()
