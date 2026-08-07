"""Training script for Bayesian Flow Networks."""

import argparse
import glob
import os
import sys

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medlatents.bayesian_flow import BayesianFlowTransformer  # noqa: E402
from training_utils import (  # noqa: E402
    MODEL_CONFIGS,
    Epoch,
    Metric,
    WandBID,
    get_loaders,
    get_lr,
    get_wandb_init_kwargs,
)


def run_model(args, train_loader, val_loader):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
        mixed_precision="fp16",
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Model setup
    config = MODEL_CONFIGS[args.model_size]
    model = BayesianFlowTransformer(
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        num_steps=args.num_steps,
        beta=args.beta,
        gradient_checkpointing=True,
        allow_dynamic_seq_length=True,
    )

    model_kwargs = {
        "model_size": args.model_size,
        "seq_length": args.seq_length,
        "vocab_size": args.vocab_size,
        "num_steps": args.num_steps,
        "beta": args.beta,
        "allow_dynamic_seq_length": True,
    }

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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if checkpoint is not None:
        opt.load_state_dict(checkpoint["opt"])

    # Prepare with accelerator
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    # Initialize wandb
    wandb_kwargs = get_wandb_init_kwargs(args, checkpoint)
    accelerator.init_trackers(
        "bayesian-flow-networks",
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
                # Compute BFN loss
                loss = model.compute_loss(batch, loss_type=args.loss_type)

                # Optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

                # Update learning rate
                lr = get_lr(global_step, args, len(train_loader))
                for param_group in opt.param_groups:
                    param_group["lr"] = lr

                opt.step()
                opt.zero_grad()

            accelerator.log({"train/loss": loss.item(), "train/lr": lr})
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.6f}"})

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = model.compute_loss(batch, loss_type=args.loss_type)
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
                "hparams": model_kwargs,
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
    parser = argparse.ArgumentParser(description="Train Bayesian Flow Network")

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

    # BFN specific parameters
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of flow steps")
    parser.add_argument("--beta", type=float, default=1.0, help="BFN beta parameter")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
        help="Loss type for training",
    )

    # System parameters
    parser.add_argument("--logdir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")

    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume from best checkpoint")
    parser.add_argument("--reset_wandb", action="store_true", help="Reset WandB run ID")

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
