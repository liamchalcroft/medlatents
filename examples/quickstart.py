"""MedLatents quickstart: train and sample a tiny model in under a minute.

This example is intentionally self-contained and dependency-light:

* It runs on **CPU only** (no GPU required).
* It uses **random integer tokens** -- no medical data, no tokenizer.
* It finishes end-to-end in **well under a minute**.

It builds a ``nano`` autoregressive transformer, trains it for a handful of
steps with a next-token cross-entropy objective, then autoregressively samples a
batch of sequences and prints the final loss and the output shape.

Run it with::

    python examples/quickstart.py

For the real training/generation/evaluation workflows on tokenized medical
imagery, see the documentation and the ``scripts/`` directory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from medlatents import AutoregressiveTransformer
from medlatents.configs import MODEL_CONFIGS


def main() -> None:
    # Reproducibility and CPU-only execution.
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Tiny problem size so the whole script runs in a few seconds on CPU.
    seq_length = 64
    vocab_size = 128
    batch_size = 8
    num_steps = 20

    # Build a "nano" autoregressive transformer from the shared size presets.
    cfg = MODEL_CONFIGS["nano"]
    model = AutoregressiveTransformer(
        seq_length=seq_length,
        vocab_size=vocab_size,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Built nano AutoregressiveTransformer with {num_params:,} parameters.")

    # Synthetic training data: random base-vocabulary token sequences.
    data = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    loss = torch.tensor(float("nan"))
    for step in range(num_steps):
        optimizer.zero_grad()

        # Next-token prediction: predict token t+1 from tokens [0..t].
        logits = model(data[:, :-1])  # [B, L-1, vocab]
        targets = data[:, 1:]  # [B, L-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 5 == 0:
            print(f"  step {step + 1:2d}/{num_steps}  loss = {loss.item():.4f}")

    print(f"Final training loss: {loss.item():.4f}")

    # Autoregressive sampling from a single beginning-of-sequence-style prompt.
    model.eval()
    prompt = torch.zeros((4, 1), dtype=torch.long, device=device)
    samples = model.generate(prompt, max_length=seq_length, temperature=1.0, top_k=50)

    print(f"Generated samples shape: {tuple(samples.shape)}")  # (4, 64)
    assert samples.shape == (4, seq_length)
    print("Quickstart complete.")


if __name__ == "__main__":
    main()
