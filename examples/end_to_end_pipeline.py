"""End-to-end latent generation pipeline in one file.

Pre-train tokenizer -> tokenize dataset -> train generator on tokens ->
generate -> FID-192. This is the full loop described in the paper, at toy
scale so it runs on CPU in a couple of minutes.

    python examples/end_to_end_pipeline.py                  # random shapes
    python examples/end_to_end_pipeline.py --dataset chestmnist

The scale knobs (image count, epochs, model width) are the only things that
change between this file and the ChestMNIST factorial in the paper repository
(github.com/liamchalcroft/tokenizer-generator-coupling); the API calls are identical.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from medtokenizers import DiscreteTokenizer
from torch.utils.data import DataLoader, TensorDataset

from medlatents.diffusion import D3PM
from medlatents.evaluation import fid_from_features
from medlatents.evaluation.memorization import InceptionPool3FeatureExtractor
from medlatents.networks import DiscreteDiT

# Shared across tokenizer and generator: 64x64 images -> 8x8 tokens.
RESOLUTION = 64
SPATIAL_COMPRESSION = 8
SEQ_LEN = (RESOLUTION // SPATIAL_COMPRESSION) ** 2
VOCAB_SIZE = 1024
# One dictionary drives every backbone in the library (see medlatents.configs).
BACKBONE = {"hidden_size": 128, "depth": 4, "num_heads": 4}


def load_images(dataset: str, n: int) -> torch.Tensor:
    """(N, 1, 64, 64) float images in [0, 1]."""
    if dataset == "random":
        g = torch.Generator().manual_seed(0)
        base = torch.rand(n, 1, 8, 8, generator=g)
        return F.interpolate(base, size=RESOLUTION, mode="bilinear", align_corners=False)

    from medmnist import INFO
    from medmnist import __dict__ as medmnist_classes

    cls = medmnist_classes[INFO[dataset]["python_class"]]
    split = cls(split="train", download=True, size=RESOLUTION)
    imgs = torch.from_numpy(split.imgs[:n]).float() / 255.0
    return imgs.unsqueeze(1) if imgs.ndim == 3 else imgs.permute(0, 3, 1, 2)[:, :1]


def pretrain_tokenizer(
    images: torch.Tensor, epochs: int, device: torch.device
) -> DiscreteTokenizer:
    """Stage 1: reconstruction-train an LFQ tokenizer."""
    tokenizer = DiscreteTokenizer(
        dim=2,
        quantizer="LFQ",
        codebook_size=VOCAB_SIZE,
        codebook_dim=VOCAB_SIZE.bit_length() - 1,
        resolution=RESOLUTION,
        spatial_compression=SPATIAL_COMPRESSION,
        channels=32,
        channels_mult=(1, 2, 4),
        num_res_blocks=1,
    ).to(device)
    opt = torch.optim.AdamW(tokenizer.parameters(), lr=3e-4)
    loader = DataLoader(TensorDataset(images), batch_size=16, shuffle=True)

    tokenizer.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            out = tokenizer(batch)
            loss = F.mse_loss(out["reconstructions"], batch) + out["quant_loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  [tokenizer] epoch {epoch}: loss {total / len(loader):.4f}")
    return tokenizer.eval()


@torch.no_grad()
def tokenize(
    tokenizer: DiscreteTokenizer, images: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Stage 2: encode the dataset to a (N, SEQ_LEN) long tensor.

    This is the same ``tokens_path`` contract that ``scripts/tokenize_medmnist.py``
    writes to disk and that medtokenizers' ``tokenize_dataset.py`` produces.
    """
    chunks = []
    for start in range(0, len(images), 32):
        codes = tokenizer.tokenize(images[start : start + 32].to(device))
        chunks.append(codes.reshape(len(codes), -1).long().cpu())
    return torch.cat(chunks)


def train_generator(
    tokens: torch.Tensor, epochs: int, device: torch.device
) -> tuple[DiscreteDiT, D3PM]:
    """Stage 3: train a matrix-free absorbing D3PM over the tokens."""
    d3pm = D3PM(
        num_classes=VOCAB_SIZE,
        num_timesteps=1000,
        schedule_type="cosine",
        transition_type="absorbing",
        device=device,
    )
    model = DiscreteDiT(
        seq_length=SEQ_LEN,
        vocab_size=d3pm.effective_num_classes,
        **BACKBONE,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(TensorDataset(tokens), batch_size=16, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            loss = d3pm.compute_loss(model, batch.to(device), loss_type="hybrid")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  [generator] epoch {epoch}: loss {total / len(loader):.4f}")
    return model.eval(), d3pm


@torch.no_grad()
def generate(
    model: DiscreteDiT,
    d3pm: D3PM,
    tokenizer: DiscreteTokenizer,
    n: int,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Stage 4: sample tokens, then decode them back to images.

    Passing ``num_timesteps=num_steps`` builds a fresh N-step cosine absorbing
    schedule rather than respacing the 1000-step chain the model was trained on.
    """
    sampler = D3PM(
        num_classes=VOCAB_SIZE,
        num_timesteps=num_steps,
        schedule_type="cosine",
        transition_type="absorbing",
        device=device,
    )
    tokens = sampler.sample(model, shape=(n, SEQ_LEN)).clamp_max(VOCAB_SIZE - 1)
    grid = int(SEQ_LEN**0.5)
    return tokenizer.detokenize(tokens.reshape(n, grid, grid)).clamp(0, 1).cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="random", help="'random' or a MedMNIST 2D name")
    p.add_argument("--num_images", type=int, default=128)
    p.add_argument("--tokenizer_epochs", type=int, default=2)
    p.add_argument("--generator_epochs", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--sampling_steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)

    print("[1/5] loading images")
    images = load_images(args.dataset, args.num_images)

    print("[2/5] pre-training tokenizer")
    tokenizer = pretrain_tokenizer(images, args.tokenizer_epochs, device)

    print("[3/5] tokenizing dataset")
    tokens = tokenize(tokenizer, images, device)
    print(f"  tokens {tuple(tokens.shape)}, range [{tokens.min()}, {tokens.max()}]")

    print("[4/5] training generator")
    model, d3pm = train_generator(tokens, args.generator_epochs, device)
    samples = generate(model, d3pm, tokenizer, args.num_samples, args.sampling_steps, device)
    print(f"  generated {tuple(samples.shape)}")

    print("[5/5] FID-192")
    extractor = InceptionPool3FeatureExtractor(device=args.device)
    real = extractor.extract(images[: args.num_samples]).numpy()
    gen = extractor.extract(samples).numpy()
    print(f"  FID-192 = {fid_from_features(real, gen):.3f}")


if __name__ == "__main__":
    main()
