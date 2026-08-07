#!/bin/bash
# Train 2D generation model on MedMNIST tokenized data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python examples/train_medmnist2d_tokens.py \
    --tokens_path "$1" \
    --token_type "${2:-discrete}" \
    --model "${3:-transformer}" \
    --vocab_size "${4:-512}" \
    --seq_len "${5:-64}" \
    --epochs "${6:-50}" \
    --batch_size "${7:-32}" \
    --lr "${8:-1e-4}" \
    --hidden_size "${9:-512}" \
    --depth "${10:-8}" \
    --num_heads "${11:-8}" \
    --use_ema \
    --use_wandb \
    --val_interval 5 \
    --generate_samples_every 10 \
    --logdir checkpoints \
    --name "${12:-medmnist2d}"
