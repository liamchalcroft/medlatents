#!/bin/bash
# Train 3D generation model on MedMNIST3D tokenized data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python examples/train_medmnist3d_tokens.py \
    --tokens_path "$1" \
    --token_type "${2:-discrete}" \
    --model "${3:-transformer}" \
    --vocab_size "${4:-512}" \
    --seq_len "${5:-512}" \
    --spatial_shape "${6:-8 8 8}" \
    --epochs "${7:-100}" \
    --batch_size "${8:-8}" \
    --lr "${9:-1e-4}" \
    --hidden_size "${10:-512}" \
    --depth "${11:-8}" \
    --num_heads "${12:-8}" \
    --use_ema \
    --use_wandb \
    --val_interval 5 \
    --generate_samples_every 10 \
    --logdir checkpoints \
    --name "${13:-medmnist3d}"
