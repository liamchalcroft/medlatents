#!/bin/bash
# Generate samples from trained MedMNIST model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

python examples/generate_medmnist.py \
    --checkpoint "$1" \
    --token_type "${2:-discrete}" \
    --model_type "${3:-transformer}" \
    --num_samples "${4:-100}" \
    --batch_size "${5:-16}" \
    --output "$6" \
    --vocab_size "${7:-512}" \
    --seq_len "${8:-64}" \
    --spatial_shape "${9:-8 8}" \
    --temperature "${10:-0.9}" \
    --device "${11:-cuda}" \
    --overwrite
