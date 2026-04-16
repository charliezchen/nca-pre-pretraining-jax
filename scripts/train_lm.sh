#!/usr/bin/env bash
# Language pretraining on 8 TPU chips, optionally warm-starting from an NCA ckpt.
set -euo pipefail

uv run python -m src.train_lm \
    --dataset Skylion007/openwebtext \
    --tokenizer_name gpt2 \
    --seq_len 1024 \
    --batch_size 512 \
    --vocab_size 50304 \
    --hidden_size 768 \
    --intermediate_size 3072 \
    --num_layers 12 \
    --num_heads 6 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --warmup_steps 1000 \
    --steps 3000 \
    --save_dir checkpoints/lm \
    --wandb --wandb_name lm --wandb_project nca-pre-pretraining-jax \
    "$@"
