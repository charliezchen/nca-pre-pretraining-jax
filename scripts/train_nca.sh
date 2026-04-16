#!/usr/bin/env bash
# NCA pre-pretraining on 8 TPU chips (single host, data parallel).
set -euo pipefail


uv run python -m src.train_nca \
    --grid 12 \
    --patch 2 \
    --num_colors 10 \
    --seq_len 1024 \
    --batch_size 16 \
    --val_batch_size 128 \
    --hidden_size 768 \
    --intermediate_size 3072 \
    --num_layers 12 \
    --num_heads 6 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --steps 5000 \
    --train_num_rules 16000 \
    --val_num_rules 500 \
    --sims_per_refill 40960 \
    --save_dir checkpoints/nca \
    --wandb --wandb_name nca --wandb_project nca-pre-pretraining-jax \
    "$@"
