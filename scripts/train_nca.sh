#!/usr/bin/env bash
# NCA pre-pretraining on 8 TPU chips (single host, data parallel).
set -euo pipefail

export XLA_FLAGS="--xla_tpu_enable_latency_hiding_scheduler=true"

uv run python -m src.train_nca \
    --grid 12 \
    --patch 2 \
    --num_colors 10 \
    --seq_len 1024 \
    --batch_size 128 \
    --val_batch_size 128 \
    --hidden_size 2048 \
    --intermediate_size 8192 \
    --num_layers 24 \
    --num_heads 32 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --steps 20000 \
    --train_num_rules 16000 \
    --val_num_rules 1000 \
    --sims_per_refill 4096 \
    --save_dir checkpoints/nca \
    "$@"
