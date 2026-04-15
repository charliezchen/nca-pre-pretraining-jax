#!/usr/bin/env bash
# Language pretraining on 8 TPU chips, optionally warm-starting from an NCA ckpt.
set -euo pipefail

export XLA_FLAGS="--xla_tpu_enable_latency_hiding_scheduler=true"

uv run python -m src.train_lm \
    --dataset Skylion007/openwebtext \
    --tokenizer_name gpt2 \
    --seq_len 1024 \
    --batch_size 128 \
    --vocab_size 50304 \
    --hidden_size 2048 \
    --intermediate_size 8192 \
    --num_layers 24 \
    --num_heads 32 \
    --lr 3e-4 \
    --warmup_steps 1000 \
    --steps 50000 \
    --save_dir checkpoints/lm \
    "$@"
