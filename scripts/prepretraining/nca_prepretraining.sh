#!/bin/bash
echo "Running NCA Pre-Training Script..."
SAVE_DIR=DIRECTORY_TO_SAVE_MODELS

python src/nca_ppt.py \
    --wandb_enable \
    --wandb_name RUN_NAME \
    --wandb_project PROJECT_NAME \
    --seed 0 \
    --grid 12 \
    --patch 2 \
    --num_colors 10 \
    --seq_len 1024 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 100 \
    --warmup 10 \
    --save_dir $SAVE_DIR \
    --model_type llama \
    --model_name llama-large \
    --n_layer 24 \
    --n_head 32 \
    --n_embd 2048 \
    --temperature 1e-4 \
    --train_num_rules 16000 \
    --val_num_rules 2000 \
    --train_num_sim 500 \
    --val_num_sim 100 \
    --eval_num_sim 160 \
    --eval_num_rules 160 \
    --dT 1 \
    --distributed \
    --log_grad \
    --log_grad_freq 100 \
    --val_freq 500 \
    --autocast \
    --mixed_precision bf16 \
    --token \
    --vocab_size 64000 \
    --filter_rules \
    --filter_rules_threshold 0.5 \
    --filter_rules_upper_bound 1.0 \
    --filter_rules_mode gzip \
    --init_rollout_steps 10 \
    --generate_train \
    --resume \
    --generate_rules 1 \
    --grad_accumulation_steps 2 \
    --interval_save