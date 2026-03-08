#!/bin/bash
echo "Running BigBench-Lite Few-Shot Fine-Tuning Script..."
MODEL_PATH=PATH_TO_PT_MODEL
MODEL_FILE=MODEL_FILE_PATH_RECOMMEND_10_CHECKPOINT
SAVE_DIR=DIRECTORY_TO_SAVE_MODELS

python src/language_train.py \
    --seed 0 \
    --device cuda:0 \
    --save_path $SAVE_DIR \
    --model_path $MODEL_PATH \
    --model_file $MODEL_FILE \
    --wandb_project BigBench \
    --wandb_name RUN_NAME \
    --wandb_enable \
    --n_shot 0 3 \
    --min_samples 100 \
    --max_samples 350 \
    --pt_vocab_size 50257 \
    --vocab_size 50257 \
    --reinit_modules embed none \
    --reinit_layer_idxs 0 24 \
    --seq_len 1024 \
    --lr 5e-6 \
    --epochs 1 \
    --warmup 0.1 \
    --val_freq 5 \
    --save_freq 5 \
    --patience -1 \
    --log_grad \
    --log_grad_freq 10 \
    --grad_clip 1.0 \
    --grad_clip_enable \
    --batch_size 16 \
    --mixed_precision fp16 \
    --autocast \
    --task bigbench-lite \
    --grad_accumulation_steps 4 \
    --eval_enable \
    --interval_save \
    --intervals 10 20 30 40 50 \
    --resume
