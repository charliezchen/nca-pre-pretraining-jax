#!/bin/bash
echo "Running GSM8K Math Fine-Tuning Script..."
MODEL_PATH=PATH_TO_PT_MODEL
MODEL_FILE=MODEL_FILE_PATH_RECOMMEND_10_CHECKPOINT
SAVE_DIR=DIRECTORY_TO_SAVE_MODELS

python src/language_train.py \
    --seed 0 \
    --device cuda:0 \
    --save_path $SAVE_DIR \
    --model_path $MODEL_PATH \
    --model_file $MODEL_FILE \
    --wandb_project GSM8K \
    --wandb_name RUN_NAME \
    --wandb_enable \
    --pretrain 1 \
    --pt_vocab_size 64000 \
    --vocab_size 64000 \
    --reinit_modules embed none \
    --reinit_layer_idxs 0 24 \
    --seq_len 1024 \
    --lr 1e-5 \
    --epochs 10 \
    --warmup 0.1 \
    --val_freq 14 \
    --patience -1 \
    --log_grad \
    --log_grad_freq 100 \
    --grad_clip 1.0 \
    --grad_clip_enable \
    --batch_size 16 \
    --mixed_precision fp16 \
    --autocast \
    --task gsm8k \
    --grad_accumulation_steps 32 \
    --eval_enable \
    --interval_save \
    --intervals 70 140 \
    --resume
