#!/bin/bash
echo "Running CodeParrot Fine-Tuning Script..."
MODEL_PATH=PATH_TO_PT_MODEL
MODEL_FILE=MODEL_FILE_PATH_RECOMMEND_10_CHECKPOINT
SAVE_DIR=DIRECTORY_TO_SAVE_MODELS
DATA_DIR=DIRECTORY_TO_DATA

python src/language_train.py \
    --seed 5 \
    --device cuda:0 \
    --data_dir $DATA_DIR \
    --wandb_project PROJECT_NAME \
    --wandb_name RUN_NAME \
    --pretrain 1 \
    --save_path $SAVE_DIR \
    --model_path $MODEL_PATH \
    --model_file $MODEL_FILE \
    --pt_vocab_size 64000 \
    --reinit_modules embed none \
    --seq_len 1024 \
    --lr 2e-4 \
    --epochs 1 \
    --warmup 750 \
    --val_freq 3000 \
    --save_freq 500 \
    --patience -1 \
    --log_grad \
    --log_grad_freq 100 \
    --grad_clip 1.0 \
    --grad_clip_enable \
    --batch_size 16 \
    --mixed_precision fp16 \
    --autocast \
    --task full-codeparrot \
    --eval_enable \
    --grad_accumulation_steps 32 \
    --resume \
    --wandb_enable