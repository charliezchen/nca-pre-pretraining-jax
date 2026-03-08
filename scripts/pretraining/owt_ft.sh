#!/bin/bash
echo "Running OpenWebText Fine-Tuning Script..."
MODEL_PATH=PATH_TO_PT_MODEL
MODEL_FILE=MODEL_FILE_PATH_RECOMMEND_10_CHECKPOINT
SAVE_DIR=DIRECTORY_TO_SAVE_MODELS
DATA_DIR=DIRECTORY_TO_DATA

python src/openwebtext_pt.py \
    --device 0 \
    --seed 5 \
    --save_dir $SAVE_DIR \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --model_file $MODEL_FILE \
    --lr 5e-4 \
    --pretrain 1 \
    --warmup 0.1 \
    --epochs 1 \
    --wandb_enable \
    --wandb_name RUN_NAME \
    --wandb_project PROJECT_NAME \
    --autocast \
    --mixed_precision fp16 \
    --log_grad 1 \
    --log_grad_freq 100 \
    --grad_clip 1.0 \
    --resume \
    --grad_clip_enable 1 \
    --freeze_modules "" \
    --reinit_modules embed none \
    --reinit_layer_idxs 0 24 \
    --weight_decay 0.0001 \
    --pt_vocab_size 64000 \
    --interval_save