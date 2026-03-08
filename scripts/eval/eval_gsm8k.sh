#!/bin/bash
echo "Running GSM8K Evaluation Script..."
MODEL_PATH=PATH_TO_FT_MODEL
MODEL_FILE=MODEL_FILE_PATH
SAVE_DIR=DIRECTORY_TO_SAVE_RESULTS

python src/eval/gsm8k.py \
    --seed 0 \
    --device cuda:0 \
    --save_path $SAVE_DIR \
    --model_path $MODEL_PATH \
    --model_file $MODEL_FILE \
    --vocab_size 50257 \
    --seq_len 1024 \
    --temperature 0.6 \
    --top_p 1.0 \
    --passes 32 \
    --max_len 250 \
    --stop_string "####" \
    --mixed_precision fp16 \
    --autocast \
    --resume
