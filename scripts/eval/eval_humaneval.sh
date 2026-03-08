#!/bin/bash
echo "Running HumanEval Evaluation Script..."
MODEL_PATH=PATH_TO_FT_MODEL
MODEL_FILE=MODEL_FILE_PATH
SAVE_DIR=DIRECTORY_TO_SAVE_RESULTS

python src/eval/humaneval.py \
    --seed 0 \
    --device cuda:0 \
    --save_path $SAVE_DIR \
    --model_path $MODEL_PATH \
    --model_file $MODEL_FILE \
    --vocab_size 64000 \
    --seq_len 1024 \
    --temperature 0.4 \
    --top_p 0.95 \
    --passes 64 \
    --eval_passes 1 2 4 8 16 32 \
    --max_len 500 \
    --weight_tying 1 \
    --reinit_modules embed none \
    --mixed_precision fp16 \
    --autocast \
    --resume
