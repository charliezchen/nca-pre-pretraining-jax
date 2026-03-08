import os
import argparse 
import sys
import json
import math

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer

import numpy as np
import itertools

import glob
from PIL import Image

import tiktoken

sys.path.append(".")
sys.path.append("..")

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from utils.util import (
    set_seed,
    setup_logger,
    load_model,
    write_jsonl,
    read_jsonl
)

from utils.models import (
    create_llama_model,
    DownstreamLlamaLM,
    create_attention_mask
)

from utils.training_args import (
    HumanEvalArgs,
    create_human_eval_parser,
    human_eval_args_to_dataclass
)

from typing import List

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#### CONSTANTS ####
log = setup_logger('eval_humaneval')

EOS_STRINGS = [
    '\nclass ',
    '\ndef ',
    '\nif __name__',
    '\nprint('
]

#### DECODING ####
def generate_code(args, model, seq):
    """
    Generate tokens autoregressively using bootstrapped generation.
    
    Args:
        args: Arguments containing max_len, temperature, top_p, device, vocab_size
        model: The language model
        seq: Initial sequence tokens [batch_size, seq_len] where batch_size = args.passes
    
    Returns:
        response: Generated tokens [batch_size, max_len]
    """
    model.eval()
    MAX_LEN = min(args.max_len, args.seq_len - seq.shape[1])
    
    # Move to device and ensure correct dtype
    seq = seq.to(args.device)
    # Clone to avoid any potential issues with in-place operations
    current_seq = seq.clone()
    
    with torch.no_grad():
        response = []
        cache = None
        
        for i in range(MAX_LEN):            
            # Forward pass
            logits, cache = model(current_seq, past_key_values=cache, use_cache=True)
            
            # Get next token prediction (last position)
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply temperature and sampling
            if args.temperature == 0.0:
                # Greedy decoding (argmax)
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                # Temperature sampling with top-p (nucleus sampling)
                probs = F.softmax(next_token_logits / args.temperature, dim=-1)
                
                # Sort probabilities for top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Get indices to remove (top-p filtering)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                sorted_probs[sorted_indices_to_remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                # Sample from filtered distribution
                if torch.any(torch.isnan(sorted_probs)) or torch.any(sorted_probs < 0):
                    # Fallback to greedy if probabilities are invalid
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, -1, next_token_sorted_idx)
            
            # Append to current sequence for next iteration
            current_seq = next_token
            
            # Store generated token for response
            response.append(next_token.detach().cpu())
        
        # Concatenate all generated tokens: [batch_size, max_len]
        response = torch.cat(response, dim=1)
    
    return response.detach().cpu()

def extract_code(text: str, stop_strings: List[str]) -> str:
    """
    Extract code from generated text by stopping at any of the stop strings.
    Returns the text up to (but not including) the first occurrence of any stop string.
    """
    min_pos = len(text)
    for stop_string in stop_strings:
        pos = text.find(stop_string)
        if pos != -1 and pos < min_pos:
            min_pos = pos
    
    return text[:min_pos]

def decode_response(args, response, prompt=None, pid=None):
    """
    Decode response into submissions for human_eval

    Args:
        args: Arguments containing tokenizer
        response: Generated tokens [batch_size, max_len]

    Returns:
        decoded_response: Decoded response [batch_size]
    """
    responses = []
    for i in range(response.shape[0]):
        decoded_response = args.tokenizer.decode(response[i].tolist(), skip_special_tokens=True)
        # Split on all EOS_STRINGS
        recorded_response = extract_code(decoded_response, EOS_STRINGS)
        
        responses.append({
            "task_id": pid,
            "completion": recorded_response,
            "question": prompt
        })

    return responses

def encode_prompt(args, prompt):
    tokens = args.tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt"
    )["input_ids"]

    tokens = torch.tile(tokens, (args.passes, 1))    
    return tokens

#### MISC ####

def build_model(args: HumanEvalArgs):
    pretrain_model = create_llama_model(
        vocab_size=args.vocab_size,
        seq_length=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    model = DownstreamLlamaLM(
        model=pretrain_model,
        vocab_size=args.vocab_size,
        frozen_modules=args.freeze_modules,
        reinit_modules=args.reinit_modules+['embed'],
        weight_tying=args.weight_tying == 1,
    )
    return model

def init_args(args: HumanEvalArgs):
    assert max(args.eval_passes) <= args.passes, "Evaluation passes must be less than or equal to the number of passes"

    # set tokenizer 
    args.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # resuming and file logic
    args.problems = read_problems()
    args.start_idx = 0 if args.start_idx is None else args.start_idx
    args.end_idx = len(args.problems) if args.end_idx is None else args.end_idx

    log.info("Saving solutions to %s", args.save_dir)

    args.solution_path = os.path.join(args.save_dir, f"solutions_{args.start_idx}_{args.end_idx}.jsonl")
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)

    # set up path to save solutions
    return args

#### MAIN ####

def main(args):
    set_seed(args.seed)
    args = init_args(args)
    log.info(f"Arguments: {args}")

    # build and load model

    model = build_model(args)
    model = load_model(model, args.model_path, args.model_file, strict=True)
    model.to(args.device)
    model.eval()

    # output model info

    log.info(f"Model loaded from {args.model_path}")
    log.info(f"Model architecture: {model}")
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {params:,}")

    # Load problems and set up resume
    problems = read_problems()
    problem_ids = list(problems.keys())
    problem_ids.sort()
    problem_ids = problem_ids[args.start_idx:args.end_idx]

    # responses
    responses = []
   
    # resume logic
    if args.resume and os.path.exists(args.solution_path):
        responses = read_jsonl(args.solution_path)
        args.start_idx += len(responses)
    else:
        responses = []

    for pid in problem_ids:
        # encode prompt (already tiled by encode_prompt)
        prompt = problems[pid]['prompt']
        tokens = encode_prompt(args, prompt)

        # generate response
        response = generate_code(args, model, tokens)
        
        # decode response
        decoded_responses = decode_response(args, response, prompt, pid)

        # output decoded response
        responses.extend(decoded_responses)

        # write responses
        write_jsonl(args.solution_path, responses)

        # output progress
        log.info(f"{decoded_responses[0]}")
        
        # Clear GPU cache between problems to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # run an evaluation portion of human eval        
    pass_at_k = evaluate_functional_correctness(args.solution_path, args.eval_passes)
    pass_at_k = {k: pass_at_k[k].item() for k in pass_at_k.keys()}
    log.info(f"Pass@k: {pass_at_k}")

    with open(os.path.join(args.save_dir, "pass_at_k.json"), "w") as f:
        json.dump(pass_at_k, f)
    log.info(f"Pass@k saved to {os.path.join(args.save_dir, 'pass_at_k.json')}")

if __name__ == "__main__":
    parser = create_human_eval_parser()
    args = parser.parse_args()
    training_args = human_eval_args_to_dataclass(args)
    main(training_args)