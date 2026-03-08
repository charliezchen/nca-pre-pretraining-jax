import os
import sys
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from math_verify import parse, verify

import tiktoken

sys.path.append(".")

from utils.training_args import (
    MathEvalArgs,
    create_math_eval_parser,
    math_eval_args_to_dataclass
)

from utils.models import (
    DownstreamLlamaLM,
    create_llama_model
)

from utils.util import (
    set_seed,
    load_model,
    setup_logger,
    write_jsonl,
    read_jsonl
)

from utils.dataset_utils import (
    load_gsm8k_dataset, 
    GSM8KDataset,
    pass_at_k
)

log = setup_logger('math_eval')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def log_output(output, verbose=False):
    if verbose: log.info(output)

# === Evaluate Model on Math Tasks === #
def eval_task(args, model, dataloader, i=0):
    # evaluate the model on a single task
    model.eval()
    prompt, answer = dataloader[i]

    question = prompt.repeat(args.passes, 1)
    question = question.to(args.device)

    response = []

    MAX_LEN = args.max_len
    #MAX_LEN = min(args.max_len, max(0, args.seq_len - question.size(1)))
    #log.info(f"MAX_LEN: {MAX_LEN}")
    cache = None

    with torch.no_grad():
        count = 0
        eot = torch.zeros(args.passes, dtype=torch.long)

        while True:
            logits, cache = model(question, past_key_values=cache, use_cache=True)
            probs = F.softmax(logits[:, -1, :] / args.temperature, dim=-1)

            # sort probabilities
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # get indices to remove
            sorted_indices_to_remove = cumulative_probs > args.top_p

            # shift indices to remove to allow first index to be included
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            sorted_probs[sorted_indices_to_remove] = 0.0 # set probabilities to remove to 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True) # renormalize probabilities
            next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1) # sample next token
            next_token = torch.gather(sorted_indices, -1, next_token_sorted_idx)
            
            # check if the next tokens are all stop tokens otherwise keep generating
            count += 1
            eot = eot | (next_token.squeeze(0).detach().cpu() == args.eos_token) # check if all tokens are stop tokens

            question = next_token
            response.append(next_token.squeeze(0).detach().cpu())

            if eot.all().item() or count >= MAX_LEN:
                break            
    
    response = torch.cat(response, dim=1)

    return prompt, response, answer

def evaluate_response(args, question, answer, response, tokenizer, verbose=False):
    text_question = tokenizer.decode(question.detach().cpu().tolist())
    text_question = text_question.split(args.eos_string)[-1].strip()

    text_answer = tokenizer.decode(answer.detach().cpu().tolist())
    numeric_answer = text_answer.split(args.eos_string)[0].strip().split(args.stop_string)[-1].strip().split(" ")[-1]

    results = {
        "question": text_question,
        "explanation": text_answer,
        "answer": numeric_answer,
        "model_answer": [],
        "model_response": [],
        "pass@k": {},
        "correct": 0
    }
    total = 0
    correct = 0
    for i in range(response.shape[0]):
        text_response = tokenizer.decode(response[i].detach().cpu().tolist())
        numeric_response = text_response.split(args.eos_string)[0].strip().split(args.stop_string)[-1].strip().split(" ")[-1]
        text_response = text_response.split(args.eos_string)[0].strip()

        results["model_answer"].append(numeric_response)
        results["model_response"].append(text_response)
        if verify(parse(numeric_response), parse(numeric_answer)):
            correct += 1
        total += 1

    for k in args.eval_passes:
        results["pass@k"][str(k)] = pass_at_k(total, correct, k)

    results["correct"] = correct
    return correct > 0, results

# === Main Functions === #

def main(args: MathEvalArgs):
    set_seed(args.seed)
    log.info(f"Using seed: {args.seed}")
    # build and implement the model
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

    model = load_model(model, args.model_path, args.model_file, strict=True)
    model.to(args.device)
    model.eval()

    log.info(f"Model loaded from {args.model_path}")
    log.info(f"Model architecture: {model}")
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {params:,}")

    test_ds = load_gsm8k_dataset(name="main", split="test")
    log.info(f"Loaded {len(test_ds)} test examples")
    
    if args.pretrained_tokenizer == "owt":
        tokenizer = tiktoken.get_encoding("gpt2")
        hf_tokenizer = False
        args.stop_token = tokenizer.encode_ordinary(args.stop_string)[-1]
        args.eos_token = tokenizer.eot_token
        args.eos_string = tokenizer.decode([args.eos_token])
    elif args.pretrained_tokenizer == "math":
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        hf_tokenizer = True
        args.stop_token = tokenizer(
            args.stop_string,
            add_special_tokens=False,
            truncation=False
        )["input_ids"][-1]
        args.stop_string = tokenizer.decode([args.stop_token])
        args.eos_token = tokenizer.eos_token_id
        args.eos_string = tokenizer.eos_token
    
    dataset = GSM8KDataset(tokenizer, test_ds, hf_tokenizer=hf_tokenizer, num_icl_examples=3)

    # json results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    start_idx = 0 if args.start_idx is None else args.start_idx
    end_idx = min(args.end_idx, len(dataset)) if args.end_idx is not None else len(dataset)
    json_file = os.path.join(args.save_dir, f"results_{start_idx}_{end_idx}.jsonl")
    json_results = [{}]
    correct = 0
    total = 0
    json_results[0]["pass@k"] = {"sum": {str(k): 0 for k in args.eval_passes}, "mean": {str(k): 0 for k in args.eval_passes}}

    if args.resume and os.path.exists(json_file):
        json_results = read_jsonl(json_file)
        start_idx += json_results[0]["total"]
        correct = json_results[0]["correct"]
        total = json_results[0]["total"]

        log.info(f"Resuming from {start_idx} to {end_idx}")
        log.info(f"Correct: {correct} | Total: {total}")
        log.info(f"Total: {total}")
        log.info(f"Jsonl contains {len(json_results)-1} results")

    log.info(f"Json results: {json_results[0]}")
    for i in range(start_idx, end_idx):
        question, response, answer = eval_task(args, model, dataset, i=i)

        check, result = evaluate_response(args, question, answer, response, tokenizer, verbose=True)
        if check: correct += 1
        total += 1

        json_results[0]["accuracy"] = correct / total
        json_results[0]["total"] = total
        json_results[0]["correct"] = correct

        for k in args.eval_passes:
            json_results[0]["pass@k"]["sum"][str(k)] += result["pass@k"][str(k)]
            json_results[0]["pass@k"]["mean"][str(k)] = json_results[0]["pass@k"]["sum"][str(k)] / total

        json_results.append(result)

        write_jsonl(json_file, json_results)

    log.info(f"Accuracy: {correct / total}\n")
    log.info(f"Total: {total}\n")
    log.info(f"Correct: {correct}\n")

### Main Script ###
if __name__ == "__main__":
    parser = create_math_eval_parser()
    parsed_args = parser.parse_args()
    training_args = math_eval_args_to_dataclass(parsed_args)

    main(training_args)