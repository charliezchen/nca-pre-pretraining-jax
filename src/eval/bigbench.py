import os
import sys
import argparse
import json
import math

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F

import tiktoken
from datasets import load_dataset

from utils.util import (
    set_seed,
    setup_logger,
    load_model,
    wandb_log,
    write_jsonl,
    read_jsonl
)

from utils.models import (
    create_llama_model,
    DownstreamLlamaLM,
    create_attention_mask
)

from utils.dataset_utils import (
    BigBenchDataset,
    get_bigbench_dataset,
    MC_LETTERS,
    pass_at_k
)

from utils.training_args import (
    BigBenchEvalArgs,
    create_bigbench_eval_parser,
    bigbench_eval_args_to_dataclass
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

### === CONSTANTS === ###
log = setup_logger('bigbench_eval')

### === DECODING FUNCTIONS === ###

def compute_logprobs(args, model, seq, targets):
    """Compute the sum of conditional log-probs for answer tokens.

    Args:
        seq:     1D LongTensor [T] — full sequence (context + candidate answer).
        targets: 1D LongTensor [T] — -100 for context positions; answer token ids
                 at positions where logits[t] predicts the next answer token
                 (i.e. targets[N-1:N+M-1] = answer_tokens for context length N,
                 answer length M).
    Returns:
        Scalar float: sum of log P(answer_token_t | prefix).
    """
    model.eval()
    with torch.no_grad():
        seq = seq.unsqueeze(0).to(args.device)    # [1, T]
        targets = targets.to(args.device)          # [T]

        seq_len = seq.size(1)
        mask = create_attention_mask(seq_len).to(args.device)

        logits = model(seq, mask).squeeze(0)       # [T, V]

        answer_mask = targets != -100              # [T] — True at answer positions
        selected_logits = logits[answer_mask]      # [M, V]
        selected_targets = targets[answer_mask]    # [M]

        log_probs = F.log_softmax(selected_logits, dim=-1)                       # [M, V]
        token_log_probs = log_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1)  # [M]
        return token_log_probs.sum().item()

def generate_response(args, model, seq):
    model.eval()
    MAX_LEN = min(args.max_len, args.seq_len - seq.shape[0])

    # Move to device and ensure correct dtype
    seq = torch.tile(seq, (args.passes, 1))
    seq = seq.to(args.device)
    # Clone to avoid any potential issues with in-place operations
    current_seq = seq.clone()

    with torch.no_grad():
        response = []

        for i in range(MAX_LEN):
            # Create attention mask for current sequence
            seq_len = current_seq.size(1)
            base_mask = create_attention_mask(seq_len).to(args.device)
            mask = base_mask.repeat(current_seq.size(0), 1, 1, 1).to(args.device)

            # Forward pass
            logits = model(current_seq, mask)

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
            current_seq = torch.cat([current_seq, next_token], dim=1)

            # Store generated token for response
            response.append(next_token.detach().cpu())

    response = torch.cat(response, dim=1)
    return response

def decode_response(args, response, subset, example, idx, num_choices, correct_answers):
    responses = []
    for i in range(response.shape[0]):
        decoded_response = args.tokenizer.decode(response[i].tolist()).split(args.eos_string)[0].strip()

        responses.append({
            "question": example["inputs"],
            "solution": decoded_response,
            "correct_answers": correct_answers,
            "num_choices": num_choices,
            "subset": subset,
            "idx": idx
        })
    return responses

def is_correct(response):
    """Check if solution matches any of the correct answers."""
    solution = response["solution"].lower().strip()
    # support both old format (single "answer") and new format ("correct_answers" list)
    if "correct_answers" in response:
        return any(solution == a.lower().strip() for a in response["correct_answers"])

    return solution == response["answer"].lower().strip()

def evaluate_pass_at_k(args, dataset, responses):
    # per-category accumulators: {subset -> {k -> {sum, count}, num_choices}}
    category_metrics = {}

    for i in range(len(dataset)):
        c = 0
        n = 0
        subset = None
        num_choices = 0
        for response in responses:
            if response["idx"] == i:
                if is_correct(response):
                    c += 1
                n += 1
                if subset is None:
                    subset = response["subset"]
                    num_choices = response.get("num_choices", 0)

        if subset not in category_metrics:
            category_metrics[subset] = {
                "sum": {str(k): 0 for k in args.eval_passes},
                "count": {str(k): 0 for k in args.eval_passes},
                "num_choices": num_choices,
            }

        for k in args.eval_passes:
            pak = pass_at_k(n, c, k)
            category_metrics[subset]["sum"][str(k)] += pak
            category_metrics[subset]["count"][str(k)] += 1

    # compute per-category metrics:
    # normalize at the task level (mean first, then normalize+clamp) to match BIG-Bench paper
    problem_metrics = {}
    for subset, vals in category_metrics.items():
        num_choices = vals["num_choices"]
        mean_k = {str(k): vals["sum"][str(k)] / vals["count"][str(k)] for k in args.eval_passes}

        if num_choices > 1:
            chance = 1.0 / num_choices
            norm_mean_k = {
                str(k): max(0.0, (mean_k[str(k)] - chance) / (1.0 - chance))
                for k in args.eval_passes
            }
        else:
            norm_mean_k = mean_k

        problem_metrics[subset] = {
            "mean": mean_k,
            "normalized_mean": norm_mean_k,
            "count": vals["count"],
        }

    # macro-average: average of per-task normalized means
    macro_mean = {}
    for k in args.eval_passes:
        task_means = [
            pm["normalized_mean"][str(k)]
            for pm in problem_metrics.values()
            if str(k) in pm.get("normalized_mean", {})
        ]
        macro_mean[str(k)] = sum(task_means) / len(task_means) if task_means else 0.0

    return macro_mean, problem_metrics

### === UTILITY FUNCTIONS === ###
def init_args(args: BigBenchEvalArgs):
    # Filter eval_passes to only include values <= passes (handles greedy/pass@1 case)
    valid_eval_passes = [k for k in args.eval_passes if k <= args.passes]
    if len(valid_eval_passes) < len(args.eval_passes):
        dropped = [k for k in args.eval_passes if k > args.passes]
        log.warning(f"Dropping eval_passes {dropped} since passes={args.passes}")
    assert len(valid_eval_passes) > 0, f"No valid eval_passes <= passes={args.passes}"
    args.eval_passes = valid_eval_passes

    # set tokenizer
    args.tokenizer = tiktoken.get_encoding("gpt2")
    args.eos_tk = args.tokenizer.eot_token
    args.eos_string = args.tokenizer.decode([args.eos_tk])
    args.solution_path = os.path.join(args.save_dir, f"solutions_{args.start_idx}_{args.end_idx}.jsonl")
    os.makedirs(args.save_dir, exist_ok=True)

    return args

def build_model(args: BigBenchEvalArgs):
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

### === MAIN FUNCTIONS === ###

def main(args):
    set_seed(args.seed)
    args = init_args(args)

    # load dataset
    dataset = get_bigbench_dataset(split='validation', min_samples=args.min_samples, max_samples=args.max_per_task)
    dataset = BigBenchDataset(
        dataset=dataset,
        seq_len=args.seq_len,
        seed=args.seed,
        shot=args.n_shot,
        eval=True,
        few_shot_prompts_path=args.few_shot_prompts_path
    )

    # build and load model
    model = build_model(args)
    log.info(f"Model built: n_layer={args.n_layer}, n_head={args.n_head}, n_embd={args.n_embd}, vocab_size={args.vocab_size}")

    # Log model keys before loading checkpoint
    model_keys = set(model.state_dict().keys())
    model = load_model(model, args.model_path, args.model_file, strict=True)
    model.to(args.device)
    model.eval()

    # output model info
    log.info(f"Model loaded from {args.model_path}")
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {params:,}")

    end_idx = len(dataset) if args.end_idx is None else args.end_idx

    if args.resume and os.path.exists(args.solution_path):
        responses = read_jsonl(args.solution_path)
        completed_idxs = set(r["idx"] for r in responses)
        log.info(f"Resuming: {len(completed_idxs)} examples already completed")
    else:
        responses = []
        completed_idxs = set()

    log.info(f"Evaluating from 0 to {end_idx} (skipping {len(completed_idxs)} completed)")

    for i in range(0, end_idx):
        if i in completed_idxs:
            continue

        seq, _tar, num_choices, correct_answers = dataset[i]

        remaining = args.seq_len - seq.shape[0]
        if remaining < args.max_len:
            subset, example = dataset.enumerated_examples[i]
            log.info(f"Skipping {subset} example {i}: seq_len={seq.shape[0]}, only {remaining} tokens left (need {args.max_len})")
            continue

        generation = generate_response(args, model, seq)
        subset, example = dataset.enumerated_examples[i]

        solution = decode_response(args, generation, subset, example, i, num_choices, correct_answers)

    ### Evaluate Pass@K ###
    macro_mean, problem_metrics = evaluate_pass_at_k(args, dataset, responses)
    log.info(f"Pass@K (normalized, macro-avg): {macro_mean}")
    log.info(f"Pass@K by task: {problem_metrics}")

    fpath = os.path.join(args.save_dir, "pass_at_k.json")
    with open(fpath, "w") as f:
        json.dump({"normalized_macro_mean": macro_mean}, f)

    fpath_by_type = os.path.join(args.save_dir, "pass_at_k_by_type.json")
    with open(fpath_by_type, "w") as f:
        json.dump(problem_metrics, f, indent=2)

    return macro_mean, problem_metrics

def main_inference(args):
    # MAIN METHOD FOR NO FINE-TUNING INFERENCE EVALUATION VIA CONDITIONAL LOG PROBABILITY
    set_seed(args.seed)
    args = init_args(args)

    # For logprob MC eval always use few-shot: ensure at least 1 in-context example
    shot = args.n_shot
    if args.few_shot_prompts_path is None and shot[0] == 0:
        shot = [1, max(1, shot[1])]
        log.info(f"Logprob eval: forcing few-shot shot range to {shot}")

    # load dataset — eval=True so __getitem__ returns the prefix (context+question, no answer)
    # mc_letter_format=True: choices formatted as A./B./C./D., answers/correct_answers as letters
    dataset = get_bigbench_dataset(split='validation', min_samples=args.min_samples, max_samples=args.max_per_task)
    dataset = BigBenchDataset(
        dataset=dataset,
        seq_len=args.seq_len,
        seed=args.seed,
        shot=shot,
        eval=True,
        few_shot_prompts_path=args.few_shot_prompts_path,
        mc_letter_format=True,
    )

    # build and load model
    model = build_model(args)
    model = load_model(model, args.model_path, args.model_file, strict=True)
    model.to(args.device)
    model.eval()

    log.info(f"Model loaded from {args.model_path}")
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {params:,}")

    end_idx = len(dataset) if args.end_idx is None else args.end_idx

    if args.resume and os.path.exists(args.solution_path):
        responses = read_jsonl(args.solution_path)
        completed_idxs = set(r["idx"] for r in responses)
        log.info(f"Resuming: {len(completed_idxs)} examples already completed")
    else:
        responses = []
        completed_idxs = set()

    log.info(f"Evaluating (logprob) from 0 to {end_idx} (skipping {len(completed_idxs)} completed)")

    for i in range(0, end_idx):
        if i in completed_idxs:
            continue

        prefix_seq, _, num_choices, correct_answers = dataset[i]
        subset, example = dataset.enumerated_examples[i]

        mc_targets = example.get('multiple_choice_targets', [])
        if not mc_targets:
            log.info(f"Skipping {subset} example {i}: no multiple_choice_targets (free-form tasks not supported for logprob eval)")
            continue

        # Use single letter tokens (A/B/C/D) as candidates — one token per choice
        letter_candidates = [MC_LETTERS[j] for j in range(min(len(mc_targets), len(MC_LETTERS)))]

        N = prefix_seq.shape[0]
        best_letter = None
        best_score = float('-inf')

        for letter in letter_candidates:
            # Encode as a single letter token (no trailing newlines — single token for clean scoring)
            answer_tokens = torch.tensor(
                dataset.tokenizer.encode_ordinary(letter), dtype=torch.long
            )
            M = answer_tokens.shape[0]

            full_seq = torch.cat([prefix_seq, answer_tokens])   # [N+M]

            if full_seq.shape[0] > args.seq_len:
                log.info(f"Skipping {subset} example {i} candidate '{letter}': exceeds seq_len ({full_seq.shape[0]} > {args.seq_len})")
                continue

            # targets[t] = what logits[t] should predict (i.e., full_seq[t+1])
            # logits[N-1] predicts answer_tokens[0], ..., logits[N+M-2] predicts answer_tokens[M-1]
            targets = torch.full((full_seq.shape[0],), -100, dtype=torch.long)
            targets[N - 1:N + M - 1] = answer_tokens

            score = compute_logprobs(args, model, full_seq, targets)
            if score > best_score:
                best_score = score
                best_letter = letter

        if best_letter is None:
            log.info(f"Skipping {subset} example {i}: all candidates exceeded seq_len")
            continue

        responses.append({
            "question": example["inputs"],
            "solution": best_letter,
            "correct_answers": correct_answers,  # already letters (A/B/C/D) from dataset
            "num_choices": num_choices,
            "subset": subset,
            "idx": i,
            "logprob_score": best_score,
        })
        write_jsonl(args.solution_path, responses)

    ### Evaluate Pass@K (pass@1 = accuracy for logprob argmax) ###
    macro_mean, problem_metrics = evaluate_pass_at_k(args, dataset, responses)
    log.info(f"Accuracy (normalized, macro-avg): {macro_mean}")
    log.info(f"Accuracy by task: {problem_metrics}")

    fpath = os.path.join(args.save_dir, "pass_at_k.json")
    with open(fpath, "w") as f:
        json.dump({"normalized_macro_mean": macro_mean}, f)

    fpath_by_type = os.path.join(args.save_dir, "pass_at_k_by_type.json")
    with open(fpath_by_type, "w") as f:
        json.dump(problem_metrics, f, indent=2)

    return macro_mean, problem_metrics

if __name__ == "__main__":
    parser = create_bigbench_eval_parser()
    args = parser.parse_args()
    training_args = bigbench_eval_args_to_dataclass(args)
    if training_args.eval_mode == 'logprob':
        main_inference(training_args)
    else:
        main(training_args)
