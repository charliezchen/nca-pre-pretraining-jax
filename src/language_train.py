import os
import json
import sys
sys.path.append("../..")
sys.path.append(".")

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn.utils import clip_grad_norm_

from collections import defaultdict
from torch.utils.data import DataLoader
from functools import partial

import datasets
from transformers import AutoTokenizer

import numpy as np
import random
import wandb
import tiktoken

import itertools

from peft import get_peft_model, LoraConfig

from utils.training_args import (
    LanguageTrainingArgs,
    create_language_ft_parser,
    language_ft_args_to_dataclass,
    language_ft_dataclass_to_args
)

from utils.models import (
    DownstreamLlamaLM,
    create_llama_model,
    create_attention_mask
)

from utils.dataset_utils import (
    BigBenchDataset,
    LanguageTaskDataset,
    ChampTaskDataset,
    FullCodeParrotLanguageDataset,
    CodeParrotLanguageDataset,
    MathLanguageDataset,
    get_bigbench_dataset,
    generate_dyck_txt_file,
    generate_shuffle_dyck_txt_file,
    generate_dyck_dataset,
    compute_k_dyck_metrics,
    compute_k_shuffle_dyck_metrics,
    get_c4_dataset,
    C4Dataset,
    load_gsm8k_dataset, 
    GSM8KTrainDataset
)

from utils.util import (
    set_seed,
    get_lr_scheduler,
    load_model,
    delete_old_checkpoint,
    sort_checkpoint,
    load_checkpoint,
    save_checkpoint,
    wandb_log,
    setup_logger
)

log = setup_logger('language_ft_train')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------- Data Methods --------------------------

def build_dataloader(
    training_args: LanguageTrainingArgs,
):
    # set worker seeds
    def seed_worker(worker_id):
        worker_seed = training_args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(training_args.seed)

    if training_args.task == 'champ':
        # generate integers
        rng = np.random.RandomState(training_args.seed)
        max_int = 2**training_args.n_bit - training_args.n # max int to generate for successive integers
        starts = rng.randint(0, max_int, training_args.num_train + training_args.num_val)
        train_ints = starts[:training_args.num_train].tolist()
        val_ints = starts[training_args.num_train:].tolist()

        train_dataset = ChampTaskDataset(
            n_bit=training_args.n_bit,
            n=training_args.n_champ_int,
            starting_numbers=train_ints,
            max_seq_len=training_args.seq_len)
        val_dataset = ChampTaskDataset(
            n_bit=training_args.n_bit,
            n=training_args.n_champ_int,
            starting_numbers=val_ints,
            max_seq_len=training_args.seq_len) 

    elif training_args.task == 'dyck' or training_args.task == 'shuffle_dyck':
        if training_args.generate_dyck:
            if training_args.task == 'dyck':
                data_dir = generate_dyck_txt_file(
                    seed=training_args.seed,
                    file_dir=training_args.data_dir,
                    num_symbols=training_args.num_symbols,
                    n=training_args.n,
                    target_length=training_args.seq_len+1,
                    min_depth=training_args.min_depth,
                    max_depth=training_args.max_depth)
            elif training_args.task == 'shuffle_dyck':
                data_dir = generate_shuffle_dyck_txt_file(
                    seed=training_args.seed,
                    file_dir=training_args.data_dir,
                    num_symbols=training_args.num_symbols,
                    n=training_args.n,
                    target_length=training_args.seq_len+1,
                    p=training_args.p_open)
        else:
            data_dir = os.path.join(training_args.data_dir, f"dyck_sequences_{training_args.num_symbols}_{training_args.min_depth}_{training_args.max_depth}.txt")

        train_tokens, val_tokens = generate_dyck_dataset(
            file_path=data_dir,
            num_train=training_args.num_train,
            num_val=training_args.num_val,
            seq_len=training_args.seq_len)

        train_dataset = LanguageTaskDataset(
            dataset = train_tokens,
            max_seq_len=training_args.seq_len)
        val_dataset = LanguageTaskDataset(
            dataset = val_tokens,
            max_seq_len=training_args.seq_len)
    
    elif training_args.task == 'codeparrot':
        train_ds = datasets.load_dataset("codeparrot/codeparrot-train-v2-near-dedup", split='train', streaming=True)
        val_ds = datasets.load_dataset("codeparrot/codeparrot-valid-v2-near-dedup", split='train', streaming=True)

        tokenizer = AutoTokenizer.from_pretrained("helmo/code-search-net-multilang-tokenizer")

        train_dataset = CodeParrotLanguageDataset(
            tokenizer=tokenizer,
            dataset=train_ds,
            infinite=False,
            seq_len=training_args.seq_len,
            seed=training_args.seed
        )

        val_dataset = CodeParrotLanguageDataset(
            tokenizer=tokenizer,
            dataset=val_ds,
            infinite=False,
            seq_len=training_args.seq_len,
            seed=training_args.seed
        )

        train_dataloader = DataLoader(train_dataset, batch_size=training_args.batch_size, num_workers=training_args.num_workers, worker_init_fn=seed_worker, generator=g)
        val_dataloader = DataLoader(val_dataset, batch_size=training_args.batch_size, num_workers=training_args.num_workers, worker_init_fn=seed_worker, generator=g)
        return train_dataloader, val_dataloader
    
    elif training_args.task == 'full-codeparrot':
        train_dataset = FullCodeParrotLanguageDataset(
            data_dir = training_args.data_dir,
            split = 'train',
            block_size = training_args.seq_len)
        val_dataset = FullCodeParrotLanguageDataset(
            data_dir = training_args.data_dir,
            split = 'test',
            block_size = training_args.seq_len)
        
        train_dataloader = DataLoader(train_dataset, batch_size=training_args.batch_size, num_workers=training_args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=training_args.batch_size, num_workers=training_args.num_workers, shuffle=False)
        return train_dataloader, val_dataloader
    
    elif training_args.task == 'math':
        train_dataset = MathLanguageDataset(
            data_dir = training_args.data_dir,
            split = 'train',
            block_size = training_args.seq_len)
        val_dataset = MathLanguageDataset(
            data_dir = training_args.data_dir,
            split = 'test',
            block_size = training_args.seq_len)
        
        train_dataloader = DataLoader(train_dataset, batch_size=training_args.batch_size, num_workers=training_args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=training_args.batch_size, num_workers=training_args.num_workers, shuffle=False)
        return train_dataloader, val_dataloader

    elif training_args.task == 'gsm8k':
        if (training_args.tokenizer_type == "owt"):
            tokenizer = tiktoken.get_encoding("gpt2")
            hf_tokenizer = False
        elif (training_args.tokenizer_type == "math"):
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            hf_tokenizer = True
        else:
            raise ValueError(f"Tokenizer type {training_args.tokenizer_type} not supported")

        train_dataset = load_gsm8k_dataset(name="main", split="train")
        val_dataset = load_gsm8k_dataset(name="main", split="test")

        train_dataset = GSM8KTrainDataset(
            tokenizer=tokenizer,
            dataset=train_dataset,
            seq_len=training_args.seq_len,
            seed=training_args.seed,
            hf_tokenizer=hf_tokenizer
        )

        val_dataset = GSM8KTrainDataset(
            tokenizer=tokenizer,
            dataset=val_dataset,
            seq_len=training_args.seq_len,
            seed=training_args.seed,
            hf_tokenizer=hf_tokenizer
        )
    
    elif training_args.task == 'bigbench-lite':
        train_dataset = get_bigbench_dataset(split='train', min_samples=training_args.min_samples, max_samples=training_args.max_samples)
        val_dataset = get_bigbench_dataset(split='validation', min_samples=training_args.min_samples, max_samples=training_args.max_samples)

        if(training_args.tokenizer_type == "owt" or training_args.tokenizer_type == "math"):
            tokenizer = tiktoken.get_encoding("gpt2")
        
        train_dataset = BigBenchDataset(
            dataset=train_dataset,
            seq_len=training_args.seq_len,
            seed=training_args.seed,
            shot=training_args.n_shot,
            eval=False
        )

        val_dataset = BigBenchDataset(
            dataset=val_dataset,
            seq_len=training_args.seq_len,
            seed=training_args.seed,
            shot=training_args.n_shot,
            eval=False
        )

    elif training_args.task == 'c4':
        train_dataset = C4Dataset(
            data_dir=training_args.data_dir,
            split='train',
            block_size=training_args.seq_len
        )

        val_dataset = C4Dataset(
            data_dir=training_args.data_dir,
            split='test',
            block_size=training_args.seq_len
        )

        train_dataloader = DataLoader(train_dataset, batch_size=training_args.batch_size, shuffle=True, num_workers=training_args.num_workers, worker_init_fn=seed_worker, generator=g)
        val_dataloader = DataLoader(val_dataset, batch_size=training_args.batch_size, shuffle=False, num_workers=training_args.num_workers, worker_init_fn=seed_worker, generator=g)

        return train_dataloader, val_dataloader
        
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.batch_size, shuffle=True, num_workers=training_args.num_workers, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=training_args.batch_size, shuffle=False, num_workers=training_args.num_workers, worker_init_fn=seed_worker, generator=g)
    return train_dataloader, val_dataloader

# -------------------------- Training Methods --------------------------

def val_epoch(args, model, dataloader, criterion, eval_function=None, max_eval_samples=300):
    model.eval()
    total_loss = 0
    total_acc = 0
    max_eval_samples = max_eval_samples if max_eval_samples is not None else float('inf')
    total_eval_metrics = defaultdict(float)
    count = 0

    base_mask = create_attention_mask(args.seq_len).to(args.device) # pre-compute mask

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(args.device)
            targets = targets.to(args.device)

            B, N = sequences.shape
            attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)

            with torch.autocast(device_type=args.device.type, enabled=args.autocast):
                logits = model(sequences, attention_mask=attention_mask)
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
                
                loss = criterion(logits, targets)
            
            total_loss += loss.detach().cpu().item()
            predictions = logits.reshape(B, N, -1).argmax(dim=-1)
            targets = targets.reshape(B, N)

            if args.eval_enable and eval_function is not None:
                eval_metrics = eval_function(sequences.detach().cpu(), targets.detach().cpu(), predictions.detach().cpu())
                for key, value in eval_metrics.items():
                    total_eval_metrics[key] += value
            count += 1

            if count >= max_eval_samples:
                break

    for key, value in total_eval_metrics.items():
        total_eval_metrics[key] /= count

    return total_loss / count, total_eval_metrics

# -------------------------- Main Methods --------------------------

def init_args(args: LanguageTrainingArgs):
    # convert device to torch.device
    args.to_device()
    
    # ensure save directory exists
    args.save_dir = os.path.join(args.save_dir)
    args.set_runtime_paths()
    
    # assert task is one of 'champ', 'dyck', 'shuffle_dyck'

    if args.task == 'champ':
        args.vocab_size = 10
    elif args.task == 'dyck' or args.task == 'shuffle_dyck':
        args.vocab_size = args.num_symbols*2
    elif args.task in ['codeparrot', 'math', 'full-codeparrot', 'c4', 'metamathqa', 'gsm8k']:
        args.vocab_size = 64000
    elif args.task in ['bigbench-lite']:
        args.vocab_size = 50257
    else:
        raise ValueError(f"Task {args.task} not supported")
    
    # load wandb info from previous run if resuming
    if os.path.exists(args.metrics_path) and not args.wandb_resume_run_id and args.wandb_enable and args.resume:
        with open(args.metrics_path, 'r') as f:
            metrics = json.load(f)
        args.wandb_name = metrics['wandb_name']
        args.wandb_resume_run_id = metrics['wandb_run_id']

    return args

def main(args: LanguageTrainingArgs):
    set_seed(args.seed)
    args = init_args(args)
    
    log.info(f"Starting Language Fine-Tuning with seed: {args.seed}")
    log.info(f"Device: {args.device}")
    log.info(f"Save directory: {args.save_dir}")
    log.info(f"Task: {args.task}")

    # save path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # initialize wandb
    if args.wandb_enable:
        if args.wandb_resume_run_id:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                resume=True,
                id=args.wandb_resume_run_id
            )
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=language_ft_dataclass_to_args(args)
            )

    # create model
    if args.model_type == 'llama':
        log.info(f"Creating model with architecture: {args.model_type}")
        pretrain_model = create_llama_model(
            vocab_size=args.pt_vocab_size,
            seq_length=args.seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd
        )

    if args.pretrain:
        print(args.model_path)
        pretrain_model = load_model(pretrain_model, args.model_path, args.model_file, pretrain_pos=False)
        log.info('Pretrained model loaded')

    if args.lora == 1:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout)
        pretrain_model = get_peft_model(pretrain_model, lora_config)
        log.info('Applying LoRA')

    if args.model_type == 'llama':
        model = DownstreamLlamaLM(
            model=pretrain_model,
            vocab_size=args.vocab_size,
            frozen_modules=args.freeze_modules,
            reinit_modules=args.reinit_modules,
            weight_tying=args.weight_tying == 1)
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    if 'attn' in args.reinit_modules:
        model.reinit_attention_weights(args.reinit_layer_idxs)
    if 'mlp' in args.reinit_modules:
        model.reinit_mlp_weights(args.reinit_layer_idxs)
    if 'ln' in args.reinit_modules:
        model.reinit_layer_norm_weights(args.reinit_layer_idxs)

    if args.lora == 1:
        model.enable_lora()

    if args.task == 'gsm8k' or args.task == 'metamathqa':
        model = load_model(model, args.model_path, args.model_file, strict=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model architecture:\n{model}")
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_dataloader, val_dataloader = build_dataloader(args)

    # set up optimizer and scheduler
    criterion = CrossEntropyLoss()
    if args.task == 'dyck':
        eval_function = partial(compute_k_dyck_metrics, num_symbols=args.num_symbols, max_depth=args.max_depth)
    elif args.task == 'shuffle_dyck':
        eval_function = partial(compute_k_shuffle_dyck_metrics, num_symbols=args.num_symbols, max_depth=args.max_depth)
    else:
        eval_function = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training_steps = (len(train_dataloader) if args.steps_per_epoch is None else (args.steps_per_epoch * args.grad_accumulation_steps)) 
    effective_training_steps = training_steps // args.grad_accumulation_steps * args.epochs

    val_steps = len(val_dataloader) // args.grad_accumulation_steps if args.steps_per_epoch is None else None

    log.info(f"Training on {effective_training_steps // args.epochs} batches per epoch")
    log.info(f"Validation on {val_steps} batches per epoch")
    log.info(f"Validation frequency: every {args.val_freq} iterations")

    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=int(args.warmup*effective_training_steps) if args.warmup <= 1 else int(args.warmup),
        total_steps=int(effective_training_steps),
        decrease_mode=args.lr_decay_mode
    )

    # METRICS
    best_val_loss = float('inf')
    best_val_loss_el = float('inf')
    count = 0
    start_epoch = 0
    total_iterations = 0
    metrics = defaultdict(list)
    
    if args.patience == -1:
        args.patience = float('inf')
    patience = args.patience

    if args.resume and os.path.exists(args.metrics_path):
        model, optimizer, scheduler, start_epoch, _, best_val_loss, best_val_loss_el, metrics, total_iterations, dataloader_state = load_checkpoint(
            model, optimizer, scheduler, args.save_dir, args.load_dir, args.device, total_iterations=True
        )
    
    if args.resume_from_checkpoint:
        model, *_ = load_checkpoint(
            model, None, None, args.save_dir, args.load_dir, args.device
        )
        model._freeze_unfreeze_modules()
    
    # Store wandb info in metrics
    if args.wandb_enable:
        metrics['wandb_name'] = args.wandb_name
        metrics['wandb_run_id'] = wandb_run.id
    
    # compile model
    if args.compile and args.device.type != "mps" and torch.__version__ >= "2.0" and torch.cuda.device_count() == 1:
        compiled_model = torch.compile(model)
        torch._inductor.config.debug = False
        torch._inductor.config.verbose_progress = False
        torch._inductor.config.triton.unique_kernel_names = True
    else:
        compiled_model = model

    compiled_model.to(args.device)

    if torch.cuda.device_count() > 1:
        log.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        compiled_model = DataParallel(compiled_model)
    else:
        log.info(f"Using single GPU/CPU: {args.device}")
    
    # Main training loop
    
    dataloader_iter = iter(train_dataloader)
    
    if args.task in ['full-codeparrot', 'c4', 'math']:
        dataloader_iter = itertools.islice(dataloader_iter, int(total_iterations)*args.grad_accumulation_steps - training_steps*start_epoch, None)
        log.info(f"Skipping to iteration {int(total_iterations)*args.grad_accumulation_steps - training_steps*start_epoch}")
    else:
        for _ in range(int(total_iterations)*args.grad_accumulation_steps - training_steps*start_epoch):
            next(dataloader_iter)

    if (args.interval_save):
        interval_save_path = os.path.join(args.save_dir, 'interval_save')
        log.info(f"Interval save path: {interval_save_path}")
        if not os.path.exists(interval_save_path):
            os.makedirs(interval_save_path)

    for epoch in range(start_epoch, args.epochs):
        log.info(f"=== epoch: {epoch+1} / {args.epochs} ===")

        # Training with validation every X iterations
        compiled_model.train()

        if args.autocast:
            scaler = torch.GradScaler(device=args.device.type)

        epoch_loss = 0.0
        batch_count = 0
        accumulated_loss = 0.0
        accumulation_step = 0
        base_mask = create_attention_mask(args.seq_len).to(args.device) # pre-compute mask

        for batch in dataloader_iter:
            sequences, targets = batch
            sequences = sequences.to(args.device)
            targets = targets.to(args.device)

            B = sequences.shape[0]
            attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)

            with torch.autocast(device_type=args.device.type, enabled=args.autocast):
                logits = compiled_model(sequences, attention_mask=attention_mask)

                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)

                loss = criterion(logits, targets)

            accumulated_loss += loss.detach().cpu().item()
            accumulation_step += 1

            if args.autocast:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if accumulation_step % args.grad_accumulation_steps == 0:
                if args.autocast:
                    if args.grad_clip_enable or args.log_grad:
                        scaler.unscale_(optimizer)
                    
                    if args.grad_clip_enable:
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    if args.log_grad and total_iterations % args.log_grad_freq == 0:
                        grad_norm = torch.norm(
                            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
                        )
                        wandb_log({'iteration': total_iterations, 'train/grad_norm': grad_norm.detach().cpu().item(), "lr": scheduler.get_last_lr()[0]}, args)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip_enable:
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    if args.log_grad and total_iterations % args.log_grad_freq == 0:
                        grad_norm = torch.norm(
                            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
                        )
                        wandb_log({'iteration': total_iterations, 'train/grad_norm': grad_norm.detach().cpu().item(), "lr": scheduler.get_last_lr()[0]}, args)
                    
                    optimizer.step()

                # log accumulated loss
                avg_accumulated_loss = accumulated_loss / args.grad_accumulation_steps
                wandb_log({'iteration': total_iterations, 'train/loss': avg_accumulated_loss}, args)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += avg_accumulated_loss
                batch_count += 1
                total_iterations += 1

                # reset accumulation
                accumulation_step = 0
                accumulated_loss = 0.0

                if args.interval_save and total_iterations in args.intervals: # every 10 epochs worth of tokens
                    save_checkpoint(total_iterations, epoch, model, optimizer, scheduler, best_val_loss, best_val_loss_el, metrics, interval_save_path, total_iterations=total_iterations)
            # Run validation every X iterations
            if accumulation_step % args.grad_accumulation_steps == 0 and total_iterations % args.val_freq == 0:
                val_loss, val_metrics = val_epoch(args, compiled_model, val_dataloader, criterion, eval_function)
                log.info(f"Iteration {total_iterations} | train loss: {avg_accumulated_loss:.8f} | val loss: {val_loss:.8f}")
                
                for key, value in val_metrics.items():
                    new_key = f'val/{key}'
                    metrics[new_key].append({total_iterations: value})
                    wandb_log({'iteration': total_iterations, new_key: value}, args)
                
                # Save checkpoint if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(total_iterations, epoch, model, optimizer, scheduler, best_val_loss, best_val_loss_el, metrics, args.save_dir, best=True, total_iterations=total_iterations)
                    delete_old_checkpoint(args.save_dir)
                
                # Save metrics
                with open(args.metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(metrics), f, indent=2)

                metrics['train/loss_mean'].append({total_iterations: avg_accumulated_loss})
                metrics['val/loss_mean'].append({total_iterations: val_loss})
                
                wandb_log({"iteration": total_iterations, "train/loss_mean": epoch_loss / batch_count, "val/loss_mean": val_loss}, args)

                # early stopping
                if val_loss - best_val_loss_el >= 1e-4:
                    count += 1
                    if count > patience:
                        log.info(f"Early stop at iteration {total_iterations}")
                        wandb_log({"iteration": total_iterations, "early stop": total_iterations}, args)
                        break
                else:
                    count = 0
                    best_val_loss_el = val_loss

            if (accumulation_step % args.grad_accumulation_steps == 0 and total_iterations % args.save_freq == 0):
                save_checkpoint(total_iterations, epoch, model, optimizer, scheduler, best_val_loss, best_val_loss_el, metrics, args.save_dir, total_iterations=total_iterations)
                delete_old_checkpoint(args.save_dir)
                # Save metrics
                with open(args.metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(metrics), f, indent=2)
 
            if(args.steps_per_epoch is not None and total_iterations >= effective_training_steps):
                break
        
        avg_epoch_loss = epoch_loss / batch_count
        log.info(f"Epoch {epoch+1} completed | average train loss: {avg_epoch_loss:.8f}")
        dataloader_iter = iter(train_dataloader)

    save_checkpoint(total_iterations, epoch, model, optimizer, scheduler, best_val_loss, best_val_loss_el, metrics, args.save_dir, total_iterations=total_iterations)
    delete_old_checkpoint(args.save_dir)
    with open(args.metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(dict(metrics), f, indent=2)
    wandb.finish()
    
if __name__ == "__main__":
    parser = create_language_ft_parser()
    parsed_args = parser.parse_args()
    training_args = language_ft_args_to_dataclass(parsed_args)
    main(training_args)
