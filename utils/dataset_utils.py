import os
import sys
import json

sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

import datasets
from datasets import load_from_disk, load_dataset, DatasetDict

from abc import ABC, abstractmethod

from typing import Dict, List, Tuple
from collections import defaultdict, Counter

from utils.tokenizers import Tokenizer
from transformers import AutoTokenizer

import numpy as np
import tiktoken

from utils.util import setup_logger
log = setup_logger('dataset_utils')

"""
Implementation of simple language tasks for evaluating downstream performance of pretrained models
"""

### ===== Base Dataset Classes ===== ###

class BaseBinaryLanguageDataset(Dataset):
    """
    Base class for language datasets that load from binary (.bin) files.
    All binary datasets use memory-mapped files and block-based sampling.
    """
    def __init__(self,
        data_dir: str,
        split: str = 'train',
        block_size: int = 1024,
        max_samples: int = None,
        seed: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.max_samples = max_samples
        self.seed = seed

        # Load the preprocessed binary file
        data_file = self._get_data_file_path()
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Memory-map the file
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        log.info(f"Loaded {split} data from {data_file} with {len(self.data)} tokens")

        # Apply max_samples limit if specified
        if self.max_samples is not None:
            self._apply_max_samples()

    def _get_data_file_path(self) -> str:
        """Get the path to the binary data file. Override if different naming convention."""
        return os.path.join(self.data_dir, f'{self.split}.bin')

    def _apply_max_samples(self):
        """Apply max_samples limit by randomly sampling blocks"""
        max_tokens = self.max_samples * self.block_size
        if len(self.data) > max_tokens:
            # Randomly sample non-overlapping blocks using the seed
            rng = np.random.RandomState(self.seed)
            num_blocks = len(self.data) // self.block_size
            max_blocks = self.max_samples
            if num_blocks > max_blocks:
                # Select random block indices
                selected_blocks = rng.choice(num_blocks, size=max_blocks, replace=False)
                selected_blocks = np.sort(selected_blocks)  # Sort to maintain some locality

                # Create index array for all tokens in selected blocks (vectorized)
                block_starts = selected_blocks * self.block_size
                indices = (block_starts[:, np.newaxis] + np.arange(self.block_size)).ravel()

                # Use fancy indexing to extract all selected tokens at once
                self.data = np.array(self.data[indices], dtype=np.uint16)

    def __len__(self):
        """Number of complete sequences we can extract"""
        return len(self.data) // (self.block_size + 1)

    def __getitem__(self, idx):
        """Get a sequence and its target"""
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        # Get a chunk of block_size + 1 tokens
        start_idx = idx * (self.block_size + 1)
        input_ids = torch.from_numpy(self.data[start_idx:start_idx + self.block_size].astype(np.int64)).clone()
        targets = torch.from_numpy(self.data[start_idx + 1:start_idx + self.block_size + 1].astype(np.int64)).clone()
        return input_ids, targets

### ===== OpenWebText Dataset ===== ###
class OpenWebTextDataset(Dataset):
    """
    Dataset for OpenWebText language modeling using binarized tokenized files
    """
    def __init__(self, data_dir, split, block_size, max_samples=None, seed=0):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.max_samples = max_samples
        
        # Load the binarized data
        if split == 'train':
            data_path = os.path.join(data_dir, 'train.bin')
        else:  # validation
            data_path = os.path.join(data_dir, 'val.bin')
            
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Binarized data file not found: {data_path}")
        
        # Load data using numpy memmap for memory efficiency
        full_data = np.memmap(data_path, dtype=np.uint16, mode='r')
        log.info(f"Loaded {split} data with {len(full_data)} tokens from {data_path}")
        
        # Apply max_samples limit if specified
        if self.max_samples is not None:
            # Randomly sample non-overlapping blocks using the seed
            rng = np.random.RandomState(seed)
            num_blocks = len(full_data) // self.block_size
            max_blocks = self.max_samples
            if num_blocks > max_blocks:
                # Select random block indices
                selected_blocks = rng.choice(num_blocks, size=max_blocks, replace=False)
                selected_blocks = np.sort(selected_blocks)  # Sort to maintain some locality
                
                # Extract the selected blocks into a contiguous array
                selected_data = []
                for block_idx in selected_blocks:
                    start_idx = block_idx * self.block_size
                    end_idx = start_idx + self.block_size
                    selected_data.append(full_data[start_idx:end_idx])
                
                self.data = np.concatenate(selected_data)
                self.selected_blocks = None
                log.info(f"Limited {split} data to {self.max_samples} samples using block sampling")
            else:
                self.data = full_data
                self.selected_blocks = None
        else:
            self.data = full_data
            self.selected_blocks = None
        
    def __len__(self):
        return len(self.data) // self.block_size
    
    def __getitem__(self, idx):
        # Sample a block of tokens
        start_idx = idx * self.block_size
        input_ids = torch.from_numpy(self.data[start_idx:start_idx + self.block_size].astype(np.int64))
        targets = torch.from_numpy(self.data[start_idx + 1:start_idx + self.block_size + 1].astype(np.int64))
        return input_ids, targets

### ===== C4 Dataset ===== ###
class C4Dataset(BaseBinaryLanguageDataset):
    """C4 dataset using binary file format"""
    def __init__(self,
        data_dir: str,
        split: str = 'train',
        block_size: int = 1024,
        max_samples: int = None,
        seed: int = 0
    ):
        super().__init__(data_dir, split, block_size, max_samples, seed)

    # Inherits all methods from BaseBinaryLanguageDataset

def get_c4_dataset(split: str = 'train', subset: str = 'en', streaming: bool = True, trust_remote_code: bool = True):
    dataset = load_dataset("allenai/c4", data_dir=subset, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
    return dataset


class BaseSequenceDataset(Dataset):
    """
    Base class for sequence datasets with padding logic.
    Handles truncation and padding to max_seq_len.
    """
    def __init__(self, max_seq_len: int = 1024):
        super().__init__()
        self.max_seq_len = max_seq_len

    def _pad_or_truncate(self, seq: torch.Tensor, targets: torch.Tensor):
        """Apply padding or truncation to sequence and targets"""
        if seq.shape[0] > self.max_seq_len:
            # Truncate
            seq = seq[:self.max_seq_len]
            targets = targets[:self.max_seq_len]
        elif seq.shape[0] < self.max_seq_len:
            # Pad
            pad_len = self.max_seq_len - seq.shape[0]
            zero_padding = torch.full((pad_len,), 0, dtype=seq.dtype)
            target_padding = torch.full((pad_len,), -100, dtype=targets.dtype)
            seq = torch.cat([seq, zero_padding], dim=0)
            targets = torch.cat([targets, target_padding], dim=0)

        return seq, targets


### ===== BIGBENCH DATASET ===== ###
MC_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

BIGBENCH_LITE = tasks = [
    "auto_debugging",
    "bbq_lite_json",
    "code_line_description",
    "conceptual_combinations",
    "conlang_translation",
    "emoji_movie",
    "formal_fallacies_syllogisms_negation",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "linguistics_puzzles",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions_russian",
    "novel_concepts",
    "operators",
    "parsinlu_reading_comprehension",
    "play_dialog_same_or_different",
    "repeat_copy_logic",
    "strange_stories",
    "strategyqa",
    "symbol_interpretation",
    "vitaminc_fact_verification",
    "winowhy"
]

BIGBENCH_LITE_ENGLISH = tasks = [
    "auto_debugging", # < 100
    "bbq_lite_json", # 16.1k
    "code_line_description", # 60
    "conceptual_combinations", # 84
    "emoji_movie", # <100
    "formal_fallacies_syllogisms_negation", # 11.4k
    "known_unknowns", #  < 100
    "logic_grid_puzzle", # 800
    "logical_deduction", # 1.2k
    "novel_concepts", # 32 rows
    "operators", # 168
    "play_dialog_same_or_different", # 2.61k
    "repeat_copy_logic", # < 100
    "strange_stories", # 140
    "strategyqa", # 1.83k
    "symbol_interpretation", # 795
    "vitaminc_fact_verification", #43.7k
    "winowhy" # 2.29k
]

class BigBenchDataset(Dataset):
    def __init__(self,
        dataset=None,
        split='train',
        subsets: List[str] = BIGBENCH_LITE_ENGLISH,
        tokenizer: Tokenizer = None,
        shot: List[int] = [0, 4], # range of additional examples to add to the prompt
        train_enable: bool = False,
        seq_len: int = 1024,
        eval: bool = False,
        seed: int = 42,
        few_shot_prompts_path: str = None,
        mc_letter_format: bool = False,  # For logprob MC eval: use A/B/C/D letters instead of full text
    ):
        super().__init__()

        self.shot = shot
        self.train_enable = train_enable
        self.eval = eval
        self.seq_len = seq_len
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mc_letter_format = mc_letter_format

        if dataset is None:
            self.dataset = get_bigbench_dataset(split=split, subsets=subsets)
            self.subsets = subsets
        else:
            self.dataset = dataset
            self.subsets = list(self.dataset.keys())
        self.tokenizer = tokenizer if tokenizer is not None else tiktoken.get_encoding("gpt2")
        if tokenizer is None: self.eos = self.tokenizer.eot_token
        else: self.eos = self.tokenizer.eos_token_id

        # Load fixed few-shot prompts if provided
        self.few_shot_prompts = None
        if few_shot_prompts_path is not None:
            with open(few_shot_prompts_path, 'r') as f:
                self.few_shot_prompts = json.load(f)
            log.info(f"Loaded fixed few-shot prompts from {few_shot_prompts_path}")
            for task_name in self.subsets:
                n = len(self.few_shot_prompts.get(task_name, []))
                if n > 0:
                    log.info(f"  {task_name}: {n} fixed few-shot examples")
                else:
                    log.info(f"  {task_name}: no fixed prompts, will use random sampling")

        # preprocessing
        self.enumerated_examples = [] # list of examples in format of (task_name, example_idx)
        for subset in self.subsets:
            for example in self.dataset[subset]:
                self.enumerated_examples.append((subset, example))

    def get_num_choices(self, example):
        """Return number of multiple choice options (0 if free response)."""
        return len(example.get('multiple_choice_targets', []))

    def get_correct_answers(self, example):
        """Return list of all correct answers for this example.
        Uses multiple_choice_targets/scores when available, falls back to targets.
        When mc_letter_format=True, returns letter labels (A/B/C/D) instead of full text."""
        mc_targets = example.get('multiple_choice_targets', [])
        mc_scores = example.get('multiple_choice_scores', [])
        if mc_targets and mc_scores:
            if self.mc_letter_format:
                correct = [
                    MC_LETTERS[i]
                    for i, (t, s) in enumerate(zip(mc_targets, mc_scores))
                    if s > 0 and i < len(MC_LETTERS)
                ]
            else:
                correct = [t for t, s in zip(mc_targets, mc_scores) if s > 0]
            if correct:
                return correct
        return example['targets']

    def get_question(self, example, target: bool = False):
        # TODO: Change formatting to remove the Question\n and Answer\n from the prompt

        prompt = example['inputs']
        prompt = prompt.replace("Question:", "").replace("Q:", "").replace("Answer:", "").replace("A:", "")
        # Pre-Process by removing any references to Question:; Q:; Answer:; A:; from the prompt
        prompt = prompt.split("choice:")[0] # Remove the choices from the prompt

        choices = example['multiple_choice_targets']

        if len(choices) > 0 and self.mc_letter_format:
            prompt = f"{prompt}\n"
            for i, choice in enumerate(choices):
                if i < len(MC_LETTERS):
                    prompt = f"{prompt}\n{MC_LETTERS[i]}. {choice}"
        elif len(choices) > 0:
            prompt = f"{prompt}\n"
            for choice in choices:
                prompt = f"{prompt}\nchoice: {choice}"

        prompt = f"Question:\n{prompt}\n\nAnswer:\n"
        encoded = self.tokenizer.encode_ordinary(prompt)
        return torch.tensor(encoded, dtype=torch.long) if not target else torch.full((len(encoded), ), -100)

    def get_answer(self, example, eval: bool = False, target: bool = False):
        # get_correct_answers returns letters when mc_letter_format=True, full text otherwise
        correct_answers = self.get_correct_answers(example)
        answer = correct_answers[self.rng.integers(0, len(correct_answers))]

        formatted_answer = f"{answer}\n\n" if not(eval) else ""
        encoded = self.tokenizer.encode_ordinary(formatted_answer)

        return torch.tensor(encoded, dtype=torch.long) if not target else torch.full((len(encoded), ), -100)

    def get_example(self, example, eval: bool = False, target: bool = False):
        question = self.get_question(example, target=target)

        if eval:
            return question
        else:
            answer = self.get_answer(example, target=target)
            return torch.cat([question, answer], dim=0)

    def __len__(self):
        return sum(len(self.dataset[subset]) for subset in self.subsets)

    def get_category(self, idx):
        subset, example = self.enumerated_examples[idx]
        return subset

    def _encode_fixed_prompts(self, subset):
        """Encode fixed few-shot prompts for a given task subset.

        Prompts are stored in HuggingFace BigBench format (with inputs,
        targets, multiple_choice_targets, multiple_choice_scores fields)
        and are processed using get_example, the same as regular dataset
        examples.
        """
        prompts = self.few_shot_prompts.get(subset, [])
        encoded = []
        for prompt_example in prompts:
            tokens = self.get_example(prompt_example, eval=False, target=False)
            encoded.append(tokens)
        return encoded

    def __getitem__(self, idx):
        # sample questions and answers
        subset, example = self.enumerated_examples[idx]
        example_idx = example['idx']

        # Use fixed few-shot prompts if available for this task
        # When enabled, ALL prompts from the file are always used (no random sampling)
        if self.few_shot_prompts is not None and subset in self.few_shot_prompts:
            fixed_prompts = self._encode_fixed_prompts(subset)
            seq = fixed_prompts.copy()
            tar = [torch.full((len(p),), -100) for p in fixed_prompts]
            seq.append(self.get_question(example, target=False))
            tar.append(self.get_question(example, target=True))
        else:
            # sample additional examples from dataset
            num_shot = self.rng.integers(low=self.shot[0], high=self.shot[1] + 1)

            while True:
                idxs = self.rng.choice(len(self.dataset[subset]), size=num_shot, replace=False)
                examples = [self.dataset[subset][int(i)] for i in idxs]
                if all([c['idx'] != example_idx for c in examples]):
                    break

            # encode additional examples
            seq = [self.get_example(c, target=False) for c in examples]
            tar = [self.get_example(c, target=True) for c in examples]
            seq.append(self.get_question(example, target=False))
            tar.append(self.get_question(example, target=True))

        seq.append(self.get_answer(example, eval=self.eval))
        # final answer is unmasked (target=False) even when few-shot answers are masked
        tar.append(self.get_answer(example, eval=self.eval, target=False))

        # append EOT after the final answer when not in eval mode
        if not self.eval:
            eot = torch.tensor([self.eos], dtype=torch.long)
            seq.append(eot)
            tar.append(eot)

        if self.eval:
            seq = torch.cat(seq, dim=0)
            tar = self.get_answer(example)
            num_choices = self.get_num_choices(example)
            correct_answers = self.get_correct_answers(example)
            return seq, tar, num_choices, correct_answers

        seq = torch.cat(seq, dim=0)
        tar = torch.cat(tar, dim=0)
        if seq.shape[0] > self.seq_len:
            seq = seq[:self.seq_len]
            tar = tar[1:self.seq_len+1]
        else:
            tar = tar[1:]
            zero_padding = torch.full((self.seq_len - seq.shape[0],), 0, dtype=seq.dtype)
            seq = torch.cat([seq, zero_padding], dim=0)
            padding = torch.full((self.seq_len - tar.shape[0],), -100, dtype=tar.dtype)
            tar = torch.cat([tar, padding], dim=0)

        return seq, tar

def get_bigbench_dataset(
    split: str = 'train',
    subsets: List[str] = BIGBENCH_LITE_ENGLISH,
    min_samples: int = 100,
    max_samples: int = 300,
    seed: int = 42
):
    # tasksource/bigbench
    dataset = DatasetDict()
    max_samples = max_samples if max_samples is not None else int(1e11)
    for subset in subsets:
        # Check that training set has enough samples
        subset_dataset = load_dataset("tasksource/bigbench", subset, split="train", streaming=False, num_proc=16, trust_remote_code=True)
        if (len(subset_dataset) < min_samples):
            log.info(f"Skipping {subset} because it has less than {min_samples} samples")
            continue

        # Load the dataset for appropriate split
        subset_dataset = load_dataset("tasksource/bigbench", subset, split=split, streaming=False, num_proc=16, trust_remote_code=True)
        dataset[subset] = subset_dataset.shuffle(seed=seed)
        if len(dataset[subset]) > max_samples:
            dataset[subset] = dataset[subset].select(range(max_samples))
    return dataset

### ===== LANGUAGE TASK DATASET Classes ===== ###

class FullCodeParrotLanguageDataset(BaseBinaryLanguageDataset):
    """Full CodeParrot dataset using binary file format"""
    def __init__(self,
        data_dir: str,
        split: str = 'train',
        block_size: int = 1024,
        max_samples: int = None,
        seed: int = 0
    ):
        super().__init__(data_dir, split, block_size, max_samples, seed)

    def _get_data_file_path(self) -> str:
        """CodeParrot uses 'test.bin' instead of 'val.bin'"""
        if self.split == 'train':
            return os.path.join(self.data_dir, 'train.bin')
        else:
            return os.path.join(self.data_dir, 'test.bin')

    # Inherits __len__() and __getitem__() from BaseBinaryLanguageDataset
        
class MathLanguageDataset(BaseBinaryLanguageDataset):
    """Math dataset using binary file format"""
    def __init__(self,
        data_dir: str,
        split: str = 'train',
        block_size: int = 1024,
        max_samples: int = None,
        seed: int = 0
    ):
        super().__init__(data_dir, split, block_size, max_samples, seed)

    # Inherits all methods from BaseBinaryLanguageDataset

class CodeParrotLanguageDataset(IterableDataset):
    def __init__(self,
        tokenizer,
        dataset,
        infinite: bool = False,
        seq_len: int = 1024,
        num_sequences: int = 1024,
        chars_per_token: float = 3.6,
        seed: int = 42
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        #self.dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        self.dataset = dataset
        self.infinite = infinite
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.epoch = 0
        self.input_chars = (seq_len + 1) * chars_per_token * num_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True

        while more_examples:
            buffer, buffer_len = [], 0

            # Fill up character buffer
            while buffer_len < self.input_chars:
                try:
                    text = next(iterator)["content"]
                    buffer.append(text)
                    buffer_len += len(text)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                    else:
                        more_examples = False
                        break

            if not buffer:
                break

            # Tokenize in batch (faster than per-sample)
            tokenized_batch = self.tokenizer(
                buffer,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            # Flatten and add EOS between docs
            all_token_ids = []
            for tokens in tokenized_batch:
                all_token_ids.extend(tokens)
                all_token_ids.append(self.concat_token_id)

            token_tensor = torch.tensor(all_token_ids, dtype=torch.long)

            # Chop into fixed-length chunks
            num_chunks = len(token_tensor) // (self.seq_len + 1)
            token_tensor = token_tensor[: num_chunks * (self.seq_len + 1)]
            token_tensor = token_tensor.view(num_chunks, self.seq_len + 1)

            for seq in token_tensor:
                x = seq[:-1]
                y = seq[1:]
                yield x, y


class LanguageTaskDataset(BaseSequenceDataset):
    """Generic language task dataset with tokenization"""
    def __init__(self,
        sequences: List[str] = None,
        targets: List[str] = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = 1024,
        dataset: Tuple[torch.tensor, torch.tensor] = None
        ):
        super().__init__(max_seq_len)
        self.text_sequences = sequences
        self.text_targets = targets
        self.tokenizer = tokenizer

        if dataset is not None:
            self.sequences, self.targets = dataset
        else:
            if self.text_targets is not None:
                self.sequences, self.targets = self.tokenizer.encode_task(self.text_sequences, self.text_targets)
            else:
                self.sequences, self.targets = self.tokenizer.encode_task(self.text_sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        targets = self.targets[idx]

        # Use inherited padding logic
        seq, targets = self._pad_or_truncate(seq, targets)

        # Ensure tensors are long type (avoid double conversion if already tensors)
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq).long()
        else:
            seq = seq.long()

        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets).long()
        else:
            targets = targets.long()

        return seq, targets

# Champ - Champernowne Constant (generating digits from successive integers)
def int_to_digits(arr: torch.Tensor) -> torch.Tensor:
    """
    Convert a 1D tensor of ints into a flat tensor of decimal digits [0-9].
    Example: [12, 305] -> [1, 2, 3, 0, 5]
    """
    # Convert each int to string, then join and map back to digits
    # This uses Python list comprehension but outputs a torch tensor
    s = ''.join(str(x.item()) for x in arr)
    digits = torch.tensor([int(ch) for ch in s], dtype=torch.long)
    return digits

class ChampTaskDataset(BaseSequenceDataset):
    """Champernowne constant dataset (successive integer digits)"""
    def __init__(
        self,
        n_bit: int = 24,
        n: int = 255,
        starting_numbers=None,
        max_seq_len: int = 1024,
    ):
        super().__init__(max_seq_len)
        self.n_bit = n_bit
        self.n = n
        self.starting_numbers = starting_numbers

    def __len__(self):
        return len(self.starting_numbers)

    def __getitem__(self, idx):
        starting_number = self.starting_numbers[idx]
        return self.make_sequence(starting_number, self.n, self.max_seq_len)

    def make_sequence(self, start, n, seq_len):
        """Generate Champernowne sequence from start"""
        # Generate consecutive integers [start, start+n)
        ints = torch.arange(start, start + n, dtype=torch.long)

        # Convert to digit sequence
        digits = int_to_digits(ints)

        seq = digits[:-1]
        target = digits[1:]

        # Use inherited padding logic
        seq, target = self._pad_or_truncate(seq, target)

        return seq, target

### === Task Data Generators ===== ###

def generate_champ_dataset(
    seed: int = 42,
    n_bit: int = 24,
    n: int = 255,
    num_train: int = 1000,
    num_val: int = 100,
    seq_len: int = 1024
):
    rng = np.random.RandomState(seed)
    max_int = 2**n_bit - n # max int to generate for successive integers

    starts = rng.randint(0, max_int, num_train + num_val)

    def make_sequence(start):
        ints = np.arange(start, start + n, dtype=np.int64)
        digits = int_to_digits(ints)

        if digits.shape[0] < seq_len + 1:
            digits = np.pad(digits, (0, seq_len + 1 - digits.shape[0]), mode='wrap')
        else:
            digits = digits[:seq_len + 1]
        return digits[:-1], digits[1:]

    # Preallocate arrays
    train_seq = np.zeros((num_train, seq_len), dtype=np.int8)
    train_tgt = np.zeros((num_train, seq_len), dtype=np.int8)
    val_seq   = np.zeros((num_val, seq_len), dtype=np.int8)
    val_tgt   = np.zeros((num_val, seq_len), dtype=np.int8)

    for i, start in enumerate(starts[:num_train]):
        train_seq[i], train_tgt[i] = make_sequence(start)
    for i, start in enumerate(starts[num_train:]):
        val_seq[i], val_tgt[i] = make_sequence(start)

    return (train_seq, train_tgt), (val_seq, val_tgt)

# Dyck - K-Dyck Lanaguage Task
# Adapted from Pre-Pretraining Language Models 

def generate_dyck(
    rng: np.random.RandomState,
    num_symbols: int,
    min_depth: int,
    max_depth: int,
    max_length: int,
    offset: int = None
):
    result = []
    stack = []

    assert min_depth <= max_depth and min_depth > 0, "Invalid depth range"
    
    if offset is None:
        offset = num_symbols
    
    # initialize minimum depth
    for _ in range(min_depth):
        opening_symbol = rng.randint(0, num_symbols)
        result.append(opening_symbol)
        stack.append(opening_symbol)
    
    # generate random depth
    while len(result) < max_length:
        if (
            len(stack) < max_depth and rng.random() < 0.5
        ):
            if (len(result) >= max_length - 1):
                closing_symbol = stack.pop() + offset
                result.append(closing_symbol)
                continue
            opening_symbol = rng.randint(0, num_symbols)
            result.append(opening_symbol)
            stack.append(opening_symbol)
        else:
            closing_symbol = stack.pop() + offset
            result.append(closing_symbol)
            if not stack:
                break
    
    # pop remaining unclosed symbols
    while stack:
        closing_symbol = stack.pop() + offset
        result.append(closing_symbol)

    return result if not stack else None

def generate_shuffle_dyck(
    rng: np.random.RandomState,
    num_symbols: int, # k in K-Dyck
    max_length: int = 1024,
    p_open: float = 0.5,
    max_depth: int = 16
):
    sequence = []
    counts = [0]*num_symbols

    while len(sequence) < max_length:
        depth = sum(counts)

        # Must open if all brackets are closed
        if depth == 0:
            bracket = rng.randint(0, num_symbols - 1)
            sequence.append(bracket)
            counts[bracket] += 1
            continue

        # If at max depth, force a close
        if depth >= max_depth:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = rng.choice(open_brackets)
            sequence.append(bracket + num_symbols)
            counts[bracket] -= 1
            continue

        # Randomly choose to open or close
        if rng.random() < p_open and depth < max_depth:
            bracket = rng.randint(0, num_symbols - 1)
            sequence.append(bracket)
            counts[bracket] += 1
        else:
            # Close an existing bracket
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket = rng.choice(open_brackets)
            sequence.append(bracket + num_symbols)
            counts[bracket] -= 1

    return sequence

def generate_shuffle_dyck_txt_file(
    seed:int,
    file_dir, 
    num_symbols=64,
    n=100000,
    target_length=2048,
    p=0.5
):
    """Generates a text file containing Dyck sequences with cross-serial dependencies.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        n: The number of sequences to generate.
        target_length: desired sequence length.
    """
    rng = np.random.RandomState(seed)

    os.makedirs(file_dir, exist_ok=True)
    file_path = f"{file_dir}/dyck_sequences_cross_serial_{num_symbols}_{p}.txt"
    with open(
        file_path, "w"
    ) as f:
        for i in range(n):
            result = generate_shuffle_dyck(rng, num_symbols, target_length, p)
            dyck_str = " ".join([str(x) for x in result[:target_length]])
            f.write(f"{dyck_str}\n")

    return file_path

def generate_dyck_txt_file(
    seed:int,
    file_dir,
    num_symbols=30,
    n=100000,
    target_length=2048,
    min_depth=1,
    max_depth=16,
):
    """Generates a text file containing Dyck sequences.

    Args:
        file_dir: The directory to save the file.
        num_symbols: The number of distinct symbol pairs (k in k-Dyck).
        n: The number of sequences to generate.
        target_length: Target length of each sequence.
        min_depth: Minimum required depth of nested brackets.
        max_depth: Maximum allowed depth of nested brackets.
    """
    rng = np.random.RandomState(seed)
    file_path = f"{file_dir}/dyck_sequences_{num_symbols}_{min_depth}_{max_depth}.txt"
    print(f"Saving to {file_path}")

    os.makedirs(file_dir, exist_ok=True)
    with open(
        file_path, "w"
    ) as f:
        for i in range(n):
            result = []
            while len(result) < target_length:

                new_seq = generate_dyck(
                    rng,
                    num_symbols,
                    min_depth=min_depth,
                    max_depth=max_depth, 
                    max_length=target_length
                )
                if new_seq is None:
                    continue
                result.extend(new_seq)

            dyck_str = " ".join(
                [str(x) for x in result[:target_length]]
            )  # truncate to target length
            assert len(dyck_str.split()) == target_length, "Sequence length is not correct"
            f.write(f"{dyck_str}\n")
    
    return file_path

def read_dyck_txt_file(
    file_path: str
):
    with open(file_path, 'r') as f:
        data = [np.array(line.strip().split(), dtype=np.int16) for line in f.readlines()]
        data = np.stack(data)
        return data

def generate_dyck_dataset(
    file_path: str,
    num_train: int,
    num_val: int,
    seq_len: int
):
    sequences = read_dyck_txt_file(file_path)
    assert sequences.shape[1] >= seq_len + 1, "Sequence length is not correct"
    train_sequences = sequences[:num_train, :seq_len]
    train_targets = sequences[:num_train, 1:seq_len+1]
    val_sequences = sequences[num_train:num_train+num_val, :seq_len]
    val_targets = sequences[num_train:num_train+num_val, 1:seq_len+1]
    return (train_sequences, train_targets), (val_sequences, val_targets)

def compute_k_dyck_metrics(
    sequences: torch.Tensor,     # ground truth: (B, N)
    targets: torch.Tensor,       # ground truth: (B, N)
    predictions: torch.Tensor,   # model outputs: (B, N)
    num_symbols: int,            # number of opening symbols
    max_depth: int               # max nesting depth allowed
):
    B, N = sequences.shape
    total_tokens = B * N

    token_correct = 0
    cond_valid_tokens = 0

    for i in range(B):
        # --- Conditional validity ---
        stack = []
        for j in range(N):
            # update stack using *ground truth*
            if sequences[i, j] < num_symbols:
                stack.append(sequences[i, j].item())
            else:
                if stack and stack[-1] == sequences[i, j].item() - num_symbols:
                    stack.pop()

            # check if prediction is valid given this stack
            pred_tok = predictions[i, j].item()
            if pred_tok < num_symbols and len(stack) < max_depth:
                cond_valid_tokens += 1
            elif stack and stack[-1] == pred_tok - num_symbols:
                cond_valid_tokens += 1

    return {
        "token_accuracy": torch.sum(targets == predictions).item() / total_tokens,
        "conditional_validity": cond_valid_tokens / total_tokens,
    }

def compute_k_shuffle_dyck_metrics(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_symbols: int,
    max_depth: int
):
    B, N = sequences.shape
    total_tokens = B*N
    token_correct = 0
    cond_valid_tokens = 0

    for i in range(B):
        counts = [0]*num_symbols
        for j in range(N):
            # --- Token accuracy ---
            token_correct += (predictions[i, j] == sequences[i, j]).sum().item()

            # --- Conditional validity ---
            if sequences[i, j] < num_symbols and counts[sequences[i, j]] < max_depth:
                counts[sequences[i, j]] += 1
            else:
                if counts[sequences[i, j] - num_symbols] > 0:
                    counts[sequences[i, j] - num_symbols] -= 1
            
            if predictions[i, j] < num_symbols and counts[predictions[i, j]] < max_depth:
                cond_valid_tokens += 1
            elif counts[predictions[i, j] - num_symbols] > 0:
                cond_valid_tokens += 1
    
    return {
        "token_accuracy": torch.sum(targets == predictions).item() / total_tokens,
        "conditional_validity": cond_valid_tokens / total_tokens
    }

# ==== Saving and Loading Coding Datasets ===== # 
def build_codeparrot_dataset(
    save_path: str,
    split: str = 'val',
    shard_enable: bool = False,
    shard_max: int = int(1e8)
):
    if split == 'train':
        dset = load_dataset("codeparrot/codeparrot-train-v2-near-dedup", split='train', streaming=True)
    else:
        dset = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup", split='train', streaming=True)
    
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab # 50257
    eot_token = enc.eot_token
    shard_idx = 0
    total_tokens = 0
    
    tok = []
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for doc in dset:
        seq = enc.encode(doc['content'], disallowed_special=())
        seq.append(eot_token)
        tok.extend(seq)

        if(shard_enable and len(tok) >= shard_max):
            arr = np.array(tok, dtype=np.uint16)
            fname = os.path.join(save_path, f"{split}_shard_{shard_idx:05d}.bin")
            arr.tofile(fname)
            print(f"wrote {len(arr):,} tokens to {fname}")
            total_tokens += len(arr)
            shard_idx += 1
            tok = []
    
    tok = np.array(tok, dtype=np.uint16)
    print(f'Total tokens: {len(tok)}')
    print(f'Total tokens: {total_tokens}')

    if shard_enable:
        tok.tofile(os.path.join(save_path, f'{split}_shard_{shard_idx:05d}.bin'))
    else:
        tok.tofile(os.path.join(save_path, f'{split}.bin'))

    print(f'Saved to {os.path.join(save_path, f"{split}.bin")}')
    return tok

### ===== Math Datasets ===== ###

class BaseMathDataset(BaseSequenceDataset):
    """
    Base class for math question-answering datasets with in-context learning.
    Handles tokenization, prompt formatting, and ICL example sampling.
    """
    def __init__(self,
        tokenizer,
        dataset,
        seq_len: int = 1024,
        seed: int = 0,
        num_icl_examples: int = 5,
        hf_tokenizer: bool = False,
        stop_string: str = '####'
    ):
        super().__init__(max_seq_len=seq_len)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hf_tokenizer = hf_tokenizer
        self.stop_string = stop_string
        self.num_icl_examples = num_icl_examples

        # Set EOS token based on tokenizer type
        if self.hf_tokenizer:
            self.eos = tokenizer.eos_token_id
        else:
            self.eos = tokenizer.eot_token

    def __len__(self):
        return len(self.dataset)

    def _encode_text(self, text: str) -> list:
        """Encode text using the appropriate tokenizer"""
        if self.hf_tokenizer:
            return self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]
        else:
            return self.tokenizer.encode_ordinary(text)

    def process_question(self, question_text: str, mask: bool = False):
        """Process question text into tokens"""
        prompt_text = f"Question:\n{question_text}\n"
        question = self._encode_text(prompt_text)
        question = torch.tensor(question, dtype=torch.long)
        if mask:
            question = torch.full(question.shape, -100, dtype=question.dtype)
        return question

    def process_answer(self, answer_text: str, include_explanation: bool = True):
        """Process answer text into tokens - override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_answer()")

    def process_example(self, example, mask: bool = False):
        """Process a full example (question + answer)"""
        raise NotImplementedError("Subclasses must implement process_example()")

    def sample_contextual_examples(self, test_question: str, num_examples: int):
        """Sample random contextual examples excluding the test question"""
        while True:
            contextual_examples = self.rng.choice(
                self.dataset,
                size=num_examples,
                replace=False
            ).tolist()
            if all([c['question'] != test_question for c in contextual_examples]):
                return contextual_examples


class GSM8KDataset(BaseMathDataset):
    """GSM8K math reasoning dataset"""
    def __init__(self,
        tokenizer,
        dataset,
        seq_len: int = 1024,
        seed: int = 0,
        num_icl_examples: int = 5,
        hf_tokenizer: bool = False,
        stop_string: str = '####'
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset=dataset,
            seq_len=seq_len,
            seed=seed,
            num_icl_examples=num_icl_examples,
            hf_tokenizer=hf_tokenizer,
            stop_string=stop_string
        )

    def process_example(self, example, mask=False):
        question_text = example['question']
        question = self.process_question(question_text, mask=mask)
        answer = self.process_answer(example['answer'])
        return torch.cat([question, answer], dim=0)

    def process_answer(self, answer, include_explanation: bool = True):
        # preprocess the text:
        chain_of_thought = answer.split(self.stop_string)[0].strip()
        final_answer = answer.split(self.stop_string)[1].strip()

        # encode the answer
        if include_explanation:
            prompt = f"Answer:\n{chain_of_thought}\nFinal Answer: {final_answer}"
        else:
            prompt = f"Answer:\n{final_answer}"

        answer = self._encode_text(prompt)
        answer.append(self.eos)
        return torch.tensor(answer, dtype=torch.long)

    def __getitem__(self, idx):
        # sample test question and answer
        test_question = self.dataset[idx]['question']
        test_answer = self.dataset[idx]['answer']

        # sample random contextual examples
        contextual_examples = self.sample_contextual_examples(
            test_question,
            self.num_icl_examples
        )

        # encode contextual examples
        contextual_examples = [self.process_example(c) for c in contextual_examples]
        contextual_examples.append(self.process_question(test_question))
        seq = torch.cat(contextual_examples, dim=0)
        target = self.process_answer(test_answer)
        return seq, target


class GSM8KTrainDataset(GSM8KDataset):
    """GSM8K training dataset with autoregressive formatting"""
    def __getitem__(self, idx):
        test_question = self.dataset[idx]['question']

        # sample random contextual examples
        contextual_examples = self.sample_contextual_examples(
            test_question,
            self.num_icl_examples
        )

        contextual_examples.append(self.dataset[idx])
        seq_examples = [self.process_example(c) for c in contextual_examples]
        tar_examples = [self.process_example(c, mask=True) for c in contextual_examples]

        seq = torch.cat(seq_examples, dim=0)
        target = torch.cat(tar_examples, dim=0)

        # Use inherited padding logic with autoregressive shift
        if seq.shape[0] > self.max_seq_len:
            seq = seq[:self.max_seq_len]
            target = target[1:self.max_seq_len+1]
        else:
            target = target[1:]
            seq, target = self._pad_or_truncate(seq, target)

        return seq.long(), target.long()


class MetaMathQADataset(BaseMathDataset):
    """MetaMathQA dataset"""
    def __init__(self,
        tokenizer,
        dataset,
        seq_len: int = 1024,
        seed: int = 0,
        num_icl_examples: int = 5,
        hf_tokenizer: bool = False,
        stop_string: str = '####',
        prompt_only: bool = False
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset=dataset,
            seq_len=seq_len,
            seed=seed,
            num_icl_examples=num_icl_examples,
            hf_tokenizer=hf_tokenizer,
            stop_string=stop_string
        )
        self.prompt_only = prompt_only

    def process_question(self, example, mask=False):
        """Override to use 'query' field"""
        question_text = example['query'] if isinstance(example, dict) else example
        prompt_text = f"Question:\n{question_text}\n"
        question = self._encode_text(prompt_text)
        question = torch.tensor(question, dtype=torch.long)
        if mask:
            question = torch.full(question.shape, -100, dtype=question.dtype)
        return question

    def process_answer(self, example, include_explanation: bool = True, prompt_only: bool = False):
        # preprocess
        answer = example['response']
        chain_of_thought = answer.split(self.stop_string)[0].split("The answer is:")[0].strip()
        final_answer = answer.split("The answer is:")[-1].strip()

        # encode the answer
        if prompt_only:
            prompt = "Answer:"
        elif include_explanation:
            prompt = f"Answer:\n{chain_of_thought}\nFinal Answer: {final_answer}"
        else:
            prompt = f"Answer:\n{final_answer}"

        answer = self._encode_text(prompt)

        # append EOS
        if not prompt_only:
            answer.append(self.eos)
        return torch.tensor(answer, dtype=torch.long)

    def process_example(self,
        example,
        mask: bool = False,
        include_explanation: bool = True,
        prompt_only: bool = False
    ):
        question = self.process_question(example, mask=mask)
        answer = self.process_answer(example, include_explanation=include_explanation, prompt_only=prompt_only)
        return torch.cat([question, answer], dim=0)

    def __getitem__(self, idx, include_explanation: bool = True, prompt_only: bool = False):
        # sample test question and answer
        test_question = self.dataset[idx]['query']
        test_example = self.dataset[idx]

        # sample random contextual examples
        while True:
            idxs = self.rng.choice(len(self.dataset), size=self.num_icl_examples, replace=False)
            contextual_examples = [self.dataset[int(i)] for i in idxs]
            if all([c['query'] != test_question for c in contextual_examples]):
                break

        # encode examples
        seq_examples = [self.process_example(c, include_explanation=include_explanation) for c in contextual_examples]
        tar_examples = [self.process_example(c, mask=True, include_explanation=include_explanation) for c in contextual_examples]

        seq_examples.append(self.process_example(test_example, include_explanation=include_explanation, prompt_only=self.prompt_only))
        tar_examples.append(self.process_example(test_example, mask=True, include_explanation=include_explanation, prompt_only=self.prompt_only))

        seq = torch.cat(seq_examples, dim=0)
        target = torch.cat(tar_examples, dim=0)

        # padding and truncation using inherited method
        if not self.prompt_only:
            if seq.shape[0] > self.max_seq_len:
                seq = seq[:self.max_seq_len]
                target = target[1:self.max_seq_len+1]
            else:
                target = target[1:]
                seq, target = self._pad_or_truncate(seq, target)

        return seq.long(), target.long()


def load_gsm8k_dataset(name: str = "main", split: str = "train"):
    """Load GSM8K dataset from HuggingFace"""
    return load_dataset(path="openai/gsm8k", name=name, split=split, streaming=False)


def load_metamathqa_dataset(split: str = "train"):
    """Load MetaMathQA dataset from HuggingFace"""
    return load_dataset(path="meta-math/MetaMathQA", split=split, streaming=False)


# Eval function for pass@k
def pass_at_k(n, c, k):
    """
    Unbiased estimate of the pass@k accuracy.
    Based on original eval LLMs on Code paper

    Args:
        n: total number of examples
        c: number of correct examples
        k: number of examples to consider
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ===== Main Methods ===== #

if __name__ == '__main__':
    shard_enable = sys.argv[1] == 'Train'
    build_codeparrot_dataset(
        save_path=sys.argv[1],
        split=sys.argv[2],
        shard_enable=shard_enable,
        shard_max=int(1e7)
    )