## Training Language Models via Neural Cellular Automata

This repository provides an implementation of the paper, [Training Language Models via Neural Cellular Automata]().

Pre-training large language models on natural language is costly, biased, and entangles knowledge with reasoning. We propose **NCA pre-pre-training**: first training a transformer on dynamics from neural cellular automata (NCA), then continuing with standard language pre-training. With only 164M NCA tokens, this improves downstream language modeling by up to 6% and accelerates convergence by up to 1.6× — outperforming 1.6B tokens of natural language (C4) as a pre-pre-training signal.

## Repository Structure

```
.
├── src/
│   ├── nca_ppt.py              - NCA pre-pre-training (data generation + transformer training)
│   ├── language_train.py       - Language pre-training and instruction fine-tuning (HF datasets)
│   ├── openwebtext_pt.py       - OpenWebText-specific pre-training
│   ├── datasets/
│   │   └── preprocess.py       - Dataset tokenization & preprocessing
│   └── eval/
│       ├── bigbench.py         - BigBench-Lite evaluation (pass@k, few-shot)
│       ├── humaneval.py        - HumanEval code generation evaluation (pass@k)
│       ├── gsm8k.py            - GSM8K math reasoning evaluation (pass@k)
│       └── bbl_prompts.json    - Few-shot prompts for BigBench-Lite
├── utils/
│   ├── nca.py                  - NCA model definitions (JAX/Flax)
│   ├── models.py               - Llama-based language model definitions
│   ├── dataset_utils.py        - Dataset loading & batching utilities
│   ├── tokenizers.py           - Tokenizer wrappers (tiktoken, JAX)
│   ├── training_args.py        - Shared training argument dataclasses
│   └── util.py                 - General helpers (seeding, logging, checkpointing)
├── scripts/
│   ├── prepretraining/
│   │   └── nca_prepretraining.sh       - Launch NCA pre-pre-training
│   ├── pretraining/
│   │   ├── owt_ft.sh                   - Launch OpenWebText pre-training
│   │   └── ft_codeparrot.sh            - Launch CodeParrot pre-training
│   ├── instruction-ft/
│   │   ├── ft_instruction_gsm8k.sh     - Fine-tune on GSM8K (math)
│   │   └── ft_instruction_bbl.sh       - Fine-tune on BigBench-Lite (reasoning)
│   └── eval/
│       ├── eval_gsm8k.sh               - Evaluate on GSM8K
│       ├── eval_humaneval.sh           - Evaluate on HumanEval
│       └── eval_bbl.sh                 - Evaluate on BigBench-Lite
├── README.md
├── requirements.txt            - pip dependencies
└── environment.yml             - Conda environment spec
```

## Setup

```bash
mamba env create -f environment.yml
mamba activate ai2
```

## Usage

### 1. NCA Pre-Pre-Training

NCA pre-pre-training generates synthetic data on-the-fly by sampling random NCA transition rules and rolling out their dynamics. The transformer is trained with next-token prediction on the serialized grid trajectories.

```bash
scripts/prepretraining/nca_prepretraining.sh
```

**Key hyperparameters:**

| Argument | Description | Default |
|---|---|---|
| `--num_colors` | NCA alphabet size (state space) | `10` |
| `--filter_rules_threshold` | Lower bound on gzip compression ratio for filtering rules (0–100%). Higher values select more complex, less compressible NCA dynamics. | `0.5` |
| `--filter_rules_upper_bound` | Upper bound on gzip compression ratio. Together with `--filter_rules_threshold`, this defines the complexity band (e.g. `0.5 1.0` = 50%+ band) | `1.0` |
| `--filter_rules_mode` | Complexity measure used for filtering. Use `gzip` (proxy for Kolmogorov complexity) | `gzip` |
| `--grid` | NCA grid size (H=W). Paper uses a 12×12 grid. | `12` |
| `--patch` | Patch size for tokenization (2×2 → 10⁴ vocabulary for num_colors = 10). | `2` |
| `--train_num_rules` | Number of unique NCA rules (i.e., distinct functions) in the training set. Default value is set to ensure every sampled trajectory is a different rule. | `16000` |
| `--generate_rules` | Re-sample the training rule set every N epochs (requires `--generate_train`). Setting to `1` draws a fresh set of rules each epoch; larger values reuse the same rules for longer before regenerating. | `1` |
| `--train_num_sim` | Number of trajectory simulations sampled per rule. | `500` |
| `--num_epochs` | Training epochs i.e. re-sampling of rules. | `100` |
| `--learning_rate` | Pre-pre-training learning rate. | `1e-4` |

**Targeting a downstream domain:** The optimal complexity band varies by domain. Use higher-complexity NCA (50%+ gzip) for web text and math; intermediate complexity (30–40% gzip) for code. Adjust `--filter_rules_threshold` and `--filter_rules_upper_bound` accordingly.

---

### 2. Language Pre-Training

After NCA pre-pre-training, transfer the model weights to a language pre-training run. Embedding layers are re-initialized for the natural language vocabulary; all other weights are carried over.

**OpenWebText:**
```bash
scripts/pretraining/owt_ft.sh
```

**CodeParrot:**
```bash
scripts/pretraining/ft_codeparrot.sh
```

Both scripts call into `src/language_train.py` or `src/openwebtext_pt.py` and accept a `--model_path` / `--model_file` pointing to the NCA pre-pre-trained checkpoint. Set `--reinit_modules embed none` to re-initialize the embedding layers while retaining all other weights. These scripts can also be used for pre-pretraining on other datasets.

**Key hyperparameters:**

| Argument | Description |
|---|---|
| `--model_path` / `--model_file` | Path to the NCA pre-pre-trained checkpoint directory and file. |
| `--reinit_modules embed none` | Re-initialize embedding layers; transfer all other weights |
| `--lr` | Pre-training learning rate. Default set to `5e-4` for math/web text, `2e-4` for code. |
| `--epochs` | Train for a single epoch over the corpus (standard for large-scale pre-training). |
| `--grad_accumulation_steps` | Effective batch size multiplier. Paper uses an effective batch size of 512. |

---

### 3. Downstream Fine-Tuning

For GSM8K and BigBench-Lite, fine-tune the pre-trained model on the respective training sets before evaluation. HumanEval is evaluated directly without fine-tuning (code completion).

**GSM8K (math reasoning):**
```bash
scripts/instruction-ft/ft_instruction_gsm8k.sh
```
Trains for 10 epochs on Chain-of-Thought traces at `lr=1e-5`.

**BigBench-Lite (reasoning):**
```bash
scripts/instruction-ft/ft_instruction_bbl.sh
```
Trains for 1 epoch at `lr=5e-6`. Tasks are sampled with `--min_samples 100 --max_samples 350` per category.

---

### 4. Evaluation

All evaluation scripts compute pass@k using the unbiased estimator from Chen et al. (2021) over 64 decodings per prompt. Results are saved to `--save_path`.

**GSM8K:**
```bash
scripts/eval/eval_gsm8k.sh
```

**HumanEval:**
```bash
scripts/eval/eval_humaneval.sh
```

**BigBench-Lite:**
```bash
scripts/eval/eval_bbl.sh
```

Common evaluation arguments:

| Argument | Description | Default |
|---|---|---|
| `--passes` | Total number of decodings per prompt | `64` |
| `--eval_passes` | Values of k to report pass@k | 1 2 4 8 16 32 |
| `--temperature` | Sampling temperature | `0.4`. |
| `--top_p` | Nucleus sampling threshold | `0.95`. |

---

## Citation

```bibtex
@misc{lee2026traininglanguagemodelsneural,
      title={Training Language Models via Neural Cellular Automata}, 
      author={Dan Lee and Seungwook Han and Akarsh Kumar and Pulkit Agrawal},
      year={2026},
      eprint={2603.10055},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.10055}, 
}
```
