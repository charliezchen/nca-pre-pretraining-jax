# Training Language Models via Neural Cellular Automata — JAX/TPU port

JAX + Flax rewrite of [Training Language Models via Neural Cellular Automata](https://arxiv.org/abs/2603.10055) (Lee et al.). Runs 100% on TPUs via `jax.sharding` data parallelism — no PyTorch, no conda.

Pre-training language models on natural language is costly, biased, and entangles knowledge with reasoning. **NCA pre-pre-training** first trains a transformer on dynamics from neural cellular automata, then continues with standard language pre-training. With 164M NCA tokens, the paper reports up to 6% downstream improvement and 1.6× faster convergence vs. 1.6B tokens of C4 as pre-pre-training.

## Repository layout

```
.
├── pyproject.toml          # uv-managed dependencies
├── src/
│   ├── model.py            # Flax Llama (RMSNorm, RoPE, SwiGLU)
│   ├── train_nca.py        # Stage 1: NCA pre-pretraining on 8 TPU chips
│   └── train_lm.py         # Stage 2: language pretraining (warm-start from NCA ckpt)
├── utils/
│   ├── nca.py              # NCA dynamics + rule sampling/filtering (JAX)
│   └── tokenizers.py       # Patch-based NCA tokenizer
└── scripts/
    ├── train_nca.sh
    └── train_lm.sh
```

## Setup (uv + TPU)

Install [`uv`](https://docs.astral.sh/uv/) if you don't already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies into a local `.venv`:

```bash
cd nca-pre-pretraining-jax
uv sync --extra tpu
```

This pulls `jax[tpu]` (TPU-backed JAX), Flax, Optax, Orbax, and the HuggingFace datasets/tokenizers used by stage 2. Verify the TPUs are visible:

```bash
uv run python -c "import jax; print(jax.devices())"
# Expected: [TpuDevice(id=0, ...), ..., TpuDevice(id=7, ...)]
```

## Parallelism

A single `jax.jit`'d training step runs on a `Mesh` with one axis `"data"` of size `len(jax.devices())`. Parameters are fully replicated; the global batch is sharded along the `"data"` axis using `NamedSharding(mesh, P("data"))`. On this 8-chip host, a `--batch_size 128` run puts 16 examples per chip.

No manual `pmap` — the training function is `jit` + sharded inputs, which lets XLA insert the all-reduce for gradients automatically.

## Stage 1 — NCA pre-pretraining

```bash
./scripts/train_nca.sh
```

The loop:

1. Samples NCA rules (optionally filtered by gzip-compression complexity band — default 50–100%).
2. Generates a pool of rollouts in JAX (`utils/nca.generate_nca_dataset`).
3. Tokenizes with `NCA_Tokenizer(patch=2, num_colors=10)` → vocab of `10^4 + 2 = 10002`.
4. Trains a 24-layer, 2048-hidden Llama with AdamW + warmup-cosine.

Important flags (see `src/train_nca.py:TrainConfig` for the full list):

| flag | meaning | default |
|---|---|---|
| `--grid`, `--patch`, `--num_colors` | NCA state space | 12 / 2 / 10 |
| `--filter_rules_threshold`, `--filter_rules_upper_bound` | gzip complexity band for rule filtering | 0.5 / 1.0 |
| `--train_num_rules` | distinct rules in training pool | 16000 |
| `--sims_per_refill` | simulations generated per host refill | 4096 |
| `--batch_size` | global batch (split across 8 chips) | 128 |
| `--steps` | total optimizer steps | 20000 |

Checkpoints (Orbax) land in `checkpoints/nca/step_*` and `checkpoints/nca/final`.

## Stage 2 — language pretraining

```bash
./scripts/train_lm.sh --nca_ckpt checkpoints/nca/final
```

Streams `Skylion007/openwebtext` through GPT-2's tokenizer, packs to `seq_len=1024`, and trains the same architecture with a language-sized vocab (50304). If `--nca_ckpt` is given, transformer blocks (attention/MLP/norms) are loaded from the stage-1 checkpoint; the embedding and LM head are fresh.

## Porting notes

Things that moved:

- `utils/models.py` (PyTorch Llama wrapper) → `src/model.py` (Flax `linen.Module`).
- `utils/dataset_utils.py`, `src/openwebtext_pt.py`, `src/language_train.py` → `src/train_lm.py` streaming loop.
- `src/nca_ppt.py` (torch DDP + torch Dataset) → `src/train_nca.py` with sharded `jax.jit`.
- `environment.yml` / `requirements.txt` / mamba → `pyproject.toml` + uv.
- `utils/training_args.py` (1000-line dataclass hierarchy + argparse) → single `tyro.cli(TrainConfig)` per entry point.

Things preserved verbatim (because they were already JAX):

- `utils/nca.py` — NCA substrate, rollout, rule filtering.
- The gzip-based complexity filter and tokenizer math.

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
