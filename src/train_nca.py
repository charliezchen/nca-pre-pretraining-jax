"""NCA pre-pretraining in JAX with data parallelism across 8 TPU chips.

Usage (single host, 8 chips):
    python -m src.train_nca --grid 12 --patch 2 --num_colors 10 \
        --seq_len 1024 --batch_size 128 --num_layers 24 --num_heads 32 \
        --hidden_size 2048 --lr 1e-4 --steps 20000

Data parallelism: parameters are replicated on every chip, batches are sharded
along the batch axis via `jax.sharding.NamedSharding`. The training step is a
single `jax.jit`'d function that uses `jax.lax.pmean` over the batch axis to
average gradients.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .checkpointing import save_checkpoint
from utils.nca import generate_nca_dataset, generate_rules_batch
from utils.tokenizers import NCA_Tokenizer
from .model import Llama, LlamaConfig


# ---------- config ---------- #

@dataclass
class TrainConfig:
    # NCA
    grid: int = 12
    patch: int = 2
    num_colors: int = 10
    identity_bias: float = 0.0
    temperature: float = 1.0
    dT: int = 2
    init_rollout_steps: int = 0
    train_num_rules: int = 16000
    val_num_rules: int = 1000
    filter_rules: bool = True
    filter_rules_threshold: float = 0.5
    filter_rules_upper_bound: float = 1.0
    filter_rules_mode: str = "gzip"
    regen_rules_every: int = 0  # 0 = never

    # data
    seq_len: int = 1024
    batch_size: int = 128          # GLOBAL batch
    val_batch_size: int = 128
    sims_per_refill: int = 4096    # host-generated sims between refills
    min_grid: int = 1

    # model
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_layers: int = 24
    num_heads: int = 32
    dropout: float = 0.0
    dtype: str = "bfloat16"

    # optim
    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    steps: int = 20000
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    b1: float = 0.9
    b2: float = 0.95

    # logging / ckpt
    log_every: int = 20
    val_every: int = 500
    ckpt_every: int = 1000
    save_dir: str = "checkpoints/nca"
    seed: int = 0
    wandb: bool = False
    wandb_project: str = "nca-pre-pretraining-jax"
    wandb_name: Optional[str] = None


# ---------- utilities ---------- #

def get_dtype(name: str):
    return {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "float16": jnp.float16}[name]


def build_model_cfg(c: TrainConfig, vocab_size: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_layers=c.num_layers,
        num_heads=c.num_heads,
        max_seq_len=c.seq_len,
        dropout=c.dropout,
        dtype=get_dtype(c.dtype),
    )


def cosine_schedule(c: TrainConfig):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=c.lr,
        warmup_steps=c.warmup_steps,
        decay_steps=c.steps,
        end_value=c.min_lr,
    )


def make_optimizer(c: TrainConfig):
    return optax.chain(
        optax.clip_by_global_norm(c.grad_clip),
        optax.adamw(
            learning_rate=cosine_schedule(c),
            b1=c.b1, b2=c.b2,
            weight_decay=c.weight_decay,
        ),
    )


# ---------- data generation ---------- #

class NCADataStream:
    """Host-side iterator that yields (inputs, targets) batches.

    Regenerates a pool of `sims_per_refill` simulations on demand; each batch
    comes from slicing the tokenized pool. Running out -> regenerate.
    """
    def __init__(self, c: TrainConfig, tokenizer: NCA_Tokenizer, rule_seeds: jnp.ndarray, key: jax.Array):
        self.c = c
        self.tok = tokenizer
        self.rule_seeds = rule_seeds
        self.key = key
        self.pool_inputs = None
        self.pool_targets = None
        self.cursor = 0
        self.grid_len = (c.grid // c.patch) ** 2 + 2
        self.num_examples = int(math.ceil(c.seq_len / self.grid_len))

    def _refill(self):
        c = self.c
        self.key, sub = jax.random.split(self.key)
        sims = generate_nca_dataset(
            sub,
            num_sims=c.sims_per_refill,
            grid=c.grid,
            d_state=c.num_colors,
            n_groups=1,
            identity_bias=c.identity_bias,
            temperature=c.temperature,
            num_examples=self.num_examples,
            dT=c.dT,
            rule_seeds=self.rule_seeds,
            num_rules=self.rule_seeds.shape[0],
            start_step=c.init_rollout_steps,
        )
        seq, targets = self.tok.encode_task(sims)
        # mask negative
        target = jnp.where(seq < 0, jnp.int32(-100), seq)
        target = target.at[:, : c.min_grid * self.grid_len].set(-100)

        inputs = seq[:, :-1]
        labels = target[:, 1:]

        L = inputs.shape[1]
        if L < c.seq_len:
            pad = jnp.full((inputs.shape[0], c.seq_len - L), -100, dtype=inputs.dtype)
            inputs = jnp.concatenate([inputs, pad], axis=1)
            labels = jnp.concatenate([labels, pad], axis=1)
        else:
            inputs = inputs[:, : c.seq_len]
            labels = labels[:, : c.seq_len]

        # inputs may contain -100 from padding/mask of seq<0 — clamp to 0 for embedding
        safe_inputs = jnp.where(inputs < 0, 0, inputs)

        self.pool_inputs = np.asarray(safe_inputs)
        self.pool_targets = np.asarray(labels)
        self.cursor = 0
        # shuffle
        idx = np.random.default_rng(int(jax.random.randint(sub, (), 0, 2**31 - 1))).permutation(self.pool_inputs.shape[0])
        self.pool_inputs = self.pool_inputs[idx]
        self.pool_targets = self.pool_targets[idx]

    def next_batch(self, batch_size: int):
        if self.pool_inputs is None or self.cursor + batch_size > self.pool_inputs.shape[0]:
            self._refill()
        s = self.cursor
        self.cursor += batch_size
        return self.pool_inputs[s : s + batch_size], self.pool_targets[s : s + batch_size]


# ---------- loss / step ---------- #

def loss_fn(params, apply_fn, inputs, labels, rng):
    logits = apply_fn({"params": params}, inputs, deterministic=False, rngs={"dropout": rng})
    logits = logits.astype(jnp.float32)
    mask = (labels != -100).astype(jnp.float32)
    labels_clamped = jnp.where(labels < 0, 0, labels)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logprobs, labels_clamped[..., None], axis=-1).squeeze(-1)
    loss = (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    return loss


@partial(jax.jit, static_argnums=(1,))
def eval_step(params, apply_fn, inputs, labels):
    logits = apply_fn({"params": params}, inputs, deterministic=True).astype(jnp.float32)
    mask = (labels != -100).astype(jnp.float32)
    labels_clamped = jnp.where(labels < 0, 0, labels)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logprobs, labels_clamped[..., None], axis=-1).squeeze(-1)
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


# ---------- main ---------- #

def main(c: TrainConfig):
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())
    assert c.batch_size % num_devices == 0, (
        f"global batch {c.batch_size} must divide # devices {num_devices}")

    os.makedirs(c.save_dir, exist_ok=True)
    with open(os.path.join(c.save_dir, "config.json"), "w") as f:
        json.dump(asdict(c), f, indent=2, default=str)

    if c.wandb:
        import wandb
        wandb.init(project=c.wandb_project, name=c.wandb_name, config=asdict(c))

    mesh = Mesh(np.array(jax.devices()).reshape(num_devices), ("data",))

    key = jax.random.PRNGKey(c.seed)
    key, k_rules, k_init, k_stream_train, k_stream_val, k_drop = jax.random.split(key, 6)

    tokenizer = NCA_Tokenizer(patch=c.patch, num_colors=c.num_colors)
    vocab_size = tokenizer.vocab_size

    # rules
    total_rules = c.train_num_rules + c.val_num_rules
    if c.filter_rules:
        print(f"Filtering {total_rules} rules (gzip ratio in "
              f"[{c.filter_rules_threshold}, {c.filter_rules_upper_bound}])...")
        rule_seeds = generate_rules_batch(
            seed=k_rules,
            num_rules=total_rules,
            tokenizer=tokenizer,
            threshold=c.filter_rules_threshold,
            upper_bound=c.filter_rules_upper_bound,
            dT=c.dT,
            n_steps=10,
            mode=c.filter_rules_mode,
            start_step=c.init_rollout_steps,
            grid=c.grid,
            d_state=c.num_colors,
            identity_bias=c.identity_bias,
            temperature=c.temperature,
        )
    else:
        rule_seeds = jax.random.split(k_rules, total_rules)

    train_rules = rule_seeds[: c.train_num_rules]
    val_rules = rule_seeds[c.train_num_rules :]

    train_stream = NCADataStream(c, tokenizer, train_rules, k_stream_train)
    val_stream = NCADataStream(c, tokenizer, val_rules, k_stream_val)

    # model + state
    cfg = build_model_cfg(c, vocab_size)
    model = Llama(cfg)
    dummy = jnp.zeros((1, c.seq_len), dtype=jnp.int32)
    params = model.init(k_init, dummy, deterministic=True)["params"]
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {num_params/1e6:.1f}M")

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=make_optimizer(c)
    )

    # replicate state across devices (data-parallel: same params, sharded batch)
    replicated_sharding = NamedSharding(mesh, P())
    state = jax.device_put(state, replicated_sharding)

    data_sharding = NamedSharding(mesh, P("data"))

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, inputs, labels, rng):
        rng, sub = jax.random.split(rng)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, state.apply_fn, inputs, labels, sub
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, rng

    # orbax checkpointer
    ckptr = ocp.StandardCheckpointer()

    rng = k_drop
    t0 = time.time()
    running_loss = 0.0
    n_running = 0
    for step in range(1, c.steps + 1):
        inputs_np, labels_np = train_stream.next_batch(c.batch_size)
        inputs = jax.device_put(inputs_np, data_sharding)
        labels = jax.device_put(labels_np, data_sharding)
        state, loss, rng = train_step(state, inputs, labels, rng)

        running_loss += float(loss)
        n_running += 1

        if c.regen_rules_every and step % c.regen_rules_every == 0:
            key, sub = jax.random.split(key)
            new_rules = generate_rules_batch(
                seed=sub, num_rules=c.train_num_rules, tokenizer=tokenizer,
                threshold=c.filter_rules_threshold, upper_bound=c.filter_rules_upper_bound,
                dT=c.dT, n_steps=10, mode=c.filter_rules_mode,
                start_step=c.init_rollout_steps, grid=c.grid,
                d_state=c.num_colors, identity_bias=c.identity_bias,
                temperature=c.temperature,
            )
            train_stream.rule_seeds = new_rules
            train_stream.pool_inputs = None

        if step % c.log_every == 0:
            avg = running_loss / n_running
            dt = time.time() - t0
            tps = c.log_every * c.batch_size * c.seq_len / dt
            print(f"step {step:6d} | loss {avg:.4f} | {tps/1e3:.1f}k tok/s")
            if c.wandb:
                import wandb
                wandb.log({"train/loss": avg, "throughput/tokens_per_s": tps}, step=step)
            running_loss = 0.0
            n_running = 0
            t0 = time.time()

        if step % c.val_every == 0:
            losses = []
            for _ in range(20):
                vin_np, vla_np = val_stream.next_batch(c.val_batch_size)
                vin = jax.device_put(vin_np, data_sharding)
                vla = jax.device_put(vla_np, data_sharding)
                losses.append(float(eval_step(state.params, state.apply_fn, vin, vla)))
            val_loss = sum(losses) / len(losses)
            print(f"           | val {val_loss:.4f}")
            if c.wandb:
                import wandb
                wandb.log({"val/loss": val_loss}, step=step)

        if step % c.ckpt_every == 0:
            path = os.path.abspath(os.path.join(c.save_dir, f"step_{step}"))
            save_checkpoint(ckptr, path, state.params)
            print(f"           | saved {path}")

    # final save
    path = os.path.abspath(os.path.join(c.save_dir, "final"))
    save_checkpoint(ckptr, path, state.params)
    print(f"done. final checkpoint: {path}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
