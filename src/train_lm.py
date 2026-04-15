"""Language pretraining in JAX with data parallelism across TPU chips.

Streams a HuggingFace dataset (default: OpenWebText) through a GPT-2 tokenizer
and trains a Flax Llama. Optionally warm-starts from an NCA checkpoint by
loading the transformer blocks while re-initializing the embedding + LM head
to match the language vocab.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
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

from .checkpointing import restore_checkpoint, save_checkpoint
from .model import Llama, LlamaConfig
from .train_nca import get_dtype, loss_fn


@dataclass
class LMConfig:
    # data
    dataset: str = "Skylion007/openwebtext"
    dataset_split: str = "train"
    tokenizer_name: str = "gpt2"
    seq_len: int = 1024
    batch_size: int = 128

    # model
    vocab_size: int = 50304  # padded gpt2 vocab to multiple of 64
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_layers: int = 24
    num_heads: int = 32
    dropout: float = 0.0
    dtype: str = "bfloat16"

    # optim
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    steps: int = 50000
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    b1: float = 0.9
    b2: float = 0.95

    # transfer
    nca_ckpt: Optional[str] = None  # path to orbax checkpoint directory

    # logging / ckpt
    log_every: int = 20
    ckpt_every: int = 2000
    save_dir: str = "checkpoints/lm"
    seed: int = 0
    wandb: bool = False
    wandb_project: str = "nca-pre-pretraining-jax"
    wandb_name: Optional[str] = None


def iter_tokenized_dataset(c: LMConfig):
    """Stream a HF dataset, tokenize, pack into fixed-length blocks."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(c.tokenizer_name)
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "<|endoftext|>"})
    eos_id = tok.eos_token_id

    ds = load_dataset(c.dataset, split=c.dataset_split, streaming=True)

    buf: list[int] = []
    block = c.seq_len + 1
    for record in ds:
        text = record.get("text") or record.get("content") or ""
        ids = tok.encode(text)
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= block:
            chunk = np.asarray(buf[:block], dtype=np.int32)
            buf = buf[block:]
            yield chunk


def batched(stream, batch_size: int, seq_len: int):
    batch = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
    i = 0
    for chunk in stream:
        batch[i] = chunk
        i += 1
        if i == batch_size:
            yield batch[:, :-1].copy(), batch[:, 1:].copy()
            i = 0


def build_model_cfg(c: LMConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_layers=c.num_layers,
        num_heads=c.num_heads,
        max_seq_len=c.seq_len,
        dropout=c.dropout,
        dtype=get_dtype(c.dtype),
    )


def _transfer_nca_params(lm_params, nca_params):
    """Copy every leaf from the NCA checkpoint into lm_params, except the
    embedding and LM head (whose shapes differ for the new language vocab)."""
    nca_flat = {
        "/".join(str(s.key) for s in path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(nca_params)[0]
    }

    def replace(path, leaf):
        key = "/".join(str(s.key) for s in path)
        if "embed" in key or "lm_head" in key:
            return leaf
        src = nca_flat.get(key)
        if src is not None and src.shape == leaf.shape:
            return src.astype(leaf.dtype)
        return leaf

    return jax.tree_util.tree_map_with_path(replace, lm_params)


def main(c: LMConfig):
    print(f"JAX devices: {jax.devices()}")
    num_devices = len(jax.devices())
    assert c.batch_size % num_devices == 0

    os.makedirs(c.save_dir, exist_ok=True)
    with open(os.path.join(c.save_dir, "config.json"), "w") as f:
        json.dump(asdict(c), f, indent=2)

    if c.wandb:
        import wandb
        wandb.init(project=c.wandb_project, name=c.wandb_name, config=asdict(c))

    mesh = Mesh(np.array(jax.devices()).reshape(num_devices), ("data",))
    data_sharding = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())

    key = jax.random.PRNGKey(c.seed)
    key, k_init, k_drop = jax.random.split(key, 3)

    cfg = build_model_cfg(c)
    model = Llama(cfg)
    params = model.init(k_init, jnp.zeros((1, c.seq_len), dtype=jnp.int32),
                        deterministic=True)["params"]
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {num_params/1e6:.1f}M")

    if c.nca_ckpt:
        print(f"Loading NCA checkpoint from {c.nca_ckpt}")
        ckptr = ocp.StandardCheckpointer()
        nca_params = restore_checkpoint(ckptr, os.path.abspath(c.nca_ckpt), params)
        params = _transfer_nca_params(params, nca_params)
        print("Transferred transformer weights; re-initialized embed + lm_head.")

    tx = optax.chain(
        optax.clip_by_global_norm(c.grad_clip),
        optax.adamw(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=c.lr,
                warmup_steps=c.warmup_steps, decay_steps=c.steps,
                end_value=c.min_lr,
            ),
            b1=c.b1, b2=c.b2, weight_decay=c.weight_decay,
        ),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax.device_put(state, replicated)

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(state, inputs, labels, rng):
        rng, sub = jax.random.split(rng)
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, state.apply_fn, inputs, labels, sub
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, rng

    ckptr = ocp.StandardCheckpointer()
    rng = k_drop
    stream = batched(iter_tokenized_dataset(c), c.batch_size, c.seq_len)

    t0 = time.time()
    running = 0.0
    n = 0
    for step in range(1, c.steps + 1):
        inputs_np, labels_np = next(stream)
        inputs = jax.device_put(inputs_np, data_sharding)
        labels = jax.device_put(labels_np, data_sharding)
        state, loss, rng = train_step(state, inputs, labels, rng)
        running += float(loss); n += 1

        if step % c.log_every == 0:
            avg = running / n
            dt = time.time() - t0
            tps = c.log_every * c.batch_size * c.seq_len / dt
            print(f"step {step:6d} | loss {avg:.4f} | {tps/1e3:.1f}k tok/s")
            if c.wandb:
                import wandb
                wandb.log({"train/loss": avg, "throughput/tokens_per_s": tps}, step=step)
            running = 0.0; n = 0; t0 = time.time()

        if step % c.ckpt_every == 0:
            path = os.path.abspath(os.path.join(c.save_dir, f"step_{step}"))
            save_checkpoint(ckptr, path, state.params)
            print(f"           | saved {path}")

    path = os.path.abspath(os.path.join(c.save_dir, "final"))
    save_checkpoint(ckptr, path, state.params)
    print(f"done. final checkpoint: {path}")


if __name__ == "__main__":
    main(tyro.cli(LMConfig))
