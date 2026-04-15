"""Minimal Flax Llama-style decoder transformer.

RMSNorm + RoPE + SwiGLU + causal multi-head attention. Pure Flax `linen` so the
training scripts can pmap/shard parameters across TPU chips with standard
`jax.sharding` primitives.

The model exposes an optional split between the input embedding vocab
(NCA patch-token vocab) and the output vocab (same by default). The language
pretraining entry point swaps the embedding/head when loading the NCA
checkpoint.
"""

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int = 10002
    output_vocab_size: Optional[int] = None
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_layers: int = 24
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    def __post_init__(self):
        if self.output_vocab_size is None:
            object.__setattr__(self, "output_vocab_size", self.vocab_size)
        if self.num_kv_heads is None:
            object.__setattr__(self, "num_kv_heads", self.num_heads)


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,), self.param_dtype)
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        return (x * scale).astype(self.dtype)


def precompute_rope(seq_len: int, head_dim: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(seq_len).astype(jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    # x: (B, T, H, D)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = jnp.stack([rot1, rot2], axis=-1)
    return out.reshape(*x.shape)


class Attention(nn.Module):
    cfg: LlamaConfig

    @nn.compact
    def __call__(self, x, cos, sin, deterministic: bool = True):
        cfg = self.cfg
        B, T, C = x.shape
        head_dim = cfg.hidden_size // cfg.num_heads

        qkv_proj = lambda n, out: nn.Dense(
            out, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name=n
        )
        q = qkv_proj("q_proj", cfg.num_heads * head_dim)(x)
        k = qkv_proj("k_proj", cfg.num_kv_heads * head_dim)(x)
        v = qkv_proj("v_proj", cfg.num_kv_heads * head_dim)(x)

        q = q.reshape(B, T, cfg.num_heads, head_dim)
        k = k.reshape(B, T, cfg.num_kv_heads, head_dim)
        v = v.reshape(B, T, cfg.num_kv_heads, head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cfg.num_kv_heads != cfg.num_heads:
            reps = cfg.num_heads // cfg.num_kv_heads
            k = jnp.repeat(k, reps, axis=2)
            v = jnp.repeat(v, reps, axis=2)

        # (B, H, T, D)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / jnp.sqrt(head_dim).astype(cfg.dtype)
        attn = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn = jnp.where(causal[None, None, :, :], attn, jnp.finfo(cfg.dtype).min)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(cfg.dtype)
        attn = nn.Dropout(cfg.dropout)(attn, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, cfg.hidden_size)
        return nn.Dense(
            cfg.hidden_size, use_bias=False, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, name="o_proj",
        )(out)


class MLP(nn.Module):
    cfg: LlamaConfig

    @nn.compact
    def __call__(self, x):
        cfg = self.cfg
        dense = lambda n, out: nn.Dense(
            out, use_bias=False, dtype=cfg.dtype, param_dtype=cfg.param_dtype, name=n
        )
        gate = dense("gate_proj", cfg.intermediate_size)(x)
        up = dense("up_proj", cfg.intermediate_size)(x)
        h = jax.nn.silu(gate) * up
        return dense("down_proj", cfg.hidden_size)(h)


class Block(nn.Module):
    cfg: LlamaConfig

    @nn.compact
    def __call__(self, x, cos, sin, deterministic: bool = True):
        cfg = self.cfg
        h = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, cfg.param_dtype, name="attn_norm")(x)
        x = x + Attention(cfg, name="attn")(h, cos, sin, deterministic)
        h = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, cfg.param_dtype, name="mlp_norm")(x)
        x = x + MLP(cfg, name="mlp")(h)
        return x


class Llama(nn.Module):
    cfg: LlamaConfig

    @nn.compact
    def __call__(self, tokens, deterministic: bool = True):
        cfg = self.cfg
        B, T = tokens.shape

        embed = nn.Embed(
            cfg.vocab_size, cfg.hidden_size,
            dtype=cfg.dtype, param_dtype=cfg.param_dtype, name="embed",
        )
        x = embed(tokens)

        head_dim = cfg.hidden_size // cfg.num_heads
        cos, sin = precompute_rope(cfg.max_seq_len, head_dim, cfg.rope_theta)
        cos = cos[:T].astype(cfg.dtype)
        sin = sin[:T].astype(cfg.dtype)

        for i in range(cfg.num_layers):
            x = Block(cfg, name=f"block_{i}")(x, cos, sin, deterministic)

        x = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.dtype, cfg.param_dtype, name="final_norm")(x)

        lm_head = nn.Dense(
            cfg.output_vocab_size, use_bias=False, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, name="lm_head",
        )
        return lm_head(x)
