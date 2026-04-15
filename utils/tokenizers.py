"""Pure-JAX NCA tokenizer.

Patch-based tokenization of a discrete cellular automaton grid. A PxP patch of
values in [0, num_colors) becomes a single integer in [0, num_colors**(P*P)).
Two extra ids serve as per-grid start/end markers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import jax.numpy as jnp


class Tokenizer(ABC):
    @abstractmethod
    def encode_task(self, grid: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...


class NCA_Tokenizer(Tokenizer):
    def __init__(self, patch: int, stride: int = None, num_colors: int = 10):
        self.patch = patch
        self.stride = stride or patch
        self.num_colors = num_colors
        self.start_tk = num_colors ** (patch ** 2)
        self.end_tk = num_colors ** (patch ** 2) + 1
        self.vocab_size = num_colors ** (patch ** 2) + 2

    def encode_task(self, grid: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Encode a batch of NCA rollouts into token/target sequences.

        grid: (B, N, H, W, C) — C is unused (first channel taken).
        returns (tokens, targets) of shape (B, N * (grid_len)), where
        grid_len = (H/P)*(W/P) + 2 for start/end.
        """
        B, N, H, W, _ = grid.shape
        P = self.patch
        N_H, N_W = H // P, W // P

        g = grid.reshape(B, N, H, W)
        g = g.reshape(B, N, N_H, P, N_W, P)
        g = g.transpose(0, 1, 2, 4, 3, 5)
        g = g.reshape(B, N, N_H * N_W, P * P)

        powers = self.num_colors ** jnp.arange(P * P)
        tokens = jnp.einsum("bnlp,p->bnl", g, powers).astype(jnp.int32)
        target = tokens

        mask = jnp.full((B, N, 1), -100, dtype=jnp.int32)
        start_tokens = jnp.full((B, N, 1), self.start_tk, dtype=jnp.int32)
        end_tokens = jnp.full((B, N, 1), self.end_tk, dtype=jnp.int32)

        tokens = jnp.concatenate([start_tokens, tokens, end_tokens], axis=-1)
        target = jnp.concatenate([mask, target, mask], axis=-1)

        return tokens.reshape(B, -1), target.reshape(B, -1)

    def decode_task(self, tokens: jnp.ndarray, dims: List[int]) -> jnp.ndarray:
        tokens = jnp.asarray(tokens)
        B, _ = tokens.shape
        P = self.patch
        N_H = dims[0] // P
        N_W = dims[1] // P

        power = self.num_colors ** jnp.arange(P * P)
        digits = (tokens[..., None] // power) % self.num_colors
        digits = digits.reshape(B, -1, N_H * N_W, P, P)
        digits = digits.reshape(B, -1, N_H, N_W, P, P)
        digits = digits.transpose(0, 1, 2, 4, 3, 5)
        digits = digits.reshape(B, -1, N_H * P, N_W * P)
        return digits
