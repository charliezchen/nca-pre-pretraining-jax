import tempfile
import unittest
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.checkpointing import restore_checkpoint, save_checkpoint
from src.model import Llama, LlamaConfig


class CheckpointingTest(unittest.TestCase):
    def test_save_and_restore_trained_replicated_params(self):
        devices = np.array(jax.devices())
        mesh = Mesh(devices.reshape(len(devices)), ("data",))
        data_sharding = NamedSharding(mesh, P("data"))
        replicated_sharding = NamedSharding(mesh, P())

        model = Llama(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_layers=1,
                num_heads=2,
                max_seq_len=8,
                dropout=0.0,
                dtype=jnp.float32,
            )
        )
        key = jax.random.PRNGKey(0)
        key, k_init = jax.random.split(key)
        params = model.init(
            k_init,
            jnp.zeros((1, 8), dtype=jnp.int32),
            deterministic=True,
        )["params"]
        target = jax.tree_util.tree_map(np.asarray, params)

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adamw(1e-3),
        )
        state = jax.device_put(state, replicated_sharding)

        @partial(jax.jit, donate_argnums=(0,))
        def train_step(state, inputs, labels):
            def loss_fn(model_params):
                logits = state.apply_fn(
                    {"params": model_params}, inputs, deterministic=True
                )
                labels_one_hot = jax.nn.one_hot(labels, logits.shape[-1])
                return jnp.mean((logits - labels_one_hot) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss

        inputs = jax.device_put(jnp.zeros((len(devices), 8), dtype=jnp.int32), data_sharding)
        labels = jax.device_put(jnp.zeros((len(devices), 8), dtype=jnp.int32), data_sharding)
        state, _ = train_step(state, inputs, labels)

        ckptr = ocp.StandardCheckpointer()
        ckpt_dir = Path(tempfile.mkdtemp()) / "ckpt"
        save_checkpoint(ckptr, str(ckpt_dir), state.params)

        restored = restore_checkpoint(ckptr, str(ckpt_dir), target)

        original_leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(np.asarray, state.params))
        restored_leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(np.asarray, restored))
        self.assertEqual(len(original_leaves), len(restored_leaves))
        for original, restored_leaf in zip(original_leaves, restored_leaves):
            np.testing.assert_allclose(restored_leaf, original)


if __name__ == "__main__":
    unittest.main()
