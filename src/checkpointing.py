from __future__ import annotations

import inspect
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp


def _to_host(value: Any) -> Any:
    """Convert JAX arrays to host-backed NumPy arrays before Orbax sees them.

    This avoids Orbax's replica-slice path for mesh-sharded arrays, which is
    incompatible with the JAX version pinned in this project.
    """
    return jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf) if isinstance(leaf, jax.Array) else leaf,
        value,
    )


def save_checkpoint(ckptr: ocp.StandardCheckpointer, directory: str, state: Any) -> None:
    """Support both old and new Orbax StandardCheckpointer.save signatures."""
    host_state = _to_host(state)
    params = inspect.signature(ckptr.save).parameters
    if "args" in params:
        ckptr.save(directory, args=ocp.args.StandardSave(host_state))
    else:
        ckptr.save(directory, host_state)
    ckptr.wait_until_finished()


def restore_checkpoint(ckptr: ocp.StandardCheckpointer, directory: str, target: Any) -> Any:
    """Support both old and new Orbax StandardCheckpointer.restore signatures."""
    host_target = _to_host(target)
    params = inspect.signature(ckptr.restore).parameters
    if "args" in params:
        return ckptr.restore(directory, args=ocp.args.StandardRestore(host_target))
    return ckptr.restore(directory, host_target)
