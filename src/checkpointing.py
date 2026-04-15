from __future__ import annotations

import inspect
from typing import Any

import orbax.checkpoint as ocp


def save_checkpoint(ckptr: ocp.StandardCheckpointer, directory: str, state: Any) -> None:
    """Support both old and new Orbax StandardCheckpointer.save signatures."""
    params = inspect.signature(ckptr.save).parameters
    if "args" in params:
        ckptr.save(directory, args=ocp.args.StandardSave(state))
        return
    ckptr.save(directory, state)


def restore_checkpoint(ckptr: ocp.StandardCheckpointer, directory: str, target: Any) -> Any:
    """Support both old and new Orbax StandardCheckpointer.restore signatures."""
    params = inspect.signature(ckptr.restore).parameters
    if "args" in params:
        return ckptr.restore(directory, args=ocp.args.StandardRestore(target))
    return ckptr.restore(directory, target)
