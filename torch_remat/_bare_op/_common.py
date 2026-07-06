# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared utilities for the bare-op detection strategies.

Everything here is used by more than one strategy in :mod:`torch_remat._bare_op`:
the :class:`_SaveOutputHandle` save-output representation, the in-place diagnostic,
storage-aliasing view classification (proxy and function mode), and the mode
suppression switch.

Suppression: a *mode* strategy sees every op, including the ones inside a
``remat.op`` body whose SAVE-output inputs the op's own consume / snapshot path has
already handled. So the ``op`` wrapper runs its processing (body included) under
:func:`_suppress_bare_op_detection`, and each mode passes straight through when the
flag is set; only user code *between* op calls runs with the mode live. The wrapper
strategies (subclass / proxy) never read the flag -- a wrapped output trips
interception on any touch.

Known gap: suppression covers the whole ``remat.op`` body, but only the op's
*arguments* are actually handled. A SAVE output the body reaches via closure capture
is therefore missed by the modes (it hits a placeholder during recompute) while the
wrapper strategies still catch it. Narrowing suppression to remat's own processing
would close the gap; until then, pass such a value as an argument or use a wrapper
strategy.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

# A thunk that records a SAVE output's real value on the remat tape so recompute can
# reproduce it. Idempotent: the first bare consumer fires it, later ones are no-ops.
PersistOutputThunk = Callable[[], None]


@dataclass(frozen=True)
class _SaveOutputHandle:
    """Type-agnostic handle for one SAVE output, stored in the region's index.

    Decouples "this tensor is a SAVE output" from *how* it is represented; consumers
    look a value up via :func:`torch_remat._region._save_output_handle` and use these
    two thunks, never the output's type:

    * ``persist_output`` records the output on the tape so recompute reproduces it
      (idempotent).
    * ``unwrap`` takes the looked-up leaf and returns its grad-connected real value
      (identity for a plain tensor, the inner for a wrapper strategy). It is handed
      the leaf rather than closing over it, so the handle -- the *value* of the weak-
      keyed save-output index -- holds no strong reference back to its own key, which
      would pin a dead SAVE output alive until backward.
    """

    persist_output: PersistOutputThunk
    unwrap: Callable[[object], torch.Tensor]


def _merge_save_output_handles(
    first: _SaveOutputHandle, second: _SaveOutputHandle
) -> _SaveOutputHandle:
    """Combine two handles for one output value shared across positions.

    Under a plain-tensor strategy, a SAVE op returning the *same* tensor at two output
    positions collapses onto one index key, but each position still owns its own tape
    slot -- so consuming the shared value must fire *both* persist thunks, else the
    shadowed position's slot stays empty and recompute serves it a placeholder.
    ``unwrap`` is identical across positions, so the first is kept.
    """

    def persist_output() -> None:
        first.persist_output()
        second.persist_output()

    return _SaveOutputHandle(persist_output=persist_output, unwrap=first.unwrap)


def _unwrap_identity(leaf: object) -> torch.Tensor:
    """Unwrap for a SAVE output that is already a plain tensor (or a deferred view of one)."""

    assert isinstance(leaf, torch.Tensor)
    return leaf


@dataclass(frozen=True)
class _BaseRetainingPersist:
    """A producer's persist-output thunk plus a strong ref to the tensor a deferred
    view was derived from.

    A deferred view fires the producer's ``persist_output`` only when poked hard,
    which can happen *after* the original output object is gone -- and the thunk
    references its output only weakly, so the save would silently no-op and recompute
    would serve a placeholder. Retaining ``base`` keeps the producer's output
    resolvable exactly as long as a view that could still demand its save is alive.

    This is view->parent retention, not the value->key self-pin
    :class:`_SaveOutputHandle` avoids -- a dead view still pins nothing. It chains
    through views of views, so a whole view chain stays served by one producer slot.
    """

    persist_output: PersistOutputThunk
    base: torch.Tensor

    def __call__(self) -> None:
        self.persist_output()


def _inplace_message(op: object) -> str:
    """Build the diagnostic for an in-place / out op on a SAVE op's output."""

    return (
        f"torch_remat: {op} tried to mutate a SAVE op's output in place. A SAVE op "
        "keeps its output for backward and reproduces it during recompute, so "
        "mutating it would corrupt both copies. Wrap the mutating op in "
        "remat.op(...) (or apply it before the value leaves the producing op)."
    )


def _storage_id(tensor: torch.Tensor) -> int:
    """Return a tensor's storage base pointer (0 for an empty / storage-less tensor)."""

    return int(tensor.untyped_storage().data_ptr())


def _all_alias(out: object, base_storage: int) -> bool:
    """Return whether every tensor leaf of ``out`` aliases ``base_storage``.

    An op whose outputs all share the producer output's storage is a view; a compute
    allocates fresh storage. Callers rule out in-place mutation first, so a result
    sharing storage is safely a (deferred) view. A result with no tensor leaf
    (``data_ptr`` -> ``int``, ``item`` -> ``float``) is not a view.
    """

    tensors = [leaf for leaf in _value_leaves(out) if isinstance(leaf, torch.Tensor)]
    if not tensors:
        return False
    return all(_storage_id(tensor) == base_storage for tensor in tensors)


def _value_leaves(out: object) -> tuple[object, ...]:
    """One-hop leaf walk of an op result (a Tensor, or a tuple/list of them).

    Local to keep this leaf module free of a ``_pytree`` dependency; the view classifiers
    only ever see a Tensor or one hop of tuple/list.
    """

    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


# True while remat's own per-op processing runs, so the forward mode ignores the SAVE
# outputs that processing legitimately touches (consume, snapshot, boundary). Task-local,
# like the region contextvars, so concurrent forwards don't clobber each other.
_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "torch_remat_bare_op_detection_suppressed",
    default=False,
)


def _bare_op_detection_suppressed() -> bool:
    """Return whether the forward bare-op detection mode is currently suppressed."""

    return _suppressed.get()


@contextlib.contextmanager
def _suppress_bare_op_detection() -> Iterator[None]:
    """Suppress the forward bare-op detection mode for the duration of the block.

    Wrapped around a ``remat.op``'s own processing (and the region boundary) so a mode
    strategy does not re-save SAVE outputs the op has already persisted / snapshotted.
    A no-op for the non-mode strategies (nothing reads the flag), so it is always safe to
    install.
    """

    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)
