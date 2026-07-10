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
``remat.region`` body whose SAVE-output inputs the op's own consume / snapshot path has
already handled. So the ``op`` wrapper runs its processing (body included) under
:func:`_suppress_bare_op_detection`, and each mode passes straight through when the
flag is set; only user code *between* op calls runs with the mode live. The wrapper
strategies (subclass / proxy) never read the flag -- a wrapped output trips
interception on any touch.

Known gap: suppression covers the whole ``remat.region`` body, but only the op's
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
from typing import Any, Callable, Iterator

import torch

# Ops that mutate an argument in place WITHOUT annotating it ``(a!)`` in their schema --
# so ``alias_info.is_write`` is False and (being invisible to the version counter too) no
# post-hoc check catches it either. Only the batch-norm family: its ``running_mean`` /
# ``running_var`` update is gated on a runtime ``training`` flag, which the static schema
# cannot express. Keyed by ``FunctionSchema.name`` (covers every overload); the arg-name
# set is matched against the schema, so positions do not matter. We treat these args as
# always-written (conservative -- a bare batch_norm in eval, where the stats are not
# actually touched, would still be rejected); an activation is never passed here in real
# checkpointing, so this only guards a contrived case. ``rrelu_with_noise`` is NOT listed:
# it annotates its ``noise`` buffer ``(b!)``, so ``alias_info`` already catches it. This
# mirrors the running-stats half of torch's ``SchemaInfo.getTrainingOps``, kept local
# rather than pulling in the whole value-aware ``_SchemaInfo`` machinery for one case.
_UNANNOTATED_INPLACE_ARGS: dict[str, frozenset[str]] = {
    name: frozenset({"running_mean", "running_var"})
    for name in (
        "aten::batch_norm",
        "aten::instance_norm",
        "aten::_batch_norm_impl_index",
        "aten::cudnn_batch_norm",
        "aten::miopen_batch_norm",
        "aten::native_batch_norm",
    )
}

# The same batch-norm family seen from the *function* surface, for the post-hoc strategies
# (proxy, function_mode). At ``__torch_function__`` time ``func`` is a bare builtin / Python
# callable with no ``_schema``, and its running-stat arg *positions* differ between entry
# points (``torch.batch_norm`` puts them at 3/4, ``torch.nn.functional.batch_norm`` at 1/2),
# so the per-arg schema match above does not apply. The ``__name__`` is stable across those
# entry points, so we gate on it and then detect the write by comparing the touched saves'
# *values* before/after -- neither the annotation nor ``_version`` reveals this mutation, but
# the value provably changes. This is precise (unlike the conservative pre-hoc carve-out): an
# eval-mode call, which does not touch the stats, leaves the value equal and is allowed.
_UNANNOTATED_INPLACE_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "batch_norm",
        "instance_norm",
        "_batch_norm_impl_index",
        "cudnn_batch_norm",
        "miopen_batch_norm",
        "native_batch_norm",
    }
)

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
        "remat.region(...) (or apply it before the value leaves the producing op)."
    )


def _mutates_selected(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    is_selected: Callable[[object], bool],
) -> bool:
    """True if ``func`` writes to an argument position holding a *selected* tensor.

    ``is_selected`` marks the tensors we must protect from in-place mutation -- SAVE-
    output wrappers for the subclass strategy, save-index members for the dispatch mode.
    A mutable-schema op is only dangerous when it mutates one of those in place; when a
    selected tensor appears only in read-only operands (e.g. ``dst.scatter_(src=save)``
    with a fresh plain ``dst``) the op reads it but never writes it, so it is safe.

    Mutation is read straight off the per-argument ``alias_info.is_write`` annotation,
    plus the :data:`_UNANNOTATED_INPLACE_ARGS` carve-out for the batch-norm family whose
    running-stat writes the schema leaves unannotated.
    """

    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    carved = _UNANNOTATED_INPLACE_ARGS.get(schema.name, frozenset())
    if not schema.is_mutable and not carved:
        return False
    for i, arg in enumerate(schema.arguments):
        alias_info = arg.alias_info
        written = (alias_info is not None and alias_info.is_write) or arg.name in carved
        if not written:
            continue
        if i < len(args):
            value = args[i]
        elif arg.name in kwargs:
            value = kwargs[arg.name]
        else:
            continue
        if _any_leaf_selected(value, is_selected):
            return True
    return False


def _any_leaf_selected(value: object, is_selected: Callable[[object], bool]) -> bool:
    """Whether ``value`` -- or, one hop in, a member of a list/tuple ``value`` -- is selected.

    The one hop reaches wrappers inside a container argument (e.g. the tensor list a
    foreach in-place op mutates).
    """

    if is_selected(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(is_selected(v) for v in value)
    return False


def _snapshot_unannotated_inplace(
    func: Callable[..., Any], tensors: list[torch.Tensor]
) -> list[torch.Tensor] | None:
    """Clone ``tensors`` iff ``func`` is a batch-norm-family op, else None (skip the cost).

    The post-hoc strategies (proxy, function_mode) can only see the write these ops make to
    a SAVE output by comparing its value before/after (see :data:`_UNANNOTATED_INPLACE_FUNC_NAMES`);
    :func:`_unannotated_inplace_mutated` consumes the returned snapshots. ``None`` when the
    op is not in the family, so the common path clones nothing.
    """

    if getattr(func, "__name__", "") not in _UNANNOTATED_INPLACE_FUNC_NAMES:
        return None
    return [tensor.detach().clone() for tensor in tensors]


def _unannotated_inplace_mutated(
    snapshots: list[torch.Tensor] | None, tensors: list[torch.Tensor]
) -> bool:
    """True if a batch-norm-family op changed a snapshotted SAVE output's value in place.

    ``snapshots`` is what :func:`_snapshot_unannotated_inplace` returned (``None`` for a
    non-family op, so this is a cheap no-op then), aligned with ``tensors``.
    """

    if snapshots is None:
        return False
    return any(
        not torch.equal(before, after) for before, after in zip(snapshots, tensors)
    )


def _storage_id(tensor: torch.Tensor) -> int:
    """Return a tensor's storage base pointer (0 for an empty / storage-less tensor)."""

    return int(tensor.untyped_storage().data_ptr())


def _view_base_index(out: object, base_storages: list[int]) -> int | None:
    """Index of the producer in ``base_storages`` that ``out`` is a pure view of, else None.

    ``out`` is a view iff every tensor leaf shares one storage and that storage belongs
    to one of the touched SAVE outputs; the returned index is which producer's slot the
    view should ride. Positional order of arguments is irrelevant -- we classify on the
    result's *observed* storage, so ``b.view_as(a)`` defers on ``a`` just as ``a.view_as(b)``
    defers on ``a``, and an arbitrary ``__torch_function__`` cannot fool us into deferring a
    non-view: a compute allocates fresh storage (no match -> None -> eager persist).

    Returns None when there is no tensor leaf (``data_ptr`` -> ``int``, ``item`` -> ``float``),
    when leaves span more than one storage (a multi-output op mixing views and computes, or
    views of different producers), or when the shared storage is not a touched producer's.
    Callers rule out in-place mutation first, so a leaf sharing a producer's storage is safely
    a (deferred) view of it. Misclassification is always conservative: at worst it declines a
    valid deferral and persists eagerly (extra memory), never a wrong or missing save.
    """

    tensors = [leaf for leaf in _value_leaves(out) if isinstance(leaf, torch.Tensor)]
    if not tensors:
        return None
    storages = {_storage_id(tensor) for tensor in tensors}
    if len(storages) != 1:
        return None
    (storage,) = storages
    if not storage:
        return None
    for index, base in enumerate(base_storages):
        if base == storage:
            return index
    return None


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

    Wrapped around a ``remat.region``'s own processing (and the region boundary) so a mode
    strategy does not re-save SAVE outputs the op has already persisted / snapshotted.
    A no-op for the non-mode strategies (nothing reads the flag), so it is always safe to
    install.
    """

    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)
