# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Checkpoint-region and phase plumbing for torch_remat.

This is the small leaf module that the core (:mod:`torch_remat._api`) and the
diagnostic tracing (:mod:`torch_remat._trace`) both depend on. It holds the
context-local phase/region state and the helpers that read it, so ``_trace`` can
ask whether a forward is being recomputed without importing ``_api`` at runtime --
``_api`` calls into ``_trace`` (to record ops), so the reverse runtime edge would
form an import cycle.

References to ``_SaveRecord`` (from ``_api``) appear only in annotations, kept
under ``TYPE_CHECKING`` so it is never imported at runtime.
"""

from __future__ import annotations

import contextlib
import contextvars
import weakref
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Protocol, runtime_checkable, TYPE_CHECKING

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.weak import WeakTensorKeyDictionary
from torch_remat._bare_op._strategy import (
    _bare_op_strategy,
    _resolve_detect_bare_ops,
)

if TYPE_CHECKING:
    from torch_remat._api import _SaveOpForwardScratch, _SaveRecord
    from torch_remat._bare_op._common import _SaveOutputHandle
    from torch_remat._bare_op._strategy import _BareOpStrategy

# A thunk that records one SAVE output's value on the remat tape so recompute can
# reproduce it. Idempotent per output slot: the first consumer fires it, later ones
# are no-ops. Stored as the value of a region's storage-keyed save-output index.
PersistOutputThunk = Callable[[], None]


@runtime_checkable
class RecomputeStateHook(Protocol):
    """Snapshot/restore hook keeping external (non-tensor) state aligned across recompute.

    ``preserve_rng_state`` realigns torch's own generators across recompute, but
    external state -- e.g. a caller's global RNG op-counter that seeds dropout and
    stochastic rounding -- is invisible to it. A ``recompute=False`` (SAVE)
    region's body runs on the original forward but is skipped on recompute (its
    outputs are served from the tape), so any such state the body advanced is left
    behind on recompute and every downstream draw shifts, silently diverging the
    recompute from the forward.

    A registered hook closes that gap. torch_remat calls :meth:`snapshot` on the
    forward and :meth:`restore` on recompute at two points:

    * at region entry (snapshot before the body, restore before the replay) --
      this is the boundary realignment that keeps bare / ``recompute=True`` draws
      *before* the first SAVE op aligned; and
    * around each SAVE op (snapshot its exit state on the forward, restore that
      exact state on recompute where the body is skipped) -- so the skipped op's
      effect on the state is reproduced.

    ``restore`` is absolute (it reinstates a captured snapshot, not a delta), so
    the state resyncs at every SAVE boundary and a ``retain_graph`` re-recompute
    replays identically. RECOMPUTE and bare ops need no hook: they rerun on
    recompute and advance the state naturally.

    Register hooks via ``checkpoint(..., recompute_state_hooks=...)``.
    """

    def snapshot(self) -> Any:
        """Return an opaque snapshot of the external state (called on the forward)."""
        ...

    def restore(self, state: Any) -> None:
        """Reinstate a snapshot returned by :meth:`snapshot` (called on recompute)."""
        ...


class _Phase(Enum):
    """Execution phase for the active checkpoint region.

    Backward is intentionally absent: the remat tape only mediates transfer of a SAVE
    op's durably-saved outputs from the original forward to checkpoint recompute.
    After recompute, ordinary PyTorch autograd owns backward (and a SAVE op's
    saved tensors, which autograd kept on the original forward graph, are unpacked
    straight into its original grad_fn).
    """

    FORWARD = 0
    RECOMPUTE = 1


# eq=False: region states are identity objects (they hold dicts/tensors, value-equality
# is meaningless) -- and identity semantics also make them hashable and weakly keyable,
# which the live-region registry below relies on. The default ``@dataclass`` would set
# ``__hash__ = None`` and make instances unusable as WeakSet members.
@dataclass(eq=False)
class _CheckpointRegionState:
    """State for one checkpointed region shared by forward and recomputation."""

    # Optional diagnostic name for the checkpoint region.
    region_name: str | None = None

    # Forward tape of SAVE-op records keyed by region-relative name. Dict insertion
    # order is the tape execution order. RECOMPUTE ops register nothing -- a missing
    # entry for a name is what marks it RECOMPUTE during recompute.
    records: dict[str, _SaveRecord] = field(default_factory=dict)

    # Bare-op detection strategy, resolved once at region creation from checkpoint's
    # ``detect_bare_ops`` (see :mod:`torch_remat._bare_op._strategy` for the options).
    bare_op_strategy: _BareOpStrategy = field(
        default_factory=lambda: _bare_op_strategy("none")
    )

    # Identity-keyed index of this region's SAVE outputs -> ``_SaveOutputHandle``. The
    # single, type-agnostic source of "this tensor is a SAVE output", so consumers
    # never test the output's type and the detection strategy stays swappable. Weak,
    # so it never keeps an output alive. (Legacy bare-op path only.)
    save_output_index: WeakTensorKeyDictionary = field(
        default_factory=WeakTensorKeyDictionary
    )

    # Whether this region uses the explicit persist-index mechanism (the
    # ``save_output_persist_index`` below) instead of the legacy ``detect_bare_ops``
    # strategy path (``bare_op_strategy`` / ``save_output_index``). Selected once at
    # region creation from ``checkpoint``'s ``detect_bare_ops`` (``None`` -> new path).
    uses_persist_index: bool = False

    # Storage-keyed index of this region's SAVE outputs -> persist thunk. Keyed by
    # ``StorageWeakRef`` (not tensor identity) so a *bare view* of a SAVE output --
    # a distinct tensor sharing the same storage -- still resolves to its producer,
    # which is how a consumer (a ``remat.region``, or an explicit
    # ``remat.recompute_needs_tensor`` call) of such a view triggers the save. The
    # weak-ref key never keeps the output's storage alive; a dead entry is a small,
    # region-scoped bookkeeping remnant (bounded by the region's output count). Used
    # only when ``uses_persist_index`` is set.
    save_output_persist_index: dict[StorageWeakRef, PersistOutputThunk] = field(
        default_factory=dict
    )

    # Recompute-scoped buffer of materialized saved-input values, keyed op_name ->
    # {slot_name: tensor}. Empty during the forward; a skipped SAVE op's replay fills
    # its entry (``_rederive_saved_inputs``) and the op's unpack hook reads it
    # (``_load_saved_input``). Each op's entry is fully replaced per replay, so a
    # ``retain_graph`` backward gets fresh values.
    rederived_saved_inputs: dict[str, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )

    # Snapshot/restore hooks for external state (e.g. an RNG op-counter) kept aligned
    # across recompute. Registered via ``checkpoint(..., recompute_state_hooks=...)``;
    # see :class:`RecomputeStateHook`. Empty when none are registered.
    state_hooks: tuple[RecomputeStateHook, ...] = ()

    # Per-hook snapshots taken at region entry on the forward (``state_hooks`` order),
    # restored at the start of recompute so pre-SAVE-op draws realign to the boundary.
    # Filled when the FORWARD phase context is entered.
    entry_snapshots: tuple[Any, ...] = ()


@dataclass(frozen=True)
class _ActiveCheckpointRegion:
    """Context-local pointer to the active checkpoint region and phase."""

    region_state: _CheckpointRegionState
    phase: _Phase

    # Op names reached in this phase, for duplicate detection. The dataclass is
    # frozen so the pointer is immutable, but the set is phase-local mutable state.
    claimed_names: set[str] = field(default_factory=set)


_state: contextvars.ContextVar[_ActiveCheckpointRegion | None] = contextvars.ContextVar(
    "torch_remat_state",
    default=None,
)


# Weak registry of checkpoint regions whose forward has run and whose backward graph is
# not yet freed -- i.e. the regions live at the pre-backward high-water mark. A region is
# added at creation (:func:`_checkpoint_context_fn`) and drops out on its own once the
# state is reclaimed: the state is kept alive only by the checkpoint frame that drives
# recompute, so membership tracks exactly "forward done, backward pending" without pinning
# anything. This is what lets a whole-model saved-for-backward report enumerate every
# transformer block's tape at once, none of which is the currently *active* region.
_live_regions: "weakref.WeakSet[_CheckpointRegionState]" = weakref.WeakSet()


def _iter_live_regions() -> list[_CheckpointRegionState]:
    """Return the currently-live checkpoint region states (order unspecified).

    "Live" means the region's forward has executed and its backward subgraph has not yet
    been released. Callers wanting a stable order should sort (e.g. by ``region_name``).
    """

    return list(_live_regions)


def is_recomputing() -> bool:
    """Return whether execution is currently in checkpoint recomputation.

    Returns:
        bool: ``True`` while a checkpoint region is replaying its forward during
        recompute, ``False`` on the original forward and outside any region.

    Example:
        ```python
        if not remat.is_recomputing():
            log_forward_only_metric(x)
        ```
    """

    state = _state.get()
    return state is not None and state.phase is _Phase.RECOMPUTE


# Forward-only pack scratch for the SAVE op whose body is currently running, so
# :func:`save_for_backward` can attach names to that op's nested-hook saves. None outside a
# SAVE op forward.
_active_save_op: contextvars.ContextVar[_SaveOpForwardScratch | None] = (
    contextvars.ContextVar(
        "torch_remat_active_save_op",
        default=None,
    )
)


class _CheckpointPhaseContext(contextlib.AbstractContextManager[None]):
    """Reusable context manager for one checkpoint phase.

    PyTorch non-reentrant checkpoint stores these and enters them around the
    original forward and replay. The FORWARD phase also installs the strategy's
    bare-op detection mode (a null context for non-mode strategies); the mode is
    deliberately absent during RECOMPUTE, where SAVE ops are skipped and their
    outputs come from the tape or a placeholder.

    The RECOMPUTE phase also brackets the replay with :class:`RecomputeStateHook`
    *fork* semantics: it snapshots the outer external state on entry, realigns to
    the region-entry snapshot, and reinstates the outer state on exit -- so the
    replay's redraws (e.g. RNG) never leak into the surrounding backward or the next
    step, matching ``torch.random.fork_rng``.
    """

    def __init__(
        self,
        region_state: _CheckpointRegionState,
        phase: _Phase,
    ) -> None:
        self._region_state = region_state
        self._phase = phase
        self._token: contextvars.Token[_ActiveCheckpointRegion | None] | None = None
        self._mode: contextlib.AbstractContextManager[None] | None = None
        # Outer-state snapshots taken at RECOMPUTE entry, reinstated at RECOMPUTE exit.
        self._outer_snapshots: tuple[Any, ...] = ()

    def __enter__(self) -> None:
        self._token = _state.set(
            _ActiveCheckpointRegion(region_state=self._region_state, phase=self._phase)
        )
        region_state = self._region_state
        if self._phase is _Phase.FORWARD:
            # Snapshot external state at region entry so recompute can realign to it
            # before replaying the body -- the boundary counterpart to per-SAVE-op
            # restore. See :class:`RecomputeStateHook`.
            region_state.entry_snapshots = tuple(
                hook.snapshot() for hook in region_state.state_hooks
            )
            mode = region_state.bare_op_strategy.forward_mode(region_state)
            mode.__enter__()
            self._mode = mode
        else:  # _Phase.RECOMPUTE
            # Recompute means backward is now consuming this region, so it leaves the
            # "live at the pre-backward high-water mark" set. Deregister deterministically
            # here rather than waiting for the weakref to drop: torch checkpoint's frame can
            # hold the region state in a reference cycle (its recompute closure captures this
            # context), so relying on reclamation would leave already-consumed regions in the
            # registry until a cyclic gc pass -- stale entries a cross-step report would count.
            # The WeakSet still backstops any region that never recomputes.
            _live_regions.discard(region_state)
            # Fork: snapshot the outer state so it can be reinstated once the replay
            # finishes, then realign to the region-entry snapshot before replaying.
            self._outer_snapshots = tuple(
                hook.snapshot() for hook in region_state.state_hooks
            )
            for hook, snapshot in zip(
                region_state.state_hooks, region_state.entry_snapshots
            ):
                hook.restore(snapshot)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc_value, traceback)
            self._mode = None
        if self._phase is _Phase.RECOMPUTE:
            # Reinstate the outer state captured on entry, completing the fork so the
            # replay's redraws don't perturb the surrounding stream.
            for hook, snapshot in zip(
                self._region_state.state_hooks, self._outer_snapshots
            ):
                hook.restore(snapshot)
            self._outer_snapshots = ()
        if self._token is not None:
            _state.reset(self._token)
            self._token = None


def _checkpoint_context_fn(
    region_name: str | None = None,
    detect_bare_ops: bool | str | None = None,
    recompute_state_hooks: tuple[RecomputeStateHook, ...] = (),
) -> tuple[
    contextlib.AbstractContextManager[None], contextlib.AbstractContextManager[None]
]:
    """Return forward/recompute context managers for non-reentrant checkpointing.

    Both contexts share one region state so op records from the original forward
    can be replayed by relative op name during recomputation.

    ``detect_bare_ops=None`` (the default) selects the explicit persist-index
    mechanism (a bare consumer is made to work by an explicit
    ``remat.recompute_needs_tensor`` call, or by regionizing the consumer); any
    explicit value keeps the legacy bare-op detection strategy path.
    """

    uses_persist_index = detect_bare_ops is None
    region_state = _CheckpointRegionState(
        region_name=region_name,
        uses_persist_index=uses_persist_index,
        bare_op_strategy=_bare_op_strategy(
            "none" if uses_persist_index else _resolve_detect_bare_ops(detect_bare_ops)
        ),
        state_hooks=recompute_state_hooks,
    )
    _live_regions.add(region_state)
    return (
        _CheckpointPhaseContext(region_state, _Phase.FORWARD),
        _CheckpointPhaseContext(region_state, _Phase.RECOMPUTE),
    )


def _display_name(region_state: _CheckpointRegionState, op_name: str) -> str:
    """Render a diagnostic op name, including the checkpoint region."""

    if region_state.region_name is None:
        return op_name
    return f"{region_state.region_name}::{op_name}"


def _save_output_handle(
    region_state: _CheckpointRegionState, leaf: object
) -> _SaveOutputHandle | None:
    """Return the SAVE-output handle for a leaf, or None if it is not a SAVE output.

    The single, strategy-agnostic "is this a SAVE output" test. A self-identifying
    output (the ``"proxy"`` strategy) is recovered by type via ``typed_handle``;
    everything else is a tensor looked up in the save-output index. The tensor check
    guards the lookup -- ``WeakTensorKeyDictionary.get`` raises on a non-tensor key.
    """

    strategy = region_state.bare_op_strategy
    typed = strategy.typed_handle(leaf)
    if typed is not None:
        return typed
    if isinstance(leaf, torch.Tensor):
        return region_state.save_output_index.get(leaf)
    return None


def _save_output_persist(
    region_state: _CheckpointRegionState, leaf: object
) -> PersistOutputThunk | None:
    """Return a leaf's producer persist thunk, or None if it is not a SAVE output.

    The single "is this (a view of) a SAVE output" test: a tensor leaf is looked up
    in the region's persist index by its storage, so both the SAVE output itself and
    any bare view sharing its storage resolve to the producer. A non-tensor leaf, or a
    tensor whose storage is not a SAVE output's, returns None.
    """

    if isinstance(leaf, torch.Tensor):
        return region_state.save_output_persist_index.get(
            StorageWeakRef(leaf.untyped_storage())
        )
    return None


def _expect_state() -> _ActiveCheckpointRegion:
    """Return the active checkpoint state or raise a useful error."""

    state = _state.get()
    if state is None:
        raise RuntimeError("No active torch_remat checkpoint region")
    return state


def _assert_phase(expected: _Phase) -> None:
    """Assert the active region runs in ``expected`` phase.

    Only for helpers that run while a phase context is installed. The backward-time
    unpack path is intentionally *not* guarded -- autograd unpacks saved tensors after
    recompute's context has exited, so no phase is active there.
    """

    state = _state.get()
    actual = None if state is None else state.phase
    assert actual is expected, (
        f"torch_remat internal error: expected {expected.name} phase, got {actual}"
    )
