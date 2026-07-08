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
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING

import torch
from torch.utils.weak import WeakTensorKeyDictionary
from torch_remat._bare_op._strategy import (
    _bare_op_strategy,
    _resolve_detect_bare_ops,
)

if TYPE_CHECKING:
    from torch_remat._api import _SaveOpForwardScratch, _SaveRecord
    from torch_remat._bare_op._common import _SaveOutputHandle
    from torch_remat._bare_op._strategy import _BareOpStrategy


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


@dataclass
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
    # so it never keeps an output alive.
    save_output_index: WeakTensorKeyDictionary = field(
        default_factory=WeakTensorKeyDictionary
    )

    # Recompute-scoped buffer of materialized saved-input values, keyed op_name ->
    # {slot_name: tensor}. Empty during the forward; a skipped SAVE op's replay fills
    # its entry (``_rederive_saved_inputs``) and the op's unpack hook reads it
    # (``_load_saved_input``). Each op's entry is fully replaced per replay, so a
    # ``retain_graph`` backward gets fresh values.
    recompute_saved_inputs: dict[str, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )


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

    def __enter__(self) -> None:
        self._token = _state.set(
            _ActiveCheckpointRegion(region_state=self._region_state, phase=self._phase)
        )
        if self._phase is _Phase.FORWARD:
            mode = self._region_state.bare_op_strategy.forward_mode(self._region_state)
            mode.__enter__()
            self._mode = mode

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc_value, traceback)
            self._mode = None
        if self._token is not None:
            _state.reset(self._token)
            self._token = None


def _checkpoint_context_fn(
    region_name: str | None = None,
    detect_bare_ops: bool | str = "subclass",
) -> tuple[
    contextlib.AbstractContextManager[None], contextlib.AbstractContextManager[None]
]:
    """Return forward/recompute context managers for non-reentrant checkpointing.

    Both contexts share one region state so op records from the original forward
    can be replayed by relative op name during recomputation.
    """

    region_state = _CheckpointRegionState(
        region_name=region_name,
        bare_op_strategy=_bare_op_strategy(_resolve_detect_bare_ops(detect_bare_ops)),
    )
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
