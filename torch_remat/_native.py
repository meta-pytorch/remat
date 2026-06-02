# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Implicit native PyTorch rematerialization helpers."""

from __future__ import annotations

import contextvars
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy as _TorchCheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch_remat._api import (
    _CheckpointRegionState,
    _display_name,
    _expect_record,
    _OpRecord,
    _Phase,
    _release_record_after_recompute_if_needed,
    _state,
    _validate_name,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class _NativeRegion:
    """Context-local native PyTorch SAVE region controlled by SAC."""

    # Region-relative name for the active native save region.
    op_name: str


_native_region: contextvars.ContextVar[_NativeRegion | None] = contextvars.ContextVar(
    "torch_remat_native_region",
    default=None,
)


def native_save_region(op_name: str, function: Callable[[], _T]) -> _T:
    """Run a native PyTorch function region under SAC-style non-recompute.

    All PyTorch ops executed by ``function`` use PyTorch selective activation
    checkpointing to avoid rerunning those ops during recomputation. The return
    value of ``function`` is the native region boundary output used in reports.

    Example:
        ```python
        y = remat.native_save_region("native.mm", lambda: torch.mm(x, w))
        ```
    """

    _validate_name(op_name, what="op_name")
    state = _state.get()
    if state is None:
        return function()

    if state.phase is _Phase.FORWARD:
        record = _begin_native_record(state.region_state, op_name)
        sac_context = _expect_native_sac_context(record, phase=state.phase)
    else:
        record = _expect_record(state.region_state, op_name)
        sac_context = _expect_native_sac_context(record, phase=state.phase)

    token = _native_region.set(_NativeRegion(op_name=op_name))
    try:
        with sac_context:
            output = function()

        if state.phase is _Phase.FORWARD:
            record.record_output_schema(_expect_native_output(output))
        else:
            _release_record_after_recompute_if_needed(record)
        return output
    finally:
        _native_region.reset(token)


def _native_region_sac_policy(
    ctx: Any,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> _TorchCheckpointPolicy:
    """Policy callback used by PyTorch selective activation checkpointing."""

    del args, kwargs
    native = _native_region.get()
    if native is None:
        return _TorchCheckpointPolicy.PREFER_RECOMPUTE

    state = _state.get()
    op_output = getattr(ctx, "op_output", None)
    if (
        state is not None
        and state.phase is _Phase.FORWARD
        and not ctx.is_recompute
        and op_output is not None
    ):
        record = state.region_state.records[native.op_name]
        _record_native_sac_outputs(record, func, op_output)

    return _TorchCheckpointPolicy.MUST_SAVE


def _begin_native_record(
    region_state: _CheckpointRegionState,
    op_name: str,
) -> _OpRecord:
    """Create an op record for a native PyTorch region."""

    if op_name in region_state.records:
        raise RuntimeError(
            f"Duplicate torch_remat op name: {_display_name(region_state, op_name)}"
        )

    record = _OpRecord(op_name=op_name)
    record.native_sac_contexts = create_selective_checkpoint_contexts(
        _native_region_sac_policy
    )
    region_state.records[op_name] = record
    return record


def _expect_native_sac_context(
    record: _OpRecord,
    *,
    phase: _Phase,
) -> Any:
    """Return the native SAC context for the current checkpoint phase."""

    if record.native_sac_contexts is None:
        raise RuntimeError(
            f"No SAC context available for native region {record.op_name}"
        )

    forward_context, recompute_context = record.native_sac_contexts
    if phase is _Phase.FORWARD:
        return forward_context
    return recompute_context


def _record_native_sac_outputs(
    record: _OpRecord,
    func: Callable[..., Any],
    output: Any,
) -> None:
    """Record live SAC output refs under unstable report labels."""

    op_name = _native_op_name(func)
    op_index = record.native_op_counts.get(op_name, 0)
    record.native_op_counts[op_name] = op_index + 1
    tensor_outputs = tuple(_iter_tensor_outputs(output))
    if not tensor_outputs:
        return

    base_name = f"{op_name}#{op_index}"
    if len(tensor_outputs) == 1:
        record.native_sac_tensors[base_name] = weakref.ref(tensor_outputs[0])
        return

    for output_index, tensor in enumerate(tensor_outputs):
        record.native_sac_tensors[f"{base_name}.{output_index}"] = weakref.ref(tensor)


def _native_op_name(func: Callable[..., Any]) -> str:
    """Return an unstable report label prefix for one dispatched op."""

    return str(func)


def _iter_tensor_outputs(output: Any) -> Iterable[torch.Tensor]:
    """Yield tensor leaves from a SAC op output."""

    if isinstance(output, torch.Tensor):
        yield output
        return

    if isinstance(output, (tuple, list)):
        for item in output:
            yield from _iter_tensor_outputs(item)


def _expect_native_output(output: _T) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Return native region output tensors or raise for unsupported schemas."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and all(
        isinstance(tensor, torch.Tensor) for tensor in output
    ):
        return output
    raise RuntimeError(
        "native_save_region function must return a Tensor or tuple of Tensors"
    )
