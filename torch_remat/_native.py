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
from functools import wraps
from typing import Any, Callable, Iterable, ParamSpec, TypeVar

import torch
from torch.utils import _pytree as pytree
from torch.utils.checkpoint import (
    CheckpointPolicy as _TorchCheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch_remat._api import (
    _display_name,
    _expect_record,
    _OpRecord,
    _Phase,
    _record_trace_op,
    _release_record_after_recompute_if_needed,
    _state,
    _validate_name,
    CheckpointPolicy,
    get_handle,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")
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


def native_op(
    function: Callable[_P, _R],
    name: str | None = None,
    *,
    policy: CheckpointPolicy,
) -> Callable[_P, _R]:
    """Annotate one native PyTorch op call for remat.

    This is the native-op analogue of :func:`remat.op`, for calls to native
    PyTorch APIs (e.g. ``torch.mm``) that cannot host the custom-autograd handle
    protocol. As with :func:`remat.op`, the call site stays close to the plain
    function call:

        ```python
        y = remat.native_op(torch.mm, "native.mm", policy=remat.CheckpointPolicy.SAVE)(x, w)
        ```

    The ``policy`` controls recompute, exactly like a custom op:

    - ``SAVE``: run ``function`` under PyTorch selective activation checkpointing
      so its outputs are saved for backward and the op is not rerun during
      recompute. This also lets the region's output be consumed by a downstream
      ``RECOMPUTE`` op without tripping the placeholder error, since the real
      output was saved.
    - ``RECOMPUTE``: rerun ``function`` during recompute, like a bare native op,
      but with one extra capability: if an input is the output of an upstream
      ``SAVE`` op (which replays as a data-inaccessible placeholder), the real
      input is saved during the original forward and loaded back at recompute so
      the op can rerun. A bare native op in that position would instead raise.

    Arguments are passed to the returned wrapper rather than captured in a
    closure on purpose: the wrapper must see the op's inputs to save/load the
    ones that would otherwise replay as placeholders.
    """

    if not callable(function):
        raise RuntimeError("native_op expects a function as its first argument")
    if name is None:
        raise RuntimeError("native_op(function, ...) expects an op_name")
    _validate_name(name, what="op_name")
    if not isinstance(policy, CheckpointPolicy):
        raise RuntimeError("native_op expects a CheckpointPolicy")

    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        _record_trace_op(name, policy=policy, source="native")
        if policy is CheckpointPolicy.SAVE:
            return _native_save_region(name, lambda: function(*args, **kwargs))
        return _native_recompute_region(name, function, args, kwargs)

    return wrapper


def _native_save_region(op_name: str, function: Callable[[], _T]) -> _T:
    """Run a native PyTorch function region under SAC-style non-recompute.

    All PyTorch ops executed by ``function`` use PyTorch selective activation
    checkpointing to avoid rerunning those ops during recomputation. The return
    value of ``function`` is the native region boundary output used in reports.
    """

    state = _state.get()
    if state is None:
        return function()

    if state.phase is _Phase.FORWARD:
        if op_name in state.region_state.records:
            raise RuntimeError(
                "Duplicate torch_remat op name: "
                f"{_display_name(state.region_state, op_name)}"
            )
        record = _OpRecord(op_name=op_name)
        record.native_sac_contexts = create_selective_checkpoint_contexts(
            _native_region_sac_policy
        )
        state.region_state.records[op_name] = record
    else:
        record = _expect_record(state.region_state, op_name)
    sac_context = _expect_native_sac_context(record, phase=state.phase)

    token = _native_region.set(_NativeRegion(op_name=op_name))
    try:
        with sac_context:
            output = function()

        if state.phase is _Phase.FORWARD:
            if isinstance(output, torch.Tensor):
                output_tensors: torch.Tensor | tuple[torch.Tensor, ...] = output
            elif isinstance(output, tuple) and all(
                isinstance(tensor, torch.Tensor) for tensor in output
            ):
                output_tensors = output
            else:
                raise RuntimeError(
                    "native_op function must return a Tensor or tuple of Tensors"
                )
            record.record_output_schema(output_tensors)
        else:
            _release_record_after_recompute_if_needed(record)
        return output
    finally:
        _native_region.reset(token)


class _RestoreInputs(torch.autograd.Function):
    """A "boring" remat op that just saves and restores its inputs.

    This is a real remat-tape op, driven by the same handle protocol as
    :func:`remat.op` (``get_handle`` then ``save_or_load_inputs`` then
    ``record_outputs``). On values it is the identity; its only job is that an
    input whose producer is a SAVE op -- which replays as a data-inaccessible
    placeholder during recompute -- is saved in the original forward and loaded
    back here on recompute.

    Being a custom autograd Function is what makes a RECOMPUTE native op work:
    ``.apply`` is called with the placeholder (so the output stays wired to the
    producer's recompute backward node), while the forward body runs under
    no-grad and substitutes the loaded real data. The native op then runs on real
    tensors that are still connected to the recompute graph -- the same thing a
    custom remat op gets for free, supplied here explicitly because a native op
    has no Function boundary of its own.
    """

    @staticmethod
    def forward(ctx: Any, op_name: str, *tensors: torch.Tensor) -> tuple[Any, ...]:
        handle = get_handle(ctx, op_name, CheckpointPolicy.RECOMPUTE)
        loaded = handle.save_or_load_inputs(*tensors)
        loaded = (loaded,) if isinstance(loaded, torch.Tensor) else tuple(loaded)
        # view_as gives a distinct object per output so this single Function node
        # attaches its grad_fn uniformly (returning an input unchanged would
        # leave that output wired to the input's grad_fn instead).
        views = tuple(tensor.view_as(tensor) for tensor in loaded)
        # Finalize with no outputs: these views are just the restored inputs
        # handed to the native op and must not be recorded or stubbed. For a
        # RECOMPUTE op record_outputs records nothing anyway -- its only effect
        # here is releasing the retained inputs from the tape after recompute.
        handle.record_outputs()
        return views

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple[Any, ...]:
        del ctx
        # No gradient for op_name; pass each tensor's gradient straight through.
        return (None, *grads)


def _native_recompute_region(
    op_name: str,
    function: Callable[_P, _R],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _R:
    """Run a native PyTorch op that should be recomputed during backward.

    The op's tensor inputs are routed through :class:`_RestoreInputs` so that any
    input which would otherwise replay as a placeholder is restored from the
    tape, then the op runs normally and rebuilds its own backward.
    """

    state = _state.get()
    if state is None:
        return function(*args, **kwargs)

    leaves, spec = pytree.tree_flatten((args, kwargs))
    tensor_positions = [
        index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)
    ]
    restored = _RestoreInputs.apply(
        op_name, *(leaves[index] for index in tensor_positions)
    )
    for position, tensor in zip(tensor_positions, restored):
        leaves[position] = tensor

    new_args, new_kwargs = pytree.tree_unflatten(leaves, spec)
    return function(*new_args, **new_kwargs)


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
        # Record live SAC output refs under unstable report labels such as
        # aten.mm.default#0. These are report-only and live only while the
        # tensors do.
        record = state.region_state.records[native.op_name]
        op_label = str(func)
        op_index = record.native_op_counts.get(op_label, 0)
        record.native_op_counts[op_label] = op_index + 1
        tensor_outputs = tuple(_iter_tensor_outputs(op_output))
        base_name = f"{op_label}#{op_index}"
        if len(tensor_outputs) == 1:
            record.native_sac_tensors[base_name] = weakref.ref(tensor_outputs[0])
        else:
            for output_index, tensor in enumerate(tensor_outputs):
                record.native_sac_tensors[f"{base_name}.{output_index}"] = weakref.ref(
                    tensor
                )

    return _TorchCheckpointPolicy.MUST_SAVE


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


def _iter_tensor_outputs(output: Any) -> Iterable[torch.Tensor]:
    """Yield tensor leaves from a SAC op output."""

    if isinstance(output, torch.Tensor):
        yield output
        return

    if isinstance(output, (tuple, list)):
        for item in output:
            yield from _iter_tensor_outputs(item)
