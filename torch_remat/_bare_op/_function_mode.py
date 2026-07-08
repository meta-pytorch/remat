# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""``TorchFunctionMode`` bare-op detection -- the mode analogue of the proxy.

The ``"function_mode"`` strategy leaves SAVE outputs as *plain* tensors and installs
:class:`_SaveFunctionMode` around the original forward. ``TorchFunctionMode``
intercepts every torch call -- operators, methods, ``data_ptr`` included -- even on
plain tensors, so none of the proxy's manual dunder plumbing is needed. Semantics
match the proxy: a **view** of a touched SAVE output *defers* (its outputs are
registered back into the save-output index under the same producer persist-output);
any other op fires each touched producer's persist-output once; an in-place op is
rejected. Unlike ``"dispatch_mode"``, ``data_ptr`` IS intercepted, so raw-pointer
kernels on a SAVE output work.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch
from torch.overrides import TorchFunctionMode
from torch_remat._bare_op._common import (
    _bare_op_detection_suppressed,
    _BaseRetainingPersist,
    _inplace_message,
    _SaveOutputHandle,
    _snapshot_unannotated_inplace,
    _storage_id,
    _unannotated_inplace_mutated,
    _unwrap_identity,
    _view_base_index,
    PersistOutputThunk,
)
from torch_remat._pytree import iter_arg_leaves, value_leaves

if TYPE_CHECKING:
    from torch_remat._region import _CheckpointRegionState


class _SaveFunctionMode(TorchFunctionMode):
    """Defer on views and persist on compute when a torch call touches a SAVE output.

    Installed only during the original forward (see the ``"function_mode"`` strategy). It
    holds the region's save-output index; a call with no SAVE-output argument passes
    straight through. Otherwise it runs the call, rejects it if it mutated a touched input,
    and -- mirroring the proxy -- either defers (view: rewraps the outputs into the index)
    or fires each touched producer's persist-output (compute).
    """

    def __init__(self, region_state: _CheckpointRegionState) -> None:
        super().__init__()
        self._region_state = region_state

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        kwargs = {} if kwargs is None else kwargs
        if _bare_op_detection_suppressed():
            return func(*args, **kwargs)
        touched = _touched_save_outputs(self._region_state, args, kwargs)
        if not touched:
            return func(*args, **kwargs)

        tensors = [tensor for tensor, _handle in touched]
        versions = [tensor._version for tensor in tensors]
        snapshots = _snapshot_unannotated_inplace(func, tensors)
        out = func(*args, **kwargs)
        if any(
            tensor._version != version for tensor, version in zip(tensors, versions)
        ) or _unannotated_inplace_mutated(snapshots, tensors):
            raise RuntimeError(_inplace_message(func))

        index = _view_base_index(out, [_storage_id(tensor) for tensor, _ in touched])
        if index is not None:
            # Defer under the producer whose storage the result aliases, whatever
            # argument position it came in at (see _view_base_index) -- mirrors the proxy.
            base_tensor, base_handle = touched[index]
            _defer_view(
                self._region_state, out, base_handle.persist_output, base_tensor
            )
            return out

        for _tensor, handle in touched:
            handle.persist_output()
        return out


def _touched_save_outputs(
    region_state: _CheckpointRegionState,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[tuple[torch.Tensor, _SaveOutputHandle]]:
    """Return each SAVE-output tensor leaf in a call paired with its handle.

    Guarded by the tensor check -- ``WeakTensorKeyDictionary.get`` raises on a non-tensor
    key. The one-hop leaf walk also reaches SAVE outputs inside a list/tuple argument.
    """

    index = region_state.save_output_index
    touched: list[tuple[torch.Tensor, _SaveOutputHandle]] = []
    for _token, value in iter_arg_leaves(args, kwargs):
        if isinstance(value, torch.Tensor):
            handle = index.get(value)
            if handle is not None:
                touched.append((value, handle))
    return touched


def _defer_view(
    region_state: _CheckpointRegionState,
    out: object,
    persist_output: PersistOutputThunk,
    base: torch.Tensor,
) -> None:
    """Register a view op's outputs in the save-output index under the producer's save.

    A view of a SAVE output is itself a (deferred) SAVE output: consuming it triggers
    the producer's persist-output just as consuming the output would -- the proxy's
    view-rewrap behavior, recorded in the index instead of a wrapper object. The save
    is wrapped to retain ``base`` because the producer's persist-output holds its
    output only weakly (see :class:`_BaseRetainingPersist`).
    """

    retaining = _BaseRetainingPersist(persist_output, base)
    for leaf in value_leaves(out):
        if isinstance(leaf, torch.Tensor):
            region_state.save_output_index[leaf] = _SaveOutputHandle(
                persist_output=retaining,
                unwrap=_unwrap_identity,
            )
