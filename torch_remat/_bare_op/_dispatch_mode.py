# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""``TorchDispatchMode`` bare-op detection -- the mode analogue of the subclass.

The ``"dispatch_mode"`` strategy leaves SAVE outputs as *plain* tensors and installs
:class:`_SaveDispatchMode` around the original forward instead. With no subclass there
is nothing to unwrap: the mode fires the producer's persist-output for any op touching
a SAVE output, then calls the op on the already-plain arguments. Like the subclass, it
fires on views too (no view deferral -- contrast the proxy / ``"function_mode"``).

Limitation: ``tensor.data_ptr()`` is a direct C++ accessor that does not go through
``__torch_dispatch__``, so a raw Triton/cutedsl kernel reading a SAVE output's pointer
is NOT intercepted here (the subclass and function mode do see it). Use one of those
if bare raw-pointer kernels consume SAVE outputs.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch_remat._bare_op._common import (
    _bare_op_detection_suppressed,
    _inplace_message,
    _SaveOutputHandle,
)
from torch_remat._pytree import iter_arg_leaves

if TYPE_CHECKING:
    from torch_remat._region import _CheckpointRegionState


class _SaveDispatchMode(TorchDispatchMode):
    """Fire a SAVE output's persist-output when a dispatched op touches it.

    Installed only during the original forward. Every op is checked for a SAVE-output
    argument; if found, a mutating op is rejected, else the producer's persist-output
    fires before the op runs on the plain arguments.
    """

    def __init__(self, region_state: _CheckpointRegionState) -> None:
        super().__init__()
        self._region_state = region_state

    def __torch_dispatch__(
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
        handles = _touched_save_handles(self._region_state, args, kwargs)
        if handles:
            _reject_if_mutating(func)
            for handle in handles:
                handle.persist_output()
        return func(*args, **kwargs)


def _touched_save_handles(
    region_state: _CheckpointRegionState,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[_SaveOutputHandle]:
    """Return the SAVE-output handle for each SAVE-output tensor leaf in a call.

    Guarded by the tensor check -- ``WeakTensorKeyDictionary.get`` raises on a non-tensor
    key. The one-hop leaf walk also reaches SAVE outputs inside a list/tuple argument
    (e.g. ``torch.cat([save, other])``).
    """

    index = region_state.save_output_index
    handles: list[_SaveOutputHandle] = []
    for _token, value in iter_arg_leaves(args, kwargs):
        if isinstance(value, torch.Tensor):
            handle = index.get(value)
            if handle is not None:
                handles.append(handle)
    return handles


def _reject_if_mutating(func: Callable[..., Any]) -> None:
    """Raise the in-place diagnostic if ``func`` is a mutating / out aten op."""

    schema = getattr(func, "_schema", None)
    if schema is not None and schema.is_mutable:
        raise RuntimeError(_inplace_message(func))
