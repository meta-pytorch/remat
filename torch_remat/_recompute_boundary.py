# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Force non-reentrant checkpoint replay to begin at the region output boundary.

Non-reentrant checkpoint starts replay lazily, when backward first unpacks a tensor
saved under checkpoint's holder hooks. For a region with nested custom autograd
Functions that is the wrong moment: replay would begin inside an inner backward body
rather than at the region output, so a skipped SAVE op's recompute-sourced saved-input
rederivation could be missed (see the load-bearing invariant in
:func:`torch_remat._api._rederive_saved_inputs`).

The gadget here forces the issue. :func:`_checkpoint_recompute_boundary` wraps every
region-output tensor in :class:`_TriggerCheckpointRecompute`, an autograd identity that
saves a zero-element tensor through checkpoint's hooks. Because it sits at the region
output it is the highest-index checkpoint holder, so the *first* backward unpack fires
its hook and drives a full replay before any inner saved-tensor unpack runs.

This is a workaround. The clean fix is a core API to tell non-reentrant checkpoint to
begin recompute at a chosen point; if that lands, this whole module deletes and
:func:`torch_remat._api.checkpoint` calls the core primitive directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_remat._pytree import map_value


class _TriggerCheckpointRecompute(torch.autograd.Function):
    """Autograd identity that installs one checkpoint-hook unpack boundary."""

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor) -> torch.Tensor:
        # Save a zero-element tensor with the same dtype/device as the output so
        # PyTorch's checkpoint unpack hook fires without retaining output storage.
        ctx.save_for_backward(
            torch.empty((0,), dtype=output.dtype, device=output.device)
        )
        return output.view_as(output)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # Trigger non-reentrant checkpoint's saved-tensor unpack hook at the
        # user-visible boundary before nested custom backward bodies run.
        (_,) = ctx.saved_tensors
        return grad_output


def _checkpoint_recompute_boundary(output: Any) -> Any:
    """Force non-reentrant checkpoint replay before nested custom backprop."""

    return map_value(_trigger_boundary, output)


def _trigger_boundary(leaf: object) -> object:
    """Install the checkpoint-recompute trigger on one region-output tensor leaf.

    A region output is a plain tensor (SAVE outputs are no longer wrapped), so the
    trigger just views the leaf. On the original forward the value rides the boundary
    trigger; on recompute the return value is discarded, so a placeholder here is fine.
    """

    if not isinstance(leaf, torch.Tensor):
        raise RuntimeError(
            "torch_remat checkpoint function must return a Tensor, or one hop of "
            "tuple/list of Tensors"
        )
    return _TriggerCheckpointRecompute.apply(leaf)
