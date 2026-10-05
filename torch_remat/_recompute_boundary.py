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
rederivation could be missed (see :func:`torch_remat._api._rederive_saved_inputs`).

The gadget here forces the issue without blocking checkpoint's early stop:

- :class:`_ReplayAnchor` runs first in the checkpointed function and saves a
  zero-element tensor through checkpoint's hooks, so it is the first checkpoint
  holder in both the forward and the replay.
- :class:`_TriggerCheckpointRecompute` wraps every region-output tensor. It saves
  nothing itself; its backward unpacks the anchor's holder, which drives the replay
  before any inner saved-tensor unpack runs.

Because the trigger holder is the *first* pack rather than the last, checkpoint's
early stop ends the replay after the last real pack, so trailing operations that save
nothing for backward (e.g. a residual add) are not replayed. :class:`_ReplayMarker`
adds a throwaway pack where the replay must reach a later point.

This is a workaround. The clean fix is a core API to tell non-reentrant checkpoint to
begin recompute at a chosen point; if that lands, this whole module deletes and
:func:`torch_remat._api.checkpoint` calls the core primitive directly.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from torch_remat._pytree import map_value


class _ReplayAnchor(torch.autograd.Function):
    """Register the first checkpoint holder; the region boundary unpacks it."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any, carrier: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(torch.empty((0,), device=carrier.device))
        # Graph tasks whose backward already unpacked the holder: checkpoint allows
        # one unpack per holder per backward, and several outputs share the anchor.
        ctx.unpacked_graph_tasks = set()
        return carrier.view_as(carrier)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_carrier: torch.Tensor
    ) -> torch.Tensor:
        return grad_carrier


class _ReplayMarker(torch.autograd.Function):
    """Add one checkpoint pack so early-stopped replay reaches this point."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any, carrier: torch.Tensor
    ) -> torch.Tensor:
        # The node and its holder die right away; a dead holder still counts toward
        # the number of packs that replay must reproduce before stopping.
        ctx.save_for_backward(torch.empty((0,), device=carrier.device))
        return carrier.view_as(carrier)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_carrier: torch.Tensor
    ) -> torch.Tensor:
        return grad_carrier


def _new_replay_anchor(device: torch.device | None) -> Any:
    """Create the replay anchor on ``device``; returns its autograd node."""

    carrier = torch.empty((0,), device=device, requires_grad=True)
    return _ReplayAnchor.apply(carrier).grad_fn


def _mark_replay_point(device: torch.device | None) -> None:
    """Record a checkpoint pack so replay does not stop before this point."""

    _ReplayMarker.apply(torch.empty((0,), device=device, requires_grad=True))


class _TriggerCheckpointRecompute(torch.autograd.Function):
    """Autograd identity at the region output whose backward triggers replay."""

    @staticmethod
    def forward(ctx: Any, output: torch.Tensor, anchor: Any) -> torch.Tensor:
        ctx.anchor = anchor
        return output.view_as(output)

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        # Trigger non-reentrant checkpoint's saved-tensor unpack hook at the
        # user-visible boundary before nested custom backward bodies run.
        anchor = ctx.anchor
        graph_task = torch._C._current_graph_task_id()
        if graph_task not in anchor.unpacked_graph_tasks:
            anchor.unpacked_graph_tasks.add(graph_task)
            (_,) = anchor.saved_tensors
        return grad_output, None


if importlib.util.find_spec("spmd_types") is not None:
    import spmd_types  # pyrefly: ignore[missing-import]

    spmd_types.register_local_autograd_function(_TriggerCheckpointRecompute)
    spmd_types.register_local_autograd_function(_ReplayAnchor)
    spmd_types.register_local_autograd_function(_ReplayMarker)


def _checkpoint_recompute_boundary(output: Any, anchor: Any) -> Any:
    """Force non-reentrant checkpoint replay before nested custom backprop."""

    return map_value(lambda leaf: _trigger_boundary(leaf, anchor), output)


def _trigger_boundary(leaf: object, anchor: Any) -> object:
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
    return _TriggerCheckpointRecompute.apply(leaf, anchor)
