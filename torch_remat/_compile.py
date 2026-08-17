# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""torch.compile support for torch_remat.

Eagerly a :func:`torch_remat.region` drives a saved-tensor tape: the region body is
skipped on recompute and its outputs are served from tensors kept on the original
forward graph. Under ``torch.compile`` there is no second Python execution -- there is
a single AOTAutograd trace and the min-cut partitioner performs recompute -- so that
tape machinery is bypassed entirely. This module instead translates a region's
``recompute`` flag into the node-level save/recompute tags the partitioner already
understands (``node.meta["recompute"]``, read by ``must_recompute`` in
``torch._functorch.partitioners``).

The mechanism, region by region:

* :func:`compiled_checkpoint` runs the whole region under an ordinary non-reentrant
  ``torch.utils.checkpoint``. Under compile the partitioner recomputes such a region
  wholesale -- that is torch_remat's "everything recomputes by default", including
  bare ops with no :func:`region` wrapper.
* :func:`compiled_region` pins a ``recompute=False`` (SAVE) region's own compute nodes
  back to ``MUST_SAVE`` so the partitioner keeps them. It routes the body through the
  ``dynamo_bypassing_wrapper`` HOP: Dynamo inlines the body into the graph (Inductor
  still fuses it) but runs our wrapper at AOT proxy-trace time, the one point where the
  body's decomposed fx nodes are reachable. Views carry infinite save-weight, so
  tagging them is a no-op and only real compute nodes are saved -- which is what makes
  this robust to composite ops (an ``nn.Linear`` on a 3-D input decomposes to
  view/addmm/view; a per-saved-tensor marker could land on a view, a whole-region tag
  cannot).

A ``recompute=True`` region needs no tag: the enclosing checkpoint already recomputes
it. A region used outside any checkpoint is a transparent call, matching eager.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper
from torch.utils.checkpoint import checkpoint as _torch_checkpoint, CheckpointPolicy

# Whether a compiled remat.checkpoint region is currently being traced. ContextVar
# mutation is unsupported by Dynamo in fullgraph mode, so this is a plain module global.
# It is read from region() as Dynamo symbolically executes the checkpoint body, so its
# value is resolved at trace time. region() tags nodes only while inside a checkpoint,
# matching eager's pass-through-outside-a-region behavior. Nesting is banned (see
# compiled_checkpoint), so a bool suffices -- no depth counter needed.
_in_compiled_checkpoint: bool = False


def in_compiled_checkpoint() -> bool:
    """Whether a compiled ``remat.checkpoint`` region is currently being traced."""

    return _in_compiled_checkpoint


def _tag_must_save(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
    """``dynamo_bypassing_wrapper`` wrapper: run ``inner_fn``, tag its nodes to save.

    Runs at AOT proxy-trace time (``get_proxy_mode()`` live): it records the graph size
    before the body runs, then stamps ``node.meta["recompute"] = MUST_SAVE`` on every
    ``call_function`` node the body appended. Off the proxy-trace path (eager execution
    under compile, or a non-proxy AOT pass) there is no graph to tag, so it just runs the
    body.
    """

    def run(*args: Any, **kwargs: Any) -> Any:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode

        proxy_mode = get_proxy_mode()
        if proxy_mode is None:
            return inner_fn(*args, **kwargs)
        graph = proxy_mode.tracer.graph
        num_nodes_before = len(graph.nodes)
        output = inner_fn(*args, **kwargs)
        for node in list(graph.nodes)[num_nodes_before:]:
            if node.op == "call_function":
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        return output

    return run


def compiled_region(
    function: Callable[..., Any],
    recompute: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run a :func:`torch_remat.region` call under ``torch.compile``.

    A ``recompute=True`` region, or any region outside a checkpoint, is a transparent
    call (the enclosing checkpoint already recomputes by default). A ``recompute=False``
    (SAVE) region routes through ``dynamo_bypassing_wrapper`` to tag its decomposed
    nodes ``MUST_SAVE``.
    """

    if recompute or not in_compiled_checkpoint():
        return function(*args, **kwargs)
    return dynamo_bypassing_wrapper(
        _tag_must_save,
        function,
        *args,
        **kwargs,
    )


def compiled_checkpoint(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    determinism_check: str,
) -> Any:
    """Run a :func:`torch_remat.checkpoint` region under ``torch.compile``.

    The region recomputes by default: run it under a plain non-reentrant
    ``torch.utils.checkpoint``, which the partitioner recomputes wholesale, and let each
    ``recompute=False`` region pin its own nodes back to ``MUST_SAVE`` (see
    :func:`compiled_region`).
    """

    global _in_compiled_checkpoint
    if _in_compiled_checkpoint:
        raise NotImplementedError(
            "nested torch_remat.checkpoint regions are not supported under torch.compile"
        )
    _in_compiled_checkpoint = True
    try:
        return _torch_checkpoint(
            function,
            *args,
            use_reentrant=False,
            determinism_check=determinism_check,
            preserve_rng_state=False,
            **kwargs,
        )
    finally:
        _in_compiled_checkpoint = False
