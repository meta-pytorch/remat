# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""PyTorch-version compatibility shims for torch_remat.

Everything here papers over a gap in the installed PyTorch and can be deleted once
the minimum supported version catches up; each shim names the upstream fix that
obsoletes it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, TypeAlias

# PyTorch non-reentrant checkpoint expects a callable returning one context for
# original forward and one context for recompute.
CheckpointContextFn: TypeAlias = Callable[
    [],
    tuple[
        contextlib.AbstractContextManager[None],
        contextlib.AbstractContextManager[None],
    ],
]


def _torch_checkpoint_with_forward_exception_cleanup(
    function: Callable[..., Any],
    function_args: tuple[Any, ...],
    function_kwargs: dict[str, Any],
    context_fn: CheckpointContextFn,
    determinism_check: str,
    preserve_rng_state: bool,
) -> Any:
    """Run PyTorch non-reentrant checkpoint with local exception cleanup.

    This is the public ``torch.utils.checkpoint.checkpoint`` non-reentrant branch
    plus ``gen.close()`` when the user forward raises, so a failing forward does
    not leave the checkpoint hook and our forward context installed.
    Upstream fix: https://github.com/pytorch/pytorch/pull/184018
    """

    from torch.utils.checkpoint import _checkpoint_without_reentrant_generator

    checkpoint_debug = False
    checkpoint_early_stop = True
    gen = _checkpoint_without_reentrant_generator(
        function,
        preserve_rng_state,
        context_fn,
        determinism_check,
        checkpoint_debug,
        checkpoint_early_stop,
        *function_args,
        **function_kwargs,
    )
    next(gen)
    try:
        ret = function(*function_args, **function_kwargs)
    except BaseException:
        gen.close()
        raise

    try:
        next(gen)
    except StopIteration:
        return ret
    raise RuntimeError("torch.utils.checkpoint generator did not stop")
