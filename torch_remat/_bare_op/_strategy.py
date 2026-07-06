# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Bare-op detection strategies: one object per way of representing / intercepting SAVE outputs.

``checkpoint(..., detect_bare_ops=...)`` selects a strategy. Each :class:`_BareOpStrategy`
answers three questions with a small callable plus one static ``wraps_outputs`` flag, so
the core (:mod:`torch_remat._api`) never branches on the strategy name -- adding a
strategy is adding one row to :func:`_strategies`.

The five strategies:

| name            | SAVE output value    | in index | wraps outputs | forward mode        |
|-----------------|----------------------|----------|---------------|---------------------|
| ``none``        | plain tensor         | yes      | no            | --                  |
| ``subclass``    | ``_SaveTensor``      | yes      | yes           | --                  |
| ``proxy``       | ``_SaveProxy``       | no       | yes           | --                  |
| ``dispatch_mode`` | plain tensor       | yes      | no            | ``TorchDispatchMode`` |
| ``function_mode`` | plain tensor       | yes      | no            | ``TorchFunctionMode`` |

The two mode strategies represent outputs exactly like ``none``; they differ only by
installing a forward-phase mode that intercepts bare consumers.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import torch
from torch_remat._bare_op._common import (
    _SaveOutputHandle,
    _unwrap_identity,
    PersistOutputThunk,
)
from torch_remat._bare_op._dispatch_mode import _SaveDispatchMode
from torch_remat._bare_op._function_mode import _SaveFunctionMode
from torch_remat._bare_op._proxy import (
    _make_save_proxy,
    _save_proxy_handle,
    _SaveProxy,
)
from torch_remat._bare_op._subclass import _make_save_tensor, _unwrap_save_tensor_leaf

if TYPE_CHECKING:
    from torch_remat._region import _CheckpointRegionState

# The value a SAVE op returns for one output, plus the handle to register in the region's
# save-output index under it -- or None when the value self-identifies by type (proxy).
_MadeOutput = tuple[Any, "_SaveOutputHandle | None"]


@dataclass(frozen=True)
class _BareOpStrategy:
    """How one bare-op detection strategy represents and intercepts SAVE outputs.

    Frozen and stateless: one shared instance per strategy name.
    """

    name: str
    # Builds the forward stand-in for a SAVE output: (value the op returns, handle to
    # register in the save-output index under it -- or None when the value
    # self-identifies by type, i.e. the proxy).
    make_output: Callable[[torch.Tensor, PersistOutputThunk], _MadeOutput]
    # Recovers the handle from a self-identifying value (the proxy); None otherwise.
    typed_handle: Callable[[object], "_SaveOutputHandle | None"]
    # Mode installed around the original forward; a null context for non-mode strategies.
    forward_mode: Callable[
        ["_CheckpointRegionState"], contextlib.AbstractContextManager[None]
    ]
    # True when make_output wraps the SAVE output in a carrier (subclass / proxy). A
    # remat.op consumer reads this to decide statically whether a SAVE-output input
    # needs unwrapping (and an arg-pytree rebuild), or only a persist-output trigger.
    wraps_outputs: bool


def _plain_output(
    real: torch.Tensor, persist_output: PersistOutputThunk
) -> _MadeOutput:
    """A plain (already grad-connected) SAVE output, indexed with an identity unwrap."""

    return real, _SaveOutputHandle(
        persist_output=persist_output, unwrap=_unwrap_identity
    )


def _subclass_output(
    real: torch.Tensor, persist_output: PersistOutputThunk
) -> _MadeOutput:
    """A ``__torch_dispatch__`` wrapper SAVE output, indexed with a grad-connected unwrap."""

    value = _make_save_tensor(real, persist_output=persist_output)
    return value, _SaveOutputHandle(
        persist_output=persist_output, unwrap=_unwrap_save_tensor_leaf
    )


def _proxy_output(
    real: torch.Tensor, persist_output: PersistOutputThunk
) -> _MadeOutput:
    """A ``__torch_function__`` proxy SAVE output; None handle -- it self-identifies by type."""

    return _make_save_proxy(real, persist_output=persist_output), None


def _proxy_typed_handle(leaf: object) -> _SaveOutputHandle | None:
    """Recover a proxy's handle, or None for anything that is not a proxy."""

    if isinstance(leaf, _SaveProxy):
        return _save_proxy_handle(leaf)
    return None


def _no_typed_handle(leaf: object) -> _SaveOutputHandle | None:
    """Handle lookup for index-backed representations (never self-identifying)."""

    del leaf
    return None


def _no_forward_mode(
    region_state: _CheckpointRegionState,
) -> contextlib.AbstractContextManager[None]:
    """Forward mode for strategies that intercept via the output representation, not a mode."""

    del region_state
    return contextlib.nullcontext()


# Built lazily on first use (avoids module-scope work at import time) and cached: the
# strategies are stateless singletons keyed by name.
_STRATEGY_CACHE: dict[str, _BareOpStrategy] | None = None


def _strategies() -> dict[str, _BareOpStrategy]:
    """Return the name -> strategy mapping, building it once."""

    global _STRATEGY_CACHE
    if _STRATEGY_CACHE is None:
        _STRATEGY_CACHE = {
            strategy.name: strategy
            for strategy in (
                _BareOpStrategy(
                    "none",
                    _plain_output,
                    _no_typed_handle,
                    _no_forward_mode,
                    wraps_outputs=False,
                ),
                _BareOpStrategy(
                    "subclass",
                    _subclass_output,
                    _no_typed_handle,
                    _no_forward_mode,
                    wraps_outputs=True,
                ),
                _BareOpStrategy(
                    "proxy",
                    _proxy_output,
                    _proxy_typed_handle,
                    _no_forward_mode,
                    wraps_outputs=True,
                ),
                _BareOpStrategy(
                    "dispatch_mode",
                    _plain_output,
                    _no_typed_handle,
                    _SaveDispatchMode,
                    wraps_outputs=False,
                ),
                _BareOpStrategy(
                    "function_mode",
                    _plain_output,
                    _no_typed_handle,
                    _SaveFunctionMode,
                    wraps_outputs=False,
                ),
            )
        }
    return _STRATEGY_CACHE


def _bare_op_strategy(name: str) -> _BareOpStrategy:
    """Return the strategy object for a resolved strategy name."""

    strategy = _strategies().get(name)
    if strategy is None:
        raise ValueError(f"unknown torch_remat bare-op strategy {name!r}")
    return strategy


def _resolve_detect_bare_ops(detect_bare_ops: bool | str) -> str:
    """Map a ``detect_bare_ops`` flag to a strategy name.

    ``True`` -> ``"subclass"`` (the default strategy); ``False`` -> ``"none"`` (opt out); a
    string names the strategy explicitly (``"subclass"`` / ``"proxy"`` / ``"dispatch_mode"``
    / ``"function_mode"``) so callers can pick between the wrapper and mode implementations.
    """

    if detect_bare_ops is False:
        return "none"
    if detect_bare_ops is True:
        return "subclass"
    if isinstance(detect_bare_ops, str) and detect_bare_ops in _strategies():
        return detect_bare_ops
    raise ValueError(
        "detect_bare_ops must be a bool or one of "
        "'subclass'/'proxy'/'dispatch_mode'/'function_mode', "
        f"got {detect_bare_ops!r}"
    )
