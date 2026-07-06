# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The ``__torch_dispatch__`` wrapper-subclass bare-op strategy (``"subclass"``, the default).

:class:`_SaveTensor` is a **wrapper** ``torch.Tensor`` subclass holding a SAVE op's
produced tensor as ``_inner``. A ``remat.op`` / boundary consumer just reads
``_inner``; a *bare* op trips :meth:`_SaveTensor.__torch_dispatch__`, which fires the
producer's ``persist_output`` (once) and runs the op on the unwrapped inner -- every
output is plain, so the wrapper never propagates past one hop.

The wrapper is autograd-connected to the producer through :class:`_WrapSave` (a plain
``_make_wrapper_subclass`` instance would be a grad *leaf*), so gradient from a bare
consumer flows wrapper -> ``_WrapSave`` -> ``_inner`` -> producer.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
from torch_remat._bare_op._common import _inplace_message, PersistOutputThunk
from torch_remat._pytree import iter_arg_leaves, map_arg_leaves


class _SaveTensor(torch.Tensor):
    """Wrapper Tensor subclass carrying a SAVE op's real output as ``_inner``.

    See the module docstring for the dispatch/unwrap flow. In-place / out ops are an
    error: mutating this tensor would corrupt both the persisted value and the copy
    autograd kept for the SAVE op's backward.
    """

    # The wrapped real output and the producer's persist-output thunk. Set in __new__
    # on the wrapper subclass instance.
    _inner: torch.Tensor
    _persist_output: PersistOutputThunk

    @staticmethod
    def __new__(
        cls,
        inner: torch.Tensor,
        persist_output: PersistOutputThunk,
    ) -> _SaveTensor:
        wrapper = cast(
            "_SaveTensor",
            torch.Tensor._make_wrapper_subclass(
                cls,
                inner.shape,
                strides=inner.stride(),
                dtype=inner.dtype,
                device=inner.device,
                requires_grad=inner.requires_grad,
            ),
        )
        wrapper._inner = inner
        wrapper._persist_output = persist_output
        return wrapper

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        kwargs = {} if kwargs is None else kwargs

        try:
            schema = object.__getattribute__(func, "_schema")
        except AttributeError:
            schema = None
        if schema is not None and schema.is_mutable:
            raise RuntimeError(_inplace_message(func))

        # Producer responsibility: the first bare op to touch a SAVE output records
        # its value so recompute can reproduce it. Fire before unwrapping so the
        # wrapper identity is still visible. The one-hop walk also reaches wrapper
        # leaves inside a list/tuple argument (e.g. torch.cat([save, other])).
        _persist_output_inputs(args, kwargs)

        new_args, new_kwargs = map_arg_leaves(
            lambda _token, value: _unwrap_leaf(value), args, kwargs
        )
        return func(*new_args, **new_kwargs)

    def data_ptr(self) -> int:
        # Handing a SAVE output straight to a Triton/cutedsl kernel reads its raw
        # pointer without a dispatchable op, so persist-output here too -- then return
        # the inner's real pointer so the kernel runs on real data and recompute
        # reproduces it.
        persist_output = _persist_output_of(self)
        if persist_output is not None:
            persist_output()
        return self._inner.data_ptr()


class _WrapSave(torch.autograd.Function):
    """Wrap a SAVE op's produced tensor into a grad-connected forward wrapper.

    A bare ``_make_wrapper_subclass`` instance is an autograd *leaf*, so gradient from a
    bare consumer would dead-end at the wrapper. Building it through this Function gives the
    wrapper a grad_fn whose backward routes gradient to ``_inner`` (hence to the producer).
    ``persist_output`` is a non-tensor argument, so backward returns ``None`` for it.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inner: torch.Tensor,
        persist_output: PersistOutputThunk,
    ) -> torch.Tensor:
        del ctx
        return _SaveTensor(inner, persist_output)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        del ctx
        return grad_output, None


def _make_save_tensor(
    real: torch.Tensor,
    *,
    persist_output: PersistOutputThunk,
) -> _SaveTensor:
    """Wrap a SAVE op's produced tensor as a forward stand-in holding it as ``_inner``."""

    return _WrapSave.apply(real, persist_output)


def _unwrap_save_tensor(save_tensor: _SaveTensor) -> torch.Tensor:
    """Return the grad-connected real value of a wrapper (unwrap only, no persist)."""

    return save_tensor._inner


def _unwrap_save_tensor_leaf(leaf: object) -> torch.Tensor:
    """Handle unwrap for the subclass strategy: the leaf is the wrapper, return its inner.

    Takes the leaf rather than closing over the wrapper, so the index handle keeps no
    strong reference back to its own key -- see :class:`_SaveOutputHandle`.
    """

    assert isinstance(leaf, _SaveTensor)
    return _unwrap_save_tensor(leaf)


def _persist_output_of(save_tensor: torch.Tensor) -> PersistOutputThunk | None:
    """Return the persist-output thunk carried by a wrapper, or None."""

    try:
        return object.__getattribute__(save_tensor, "_persist_output")
    except AttributeError:
        return None


def _unwrap_leaf(value: Any) -> Any:
    """Unwrap one ``__torch_dispatch__`` leaf to its plain inner tensor.

    Autograd was already recorded above the dispatcher on the wrapper inputs, so running the
    op on the grad-tracking inner adds no second autograd node.
    """

    if isinstance(value, _SaveTensor):
        return value._inner
    return value


def _persist_output_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Fire the persist-output thunk of every wrapper leaf in a call."""

    for _token, value in iter_arg_leaves(args, kwargs):
        if isinstance(value, _SaveTensor):
            persist_output = _persist_output_of(value)
            if persist_output is not None:
                persist_output()
