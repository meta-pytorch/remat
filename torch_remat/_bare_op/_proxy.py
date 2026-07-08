# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The ``__torch_function__`` proxy bare-op strategy (``"proxy"``).

An alternative to the wrapper subclass in :mod:`._subclass`. :class:`_SaveProxy` is a
plain object -- like ``torch.fx.Proxy`` -- that carries the real grad-connected output
as ``_inner``. Not being a tensor, it never enters the autograd graph: the instant an
op touches it, the op runs on ``_inner``, so gradient flows producer -> ``_inner`` ->
consumer with no bridge node. The cost is that operator dunders (``+``, ``@``, ``[]``
...) and method access must be installed manually and routed through
:func:`_dispatch_proxy_op` -- the ``fx.Proxy`` pattern.

Poke semantics (all in :func:`_dispatch_proxy_op`): a **view** op returns a new proxy
and *defers* the save until the view is actually used; any other op ("poked hard")
fires the producer's persist-output once and returns the plain result; an in-place op
is rejected.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import torch
from torch_remat._bare_op._common import (
    _BaseRetainingPersist,
    _inplace_message,
    _SaveOutputHandle,
    _snapshot_unannotated_inplace,
    _storage_id,
    _unannotated_inplace_mutated,
    _view_base_index,
    PersistOutputThunk,
)
from torch_remat._pytree import map_arg_leaves, map_value

# Tensor attributes that read only metadata (never the data), forwarded straight to
# the inner without poking the proxy. Anything not here that resolves to a callable
# is treated as a method and routed through the proxy; a non-callable tensor property
# (``.T``, ``.data`` ...) is read off the inner directly (rare, loses proxy-ness).
_FORWARDED_ATTRS: frozenset[str] = frozenset(
    {
        "shape",
        "dtype",
        "device",
        "layout",
        "requires_grad",
        "ndim",
        "is_cuda",
        "is_meta",
        "is_nested",
        "is_sparse",
        "is_quantized",
        "is_leaf",
        "grad_fn",
        "grad",
        "names",
        "itemsize",
    }
)


class _SaveProxy:
    """A ``__torch_function__`` proxy carrying a SAVE op's real output as ``_inner``.

    Not a ``torch.Tensor`` (so never on the autograd graph). ``_inner`` is the real,
    grad-connected value; ``_persist_output`` records the *producer's* output slot on the
    tape (shared by every view proxy derived from this one); ``_base_storage`` is the
    producer output's storage pointer, against which :func:`_dispatch_proxy_op` classifies
    an op result as a (storage-aliasing) view versus a compute.
    """

    # Not tensors, so identity hashing must be preserved even though ``__eq__`` is
    # overridden below to return an elementwise comparison (tensor semantics).
    __hash__ = object.__hash__

    def __init__(
        self,
        inner: torch.Tensor,
        persist_output: PersistOutputThunk,
        base_storage: int,
    ) -> None:
        self._inner = inner
        self._persist_output = persist_output
        self._base_storage = base_storage
        self._handle = _SaveOutputHandle(
            persist_output=persist_output, unwrap=_unwrap_proxy_inner
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        return _dispatch_proxy_op(func, args, kwargs)

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not found normally (i.e. not the instance attrs set in
        # __init__ nor the dunders below). A metadata attribute is forwarded to the
        # inner; a tensor method is routed through the proxy so its result is
        # view-classified / persisted like any other op.
        inner = self.__dict__["_inner"]
        if name in _FORWARDED_ATTRS:
            return getattr(inner, name)
        tensor_attr = getattr(torch.Tensor, name, None)
        if callable(tensor_attr):
            return lambda *a, **k: _dispatch_proxy_op(tensor_attr, (self, *a), k)
        return getattr(inner, name)

    def __setitem__(self, index: Any, value: Any) -> None:
        raise RuntimeError(_inplace_message("__setitem__"))

    def __len__(self) -> int:
        return len(self._inner)

    def __bool__(self) -> bool:
        # Reading a scalar for control flow consumes the value, so poke the producer.
        self._persist_output()
        return bool(self._inner)

    def __repr__(self) -> str:
        return (
            f"_SaveProxy(shape={tuple(self._inner.shape)}, dtype={self._inner.dtype})"
        )


def _make_save_proxy(
    real: torch.Tensor,
    *,
    persist_output: PersistOutputThunk,
) -> _SaveProxy:
    """Wrap a SAVE op's produced tensor as a ``__torch_function__`` forward stand-in.

    ``real`` is the plain, grad-connected tensor the SAVE op body produced. The proxy holds
    it as ``_inner`` and keys view classification on its storage.
    """

    return _SaveProxy(real, persist_output, _storage_id(real))


def _unwrap_proxy_inner(leaf: object) -> torch.Tensor:
    """Handle unwrap for the proxy strategy: the leaf is the proxy, return its ``_inner``.

    Takes the looked-up leaf rather than closing over the proxy, so the proxy's own handle
    holds no strong reference back to it -- see :class:`_SaveOutputHandle`.
    """

    assert isinstance(leaf, _SaveProxy)
    return leaf._inner


def _save_proxy_handle(proxy: _SaveProxy) -> _SaveOutputHandle:
    """Return the SAVE-output handle a proxy carries (persist-output + grad-connected unwrap).

    The proxy self-identifies by type, so -- unlike the plain / subclass / mode strategies --
    it needs no entry in ``_CheckpointRegionState.save_output_index``; the RECOMPUTE-op
    consume path, SAVE-input snapshot, and region boundary read this handle off the proxy
    directly.
    """

    return proxy._handle


def _dispatch_proxy_op(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> Any:
    """Run one op over proxy arguments, deferring on views and saving on compute.

    Unwrap every proxy leaf to its ``_inner`` and run ``func`` on the reals. An op
    that bumped an inner's ``_version`` mutated a SAVE output in place and is
    rejected. A result aliasing the producer output's storage is a view -- rewrap and
    defer. Otherwise fire each touched producer's persist-output and return the plain
    result.
    """

    kwargs = {} if kwargs is None else kwargs

    saves: list[PersistOutputThunk] = []
    base_storages: list[int] = []
    inners: list[torch.Tensor] = []
    versions: list[int] = []

    def unwrap(_token: object, value: object) -> object:
        if isinstance(value, _SaveProxy):
            saves.append(value._persist_output)
            base_storages.append(value._base_storage)
            inners.append(value._inner)
            versions.append(value._inner._version)
            return value._inner
        return value

    new_args, new_kwargs = map_arg_leaves(unwrap, args, kwargs)
    snapshots = _snapshot_unannotated_inplace(func, inners)
    out = func(*new_args, **new_kwargs)

    if any(
        inner._version != version for inner, version in zip(inners, versions)
    ) or _unannotated_inplace_mutated(snapshots, inners):
        raise RuntimeError(_inplace_message(getattr(func, "__name__", func)))

    index = _view_base_index(out, base_storages)
    if index is not None:
        # Defer: the view rides its producer's save, retaining the tensor it was derived
        # from so the producer output stays resolvable until the view is actually poked --
        # the producer's persist-output holds its output only weakly (see _BaseRetainingPersist).
        # The producer is the one whose storage the result aliases, whatever argument
        # position it came in at (see _view_base_index).
        persist_output = _BaseRetainingPersist(saves[index], inners[index])
        base_storage = base_storages[index]
        return map_value(
            lambda leaf: _rewrap_view(leaf, persist_output, base_storage), out
        )

    for persist_output in saves:
        persist_output()
    return out


def _rewrap_view(
    leaf: object, persist_output: PersistOutputThunk, base_storage: int
) -> object:
    """Wrap one storage-aliasing view leaf back into a proxy over the same producer slot."""

    if isinstance(leaf, torch.Tensor):
        return _SaveProxy(leaf, persist_output, base_storage)
    return leaf


def _install_operator_dunders() -> None:
    """Install the arithmetic / comparison / indexing dunders on the proxy class.

    Python resolves operators (``+``, ``@``, ``[]`` ...) via type-level dunders, which --
    unlike ``torch.*`` calls -- do not consult ``__torch_function__``. Each dunder routes
    through :func:`_dispatch_proxy_op`, so an operator on a proxy is view-classified and
    persisted exactly like a ``torch.*`` call. This mirrors ``torch.fx.Proxy``'s
    magic-method installation.
    """

    binary = [
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "matmul",
        "and_",
        "or_",
        "xor",
        "lshift",
        "rshift",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "getitem",  # indexing / slicing -- a view, classified by _dispatch_proxy_op
    ]
    reflectable = {"add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "matmul"}
    unary = ["neg", "pos", "abs", "invert"]

    for name in binary:
        op = getattr(operator, name)
        dunder = f"__{name.rstrip('_')}__"
        setattr(_SaveProxy, dunder, _make_binary(op))
        if name in reflectable:
            setattr(_SaveProxy, f"__r{name}__", _make_reflected(op))
    for name in unary:
        op = getattr(operator, name)
        setattr(_SaveProxy, f"__{name}__", _make_unary(op))


def _make_binary(op: Callable[..., Any]) -> Callable[..., Any]:
    def dunder(self: _SaveProxy, other: object) -> object:
        return _dispatch_proxy_op(op, (self, other), {})

    return dunder


def _make_reflected(op: Callable[..., Any]) -> Callable[..., Any]:
    def dunder(self: _SaveProxy, other: object) -> object:
        return _dispatch_proxy_op(op, (other, self), {})

    return dunder


def _make_unary(op: Callable[..., Any]) -> Callable[..., Any]:
    def dunder(self: _SaveProxy) -> object:
        return _dispatch_proxy_op(op, (self,), {})

    return dunder


_install_operator_dunders()
