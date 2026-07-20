# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""A deliberately tiny pytree variant for torch_remat.

``torch.utils._pytree`` is famously quite slow, and torch_remat can lean
a bit on autograd.Function conventions for what inputs it actually knows
about.  So this is a lean implementation that doesn't support TreeSpec style
flatten/unflatten and has a limited set of things it traverses into.  Formally
what we support is:

* A **value** is a leaf, or *one* hop of ``list`` / ``tuple`` whose elements are
  leaves. This is the shape of an ATen operator's arguments and results
  (``Tensor``, ``Tensor[]``, ``(Tensor, Tensor)``) and of a ``remat.region`` call's
  return. The container's own type is kept across a map-and-rebuild (see
  :func:`container_type`), so a structured one-hop return -- a ``NamedTuple`` like
  ``RouterOutput`` or a ``structseq`` like ``torch.return_types.max`` -- survives the
  round-trip with its named fields intact.
* A call's **arguments** are ``args`` (positional) plus ``kwargs`` (keyword),
  where each argument is a value. This is what a ``remat.region``-wrapped callable and
  ``__torch_dispatch__`` receive.

A **leaf** is anything else -- a ``Tensor``, but also ``None``, an ``int``, a
``dict``, or a container nested one hop deeper, which is handed to callers whole
as an opaque leaf. Leaves are typed ``object`` so callers must narrow (usually
``isinstance(leaf, torch.Tensor)``) before use; a structure remat does not
understand is thus skipped rather than rejected.

Unlike ``tree_flatten``/``tree_unflatten``, there is no persisted "treespec":
remat re-receives the live ``args``/``kwargs`` on recompute, so structure is only
ever walked, or mapped-and-rebuilt, in place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TypeAlias

# A path to one leaf within a call's arguments. The head identifies the argument
# (``int`` -> ``args[i]``, ``str`` -> ``kwargs["k"]``); any tail entry indexes one
# hop into a container argument (``int`` -> sequence index).
# It doubles as a stable, hashable identity for a leaf position across a forward
# and its recompute, and renders to a human path via :func:`path_str`.
PathToken: TypeAlias = tuple[int | str, ...]


def value_leaves(value: object) -> tuple[object, ...]:
    """Return the leaves of one value, in order; a bare leaf yields ``(value,)``."""

    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def container_type(value: object) -> type | None:
    """Return the type to rebuild one-hop ``value`` with, or ``None`` for a leaf.

    The value's own type is kept, so a one-hop container round-trips a map-and-rebuild
    as itself: a plain ``list`` / ``tuple`` unchanged, and a structured container -- a
    ``NamedTuple`` (e.g. ``RouterOutput``) or a ``structseq`` (``torch.return_types.*``)
    -- with its named fields intact. :func:`rebuild_container` knows how to reconstruct
    each; a subclass is preserved only if it is likewise constructible from one iterable.
    """

    if isinstance(value, (list, tuple)):
        return type(value)
    return None


def rebuild_container(container: type, items: list[object]) -> object:
    """Build a one-hop ``container`` from already-mapped ``items``.

    ``container`` is what :func:`container_type` returned for the original value. A
    ``NamedTuple`` is built via ``_make`` -- the documented namedtuple constructor
    (underscore-prefixed only to avoid clashing with user field names) that takes the
    fields as one iterable rather than positionally. Every other one-hop container --
    ``list``, ``tuple``, a ``structseq``, or a subclass of these -- is built by calling
    its type with the iterable directly.
    """

    make = getattr(container, "_make", None)
    if make is not None:
        return make(items)
    return container(items)


def map_value(fn: Callable[[object], object], value: object) -> object:
    """Map ``fn`` over the leaves of one value, rebuilding the same container.

    ``fn`` is applied to every leaf (tensor or not). The container is rebuilt per
    :func:`container_type`, keeping the value's own type (``list``, ``tuple``,
    ``NamedTuple``, ``structseq``, ...), matching how op output schemas are recorded.
    """

    container = container_type(value)
    if container is None:
        return fn(value)
    return rebuild_container(container, [fn(leaf) for leaf in value_leaves(value)])


def iter_arg_leaves(
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> Iterator[tuple[PathToken, object]]:
    """Yield ``(path, leaf)`` for every leaf of a call's arguments, in order.

    Positional args come first (``(i, ...)``), then keyword args (``(name, ...)``).
    """

    for index, arg in enumerate(args):
        yield from _iter_arg(arg, (index,))
    for key, value in kwargs.items():
        yield from _iter_arg(value, (key,))


def map_arg_leaves(
    fn: Callable[[PathToken, object], object],
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Map ``fn`` over a call's argument leaves, rebuilding ``args`` and ``kwargs``.

    ``fn`` receives ``(path, leaf)`` and returns the replacement leaf. Each container
    is rebuilt with its own type preserved (see :func:`container_type`), matching
    :func:`map_value`.
    """

    new_args = tuple(_map_arg(fn, arg, (index,)) for index, arg in enumerate(args))
    new_kwargs = {key: _map_arg(fn, value, (key,)) for key, value in kwargs.items()}
    return new_args, new_kwargs


def path_str(token: PathToken) -> str:
    """Render a :data:`PathToken` as ``args[0]`` / ``args[1][0]`` / ``kwargs["k"]``.

    Cold path only (memory reports, slot error messages); never call it on the
    per-op hot path -- pass the token around and render once, at display time.
    """

    head, *rest = token
    parts = [f"args[{head}]" if isinstance(head, int) else f'kwargs["{head}"]']
    for entry in rest:
        parts.append(f"[{entry}]")
    return "".join(parts)


def _iter_arg(
    value: object,
    prefix: PathToken,
) -> Iterator[tuple[PathToken, object]]:
    """Yield ``(path, leaf)`` for one argument, prefixing paths with ``prefix``."""

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield (*prefix, index), item
    else:
        yield prefix, value


def _map_arg(
    fn: Callable[[PathToken, object], object],
    value: object,
    prefix: PathToken,
) -> object:
    """Map ``fn`` over one argument's leaves, rebuilding the container."""

    container = container_type(value)
    if container is None:
        return fn(prefix, value)
    items = [
        fn((*prefix, index), leaf) for index, leaf in enumerate(value_leaves(value))
    ]
    return rebuild_container(container, items)
