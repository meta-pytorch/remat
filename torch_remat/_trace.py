# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Reporting-only diagnostic tracing for torch_remat checkpoint regions.

:func:`collect_trace` records the :func:`torch_remat.region` annotations and
:func:`trace_scope` hierarchy seen during one original forward, for inspection and
debugging. Tracing never changes rematerialization behavior, region names, or
recompute settings, and recomputation is skipped so replayed work is not
double-counted.

This module depends only on the region/phase plumbing in :mod:`torch_remat._region`;
the core (:mod:`torch_remat._api`) imports the recording hook from here, so the
dependency runs one way (``_api`` -> ``_trace`` -> ``_region``) with no cycle.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Iterator, ParamSpec, TypeVar

from torch_remat._region import is_recomputing

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass
class OpTrace:
    """One remat region observed by diagnostic tracing."""

    name: str
    recompute: bool


@dataclass
class ScopeTrace:
    """One reporting-only diagnostic scope in a remat trace."""

    name: str
    metadata: str | None
    entries: list[OpTrace | ScopeTrace] = field(default_factory=list)


@dataclass
class RematTrace:
    """Diagnostic trace collected around one user region."""

    entries: list[OpTrace | ScopeTrace] = field(default_factory=list)
    _scope_stack: list[ScopeTrace] = field(default_factory=list)

    def format(self) -> str:
        lines = ["torch_remat trace"]
        for entry in self.entries:
            _append_trace_entry(lines, entry, indent=0)
        return "\n".join(lines)

    def _current_entries(self) -> list[OpTrace | ScopeTrace]:
        if self._scope_stack:
            return self._scope_stack[-1].entries
        return self.entries


_active_trace: contextvars.ContextVar[RematTrace | None] = contextvars.ContextVar(
    "torch_remat_trace",
    default=None,
)


@contextlib.contextmanager
def collect_trace() -> Iterator[RematTrace]:
    """Collect a reporting-only trace of remat annotations.

    Records calls to :func:`region` and :func:`trace_scope`
    during the original forward. Recomputation is skipped so reports do not
    double-count replayed work.

    Yields:
        RematTrace: The trace being collected. It is populated with the ops and
        scopes seen within the ``with`` block; call :meth:`RematTrace.format` on
        it afterward to render the hierarchy.
    """

    trace = RematTrace()
    token = _active_trace.set(trace)
    try:
        yield trace
    finally:
        _active_trace.reset(token)


def trace_scope(
    function: Callable[_P, _R],
    name: str,
    *,
    metadata: str | None = None,
) -> Callable[_P, _R]:
    """Add a diagnostic hierarchy scope to the active op trace.

    Reporting-only: it does not change rematerialization behavior, region names, or
    recompute settings.

    Args:
        function (Callable): The callable to wrap, used unmodified. Ops and
            nested scopes it reaches nest under this scope in the trace.
        name (str): Scope name shown in the trace. Must be a non-empty string.
        metadata (str, optional): Extra annotation rendered beside the scope
            name (as ``name [metadata]``). Must be a non-empty string when given.
            Keyword-only. Default: ``None``.

    Returns:
        Callable: A wrapper with the same signature as ``function``. When no
        trace is being collected (or during recompute) it just calls
        ``function``.

    Raises:
        RuntimeError: If ``function`` is not callable, or ``name`` / ``metadata``
            is not a string.
        ValueError: If ``name`` or ``metadata`` is an empty string.
    """

    if not callable(function):
        raise RuntimeError("trace_scope expects a function as its first argument")
    _validate_trace_scope_text(name, what="trace scope name")
    if metadata is not None:
        _validate_trace_scope_text(metadata, what="trace scope metadata")

    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        trace = _active_trace.get()
        if trace is None or is_recomputing():
            return function(*args, **kwargs)

        scope = ScopeTrace(name=name, metadata=metadata)
        trace._current_entries().append(scope)
        trace._scope_stack.append(scope)
        try:
            return function(*args, **kwargs)
        finally:
            popped_scope = trace._scope_stack.pop()
            if popped_scope is not scope:
                raise RuntimeError("torch_remat trace scope stack was corrupted")

    return wrapper


def _record_trace_op(
    name: str,
    *,
    recompute: bool,
) -> None:
    """Append a region to the active trace, if tracing original forward."""

    trace = _active_trace.get()
    if trace is None or is_recomputing():
        return
    trace._current_entries().append(OpTrace(name=name, recompute=recompute))


def _append_trace_entry(
    lines: list[str],
    entry: OpTrace | ScopeTrace,
    *,
    indent: int,
) -> None:
    prefix = "  " * indent
    if isinstance(entry, ScopeTrace):
        metadata = "" if entry.metadata is None else f" [{entry.metadata}]"
        lines.append(f"{prefix}{entry.name}{metadata}")
        for child in entry.entries:
            _append_trace_entry(lines, child, indent=indent + 1)
        return

    details = "recompute" if entry.recompute else "save"
    lines.append(f"{prefix}{entry.name}: {details}")


def _validate_trace_scope_text(text: str, *, what: str) -> None:
    """Validate user-facing trace scope text."""

    if not isinstance(text, str):
        raise RuntimeError(f"torch_remat {what} must be a string")
    if not text:
        raise ValueError(f"torch_remat {what} must be non-empty")
