# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""A default :data:`~torch_remat._reporting.Annotate` backed by the CUDA memory snapshot.

The saved-tensors report can tag each storage row with a caller-supplied label. By far the
most common label is *where the tensor was allocated*, which the CUDA caching allocator
already records (when memory-history recording is on). This module turns that snapshot into a
ready-made annotator -- the "which tensor is this" that a positional ``saved.<i>`` row cannot
give -- so callers don't reinvent the storage-pointer -> allocation-site mapping.

The snapshot walk is expensive, so the fetch is memoized (:func:`clear_memory_snapshot_cache`
resets it) and the built annotator indexes the snapshot once, reusing it across every
per-tensor lookup -- a whole report costs a single snapshot fetch.
"""

from __future__ import annotations

import bisect
from typing import Any, Mapping

import torch
from torch_remat._reporting import Annotate

_snapshot_cache: Mapping[str, Any] | None = None


def _memory_snapshot(*, refresh: bool = False) -> Mapping[str, Any]:
    """Return the CUDA allocator memory snapshot, memoized after the first fetch.

    ``torch.cuda.memory._snapshot()`` walks the whole allocator and is expensive, so the
    result is cached. Pass ``refresh=True`` (or call :func:`clear_memory_snapshot_cache`) to
    re-fetch after more allocation. The snapshot carries allocation backtraces only while
    memory-history recording is enabled.
    """
    global _snapshot_cache
    if refresh or _snapshot_cache is None:
        _snapshot_cache = torch.cuda.memory._snapshot()
    return _snapshot_cache


def clear_memory_snapshot_cache() -> None:
    """Drop the memoized snapshot so the next annotator re-fetches (see :func:`_memory_snapshot`)."""
    global _snapshot_cache
    _snapshot_cache = None


def _alloc_site_from_frames(frames: list[dict[str, Any]] | None) -> str | None:
    """Pick a short ``file:line name`` label from an allocation backtrace (innermost first).

    Prefer the innermost frame outside library internals (torch and torch_remat), which is
    usually the model/kernel code that produced the tensor; fall back to the innermost frame
    of any kind.
    """
    for frame in frames or []:
        filename = frame.get("filename", "")
        if "/torch/" not in filename and "/torch_remat/" not in filename:
            base = filename.rsplit("/", 1)[-1]
            return f"{base}:{frame.get('line')} {frame.get('name', '')}".rstrip()
    if frames:
        frame = frames[0]
        return f"{frame.get('filename', '?').rsplit('/', 1)[-1]}:{frame.get('line')}"
    return None


def memory_snapshot_annotate(snapshot: Mapping[str, Any] | None = None) -> Annotate:
    """Return an :data:`Annotate` mapping a saved tensor's storage to its allocation site.

    Each live allocator block carries its address and allocation backtrace, so a saved
    tensor's storage pointer (which falls inside exactly one live block) resolves to the code
    that allocated it. Pass the result as ``annotate=`` to
    :func:`torch_remat.format_saved_tensors_report`.

    Args:
        snapshot: A CUDA memory snapshot (``torch.cuda.memory._snapshot()``); defaults to the
            cached :func:`_memory_snapshot`. The block index is built once here and reused by
            every per-tensor lookup, so a whole report costs a single snapshot fetch.
    """

    if snapshot is None:
        snapshot = _memory_snapshot()

    blocks: list[tuple[int, int, str]] = []  # (address, size, site), sorted by address
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            site = _alloc_site_from_frames(block.get("frames"))
            if site is not None:
                blocks.append((block["address"], block["size"], site))
    blocks.sort()
    addrs = [address for address, _, _ in blocks]

    def annotate(tensor: torch.Tensor) -> str | None:
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except Exception:
            return None
        index = bisect.bisect_right(addrs, ptr) - 1
        if 0 <= index < len(blocks):
            address, size, site = blocks[index]
            if address <= ptr < address + size:
                return site
        return None

    return annotate
