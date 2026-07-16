# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Debug reporting helpers for torch_remat checkpoint regions.

The memory report is a *view over* whatever ``region_state`` already holds -- it never
asks torch_remat to record anything extra for its benefit. Its job is to answer "why is
this byte resident, and how much am I keeping?", so it is organised around the three
levels that actually determine that:

* **storage** -- the only thing that owns bytes; each appears exactly once, and it is the
  only kind of row that carries a byte figure. The region total is the literal sum of
  those figures.
* **value** -- a ``(shape, stride, offset)`` over a storage. Several *exact aliases* of one
  value (same value under several names) fold into a single row; a *view* (shares the
  storage, different extent) hangs underneath as a child with no byte figure of its own.
* **name/role** -- an annotation on a value (the ``save_for_backward`` name, the durable
  ``output.<i>`` slot). Names never carry bytes.

Everything that is *not* resident -- saved inputs rebuilt on recompute, offloaded outputs --
is reported as a non-additive footer line so the byte column stays a clean bijection with
storages. A note on lifetime: SAVE saves are held weakly by autograd through the region
output's grad_fn, so a report taken after that graph is released (or after backward) shows
them as already gone.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, NamedTuple, TextIO

# An optional per-tensor annotator: given a saved tensor, return a short label (e.g. its
# allocation site) to append to that storage's row, or None. Lets a caller enrich the report
# from a memory snapshot without torch_remat knowing about snapshots.
Annotate = Callable[["torch.Tensor"], "str | None"]

import torch
from torch_remat._api import (
    _output_slot_name,
    _SaveRecord,
)
from torch_remat._region import (
    _CheckpointRegionState,
    _display_name,
    _expect_state,
)

# A view whose addressed bytes fan out into more runs than this is summarised by its
# [min, max] byte extent instead of an exact interval union. The extent is a conservative
# superset, so it can only *under*-report waste -- never cry wolf -- and it keeps the
# report O(1) on pathologically strided tensors rather than O(numel).
_INTERVAL_FANOUT_CAP: int = 4096


@dataclass
class _Value:
    """One distinct ``(shape, stride, offset)`` over a storage, with the names it wears.

    ``covered`` is the number of storage bytes this value actually addresses (its logical
    size for any non-overlapping view). A value that covers the whole storage is the
    storage's *owner* and supplies the storage row's name; the rest are views rendered as
    children.
    """

    tensor: torch.Tensor
    names: list[str] = field(default_factory=list)
    covered: int = 0


@dataclass
class _Storage:
    """One backing storage and every resident value that references it."""

    nbytes: int
    dtype: torch.dtype
    device: torch.device
    values: list[_Value] = field(default_factory=list)


class _ViewChild(NamedTuple):
    """A view pinned under a storage, rendered as a child row beneath its base."""

    name: str
    shape: str
    spans: str


@dataclass
class _StorageRow:
    """One storage's rendered fields, pre-layout, so a caller can align columns.

    Split from rendering so the report can compute per-region column widths in one pass
    and emit vertically-aligned rows in a second. ``device`` is intentionally absent from
    the rendered columns -- one training process owns one device, so it is noise.
    """

    nbytes: int
    byte_str: str
    name: str
    shape: str
    dtype: str
    flag: str
    annot: str
    # A "base of <views>" row whose name is a list of every pinning view -- often long. It is
    # excluded from the name-column width so one outlier doesn't pad every normal row; it just
    # overflows its own column.
    wide_name: bool = False
    # View children pinned under this storage.
    children: list[_ViewChild] = field(default_factory=list)


def format_current_memory_report(annotate: Annotate | None = None) -> str:
    """Return a memory report for the currently active checkpoint region.

    Must be called from inside a checkpoint region (typically within the region
    ``function``, guarded by ``not remat.is_recomputing()`` so it runs on the
    original forward only).

    Args:
        annotate: Optional per-tensor labeller (see :data:`Annotate`); its result is
            appended to each storage's row. Default: ``None``.

    Returns:
        str: The multi-line report, tallying the tensors each SAVE op keeps
        resident and how many bytes they cost.

    Raises:
        RuntimeError: If no checkpoint region is currently active.

    Example:
        ```python
        if not remat.is_recomputing():
            report = remat.format_current_memory_report()
        ```
    """

    state = _expect_state()
    return _format_memory_report(state.region_state, annotate)


def print_current_memory_report(
    file: TextIO | None = None, annotate: Annotate | None = None
) -> None:
    """Print a memory report for the currently active checkpoint region.

    Must be called from inside a checkpoint region (see
    :func:`format_current_memory_report`, whose output this writes).

    Args:
        file (TextIO, optional): Destination stream; the report and a trailing
            newline are written to it. When ``None``, writes to ``sys.stdout``.
            Default: ``None``.
        annotate: Optional per-tensor labeller (see :data:`Annotate`).

    Raises:
        RuntimeError: If no checkpoint region is currently active.

    Example:
        ```python
        if not remat.is_recomputing():
            remat.print_current_memory_report()
        ```
    """

    output_file = sys.stdout if file is None else file
    output_file.write(format_current_memory_report(annotate))
    output_file.write("\n")


def _format_memory_report(
    region_state: _CheckpointRegionState,
    annotate: Annotate | None = None,
    exclude_keys: frozenset[tuple[torch.device, int]] = frozenset(),
) -> str:
    """Format a storage-oriented memory report for one checkpoint region.

    Columns are aligned vertically: rows are built once as :class:`_StorageRow`, the
    per-region column widths are measured, then rows are rendered padded to those widths.

    ``exclude_keys`` are storages already attributed to an earlier region by the whole-model
    report (:func:`torch_remat.format_saved_tensors_report`); they are dropped from this
    region's rows, subtotals, and total so a storage shared across regions is billed exactly
    once. Empty (the default) for a standalone single-region report.
    """

    region = (
        region_state.region_name
        if region_state.region_name is not None
        else "<unnamed>"
    )

    # Collect every op's rows up front so the header total dedupes storages shared across
    # ops (counted once), each op subtotal sums the storages it references, and column
    # widths can be measured across the whole region before rendering. Storages in
    # ``exclude_keys`` are billed to an earlier region and skipped here entirely.
    op_sections: list[tuple[str, list[_StorageRow]]] = []
    seen_storage: set[tuple[torch.device, int]] = set()
    excluded_any = False
    total_bytes = 0
    for record in region_state.records.values():
        rows: list[_StorageRow] = []
        for key, storage in _collect_storages(record).items():
            if key in exclude_keys:
                excluded_any = True
                continue
            rows.append(_storage_row(storage, annotate))
            if key not in seen_storage:
                seen_storage.add(key)
                total_bytes += storage.nbytes
        if not rows:
            continue
        op_sections.append((_display_name(region_state, record.op_name), rows))

    widths = _column_widths(row for _, rows in op_sections for row in rows)

    lines = [
        f"{region}: {_format_bytes(total_bytes)} resident in "
        f"{len(seen_storage)} storage(s)"
    ]
    for op_name, rows in op_sections:
        op_bytes = sum(row.nbytes for row in rows)
        lines.append(f"{op_name}: {_format_bytes(op_bytes)}")
        for row in rows:
            lines.extend(_render_storage_row(row, widths))

    footer = _format_not_resident(region_state)
    lines.extend(footer)

    # SAVE saves vanish from the weak index once the region output's grad graph is
    # released, so a completed SAVE op (``output_schema`` set) with nothing resident
    # and nothing deferred means the graph is gone -- flag it rather than let a bare
    # 0 B read as "nothing was retained".
    completed = any(
        record.output_schema is not None for record in region_state.records.values()
    )
    if completed and total_bytes == 0 and not footer:
        if excluded_any:
            lines.append(
                "  (resident storages all billed to an earlier region -- counted there)"
            )
        else:
            lines.append(
                "  ! region output no longer alive -- saved tensors already released; "
                "report reflects that"
            )
    return "\n".join(lines)


def _collect_storages(record: _SaveRecord) -> dict[tuple[torch.device, int], _Storage]:
    """Group a SAVE op's resident tensors by storage, folding exact aliases into one value.

    Sources, all resident and real: the weak ``saved_tensor_names`` index (autograd-owned
    saves, live ones only) and the non-offloaded durable ``output_slots``. Tensors sharing
    a storage land in the same :class:`_Storage`; within it, tensors sharing an exact
    ``(shape, stride, offset)`` fold into one :class:`_Value` wearing all their names.
    """

    storages: dict[tuple[torch.device, int], _Storage] = {}
    # (name, tensor) in a stable order: save names first (pack order), then durable slots.
    entries: list[tuple[str, torch.Tensor]] = [
        (name, tensor) for tensor, name in record.saved_tensor_names.items()
    ]
    for index, slot in record.output_slots.items():
        if isinstance(slot.tensor, torch.Tensor):
            entries.append((_output_slot_name(index), slot.tensor))

    for name, tensor in entries:
        key = _storage_key(tensor)
        if key is None:
            continue
        storage = storages.get(key)
        if storage is None:
            storage = _Storage(
                nbytes=tensor.untyped_storage().nbytes(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            storages[key] = storage
        value = _find_value(storage, tensor)
        if value is None:
            value = _Value(
                tensor=tensor, covered=_covered_bytes(_addressed_intervals(tensor))
            )
            storage.values.append(value)
        value.names.append(name)

    return storages


def _find_value(storage: _Storage, tensor: torch.Tensor) -> _Value | None:
    """Return the existing value for ``tensor``'s exact extent, or None to make a new one."""

    for value in storage.values:
        existing = value.tensor
        if (
            existing.shape == tensor.shape
            and existing.stride() == tensor.stride()
            and existing.storage_offset() == tensor.storage_offset()
        ):
            return value
    return None


def _storage_row(storage: _Storage, annotate: Annotate | None) -> _StorageRow:
    """Extract one storage's fields (unpadded) for later aligned rendering.

    The byte figure lives on this row and nowhere else. A storage always has at least one
    named value, so the row is always named after one -- there is no "unnamed" case:

    * a value that spans the whole storage (the *owner*) names the row; other values, if
      any, hang beneath it as view children;
    * a lone value that doesn't span the whole storage still names the row -- it is the sole
      reason the storage is resident -- and the ``! held for`` flag reports the slack;
    * a storage whose own base tensor is gone, held only by several partial views, is named
      ``base of <views>`` (the pinning pathology -- the row shouts the ratio).
    """

    # Prefer a value spanning the whole storage; else the sole value; else a base held only
    # by partial views. Every branch yields a real name -- storages are never "unnamed".
    owner: _Value | None = next(
        (value for value in storage.values if value.covered == storage.nbytes), None
    )
    if owner is not None:
        row_value: _Value | None = owner
        children = [value for value in storage.values if value is not owner]
    elif len(storage.values) == 1:
        row_value = storage.values[0]
        children = []
    else:
        row_value = None
        children = list(storage.values)
    children.sort(key=lambda value: value.tensor.storage_offset())

    dtype = str(storage.dtype).replace("torch.", "")
    used = _covered_bytes(
        [iv for value in storage.values for iv in _addressed_intervals(value.tensor)]
    )
    flag = ""
    if used < storage.nbytes:
        flag = f"  ! held for {_format_bytes(used)} of {_format_bytes(storage.nbytes)}"

    if row_value is not None:
        name = _format_value_name(row_value.names)
        shape = f"{tuple(row_value.tensor.shape)}"
        repr_tensor: torch.Tensor | None = row_value.tensor
    else:
        name = "base of " + ", ".join(_order_names(v.names)[0] for v in children)
        shape = ""
        repr_tensor = children[0].tensor if children else None

    annot = ""
    if annotate is not None and repr_tensor is not None:
        site = annotate(repr_tensor)
        if site:
            annot = f"  {site}"

    child_rows = [
        _ViewChild(
            name=_format_value_name(view.names),
            shape=f"{tuple(view.tensor.shape)}",
            spans=_format_bytes(view.covered),
        )
        for view in children
    ]
    return _StorageRow(
        nbytes=storage.nbytes,
        byte_str=_format_bytes(storage.nbytes),
        name=name,
        shape=shape,
        dtype=dtype,
        flag=flag,
        annot=annot,
        wide_name=row_value is None,
        children=child_rows,
    )


def _column_widths(rows: Iterable[_StorageRow]) -> tuple[int, int, int, int]:
    """Max width of each aligned column (bytes, name, shape, dtype) across ``rows``.

    The name width ignores ``wide_name`` rows (long ``base of ...`` callouts) so a single
    outlier doesn't pad every normal row; those rows overflow their own name column instead.
    """

    rows = list(rows)
    byte_w = max((len(row.byte_str) for row in rows), default=0)
    name_w = max((len(row.name) for row in rows if not row.wide_name), default=0)
    shape_w = max((len(row.shape) for row in rows), default=0)
    dtype_w = max((len(row.dtype) for row in rows), default=0)
    return byte_w, name_w, shape_w, dtype_w


def _render_storage_row(
    row: _StorageRow, widths: tuple[int, int, int, int]
) -> list[str]:
    """Render one storage row (and its view children) padded to the given column widths."""

    byte_w, name_w, shape_w, dtype_w = widths
    main = (
        f"  {row.byte_str:>{byte_w}}  {row.name:<{name_w}}  "
        f"{row.shape:<{shape_w}}  {row.dtype:<{dtype_w}}{row.flag}{row.annot}"
    ).rstrip()
    lines = [main]
    for child in row.children:
        lines.append(
            f"    - {child.name:<{name_w}}  view {child.shape:<{shape_w}}  "
            f"spans {child.spans}".rstrip()
        )
    return lines


def _format_not_resident(region_state: _CheckpointRegionState) -> list[str]:
    """Non-additive footer: saves rebuilt on recompute, and outputs offloaded off-device.

    Kept out of the byte column on purpose -- these bytes are not resident, so counting them
    would break the "rows sum to the total" invariant. Offloaded is distinct from rebuilt:
    an offloaded output still occupies host memory (we cannot size it here -- the durable
    slot holds no live tensor), whereas a rebuilt save costs nothing until backward.
    """

    rebuilt = 0
    offloaded = 0
    for record in region_state.records.values():
        rebuilt += len(record.saved_input_recipes)
        offloaded += sum(
            1 for slot in record.output_slots.values() if slot.tensor is None
        )

    lines: list[str] = []
    if rebuilt:
        noun = "save" if rebuilt == 1 else "saves"
        lines.append(f"  + {rebuilt} {noun} rebuilt on recompute, not resident")
    if offloaded:
        noun = "output" if offloaded == 1 else "outputs"
        lines.append(f"  + {offloaded} {noun} offloaded, not resident on device")
    return lines


def _order_names(names: list[str]) -> list[str]:
    """Order a value's folded names: programmer/save names first, durable slots last."""

    saves = [n for n in names if not n.startswith("output.")]
    slots = sorted(n for n in names if n.startswith("output."))
    return saves + slots


def _format_value_name(names: list[str]) -> str:
    """Render a value's folded names for a report row.

    Programmer save names (from ``remat.save_for_backward``) join with ``=`` to show they are
    aliases of one value. A durable ``output.<i>`` slot is a *role*, not a save name, so it is
    shown parenthesised -- ``y (output at idx 0)`` -- rather than ``y = output.0``, which reads
    like an assignment. A value with only slot names (no programmer save) renders as the bare
    role.
    """

    saves = [n for n in names if not n.startswith("output.")]
    slots = sorted(n for n in names if n.startswith("output."))
    role = ""
    if slots:
        idxs = ", ".join(n[len("output.") :] for n in slots)
        role = f"output at idx {idxs}"
    if saves and role:
        return f"{' = '.join(saves)} ({role})"
    if saves:
        return " = ".join(saves)
    return role


def _storage_key(tensor: torch.Tensor) -> tuple[torch.device, int] | None:
    """Return a live tensor's storage identity for grouping, or None if unusable."""

    try:
        data_ptr = int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return None
    if data_ptr == 0:
        return None
    return tensor.device, data_ptr


def _addressed_intervals(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """Return the byte intervals a tensor's elements address within its storage.

    Contiguous inner dimensions collapse into one run, so a plain slice yields a handful of
    intervals rather than one per element. A view that fans out past
    :data:`_INTERVAL_FANOUT_CAP` runs is summarised by its [min, max] extent -- a superset,
    so waste is only ever under-reported.
    """

    sizes = list(tensor.shape)
    strides = list(tensor.stride())
    if any(size == 0 for size in sizes):
        return []
    element_size = tensor.element_size()
    base = tensor.storage_offset()

    run = 1
    expected = 1
    outer: list[tuple[int, int]] = []
    for stride, size in sorted(zip(strides, sizes)):
        if stride == expected:
            run *= size
            expected *= size
        else:
            outer.append((stride, size))

    fanout = 1
    for _, size in outer:
        fanout *= size
    if fanout > _INTERVAL_FANOUT_CAP:
        extent = sum((size - 1) * stride for stride, size in zip(strides, sizes))
        start = base * element_size
        return [(start, start + (extent + 1) * element_size)]

    # Iterative product walk (not a recursive closure: a self-referential cell would
    # be a gc cycle on every report call). Fanout is capped above, so the offset
    # list stays small.
    offsets = [0]
    for stride, size in outer:
        offsets = [offset + i * stride for offset in offsets for i in range(size)]
    intervals: list[tuple[int, int]] = []
    for offset in offsets:
        start = (base + offset) * element_size
        intervals.append((start, start + run * element_size))
    return intervals


def _covered_bytes(intervals: list[tuple[int, int]]) -> int:
    """Return the total length of a set of byte intervals after merging overlaps."""

    total = 0
    last_end = -1
    for start, end in sorted(intervals):
        if start >= last_end:
            total += end - start
            last_end = end
        elif end > last_end:
            total += end - last_end
            last_end = end
    return total


def _format_bytes(nbytes: int) -> str:
    """Format bytes for debug reports."""

    if nbytes < 1024:
        return f"{nbytes} B"

    kib = nbytes / 1024
    if kib < 1024:
        return f"{kib:.2f} KiB"

    mib = kib / 1024
    if mib < 1024:
        return f"{mib:.2f} MiB"

    return f"{mib / 1024:.2f} GiB"
