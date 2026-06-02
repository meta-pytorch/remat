# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Debug reporting helpers for torch_remat checkpoint regions."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TextIO

import torch
from torch_remat._api import (
    _CheckpointRegionState,
    _display_name,
    _expect_state,
    _OpRecord,
    _REPORT_OUTPUT_NAME_ATTR,
    _TensorMetadata,
    CheckpointPolicy,
)


@dataclass
class _StorageSummary:
    """Tensor rows that share one storage in a report section."""

    # Representative metadata for the shared storage.
    metadata: _TensorMetadata

    # Backing storage bytes.
    nbytes: int

    # Report row names that reference the same storage.
    names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ReportRow:
    """One tensor row in the memory report."""

    metadata: _TensorMetadata
    tensor: torch.Tensor | None = None
    policy: CheckpointPolicy | None = None
    alias_names: tuple[str, ...] = ()
    is_native: bool = False


def format_current_memory_report() -> str:
    """Return a memory report for the currently active checkpoint region.

    Example:
        ```python
        if not remat.is_recomputing():
            report = remat.format_current_memory_report()
        ```
    """

    state = _expect_state()
    return _format_memory_report(state.region_state)


def print_current_memory_report(file: TextIO | None = None) -> None:
    """Print a memory report for the currently active checkpoint region.

    Example:
        ```python
        if not remat.is_recomputing():
            remat.print_current_memory_report()
        ```
    """

    output_file = sys.stdout if file is None else file
    output_file.write(format_current_memory_report())
    output_file.write("\n")


def _format_memory_report(region_state: _CheckpointRegionState) -> str:
    """Format a hierarchical memory report for one checkpoint region."""

    region = (
        region_state.region_name
        if region_state.region_name is not None
        else "<unnamed>"
    )
    lines = [f"torch_remat checkpoint region: {region}"]
    total_bytes = _unique_storage_nbytes(_iter_region_retained_rows(region_state))
    lines.append(f"total: {_format_bytes(total_bytes)}")

    for record in region_state.records.values():
        saved_rows = _saved_rows_by_name(record)
        if not saved_rows:
            continue
        lines.append(
            f"{_display_name(region_state, record.op_name)} "
            f"total={_format_bytes(_record_nbytes(record))}"
        )
        lines.extend(_format_metadata_lines(saved_rows))
        lines.extend(_format_storage_lines(saved_rows))

    return "\n".join(lines)


def _record_nbytes(
    record: _OpRecord,
) -> int:
    """Return total retained bytes for an op record."""

    return _unique_storage_nbytes(_saved_rows_by_name(record).values())


def _unique_storage_nbytes(
    rows: Iterable[_ReportRow],
) -> int:
    """Return unique storage bytes for a set of tensor metadata."""

    seen: set[tuple[torch.device, int]] = set()
    total = 0
    for row in rows:
        storage_key = _storage_identity(row.tensor)
        if storage_key is None:
            total += _logical_nbytes(row.metadata)
            continue
        if storage_key in seen:
            continue
        seen.add(storage_key)
        total += row.tensor.untyped_storage().nbytes()
    return total


def _iter_region_retained_rows(
    region_state: _CheckpointRegionState,
) -> Iterable[_ReportRow]:
    """Yield metadata that corresponds to retained region storage."""

    for record in region_state.records.values():
        yield from _saved_rows_by_name(record).values()


def _saved_rows_by_name(record: _OpRecord) -> dict[str, _ReportRow]:
    """Return live saved tensors keyed by report names."""

    rows: dict[str, _ReportRow] = {}
    for slot_name, slot in record.tensor_slots.items():
        if not isinstance(slot.tensor, torch.Tensor):
            continue
        rows[slot_name] = _ReportRow(
            _metadata_from_live_tensor(slot.tensor),
            slot.tensor,
            record.policy,
            _preferred_alias_names(slot.tensor, name=slot_name),
        )

    for tensor_name, tensor_ref in record.native_sac_tensors.items():
        tensor = tensor_ref()
        if tensor is None:
            continue
        rows[tensor_name] = _ReportRow(
            _metadata_from_live_tensor(tensor),
            tensor,
            None,
            _preferred_alias_names(tensor, name=tensor_name),
            is_native=True,
        )

    return rows


def _format_storage_lines(rows: Mapping[str, _ReportRow]) -> list[str]:
    """Format storage rows only when they add information beyond tensor rows."""

    summaries: dict[tuple[torch.device, int], _StorageSummary] = {}
    _add_storage_summaries(summaries, rows)

    lines: list[str] = []
    storage_index = 0
    for summary in summaries.values():
        if (
            len(summary.names) == 1
            and _logical_nbytes(summary.metadata) == summary.nbytes
        ):
            continue

        lines.append(
            "  "
            f"storage.s{storage_index}: {_format_bytes(summary.nbytes)} "
            f"shared_by={','.join(summary.names)}"
        )
        storage_index += 1

    return lines


def _add_storage_summaries(
    summaries: dict[tuple[torch.device, int], _StorageSummary],
    rows_by_name: Mapping[str, _ReportRow],
) -> None:
    """Add tensor metadata to storage sharing summaries."""

    for tensor_name, row in rows_by_name.items():
        storage_key = _storage_identity(row.tensor)
        if storage_key is None or row.tensor is None:
            continue
        summary = summaries.setdefault(
            storage_key,
            _StorageSummary(
                metadata=row.metadata,
                nbytes=row.tensor.untyped_storage().nbytes(),
            ),
        )
        summary.names.extend(row.alias_names or (tensor_name,))


def _format_metadata_lines(
    rows_by_name: Mapping[str, _ReportRow],
) -> list[str]:
    """Format per-tensor memory report rows."""

    lines: list[str] = []
    for tensor_name, row in rows_by_name.items():
        annotations: list[str] = []
        if row.policy is not None:
            annotations.append(f"policy={row.policy.name}")
        if row.is_native:
            annotations.append("source=native")
        if row.alias_names:
            annotations.append(f"aliases={','.join(row.alias_names)}")
        annotation_str = "" if not annotations else " " + " ".join(annotations)
        lines.append(
            "  "
            f"{tensor_name}: {_format_bytes(_logical_nbytes(row.metadata))} "
            f"shape={row.metadata.shape} dtype={row.metadata.dtype} "
            f"device={row.metadata.device}"
            f"{annotation_str}"
        )
    return lines


def _storage_identity(tensor: torch.Tensor | None) -> tuple[torch.device, int] | None:
    """Return a live tensor's storage identity for report-time grouping."""

    if tensor is None:
        return None
    try:
        data_ptr = int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return None
    if data_ptr == 0:
        return None
    return tensor.device, data_ptr


def _preferred_alias_names(
    tensor: torch.Tensor,
    *,
    name: str,
) -> tuple[str, ...]:
    """Return output-side aliases for a saved activation row."""

    output_name = getattr(tensor, _REPORT_OUTPUT_NAME_ATTR, None)
    if isinstance(output_name, str) and output_name != name:
        return (output_name,)
    return ()


def _metadata_from_live_tensor(tensor: torch.Tensor) -> _TensorMetadata:
    """Return the current shape/device data for a live report tensor."""

    return _TensorMetadata(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _logical_nbytes(metadata: _TensorMetadata) -> int:
    """Return logical bytes occupied by one tensor view."""

    numel = 1
    for size in metadata.shape:
        numel *= size
    return numel * torch._utils._element_size(metadata.dtype)  # type: ignore[attr-defined]


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
