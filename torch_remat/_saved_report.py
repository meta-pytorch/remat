# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Whole-model "everything saved for backward" report.

:func:`format_current_memory_report` (in :mod:`torch_remat._reporting`) reports a single
*active* checkpoint region. This module answers the broader question you want at the
pre-backward high-water mark -- "print everything saved for backward" -- spanning every
checkpoint region (all transformer blocks at once) *and* the model code outside any region
(embeddings, the output head, the loss). Call it after the full forward, before backward,
with the loss as the root:

    if not remat.is_recomputing():
        remat.print_saved_tensors_report(loss)

It fuses two sources and de-duplicates them **by storage**:

* **remat region tapes** -- every live region state (see
  :func:`torch_remat._region._iter_live_regions`). These carry op-level names and the
  resident / rebuilt-on-recompute / offloaded distinction the raw autograd graph cannot
  express.
* **an autograd graph walk** from the given roots. This reaches saved tensors that live on
  the ordinary graph, outside any region.

De-duplication is **by storage**, so every resident storage is billed exactly once and the
parts reconcile with the header total. Three cases:

* **region vs graph**: a SAVE op's tensors sit on the autograd graph *and* in the region
  tape. The region tapes claim their storages first and the graph walk skips any storage
  already claimed.
* **across regions**: a storage saved by more than one region (e.g. a genuinely shared
  activation, or a weight referenced by a ``save_for_backward`` in every block) is
  attributed to the *first* region that claims it, in the natural region sort order. So the
  per-region summary and per-region detail bill it once -- to that first region -- and their
  bytes sum to the header total. A region that ends up owning nothing (all its storages
  attributed to earlier regions) is shown with a note in its detail.
* **within a region**: the single-region reporter already dedupes storages shared across
  ops for its region total.

The union is exact, never double-counted.

Two deliberate scope choices:

* The graph walk skips leaf tensors that require grad (parameters and graph inputs). Those
  are legitimately "saved for backward" by ops like matmul, but they are weights/inputs,
  not activations, and counting them would drown the signal. Region detail (rendered by the
  unchanged single-region reporter) may still show a weight a SAVE op references.
* Non-reentrant checkpoint nodes are **opaque before backward**: their inner graph and
  their link to a region's inputs only exist during recompute. So the pre-backward graph
  walk reaches non-region saves *downstream of* or disconnected from the regions (the
  output head, the loss) but not those *upstream of* or *between* regions (e.g. the token
  embeddings) -- the walk dead-ends at the topmost checkpoint node. This is the price of
  the remat tape living off the autograd tape; the dominant intra-region activations are
  covered by the tapes regardless, so the blind spot is the (usually small) pre-layer and
  inter-layer non-checkpointed saves.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, TextIO

import torch
from torch_remat._region import _CheckpointRegionState, _iter_live_regions
from torch_remat._reporting import (
    _collect_storages,
    _format_bytes,
    _format_memory_report,
    _Storage,
    _storage_key,
    Annotate,
)

_StorageKey = tuple[torch.device, int]


@dataclass
class _RegionView:
    """A region plus its storages after cross-region attribution.

    ``owned`` holds only the storages this region is the first (in natural sort order) to
    claim; ``excluded`` are keys an earlier region already claimed and therefore bills. Every
    resident storage is owned by exactly one region, so summing ``owned`` bytes over all
    views equals the header total.
    """

    region: _CheckpointRegionState
    owned: "dict[_StorageKey, _Storage]"
    excluded: "frozenset[_StorageKey]"


# One group of regions that share an owned-storage signature (same op / owned-storage-size
# shape), rendered once from the group's first (representative) view.
_RegionGroup = list[_RegionView]


@dataclass
class _GraphStorage:
    """One storage found on the autograd graph, outside any region tape."""

    nbytes: int
    dtype: torch.dtype
    device: torch.device
    node_types: set[str] = field(default_factory=set)


def format_saved_tensors_report(
    roots: torch.Tensor | Iterable[torch.Tensor] | None = None,
    annotate: Annotate | None = None,
) -> str:
    """Return a whole-model report of everything saved for backward.

    Args:
        roots: The tensor(s) whose autograd graph is walked for non-region saves --
            typically the loss. A single tensor or an iterable of tensors. When ``None``
            the graph walk is skipped and only the live region tapes are reported.
        annotate: Optional per-tensor labeller (see :data:`torch_remat._reporting.Annotate`)
            appended to each region-detail storage row -- e.g. an allocation site derived
            from a memory snapshot, to say *which* tensor a ``saved.<i>`` row is.

    Returns:
        str: A multi-line report: a header total, a per-region summary (regions that share a
        structural signature collapsed into one line), the non-region graph breakdown by
        autograd node type, a not-resident footer, and per-signature representative detail.
    """

    root_list = _normalize_roots(roots)
    regions = sorted(_iter_live_regions(), key=_region_sort_key)
    # Attribute each storage to the first region (sort order) that saves it, so a storage
    # shared across regions is billed once and the summary/detail reconcile with the total.
    region_views = _attribute_regions(regions)
    groups = _group_regions(region_views)

    # Owned maps are disjoint by construction, so their union is every region storage counted
    # once -- the header total, and the claimed set the graph walk de-dupes against.
    claimed: set[_StorageKey] = set()
    region_resident = 0
    for view in region_views:
        for key, storage in view.owned.items():
            claimed.add(key)
            region_resident += storage.nbytes

    walked = bool(root_list)
    graph_storages = _walk_graph(root_list, claimed) if walked else {}
    graph_resident = sum(gs.nbytes for gs in graph_storages.values())
    grand_total = region_resident + graph_resident

    # With no roots the graph is not walked, so say "not walked" rather than "outside regions
    # 0 B" -- a zero there reads as "there are none", when the non-region saves are merely
    # uncounted (the section below says how to count them).
    outside = (
        f"outside regions {_format_bytes(graph_resident)}"
        if walked
        else "outside regions not walked"
    )
    lines: list[str] = [
        f"saved for backward: {_format_bytes(grand_total)} resident -- "
        f"{len(regions)} region(s) {_format_bytes(region_resident)}, "
        f"{outside}"
    ]

    lines.extend(_format_region_summary(groups, region_views))
    lines.extend(_format_graph_section(graph_storages, graph_resident, walked))
    lines.extend(_format_region_detail(groups, annotate))
    return "\n".join(lines)


def print_saved_tensors_report(
    roots: torch.Tensor | Iterable[torch.Tensor] | None = None,
    file: TextIO | None = None,
    annotate: Annotate | None = None,
) -> None:
    """Print :func:`format_saved_tensors_report` (plus a trailing newline).

    Args:
        roots: See :func:`format_saved_tensors_report`.
        file: Destination stream; defaults to ``sys.stdout``.
        annotate: See :func:`format_saved_tensors_report`.
    """

    output_file = sys.stdout if file is None else file
    output_file.write(format_saved_tensors_report(roots, annotate))
    output_file.write("\n")


def _region_sort_key(region: _CheckpointRegionState) -> tuple[str, int]:
    """Natural-sort key so ``layer.2`` precedes ``layer.10`` (not string order)."""

    name = region.region_name or ""
    match = re.match(r"^(.*?)(\d+)$", name)
    return (match.group(1), int(match.group(2))) if match else (name, -1)


def _attribute_regions(
    regions: list[_CheckpointRegionState],
) -> list[_RegionView]:
    """Attribute each storage to the first region (sort order) that saves it -- policy #1.

    A storage saved by more than one region is owned by the earliest region and excluded from
    every later one, so the per-region summary and detail bill it once and reconcile with the
    header total. Non-aliasing regions own all their storages, exactly as before.
    """

    claimed: set[_StorageKey] = set()
    views: list[_RegionView] = []
    for region in regions:
        full_map = _region_storage_map(region)
        excluded = frozenset(key for key in full_map if key in claimed)
        owned = {key: s for key, s in full_map.items() if key not in claimed}
        claimed.update(full_map)
        views.append(_RegionView(region=region, owned=owned, excluded=excluded))
    return views


def _group_regions(region_views: list[_RegionView]) -> list[_RegionGroup]:
    """Bucket regions by owned-storage signature, preserving first-seen order.

    Identical transformer blocks collapse into one group; a heterogeneous stack (e.g.
    sliding-window vs global attention in a 3:1 pattern) yields one group per distinct
    shape, so the report shows the pattern instead of N near-duplicate dumps. Because the
    signature is over *owned* storages, two otherwise-identical regions split into different
    groups when one of them shares a storage billed to an earlier region -- keeping the
    per-group "xN ... each" figure honest (every member of a group owns the same storages).
    """

    groups: dict[tuple[object, ...], _RegionGroup] = {}
    order: list[tuple[object, ...]] = []
    for view in region_views:
        signature = _region_owned_signature(view.region, view.excluded)
        if signature not in groups:
            groups[signature] = []
            order.append(signature)
        groups[signature].append(view)
    return [groups[signature] for signature in order]


def _compact_region_names(names: list[str]) -> str:
    """Compact region names sharing a ``prefix<int>`` shape into ranges.

    ``layer.0..layer.7`` -> ``layer.0-7``; a split set -> ``layer.0-2, layer.4-6``. Names
    that don't fit the pattern are listed verbatim.
    """

    parsed: list[tuple[str, int]] = []
    for name in names:
        match = re.match(r"^(.*?)(\d+)$", name or "")
        if not match:
            return ", ".join(name or "<unnamed>" for name in names)
        parsed.append((match.group(1), int(match.group(2))))
    if len({prefix for prefix, _ in parsed}) != 1:
        return ", ".join(name or "<unnamed>" for name in names)

    prefix = parsed[0][0]
    nums = sorted(num for _, num in parsed)
    ranges: list[tuple[int, int]] = []
    start = prev = nums[0]
    for num in nums[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append((start, prev))
            start = prev = num
    ranges.append((start, prev))
    return ", ".join(
        f"{prefix}{lo}" if lo == hi else f"{prefix}{lo}-{hi}" for lo, hi in ranges
    )


def _normalize_roots(
    roots: torch.Tensor | Iterable[torch.Tensor] | None,
) -> list[torch.Tensor]:
    if roots is None:
        return []
    if isinstance(roots, torch.Tensor):
        return [roots]
    return [t for t in roots if isinstance(t, torch.Tensor)]


def _region_storage_map(
    region_state: _CheckpointRegionState,
) -> dict[_StorageKey, _Storage]:
    """Merge a region's per-record resident storages, deduped by storage key."""

    merged: dict[_StorageKey, _Storage] = {}
    for record in region_state.records.values():
        for key, storage in _collect_storages(record).items():
            merged.setdefault(key, storage)
    return merged


def _is_activation_save(tensor: torch.Tensor) -> bool:
    """A saved tensor worth reporting: has bytes and is not a parameter/graph input.

    Leaf tensors that require grad are parameters or user inputs -- saved by ops like
    matmul, but not activations -- so they are excluded to keep the signal readable.
    """

    return tensor.numel() > 0 and not (tensor.is_leaf and tensor.requires_grad)


def _node_saved_tensors(node: torch.autograd.graph.Node) -> list[torch.Tensor]:
    """Return a node's saved-for-backward activation tensors, best-effort.

    Reads the *raw* saves via the ``_raw_saved_*`` accessors, which expose each save as a
    ``SavedTensor`` (its ``.data`` is the packed value) *without unpacking it*. This is
    deliberate: unpacking (``node.saved_tensors`` / ``node._saved_<name>``) fires the save's
    unpack hook, and for a remat- or checkpoint-saved tensor that hook triggers a recompute
    mid-report -- corrupting the real backward. A save that carries an unpack hook is exactly
    such a hook-saved tensor (and is billed by the region tapes anyway), so it is skipped;
    only a plain save, whose ``.data`` is the resident tensor, is reported here.

    ``_raw_saved_tensors`` is the whole tuple for a custom ``autograd.Function`` node;
    built-in nodes instead expose one ``_raw_saved_<name>`` accessor per save.
    """

    raw_saves: list[object] = list(getattr(node, "_raw_saved_tensors", ()) or ())
    for attr in dir(node):
        if attr.startswith("_raw_saved_") and attr != "_raw_saved_tensors":
            saved = getattr(node, attr, None)
            if saved is not None:
                raw_saves.append(saved)

    tensors: list[torch.Tensor] = []
    for saved in raw_saves:
        if getattr(saved, "unpack_hook", None) is not None:
            continue  # hook-saved (remat / checkpoint) -- covered by the region tapes
        data = getattr(saved, "data", None)
        if isinstance(data, torch.Tensor):
            tensors.append(data)
    return [t for t in tensors if _is_activation_save(t)]


def _walk_graph(
    roots: list[torch.Tensor], claimed: set[_StorageKey]
) -> dict[_StorageKey, _GraphStorage]:
    """Walk the autograd graph from ``roots``, collecting resident saved storages.

    Storages already claimed by a region tape are skipped, so the result is exactly the
    non-region remainder.
    """

    result: dict[_StorageKey, _GraphStorage] = {}
    # Hold the visited Node objects, NOT their id(): an autograd Node's Python wrapper is
    # ephemeral (recreated per access) while the underlying node is cached, so once a wrapper
    # is dropped its id() is reused by a different node -- an id-keyed set would then mark a
    # fresh node "seen" and skip its whole subgraph (multi-root walks hit this immediately).
    # Keeping the objects pins each node's stable cached wrapper, so identity dedup is correct.
    seen: set[torch.autograd.graph.Node] = set()
    todo: list[torch.autograd.graph.Node] = [
        t.grad_fn for t in roots if t.grad_fn is not None
    ]
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        for tensor in _node_saved_tensors(node):
            key = _storage_key(tensor)
            if key is None or key in claimed:
                continue
            entry = result.get(key)
            if entry is None:
                entry = _GraphStorage(
                    nbytes=tensor.untyped_storage().nbytes(),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                result[key] = entry
            entry.node_types.add(type(node).__name__)
        todo.extend(nf[0] for nf in node.next_functions)
    return result


def _format_region_summary(
    groups: list[_RegionGroup],
    region_views: list[_RegionView],
) -> list[str]:
    if not groups:
        return []

    # Every member of a group owns the same storages (grouped by owned signature), so the
    # per-member ``owned`` bytes stand for the whole group and ``bytes x count`` reconciles.
    def group_bytes(group: _RegionGroup) -> int:
        return sum(s.nbytes for s in group[0].owned.values())

    byte_w = max(len(_format_bytes(group_bytes(group))) for group in groups)
    lines = ["", "regions:"]
    for group in groups:
        rep_owned = group[0].owned
        count = len(group)
        names = _compact_region_names(
            [v.region.region_name or "<unnamed>" for v in group]
        )
        noun = "storage" if len(rep_owned) == 1 else "storages"
        count_col = f"x{count}  " if count > 1 else ""
        each = " each" if count > 1 else ""
        lines.append(
            f"  {_format_bytes(group_bytes(group)):>{byte_w}}  "
            f"{count_col}{names}  ({len(rep_owned)} {noun}{each})"
        )
    lines.extend(_aggregate_footer([v.region for v in region_views]))
    return lines


def _aggregate_footer(regions: list[_CheckpointRegionState]) -> list[str]:
    """Non-resident totals across all regions: rebuilt-on-recompute and offloaded."""

    rebuilt = 0
    offloaded = 0
    for region in regions:
        for record in region.records.values():
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


def _format_graph_section(
    graph_storages: dict[_StorageKey, _GraphStorage],
    graph_resident: int,
    walked: bool,
) -> list[str]:
    if not walked:
        return [
            "",
            "outside regions: not walked (pass roots=<loss>, or "
            "roots=remat.discover_autograd_roots() for everything)",
        ]

    noun = "storage" if len(graph_storages) == 1 else "storages"
    lines = [
        "",
        f"outside regions: {_format_bytes(graph_resident)} in "
        f"{len(graph_storages)} {noun}",
    ]
    by_label: dict[str, list[_GraphStorage]] = defaultdict(list)
    for gs in graph_storages.values():
        by_label[" / ".join(sorted(gs.node_types))].append(gs)
    for label, group in sorted(
        by_label.items(), key=lambda kv: -sum(g.nbytes for g in kv[1])
    ):
        total = sum(g.nbytes for g in group)
        lines.append(f"  {_format_bytes(total):>9}  {label} (x{len(group)})")
    return lines


def _region_owned_signature(
    region_state: _CheckpointRegionState,
    excluded: frozenset[_StorageKey],
) -> tuple[object, ...]:
    """A name-independent signature over the region's *owned* storages so identical blocks
    group together, while a region that shares a storage billed to an earlier region (fewer
    owned storages) splits off into its own group. Storages in ``excluded`` are billed to an
    earlier region; storages already seen within this region are deduped, matching the total.
    """

    parts: list[object] = []
    seen: set[_StorageKey] = set()
    for op_name, record in region_state.records.items():
        sizes: list[int] = []
        for key, storage in _collect_storages(record).items():
            if key in excluded or key in seen:
                continue
            seen.add(key)
            sizes.append(storage.nbytes)
        parts.append((op_name, tuple(sorted(sizes))))
    return tuple(parts)


def _format_region_detail(
    groups: list[_RegionGroup],
    annotate: Annotate | None,
) -> list[str]:
    if not groups:
        return []

    lines = ["", "region detail:"]
    for group in groups:
        rep = group[0]
        if len(group) > 1:
            names = _compact_region_names(
                [v.region.region_name or "<unnamed>" for v in group]
            )
            lines.append(f"[x{len(group)}: {names}]")
        lines.append(
            _format_memory_report(rep.region, annotate, exclude_keys=rep.excluded)
        )
    return lines
