# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``remat.format_saved_tensors_report``: the whole-model, all-regions
saved-for-backward report. Covers the regions-only path (no graph walk), the fused
region + autograd-graph walk with storage de-duplication, identical-region grouping, and
deterministic (gc-independent) eviction of a region from the live registry once backward
consumes it."""

from __future__ import annotations

import gc
import re
import types
from typing import Any

import expecttest
import torch
import torch_remat as remat
from torch_remat._region import (
    _CheckpointRegionState,
    _iter_live_regions,
    _live_regions,
)


def _cell_contents(cell: types.CellType) -> list[object]:
    try:
        return [cell.cell_contents]
    except ValueError:  # empty cell
        return []


def _data_referents(obj: object) -> list[object]:
    """References that could close a cycle *within remat's data*, and only those.

    Follows containers, object ``__dict__``, and closure cells (which is how a stray
    ``region_state``-capturing closure stored on the tape would show up). Deliberately does
    NOT follow a function's ``__globals__`` (that escapes into whole-module namespaces) or a
    tensor's grad graph (native autograd, not remat data) -- so the walk stays bounded and
    precise.
    """

    if isinstance(obj, types.FunctionType):
        return [c for cell in (obj.__closure__ or ()) for c in _cell_contents(cell)]
    if isinstance(obj, types.CellType):
        return _cell_contents(obj)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return list(obj)
    if isinstance(obj, dict):
        return list(obj.keys()) + list(obj.values())
    if isinstance(obj, torch.Tensor):
        return []
    slots = getattr(obj, "__dict__", None)
    return list(slots.values()) if isinstance(slots, dict) else []


def _reaches_self(root: object) -> bool:
    """Whether ``root`` is reachable from itself through remat's own data structures.

    Detects a *Python* reference cycle in the tape/region data. It does not -- and cannot
    -- see native autograd (C++) cycles; those are handled by deterministic deregistration,
    not by refcounting.
    """

    root_id = id(root)
    seen: set[int] = set()
    stack = _data_referents(root)
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id == root_id:
            return True
        if obj_id in seen:
            continue
        seen.add(obj_id)
        stack.extend(_data_referents(obj))
    return False


class CycleDetectionApparatusTest(expecttest.TestCase):
    """Tests for the cycle detector itself (``_reaches_self`` / ``_data_referents``).

    ``SavedTensorsReportTest.test_region_tape_has_no_python_reference_cycle`` asserts the
    detector finds *no* cycle in real region data. That guarantee is only meaningful if the
    detector can actually find a cycle when one exists -- a detector hard-wired to return
    ``False`` would pass that test while catching nothing. These tests build the kinds of
    cycles the detector is meant to catch (constructed directly from plain Python objects,
    with no remat internals) and confirm it ignores the references it deliberately excludes
    (tensor autograd graphs, function ``__globals__``) so the walk stays bounded.
    """

    def test_detects_closure_on_tape_capturing_state(self) -> None:
        # The exact shape the real invariant guards against: a persist-thunk closure stored
        # on a region's tape that captures the region state, closing the loop
        # state -> tape(list) -> thunk(function) -> closure cell -> state.
        class RegionState:
            def __init__(self) -> None:
                self.tape: list[object] = []

        state = RegionState()

        def persist_thunk() -> object:
            return state  # captures `state` in a closure cell

        state.tape.append(persist_thunk)
        self.assertTrue(_reaches_self(state))

    def test_detects_mutual_reference_through_object_dict(self) -> None:
        # Two objects referencing each other through their attributes (``__dict__``) -- the
        # plain-attribute analog of a region <-> tape back-pointer.
        class Node:
            def __init__(self) -> None:
                self.peer: object | None = None

        a, b = Node(), Node()
        a.peer = b
        b.peer = a
        self.assertTrue(_reaches_self(a))

    def test_detects_self_referential_container(self) -> None:
        # A tape (list) that transitively contains itself via a dict value -- exercises the
        # list and dict branches of ``_data_referents`` closing a cycle.
        tape: list[object] = []
        tape.append({"back": tape})
        self.assertTrue(_reaches_self(tape))

    def test_acyclic_dag_not_detected(self) -> None:
        # A DAG whose shared leaf is reached by two paths but never loops back: the detector
        # must not false-positive merely because a node is visited via multiple routes.
        shared = {"leaf": torch.randn(2)}
        root = {"left": {"child": shared}, "right": {"child": shared}}
        self.assertFalse(_reaches_self(root))

    def test_ignores_tensor_autograd_graph(self) -> None:
        # A tensor with a live ``grad_fn`` can reach a large native-autograd graph, but the
        # detector stops at tensors on purpose (autograd cycles are handled by deterministic
        # deregistration, not refcounting). So a tensor contributes no data referents, and a
        # cycle that closes only through a tensor's grad graph is invisible to the detector.
        x = torch.randn(3, requires_grad=True)
        y = (x * x).sum()
        self.assertIsNotNone(y.grad_fn)
        self.assertEqual(_data_referents(y), [])

    def test_ignores_function_globals(self) -> None:
        # The walk follows closure cells but never ``__globals__`` -- otherwise it would
        # escape into the whole module namespace and lose all precision. A module global
        # referenced by a function (not captured as a free variable) is not a data referent.
        def uses_global() -> object:
            return _live_regions  # module global lookup, not a closure capture

        self.assertIsNone(uses_global.__closure__)
        self.assertEqual(_data_referents(uses_global), [])


class _Sq(torch.autograd.Function):
    """A SAVE op that keeps two internal tensors resident (``y`` and ``gf``)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = x * x
        gf = x * 2
        remat.save_for_backward(ctx, {"y": y, "gf": gf})
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (y, gf) = ctx.saved_tensors
        del y
        return grad_output * gf


class _Scale(torch.autograd.Function):
    """A SAVE op that keeps a single internal tensor resident (``s``)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        s = x + x
        remat.save_for_backward(ctx, {"s": s})
        return s

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (s,) = ctx.saved_tensors
        del s
        return grad_output + grad_output


class _SaveShared(torch.autograd.Function):
    """A SAVE op saving a per-call tensor ``y`` plus a caller-provided ``shared`` tensor.

    Used to force the exact same storage into two different regions -- the cross-region
    aliasing the whole-model report must attribute to a single region, not double-bill.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        y = x * x
        remat.save_for_backward(ctx, {"y": y, "shared": shared})
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (y, shared) = ctx.saved_tensors
        del y, shared
        return grad_output, None


class _StdSave(torch.autograd.Function):
    """A SAVE op using PyTorch's standard ``ctx.save_for_backward`` -- NOT remat's named
    ``remat.save_for_backward`` -- so its resident tensor carries no programmer name and the
    report falls back to autograd's positional label (``saved.0``)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = x * x
        ctx.save_for_backward(y)  # standard autograd save: no name applied
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return grad_output * y


def _header_total_bytes(report: str) -> int:
    match = re.search(r"saved for backward: (\d+) B resident", report)
    assert match, report
    return int(match.group(1))


def _header_outside_bytes(report: str) -> int:
    if "outside regions not walked" in report:
        return 0  # walk skipped (no roots) -- no non-region bytes counted
    match = re.search(r"outside regions (\d+) B", report)
    assert match, report
    return int(match.group(1))


def _sum_summary_region_bytes(report: str) -> int:
    """Sum the per-group bytes (bytes x count) listed under the ``regions:`` summary."""
    total = 0
    in_summary = False
    for line in report.splitlines():
        if line == "regions:":
            in_summary = True
            continue
        if not in_summary:
            continue
        if not line.strip():
            break
        match = re.match(r"\s*(\d+) B\s+(?:x(\d+)\s+)?\S", line)
        if not match:  # skip non-byte footer lines (e.g. "+ N saves rebuilt ...")
            continue
        count = int(match.group(2)) if match.group(2) else 1
        total += int(match.group(1)) * count
    return total


class SavedTensorsReportTest(expecttest.TestCase):
    def setUp(self) -> None:
        # Deterministic isolation: drop any region states left registered by earlier tests
        # so the whole-model report sees exactly this test's regions -- no ``gc.collect()``
        # crutch. (A region held alive in a torch-checkpoint frame cycle would otherwise
        # linger in the registry until a cyclic gc pass.)
        _live_regions.clear()

    def test_regions_only_without_roots(self) -> None:
        # With no roots the graph walk is skipped; only the live region tapes are reported.
        # Driven through the real ``remat.checkpoint`` API: the region stays live (held by
        # the checkpoint frame) after the call returns, and its SAVE tensors stay resident
        # while the output is held. The detail matches the single-region reporter verbatim.
        def layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(_Sq.apply, "sq", recompute=False)(h)

        out = remat.checkpoint(region_name="layers.0")(layer)(
            torch.tensor([1.0, 2.0], requires_grad=True)
        )
        self.assertExpectedInline(
            remat.format_saved_tensors_report(),
            """\
saved for backward: 16 B resident -- 1 region(s) 16 B, outside regions not walked

regions:
  16 B  layers.0  (2 storages)

outside regions: not walked (pass roots=<loss>, or roots=remat.discover_autograd_roots() for everything)

region detail:
layers.0: 16 B resident in 2 storage(s)
layers.0::sq: 16 B
  8 B  y (output at idx 0)  (2,)  float32
  8 B  gf                   (2,)  float32""",
        )
        del out

    def test_region_save_without_named_save_for_backward(self) -> None:
        # A region op that saves via PyTorch's standard ctx.save_for_backward instead of remat's
        # named remat.save_for_backward: the save carries no programmer name, so the report
        # falls back to autograd's positional label (saved.0) alongside the durable output-slot
        # role -- contrast test_regions_only_without_roots, whose _Sq names its saves "y"/"gf".
        # (Requested on the diff: an example that didn't use the special name-applying save.)
        def layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(_StdSave.apply, "sq", recompute=False)(h)

        out = remat.checkpoint(region_name="L")(layer)(
            torch.randn(3, 4, requires_grad=True)
        )
        self.assertExpectedInline(
            remat.format_saved_tensors_report(),
            """\
saved for backward: 48 B resident -- 1 region(s) 48 B, outside regions not walked

regions:
  48 B  L  (1 storage)

outside regions: not walked (pass roots=<loss>, or roots=remat.discover_autograd_roots() for everything)

region detail:
L: 48 B resident in 1 storage(s)
L::sq: 48 B
  48 B  saved.0 (output at idx 0)  (3, 4)  float32""",
        )
        del out

    def test_spans_regions_and_graph_with_dedup(self) -> None:
        # Saves both inside remat regions (two identical "layers") and outside any region
        # (a tanh "head" downstream of the layers -- reachable by the pre-backward walk,
        # unlike anything upstream of a checkpoint node). The report shows both, groups the
        # identical layers, and -- crucially -- the region SAVE storages are not re-counted
        # by the graph walk (the tapes claim them first).
        x = torch.randn(3, 4, requires_grad=True)

        def layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(_Sq.apply, "sq", recompute=False)(h)

        h = x
        for i in range(2):
            h = remat.checkpoint(region_name=f"layer.{i}")(layer)(h)
        loss = (
            h.tanh().sum()
        )  # TanhBackward: a non-region save downstream of the regions

        self.assertExpectedInline(
            remat.format_saved_tensors_report(loss),
            """\
saved for backward: 240 B resident -- 2 region(s) 192 B, outside regions 48 B

regions:
  96 B  x2  layer.0-1  (2 storages each)

outside regions: 48 B in 1 storage
       48 B  TanhBackward0 (x1)

region detail:
[x2: layer.0-1]
layer.0: 96 B resident in 2 storage(s)
layer.0::sq: 96 B
  48 B  y (output at idx 0)  (3, 4)  float32
  48 B  gf                   (3, 4)  float32""",
        )

        # Sanity: internally consistent enough to drive a real backward.
        loss.backward()

    def test_cross_region_shared_storage_billed_once(self) -> None:
        # Two regions that both save the SAME storage (a shared activation / weight-like
        # tensor). Under first-region attribution it is billed only to layer.0, so the
        # per-region summary and detail sum exactly to the header total (144 B) rather than
        # 192 B (96 x2) as a naive per-region tally would double-count. layer.1 owns only its
        # own y, so the two regions split into separate summary groups.
        shared = torch.randn(3, 4)

        def layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(
                lambda t: _SaveShared.apply(t, shared), "sq", recompute=False
            )(h)

        h = torch.randn(3, 4, requires_grad=True)
        for i in range(2):
            h = remat.checkpoint(region_name=f"layer.{i}")(layer)(h)

        report = remat.format_saved_tensors_report()
        self.assertExpectedInline(
            report,
            """\
saved for backward: 144 B resident -- 2 region(s) 144 B, outside regions not walked

regions:
  96 B  layer.0  (2 storages)
  48 B  layer.1  (1 storage)

outside regions: not walked (pass roots=<loss>, or roots=remat.discover_autograd_roots() for everything)

region detail:
layer.0: 96 B resident in 2 storage(s)
layer.0::sq: 96 B
  48 B  y (output at idx 0)  (3, 4)  float32
  48 B  shared               (3, 4)  float32
layer.1: 48 B resident in 1 storage(s)
layer.1::sq: 48 B
  48 B  y (output at idx 0)  (3, 4)  float32""",
        )

        # The summary + outside-regions bytes must reconcile to the header total.
        self.assertEqual(
            _sum_summary_region_bytes(report) + _header_outside_bytes(report),
            _header_total_bytes(report),
        )

        h.sum().backward()

    def test_alternating_attention_layers_group_by_signature(self) -> None:
        # An alternating global/local attention stack: the even layers (0, 2) run one SAVE op
        # (two resident tensors, like full attention) and the odd layers (1, 3) run another
        # (a single resident tensor, like a windowed one). The report must collapse this into
        # two signature groups, each rendering its layers as a *split* range -- layer.0,
        # layer.2 and layer.1, layer.3 -- not one bogus 0-3 range and not four separate dumps.
        def global_layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(_Sq.apply, "attn", recompute=False)(h)

        def local_layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(_Scale.apply, "attn", recompute=False)(h)

        h = torch.randn(3, 4, requires_grad=True)
        for i in range(4):
            layer = global_layer if i % 2 == 0 else local_layer
            h = remat.checkpoint(region_name=f"layer.{i}")(layer)(h)

        self.assertExpectedInline(
            remat.format_saved_tensors_report(),
            """\
saved for backward: 288 B resident -- 4 region(s) 288 B, outside regions not walked

regions:
  96 B  x2  layer.0, layer.2  (2 storages each)
  48 B  x2  layer.1, layer.3  (1 storage each)

outside regions: not walked (pass roots=<loss>, or roots=remat.discover_autograd_roots() for everything)

region detail:
[x2: layer.0, layer.2]
layer.0: 96 B resident in 2 storage(s)
layer.0::attn: 96 B
  48 B  y (output at idx 0)  (3, 4)  float32
  48 B  gf                   (3, 4)  float32
[x2: layer.1, layer.3]
layer.1: 48 B resident in 1 storage(s)
layer.1::attn: 48 B
  48 B  s (output at idx 0)  (3, 4)  float32""",
        )
        h.sum().backward()

    def test_no_live_regions_pure_graph(self) -> None:
        # No remat regions at all: the report degrades to a pure autograd-graph walk.
        x = torch.randn(2, 3, requires_grad=True)
        loss = x.tanh().sum()
        self.assertExpectedInline(
            remat.format_saved_tensors_report(loss),
            """\
saved for backward: 24 B resident -- 0 region(s) 0 B, outside regions 24 B

outside regions: 24 B in 1 storage
       24 B  TanhBackward0 (x1)""",
        )
        loss.backward()

    def test_graph_walk_does_not_trigger_unpack_hooks(self) -> None:
        # The graph walk must read saves WITHOUT unpacking them: firing a save's unpack hook
        # on a remat/checkpoint save would trigger a recompute mid-report and corrupt the real
        # backward. Build a graph whose saves are hook-saved, take the report rooted at it, and
        # assert the unpack hook never fired -- the pre-fix code, which read ``_saved_*``, did.
        unpacked: list[torch.Tensor] = []

        def pack(t: torch.Tensor) -> torch.Tensor:
            return t

        def unpack(t: torch.Tensor) -> torch.Tensor:
            unpacked.append(t)
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            x = torch.randn(3, 4, requires_grad=True)
            loss = x.tanh().sum()  # TanhBackward saves its result under the hook

        remat.format_saved_tensors_report(loss)
        self.assertEqual(unpacked, [])  # the report never unpacked

        # The save is intact: the hook fires exactly once, during the real backward.
        loss.backward()
        self.assertEqual(len(unpacked), 1)

    def test_region_tape_has_no_python_reference_cycle(self) -> None:
        # remat's own tape/region data structures must not form a Python reference cycle
        # (which would force cyclic gc to reclaim them). Exercised with the default
        # ``subclass`` bare-op strategy, whose SAVE-output wrapper carries a persist thunk.
        #
        # Note: the region can still be pinned past backward by a *native* autograd cycle
        # (the grad-connected subclass wrapper -> torch's checkpoint frame), invisible to
        # refcounting -- which is exactly why regions are deregistered deterministically at
        # recompute (see test_region_evicted_after_backward_without_gc). This test guards the
        # separate, refcount-relevant invariant: no Python cycle in remat's data itself.
        def layer(h: torch.Tensor) -> torch.Tensor:
            return remat.region(_Sq.apply, "sq", recompute=False)(h)

        out = remat.checkpoint(region_name="L")(layer)(
            torch.randn(3, 4, requires_grad=True)
        )
        regions = _iter_live_regions()
        self.assertEqual(len(regions), 1)
        region_state = regions[0]
        self.assertIsInstance(region_state, _CheckpointRegionState)
        self.assertFalse(
            _reaches_self(region_state),
            "region tape forms a Python reference cycle back to the region state",
        )
        del out

    def test_region_evicted_after_backward_without_gc(self) -> None:
        # A region leaves the live registry the moment backward recomputes it -- by refcount,
        # not cyclic gc. With gc disabled, the registry must still be empty after backward.
        def layer(h: torch.Tensor) -> torch.Tensor:
            a = remat.region(_Sq.apply, "sq", recompute=False)(h)
            # A RECOMPUTE op guarantees the region is recomputed during backward.
            return remat.region(lambda t: t * 3, "scale", recompute=True)(a)

        gc.disable()
        try:
            h = remat.checkpoint(region_name="layer.0")(layer)(
                torch.randn(3, 4, requires_grad=True)
            )
            loss = h.sum()
            self.assertEqual(len(_iter_live_regions()), 1)
            loss.backward()
            self.assertEqual(len(_iter_live_regions()), 0)
        finally:
            gc.enable()

    def test_disconnected_graphs_multiple_roots(self) -> None:
        # Two independent autograd graphs with no shared tensors. The multi-root walk covers
        # both from their two roots; the shared visited-set only prevents re-walking within a
        # graph (there is nothing shared to collide on here). Distinct ops -> distinct
        # backward node types so each graph's contribution is identifiable.
        a = torch.randn(4, 4, requires_grad=True)
        b = torch.randn(8, requires_grad=True)
        loss_a = a.tanh().sum()  # TanhBackward saves the (4, 4) result -> 64 B
        loss_b = b.sigmoid().sum()  # SigmoidBackward saves the (8,) result -> 32 B

        self.assertExpectedInline(
            remat.format_saved_tensors_report([loss_a, loss_b]),
            """\
saved for backward: 96 B resident -- 0 region(s) 0 B, outside regions 96 B

outside regions: 96 B in 2 storages
       64 B  TanhBackward0 (x1)
       32 B  SigmoidBackward0 (x1)""",
        )

        # Disconnected means a single root reaches only its own graph: from loss_a the walk
        # sees only TanhBackward, never loss_b's SigmoidBackward.
        self.assertExpectedInline(
            remat.format_saved_tensors_report([loss_a]),
            """\
saved for backward: 64 B resident -- 0 region(s) 0 B, outside regions 64 B

outside regions: 64 B in 1 storage
       64 B  TanhBackward0 (x1)""",
        )

        loss_a.backward()
        loss_b.backward()

    def test_disconnected_graphs_each_with_regions(self) -> None:
        # Two disconnected graphs, each routed through its own remat.checkpoint region with a
        # different-shaped SAVE op. Region enumeration is global (via the live registry), so
        # both regions appear regardless of roots; the walk adds each graph's non-region tail.
        # The differing shapes keep them in separate signature groups.
        def make(name: str, x: torch.Tensor) -> torch.Tensor:
            return remat.checkpoint(region_name=name)(
                lambda h: remat.region(_Sq.apply, "sq", recompute=False)(h)
            )(x)

        out_a = make("g_a", torch.randn(3, 4, requires_grad=True))
        out_b = make("g_b", torch.randn(5, 6, requires_grad=True))
        loss_a = out_a.tanh().sum()
        loss_b = out_b.tanh().sum()

        # Both regions from the two disconnected graphs are enumerated (global registry),
        # kept in separate signature groups by their differing shapes, and both graphs'
        # non-region tails are reached (two distinct TanhBackward storages).
        self.assertExpectedInline(
            remat.format_saved_tensors_report([loss_a, loss_b]),
            """\
saved for backward: 504 B resident -- 2 region(s) 336 B, outside regions 168 B

regions:
   96 B  g_a  (2 storages)
  240 B  g_b  (2 storages)

outside regions: 168 B in 2 storages
      168 B  TanhBackward0 (x2)

region detail:
g_a: 96 B resident in 2 storage(s)
g_a::sq: 96 B
  48 B  y (output at idx 0)  (3, 4)  float32
  48 B  gf                   (3, 4)  float32
g_b: 240 B resident in 2 storage(s)
g_b::sq: 240 B
  120 B  y (output at idx 0)  (5, 6)  float32
  120 B  gf                   (5, 6)  float32""",
        )

        loss_a.backward()
        loss_b.backward()
