# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``remat.Box``: returning a non-tensor value from a region. A box from a
``recompute=False`` region is kept for recompute only when code that reruns reads its
``value`` on the forward, and otherwise replays as a placeholder box."""

from __future__ import annotations

import gc
import weakref
from typing import Any

import expecttest
import pytest
import torch
import torch_remat as remat
from remat_test_helpers import (  # pyrefly: ignore[missing-import]
    checkpoint_for_test,
    IS_COMPILE_TEST,
)
from torch._dynamo.testing import CompileCounter
from torch.utils._pytree import tree_flatten
from torch_remat._region import _CheckpointRegionState, _state

_EAGER_ONLY = pytest.mark.skipif(
    IS_COMPILE_TEST, reason="inspects the eager recompute tape"
)


class _Handle:
    """An opaque, non-tensor value a dispatch kernel hands to its combine kernel."""

    def __init__(self, scale: float) -> None:
        self.scale = scale


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> tuple[torch.Tensor, remat.Box[_Handle]]:
        del ctx
        return x * 2, remat.Box(_Handle(3.0))

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_y: torch.Tensor, grad_box: None
    ) -> torch.Tensor:
        del ctx, grad_box
        return grad_y * 2


class _Combine(torch.autograd.Function):
    """Uses the handle in forward and keeps the box for its backward."""

    @staticmethod
    def forward(ctx: Any, y: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
        ctx.box = box
        return y * box.value.scale

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return grad_out * ctx.box.value.scale, None


def _capture_region(regions: list[_CheckpointRegionState]) -> None:
    if IS_COMPILE_TEST or remat.is_recomputing():
        return
    active = _state.get()
    if active is not None:
        regions.append(active.region_state)


def _reference_grad(x: torch.Tensor) -> torch.Tensor:
    # out = square(2x) * 3
    return 24 * x


class BoxTest(expecttest.TestCase):
    def test_box_is_a_pytree_leaf(self) -> None:
        box = remat.Box((torch.ones(2), {"k": 1}))
        leaves, _ = tree_flatten((box, torch.zeros(1)))
        self.assertIs(box, leaves[0])
        self.assertEqual(2, len(leaves))

    def test_box_outside_checkpoint_is_plain(self) -> None:
        handle = _Handle(1.0)
        y, box = remat.region(lambda x: (x, remat.Box(handle)), "r", recompute=False)(
            torch.ones(1)
        )
        self.assertIs(handle, box.value)
        self.assertFalse(remat.is_placeholder(box))

    def test_box_read_by_recompute_region_is_kept(self) -> None:
        regions: list[_CheckpointRegionState] = []
        handles: list[_Handle] = []

        def combine(y: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
            if not IS_COMPILE_TEST:
                handles.append(box.value)
            return y * box.value.scale

        def block(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            z = remat.region(torch.square, "expert", recompute=True)(y)
            return remat.region(combine, "combine", recompute=True)(z, box)

        x = torch.arange(3.0, requires_grad=True)
        out = checkpoint_for_test(region_name="layer")(block)(x)
        if not IS_COMPILE_TEST:
            self.assertEqual([1], list(regions[0].records["dispatch"].box_slots))
        out.sum().backward()

        torch.testing.assert_close(x.grad, _reference_grad(x.detach()))
        if not IS_COMPILE_TEST:
            # The replayed combine sees the same handle object the forward produced,
            # and the tape slot is freed once replay serves it.
            self.assertEqual(2, len(handles))
            self.assertIs(handles[0], handles[1])
            self.assertEqual({}, regions[0].records["dispatch"].box_slots)

    def test_box_read_by_bare_code_is_kept(self) -> None:
        def block(x: torch.Tensor) -> torch.Tensor:
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            remat.recompute_needs_tensor(y)
            return torch.square(y) * box.value.scale

        x = torch.arange(3.0, requires_grad=True)
        checkpoint_for_test()(block)(x).sum().backward()
        torch.testing.assert_close(x.grad, _reference_grad(x.detach()))

    def test_box_read_only_by_save_region_is_not_kept(self) -> None:
        regions: list[_CheckpointRegionState] = []
        replayed: list[remat.Box[_Handle]] = []

        def block(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            if remat.is_recomputing():
                replayed.append(box)
            z = remat.region(torch.square, "expert", recompute=True)(y)
            return remat.region(_Combine.apply, "combine", recompute=False)(z, box)

        x = torch.arange(3.0, requires_grad=True)
        out = checkpoint_for_test()(block)(x)
        if not IS_COMPILE_TEST:
            self.assertEqual({}, regions[0].records["dispatch"].box_slots)
        out.sum().backward()

        # combine's backward reads the handle through the box its forward kept.
        torch.testing.assert_close(x.grad, _reference_grad(x.detach()))
        if not IS_COMPILE_TEST:
            (box,) = replayed
            self.assertTrue(remat.is_placeholder(box))

    @_EAGER_ONLY
    def test_reading_placeholder_box_during_recompute_raises(self) -> None:
        def block(x: torch.Tensor) -> torch.Tensor:
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            # A read only during recompute is one the forward never saw.
            remat.recompute_needs_tensor(y)
            scale = box.value.scale if remat.is_recomputing() else 3.0
            return torch.square(y) * scale

        x = torch.arange(3.0, requires_grad=True)
        out = remat.checkpoint(region_name="layer")(block)(x)
        with self.assertRaisesRegex(
            RuntimeError,
            r"layer::dispatch\.1 is a placeholder for a remat\.Box returned by "
            r"remat\.region 'layer::dispatch'",
        ):
            out.sum().backward()

    def test_box_kept_across_retain_graph(self) -> None:
        def block(x: torch.Tensor) -> torch.Tensor:
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            remat.recompute_needs_tensor(y)
            return torch.square(y) * box.value.scale

        x = torch.arange(3.0, requires_grad=True)
        out = checkpoint_for_test()(block)(x).sum()
        out.backward(retain_graph=True)
        out.backward()
        torch.testing.assert_close(x.grad, 2 * _reference_grad(x.detach()))

    def test_bare_box_output_and_checkpoint_output(self) -> None:
        handle = _Handle(3.0)

        def block(x: torch.Tensor) -> tuple[torch.Tensor, remat.Box[_Handle]]:
            box = remat.region(lambda: remat.Box(handle), "make", recompute=False)()
            return x * box.value.scale, box

        x = torch.arange(3.0, requires_grad=True)
        out, box = checkpoint_for_test()(block)(x)
        self.assertIs(handle, box.value)
        out.sum().backward()
        torch.testing.assert_close(x.grad, torch.full((3,), 3.0))

    @_EAGER_ONLY
    def test_box_from_other_checkpoint_is_not_kept_there(self) -> None:
        regions: list[_CheckpointRegionState] = []

        def producer(x: torch.Tensor) -> tuple[torch.Tensor, remat.Box[_Handle]]:
            _capture_region(regions)
            return remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)

        def consumer(y: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
            return y * box.value.scale

        x = torch.arange(3.0, requires_grad=True)
        y, box = remat.checkpoint()(producer)(x)
        out = remat.checkpoint()(consumer)(y, box)
        self.assertEqual({}, regions[0].records["dispatch"].box_slots)
        out.sum().backward()
        torch.testing.assert_close(x.grad, torch.full((3,), 6.0))

    @_EAGER_ONLY
    def test_memory_report_counts_boxed_outputs(self) -> None:
        reports: list[str] = []

        def block(x: torch.Tensor) -> torch.Tensor:
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            remat.recompute_needs_tensor(y)
            out = torch.square(y) * box.value.scale
            if not remat.is_recomputing():
                reports.append(remat.format_current_memory_report())
            return out

        x = torch.arange(3.0, requires_grad=True)
        remat.checkpoint(region_name="layer")(block)(x).sum().backward()
        self.assertIn("+ 1 boxed output kept for recompute", reports[0])

    @_EAGER_ONLY
    def test_box_passed_to_compiled_recompute_region_is_kept(self) -> None:
        torch._dynamo.reset()

        @torch.compile(backend="eager", fullgraph=True)
        def combine(z: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
            return z * box.value.scale

        def block(x: torch.Tensor) -> torch.Tensor:
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            z = remat.region(torch.square, "expert", recompute=True)(y)
            return remat.region(combine, "combine", recompute=True)(z, box)

        x = torch.arange(3.0, requires_grad=True)
        remat.checkpoint()(block)(x).sum().backward()
        torch.testing.assert_close(x.grad, _reference_grad(x.detach()))

    @_EAGER_ONLY
    def test_box_read_in_compiled_function_is_kept(self) -> None:
        if torch.__version__ < (2, 14):
            self.skipTest("Dynamo traces ContextVar.get() from torch 2.14")
        torch._dynamo.reset()
        counter = CompileCounter()
        regions: list[_CheckpointRegionState] = []

        # Called from bare checkpoint code: the eager checkpoint replays it on
        # recompute, so its read keeps the handle.
        @torch.compile(backend=counter, fullgraph=True)
        def combine(z: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
            return z * box.value.scale

        def block(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            z = remat.region(torch.square, "expert", recompute=True)(y)
            return combine(z, box)

        for _ in range(2):
            x = torch.arange(3.0, requires_grad=True)
            out = remat.checkpoint()(block)(x)
            self.assertEqual([1], list(regions[-1].records["dispatch"].box_slots))
            out.sum().backward()
            torch.testing.assert_close(x.grad, _reference_grad(x.detach()))
        # One graph for the forward and one for recompute; later checkpoints reuse them.
        self.assertEqual(2, counter.frame_count)

    @_EAGER_ONLY
    def test_box_read_in_compiled_save_region_is_not_kept(self) -> None:
        if torch.__version__ < (2, 14):
            self.skipTest("Dynamo traces ContextVar.get() from torch 2.14")
        torch._dynamo.reset()
        regions: list[_CheckpointRegionState] = []

        @torch.compile(backend="eager", fullgraph=True)
        def combine(z: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
            return z * box.value.scale

        def block(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            z = remat.region(torch.square, "expert", recompute=True)(y)
            return remat.region(combine, "combine", recompute=False)(z, box)

        x = torch.arange(3.0, requires_grad=True)
        out = remat.checkpoint()(block)(x)
        self.assertEqual({}, regions[0].records["dispatch"].box_slots)
        out.sum().backward()
        torch.testing.assert_close(x.grad, _reference_grad(x.detach()))

    @_EAGER_ONLY
    def test_long_lived_box_does_not_pin_regions(self) -> None:
        handle = remat.Box(_Handle(2.0))
        regions: list[weakref.ReferenceType[_CheckpointRegionState]] = []

        def block(x: torch.Tensor, box: remat.Box[_Handle]) -> torch.Tensor:
            active = _state.get()
            if active is not None and not remat.is_recomputing():
                regions.append(weakref.ref(active.region_state))
            y, same = remat.region(
                lambda x, box: (x * 2, box), "passthrough", recompute=False
            )(x, box)
            return remat.region(_Combine.apply, "combine", recompute=False)(y, same)

        for _ in range(3):
            x = torch.arange(3.0, requires_grad=True)
            remat.checkpoint()(block)(x, handle).sum().backward()
        gc.collect()
        self.assertEqual(3, len(regions))
        self.assertTrue(all(region() is None for region in regions))

    @_EAGER_ONLY
    def test_recompute_needs_tensor_keeps_box(self) -> None:
        regions: list[_CheckpointRegionState] = []

        def block(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            y, box = remat.region(_Dispatch.apply, "dispatch", recompute=False)(x)
            remat.recompute_needs_tensor(y, box)
            # A read only during recompute, which remat cannot see on the forward.
            scale = box.value.scale if remat.is_recomputing() else 3.0
            return torch.square(y) * scale

        x = torch.arange(3.0, requires_grad=True)
        out = remat.checkpoint()(block)(x)
        self.assertEqual([1], list(regions[0].records["dispatch"].box_slots))
        out.sum().backward()
        torch.testing.assert_close(x.grad, _reference_grad(x.detach()))

    def test_recompute_needs_tensor_rejects_other_values(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expects Tensor or remat.Box"):
            remat.recompute_needs_tensor(
                _Handle(1.0)
            )  # pyrefly: ignore[bad-argument-type]
