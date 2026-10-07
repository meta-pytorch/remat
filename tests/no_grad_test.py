# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for grad-free code and checkpoints: a whole checkpoint under ``no_grad`` /
``inference_mode`` (runs once, records nothing), grad enabled but nothing requiring grad,
and ``no_grad`` blocks inside a recording checkpoint. Checks that gradients match eager,
that grad-free SAVE regions are never replayed and persist an output only for a
consumer that reruns, and when a grad-free RECOMPUTE region must still be replayed."""

from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager
from typing import Callable

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import checkpoint_for_test, IS_COMPILE_TEST
from torch_remat._region import _CheckpointRegionState, _iter_live_regions, _state


class _Calls:
    """Counts each wrapped region body's forward and replay invocations."""

    def __init__(self) -> None:
        self.forward: Counter[str] = Counter()
        self.replay: Counter[str] = Counter()

    def region(
        self,
        function: Callable[..., torch.Tensor],
        name: str,
        *,
        recompute: bool,
    ) -> Callable[..., torch.Tensor]:
        # A compiled body cannot mutate the counters (and would count at trace time).
        if IS_COMPILE_TEST:
            return remat.region(function, name, recompute=recompute)

        def counted(*args: torch.Tensor) -> torch.Tensor:
            (self.replay if remat.is_recomputing() else self.forward)[name] += 1
            return function(*args)

        return remat.region(counted, name, recompute=recompute)

    def assert_counts(
        self,
        test: expecttest.TestCase,
        forward: dict[str, int],
        replay: dict[str, int],
    ) -> None:
        if IS_COMPILE_TEST:
            return
        test.assertEqual(forward, dict(self.forward))
        test.assertEqual(replay, dict(self.replay))


def _capture_region(box: list[_CheckpointRegionState]) -> None:
    """From a checkpoint body, record the active region's state on the forward (a no-op
    when the body runs uncheckpointed as the eager reference)."""

    if IS_COMPILE_TEST or remat.is_recomputing():
        return
    active = _state.get()
    if active is not None:
        box.append(active.region_state)


def _mask(t: torch.Tensor) -> torch.Tensor:
    return (t > 0).to(t.dtype)


def _square(t: torch.Tensor) -> torch.Tensor:
    return t * t


def _eager_grad(body: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """The input grad of ``body(x).sum()`` run uncheckpointed, where regions are plain
    calls."""

    x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
    body(x).sum().backward()
    assert x.grad is not None
    return x.grad


class NoGradTest(expecttest.TestCase):
    def test_checkpoint_under_ambient_no_grad_runs_body_once(self) -> None:
        for mode in (torch.no_grad, torch.inference_mode):
            with self.subTest(mode=mode.__name__):
                self._check_ambient_mode_runs_body_once(mode)

    def _check_ambient_mode_runs_body_once(
        self, mode: Callable[[], AbstractContextManager[None]]
    ) -> None:
        calls = _Calls()
        in_region: list[bool] = []

        def body(x: torch.Tensor) -> torch.Tensor:
            if not IS_COMPILE_TEST:
                in_region.append(_state.get() is not None)
            h = calls.region(_square, "save", recompute=False)(x)
            h = calls.region(torch.sin, "recompute", recompute=True)(h)
            return h * 2

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        with mode():
            out = checkpoint_for_test(region_name="r")(body)(x)

        self.assertFalse(out.requires_grad)
        torch.testing.assert_close(out, torch.sin(x.detach() ** 2) * 2)
        calls.assert_counts(self, {"save": 1, "recompute": 1}, {})
        if not IS_COMPILE_TEST:
            # Regions ran as plain calls: no active region, nothing registered.
            self.assertEqual([False], in_region)
        self.assertEqual([], _iter_live_regions())

    def test_checkpoint_under_ambient_no_grad_packs_nothing(self) -> None:
        kinds: list[remat.SavedTensorKind] = []

        def pack(t: torch.Tensor) -> torch.Tensor:
            info = remat.current_saved_tensor_info()
            assert info is not None
            kinds.append(info.kind)
            return t

        def body(x: torch.Tensor) -> torch.Tensor:
            h = remat.region(_square, "save", recompute=False)(x)
            return remat.region(torch.sin, "recompute", recompute=True)(h)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        with torch.no_grad():
            remat.checkpoint(saved_tensors_hooks=(pack, lambda t: t))(body)(x)
        self.assertEqual([], kinds)

    def test_enable_grad_inside_no_grad_checkpoint(self) -> None:
        # Under ambient no_grad the checkpoint does not record, so a body that re-enables
        # grad builds an ordinary graph: regions are plain calls and nothing replays.
        calls = _Calls()

        def body(x: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                h = calls.region(_square, "save", recompute=False)(x)
                return calls.region(torch.sin, "recompute", recompute=True)(h)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        with torch.no_grad():
            out = checkpoint_for_test(region_name="r")(body)(x)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        torch.testing.assert_close(x.grad, torch.cos(x.detach() ** 2) * 2 * x.detach())
        calls.assert_counts(self, {"save": 1, "recompute": 1}, {})

    def test_nothing_requires_grad_never_replays(self) -> None:
        calls = _Calls()

        def body(x: torch.Tensor) -> torch.Tensor:
            h = calls.region(_square, "save", recompute=False)(x)
            h = calls.region(torch.sin, "recompute", recompute=True)(h)
            return h * 2

        out = checkpoint_for_test(region_name="r")(body)(torch.tensor([1.0, 2.0]))
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        calls.assert_counts(self, {"save": 1, "recompute": 1}, {})

    def test_no_grad_save_region_feeding_save_region_is_not_persisted(self) -> None:
        calls = _Calls()
        regions: list[_CheckpointRegionState] = []

        def body(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            with torch.no_grad():
                mask = calls.region(_mask, "mask", recompute=False)(x)
            return calls.region(torch.mul, "apply", recompute=False)(x, mask)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(body)(x)
        if not IS_COMPILE_TEST:
            self.assertEqual({}, regions[0].records["mask"].output_slots)
        out.sum().backward()

        calls.assert_counts(self, {"mask": 1, "apply": 1}, {})
        torch.testing.assert_close(x.grad, _eager_grad(body))

    def test_no_grad_save_region_feeding_recompute_region_is_persisted(self) -> None:
        calls = _Calls()
        regions: list[_CheckpointRegionState] = []

        def body(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            with torch.no_grad():
                mask = calls.region(_mask, "mask", recompute=False)(x)
            return calls.region(torch.mul, "apply", recompute=True)(x, mask)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(body)(x)
        if not IS_COMPILE_TEST:
            self.assertEqual([0], list(regions[0].records["mask"].output_slots))
        out.sum().backward()

        # The mask is ferried to the replayed consumer, not recomputed.
        calls.assert_counts(self, {"mask": 1, "apply": 1}, {"apply": 1})
        torch.testing.assert_close(x.grad, _eager_grad(body))

    def test_no_grad_save_region_feeding_bare_op(self) -> None:
        calls = _Calls()

        def body(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                mask = calls.region(_mask, "mask", recompute=False)(x)
            remat.recompute_needs_tensor(mask)
            return x * mask

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(body)(x)
        out.sum().backward()

        calls.assert_counts(self, {"mask": 1}, {})
        torch.testing.assert_close(x.grad, _eager_grad(body))

    def test_no_grad_recompute_region_replays_for_save_consumer_input(self) -> None:
        # The SAVE consumer's backward needs the mask, which it rederives by replaying the
        # grad-free producer rather than retaining it.
        calls = _Calls()
        regions: list[_CheckpointRegionState] = []

        def body(x: torch.Tensor) -> torch.Tensor:
            _capture_region(regions)
            with torch.no_grad():
                mask = calls.region(_mask, "mask", recompute=True)(x)
            return calls.region(torch.mul, "apply", recompute=False)(x, mask)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(body)(x)
        if not IS_COMPILE_TEST:
            self.assertEqual(1, len(regions[0].records["apply"].saved_input_recipes))
        out.sum().backward()

        calls.assert_counts(self, {"mask": 1, "apply": 1}, {"mask": 1})
        torch.testing.assert_close(x.grad, _eager_grad(body))

    def test_grad_free_code_skips_replay_only_in_save_region(self) -> None:
        # Grad-free code saves nothing for backward either way; only recompute=False
        # keeps replay from rerunning it. Remat cannot prove a grad-free RECOMPUTE result
        # unread (a bare op may consume it), so it replays it.
        for recompute in (True, False):
            with self.subTest(recompute=recompute):
                self._check_grad_free_stat(recompute)

    def _check_grad_free_stat(self, recompute: bool) -> None:
        calls = _Calls()

        def body(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                stat = calls.region(torch.sum, "stat", recompute=recompute)(x)
            del stat
            h = calls.region(torch.sin, "sin", recompute=True)(x)
            return calls.region(_square, "square", recompute=False)(h)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(body)(x)
        out.sum().backward()

        replay = {"sin": 1, **({"stat": 1} if recompute else {})}
        calls.assert_counts(self, {"stat": 1, "sin": 1, "square": 1}, replay)
        torch.testing.assert_close(x.grad, _eager_grad(body))

    def test_no_grad_inside_region_body_is_honored_on_replay(self) -> None:
        for recompute in (True, False):
            with self.subTest(recompute=recompute):
                self._check_no_grad_inside_region_body(recompute)

    def _check_no_grad_inside_region_body(self, recompute: bool) -> None:
        # Replay runs under enable_grad, but a no_grad block inside the body still applies,
        # so the replayed stat stays out of the gradient exactly as on the forward.
        def normalize(t: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                scale = t.abs().sum()
            return t * t / scale

        def body(x: torch.Tensor) -> torch.Tensor:
            h = remat.region(torch.sin, "sin", recompute=True)(x)
            return remat.region(normalize, "norm", recompute=recompute)(h)

        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(body)(x)
        out.sum().backward()
        torch.testing.assert_close(x.grad, _eager_grad(body))
