# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for checkpoints run under ambient ``no_grad``."""

from __future__ import annotations

from collections import Counter
from typing import Callable

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import checkpoint_for_test, IS_COMPILE_TEST


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


def _square(t: torch.Tensor) -> torch.Tensor:
    return t * t


class NoGradTest(expecttest.TestCase):
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
