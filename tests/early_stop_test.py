# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint early stop: replay ends at the last op that saves for backward."""

from __future__ import annotations

import gc
import weakref

import expecttest
import pytest
import torch
import torch_remat as remat
from remat_test_helpers import (  # pyrefly: ignore[missing-import]
    checkpoint_for_test,
    IS_COMPILE_TEST,
)

pytestmark = pytest.mark.skipif(
    IS_COMPILE_TEST, reason="eager replay behavior; compile has its own partitioner"
)


def _scale(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0


class EarlyStopTest(expecttest.TestCase):
    def test_trailing_bare_consumer_is_not_replayed_and_not_persisted(self) -> None:
        tail_runs = 0
        saved_outputs: list[weakref.ReferenceType[torch.Tensor]] = []

        def tail(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            nonlocal tail_runs
            tail_runs += 1
            return a + x

        def body(x: torch.Tensor) -> torch.Tensor:
            a = remat.region(_scale, "scale", recompute=False)(x)
            if not remat.is_recomputing():
                saved_outputs.append(weakref.ref(a))
            # The add saves nothing for backward, so replay never reaches it and the
            # pin on its input is dropped at the end of the forward.
            remat.recompute_needs_tensor(a)
            return tail(a, x)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = checkpoint_for_test()(body)(x)
        gc.collect()
        self.assertIsNone(saved_outputs[0]())
        y.sum().backward()

        self.assertEqual(1, tail_runs)
        # pyrefly: ignore[bad-argument-type]
        self.assertTrue(torch.equal(x.grad, torch.tensor([3.0, 3.0])))

    def test_pinned_output_read_before_the_last_save_is_kept(self) -> None:
        def body(x: torch.Tensor) -> torch.Tensor:
            a = remat.region(_scale, "scale", recompute=False)(x)
            remat.recompute_needs_tensor(a)
            # sin saves its input, so replay reaches it and needs the real output.
            return a.sin()

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = checkpoint_for_test()(body)(x)
        y.sum().backward()

        # pyrefly: ignore[bad-argument-type]
        self.assertTrue(torch.allclose(x.grad, 2.0 * torch.cos(2.0 * x.detach())))

    def test_saved_input_from_recompute_is_rederived_without_later_saves(self) -> None:
        def body(x: torch.Tensor) -> torch.Tensor:
            r = remat.region(_scale, "double", recompute=True)(x)
            # sin saves its input r, which comes from a recompute region, so replay
            # must reach this skipped op to rederive it even though nothing after it
            # saves for backward.
            return remat.region(torch.sin, "sin", recompute=False)(r)

        x = torch.tensor([0.5, 1.0], requires_grad=True)
        y = checkpoint_for_test()(body)(x)
        y.sum().backward()

        # pyrefly: ignore[bad-argument-type]
        self.assertTrue(torch.allclose(x.grad, 2.0 * torch.cos(2.0 * x.detach())))

    def test_retain_graph_replays_each_backward(self) -> None:
        def body(x: torch.Tensor) -> torch.Tensor:
            a = remat.region(_scale, "scale", recompute=False)(x)
            remat.recompute_needs_tensor(a)
            return a.sin() + x

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = checkpoint_for_test()(body)(x)
        y.sum().backward(retain_graph=True)
        # pyrefly: ignore[missing-attribute]
        first = x.grad.clone()
        x.grad = None
        y.sum().backward()

        # pyrefly: ignore[bad-argument-type]
        self.assertTrue(torch.equal(first, x.grad))
        # pyrefly: ignore[bad-argument-type]
        self.assertTrue(torch.allclose(x.grad, 2.0 * torch.cos(2.0 * x.detach()) + 1.0))


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
