# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the ``remat.recompute_needs_tensor`` API -- the explicit, consumer-side
replacement for bare-op detection. A SAVE region's output that feeds a *bare*
(un-``remat.region``-wrapped) consumer is not persisted by default, so it reads a
placeholder during recompute and raises; calling ``remat.recompute_needs_tensor(t)`` on
the output right before the bare op forces the producer to save it. Placing the call on
the consumer side means the output is persisted only when that code path runs -- you can
never over-save. A ``remat.region`` consumer (including one that receives a bare view of
the output, resolved by storage) still triggers the save automatically."""

from __future__ import annotations

from typing import Callable

import expecttest
import pytest
import torch
import torch_remat as remat
from remat_test_helpers import _ref_grad, checkpoint_for_test


def _recompute_error(
    body: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> str:
    """Run ``body`` under checkpoint, expect a placeholder ``RuntimeError`` during recompute,
    and return its message."""

    try:
        checkpoint_for_test()(body)(x).sum().backward()
    except RuntimeError as error:
        return str(error)
    raise AssertionError("expected a placeholder RuntimeError during recompute")


# The SAVE region bodies below are plain ops on purpose: `t * 2` (and `(t * 2, t * 3)`)
# save nothing for backward, so the region's output is not resident during recompute --
# which is exactly the condition that produces a placeholder for a bare consumer to hit.
class RecomputeNeedsTensorTest(expecttest.TestCase):
    @pytest.mark.compile_xfail("compiled graphs carry saved-region outputs directly")
    def test_bare_consumer_without_annotation_raises_during_recompute(self) -> None:
        # A bare consumer of a SAVE output is not detected, so during recompute it reads
        # the skipped op's placeholder and raises. The message names the producing region
        # and tells you to call recompute_needs_tensor right before the bare consumer.
        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(lambda t: t * 2, "mul", recompute=False)(x)
            return torch.relu(y)  # bare consumer of the SAVE output

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        self.assertExpectedInline(
            _recompute_error(body, x),
            """\
mul.out is a placeholder for the output of remat.region 'mul' (recompute=False): a saved region is skipped during recompute, so its output is not recomputed -- only a metadata placeholder stands in, and something read its data.
A saved region's output is real during recompute only if some consumer made the producer durably save it. A remat.region consumer (including one that receives a bare view of the output) does this automatically. A bare, unwrapped op -- a residual add, a view then .contiguous(), a raw kernel, anything not wrapped in remat.region -- cannot be detected, and hits this placeholder instead.
This depends on recompute: it appears only because 'mul' has recompute=False; the same code works with recompute=True (its output is then real).
To fix it, call remat.recompute_needs_tensor(t) on the output tensor, right before the bare op that reads it, to force the producer to durably save it. Placing the call at the consumer means the output is saved only when that code path actually runs, so you can never over-save.""",
        )

    def test_recompute_needs_tensor_lets_a_bare_consumer_work(self) -> None:
        # recompute_needs_tensor on the output, right before the bare relu, forces the
        # producer to persist it, so the consumer reads real data during recompute.
        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(lambda t: t * 2, "mul", recompute=False)(x)
            remat.recompute_needs_tensor(y)  # persist before the bare consumer
            return torch.relu(y)

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        checkpoint_for_test()(body)(x).sum().backward()
        # d/dx relu(2x) = 2 where 2x > 0.
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))

    def test_recompute_needs_tensor_outside_a_region_is_a_noop(self) -> None:
        # Safe to call anywhere: with no active checkpoint region it does nothing (and does
        # not raise), so model code that calls it works whether or not it is checkpointed.
        remat.recompute_needs_tensor(torch.ones(2))  # no active region -> no-op

    def test_recompute_needs_tensor_on_non_save_output_is_a_noop(self) -> None:
        # Called on a tensor that is not a SAVE region's output (an ordinary recomputed
        # tensor), it is a no-op -- the tensor is recomputed normally, so nothing breaks.
        def body(x: torch.Tensor) -> torch.Tensor:
            y = x * 2  # bare, recomputed -- not a SAVE output
            remat.recompute_needs_tensor(y)  # no-op: y is not a SAVE output
            return torch.relu(y)

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        checkpoint_for_test()(body)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))

    def test_recompute_needs_tensor_on_one_of_two_outputs(self) -> None:
        # A multi-output region: persist only position 0 (consumed by a bare relu); position
        # 1 is consumed by nothing, so it stays unpersisted -- no over-save.
        def body(x: torch.Tensor) -> torch.Tensor:
            a, _b = remat.region(lambda t: (t * 2, t * 3), "split", recompute=False)(x)
            remat.recompute_needs_tensor(a)  # only position 0 is needed
            return torch.relu(a)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        checkpoint_for_test()(body)(x).sum().backward()
        # Only a = 2x is used: d/dx relu(2x) = 2 (both positive); b unused -> grad 0.
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 2.0])))

    def test_recompute_needs_tensor_persists_both_outputs(self) -> None:
        # Marking both outputs persists each, so bare consumers of both positions work.
        def body(x: torch.Tensor) -> torch.Tensor:
            a, b = remat.region(lambda t: (t * 2, t * 3), "split", recompute=False)(x)
            remat.recompute_needs_tensor(a, b)
            return torch.relu(a) + torch.relu(b)  # bare consumers of both

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        checkpoint_for_test()(body)(x).sum().backward()
        # d/dx (relu(2x) + relu(3x)) = 2 + 3 = 5 (all positive).
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))

    def test_recompute_needs_tensor_on_bare_view_persists_the_base(self) -> None:
        # The tensor is resolved to its producer by storage, so marking a bare *view* of
        # the SAVE output persists the underlying output; a bare consumer of the view works.
        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(lambda t: t * 2, "mul", recompute=False)(x)
            v = y.reshape(-1)  # bare view of the SAVE output
            remat.recompute_needs_tensor(v)  # mark the view -> persists the base
            return torch.relu(v)  # bare consumer of the view

        def reference(x: torch.Tensor) -> torch.Tensor:
            return torch.relu((x * 2).reshape(-1))

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        checkpoint_for_test()(body)(x).sum().backward()
        self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, x)))

    def test_region_consuming_bare_view_of_save_output_works(self) -> None:
        # A remat.region consuming a *bare view* of a SAVE output needs no annotation: the
        # view is resolved to its producer by storage, so the consumer triggers the
        # producer's save on the forward, and recompute reproduces the base and the view.
        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(lambda t: t * 2, "mul", recompute=False)(x)
            v = y.reshape(-1)  # bare view of the SAVE output
            return remat.region(torch.mul, "consume", recompute=True)(v, 3.0)

        def reference(x: torch.Tensor) -> torch.Tensor:
            return (x * 2).reshape(-1) * 3.0

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        checkpoint_for_test(region_name="r")(body)(x).sum().backward()
        # d/dx (2x * 3) = 6.
        self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, x)))

    def test_recompute_needs_tensor_is_a_noop_with_recompute_true(self) -> None:
        # A recompute=True region reruns during recompute, so its output is always real --
        # recompute_needs_tensor on it is harmless, letting a config-driven call site invoke
        # it unconditionally regardless of the producer's recompute setting.
        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(lambda t: t * 2, "mul", recompute=True)(x)
            remat.recompute_needs_tensor(y)  # no-op: mul reruns in recompute
            return torch.relu(y)  # bare consumer is fine

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        checkpoint_for_test()(body)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))
