# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the ``remat.op`` call surface and policy semantics: kwarg forwarding and
state cleanup outside a region, non-callable / missing-name rejection, the SAVE
default policy, SAVE running once vs RECOMPUTE rerunning its body, forward/recompute
policy agreement, duplicate op-name detection, leaf-requires-grad rejection, and the
identity node's backprop guard."""

from __future__ import annotations

from typing import Any, cast

import expecttest
import torch
import torch_remat as remat
from torch_remat._api import _MakeNonLeaf
from torch_remat._region import (
    _checkpoint_context_fn,
    _state,
)


class OpBasicsTest(expecttest.TestCase):
    def test_op_outside_checkpoint_supports_kwargs(self) -> None:
        def scale(x: torch.Tensor, *, factor: float) -> torch.Tensor:
            return x * factor

        # Outside a checkpoint region, op() is a transparent call that forwards
        # positional and keyword arguments to the wrapped function and leaves no
        # active remat state behind.
        x = torch.tensor([2.0])
        y = remat.op(
            scale,
            "scale",
            policy=remat.RECOMPUTE,
        )(x, factor=3.0)

        self.assertTrue(torch.equal(y, torch.tensor([6.0])))
        self.assertIsNone(_state.get())

        def fail() -> None:
            raise RuntimeError("intentional failure")

        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            remat.op(
                fail,
                "failure",
                policy=remat.RECOMPUTE,
            )()

        self.assertIsNone(_state.get())

    def test_op_rejects_non_callable_and_missing_name(self) -> None:
        # A non-callable first argument is rejected with a clear message. A name
        # must be supplied to get past the signature and reach this check.
        with self.assertRaisesRegex(
            RuntimeError,
            "op expects a function as its first argument",
        ):
            remat.op(
                cast(Any, "context.style"),
                "context.style",
                policy=remat.RECOMPUTE,
            )
        # op_name is a required positional; omitting it is a plain TypeError.
        with self.assertRaises(TypeError):
            remat.op(torch.sin, policy=remat.RECOMPUTE)

    def test_op_policy_defaults_to_save(self) -> None:
        # A checkpoint region recomputes by default, so an explicit op annotation
        # defaults to SAVE. Only SAVE ops get a record, so a policy-less op that
        # produces a record proves the default.
        forward_context, _ = _checkpoint_context_fn("r")
        with forward_context:
            remat.op(lambda t: t + 1, "defaulted")(
                torch.tensor([1.0], requires_grad=True)
            )
            active = _state.get()
            assert active is not None
            self.assertIn("defaulted", active.region_state.records)

    def test_save_op_runs_once_and_skips_recompute_body(self) -> None:
        class FunctionStyleSquare(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                FunctionStyleSquare.forward_runs += 1
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.op(
                FunctionStyleSquare.apply,
                "function.style.square",
                policy=remat.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(1, FunctionStyleSquare.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_recompute_op_reruns_body(self) -> None:
        class ReadmeSquare(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ReadmeSquare.forward_runs += 1
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.op(
                ReadmeSquare.apply,
                "readme.square",
                policy=remat.RECOMPUTE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(2, ReadmeSquare.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_recompute_policy_must_match_forward_policy(self) -> None:
        class PolicyDrift(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def run(x: torch.Tensor) -> torch.Tensor:
            policy = remat.RECOMPUTE if remat.is_recomputing() else remat.SAVE
            return remat.op(PolicyDrift.apply, "policy.drift", policy=policy)(x)

        y = remat.checkpoint()(run)(torch.ones(1, requires_grad=True))

        with self.assertRaisesRegex(RuntimeError, "Conflicting checkpoint policies"):
            y.sum().backward()

    def test_leaf_requires_grad_output_from_remat_op_errors(self) -> None:
        # A remat.op that returns a requires-grad *leaf* (a bare allocation, not the
        # result of a real computation) is rejected: an autograd.Function or a
        # differentiable op always gives its output a grad_fn, so a leaf here is
        # meaningless (disconnected from the region's inputs). Caught in the original
        # forward, for both SAVE and RECOMPUTE ops.
        def alloc(t: torch.Tensor) -> torch.Tensor:
            return torch.full_like(t, 2.0).requires_grad_(True)

        x = torch.randn(4, dtype=torch.float64, requires_grad=True)

        def save_block(x: torch.Tensor) -> torch.Tensor:
            return remat.op(alloc, "alloc", policy=remat.SAVE)(x)

        with self.assertRaisesRegex(
            RuntimeError, "returned a leaf tensor that requires grad"
        ):
            remat.checkpoint()(save_block)(x)

        def recompute_block(x: torch.Tensor) -> torch.Tensor:
            return remat.op(alloc, "alloc", policy=remat.RECOMPUTE)(x)

        with self.assertRaisesRegex(
            RuntimeError, "returned a leaf tensor that requires grad"
        ):
            remat.checkpoint()(recompute_block)(x)

    def test_op_outside_checkpoint_preserves_saved_tensors_hooks(self) -> None:
        packed_shapes: list[tuple[int, ...]] = []

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            packed_shapes.append(tuple(tensor.shape))
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        class UncheckpointedSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.scale = 2
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * ctx.scale * x

        # Outside a checkpoint region op() is a plain call, so the wrapped
        # Function's own save_for_backward flows through user saved_tensors_hooks
        # untouched by remat.
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = remat.op(
                UncheckpointedSquare.apply,
                "sq",
                policy=remat.SAVE,
            )(x)
            y.sum().backward()

        self.assertTrue(torch.equal(y.detach(), torch.tensor([4.0, 9.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))
        self.assertEqual([(2,)], packed_shapes)

    def test_duplicate_op_name_errors_in_forward(self) -> None:
        class FirstDuplicate(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        class SecondDuplicate(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            y = remat.op(
                FirstDuplicate.apply,
                "duplicate.forward",
                policy=remat.SAVE,
            )(x)
            return remat.op(
                SecondDuplicate.apply,
                "duplicate.forward",
                policy=remat.SAVE,
            )(y)

        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate torch_remat op name.*during forward",
        ):
            remat.checkpoint()(checkpoint_body)(torch.ones(1, requires_grad=True))

    def test_identity_node_rejects_backprop(self) -> None:
        # _MakeNonLeaf is fabricated only during recompute to reshape autograd
        # metadata, and non-reentrant checkpoint discards that graph -- so its
        # backward must never run. Apply it directly and backprop to assert the
        # guard fires rather than silently passing gradient through.
        x = torch.ones(3, requires_grad=True)
        y = _MakeNonLeaf.apply(x)
        with self.assertRaisesRegex(
            RuntimeError, "must never be backpropagated through"
        ):
            y.sum().backward()
