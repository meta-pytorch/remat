# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the ``remat.checkpoint`` region wrapper: option/user-kwarg isolation,
forward-exception state unwinding, trace collection, forced recompute before an inner
custom backward, the one-hop pytree rules at the region boundary, the
forward/recompute phase helpers, and the default saved-tensor release /
``retain_graph`` behavior."""

from __future__ import annotations

import gc
import weakref
from typing import Any

import expecttest
import torch
import torch_remat as remat
from torch_remat._recompute_boundary import _checkpoint_recompute_boundary
from torch_remat._region import _checkpoint_context_fn


class CheckpointTest(expecttest.TestCase):
    def test_checkpoint_options_do_not_collide_with_user_kwargs(self) -> None:
        def fn(x: torch.Tensor, *, region_name: str) -> torch.Tensor:
            self.assertEqual("user.kwarg", region_name)
            return x * 2

        x = torch.tensor([3.0], requires_grad=True)
        y = remat.checkpoint(region_name="checkpoint.region")(fn)(
            x,
            region_name="user.kwarg",
        )
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0])))

    def test_checkpoint_forward_exception_unwinds_remat_state(self) -> None:
        class FailingForward(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                raise RuntimeError("intentional forward failure")

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        def failing_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                FailingForward.apply,
                "failing.forward",
                recompute=False,
            )(x)

        caught_exception: RuntimeError | None = None
        try:
            remat.checkpoint(
                region_name="leak.check",
            )(failing_body)(torch.ones(1, requires_grad=True))
        except RuntimeError as exc:
            caught_exception = exc

        self.assertIsNotNone(caught_exception)

        with self.assertRaisesRegex(
            RuntimeError,
            "No active torch_remat checkpoint region",
        ):
            remat.format_current_memory_report()

        class FollowupSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def followup_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                FollowupSquare.apply,
                "followup.square",
                recompute=False,
            )(x)

        x = torch.tensor([2.0], requires_grad=True)
        y = remat.checkpoint(
            region_name="followup.check",
        )(followup_body)(x)
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0])))

    def test_collect_trace_records_original_forward_annotations(self) -> None:
        def scope_body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                torch.sin,
                "sin",
                recompute=False,
            )(x)
            return remat.region(
                torch.cos,
                "cos",
                recompute=True,
            )(y)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.trace_scope(
                scope_body,
                "scope",
                metadata="test_flag",
            )(x)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with remat.collect_trace() as trace:
            y = remat.checkpoint()(checkpoint_body)(x)
            y.sum().backward()

        self.assertExpectedInline(
            trace.format(),
            """\
torch_remat trace
scope [test_flag]
  sin: save
  cos: recompute""",
        )

    def test_checkpoint_forces_recompute_before_inner_custom_backward(self) -> None:
        events: list[str] = []

        class Inner(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                events.append("inner_forward")
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                events.append("inner_backward_before_unpack")
                (x,) = ctx.saved_tensors
                del x
                events.append("inner_backward_after_unpack")
                return grad_output * 3

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            events.append("checkpoint_body")
            return Inner.apply(x)

        y = remat.checkpoint()(checkpoint_body)(
            torch.tensor([1.0, 2.0], requires_grad=True)
        )
        y.sum().backward()

        self.assertExpectedInline(
            "\n".join(events),
            """\
checkpoint_body
inner_forward
checkpoint_body
inner_forward
inner_backward_before_unpack
inner_backward_after_unpack""",
        )

    def test_checkpoint_boundary_saves_zero_element_trigger(self) -> None:
        packed_numels: list[int] = []
        packed_nbytes: list[int] = []
        unpacked_numels: list[int] = []
        unpacked_nbytes: list[int] = []

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            packed_numels.append(tensor.numel())
            packed_nbytes.append(tensor.untyped_storage().nbytes())
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            unpacked_numels.append(tensor.numel())
            unpacked_nbytes.append(tensor.untyped_storage().nbytes())
            return tensor

        x = torch.ones(1024, requires_grad=True)
        output = x * 3
        self.assertGreater(output.untyped_storage().nbytes(), 0)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = _checkpoint_recompute_boundary(output)
            y.sum().backward()

        self.assertEqual([0], packed_numels)
        self.assertEqual([0], packed_nbytes)
        self.assertEqual([0], unpacked_numels)
        self.assertEqual([0], unpacked_nbytes)
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 3)))

    def test_checkpoint_boundary_rejects_non_tensor_leaf(self) -> None:
        x = torch.ones(1, requires_grad=True)

        # A non-tensor leaf in the one-hop container has no place at the boundary.
        with self.assertRaisesRegex(RuntimeError, "must return a Tensor"):
            remat.checkpoint()(lambda value: (value, None))(x)

    def test_checkpoint_boundary_preserves_tuple_subclass(self) -> None:
        # _pytree keeps a one-hop container's own type across the round-trip, so a
        # tuple subclass constructible from one iterable (namedtuple,
        # torch.return_types, ...) is rebuilt as itself, not collapsed to plain tuple.
        class TensorTuple(tuple):
            pass

        x = torch.ones(1, requires_grad=True)
        output = remat.checkpoint()(lambda value: TensorTuple((value * 2,)))(x)

        self.assertIs(type(output), TensorTuple)
        output[0].sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2)))

    def test_checkpoint_boundary_supports_one_hop_builtin_containers(
        self,
    ) -> None:
        def fn(
            x: torch.Tensor,
        ) -> list[torch.Tensor]:
            return [x * 2, x * 3]

        x = torch.tensor([2.0], requires_grad=True)
        output = remat.checkpoint()(fn)(x)
        loss = output[0].sum() + output[1].sum()
        loss.backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0])))

    def test_checkpoint_boundary_rejects_dict(self) -> None:
        # dict is not a recognized container -- it matches the autograd.Function /
        # ATen op support level, so a dict return is an opaque, non-tensor leaf and
        # is rejected at the boundary.
        def fn(x: torch.Tensor) -> dict[str, Any]:
            return {"left": x * 2, "right": x * 3}

        x = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "must return a Tensor, or one hop of tuple/list of Tensors",
        ):
            remat.checkpoint()(fn)(x)

    def test_checkpoint_boundary_rejects_deeply_nested_containers(self) -> None:
        # The region output is a single _pytree value: a Tensor or one hop of
        # tuple/list of Tensors. A container nested one hop deeper is not
        # traversed, so its leaves are rejected at the boundary.
        def fn(x: torch.Tensor) -> list[Any]:
            return [x * 3, (x * 4,)]

        x = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "must return a Tensor, or one hop of tuple/list of Tensors",
        ):
            remat.checkpoint()(fn)(x)

    def test_phase_helpers_report_forward_and_recompute(self) -> None:
        forward_context, recompute_context = _checkpoint_context_fn()

        with forward_context:
            self.assertFalse(remat.is_recomputing())

        with recompute_context:
            self.assertTrue(remat.is_recomputing())

    def test_save_policy_checkpoint_releases_saved_tensors_by_default(self) -> None:
        saved_activation_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class SavedTensorProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                y = x * x
                saved_activation = x + 1
                nonlocal saved_activation_ref
                saved_activation_ref = weakref.ref(saved_activation)
                ctx.save_for_backward(x, saved_activation)
                return y

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                x, saved_activation = ctx.saved_tensors
                del saved_activation
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                SavedTensorProbe.apply,
                "saved.probe",
                recompute=False,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

        activation_ref = saved_activation_ref
        self.assertIsNotNone(activation_ref)
        assert activation_ref is not None
        gc.collect()
        self.assertIsNone(activation_ref())

    def test_save_policy_checkpoint_retain_graph(self) -> None:
        class RetainGraphSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                x, y = ctx.saved_tensors
                del y
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                RetainGraphSquare.apply,
                "retain.square",
                recompute=False,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward(retain_graph=True)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

        x.grad = None
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))
