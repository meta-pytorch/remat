# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the data-shape mechanics of a save/recompute region: named saves, ``None``
saved-tensor slots, tuple/list output schemas, multi-op spans, mixed/nested RECOMPUTE
inputs, reshape-in-region, in-place mutation detection, and the full family of
saved-view / input-ferrying cases (distinct-object aliases, slice views,
non-contiguous bases, ``retain_graph``, and layout-mismatch errors)."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast, NamedTuple

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import _ref_grad
from torch_remat._region import _checkpoint_context_fn


class SaveRegionTest(expecttest.TestCase):
    def test_save_op_restores_saved_output_view(self) -> None:
        class SavesOutputView(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> torch.Tensor:
                if remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                SavesOutputView.forward_runs += 1
                query = x
                key = x + 1
                value = x + 2
                attn_vis = torch.ones_like(x)
                raw_out = (x * x).view(2, 2)
                out = raw_out.view_as(x)
                softmax_lse = raw_out.sum(dim=1)
                ctx.save_for_backward(
                    query,
                    key,
                    value,
                    attn_vis,
                    out,
                    softmax_lse,
                )
                return out

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                query, key, value, attn_vis, out, softmax_lse = ctx.saved_tensors
                del query, key, value, attn_vis
                self.assertGreater(out.contiguous().data_ptr(), 0)
                self.assertGreater(softmax_lse.data_ptr(), 0)
                return grad_output * out.contiguous().view_as(grad_output)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                SavesOutputView.apply,
                "saves.output",
                recompute=False,
            )(x)

        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(1, SavesOutputView.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([1.0, 4.0, 9.0, 16.0])))

    def test_save_preserves_none_saved_tensor_slots(self) -> None:
        class OptionalSavedTensor(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                right = x + 1
                ctx.save_for_backward(x, None, right)
                return right

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                left, missing, right = ctx.saved_tensors
                self.assertIsNone(missing)
                return grad_output * (right - left + 1)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                OptionalSavedTensor.apply,
                "optional.save",
                recompute=False,
            )(x)

        x = torch.tensor([3.0, 4.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2.0)))

    def test_save_op_tuple_output_schema_and_grad(self) -> None:
        class TupleReturn(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                TupleReturn.forward_runs += 1
                left = x * x
                right = x + 1
                ctx.save_for_backward(x, left, right)
                return left, right

            @staticmethod
            def backward(
                ctx: Any,
                grad_left: torch.Tensor,
                grad_right: torch.Tensor,
            ) -> torch.Tensor:
                x, left, right = ctx.saved_tensors
                del left, right
                return grad_left * 2 * x + grad_right

        x = torch.tensor([2.0, 3.0], requires_grad=True)

        def checkpoint_body(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return remat.region(
                TupleReturn.apply,
                "tuple.return",
                recompute=False,
            )(value)

        left, right = remat.checkpoint()(checkpoint_body)(x)
        (left + right).sum().backward()

        self.assertEqual(1, TupleReturn.forward_runs)
        self.assertTrue(torch.equal(left.detach(), torch.tensor([4.0, 9.0])))
        self.assertTrue(torch.equal(right.detach(), torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 7.0])))

    def test_skipped_output_view_of_recomputed_tensor_replays_as_zero_storage(
        self,
    ) -> None:
        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 3
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class ViewConsumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward()
                return x[:1]

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return torch.cat([grad_output, torch.zeros_like(grad_output)])

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            produced = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(
                ViewConsumer.apply,
                "view.consumer",
                recompute=False,
            )(produced)

        y = remat.checkpoint()(checkpointed_region)(torch.ones(2, requires_grad=True))

        y.sum().backward()

    def test_save_op_saving_bare_view_of_stub_materializes(self) -> None:
        # A SAVE op saving an input that is a *bare* view of an upstream SAVE op's
        # output. The consuming SAVE op resolves the view by storage to the producer's
        # SAVE output, so it is a stub input (not reproduced by replay) and is retained
        # like any other save on the autograd graph -- it just works, producing the
        # correct gradient.
        class ProducerSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class SaveInput(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, v: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(v)
                return v * v

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (v,) = ctx.saved_tensors
                return grad_output * 2 * v

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(ProducerSave.apply, "producer", recompute=False)(x)
            v = y.view_as(y)  # bare metadata view of a SAVE output
            return remat.region(SaveInput.apply, "consumer", recompute=False)(v)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        remat.checkpoint(region_name="r")(region)(x).sum().backward()
        # d/dx (3x)^2 = 18x
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 36.0])))

    def test_save_op_recomputed_input_is_not_offloaded(self) -> None:
        # With remat offload hooks active, a SAVE op that saves a RECOMPUTE-sourced
        # input does NOT offload it (case A): it is recomputed instead. Only the
        # op's internally-produced saves reach the offloader.
        packed: list[str] = []
        stash: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> object:
            index = len(stash)
            stash.append(tensor.detach().clone())
            packed.append(f"shape{tuple(tensor.shape)}")
            return index

        def unpack(packed_obj: object) -> torch.Tensor:
            return stash[cast(int, packed_obj)]

        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                internal = y + 1
                # y is the input (case A, recomputed); internal is offloaded.
                ctx.save_for_backward(y, internal)
                return y * y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y, internal) = ctx.saved_tensors
                del internal
                return grad_output * 2 * y

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(
                Consumer.apply,
                "consumer",
                recompute=False,
            )(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with remat.saved_tensors_hooks(pack, unpack):
            out = remat.checkpoint(region_name="r")(region)(x)
            out.sum().backward()

        # Only the internal save reached the offloader; the recomputed input did not.
        self.assertEqual(["shape(2,)"], packed)
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 36.0])))

    def test_named_save_for_backward_round_trips(self) -> None:
        class Affine(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Affine.runs += 1
                z = x + 1
                remat.save_for_backward(ctx, {"x": x, "missing": None, "z": z})
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                x, missing, z = ctx.saved_tensors
                assert missing is None
                assert torch.equal(z, x + 1)  # names mapped to the right tensors
                return grad_output * 2 * x

        base = torch.tensor([2.0, 3.0], dtype=torch.float64)
        x = base.clone().requires_grad_(True)
        Affine.runs = 0
        y = remat.checkpoint()(
            lambda t: remat.region(Affine.apply, "affine", recompute=False)(t)
        )(x)
        y.sum().backward()
        self.assertEqual(1, Affine.runs)  # SAVE: body not rerun during recompute
        self.assertTrue(torch.allclose(x.grad, _ref_grad(lambda t: t * t, base)))

    def test_named_save_in_multi_op_span_keys_by_identity(self) -> None:
        # Regression: a multi-op SAVE span whose first op builds no grad_fn (its
        # input does not require grad) so its named save never packs. The dead
        # name must not shift onto the second op's saved tensor -- names are keyed
        # by tensor identity, not pack order. With order-based naming, RealSave's
        # tensor would be mislabeled "dead".
        class NoGradSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, c: torch.Tensor) -> torch.Tensor:
                remat.save_for_backward(ctx, {"dead": c})
                return c * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        class RealSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                # Save an internal grad factor, not the input x: a saved input is
                # diverted off the identity hook and recomputed (case A), so it would
                # not appear in the report -- "live" must name a retained internal.
                two_x = x * 2
                remat.save_for_backward(ctx, {"live": two_x})
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (two_x,) = ctx.saved_tensors
                return grad_output * two_x

        def span(x: torch.Tensor) -> torch.Tensor:
            const = torch.ones(2, dtype=torch.float32)  # requires_grad=False
            NoGradSave.apply(const)  # no grad_fn -> "dead" never packs
            return RealSave.apply(x)

        forward_context, _ = _checkpoint_context_fn("blk")
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        with forward_context:
            # Hold the op output so RealSave's grad_fn keeps the saved internal
            # "live" alive for the report (a saved input would instead be diverted
            # off the identity hook and not reported).
            out = remat.region(span, "span", recompute=False)(x)
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
blk: 12 B resident in 1 storage(s)
blk::span: 12 B
       12 B  live                   (3,)       float32  cpu""",
            )
            self.assertEqual((3,), tuple(out.shape))

        # And the span still produces correct gradients end to end.
        base = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        xc = base.clone().requires_grad_(True)
        remat.checkpoint(region_name="blk")(span)(xc).sum().backward()
        self.assertTrue(torch.allclose(xc.grad, _ref_grad(lambda t: t * t, base)))

    def test_recompute_op_loads_mixed_flat_inputs(self) -> None:
        """RECOMPUTE op with a placeholder input, a real input, and a kwarg.

        Only the SAVE-produced input replays as a placeholder and is ferried
        back; the recomputed input flows through directly, and the non-tensor
        ``alpha`` keyword survives the flatten/unflatten round-trip.
        """

        class SavedDouble(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            saved = remat.region(SavedDouble.apply, "double", recompute=False)(x)
            recomputed = x * 3
            # add(saved, recomputed, alpha=2) = 2x + 2 * 3x = 8x
            return remat.region(torch.add, "add", recompute=True)(
                saved, recomputed, alpha=2.0
            )

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint()(body)(x)
        out.sum().backward()

        self.assertTrue(torch.equal(out.detach(), torch.tensor([8.0, 16.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([8.0, 8.0])))

    def test_recompute_op_loads_nested_list_input(self) -> None:
        """RECOMPUTE op whose single argument is a list holding a SAVE output.

        A native-style op (here ``torch.stack``) takes a one-hop list of tensors.
        The SAVE-produced element replays as a placeholder and must be ferried back
        by leaf position even though it is nested one level inside the list.
        """

        class SavedDouble(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            saved = remat.region(SavedDouble.apply, "double", recompute=False)(x)
            recomputed = x * 3
            # stack([2x, 3x]).sum(0) = 5x
            return remat.region(
                lambda tensors: torch.stack(tensors).sum(0),
                "stack",
                recompute=True,
            )([saved, recomputed])

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint()(body)(x)
        out.sum().backward()

        self.assertTrue(torch.equal(out.detach(), torch.tensor([5.0, 10.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))

    def test_save_op_returns_list_output(self) -> None:
        """A SAVE op may return a list of tensors; recompute rebuilds it as a list."""

        seen_container: list[type] = []

        def body(x: torch.Tensor) -> torch.Tensor:
            pair = remat.region(
                lambda t: [t * 2, t * 3],
                "split",
                recompute=False,
            )(x)
            seen_container.append(type(pair))
            first, second = pair
            # add(2x, 3x) = 5x; both inputs are SAVE outputs ferried on recompute.
            return remat.region(torch.add, "add", recompute=True)(first, second)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint()(body)(x)
        out.sum().backward()

        self.assertTrue(torch.equal(out.detach(), torch.tensor([5.0, 10.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))
        # Recorded on the original forward and again on recompute (where the SAVE
        # op replays as placeholders): both must rebuild a list, not a tuple.
        self.assertEqual(seen_container, [list, list])

    def test_save_op_returns_namedtuple_output(self) -> None:
        """A SAVE op may return a NamedTuple; recompute rebuilds the same named type.

        This is the ``RouterOutput`` shape: a structured, all-Tensor return wrapped
        directly in a region. Its type must survive the round-trip on both the
        original forward and the tape replay so callers keep field access.
        """

        class Pair(NamedTuple):
            double: torch.Tensor
            triple: torch.Tensor

        seen_container: list[type] = []

        def body(x: torch.Tensor) -> torch.Tensor:
            pair = remat.region(
                lambda t: Pair(double=t * 2, triple=t * 3),
                "split",
                recompute=False,
            )(x)
            seen_container.append(type(pair))
            # Named-field access must work on the forward value and the recompute
            # reconstruction alike; both fields are SAVE outputs ferried on recompute.
            return remat.region(torch.add, "add", recompute=True)(
                pair.double, pair.triple
            )

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint()(body)(x)
        out.sum().backward()

        self.assertTrue(torch.equal(out.detach(), torch.tensor([5.0, 10.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))
        # Rebuilt as the NamedTuple type on the original forward and again on
        # recompute (where the SAVE op replays from the tape), not a plain tuple.
        self.assertEqual(seen_container, [Pair, Pair])

    def test_multi_op_save_region(self) -> None:
        base = torch.randn(5, dtype=torch.float64)

        def block(t: torch.Tensor) -> torch.Tensor:
            # A SAVE region of several native ops: every internal SavedVariable is
            # taped under one nested hook and the whole span is skipped on recompute.
            saved = remat.region(
                lambda u: torch.exp(torch.sin(u)) * u,
                "blk",
                recompute=False,
            )(t)
            return remat.region(lambda u: u * u, "sq", recompute=True)(saved)

        def reference(t: torch.Tensor) -> torch.Tensor:
            return (torch.exp(torch.sin(t)) * t) ** 2

        x = base.clone().requires_grad_(True)
        y = remat.checkpoint()(block)(x)
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, base)))

    def test_save_op_with_reshape_in_region_feeds_recompute(self) -> None:
        # The attention pattern done right for the tape model: the qkv projection's
        # view/unbind/contiguous live *inside* the SAVE op, so its outputs (not bare
        # views of them) are what flow downstream, and every consumer is a remat.region
        # that ferries them past the placeholders during recompute.
        class Rope(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any, q: torch.Tensor, k: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                ctx.save_for_backward(q, k)
                return q * 2, k * 3

            @staticmethod
            def backward(
                ctx: Any, grad_q: torch.Tensor, grad_k: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del ctx
                return grad_q * 2, grad_k * 3

        w = torch.randn(6, 12, dtype=torch.float64)
        base = torch.randn(2, 3, 6, dtype=torch.float64)

        def qkv_proj(
            u: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            qkv = (u @ w).view(2, 3, 3, 2, 2)
            q, k, v = qkv.unbind(dim=2)
            return q.contiguous(), k.contiguous(), v.contiguous()

        def attention(t: torch.Tensor) -> torch.Tensor:
            q, k, v = remat.region(qkv_proj, "qkv", recompute=False)(t)
            q, k = remat.region(Rope.apply, "rope", recompute=True)(q, k)
            return remat.region(
                lambda a, b, c: (a + b + c).sum(),
                "combine",
                recompute=True,
            )(q, k, v)

        x = base.clone().requires_grad_(True)
        remat.checkpoint(region_name="attn")(attention)(x).backward()
        self.assertTrue(torch.allclose(x.grad, _ref_grad(attention, base)))

    def test_save_op_detects_in_place_mutation_of_saved_tensor(self) -> None:
        # A SAVE op's saved tensor is packed through an identity saved_tensors_hooks
        # so autograd owns it -- but autograd's native version-counter guard does
        # NOT fire for tensors packed through custom hooks, so the SAVE-op unpack
        # hook records the version at save and re-checks it. Capture the saved
        # intermediate, mutate it in place after the forward, and confirm backward
        # raises rather than silently using stale data.
        saved_holder: list[torch.Tensor] = []

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                grad_factor = x * 2
                saved_holder.append(grad_factor)
                ctx.save_for_backward(grad_factor)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (grad_factor,) = ctx.saved_tensors
                return grad_output * grad_factor

        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(Square.apply, "sq", recompute=False)(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(body)(x)
        with torch.no_grad():
            saved_holder[0].mul_(5.0)  # invalidate the saved tensor after the forward
        with self.assertRaisesRegex(RuntimeError, "modified in-place"):
            y.sum().backward()

    def test_save_op_saving_input_mutated_in_place_is_retained(self) -> None:
        # A SAVE op that mutates a RECOMPUTE-sourced input in place and then saves it
        # (the fused in-place kernel shape: x.exp_(); mark_dirty; save_for_backward).
        # The saved tensor aliases the input by storage, but its data is no longer
        # what replay reproduces at op entry -- the version-counter guard in
        # _classify_saved_input must decline the ferry match and retain the
        # post-mutation value, or backward silently uses pre-mutation data.
        class InplaceExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                x.exp_()
                ctx.mark_dirty(x)
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                return grad_output * y

        def region(x: torch.Tensor) -> torch.Tensor:
            h = x * 1.0  # RECOMPUTE-produced tensor feeding the SAVE op
            y = remat.region(InplaceExp.apply, "exp", recompute=False)(h)
            return y * 1.0

        base = torch.randn(4, dtype=torch.float64)
        x = base.clone().requires_grad_(True)
        remat.checkpoint(region_name="r")(region)(x).sum().backward()
        self.assertTrue(torch.allclose(x.grad, _ref_grad(region, base)))

    def test_save_op_saves_distinct_object_alias_of_input_is_ferried(self) -> None:
        # Inputs are matched by storage, not Python object identity: a SAVE op that
        # saves ``y.view_as(y)`` -- a DISTINCT object aliasing a RECOMPUTE-sourced
        # input with identical layout -- is still recognized as that input and
        # ferried, not retained. This is the robustness the storage match buys: pack
        # may be handed a re-wrapped TensorImpl rather than the original input object.
        producer_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class Producer(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Producer.runs += 1
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                nonlocal producer_output_ref
                if not remat.is_recomputing():
                    producer_output_ref = weakref.ref(y)
                alias = y.view_as(y)  # distinct object, same storage + layout
                assert alias is not y
                ctx.save_for_backward(alias)
                return y * y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                return grad_output * 2 * y

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(Consumer.apply, "consumer", recompute=False)(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(region)(x)

        ref = producer_output_ref
        assert ref is not None
        gc.collect()
        # The aliased input was ferried (recognized by storage), not retained.
        self.assertIsNone(ref())

        out.sum().backward()
        self.assertEqual(2, Producer.runs)  # recomputed once at backward
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 36.0])))

    def test_save_op_saves_slice_views_of_recomputed_input_is_ferried(self) -> None:
        # A SAVE op that saves slice VIEWS of a RECOMPUTE-sourced input does not
        # retain them: the input's storage dies after the forward, and each view is
        # rebuilt from the reproduced base during recompute (here with a nonzero
        # relative storage offset for the second half).
        producer_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class Producer(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Producer.runs += 1
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                nonlocal producer_output_ref
                if not remat.is_recomputing():
                    producer_output_ref = weakref.ref(y)
                first = y[:2]  # view, storage offset 0
                second = y[2:]  # view, storage offset 2
                ctx.save_for_backward(first, second)
                return (y * y).sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                first, second = ctx.saved_tensors
                return grad_output * 2 * torch.cat([first, second])

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(Consumer.apply, "consumer", recompute=False)(y)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(region)(x)

        ref = producer_output_ref
        assert ref is not None
        gc.collect()
        # The views were ferried, so the producer output was not kept resident.
        self.assertIsNone(ref())

        out.backward()
        self.assertEqual(2, Producer.runs)
        # d/dx sum((3x)^2) = 18x
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 36.0, 54.0, 72.0])))

    def test_save_op_saved_views_survive_retain_graph(self) -> None:
        # A ferried saved view is rebuilt on each replay, so a second backward under
        # retain_graph=True recovers it again.
        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(y[:2], y[2:])
                return (y * y).sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                first, second = ctx.saved_tensors
                return grad_output * 2 * torch.cat([first, second])

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(Consumer.apply, "consumer", recompute=False)(y)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(region)(x)
        expected = torch.tensor([18.0, 36.0, 54.0, 72.0])

        out.backward(retain_graph=True)
        self.assertTrue(torch.equal(x.grad, expected))

        x.grad = None
        out.backward()
        self.assertTrue(torch.equal(x.grad, expected))

    def test_save_op_saved_view_of_noncontiguous_input_is_ferried(self) -> None:
        # A saved view of a NON-CONTIGUOUS input is still reconstructable: as_strided
        # reads the base's raw storage, so recompute rebuilds the view from the
        # reproduced base as long as it reproduces the same (shape, stride) layout --
        # contiguity is not required. So the views are ferried (not retained): the
        # non-contiguous input's storage is freed after the forward, and grads stay exact.
        base = torch.randn(2, 2, dtype=torch.float64)
        producer_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class Producer(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Producer.runs += 1
                ctx.save_for_backward(x)
                return (x * 3).t()  # non-contiguous output

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return (grad_output * 3).t()

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                nonlocal producer_output_ref
                if not remat.is_recomputing():
                    producer_output_ref = weakref.ref(y)
                # Save two row views of the non-contiguous base.
                ctx.save_for_backward(y[0], y[1])
                return (y * y).sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                row0, row1 = ctx.saved_tensors
                return grad_output * 2 * torch.stack([row0, row1])

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(Consumer.apply, "consumer", recompute=False)(y)

        def reference(x: torch.Tensor) -> torch.Tensor:
            return ((x * 3).t() ** 2).sum()

        x = base.clone().requires_grad_(True)
        out = remat.checkpoint(region_name="r")(region)(x)

        ref = producer_output_ref
        assert ref is not None
        gc.collect()
        # The views were ferried, so the non-contiguous input was not kept resident.
        self.assertIsNone(ref())

        out.backward()
        self.assertEqual(2, Producer.runs)
        self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, base)))

    def test_save_op_saved_view_layout_mismatch_on_recompute_errors(self) -> None:
        # If a ferried saved view's base reproduces with a DIFFERENT layout than the
        # forward recorded (here forced by a producer that returns a non-contiguous
        # base only on recompute), the view cannot be rebuilt -- the reconstruction
        # guard raises a clear error rather than reading the wrong elements.
        class DriftingProducer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                if remat.is_recomputing():
                    # Same values (x*3), same shape, but non-contiguous layout.
                    return torch.repeat_interleave(x * 3, 2)[::2]
                return x * 3  # contiguous on the original forward

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(y[:1])  # a view -> ferried (base is contiguous)
                return (y * y).sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (head,) = ctx.saved_tensors
                return grad_output * 2 * torch.cat([head, head])

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(
                DriftingProducer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(Consumer.apply, "consumer", recompute=False)(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(region)(x)
        with self.assertRaisesRegex(RuntimeError, "different layout"):
            out.backward()

    def test_nested_save_region_inside_save_runs_inert(self) -> None:
        # A remat.region nested inside an enclosing SAVE (recompute=False) region runs
        # inert: its saves ride the enclosing region's hooks. This is the MoE dedup
        # dispatch pattern -- an inner SAVE (the grouped GEMM) consuming a bare op's
        # output (the dispatched tokens) produced *inside* the outer SAVE. If the inner
        # region were not inert it would record a rederive recipe for that input and
        # then fail at backward ("No saved input ... different code paths"): the
        # enclosing SAVE skips the bare producer on recompute, so the recipe is never
        # filled.
        inner_forward_runs = [0]

        class InnerSquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if remat.is_recomputing():
                    raise AssertionError("enclosing SAVE must skip the nested op")
                inner_forward_runs[0] += 1
                ctx.save_for_backward(x)  # a bare-op output inside the outer SAVE
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def outer_body(x: torch.Tensor) -> torch.Tensor:
            dispatched = x * 3  # a bare (recompute-by-default) op inside the outer SAVE
            squared = remat.region(InnerSquare.apply, "inner", recompute=False)(
                dispatched
            )
            return squared + 1  # a trailing bare op (the "combine" analog)

        base = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x = base.clone().requires_grad_(True)
        y = remat.checkpoint(region_name="outer_ckpt")(
            lambda t: remat.region(outer_body, "outer", recompute=False)(t)
        )(x)
        y.sum().backward()

        # Inert: the inner op ran once on the forward and never re-ran on recompute
        # (the enclosing SAVE region is skipped wholesale).
        self.assertEqual(1, inner_forward_runs[0])
        # d/dx ((3x)^2 + 1) = 18x
        self.assertTrue(
            torch.allclose(x.grad, _ref_grad(lambda t: (t * 3) ** 2 + 1, base))
        )

    def test_recompute_region_inside_save_region_raises(self) -> None:
        # A recompute=True region nested inside a SAVE (recompute=False) region cannot
        # be honored -- the enclosing SAVE never recomputes -- so it is rejected.
        def outer_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(lambda t: t * 2, "inner", recompute=True)(x)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "recompute=True.*nested inside a recompute=False"
        ):
            remat.checkpoint(region_name="outer_ckpt")(
                lambda t: remat.region(outer_body, "outer", recompute=False)(t)
            )(x)

    def test_save_output_register_hook_fires_with_grad(self) -> None:
        # A backward hook registered on a SAVE region's output must fire with the
        # gradient w.r.t. that output when the output is consumed by a remat.region.
        # SAVE outputs are plain tensors, so the hook rides the output's own grad_fn
        # normally. Regression: activation-gradient (.dx) metrics that register_hook on
        # a SAVE output were silently dropped (nan) under whole-layer remat.
        captured: list[torch.Tensor] = []

        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(Producer.apply, "producer", recompute=False)(x)
            y.register_hook(lambda g: captured.append(g.detach().clone()) or g)
            # Consume via a remat.region.
            return remat.region(lambda t: t * 2, "consumer", recompute=True)(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        remat.checkpoint(region_name="r")(body)(x).sum().backward()

        # The hook fired once, with the gradient w.r.t. the SAVE output y = 3x:
        # d(sum(2 * y))/dy = 2.
        self.assertEqual(1, len(captured))
        self.assertTrue(torch.equal(captured[0], torch.tensor([2.0, 2.0])))
        # End-to-end gradient is still correct: d(sum(2 * 3x))/dx = 6.
        self.assertTrue(torch.equal(x.grad, torch.tensor([6.0, 6.0])))
