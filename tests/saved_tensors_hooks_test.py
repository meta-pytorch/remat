# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``remat.saved_tensors_hooks``: pack/unpack fire at save and load but not on
recompute, unpack is bound per tensor, and two worked activation-offload engines -- a
fine-grained per-tensor wedge and a coarse per-group bulk offloader -- round-trip a
SAVE -> RECOMPUTE -> SAVE model bitwise-identically. The offloader machinery lives in
``remat_test_helpers``."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any, cast

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import (
    _BulkOffloader,
    _run_bulk_model,
    _run_wedge_model,
    _WedgeOffloader,
)
from torch_remat._api import _active_saved_tensors_hooks


class SavedTensorsHooksTest(expecttest.TestCase):
    def test_saved_tensors_hooks_match_native_outside_checkpoint(self) -> None:
        def run(
            context: contextlib.AbstractContextManager[None],
            packed_shapes: list[tuple[int, ...]],
            unpacked_shapes: list[tuple[int, ...]],
        ) -> torch.Tensor:
            x = torch.tensor([2.0, 3.0], requires_grad=True)
            self.assertIsNone(_active_saved_tensors_hooks.get())
            with context:
                self.assertIsNone(_active_saved_tensors_hooks.get())
                y = x * x
            y.sum().backward()
            self.assertIsNotNone(x.grad)
            return cast(torch.Tensor, x.grad)

        native_packed: list[tuple[int, ...]] = []
        native_unpacked: list[tuple[int, ...]] = []
        remat_packed: list[tuple[int, ...]] = []
        remat_unpacked: list[tuple[int, ...]] = []

        def hooks(
            packed: list[tuple[int, ...]],
            unpacked: list[tuple[int, ...]],
        ) -> tuple[Callable[[torch.Tensor], object], Callable[[object], torch.Tensor]]:
            def pack(tensor: torch.Tensor) -> object:
                packed.append(tuple(tensor.shape))
                return tensor

            def unpack(payload: object) -> torch.Tensor:
                tensor = cast(torch.Tensor, payload)
                unpacked.append(tuple(tensor.shape))
                return tensor

            return pack, unpack

        native_pair = hooks(native_packed, native_unpacked)
        native_grad = run(
            torch.autograd.graph.saved_tensors_hooks(*native_pair),
            native_packed,
            native_unpacked,
        )
        remat_pair = hooks(remat_packed, remat_unpacked)
        remat_grad = run(
            remat.saved_tensors_hooks(*remat_pair),
            remat_packed,
            remat_unpacked,
        )

        self.assertEqual(native_packed, remat_packed)
        self.assertEqual(native_unpacked, remat_unpacked)
        self.assertTrue(torch.equal(native_grad, remat_grad))

    def test_saved_tensors_hooks_expose_native_save_metadata(self) -> None:
        kinds: list[remat.SavedTensorKind] = []
        contexts: list[object] = []
        capture_calls = [0]

        def capture_context() -> object:
            capture_calls[0] += 1
            return "captured"

        def pack(tensor: torch.Tensor) -> object:
            info = remat.current_saved_tensor_info()
            kinds.append(info.kind)
            contexts.append(info.context)
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with remat.saved_tensors_hooks(
            pack,
            unpack,
            capture_context=capture_context,
        ):
            y = x * x
        y.sum().backward()

        self.assertEqual(
            [remat.SavedTensorKind.BACKWARD, remat.SavedTensorKind.BACKWARD],
            kinds,
        )
        self.assertEqual([None, None], contexts)
        self.assertEqual(0, capture_calls[0])
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_raw_push_captures_context_for_native_saves(self) -> None:
        contexts: list[object] = []

        def pack(tensor: torch.Tensor) -> object:
            contexts.append(remat.current_saved_tensor_info().context)
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        remat._push_saved_tensors_hooks(
            pack,
            unpack,
            capture_context=lambda: "native-context",
        )
        try:
            x = torch.tensor([2.0, 3.0], requires_grad=True)
            y = x * x
        finally:
            remat._pop_saved_tensors_hooks()
        y.sum().backward()

        self.assertEqual(["native-context", "native-context"], contexts)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_fire_at_save_and_load_not_recompute(
        self,
    ) -> None:
        # Inside a checkpoint, the wrapper preserves checkpoint's native hooks while
        # delegating the SAVE region's retained tensors through the user pair.
        pack_shapes: list[tuple[int, ...]] = []
        pack_kinds: list[remat.SavedTensorKind] = []
        unpack_tags: list[str] = []
        stash: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> object:
            pack_shapes.append(tuple(tensor.shape))
            pack_kinds.append(remat.current_saved_tensor_info().kind)
            index = len(stash)
            stash.append(tensor.detach().clone())
            return ("stashed", index)

        def unpack(packed: object) -> torch.Tensor:
            tag, index = cast(tuple[str, int], packed)
            unpack_tags.append(tag)
            return stash[index]

        class Square(torch.autograd.Function):
            # Saves two *internal* tensors, not the input x: a SAVE op's saved input
            # is diverted off the offload hook and recomputed instead (case A), so
            # only internally-produced saves exercise the hooks here.
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                two_x = x * 2
                y = x * x
                ctx.save_for_backward(two_x, y)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (two_x, y) = ctx.saved_tensors
                del y
                return grad_output * two_x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            with remat.saved_tensors_hooks(pack, unpack):
                return remat.region(
                    Square.apply,
                    "sq",
                    recompute=False,
                )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(2, len(pack_shapes))
        self.assertEqual([remat.SavedTensorKind.BACKWARD] * 2, pack_kinds)
        self.assertEqual(["stashed", "stashed"], unpack_tags)
        # The custom pack/unpack round-trip leaves the gradient unchanged.
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_unpack_is_bound_per_tensor(self) -> None:
        # The unpack hook used at load time must be the one bound when the tensor
        # was packed, even if a different pack/unpack pair is active later (here:
        # no hooks at all during backward, because the with-block has exited).
        unpack_calls: list[str] = []
        stash: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> object:
            index = len(stash)
            stash.append(tensor.detach().clone())
            return index

        def unpack(packed: object) -> torch.Tensor:
            unpack_calls.append("bound")
            return stash[cast(int, packed)]

        class Square(torch.autograd.Function):
            # Saves two *internal* tensors, not the input x: a SAVE op's saved input
            # is diverted off the offload hook and recomputed instead (case A), so
            # only internally-produced saves exercise the hooks here.
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                two_x = x * 2
                y = x * x
                ctx.save_for_backward(two_x, y)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (two_x, y) = ctx.saved_tensors
                del y
                return grad_output * two_x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(
                Square.apply,
                "sq",
                recompute=False,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        # Pack inside the hook scope; recompute/backward runs OUTSIDE it.
        with remat.saved_tensors_hooks(pack, unpack):
            y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        # unpack still ran (bound at pack time) despite no active hook at backward.
        self.assertEqual(["bound", "bound", "bound"], unpack_calls)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_offload_through_save_recompute_save_wedge(
        self,
    ) -> None:
        # The shape that actually motivated remat-level saved-tensor hooks, in
        # miniature for readers without access to a real offloader. The model is
        # two "blocks" (each a checkpoint region == one offload group), and inside
        # every block a SAVE -> RECOMPUTE -> SAVE wedge of ops on x = [1.0, 2.0]:
        #
        #   block.k:  in = Sq[SAVE]  ->  mid = Relu[RECOMPUTE]  ->  out = Sq[SAVE]
        #
        # where Sq(t) = t*t (saves the gradient factor 2*t) and Relu(t) = relu(t)
        # (saves its 0/1 mask). A toy offload engine is wired in via each
        # checkpoint's saved_tensors_hooks kwarg. It passes through checkpoint
        # inputs and offloads the SAVE tensors. Each
        # packed value binds its own unpack hook (a SAVE save's autograd payload and
        # a ferried input's tape slot both capture it at pack), so -- unlike
        # autograd's blanket default hooks, which a real engine must install
        # per-layer INSIDE the checkpoint -- a single install suffices. Each block's
        # offloaded storage is freed a block late (the next block's commit frees the
        # previous group) and reloaded in backward.
        #
        # Working the forward by hand (x = [1, 2], so relu is the identity here):
        #   block.0.in  Sq[SAVE]:   y0 = x*x = [1, 4]   -> offload: 2*x = [2, 4]
        #                           (gf), and y0 = [1, 4] (its output feeds the
        #                           RECOMPUTE op, ferried so recompute has it)
        #   block.0.mid Relu[RECOMPUTE]: relu([1,4]) = [1, 4]; its mask is NOT
        #                           offloaded (it is regenerated in backward)
        #   block.0.out Sq[SAVE]:   y1 = [1, 16] -> offload: 2*[1,4] = [2, 8] (gf
        #                           only; the block output is not offloaded)
        #   block.1 repeats on [1, 16] -> ... -> [1, 256] -> [1, 65536]
        #
        # So exactly the SAVE ops reach the offloader (four grad_factors + the two
        # *.in outputs); nothing named *.mid is ever packed. Rather than assert
        # those facts piecemeal, we capture the engine's event trace and pin it
        # whole: the absence of any "pack ... *.mid" line IS the proof that
        # RECOMPUTE intermediates never reach the offloader, and the compute lines
        # show SAVE bodies are skipped on recompute while RECOMPUTE bodies rerun.
        #
        # Read the trace below as the worked execution. In the forward, every
        # pack line names a SAVE tensor (*.gf or the *.in output) -- there is no
        # "pack ... *.mid" line, which IS the proof that the RECOMPUTE op's saved
        # tensors never reach the offloader. Each block's tensors are freed a block
        # late (commit block.1 frees block.0's t0..t2; flush frees block.1's).
        # In the backward, only the *.mid (RECOMPUTE) compute lines reappear --
        # the SAVE bodies are not rerun -- and every freed activation is restored
        # through unpack before it is read, in the reverse, recompute-driven order.
        base_loss, base_grad, _ = _run_wedge_model(None)

        offloader = _WedgeOffloader()
        off_loss, off_grad, trace = _run_wedge_model(offloader)

        self.assertExpectedInline(
            trace,
            """\
== forward ==
compute block.0.in [save]
  pack t0 = block.0.in.gf
  pack t1 = block.0.in.y
compute block.0.mid [recompute]
compute block.0.out [save]
  pack t2 = block.0.out.gf
  commit block.0: free []
compute block.1.in [save]
  pack t3 = block.1.in.gf
  pack t4 = block.1.in.y
compute block.1.mid [recompute]
compute block.1.out [save]
  pack t5 = block.1.out.gf
  commit block.1: free [t0, t1, t2]
  flush: free [t3, t4, t5]
== backward ==
  unpack t4 = block.1.in.y
compute block.1.mid [recompute] (recompute)
  unpack t5 = block.1.out.gf
  unpack t3 = block.1.in.gf
  unpack t1 = block.0.in.y
compute block.0.mid [recompute] (recompute)
  unpack t2 = block.0.out.gf
  unpack t0 = block.0.in.gf""",
        )

        # Every offloaded activation's storage was freed before backward ran, yet
        # the recompute above reloaded each one through unpack, so the offloaded
        # run is bitwise-identical to the no-offload baseline.
        self.assertTrue(
            all(t.untyped_storage().nbytes() == 0 for t in offloader.originals)
        )
        self.assertTrue(torch.equal(base_loss, off_loss))
        self.assertTrue(torch.equal(base_grad, off_grad))

    def test_saved_tensors_hooks_bulk_offload_group_onloads_at_recompute_start(
        self,
    ) -> None:
        # The same SAVE -> RECOMPUTE -> SAVE wedge as the test above, but driven
        # by a *bulk* offloader instead of the fine-grained wedge one. The point
        # is to exercise the coarse, batched shape a real activation-offload
        # engine actually uses -- one device<->host copy per layer group rather
        # than per tensor -- and, crucially, behavioral difference #1 of
        # remat.saved_tensors_hooks: because a SAVE *output* saved for a later
        # RECOMPUTE region is unpacked during the recompute phase (before
        # backward starts), the group's onload has to be triggered from the top
        # of the region body under remat.is_recomputing(), NOT from an autograd
        # function (which would run too late for that first unpack).
        #
        # Read the trace as: pack merely records each SAVE tensor into its group;
        # after each block's forward, one "offload block.k: D2H ..." line flushes
        # the whole group to host and frees the device storages; in backward, each
        # block's recompute opens with a single "onload block.k: H2D ..." (fired
        # by the region body before any unpack for that group), after which the
        # group's tensors are served straight from the reloaded copies. As in the
        # wedge test, no "pack ... *.mid" line appears -- the RECOMPUTE op's saved
        # mask never reaches the offloader -- and the SAVE bodies do not recompute.
        base_loss, base_grad, _ = _run_bulk_model(None)

        offloader = _BulkOffloader()
        off_loss, off_grad, trace = _run_bulk_model(offloader)

        self.assertExpectedInline(
            trace,
            """\
== forward ==
compute block.0.in [save]
  pack block.0.in.gf
  pack block.0.in.y
compute block.0.mid [recompute]
compute block.0.out [save]
  pack block.0.out.gf
offload block.0: D2H 3 tensors, free device
compute block.1.in [save]
  pack block.1.in.gf
  pack block.1.in.y
compute block.1.mid [recompute]
compute block.1.out [save]
  pack block.1.out.gf
offload block.1: D2H 3 tensors, free device
== backward ==
onload block.1: H2D 3 tensors
  unpack block.1.in.y
compute block.1.mid [recompute] (recompute)
  unpack block.1.out.gf
  unpack block.1.in.gf
onload block.0: H2D 3 tensors
  unpack block.0.in.y
compute block.0.mid [recompute] (recompute)
  unpack block.0.out.gf
  unpack block.0.in.gf""",
        )

        # Every group was flushed to host and its device storages freed before
        # backward ran, yet the per-group onloads restored each tensor in time,
        # so the bulk-offloaded run is bitwise-identical to the baseline.
        self.assertTrue(
            all(t.untyped_storage().nbytes() == 0 for t in offloader.all_originals())
        )
        self.assertTrue(torch.equal(base_loss, off_loss))
        self.assertTrue(torch.equal(base_grad, off_grad))

    def test_checkpoint_saved_tensors_hooks_include_inputs_and_save_tensors(
        self,
    ) -> None:
        outer_pack_calls = [0]
        pack_kinds: list[remat.SavedTensorKind] = []
        unpack_kinds: list[remat.SavedTensorKind] = []
        stash: list[torch.Tensor] = []

        def outer_pack(tensor: torch.Tensor) -> object:
            outer_pack_calls[0] += 1
            return tensor

        def outer_unpack(packed: object) -> torch.Tensor:
            return cast(torch.Tensor, packed)

        def pack(tensor: torch.Tensor) -> object:
            kind = remat.current_saved_tensor_info().kind
            pack_kinds.append(kind)
            stash.append(tensor.detach().clone())
            return (len(stash) - 1, kind)

        def unpack(packed: object) -> torch.Tensor:
            index, kind = cast(tuple[int, remat.SavedTensorKind], packed)
            unpack_kinds.append(kind)
            return stash[index]

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                two_x = x * 2
                ctx.save_for_backward(two_x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (two_x,) = ctx.saved_tensors
                return grad_output * two_x

        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(Square.apply, "sq", recompute=False)(x)

        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        region_input = leaf * 1.0  # non-leaf, so it is a real saved input
        with remat.saved_tensors_hooks(outer_pack, outer_unpack):
            y = remat.checkpoint(saved_tensors_hooks=(pack, unpack))(body)(region_input)
        y.sum().backward()

        self.assertEqual(0, outer_pack_calls[0])
        self.assertCountEqual(
            [
                remat.SavedTensorKind.CHECKPOINT_INPUT,
                remat.SavedTensorKind.BACKWARD,
            ],
            pack_kinds,
        )
        self.assertCountEqual(
            [
                remat.SavedTensorKind.CHECKPOINT_INPUT,
                remat.SavedTensorKind.BACKWARD,
            ],
            unpack_kinds,
        )
        self.assertTrue(torch.equal(leaf.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_entered_inside_save_body_take_precedence(
        self,
    ) -> None:
        outer_packs = [0]
        inner_packs = [0]

        def outer_pack(tensor: torch.Tensor) -> object:
            outer_packs[0] += 1
            return tensor

        def inner_pack(tensor: torch.Tensor) -> object:
            inner_packs[0] += 1
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                doubled = x * 2
                ctx.save_for_backward(doubled)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (doubled,) = ctx.saved_tensors
                return grad_output * doubled

        def save_body(x: torch.Tensor) -> torch.Tensor:
            with remat.saved_tensors_hooks(inner_pack, unpack):
                return Square.apply(x)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(save_body, "square", recompute=False)(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with remat.saved_tensors_hooks(outer_pack, unpack):
            y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(1, outer_packs[0])
        self.assertEqual(1, inner_packs[0])
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_deprecated_input_hooks_remain_input_only(self) -> None:
        packed: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> object:
            packed.append(tensor)
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                doubled = x * 2
                ctx.save_for_backward(doubled)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (doubled,) = ctx.saved_tensors
                return grad_output * doubled

        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(Square.apply, "square", recompute=False)(x)

        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        region_input = leaf * 1.0
        y = remat.checkpoint(input_saved_tensors_hooks=(pack, unpack))(body)(
            region_input
        )
        y.sum().backward()

        self.assertEqual(1, len(packed))
        self.assertTrue(torch.equal(leaf.grad, torch.tensor([4.0, 6.0])))

    def test_deprecated_input_hooks_preserve_outer_native_hooks(self) -> None:
        input_packed: list[torch.Tensor] = []
        outer_packed: list[torch.Tensor] = []

        def input_pack(tensor: torch.Tensor) -> object:
            input_packed.append(tensor)
            return tensor

        def outer_pack(tensor: torch.Tensor) -> object:
            outer_packed.append(tensor)
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                doubled = x * 2
                ctx.save_for_backward(doubled)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (doubled,) = ctx.saved_tensors
                return grad_output * doubled

        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(Square.apply, "square", recompute=False)(x)

        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        region_input = leaf * 1.0
        with torch.autograd.graph.saved_tensors_hooks(outer_pack, unpack):
            y = remat.checkpoint(input_saved_tensors_hooks=(input_pack, unpack))(body)(
                region_input
            )
        y.sum().backward()

        self.assertEqual(1, len(input_packed))
        self.assertEqual(1, len(outer_packed))
        self.assertTrue(torch.equal(leaf.grad, torch.tensor([4.0, 6.0])))

    def test_checkpoint_hook_options_are_mutually_exclusive(self) -> None:
        def pack(tensor: torch.Tensor) -> object:
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            remat.checkpoint(
                saved_tensors_hooks=(pack, unpack),
                input_saved_tensors_hooks=(pack, unpack),
            )

    def test_nested_native_hooks_take_precedence(self) -> None:
        outer_packs = [0]
        inner_packs = [0]

        def outer_pack(tensor: torch.Tensor) -> object:
            outer_packs[0] += 1
            return tensor

        def inner_pack(tensor: torch.Tensor) -> object:
            inner_packs[0] += 1
            return tensor

        def unpack(payload: object) -> torch.Tensor:
            return cast(torch.Tensor, payload)

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                doubled = x * 2
                ctx.save_for_backward(doubled)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (doubled,) = ctx.saved_tensors
                return grad_output * doubled

        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(Square.apply, "square", recompute=False)(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with remat.saved_tensors_hooks(outer_pack, unpack):
            with torch.autograd.graph.saved_tensors_hooks(inner_pack, unpack):
                y = remat.checkpoint()(body)(x)
        y.sum().backward()

        self.assertEqual(0, outer_packs[0])
        self.assertEqual(2, inner_packs[0])
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_retain_parameter_ness(self) -> None:
        # A SAVE op that saves an nn.Parameter for backward (e.g. a linear weight for
        # its wgrad) must hand that tensor to the user pack hook AS an nn.Parameter.
        # remat detaches saved tensors before the hook (to break the Node<->payload
        # cycle); a plain detach() drops the Parameter type, so an activation-offload
        # engine that skips FSDP-managed weights via isinstance(t, nn.Parameter) would
        # mistake the unsharded weight for an activation, offload it, and race FSDP's
        # reshard-after-forward that frees its storage. The saved non-parameter
        # activation must still arrive as a plain Tensor.
        packed_is_param: list[bool] = []

        def pack(tensor: torch.Tensor) -> object:
            packed_is_param.append(isinstance(tensor, torch.nn.Parameter))
            return tensor

        def unpack(packed: object) -> torch.Tensor:
            return cast(torch.Tensor, packed)

        # Mirror the real attn qkv_proj SAVE op: a custom autograd Function whose
        # forward does ctx.save_for_backward(activation, weight) with the weight an
        # nn.Parameter captured from the enclosing module (not a region input).
        # An internal activation is saved alongside it as a plain-Tensor control.
        weight = torch.nn.Parameter(torch.randn(3, 3))

        class Linear(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx: Any, x: torch.Tensor, w: torch.nn.Parameter
            ) -> torch.Tensor:
                act = x * 2  # an internal, non-parameter save
                ctx.save_for_backward(act, w)
                return act @ w.t()

            @staticmethod
            def backward(
                ctx: Any, grad_output: torch.Tensor
            ) -> tuple[torch.Tensor, None]:
                (act, w) = ctx.saved_tensors
                del act
                return (grad_output @ w) * 2, None

        def body(x: torch.Tensor) -> torch.Tensor:
            def fn(inp: torch.Tensor) -> torch.Tensor:
                return Linear.apply(inp, weight)

            return remat.region(fn, "linear", recompute=False)(x)

        x = torch.randn(4, 3, requires_grad=True)
        with remat.saved_tensors_hooks(pack, unpack):
            y = remat.checkpoint()(body)(x)
            y.sum().backward()

        # The native checkpoint-input save arrives first, followed by the activation
        # and parameter retained by the SAVE region.
        self.assertEqual([False, False, True], packed_is_param)
        # Grad still flows through the parameter-wrapped saved weight.
        self.assertIsNotNone(x.grad)

    def test_capture_context_binds_producer_context_for_deferred_output_save(
        self,
    ) -> None:
        # capture_context runs in-window -- where the SAVE output is produced -- and its
        # result is exposed through current_saved_tensor_info. A SAVE output whose pack is
        # DEFERRED to a bare consumer (fired by remat.recompute_needs_tensor after the hook
        # scope has exited) must still pack against the context captured at the producer,
        # not what is live at the consumer. This is the offload case: the pack ("which
        # chunk") binds the producer's context even though it runs late.
        packed_with: list[int] = []
        packed_kinds: list[remat.SavedTensorKind] = []
        # A mutable "current chunk id" the model advances after the region, mimicking an
        # offloader whose per-region context is gone by the time the consumer fires.
        current_chunk = {"id": 7}

        def capture_context() -> int:
            return current_chunk["id"]

        def pack(tensor: torch.Tensor) -> object:
            info = remat.current_saved_tensor_info()
            packed_with.append(cast(int, info.context))
            packed_kinds.append(info.kind)
            return (tensor.detach(), info.context)

        def unpack(payload: object) -> torch.Tensor:
            tensor, _chunk_id = cast("tuple[torch.Tensor, int]", payload)
            return tensor

        def body(x: torch.Tensor) -> torch.Tensor:
            with remat.saved_tensors_hooks(
                pack, unpack, capture_context=capture_context
            ):
                # `mul` saves nothing for backward, so its output is kept only if a consumer
                # asks -- i.e. the pack is deferred, not eager at region exit.
                y = remat.region(lambda t: t * 2, "mul", recompute=False)(x)
            # The producer's hook scope has exited and its context has moved on...
            current_chunk["id"] = 999
            # ...but this deferred save must still pack against chunk 7, captured in-window.
            remat.recompute_needs_tensor(y)
            return torch.relu(y)  # bare consumer, reads y during recompute

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        remat.checkpoint()(body)(x).sum().backward()
        # Packed once, with the context captured at the producer (7), not the later 999.
        self.assertEqual([7], packed_with)
        # ...and pack was told this entry is a persisted SAVE-region output, not a
        # save-for-backward tensor, so a policy hook can treat it differently.
        self.assertEqual([remat.SavedTensorKind.SAVE_OUTPUT], packed_kinds)
        # d/dx relu(2x) = 2 where 2x > 0.
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))
