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


class SavedTensorsHooksTest(expecttest.TestCase):
    def test_saved_tensors_hooks_fire_at_save_and_load_not_recompute(
        self,
    ) -> None:
        # A SAVE-policy op routes its saved tensors through the active remat
        # saved_tensors_hooks. pack must fire once per saved tensor in the original
        # forward, unpack once per saved tensor when backward reads them back, and
        # pack must NOT fire again during recompute. A custom pack/unpack pair that
        # round-trips through a Python list must leave gradients unchanged.
        pack_shapes: list[tuple[int, ...]] = []
        unpack_tags: list[str] = []
        stash: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> object:
            pack_shapes.append(tuple(tensor.shape))
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
            return remat.op(
                Square.apply,
                "sq",
                policy=remat.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with remat.saved_tensors_hooks(pack, unpack):
            y = remat.checkpoint()(checkpoint_body)(x)
            y.sum().backward()

        # pack fires for x and y in the original forward only (not on recompute).
        self.assertEqual(2, len(pack_shapes))
        # unpack fires for x and y when backward reads them back.
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
            return remat.op(
                Square.apply,
                "sq",
                policy=remat.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        # Pack inside the hook scope; recompute/backward runs OUTSIDE it.
        with remat.saved_tensors_hooks(pack, unpack):
            y = remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        # unpack still ran (bound at pack time) despite no active hook at backward.
        self.assertEqual(["bound", "bound"], unpack_calls)
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
        # (saves its 0/1 mask). A toy offload engine is wired in via remat
        # saved_tensors_hooks; we install ONE scope around the whole forward. Each
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
compute block.0.in [SAVE]
  pack t0 = block.0.in.gf
  pack t1 = block.0.in.y
compute block.0.mid [RECOMPUTE]
compute block.0.out [SAVE]
  pack t2 = block.0.out.gf
  commit block.0: free []
compute block.1.in [SAVE]
  pack t3 = block.1.in.gf
  pack t4 = block.1.in.y
compute block.1.mid [RECOMPUTE]
compute block.1.out [SAVE]
  pack t5 = block.1.out.gf
  commit block.1: free [t0, t1, t2]
  flush: free [t3, t4, t5]
== backward ==
  unpack t4 = block.1.in.y
compute block.1.mid [RECOMPUTE] (recompute)
  unpack t5 = block.1.out.gf
  unpack t3 = block.1.in.gf
  unpack t1 = block.0.in.y
compute block.0.mid [RECOMPUTE] (recompute)
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
compute block.0.in [SAVE]
  pack block.0.in.gf
  pack block.0.in.y
compute block.0.mid [RECOMPUTE]
compute block.0.out [SAVE]
  pack block.0.out.gf
offload block.0: D2H 3 tensors, free device
compute block.1.in [SAVE]
  pack block.1.in.gf
  pack block.1.in.y
compute block.1.mid [RECOMPUTE]
compute block.1.out [SAVE]
  pack block.1.out.gf
offload block.1: D2H 3 tensors, free device
== backward ==
onload block.1: H2D 3 tensors
  unpack block.1.in.y
compute block.1.mid [RECOMPUTE] (recompute)
  unpack block.1.out.gf
  unpack block.1.in.gf
onload block.0: H2D 3 tensors
  unpack block.0.in.y
compute block.0.mid [RECOMPUTE] (recompute)
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

    def test_input_saved_tensors_hooks_fire_only_for_region_inputs(self) -> None:
        # checkpoint(input_saved_tensors_hooks=...) installs autograd saved_tensors_hooks
        # around the region. torch.utils.checkpoint saves the region INPUTS as autograd
        # SavedTensors at entry (via _make_saved_tensor), before it installs its own
        # recompute hook; the body's saves then run under that shadowing hook. So the pair
        # fires exactly once -- for the region input -- and not for the body's
        # save_for_backward. The round-trip must leave gradients unchanged.
        input_pack_shapes: list[tuple[int, ...]] = []
        input_unpacks = [0]
        stash: list[torch.Tensor] = []

        def pack(tensor: torch.Tensor) -> object:
            input_pack_shapes.append(tuple(tensor.shape))
            stash.append(tensor.detach().clone())
            return len(stash) - 1

        def unpack(packed: object) -> torch.Tensor:
            input_unpacks[0] += 1
            return stash[cast(int, packed)]

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(
                    x
                )  # a BODY save -- must NOT reach the input hooks
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.op(Square.apply, "sq", policy=remat.RECOMPUTE)(x)

        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        region_input = leaf * 1.0  # non-leaf, so it is a real saved input
        y = remat.checkpoint(input_saved_tensors_hooks=(pack, unpack))(body)(
            region_input
        )
        y.sum().backward()

        # pack fired exactly once -- for the region input, not Square's save_for_backward.
        self.assertEqual([(2,)], input_pack_shapes)
        # unpack fired once, when recompute reloaded the input.
        self.assertEqual(1, input_unpacks[0])
        # grad of sum(x^2) is 2x -- the round-tripped input fed forward and recompute.
        self.assertTrue(torch.equal(leaf.grad, torch.tensor([4.0, 6.0])))
