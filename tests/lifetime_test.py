# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for retention and GC semantics across policy crossings: a RECOMPUTE op retains
none of its original saved tensors, a saved-own-output is freed without gc, and the
RECOMPUTE/SAVE producer-consumer chains keep or drop the producer output exactly as
the tape model requires -- plus dead SAVE outputs are freed and SAVE saved tensors
live on autograd, not the tape."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import _BARE_OP_STRATEGIES, assert_reclaimed_without_gc
from torch_remat._region import (
    _checkpoint_context_fn,
    _state,
)


class LifetimeTest(expecttest.TestCase):
    def test_recompute_policy_does_not_retain_original_saved_tensors_after_forward(
        self,
    ) -> None:
        original_saved_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class SavedTensorLifetimeProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                nonlocal original_saved_ref

                saved_activation = x + 1
                if not remat.is_recomputing():
                    original_saved_ref = weakref.ref(saved_activation)
                ctx.save_for_backward(saved_activation)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (saved_activation,) = ctx.saved_tensors
                del saved_activation
                return grad_output * 2

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return remat.op(
                SavedTensorLifetimeProbe.apply,
                "saved.tensor.lifetime",
                policy=remat.RECOMPUTE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = remat.checkpoint()(checkpoint_body)(x)

        saved_ref = original_saved_ref
        self.assertIsNotNone(saved_ref)
        assert saved_ref is not None
        gc.collect()
        self.assertIsNone(saved_ref())

        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 2.0])))

    def test_save_output_for_backward_freed_without_gc(self) -> None:
        # A SAVE op that saves one of its OWN OUTPUTS for backward. The identity SAVE
        # hook detaches on pack (see _default_pack), so autograd's SavedVariable
        # holds a grad_fn-less copy sharing storage -- NOT the live output. Handing it the
        # live output would pin the grad_fn-bearing tensor and close a C++ Node <->
        # TensorImpl refcount cycle (through the hook payload) that not even Python's
        # cyclic gc can reclaim, leaking the output whenever the graph is dropped without
        # a backward. Detaching severs that cycle, so dropping the sole reference to the
        # graph reclaims the output promptly by refcounting alone -- no gc, no backward.
        saved_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class SavesOwnOutput(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                nonlocal saved_output_ref
                y = x * 2
                saved_output_ref = weakref.ref(y)
                ctx.save_for_backward(y)  # its own output -> cycle under identity hook
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                del y
                return grad_output * 2

        def region(x: torch.Tensor) -> torch.Tensor:
            return remat.op(
                SavesOwnOutput.apply, "saves.own.output", policy=remat.SAVE
            )(x)

        was_enabled = gc.isenabled()
        gc.disable()  # isolate refcount reclamation from cyclic collection
        try:
            y = remat.checkpoint(detect_bare_ops=False)(region)(
                torch.tensor([1.0, 2.0], requires_grad=True)
            )
            ref = saved_output_ref
            assert ref is not None
            self.assertIsNotNone(ref())  # graph alive -> saved output still resident

            # Drop the sole owning reference to the autograd graph; no backward ran.
            del y
            # Refcounting alone should reclaim the saved output immediately.
            self.assertIsNone(ref())
        finally:
            if was_enabled:
                gc.enable()

    def test_save_output_through_user_identity_hook_freed_without_gc(self) -> None:
        # The same C++ Node <-> payload cycle as the test above, but reached through a
        # *user* remat.saved_tensors_hooks whose pack is the identity. remat detaches the
        # tensor before handing it to the user pack hook (like _default_pack), so even an
        # identity pack -- whose payload IS the returned tensor -- holds a grad_fn-less
        # copy, not the live output. Drop the detach and the payload pins the output's
        # grad_fn, closing the gc-invisible cycle so the graph leaks when dropped without
        # a backward. This is the regression guard for that detach in the user-hook path.
        def make_graph() -> tuple[object, weakref.ReferenceType[torch.Tensor]]:
            saved_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

            class SavesOwnOutput(torch.autograd.Function):
                @staticmethod
                def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                    nonlocal saved_output_ref
                    y = x * 2
                    saved_output_ref = weakref.ref(y)
                    ctx.save_for_backward(y)  # own output -> cycle under identity hook
                    return y

                @staticmethod
                def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                    (y,) = ctx.saved_tensors
                    del y
                    return grad_output * 2

            def region(x: torch.Tensor) -> torch.Tensor:
                return remat.op(
                    SavesOwnOutput.apply, "saves.own.output", policy=remat.SAVE
                )(x)

            with remat.saved_tensors_hooks(lambda t: t, lambda t: t):
                y = remat.checkpoint(detect_bare_ops=False)(region)(
                    torch.tensor([1.0, 2.0], requires_grad=True)
                )
            assert saved_output_ref is not None
            return y, saved_output_ref

        assert_reclaimed_without_gc(self, make_graph)

    def test_save_output_through_user_offload_hook_freed_without_gc(self) -> None:
        # Complement to the identity-hook test: an *offloading* user pack returns an opaque
        # token (an index), never the tensor, so its payload cannot pin the output's
        # grad_fn regardless of the detach. This path was always cycle-free; the test pins
        # that invariant so a future payload change that starts retaining the tensor is
        # caught here too. (The stash keeps a detached clone alive -- that shares storage,
        # but not the output *object*, so the weakref on the object still dies.)
        def make_graph() -> tuple[object, weakref.ReferenceType[torch.Tensor]]:
            saved_output_ref: weakref.ReferenceType[torch.Tensor] | None = None
            stash: list[torch.Tensor] = []

            def pack(tensor: torch.Tensor) -> object:
                stash.append(tensor.detach().clone())
                return len(stash) - 1

            class SavesOwnOutput(torch.autograd.Function):
                @staticmethod
                def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                    nonlocal saved_output_ref
                    y = x * 2
                    saved_output_ref = weakref.ref(y)
                    ctx.save_for_backward(y)
                    return y

                @staticmethod
                def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                    (y,) = ctx.saved_tensors
                    del y
                    return grad_output * 2

            def region(x: torch.Tensor) -> torch.Tensor:
                return remat.op(
                    SavesOwnOutput.apply, "saves.own.output", policy=remat.SAVE
                )(x)

            with remat.saved_tensors_hooks(pack, lambda index: stash[cast(int, index)]):
                y = remat.checkpoint(detect_bare_ops=False)(region)(
                    torch.tensor([1.0, 2.0], requires_grad=True)
                )
            assert saved_output_ref is not None
            return y, saved_output_ref

        assert_reclaimed_without_gc(self, make_graph)

    def test_recompute_chain_does_not_retain_producer_output(
        self,
    ) -> None:
        class Producer(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Producer.runs += 1
                y = x * 3
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        test_case = self

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                test_case.assertEqual(2, x.numel())
                test_case.assertGreater(x.untyped_storage().nbytes(), 0)
                ctx.save_for_backward(x)
                if not remat.is_recomputing():
                    test_case.assertExpectedInline(
                        remat.format_current_memory_report(),
                        """inputs: 0 B resident in 0 storage(s)""",
                    )
                return x.sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * torch.ones_like(x)

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            y = remat.op(
                Producer.apply,
                "producer",
                policy=remat.RECOMPUTE,
            )(x)
            return remat.op(
                Consumer.apply,
                "consumer",
                policy=remat.RECOMPUTE,
            )(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)

        y = remat.checkpoint(
            region_name="inputs",
        )(checkpointed_region)(x)
        y.backward()

        self.assertEqual(2, Producer.runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([3.0, 3.0])))

    def test_recompute_to_save_input_is_not_retained(self) -> None:
        # RECOMPUTE -> SAVE crossing where the SAVE op saves its input for backward.
        # The input is a RECOMPUTE op's output, reproduced during replay, so it must
        # not be retained: the memory report shows 0 B, the producer's output dies
        # after the forward, and the value is recovered by recompute at backward.
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

        test_case = self

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                nonlocal producer_output_ref
                if not remat.is_recomputing():
                    producer_output_ref = weakref.ref(y)
                    test_case.assertExpectedInline(
                        remat.format_current_memory_report(),
                        """r: 0 B resident in 0 storage(s)""",
                    )
                ctx.save_for_backward(y)  # saves its input (a RECOMPUTE output)
                return y * y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                return grad_output * 2 * y

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.op(
                Producer.apply,
                "producer",
                policy=remat.RECOMPUTE,
            )(x)
            return remat.op(
                Consumer.apply,
                "consumer",
                policy=remat.SAVE,
            )(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(region)(x)

        ref = producer_output_ref
        assert ref is not None
        gc.collect()
        # The SAVE op saved this RECOMPUTE output but did not retain it.
        self.assertIsNone(ref())

        out.sum().backward()
        self.assertEqual(2, Producer.runs)  # recomputed once at backward
        # d/dx (3x)^2 = 18x
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 36.0])))

    def test_save_to_save_input_lives_on_autograd(self) -> None:
        # SAVE -> SAVE crossing: the consumer saves its input, which is the producer
        # SAVE op's output (a stub during recompute). Recompute does not reproduce it,
        # so there is nothing to divert to the tape -- it is retained like any other
        # save, on the autograd graph, and recovered at backward.
        class ProducerSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class ConsumerSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(y)  # saves a stub input (SAVE -> SAVE)
                return y * y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                return grad_output * 2 * y

        def consumes_save(x: torch.Tensor) -> torch.Tensor:
            y = remat.op(ProducerSave.apply, "producer", policy=remat.SAVE)(x)
            return remat.op(ConsumerSave.apply, "consumer", policy=remat.SAVE)(y)

        # Introspect the forward: the stub input lands on the autograd-owned
        # attribution index, not on the consumer's tape.
        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                forward_context, _ = _checkpoint_context_fn(
                    "r", detect_bare_ops=strategy
                )
                with forward_context:
                    # Keep the output (and thus the autograd graph) alive so the
                    # weakly-held attribution below is not collected before we read it.
                    out = consumes_save(torch.tensor([1.0, 2.0], requires_grad=True))
                    active = _state.get()
                    assert active is not None
                    consumer_record = active.region_state.records["consumer"]
                    # The stub input is autograd-owned (weak attribution), not diverted;
                    # so the consumer records no saved-input recipe, and its own output
                    # has no bare consumer (the region output is unwrapped at the boundary,
                    # not materialized), so no output.<i> slot is created either.
                    self.assertEqual([], list(consumer_record.saved_input_recipes))
                    self.assertEqual([], list(consumer_record.output_slots))
                    self.assertEqual(1, len(consumer_record.saved_tensor_names))
                    self.assertEqual({}, active.region_state.recompute_saved_inputs)
                    del out

        # And it produces correct gradients end to end.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(consumes_save)(x)
        out.sum().backward()
        # d/dx (3x)^2 = 18x
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 36.0])))

    def test_recompute_consumers_share_one_producer_durable_output(self) -> None:
        # A SAVE output consumed by RECOMPUTE ops is the *producer's* responsibility: each
        # consumer (reached through a positional, list, or kwarg argument) triggers the
        # producer's durable save, which is idempotent -- so one shared ``output.0`` slot
        # on the producer serves all three consumers, and the consumers hold no tape state
        # of their own.
        class ProducerSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                del ctx
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        def bare(t: torch.Tensor) -> torch.Tensor:
            return t * 2

        def in_list(pair: list[torch.Tensor]) -> torch.Tensor:
            return pair[0] * 2

        def in_kwargs(*, y: torch.Tensor) -> torch.Tensor:
            return y * 2

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.op(ProducerSave.apply, "producer", policy=remat.SAVE)(x)
            a = remat.op(bare, "bare", policy=remat.RECOMPUTE)(y)
            b = remat.op(in_list, "in_list", policy=remat.RECOMPUTE)([y])
            c = remat.op(in_kwargs, "in_kwargs", policy=remat.RECOMPUTE)(y=y)
            return a + b + c

        # Introspect the forward: the producer holds the single durably saved output.0,
        # regardless of how each RECOMPUTE consumer received it (positional / list / kwarg).
        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                forward_context, _ = _checkpoint_context_fn(
                    "r", detect_bare_ops=strategy
                )
                with forward_context:
                    region(torch.tensor([1.0, 2.0], requires_grad=True))
                    active = _state.get()
                    assert active is not None
                    records = active.region_state.records
                    self.assertEqual([0], list(records["producer"].output_slots))

        # And it still round-trips through recompute at backward: each consumer computes
        # 2*(3x) = 6x, so d/dx sum(a+b+c) = 18 per element.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = remat.checkpoint(region_name="r")(region)(x)
        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([18.0, 18.0])))

    def test_save_op_saved_tensors_live_on_autograd_not_tape(self) -> None:
        # SAVE saved tensors are owned by autograd (ordinary saved tensors on the
        # original forward graph), not the remat tape: the op record keeps no tape
        # slot for them (only a weak, report-only attribution), and a normal
        # backward frees them via autograd -- no manual tape pop. The identity hook
        # retains a detached, storage-sharing copy (see _default_pack), so it
        # is that resident copy -- the attribution key -- whose lifetime we track.
        class Probe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                saved = x + 1
                ctx.save_for_backward(saved)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (saved,) = ctx.saved_tensors
                return grad_output * 2 * (saved - 1)

        forward_context, _ = _checkpoint_context_fn("blk")
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with forward_context:
            y = remat.op(Probe.apply, "probe", policy=remat.SAVE)(x)
            active = _state.get()
            assert active is not None
            record = active.region_state.records["probe"]
            # No tape slot for the SAVE save; autograd owns it. This SAVE op saved no
            # input and its output has no bare consumer, so it records no output slot
            # and no saved-input recipe.
            self.assertEqual({}, dict(record.output_slots))
            self.assertEqual([], list(record.saved_input_recipes))
            self.assertEqual(1, len(record.saved_tensor_names))
            (resident,) = record.saved_tensor_names.keys()
            resident_ref = weakref.ref(resident)
            del resident

        # Held by the autograd graph (reachable from y) until backward runs.
        gc.collect()
        self.assertIsNotNone(resident_ref())

        y.sum().backward()
        del y
        gc.collect()
        # Freed by autograd after a normal (non-retain_graph) backward; no pop.
        self.assertIsNone(resident_ref())
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_save_op_dead_output_is_freed_after_forward(self) -> None:
        # A SAVE op output that nothing consumes -- not fed downstream, not saved for
        # backward, not returned as the region output -- must not stay resident from
        # forward to backward. The region's weak-keyed save-output index holds each
        # output's handle as the *value* of its own weak entry, so the handle must not
        # strong-reference the key: neither via an unwrap that closes over the output nor
        # via a durable-save that eagerly snapshots (and so pins the storage of) the
        # output. Otherwise a large dead auxiliary output (e.g. an unused attention LSE)
        # silently costs its full size across the whole peak-memory window an activation
        # checkpointing library exists to shrink. Checked for the no-detect path
        # and every bare-op strategy; before the fix only "proxy" freed the output.
        for strategy in (False, *_BARE_OP_STRATEGIES):
            with self.subTest(strategy=strategy):
                aux_ref: weakref.ReferenceType[torch.Tensor] | None = None

                class TwoOutputSave(torch.autograd.Function):
                    @staticmethod
                    def forward(
                        ctx: Any, x: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        del ctx
                        return x * 2, x * 3  # second output is an unused auxiliary

                    @staticmethod
                    def backward(
                        ctx: Any, grad_main: torch.Tensor, grad_aux: torch.Tensor
                    ) -> torch.Tensor:
                        del ctx, grad_aux
                        return grad_main * 2

                def region(x: torch.Tensor) -> torch.Tensor:
                    nonlocal aux_ref
                    main, aux = remat.op(
                        TwoOutputSave.apply, "twoout", policy=remat.SAVE
                    )(x)
                    if not remat.is_recomputing():
                        # Weakref the real produced tensor (the wrapper's inner for the
                        # subclass / proxy strategies; the value itself otherwise).
                        aux_ref = weakref.ref(getattr(aux, "_inner", aux))
                    return main  # aux is consumed by nothing

                x = torch.ones(1024, requires_grad=True)
                out = remat.checkpoint(region_name="r", detect_bare_ops=strategy)(
                    region
                )(x)

                ref = aux_ref
                assert ref is not None
                gc.collect()
                # The dead auxiliary output was not pinned by the save-output index.
                self.assertIsNone(ref())

                out.sum().backward()
                self.assertTrue(torch.equal(x.grad, torch.full((1024,), 2.0)))
