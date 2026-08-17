# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for retention and GC semantics across recompute/save crossings: a
recompute=True region retains none of its original saved tensors, a saved-own-output
is freed without gc, and the recompute/save producer-consumer chains keep or drop the
producer output exactly as the tape model requires -- plus dead saved outputs are
freed and saved tensors live on autograd, not the tape."""

from __future__ import annotations

import gc
import weakref
from typing import Any, cast

import expecttest
import pytest
import torch
import torch_remat as remat
from remat_test_helpers import (
    _ref_grad,
    assert_reclaimed_without_gc,
    checkpoint_for_test,
)
from torch_remat._region import (
    _checkpoint_context_fn,
    _state,
)


def _make_exp_save_op(saves: str) -> type[torch.autograd.Function]:
    """An op computing exp(x) that saves for backward per ``saves``: an internal
    (a distinct exp(x)), its own output, or its input. exp keeps all three roles
    correct with the same op -- the internal/output saves already hold the grad
    factor exp(x); the input save re-applies exp."""

    class Op(torch.autograd.Function):
        runs: int = 0

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            Op.runs += 1
            out = torch.exp(x)
            saved = {"internal": torch.exp(x), "output": out, "input": x}[saves]
            ctx.save_for_backward(saved)
            ctx.saves = saves
            return out

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            (saved,) = ctx.saved_tensors
            factor = saved.exp() if ctx.saves == "input" else saved
            return grad_output * factor

    return Op


def _make_recompute_doubler() -> type[torch.autograd.Function]:
    """A RECOMPUTE producer: doubles its input, counting its forward runs."""

    class Doubler(torch.autograd.Function):
        runs: int = 0

        @staticmethod
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            Doubler.runs += 1
            del ctx
            return x * 2

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            del ctx
            return grad_output * 2

    return Doubler


def _make_taxonomy_op() -> type[torch.autograd.Function]:
    """An op that saves an internal (3w) and its input (w) and emits a dead
    auxiliary output (w + 7), recording a weakref to the internal on the forward."""

    class Op(torch.autograd.Function):
        internal_ref: weakref.ReferenceType[torch.Tensor] | None = None

        @staticmethod
        def forward(ctx: Any, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            internal = w * 3
            if not remat.is_recomputing():
                Op.internal_ref = weakref.ref(internal)
            aux = w + 7  # dead auxiliary output
            out = internal * w  # 3 * w**2
            ctx.save_for_backward(internal, w)
            return out, aux

        @staticmethod
        def backward(
            ctx: Any, grad_out: torch.Tensor, grad_aux: torch.Tensor
        ) -> torch.Tensor:
            (internal, w) = ctx.saved_tensors
            del grad_aux
            return grad_out * (internal + 3 * w)  # d(3 w**2)/dw = 6w

    return Op


def _exp_exp_times3(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.exp(t)) * 3  # A = exp, B = exp, tail = * 3


def _twelve_t_squared(t: torch.Tensor) -> torch.Tensor:
    return 3 * (t * 2) ** 2  # producer w = 2t, op out = 3 w**2 -> 12 t**2


class LifetimeTest(expecttest.TestCase):
    @pytest.mark.compile_xfail(
        "compiled regions do not retain the eager forward's saved tensor"
    )
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
            return remat.region(
                SavedTensorLifetimeProbe.apply,
                "saved.tensor.lifetime",
                recompute=True,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = checkpoint_for_test()(checkpoint_body)(x)

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
            return remat.region(
                SavesOwnOutput.apply, "saves.own.output", recompute=False
            )(x)

        was_enabled = gc.isenabled()
        gc.disable()  # isolate refcount reclamation from cyclic collection
        try:
            y = checkpoint_for_test()(region)(
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
                return remat.region(
                    SavesOwnOutput.apply, "saves.own.output", recompute=False
                )(x)

            with remat.saved_tensors_hooks(lambda t: t, lambda t: t):
                y = checkpoint_for_test()(region)(
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
                return remat.region(
                    SavesOwnOutput.apply, "saves.own.output", recompute=False
                )(x)

            with remat.saved_tensors_hooks(pack, lambda index: stash[cast(int, index)]):
                y = checkpoint_for_test()(region)(
                    torch.tensor([1.0, 2.0], requires_grad=True)
                )
            assert saved_output_ref is not None
            return y, saved_output_ref

        assert_reclaimed_without_gc(self, make_graph)

    @pytest.mark.compile_xfail(
        "compiled regions do not expose the eager producer-output lifetime"
    )
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
            y = remat.region(
                Producer.apply,
                "producer",
                recompute=True,
            )(x)
            return remat.region(
                Consumer.apply,
                "consumer",
                recompute=True,
            )(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)

        y = checkpoint_for_test(
            region_name="inputs",
        )(checkpointed_region)(x)
        y.backward()

        self.assertEqual(2, Producer.runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([3.0, 3.0])))

    @pytest.mark.compile_xfail(
        "compiled regions do not expose the eager tape's input retention"
    )
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
        out = checkpoint_for_test(region_name="r")(region)(x)

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
            y = remat.region(ProducerSave.apply, "producer", recompute=False)(x)
            return remat.region(ConsumerSave.apply, "consumer", recompute=False)(y)

        # Introspect the forward: the stub input lands on the autograd-owned
        # attribution index, not on the consumer's tape.
        forward_context, _ = _checkpoint_context_fn("r")
        with forward_context:
            # Keep the output (and thus the autograd graph) alive so the
            # weakly-held attribution below is not collected before we read it.
            out = consumes_save(torch.tensor([1.0, 2.0], requires_grad=True))
            active = _state.get()
            assert active is not None
            consumer_record = active.region_state.records["consumer"]
            # The stub input is autograd-owned (weak attribution), not diverted;
            # so the consumer records no saved-input recipe, and its own output
            # has no bare consumer, so no output.<i> slot is created either.
            self.assertEqual([], list(consumer_record.saved_input_recipes))
            self.assertEqual([], list(consumer_record.output_slots))
            self.assertEqual(1, len(consumer_record.saved_tensor_names))
            self.assertEqual({}, active.region_state.rederived_saved_inputs)
            del out

        # And it produces correct gradients end to end.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(consumes_save)(x)
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
            y = remat.region(ProducerSave.apply, "producer", recompute=False)(x)
            a = remat.region(bare, "bare", recompute=True)(y)
            b = remat.region(in_list, "in_list", recompute=True)([y])
            c = remat.region(in_kwargs, "in_kwargs", recompute=True)(y=y)
            return a + b + c

        # Introspect the forward: the producer holds the single durably saved output.0,
        # regardless of how each RECOMPUTE consumer received it (positional / list / kwarg).
        forward_context, _ = _checkpoint_context_fn("r")
        with forward_context:
            region(torch.tensor([1.0, 2.0], requires_grad=True))
            active = _state.get()
            assert active is not None
            records = active.region_state.records
            self.assertEqual([0], list(records["producer"].output_slots))

        # And it still round-trips through recompute at backward: each consumer computes
        # 2*(3x) = 6x, so d/dx sum(a+b+c) = 18 per element.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = checkpoint_for_test(region_name="r")(region)(x)
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
            y = remat.region(Probe.apply, "probe", recompute=False)(x)
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

    @pytest.mark.compile_xfail(
        "compiled regions do not expose the eager dead-output lifetime"
    )
    def test_save_op_dead_output_is_freed_after_forward(self) -> None:
        # A SAVE op output that nothing consumes -- not fed downstream, not saved for
        # backward, not returned as the region output -- must not stay resident from
        # forward to backward. The region's storage-keyed save-output index holds each
        # output's persist thunk, which references the output only weakly, so an output
        # nothing consumes is not pinned. Otherwise a large dead auxiliary output (e.g. an
        # unused attention LSE) silently costs its full size across the whole peak-memory
        # window an activation checkpointing library exists to shrink.
        aux_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class TwoOutputSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
            main, aux = remat.region(TwoOutputSave.apply, "twoout", recompute=False)(x)
            if not remat.is_recomputing():
                aux_ref = weakref.ref(aux)
            return main  # aux is consumed by nothing

        x = torch.ones(1024, requires_grad=True)
        out = checkpoint_for_test(region_name="r")(region)(x)

        ref = aux_ref
        assert ref is not None
        gc.collect()
        # The dead auxiliary output was not pinned by the save-output index.
        self.assertIsNone(ref())

        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.full((1024,), 2.0)))

    @pytest.mark.compile_xfail("compiled regions do not expose eager tensor lifetimes")
    def test_interior_recompute_output_freed_but_escaping_output_retained(
        self,
    ) -> None:
        # Two recompute=True regions chained in ONE checkpoint scope. The first region's
        # output stays INTERIOR (consumed by the second region), so it is freed after the
        # forward and recomputed on the backward replay. The second region's output is
        # also a recompute=True output, but it ESCAPES the scope (it is returned), so its
        # value survives the forward and is directly readable with no recompute. Escaping
        # the ckpt scope, not the recompute flag, is what decides retention.
        interior_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class Stage0(torch.autograd.Function):  # produces the interior output
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Stage0.runs += 1
                ctx.save_for_backward(x * 2)  # d(x*x)/dx = 2x
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (grad_factor,) = ctx.saved_tensors
                return grad_output * grad_factor

        class Stage1(torch.autograd.Function):  # produces the escaping output
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Stage1.runs += 1
                ctx.save_for_backward(x * 2)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (grad_factor,) = ctx.saved_tensors
                return grad_output * grad_factor

        def body(x: torch.Tensor) -> torch.Tensor:
            nonlocal interior_ref
            interior = remat.region(Stage0.apply, "stage.0", recompute=True)(x)
            if not remat.is_recomputing():
                interior_ref = weakref.ref(interior)
            return remat.region(Stage1.apply, "stage.1", recompute=True)(interior)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = checkpoint_for_test(region_name="scope")(body)(x)

        ref = interior_ref
        assert ref is not None
        gc.collect()
        # The interior output is freed after the forward.
        self.assertIsNone(ref())
        # Neither region has recomputed yet (reading the escaped output does not trigger
        # recompute), and the escaping output (== x**4) survived the forward intact.
        self.assertEqual(1, Stage0.runs)
        self.assertEqual(1, Stage1.runs)
        self.assertTrue(torch.equal(out.detach(), torch.tensor([1.0, 16.0])))

        out.sum().backward()
        # Both recompute=True regions rerun on the backward replay.
        self.assertEqual(2, Stage0.runs)
        self.assertEqual(2, Stage1.runs)
        # d/dx sum(x**4) = 4 * x**3
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 32.0])))

    @pytest.mark.compile_xfail("compiled regions do not expose eager tensor lifetimes")
    def test_recompute_output_shared_by_save_and_recompute_consumers_is_freed(
        self,
    ) -> None:
        # A recompute=True region output consumed by BOTH a SAVE region (recompute=False)
        # and a RECOMPUTE region (recompute=True) -- mixed-policy fan-out on one producer
        # output. It stays interior, so it is freed after the forward and reproduced on
        # the backward replay: the SAVE consumer saved it as an INPUT (a RECOMPUTE -> SAVE
        # crossing, captured during replay, not retained), and the RECOMPUTE consumer
        # reruns.
        producer_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class Producer(torch.autograd.Function):  # recompute=True
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                Producer.runs += 1
                del ctx
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        class SaveConsumer(torch.autograd.Function):  # SAVE; saves its input
            @staticmethod
            def forward(ctx: Any, h: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(h)
                return h * h

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (h,) = ctx.saved_tensors
                return grad_output * 2 * h

        class RecomputeConsumer(torch.autograd.Function):  # RECOMPUTE
            @staticmethod
            def forward(ctx: Any, h: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(h)
                return h * h * h

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (h,) = ctx.saved_tensors
                return grad_output * 3 * h * h

        def body(x: torch.Tensor) -> torch.Tensor:
            nonlocal producer_output_ref
            h = remat.region(Producer.apply, "producer", recompute=True)(x)
            if not remat.is_recomputing():
                producer_output_ref = weakref.ref(h)
            saved = remat.region(SaveConsumer.apply, "save_consumer", recompute=False)(
                h
            )
            recomputed = remat.region(
                RecomputeConsumer.apply, "recompute_consumer", recompute=True
            )(h)
            # Combine via a region: `saved` is a SAVE output, so a bare consumer would
            # need remat.recompute_needs_tensor; a region consumer ferries it directly.
            return remat.region(torch.add, "combine", recompute=True)(saved, recomputed)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = checkpoint_for_test(region_name="scope")(body)(x)

        ref = producer_output_ref
        assert ref is not None
        gc.collect()
        # The producer output is a recompute output consumed only inside the scope ->
        # freed, even though the SAVE consumer saved it as an input.
        self.assertIsNone(ref())

        out.sum().backward()
        self.assertEqual(2, Producer.runs)  # recomputed at backward
        # out = (2x)**2 + (2x)**3 = 4x**2 + 8x**3; d/dx = 8x + 24x**2
        self.assertTrue(torch.equal(x.grad, torch.tensor([32.0, 112.0])))

    @pytest.mark.compile_xfail("compiled regions do not expose eager tensor lifetimes")
    def test_bare_op_output_is_freed_and_recomputed(self) -> None:
        # A plain op -- not wrapped in remat.region -- inside a checkpoint scope recomputes
        # by default. Its output is consumed by a downstream plain op that saves it for
        # backward, but under the whole-region checkpoint that save is diverted, so the
        # output is freed after the forward and recomputed on the backward replay. No SAVE
        # region is involved -- this is the default-recompute path for unwrapped ops.
        producer_output_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class BareProducer(torch.autograd.Function):  # run without remat.region
            runs: int = 0

            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                BareProducer.runs += 1
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        class BareConsumer(torch.autograd.Function):  # plain consumer
            @staticmethod
            def forward(ctx: Any, y: torch.Tensor) -> torch.Tensor:
                nonlocal producer_output_ref
                if not remat.is_recomputing():
                    producer_output_ref = weakref.ref(y)
                ctx.save_for_backward(y)
                return y * y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                return grad_output * 2 * y

        def body(x: torch.Tensor) -> torch.Tensor:
            produced = BareProducer.apply(x)  # not wrapped in remat.region
            return BareConsumer.apply(produced)  # plain consumer

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = checkpoint_for_test(region_name="scope")(body)(x)

        ref = producer_output_ref
        assert ref is not None
        gc.collect()
        # The unwrapped recompute op's output was freed after the forward.
        self.assertIsNone(ref())

        out.sum().backward()
        self.assertEqual(2, BareProducer.runs)  # recomputed at backward
        # out = (2x)**2 = 4x**2; d/dx sum = 8x
        self.assertTrue(torch.equal(x.grad, torch.tensor([8.0, 16.0])))

    def _check_retention_cell(
        self, recompute_a: bool, recompute_b: bool, saves_a: str
    ) -> None:
        a_op = _make_exp_save_op(saves_a)
        b_op = _make_exp_save_op(
            "internal"
        )  # B fixed; its role is in the taxonomy test
        y_ref: weakref.ReferenceType[torch.Tensor] | None = None
        z_ref: weakref.ReferenceType[torch.Tensor] | None = None

        def body(x: torch.Tensor) -> torch.Tensor:
            nonlocal y_ref, z_ref
            y = remat.region(a_op.apply, "a", recompute=recompute_a)(x)
            z = remat.region(b_op.apply, "b", recompute=recompute_b)(y)
            if not remat.is_recomputing():
                y_ref = weakref.ref(y)
                z_ref = weakref.ref(z)
            return remat.region(lambda t: t * 3, "tail", recompute=True)(z)

        base = torch.tensor([0.5, 1.0], dtype=torch.float64)
        x = base.clone().requires_grad_(True)
        out = checkpoint_for_test(region_name="scope")(body)(x)

        assert y_ref is not None and z_ref is not None
        gc.collect()
        # A RECOMPUTE region frees its interior output regardless of save role.
        for recompute, output_ref in ((recompute_a, y_ref), (recompute_b, z_ref)):
            if recompute:
                self.assertIsNone(output_ref())

        out.sum().backward()

        # SAVE runs once (body skipped on recompute); RECOMPUTE runs twice.
        self.assertEqual(2 if recompute_a else 1, a_op.runs)
        self.assertEqual(2 if recompute_b else 1, b_op.runs)
        # Gradient matches the uncheckpointed reference in every cell.
        self.assertTrue(torch.allclose(x.grad, _ref_grad(_exp_exp_times3, base)))

    @pytest.mark.compile_xfail("compiled regions do not expose eager tensor lifetimes")
    def test_retention_matrix_over_recompute_flags_and_save_roles(self) -> None:
        # Retention across a two-region chain A -> B (plus a trailing RECOMPUTE op so that
        # BOTH A's output and B's output stay *interior* -- consumed downstream, never
        # returned), swept over two orthogonal axes:
        #
        #   * each region's policy: recompute_a, recompute_b in {False (SAVE), True};
        #   * what the FIRST op saves for backward: an internal intermediate, its own
        #     output, or its input.
        #
        # The invariant under test is that the save role does NOT change interior-output
        # retention: a RECOMPUTE region frees its interior output no matter what it saved
        # (a dead weakref witnesses it), a SAVE region keeps it (runs once, never
        # recomputes), and the gradient matches the uncheckpointed reference in every cell.
        # Per-save-role retention of the *saved tensor itself* is pinned by the taxonomy
        # test below.
        for recompute_a in (False, True):
            for recompute_b in (False, True):
                for saves_a in ("internal", "output", "input"):
                    with self.subTest(
                        recompute_a=recompute_a,
                        recompute_b=recompute_b,
                        saves_a=saves_a,
                    ):
                        self._check_retention_cell(recompute_a, recompute_b, saves_a)

    def _run_taxonomy_pass(self, recompute_op: bool) -> None:
        producer = _make_recompute_doubler()
        op = _make_taxonomy_op()
        aux_ref: weakref.ReferenceType[torch.Tensor] | None = None
        w_ref: weakref.ReferenceType[torch.Tensor] | None = None

        def body(x: torch.Tensor) -> torch.Tensor:
            nonlocal aux_ref, w_ref
            w = remat.region(producer.apply, "producer", recompute=True)(x)
            if not remat.is_recomputing():
                w_ref = weakref.ref(w)
            out, aux = remat.region(op.apply, "op", recompute=recompute_op)(w)
            if not remat.is_recomputing():
                aux_ref = weakref.ref(aux)
            return out  # aux dropped

        base = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x = base.clone().requires_grad_(True)
        out = checkpoint_for_test(region_name="scope")(body)(x)

        assert aux_ref is not None and w_ref is not None
        gc.collect()
        self.assertIsNone(w_ref())  # a saved input (from a RECOMPUTE producer) is freed
        self.assertIsNone(aux_ref())  # a dead auxiliary output is freed
        if recompute_op:
            # A RECOMPUTE op frees its internal too (a SAVE op keeps it -- see below).
            assert op.internal_ref is not None
            self.assertIsNone(op.internal_ref())

        out.sum().backward()
        self.assertEqual(2, producer.runs)  # the producer reran to reproduce w
        self.assertTrue(torch.allclose(x.grad, _ref_grad(_twelve_t_squared, base)))

    @pytest.mark.compile_xfail("compiled regions do not expose eager tensor lifetimes")
    def test_save_role_retention_taxonomy_is_per_tensor(self) -> None:
        # Orthogonality: within a SINGLE op, each saved/produced tensor's fate is decided
        # by its own role, independent of the others. One op saves an internal AND its own
        # input (sourced from a RECOMPUTE producer), and also emits a dead auxiliary
        # output. As a RECOMPUTE op every one of those is freed and reproduced on the
        # replay; as a SAVE op the roles diverge -- the internal stays resident while the
        # saved input is freed (recomputed) and the dead aux is freed.
        self._run_taxonomy_pass(recompute_op=True)
        self._run_taxonomy_pass(recompute_op=False)

        # Direct residency view of the SAVE op: only the internal is resident (the saved
        # input is rebuilt on recompute; the dead aux and the output carry no bytes here).
        forward_context, _ = _checkpoint_context_fn("scope")
        base = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x = base.clone().requires_grad_(True)
        with forward_context:
            doubler = _make_recompute_doubler()
            w = remat.region(doubler.apply, "producer", recompute=True)(x)
            # Hold the op output so its grad_fn keeps the SAVE saves live for the report.
            held = remat.region(_make_taxonomy_op().apply, "op", recompute=False)(w)
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
scope: 16 B resident in 1 storage(s)
scope::op: 16 B
  16 B  saved.0  (2,)  float64
  + 1 save rebuilt on recompute, not resident""",
            )
            del held
