# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``checkpoint(..., recompute_state_hooks=...)`` and the built-in
``preserve_rng_state`` hook.

A ``recompute=False`` (SAVE) region's body runs on the forward but is skipped on
recompute, so any external state it advanced (e.g. a global RNG op-counter, or
torch's own generator) would be left behind and every downstream draw would shift.
A :class:`RecomputeStateHook` is snapshotted at region entry and after each SAVE op
on the forward, restored at the matching points on recompute, and the region is
forked so the replay's redraws don't leak into the surrounding stream. These tests
exercise all of that on CPU.

Each op saves a tensor for backward (and reads it back), which is what makes the
checkpoint region actually recompute during ``backward`` -- without a saved tensor
to unpack, no recompute fires and the test would pass vacuously."""

from __future__ import annotations

from typing import Any

import expecttest
import torch
import torch_remat as remat


class _IntCounter:
    """Minimal :class:`remat.RecomputeStateHook` -- a mutable integer position.

    Ops mutate ``value`` directly (standing in for RNG-seed draws); the hook
    snapshots and restores that integer.
    """

    def __init__(self) -> None:
        self.value = 0

    def snapshot(self) -> int:
        return self.value

    def restore(self, state: int) -> None:
        self.value = state


def _probe_op(counter: _IntCounter, observed: dict[str, list[int]], draw: int) -> Any:
    """A ``recompute=True`` op that records the counter it sees, then advances it.

    Reruns on both passes (like a bare / RECOMPUTE draw), so it advances the counter
    naturally and needs no hook -- it exists to observe whether the *hook-managed*
    restores line the counter up at its position. Saves its input so backward forces
    the region to recompute.
    """

    class _Probe(torch.autograd.Function):
        @staticmethod
        # pyre-ignore[14]
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            key = "recompute" if remat.is_recomputing() else "forward"
            observed[key].append(counter.value)
            counter.value += draw
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        # pyre-ignore[14]
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            return grad_output * 2 * x

    return _Probe.apply


def _save_advance_op(counter: _IntCounter, draw: int) -> Any:
    """A ``recompute=False`` (SAVE) op that advances the counter on its forward.

    Stands in for a norm / residual add that draws an SR seed. Its body must not run
    on recompute; the hook restores the counter to its forward exit instead.
    """

    class _SaveAdvance(torch.autograd.Function):
        @staticmethod
        # pyre-ignore[14]
        def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
            if remat.is_recomputing():
                raise AssertionError("SAVE replay must skip the forward body")
            counter.value += draw
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        # pyre-ignore[14]
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            (x,) = ctx.saved_tensors
            return grad_output * 2 * x

    return _SaveAdvance.apply


class RecomputeStateHookTest(expecttest.TestCase):
    def test_entry_and_save_op_state_restored_on_recompute(self) -> None:
        # probe_a runs BEFORE any SAVE op: it can only see the same counter on both
        # passes if the region-entry snapshot is restored at recompute start. The
        # SAVE op then advances the counter; probe_b runs AFTER it and can only agree
        # across passes if the skipped SAVE op's exit state is restored too.
        counter = _IntCounter()
        observed: dict[str, list[int]] = {"forward": [], "recompute": []}
        probe_a = _probe_op(counter, observed, draw=1)
        save_op = _save_advance_op(counter, draw=2)
        probe_b = _probe_op(counter, observed, draw=1)

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(probe_a, "a", recompute=True)(x)
            z = remat.region(save_op, "save", recompute=False)(y)
            return remat.region(probe_b, "b", recompute=True)(z)

        x = torch.tensor([2.0, 5.0], requires_grad=True)
        out = remat.checkpoint(
            region_name="m",
            preserve_rng_state=False,
            recompute_state_hooks=(counter,),
        )(body)(x)
        # Unrelated draws happen between the forward and the backward recompute (as
        # other layers would in a real model); the entry restore must undo this.
        counter.value += 100
        out.sum().backward()

        # forward: probe_a @0, save +2 -> counter 3, probe_b @3.
        # recompute: entry restore -> 0, probe_a @0, save skipped -> exit restore 3,
        #            probe_b @3.
        self.assertEqual([0, 3], observed["forward"])
        self.assertEqual([0, 3], observed["recompute"])

    def test_without_hooks_state_desyncs(self) -> None:
        # Control: with no hook registered, neither the entry nor the skipped SAVE
        # op's state is restored, so the probes see shifted positions on recompute --
        # exactly the divergence recompute_state_hooks exists to prevent.
        counter = _IntCounter()
        observed: dict[str, list[int]] = {"forward": [], "recompute": []}
        probe_a = _probe_op(counter, observed, draw=1)
        save_op = _save_advance_op(counter, draw=2)
        probe_b = _probe_op(counter, observed, draw=1)

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(probe_a, "a", recompute=True)(x)
            z = remat.region(save_op, "save", recompute=False)(y)
            return remat.region(probe_b, "b", recompute=True)(z)

        x = torch.tensor([2.0], requires_grad=True)
        out = remat.checkpoint(region_name="m", preserve_rng_state=False)(body)(x)
        counter.value += 100
        out.sum().backward()

        self.assertEqual([0, 3], observed["forward"])
        # No entry restore: recompute resumes at the post-forward 104 (counter ended
        # at 4, plus the 100 unrelated draws). probe_a sees 104 and advances to 105;
        # the skipped SAVE op restores nothing, so probe_b sees 105. Both desynced
        # from the forward's [0, 3].
        self.assertEqual([104, 105], observed["recompute"])

    def test_multiple_hooks_restored_independently(self) -> None:
        # Each registered hook is snapshotted/restored independently, in order.
        a = _IntCounter()
        b = _IntCounter()
        observed_a: dict[str, list[int]] = {"forward": [], "recompute": []}
        observed_b: dict[str, list[int]] = {"forward": [], "recompute": []}

        class _SaveAdvance(torch.autograd.Function):
            @staticmethod
            # pyre-ignore[14]
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                a.value += 1
                b.value += 4
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            # pyre-ignore[14]
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        class _Probe(torch.autograd.Function):
            @staticmethod
            # pyre-ignore[14]
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                key = "recompute" if remat.is_recomputing() else "forward"
                observed_a[key].append(a.value)
                observed_b[key].append(b.value)
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            # pyre-ignore[14]
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(_SaveAdvance.apply, "save", recompute=False)(x)
            return remat.region(_Probe.apply, "probe", recompute=True)(y)

        x = torch.tensor([2.0], requires_grad=True)
        out = remat.checkpoint(
            region_name="m", preserve_rng_state=False, recompute_state_hooks=(a, b)
        )(body)(x)
        a.value += 100
        b.value += 100
        out.sum().backward()

        self.assertEqual([1], observed_a["forward"])
        self.assertEqual([1], observed_a["recompute"])
        self.assertEqual([4], observed_b["forward"])
        self.assertEqual([4], observed_b["recompute"])

    def test_fork_restores_outer_state_after_recompute(self) -> None:
        # Fork semantics: the replay realigns to the region's forward state, but on
        # exit the *outer* state (as it was just before recompute ran) is reinstated
        # so the replay's effect does not leak into later backward / the next step.
        counter = _IntCounter()
        observed: dict[str, list[int]] = {"forward": [], "recompute": []}
        save_op = _save_advance_op(counter, draw=2)
        # A downstream recompute=True op forces the region to recompute during backward.
        probe = _probe_op(counter, observed, draw=0)

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(save_op, "save", recompute=False)(x)
            return remat.region(probe, "probe", recompute=True)(y)

        x = torch.tensor([1.0], requires_grad=True)
        out = remat.checkpoint(
            region_name="m", preserve_rng_state=False, recompute_state_hooks=(counter,)
        )(body)(x)
        self.assertEqual(2, counter.value)  # forward advanced it
        counter.value = 777  # stand-in for the outer state at recompute time
        out.sum().backward()
        # entry(0) + skipped-SAVE restore(2) drive the replay, then the outer 777 is
        # reinstated -- recompute leaves the surrounding stream untouched.
        self.assertEqual(777, counter.value)


class PreserveRngStateTest(expecttest.TestCase):
    """torch_remat does not preserve torch's RNG state; True must fail loud."""

    def test_preserve_rng_state_true_raises(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "does not preserve torch's RNG state"
        ):
            remat.checkpoint(region_name="m", preserve_rng_state=True)

    def test_preserve_rng_state_defaults_off(self) -> None:
        # The default must not error: a plain checkpoint builds and runs.
        def body(x: torch.Tensor) -> torch.Tensor:
            return remat.region(lambda t: t * t, "sq", recompute=True)(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        out = remat.checkpoint(region_name="m")(body)(x)
        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))
