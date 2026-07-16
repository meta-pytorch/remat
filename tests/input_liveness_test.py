# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for per-input liveness at the checkpoint region boundary.

A region input consumed only by ``recompute=False`` regions (or nothing) is never read
during recompute, so ``remat.checkpoint`` routes inputs through a synthetic SAVE op and
lets producer-responsibility keep only the ones a consumer touches -- dropping the rest
instead of pinning every input for the whole backward (the prior ``torch.utils.checkpoint``
behavior). These tests pin down: the drop actually happens; an input a ``RECOMPUTE`` region
(or a ``recompute_needs_tensor`` flag) consumes is kept; the routing is numerically
transparent across every consumer shape; and the ``input_saved_tensors_hooks`` path falls
back to saving every input."""

from __future__ import annotations

import weakref
from typing import Any, Callable

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import _assert_byte_column_sums


class _SaveIntermediate(torch.autograd.Function):
    """SAVE op that saves an *intermediate* (``x * 2``) for backward, never its input.

    Its input is therefore only consumed by a ``recompute=False`` region that does not
    save it -- the case a dead region input should be dropped rather than pinned.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x * 2.0)
        return x * x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (grad_factor,) = ctx.saved_tensors
        return grad_output * grad_factor


class _SaveInput(torch.autograd.Function):
    """SAVE op that saves its own input for backward (the divert-to-recipe / stub path)."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * 2.0 * x


class _RecomputeMul(torch.autograd.Function):
    """A plain op driven as a ``recompute=True`` region: reruns and reads its input."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        del ctx
        return x * 3.0

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx
        return grad_output * 3.0


def _save_only(x: torch.Tensor) -> torch.Tensor:
    return remat.region(_SaveIntermediate.apply, "s", recompute=False)(x)


def _save_input(x: torch.Tensor) -> torch.Tensor:
    return remat.region(_SaveInput.apply, "s", recompute=False)(x)


def _recompute(x: torch.Tensor) -> torch.Tensor:
    return remat.region(_RecomputeMul.apply, "r", recompute=True)(x)


def _needs_tensor_bare(x: torch.Tensor) -> torch.Tensor:
    # A bare consumer of the input: flag it so the producer (the synthetic input op)
    # persists it, exactly as a bare consumer of any SAVE output must. Without the flag
    # the input would be dropped and this bare op would read a placeholder on recompute.
    remat.recompute_needs_tensor(x)
    return x * 2.0


class InputLivenessTest(expecttest.TestCase):
    def _assert_checkpoint_transparent(
        self,
        region_fn: Callable[..., torch.Tensor],
        build_call: Callable[[torch.Tensor], tuple[tuple[Any, ...], dict[str, Any]]],
    ) -> None:
        """Checkpointing ``region_fn`` must be bitwise-transparent.

        The reference is ``region_fn`` run with *no* checkpoint (``remat.region`` falls
        through to a plain call), so the exact same ops -- and their hand-written backwards
        -- define the expected gradient; checkpoint recompute must reproduce them bitwise.
        ``build_call`` maps a fresh leaf to the ``(args, kwargs)`` for the region so a test
        can shape the input (positional, kwargs, container, non-leaf, ...).
        """

        base = torch.randn(16, requires_grad=True)
        ref_leaf = base.detach().clone().requires_grad_(True)
        ref_args, ref_kwargs = build_call(ref_leaf)
        region_fn(*ref_args, **ref_kwargs).sum().backward()
        assert ref_leaf.grad is not None
        reference = ref_leaf.grad

        leaf = base.detach().clone().requires_grad_(True)
        args, kwargs = build_call(leaf)
        out = remat.checkpoint(region_name="r")(region_fn)(*args, **kwargs)
        out.sum().backward()
        assert leaf.grad is not None
        self.assertTrue(torch.equal(leaf.grad, reference))

    def _run_and_report_input_dropped(
        self,
        region_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        input_saved_tensors_hooks: tuple[Any, Any] | None = None,
    ) -> bool:
        """Run ``region_fn`` on a non-leaf input held only by the region; report whether
        its storage was freed by the time the forward returned (i.e. it was dropped)."""

        base = torch.randn(8, requires_grad=True)
        witness: dict[str, weakref.ref[torch.UntypedStorage]] = {}

        def call() -> torch.Tensor:
            x = base * 3.0  # non-leaf, requires grad; the only live reference is local
            witness["storage"] = weakref.ref(x.untyped_storage())
            return remat.checkpoint(
                region_name="r", input_saved_tensors_hooks=input_saved_tensors_hooks
            )(region_fn)(x)

        out = call()
        dropped = witness["storage"]() is None
        out.sum().backward()  # must still succeed with the input dropped / fed
        assert base.grad is not None
        return dropped

    def test_dead_input_is_dropped(self) -> None:
        self.assertTrue(
            self._run_and_report_input_dropped(_save_only),
            "an input consumed only by a non-saving SAVE region should be freed once the "
            "forward returns",
        )

    def test_input_saved_tensors_hooks_disables_liveness(self) -> None:
        # With input offload hooks installed the liveness path is bypassed, so checkpoint
        # saves every input -- even one that would otherwise be dropped.
        identity_hooks = (lambda t: t, lambda t: t)
        self.assertFalse(
            self._run_and_report_input_dropped(
                _save_only, input_saved_tensors_hooks=identity_hooks
            )
        )

    def test_recompute_consumed_input_is_kept(self) -> None:
        self.assertFalse(
            self._run_and_report_input_dropped(_recompute),
            "an input a RECOMPUTE region reruns from must survive to recompute",
        )

    def test_recompute_needs_tensor_input_is_kept(self) -> None:
        self.assertFalse(
            self._run_and_report_input_dropped(_needs_tensor_bare),
            "an input flagged with recompute_needs_tensor must survive to recompute",
        )

    def test_numerics_save_only(self) -> None:
        self._assert_checkpoint_transparent(
            _save_only, lambda leaf: ((leaf * 3.0,), {})
        )

    def test_numerics_save_input_stub(self) -> None:
        self._assert_checkpoint_transparent(
            _save_input, lambda leaf: ((leaf * 3.0,), {})
        )

    def test_numerics_recompute_consumer(self) -> None:
        self._assert_checkpoint_transparent(
            _recompute, lambda leaf: ((leaf * 3.0,), {})
        )

    def test_numerics_recompute_needs_tensor(self) -> None:
        self._assert_checkpoint_transparent(
            _needs_tensor_bare, lambda leaf: ((leaf * 3.0,), {})
        )

    def test_numerics_kwargs_input(self) -> None:
        def region_fn(*, inp: torch.Tensor) -> torch.Tensor:
            return remat.region(_SaveInput.apply, "s", recompute=False)(inp)

        self._assert_checkpoint_transparent(
            region_fn, lambda leaf: ((), {"inp": leaf * 3.0})
        )

    def test_numerics_list_input(self) -> None:
        def region_fn(xs: list[torch.Tensor]) -> torch.Tensor:
            return remat.region(_SaveInput.apply, "s", recompute=False)(xs[0])

        self._assert_checkpoint_transparent(
            region_fn, lambda leaf: (([leaf * 3.0],), {})
        )

    def test_numerics_mixed_multi_input(self) -> None:
        # a: dead (save-only), b: live (recompute), scale: non-tensor. Exercises routing a
        # dropped and a kept tensor plus an opaque leaf through one synthetic input op.
        def region_fn(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
            dead = remat.region(_SaveIntermediate.apply, "dead", recompute=False)(a)
            live = remat.region(_RecomputeMul.apply, "live", recompute=True)(b)
            # dead is a SAVE output consumed by the bare add below; flag it to persist.
            remat.recompute_needs_tensor(dead)
            return dead + live * scale

        self._assert_checkpoint_transparent(
            region_fn, lambda leaf: ((leaf * 3.0, leaf * 5.0, 2.0), {})
        )

    def test_retain_graph_double_backward(self) -> None:
        base = torch.randn(16, requires_grad=True)
        ref_leaf = base.detach().clone().requires_grad_(True)
        _save_input(ref_leaf * 3.0).sum().backward()
        assert ref_leaf.grad is not None
        reference = ref_leaf.grad

        leaf = base.detach().clone().requires_grad_(True)
        out = remat.checkpoint(region_name="r")(_save_input)(leaf * 3.0)
        first = torch.autograd.grad(out.sum(), leaf, retain_graph=True)[0]
        second = torch.autograd.grad(out.sum(), leaf)[0]
        self.assertTrue(torch.equal(first, reference))
        self.assertTrue(torch.equal(second, reference))

    def test_requires_grad_leaf_input_kept_and_correct(self) -> None:
        # A requires-grad leaf (e.g. a Parameter) cannot pass through a remat.region, so it
        # takes the _KeptInput bypass -- it must still be fed and differentiated correctly.
        self._assert_checkpoint_transparent(_save_input, lambda leaf: ((leaf,), {}))

    def test_determinism_check_default(self) -> None:
        base = torch.randn(16, requires_grad=True)
        ref_leaf = base.detach().clone().requires_grad_(True)
        _save_input(ref_leaf * 3.0).sum().backward()
        assert ref_leaf.grad is not None
        reference = ref_leaf.grad
        leaf = base.detach().clone().requires_grad_(True)
        out = remat.checkpoint(region_name="r", determinism_check="default")(
            _save_input
        )(leaf * 3.0)
        out.sum().backward()
        assert leaf.grad is not None
        self.assertTrue(torch.equal(leaf.grad, reference))

    def test_memory_report_shows_kept_input_and_omits_dead(self) -> None:
        # A kept (recompute-consumed) input appears as a <region_inputs> op row; a dead
        # (save-only) input does not, so its bytes leave the resident total entirely.
        reports: dict[str, str] = {}

        def keep(x: torch.Tensor) -> torch.Tensor:
            out = remat.region(_RecomputeMul.apply, "r", recompute=True)(x)
            if not remat.is_recomputing():
                reports["keep"] = remat.format_current_memory_report()
            return out

        def drop(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(_SaveIntermediate.apply, "s", recompute=False)(x)
            if not remat.is_recomputing():
                reports["drop"] = remat.format_current_memory_report()
            return y

        for region_fn in (keep, drop):
            base = torch.randn(64, requires_grad=True)
            remat.checkpoint(region_name="blk")(region_fn)(base * 3.0).sum().backward()

        self.assertIn("blk::<region_inputs>", reports["keep"])
        self.assertNotIn("<region_inputs>", reports["drop"])
        _assert_byte_column_sums(self, reports["keep"])
        _assert_byte_column_sums(self, reports["drop"])
