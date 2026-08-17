# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""torch.compile support for torch_remat.

Verifies that a compiled ``remat.checkpoint`` region (a) produces gradients bitwise-
identical to the same math run eagerly without checkpointing, and (b) drives the
min-cut partitioner's save/recompute decision from each region's ``recompute`` flag: a
``recompute=False`` (SAVE) region's op is kept in the forward and not recomputed in
backward, a ``recompute=True`` region's op is recomputed in backward. Both plain
functions and custom ``autograd.Function``\\s (with :func:`remat.save_for_backward`)
are covered as region bodies.

The save/recompute check inspects the partitioned forward and backward graphs directly
(via a capturing backend around the min-cut partitioner) and looks at where a marker op
(``aten.exp``) lives, rather than counting backward dispatches: min-cut is free to move
a recomputed op into the forward, so a backward-only count cannot tell "saved" from
"relocated to forward".
"""

from __future__ import annotations

from typing import Any, Callable

import expecttest
import torch
import torch_remat as remat


def _block(x: torch.Tensor, w: torch.Tensor, *, recompute: bool) -> torch.Tensor:
    # matmul recomputes by default (enclosing checkpoint); exp is the marker op whose
    # fate is set by the region's recompute flag.
    a = torch.matmul(x, w)
    e = remat.region(torch.exp, "exp", recompute=recompute)(a)
    return e.sum()


class _ExpFn(torch.autograd.Function):
    """Marker op as a custom autograd.Function: forward runs exp, saves it by name."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        e = torch.exp(x)
        remat.save_for_backward(ctx, {"e": e})
        return e

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> torch.Tensor:
        (e,) = ctx.saved_tensors
        return g * e


def _fn_block(x: torch.Tensor, w: torch.Tensor, *, recompute: bool) -> torch.Tensor:
    a = torch.matmul(x, w)
    e = remat.region(_ExpFn.apply, "exp", recompute=recompute)(a)
    return e.sum()


class CompileTest(expecttest.TestCase):
    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        x = torch.randn(8, 8, requires_grad=True)
        w = torch.randn(8, 8, requires_grad=True)
        return x, w

    def _grads(
        self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, w = self._inputs()
        fn(x, w).backward()
        assert x.grad is not None and w.grad is not None
        return x.grad, w.grad

    def _assert_grads_match_eager(
        self, block: Callable[..., torch.Tensor], recompute: bool
    ) -> None:
        eager_gx, eager_gw = self._grads(lambda x, w: block(x, w, recompute=recompute))
        compiled = torch.compile(
            lambda x, w: remat.checkpoint()(
                lambda a, b: block(a, b, recompute=recompute)
            )(x, w),
            backend="aot_eager",
            fullgraph=True,
        )
        comp_gx, comp_gw = self._grads(compiled)
        self.assertTrue(torch.equal(eager_gx, comp_gx))
        self.assertTrue(torch.equal(eager_gw, comp_gw))

    def test_compiled_grads_match_eager(self) -> None:
        for recompute in (True, False):
            with self.subTest(recompute=recompute):
                self._assert_grads_match_eager(_block, recompute)

    def test_autograd_function_region_grads_match_eager(self) -> None:
        # A region wrapping a custom autograd.Function that uses save_for_backward --
        # the primary region() use case -- compiles and matches eager grads.
        for recompute in (True, False):
            with self.subTest(recompute=recompute):
                self._assert_grads_match_eager(_fn_block, recompute)

    def test_eagerly_constructed_region_wrapper_under_compile(self) -> None:
        # region() built OUTSIDE compile (so it runs eagerly and returns the @wraps
        # wrapper), then invoked inside a compiled checkpoint. This hits the wrapper's
        # own is_compiling() branch, not the top-of-region() branch that inline
        # construction takes -- the two cover different construct-vs-call timings.
        r = remat.region(_ExpFn.apply, "exp", recompute=False)

        def block(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return r(torch.matmul(x, w)).sum()

        eager_gx, eager_gw = self._grads(block)
        compiled = torch.compile(
            lambda x, w: remat.checkpoint()(block)(x, w),
            backend="aot_eager",
            fullgraph=True,
        )
        comp_gx, comp_gw = self._grads(compiled)
        self.assertTrue(torch.equal(eager_gx, comp_gx))
        self.assertTrue(torch.equal(eager_gw, comp_gw))

    def _marker_in_fwd_bwd(
        self, block: Callable[..., torch.Tensor], recompute: bool
    ) -> tuple[int, int]:
        """Return (# exp in partitioned forward, # exp in partitioned backward)."""

        fw_graphs: list[torch.fx.GraphModule] = []
        bw_graphs: list[torch.fx.GraphModule] = []

        def backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
            from functorch.compile import (
                make_boxed_func,
                min_cut_rematerialization_partition,
            )
            from torch._functorch.aot_autograd import aot_module_simplified

            def fw(g: torch.fx.GraphModule, _: Any) -> Any:
                fw_graphs.append(g)
                return make_boxed_func(g.forward)

            def bw(g: torch.fx.GraphModule, _: Any) -> Any:
                bw_graphs.append(g)
                return make_boxed_func(g.forward)

            return aot_module_simplified(
                gm,
                example_inputs,
                fw_compiler=fw,
                bw_compiler=bw,
                partition_fn=min_cut_rematerialization_partition,
            )

        x, w = self._inputs()
        compiled = torch.compile(
            lambda a, b: remat.checkpoint()(
                lambda p, q: block(p, q, recompute=recompute)
            )(a, b),
            backend=backend,
            fullgraph=True,
        )
        compiled(x, w).backward()

        def count(gm: torch.fx.GraphModule) -> int:
            return sum(
                1
                for n in gm.graph.nodes
                if n.op == "call_function" and n.target is torch.ops.aten.exp.default
            )

        return count(fw_graphs[-1]), count(bw_graphs[-1])

    def test_save_region_not_recomputed(self) -> None:
        for block in (_block, _fn_block):
            with self.subTest(block=block.__name__):
                fwd, bwd = self._marker_in_fwd_bwd(block, recompute=False)
                # SAVE: exp is computed once in forward and kept, not recomputed.
                self.assertEqual(fwd, 1)
                self.assertEqual(bwd, 0)

    def test_recompute_region_is_recomputed(self) -> None:
        for block in (_block, _fn_block):
            with self.subTest(block=block.__name__):
                fwd, bwd = self._marker_in_fwd_bwd(block, recompute=True)
                # RECOMPUTE: exp is rematerialized in backward.
                self.assertGreaterEqual(bwd, 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
