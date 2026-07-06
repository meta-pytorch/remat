# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for ``remat.format_current_memory_report``: how it names a producer's durable
output, folds exact save/output aliases, flags view-pinned storage, names a single
saved view, footnotes rebuilt saves, refuses to fold different shapes on one storage,
flags a released graph, groups by region/op/tensor, and keeps its byte column summing
to the header total."""

from __future__ import annotations

import gc
from typing import Any

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import _assert_byte_column_sums
from torch_remat._region import _checkpoint_context_fn


class MemoryReportTest(expecttest.TestCase):
    def test_memory_report_names_producer_durable_output(self) -> None:
        # A SAVE output consumed by a RECOMPUTE op is durably saved on the producer, so
        # the memory report shows it as the producer's output slot -- not a consumer-side
        # ferried input.
        class ProducerSave(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                del ctx
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        def consumer(t: torch.Tensor) -> torch.Tensor:
            return t * 2

        def region(x: torch.Tensor) -> torch.Tensor:
            y = remat.op(ProducerSave.apply, "producer", policy=remat.SAVE)(x)
            return remat.op(consumer, "consumer", policy=remat.RECOMPUTE)(y)

        forward_context, _ = _checkpoint_context_fn("layers.0")
        with forward_context:
            region(torch.tensor([1.0, 2.0], requires_grad=True))
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
layers.0: 8 B resident in 1 storage(s)
layers.0::producer: 8 B
        8 B  output.0               (2,)       float32  cpu    SAVE""",
            )

    def test_memory_report_folds_exact_alias_save_and_output(self) -> None:
        # A SAVE op that saves its own output: the value is one storage known by two
        # names (the save name ``y`` and the durable slot ``output.0``). They share ptr,
        # shape, stride, and offset, so the report folds them into a single byte-bearing
        # row -- ``y = output.0`` -- rather than two rows plus a shared-storage join. The
        # internal ``gf`` is a separate storage. Rows sum to the 16 B total.
        class Sq(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                gf = x * 2
                remat.save_for_backward(ctx, {"y": y, "gf": gf})
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y, gf) = ctx.saved_tensors
                del y
                return grad_output * gf

        forward_context, _ = _checkpoint_context_fn("layers.0")
        with forward_context:
            # Hold the output so the SAVE saves stay live for the report.
            out = remat.op(Sq.apply, "sq", policy=remat.SAVE)(
                torch.tensor([1.0, 2.0], requires_grad=True)
            )
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
layers.0: 16 B resident in 2 storage(s)
layers.0::sq: 16 B
        8 B  y = output.0           (2,)       float32  cpu    SAVE
        8 B  gf                     (2,)       float32  cpu    SAVE""",
            )
            self.assertEqual((2,), tuple(out.shape))

    def test_memory_report_flags_view_pinned_storage(self) -> None:
        # A SAVE op saves two views of an *internal* tensor whose base object then dies.
        # Only the base's storage survives -- pinned resident by the small saved views --
        # so the report gives it a byte-bearing row named after the views it backs
        # (``base of q, k``), lists the views as children, and shouts the waste ratio:
        # 96 B held for the 64 B the views span.
        class TwoViews(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                big = x.new_ones(2, 12)  # 96 B internal base, not a saved value itself
                q = big[:, 0:4]
                k = big[:, 4:8]
                remat.save_for_backward(ctx, {"q": q, "k": k})
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (q, k) = ctx.saved_tensors
                del q, k
                return grad_output

        forward_context, _ = _checkpoint_context_fn("layers.0")
        with forward_context:
            out = remat.op(TwoViews.apply, "attn", policy=remat.SAVE)(
                torch.ones(2, 3, requires_grad=True)
            )
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
layers.0: 96 B resident in 1 storage(s)
layers.0::attn: 96 B
       96 B  base of q, k                      float32  cpu    SAVE   ! held for 64 B of 96 B
          - q          view (2, 4)     SAVE      spans 32 B
          - k          view (2, 4)     SAVE      spans 32 B""",
            )
            self.assertEqual((2, 3), tuple(out.shape))

    def test_memory_report_names_a_single_saved_view(self) -> None:
        # A single saved slice of a larger buffer -- the common view-save case -- is named
        # after the save itself (``v``), NOT demoted under an "unnamed storage" row. The
        # byte column carries the whole backing storage and the flag reports the slack, so
        # the pinning is still visible without inventing a nameless row for a named tensor.
        class OneSlice(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                big = x.new_ones(10)  # 40 B buffer; only a 16 B slice is saved
                remat.save_for_backward(ctx, {"v": big[0:4]})
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (v,) = ctx.saved_tensors
                del v
                return grad_output

        forward_context, _ = _checkpoint_context_fn("layers.0")
        with forward_context:
            out = remat.op(OneSlice.apply, "op", policy=remat.SAVE)(
                torch.ones(3, requires_grad=True)
            )
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
layers.0: 40 B resident in 1 storage(s)
layers.0::op: 40 B
       40 B  v                      (4,)       float32  cpu    SAVE   ! held for 16 B of 40 B""",
            )
            self.assertEqual((3,), tuple(out.shape))

    def test_memory_report_reports_rebuilt_saves_in_footer(self) -> None:
        # A SAVE op that saves a view of its *input* is not retained -- remat records a
        # rebuild recipe and reproduces the view on recompute. So it holds zero resident
        # bytes and appears only as a non-additive footer line, never in the byte column.
        class SaveInputView(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                v = x[:, 0:4]
                remat.save_for_backward(ctx, {"v": v})
                return (x * 2).sum().reshape(1)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return torch.zeros(2, 12)

        forward_context, _ = _checkpoint_context_fn("layers.0")
        with forward_context:
            out = remat.op(SaveInputView.apply, "proj", policy=remat.SAVE)(
                torch.ones(2, 12, requires_grad=True)
            )
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
layers.0: 0 B resident in 0 storage(s)
  + 1 save rebuilt on recompute, not resident""",
            )
            self.assertEqual((1,), tuple(out.shape))

    def test_memory_report_does_not_fold_different_shapes_on_one_storage(self) -> None:
        # The alias fold keys on ptr+shape+stride+offset, not ptr alone: two values that
        # share a storage at offset 0 but have different shapes are NOT the same value, so
        # they must render as owner-plus-view, never as an ``a = b`` exact-alias chain
        # (which would falsely claim they are interchangeable). Here ``a`` is also the
        # durable output (a true exact alias -> folds to ``a = output.0``) while ``av`` is
        # a reshape of the same storage (different shape -> a view child).
        class TwoShapes(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                a = x * x
                av = a.view(2, 2)  # same storage, offset 0, different shape
                remat.save_for_backward(ctx, {"a": a, "av": av})
                return a

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (a, av) = ctx.saved_tensors
                del a, av
                return grad_output * 2

        forward_context, _ = _checkpoint_context_fn("blk")
        with forward_context:
            out = remat.op(TwoShapes.apply, "op", policy=remat.SAVE)(
                torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
            )
            report = remat.format_current_memory_report()
            self.assertExpectedInline(
                report,
                """\
blk: 16 B resident in 1 storage(s)
blk::op: 16 B
       16 B  a = output.0           (4,)       float32  cpu    SAVE
          - av         view (2, 2)     SAVE      spans 16 B""",
            )
            self.assertEqual((4,), tuple(out.shape))
        # The differently-shaped view is a child, never folded into the ``=`` chain.
        self.assertNotIn("a = av", report)

    def test_memory_report_flags_released_graph(self) -> None:
        # SAVE saves are held weakly through the region output's grad_fn. If that output
        # (and graph) is released while the region is still active, the saves vanish and a
        # naive report would read a misleading bare "0 B". A completed SAVE op with nothing
        # resident and nothing deferred is flagged instead.
        class SaveInternal(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                gf = x * 2  # internal save; the output (x + 1) is not itself saved
                remat.save_for_backward(ctx, {"gf": gf})
                return x + 1

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (gf,) = ctx.saved_tensors
                return grad_output * gf

        forward_context, _ = _checkpoint_context_fn("blk")
        with forward_context:
            out = remat.op(SaveInternal.apply, "op", policy=remat.SAVE)(
                torch.ones(3, requires_grad=True)
            )
            del out  # drop the graph; the weakly-held save is collected
            gc.collect()
            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
blk: 0 B resident in 0 storage(s)
  ! region output no longer alive -- saved tensors already released; report reflects that""",
            )

    def test_memory_report_byte_column_sums_to_header(self) -> None:
        # Fable's invariant as executable code: the header total is the literal sum of the
        # byte-bearing storage rows. Exercised on a folded exact alias (one storage, two
        # names), a plain internal save (a second storage), and a view-pinned storage
        # (nameless row carries the bytes; its view children carry none).
        class Sq(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                gf = x * 2
                remat.save_for_backward(ctx, {"y": y, "gf": gf})
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y, gf) = ctx.saved_tensors
                del y
                return grad_output * gf

        class TwoViews(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                big = x.new_ones(2, 12)
                remat.save_for_backward(ctx, {"q": big[:, 0:4], "k": big[:, 4:8]})
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (q, k) = ctx.saved_tensors
                del q, k
                return grad_output

        forward_context, _ = _checkpoint_context_fn("layers.0")
        with forward_context:
            out_a = remat.op(Sq.apply, "sq", policy=remat.SAVE)(
                torch.tensor([1.0, 2.0], requires_grad=True)
            )
            out_b = remat.op(TwoViews.apply, "attn", policy=remat.SAVE)(
                torch.ones(2, 3, requires_grad=True)
            )
            report = remat.format_current_memory_report()
            _assert_byte_column_sums(self, report)
            del out_a, out_b

    def test_memory_report_groups_by_region_op_and_tensor(self) -> None:
        class Probe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                lse = torch.zeros(3, dtype=torch.float32)
                probs = torch.zeros(4, dtype=torch.float32)
                # Named saves surface as report row keys instead of saved.0/saved.1.
                remat.save_for_backward(ctx, {"lse": lse, "probs": probs})
                return x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        forward_context, _ = _checkpoint_context_fn("layers.0")
        x = torch.tensor([1.0], requires_grad=True)

        with forward_context:
            # Hold the op output: its grad_fn now owns the SAVE saved tensors via
            # autograd (not the remat tape), so the report can see lse/probs only
            # while that graph is alive.
            out = remat.op(
                Probe.apply,
                "attn.softmax",
                policy=remat.SAVE,
            )(x)

            self.assertExpectedInline(
                remat.format_current_memory_report(),
                """\
layers.0: 28 B resident in 2 storage(s)
layers.0::attn.softmax: 28 B
       12 B  lse                    (3,)       float32  cpu    SAVE
       16 B  probs                  (4,)       float32  cpu    SAVE""",
            )
            self.assertEqual((1,), tuple(out.shape))
