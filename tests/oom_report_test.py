# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the out-of-the-box OOM reporting helpers (``torch_remat._oom``).

These run on CPU: the OOM path has no loss handle, so it recovers autograd roots from the
live object graph via ``discover_autograd_roots`` and folds them into the whole-model report.
The helpers are pure Python (no CUDA), so the discovery + report + print composition is
exercisable without a GPU or an actual allocation failure.

``discover_autograd_roots`` sweeps the *whole* live object graph, so the grand total and the
entire ``outside regions`` section depend on tensors other tests leave alive in the process.
``_redact`` masks exactly those volatile pieces so the stable parts of the report can still be
pinned with ``assertExpectedInline``. The report body itself (region enumeration, the graph
walk, byte accounting) is content-tested from explicit roots in ``saved_report_test``; here we
only cover the OOM-specific glue -- gc root discovery and the observer's print/never-raise
contract."""

from __future__ import annotations

import contextlib
import io
import re
from unittest import mock

import expecttest
import torch
import torch_remat as remat
from torch_remat._region import _live_regions


def _redact(report: str) -> str:
    """Mask the parts of a gc-discovered OOM report that vary with unrelated live tensors.

    Kept verbatim: the region-tape summary/detail and the ``N region(s) X B`` region tally
    (all deterministic once the registry is cleared). Masked: the grand total, the header's
    ``outside regions`` byte figure, and the whole ``outside regions`` section body.
    """

    lines = report.splitlines()
    out: list[str] = []
    dropping_outside = False
    for line in lines:
        if line.startswith("saved for backward:"):
            line = re.sub(
                r"saved for backward: \S+ \S+ resident",
                "saved for backward: <total> resident",
                line,
            )
            line = re.sub(r"outside regions \S+ \S+$", "outside regions <bytes>", line)
            out.append(line)
        elif line.startswith("outside regions:"):
            out.append("outside regions: <redacted>")
            dropping_outside = True
        elif dropping_outside:
            if not line.strip():  # blank line ends the outside-regions section
                dropping_outside = False
                out.append(line)
            # else: drop the volatile per-node rows
        else:
            out.append(line)
    return "\n".join(out)


class OomReportTest(expecttest.TestCase):
    def setUp(self) -> None:
        # Only this test's regions should appear in the report (see saved_report_test).
        _live_regions.clear()

    def test_discover_autograd_roots_selects_grad_fn_tensors(self) -> None:
        # A leaf (even one requiring grad) has no grad_fn -- it is a weight/input, not a
        # graph output -- so it is not a root; an intermediate with a grad_fn is. This asserts
        # membership of specific tensor objects in a process-wide gc sweep, so it stays a
        # boolean check (expecttest does not apply to object identity).
        leaf = torch.randn(3, requires_grad=True)
        intermediate = leaf * 2
        no_grad = torch.randn(3)

        ids = {id(t) for t in remat.discover_autograd_roots()}
        self.assertIn(id(intermediate), ids)
        self.assertNotIn(id(leaf), ids)
        self.assertNotIn(id(no_grad), ids)

    def test_discover_autograd_roots_skips_fake_tensors(self) -> None:
        # Under --fake_pg the process is full of FakeTensors carrying grad_fns; feeding them to
        # the walk would explode (fake ops under a dead FakeTensorMode). Discovery must exclude
        # them -- the isinstance(FakeTensor) guard -- while still finding real roots. If that
        # guard regressed, the fake intermediate below would appear as a root.
        from torch._subclasses.fake_tensor import FakeTensorMode

        real = torch.randn(3, requires_grad=True) * 2  # real intermediate -> a root
        with FakeTensorMode():
            fake = torch.randn(3, requires_grad=True) * 2  # fake, has grad_fn
        self.assertTrue(fake.grad_fn is not None)  # would qualify but for the guard

        ids = {id(t) for t in remat.discover_autograd_roots()}
        self.assertIn(id(real), ids)
        self.assertNotIn(id(fake), ids)

    def test_oom_observer_prints_report_and_never_raises(self) -> None:
        # The observer is a best-effort diagnostic on a terminal path: it must print the
        # report (prefixed with the failing request) to stderr and, above all, never raise
        # (the allocator raises the real OOM the instant it returns).
        x = torch.randn(2, 3, requires_grad=True)
        loss = x.tanh().sum()

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            remat.oom_observer(0, 1024, 2048, 512)

        # The report was really produced: the top-level guard (which prints this banner on any
        # failure) must NOT have fired. Without this, a report that silently broke and fell
        # into the guard could still pass a looser check -- exactly the suppression we fear.
        self.assertNotIn("failed to produce", buf.getvalue())
        self.assertExpectedInline(
            _redact(buf.getvalue()),
            """\
CUDA OOM on device 0: requested 1024 B (device_alloc=2048 B, device_free=512 B). torch_remat saved-for-backward report:
saved for backward: <total> resident -- 0 region(s) 0 B, outside regions <bytes>

outside regions: <redacted>""",
        )

        loss.backward()

    def test_oom_observer_surfaces_failure_instead_of_masking_it(self) -> None:
        # The guard's real job, on the failure path: if report production raises, the observer
        # must NOT re-raise (that would mask the real OOM the allocator throws the instant we
        # return) -- yet it must make the breakage LOUD (banner + traceback to stderr) rather
        # than silently yielding no report. This is the one case the happy-path test cannot see.
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("synthetic report failure")

        buf = io.StringIO()
        with (
            contextlib.redirect_stderr(buf),
            mock.patch("torch_remat._oom._print_oom_report", boom),
        ):
            remat.oom_observer(0, 1024, 2048, 512)  # must not raise

        out = buf.getvalue()
        self.assertIn("failed to produce the saved-for-backward report", out)
        self.assertIn("synthetic report failure", out)  # the traceback is surfaced
