# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Out-of-the-box CUDA out-of-memory (OOM) reporting.

An OOM observer is a bare allocator callback -- it fires with no reference to the loss, so
it cannot seed :func:`torch_remat.format_saved_tensors_report`'s autograd graph walk the
way a normal ``print_saved_tensors_report(loss)`` call would. This module bridges that gap:
:func:`discover_autograd_roots` recovers roots from the live object graph, and
:func:`format_oom_saved_tensors_report` composes them into the report so you get the
non-region tail (output head, loss) as well as the region tapes.

Two entry points, by how much you want to own:

* :func:`oom_observer` -- attach it directly and it prints the report on OOM::

      torch._C._cuda_attach_out_of_memory_observer(torch_remat.oom_observer)

* :func:`format_oom_saved_tensors_report` -- returns the report string so you decide how to
  surface it (log it, write it beside the allocator snapshot, add allocation-site
  annotations). Compose it into your own observer when a bare print is not enough.
"""

from __future__ import annotations

import gc
import sys
from typing import TextIO

import torch
from torch_remat._annotate import memory_snapshot_annotate
from torch_remat._reporting import Annotate
from torch_remat._saved_report import format_saved_tensors_report


def discover_autograd_roots() -> list[torch.Tensor]:
    """Best-effort autograd roots for the saved-for-backward walk when no loss is in hand.

    Seeds the walk from every live tensor with a ``grad_fn`` (a graph output or
    intermediate). Over-complete on purpose -- the walk de-dupes nodes and storages -- so it
    still reaches the non-region tail (output head, loss) that the region tapes don't cover.
    Scanning ``gc`` is heavy, so this is meant for terminal contexts (an OOM handler) where
    there is no loss handle to seed the walk directly; when you do have the loss, pass it to
    :func:`torch_remat.format_saved_tensors_report` instead.
    """

    roots: list[torch.Tensor] = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            if isinstance(obj, torch._subclasses.FakeTensor):
                continue
            if obj.grad_fn is not None:
                roots.append(obj)
        except Exception:  # exotic objects can raise on isinstance / attribute access
            continue
    return roots


def format_oom_saved_tensors_report(annotate: Annotate | None = None) -> str:
    """Return the whole-model saved-for-backward report for a no-loss-handle context.

    A convenience over :func:`torch_remat.format_saved_tensors_report` that discovers its
    own roots via :func:`discover_autograd_roots` (since an OOM observer has no loss to pass).

    Args:
        annotate: Optional per-tensor labeller appended to each region-detail storage row.
            Defaults to :func:`torch_remat.memory_snapshot_annotate` -- the allocation site
            from the CUDA memory snapshot, which an OOM handler almost always wants -- best
            effort: if the snapshot is unavailable (recording off, or no CUDA) annotations are
            silently skipped. Pass an explicit annotator, or a no-op, to override.
    """

    if annotate is None:
        # No CUDA / memory-history recording off -- fall back to an unannotated report.
        try:
            annotate = memory_snapshot_annotate()
        except Exception:
            annotate = None
    return format_saved_tensors_report(
        roots=discover_autograd_roots(), annotate=annotate
    )


def oom_observer(device: int, alloc: int, device_alloc: int, device_free: int) -> None:
    """A ready-to-attach CUDA OOM observer that prints the saved-for-backward report.

    Attach it as the allocator's out-of-memory observer::

        torch._C._cuda_attach_out_of_memory_observer(torch_remat.oom_observer)

    Its signature matches the observer callback torch invokes (device index and the alloc /
    device-allocated / device-free byte counts of the failing request). On OOM it prints
    :func:`format_oom_saved_tensors_report` to ``stderr``.

    Fully guarded: the allocator raises the real OOM the instant this returns, so a failure
    in here must never mask it. For richer handling (writing the report to disk, adding
    allocation-site annotations), compose :func:`format_oom_saved_tensors_report` into your
    own observer instead of attaching this one.
    """

    # A failure here must not re-raise -- that would mask the real OOM the allocator raises
    # the instant we return. But don't hide it either: print it (it lands above the OOM in
    # the log) so a broken diagnostic is visible rather than silently yielding no report.
    try:
        _print_oom_report(device, alloc, device_alloc, device_free)
    except Exception:
        import traceback

        print(
            "torch_remat.oom_observer: failed to produce the saved-for-backward report:",
            file=sys.stderr,
        )
        traceback.print_exc()


def _print_oom_report(
    device: int,
    alloc: int,
    device_alloc: int,
    device_free: int,
    file: TextIO | None = None,
) -> None:
    output = sys.stderr if file is None else file
    output.write(
        f"CUDA OOM on device {device}: requested {alloc} B "
        f"(device_alloc={device_alloc} B, device_free={device_free} B). "
        "torch_remat saved-for-backward report:\n"
    )
    output.write(format_oom_saved_tensors_report())
    output.write("\n")
