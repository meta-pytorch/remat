# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared, non-test helpers for the torch_remat API test suite. Not collected by pytest
(the filename does not match ``*_test.py``). Holds the byte-report parsing and
column-sum invariant, the placeholder assertion, the ``_ref_grad`` reference, the
``_BARE_OP_STRATEGIES`` matrix, and the two worked activation-offload engines (the
fine-grained wedge and the coarse bulk offloader) that the saved-tensors-hooks tests
drive."""

from __future__ import annotations

import contextlib
import re
from typing import Any, Callable, cast

import expecttest
import torch
import torch_remat as remat
from torch_remat._placeholder import _placeholder_message


def _numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for size in shape:
        numel *= size
    return numel


# The four opt-in bare-op detection strategies. Behavioral bare-op tests run under all
# of them via ``self.subTest(strategy=...)``: the ``__torch_dispatch__`` tensor subclass
# (:mod:`torch_remat._bare_op._subclass`), the ``__torch_function__`` proxy
# (:mod:`torch_remat._bare_op._proxy`), and their mode analogues -- the ``TorchDispatchMode``
# (``"dispatch_mode"``) and ``TorchFunctionMode`` (``"function_mode"``,
# :mod:`torch_remat._bare_op._function_mode`). For a bare op consuming a SAVE output passed to
# it as an argument -- what these behavioral tests exercise -- all four produce identical
# observable behavior (gradients, tape slots); their internals are covered separately. They are
# NOT identical for a SAVE output consumed *inside* a ``remat.op`` body via closure capture: the
# wrapper strategies (subclass, proxy) catch it, the modes (which are suppressed for the whole
# ``remat.op`` body) miss it and raise a placeholder error during recompute. See the
# ``_suppress_bare_op_detection`` note in :mod:`torch_remat._bare_op._common`.
_BARE_OP_STRATEGIES: tuple[str, ...] = (
    "subclass",
    "proxy",
    "dispatch_mode",
    "function_mode",
)


# Shared execution trace for the wedge test below. The ops and the toy offloader
# append human-readable events here; _run_wedge_model resets it per run and joins
# it into the string the test asserts with assertExpectedInline. _WEDGE_LABEL /
# _WEDGE_POLICY carry the current op's region label + policy into the op's forward
# (the wrapped forward(ctx, x) does not receive the op_name), set by each
# _wedge_step right before the op runs -- safe because execution is synchronous.
_WEDGE_TRACE: list[str] = []
_WEDGE_LABEL: str = ""
_WEDGE_POLICY: str = ""
# Maps id(tensor) -> human label for tensors the ops save, so pack can name each
# packed tensor in the trace. Keyed by id (not a tensor attribute) to avoid
# B009/B010; safe because every saved tensor is still alive when it is packed.
_WEDGE_TAGS: dict[int, str] = {}


def _wedge_log(message: str) -> None:
    _WEDGE_TRACE.append(message)


class _WedgeOffloader:
    """Minimal stand-in for an activation-offload engine on CPU, wired in via remat
    saved_tensors_hooks. pack records the live tensor (no copy) and stashes a
    backup; a block's tensors are freed when the NEXT block commits (the previous
    group's D2H is done by then); unpack reloads a fresh tensor.

    Used by test_saved_tensors_hooks_offload_through_save_recompute_save_wedge.
    """

    def __init__(self) -> None:
        self.backups: list[torch.Tensor] = []
        self.originals: list[torch.Tensor] = []
        self.labels: list[str] = []  # per-tag wedge_tag, for legible trace lines
        self.pending: list[int] = []  # tags packed in the current block
        self.committed: list[int] = []  # tags from the last committed block

    def pack(self, tensor: torch.Tensor) -> object:
        tag = len(self.backups)
        # Key by storage, not object identity: a producer's durable save offloads a
        # *detached* snapshot of the output (shares storage with the real value but is a
        # distinct Python object), so id() would miss the tag.
        label = _WEDGE_TAGS.get(tensor.untyped_storage().data_ptr(), "<untagged>")
        self.labels.append(label)
        self.backups.append(tensor.detach().clone())
        self.originals.append(tensor)
        self.pending.append(tag)
        _wedge_log(f"  pack t{tag} = {label}")
        return tag

    def _free(self, tags: list[int]) -> None:
        for tag in tags:
            self.originals[tag].untyped_storage().resize_(0)

    def commit_group(self, block: str) -> None:
        freed = "[" + ", ".join(f"t{tag}" for tag in self.committed) + "]"
        _wedge_log(f"  commit {block}: free {freed}")
        self._free(self.committed)
        self.committed, self.pending = self.pending, []

    def flush(self) -> None:
        freed = "[" + ", ".join(f"t{tag}" for tag in self.committed) + "]"
        _wedge_log(f"  flush: free {freed}")
        self._free(self.committed)
        self.committed = []

    def unpack(self, packed: object) -> torch.Tensor:
        tag = cast(int, packed)
        _wedge_log(f"  unpack t{tag} = {self.labels[tag]}")
        return self.backups[tag].clone()


def _wedge_compute_log() -> None:
    suffix = " (recompute)" if remat.is_recomputing() else ""
    _wedge_log(f"compute {_WEDGE_LABEL} [{_WEDGE_POLICY}]{suffix}")


class _WedgeSq(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        _wedge_compute_log()
        # Save an intermediate (d(x*x)/dx), never the region input -- the input is
        # PyTorch's checkpoint recompute-input and must not be freed out from under
        # it. Tag the saved tensors so the trace names each pack.
        grad_factor = x * 2
        _WEDGE_TAGS[grad_factor.untyped_storage().data_ptr()] = f"{_WEDGE_LABEL}.gf"
        y = x * x
        _WEDGE_TAGS[y.untyped_storage().data_ptr()] = f"{_WEDGE_LABEL}.y"
        ctx.save_for_backward(grad_factor)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (grad_factor,) = ctx.saved_tensors
        return grad_output * grad_factor


class _WedgeRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        _wedge_compute_log()
        # A RECOMPUTE op: its saved mask is regenerated in backward and is never
        # offloaded, so the trace shows no pack line for any *.mid tensor.
        mask = (x > 0).to(x.dtype)
        _WEDGE_TAGS[mask.untyped_storage().data_ptr()] = f"{_WEDGE_LABEL}.mask"
        ctx.save_for_backward(mask)
        return torch.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (mask,) = ctx.saved_tensors
        return grad_output * mask


def _wedge_step(  # pyre-ignore[3]
    t: torch.Tensor, label: str, op: Any, policy: remat.CheckpointPolicy
):
    global _WEDGE_LABEL, _WEDGE_POLICY
    _WEDGE_LABEL = label
    _WEDGE_POLICY = policy.name
    return remat.op(op, label, policy=policy)(t)


def _wedge_block_body(prefix: str):  # pyre-ignore[3]
    """A SAVE -> RECOMPUTE -> SAVE wedge: Sq[SAVE] -> Relu[RECOMPUTE] -> Sq[SAVE]."""

    def body(t: torch.Tensor) -> torch.Tensor:
        save = remat.SAVE
        recompute = remat.RECOMPUTE
        t = _wedge_step(t, f"{prefix}.in", _WedgeSq.apply, save)
        t = _wedge_step(t, f"{prefix}.mid", _WedgeRelu.apply, recompute)
        t = _wedge_step(t, f"{prefix}.out", _WedgeSq.apply, save)
        return t

    return body


def _run_wedge_model(
    offloader: _WedgeOffloader | None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Run two SAVE->RECOMPUTE->SAVE blocks (each a checkpoint region == one
    offload group). With an offloader installed, route its saves through it and
    free each block's activations a block late. Returns (loss, x.grad, trace)."""
    global _WEDGE_TRACE
    _WEDGE_TRACE = []
    _WEDGE_TAGS.clear()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    hooks: contextlib.AbstractContextManager[object] = (
        remat.saved_tensors_hooks(offloader.pack, offloader.unpack)
        if offloader is not None
        else contextlib.nullcontext()
    )
    _wedge_log("== forward ==")
    with hooks:
        h = x
        for block_id in range(2):
            block = f"block.{block_id}"
            h = remat.checkpoint(region_name=block)(_wedge_block_body(block))(h)
            if offloader is not None:
                offloader.commit_group(block)  # deferred cleanup of the prior block
    if offloader is not None:
        offloader.flush()
    _wedge_log("== backward ==")
    loss = h.sum()
    loss.backward()
    assert x.grad is not None
    return loss.detach().clone(), x.grad.detach().clone(), "\n".join(_WEDGE_TRACE)


class _BulkOffloader:
    """Coarse, batched stand-in for an activation-offload engine -- the bulk
    counterpart to _WedgeOffloader. Where the wedge acts on each tensor as pack
    sees it, a bulk offloader defers the copies and moves a whole group at once:

    * ``pack`` only *records* the tensor into the current group; it copies
      nothing.
    * ``offload_group`` runs ONE device->host copy for the whole group and frees
      every device storage in it -- the analogue of a real engine fusing a
      layer's saves into a single ``torch._foreach_copy_`` on a side stream and
      then reclaiming the source blocks.
    * ``onload_group`` runs ONE host->device copy for the whole group at the
      *start* of that group's recompute, before any of its unpacks fire.

    That last point is why a bulk engine must key off ``remat.is_recomputing()``
    inside the region body (behavioral difference #1 in ``saved_tensors_hooks``):
    a saved *output* is unpacked during recompute -- before backward begins -- so
    an onload staged from an autograd function would arrive too late for the
    group's first tensor. ``unpack`` then just hands back the already-onloaded
    tensor; the packed-slot load path does not re-check versions, so returning a
    fresh device tensor is fine.

    Simplified for a CPU test: "device -> host" is ``detach().clone()``, freeing
    device memory is ``untyped_storage().resize_(0)``, and there are no real
    streams or events. Used by
    test_saved_tensors_hooks_bulk_offload_group_onloads_at_recompute_start.
    """

    def __init__(self) -> None:
        self.current: str = ""
        self.originals: dict[str, list[torch.Tensor]] = {}
        self.labels: dict[str, list[str]] = {}  # per-slot label, for the trace
        self.host: dict[str, list[torch.Tensor]] = {}  # host-side backups
        self.onloaded: dict[str, list[torch.Tensor]] = {}  # reloaded device copies

    def begin_group(self, block: str) -> None:
        self.current = block
        self.originals[block] = []
        self.labels[block] = []

    def pack(self, tensor: torch.Tensor) -> object:
        block = self.current
        index = len(self.originals[block])
        # Key by storage, not object identity: a producer's durable save offloads
        # a *detached* snapshot of the output (shares storage, distinct object).
        label = _WEDGE_TAGS.get(tensor.untyped_storage().data_ptr(), "<untagged>")
        self.labels[block].append(label)
        self.originals[block].append(tensor)
        _wedge_log(f"  pack {label}")
        return (block, index)

    def offload_group(self, block: str) -> None:
        originals = self.originals[block]
        self.host[block] = [t.detach().clone() for t in originals]
        for t in originals:
            t.untyped_storage().resize_(0)
        _wedge_log(f"offload {block}: D2H {len(originals)} tensors, free device")

    def onload_group(self, block: str) -> None:
        host = self.host[block]
        self.onloaded[block] = [t.clone() for t in host]
        _wedge_log(f"onload {block}: H2D {len(host)} tensors")

    def all_originals(self) -> list[torch.Tensor]:
        return [t for group in self.originals.values() for t in group]

    def unpack(self, packed: object) -> torch.Tensor:
        block, index = cast(tuple[str, int], packed)
        _wedge_log(f"  unpack {self.labels[block][index]}")
        return self.onloaded[block][index]


def _bulk_block_body(offloader: _BulkOffloader | None, block: str):  # pyre-ignore[3]
    """The wedge block body, prefixed with a recompute-time bulk onload: at the
    start of this group's recompute, bring the whole group back to device before
    any saved-output unpack fires (behavioral difference #1)."""

    inner = _wedge_block_body(block)

    def body(t: torch.Tensor) -> torch.Tensor:
        if offloader is not None and remat.is_recomputing():
            offloader.onload_group(block)
        return inner(t)

    return body


def _run_bulk_model(
    offloader: _BulkOffloader | None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Like _run_wedge_model, but the offloader batches per group: pack records,
    offload_group flushes the whole group's D2H after each region forward, and
    the region body reloads the group in one H2D at recompute start."""
    global _WEDGE_TRACE
    _WEDGE_TRACE = []
    _WEDGE_TAGS.clear()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    hooks: contextlib.AbstractContextManager[object] = (
        remat.saved_tensors_hooks(offloader.pack, offloader.unpack)
        if offloader is not None
        else contextlib.nullcontext()
    )
    _wedge_log("== forward ==")
    with hooks:
        h = x
        for block_id in range(2):
            block = f"block.{block_id}"
            if offloader is not None:
                offloader.begin_group(block)
            h = remat.checkpoint(region_name=block)(_bulk_block_body(offloader, block))(
                h
            )
            if offloader is not None:
                offloader.offload_group(block)  # one bulk D2H per group
    _wedge_log("== backward ==")
    loss = h.sum()
    loss.backward()
    assert x.grad is not None
    return loss.detach().clone(), x.grad.detach().clone(), "\n".join(_WEDGE_TRACE)


def _ref_grad(
    fn: Callable[[torch.Tensor], torch.Tensor], base: torch.Tensor
) -> torch.Tensor:
    """Return the gradient of ``fn(x).sum()`` with no checkpointing."""

    x = base.detach().clone().requires_grad_(True)
    fn(x).sum().backward()
    assert x.grad is not None
    return x.grad


_BYTE_UNITS: dict[str, int] = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}


def _parse_bytes(text: str) -> int:
    """Parse a leading ``_format_bytes`` figure (``"96 B"``, ``"1.50 KiB"``) to bytes."""

    match = re.match(r"\s*([\d.]+)\s*(B|KiB|MiB|GiB)", text)
    assert match is not None, f"no byte figure in {text!r}"
    return round(float(match.group(1)) * _BYTE_UNITS[match.group(2)])


def _assert_byte_column_sums(test: expecttest.TestCase, report: str) -> None:
    """The report's core invariant, as executable code.

    Every storage row carries a byte figure and each storage appears once, so the header
    total must equal the sum of the byte-bearing rows. View children (tree glyphs), op
    headers (start at column 0), and footer lines (``+``/``!``) carry no bytes and are
    excluded -- if any of them leaked into the byte column the totals would diverge.
    """

    header, *rest = report.splitlines()
    header_total = _parse_bytes(header.split(":", 1)[1])
    row_total = 0
    for line in rest:
        if not line.startswith("  "):  # op header / region header
            continue
        stripped = line.strip()
        if stripped[:1] in ("+", "!", "-"):  # footer / warning / view child
            continue
        row_total += _parse_bytes(stripped)
    test.assertEqual(header_total, row_total)


def assert_placeholder(
    test: expecttest.TestCase,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    test.assertEqual(expected_shape, tuple(tensor.shape))
    test.assertEqual(_numel(expected_shape), tensor.numel())
    message = _placeholder_message(cast(Any, tensor))
    test.assertIn("skipped during recompute", message)
    with test.assertRaisesRegex(RuntimeError, "skipped during recompute"):
        torch.sin(tensor)
    with test.assertRaisesRegex(RuntimeError, "skipped during recompute"):
        tensor.data_ptr()
    with test.assertRaisesRegex(RuntimeError, "skipped during recompute"):
        tensor.untyped_storage().data_ptr()
