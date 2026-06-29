# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for torch_remat exercised exclusively through its high-level public
surface: ``auto_forward``, ``op``, ``native_op``, and ``checkpoint``. The
low-level handle API (``get_handle`` / ``maybe_load_saved`` /
``save_or_load_inputs`` / ``record_outputs``) is intentionally not used here.
"""

from __future__ import annotations

import contextlib
import gc
import weakref
from typing import Any, cast

import expecttest
import torch
import torch_remat
from torch.utils._python_dispatch import TorchDispatchMode
from torch_remat._api import (
    _active_op,
    _checkpoint_context_fn,
    _checkpoint_recompute_boundary,
    _is_stub_on_recompute,
    _make_placeholder_tensor,
    _placeholder_message,
    _state,
    _TensorMetadata,
)


def _numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for size in shape:
        numel *= size
    return numel


# Shared execution trace for the wedge test below. The ops and the toy offloader
# append human-readable events here; _run_wedge_model resets it per run and joins
# it into the string the test asserts with assertExpectedInline. _WEDGE_LABEL /
# _WEDGE_POLICY carry the current op's region label + policy into the op's forward
# (auto_forward's forward(ctx, x) does not receive the op_name), set by each
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
    """Minimal stand-in for an activation-offload engine on CPU, wired to the
    remat tape via saved_tensors_hooks. pack records the live tensor (no copy)
    and stashes a backup; a block's tensors are freed when the NEXT block commits
    (the previous group's D2H is done by then); unpack reloads a fresh tensor.

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
        label = _WEDGE_TAGS.get(id(tensor), "<untagged>")
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
    suffix = " (recompute)" if torch_remat.is_recomputing() else ""
    _wedge_log(f"compute {_WEDGE_LABEL} [{_WEDGE_POLICY}]{suffix}")


class _WedgeSq(torch.autograd.Function):
    @staticmethod
    @torch_remat.auto_forward("x")
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        _wedge_compute_log()
        # Save a tape-owned intermediate (d(x*x)/dx), never the region input --
        # the input is PyTorch's checkpoint recompute-input and must not be freed
        # out from under it. Tag the saved tensors so the trace names each pack.
        grad_factor = x * 2
        _WEDGE_TAGS[id(grad_factor)] = f"{_WEDGE_LABEL}.gf"
        y = x * x
        _WEDGE_TAGS[id(y)] = f"{_WEDGE_LABEL}.y"
        ctx.save_for_backward(grad_factor)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (grad_factor,) = ctx.saved_tensors
        return grad_output * grad_factor


class _WedgeRelu(torch.autograd.Function):
    @staticmethod
    @torch_remat.auto_forward("x")
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        _wedge_compute_log()
        # A RECOMPUTE op: its saved mask is regenerated in backward and never
        # reaches the tape, so the trace shows no pack line for any *.mid tensor.
        mask = (x > 0).to(x.dtype)
        _WEDGE_TAGS[id(mask)] = f"{_WEDGE_LABEL}.mask"
        ctx.save_for_backward(mask)
        return torch.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (mask,) = ctx.saved_tensors
        return grad_output * mask


def _wedge_step(  # pyre-ignore[3]
    t: torch.Tensor, label: str, op: Any, policy: torch_remat.CheckpointPolicy
):
    global _WEDGE_LABEL, _WEDGE_POLICY
    _WEDGE_LABEL = label
    _WEDGE_POLICY = policy.name
    return torch_remat.op(op, label, policy=policy)(t)


def _wedge_block_body(prefix: str):  # pyre-ignore[3]
    """A SAVE -> RECOMPUTE -> SAVE wedge: Sq[SAVE] -> Relu[RECOMPUTE] -> Sq[SAVE]."""

    def body(t: torch.Tensor) -> torch.Tensor:
        save = torch_remat.CheckpointPolicy.SAVE
        recompute = torch_remat.CheckpointPolicy.RECOMPUTE
        t = _wedge_step(t, f"{prefix}.in", _WedgeSq.apply, save)
        t = _wedge_step(t, f"{prefix}.mid", _WedgeRelu.apply, recompute)
        t = _wedge_step(t, f"{prefix}.out", _WedgeSq.apply, save)
        return t

    return body


def _run_wedge_model(
    offloader: _WedgeOffloader | None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Run two SAVE->RECOMPUTE->SAVE blocks (each a checkpoint region == one
    offload group). With an offloader installed, route tape saves through it and
    free each block's activations a block late. Returns (loss, x.grad, trace)."""
    global _WEDGE_TRACE
    _WEDGE_TRACE = []
    _WEDGE_TAGS.clear()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    hooks: contextlib.AbstractContextManager[object] = (
        torch_remat.saved_tensors_hooks(offloader.pack, offloader.unpack)
        if offloader is not None
        else contextlib.nullcontext()
    )
    _wedge_log("== forward ==")
    with hooks:
        h = x
        for block_id in range(2):
            block = f"block.{block_id}"
            h = torch_remat.checkpoint(region_name=block)(_wedge_block_body(block))(h)
            if offloader is not None:
                offloader.commit_group(block)  # deferred cleanup of the prior block
    if offloader is not None:
        offloader.flush()
    _wedge_log("== backward ==")
    loss = h.sum()
    loss.backward()
    assert x.grad is not None
    return loss.detach().clone(), x.grad.detach().clone(), "\n".join(_WEDGE_TRACE)


class ApiTest(expecttest.TestCase):
    def assert_placeholder(
        self,
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...],
    ) -> None:
        self.assertEqual(expected_shape, tuple(tensor.shape))
        self.assertEqual(_numel(expected_shape), tensor.numel())
        message = _placeholder_message(cast(Any, tensor))
        self.assertIn("skipped during recompute", message)
        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            torch.sin(tensor)
        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            tensor.data_ptr()
        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            tensor.untyped_storage().data_ptr()

    def test_placeholder_allows_metadata_only_aliasing_ops(self) -> None:
        placeholder = _make_placeholder_tensor(
            _TensorMetadata(
                shape=(2, 3),
                stride=(3, 1),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
            "placeholder source was skipped during recompute",
        )

        detached = placeholder.detach()
        self.assert_placeholder(detached, (2, 3))
        self.assertFalse(detached.requires_grad)

        viewed = placeholder.view(6)
        self.assert_placeholder(viewed, (6,))
        self.assertEqual((1,), viewed.stride())

        transposed = placeholder.t()
        self.assert_placeholder(transposed, (3, 2))
        self.assertEqual((1, 3), transposed.stride())

        sliced = placeholder[:, :2]
        self.assert_placeholder(sliced, (2, 2))
        self.assertEqual((3, 1), sliced.stride())

    def test_placeholder_rejects_data_producing_ops(self) -> None:
        placeholder = _make_placeholder_tensor(
            _TensorMetadata(
                shape=(2, 3),
                stride=(3, 1),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
            "placeholder source was skipped during recompute",
        )

        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            torch.sin(placeholder)

        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            placeholder.clone()

        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            placeholder.add_(1)

    def test_checkpoint_options_do_not_collide_with_user_kwargs(self) -> None:
        def fn(x: torch.Tensor, *, region_name: str) -> torch.Tensor:
            self.assertEqual("user.kwarg", region_name)
            return x * 2

        x = torch.tensor([3.0], requires_grad=True)
        y = torch_remat.checkpoint(region_name="checkpoint.region")(fn)(
            x,
            region_name="user.kwarg",
        )
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0])))

    def test_checkpoint_forward_exception_unwinds_remat_state(self) -> None:
        class FailingForward(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                raise RuntimeError("intentional forward failure")

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        def failing_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                FailingForward.apply,
                "failing.forward",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        caught_exception: RuntimeError | None = None
        try:
            torch_remat.checkpoint(
                region_name="leak.check",
            )(failing_body)(torch.ones(1, requires_grad=True))
        except RuntimeError as exc:
            caught_exception = exc

        self.assertIsNotNone(caught_exception)

        with self.assertRaisesRegex(
            RuntimeError,
            "No active torch_remat checkpoint region",
        ):
            torch_remat.format_current_memory_report()

        class FollowupSquare(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def followup_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                FollowupSquare.apply,
                "followup.square",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([2.0], requires_grad=True)
        y = torch_remat.checkpoint(
            region_name="followup.check",
        )(followup_body)(x)
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0])))

    def test_native_op_save_does_not_rerun_native_op(self) -> None:
        # This one needs dispatch-level counting, not an invocation counter:
        # under SAC the wrapped callable is still re-invoked during recompute,
        # but the cached output is served so aten.sin never actually re-runs.
        # Only counting real dispatches shows it stayed at 1.
        class CountSinMode(TorchDispatchMode):
            def __init__(self) -> None:
                self.sin_calls = 0

            def __torch_dispatch__(
                self,
                func: Any,
                types: tuple[type[Any], ...],
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                del types
                if str(func) == "aten.sin.default":
                    self.sin_calls += 1
                return func(*args, **({} if kwargs is None else kwargs))

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.native_op(
                torch.sin,
                "native.sin",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)
            return y * y

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        mode = CountSinMode()

        with mode:
            y = torch_remat.checkpoint()(checkpoint_body)(x)
            self.assertEqual(1, mode.sin_calls)
            y.sum().backward()

        self.assertEqual(1, mode.sin_calls)

    def test_native_op_recompute_is_inert(self) -> None:
        # A RECOMPUTE native op simply re-invokes the wrapped callable during
        # recompute (no SAC short-circuit), so counting invocations is enough to
        # show it reran -- no need to observe dispatch. (The trace cannot show
        # this: it is only recorded during the original forward.)
        sin_calls = 0

        def counting_sin(x: torch.Tensor) -> torch.Tensor:
            nonlocal sin_calls
            sin_calls += 1
            return torch.sin(x)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.native_op(
                counting_sin,
                "native.sin",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(x)
            return y * y

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)
        self.assertEqual(1, sin_calls)
        y.sum().backward()

        # RECOMPUTE is inert: the native op is rerun during recompute, exactly
        # as it would be without any native_op annotation.
        self.assertEqual(2, sin_calls)

    def test_collect_trace_records_original_forward_annotations(self) -> None:
        def scope_body(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.native_op(
                torch.sin,
                "native.sin",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)
            return torch_remat.op(
                torch.cos,
                "custom.cos",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(y)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.trace_scope(
                scope_body,
                "scope",
                metadata="test_flag",
            )(x)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch_remat.collect_trace() as trace:
            y = torch_remat.checkpoint()(checkpoint_body)(x)
            y.sum().backward()

        self.assertExpectedInline(
            trace.format(),
            """\
torch_remat trace
scope [test_flag]
  native.sin: native SAVE
  custom.cos: RECOMPUTE""",
        )

    def test_checkpoint_forces_recompute_before_inner_custom_backward(self) -> None:
        events: list[str] = []

        class Inner(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                events.append("inner_forward")
                ctx.save_for_backward(x)
                return x * 3

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                events.append("inner_backward_before_unpack")
                (x,) = ctx.saved_tensors
                del x
                events.append("inner_backward_after_unpack")
                return grad_output * 3

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            events.append("checkpoint_body")
            return Inner.apply(x)

        y = torch_remat.checkpoint()(checkpoint_body)(
            torch.tensor([1.0, 2.0], requires_grad=True)
        )
        y.sum().backward()

        self.assertExpectedInline(
            "\n".join(events),
            """\
checkpoint_body
inner_forward
checkpoint_body
inner_forward
inner_backward_before_unpack
inner_backward_after_unpack""",
        )

    def test_checkpoint_boundary_saves_zero_element_trigger(self) -> None:
        packed_numels: list[int] = []
        packed_nbytes: list[int] = []
        unpacked_numels: list[int] = []
        unpacked_nbytes: list[int] = []

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            packed_numels.append(tensor.numel())
            packed_nbytes.append(tensor.untyped_storage().nbytes())
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            unpacked_numels.append(tensor.numel())
            unpacked_nbytes.append(tensor.untyped_storage().nbytes())
            return tensor

        x = torch.ones(1024, requires_grad=True)
        output = x * 3
        self.assertGreater(output.untyped_storage().nbytes(), 0)

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = _checkpoint_recompute_boundary(output)
            y.sum().backward()

        self.assertEqual([0], packed_numels)
        self.assertEqual([0], packed_nbytes)
        self.assertEqual([0], unpacked_numels)
        self.assertEqual([0], unpacked_nbytes)
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 3)))

    def test_checkpoint_boundary_rejects_unsupported_output_schema(self) -> None:
        class TensorTuple(tuple):
            pass

        x = torch.ones(1, requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "must return a Tensor"):
            torch_remat.checkpoint()(lambda value: TensorTuple((value,)))(x)

        with self.assertRaisesRegex(RuntimeError, "must return a Tensor"):
            torch_remat.checkpoint()(lambda value: (value, None))(x)

    def test_checkpoint_boundary_supports_exact_nested_builtin_containers(
        self,
    ) -> None:
        def fn(
            x: torch.Tensor,
        ) -> dict[str, Any]:
            return {
                "left": x * 2,
                "nested": [x * 3, (x * 4,)],
            }

        x = torch.tensor([2.0], requires_grad=True)
        output = torch_remat.checkpoint()(fn)(x)
        loss = output["left"].sum()
        nested = output["nested"]
        assert isinstance(nested, list)
        loss = loss + nested[0].sum()
        nested_tuple = nested[1]
        assert isinstance(nested_tuple, tuple)
        loss = loss + nested_tuple[0].sum()
        loss.backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([9.0])))

    def test_phase_helpers_report_forward_and_recompute(self) -> None:
        forward_context, recompute_context = _checkpoint_context_fn()

        with forward_context:
            self.assertFalse(torch_remat.is_recomputing())

        with recompute_context:
            self.assertTrue(torch_remat.is_recomputing())

    def test_op_function_style_supports_kwargs_and_restores_context(self) -> None:
        def scale(
            x: torch.Tensor,
            *,
            factor: float,
        ) -> torch.Tensor:
            self.assertIsNotNone(_active_op.get())
            return x * factor

        x = torch.tensor([2.0])
        y = torch_remat.op(
            scale,
            "function.style.scale",
            policy=torch_remat.CheckpointPolicy.RECOMPUTE,
        )(x, factor=3.0)

        self.assertTrue(torch.equal(y, torch.tensor([6.0])))
        self.assertIsNone(_active_op.get())

        def fail() -> None:
            self.assertIsNotNone(_active_op.get())
            raise RuntimeError("intentional failure")

        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            torch_remat.op(
                fail,
                "function.style.failure",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )()

        self.assertIsNone(_active_op.get())

    def test_op_rejects_context_manager_style(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "op expects a function as its first argument",
        ):
            torch_remat.op(
                cast(Any, "context.style"),
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )

    def test_op_function_style_auto_forward_save_policy(self) -> None:
        class FunctionStyleSquare(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if torch_remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                FunctionStyleSquare.forward_runs += 1
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                FunctionStyleSquare.apply,
                "function.style.square",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(1, FunctionStyleSquare.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_fire_at_tape_save_and_load_not_recompute(
        self,
    ) -> None:
        # A SAVE-policy op stores tensors on the remat tape. saved_tensors_hooks
        # must fire pack once per saved tensor in the original forward, fire
        # unpack once per saved tensor when recompute reads them back, and must
        # NOT fire pack again during recompute. A custom pack/unpack pair that
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
            @staticmethod
            @torch_remat.auto_forward("x", "y")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x, y) = ctx.saved_tensors
                del y
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                Square.apply,
                "sq",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with torch_remat.saved_tensors_hooks(pack, unpack):
            y = torch_remat.checkpoint()(checkpoint_body)(x)
            y.sum().backward()

        # pack fires for x and y in the original forward only (not on recompute).
        self.assertEqual(2, len(pack_shapes))
        # unpack fires for x and y when recompute reads the tape back.
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
            @staticmethod
            @torch_remat.auto_forward("x", "y")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x, y) = ctx.saved_tensors
                del y
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                Square.apply,
                "sq",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        # Pack inside the hook scope; recompute/backward runs OUTSIDE it.
        with torch_remat.saved_tensors_hooks(pack, unpack):
            y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        # unpack still ran (bound to the slot at pack time) despite no active hook.
        self.assertEqual(["bound", "bound"], unpack_calls)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_saved_tensors_hooks_offload_through_save_recompute_save_wedge(
        self,
    ) -> None:
        # The shape that actually motivated tape-level hooks, in miniature for
        # readers without access to a real offloader. The model is two "blocks"
        # (each a checkpoint region == one offload group), and inside every block
        # a SAVE -> RECOMPUTE -> SAVE wedge of ops on x = [1.0, 2.0]:
        #
        #   block.k:  in = Sq[SAVE]  ->  mid = Relu[RECOMPUTE]  ->  out = Sq[SAVE]
        #
        # where Sq(t) = t*t (saves the gradient factor 2*t) and Relu(t) = relu(t)
        # (saves its 0/1 mask). A toy offload engine is wired to the remat tape
        # via saved_tensors_hooks; we install ONE scope around the whole forward
        # (the tape binds each unpack to its slot, so -- unlike autograd's blanket
        # default hooks, which a real engine must install per-layer INSIDE the
        # checkpoint -- a single install suffices). Each block's offloaded storage
        # is freed a block late (the next block's commit frees the previous group)
        # and reloaded in backward.
        #
        # Working the forward by hand (x = [1, 2], so relu is the identity here):
        #   block.0.in  Sq[SAVE]:   y0 = x*x = [1, 4]   -> tape: 2*x = [2, 4] (gf),
        #                           and y0 = [1, 4] (its output feeds the RECOMPUTE
        #                           op, so the tape must keep it for recompute)
        #   block.0.mid Relu[RECOMPUTE]: relu([1,4]) = [1, 4]; its mask is NOT
        #                           taped (it is regenerated in backward)
        #   block.0.out Sq[SAVE]:   y1 = [1, 16] -> tape: 2*[1,4] = [2, 8] (gf only;
        #                           the block output is not taped)
        #   block.1 repeats on [1, 16] -> ... -> [1, 256] -> [1, 65536]
        #
        # So exactly the SAVE ops reach the offloader (four grad_factors + the two
        # *.in outputs); nothing named *.mid is ever packed. Rather than assert
        # those facts piecemeal, we capture the engine's event trace and pin it
        # whole: the absence of any "pack ... *.mid" line IS the proof that
        # RECOMPUTE intermediates never reach the tape, and the compute lines show
        # SAVE bodies are skipped on recompute while RECOMPUTE bodies rerun.
        #
        # Read the trace below as the worked execution. In the forward, every
        # pack line names a SAVE tensor (*.gf or the *.in output) -- there is no
        # "pack ... *.mid" line, which IS the proof that the RECOMPUTE op's saved
        # tensors never reach the tape. Each block's tensors are freed a block
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
  unpack t3 = block.1.in.gf
  unpack t4 = block.1.in.y
compute block.1.mid [RECOMPUTE] (recompute)
  unpack t5 = block.1.out.gf
  unpack t0 = block.0.in.gf
  unpack t1 = block.0.in.y
compute block.0.mid [RECOMPUTE] (recompute)
  unpack t2 = block.0.out.gf""",
        )

        # Every offloaded activation's storage was freed before backward ran, yet
        # the recompute above reloaded each one through unpack, so the offloaded
        # run is bitwise-identical to the no-offload baseline.
        self.assertTrue(
            all(t.untyped_storage().nbytes() == 0 for t in offloader.originals)
        )
        self.assertTrue(torch.equal(base_loss, off_loss))
        self.assertTrue(torch.equal(base_grad, off_grad))

    def test_auto_forward_save_policy_restores_saved_output_view(self) -> None:
        class SavesOutputView(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            @torch_remat.auto_forward(
                "query",
                "key",
                "value",
                "attn_vis",
                "out",
                "softmax_lse",
            )
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> torch.Tensor:
                if torch_remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                SavesOutputView.forward_runs += 1
                query = x
                key = x + 1
                value = x + 2
                attn_vis = torch.ones_like(x)
                raw_out = (x * x).view(2, 2)
                out = raw_out.view_as(x)
                softmax_lse = raw_out.sum(dim=1)
                ctx.save_for_backward(
                    query,
                    key,
                    value,
                    attn_vis,
                    out,
                    softmax_lse,
                )
                return out

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                query, key, value, attn_vis, out, softmax_lse = ctx.saved_tensors
                del query, key, value, attn_vis
                self.assertFalse(_is_stub_on_recompute(out))
                self.assertFalse(_is_stub_on_recompute(softmax_lse))
                self.assertGreater(out.contiguous().data_ptr(), 0)
                self.assertGreater(softmax_lse.data_ptr(), 0)
                return grad_output * out.contiguous().view_as(grad_output)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                SavesOutputView.apply,
                "saves.output",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(1, SavesOutputView.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([1.0, 4.0, 9.0, 16.0])))

    def test_readme_style_recompute_policy_reruns_body(self) -> None:
        class ReadmeSquare(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ReadmeSquare.forward_runs += 1
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                ReadmeSquare.apply,
                "readme.square",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(2, ReadmeSquare.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_handles_are_inert_when_no_inputs_need_grad(self) -> None:
        record_counts: list[int] = []

        class NoGradInputProbe(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                active_state = _state.get()
                record_counts.append(
                    0
                    if active_state is None
                    else len(active_state.region_state.records)
                )
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                return grad_output * 2

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                NoGradInputProbe.apply,
                "no.grad.input.probe",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([1.0], requires_grad=False)
        y = torch_remat.checkpoint(region_name="auto.no.grad.input")(checkpoint_body)(x)

        self.assertFalse(y.requires_grad)
        self.assertEqual([0], record_counts)

    def test_recompute_policy_does_not_retain_original_saved_tensors_after_forward(
        self,
    ) -> None:
        original_saved_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class SavedTensorLifetimeProbe(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("saved_activation")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                nonlocal original_saved_ref

                saved_activation = x + 1
                if not torch_remat.is_recomputing():
                    original_saved_ref = weakref.ref(saved_activation)
                ctx.save_for_backward(saved_activation)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (saved_activation,) = ctx.saved_tensors
                del saved_activation
                return grad_output * 2

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                SavedTensorLifetimeProbe.apply,
                "saved.tensor.lifetime",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)

        saved_ref = original_saved_ref
        self.assertIsNotNone(saved_ref)
        assert saved_ref is not None
        gc.collect()
        self.assertIsNone(saved_ref())

        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 2.0])))

    def test_save_or_load_inputs_does_not_retain_recomputed_producer_output(
        self,
    ) -> None:
        class Producer(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            @torch_remat.auto_forward("x")
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
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                test_case.assertEqual(2, x.numel())
                test_case.assertGreater(x.untyped_storage().nbytes(), 0)
                ctx.save_for_backward(x)
                if not torch_remat.is_recomputing():
                    test_case.assertExpectedInline(
                        torch_remat.format_current_memory_report(),
                        """\
torch_remat checkpoint region: inputs
total: 0 B""",
                    )
                return x.sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * torch.ones_like(x)

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                Producer.apply,
                "producer",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(x)
            return torch_remat.op(
                Consumer.apply,
                "consumer",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(y)

        x = torch.tensor([1.0, 2.0], requires_grad=True)

        y = torch_remat.checkpoint(
            region_name="inputs",
        )(checkpointed_region)(x)
        y.backward()

        self.assertEqual(2, Producer.runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([3.0, 3.0])))

    def test_recompute_policy_must_match_forward_policy(self) -> None:
        class PolicyDrift(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def run(x: torch.Tensor) -> torch.Tensor:
            policy = (
                torch_remat.CheckpointPolicy.RECOMPUTE
                if torch_remat.is_recomputing()
                else torch_remat.CheckpointPolicy.SAVE
            )
            return torch_remat.op(PolicyDrift.apply, "policy.drift", policy=policy)(x)

        y = torch_remat.checkpoint()(run)(torch.ones(1, requires_grad=True))

        with self.assertRaisesRegex(RuntimeError, "Conflicting checkpoint policies"):
            y.sum().backward()

    def test_memory_report_groups_by_region_op_and_tensor(self) -> None:
        class Probe(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("lse", "probs")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                lse = torch.zeros(3, dtype=torch.float32)
                probs = torch.zeros(4, dtype=torch.float32)
                ctx.save_for_backward(lse, probs)
                return x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        forward_context, _ = _checkpoint_context_fn("layers.0")
        x = torch.tensor([1.0], requires_grad=True)

        with forward_context:
            torch_remat.op(
                Probe.apply,
                "attn.softmax",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

            self.assertExpectedInline(
                torch_remat.format_current_memory_report(),
                """\
torch_remat checkpoint region: layers.0
total: 28 B
layers.0::attn.softmax total=28 B
  lse: 12 B shape=(3,) dtype=torch.float32 device=cpu policy=SAVE
  probs: 16 B shape=(4,) dtype=torch.float32 device=cpu policy=SAVE""",
            )

    def test_native_memory_report_observes_live_sac_outputs(self) -> None:
        from torch.utils.checkpoint import SelectiveCheckpointContext

        has_op_output = hasattr(
            SelectiveCheckpointContext(is_recompute=False), "op_output"
        )

        forward_context, _ = _checkpoint_context_fn("native.report")
        x = torch.tensor([1.0, 2.0], requires_grad=True)

        with forward_context:
            y = torch_remat.native_op(
                torch.exp, "native.exp", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            self.assertEqual((2,), tuple(y.shape))

            report = torch_remat.format_current_memory_report()
            if has_op_output:
                self.assertExpectedInline(
                    report,
                    """\
torch_remat checkpoint region: native.report
total: 8 B
native.report::native.exp total=8 B
  aten.exp.default#0: 8 B shape=(2,) dtype=torch.float32 device=cpu source=native aliases=out.0""",
                )

    def test_save_preserves_none_saved_tensor_slots(self) -> None:
        class OptionalSavedTensor(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("left", "missing", "right")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if torch_remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                right = x + 1
                ctx.save_for_backward(x, None, right)
                return right

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                left, missing, right = ctx.saved_tensors
                self.assertIsNone(missing)
                return grad_output * (right - left + 1)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                OptionalSavedTensor.apply,
                "optional.save",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([3.0, 4.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2.0)))

    def test_checkpoint_recompute_errors_on_unreleased_tape_entries(self) -> None:
        class UnusedSaveProducer(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward()
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward()
                return x + 1

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        class UnusedRecomputeConsumer(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward()
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward()
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            y = torch.sin(x)
            producer = torch_remat.op(
                UnusedSaveProducer.apply,
                "unused.save.producer",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)
            torch_remat.op(
                UnusedRecomputeConsumer.apply,
                "unused.recompute.consumer",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(producer)
            return y

        y = torch.utils.checkpoint.checkpoint(
            checkpointed_region,
            torch.ones(2, requires_grad=True),
            use_reentrant=False,
            context_fn=lambda: _checkpoint_context_fn("early.stop"),
        )

        with self.assertRaises(RuntimeError) as cm:
            y.sum().backward()

        self.assertExpectedInline(
            str(cm.exception),
            """\
torch_remat checkpoint region early.stop finished recompute with unreleased remat tape entries.
This usually means forward recorded saved tensors for an op that was not executed during recompute, often because PyTorch checkpoint early-stop skipped work that backward did not need.
Unreleased records:
  early.stop::unused.save.producer: policy=SAVE output_placeholders=1
  early.stop::unused.recompute.consumer: policy=RECOMPUTE saved_tensors=input.0(shape=(2,) dtype=torch.float32 device=cpu)
Retained memory report:
torch_remat checkpoint region: early.stop
total: 8 B
early.stop::unused.recompute.consumer total=8 B
  input.0: 8 B shape=(2,) dtype=torch.float32 device=cpu policy=RECOMPUTE aliases=observed_output.out""",
        )

    def test_record_outputs_tuple_can_be_returned_from_forward(self) -> None:
        class TupleReturn(torch.autograd.Function):
            forward_runs: int = 0

            @staticmethod
            @torch_remat.auto_forward("x", "left", "right")
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if torch_remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                TupleReturn.forward_runs += 1
                left = x * x
                right = x + 1
                ctx.save_for_backward(x, left, right)
                return left, right

            @staticmethod
            def backward(
                ctx: Any,
                grad_left: torch.Tensor,
                grad_right: torch.Tensor,
            ) -> torch.Tensor:
                x, left, right = ctx.saved_tensors
                del left, right
                return grad_left * 2 * x + grad_right

        x = torch.tensor([2.0, 3.0], requires_grad=True)

        def checkpoint_body(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch_remat.op(
                TupleReturn.apply,
                "tuple.return",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(value)

        left, right = torch_remat.checkpoint()(checkpoint_body)(x)
        (left + right).sum().backward()

        self.assertEqual(1, TupleReturn.forward_runs)
        self.assertTrue(torch.equal(left.detach(), torch.tensor([4.0, 9.0])))
        self.assertTrue(torch.equal(right.detach(), torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 7.0])))

    def test_skipped_output_view_of_recomputed_tensor_replays_as_zero_storage(
        self,
    ) -> None:
        class Producer(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 3
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class ViewConsumer(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward()
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward()
                return x[:1]

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return torch.cat([grad_output, torch.zeros_like(grad_output)])

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            produced = torch_remat.op(
                Producer.apply,
                "producer",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(x)
            return torch_remat.op(
                ViewConsumer.apply,
                "view.consumer",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(produced)

        y = torch_remat.checkpoint()(checkpointed_region)(
            torch.ones(2, requires_grad=True)
        )

        y.sum().backward()

    def test_remat_handle_works_without_checkpoint(self) -> None:
        packed_shapes: list[tuple[int, ...]] = []

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            packed_shapes.append(tuple(tensor.shape))
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        class UncheckpointedSquare(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.scale = 2
                y = x * x
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * ctx.scale * x

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            y = UncheckpointedSquare.apply(x)
            y.sum().backward()

        self.assertTrue(torch.equal(y.detach(), torch.tensor([4.0, 9.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))
        self.assertEqual([(2,)], packed_shapes)

    def test_duplicate_handle_name_errors_in_forward(self) -> None:
        class FirstDuplicate(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        class SecondDuplicate(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                FirstDuplicate.apply,
                "duplicate.forward",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)
            return torch_remat.op(
                SecondDuplicate.apply,
                "duplicate.forward",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(y)

        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate torch_remat handle retrieval.*during forward",
        ):
            torch_remat.checkpoint()(checkpoint_body)(torch.ones(1, requires_grad=True))

    def test_auto_forward_saves_named_tensors_and_validates_names(self) -> None:
        class AutoSquare(torch.autograd.Function):
            runs: int = 0

            @staticmethod
            @torch_remat.auto_forward("x", "y")
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> torch.Tensor:
                AutoSquare.runs += 1
                y = x * x
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                x, y = ctx.saved_tensors
                del y
                return grad_output * 2 * x

        x = torch.tensor([2.0, 3.0], requires_grad=True)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                AutoSquare.apply,
                "auto.square",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertEqual(1, AutoSquare.runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

        class BadAutoSquare(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x", "y")
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                del ctx
                return grad_output

        def bad_checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                BadAutoSquare.apply,
                "auto.bad",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        with self.assertRaisesRegex(
            RuntimeError,
            "auto_forward save_for_backward names must match",
        ):
            torch_remat.checkpoint()(bad_checkpoint_body)(
                torch.ones(1, requires_grad=True)
            )

    def test_auto_forward_rejects_duplicate_saved_tensor_names(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate auto_forward save_for_backward name: x",
        ):
            torch_remat.auto_forward("x", "x")

    def test_auto_forward_forwards_ctx_attribute_writes(self) -> None:
        class AutoCtxScale(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(
                ctx: Any,
                x: torch.Tensor,
            ) -> torch.Tensor:
                ctx.scale = 5
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * ctx.scale * x

        x = torch.tensor([2.0, 3.0], requires_grad=True)

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                AutoCtxScale.apply,
                "auto.ctx.scale",
                policy=torch_remat.CheckpointPolicy.RECOMPUTE,
            )(x)

        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([10.0, 15.0])))

    def test_save_policy_checkpoint_releases_saved_tensors_by_default(self) -> None:
        saved_activation_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class SavedTensorProbe(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x", "saved_activation")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                if torch_remat.is_recomputing():
                    raise AssertionError("SAVE replay must skip the forward body")

                y = x * x
                saved_activation = x + 1
                nonlocal saved_activation_ref
                saved_activation_ref = weakref.ref(saved_activation)
                ctx.save_for_backward(x, saved_activation)
                return y

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                x, saved_activation = ctx.saved_tensors
                del saved_activation
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                SavedTensorProbe.apply,
                "saved.probe",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

        activation_ref = saved_activation_ref
        self.assertIsNotNone(activation_ref)
        assert activation_ref is not None
        gc.collect()
        self.assertIsNone(activation_ref())

    def test_save_policy_checkpoint_retain_graph(self) -> None:
        class RetainGraphSquare(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x", "y")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * x
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> torch.Tensor:
                x, y = ctx.saved_tensors
                del y
                return grad_output * 2 * x

        def checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                RetainGraphSquare.apply,
                "retain.square",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(checkpoint_body)(x)
        y.sum().backward(retain_graph=True)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

        x.grad = None
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_native_op_after_save_op_errors_on_recompute(self) -> None:
        """A bare native op consuming a SAVE op's placeholder must error.

        Also verifies the three proposed fixes from the error message:
        (1) native_op SAVE, (2) custom autograd Function, (3) RECOMPUTE.
        """

        class SavedMul(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        class ReluOp(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = torch.relu(x)
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * (x > 0).float()

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        expected_grad = torch.tensor([2.0, 0.0])

        # Bare native op after SAVE op: errors.
        def body_bare(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                SavedMul.apply, "mul", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return torch.relu(y)

        y = torch_remat.checkpoint()(body_bare)(x)
        with self.assertRaisesRegex(RuntimeError, "native_op"):
            y.sum().backward()

        # Fix 1: wrap native op in native_op with SAVE policy.
        x.grad = None

        def body_native_save(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                SavedMul.apply, "mul", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return torch_remat.native_op(
                torch.relu, "relu", policy=torch_remat.CheckpointPolicy.SAVE
            )(y)

        torch_remat.checkpoint()(body_native_save)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, expected_grad))

        # Fix 2: move native op into a custom autograd Function.
        x.grad = None

        def body_custom_op(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                SavedMul.apply, "mul", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return torch_remat.op(
                ReluOp.apply, "relu", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(y)

        torch_remat.checkpoint()(body_custom_op)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, expected_grad))

        # Fix 3: change upstream op's policy to RECOMPUTE.
        x.grad = None

        def body_recompute(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                SavedMul.apply, "mul", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(x)
            return torch.relu(y)

        torch_remat.checkpoint()(body_recompute)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, expected_grad))

    def test_native_op_recompute_loads_save_op_output(self) -> None:
        """A RECOMPUTE native op can consume a SAVE op output and recompute.

        The SAVE producer is skipped during recompute (its output replays as a
        placeholder), but the native op's real input is saved in the original
        forward and loaded back, so the native op reruns on real data with
        gradients flowing to the producer.
        """

        # The native op is RECOMPUTE, so during recompute its wrapped callable is
        # re-invoked -- counting invocations is enough (no dispatch mode needed).
        # The SAVE op short-circuits before its body, so its counter stays at 1.
        calls = {"mul": 0, "relu": 0}

        def counting_relu(x: torch.Tensor) -> torch.Tensor:
            calls["relu"] += 1
            return torch.relu(x)

        class SavedMul(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                calls["mul"] += 1
                y = x * 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                SavedMul.apply, "mul", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return torch_remat.native_op(
                counting_relu, "relu", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(y)

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        out = torch_remat.checkpoint()(body)(x)
        # Forward runs the SAVE op body and the native op once each.
        self.assertEqual({"mul": 1, "relu": 1}, calls)
        out.sum().backward()

        # The SAVE op body is not rerun during recompute, but the RECOMPUTE
        # native op is rerun on the loaded real input.
        self.assertEqual({"mul": 1, "relu": 2}, calls)
        self.assertTrue(torch.equal(out.detach(), torch.tensor([2.0, 0.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))

    def test_native_op_recompute_loads_mixed_pytree_inputs(self) -> None:
        """RECOMPUTE native op with a placeholder input, a real input, a kwarg.

        Only the SAVE-produced input replays as a placeholder and is saved and
        bridged back; the recomputed input flows through directly, and the
        non-tensor ``alpha`` keyword survives the pytree round-trip.
        """

        class SavedDouble(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            saved = torch_remat.op(
                SavedDouble.apply, "double", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            recomputed = x * 3
            # add(saved, recomputed, alpha=2) = 2x + 2 * 3x = 8x
            return torch_remat.native_op(
                torch.add, "add", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(saved, recomputed, alpha=2.0)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = torch_remat.checkpoint()(body)(x)
        out.sum().backward()

        self.assertTrue(torch.equal(out.detach(), torch.tensor([8.0, 16.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([8.0, 8.0])))

    def test_native_op_save_multiple_outputs(self) -> None:
        """A SAVE native op may return a tuple of tensors.

        Backward triggers recompute, where SAC serves both cached outputs, so
        correct output + gradient also exercises the tuple-output replay path.
        """

        def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x * 3

        def body(x: torch.Tensor) -> torch.Tensor:
            a, b = torch_remat.native_op(
                split, "native.split", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return a + b

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = torch_remat.checkpoint()(body)(x)
        out.sum().backward()

        # a + b = 2x + 3x = 5x
        self.assertTrue(torch.equal(out.detach(), torch.tensor([5.0, 10.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))

    def test_native_op_recompute_multiple_outputs(self) -> None:
        """A RECOMPUTE native op may return a tuple of tensors and is rerun."""

        calls = 0

        def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            return x * 2, x * 3

        def body(x: torch.Tensor) -> torch.Tensor:
            a, b = torch_remat.native_op(
                split, "native.split", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(x)
            return a + b

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = torch_remat.checkpoint()(body)(x)
        self.assertEqual(1, calls)
        out.sum().backward()

        # Rerun during recompute, and the tuple output is correct end to end.
        self.assertEqual(2, calls)
        self.assertTrue(torch.equal(out.detach(), torch.tensor([5.0, 10.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))

    def test_native_op_recompute_retain_graph(self) -> None:
        """A RECOMPUTE native op survives backward(retain_graph=True) + replay."""

        class SavedMul(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            y = torch_remat.op(
                SavedMul.apply, "mul", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return torch_remat.native_op(
                torch.relu, "relu", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(y)

        x = torch.tensor([1.0, -1.0], requires_grad=True)
        out = torch_remat.checkpoint()(body)(x)

        out.sum().backward(retain_graph=True)
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))

        # The remat tape is preserved under retain_graph, so a second backward
        # recomputes again and produces the same gradient.
        x.grad = None
        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))

    def test_native_op_recompute_loads_two_save_op_outputs(self) -> None:
        """A RECOMPUTE native op can restore several placeholder inputs at once."""

        class SaveScale(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = x * 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            a = torch_remat.op(
                SaveScale.apply, "a", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            b = torch_remat.op(
                SaveScale.apply, "b", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            # Both inputs replay as placeholders; add must restore input.0 and
            # input.1 from the tape.
            return torch_remat.native_op(
                torch.add, "add", policy=torch_remat.CheckpointPolicy.RECOMPUTE
            )(a, b)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        out = torch_remat.checkpoint()(body)(x)
        out.sum().backward()

        # a + b = 2x + 2x = 4x; grad flows back through both producers.
        self.assertTrue(torch.equal(out.detach(), torch.tensor([4.0, 8.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 4.0])))

    def test_native_op_outside_checkpoint_runs_plainly(self) -> None:
        """Outside a checkpoint region, native_op just runs the function."""

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        saved = torch_remat.native_op(
            torch.sin, "s", policy=torch_remat.CheckpointPolicy.SAVE
        )(x)
        recomputed = torch_remat.native_op(
            torch.cos, "r", policy=torch_remat.CheckpointPolicy.RECOMPUTE
        )(x)

        self.assertTrue(torch.allclose(saved, x.sin()))
        self.assertTrue(torch.allclose(recomputed, x.cos()))

        (saved.sum() + recomputed.sum()).backward()
        self.assertTrue(torch.allclose(x.grad, x.cos() - x.sin()))

    def test_native_op_rejects_bad_arguments(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expects a function"):
            torch_remat.native_op(
                123,  # type: ignore[arg-type]
                "native.bad",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )
        with self.assertRaisesRegex(RuntimeError, "expects an op_name"):
            torch_remat.native_op(torch.sin, policy=torch_remat.CheckpointPolicy.SAVE)
        with self.assertRaisesRegex(RuntimeError, "CheckpointPolicy"):
            torch_remat.native_op(
                torch.sin,
                "native.bad",
                policy="SAVE",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            torch_remat.native_op(
                torch.sin, "", policy=torch_remat.CheckpointPolicy.SAVE
            )

    def test_native_op_duplicate_name_errors(self) -> None:
        def body(x: torch.Tensor) -> torch.Tensor:
            a = torch_remat.native_op(
                torch.sin, "dup", policy=torch_remat.CheckpointPolicy.SAVE
            )(x)
            return torch_remat.native_op(
                torch.cos, "dup", policy=torch_remat.CheckpointPolicy.SAVE
            )(a)

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Duplicate torch_remat op name"):
            torch_remat.checkpoint()(body)(x)
