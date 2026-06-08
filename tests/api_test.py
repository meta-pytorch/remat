# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

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
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "failing.forward"
                policy = torch_remat.CheckpointPolicy.SAVE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                handle.save_for_backward({"x": x})
                raise RuntimeError("intentional forward failure")

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        caught_exception: RuntimeError | None = None
        try:
            torch_remat.checkpoint(
                region_name="leak.check",
            )(FailingForward.apply)(torch.ones(1, requires_grad=True))
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
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "followup.square"
                policy = torch_remat.CheckpointPolicy.SAVE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                y = x * x
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        x = torch.tensor([2.0], requires_grad=True)
        y = torch_remat.checkpoint(
            region_name="followup.check",
        )(FollowupSquare.apply)(x)
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

    def test_readme_style_save_policy_skips_recompute_body(self) -> None:
        class ReadmeSquare(torch.autograd.Function):
            forward_runs: int = 0
            load_runs: int = 0

            @staticmethod
            def forward(
                ctx: Any,
                x: torch.Tensor,
                op_name: str,
                policy: torch_remat.CheckpointPolicy,
            ) -> torch.Tensor:
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    ReadmeSquare.load_runs += 1
                    assert isinstance(ret, torch.Tensor)
                    self.assertTrue(torch_remat.is_recomputing())
                    self.assert_placeholder(ret, (2,))
                    return ret

                self.assertFalse(torch_remat.is_recomputing())
                x = handle.save_or_load_inputs(x)
                assert isinstance(x, torch.Tensor)
                ReadmeSquare.forward_runs += 1
                y = x * x
                handle.save_for_backward({"x": x, "y": y})
                return handle.record_outputs(y)

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> tuple[torch.Tensor, None, None]:
                (x, y) = ctx.saved_tensors
                del y
                return grad_output * 2 * x, None, None

        x = torch.tensor([2.0, 3.0], requires_grad=True)

        y = torch_remat.checkpoint()(ReadmeSquare.apply)(
            x,
            "readme.square",
            torch_remat.CheckpointPolicy.SAVE,
        )
        y.sum().backward()

        self.assertEqual(1, ReadmeSquare.forward_runs)
        self.assertEqual(1, ReadmeSquare.load_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

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
            def forward(
                ctx: Any,
                x: torch.Tensor,
                op_name: str,
                policy: torch_remat.CheckpointPolicy,
            ) -> torch.Tensor:
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                x = handle.save_or_load_inputs(x)
                assert isinstance(x, torch.Tensor)
                ReadmeSquare.forward_runs += 1
                y = x * x
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> tuple[torch.Tensor, None, None]:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x, None, None

        x = torch.tensor([2.0, 3.0], requires_grad=True)

        y = torch_remat.checkpoint()(ReadmeSquare.apply)(
            x,
            "readme.square",
            torch_remat.CheckpointPolicy.RECOMPUTE,
        )
        y.sum().backward()

        self.assertEqual(2, ReadmeSquare.forward_runs)
        self.assertTrue(torch.equal(x.grad, torch.tensor([4.0, 6.0])))

    def test_handles_are_inert_when_no_inputs_need_grad(self) -> None:
        auto_forward_record_counts: list[int] = []
        manual_record_counts: list[int] = []

        class NoGradInputProbe(torch.autograd.Function):
            @staticmethod
            @torch_remat.auto_forward("x")
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                active_state = _state.get()
                auto_forward_record_counts.append(
                    0
                    if active_state is None
                    else len(active_state.region_state.records)
                )
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                return grad_output * 2

        class ManualNoGradInputProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                handle = torch_remat.get_handle(
                    ctx,
                    "manual.no.grad.input.probe",
                    torch_remat.CheckpointPolicy.SAVE,
                )
                active_state = _state.get()
                manual_record_counts.append(
                    0
                    if active_state is None
                    else len(active_state.region_state.records)
                )
                if (ret := handle.maybe_load_saved()) is not None:
                    return ret
                handle.save_for_backward({"x": x})
                return handle.record_outputs(x * 3)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                return grad_output * 3

        def auto_forward_checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return torch_remat.op(
                NoGradInputProbe.apply,
                "no.grad.input.probe",
                policy=torch_remat.CheckpointPolicy.SAVE,
            )(x)

        def manual_checkpoint_body(x: torch.Tensor) -> torch.Tensor:
            return ManualNoGradInputProbe.apply(x)

        x = torch.tensor([1.0], requires_grad=False)
        y = torch_remat.checkpoint(region_name="auto.no.grad.input")(
            auto_forward_checkpoint_body
        )(x)
        z = torch_remat.checkpoint(region_name="manual.no.grad.input")(
            manual_checkpoint_body
        )(x)

        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)
        self.assertEqual([0], auto_forward_record_counts)
        self.assertEqual([0], manual_record_counts)

    def test_recompute_policy_does_not_retain_original_saved_tensors_after_forward(
        self,
    ) -> None:
        original_saved_ref: weakref.ReferenceType[torch.Tensor] | None = None

        class SavedTensorLifetimeProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                nonlocal original_saved_ref

                handle = torch_remat.get_handle(
                    ctx,
                    "saved.tensor.lifetime",
                    torch_remat.CheckpointPolicy.RECOMPUTE,
                )
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                x = handle.save_or_load_inputs(x)
                assert isinstance(x, torch.Tensor)

                saved_activation = x + 1
                if not torch_remat.is_recomputing():
                    original_saved_ref = weakref.ref(saved_activation)
                handle.save_for_backward({"saved_activation": saved_activation})
                return handle.record_outputs(x * 2)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (saved_activation,) = ctx.saved_tensors
                del saved_activation
                return grad_output * 2

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(SavedTensorLifetimeProbe.apply)(x)

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
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "producer"
                policy = torch_remat.CheckpointPolicy.RECOMPUTE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                x = handle.save_or_load_inputs(x)
                assert isinstance(x, torch.Tensor)
                Producer.runs += 1
                y = x * 3
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class Consumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "consumer"
                policy = torch_remat.CheckpointPolicy.RECOMPUTE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                self.assertEqual(2, x.numel())
                self.assertGreater(x.untyped_storage().nbytes(), 0)
                x = handle.save_or_load_inputs(x)
                assert isinstance(x, torch.Tensor)
                self.assertEqual(2, x.numel())
                self.assertGreater(x.untyped_storage().nbytes(), 0)
                handle.save_for_backward({"x": x})
                if not torch_remat.is_recomputing():
                    self.assertExpectedInline(
                        torch_remat.format_current_memory_report(),
                        """\
torch_remat checkpoint region: inputs
total: 0 B""",
                    )
                return handle.record_outputs(x.sum())

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * torch.ones_like(x)

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            return Consumer.apply(Producer.apply(x))

        x = torch.tensor([1.0, 2.0], requires_grad=True)

        y = torch_remat.checkpoint(
            region_name="inputs",
        )(checkpointed_region)(x)
        y.backward()

        self.assertEqual(2, Producer.runs)

    def test_recompute_policy_must_match_forward_policy(self) -> None:
        class PolicyDrift(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                policy = (
                    torch_remat.CheckpointPolicy.RECOMPUTE
                    if torch_remat.is_recomputing()
                    else torch_remat.CheckpointPolicy.SAVE
                )
                handle = torch_remat.get_handle(ctx, "policy.drift", policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                y = x * x
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        y = torch_remat.checkpoint()(PolicyDrift.apply)(
            torch.ones(1, requires_grad=True)
        )

        with self.assertRaisesRegex(RuntimeError, "Conflicting checkpoint policies"):
            y.sum().backward()

    def test_memory_report_groups_by_region_op_and_tensor(self) -> None:
        class Probe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "attn.softmax"
                policy = torch_remat.CheckpointPolicy.SAVE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                out = torch.zeros(2, dtype=torch.float32)
                lse = torch.zeros(3, dtype=torch.float32)
                probs = torch.zeros(4, dtype=torch.float32)
                handle.save_for_backward({"lse": lse, "probs": probs})
                self.assertExpectedInline(
                    torch_remat.format_current_memory_report(),
                    """\
torch_remat checkpoint region: layers.0
total: 28 B
layers.0::attn.softmax total=28 B
  lse: 12 B shape=(3,) dtype=torch.float32 device=cpu policy=SAVE
  probs: 16 B shape=(4,) dtype=torch.float32 device=cpu policy=SAVE""",
                )
                handle.record_outputs(out)
                return x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        torch_remat.checkpoint(
            region_name="layers.0",
        )(Probe.apply)(torch.tensor([1.0], requires_grad=True))

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
            def forward(
                ctx: Any,
                x: torch.Tensor,
                op_name: str,
                policy: torch_remat.CheckpointPolicy,
            ) -> torch.Tensor:
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    self.assertTrue(torch_remat.is_recomputing())
                    self.assert_placeholder(ret, (2,))
                    return ret

                self.assertFalse(torch_remat.is_recomputing())
                right = x + 1
                handle.save_for_backward(
                    {"left": x, "missing": None, "right": right},
                )
                return handle.record_outputs(right)

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> tuple[torch.Tensor, None, None]:
                saved_tensors = ctx.saved_tensors
                left, missing, right = saved_tensors
                self.assertTrue(torch.equal(left, x.detach()))
                self.assertIsNone(missing)
                self.assertTrue(torch.equal(right, x.detach() + 1))
                return grad_output * (right - left + 1), None, None

        x = torch.tensor([3.0, 4.0], requires_grad=True)
        y = torch_remat.checkpoint()(OptionalSavedTensor.apply)(
            x,
            "optional.save",
            torch_remat.CheckpointPolicy.SAVE,
        )
        y.sum().backward()

        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2.0)))

    def test_save_policy_requires_maybe_load_saved_during_recompute(self) -> None:
        class MissingMaybeLoad(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                handle = torch_remat.get_handle(
                    ctx,
                    "missing.maybe_load",
                    torch_remat.CheckpointPolicy.SAVE,
                )
                y = x * x
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        y = torch_remat.checkpoint()(MissingMaybeLoad.apply)(
            torch.ones(1, requires_grad=True)
        )

        with self.assertRaisesRegex(RuntimeError, "must call maybe_load_saved"):
            y.sum().backward()

    def test_checkpoint_recompute_errors_on_unreleased_tape_entries(self) -> None:
        class UnusedSaveProducer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                handle = torch_remat.get_handle(
                    ctx,
                    "unused.save.producer",
                    torch_remat.CheckpointPolicy.SAVE,
                )
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                handle.save_for_backward({})
                return handle.record_outputs(x + 1)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        class UnusedRecomputeConsumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                handle = torch_remat.get_handle(
                    ctx,
                    "unused.recompute.consumer",
                    torch_remat.CheckpointPolicy.RECOMPUTE,
                )
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                x = handle.save_or_load_inputs(x)
                assert isinstance(x, torch.Tensor)
                handle.save_for_backward({})
                return handle.record_outputs(x * 2)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            y = torch.sin(x)
            UnusedRecomputeConsumer.apply(UnusedSaveProducer.apply(x))
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

    def test_skipped_outputs_preserve_tuple_schema(self) -> None:
        class SchemaProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                policy = torch_remat.CheckpointPolicy.SAVE
                pair_op = torch_remat.get_handle(ctx, "pair.output", policy)
                triple_op = torch_remat.get_handle(ctx, "triple.output", policy)
                if (pair_output := pair_op.maybe_load_saved()) is not None:
                    triple_output = triple_op.maybe_load_saved()
                    assert isinstance(pair_output, tuple)
                    assert isinstance(triple_output, tuple)
                    self.assertTrue(torch_remat.is_recomputing())
                    self.assertEqual(2, len(pair_output))
                    self.assert_placeholder(pair_output[0], (2,))
                    self.assert_placeholder(pair_output[1], (3,))
                    self.assertEqual(3, len(triple_output))
                    self.assert_placeholder(triple_output[0], (4,))
                    self.assert_placeholder(triple_output[1], (5,))
                    self.assert_placeholder(triple_output[2], (6,))
                    return x

                self.assertFalse(torch_remat.is_recomputing())
                pair_op.save_for_backward({})
                pair_op.record_outputs(
                    torch.ones(2),
                    torch.ones(3),
                )
                triple_op.save_for_backward({})
                triple_op.record_outputs(
                    torch.ones(4),
                    torch.ones(5),
                    torch.ones(6),
                )
                return x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        y = torch_remat.checkpoint()(SchemaProbe.apply)(
            torch.ones(2, requires_grad=True)
        )
        y.sum().backward()

    def test_record_outputs_tuple_can_be_returned_from_forward(self) -> None:
        class TupleReturn(torch.autograd.Function):
            forward_runs: int = 0
            load_runs: int = 0

            @staticmethod
            def forward(
                ctx: Any,
                x: torch.Tensor,
                op_name: str,
                policy: torch_remat.CheckpointPolicy,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    TupleReturn.load_runs += 1
                    assert isinstance(ret, tuple)
                    self.assertTrue(torch_remat.is_recomputing())
                    self.assert_placeholder(ret[0], (2,))
                    self.assert_placeholder(ret[1], (2,))
                    return ret

                self.assertFalse(torch_remat.is_recomputing())
                TupleReturn.forward_runs += 1
                left = x * x
                right = x + 1
                handle.save_for_backward({"x": x, "left": left, "right": right})
                output = handle.record_outputs(left, right)
                assert isinstance(output, tuple)
                left_out, right_out = output
                return left_out, right_out

            @staticmethod
            def backward(
                ctx: Any,
                grad_left: torch.Tensor,
                grad_right: torch.Tensor,
            ) -> tuple[torch.Tensor, None, None]:
                x, left, right = ctx.saved_tensors
                del left, right
                return grad_left * 2 * x + grad_right, None, None

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        left, right = torch_remat.checkpoint()(TupleReturn.apply)(
            x,
            "tuple.return",
            torch_remat.CheckpointPolicy.SAVE,
        )
        (left + right).sum().backward()

        self.assertEqual(1, TupleReturn.forward_runs)
        self.assertEqual(1, TupleReturn.load_runs)
        self.assertTrue(torch.equal(left.detach(), torch.tensor([4.0, 9.0])))
        self.assertTrue(torch.equal(right.detach(), torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 7.0])))

    def test_skipped_output_view_replays_as_fresh_zero_storage(self) -> None:
        base = torch.arange(8, dtype=torch.float32)

        class ViewProbe(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, base: torch.Tensor) -> torch.Tensor:
                op_name = "view.output"
                policy = torch_remat.CheckpointPolicy.SAVE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    self.assertTrue(torch_remat.is_recomputing())
                    self.assert_placeholder(ret, (4,))
                    return base.sum()

                self.assertFalse(torch_remat.is_recomputing())
                handle.save_for_backward({"saved": base[:2]})
                handle.record_outputs(base[2:6])
                return base.sum()

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (saved,) = ctx.saved_tensors
                self.assertGreater(saved.untyped_storage().nbytes(), 0)
                return torch.ones(8, dtype=grad_output.dtype) * grad_output

        y = torch_remat.checkpoint()(ViewProbe.apply)(
            base.detach().clone().requires_grad_()
        )
        y.backward()

    def test_skipped_output_view_of_recomputed_tensor_replays_as_zero_storage(
        self,
    ) -> None:
        class Producer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "producer"
                policy = torch_remat.CheckpointPolicy.RECOMPUTE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    self.assert_placeholder(ret, (2,))
                    return x.sum()

                y = x * 3
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 3

        class ViewConsumer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                op_name = "view.consumer"
                policy = torch_remat.CheckpointPolicy.SAVE
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                handle.save_for_backward({})
                return handle.record_outputs(x[:1])

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return torch.cat([grad_output, torch.zeros_like(grad_output)])

        def checkpointed_region(x: torch.Tensor) -> torch.Tensor:
            return ViewConsumer.apply(Producer.apply(x))

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
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                handle = torch_remat.get_handle(
                    ctx,
                    "uncheckpointed.square",
                    torch_remat.CheckpointPolicy.SAVE,
                )
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                ctx.scale = 2
                y = x * x
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

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
        class DuplicateForwardHandle(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                policy = torch_remat.CheckpointPolicy.SAVE
                torch_remat.get_handle(ctx, "duplicate.forward", policy)
                torch_remat.get_handle(ctx, "duplicate.forward", policy)
                return x

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output

        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate torch_remat handle retrieval.*during forward",
        ):
            torch_remat.checkpoint()(DuplicateForwardHandle.apply)(
                torch.ones(1, requires_grad=True)
            )

    def test_duplicate_handle_name_errors_in_recompute(self) -> None:
        class DuplicateRecomputeHandle(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                policy = torch_remat.CheckpointPolicy.RECOMPUTE
                handle = torch_remat.get_handle(ctx, "duplicate.recompute", policy)
                if torch_remat.is_recomputing():
                    torch_remat.get_handle(ctx, "duplicate.recompute", policy)

                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                y = x * x
                handle.save_for_backward({"x": x})
                return handle.record_outputs(y)

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2 * x

        y = torch_remat.checkpoint()(DuplicateRecomputeHandle.apply)(
            torch.ones(1, requires_grad=True)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate torch_remat handle retrieval.*during recompute",
        ):
            y.sum().backward()

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
            def forward(
                ctx: Any,
                x: torch.Tensor,
                op_name: str,
                policy: torch_remat.CheckpointPolicy,
            ) -> torch.Tensor:
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                y = x * x
                saved_activation = x + 1
                nonlocal saved_activation_ref
                if not torch_remat.is_recomputing():
                    saved_activation_ref = weakref.ref(saved_activation)
                handle.save_for_backward({"x": x, "saved_activation": saved_activation})
                return handle.record_outputs(y)

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> tuple[torch.Tensor, None, None]:
                x, saved_activation = ctx.saved_tensors
                del saved_activation
                return grad_output * 2 * x, None, None

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(SavedTensorProbe.apply)(
            x,
            "saved.probe",
            torch_remat.CheckpointPolicy.SAVE,
        )
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
            def forward(
                ctx: Any,
                x: torch.Tensor,
                op_name: str,
                policy: torch_remat.CheckpointPolicy,
            ) -> torch.Tensor:
                handle = torch_remat.get_handle(ctx, op_name, policy)
                if (ret := handle.maybe_load_saved()) is not None:
                    assert isinstance(ret, torch.Tensor)
                    return ret

                y = x * x
                handle.save_for_backward({"x": x, "y": y})
                return handle.record_outputs(y)

            @staticmethod
            def backward(
                ctx: Any,
                grad_output: torch.Tensor,
            ) -> tuple[torch.Tensor, None, None]:
                x, y = ctx.saved_tensors
                del y
                return grad_output * 2 * x, None, None

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch_remat.checkpoint()(RetainGraphSquare.apply)(
            x,
            "retain.square",
            torch_remat.CheckpointPolicy.SAVE,
        )
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
