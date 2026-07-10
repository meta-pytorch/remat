# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for bare-op detection -- consuming a SAVE op's output outside ``remat.region`` --
across all four strategies (the ``__torch_dispatch__`` subclass, the
``__torch_function__`` proxy, and the dispatch/function modes): a bare op
materializes on recompute, views defer, producer responsibility fills ``output.<i>``
selectively (including a tensor returned at two positions), deferred views outliving
their output still recompute, and the forward-save carriers save on the right
touches."""

from __future__ import annotations

import contextlib
import gc
import weakref
from typing import Any, Callable

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import (
    _BARE_OP_STRATEGIES,
    _ref_grad,
)
from torch_remat._bare_op._common import (
    _SaveOutputHandle,
    _suppress_bare_op_detection,
)
from torch_remat._bare_op._dispatch_mode import _SaveDispatchMode
from torch_remat._bare_op._function_mode import _SaveFunctionMode
from torch_remat._bare_op._proxy import (
    _make_save_proxy,
    _save_proxy_handle,
    _SaveProxy,
)
from torch_remat._bare_op._subclass import (
    _make_save_tensor,
    _SaveTensor,
    _unwrap_save_tensor,
)
from torch_remat._region import (
    _checkpoint_context_fn,
    _CheckpointRegionState,
    _state,
)


class BareOpTest(expecttest.TestCase):
    def test_bare_op_after_save_op_materializes_and_variants(self) -> None:
        """A bare op consuming a SAVE op's output works, materializing on recompute.

        The SAVE output carries real data in the forward, so the bare (unwrapped)
        consumer just runs on it; touching it durably saves the output so recompute
        reproduces the real value -- no remat.region needed on the consumer. The two
        explicit variants (wrap the consumer in a recompute=True region so it ferries
        the saved value, or give the producer recompute=True so its output is real)
        produce the same gradient.
        """

        class SavedMul(torch.autograd.Function):
            @staticmethod
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
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = torch.relu(x)
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * (x > 0).float()

        expected_grad = torch.tensor([2.0, 0.0])

        # Bare op after a SAVE op: the bare relu consumes the SAVE op's placeholder
        # during recompute and materializes it from the taped output -- works with no
        # remat.region on the consumer.
        def body_bare(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(SavedMul.apply, "mul", recompute=False)(x)
            return torch.relu(y)

        # Bare consumption of a SAVE output needs the bare-op detection strategy.
        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                x = torch.tensor([1.0, -1.0], requires_grad=True)
                remat.checkpoint(detect_bare_ops=strategy)(body_bare)(
                    x
                ).sum().backward()
                self.assertTrue(torch.equal(x.grad, expected_grad))

        x = torch.tensor([1.0, -1.0], requires_grad=True)

        # Variant 1: wrap the consumer in a RECOMPUTE op (it ferries the saved value).
        x.grad = None

        def body_recompute_consumer(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(SavedMul.apply, "mul", recompute=False)(x)
            return remat.region(ReluOp.apply, "relu", recompute=True)(y)

        remat.checkpoint()(body_recompute_consumer)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, expected_grad))

        # Variant 2: give the producer RECOMPUTE, so its output is real during replay.
        x.grad = None

        def body_recompute_producer(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(SavedMul.apply, "mul", recompute=True)(x)
            return torch.relu(y)

        remat.checkpoint()(body_recompute_producer)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, expected_grad))

    def test_bare_op_detection_defaults_on_and_opt_out_raises(self) -> None:
        """Bare-op detection is on by default; ``detect_bare_ops=False`` opts back out.

        The default (``"subclass"``) intercepts a bare consumer so it just works, while
        the opt-out leaves SAVE outputs plain and a bare consumer meets the recompute
        placeholder -- the tight prod path for callers who wrap every consumer.
        """

        class SavedMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(SavedMul.apply, "mul", recompute=False)(x)
            return torch.relu(y)  # bare consumer of the SAVE output

        # Default: bare consumer is intercepted, so it just works.
        x = torch.tensor([1.0, -1.0], requires_grad=True)
        remat.checkpoint()(body)(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([2.0, 0.0])))

        # Opt out: the bare consumer meets a placeholder during recompute and raises.
        x_opt = torch.tensor([1.0, -1.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            remat.checkpoint(detect_bare_ops=False)(body)(x_opt).sum().backward()

    def test_bare_view_then_compute_after_save_materializes(self) -> None:
        """A bare view of a SAVE output, then bare compute, materializes on recompute.

        The SAVE output carries real data, so the view and the later compute both just
        run; touching it durably saves the base, and recompute reproduces the base and
        re-runs the chain -- so a reshape + add + relu written outside remat.region works.
        """

        class SavedMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return grad_output * 2

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(SavedMul.apply, "mul", recompute=False)(x)
            # Bare view (delayed) feeding a bare compute op (materialized).
            return torch.relu(y.reshape(4) + 1.0)

        def reference(x: torch.Tensor) -> torch.Tensor:
            return torch.relu((x * 2).reshape(4) + 1.0)

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                x = torch.tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)
                remat.checkpoint(detect_bare_ops=strategy)(body)(x).sum().backward()
                self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, x)))

    def test_bare_op_after_multi_output_save_materializes(self) -> None:
        """A bare op consuming several outputs of one multi-output SAVE op.

        Each output is a forward save stand-in; the bare op touching both durably saves
        each under output.<i>, so recompute reproduces both real values.
        """

        class SplitScale(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                ctx.save_for_backward(x)
                return x * 2, x * 3

            @staticmethod
            def backward(
                ctx: Any, grad_a: torch.Tensor, grad_b: torch.Tensor
            ) -> torch.Tensor:
                del ctx
                return grad_a * 2 + grad_b * 3

        def body(x: torch.Tensor) -> torch.Tensor:
            a, b = remat.region(SplitScale.apply, "split", recompute=False)(x)
            return a + b  # bare add of two SAVE outputs

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                x = torch.tensor([1.0, 2.0], requires_grad=True)
                remat.checkpoint(detect_bare_ops=strategy)(body)(x).sum().backward()
                # d/dx (2x + 3x) = 5
                self.assertTrue(torch.equal(x.grad, torch.tensor([5.0, 5.0])))

    def test_bare_list_arg_op_over_save_outputs(self) -> None:
        """A bare op taking a *list* of SAVE outputs (torch.cat) durably saves each.

        The forward save stand-in is reached through a one-hop list argument, so both
        outputs are saved and recompute reproduces them -- the residual cat works
        outside remat.region.
        """

        class SplitScale(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                ctx.save_for_backward()
                return x * 2, x * 3

            @staticmethod
            def backward(
                ctx: Any, grad_a: torch.Tensor, grad_b: torch.Tensor
            ) -> torch.Tensor:
                del ctx
                return grad_a * 2 + grad_b * 3

        def body(x: torch.Tensor) -> torch.Tensor:
            a, b = remat.region(SplitScale.apply, "split", recompute=False)(x)
            return torch.cat([a, b])  # bare op over a list of two SAVE outputs

        def reference(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x * 2, x * 3])

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                x = torch.tensor([1.0, 2.0], requires_grad=True)
                remat.checkpoint(detect_bare_ops=strategy)(body)(x).sum().backward()
                self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, x)))

    def test_output_saving_is_producer_responsible_and_selective(self) -> None:
        """Only a SAVE output actually touched by a bare op fills output.<i> on its producer.

        Producer responsibility: a bare consumer trips the forward save stand-in's
        interception, which durably saves the producer's output slot (so recompute can
        reproduce it). A remat.region consumer unwraps up front and ferries the value onto
        its *own* record instead, so the producer keeps no output.<i> slot -- the save
        is selective, not producer-eager for every output.
        """

        class Save(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                del ctx
                return x * 2

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                del ctx
                return grad_output * 2

        x = torch.tensor([1.0, 2.0], requires_grad=True)

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                # Bare consumer: durably saves producer output.0.
                forward_context, _ = _checkpoint_context_fn(
                    "r", detect_bare_ops=strategy
                )
                with forward_context:
                    y = remat.region(Save.apply, "producer", recompute=False)(x)
                    torch.relu(
                        y
                    )  # bare touch of the stand-in -> durably saves output.0
                    active = _state.get()
                    assert active is not None
                    producer = active.region_state.records["producer"]
                    self.assertEqual([0], list(producer.output_slots))

                # remat.region consumer: also the producer's responsibility -- the consumer
                # triggers the same durable output.0, and holds no tape state itself.
                forward_context, _ = _checkpoint_context_fn(
                    "r", detect_bare_ops=strategy
                )
                with forward_context:
                    y = remat.region(Save.apply, "producer", recompute=False)(x)
                    remat.region(
                        lambda t: t + 1,
                        "consumer",
                        recompute=True,
                    )(y)
                    active = _state.get()
                    assert active is not None
                    self.assertEqual(
                        [0],
                        list(active.region_state.records["producer"].output_slots),
                    )

    def test_bare_consumer_of_saved_for_backward_output_works_without_detect(
        self,
    ) -> None:
        """A SAVE output that is itself saved for backward is resident, so
        it is eagerly durably saved -- a bare consumer of it works during recompute even
        without ``detect_bare_ops`` (which the same code would need if the output were
        not saved for backward)."""

        class SaveExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                y = torch.exp(x)
                ctx.save_for_backward(y)  # saves the OUTPUT (d exp/dx = exp(x) = y)
                return y

            @staticmethod
            def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
                (y,) = ctx.saved_tensors
                return grad_output * y

        def body(x: torch.Tensor) -> torch.Tensor:
            y = remat.region(SaveExp.apply, "exp", recompute=False)(x)
            return y + 1.0  # bare consumer of the SAVE output, no detect_bare_ops

        def reference(x: torch.Tensor) -> torch.Tensor:
            return torch.exp(x) + 1.0

        # The producer eagerly fills output.0 because the output is saved for backward.
        forward_context, _ = _checkpoint_context_fn("r")
        with forward_context:
            body(torch.tensor([0.5, -0.5], requires_grad=True))
            active = _state.get()
            assert active is not None
            self.assertEqual([0], list(active.region_state.records["exp"].output_slots))

        x = torch.tensor([0.5, -0.5], requires_grad=True)
        remat.checkpoint()(body)(x).sum().backward()
        self.assertTrue(torch.allclose(x.grad, _ref_grad(reference, x)))

    def test_save_op_returning_same_tensor_at_two_positions(self) -> None:
        # A SAVE op may return the *same* tensor object at more than one output position
        # (``return y, y``). The plain-tensor strategies ("none" and the two modes) key the
        # save-output index by the output itself, so both positions collapse onto one key --
        # a naive per-position registration would let position 1's handle shadow position
        # 0's, leaving output.0 with no durable save and a recompute placeholder for a
        # consumer of position 0. The handles must merge so consuming *either* position
        # durably saves the consumed position's slot. Exercised across every strategy,
        # including "none"; the wrapper strategies give each position a distinct carrier and
        # never collide, so this also guards that they keep working.
        class DupProducer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                del ctx
                y = x * 3
                return y, y

            @staticmethod
            def backward(
                ctx: Any, grad0: torch.Tensor, grad1: torch.Tensor
            ) -> torch.Tensor:
                del ctx
                return (grad0 + grad1) * 3

        def consumer(t: torch.Tensor) -> torch.Tensor:
            return t * 2

        def region(x: torch.Tensor) -> torch.Tensor:
            # Consume position 0 (the position historically served a placeholder); leave
            # position 1 dead so the whole guarantee rides on position 0's slot filling.
            first, _second = remat.region(
                DupProducer.apply, "producer", recompute=False
            )(x)
            return remat.region(consumer, "consumer", recompute=True)(first)

        strategies = ("none", *_BARE_OP_STRATEGIES)

        # Introspect the forward: consuming position 0 durably saves output.0 under every
        # strategy, rather than shadowing it onto output.1.
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                forward_context, _ = _checkpoint_context_fn(
                    "r", detect_bare_ops=strategy
                )
                with forward_context:
                    region(torch.tensor([1.0, 2.0], requires_grad=True))
                    active = _state.get()
                    assert active is not None
                    self.assertIn(
                        0,
                        active.region_state.records["producer"].output_slots,
                    )

        # And it round-trips through recompute at backward: consumer computes 2*(3x) = 6x,
        # so d/dx sum(6x) = 6 per element -- for every strategy.
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                x = torch.tensor([1.0, 2.0], requires_grad=True)
                out = remat.checkpoint(region_name="r", detect_bare_ops=strategy)(
                    region
                )(x)
                out.sum().backward()
                self.assertTrue(torch.equal(x.grad, torch.tensor([6.0, 6.0])))

    def test_forward_save_tensor_durably_saves_on_bare_touch(self) -> None:
        """The forward save subclass carries real data (producer responsibility).

        A bare op runs on the real data and returns a plain tensor (one hop), firing
        the producer's durable-save on every touch; the grad-connected unwrap a
        remat.region uses does not save; an in-place op errors -- it would corrupt the
        copy the SAVE op keeps for backward.
        """

        real = torch.tensor([1.0, 2.0], requires_grad=True) * 2
        calls = {"n": 0}

        def persist_output() -> None:
            calls["n"] += 1

        save_tensor = _make_save_tensor(real, persist_output=persist_output)
        self.assertIsInstance(save_tensor, _SaveTensor)

        # Carries real data: data_ptr aliases the source and reads succeed (a recompute
        # placeholder raises on both). data_ptr also counts as a touch (Triton path).
        self.assertEqual(real.data_ptr(), save_tensor.data_ptr())

        # A bare compute op runs on real data and returns a plain tensor (one hop).
        out = torch.sin(save_tensor)
        self.assertNotIsInstance(out, _SaveTensor)
        self.assertTrue(torch.equal(out, torch.sin(real.detach())))
        # A bare view is also plain and one-hop.
        self.assertNotIsInstance(save_tensor.reshape(2, 1), _SaveTensor)

        # Every bare touch fires the (idempotent-in-practice) durable save.
        self.assertGreaterEqual(calls["n"], 3)

        # The remat.region / boundary unwrap is grad-connected and does NOT durable save.
        calls["n"] = 0
        plain = _unwrap_save_tensor(save_tensor)
        self.assertEqual(0, calls["n"])
        self.assertNotIsInstance(plain, _SaveTensor)
        self.assertTrue(torch.equal(plain, real.detach()))

        # In-place mutation is rejected: it would corrupt the durably saved / autograd
        # copies of the SAVE output.
        with self.assertRaisesRegex(RuntimeError, "mutate a SAVE op's output"):
            save_tensor.add_(1.0)

    def test_forward_save_proxy_defers_on_views_and_saves_on_compute(self) -> None:
        """The forward save proxy defers on views and durably saves only when poked hard.

        Unlike the ``__torch_dispatch__`` subclass, the ``__torch_function__`` proxy is
        not a tensor: a *view* (``reshape``, slice) returns a new proxy and does NOT
        fire the producer's durable-save; a real compute, an operator, or ``data_ptr``
        ("poked hard") unwraps to the grad-connected inner, fires the save once, and
        returns a plain result. The handle's unwrap is grad-connected and never saves;
        an in-place op errors.
        """

        real = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True) * 2
        calls = {"n": 0}

        def persist_output() -> None:
            calls["n"] += 1

        proxy = _make_save_proxy(real, persist_output=persist_output)
        self.assertIsInstance(proxy, _SaveProxy)

        # A view returns a NEW proxy and defers the save (the key difference from the
        # subclass, which durably saves on every touch including views).
        view = proxy.reshape(2, 2)
        self.assertIsInstance(view, _SaveProxy)
        self.assertEqual(0, calls["n"])
        # Indexing (getitem) is a view too.
        self.assertIsInstance(proxy[0:2], _SaveProxy)
        self.assertEqual(0, calls["n"])
        # A no-op that returns the SAME tensor (already contiguous) did not mutate it
        # (version unchanged), so it stays a deferred view rather than firing the save.
        self.assertIsInstance(proxy.contiguous(), _SaveProxy)
        self.assertEqual(0, calls["n"])

        # A bare compute op unwraps to the inner, returns a plain tensor, and saves.
        out = torch.sin(proxy)
        self.assertNotIsInstance(out, _SaveProxy)
        self.assertIsInstance(out, torch.Tensor)
        self.assertTrue(torch.equal(out, torch.sin(real.detach())))
        self.assertEqual(1, calls["n"])

        # An operator (the common residual add) is intercepted via the magic-method
        # dunder and pokes hard too.
        calls["n"] = 0
        added = proxy + 1.0
        self.assertNotIsInstance(added, _SaveProxy)
        self.assertTrue(torch.equal(added, real.detach() + 1.0))
        self.assertEqual(1, calls["n"])

        # Poking a deferred view fires the base producer's save (once) and returns plain.
        calls["n"] = 0
        materialized = view + 1.0
        self.assertNotIsInstance(materialized, _SaveProxy)
        self.assertEqual(1, calls["n"])

        # data_ptr aliases the inner (Triton path) and counts as a poke.
        calls["n"] = 0
        self.assertEqual(real.data_ptr(), proxy.data_ptr())
        self.assertEqual(1, calls["n"])

        # A metadata attribute is forwarded without poking.
        calls["n"] = 0
        self.assertEqual(real.shape, proxy.shape)
        self.assertEqual(0, calls["n"])

        # The remat.region / boundary unwrap is grad-connected and does NOT durable save.
        handle = _save_proxy_handle(proxy)
        unwrapped = handle.unwrap(proxy)
        self.assertEqual(0, calls["n"])
        self.assertNotIsInstance(unwrapped, _SaveProxy)
        self.assertTrue(torch.equal(unwrapped, real.detach()))

        # In-place mutation is rejected: it bumps the inner's ``_version``.
        with self.assertRaisesRegex(RuntimeError, "mutate a SAVE op's output"):
            proxy.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "mutate a SAVE op's output"):
            proxy[0] = 5.0

        # An ``out=`` write whose target is NOT the SAVE output does not mutate the
        # proxy -- it is a compute and must be allowed (mutation is keyed on the inner's
        # version bump, not the mere presence of an ``out=`` kwarg).
        calls["n"] = 0
        fresh = _make_save_proxy(
            torch.tensor([1.0, 2.0, 3.0, 4.0]) * 2,
            persist_output=persist_output,
        )
        external = torch.empty(4)
        result = torch.add(fresh, 1.0, out=external)
        self.assertIs(result, external)
        self.assertNotIsInstance(result, _SaveProxy)
        self.assertEqual(1, calls["n"])

    def test_bare_binary_view_over_two_saves_rides_viewed_producer(self) -> None:
        # A bare binary view op over two SAVE outputs -- ``a.view_as(b)`` -- is a view of
        # ``a``; ``b`` is only a shape spectator. The deferred view must ride ``a``'s
        # producer, so recompute rebuilds it from ``a`` (== x*2), not ``b`` (== x*3):
        # riding the wrong operand would silently reproduce the wrong values. The gradient
        # (checked for every strategy) pins that the view rebuilds from ``a``; the spectator
        # ``b`` carrying no durable output slot (checked for the deferring strategies) pins
        # that the view op did not fall to the eager path and over-persist both producers.
        class SaveMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor, k: float) -> torch.Tensor:
                ctx.k = k
                return x * k

            @staticmethod
            def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
                return grad * ctx.k, None

        def region(x: torch.Tensor) -> torch.Tensor:
            a = remat.region(SaveMul.apply, "a", recompute=False)(x, 2.0)
            b = remat.region(SaveMul.apply, "b", recompute=False)(x, 3.0)
            c = a.view_as(b)  # bare binary view: view of `a`, `b` a shape spectator
            return remat.region(torch.mul, "consume", recompute=True)(c, 5.0)

        def reference(x: torch.Tensor) -> torch.Tensor:
            return (x * 2.0) * 5.0

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                x = torch.randn(4, requires_grad=True)
                out = remat.checkpoint(region_name="r", detect_bare_ops=strategy)(
                    region
                )(x)
                out.sum().backward()
                self.assertTrue(
                    torch.allclose(x.grad, _ref_grad(reference, x.detach()))
                )

        # The deferring strategies persist only the viewed producer `a`; the spectator `b`
        # is never touched durably, so its record carries no output slot.
        for strategy in ("proxy", "function_mode"):
            with self.subTest(strategy=strategy):
                forward_context, _ = _checkpoint_context_fn(
                    "r", detect_bare_ops=strategy
                )
                with forward_context:
                    region(torch.randn(4, requires_grad=True))
                    active = _state.get()
                    assert active is not None
                    records = active.region_state.records
                    self.assertEqual([0], list(records["a"].output_slots))
                    self.assertEqual([], list(records["b"].output_slots))

    def test_forward_save_dispatch_mode_saves_on_touch(self) -> None:
        """The dispatch-mode strategy fires the producer's save on any dispatched touch.

        Mirrors the ``__torch_dispatch__`` subclass: SAVE outputs stay plain tensors and the
        installed ``TorchDispatchMode`` fires the producer's durable-save for any op that
        touches one -- views included (no deferral). An in-place op errors before it runs,
        and detection is suppressed inside a ``remat.region``'s own processing. ``data_ptr`` is a
        raw accessor that bypasses ``__torch_dispatch__``, so -- unlike the subclass and the
        function mode -- it is NOT intercepted here.
        """

        region_state = _CheckpointRegionState()
        calls = {"n": 0}

        def persist_output() -> None:
            calls["n"] += 1

        def register(tensor: torch.Tensor) -> torch.Tensor:
            region_state.save_output_index[tensor] = _SaveOutputHandle(
                persist_output=persist_output, unwrap=lambda leaf: leaf
            )
            return tensor

        real = register(torch.tensor([1.0, 2.0], requires_grad=True) * 2)

        with _SaveDispatchMode(region_state):
            # A bare compute fires the save and returns a plain tensor.
            out = torch.sin(real)
            self.assertNotIsInstance(out, _SaveTensor)
            self.assertGreaterEqual(calls["n"], 1)

            # A view fires too -- no deferral (the key contrast with the function mode).
            calls["n"] = 0
            _ = real.reshape(2, 1)
            self.assertGreaterEqual(calls["n"], 1)

            # data_ptr bypasses __torch_dispatch__, so it is not intercepted.
            calls["n"] = 0
            _ = real.data_ptr()
            self.assertEqual(0, calls["n"])

            # Suppressed inside a remat.region's own processing: no save.
            calls["n"] = 0
            with _suppress_bare_op_detection():
                _ = torch.sin(real)
            self.assertEqual(0, calls["n"])

            # In-place is rejected before it runs (real is left unmutated).
            with self.assertRaisesRegex(RuntimeError, "mutate a SAVE op's output"):
                real.add_(1.0)

    def test_inplace_mutation_detection_is_precise(self) -> None:
        """Every strategy rejects only a genuine *write* to a SAVE output, batch_norm included.

        Across all four strategies, three edge cases -- untested before -- behave identically:

        * ``dst.scatter_(src=save)`` with a fresh plain ``dst`` only *reads* the SAVE
          output, so it is allowed and fires the producer's save (a per-argument
          refinement over "any mutable op touching a save").
        * an in-place foreach reaching a SAVE output through its *list* argument is a
          write and is rejected.
        * ``native_batch_norm`` updates ``running_mean`` in place, yet the write is both
          unannotated (``alias_info.is_write`` is False) and version-counter-invisible, so
          neither the plain schema check nor a version bump catches it. The batch-norm
          carve-out does: pre-hoc for the schema strategies (via ``_UNANNOTATED_INPLACE_ARGS``,
          matched by arg name) and post-hoc for the version strategies (via
          ``_UNANNOTATED_INPLACE_FUNC_NAMES``, matched by value change). A SAVE output passed
          as ``running_mean`` is rejected; a SAVE output passed as the (read-only) input is not.

        The pre-hoc strategies reject *before* the op runs, so the value is preserved; the
        post-hoc ones detect the write after it lands (see the ``_PRE_HOC`` assertion).
        """

        _PRE_HOC = ("subclass", "dispatch_mode")

        # Each strategy gets a context to install and a factory that marks a plain tensor as
        # a SAVE output (a wrapper for the subclass/proxy, an index entry for the modes),
        # plus a shared save-fire counter.
        def harness(
            strategy: str,
        ) -> tuple[
            contextlib.AbstractContextManager[Any],
            Callable[[torch.Tensor], torch.Tensor],
            dict[str, int],
        ]:
            calls = {"n": 0}

            def persist_output() -> None:
                calls["n"] += 1

            if strategy == "subclass":
                return (
                    contextlib.nullcontext(),
                    lambda t: _make_save_tensor(t, persist_output=persist_output),
                    calls,
                )
            if strategy == "proxy":
                return (
                    contextlib.nullcontext(),
                    lambda t: _make_save_proxy(t, persist_output=persist_output),
                    calls,
                )

            region_state = _CheckpointRegionState()

            def register(t: torch.Tensor) -> torch.Tensor:
                region_state.save_output_index[t] = _SaveOutputHandle(
                    persist_output=persist_output, unwrap=lambda leaf: leaf
                )
                return t

            mode = (
                _SaveDispatchMode(region_state)
                if strategy == "dispatch_mode"
                else _SaveFunctionMode(region_state)
            )
            return mode, register, calls

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                active, make_save, calls = harness(strategy)
                with active:
                    # Read-only save (scatter src): allowed, and it fires the save.
                    save = make_save(torch.tensor([[5.0, 6.0]]))
                    dst = torch.zeros(1, 2)
                    dst.scatter_(1, torch.tensor([[0, 1]]), save)
                    self.assertGreater(calls["n"], 0)

                    # In-place through a list argument (foreach): a write -> rejected.
                    listed = make_save(torch.tensor([1.0, 2.0]))
                    with self.assertRaisesRegex(
                        RuntimeError, "mutate a SAVE op's output"
                    ):
                        torch._foreach_add_([listed], 1.0)

                    # A SAVE output as the read-only input is fine (input is not mutated).
                    calls["n"] = 0
                    save_input = make_save(torch.randn(2, 3, 4))
                    torch.native_batch_norm(
                        save_input,
                        torch.randn(3),
                        torch.randn(3),
                        torch.zeros(3),
                        torch.ones(3),
                        True,
                        0.1,
                        1e-5,
                    )
                    self.assertGreater(calls["n"], 0)

                    # native_batch_norm mutates running_mean in place though the static
                    # schema hides it and the version counter never bumps -- rejected by the
                    # carve-out. Pre-hoc strategies leave the value unmutated.
                    running_mean = make_save(torch.zeros(3))
                    before = (
                        running_mean.clone().detach() if strategy in _PRE_HOC else None
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "mutate a SAVE op's output"
                    ):
                        torch.native_batch_norm(
                            torch.randn(2, 3, 4),
                            torch.randn(3),
                            torch.randn(3),
                            running_mean,
                            torch.ones(3),
                            True,
                            0.1,
                            1e-5,
                        )
                    if before is not None:
                        self.assertTrue(
                            torch.equal(running_mean.clone().detach(), before)
                        )

    def test_forward_save_function_mode_defers_on_views_and_saves_on_compute(
        self,
    ) -> None:
        """The function-mode strategy defers on views and saves on compute, like the proxy.

        SAVE outputs stay plain tensors; the installed ``TorchFunctionMode`` intercepts every
        torch call. A *view* is deferred -- its outputs are registered in the save-output
        index under the producer's save and no save fires; a real compute, an operator, or
        ``data_ptr`` ("poked hard") fires the save once and returns a plain result. An
        in-place op errors, and detection is suppressed inside a ``remat.region``.
        """

        region_state = _CheckpointRegionState()
        calls = {"n": 0}

        def persist_output() -> None:
            calls["n"] += 1

        def register(tensor: torch.Tensor) -> torch.Tensor:
            region_state.save_output_index[tensor] = _SaveOutputHandle(
                persist_output=persist_output, unwrap=lambda leaf: leaf
            )
            return tensor

        real = register(torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True) * 2)
        # Reference values computed outside the mode -- a ``detach`` under the mode is
        # itself a (deferred) view that would register in the index and skew the counts.
        expected_sin = torch.sin(real.detach())
        expected_add = real.detach() + 1.0

        with _SaveFunctionMode(region_state):
            # A view defers: no save, and the view is itself registered as a SAVE output.
            view = real.reshape(2, 2)
            self.assertEqual(0, calls["n"])
            self.assertIn(view, region_state.save_output_index)
            # Indexing (getitem) is a view too.
            self.assertIn(real[0:2], region_state.save_output_index)
            self.assertEqual(0, calls["n"])

            # A bare compute fires the save once and returns a plain tensor.
            out = torch.sin(real)
            self.assertEqual(1, calls["n"])
            self.assertIsInstance(out, torch.Tensor)
            self.assertNotIn(out, region_state.save_output_index)
            self.assertTrue(torch.equal(out, expected_sin))

            # An operator (the common residual add) pokes hard too.
            calls["n"] = 0
            added = real + 1.0
            self.assertEqual(1, calls["n"])
            self.assertTrue(torch.equal(added, expected_add))

            # Poking a deferred view fires the base producer's save once.
            calls["n"] = 0
            _ = view + 1.0
            self.assertEqual(1, calls["n"])

            # data_ptr flows through __torch_function__ and counts as a poke (unlike the
            # dispatch mode, which cannot see it).
            calls["n"] = 0
            _ = real.data_ptr()
            self.assertEqual(1, calls["n"])

            # Suppressed inside a remat.region's own processing: no save.
            calls["n"] = 0
            with _suppress_bare_op_detection():
                _ = torch.sin(real)
            self.assertEqual(0, calls["n"])

        # In-place is rejected. The version check runs after the op (mirroring the proxy),
        # so use a fresh tensor: the mutation lands before the error is raised.
        fresh = register(torch.tensor([1.0, 2.0]) * 2)
        with _SaveFunctionMode(region_state):
            with self.assertRaisesRegex(RuntimeError, "mutate a SAVE op's output"):
                fresh.add_(1.0)

    def test_deferred_view_outliving_its_output_recomputes(self) -> None:
        # A bare view of a SAVE op output DEFERS the producer's durable save (proxy rewrap /
        # function-mode _defer_view), firing it only when the view is poked. Because
        # _PersistOutputThunk references its output weakly (a dead unconsumed output is not pinned to
        # backward), a deferred view poked AFTER its output object is gone would find a dead
        # weakref and silently no-op -- serving recompute a data-inaccessible placeholder
        # where a correct gradient belonged. This bites when the SAVE output is itself a view
        # of an op-internal tensor: the bare view's storage roots at that internal tensor, so
        # nothing keeps the output object alive. Here the output is never bound to a name (only
        # the bare view survives), so it drops the instant the view is taken; a downstream
        # RECOMPUTE op then consumes the view and backward must recompute it. The deferral
        # retains the tensor it was derived from (_BaseRetainingPersist); pre-fix the deferring
        # strategies raised the placeholder error. The non-deferring strategies (subclass,
        # dispatch_mode) never had the bug and confirm the shared region shape.
        class SaveTransposeView(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                return (x * 2).t()  # output is a view of an op-internal tensor

            @staticmethod
            def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
                return (grad * 2).t()

        def region(x: torch.Tensor) -> torch.Tensor:
            view = remat.region(SaveTransposeView.apply, "save", recompute=False)(x)[
                0:3
            ]
            return remat.region(torch.mul, "consume", recompute=True)(view, 3.0)

        def reference(x: torch.Tensor) -> torch.Tensor:
            return (x * 2).t()[0:3] * 3.0

        for strategy in _BARE_OP_STRATEGIES:
            with self.subTest(strategy=strategy):
                x = torch.randn(4, 4, requires_grad=True)
                out = remat.checkpoint(region_name="r", detect_bare_ops=strategy)(
                    region
                )(x)
                out.sum().backward()
                self.assertTrue(
                    torch.allclose(x.grad, _ref_grad(reference, x.detach()))
                )

    def test_deferred_view_dropped_unused_pins_nothing(self) -> None:
        # The deferred-view base retention (_BaseRetainingPersist) must not become a leak: a bare
        # view created but never consumed, then dropped, must free both the view and the SAVE
        # output it retained. The retention is bounded by the view's lifetime (the view is the
        # weak index key, the retained base is its parent), so a dead view pins nothing.
        class SaveTransposeView(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
                return (x * 2).t()

            @staticmethod
            def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
                return (grad * 2).t()

        for strategy in ("proxy", "function_mode"):
            with self.subTest(strategy=strategy):
                output_ref: weakref.ReferenceType[torch.Tensor] | None = None

                def region(x: torch.Tensor) -> torch.Tensor:
                    nonlocal output_ref
                    y = remat.region(SaveTransposeView.apply, "save", recompute=False)(
                        x
                    )
                    view = y[0:3]  # bare deferred view, created then dropped unused
                    if not remat.is_recomputing():
                        # Reach the real output (the proxy's inner, or the value itself).
                        output_ref = weakref.ref(getattr(y, "_inner", y))
                    del view
                    return x * 5  # region output independent of the SAVE op

                x = torch.randn(4, 4, requires_grad=True)
                out = remat.checkpoint(region_name="r", detect_bare_ops=strategy)(
                    region
                )(x)

                ref = output_ref
                assert ref is not None
                gc.collect()
                # The dropped deferred view released the base it retained.
                self.assertIsNone(ref())

                out.sum().backward()
                self.assertTrue(torch.equal(x.grad, torch.full((4, 4), 5.0)))
