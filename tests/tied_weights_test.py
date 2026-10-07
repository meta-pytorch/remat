# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for weights shared across several uses: a weight and its transpose (tied
embedding / unembedding), closed over or passed as a region argument, under every
SAVE / RECOMPUTE policy pairing, repeated inside one region or checkpoint, shared by two
checkpoints, used both outside and inside a checkpoint, frozen, and backpropagated twice
with ``retain_graph``. Each run must match eager loss and gradients; the eager-only
tests pin how a SAVE op holds the weight (rederived vs identity-saved)."""

from __future__ import annotations

import itertools
from typing import Any, Callable

import expecttest
import torch
import torch_remat as remat
from remat_test_helpers import checkpoint_for_test
from torch_remat._region import _checkpoint_context_fn, _state

_Decorate = Callable[..., Callable[[Callable[..., Any]], Callable[..., Any]]]
_Model = Callable[[torch.Tensor, torch.Tensor, _Decorate], torch.Tensor]

_POLICIES: list[tuple[bool, bool]] = list(itertools.product((False, True), repeat=2))


def _no_checkpoint(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Stand-in for ``checkpoint_for_test`` in the eager reference: regions run as plain
    calls outside a checkpoint."""

    del kwargs
    return lambda function: function


def _closed_over_model(embed_recompute: bool, unembed_recompute: bool) -> _Model:
    """Tied embedding / unembedding reading the weight from the enclosing scope."""

    def model(x: torch.Tensor, w: torch.Tensor, decorate: _Decorate) -> torch.Tensor:
        def body(t: torch.Tensor) -> torch.Tensor:
            h = remat.region(lambda a: a @ w, "embed", recompute=embed_recompute)(t)
            return remat.region(
                lambda a: torch.tanh(a) @ w.t(), "unembed", recompute=unembed_recompute
            )(h)

        return decorate(region_name="r")(body)(x).sum()

    return model


def _argument_model(embed_recompute: bool, unembed_recompute: bool) -> _Model:
    """Tied embedding / unembedding receiving the weight as a region argument."""

    def model(x: torch.Tensor, w: torch.Tensor, decorate: _Decorate) -> torch.Tensor:
        def body(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            h = remat.region(torch.matmul, "embed", recompute=embed_recompute)(t, v)
            return remat.region(
                lambda a, b: torch.tanh(a) @ b.t(),
                "unembed",
                recompute=unembed_recompute,
            )(h, v)

        return decorate(region_name="r")(body)(x, w).sum()

    return model


def _weight_and_view_model(recompute: bool) -> _Model:
    """One region receiving both the weight and its transpose."""

    def model(x: torch.Tensor, w: torch.Tensor, decorate: _Decorate) -> torch.Tensor:
        def body(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            h = remat.region(
                lambda a, b, bt: a @ b + torch.sin(a @ bt), "both", recompute=recompute
            )(t, v, v.t())
            return remat.region(lambda a: a @ w, "proj", recompute=False)(h)

        return decorate(region_name="r")(body)(x, w).sum()

    return model


def _gram_model(recompute: bool) -> _Model:
    """One op receiving the weight as both operands."""

    def model(x: torch.Tensor, w: torch.Tensor, decorate: _Decorate) -> torch.Tensor:
        def body(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            gram = remat.region(torch.mul, "gram", recompute=recompute)(v, v)
            return remat.region(torch.matmul, "apply", recompute=False)(t, gram)

        return decorate(region_name="r")(body)(x, w).sum()

    return model


class TiedWeightsTest(expecttest.TestCase):
    def _assert_matches_eager(
        self,
        model: _Model,
        *,
        w_requires_grad: bool = True,
        backward_passes: int = 1,
    ) -> None:
        """Run ``model(x, w, decorate)`` eagerly and checkpointed; compare loss and grads."""

        results: list[
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        ] = []
        for decorate in (_no_checkpoint, checkpoint_for_test):
            x = torch.linspace(-1.0, 1.0, 12).reshape(3, 4).requires_grad_()
            w = torch.linspace(-0.5, 0.7, 16).reshape(4, 4)
            w.requires_grad_(w_requires_grad)
            loss = model(x, w, decorate)
            for _ in range(backward_passes):
                loss.backward(retain_graph=backward_passes > 1)
            results.append((loss.detach(), x.grad, w.grad))
        (ref_loss, ref_x_grad, ref_w_grad), (loss, x_grad, w_grad) = results
        torch.testing.assert_close(loss, ref_loss)
        torch.testing.assert_close(x_grad, ref_x_grad)
        torch.testing.assert_close(w_grad, ref_w_grad)

    def test_closed_over_weight_and_transpose(self) -> None:
        for embed_recompute, unembed_recompute in _POLICIES:
            with self.subTest(embed=embed_recompute, unembed=unembed_recompute):
                self._assert_matches_eager(
                    _closed_over_model(embed_recompute, unembed_recompute)
                )

    def test_weight_passed_to_regions_as_argument(self) -> None:
        for embed_recompute, unembed_recompute in _POLICIES:
            with self.subTest(embed=embed_recompute, unembed=unembed_recompute):
                self._assert_matches_eager(
                    _argument_model(embed_recompute, unembed_recompute)
                )

    def test_weight_and_its_view_in_one_region(self) -> None:
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                self._assert_matches_eager(_weight_and_view_model(recompute))

    def test_weight_multiplied_with_itself(self) -> None:
        for recompute in (False, True):
            with self.subTest(recompute=recompute):
                self._assert_matches_eager(_gram_model(recompute))

    def test_same_weight_passed_twice_to_checkpoint(self) -> None:
        def model(
            x: torch.Tensor, w: torch.Tensor, decorate: _Decorate
        ) -> torch.Tensor:
            def body(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                h = remat.region(torch.matmul, "first", recompute=False)(t, a)
                return remat.region(torch.matmul, "second", recompute=True)(h, b.t())

            return decorate(region_name="r")(body)(x, w, w).sum()

        self._assert_matches_eager(model)

    def test_weight_used_outside_and_inside_checkpoint(self) -> None:
        def model(
            x: torch.Tensor, w: torch.Tensor, decorate: _Decorate
        ) -> torch.Tensor:
            def body(t: torch.Tensor) -> torch.Tensor:
                h = remat.region(torch.sin, "act", recompute=True)(t)
                return remat.region(lambda a: a @ w.t(), "unembed", recompute=False)(h)

            return decorate(region_name="r")(body)(x @ w).sum()

        self._assert_matches_eager(model)

    def test_two_checkpoints_share_weight(self) -> None:
        def model(
            x: torch.Tensor, w: torch.Tensor, decorate: _Decorate
        ) -> torch.Tensor:
            def layer(t: torch.Tensor) -> torch.Tensor:
                h = remat.region(lambda a: a @ w, "proj", recompute=False)(t)
                return remat.region(torch.tanh, "act", recompute=True)(h)

            h = decorate(region_name="layer.0")(layer)(x)
            return decorate(region_name="layer.1")(layer)(h).sum()

        self._assert_matches_eager(model)

    def test_shared_block_called_twice_in_one_checkpoint(self) -> None:
        def model(
            x: torch.Tensor, w: torch.Tensor, decorate: _Decorate
        ) -> torch.Tensor:
            def block(t: torch.Tensor) -> torch.Tensor:
                h = remat.region(lambda a: a @ w, "proj", recompute=False)(t)
                return remat.region(torch.tanh, "act", recompute=True)(h)

            def body(t: torch.Tensor) -> torch.Tensor:
                for index in range(2):
                    with remat.name_scope(f"block.{index}"):
                        t = block(t)
                return t

            return decorate(region_name="r")(body)(x).sum()

        self._assert_matches_eager(model)

    def test_frozen_tied_weight(self) -> None:
        def model(
            x: torch.Tensor, w: torch.Tensor, decorate: _Decorate
        ) -> torch.Tensor:
            def body(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                h = remat.region(torch.matmul, "embed", recompute=False)(t, v)
                h = remat.region(torch.tanh, "act", recompute=True)(h)
                return remat.region(lambda a: a @ w.t(), "unembed", recompute=False)(h)

            return decorate(region_name="r")(body)(x, w).sum()

        self._assert_matches_eager(model, w_requires_grad=False)

    def test_tied_weight_retain_graph_backward_twice(self) -> None:
        def model(
            x: torch.Tensor, w: torch.Tensor, decorate: _Decorate
        ) -> torch.Tensor:
            def body(t: torch.Tensor) -> torch.Tensor:
                h = remat.region(lambda a: a @ w, "embed", recompute=True)(t)
                return remat.region(
                    lambda a: torch.tanh(a) @ w.t(), "unembed", recompute=False
                )(h)

            return decorate(region_name="r")(body)(x).sum()

        self._assert_matches_eager(model, backward_passes=2)

    def test_weight_region_argument_is_rederived_not_retained(self) -> None:
        # Passed as an argument, the weight is a region input like any other, so a SAVE
        # op that saves it records a recipe instead of holding a reference.
        w = torch.ones(4, 4, requires_grad=True)
        forward_context, _ = _checkpoint_context_fn("r")
        with forward_context:
            remat.region(torch.matmul, "embed", recompute=False)(
                torch.ones(3, 4, requires_grad=True), w
            )
            active = _state.get()
            assert active is not None
            record = active.region_state.records["embed"]
            self.assertEqual(2, len(record.saved_input_recipes))
            self.assertEqual(0, len(record.saved_tensor_names))

    def test_closed_over_weight_is_saved_by_identity(self) -> None:
        # Closed over, the weight is invisible to the region's input tracking, so the
        # SAVE op holds it (here as its transpose) -- a storage-sharing alias, not a copy.
        w = torch.ones(4, 4, requires_grad=True)
        forward_context, _ = _checkpoint_context_fn("r")
        with forward_context:
            out = remat.region(lambda a: a @ w.t(), "unembed", recompute=False)(
                torch.ones(3, 4, requires_grad=True)
            )
            active = _state.get()
            assert active is not None
            record = active.region_state.records["unembed"]
            (saved,) = record.parameter_saves.keys()
            self.assertEqual(
                w.untyped_storage().data_ptr(), saved.untyped_storage().data_ptr()
            )
            del out
