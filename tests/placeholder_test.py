# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the recompute placeholder tensor -- the storage-free stand-in a SAVE op's
skipped output becomes during recompute. Covers the metadata-only aliasing ops it
permits (detach, view, transpose, slice) and the data-producing ops it rejects."""

from __future__ import annotations

import expecttest
import torch
from remat_test_helpers import assert_placeholder
from torch_remat._placeholder import (
    _make_placeholder_tensor,
    _TensorMetadata,
)


class PlaceholderTest(expecttest.TestCase):
    def test_placeholder_allows_metadata_only_aliasing_ops(self) -> None:
        placeholder = _make_placeholder_tensor(
            _TensorMetadata(
                shape=(2, 3),
                stride=(3, 1),
                dtype=torch.float32,
                device=torch.device("cpu"),
                storage_nbytes=24,
            ),
            "placeholder source was skipped during recompute",
        )

        detached = placeholder.detach()
        assert_placeholder(self, detached, (2, 3))
        self.assertFalse(detached.requires_grad)

        viewed = placeholder.view(6)
        assert_placeholder(self, viewed, (6,))
        self.assertEqual((1,), viewed.stride())

        transposed = placeholder.t()
        assert_placeholder(self, transposed, (3, 2))
        self.assertEqual((1, 3), transposed.stride())

        sliced = placeholder[:, :2]
        assert_placeholder(self, sliced, (2, 2))
        self.assertEqual((3, 1), sliced.stride())

    def test_placeholder_rejects_data_producing_ops(self) -> None:
        placeholder = _make_placeholder_tensor(
            _TensorMetadata(
                shape=(2, 3),
                stride=(3, 1),
                dtype=torch.float32,
                device=torch.device("cpu"),
                storage_nbytes=24,
            ),
            "placeholder source was skipped during recompute",
        )

        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            torch.sin(placeholder)

        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            placeholder.clone()

        with self.assertRaisesRegex(RuntimeError, "skipped during recompute"):
            placeholder.add_(1)
