# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the default memory-snapshot annotator (``torch_remat._annotate``).

The annotator maps a saved tensor's storage pointer to the code that allocated it, read from
a CUDA memory snapshot. It takes the snapshot as an argument, so the pointer arithmetic is
exercisable on CPU with a synthetic snapshot -- no GPU required."""

from __future__ import annotations

import expecttest
import torch
from torch_remat._annotate import (
    _alloc_site_from_frames,
    _memory_snapshot,
    clear_memory_snapshot_cache,
    memory_snapshot_annotate,
)


class AllocSiteFromFramesTest(expecttest.TestCase):
    def test_prefers_innermost_non_library_frame(self) -> None:
        # Frames are innermost-first. Skip torch / torch_remat internals; pick the innermost
        # real model frame.
        frames = [
            {"filename": "/x/torch/_ops.py", "line": 9, "name": "call"},
            {"filename": "/x/torch_remat/_region.py", "line": 5, "name": "skip"},
            {"filename": "/x/mymodel/moe.py", "line": 470, "name": "experts"},
        ]
        self.assertEqual(_alloc_site_from_frames(frames), "moe.py:470 experts")

    def test_falls_back_to_innermost_when_all_library(self) -> None:
        frames = [{"filename": "/x/torch/_ops.py", "line": 9, "name": "call"}]
        self.assertEqual(_alloc_site_from_frames(frames), "_ops.py:9")

    def test_empty_or_none(self) -> None:
        self.assertIsNone(_alloc_site_from_frames(None))
        self.assertIsNone(_alloc_site_from_frames([]))


class MemorySnapshotAnnotateTest(expecttest.TestCase):
    def test_maps_storage_and_views_to_site(self) -> None:
        tensor = torch.arange(16, dtype=torch.float32)
        ptr = tensor.untyped_storage().data_ptr()
        size = tensor.untyped_storage().nbytes()
        # A decoy block before the real one exercises the bisect selection.
        snapshot = {
            "segments": [
                {
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "address": 1,
                            "size": 1,
                            "frames": [{"filename": "/m/a.py", "line": 1, "name": "x"}],
                        },
                        {
                            "state": "active_allocated",
                            "address": ptr,
                            "size": size,
                            "frames": [
                                {
                                    "filename": "/m/model/foo.py",
                                    "line": 42,
                                    "name": "fwd",
                                }
                            ],
                        },
                    ]
                }
            ]
        }
        annotate = memory_snapshot_annotate(snapshot)
        self.assertEqual(annotate(tensor), "foo.py:42 fwd")
        # A view shares the storage base, so it resolves to the same site.
        self.assertEqual(annotate(tensor[4:8]), "foo.py:42 fwd")

    def test_returns_none_when_uncovered(self) -> None:
        snapshot = {
            "segments": [
                {
                    "blocks": [
                        {
                            "state": "active_allocated",
                            "address": 1,
                            "size": 1,
                            "frames": [{"filename": "/m/a.py", "line": 1, "name": "x"}],
                        }
                    ]
                }
            ]
        }
        self.assertIsNone(memory_snapshot_annotate(snapshot)(torch.zeros(4)))

    def test_ignores_non_allocated_blocks(self) -> None:
        tensor = torch.arange(8, dtype=torch.float32)
        snapshot = {
            "segments": [
                {
                    "blocks": [
                        {
                            "state": "inactive",
                            "address": tensor.untyped_storage().data_ptr(),
                            "size": tensor.untyped_storage().nbytes(),
                            "frames": [{"filename": "/m/x.py", "line": 1, "name": "f"}],
                        }
                    ]
                }
            ]
        }
        self.assertIsNone(memory_snapshot_annotate(snapshot)(tensor))

    def test_empty_snapshot(self) -> None:
        # No segments at all -> annotator is a total function that returns None.
        self.assertIsNone(memory_snapshot_annotate({})(torch.zeros(2)))


class MemorySnapshotCacheTest(expecttest.TestCase):
    def setUp(self) -> None:
        clear_memory_snapshot_cache()

    def tearDown(self) -> None:
        clear_memory_snapshot_cache()

    def test_fetch_is_memoized_until_refresh_or_clear(self) -> None:
        # The expensive torch.cuda.memory._snapshot() fetch is cached (works on CPU too --
        # it just reports zero segments), so repeated report calls don't re-walk the allocator.
        first = _memory_snapshot()
        self.assertIs(_memory_snapshot(), first)  # memoized
        self.assertIsNot(_memory_snapshot(refresh=True), first)  # forced re-fetch
        clear_memory_snapshot_cache()
        self.assertIsNot(_memory_snapshot(), first)  # cleared -> re-fetch
