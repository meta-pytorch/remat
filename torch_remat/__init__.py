# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch_remat._api import (  # noqa: F401
    auto_forward,
    checkpoint,
    CheckpointPolicy,
    collect_trace,
    get_handle,
    is_recomputing,
    op,
    RematHandle,
    saved_tensors_hooks,
    set_saved_tensors_hooks,
    trace_scope,
)
from torch_remat._native import native_op  # noqa: F401
from torch_remat._reporting import (  # noqa: F401
    format_current_memory_report,
    print_current_memory_report,
)
