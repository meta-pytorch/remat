# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch_remat._api import (  # noqa: F401
    checkpoint,
    CheckpointPolicy,
    op,
    save_for_backward,
    saved_tensors_hooks,
)
from torch_remat._region import is_recomputing  # noqa: F401
from torch_remat._reporting import (  # noqa: F401
    format_current_memory_report,
    print_current_memory_report,
)
from torch_remat._trace import collect_trace, trace_scope  # noqa: F401

# Convenience aliases for the two checkpoint policies, so callers can write
# ``remat.SAVE`` / ``remat.RECOMPUTE`` instead of ``remat.CheckpointPolicy.SAVE``.
SAVE = CheckpointPolicy.SAVE
RECOMPUTE = CheckpointPolicy.RECOMPUTE
