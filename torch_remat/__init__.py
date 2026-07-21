# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch_remat._annotate import (  # noqa: F401
    clear_memory_snapshot_cache,
    memory_snapshot_annotate,
)
from torch_remat._api import (  # noqa: F401
    _pop_saved_tensors_hooks,
    _push_saved_tensors_hooks,
    checkpoint,
    current_saved_tensor_info,
    recompute_needs_tensor,
    region,
    save_for_backward,
    saved_tensors_hooks,
)
from torch_remat._oom import (  # noqa: F401
    discover_autograd_roots,
    format_oom_saved_tensors_report,
    oom_observer,
)
from torch_remat._region import is_recomputing, RecomputeStateHook  # noqa: F401
from torch_remat._reporting import (  # noqa: F401
    format_current_memory_report,
    print_current_memory_report,
)
from torch_remat._saved_report import (  # noqa: F401
    format_saved_tensors_report,
    print_saved_tensors_report,
)
from torch_remat._trace import collect_trace, trace_scope  # noqa: F401
from torch_remat._types import SavedTensorInfo, SavedTensorKind  # noqa: F401
