# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Environment-conditional xfail for the recompute-placeholder path.

``torch_remat._placeholder`` builds a storage-free recompute placeholder by binding a
tensor onto a null-pointer storage (``torch._C._construct_storage_from_data_pointer``)
and poisoning its data-pointer access (``torch._C._set_storage_data_ptr_access_error_msg``).
Older / unpatched torch builds lack those functions, so any test that materializes such
a placeholder raises ``AttributeError`` from that path. When a function is absent we
convert *that specific* failure to an expected failure (xfail) so the suite stays green
until the torch build catches up; on a build that has them the tests run normally, so a
genuine regression still fails.

Keyed on the functions' presence rather than on the device or CUDA availability: the
placeholder path is device-agnostic, but the missing-attribute failure can only happen
when one of these functions does not exist.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch

_PLACEHOLDER_PRIMITIVES: tuple[str, ...] = (
    "_construct_storage_from_data_pointer",
    "_set_storage_data_ptr_access_error_msg",
)
_HAS_PLACEHOLDER_PRIMITIVES: bool = all(
    hasattr(torch._C, name) for name in _PLACEHOLDER_PRIMITIVES
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, None, None]:
    outcome = yield
    if _HAS_PLACEHOLDER_PRIMITIVES or call.when != "call" or call.excinfo is None:
        return
    exc = call.excinfo.value
    if isinstance(exc, AttributeError) and any(
        name in str(exc) for name in _PLACEHOLDER_PRIMITIVES
    ):
        report = outcome.get_result()
        report.outcome = "skipped"
        report.wasxfail = (
            "torch build lacks the null-storage placeholder primitives "
            f"({', '.join(_PLACEHOLDER_PRIMITIVES)}); the recompute placeholder "
            "path in torch_remat._placeholder cannot run"
        )
