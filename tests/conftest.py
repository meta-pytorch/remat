# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Conditional expected-failure handling for the API suite.

In the explicit compile test target, ``compile_xfail`` tests run under strict xfail:
the documented incompatibility must reproduce, and an unexpected pass fails the test.

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
from remat_test_helpers import IS_COMPILE_TEST

_PLACEHOLDER_PRIMITIVES: tuple[str, ...] = (
    "_construct_storage_from_data_pointer",
    "_set_storage_data_ptr_access_error_msg",
)
_HAS_PLACEHOLDER_PRIMITIVES: bool = all(
    hasattr(torch._C, name) for name in _PLACEHOLDER_PRIMITIVES
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "compile_xfail(reason): behavior that is unsupported under torch.compile",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    if not IS_COMPILE_TEST:
        return
    for item in items:
        marker = item.get_closest_marker("compile_xfail")
        if marker is None:
            continue
        reason = marker.args[0] if marker.args else "not supported under torch.compile"
        item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, None, None]:
    outcome = yield
    report = outcome.get_result()
    if (
        IS_COMPILE_TEST
        and call.when == "call"
        and report.outcome == "skipped"
        and hasattr(report, "wasxfail")
    ):
        # TPX currently reports pytest xfails as skips, which opens test-health
        # issues for this intentionally strict expected-failure catalogue. Pytest
        # has already verified that the test failed here; a strict unexpected pass
        # has outcome "failed" and remains a failure.
        report.outcome = "passed"
        report.longrepr = None
        delattr(report, "wasxfail")
        return
    if _HAS_PLACEHOLDER_PRIMITIVES or call.when != "call" or call.excinfo is None:
        return
    exc = call.excinfo.value
    if isinstance(exc, AttributeError) and any(
        name in str(exc) for name in _PLACEHOLDER_PRIMITIVES
    ):
        report.outcome = "skipped"
        report.wasxfail = (
            "torch build lacks the null-storage placeholder primitives "
            f"({', '.join(_PLACEHOLDER_PRIMITIVES)}); the recompute placeholder "
            "path in torch_remat._placeholder cannot run"
        )
