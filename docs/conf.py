# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

project = "torch_remat"
copyright = "Meta Platforms, Inc."

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_commonmark_compat = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

autodoc_member_order = "bysource"
napoleon_google_docstring = True

nitpick_ignore = [
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.device"),
    ("py:class", "torch_remat._reporting.Annotate"),
]
