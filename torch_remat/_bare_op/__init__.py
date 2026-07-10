# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Bare-op detection strategies for a SAVE op's outputs.

A *bare* consumer of a SAVE output -- one not wrapped in :func:`torch_remat.region` -- is
intercepted per ``checkpoint(..., detect_bare_ops=...)`` (default ``"subclass"``;
``False`` opts out). This package holds the four intercepting strategies plus their
shared plumbing; :mod:`torch_remat._api` selects one via :mod:`._strategy` and
otherwise never branches on the strategy.

* :mod:`._common` -- shared utilities: the ``_SaveOutputHandle`` / ``PersistOutputThunk``
  save-output representation, the in-place diagnostic, storage-aliasing view classification,
  and the suppression switch the mode strategies use.
* :mod:`._strategy` -- the ``_BareOpStrategy`` abstraction (make-output / typed-handle /
  forward-mode hooks) and the ``detect_bare_ops`` resolver.
* :mod:`._subclass` / :mod:`._proxy` -- the wrapper strategies (a ``__torch_dispatch__``
  tensor subclass and a ``__torch_function__`` proxy object).
* :mod:`._dispatch_mode` / :mod:`._function_mode` -- their mode analogues, which leave SAVE
  outputs plain and install a ``TorchDispatchMode`` / ``TorchFunctionMode`` for the forward.
"""
