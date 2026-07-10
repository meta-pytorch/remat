# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Storage-free placeholder tensors for skipped SAVE op outputs during recompute.

By default, all compute inside `remat.checkpoint` is recomputed.  However, you can
use `remat.region` with `recompute=False` to mark some regions as saved, in which
case they shouldn't be recomputed.  This poses a problem: what do we *return* from
`remat.region` when its internal compute has been skipped?  In some cases, we will
have saved these outputs for other reasons (e.g., `remat.region` saved it for
backwards, or in forward we found out some other `recompute=True` region would need
it and we have to save it so we can actually run the recompute.)  But sometimes, the
output is simply *not needed at all*, e.g., if it is to be fed immediately into
another saved region.  But we have to return something, and it has to be a convincing
enough facsimile of the real thing that benign things (like checking its size) still
work.  Thus the placeholder tensor.

It's important to note that a placeholder ONLY exists during recompute, and should
ONLY be produced in situations where we know that the tensor will never be used (as
determined by forwards).  It is an error to try to do actual compute on a placeholder.

One small complication is that if I do a (bare) view operation on the output of a
saved (`recompute=False`) region, I shouldn't force this output to be saved; it could
be that the view never gets used in any useful way.  This implies that we should also
support (metadata only) view operations on a placeholder.

Rather than a tensor subclass with a ``__torch_dispatch__`` that hand-classifies
each op as view-vs-compute, a placeholder is a plain ``torch.Tensor`` backed by a
storage with a **null data pointer** and the storage's data-pointer access flag set
to raise our diagnostic (the same mechanism CUDA graphs uses for its overwritten
outputs).  The C++ dispatcher then gives us the behavior for free: a metadata/view
op never reads ``data_ptr`` and passes through -- returning a view over the *same*
poisoned storage, so a chain of views stays storage-free and stays a placeholder --
while any op that would read or produce data hits the access flag and raises.  The
storage declares its true byte extent (so shape-only ops that bounds-check the
storage still pass) but never allocates it, so even a large skipped output costs no
memory.  Because a placeholder is a plain tensor, "is this a placeholder?" is
answered by probing the storage (:func:`_is_placeholder`), which also makes a view
of a placeholder test true.

Design note: crossing the save->recompute boundary is the *producer's*
responsibility -- a saved region durably saves any output a consumer needs, so on
replay the real tensor simply shows up in the dataflow. The rejected alternative (the
consumer ferries the value onto its own tape slot and substitutes it by position)
is worse for debugging: you emit placeholders with no real data, then magically
swap in real data when they are used. Better to have the real tensor present from
the start, with a placeholder only where we legitimately have no data at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _TensorMetadata:
    """Tensor metadata used for data-inaccessible replay outputs."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    storage_nbytes: int


def _make_placeholder_tensor(
    metadata: _TensorMetadata,
    message: str,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Return a shaped, storage-free placeholder whose tensor-data access raises.

    The result is a plain tensor over a null-pointer storage: metadata/view ops pass
    through (a view is another placeholder sharing this storage), while any data
    access raises ``message``. No device memory is allocated for the (potentially
    large) skipped output.
    """

    device = metadata.device

    # Rebind an empty tensor onto the null storage in one ``set_``: the bounds check
    # compares (shape, stride) against the storage's *declared* extent (the real
    # output's size, captured in forward), so the layout is accepted without ever
    # allocating -- zero peak memory on any device, no temporary.
    placeholder = torch.empty(0, dtype=metadata.dtype, device=device)
    placeholder.set_(
        _null_storage(metadata.storage_nbytes, device),
        0,
        torch.Size(metadata.shape),
        tuple(metadata.stride),
    )

    torch._C._set_storage_data_ptr_access_error_msg(
        placeholder.untyped_storage()._cdata, message
    )
    if requires_grad:
        # A freshly built placeholder is a leaf; requires_grad_ only touches
        # autograd metadata, so it does not trip the poisoned data pointer.
        placeholder.requires_grad_(True)
    return placeholder


def _is_placeholder(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is (a view of) a placeholder.

    A placeholder's storage raises on data-pointer access; a real tensor's returns a
    pointer (0 for an empty tensor). Probing the storage -- rather than the tensor
    type -- is what makes a view of a placeholder, a distinct object over the same
    poisoned storage, test true as well.
    """

    storage = tensor.untyped_storage()
    try:
        storage.data_ptr()
    except RuntimeError:
        return True
    return False


def _placeholder_message(tensor: torch.Tensor) -> str:
    """Return the diagnostic carried by a placeholder's poisoned storage.

    The message is read back from the storage's data-pointer access flag, so it is
    the single source of truth shared by every view of the placeholder.
    """

    storage = tensor.untyped_storage()
    try:
        storage.data_ptr()
    except RuntimeError as error:
        return str(error)
    raise AssertionError("tensor is not a placeholder")


def _placeholder_message_text(source: str, op_display_name: str) -> str:
    """Build the diagnostic carried by a skipped saved region's placeholder output.

    The failure it describes is config-dependent and only surfaces during the
    recompute pass (inside backward), so the message has to stand on its own --
    the traceback points into checkpoint recompute, not the user's code.
    """

    return (
        f"{source} is a placeholder for the output of remat.region '{op_display_name}' "
        "(recompute=False): a saved region is skipped during recompute, so its output "
        "is not recomputed -- only a metadata placeholder stands in, and something "
        "read its data.\n"
        "A saved region's output is real during recompute only if some consumer made "
        "the producer durably save it: a remat.region consumer does this automatically, "
        "as does a bare consumer when detect_bare_ops is enabled. With detect_bare_ops "
        "disabled, a plain bare/unwrapped op -- a view then .contiguous(), a residual "
        "add, anything not wrapped in remat.region -- does not, and hits this "
        "placeholder instead.\n"
        f"This depends on recompute: it appears only because '{op_display_name}' has "
        "recompute=False; the same code works with recompute=True (its output is then "
        "real). That is why a region can pass with everything recompute=True and fail "
        "once a region is switched to recompute=False.\n"
        "Fix one of:\n"
        "  (1) wrap the consuming op in remat.region(..., recompute=True) so it loads "
        "the saved value during recompute (e.g. make a residual add a recompute=True "
        "region);\n"
        "  (2) move the consuming op into a remat.region (do the reshape/add "
        "inside the producing or consuming region);\n"
        f"  (3) give '{op_display_name}' recompute=True."
    )


def _null_storage(nbytes: int, device: torch.device) -> torch.UntypedStorage:
    """Return a storage that declares ``nbytes`` but holds a null data pointer."""

    return torch._C._construct_storage_from_data_pointer(0, device, nbytes)
