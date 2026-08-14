# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Storage/view classification for a SAVE op's saved inputs.

We'd like to apply the following optimization.  Suppose that you have an
autograd function in a SAVE region that saves one of its inputs for backwards.
If that input comes from a RECOMPUTE region, we don't actually need to save
the input for backwards: it will show up as a real input during recompute and
we can just use it directly.  This will save us memory overall.  To do this
optimization, we have to be able to identify when an input to a remat.region is
saved for backwards (via saved tensor hooks).  This is not a big deal.

Now, let's suppose that the input wasn't directly saved for backward; instead,
some *view* of the input was saved for backwards.  Assume this view was
computed from inside the remat.region region.  We still shouldn't save this view
for backwards.  But we can't use the input directly; we have to reapply the
view to it so that we have a backwards of the correct extent and shape.  We
need a view replay.  Oh no!

This is not a hypothetical problem either.  Consider an N-D linear.  To feed
this into torch.addmm, we flatten batch dimensions into a single dimension:

    def linear(input, weight, bias):
        flattened_nrows = prod(input.shape[:-1])   # all batch dims folded
        ncols = input.shape[-1]                     # = in_features
        input_2d = input.view(flattened_nrows, ncols)
        out_2d = torch.addmm(bias, input_2d, weight.t())
        return out_2d.view(*input.shape[:-1], weight.shape[0])  # reshape back

The `input_2d` is saved for backwards, and it is a view of `input`.  So we need
to replay this view.

Fortunately, these view replays happen during recompute, when we don't care about
creating an autograd graph (only the forward autograd graph gets used).  So we can
directly use `as_strided` to do this.  Here is the recipe:

- Determine that something saved for backwards aliases with the input.

- Compute the RELATIVE `as_strided` arguments that compute the save for
  backward tensor from the input tensor.  The "relative" here refers to
  the storage offset; `as_strided` ignores the storage offset on base, and though
  we would usually expect base in recompute to have the same storage offset as
  forward, it is a small safety improvement if we can handle the situation
  when the storage offsets are different.

- Apply `as_strided` during recompute to generate the save for backwards
  tensor.
"""

from __future__ import annotations

import torch
from torch_remat._placeholder import _is_placeholder
from torch_remat._region import _CheckpointRegionState, _display_name
from torch_remat._types import _InputInfo, _ViewSpec


def _addressed_extent(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    """Return ``1 + max element offset`` addressed by (shape, stride); 0 if empty."""

    extent = 0
    for size, step in zip(shape, stride):
        if size == 0:
            return 0
        extent += (size - 1) * step
    return extent + 1


def _classify_saved_input(
    saved: torch.Tensor,
    inputs: list[_InputInfo],
) -> tuple[_InputInfo, _ViewSpec | None] | None:
    """Match a SAVE op's saved tensor to one of its inputs by shared storage.

    Aliasing is decided by storage-object identity (``is`` on the live
    ``UntypedStorage``), never the data pointer, so a freed-and-recycled address
    cannot mis-match and a saved tensor matches even if autograd hands pack a
    different Python object wrapping the input's TensorImpl.

    Returns ``(info, None)`` when the saved tensor *is* an input (same storage and
    layout); ``(info, view_spec)`` when it is a reconstructable view of a non-stub
    input; ``None`` to leave it on the identity hook -- a disjoint view, a dtype
    reinterpret, a stub or negatively-strided base, or an input already freed --
    so correctness never depends on this match.
    """

    if _is_placeholder(saved):
        return None
    saved_storage = saved.untyped_storage()
    saved_offset = saved.storage_offset()
    saved_shape = tuple(saved.shape)
    saved_stride = tuple(saved.stride())

    # A saved tensor mutated in place since op entry (its version counter moved past
    # the entry snapshot) no longer holds the value replay reproduces at op entry --
    # e.g. an op that exps its input in place and saves the result. Leave it on the
    # identity hook so the post-mutation value is retained.
    saved_version = saved._version

    # Exact input: shared storage and identical layout/dtype.
    for info in inputs:
        base_storage = info.storage_ref()
        if (
            base_storage is not None
            and base_storage is saved_storage
            and info.dtype == saved.dtype
            and info.storage_offset == saved_offset
            and info.shape == saved_shape
            and info.stride == saved_stride
        ):
            if saved_version != info.version:
                return None
            return info, None

    # View of an input: a non-stub base whose footprint fully contains the view can be
    # rebuilt at recompute via as_strided (contiguity doesn't matter -- as_strided
    # works off the storage). The extent arithmetic assumes non-negative strides, so
    # negatively-strided tensors are left out.
    if saved.numel() == 0 or any(step < 0 for step in saved_stride):
        return None
    view_lo = saved_offset
    view_hi = saved_offset + _addressed_extent(saved_shape, saved_stride)
    for info in inputs:
        base_storage = info.storage_ref()
        if (
            base_storage is None
            or base_storage is not saved_storage
            or info.is_stub
            or info.dtype != saved.dtype
            or any(step < 0 for step in info.stride)
        ):
            continue
        base_lo = info.storage_offset
        base_hi = base_lo + _addressed_extent(info.shape, info.stride)
        if base_lo <= view_lo and view_hi <= base_hi:
            if saved_version != info.version:
                return None
            return info, _ViewSpec(
                size=saved_shape,
                stride=saved_stride,
                rel_offset=view_lo - base_lo,
                base_shape=info.shape,
                base_stride=info.stride,
            )
    return None


def _rebuild_input_view(
    region_state: _CheckpointRegionState,
    op_name: str,
    base: torch.Tensor,
    view_spec: _ViewSpec,
) -> torch.Tensor:
    """Rebuild a view of an input from its recompute-reproduced base.

    ``as_strided`` reproduces the saved view's elements only if the reproduced base
    has the layout the spec was recorded against; verify it and raise rather than
    silently read the wrong elements -- the view was discarded in the forward, so
    there is nothing to fall back to.
    """

    if (
        tuple(base.shape) != view_spec.base_shape
        or tuple(base.stride()) != view_spec.base_stride
    ):
        raise RuntimeError(
            f"{_display_name(region_state, op_name)} (recompute=False) saved a "
            "view of an input for backward, but recompute reproduced that input "
            f"with a different layout (expected shape {view_spec.base_shape} "
            f"stride {view_spec.base_stride}, got shape {tuple(base.shape)} "
            f"stride {tuple(base.stride())}). The saved view cannot be reconstructed."
        )
    return base.as_strided(
        view_spec.size,
        view_spec.stride,
        base.storage_offset() + view_spec.rel_offset,
    )


def _rebuild_saved_view(
    region_state: _CheckpointRegionState,
    op_name: str,
    base: torch.Tensor,
    view_spec: _ViewSpec,
) -> torch.Tensor:
    """Rebuild and detach a saved-for-backward view of a recomputed input."""

    return _rebuild_input_view(region_state, op_name, base, view_spec).detach()
