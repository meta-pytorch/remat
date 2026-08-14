# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plain-data types for the torch_remat tape.

The saved-tensor pack/unpack hook aliases and the pure-data records that
:mod:`torch_remat._api` builds during forward and consults during recompute
(:class:`~torch_remat._api._SaveRecord` owns them on the tape).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, TypeAlias

import torch
from torch_remat._placeholder import _TensorMetadata
from torch_remat._pytree import PathToken

# Same contract as ``torch.autograd.graph.saved_tensors_hooks``: pack returns an
# opaque object stored in place of the tensor, and unpack recovers it.
PackHook: TypeAlias = Callable[[torch.Tensor], object]
UnpackHook: TypeAlias = Callable[[object], torch.Tensor]

# Optional companion to a pack hook (see :func:`torch_remat.saved_tensors_hooks`'s
# ``capture_context``). Called *in-window*, where a saved tensor is produced, to snapshot
# whatever context the pack needs (e.g. an offloader's current chunk). The captured value
# is available from :func:`torch_remat.current_saved_tensor_info` while the one-argument
# pack hook runs. This lets a *deferred* SAVE-output save -- one whose pack fires later, at
# the consumer -- still observe the context that was live where the output was born.
CaptureContext: TypeAlias = Callable[[], object]


class SavedTensorKind(Enum):
    """Why remat is packing a tensor for a later load."""

    CHECKPOINT_INPUT = auto()
    BACKWARD = auto()
    SAVE_OUTPUT = auto()


@dataclass(frozen=True)
class SavedTensorInfo:
    """Metadata for the tensor whose remat pack hook is currently running.

    Obtain this from :func:`torch_remat.current_saved_tensor_info`; it is valid only
    during a pack-hook invocation.

    Attributes:
        kind: Why remat is packing the tensor. ``CHECKPOINT_INPUT`` denotes a
            checkpoint input retained for recomputation. ``BACKWARD`` denotes a
            tensor retained for backward, including ordinary native saves outside
            a checkpoint. ``SAVE_OUTPUT`` denotes a ``recompute=False`` region's
            output persisted as an input to later recomputation.
        context: The value returned by the hook's ``capture_context`` callback where
            the tensor was produced, or ``None`` when no callback was supplied. In
            particular, a deferred ``SAVE_OUTPUT`` pack receives its producer-time
            context even if a later consumer triggers the pack after the producer's
            hook scope has exited. Native saves outside a checkpoint always receive
            ``None`` because ``capture_context`` is remat-specific.
    """

    kind: SavedTensorKind
    context: object = None


@dataclass
class _OutputSlot:
    """One durably-saved SAVE-output slot, keyed by output position in an op record.

    Populated when a SAVE output is durably saved for recompute and read back by
    :func:`torch_remat._api._load_output_slot`. Not to be confused with :class:`_SavedTensor`
    and its siblings, which are the pack payloads for a SAVE op's saved-for-backward
    tensors; this is the record for the op's *outputs*.
    """

    # The saved tensor. None is a valid value for an explicitly retained input.
    tensor: torch.Tensor | None

    # Version counter observed when the tensor was saved. PyTorch's own in-place
    # guard does not fire for tensors packed through custom saved_tensors_hooks,
    # so we replicate it here.
    version: int | None

    # The unpack hook bound to this slot at pack time; its presence marks the slot
    # as offloaded. Bound per-slot so the tensor is recovered by the pair that packed
    # it, not whatever hooks are active at load time.
    unpack_hook: UnpackHook | None = None

    # The unpack hook's opaque packed payload; only set for offloaded slots.
    packed: object = None

    # Autograd metadata observed when the tensor was saved, reproduced when a
    # durably-saved SAVE output is reloaded for recompute (see _fabricate_recompute_input).
    # Both are tracked: a fresh requires-grad leaf allocated in a forward body must
    # replay as a leaf, not a view.
    requires_grad: bool = False
    is_leaf: bool = True


@dataclass
class _SavedHookData:
    """Autograd-held pack payload for a SAVE saved tensor packed via user remat hooks.

    Autograd holds only the user hook's ``packed`` result -- the original tensor is
    free to be dropped (offloading is the motivating case) -- and unpack restores it
    through the bound ``unpack_hook``, the pair that packed it. Mirrors
    :class:`_OutputSlot`'s offloaded form.
    """

    packed: object
    unpack_hook: UnpackHook


@dataclass(frozen=True)
class _SavedTensor:
    """Autograd-held pack payload for a SAVE op's identity-retained saved tensor.

    The default fate of a SAVE op's ``save_for_backward`` tensor: kept resident as a
    detached copy (see :func:`torch_remat._api._default_pack`) and handed back
    verbatim at unpack. ``version`` is the save-time version counter, re-checked at
    backward to catch in-place mutation -- autograd's own version guard does not fire
    for tensors packed through custom ``saved_tensors_hooks``.
    """

    tensor: torch.Tensor
    version: int


@dataclass(frozen=True)
class _SavedInputRef:
    """Autograd-held pack payload for a SAVE op's saved *input* that replay reproduces.

    A recompute-sourced tensor that crosses into a SAVE op from outside and is saved
    for backward is not retained on the identity hook: pack returns this ref instead of
    the tensor, and unpack resolves the value from the op's tape slot, which
    :func:`_rederive_saved_inputs` fills during recompute. This is what lets a
    RECOMPUTE->SAVE crossing avoid keeping the recompute output resident. (A
    SAVE-sourced stub input, which replay does not reproduce, is retained like any other
    save rather than diverted here.)
    """

    slot_name: str


@dataclass(frozen=True)
class _ViewSpec:
    """How to reconstruct an alias or view of an input at recompute.

    The view is rebuilt from the reproduced base with ``as_strided``. ``rel_offset``
    is the view's storage offset relative to the base's (absolute offsets are not
    stable across recompute). ``base_shape`` / ``base_stride`` pin the base layout the
    view was recorded against; recompute verifies the reproduced base matches before
    rebuilding (see :func:`torch_remat._view._rebuild_input_view`).
    """

    size: tuple[int, ...]
    stride: tuple[int, ...]
    rel_offset: int
    base_shape: tuple[int, ...]
    base_stride: tuple[int, ...]


@dataclass(frozen=True)
class _InputReplay:
    """How to reproduce a tensor from one of a region's recomputed inputs."""

    # Path token locating the source input in the op's call (e.g. ``(0,)`` ->
    # ``args[0]``); stable across forward and recompute.
    path: PathToken
    # ``None`` reuses the input directly; otherwise rebuild this alias or view with
    # ``as_strided`` relative to the input's replayed layout. The consumer decides
    # whether its replayed value must be detached.
    view_spec: _ViewSpec | None


@dataclass(frozen=True)
class _SavedInputRecipe(_InputReplay):
    """Forward recipe for one recompute-sourced saved input a SAVE op diverted off the
    identity hook (recorded in :func:`torch_remat._api._run_save_op`'s pack).

    Pure data, no tensor: the value is rederived at recompute from the reproduced
    input rather than retained -- the whole point of diverting.
    """

    # Tape-buffer key (``saved_input.<i>``); the same string rides the autograd pack
    # payload (:class:`_SavedInputRef`), and is what links forward to recompute --
    # this entry's list position is not load-bearing.
    slot_name: str
    # Report name (diagnostics only; the value is fetched via ``slot_name``).
    name: str


@dataclass(frozen=True)
class _InputInfo:
    """Layout snapshot of one SAVE-op input, taken before the body runs.

    Captured without keeping the input alive: the storage is held by a *weak*
    reference and every other field is plain data -- the pack closure outlives the
    call on the autograd graph, so a strong reference would pin every SAVE-op input
    until backward. pack classifies saved tensors against these by storage-object
    identity (``is`` on the live ``UntypedStorage``), robust to a saved tensor
    arriving as a different Python object wrapping the input's ``TensorImpl``.
    """

    # Path token locating this input in the call (e.g. ``(0,)`` -> ``args[0]``);
    # stable across forward and recompute.
    path: PathToken
    # Weak, so the snapshot never keeps an input alive; a dead reference simply
    # fails to match.
    storage_ref: weakref.ref[torch.UntypedStorage]
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    # Version counter observed at op entry. A saved tensor whose version has moved
    # past this was mutated in place by the op body after entry, so its data is NOT
    # what replay reproduces at op entry -- it must be retained, not diverted to a
    # rebuild recipe (see _classify_saved_input).
    version: int
    # Whether this input is a skipped SAVE op's output (a placeholder in replay), so
    # its value is stored now rather than reproduced by recompute.
    is_stub: bool


@dataclass(frozen=True)
class _OutputSpec:
    """A tensor output reconstructed from a persisted value or placeholder."""

    metadata: _TensorMetadata

    # Requires-grad observed on the original forward. A placeholder must reproduce
    # it so a downstream consumer (a RECOMPUTE op, the recompute boundary trigger,
    # ...) builds the same backward node and packs the same saved tensors during
    # replay, keeping checkpoint's saved-tensor count aligned.
    requires_grad: bool


@dataclass(frozen=True)
class _OutputSchema:
    """A skipped SAVE op's observed output schema, replayed as placeholders.

    Recorded on the original forward and consulted during recompute to rebuild the
    op's outputs as data-inaccessible placeholders. Output aliasing is
    intentionally not preserved.
    """

    # The observed output container's own type, kept verbatim: ``None`` for a bare
    # tensor, else the value's type -- ``tuple`` / ``list``, a ``NamedTuple``, or a
    # ``structseq`` (``torch.return_types.*``) -- so its named fields survive recompute.
    # Recompute rebuilds the placeholders in this container (see ``rebuild_container``).
    container: type | None

    # Per-output entries in return-schema order. None reproduces a static None;
    # every tensor has a spec describing its replay value or placeholder.
    specs: tuple[_OutputSpec | None, ...]
