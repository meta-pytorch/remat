# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Explicit activation rematerialization for checkpoint regions.

:func:`checkpoint` wraps a region that recomputes by default; :func:`op`
annotates one call inside it with a :class:`CheckpointPolicy` (``SAVE`` or
``RECOMPUTE``). The wrapped callable is used unmodified.

The region runs under PyTorch non-reentrant checkpointing: backward executes
the *original* forward grad_fns, recompute only refills their saved tensors.
A ``SAVE`` op installs a nested ``saved_tensors_hooks`` that shadows
checkpoint's hooks, so its saved tensors stay ordinary autograd saved tensors
on the original graph; the op is skipped during recompute (returning
placeholder outputs). A ``RECOMPUTE`` op runs normally in both passes; its one
extra duty is at the boundary: an input that is an upstream ``SAVE`` op's
output is made the *producer's* responsibility to persist, so replay can
reproduce it into the dataflow.
"""

from __future__ import annotations

import contextlib
import contextvars
import weakref
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    cast,
    Iterator,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.weak import WeakTensorKeyDictionary
from torch_remat._bare_op._common import (
    _merge_save_output_handles,
    _SaveOutputHandle,
    _suppress_bare_op_detection,
)
from torch_remat._compat import _torch_checkpoint_with_forward_exception_cleanup
from torch_remat._placeholder import (
    _is_placeholder,
    _make_placeholder_tensor,
    _placeholder_message_text,
    _TensorMetadata,
)
from torch_remat._pytree import (
    iter_arg_leaves,
    map_arg_leaves,
    PathToken,
    value_leaves,
)
from torch_remat._recompute_boundary import _checkpoint_recompute_boundary
from torch_remat._region import (
    _active_save_op,
    _ActiveCheckpointRegion,
    _assert_phase,
    _checkpoint_context_fn,
    _CheckpointRegionState,
    _display_name,
    _Phase,
    _save_output_handle,
    _state,
)
from torch_remat._trace import _record_trace_op
from torch_remat._types import (
    _InputInfo,
    _OutputSchema,
    _OutputSlot,
    _OutputSpec,
    _SavedHookData,
    _SavedInputRecipe,
    _SavedInputRef,
    _SavedTensor,
    CheckpointPolicy,
    PackHook,
    UnpackHook,
)
from torch_remat._view import (
    _classify_saved_input,
    _rebuild_saved_view,
)

# A remat-aware op call returns a tensor, or a flat tuple or list of tensors --
# the shapes autograd.Function.apply and native ops commonly produce. We only need
# to locate the tensors; the container type is preserved for the caller.
Output: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]

_P = ParamSpec("_P")
_R = TypeVar("_R")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def checkpoint(
    *,
    region_name: str | None = None,
    determinism_check: str = "none",
    preserve_rng_state: bool = True,
    detect_bare_ops: bool | str = "subclass",
    input_saved_tensors_hooks: tuple[PackHook, UnpackHook] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that builds a torch_remat checkpoint wrapper.

    Checkpoint options, the function, and user arguments are supplied in three
    separate calls: ``checkpoint(...)(function)(*args, **kwargs)``.
    ``checkpoint(function)(...)`` is intentionally unsupported to avoid
    confusion with ``torch.utils.checkpoint.checkpoint``.

    Everything inside the region recomputes by default; annotate calls with
    :func:`op` to ``SAVE`` (keep activations, skip recompute) instead.

    Region arguments are forwarded unchanged (``torch.utils.checkpoint`` handles
    them), but the region *output* must be a Tensor or a one-hop ``tuple`` /
    ``list`` of Tensors; anything else raises at the region boundary.

    Args:
        region_name (str, optional): Label for the region, shown in memory
            reports, traces, and error messages to identify which region an op
            belongs to. When ``None`` the region renders as ``<unnamed>``.
            Keyword-only. Default: ``None``.
        determinism_check (str, optional): Forwarded verbatim to
            ``torch.utils.checkpoint.checkpoint``; selects the check that
            compares tensor metadata between the forward and the recompute to
            catch nondeterministic regions. ``"none"`` disables it. Keyword-only.
            Default: ``"none"``.
        preserve_rng_state (bool, optional): Forwarded verbatim to
            ``torch.utils.checkpoint.checkpoint``. If ``True``, the forward RNG
            state is stashed and restored before recompute so RNG-using ops
            replay identically. Keyword-only. Default: ``True``.
        detect_bare_ops (bool or str, optional): How *bare* (un-``op``-wrapped)
            consumers of a SAVE op's outputs are intercepted, so e.g. a residual
            add or raw Triton kernel works without a :func:`op` wrapper. One of
            ``True`` / ``"subclass"`` (wrap outputs in a tensor subclass),
            ``"proxy"`` (wrap in a proxy object), ``"dispatch_mode"``,
            ``"function_mode"`` (intercept via a torch mode), or ``False`` to opt
            out -- a bare consumer then raises during recompute. See
            :mod:`torch_remat._bare_op._strategy` for the trade-offs, and the
            README for the full comparison. Keyword-only. Default: ``"subclass"``.
        input_saved_tensors_hooks (tuple, optional): A ``(pack_hook, unpack_hook)`` pair
            (same signature as ``torch.autograd.graph.saved_tensors_hooks``) applied to the
            region's *input* tensors -- e.g. to offload a large residual stream to CPU.
            ``pack_hook`` fires once per input at region entry, *before the body runs*, so it
            must not synchronously free storage the body still reads (defer the free);
            ``unpack_hook`` restores each input when the region is replayed for recompute.
            Keyword-only. Default: ``None``.

    Returns:
        Callable: A decorator that takes the region ``function`` and returns a
        checkpointed callable; call that with the region's own ``*args`` /
        ``**kwargs``.

    Example:
        ```python
        import torch_remat as remat

        y = remat.checkpoint(region_name="layers.0")(block)(x)
        ```
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped_function(*inner_args: Any, **inner_kwargs: Any) -> Any:
            output = function(*inner_args, **inner_kwargs)
            return _checkpoint_recompute_boundary(output)

        @wraps(function)
        def checkpointed_function(*args: Any, **kwargs: Any) -> Any:
            # We install these around the WHOLE region but they fire only for the region
            # inputs: torch.utils.checkpoint saves the inputs (via _make_saved_tensor) at
            # region entry, then enters its own saved_tensors_hooks for recompute which
            # shadows ours for every save in the body -- only the input save, before
            # checkpoint's hook is installed, reaches ours. Can't scope tighter than the
            # whole region: checkpoint's hook nests inside and stays across the body, so
            # popping ours earlier would break the hook stack's LIFO order.
            input_hooks_ctx: contextlib.AbstractContextManager[Any] = (
                torch.autograd.graph.saved_tensors_hooks(*input_saved_tensors_hooks)
                if input_saved_tensors_hooks is not None
                else contextlib.nullcontext()
            )
            with input_hooks_ctx:
                return _torch_checkpoint_with_forward_exception_cleanup(
                    wrapped_function,
                    function_args=args,
                    function_kwargs=kwargs,
                    context_fn=lambda: _checkpoint_context_fn(
                        region_name, detect_bare_ops
                    ),
                    determinism_check=determinism_check,
                    preserve_rng_state=preserve_rng_state,
                )

        return checkpointed_function

    return decorate


def op(
    function: Callable[_P, _R],
    name: str,
    *,
    policy: CheckpointPolicy = CheckpointPolicy.SAVE,
) -> Callable[_P, _R]:
    """Annotate one call inside a checkpoint region with a remat policy.

    ``function`` is used unmodified -- it may be a custom ``autograd.Function``'s
    ``.apply``, a bare native op, or any callable taking a flat list of
    Tensor/non-Tensor arguments and returning a Tensor or tuple of Tensors.

        ```python
        y = remat.op(MyOp.apply, "my.op")(x)
        ```

    With ``SAVE`` (the default -- the region already recomputes everything, so an
    annotation normally marks the exception), the tensors the call's autograd
    nodes save are kept by autograd on the original forward graph and the call
    is not rerun during recompute. With ``RECOMPUTE``, the call is rerun during
    recompute; any input that is an upstream ``SAVE`` op's output is made that
    producer's responsibility to persist, so the rerun has real data.

    Despite its name, a `remat.op` doesn't have to correspond to a true PyTorch
    operator; you can scope it as large as you like. We recommend about the
    granularity of an autograd function, since this gives you the most accurate
    reporting of where save-for-backward costs are going.

    Tensor inputs and outputs follow ATen conventions -- a Tensor, or a
    *one-hop* ``tuple`` / ``list`` of Tensors -- deliberately not full pytree
    (nor ``dict``). Arguments are walked leniently: anything else (a ``dict``, a
    custom object, deeper nesting) is an opaque leaf handed to ``function``
    untouched. That leniency has a cost: if an upstream ``SAVE`` op's output is
    hidden inside such a leaf, remat's argument walk never finds it and so never
    arranges for the producer to persist it. A ``RECOMPUTE`` consumer of that
    output then gets the skipped producer's stand-in placeholder instead of real
    data, which raises -- but only during recompute (inside backward), not at
    this call, so keep SAVE outputs at the top level or one hop deep. The
    *return* is validated strictly: a non-Tensor or nested return raises
    ``RuntimeError``.

    NB: if you smuggle an input into the callable (e.g., via a global or via a
    closure), you had better ensure that it is available/recomputed in
    recompute, otherwise we may fail to save it for backwards (only direct
    inputs induce save.)

    ``torch.compile`` is not supported yet.

    Args:
        function (Callable): The callable to annotate, used unmodified. It may be
            a custom ``autograd.Function``'s ``.apply``, a bare native op, or any
            callable taking a flat list of Tensor/non-Tensor arguments and
            returning a Tensor or a one-hop ``tuple`` / ``list`` of Tensors.
        name (str): Region-relative name for this op, shown in memory reports,
            traces, and error messages. Must be non-empty and unique among the
            op names reached in a single phase.
        policy (CheckpointPolicy, optional): How the call is handled under
            recompute. ``SAVE`` keeps the tensors the call saves for backward on
            the original forward graph and skips rerunning the call during
            recompute; ``RECOMPUTE`` reruns the call, and any input that is an
            upstream ``SAVE`` op's output is made that producer's responsibility
            to persist. Keyword-only. Default: ``CheckpointPolicy.SAVE``.

    Returns:
        Callable: A wrapper with the same signature as ``function``. Called
        outside an active checkpoint region it simply forwards to ``function``.

    Raises:
        RuntimeError: If ``function`` is not callable, or ``policy`` is not a
            :class:`CheckpointPolicy`.
        ValueError: If ``name`` is empty.
    """

    if not callable(function):
        raise RuntimeError("op expects a function as its first argument")
    _validate_name(name, what="op_name")
    if not isinstance(policy, CheckpointPolicy):
        raise RuntimeError("op expects a CheckpointPolicy")

    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        state = _state.get()
        if state is None:
            # Outside a checkpoint region: behave as a plain call.
            return function(*args, **kwargs)

        # Suppress the bare-op detection mode (if any) for this op's own processing,
        # including the wrapped body: the consume/snapshot path already handles the op's
        # SAVE-output arguments, so the mode must not re-handle them. Known gap: a SAVE
        # output the body reaches via closure capture is handled by neither -- see the
        # _suppress_bare_op_detection note in torch_remat._bare_op._common.
        with _suppress_bare_op_detection():
            _record_trace_op(name, policy=policy)

            # Record this op invocation in the current phase, rejecting duplicates.
            if name in state.claimed_names:
                raise RuntimeError(
                    f"Duplicate torch_remat op name "
                    f"{_display_name(state.region_state, name)} during "
                    f"{state.phase.name.lower()}"
                )
            state.claimed_names.add(name)

            if policy is CheckpointPolicy.SAVE:
                return cast(_R, _run_save_op(state, name, function, args, kwargs))
            return cast(_R, _run_recompute_op(state, name, function, args, kwargs))

    return wrapper


def save_for_backward(
    ctx: Any,
    saved: Mapping[str, torch.Tensor | None],
) -> None:
    """Named ``ctx.save_for_backward`` for use inside a :func:`op` forward.

    Call this in place of ``ctx.save_for_backward(...)`` to give the saved tensors
    stable, meaningful names. Under a ``SAVE`` op the names label the tensors in
    :func:`format_current_memory_report` instead of positional ``saved.0`` /
    ``saved.1`` keys. ``None`` values are preserved positionally (as
    ``ctx.save_for_backward`` does) but carry no name.

    Using it is optional and always safe: outside a ``SAVE`` op (a ``RECOMPUTE`` op,
    or no active checkpoint region) it simply forwards to ``ctx.save_for_backward``
    and the names are ignored.

    Args:
        ctx: The ``autograd.Function`` context object handed to ``forward``; the
            tensors are forwarded to its ``ctx.save_for_backward``.
        saved (Mapping[str, Tensor or None]): Mapping from report name to tensor.
            Each name labels its tensor in :func:`format_current_memory_report`
            in place of a positional ``saved.<i>`` key; it must be non-empty and
            must not contain ``'.'``. A ``None`` value is preserved positionally
            (as ``ctx.save_for_backward`` does) but carries no name.

    Raises:
        ValueError: If a name is empty.
        RuntimeError: If a name contains ``'.'``, or a value is neither a Tensor
            nor ``None``.

    Example:
        ```python
        def forward(ctx, x):
            y = f(x)
            remat.save_for_backward(ctx, {"x": x, "y": y})
            return g(y)
        ```
    """

    scratch = _active_save_op.get()
    for tensor_name, tensor in saved.items():
        _validate_name(tensor_name, what="save_for_backward name")
        if "." in tensor_name:
            raise RuntimeError(
                f"save_for_backward name {tensor_name!r} must not contain '.'"
            )
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"save_for_backward[{tensor_name!r}] must be a Tensor or None"
            )
        if scratch is not None and isinstance(tensor, torch.Tensor):
            # Record the name in the op's forward scratch, keyed by the exact tensor
            # object: pack looks the name up for the tensor it is handed (see
            # :func:`_resolve_save_name`), so a name cannot shift onto another op's
            # tensor -- the scratch dies with this op's forward, so the name cannot
            # outlive the op (or mutate the user's tensor). Gated on being inside a
            # SAVE op so a plain ``save_for_backward`` outside a remat region
            # records nothing.
            scratch.pending_save_names[tensor] = tensor_name
    ctx.save_for_backward(*saved.values())


# Task-local so concurrent forwards don't clobber each other's hooks. Only the
# store (forward) side reads this; the load side uses the unpack hook bound to
# each slot at pack time, so we never rely on this contextvar propagating into
# a separate backward/recompute thread.
_active_saved_tensors_hooks: contextvars.ContextVar[
    tuple[PackHook, UnpackHook] | None
] = contextvars.ContextVar("torch_remat_saved_tensors_hooks", default=None)


# LIFO stack of reset tokens for the tokenless _push/_pop mutators below, so
# _pop can restore the prior hooks without the caller threading a token through.
# Task-local for the same reason as _active_saved_tensors_hooks: concurrent
# forwards must not pop each other's tokens. Immutable tuple so ContextVar's
# copy-on-set semantics hold.
_saved_tensors_hooks_tokens: contextvars.ContextVar[tuple[contextvars.Token, ...]] = (
    contextvars.ContextVar("torch_remat_saved_tensors_hooks_tokens", default=())
)


def _push_saved_tensors_hooks(
    pack_hook: PackHook,
    unpack_hook: UnpackHook,
) -> None:
    """Raw mutator: install ``(pack_hook, unpack_hook)`` until a matching
    :func:`_pop_saved_tensors_hooks`.

    Prefer the :func:`saved_tensors_hooks` context manager. This exists for
    callers whose install and uninstall span two methods (e.g. a manager's
    ``__enter__`` / ``__exit__``) and so cannot hold a single ``with`` block.
    Calls must nest LIFO -- each ``_pop`` undoes the most recent ``_push``.
    """
    token = _active_saved_tensors_hooks.set((pack_hook, unpack_hook))
    _saved_tensors_hooks_tokens.set(_saved_tensors_hooks_tokens.get() + (token,))


def _pop_saved_tensors_hooks() -> None:
    """Raw mutator: undo the most recent :func:`_push_saved_tensors_hooks`."""
    tokens = _saved_tensors_hooks_tokens.get()
    if not tokens:
        raise RuntimeError(
            "_pop_saved_tensors_hooks called without a matching _push_saved_tensors_hooks"
        )
    token = tokens[-1]
    _saved_tensors_hooks_tokens.set(tokens[:-1])
    _active_saved_tensors_hooks.reset(token)


@contextlib.contextmanager
def saved_tensors_hooks(
    pack_hook: PackHook,
    unpack_hook: UnpackHook,
) -> Iterator[None]:
    """Apply hooks that trigger when tensors are saved in forwards for load in
    recompute/backward.

    The remat analogue of ``torch.autograd.graph.saved_tensors_hooks``:
    ``pack_hook(tensor) -> packed`` runs for every tensor which is saved for
    recompute/backward (this is both normal `save_for_backward` tensors as
    well as outputs of SAVE regions which are needed for RECOMPUTE regions),
    and ``unpack_hook(packed) -> tensor`` at each load.  The autograd API
    cannot be used for this, as saved tensor hooks are not compositional and
    you will override the saved tensor hooks that ``torch_remat`` uses for
    its functionality.

    Although saved tensor hooks are a versatile mechanism, we originally
    designed this with the intent that it can be used for activation
    offloading type use cases.

    There are two important behavioral differences from traditional autograd
    hooks which need to be emphasized:

    1. If you want to run some code before any unpack hooks are triggered,
       you should do so at the start of the forward logic in ``remat.checkpoint``
       when ``remat.is_recomputing()``.  This is because saved output unpack
       hooks will trigger during recompute phase, before we start executing
       backwards.  Don't use an autograd function, this will be too late!

    2. Unpack hooks are not guaranteed to run in reverse order of pack hooks;
       indeed, unpack hooks for saved outputs will always trigger in the same
       order as their pack hooks, before all other hooks (as they are needed
       in the same order for forwards.)

    Args:
        pack_hook (Callable[[Tensor], Any]): Runs once for every tensor saved
            for recompute/backward (both ordinary ``save_for_backward`` tensors
            and SAVE-region outputs needed by a RECOMPUTE region), returning an
            opaque payload that autograd holds in place of the tensor -- so the
            original may be dropped or offloaded.
        unpack_hook (Callable[[Any], Tensor]): Runs at each load, taking the
            payload ``pack_hook`` returned and reconstructing the tensor. Bound
            per-save, so it fires with the payload it packed even after the
            ``with`` block has exited.

    Yields:
        None: The hooks apply to saves that occur within the ``with`` block.
    """

    _push_saved_tensors_hooks(pack_hook, unpack_hook)
    try:
        yield
    finally:
        _pop_saved_tensors_hooks()


# --------------------------------------------------------------------------
# SAVE-op records and record lifecycle
# --------------------------------------------------------------------------


@dataclass
class _SaveRecord:
    """Per remat.op information (metadata and tensors) generated during
    forward that is needed for recompute.  Note that not every Tensor saved
    between forward and recompute is stored here.

    When you SAVE a remat.op, so that it doesn't reexecute during recompute,
    we need to record three main pieces of information to actually run
    recompute and backwards later:

    - The tensors that are `save_for_backward`, for obvious reasons!
      These actually aren't stored in this record: we keep them directly as
      saved tensors in the autograd graph, the way a normal tensor that is
      saved for backwards would be saved (in principle, we could have saved it
      in this record, but there are small benefits to keeping it in the
      autograd graph, such as prompt deallocation of tensors saved for
      backward if their autograd node goes dead).  For debugging purposes, we
      do maintain a weak `saved_tensor_names` that lets us enumerate all tensors
      we've saved for backwards.

    - However, there is an exception to the above: if a tensor that is saved
      for backward is an alias of an input tensor to the input region, AND
      that input tensor comes from a RECOMPUTE region, we can avoid saving
      it altogether since we can always rederive the alias at recompute time.
      `saved_input_recipes` records the information to do this recomputation.
      There's some special logic in the saved tensor hooks to divert
      pack/unpack when this situation is detected.

    - Normally, autograd doesn't save the output of an operation (unless it
      is specifically needed for backwards).  But if the output of a SAVE
      region passes to a RECOMPUTE region, we must save it so we can run
      the recompute!  `output_slots` records tensors we've saved in this way.
      Additionally, if it turns out an output of a SAVE region doesn't need
      to be used for anything in recompute (e.g., it passes to a SAVE region),
      we still need to construct a placeholder for it, since we do have to
      return *something*; `output_schema` records this information.

    There's no corresponding _RecomputeRecord, because we never save anything
    on a RECOMPUTE op.  A RECOMPUTE op can induce a prior SAVE op to need to
    save some outputs it wouldn't have needed to save otherwise (SAVE-RECOMPUTE
    crossing), but it's always "producer's responsibility" to save a tensor for
    backwards.  Each output of a SAVE op has a `persist_output` thunk that a
    consumer calls if it turns out to need that output; calling it fills in the
    corresponding `output_slots` entry on the producer's record.

    This struct only holds durable state that is needed during recompute; there's also
    some transient state in `_SaveOpForwardScratch` that we use for bookkeeping within
    the forward execution of a SAVE remat.op only.
    """

    # Region-relative name for this op.
    op_name: str

    # --- Metadata about save_for_backward tensors
    # The actual tensors are stored in the autograd graph; however, remat
    # supports reporting how much memory was saved for backwards without
    # reference to the autograd graph, so we keep (weakly) track of every
    # tensor that was saved in the op so we can enumerate them.  If you use
    # `remat.save_for_backward`, you also get to attach names to saved
    # tensors (otherwise they get defaulted names), which makes reports more
    # readable.  NB: this does NOT include save for backward tensors that are
    # aliases of RECOMPUTE inputs, because those don't get saved at all! (so no
    # weak tensor to key on).
    saved_tensor_names: MutableMapping[torch.Tensor, str] = field(
        default_factory=WeakTensorKeyDictionary
    )

    # --- Metadata about save_for_backward tensors that are aliases of RECOMPUTE inputs
    # See _SavedInputRecipe for details.  The order of this list doesn't
    # matter (it's populated in the order inputs are saved for backwards.)
    saved_input_recipes: list[_SavedInputRecipe] = field(default_factory=list)

    # --- Metadata about outputs, and saved output tensors needed for recompute
    # The "schema" of the output, specifying metadata of the tensors and the
    # concrete values of all non-Tensor outputs, so we can reconstruct the
    # output during recompute to continue eager execution.
    output_schema: _OutputSchema | None = None

    # Outputs that are saved for recompute, so we can produce an output tensor
    # of a SAVE region that will be needed for RECOMPUTE.  This is indexed by
    # flattened position order in `output_schema`.  Not all outputs will be
    # saved, only the ones consumed by RECOMPUTE regions.
    #
    # Note that outputs saved for recompute DO trigger saved tensor hooks, see
    # :func:`saved_tensors_hooks` for more details.
    output_slots: dict[int, _OutputSlot] = field(default_factory=dict)


@dataclass
class _SaveOpForwardScratch:
    """Forward-only pack bookkeeping for one SAVE op, discarded when its forward returns."""

    # Every time a tensor is saved for backwards, and it wasn't directly named
    # using ``remat.save_for_backward``, we assign it a fresh name per this
    # counter.  Note this counter is per-op!
    unnamed_save_counter: int = 0

    # Names attached via ``remat.save_for_backward``, keyed weakly by tensor
    # identity. Weak keys never pin the tensor, so no entry can form the gc-invisible
    # Node <-> payload cycle a strong ref would (nor need a teardown to break it), and
    # a dead tensor's entry drops itself before its id can be recycled. The whole map
    # is discarded with this scratch when the op's forward returns, so a name can
    # neither leak onto another op's saves nor onto the user's tensor (as a setattr
    # tag would).
    pending_save_names: WeakTensorKeyDictionary = field(
        default_factory=WeakTensorKeyDictionary
    )

    # Storages of this op's identity-hook (resident) saves, weakly held. An output
    # whose storage is in here is already kept alive by autograd for backward, so it
    # is persisted eagerly (see :func:`_prepare_outputs`) at no memory cost.
    saved_identity_storages: set[StorageWeakRef] = field(default_factory=set)


# --------------------------------------------------------------------------
# Op dispatch: forward and recompute entry points
# --------------------------------------------------------------------------


def _run_save_op(
    state: _ActiveCheckpointRegion,
    name: str,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Output:
    """Run (forward) or skip (recompute) a SAVE op.

    Forward installs a nested ``saved_tensors_hooks`` that shadows checkpoint's
    hooks. The pack hook sorts each saved tensor into one of three fates:

    * a RECOMPUTE-sourced input (or a view of one) is *not* retained -- replay
      reproduces it, so pack records a rebuild recipe instead (this is what
      avoids pinning an upstream RECOMPUTE output just because we saved it);
    * with user :func:`saved_tensors_hooks` installed, delegate to them
      (autograd holds the packed payload, so the tensor can be offloaded);
    * otherwise identity: the tensor stays a normal autograd saved tensor on
      the original forward graph, and remat keeps only a weak report-only name.

    Recompute skips the call and returns each output from its persisted value,
    or a placeholder when none was saved (see :func:`_load_saved_outputs`).
    """

    region_state = state.region_state
    if state.phase is _Phase.RECOMPUTE:
        record = region_state.records.get(name)
        if record is None:
            raise RuntimeError(
                f"No SAVE op record for {_display_name(region_state, name)} during "
                "recompute; forward and recompute followed different code paths (or the "
                "op's policy differs between them)"
            )
        if record.saved_input_recipes:
            _rederive_saved_inputs(record, region_state, args, kwargs)
        return _load_saved_outputs(record, region_state)

    # The wrapper's duplicate-name check is phase-local; this guards the cross-phase
    # records dict against a phase-dispatch bug clobbering the forward record.
    assert name not in region_state.records, (
        f"torch_remat internal error: a SAVE record for {name!r} already exists"
    )
    record = _SaveRecord(op_name=name)
    region_state.records[name] = record
    scratch = _SaveOpForwardScratch()

    args, kwargs, input_infos = _unwrap_and_snapshot_inputs(region_state, args, kwargs)

    def pack(tensor: torch.Tensor) -> object:
        # Resolve the report name before branching on the save's fate, so the
        # unnamed-save counter advances in pack order and a save's saved.<i> label
        # doesn't shift depending on whether saved-tensor hooks are installed.
        name = _resolve_save_name(scratch, tensor)

        # A recompute-sourced saved input, or any saved view (whose base is always
        # non-stub), is not retained: replay reproduces it, so record a rebuild recipe
        # instead. A SAVE-sourced stub input is not reproduced by replay, so it falls
        # through and is retained like any other save.
        match = _classify_saved_input(tensor, input_infos)
        if match is not None:
            info, view_spec = match
            if view_spec is not None or not info.is_stub:
                slot_name = f"saved_input.{len(record.saved_input_recipes)}"
                record.saved_input_recipes.append(
                    _SavedInputRecipe(
                        path=info.path,
                        slot_name=slot_name,
                        view_spec=view_spec,
                        name=name,
                    )
                )
                return _SavedInputRef(slot_name)

        # User remat-level saved-tensor hooks: autograd holds the packed payload (the
        # original tensor may be dropped, e.g. offloaded). No version check and no
        # memory-report row -- the payload is opaque. The hook gets a *detached*
        # tensor (same storage/version), like the persist-output path: a payload
        # that retains the tensor (e.g. an identity pack) must not close the
        # gc-invisible C++ Node <-> payload cycle that _default_pack's detach
        # breaks -- without this, an op saving its own output through such a hook
        # leaks the graph if dropped without backward.
        hooks = _active_saved_tensors_hooks.get()
        if hooks is not None:
            pack_hook, unpack_hook = hooks
            return _SavedHookData(pack_hook(tensor.detach()), unpack_hook)

        # Default (identity hook): the tensor stays a normal autograd saved tensor.
        return _default_pack(record, scratch, tensor, name)

    def unpack(saved: object) -> torch.Tensor:
        if isinstance(saved, _SavedInputRef):
            return _load_saved_input(record, region_state, saved.slot_name)
        if isinstance(saved, _SavedHookData):
            # Restore via the pair that packed it -- works even after the user's
            # hook scope has exited.
            return saved.unpack_hook(saved.packed)
        assert isinstance(saved, _SavedTensor), saved
        return _default_unpack(region_state, record, saved)

    # Expose the forward scratch so save_for_backward (called inside the body) can
    # name this op's saves.
    token = _active_save_op.set(scratch)
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            output = function(*args, **kwargs)
    finally:
        _active_save_op.reset(token)

    validated_output = _validate_output(output, reject_leaves_for=(region_state, name))
    _record_output_schema(record, validated_output)
    return _prepare_outputs(region_state, record, scratch, validated_output)


def _run_recompute_op(
    state: _ActiveCheckpointRegion,
    name: str,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Output:
    """Run a RECOMPUTE op in both phases.

    On the original forward, an input that is an upstream SAVE op's output triggers
    that producer's persist-output thunk (so the skipped producer can reproduce it on
    replay) and is unwrapped to the grad-connected real for the body. During recompute
    the op simply reruns: skipped producers have already returned real outputs into
    the replay dataflow, so it holds no tape state and needs no substitution of its own.
    """

    region_state = state.region_state
    if state.phase is _Phase.RECOMPUTE:
        if name in region_state.records:
            raise RuntimeError(
                f"Conflicting checkpoint policies for {_display_name(region_state, name)}: "
                "the forward ran it as SAVE but recompute runs it as RECOMPUTE"
            )
        return _validate_output(function(*args, **kwargs))

    # Forward. Any SAVE-output input must trigger its producer's persist-output thunk.
    # A wrapping strategy (subclass / proxy) additionally hands the body a carrier we
    # must unwrap to the grad-connected real, rebuilding the arg pytree; a plain-output
    # strategy leaves SAVE outputs as real tensors, so we only walk for the
    # persist-output side effect and pass args through.
    if region_state.bare_op_strategy.wraps_outputs:

        def consume(_token: PathToken, leaf: object) -> object:
            handle = _save_output_handle(region_state, leaf)
            if handle is None:
                return leaf
            handle.persist_output()
            return handle.unwrap(leaf)

        args, kwargs = map_arg_leaves(consume, args, kwargs)
    else:
        for _token, leaf in iter_arg_leaves(args, kwargs):
            handle = _save_output_handle(region_state, leaf)
            if handle is not None:
                handle.persist_output()

    return _validate_output(
        function(*args, **kwargs), reject_leaves_for=(region_state, name)
    )


# --------------------------------------------------------------------------
# SAVE op forward: inputs, saved tensors, outputs
# --------------------------------------------------------------------------


def _unwrap_and_snapshot_inputs(
    region_state: _CheckpointRegionState,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], list[_InputInfo]]:
    """Unwrap a SAVE op's upstream-SAVE-output inputs and snapshot their layout.

    An input that is another SAVE op's output (found via the region's save-output
    index) is marked a *stub*: it is not reproduced by recompute, so pack must retain
    a saved copy rather than divert it to a rebuild recipe. Any other input is
    reproduced by replay. Unlike a RECOMPUTE consumer, this does NOT trigger the
    producer's persist-output thunk -- a skipped SAVE op does not rerun on replay, so
    its body needs no reproduced input.

    The snapshot is plain data (weak storage ref, no tensor reference), so the pack
    closure -- which outlives this call on the autograd graph -- keeps no input alive;
    pack classifies each saved tensor against it by storage. As in
    :func:`_run_recompute_op`, a wrapping strategy also unwraps each stub to its
    grad-connected real (rebuilding the arg pytree); a plain strategy passes args
    through untouched.
    """

    _assert_phase(_Phase.FORWARD)

    wraps_outputs = region_state.bare_op_strategy.wraps_outputs
    infos: list[_InputInfo] = []

    def visit(token: PathToken, leaf: object) -> object:
        handle = _save_output_handle(region_state, leaf)
        is_stub = handle is not None
        if is_stub and wraps_outputs:
            leaf = handle.unwrap(leaf)
        if isinstance(leaf, torch.Tensor) and not _is_placeholder(leaf):
            infos.append(
                _InputInfo(
                    path=token,
                    storage_ref=weakref.ref(leaf.untyped_storage()),
                    dtype=leaf.dtype,
                    shape=tuple(leaf.shape),
                    stride=tuple(leaf.stride()),
                    storage_offset=leaf.storage_offset(),
                    version=leaf._version,
                    is_stub=is_stub,
                )
            )
        return leaf

    if wraps_outputs:
        new_args, new_kwargs = map_arg_leaves(visit, args, kwargs)
        return new_args, new_kwargs, infos
    for token, leaf in iter_arg_leaves(args, kwargs):
        visit(token, leaf)
    return args, kwargs, infos


def _resolve_save_name(scratch: _SaveOpForwardScratch, tensor: torch.Tensor) -> str:
    """Return the report name for a tensor a SAVE op saved for backward.

    The :func:`save_for_backward` name recorded in this op's forward scratch for the
    exact tensor object, else a positional ``saved.<i>`` from the op's unnamed-save
    counter.
    """

    _assert_phase(_Phase.FORWARD)

    name = scratch.pending_save_names.get(tensor)
    if name is not None:
        return name
    name = f"saved.{scratch.unnamed_save_counter}"
    scratch.unnamed_save_counter += 1
    return name


def _default_pack(
    record: _SaveRecord,
    scratch: _SaveOpForwardScratch,
    tensor: torch.Tensor,
    name: str,
) -> _SavedTensor:
    """Default pack for a SAVE saved tensor: hand autograd an ordinary *detached* copy.

    Detaching matters: when an op saves one of its *own* outputs, handing autograd the
    live grad_fn-bearing tensor closes a C++ ``Node`` <-> ``TensorImpl`` refcount cycle
    (through the hook payload) that Python's gc cannot reclaim, so a graph dropped
    without a backward would leak. This is the Python equivalent of autograd's own
    ``tensor_data()`` cycle-break. The grad_fn-less tensor handed back at backward is
    fine because remat does not support double backward through a SAVE op.

    The payload also carries the save-time version for the in-place guard (see
    :func:`_default_unpack`); the report ``name`` goes in the weak report-only index.
    """

    _assert_phase(_Phase.FORWARD)

    saved = tensor.detach()
    record.saved_tensor_names[saved] = name
    # Note this tensor's storage (shared with the original) as resident, so an output
    # sharing it can be eagerly persisted (see :func:`_prepare_outputs`).
    scratch.saved_identity_storages.add(StorageWeakRef(saved.untyped_storage()))
    return _SavedTensor(saved, saved._version)


def _prepare_outputs(
    region_state: _CheckpointRegionState,
    record: _SaveRecord,
    scratch: _SaveOpForwardScratch,
    output: Output,
) -> Output:
    """Return a SAVE op's outputs, representing each per the region's bare-op strategy.

    ``strategy.make_output`` builds the forward stand-in for each output: ``value`` is
    what the op returns to its caller, and ``handle`` is registered in
    ``region_state.save_output_index`` under ``value`` -- unless it is ``None`` (the
    proxy, which self-identifies by type). Downstream consumers identify SAVE outputs
    uniformly through :func:`torch_remat._region._save_output_handle`.

    An output that is *itself* saved for backward (its storage is in
    ``scratch.saved_identity_storages``) is persisted eagerly. It is resident for
    backward anyway, so this costs no extra memory and lets a bare consumer of it work
    during recompute even without an opt-in strategy.
    """

    strategy = region_state.bare_op_strategy

    def make(index: int, real: torch.Tensor) -> tuple[Any, _SaveOutputHandle | None]:
        # Weak ref + lazy snapshot (see _PersistOutputThunk), so a SAVE output nothing
        # consumes is never pinned alive by its own persist-output thunk.
        persist_output = _PersistOutputThunk(
            record=record, slot_index=index, output_ref=weakref.ref(real)
        )
        value, handle = strategy.make_output(real, persist_output)
        if StorageWeakRef(real.untyped_storage()) in scratch.saved_identity_storages:
            persist_output()  # saved for backward, so resident anyway -- persist eagerly
        return value, handle

    tensors = _output_tensors(output)

    if not strategy.wraps_outputs:
        # Plain strategy: values are the real tensors, container unchanged -- register
        # only. Two output positions returning the *same* tensor collapse onto one
        # index key, so merge the handles: each position owns its own tape slot, and
        # consuming the shared value must persist all of them.
        for index, real in enumerate(tensors):
            _value, handle = make(index, real)
            existing = region_state.save_output_index.get(real)
            if existing is not None:
                handle = _merge_save_output_handles(existing, handle)
            region_state.save_output_index[real] = handle
        return output

    # Wrapping strategy: each output becomes a distinct carrier (never colliding in
    # the index, so no merge) -- rebuild the container around them.
    returned: list[Any] = []
    for index, real in enumerate(tensors):
        value, handle = make(index, real)
        if handle is not None:
            region_state.save_output_index[value] = handle
        returned.append(value)
    if isinstance(output, torch.Tensor):
        return returned[0]
    return (list if isinstance(output, list) else tuple)(returned)


@dataclass
class _PersistOutputThunk:
    """Callable that records one SAVE output on the tape, once (producer responsibility).

    A SAVE op is skipped during recompute, so any consumer that needs its output during
    replay fires this thunk during the forward: a bare consumer via the detection
    strategy, a ``remat.op`` consumer via the handle, or the eager persist of an
    output that is itself saved for backward.
    All consumer kinds share one tape slot, the output's position.

    The output is referenced *weakly* and snapshotted (detached) only when called:
    every consumer holds the output live at the moment it fires this, and until then
    nothing pins it -- a SAVE output consumed by nothing is freed after the forward
    rather than kept resident to backward. A dead weakref means no consumer, so the
    call is a no-op.

    When a remat saved-tensor hook is installed, the persisted value is packed through
    it (offload); otherwise it is held resident.
    """

    record: _SaveRecord
    slot_index: int
    output_ref: weakref.ReferenceType[torch.Tensor]

    def __call__(self) -> None:
        if self.slot_index in self.record.output_slots:
            return
        real = self.output_ref()
        if real is None:
            return
        detached = real.detach()
        hooks = _active_saved_tensors_hooks.get()
        if hooks is not None:
            # Bind the matching unpack hook to the slot so replay reloads via the pair
            # that packed it, not whatever hooks are active at load time.
            pack_hook, unpack_hook = hooks
            self.record.output_slots[self.slot_index] = _OutputSlot(
                tensor=None,
                version=None,
                packed=pack_hook(detached),
                unpack_hook=unpack_hook,
                requires_grad=real.requires_grad,
                is_leaf=real.is_leaf,
            )
            return
        self.record.output_slots[self.slot_index] = _OutputSlot(
            tensor=detached,
            version=detached._version,
            requires_grad=real.requires_grad,
            is_leaf=real.is_leaf,
        )


def _record_output_schema(record: _SaveRecord, output: Output) -> None:
    """Record a SAVE op's boundary output metadata and report labels on ``record``."""

    tensors = _output_tensors(output)
    container: type | None = None
    if not isinstance(output, torch.Tensor):
        container = list if isinstance(output, list) else tuple
    record.output_schema = _OutputSchema(
        container=container,
        specs=tuple(
            _OutputSpec(
                metadata=_TensorMetadata(
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    device=tensor.device,
                    storage_nbytes=tensor.untyped_storage().nbytes(),
                ),
                requires_grad=tensor.requires_grad,
            )
            for tensor in tensors
        ),
    )


# --------------------------------------------------------------------------
# SAVE op recompute: reload saved values and reproduce outputs
# --------------------------------------------------------------------------


def _rederive_saved_inputs(
    record: _SaveRecord,
    region_state: _CheckpointRegionState,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Fill recompute-sourced saved-input slots during a skipped SAVE replay.

    The op's saved inputs were not retained in the forward; capture their values
    now from the reproduced args -- detached, so the throwaway recompute graph is
    not kept alive -- for the op's unpack hook to hand back at backward. A saved
    view is rebuilt from the reproduced base with ``as_strided``.

    Load-bearing invariant: this fires only if replay actually reaches this
    skipped op. That holds because ``_TriggerCheckpointRecompute`` at the region
    output is the highest-index checkpoint holder, so the first backward unpack
    forces a full replay before any saved-input unpack reads a slot. A change
    that lets recompute stop earlier would silently skip rederiving them.
    """

    _assert_phase(_Phase.RECOMPUTE)

    leaves_by_path = dict(iter_arg_leaves(args, kwargs))
    captured_slots: dict[str, torch.Tensor] = {}
    for recipe in record.saved_input_recipes:
        captured = leaves_by_path[recipe.path]
        if _is_placeholder(captured):
            raise RuntimeError(
                f"{_display_name(region_state, record.op_name)} (policy SAVE) saved "
                f"an input ({recipe.name}) for backward that recompute was expected to "
                "reproduce, but it replays as a placeholder. The saved input is a bare "
                "(unwrapped) view or derivative of an upstream SAVE op's output, "
                "which is not reproduced during recompute. Wrap that producer in "
                "remat.op(..., policy=RECOMPUTE) so its output is real during "
                "replay."
            )
        if recipe.view_spec is None:
            value = captured.detach()
        else:
            value = _rebuild_saved_view(
                region_state, record.op_name, captured, recipe.view_spec
            )
        captured_slots[recipe.slot_name] = value
    # Replace the op's whole entry so a re-replay (retain_graph) starts from fresh values.
    region_state.rederived_saved_inputs[record.op_name] = captured_slots


def _load_saved_outputs(
    record: _SaveRecord, region_state: _CheckpointRegionState
) -> Output:
    """Return a skipped SAVE op's outputs during recompute, preserving the container.

    A persisted output is reproduced straight from the tape into the replay dataflow,
    with its autograd metadata restored so a downstream op rebuilds the same backward
    node (and its holders refill) it did on the forward. An output with no saved
    value -- genuinely dead -- is a storage-free placeholder that raises if its data
    is actually read.
    """

    _assert_phase(_Phase.RECOMPUTE)

    schema = record.output_schema
    if schema is None:
        raise RuntimeError(
            f"No output metadata available for "
            f"{_display_name(region_state, record.op_name)}"
        )

    # Free each output slot as it is served, except under retain_graph where a later
    # backward replays and reads it again. Outside a backward graph task the C++
    # accessor returns True (conservative "keep"), so no extra guard is needed.
    pop = not torch._C._autograd._get_current_graph_task_keep_graph()  # type: ignore[attr-defined]
    display_name = _display_name(region_state, record.op_name)
    outputs: list[torch.Tensor] = []
    for index, spec in enumerate(schema.specs):
        slot = record.output_slots.get(index)
        if slot is not None:
            loaded = _load_output_slot(
                record.output_slots,
                region_state,
                record.op_name,
                index,
                pop=pop,
            )
            outputs.append(
                _fabricate_recompute_input(
                    loaded, requires_grad=slot.requires_grad, is_leaf=slot.is_leaf
                )
            )
            continue

        source = (
            f"{display_name}."
            f"{_output_name(index, is_sequence=schema.container is not None)}"
        )
        placeholder = _make_placeholder_tensor(
            spec.metadata,
            _placeholder_message_text(source, display_name),
            requires_grad=spec.requires_grad,
        )
        if spec.requires_grad:
            # Make the placeholder non-leaf with a fresh grad_fn (saving
            # nothing) so a consumer that builds on it during replay sees the
            # same autograd shape the original output had.
            placeholder = _MakeNonLeaf.apply(placeholder)
        outputs.append(placeholder)

    if schema.container is None:
        return outputs[0]
    return schema.container(outputs)


def _load_output_slot(
    slots: dict[int, _OutputSlot],
    region_state: _CheckpointRegionState,
    op_name: str,
    index: int,
    *,
    pop: bool = False,
) -> torch.Tensor:
    """Load one persisted SAVE-output tensor by its output position.

    A slot is either resident (holds a live tensor, checked against its recorded
    version counter for in-place mutation -- PyTorch's own guard does not fire for
    tensors packed through custom hooks) or offloaded (recovered through the unpack
    hook bound at pack time). ``pop`` deletes the slot after loading so the tensor
    is freed as soon as replay serves it; the caller passes ``pop=False`` under
    ``retain_graph=True``, where a later backward reads the slot again.
    """

    name = _output_slot_name(index)
    slot = slots.get(index)
    if slot is None:
        saved_names = ", ".join(_output_slot_name(i) for i in sorted(slots)) or "(none)"
        raise RuntimeError(
            f"No saved tensor {name} for "
            f"{_display_name(region_state, op_name)} "
            f"(saved tensors: {saved_names}). "
            "This usually means forward and recompute followed different code paths."
        )
    unpack_hook = slot.unpack_hook
    if unpack_hook is not None:
        tensor = unpack_hook(slot.packed)
    else:
        tensor = slot.tensor
        assert tensor is not None  # resident slots always hold a real tensor
        if slot.version is not None and tensor._version != slot.version:
            raise RuntimeError(
                f"Saved tensor {name} for "
                f"{_display_name(region_state, op_name)} was modified in-place "
                "after it was saved"
            )
    if pop:
        del slots[index]
    return tensor


def _load_saved_input(
    record: _SaveRecord,
    region_state: _CheckpointRegionState,
    slot_name: str,
) -> torch.Tensor:
    """Return a saved input's recompute-materialized value for the op's unpack hook.

    :func:`_rederive_saved_inputs` filled the region's recompute buffer when replay
    reached the skipped op; a missing entry means replay never reached it. Not popped:
    it may back more than one saved-tensor reference, and is rebuilt each replay
    under ``retain_graph``.
    """

    slots = region_state.rederived_saved_inputs.get(record.op_name)
    tensor = slots.get(slot_name) if slots is not None else None
    if tensor is None:
        saved_names = ", ".join(slots) if slots else "(none)"
        raise RuntimeError(
            f"No saved input {slot_name} for "
            f"{_display_name(region_state, record.op_name)} "
            f"(saved inputs: {saved_names}). "
            "This usually means forward and recompute followed different code paths."
        )
    return tensor


def _default_unpack(
    region_state: _CheckpointRegionState,
    record: _SaveRecord,
    packed: _SavedTensor,
) -> torch.Tensor:
    """Return the detached SAVE copy autograd kept, re-checking for in-place mutation.

    Autograd's native version-counter guard does not fire for tensors packed through
    custom ``saved_tensors_hooks``, so compare the save-time version ourselves. The
    report index is consulted only to name the tensor in the error; a miss there
    degrades the message, never the check.
    """

    tensor = packed.tensor
    if tensor._version != packed.version:
        name = record.saved_tensor_names.get(tensor, "(unknown)")
        raise RuntimeError(
            f"Saved tensor {name} for "
            f"{_display_name(region_state, record.op_name)} was modified "
            "in-place after it was saved"
        )
    return tensor


def _fabricate_recompute_input(
    tensor: torch.Tensor,
    *,
    requires_grad: bool,
    is_leaf: bool,
) -> torch.Tensor:
    """Return a recompute stand-in reproducing the saved input's autograd metadata.

    ``requires_grad`` (captured at save time) must match so the RECOMPUTE op rebuilds
    its backward node and its ``save_for_backward`` packs refill checkpoint's holders.
    A requires-grad input is always non-leaf here -- a remat.op never yields a
    requires-grad *leaf* (see :func:`_reject_grad_leaf`) -- so it gets a fresh
    grad_fn via :class:`_MakeNonLeaf` (which saves nothing, so it adds no checkpoint pack);
    ``is_leaf`` is carried only to assert that invariant.
    """

    base = tensor.detach()
    if not requires_grad:
        # Non-requires-grad tensors are always leaves; nothing more to reproduce.
        return base
    base.requires_grad_(True)
    assert not is_leaf, "a requires-grad recompute input must be non-leaf"
    return _MakeNonLeaf.apply(base)


class _MakeNonLeaf(torch.autograd.Function):
    """Identity that makes its output non-leaf via a fresh grad_fn that saves nothing.

    Used during recompute to turn a tape-loaded SAVE output into a non-leaf
    requires-grad tensor without emitting any save_for_backward pack -- so the
    downstream RECOMPUTE op builds its backward node (and its holders fill)
    without desyncing checkpoint's saved-tensor counter.

    This node is fabricated only while recompute rebuilds the forward graph, and
    non-reentrant checkpoint discards that graph once it has refilled saved
    tensors. Its backward must therefore never run; reaching it means the
    synthetic node leaked into the real backward graph, so it raises rather than
    silently passing gradient through.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        del ctx
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        raise RuntimeError(
            "torch_remat internal error: _MakeNonLeaf.backward was invoked. This "
            "synthetic autograd node exists only to reproduce a saved tensor's "
            "autograd metadata during recompute and must never be "
            "backpropagated through."
        )


# --------------------------------------------------------------------------
# Shared output and validation helpers
# --------------------------------------------------------------------------


_OP_OUTPUT_TYPE_MESSAGE = (
    "remat.op function must return a Tensor, or a tuple or list of Tensors"
)


def _validate_output(
    output: Any,
    reject_leaves_for: tuple[_CheckpointRegionState, str] | None = None,
) -> Output:
    """Check a remat op returned a Tensor, or a one-hop tuple/list of Tensors.

    Returns the output unchanged. When ``reject_leaves_for`` (the ``(region_state,
    op_name)`` of an eager forward op) is given, the same walk also rejects a
    requires-grad *leaf* output (see :func:`_reject_grad_leaf`); the recompute rerun
    passes ``None`` since leaf rejection already ran on the forward.
    """

    if isinstance(output, torch.Tensor):
        if reject_leaves_for is not None:
            _reject_grad_leaf(output, reject_leaves_for)
        return output
    if isinstance(output, (tuple, list)):
        for leaf in output:
            if not isinstance(leaf, torch.Tensor):
                raise RuntimeError(_OP_OUTPUT_TYPE_MESSAGE)
            if reject_leaves_for is not None:
                _reject_grad_leaf(leaf, reject_leaves_for)
        return output
    raise RuntimeError(_OP_OUTPUT_TYPE_MESSAGE)


def _reject_grad_leaf(
    tensor: torch.Tensor, region_and_op: tuple[_CheckpointRegionState, str]
) -> None:
    """Raise if a remat.op returned a requires-grad *leaf* (a graph endpoint).

    An ``autograd.Function`` or any differentiable op always gives its output a grad_fn, so a
    requires-grad leaf out of a remat.op means it wrapped a bare allocation (e.g.
    ``torch.zeros(..., requires_grad=True)``) rather than a real computation. That has no
    legitimate use inside a checkpoint region -- the value is disconnected from the region's
    inputs -- and it makes recompute's autograd shape ambiguous, so we reject it rather than
    silently reproduce it.
    """

    if tensor.requires_grad and tensor.is_leaf:
        region_state, op_name = region_and_op
        raise RuntimeError(
            f"{_display_name(region_state, op_name)} returned a leaf tensor that "
            "requires grad (a requires_grad tensor with no grad_fn). A remat.op "
            "must return the result of a real computation; this usually means it "
            "wrapped a bare allocation like torch.zeros(..., requires_grad=True). "
            "Move the allocation outside the checkpoint region (or do not wrap it "
            "in remat.op)."
        )


def _output_tensors(output: Output) -> tuple[torch.Tensor, ...]:
    """Return output tensors in return-schema order.

    ``output`` is validated to a Tensor or a flat tuple/list of Tensors, so every
    value leaf is a Tensor.
    """

    return cast("tuple[torch.Tensor, ...]", value_leaves(output))


def _output_name(index: int, *, is_sequence: bool) -> str:
    """Return the canonical report name for one output position."""

    if not is_sequence:
        return "out"
    return str(index)


def _output_slot_name(index: int) -> str:
    """Render a persisted output slot's report/error name from its position index.

    Display only (memory reports, error messages); slots are keyed by the raw int.
    """

    return f"output.{index}"


def _validate_name(name: str, *, what: str) -> None:
    """Validate a relative op or tensor name passed by user code."""

    if not name:
        raise ValueError(f"torch_remat {what} must be non-empty")
