# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Explicit activation rematerialization helpers for custom autograd functions."""

from __future__ import annotations

import contextlib
import contextvars
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from types import TracebackType
from typing import (
    Any,
    Callable,
    cast,
    Iterator,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
)

import torch
from torch.utils import _pytree as pytree

# Custom autograd Function outputs handled by RematHandle. The top-level
# checkpoint wrapper supports richer containers, but one remat-aware Function
# boundary is intentionally limited to the schemas PyTorch autograd.Function
# forwards commonly return and that record_outputs can replay precisely.
Output: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]

# PyTorch's save_for_backward path preserves None entries in saved_tensors. We
# model that explicitly because optional saved activations need stable names and
# positions just like tensor activations.
SavedTensor: TypeAlias = torch.Tensor | None

# PyTorch non-reentrant checkpoint expects a callable returning one context for
# original forward and one context for recompute.
CheckpointContextFn: TypeAlias = Callable[
    [],
    tuple[
        contextlib.AbstractContextManager[None],
        contextlib.AbstractContextManager[None],
    ],
]

_P = ParamSpec("_P")
_R = TypeVar("_R")


class CheckpointPolicy(Enum):
    """Policy controlling how an activation record is handled under checkpointing."""

    # Rerun this custom autograd forward during checkpoint recompute.
    RECOMPUTE = 0

    # Skip this custom autograd forward during checkpoint recompute and replay
    # the saved backward inputs plus placeholder outputs from the remat tape.
    SAVE = 1


class AutogradCtx(Protocol):
    """Minimal protocol for the context object passed to autograd.Function.forward."""

    needs_input_grad: tuple[bool, ...]

    def save_for_backward(self, *tensors: SavedTensor) -> None:
        """Save tensors for the custom autograd backward."""
        ...


class _Phase(Enum):
    """Execution phase for the active checkpoint region.

    Backward is intentionally absent: torch_remat's private tape only mediates
    transfer from original forward to checkpoint recompute. After recompute
    calls ctx.save_for_backward, ordinary PyTorch autograd owns backward.
    """

    FORWARD = 0
    RECOMPUTE = 1


@dataclass(frozen=True)
class _TensorMetadata:
    """Tensor metadata used for data-inaccessible replay outputs.

    Requires-grad is not stored here because autograd.Function.forward runs
    under no-grad semantics; the autograd engine attaches grad_fn information to
    returned tensors after forward returns.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass
class _SavedTensorSlot:
    """One tensor slot owned by an op record."""

    # Tensor passed to ctx.save_for_backward. None is a valid saved slot value.
    tensor: SavedTensor

    # Version counter observed when the tensor was saved.
    version: int | None


@dataclass
class _OpRecord:
    """One custom autograd op record in the forward tape."""

    # Region-relative name for this activation record.
    op_name: str

    # Checkpoint policy established for this custom autograd call. Native
    # regions and low-level records do not use the high-level policy flow.
    policy: CheckpointPolicy | None = None

    # Unified namespace of named tensor slots. User-saved tensors use their
    # provided names; retained recompute inputs use input.<arg_index>. A
    # single namespace ensures name collisions are caught across both types.
    tensor_slots: dict[str, _SavedTensorSlot] = field(default_factory=dict)

    # Ordered names of slots that ctx.save_for_backward should receive during
    # recompute. Only populated by save_for_backward (not save_or_load_inputs).
    saved_for_backward_names: list[str] = field(default_factory=list)

    # Whether the observed output schema was a tuple rather than a single tensor.
    output_is_tuple: bool = False

    # Ordered output metadata used to build fresh data-inaccessible outputs during
    # recompute. We intentionally do not preserve output aliasing relationships
    # for simplicity.
    output_metadata: tuple[_TensorMetadata, ...] = ()

    # The fields below only apply native saved regions, and are empty for
    # conventional custom autograd calls.

    # Weak refs to SAC-cached tensor outputs in native function regions, keyed
    # by unstable report labels such as aten.mm.default#0. These are report-only
    # and only appear while the tensors are live.
    native_sac_tensors: dict[str, weakref.ReferenceType[torch.Tensor]] = field(
        default_factory=dict
    )

    # Native-only per-op occurrence counts used to build unstable report labels.
    native_op_counts: dict[str, int] = field(default_factory=dict)

    # Native-only SAC contexts. They are created for native_save_region records
    # and entered only while that native function region is running.
    native_sac_contexts: (
        tuple[
            contextlib.AbstractContextManager[None],
            contextlib.AbstractContextManager[None],
        ]
        | None
    ) = None

    # Whether this record has released remat-owned state after recompute.
    released: bool = False

    def store_saved_tensor_slot(
        self,
        region_state: _CheckpointRegionState,
        tensor_name: str,
        tensor: SavedTensor,
    ) -> None:
        """Store one named tensor slot on this record.

        ``None`` is a valid tensor value.
        """

        if tensor_name in self.tensor_slots:
            raise RuntimeError(
                f"Duplicate saved tensor {tensor_name} for "
                f"{_display_name(region_state, self.op_name)}"
            )

        version = tensor._version if isinstance(tensor, torch.Tensor) else None
        self.tensor_slots[tensor_name] = _SavedTensorSlot(
            tensor=tensor,
            version=version,
        )

    def load_saved_tensor(
        self,
        region_state: _CheckpointRegionState,
        tensor_name: str,
    ) -> SavedTensor:
        """Load one named saved tensor from this record.

        Called during recompute by maybe_load_saved (for user-saved tensors
        to pass to ctx.save_for_backward) and by save_or_load_inputs (for
        retained input tensors). The slot keeps its reference until
        _release_record_after_recompute_if_needed clears the record at the
        end of the op's recompute.
        """

        if self.released:
            raise RuntimeError(
                f"No saved tensor {tensor_name} for "
                f"{_display_name(region_state, self.op_name)}; "
                "the remat tape was already released"
            )

        if tensor_name not in self.tensor_slots:
            saved_names = (
                ", ".join(self.tensor_slots) if self.tensor_slots else "(none)"
            )
            raise RuntimeError(
                f"No saved tensor {tensor_name} for "
                f"{_display_name(region_state, self.op_name)} "
                f"(saved tensors: {saved_names}). "
                "This usually means forward and recompute followed different code paths, "
                "or save_for_backward was not called with this tensor name during the "
                "original forward."
            )

        slot = self.tensor_slots[tensor_name]
        tensor = slot.tensor
        if tensor is None:
            return None

        if slot.version is not None and tensor._version != slot.version:
            raise RuntimeError(
                f"Saved tensor {tensor_name} for "
                f"{_display_name(region_state, self.op_name)} was modified in-place "
                "after it was saved"
            )
        return tensor

    def record_output_schema(self, output: Output) -> None:
        """Record boundary output metadata and report labels."""

        tensors = _output_tensors(output)
        self.output_is_tuple = isinstance(output, tuple)
        self.output_metadata = tuple(
            _TensorMetadata(
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for tensor in tensors
        )
        for index, tensor in enumerate(tensors):
            setattr(
                tensor,
                _REPORT_OUTPUT_NAME_ATTR,
                f"observed_output.{_output_name(index, output_is_tuple=self.output_is_tuple)}",
            )

    def placeholder_output(self, region_state: _CheckpointRegionState) -> Output:
        """Return placeholder output for a skipped non-recompute call."""

        if self.released:
            raise RuntimeError(
                f"The remat tape for {_display_name(region_state, self.op_name)} "
                "was already released"
            )

        if not self.output_metadata:
            raise RuntimeError(
                f"No output metadata available for "
                f"{_display_name(region_state, self.op_name)}"
            )

        placeholders: list[torch.Tensor] = []
        for index, metadata in enumerate(self.output_metadata):
            source = (
                f"{_display_name(region_state, self.op_name)}."
                f"{_output_name(index, output_is_tuple=self.output_is_tuple)}"
            )
            placeholders.append(
                _make_placeholder_tensor(
                    metadata,
                    f"{source} was skipped during recompute. This is likely "
                    "because the output of a remat-aware autograd Function with "
                    "policy SAVE was consumed by a native PyTorch op not wrapped in "
                    "remat.native_save_region. To fix, either: (1) wrap the "
                    "native op with remat.native_save_region, (2) move it into "
                    "a custom autograd Function with auto_forward, or "
                    "(3) change the upstream op's policy to RECOMPUTE.",
                )
            )

        if not self.output_is_tuple:
            return placeholders[0]
        return tuple(placeholders)


@dataclass
class _CheckpointRegionState:
    """State for one checkpointed region shared by forward and recomputation."""

    # Optional diagnostic name for the checkpoint region.
    region_name: str | None = None

    # Forward tape of op records keyed by region-relative name. Dict insertion
    # order is the tape execution order.
    records: dict[str, _OpRecord] = field(default_factory=dict)


@dataclass(frozen=True)
class _ActiveCheckpointRegion:
    """Context-local pointer to the active checkpoint region and phase."""

    # Shared state for the active checkpoint region.
    region_state: _CheckpointRegionState

    # Whether execution is original forward or recomputation.
    phase: _Phase

    # Op names that have called get_handle in this phase, for duplicate detection.
    # The dataclass is frozen so the context pointer is immutable, but the set
    # itself is phase-local mutable state accumulated while the context is active.
    handle_names: set[str] = field(default_factory=set)


_state: contextvars.ContextVar[_ActiveCheckpointRegion | None] = contextvars.ContextVar(
    "torch_remat_state",
    default=None,
)
_active_op: contextvars.ContextVar[tuple[str, CheckpointPolicy] | None] = (
    contextvars.ContextVar(
        "torch_remat_op",
        default=None,
    )
)
# These tensor attrs are deliberately lightweight projections of the producing
# OpRecord plus output index, not references back to the record. A direct
# tensor -> record pointer would create cycles through record.tensor_slots when
# a saved tensor is also a remat output.
#
# Outputs from a producer with SAVE policy replay as placeholders. A downstream
# RECOMPUTE consumer that receives one during replay needs a real tensor instead,
# so forward marks these outputs and save_or_load_inputs retains only the inputs
# whose producer policy is SAVE. This is deliberately narrower than saving every
# input to every RECOMPUTE op; outputs from RECOMPUTE producers will be real
# during replay and do not need extra tape storage.
_STUB_ON_RECOMPUTE_ATTR = "_torch_remat_stub_on_recompute"
_REPORT_PLACEHOLDER_MESSAGE_ATTR = "_torch_remat_placeholder"
_REPORT_OUTPUT_NAME_ATTR = "_torch_remat_report_output_name"


def _is_stub_on_recompute(tensor: torch.Tensor) -> bool:
    """Return whether recompute should replace this tensor with a placeholder."""

    try:
        value = object.__getattribute__(tensor, _STUB_ON_RECOMPUTE_ATTR)
    except AttributeError:
        return False
    return bool(value)


class _PlaceholderTensor(torch.Tensor):
    """Storage-free placeholder for skipped recompute outputs.

    Placeholder outputs exist so a skipped SAVE op can satisfy the autograd
    Function output schema without retaining its output data. Framework code may
    still apply metadata-only aliasing operations such as detach, view, or slice
    before a later RECOMPUTE op has a chance to replace the placeholder with a
    saved real input. Those operations should keep working. Operations that
    would create fresh data, such as sin or clone, must fail with the remat
    diagnostic because consuming placeholder values is a user-code bug.

    This is a Tensor subclass because ordinary Python tensor construction APIs
    cannot represent a non-empty CPU/CUDA tensor with arbitrary size/stride and
    no backing allocation. ``empty_strided`` allocates the implied storage, and
    ``set_``/``as_strided`` either resize storage to fit or reject undersized
    non-resizable storage. ``_make_wrapper_subclass`` is the Python-level PyTorch
    mechanism for a tensor-shaped object whose metadata is real but whose data
    is intentionally absent.

    Mutating ops are rejected from schema alias metadata. To avoid maintaining
    a hard-coded list of view ops, non-mutating dispatch runs the same op on
    meta mirrors. Outputs that share meta storage with a placeholder input are
    treated as metadata-only aliases and wrapped back into placeholders; outputs
    with fresh meta storage are data-producing and rejected.
    """

    @staticmethod
    def __new__(
        cls,
        metadata: _TensorMetadata,
        message: str,
        *,
        requires_grad: bool = False,
    ) -> _PlaceholderTensor:
        placeholder = torch.Tensor._make_wrapper_subclass(
            cls,
            metadata.shape,
            strides=metadata.stride,
            dtype=metadata.dtype,
            device=metadata.device,
            requires_grad=requires_grad,
        )
        setattr(placeholder, _REPORT_PLACEHOLDER_MESSAGE_ATTR, message)
        return placeholder

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        kwargs = {} if kwargs is None else kwargs
        try:
            schema = object.__getattribute__(func, "_schema")
        except AttributeError:
            schema = None
        if schema is not None and schema.is_mutable:
            raise RuntimeError(_placeholder_message_from_args(args, kwargs))

        sources_by_meta_storage: dict[int, _PlaceholderTensor] = {}

        def unwrap_placeholder(value: Any) -> Any:
            if not isinstance(value, _PlaceholderTensor):
                return value

            meta_value = torch.empty_strided(
                tuple(value.shape),
                tuple(value.stride()),
                dtype=value.dtype,
                device="meta",
                requires_grad=value.requires_grad,
            )
            # Meta tensors preserve storage identity across aliasing/view ops.
            # This gives us a generic "metadata-only" test without naming every
            # view-like aten operator in this dispatch handler.
            sources_by_meta_storage[meta_value.untyped_storage()._cdata] = value
            return meta_value

        try:
            meta_output = func(
                *pytree.tree_map(unwrap_placeholder, args),
                **pytree.tree_map(unwrap_placeholder, kwargs),
            )
        except Exception as error:
            raise RuntimeError(_placeholder_message_from_args(args, kwargs)) from error

        def wrap_meta_output(value: Any) -> Any:
            if not isinstance(value, torch.Tensor):
                return value

            source = sources_by_meta_storage.get(value.untyped_storage()._cdata)
            if source is None:
                raise RuntimeError(_placeholder_message_from_args(args, kwargs))

            return _make_placeholder_tensor(
                _TensorMetadata(
                    shape=tuple(value.shape),
                    stride=tuple(value.stride()),
                    dtype=value.dtype,
                    device=source.device,
                ),
                _placeholder_message(source),
                requires_grad=value.requires_grad,
            )

        return pytree.tree_map(wrap_meta_output, meta_output)


def _checkpoint_context_fn(
    region_name: str | None = None,
) -> tuple[
    contextlib.AbstractContextManager[None], contextlib.AbstractContextManager[None]
]:
    """Return context managers for PyTorch non-reentrant checkpointing.

    Pass this as ``context_fn`` to ``torch.utils.checkpoint.checkpoint``. The two
    contexts share one checkpoint region state so cached op records from the
    original forward can be replayed by relative op name during recomputation.
    """

    region_state = _CheckpointRegionState(region_name=region_name)
    return (
        _CheckpointPhaseContext(
            region_state,
            _Phase.FORWARD,
        ),
        _CheckpointPhaseContext(
            region_state,
            _Phase.RECOMPUTE,
        ),
    )


class _TriggerCheckpointRecompute(torch.autograd.Function):
    """Autograd identity that installs one checkpoint-hook unpack boundary."""

    @staticmethod
    def forward(ctx: AutogradCtx, output: torch.Tensor) -> torch.Tensor:
        # Save a zero-element tensor with the same dtype/device as the output so
        # PyTorch's checkpoint unpack hook is triggered without retaining output
        # storage or forcing a device transfer at the checkpoint boundary.
        ctx.save_for_backward(
            torch.empty((0,), dtype=output.dtype, device=output.device)
        )
        return output.view_as(output)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # Trigger PyTorch non-reentrant checkpoint's saved-tensor unpack hook at
        # the user-visible checkpoint boundary before nested custom autograd
        # Function backward bodies unpack their own saved tensors.
        (_,) = ctx.saved_tensors
        return grad_output


def _checkpoint_recompute_boundary(output: Any) -> Any:
    """Force non-reentrant checkpoint replay before nested custom backprop.

    Non-reentrant PyTorch checkpoint starts replay lazily when backward first
    unpacks a tensor saved under checkpoint hooks. For remat regions containing
    nested custom autograd Functions, replay should start at the checkpoint
    output boundary rather than from inside an inner backward body. This helper
    preserves values while inserting that boundary trigger on tensor outputs.
    """

    if isinstance(output, torch.Tensor):
        return _TriggerCheckpointRecompute.apply(output)
    # Exact builtin container checks keep the public contract small and avoid
    # silently changing the type or invariants of subclasses such as namedtuple,
    # custom mappings, or domain objects with constructor requirements.
    if type(output) is tuple:
        return tuple(_checkpoint_recompute_boundary(item) for item in output)
    if type(output) is list:
        return [_checkpoint_recompute_boundary(item) for item in output]
    if type(output) is dict:
        return {
            key: _checkpoint_recompute_boundary(value) for key, value in output.items()
        }
    raise RuntimeError(
        "torch_remat checkpoint function must return a Tensor or an exact "
        "tuple/list/dict containing only supported checkpoint outputs"
    )


def checkpoint(
    *,
    region_name: str | None = None,
    determinism_check: str = "none",
    preserve_rng_state: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator that builds a torch_remat checkpoint wrapper.

    Checkpoint options, the function, and user function arguments are supplied
    in three separate calls: ``checkpoint(...)(function)(*args, **kwargs)``.
    This avoids collisions between checkpoint option names, function attributes,
    and user keyword arguments.

    NB: ``checkpoint(function)(*args, **kwargs)`` is intentionally unsupported
    to avoid confusion with ``torch.utils.checkpoint.checkpoint``, which cannot
    support that calling convention for BC reasons. ``torch_remat`` always uses
    non-reentrant checkpointing internally and only exposes the PyTorch knobs
    that are expected to matter to remat users.

    Keyword args:
        region_name: Optional diagnostic name for this checkpoint region. This
            name appears in torch_remat errors and memory reports.
        determinism_check: A string specifying the PyTorch determinism check to
            perform during non-reentrant checkpoint recomputation. ``"default"``
            compares the shapes, dtypes, and devices of recomputed tensors
            against the saved tensors. ``"none"`` disables this check.
            Currently these are the only two supported PyTorch values.
            Default: ``"none"``
        preserve_rng_state: If ``False``, omit stashing and restoring the RNG
            state during each checkpoint. Note that under ``torch.compile``,
            this flag does not take effect and PyTorch always preserves RNG
            state. Default: ``True``

    Example:
        ```python
        import torch_remat as remat

        y = remat.checkpoint(region_name="layers.0")(block)(x)
        ```
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped_function(*inner_args: Any, **inner_kwargs: Any) -> Any:
            # The boundary trigger must be inside the function passed to PyTorch
            # checkpoint so its saved tensor is covered by PyTorch's checkpoint
            # hooks and can force replay before nested custom backward code runs.
            output = function(*inner_args, **inner_kwargs)
            return _checkpoint_recompute_boundary(output)

        def checkpointed_function(*args: Any, **kwargs: Any) -> Any:
            return _torch_checkpoint_with_forward_exception_cleanup(
                wrapped_function,
                function_args=args,
                function_kwargs=kwargs,
                context_fn=lambda: _checkpoint_context_fn(region_name),
                determinism_check=determinism_check,
                preserve_rng_state=preserve_rng_state,
            )

        return checkpointed_function

    return decorate


def _keep_graph() -> bool:
    """Return whether the current backward pass uses retain_graph=True."""

    return torch._C._autograd._get_current_graph_task_keep_graph()  # type: ignore[attr-defined]


def _torch_checkpoint_with_forward_exception_cleanup(
    function: Callable[..., Any],
    function_args: tuple[Any, ...],
    function_kwargs: dict[str, Any],
    context_fn: CheckpointContextFn,
    determinism_check: str,
    preserve_rng_state: bool,
) -> Any:
    """Run PyTorch non-reentrant checkpoint with local exception cleanup.

    This is intentionally just the public ``torch.utils.checkpoint.checkpoint``
    non-reentrant branch, plus ``gen.close()`` when the user forward raises.
    Once PyTorch's public implementation closes the generator on that path, this
    helper can be replaced with a direct call to ``torch.utils.checkpoint``.
    Upstream fix: https://github.com/pytorch/pytorch/pull/184018
    """

    # Import locally so importing torch_remat does not eagerly bind PyTorch's
    # private checkpoint implementation. This helper is the only compatibility
    # layer that should depend on that private symbol.
    from torch.utils.checkpoint import _checkpoint_without_reentrant_generator

    # Match the public non-reentrant checkpoint defaults for the private
    # generator parameters that appear after determinism_check.
    checkpoint_debug = False
    checkpoint_early_stop = True
    gen = _checkpoint_without_reentrant_generator(
        function,
        preserve_rng_state,
        context_fn,
        determinism_check,
        checkpoint_debug,
        checkpoint_early_stop,
        *function_args,
        **function_kwargs,
    )
    next(gen)
    try:
        ret = function(*function_args, **function_kwargs)
    except BaseException:
        # PyTorch's public checkpoint() currently leaves the non-reentrant
        # generator suspended if the user forward raises. That keeps both the
        # PyTorch checkpoint hook and our forward context installed as long as
        # the traceback is alive, which can corrupt later checkpoint regions.
        # Drive the same private generator directly so we can close it on the
        # exceptional path; this mirrors the upstream fix.
        gen.close()
        raise

    try:
        next(gen)
    except StopIteration:
        return ret
    raise RuntimeError("torch.utils.checkpoint generator did not stop")


def is_recomputing() -> bool:
    """Return whether execution is currently in checkpoint recomputation.

    Example:
        ```python
        if not remat.is_recomputing():
            log_forward_only_metric(x)
        ```
    """

    # Outside remat.checkpoint there is no active state, and user code should
    # treat execution as ordinary forward rather than recompute.
    state = _state.get()
    return state is not None and state.phase is _Phase.RECOMPUTE


@contextlib.contextmanager
def _active_op_context(
    name: str,
    policy: CheckpointPolicy,
) -> Iterator[None]:
    token = _active_op.set((name, policy))
    try:
        yield
    finally:
        _active_op.reset(token)


def op(
    function: Callable[_P, _R],
    name: str | None = None,
    *,
    policy: CheckpointPolicy,
) -> Callable[_P, _R]:
    """Annotate one remat-aware custom autograd op call.

    This lets call sites keep the remat name and policy out of the
    ``autograd.Function.apply`` argument list.

        ```python
        return remat.op(MyOp.apply, "my.op", policy=remat.CheckpointPolicy.SAVE)(
            x,
            y,
        )
        ```
    """

    if not callable(function):
        raise RuntimeError("op expects a function as its first argument")
    if name is None:
        raise RuntimeError("op(function, ...) expects an op_name")
    _validate_name(name, what="op_name")
    if not isinstance(policy, CheckpointPolicy):
        raise RuntimeError("op expects a CheckpointPolicy")

    @wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _active_op_context(name, policy):
            return function(*args, **kwargs)

    return wrapper


class _InertRematHandle:
    """Handle for autograd Function forwards outside a checkpoint region.

    All methods delegate to ordinary autograd behavior without recording
    anything on a remat tape.
    """

    def __init__(self, ctx: AutogradCtx) -> None:
        self._ctx = ctx

    def maybe_load_saved(self) -> Output | None:
        return None

    def save_or_load_inputs(
        self,
        *args: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return _return_like_args(args)

    def save_for_backward(
        self,
        saved_tensors: Mapping[str, torch.Tensor | None],
    ) -> None:
        tensors_to_save: dict[str, SavedTensor] = dict(saved_tensors)
        for tensor_name, tensor in tensors_to_save.items():
            _validate_name(tensor_name, what="save_for_backward key")
            if "." in tensor_name:
                raise RuntimeError(
                    f"save_for_backward key {tensor_name!r} must not contain '.'"
                )
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"save_for_backward.{tensor_name} must be a tensor or None"
                )
        self._ctx.save_for_backward(*tensors_to_save.values())

    def record_outputs(self, *outs: Output) -> Output:
        if len(outs) == 1:
            return outs[0]
        if all(isinstance(out, torch.Tensor) for out in outs):
            return cast(tuple[torch.Tensor, ...], outs)
        raise RuntimeError("record_outputs accepts tensors or a single output tuple")


class RematHandle:
    """Handle for one remat-aware autograd Function forward call.

    This handle lets us do materialization related save/load inside of a
    custom autograd function forwards.  Obtain this at the start of a custom
    ``autograd.Function.forward`` using :func:`get_handle`.

    Example:
        ```python
        handle = remat.get_handle(ctx, op_name, remat_policy)
        if (ret := handle.maybe_load_saved()) is not None:
            return ret

        x = handle.save_or_load_inputs(x)
        y = my_op_fwd1(x)
        z = my_op_fwd2(y)
        handle.save_for_backward({"x": x, "y": y})
        return handle.record_outputs(z)
        ```
    """

    def __init__(
        self,
        ctx: AutogradCtx,
        op_name: str,
        policy: CheckpointPolicy,
        active_state: _ActiveCheckpointRegion,
        record: _OpRecord,
    ) -> None:
        self._ctx = ctx
        self._op_name = op_name
        self._policy = policy
        self._active_state = active_state
        self._record = record

    def maybe_load_saved(self) -> Output | None:
        """Load saved tensors and return stub outputs during replay, if possible.

        The idiomatic use of this method is to test whether a custom autograd
        forward can short-circuit its expensive body. If this returns a
        non-None result, return it directly.

        In the forward phase this always returns ``None``. During recompute,
        this returns data-inaccessible placeholder outputs for ``SAVE`` ops and
        loads the tensors previously recorded by :meth:`save_for_backward`
        into ``ctx``.  Outputs are never retained solely because they were
        returned; if a later ``RECOMPUTE`` op needs a real value that would
        otherwise be a placeholder, that use site is responsible for
        saving/loading it via :meth:`save_or_load_inputs`.

        Example:
            ```python
            handle = remat.get_handle(ctx, op_name, remat_policy)
            if (ret := handle.maybe_load_saved()) is not None:
                return ret
            ```
        """

        active_state = self._active_state
        if (
            active_state.phase is _Phase.FORWARD
            or self._policy is CheckpointPolicy.RECOMPUTE
        ):
            return None

        record = self._record
        output = record.placeholder_output(active_state.region_state)

        # SAVE-policy recompute skips the user forward body, so it must still
        # populate the recompute autograd graph's ctx in the same order as the
        # original ctx.save_for_backward call.
        saved_tensors: list[SavedTensor] = []
        for tensor_name in record.saved_for_backward_names:
            saved_tensors.append(
                record.load_saved_tensor(active_state.region_state, tensor_name)
            )
        if saved_tensors:
            self._ctx.save_for_backward(*saved_tensors)
        _release_record_after_recompute_if_needed(record)
        return output

    def save_or_load_inputs(
        self,
        *args: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Save unavailable recompute inputs in forward, or load them in replay.

        This is only active for ``RECOMPUTE`` ops. In the original forward, any
        input that would replay as a data-inaccessible placeholder is retained by
        this handle. During recompute, the retained real input is loaded back
        and returned in place of the placeholder. For ``SAVE`` ops, inputs are
        returned unchanged.

        Place this immediately after :meth:`maybe_load_saved` and before the
        expensive forward body.

        Example:
            ```python
            x = handle.save_or_load_inputs(x)
            y = my_op_fwd1(x)
            ```
        """

        if self._policy is not CheckpointPolicy.RECOMPUTE:
            assert self._active_state.phase is not _Phase.RECOMPUTE, (
                f"save_or_load_inputs called on {self._op_name} with policy "
                f"{self._policy} during recompute; SAVE ops must short-circuit "
                f"via maybe_load_saved()"
            )
            return _return_like_args(args)

        active_state = self._active_state
        if active_state.phase is _Phase.RECOMPUTE:
            loaded_args: list[torch.Tensor] = []
            record = self._record
            for index, tensor in enumerate(args):
                slot_name = f"input.{index}"
                if slot_name not in record.tensor_slots:
                    # This input was already real during forward replay, either
                    # because it came from a RECOMPUTE producer or because it was
                    # not a torch_remat placeholder source.
                    loaded_args.append(tensor)
                    continue

                loaded_tensor = record.load_saved_tensor(
                    active_state.region_state, slot_name
                )
                if loaded_tensor is None:
                    raise RuntimeError(
                        f"input {self._op_name}.{index} was saved as None"
                    )
                loaded_args.append(loaded_tensor)

            return _return_like_args(tuple(loaded_args))

        record = self._record
        for index, tensor in enumerate(args):
            if not _is_stub_on_recompute(tensor):
                continue

            # Only outputs whose producer policy is SAVE carry this marker.
            # Retaining these inputs makes this RECOMPUTE op independent of
            # the skipped producer's placeholder output during replay.
            record.store_saved_tensor_slot(
                active_state.region_state,
                f"input.{index}",
                tensor,
            )

        return _return_like_args(args)

    def save_for_backward(
        self,
        saved_tensors: Mapping[str, torch.Tensor | None],
    ) -> None:
        """Named replacement for ``ctx.save_for_backward``.

        The mapping gives names to every tensor passed to
        ``ctx.save_for_backward``; insertion order determines the order seen by
        ``ctx.saved_tensors`` in backward. For ``SAVE`` ops, the tensors are
        also recorded on the torch_remat tape so :meth:`maybe_load_saved` can
        load them into the recompute autograd graph. For ``RECOMPUTE`` ops,
        they are only saved to the current autograd graph.

        ``None`` is a valid saved slot value and is preserved in
        ``ctx.saved_tensors``.

        Example:
            ```python
            handle.save_for_backward(
                {"x": x, "y": y},  # order matches ctx.saved_tensors
            )
            ```
        """

        active_state = self._active_state
        tensors_to_save: dict[str, SavedTensor] = dict(saved_tensors)
        for tensor_name, tensor in tensors_to_save.items():
            _validate_name(tensor_name, what="save_for_backward key")
            if "." in tensor_name:
                raise RuntimeError(
                    f"save_for_backward key {tensor_name!r} must not contain '.'"
                )
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"save_for_backward.{tensor_name} must be a tensor or None"
                )

        if (
            active_state.phase is _Phase.RECOMPUTE
            and self._policy is not CheckpointPolicy.RECOMPUTE
        ):
            # For SAVE-policy recompute, maybe_load_saved is the only path that
            # both restores ctx.saved_tensors and returns the placeholder output
            # schema. Reaching save_for_backward means user code failed to take
            # that short-circuit path.
            raise RuntimeError(
                "SAVE-policy torch_remat forwards must call maybe_load_saved() "
                "and return its placeholder output during recompute before "
                "calling save_for_backward"
            )

        if self._policy is not CheckpointPolicy.RECOMPUTE:
            record = self._record
            for tensor_name, tensor in tensors_to_save.items():
                record.store_saved_tensor_slot(
                    active_state.region_state,
                    tensor_name,
                    tensor,
                )
                record.saved_for_backward_names.append(tensor_name)

        if tensors_to_save:
            # Preserve ordinary autograd behavior for the current graph. In
            # original forward this supports uncheckpointed use; in recompute it
            # transfers ownership to PyTorch's backward graph.
            self._ctx.save_for_backward(*tensors_to_save.values())

    def record_outputs(self, *outs: Output) -> Output:
        """Record output metadata for this autograd Function call.

        For ``SAVE`` ops in the original forward, this records enough output
        schema and tensor metadata to synthesize data-inaccessible placeholders
        during recompute. The returned value preserves the single-tensor versus
        tuple schema expected by the autograd engine. Output tensor storage is
        not retained; include any output needed by backward in
        :meth:`save_for_backward`.

        A single output is conventionally named ``out`` in memory reports; tuple
        outputs are named by position.

        Example:
            ```python
            z = my_op_fwd2(y)
            return handle.record_outputs(z)
            ```
        """

        active_state = self._active_state
        if len(outs) == 1:
            output = outs[0]
        elif all(isinstance(out, torch.Tensor) for out in outs):
            output = cast(tuple[torch.Tensor, ...], outs)
        else:
            raise RuntimeError(
                "record_outputs accepts tensors or a single output tuple"
            )

        record = self._record
        assert (
            active_state.phase is not _Phase.RECOMPUTE
            or self._policy is CheckpointPolicy.RECOMPUTE
        ), (
            f"record_outputs called on {self._op_name} with policy "
            f"{self._policy} during recompute; SAVE ops must short-circuit "
            f"via maybe_load_saved()"
        )

        if isinstance(output, torch.Tensor):
            output_tensors: Output = output
        elif isinstance(output, tuple) and all(
            isinstance(tensor, torch.Tensor) for tensor in output
        ):
            output_tensors = cast(tuple[torch.Tensor, ...], output)
        else:
            raise RuntimeError(
                "output must be a Tensor or tuple of Tensors when checkpoint forward runs"
            )

        if self._policy is not CheckpointPolicy.RECOMPUTE:
            record.record_output_schema(output_tensors)
            for tensor in _output_tensors(output_tensors):
                # Downstream RECOMPUTE handles use this marker to retain only
                # inputs that would otherwise be placeholder outputs in replay.
                setattr(tensor, _STUB_ON_RECOMPUTE_ATTR, True)

        if active_state.phase is _Phase.RECOMPUTE:
            _release_record_after_recompute_if_needed(record)

        return output_tensors


def get_handle(
    ctx: AutogradCtx,
    op_name: str,
    policy: CheckpointPolicy,
) -> RematHandle | _InertRematHandle:
    """Return a ``RematHandle`` for one remat-aware autograd Function call.

    Call this at the start of a custom ``autograd.Function.forward``. In the
    original forward of an active checkpoint region, this creates or finds the
    call's private tape record and establishes its checkpoint policy. During
    recompute, it looks up the existing tape record. Outside a checkpoint
    region, it returns an inert handle whose methods behave like normal
    autograd-forward helpers.

    The ``op_name`` is relative to the current checkpoint region and must be
    stable and unique for this forward invocation. Conflicting policies for
    the same name in one region are rejected.

    Example:
        ```python
        handle = remat.get_handle(ctx, op_name, remat_policy)
        ```
    """

    _validate_name(op_name, what="op_name")
    if not any(ctx.needs_input_grad):
        return _InertRematHandle(ctx)

    active_state = _state.get()
    if active_state is None:
        return _InertRematHandle(ctx)

    # A remat-aware custom Function should create exactly one handle for its
    # stable op_name in each phase. Duplicates usually mean two logical ops are
    # sharing a name or one forward body is drifting between phases.
    _claim_handle_name(active_state, op_name)
    if active_state.phase is _Phase.FORWARD:
        record = _get_or_create_policy_record(
            active_state.region_state,
            op_name,
            policy,
        )
    else:
        record = _expect_policy_record(
            active_state.region_state,
            op_name,
            policy,
            "get_handle",
        )

    return RematHandle(ctx, op_name, policy, active_state, record)


def auto_forward(
    *save_for_backward_names: str,
) -> Callable[[Callable[..., Output]], Callable[..., Output]]:
    """Decorate a high-level ``autograd.Function.forward`` implementation.

    The decorated forward takes the same arguments as the underlying
    ``autograd.Function.forward``. Calls to ``ctx.save_for_backward`` are
    captured and named with ``save_for_backward_names``. Only direct positional tensor
    arguments are passed through :meth:`RematHandle.save_or_load_inputs`; nested
    tensor containers should use the explicit handle API so their replay inputs
    can be named intentionally.

    Example:
        ```python
        class MyOp(torch.autograd.Function):
            @staticmethod
            @remat.auto_forward("x", "y")
            def forward(ctx, x):
                y = my_op_fwd1(x)
                z = my_op_fwd2(y)
                ctx.save_for_backward(x, y)
                return z
        ```
    """

    seen_names: set[str] = set()
    for tensor_name in save_for_backward_names:
        _validate_name(tensor_name, what="auto_forward save_for_backward name")
        if "." in tensor_name:
            raise RuntimeError(
                f"auto_forward save_for_backward name {tensor_name!r} must not "
                "contain '.'"
            )
        if tensor_name in seen_names:
            raise RuntimeError(
                f"Duplicate auto_forward save_for_backward name: {tensor_name}"
            )
        seen_names.add(tensor_name)

    def decorator(forward: Callable[..., Output]) -> Callable[..., Output]:
        @wraps(forward)
        def wrapper(ctx: AutogradCtx, *args: Any) -> Output:
            context = _active_op.get()
            if context is None:
                if _state.get() is None:
                    return forward(ctx, *args)
                raise RuntimeError("auto_forward expects an active remat.op call")
            op_name, remat_policy = context
            body_args = args

            handle = get_handle(ctx, op_name, remat_policy)
            if (ret := handle.maybe_load_saved()) is not None:
                return ret

            tensor_args = tuple(
                arg for arg in body_args if isinstance(arg, torch.Tensor)
            )
            loaded_args = handle.save_or_load_inputs(*tensor_args)
            # save_or_load_inputs only operates on tensors, but the wrapped
            # forward must still receive its non-tensor positional arguments in
            # their original positions.
            loaded_tensor_iter = iter(
                (loaded_args,) if isinstance(loaded_args, torch.Tensor) else loaded_args
            )
            body_args = tuple(
                next(loaded_tensor_iter) if isinstance(arg, torch.Tensor) else arg
                for arg in body_args
            )
            proxy_ctx = _SaveForBackwardProxy(ctx)
            output = forward(proxy_ctx, *body_args)
            named_saved_tensors = proxy_ctx.named_saved_tensors(save_for_backward_names)
            handle.save_for_backward(named_saved_tensors)
            return handle.record_outputs(output)

        return wrapper

    return decorator


class _CheckpointPhaseContext(contextlib.AbstractContextManager[None]):
    """Reusable context manager for one checkpoint phase.

    PyTorch non-reentrant checkpoint stores these context managers and enters
    them later around the original forward and replay.
    """

    def __init__(
        self,
        region_state: _CheckpointRegionState,
        phase: _Phase,
    ) -> None:
        self._region_state = region_state
        self._phase = phase
        self._token: contextvars.Token[_ActiveCheckpointRegion | None] | None = None

    def __enter__(self) -> None:
        self._token = _state.set(
            _ActiveCheckpointRegion(region_state=self._region_state, phase=self._phase)
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if (
                (exc_type is None or _is_checkpoint_early_stop_exception(exc_type))
                and self._phase is _Phase.RECOMPUTE
                and not _keep_graph()
            ):
                _check_remat_tape_released_after_recompute(self._region_state)
        finally:
            if self._token is not None:
                _state.reset(self._token)
                self._token = None


class _SaveForBackwardProxy:
    """Capture decorator-body save_for_backward calls.

    The proxy forwards arbitrary ctx attributes to the real autograd context so
    decorator users can keep normal ctx.foo assignments, while intercepting only
    save_for_backward to attach names before the real ctx sees the tensors.
    """

    _ctx: AutogradCtx
    _saved_tensors: tuple[SavedTensor, ...] | None

    def __init__(self, ctx: AutogradCtx) -> None:
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_saved_tensors", None)

    def save_for_backward(self, *tensors: SavedTensor) -> None:
        self._saved_tensors = tensors

    def named_saved_tensors(
        self,
        names: tuple[str, ...],
    ) -> dict[str, SavedTensor]:
        tensors = () if self._saved_tensors is None else self._saved_tensors
        if len(names) != len(tensors):
            raise RuntimeError(
                "auto_forward save_for_backward names must match "
                "ctx.save_for_backward tensor count"
            )

        return dict(zip(names, tensors))

    def __getattr__(self, name: str) -> Any:
        return object.__getattribute__(
            object.__getattribute__(self, "_ctx"),
            name,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        setattr(self._ctx, name, value)


def _return_like_args(
    args: tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Return one tensor as itself, or multiple tensors as a tuple."""

    if len(args) == 1:
        return args[0]
    return args


def _expect_record(
    region_state: _CheckpointRegionState,
    op_name: str,
) -> _OpRecord:
    """Return a forward record by name or raise."""

    record = region_state.records.get(op_name)
    if record is None:
        raise RuntimeError(
            f"No saved torch_remat op record for {_display_name(region_state, op_name)}"
        )

    return record


def _expect_policy_record(
    region_state: _CheckpointRegionState,
    op_name: str,
    policy: CheckpointPolicy,
    caller: str,
) -> _OpRecord:
    """Return a high-level op record with an established policy."""

    record = region_state.records.get(op_name)
    if record is None or record.policy is None:
        raise RuntimeError(
            f"{caller} must follow maybe_load_saved or save_for_backward for {op_name}"
        )

    if record.policy is not policy:
        raise RuntimeError(
            f"Conflicting checkpoint policies for {_display_name(region_state, op_name)} "
            f"during recompute: forward used {record.policy.name}, "
            f"recompute used {policy.name}"
        )

    return record


def _get_or_create_policy_record(
    region_state: _CheckpointRegionState,
    op_name: str,
    policy: CheckpointPolicy,
) -> _OpRecord:
    """Return an op record and establish its checkpoint policy."""

    record = region_state.records.get(op_name)
    if record is None:
        record = _OpRecord(op_name=op_name)
        region_state.records[op_name] = record
    if record.policy is not None and record.policy is not policy:
        raise RuntimeError(
            f"Conflicting checkpoint policies for {_display_name(region_state, op_name)}"
        )

    record.policy = policy
    return record


def _claim_handle_name(
    active_state: _ActiveCheckpointRegion,
    op_name: str,
) -> None:
    """Record one handle retrieval in the current checkpoint phase."""

    if op_name in active_state.handle_names:
        raise RuntimeError(
            "Duplicate torch_remat handle retrieval for "
            f"{_display_name(active_state.region_state, op_name)} "
            f"during {active_state.phase.name.lower()}"
        )

    active_state.handle_names.add(op_name)


def _make_placeholder_tensor(
    metadata: _TensorMetadata,
    message: str,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Return a shaped placeholder whose tensor-data access raises."""

    placeholder = _PlaceholderTensor(metadata, message, requires_grad=requires_grad)
    _set_placeholder_storage_data_ptr_error(placeholder)
    return placeholder


def _set_placeholder_storage_data_ptr_error(tensor: _PlaceholderTensor) -> None:
    """Make data_ptr access on placeholder storage raise the remat diagnostic."""

    torch._C._set_storage_data_ptr_access_error_msg(
        tensor.untyped_storage()._cdata,
        _placeholder_message(tensor),
    )


def _placeholder_message(tensor: _PlaceholderTensor) -> str:
    """Return the diagnostic message carried by one placeholder."""

    message = object.__getattribute__(tensor, _REPORT_PLACEHOLDER_MESSAGE_ATTR)
    assert isinstance(message, str)
    return message


def _placeholder_message_from_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Return a placeholder message from dispatch inputs."""

    for value in pytree.tree_leaves((args, kwargs)):
        if isinstance(value, _PlaceholderTensor):
            return _placeholder_message(value)

    return "Attempted to use a torch_remat placeholder tensor"


def _release_record_after_recompute_if_needed(record: _OpRecord) -> None:
    """Release remat-owned state once recompute has transferred ownership."""

    if _keep_graph():
        return

    # At this point SAVE-policy tensors have been copied into recompute ctx and
    # RECOMPUTE-policy tensors were produced in the replay graph. Clearing the
    # remat tape here matches PyTorch's default non-retained backward lifetime.
    record.tensor_slots.clear()
    record.saved_for_backward_names.clear()
    record.output_metadata = ()
    record.native_sac_tensors.clear()
    record.native_op_counts.clear()
    record.native_sac_contexts = None
    record.released = True


def _is_checkpoint_early_stop_exception(
    exc_type: type[BaseException] | None,
) -> bool:
    """Return whether an exception is PyTorch checkpoint's early-stop signal."""

    if exc_type is None:
        return False

    from torch.utils.checkpoint import _StopRecomputationError

    return issubclass(exc_type, _StopRecomputationError)


def _check_remat_tape_released_after_recompute(
    region_state: _CheckpointRegionState,
) -> None:
    """Raise if recompute exited while the remat tape still owns entries."""

    unreleased_records = [
        record
        for record in region_state.records.values()
        if _record_retains_tape(record)
    ]
    if not unreleased_records:
        return

    # Import locally to avoid a module import cycle; reporting imports private
    # structures from this module.
    from torch_remat._reporting import _format_memory_report

    region = (
        region_state.region_name
        if region_state.region_name is not None
        else "<unnamed>"
    )
    lines = [
        "torch_remat checkpoint region "
        f"{region} finished recompute with unreleased remat tape entries.",
        "This usually means forward recorded saved tensors for an op that was "
        "not executed during recompute, often because PyTorch checkpoint "
        "early-stop skipped work that backward did not need.",
        "Unreleased records:",
    ]
    lines.extend(
        f"  {_display_name(region_state, record.op_name)}: "
        f"{_format_unreleased_record_summary(record)}"
        for record in unreleased_records
    )
    lines.append("Retained memory report:")
    lines.append(_format_memory_report(region_state))
    raise RuntimeError("\n".join(lines))


def _record_retains_tape(record: _OpRecord) -> bool:
    """Return whether a record still owns tape state after recompute."""

    return bool(
        record.tensor_slots
        or record.saved_for_backward_names
        or record.output_metadata
        or record.native_sac_tensors
        or record.native_op_counts
        or record.native_sac_contexts is not None
    )


def _format_unreleased_record_summary(record: _OpRecord) -> str:
    """Format one unreleased record for the recompute-exit diagnostic."""

    pieces: list[str] = []
    if record.policy is not None:
        pieces.append(f"policy={record.policy.name}")
    if record.tensor_slots:
        pieces.append(
            "saved_tensors="
            + ",".join(
                f"{name}({_format_saved_tensor_slot(slot)})"
                for name, slot in record.tensor_slots.items()
            )
        )
    if record.output_metadata:
        pieces.append(f"output_placeholders={len(record.output_metadata)}")
    if record.native_sac_tensors:
        pieces.append(f"native_saved_outputs={len(record.native_sac_tensors)}")
    if record.native_sac_contexts is not None:
        pieces.append("native_sac_contexts=live")
    return " ".join(pieces)


def _format_saved_tensor_slot(slot: _SavedTensorSlot) -> str:
    """Format one saved tensor slot for diagnostics."""

    tensor = slot.tensor
    if tensor is None:
        return "None"
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"


def _validate_name(name: str, *, what: str) -> None:
    """Validate a relative op or tensor name passed by user code."""

    if not name:
        raise ValueError(f"torch_remat {what} must be non-empty")


def _display_name(region_state: _CheckpointRegionState, op_name: str) -> str:
    """Render a diagnostic op name, including the checkpoint region."""

    if region_state.region_name is None:
        return op_name

    return f"{region_state.region_name}::{op_name}"


def _output_tensors(output: Output) -> tuple[torch.Tensor, ...]:
    """Return output tensors in return-schema order."""

    if isinstance(output, torch.Tensor):
        return (output,)
    return output


def _output_name(index: int, *, output_is_tuple: bool) -> str:
    """Return the canonical report name for one output position."""

    if not output_is_tuple:
        return "out"
    return str(index)


def _expect_state() -> _ActiveCheckpointRegion:
    """Return the active checkpoint state or raise a useful error."""

    state = _state.get()
    if state is None:
        raise RuntimeError("No active torch_remat checkpoint region")

    return state
