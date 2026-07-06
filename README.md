# torch_remat

`torch_remat` is a small library for activation checkpointing in a style where
all tensors are recomputed by default, and you explicitly mark the specific
ops whose activations you want to **save** for backward instead. This style of
API gives fine-grained, explicit control over what is kept in memory, as what
is saved for backwards is precisely (1) everything *internally* saved for
backwards in `SAVE` regions (a tensor a `SAVE` op saves that is merely one of
its own inputs is recomputed or ferried instead, not kept), and (2) the tensors
that pass from a `SAVE` to a `RECOMPUTE` region (since they ordinarily aren't
available during recompute, since you skipped running the code that produces
them).

Broadly speaking, here is how you use `torch_remat`:

- Use `remat.checkpoint(...)` to wrap the full region you want to recompute
  (usually a transformer block).  This will cause everything in the region to
  `RECOMPUTE`.

- Wrap operations that you want to instead save values for backward with
  `remat.op(fn, name)`.  (You can also explicitly pass in `policy` argument
  with `remat.SAVE` or `remat.RECOMPUTE` to conveniently have configs to
  toggle between save or not.)

By default, `torch_remat` detects that an output of a `SAVE` region has passed
to a `RECOMPUTE` region by wrapping all outputs from `SAVE` regions into a
tensor subclass that acts like a normal tensor, except it helps us tell if
the tensor is used in a subsequent `RECOMPUTE` region.  If your code is
not compatible with tensor subclasses, you can turn this off via
`detect_bare_ops=False` (you must then explicitly annotate all downstream
consumers with an explicit `remat.op`), or try another `detect_bare_ops`
strategy (described in the [Detect bare ops](#detect-bare-ops) section.)

How does this compare to existing PyTorch checkpointing APIs?

* Compared to non-reentrant activation checkpointing (AC): this is essentially
  the same API, but with an extra `remat.op` API!  (Unfortunately, we did have
  to provide our own `remat.checkpoint` entry because we make some slightly
  different choices compared to AC; for example, we immediately run the
  recompute at the beginning of backwards for the entire region.)  Whereas the
  classic AC recomputes everything, this API gives you the ability to
  selectively save tensors in regions of code so you can skip recomputing
  them, and we automatically take care of saving input tensors that cross from
  recompute to save regions so that recompute can continue (these tensors are
  maintained on a dedicated, remat-specific tape).

* Compared to selective activation checkpointing (SAC): SAC decides what to save
  with a policy function that classifies each *aten op* as cheap-or-expensive to
  recompute, via a `TorchDispatchMode`. `torch_remat` instead lets you mark
  specific *regions* of code, so you can easily ask to save one matmul but
  recompute another.  Additionally, this API does not require the use of
  a `TorchDispatchMode` and works with custom kernels that weren't registered
  to the dispatcher.

## API

At the top-level unit of recompute (e.g. a transformer block), write:

```python
import torch_remat as remat

y = remat.checkpoint(region_name="layers.0")(block)(x)
```

The first call binds checkpoint options, the second binds the function, and
The third passes user arguments to `block`.  (Although seemingly natural,
`remat.checkpoint(block)` is intentionally not supported for better
compatibility with `torch.utils.checkpoint`, which would interpret this as
invoking `block` with no arguments!)

Like `torch.utils.checkpoint`, everything in the region recomputes (by
default) on backward: thus, it must be safe to run the forward again (no
double side effects), RNG must be synchronized, you shouldn't compute metrics
in the recompute, and recompute must run the same series of operations as the
original forward.

See "pytree semantics" for requirements on inputs/outputs to checkpoint
region.

### Saving specific activations with `remat.op`

Inside the region, wrap any call you want to control with `remat.op`:

```python
y = remat.op(my_op, "my_op", policy=remat.CheckpointPolicy.SAVE)(x)
```

- `fn` (here `my_op`) is any operator you want to control checkpoint policy on.
  We suggest wrapping a single custom autograd function per `remat.op`, as
  this is the finest granularity save/recompute decisions can be made in.
  See "pytree semantics" for requirements on inputs/outputs to `fn`.

- `name` must be unique within the checkpoint region. Unique names give desync
  protection and make it easier to read memory reports.

- `policy` is `SAVE` or `RECOMPUTE`:
  - **`SAVE`**: the tensors `fn`'s autograd nodes save for backward are kept by
    PyTorch autograd (on the original forward graph), and `fn` is **not** rerun
    during recompute.
  - **`RECOMPUTE`**: `fn` is rerun during recompute (default).

Optionally, inside a custom `autograd.Function`'s `forward` you may call
`remat.save_for_backward(ctx, {"name": tensor, ...})` in place of
`ctx.save_for_backward(...)` to give the saved activations names. The names label
the saved tensors, so they appear in `remat.format_current_memory_report()` instead
of positional `saved.0` / `saved.1` keys.  This works best if the `remat.op`
was scoped immediately around the custom autograd function.

`remat.op` is not allowed to be nested; holler if you want this to work.

### pytree semantics

To work, `torch_remat` needs to be able to identify Tensor inputs/outputs into
`remat.checkpoint` and `remat.op`.  We match `autograd.Function` / ATen op allowed
inputs/outputs: Tensor and a (one-hop) tuple/list of Tensor are supported for
input/output.  Keyword arguments are supported.  We chose not to support full
pytree (nor `dict`) for efficiency and predictability reasons.

Nesting a recognized container inside another (e.g. a list of lists of Tensor)
is *not* supported and raises `TypeError` early — we traverse exactly one hop of
`tuple`/`list`, no deeper.

Similar to `autograd.Function`, it is permissible to pass Tensor via structures
that are *opaque* to these semantics — a `dict`, or a custom object we don't
recognize as a container (not a `tuple`/`list`), which we treat as a single leaf
and hand to your `fn` untouched. As long as such a Tensor is not differentiable and always
recomputed, this will work fine. If it is instead a `SAVE` output later used in a
`RECOMPUTE` region, we can't ferry a value we never saw, so you get a placeholder
error — but only surfaced at a later time (during recompute), not at the call.

If you think we should support full pytree, give us a holler.

## How SAVE and RECOMPUTE work

The way classic PyTorch non-reentrant checkpointing works is that in the
initial forwards, all tensors saved for backwards are discarded (via saved
tensor hooks); when we run the recompute, we recompute all of these saved
for backwards tensors so that the eventual backwards can access all of them.
By default, everything is a `RECOMPUTE` region and just executes in this way.
(NB: the autograd graph that eventually gets run for backwards is the
*original* forward autograd graph, not the recompute one.)

`torch_remat` works in the same way, except we now want to undo the recompute
policy and go back to *saving* some tensors for backwards in the `SAVE`
regions.  Additionally, we also might need to save some outputs of a `SAVE`
region, in case we transition back into a `RECOMPUTE` region (since we need
the inputs to the recompute region to actually recompute it.)  So we just
introduce two new mechanisms to make this work:

1. Inside a `SAVE` region, we install a nested identity saved-tensor hook. This
   suspends checkpoint's own saved-tensor hooks, so the region just saves for
   backwards normally and PyTorch autograd owns those tensors on the original
   forward graph (present at backward with zero recompute, freed by autograd
   after backward — no tape bookkeeping). You can override this hook with
   `remat.saved_tensors_hooks(...)` — e.g. to offload `SAVE` activations — since
   the identity hook would otherwise shadow PyTorch's own `saved_tensors_hooks`.

2. We register each output of a `SAVE` region in a per-region **save-output index**
   that marks it as needing to be saved if it flows into a `RECOMPUTE` region. A
   `RECOMPUTE` `remat.op` looks the value up in the index and ferries it onto a
   special remat-specific tape (by the way, this is why you need to apply `remat.op`
   to both `SAVE` and `RECOMPUTE` regions, not just `RECOMPUTE`.) A *bare* op that
   touches a SAVE output instead makes the **producer** save the value on the tape
   (producer responsibility), via one of the `torch_remat._bare_op` strategies (by
   default the `_SaveTensor` subclass, unless disabled with `detect_bare_ops=False`).
   See "SAVE outputs: forward vs recompute".

For any intermediate tensor, this is how it is made available during recompute:

* Input to the overall region: provided by checkpoint (always available)
* Output of a `RECOMPUTE` op: recomputed
* Saved-for-backward of a `RECOMPUTE` op: recomputed
* Saved-for-backward of a `SAVE` op, when it is an *internally produced* tensor:
  kept by autograd on the original forward graph
* Saved-for-backward of a `SAVE` op, when it is one of the op's own *inputs*: not
  kept — recomputed if it came from a `RECOMPUTE` region (captured during replay),
  or ferried on the remat tape if it came from another `SAVE` region. This avoids
  retaining a `RECOMPUTE` region's output merely because a downstream `SAVE` op
  saved it for backward.
* Output of a `SAVE` op: identified via the region's save-output index (by tensor
  identity, not type). A `RECOMPUTE` op consumer ferries the real value through the
  remat tape. A *bare* (unwrapped) op consumer is intercepted by default (the
  **producer** then persists the value the first time it is touched); under
  `checkpoint(..., detect_bare_ops=False)` it is not intercepted and instead meets a
  placeholder during recompute and raises. In recompute the output is the real
  persisted value, or a storage-free placeholder when none was saved (see "SAVE
  outputs: forward vs recompute").

## SAVE outputs: forward vs recompute

Every `SAVE` output is registered in a per-region **save-output index** keyed by tensor
identity (not type), recording how to persist and unwrap it. The ferry (a
`remat.op` consumer), the SAVE-input snapshot, and the region boundary all consult the
index, so none of them depends on the output's representation — which is chosen per
region:

- **Opt out** (`detect_bare_ops=False`): a `SAVE` op returns its outputs as **plain
  tensors**. A `remat.op` consumer is ferried via the index; a *bare* (unwrapped)
  consumer is not intercepted, so during recompute it meets a storage-free placeholder
  and raises an actionable error telling you to wrap it in `remat.op` (or re-enable
  bare-op detection). This is the tight prod path — no tensor subclass, and it works
  uniformly for any tensor type (`DTensor`, etc.).

- **Default** (`detect_bare_ops=True`, i.e. `"subclass"`): outputs are wrapped in
  `_SaveTensor` (a *wrapper* subclass in `torch_remat._bare_op._subclass`, holding the
  real output as `_inner` and grad-connected to the producer). A bare op consuming it
  trips `__torch_dispatch__`, which fires the **producer's** persist-output (recording the
  value so recompute can reproduce it) and runs the op on the unwrapped inner — one hop,
  every output plain. `data_ptr()` is overridden to persist then return the inner's
  real pointer, so raw Triton/cutedsl kernels on a `SAVE` output also work. It is the
  default because it costs only O(SAVE outputs) rather than intercepting every op (as the
  modes do), sees `data_ptr()`, and is a real tensor so all torch/Python protocols work;
  opt out with `detect_bare_ops=False` if your tensors aren't subclass-compatible.

- **`checkpoint(..., detect_bare_ops="proxy")`**: outputs are wrapped in
  `_SaveProxy` (a `__torch_function__` object in `torch_remat._bare_op._proxy`,
  `fx.Proxy`-style), an alternative to the subclass. Because it is **not** a tensor it
  never enters the autograd graph — the moment an op touches it, it unwraps to the
  grad-connected `_inner` and the op runs on that, so gradient flows
  producer → `_inner` → consumer with no `_WrapSave` bridge needed. A *view* op (its
  result aliases the producer output's storage — `reshape`, slice, `transpose`) returns a
  **new proxy** and *defers* the save, so a bare view later ferried by a `remat.op` never
  forces the producer to keep a slot; any other op ("poked hard" — a real compute, an
  operator, `data_ptr()`, `item()`) fires the persist-output once and returns the plain
  result. The cost of not being a tensor is that operator dunders (`+`, `@`, `[]` …) and
  method access are installed manually and routed through one dispatcher.

- **`checkpoint(..., detect_bare_ops="dispatch_mode" | "function_mode")`**: the *mode*
  analogues of the subclass and proxy. SAVE outputs stay **plain tensors** (indexed exactly
  like the opt-out path); instead of a per-output wrapper, a `TorchDispatchMode` /
  `TorchFunctionMode` is installed for the duration of the original forward and fires the
  producer's persist-output when an op touches a SAVE output. Because there is no subclass on
  the graph, there is nothing to unwrap — the op just runs on its already-plain arguments
  (redispatch is trivial), which is the main appeal. The trade-off is that a mode intercepts
  **every** op in the region, so remat's own per-op processing (ferry, snapshot, boundary)
  runs under a suppression flag so it is not mistaken for a bare consumer. `dispatch_mode`
  mirrors the subclass (fires on every touch, views included) but **cannot see
  `data_ptr()`** — a raw-pointer kernel bypasses `__torch_dispatch__`; `function_mode`
  mirrors the proxy (defers on views, registering them back into the save-output index) and
  **does** see `data_ptr()` through `__torch_function__`.

  For the common case — a *bare* op consuming a SAVE output passed to it as an argument — all
  four intercepting strategies produce identical observable behavior (gradients, tape slots);
  they differ only in overhead and in the `data_ptr` reach noted above. They are **not**
  identical for a SAVE output consumed *inside* a `remat.op` body via **closure capture**
  (read from the enclosing scope rather than passed as an argument). remat runs the entire
  `remat.op` body — user code included — under `_suppress_bare_op_detection` on the theory that
  everything inside an `op` is explicitly handled; but only the op's *arguments* are handled by
  the consume/snapshot path, not values it reaches through a closure. The wrapper strategies
  (`subclass`, `proxy`) still catch such a value because a wrapped output trips interception on
  *any* touch, regardless of the suppression flag (which they never read), so the producer
  persists it and recompute succeeds. The mode strategies (`dispatch_mode`,
  `function_mode`) honor the suppression flag, so the closure-captured touch is invisible, the
  producer never saves, and the value meets a placeholder during recompute and **raises**. Pass
  such a value as an argument to the `remat.op` (so the consume path handles it) rather than
  capturing it, or use a wrapper strategy.

In **recompute** a skipped `SAVE` op returns each output from its persisted value,
or a storage-free `_PlaceholderTensor` when none was saved — it was dead, or consumed
only by a ferrying `remat.op` (which substitutes its value by argument position before
any op runs). A placeholder supports metadata/view ops (a view of a placeholder is
another placeholder) but raises if its data is actually read.

This is **producer responsibility**: the consumer does no bookkeeping; it just uses the
tensor, and the producer decides what to keep. A `remat.op` consumer unwraps via the
index up front (grad-connected — the wrapper's `_inner`, or the plain tensor itself) and
ferries the value on its own record, so it does *not* trigger the producer's
persist-output thunk.

Consequences:

- Only `SAVE` outputs actually touched by a bare op are kept resident (they show up as
  `output.<i>` rows in `remat.format_current_memory_report()`, attributed to the
  producing op). An output consumed only by `remat.op`s is ferried and does **not**
  create an `output.<i>` row; an output consumed by nothing costs nothing.

One thing remains an **error** (even with `detect_bare_ops`):

- **In-place / mutating ops** on a `SAVE` output — mutating it would corrupt both the
  persisted value and the copy autograd kept for the op's backward. Wrap the
  mutation in a `remat.op` (or apply it before the value leaves the producing op).

## Developer notes

* Version counters: Claude says that when you use a saved-tensor hook,
  autograd's own version-counter guard against in-place mutation does **not**
  fire for them. `torch_remat` therefore records each tensor's version at save
  time and re-checks it at backward, raising if the tensor was mutated in
  between.

## torch.compile support

Not supported yet. This is the eager implementation; `torch.compile` support
(translating remat policies into min-cut-partitioner annotations) is a separate
change.

## Detect bare ops

`detect_bare_ops` selects the bare-op detection strategy. For a bare op consuming a SAVE
output passed to it as an argument, the four intercepting strategies produce identical
observable behavior (gradients, tape slots) and differ only in overhead and `data_ptr`
reach; `False` disables interception:

| value | mechanism | `data_ptr()` kernels | notes |
|---|---|---|---|
| `False` | none — plain tensors | n/a | tightest prod path; bare consumers raise |
| `True` / `"subclass"` (default) | `_SaveTensor` wrapper subclass | ✅ seen | fires on every touch, views included |
| `"proxy"` | `_SaveProxy` `__torch_function__` object | ✅ seen | defers on views (a bare view forces no save) |
| `"dispatch_mode"` | `TorchDispatchMode` (plain tensors) | ❌ bypassed | mode-based analogue of the subclass |
| `"function_mode"` | `TorchFunctionMode` (plain tensors) | ✅ seen | mode-based analogue of the proxy; defers on views |

The wrapper strategies (`subclass`, `proxy`) and the mode strategies (`dispatch_mode`,
`function_mode`) diverge on one corner: a SAVE output consumed *inside* a `remat.op` body via
**closure capture** (rather than passed as an argument) is caught by the wrappers but missed by
the modes — the modes are suppressed for the whole `remat.op` body, and only the op's arguments
are otherwise handled, so under a mode the value hits a placeholder during recompute and raises.
See "SAVE outputs: forward vs recompute" for the full mechanics of each strategy.


## Diagnostics

TODO: we should describe this more

- `remat.is_recomputing()` returns whether execution is in the recompute pass.
- `remat.collect_trace()` / `remat.trace_scope(...)` collect a reporting-only
  tree of the `op` annotations seen during the original forward.
- `remat.format_current_memory_report()` /
  `remat.print_current_memory_report()` summarize the activations retained for the
  active region (autograd-owned `SAVE` saves and tape-owned ferried inputs),
  grouped by op and tensor, with storage-sharing (aliasing) accounting.

## Offloading

* Saved tensor hooks extension point
* Wedge as a worked example
* Subtlety: we save outputs for recompute; timing is different
* Subtlety: when to trigger onload (do it on is recompute, not as an autograd
  function)

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
