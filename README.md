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
  recompute.

- Wrap operations whose activations you want to save for backward instead with
  `remat.region(fn, name, recompute=False)`.  The `recompute` keyword is
  required: pass `recompute=False` to save (keep activations, skip recompute) or
  `recompute=True` to recompute.  Driving `recompute` from a config flag is the
  natural way to toggle an op between saving and recomputing.

`torch_remat` detects that an output of a `SAVE` region has passed to a
`RECOMPUTE` region by keeping a per-region index of `SAVE` outputs keyed by
storage: when a `remat.region` consumes such an output (or a bare view of it),
the producer is made to durably save it. A *bare* op (something not wrapped in
`remat.region` — a residual add, a `.reshape`, a raw kernel) cannot be detected;
if a `SAVE` region's output flows into one, call `remat.recompute_needs_tensor(t)`
on the output right before the bare op so the producer persists it (see the
[recompute_needs_tensor](#recompute_needs_tensor) section). Otherwise a bare
consumer meets a placeholder during recompute and raises with a message pointing
back at the producing region.

How does this compare to existing PyTorch checkpointing APIs?

* Compared to non-reentrant activation checkpointing (AC): this is essentially
  the same API, but with an extra `remat.region` API!  (Unfortunately, we did have
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

### Saving specific activations with `remat.region`

Inside the region, wrap any call you want to control with `remat.region`:

```python
y = remat.region(my_op, "my_op", recompute=False)(x)
```

- `fn` (here `my_op`) is any operator you want to control recompute for.
  We suggest wrapping a single custom autograd function per `remat.region`, as
  this is the finest granularity save/recompute decisions can be made in.
  See "pytree semantics" for requirements on inputs/outputs to `fn`.

- `name` must be unique within the checkpoint region. Unique names give desync
  protection and make it easier to read memory reports.

- `recompute` (keyword-only, required) is a bool:
  - **`recompute=False`** (save): the tensors `fn`'s autograd nodes save for
    backward are kept by PyTorch autograd (on the original forward graph), and
    `fn` is **not** rerun during recompute.
  - **`recompute=True`**: `fn` is rerun during recompute, exactly as the enclosing
    `remat.checkpoint` region would have done anyway.

Optionally, inside a custom `autograd.Function`'s `forward` you may call
`remat.save_for_backward(ctx, {"name": tensor, ...})` in place of
`ctx.save_for_backward(...)` to give the saved activations names. The names label
the saved tensors, so they appear in `remat.format_current_memory_report()` instead
of positional `saved.0` / `saved.1` keys.  This works best if the `remat.region`
was scoped immediately around the custom autograd function.

`remat.region` is not allowed to be nested; holler if you want this to work.

### pytree semantics

To work, `torch_remat` needs to be able to identify Tensor inputs/outputs into
`remat.checkpoint` and `remat.region`.  We match `autograd.Function` / ATen op allowed
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
default and go back to *saving* some tensors for backwards in the `SAVE`
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

2. We register each output of a `SAVE` region in a per-region **save-output index**,
   keyed by storage, that marks it as needing to be saved if it flows into a
   `RECOMPUTE` region. A `RECOMPUTE` `remat.region` looks the value up in the index —
   by storage, so a bare view of the output resolves too — and makes the **producer**
   save it on a special remat-specific tape. A *bare* op (not wrapped in
   `remat.region`) cannot be detected; to feed a SAVE output into one, call
   `remat.recompute_needs_tensor(t)` on the output right before the bare op so
   the producer persists it. See "SAVE outputs: forward vs recompute".

For any intermediate tensor, this is how it is made available during recompute:

* Input to the overall region: treated like a `SAVE` op's output (see "Region inputs
  are liveness-tracked"). Kept and served real during recompute if a `RECOMPUTE` region
  consumes it (or a bare consumer flags it with `recompute_needs_tensor`); served a
  storage-free placeholder if only `SAVE` regions (which are skipped) consume it, so a
  dead-into-`SAVE` input is not pinned. With `input_saved_tensors_hooks` set it falls back
  to checkpoint saving every input (always available).
* Output of a `RECOMPUTE` op: recomputed
* Saved-for-backward of a `RECOMPUTE` op: recomputed
* Saved-for-backward of a `SAVE` op, when it is an *internally produced* tensor:
  kept by autograd on the original forward graph
* Saved-for-backward of a `SAVE` op, when it is one of the op's own *inputs*: not
  kept — recomputed if it came from a `RECOMPUTE` region (captured during replay),
  or ferried on the remat tape if it came from another `SAVE` region. This avoids
  retaining a `RECOMPUTE` region's output merely because a downstream `SAVE` op
  saved it for backward.
* Output of a `SAVE` op: identified via the region's save-output index (by storage,
  not type). A `RECOMPUTE` op consumer (or one receiving a bare view of the output)
  makes the **producer** persist the value on the remat tape. A *bare* op consumer
  cannot be detected: call `remat.recompute_needs_tensor(t)` on the output right before
  it so the producer persists the value; otherwise it meets a placeholder during
  recompute and raises. In recompute the output is the real persisted value, or a
  storage-free placeholder when none was saved (see "SAVE outputs: forward vs recompute").

## SAVE outputs: forward vs recompute

A `SAVE` op returns its outputs as **plain tensors** — there is no wrapper subclass or
proxy. Every output is registered in a per-region **save-output index** keyed by
**storage**, whose value is a *persist-output thunk* that, when fired, records the output
on the remat tape so recompute can reproduce it. Because the index is keyed by storage
(not tensor identity, not type), both the output itself and any bare view sharing its
storage resolve to the same producer thunk, and the mechanism works uniformly for any
tensor type (`DTensor`, etc.).

Every output is registered in the index. Who fires its thunk depends on the consumer:

- A **`remat.region` consumer** (a `RECOMPUTE` op, or a `SAVE` op receiving another SAVE
  op's output) — including one that receives a **bare view** of the output — looks the
  value up in the index on the forward and fires the producer's thunk. This is
  on-demand: an output no `remat.region` consumes is not persisted for this reason.

- An output that is **itself saved for backward** is already resident (autograd keeps it
  on the original forward graph), so it is persisted eagerly at region exit for free.

- **`remat.recompute_needs_tensor(t)`** fires the thunk explicitly for `t` (or any bare
  view of it, resolved by storage). Placed right before a **bare** op consumer, it is what
  makes that consumer work — a bare op cannot be detected, so without it the output would
  not be saved. Because the call sits on the consumer side, the output is persisted only
  when that code path runs: you can never over-save.

A bare consumer of an output that nothing persisted meets a placeholder during recompute
and raises, with a message that names the producing region and tells you to call
`remat.recompute_needs_tensor(t)` (or regionize the consumer).

In **recompute** a skipped `SAVE` op returns each output from its persisted value, or a
storage-free `_PlaceholderTensor` when none was saved — it was dead, or consumed only by a
`remat.region` (whose consume path made the producer persist it, so replay actually serves
the real value). A placeholder supports metadata/view ops (a view of a placeholder is
another placeholder) but raises if its data is actually read.

This is **producer responsibility**: the consumer does no bookkeeping; it just uses the
tensor, and the producer decides what to keep.

Consequences:

- Only `SAVE` outputs actually needed (consumed by a `remat.region`, saved for backward,
  or explicitly marked with `remat.recompute_needs_tensor`) are kept resident; they show up
  as `output.<i>` rows in `remat.format_current_memory_report()`, attributed to the
  producing op. An output consumed by nothing costs nothing.

One thing remains an **error**:

- **In-place / mutating ops** on a persisted `SAVE` output — mutating it would corrupt both
  the persisted value and the copy autograd kept for the op's backward. remat's version
  counter catches this at backward. Wrap the mutation in a `remat.region` (or apply it
  before the value leaves the producing op).

## Region inputs are liveness-tracked

Non-reentrant checkpoint pins **every** region input for the whole backward: it saves each
one at region entry and closes over it for the recompute rerun, with no notion of whether
recompute actually reads it. But an input consumed only by `SAVE` (`recompute=False`)
regions is never read during recompute — those regions are skipped and serve their outputs
from the tape — so pinning it is pure waste (a transformer block that saves its whole
attention/MLP output but only *recomputes* from a normed copy leaves the raw residual input
resident for nothing).

`torch_remat` closes this by treating the region's inputs as the outputs of one synthetic
`SAVE` op at the region boundary, reusing the exact producer-responsibility machinery above:

- checkpoint is driven with **zero** region inputs, so it neither saves them via
  `_make_saved_tensor` nor closes over them — the library owns capturing them on the forward
  and feeding them on recompute.
- Each input is registered in the save-output persist index with a persist thunk. A
  `RECOMPUTE` region consuming it fires that thunk (the input is kept and served real during
  recompute); an input only `SAVE` regions consume is never persisted and is served a
  storage-free placeholder — and since its only consumers are skipped, the placeholder is
  never read. A kept input shows up as a `<region_inputs>` op with `output.<i>` rows in
  `format_current_memory_report()`; a dropped one shows nothing.
- A requires-grad **leaf** input (e.g. a `Parameter`) bypasses the synthetic op — a
  `remat.region` may not yield a requires-grad leaf, and a leaf is owned elsewhere anyway, so
  it is fed directly at no extra memory.

An input consumed by a *bare* op is not detected automatically (same as any SAVE output);
flag it with `recompute_needs_tensor` to keep it. When `input_saved_tensors_hooks` is set the
whole mechanism is bypassed (composing per-input liveness with input offload is future work)
and checkpoint saves every input, exactly as before.

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

## recompute_needs_tensor

`remat.recompute_needs_tensor(*tensors)` forces a `SAVE` region's output to be durably
persisted for recompute. It exists because a *bare* op (anything not wrapped in
`remat.region` — a residual add, a `.reshape`, a raw kernel) consuming a `SAVE` output
cannot be detected, so the output would not be saved and the consumer would read a
placeholder during recompute.

Call it on the output tensor, placed **right before the bare op that consumes it**:

```python
y = remat.region(my_op, "my_op", recompute=False)(x)
remat.recompute_needs_tensor(y)   # persist y for the bare consumer below
z = torch.relu(y)                 # bare op — reads real data during recompute
```

Each tensor is resolved to its producer **by storage**, so passing the output itself or
any bare view of it works. Because the call sits on the *consumer* side, the output is
persisted only when this code path actually runs — you can never over-save (unlike
declaring persistence on the producer, which pays even in configs where the consumer is
absent).

It is always safe to call: on a tensor that is not a `SAVE` region's output (an ordinary
recomputed tensor, a region input), on a `recompute=True` region's output (which is real
during recompute anyway), or outside any checkpoint region, it is a no-op. So model code
may call it unconditionally, whether or not it is being checkpointed and regardless of the
producer's `recompute` setting.

The typical workflow: write the model without any annotations; if a run raises a
placeholder error during recompute, the message names the producing region and tells you
to call `remat.recompute_needs_tensor(t)` on the output, right before the bare op that
reads it, to force the producer to save it.

Wrapping the consuming op in `remat.region(..., recompute=True)` is an equivalent
alternative — remat then detects the crossing and saves the output on demand — worth
preferring when the consumer is naturally a region boundary anyway (e.g. a shared-expert
combine add). Both live on the consumer, so both persist the output only when the consumer
is present.


## Diagnostics

TODO: we should describe this more

- `remat.is_recomputing()` returns whether execution is in the recompute pass.
- `remat.collect_trace()` / `remat.trace_scope(...)` collect a reporting-only
  tree of the `remat.region` annotations seen during the original forward.
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
* Subtlety: **deferred SAVE-output saves and `capture_context`.** A SAVE region's
  output is packed only when a consumer claims it (a `remat.region`, or an explicit
  `remat.recompute_needs_tensor`), so that `pack` can fire *after* the producing region's
  `saved_tensors_hooks` scope has exited — later, at the consumer. remat still packs it
  with the hooks that were installed *where the output was produced* (snapshotted at region
  exit), so an offloader gets a consistent view. But the *ambient* state your `pack` reads
  (e.g. "the current chunk") is gone by then. Pass `capture_context` to
  `saved_tensors_hooks`: remat calls it in-window (where the output is produced) and hands
  its result to `pack(tensor, context)`. Bind your offload target there — e.g.
  `capture_context=self.cur_forward_chunk` and `pack=lambda t, chunk: chunk.tensor_push(t)`
  — and a deferred save routes to the right chunk even though it runs at the consumer. Hooks
  that don't set `capture_context` are called `pack(tensor)` as before.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
