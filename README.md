# torch_remat

`torch_remat` is a small library of helper functions for writing activation
checkpointing in a style where all tensors are recomputed by default, and
users explicitly specify what tensors that they want to save for backwards.
This is good for users who wish to have fine-grained control over saved
activations, and want the specification of what is saved for backwards to be
explicit (at the cost of having what to recompute determined implicitly.)
In LLM training, it would be typical for the entire transformer block to be
the unit of recompute.

How does this compare to existing PyTorch checkpointing APIs?

* Compared to non-reentrant AC: in fact, this API is built on top of
  non-reentrant AC!  We do provide our own top-level `checkpoint` API to
  enforce that the forward recompute is triggered immediately at the beginning
  of the recompute block backwards, rather than lazily upon the first load
  tensor hook, as is the default for non-reentrant checkpointing.  But one
  good way of thinking about this API is that, non-reentrant AC forces you to
  recompute everything, and this API maintains a tape that lets you recompute
  less than everything for some subregions of the AC region.

* Compared to SAC: there are two big differences.  First, SAC requires use
  of a TorchDispatchMode to give it the ability to skip operations during
  recompute; idiomatic use of `torch_remat` instead asks you to manually
  modify your autograd functions to add the capability of skipping recompute.
  You can optionally make use of a TorchDispatchMode to have native PyTorch
  operations save for backwards, but this is not recommended because what
  exactly is saved for backwards is not explicit when you do this.  Second,
  SAC currently operates via a policy function which makes a determination by
  classifying an operation as cheap to recompute or not.  `torch_remat` allows
  for fine-grained choices on a tensor-by-tensor basis if you want to save
  them for backwards.  In principle, SAC could support this mode of operation
  too, but this style of API hasn't made it to upstream yet.

## API

At the top level unit of recompute (e.g., a transformer block), write this:

```python
import torch_remat as remat

y = remat.checkpoint()(block)(x)
```

The first call binds checkpoint options, the second call binds the function,
and the third call passes user arguments to `block`. This avoids collisions
between checkpoint option names, function attributes, and keyword arguments
that the user function wants to receive. `remat.checkpoint(block)(x)` is
intentionally not supported: requiring the empty `checkpoint()` call avoids
making this look interchangeable with `torch.utils.checkpoint.checkpoint`,
which cannot accept this calling convention for backward-compatibility
reasons.

The behavior is otherwise similar to `torch.utils.checkpoint`, except that the
recompute will happen immediately upon the backwards of `block` (and we also
reserve the right to make internal implementation strategy changes in the
future.) `remat.checkpoint` intentionally exposes only the PyTorch checkpoint
options that are expected to matter for remat users, and always uses
non-reentrant checkpointing internally. By default, all contents transformer
block will now get recomputed immediately before backwards.  The same
correctness requirements of `torch.utils.checkpoint` apply here: it must be
safe to run the forwards again (no side effects that run twice), RNG must be
synchronized, you shouldn't compute metrics in the recompute, the recompute
must run the same series of operations as the original.

The checkpointed function must return a `Tensor`, or an exact builtin `tuple`,
`list`, or `dict` whose values recursively satisfy the same rule. Subclasses
such as namedtuples, custom mappings/sequences, and non-tensor leaves are
rejected instead of being passed through silently.

By default, `torch_remat` releases remat-owned saved tensors as the remat tape
is consumed during recompute. This keeps memory lifetime tied to the backward
pass. `backward(retain_graph=True)` is detected automatically via
`torch._C._autograd._get_current_graph_task_keep_graph()`, so no manual
opt-in is needed — the remat tape is preserved when `retain_graph=True`.

`torch_remat` maintains its own autograd tape, analogous to the classic
PyTorch autograd graph.  This tape is responsible for ferrying saved
activations and tensors needed from recompute from the forward to the
recompute phases.  Unlike the classic autograd tape, all saved activations are
explicitly named.  We then use the classic PyTorch autograd graph to ferry
saved activations from recompute to backwards.

We can think of the inside of the checkpoint as a series of `SAVE` and
`RECOMPUTE` blocks.  During recompute, the calling convention across these
blocks is that `RECOMPUTE` blocks are specifically responsible for
saving/loading their inputs, if they were not already available (because they
were recomputed or already saved for backwards.)  In practice, `SAVE` region
outputs are the interesting unavailable case: during replay they are represented
by placeholder tensors, and a downstream `RECOMPUTE` region that needs the real
value is responsible for saving that input during the original forward and
loading it back at its own boundary.  This means `SAVE` blocks are
compositional: you can chain as many `SAVE` blocks as you like together, and
we will not unnecessarily save their outputs for recompute.

### Forward and Recompute Flow

Each remat-aware autograd function participates in two executions of the
checkpointed region: the original forward, and the recompute forward that runs
during backward.  A policy specifies whether or not the op is `RECOMPUTE`d
during recompute, or skipped because we `SAVE`d everything we need for
backwards.  While the policy of an op specifies if we save things it needs for
backwards, the *outputs* of an op are saved by later consumer ops.

During the original forward:

- A `SAVE` op runs normally, saves its named backward tensors to the remat
  tape, records output metadata, and marks its outputs as coming from a skipped
  producer.
- A `RECOMPUTE` op runs normally and does not save its backward tensors to the
  remat tape. However, `save_or_load_inputs()` may save any input whose
  producer is a `SAVE` op, because that producer will not recreate the real
  value during recompute.

During recompute:

- A `SAVE` op does not run its real forward. `maybe_load_saved()` restores its
  saved backward tensors into the recompute autograd context and returns
  metadata-only output placeholders.
- A `RECOMPUTE` op runs its real forward. `save_or_load_inputs()` loads any
  inputs that were saved during the original forward because their producer was
  skipped.

For any given intermediate tensor, this is how it is made available during
recompute:

* Inputs to the overall checkpointed region: unconditionally saved
* Output of a RECOMPUTE op: recomputed
* Saved for backward of a RECOMPUTE op: recomputed
* Output of a SAVE op: saved to the remat tape, but only if a RECOMPUTE op consumes it
* Saved for backwards of a SAVE op: saved to the remat tape

## How to avoid recomputing autograd.Function

Everything inside a `remat.checkpoint` gets recomputed.  To avoid recomputing
an expensive autograd function, you need to write your autograd function in
a particular stylized way.  The idea is the autograd function forwards will get called
twice: once in the initial forwards, and then again in the recompute.  We need
to appropriately save/load tensors depending on whether or not we wish to
recompute or save the activations of this operation.

Let's suppose you had an autograd function that previously looked like this:

```python
class MyOp(autograd.Function):
    def forward(ctx, x):
        y = my_op_fwd1(x)
        z = my_op_fwd2(y)
        ctx.save_for_backward(x, y)
        return z

    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        return my_op_bwd(x, y, grad_z)
```

We need to make two public facing API changes for the function:

1. We need a way to tell if the activations needed for backwards should be
   saved or recomputed.  This can be done in any way you want, although the
   most straightforward way is to add an extra `remat_policy` argument to
   forwards so you can control this from the call site.  We give a stock
   policy enum `CheckpointPolicy` which can be `RECOMPUTE` or `SAVE`.

2. We need a way to name the specific operator call, such that it is unique
   in the transformer block.  This is because `torch_remat` takes the opinion
   that you should have a unique, stable name for every saved activation,
   and enforces uniqueness of names in its tape representation.  Unique names
   give stronger desync protection between forward and recompute, let memory
   reports localize usage to exact call sites, and lay the groundwork for a
   future API where users specify what to save by name.  If an autograd
   function is called only once in a transformer block, you can hardcode a
   name for it inside the function; otherwise, consider making the string
   name an argument that can be passed in.

With these new arguments, we can then restructure the inside of the autograd
forward function as so:

```python
class MyOp(autograd.Function):
    def forward(ctx, x, op_name, remat_policy):
        handle = remat.get_handle(ctx, op_name, remat_policy)
        if (ret := handle.maybe_load_saved()) is not None:
            return ret
        x = handle.save_or_load_inputs(x)
        y = my_op_fwd1(x)
        z = my_op_fwd2(y)
        handle.save_for_backward(
            {"x": x, "y": y},  # order matters!
        )
        return handle.record_outputs(z)

    # Unchanged!
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        return my_op_bwd(x, y, grad_z)
```

Let's walk through what each API does.  They do different things depending on
if you are doing forward or recompute, and what the rematerialization policy
is for the function.

### `remat.get_handle(ctx, op_name, remat_policy)`

We always construct a `RematHandle` at the beginning of forwards.  This records
the policy for the named autograd Function call and gives the rest of the
forward a handle for interacting with that call's tape record.  The `op_name`
must be unique within the checkpoint region.

### `handle.maybe_load_saved()`

After constructing the `RematHandle`, call this method to see if you can
short-circuit performing actual compute.

In forwards, this always returns None (since we cannot have saved anything).

In recompute, this will short circuit the execution of this function when
the policy is `SAVE`, since we have saved the necessary activations for
backwards.  We'll load them straight into `ctx` and then short circuit
execution.

Note that `ret` is NOT guaranteed to have real data: we can generate
data-inaccessible placeholder tensors, if the output wasn't saved for
backwards. These placeholders preserve size, stride, dtype, and device
metadata, but throw if data pointer access or real computation is attempted.
This is because the output may not actually be needed at all to finish the rest
of the recompute, so we want to wait until the first usage
(`save_or_load_inputs`) to save/load it.  For simplicity, these placeholder
tensors do not have accurate aliasing relationships until they are loaded.

### `handle.save_or_load_inputs(*args)`

When the policy is `RECOMPUTE`, in the initial forwards, we check if any input
would be unavailable during recompute because it is the output of a `SAVE`
region.  Those inputs would replay as data-inaccessible placeholder tensors, so
we save the real tensors here and load them back during recompute.  Inputs
produced by `RECOMPUTE` regions are recomputed as real tensors and do not need
extra tape storage here.

Note that we order this after `maybe_load_saved`, so this is a no-op when the
policy is `SAVE`.

### `handle.save_for_backward(saved_tensors)`

This intuitively does the same thing as `ctx.save_for_backward` but it gives
names to all the saved activations (we require a dict of string names to saved tensors,
with the convention that the order of keys in the dict corresponds to the
original order on `ctx`) and knows how to save activations on the
`torch_remat` tape, so that `handle.maybe_load_saved` can load the activations
back into `ctx` (as a reminder: in classic non-reentrant activation checkpoint, we
construct PyTorch's autograd graph twice; once in forwards, and once in
recompute, but it's the recompute autograd graph that actually gets executed
in backwards.)

Note that when the policy is `RECOMPUTE`, the original forward activations are
not saved into the `torch_remat` tape. The recompute forward still calls
`ctx.save_for_backward` for the ordinary PyTorch autograd graph that will run
backward.

### `handle.record_outputs(*outs)`

This gives names to all outputs (`save_for_backward` isn't guaranteed to have
done so, as not all outputs are necessarily saved for backwards) and, if
the policy is `SAVE`, records metadata for them so that `handle.maybe_load_saved`
can generate data-inaccessible placeholder tensors to return.

A singular output of a custom autograd Function call is conventionally known as
`out` in memory reports. If there are multiple outputs, they are named by
position: `0`, `1`, etc.
The return value of this function preserves the single-tensor versus tuple
schema expected by the autograd engine.

### Decorator style API

If you don't want to write the inside of your forward function, we offer a
magical decorator that takes care of everything:

```
class MyOp(autograd.Function):
    @remat.auto_forward("x", "y")
    def forward(ctx, x, op_name, remat_policy):
        ...
```

This decorator assumes the last two arguments of the forward function are
`op_name` and `remat_policy`, and takes care of constructing the `RematHandle`
and calling its methods (including passing a special proxy `ctx` object to
intercept the `ctx.save_for_backward` call).

## How to avoid recomputing native PyTorch APIs

The above APIs only work if you can put them inside a custom autograd
function.  For calls to native PyTorch APIs (e.g., `torch.mm`), they do not
work.  We will simply assume by default that all of these calls should be
recomputed; an often reasonable assumption as extremely computationally
expensive operations are frequently implemented from scratch and thus have
custom autograd functions.

**Important limitation:** a native PyTorch op cannot consume the output of a
remat-aware autograd Function with policy `SAVE`.  Attempting this will raise
a `RuntimeError`.  To fix, either:

1. Wrap the native op with `remat.native_save_region` so it is also saved
   (its output is replayed during recompute without reading the placeholder).
2. Move the native op into a custom autograd function with `auto_forward`.
3. Change the upstream op's policy to `RECOMPUTE`.

In the rare situation where you want to avoid recomputing some basic PyTorch
compute (e.g., a matrix multiply), we support a `remat.native_save_region`
function wrapper which you can use to specify that this region should not be
recomputed in backwards.  To prevent recompute, this wrapper uses PyTorch SAC.

## Offloading (TODO)

This API should support offloading.  The idea is that instead of saving to the
tape, we offload the activations, and then onload them when we would have
loaded them.  `CheckpointPolicy.OFFLOAD` would let us indicate we want this.

The actual offload implementation isn't in this package.  So we should have
hooks so you can put in your own offload implementation.

There is still some softness in our offloading plan.  In particular, it's
not obvious how to prevent blocking on offloading until backwards actually
needs to use it.  We will need to work this in more detail and refine this
API.  Currently, offloading is not implemented.

## Tape runtime details

`torch_remat` maintains its own tape which it uses to transfer tensor from
forward to recompute, before passing them off to the traditional autograd
tape.  We take some care to make sure that we handle a number of PyTorch edge
cases around aliasing and mutation correctly, as well as to ensure prompt
deallocation, so we describe the design here.

The crux of the matter is that we need to save tensors for recompute/backward
for a variety of reasons:

* Our policy is `SAVE` and a tensor is needed for backwards
* Our policy is `SAVE`, and an output tensor (not saved for backwards) is needed
  for a subsequent `RECOMPUTE` region

Aliasing can also be quite complicated.  In general, the same tensor can be
saved for backwards multiple times.  We can also save aliases into the same
underlying storage.  We can also return an alias into the input tensor from
an autograd function.

Finally, inplace mutation can invalidate a saved tensor.  Traditional autograd
tape uses version counters to detect if this situation has occurred; we need
to replicate this logic for recomputation.

Here is our general strategy:

* It harmless to save multiple views of the same underlying storage.  Internal
  refcounting will ensure we deallocate the storage after the last tensor
  referencing is deallocated.  Understanding the aliasing structure is useful
  when we are printing the memory usage of saved activations, but otherwise
  tensors saved on the tape are plain tensors.

* The tape is composed of a sequence of internal records, one per autograd
  Function call which executes during forward.  By default, remat-owned slots
  are released as each record is consumed during recompute, after the saved
  tensors have been transferred to the recompute autograd graph.  When
  `retain_graph=True` is active, slots are preserved automatically so the
  tape remains available for later traversals.
  Tensors can still live beyond recompute due to graph retention, aliasing, or
  being needed for backwards.

* Any time there is an output of a `SAVE` region in forwards, during recompute
  phase we will always generate a fresh data-inaccessible placeholder tensor
  for that output.  We intentionally do not preserve output aliasing
  relationships here: if the original output was a view of an input or saved
  tensor, the replayed output is still a fresh placeholder.  Downstream
  torch_remat-aware custom autograd functions must save and directly use the
  real tensors they need during the original forward instead of relying on
  replayed output storage or aliasing.

## Ownership and memory lifetime

First, let's describe the easy situation.  In the easy case:

- The remat tape's is owned by the output tensors of the checkpointed
  region (via their autograd graph, which is responsible for triggering
  recompute when backwards is executed.)
- We run forward, recompute, backward, in exactly this sequence.  No double
  backwards. `retain_graph=False`. Every allocated autograd node gets run in
  backward, we don't have to worry about the user not calling backward.
- We expect saved for backwards tensors to get deallocated after the
  backward node that needs them has executed.
- We expect inputs saved for recompute to get deallocated after (all)
  the recompute that needs them has executed.

Intuitively, all we need to do is make sure we free saved for backwards
tensors and input tensors right after we use them.  Conventional autograd
works in the same way: we free the graph as we execute it, to provide
guarantees about when saved for backward tensors get deallocated.

We might worry about these two situatiosn:

- What if `backward()` is never called, and instead the autograd saved state
  goes out of scope and becomes dead?  We would hope saved tensors can be
  deallocated in this case.
- What if `retain_graph=True` is called?  This is detected automatically via
  `torch._C._autograd._get_current_graph_task_keep_graph()`, so the remat
  tape is preserved without any manual opt-in.

A refined memory model prefers us to associate lifetimes with the autograd
saved state itself.  If the autograd saved state dies (because the `grad_fn`
because dead, or because we ran `backward` with `retain_graph=False`), this
naturally ensures things get deallocated.  However, this is a bit complicated
to implement, and there is always a "clean" version of the user code that
doesn't have this problem (in particular, by ensuring you `detach()` before
running operations that won't get fed into the autograd graph).  So we do NOT
do this, and instead stick to the simplified model above which keeps our code
simple.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
