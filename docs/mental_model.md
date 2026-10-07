# Mental model

At a first approximation, you can simply think of `torch_remat` as simply
`torch.utils.checkpoint` (recompute everything), but with the ability to trade
increased memory (`recompute=False`) to reduce compute.  However, there is a
subtlety about what *exactly* is saved when you mark a region as
`recompute=False`; in particular, we may need to save output tensors of a
`recompute=False` region, so that we can resume recomputing after a save
region.  The purpose of this doc is to walk you through the rest of the
subtleties.

## Basics of activation checkpointing

This section recaps some basic properties of activation checkpointing that are
true for all of PyTorch's activation checkpointing implementations.

When we do activation checkpointing, instead of saving tensors for backwards
as we would normally do in autograd, we instead recompute the forward function
right before performing backwards.  To do this, we only have to save the
inputs into the recompute region.

In a typical LLM, we perform activation checkpointing on a per-transformer
block basis (in general, when you have a repeated layer structure, it's good
to AC your layers).

Suppose I have a network with two checkpointed layers.  Without AC, their
execution order looks like:

```text
layer 0 forward
layer 1 forward
loss
layer 1 backward
layer 0 backward
```

With AC, the execution order is:

```text
layer 0 forward
layer 1 forward
loss
layer 1 recompute
layer 1 backward
layer 0 recompute
layer 0 backward
```

Activation checkpointing reduces overall memory usage, because we no longer
retain the saved activations for all layers at the usual high watermark (the
loss compute); instead we only retain inputs into the checkpointed layers and
materialize their save for backward tensors on a layer-by-layer basis.

## How SAVE regions (`recompute=False`) work

The twist for `torch_remat` is that, inside of a `remat.checkpoint` layer, you
can write this:

```python
z = remat.region(func, "func", recompute=False)(y)
```

And instead of recomputing it, we guarantee that we will skip running `func`
during recompute.  We colloquially call these SAVE regions, as opposed to
RECOMPUTE regions (`recompute=True` regions, or also any operation that isn't
in a `remat.region`.)

If you think about this API carefully, there are three immediate implications
that fall out from this:

1. A SAVE region will save all tensors it needs for backwards (as opposed to
   dropping them, as is the case for normal AC).

2. Because a SAVE region doesn't actually run during recompute, we may not
   actually have the output tensors during recompute!  `torch_remat` will
   only produce output tensors of SAVE regions if we needed to save them
   anyway (e.g., for backwards); otherwise, it will return a "placeholder"
   tensor which has the same metadata as the real tensor but doesn't contain
   any real data.  If you try to run a kernel on it, it will error.

3. Many operations don't need to save their output for backwards, but if
   we are going to use that output to RECOMPUTE something later, we need
   to save that output.

During forwards, we try to infer whether or not an output of a SAVE region
will need to be saved by tracking if it ends up being used by a RECOMPUTE
region.  If you have an explicit `remat.region(..., recompute=True)` that
consumes this tensor, we can infer it automatically; otherwise, you should
call `remat.recompute_needs_tensor` write before its usage (we could have
used a tensor subclass to track usage in forwards, but this ends up being
pretty complicated, so we think this explicit API call is a good trade-off).

## What remains resident

The description above technically tells you everything you need to know to
figure out if a tensor will be saved for backwards or not, but it's helpful to
work through some examples to get some intuition.  Consider three regions
inside one checkpoint, where region C is RECOMPUTE (the third region ensures
`z` is an interior value rather than the checkpoint output, which is treated
specially):

```text
checkpoint input
      x
      |
      v
  +----------+      +----------+      +----------+
  | region A |--y-->| region B |--z-->| region C |--> output
  | saves p  |      | saves q  |      +----------+
  +----------+      +----------+
```

Suppose A's backward needs its internal value `p`, and B's backward needs its
internal value `q`.  For every combination of A/B policy, we can write
out exactly what tensors are saved for backwards versus recomputed.

| A policy | B policy | Resident for these regions | Recreated |
| --- | --- | --- | --- |
| SAVE | SAVE | `z`, `p`, `q` | none |
| SAVE | RECOMPUTE | `y`, `p` | `z`, `q` |
| RECOMPUTE | SAVE | checkpoint anchor `x`, `z`, `q` | `y`, `p` |
| RECOMPUTE | RECOMPUTE | checkpoint anchor `x` | `y`, `z`, `p`, `q` |

When both A and B are SAVE, `y` is neither resident nor recreated: neither
backward needs it, and B is skipped during replay, so B only sees a
placeholder for it.  Likewise, `z` is resident whenever B is SAVE only because
region C recomputes and reads it; if C were also SAVE (and its backward did not
need `z`), `z` would not be kept either.

In general, the producer and consumer policies determine how an intermediate
reaches replay:

| Producer | Consumer | What happens |
| --- | --- | --- |
| RECOMPUTE | RECOMPUTE | The producer recreates the value. |
| RECOMPUTE | SAVE | The SAVE call is skipped, while any input its backward needs is rederived during replay. |
| SAVE | SAVE | The producer's output is ferried to the skipped consumer as needed. |
| SAVE | RECOMPUTE | The producer persists its output so the consumer can rerun. |

One thing to note is that when a RECOMPUTE region requires one of its input tensors
to be saved, we attribute the cost of saving it to the producer of the tensor
(so called "producer responsibility").  This keeps the invariant that only
SAVE regions save tensors, and also implies that if an output is needed for
any recompute, it *always* is available in all code in the recompute that has
a reference to it.

## No early stop: wrap the tail in a SAVE region

`torch.utils.checkpoint` stops replay early, as soon as every tensor saved
under the checkpoint has been recomputed.  If the last op in the checkpointed
function saves only tensors produced earlier (e.g. a final matmul saving its
inputs), native checkpoint never reruns it.

`remat.checkpoint` always replays the whole function.  It saves a zero-element
tensor at the function's output to force replay to start before any inner
backward runs, and since that is the last tensor saved, early stop never
triggers before the end.  To skip the trailing ops, wrap them in a SAVE region:

```python
def block(x, w):
    h = x.sin()
    return remat.region(torch.matmul, "out_proj", recompute=False)(h, w)
```

This retains no extra memory in the case where early stop would have skipped
the tail: the matmul's saved `h` comes from a RECOMPUTE producer, so it is
rederived during replay (see the table above) rather than kept resident.

## Code that needs no gradient

`torch_remat` only acts on code that autograd records.  If the whole checkpoint
runs under `torch.no_grad()` or `torch.inference_mode()`, the function runs once
as a plain call: regions are ordinary calls, nothing is saved, and nothing is
ever recomputed.

Inside a checkpoint that does record, grad-free code (a `torch.no_grad()`
block, or ops on tensors that don't require grad, like a mask or a statistic)
saves nothing for backward.  If it runs in a RECOMPUTE region, though, replay
still reruns it: `torch_remat` cannot tell whether a bare op later in the replay
reads its result.  Put such code in a SAVE region instead and replay skips it.
Since it saved nothing, it costs nothing, unless a recompute needs its output,
which is then persisted as described above.
