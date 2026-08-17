# torch_remat

`torch_remat` is a modern activation checkpointing implementation for PyTorch.
It takes the classic recipe of non-reentrant `torch.utils.checkpoint` (recompute
everything before backwards) and enhances it with the ability to mark regions
inside the recompute region to save their activations for backward, so they
don't have to be recomputed.

Compared with selective activation checkpointing (SAC), `torch_remat` allows you
to specify save/recompute decisions on regions of source code, instead of writing
a generic policy function that operates on a per-ATen operator basis.  This
allows for finer-grained recompute policies (e.g., you can easily express that
one matmul should be saved while another should be recomputed).  It's also
easier to use `torch_remat` with code that has many custom kernels (you don't
have to custom op'ify them), and the fact that `torch_remat` doesn't use a
`TorchDispatchMode` means CPU overhead is lower.

## Quick start

Put one checkpoint around the unit you want to replay, usually a transformer
block:

```python
import torch_remat as remat

# torch.utils.checkpoint.checkpoint compatible API:
# remat.checkpoint(**remat_kwargs)(func)(*args, **kwargs)
output = remat.checkpoint(region_name="layers.0")(block)(hidden_states)
```

(`remat.checkpoint(block)(hidden_states)` is intentionally not supported,
because this phrasing of the API is ambiguous with `torch.utils.checkpoint`.)

Inside the block's `forward` method, annotate calls whose backward activations
should be saved with `remat.region(..., recompute=False)`, and annotate
outputs of save regions which will be needed for recompute with
`remat.recompute_needs_tensor` (you can also omit these and `torch_remat` will
tell you which ones you need to mark):

```python
def forward(self, hidden_states):
    x = self.attention_norm(hidden_states)
    attn = remat.region(
        self.attention,
        "attention",
        recompute=False,
    )(x)
    remat.recompute_needs_tensor(attn)
    hidden_states = hidden_states + attn

    x = self.ffn_norm(hidden_states)
    moe = remat.region(
        self.moe,
        "moe",
        recompute=False,
    )(x)
    remat.recompute_needs_tensor(moe)
    return hidden_states + moe
```

The interaction between recompute and save regions is somewhat subtle;
check [Mental model](docs/mental_model.md) for more details.

## `torch.compile`

The core checkpoint and save-versus-recompute policy works under
`torch.compile` with AOTAutograd's min-cut partitioner. Eager-only hooks and
diagnostics are not available in compiled regions. See
[Compilation](docs/compilation.md) for the complete compatibility table.

## State and side effects

Code that is replayed must behave consistently with the original forward, even
if saved region bodies are skipped during replay.  In particular, if you rely
on mutable state in forwards (e.g., for RNG), you need to ensure you can
snapshot and restore this state.  Code `remat.RecomputeStateHook` and pass it
with `recompute_state_hooks=`. The hook restores state at checkpoint entry and
at every non-recomputed function.  Here is an example that takes care of
setting both a custom user RNG counter as well as standard PyTorch RNG state.

```python
from contextvars import ContextVar

rng_counter = ContextVar("rng_counter", default=0)


class CudaRNGStateHook:
    def __init__(self, device):
        self.device = device

    def snapshot(self):
        return torch.cuda.get_rng_state(self.device), rng_counter.get()

    def restore(self, state):
        cuda_rng_state, counter = state
        torch.cuda.set_rng_state(cuda_rng_state, self.device)
        rng_counter.set(counter)


rng_hook = CudaRNGStateHook(hidden_states.device)
output = remat.checkpoint(
    region_name="layers.0",
    recompute_state_hooks=(rng_hook,),
)(block)(hidden_states)
```

We don't provide a "stock" save/restore hook; in particular, `torch_remat`
doesn't support the `preserve_rng_state=True` kwarg that
`torch.utils.checkpoint` supports.  The primary reason for this is that
`preserve_rng_state` is documented to also save CPU RNG state, but in modern
PyTorch code this is unnecessary (RNG should be sampled on-device) and
expensive (a 5KB allocation is needed to snapshot the CPU MT19937 state)--and
unlike `torch.utils.checkpoint`, we will repeatedly save/load RNG state many
times per a `remat.checkpoint`.

Separately, you can check whether other code is being replayed with
`remat.is_recomputing()`, for example to suppress forward-only logging or
metrics:

```python
if not remat.is_recomputing():
    record_metric(value)
```

TODO: We should offer a simple way of checking that the replay is bitwise
equivalent to the original.

## Diagnostics

`torch_remat` also comes with a number tools for understanding the recompute/save behavior
and memory usage of your program.  Here are some things you can do:

**Trace the configured region hierarchy to see what is being saved/recomputed:**

```python
with remat.collect_trace() as trace:
    output = model(inputs)
print(trace.format())
```

```text
torch_remat trace
scope [test_flag]
  sin: save
  cos: recompute
```

**Inspect retained activations from inside a checkpoint forward:**

```python
if not remat.is_recomputing():
    remat.print_current_memory_report()
```

```text
layers.0: 28 B resident in 2 storage(s)
layers.0::attn.softmax: 28 B
  12 B  lse    (3,)  float32
  16 B  probs  (4,)  float32
```

**Inspect all live checkpoint regions plus saves reachable from a loss after
the full forward:**

```python
remat.print_saved_tensors_report(loss)
```

```text
saved for backward: 240 B resident -- 2 region(s) 192 B, outside regions 48 B

regions:
  96 B  x2  layer.0-1  (2 storages each)

outside regions: 48 B in 1 storage
       48 B  TanhBackward0 (x1)

region detail:
[x2: layer.0-1]
layer.0: 96 B resident in 2 storage(s)
layer.0::sq: 96 B
  48 B  y (output at idx 0)  (3, 4)  float32
  48 B  gf                   (3, 4)  float32
```

Allocation-site annotations and an attachable CUDA OOM observer are also
provided. See the [Diagnostics API](docs/api/diagnostics.md) for more details.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
