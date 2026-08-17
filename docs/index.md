# `torch_remat`

`torch_remat` is an activation checkpointing implementation built on PyTorch's
non-reentrant checkpoint. It replays a checkpoint body during backward by
default, while letting model authors mark calls whose backward activations
should be saved instead.

Save/recompute decisions apply to regions of source code rather than through a
generic per-operator policy. This makes it straightforward to distinguish
similar operations and to cover custom kernels without wrapping them as custom
operators.

```python
import torch_remat as remat


def block(x):
    y = remat.region(expensive_op, "expensive", recompute=False)(x)
    remat.recompute_needs_tensor(y)
    return x + y


output = remat.checkpoint(region_name="layers.0")(block)(hidden_states)
```

```{toctree}
:maxdepth: 2

mental_model
offloading
compilation
api/index
```
