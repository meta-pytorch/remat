# Compilation

`torch_remat` supports its core save-versus-recompute policy under
`torch.compile`. The compiled implementation does not run the eager
saved-tensor tape: Dynamo traces the body once, then AOTAutograd's min-cut
partitioner decides which graph nodes to keep for backward and which to
rematerialize.

The policy is meaningful for compiler backends that use the min-cut
partitioner, including the normal Inductor path. Only operations captured in
the compiled graph receive the policy; ordinary `torch.compile` restrictions
on graph breaks, Python side effects, and data-dependent Python control flow
still apply.

There are two ways to combine them, and features can behave differently in each:

- **A compiled checkpoint**: `torch.compile` traces a `remat.checkpoint` call.
  There is no eager tape, and the body is never replayed; the partitioner
  decides what to recompute.
- **A compiled function inside an eager checkpoint**: an eager
  `remat.checkpoint` calls a `torch.compile`'d function. The eager tape is live,
  and the checkpoint replays the compiled function on recompute.

Rows describe a compiled checkpoint unless they say otherwise.

| Feature | Status under `torch.compile` | Behavior |
|---|---|---|
| `remat.checkpoint()(fn)(...)` | Supported | Runs as a non-reentrant PyTorch checkpoint. Captured operations recompute by default. |
| Unannotated operations inside `checkpoint` | Supported | They are part of the checkpointed graph and recompute by default. |
| `remat.region(..., recompute=True)` | Supported | Adds no override; the enclosing checkpoint's recompute policy applies. |
| `remat.region(..., recompute=False)` | Supported | Tags every decomposed compute node in the region `MUST_SAVE`, so the partitioner keeps it in the forward graph instead of recomputing it. Native and composite callables and custom `autograd.Function.apply` bodies are supported. |
| A `region` constructed before compilation | Supported | Both eagerly constructed wrappers and `region(...)` calls made while Dynamo traces are supported. |
| `region` outside `checkpoint` | Supported, as a no-op | Calls the wrapped function without adding a partitioner policy, matching eager behavior. |
| `remat.recompute_needs_tensor(...)` | Supported, as a no-op | Compiled graphs carry values directly, so there is no eager placeholder to persist. It is also a no-op in a compiled function inside an eager checkpoint, where the tape is live; call it from eager code before the compiled call. |
| `remat.Box` | Supported | Boxes pass through the compiled graph like any other Python object, with no eager tape to keep them on. In a compiled function inside an eager checkpoint, reading a SAVE region's box keeps its contents as in eager; before torch 2.14 the read graph-breaks (raises under `fullgraph=True`). |
| `remat.save_for_backward` | Supported, names ignored | Forwards directly to `ctx.save_for_backward`; mapping keys only feed eager memory reports. |
| `checkpoint(determinism_check=...)` | Supported | Forwarded to the underlying non-reentrant checkpoint. |
| `region_name` and region `name` | Accepted, diagnostics unavailable | Names do not affect the compiled policy because there is no eager tape or region report. |
| `remat.is_recomputing()` | `False` inside a compiled `checkpoint` | Python executes once during tracing; rematerialization happens in the partitioned backward graph, not by replaying the Python body. Elsewhere it reads the eager phase, which graph-breaks before torch 2.14 (raises under `fullgraph=True`). |
| A compiled function inside an eager `checkpoint` | Supported, `region` is a no-op | The eager checkpoint replays the function on recompute, so `recompute=False` region bodies rerun. Guard forward-only side effects with `remat.is_recomputing()`, which reports the eager phase. |
| Eager checkpoint output validation | Not applied | Compiled outputs follow Dynamo's pytree and aliasing rules rather than the eager tape's one-hop Tensor container validation. |
| `checkpoint(saved_tensors_hooks=...)` | Unsupported | Rejected during compilation with `NotImplementedError`. |
| `recompute_state_hooks` | Unsupported | Rejected during compilation with `NotImplementedError`. |
| `remat.saved_tensors_hooks(...)` | Unsupported in compiled regions | The eager remat hook stack does not mediate tensors saved by the compiler. |
| Memory reports, OOM reports, and `collect_trace` | Unsupported for compiled regions | Compiled regions do not populate the eager tape and therefore do not appear in these diagnostics. |
| Nested `remat.checkpoint` regions | Unsupported | A checkpoint region must not contain another checkpoint region. |

`preserve_rng_state=True` remains unsupported in both eager and compiled modes.
Use `recompute_state_hooks` in eager mode when external RNG or other non-tensor
state must be realigned; there is not yet a compiled equivalent.

The API suite runs in eager mode and in strict compile mode. Compile-aware test
call sites use `checkpoint_for_test`, which is the ordinary `remat.checkpoint`
decorator in eager mode and wraps the checkpointed callable in
`torch.compile(backend="aot_eager", fullgraph=True)` in compile mode. Full-graph
compilation prevents graph breaks from turning a passing test into accidental
eager coverage. Known incompatibilities are strict expected failures: an
unexpected pass fails the suite, keeping the unsupported catalogue tied to
reproduced behavior. Focused compile tests also inspect the AOTAutograd forward
and backward graphs to verify partition placement.
