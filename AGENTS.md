# External Synchronization Boundary

This directory is externally synchronized.

- Keep changes in this directory in commits separate from caller-specific or
  integration changes outside this directory.
- Commit messages, code comments, tests, and documentation for changes in this
  directory must be suitable for public release. Do not refer to non-public
  projects, products, repositories, infrastructure, URLs, identifiers, or code
  names.
- Put integration changes and their rationale in caller-owned commits stacked
  above the generic change.
- Before submitting, inspect the changed paths and split any commit that mixes
  this directory with caller-specific code.

# torch.compile Contexts

`torch.compiler.is_compiling()` is true in two different situations, and code in
this directory must handle each (see the top of `docs/compilation.md`):

1. A compiled checkpoint (`in_compiled_checkpoint()` in
   `torch_remat/_compile.py`): there is no eager tape and the body is never
   replayed; the partitioner decides what to recompute.
2. A compiled function inside an eager checkpoint: the eager tape is live, the
   checkpoint replays the function on recompute, and `is_recomputing()` reports
   the eager phase.

- Don't read `is_compiling()` as "there is no eager tape". Use
  `in_compiled_checkpoint()` for case 1.
- In case 2, Python in a traced function runs once per compile, not once per
  call. A per-call effect on the tape must either be something Dynamo replays on
  every call (a plain attribute write on an input, like `Box`'s read flag) or
  run in eager code.
- Test case 2 explicitly. The `TORCH_REMAT_TEST_COMPILE=1` suite compiles whole
  checkpoints, so it only covers case 1.
- When a feature behaves differently in the two cases, say so in its row of
  `docs/compilation.md`.
