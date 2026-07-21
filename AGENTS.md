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
