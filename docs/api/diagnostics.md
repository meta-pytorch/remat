# Diagnostics API

## Policy traces

```{eval-rst}
.. autofunction:: torch_remat.collect_trace

.. autofunction:: torch_remat.trace_scope

.. autoclass:: torch_remat._trace.RematTrace
   :members: format
```

## Memory reports

```{eval-rst}
.. autofunction:: torch_remat.format_current_memory_report

.. autofunction:: torch_remat.print_current_memory_report

.. autofunction:: torch_remat.format_saved_tensors_report

.. autofunction:: torch_remat.print_saved_tensors_report
```

## CUDA memory annotations

```{eval-rst}
.. autofunction:: torch_remat.memory_snapshot_annotate

.. autofunction:: torch_remat.clear_memory_snapshot_cache
```

## OOM helpers

```{eval-rst}
.. autofunction:: torch_remat.discover_autograd_roots

.. autofunction:: torch_remat.format_oom_saved_tensors_report

.. autofunction:: torch_remat.oom_observer
```
