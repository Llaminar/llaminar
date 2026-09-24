# Perf Stats Collector

`PerfStatsCollector` is the unified structured counter/timer path for new profiling work.
It is intentionally generic so older ad hoc profilers can be bridged into it over time.

## Environment

- `LLAMINAR_PERF_STATS_SUMMARY=1` prints the unified human-readable table.
- `LLAMINAR_PERF_STATS_JSON=/path/file.json` writes machine-readable JSON.
- `LLAMINAR_PERF_STATS_CSV=/path/file.csv` writes machine-readable CSV.
- `LLAMINAR_PERF_STATS_FILTER=mtp,prefix_cache` exports only matching domains or qualified prefixes.
- `LLAMINAR_GPU_STAGE_TIMING=1` enables GPU event timing on the production graph path.
- `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` enables GPU event timing for graph replay and eager setup.
- `LLAMINAR_PERF_STATS_FILTER=stage_gpu` also enables GPU event timing for structured stage profiling.
- `LLAMINAR_PROFILING=1` is deprecated. It aliases summary plus graph-safe GPU
  replay timing, emits a warning, and no longer changes graph capture or
  executor topology.
- `LLAMINAR_PROFILE_KERNELS=1` explicitly enables the legacy
  hand-instrumented kernel tables. It is reserved for focused diagnostics and
  is not implied by PerfStats or the deprecated compatibility alias.

Truthy JSON/CSV values such as `1`, `true`, `on`, or `yes` use:

- `/tmp/llaminar_perf_stats.json`
- `/tmp/llaminar_perf_stats.csv`

Use `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` for trustworthy production-graph
GPU timing. `LLAMINAR_GPU_STAGE_TIMING=1` remains a human timeline diagnostic.
New tooling must not use deprecated `LLAMINAR_PROFILING`.

## Decode Loop Records

The `decode_loop` domain reports host-visible transaction boundaries without
instrumenting inside a captured graph:

- `orchestrated_step` measures a scalar MTP decode transaction, including
  proposal, grouped verification, acceptance, and result publication.
- `orchestrated_maintenance` measures scalar decode-boundary maintenance.
- `request_batch_step` and `request_batch_maintenance` provide the equivalent
  records for request-batched MTP.
- `sampler` and `inter_step` retain the non-orchestrated decode-loop breakdown.

These timers are enabled only by an explicit timing request. Plain JSON/CSV
counter export remains passive.

## GPU Stage Records

The `stage_gpu` domain uses GPU events, not host enqueue time.

- `source=stage_timeline`, `graph_capture_scope=eager_per_stage_events` records
  per-stage timings for eager or warmup/capture setup execution.
- `source=full_graph_capture`, `graph_capture_scope=full_graph_replay_events`
  records monolithic graph replay totals.
- `source=segmented_graph_capture`, `graph_capture_scope=segmented_replay_events`
  is reserved for explicitly admitted heterogeneous collective domains and
  records graph replay totals and segment timings.
- Capture-plan records retain stage-type inventory so replay timing can be
  interpreted without pretending graph-captured internals were individually
  timed.

## Kernel Route Records

Kernel route counters explain which backend path ran without forcing legacy
profiling. They are CPU-side counters around explicit-stream GPU launches, so
they are safe to export during graph-safe diagnostic runs.

- `kernel.cuda_native_vnni_prefill_calls` records CUDA NativeVNNI prompt-prefill
  route selection with `codebook`, `m`, `n`, `k`, `tile_id`, `split_k`, `bk256`,
  and `streamk` tags.

## Current MTP Records

The first instrumented domain is `mtp`. It records request-level decode phases such as:

- `capture_live_prefix_state`
- `condition_forward`
- `sidecar_forward`
- `verifier_forward`
- `restore_live_prefix_state`
- `replay_forward`

It also records sidecar graph internals on each participant:

- `sidecar_resolve_weight_bindings`
- `sidecar_build_graph`
- `sidecar_execute_graph`

These records are meant to make MTP regressions explainable without scraping tables or logs.
