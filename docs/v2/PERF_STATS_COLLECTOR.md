# Perf Stats Collector

`PerfStatsCollector` is the unified structured counter/timer path for new profiling work.
It is intentionally generic so older ad hoc profilers can be bridged into it over time.

## Environment

- `LLAMINAR_PROFILING=1` enables collection.
- `LLAMINAR_PERF_STATS_JSON=/path/file.json` writes machine-readable JSON.
- `LLAMINAR_PERF_STATS_CSV=/path/file.csv` writes machine-readable CSV.
- `LLAMINAR_PERF_STATS_FILTER=mtp,prefix_cache` exports only matching domains or qualified prefixes.
- `LLAMINAR_GPU_STAGE_TIMING=1` enables GPU event timing on the production graph path.
- `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1` enables GPU event timing for perf stats export without legacy profiling.
- `LLAMINAR_PERF_STATS_FILTER=stage_gpu` also enables GPU event timing for structured stage profiling.

Truthy JSON/CSV values such as `1`, `true`, `on`, or `yes` use:

- `/tmp/llaminar_perf_stats.json`
- `/tmp/llaminar_perf_stats.csv`

Use `LLAMINAR_GPU_STAGE_TIMING=1` or `LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1`
for trustworthy GPU stage timing. Avoid `LLAMINAR_PROFILING=1` when measuring
production decode throughput because it enables legacy executor profiling and can
change graph-capture behavior.

## GPU Stage Records

The `stage_gpu` domain uses GPU events, not host enqueue time.

- `source=stage_timeline`, `graph_capture_scope=eager_per_stage_events` records
  per-stage timings for eager or warmup/capture setup execution.
- `source=segmented_graph_capture`, `graph_capture_scope=segmented_replay_events`
  records graph-captured replay totals and segment timings. These are accurate
  captured-graph event timings, not per-stage attributions inside the captured graph.
- `source=segmented_graph_capture`, `graph_capture_scope=segmented_capture_plan`
  records the segment plan and stage-type inventory so replay segment timing can
  be interpreted without pretending graph-captured internals were individually timed.

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
