# Perf Stats Collector

`PerfStatsCollector` is the unified structured counter/timer path for new profiling work.
It is intentionally generic so older ad hoc profilers can be bridged into it over time.

## Environment

- `LLAMINAR_PROFILING=1` enables collection.
- `LLAMINAR_PERF_STATS_JSON=/path/file.json` writes machine-readable JSON.
- `LLAMINAR_PERF_STATS_CSV=/path/file.csv` writes machine-readable CSV.
- `LLAMINAR_PERF_STATS_FILTER=mtp,prefix_cache` exports only matching domains or qualified prefixes.

Truthy JSON/CSV values such as `1`, `true`, `on`, or `yes` use:

- `/tmp/llaminar_perf_stats.json`
- `/tmp/llaminar_perf_stats.csv`

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
