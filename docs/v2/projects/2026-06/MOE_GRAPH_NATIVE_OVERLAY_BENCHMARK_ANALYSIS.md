# MoE Graph-Native Overlay Benchmark Analysis

This page documents the Phase 20 analysis tooling for graph-native MoE overlay benchmark sweeps. It is intentionally conservative: the analyzer reports missing evidence as `insufficient data` rather than turning partial logs into performance claims.

## Run The Phase 19 Sweep

Build the Release binary with the same backend options used by the target hardware lane:

```bash
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel
```

Run the curated sweep. The script writes one directory per config under `benchmark_results/moe_overlay/<timestamp>-<git-hash>/` and records `command.txt`, `exit_code.txt`, `stdout_stderr.log`, and `moe_overlay_profile.csv` when the profiler flushes successfully.

```bash
LLAMINAR_MOE_OVERLAY_MODEL=/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  scripts/run_moe_graph_native_overlay_benchmarks.sh
```

Do not use `--no-mpi-bootstrap` for benchmark sweeps. That flag is for direct profiling/debugging only and disables runtime placement behavior that benchmarks need to measure.

## Run The Analyzer

Analyze a real sweep directory:

```bash
python3 scripts/analyze_moe_overlay_benchmarks.py \
  benchmark_results/moe_overlay/<timestamp>-<git-hash> \
  --output benchmark_results/moe_overlay/<timestamp>-<git-hash>/analysis.md
```

Run the synthetic gate fixture:

```bash
python3 scripts/analyze_moe_overlay_benchmarks.py \
  tests/v2/performance/moe/fixtures/moe_overlay_sample_run \
  --output /tmp/moe_overlay_analysis.md
```

Run the Release CTest gate:

```bash
cmake --build build_v2_release --target v2_perf_moe_graph_native_overlay_analysis --parallel
ctest --test-dir build_v2_release -R "V2_Perf_MoEGraphNativeOverlayAnalysis" --verbose
```

## Metrics

`selected_rows` counts routed token rows sent by a graph-native sparse dispatch or local expert row. `inbound_rows` counts rows received by a participant or reduce root. `compact_dispatch_bytes` and `compact_return_bytes` are sparse payload sizes. `dense_bytes_avoided` is the dense transfer volume minus compact transfer volume; it is a savings counter, not a throughput result. `cpu_fallback_rows` counts rows handled by CPU expert participants. `gpu_cached_rows` counts rows handled by GPU resident expert tiers.

Timing columns are summed by name from `moe_overlay_profile.csv`. `compute_ms` is local expert compute time when present. `domain_reduce_ms`, `cross_domain_reduce_ms`, and fields containing `wait` are reported as sparse wait time. Fields containing `scatter` and `import` are reported separately so sparse return-reduce overhead can be diagnosed before GPU-native pack/unpack work.

Prefill and decode wall timing are parsed from each config `stdout_stderr.log` when BenchmarkRunner prints `PREFILL`, `DECODE`, and `TOTAL` tables.

## Evidence Required For Expected Observations

All-GPU versus mixed GPU/CPU requires parsed benchmark throughput for at least one all-GPU config and one mixed config. The analyzer uses all-fit all-GPU configs when present, then reports whether the comparable prefill, decode, and overall throughput values favor all-GPU.

GPU expert budget scaling requires two or more low/medium/high/all-fit configs in the same family with parsed timing. Without that, the analyzer reports `GPU budget throughput trend: insufficient data`.

CPU fallback correlation requires profiler CSV rows with `cpu_fallback_rows`. Mixed configs should carry fallback rows while all-GPU configs should not. A nonzero fallback count is expected to explain mixed GPU/CPU overhead, but it is not by itself a latency proof.

Prefill GEMM-like and decode GEMV-like behavior requires kernel-class evidence in logs or profiler output in addition to prefill/decode timing. If only wall timing is present, the analyzer reports `Prefill/decode kernel shape: insufficient data`.

Sparse transport bottleneck diagnosis requires profiler timing for wait, scatter, import, or reduce fields. If sparse transport time is comparable to or larger than compute time, investigate sparse pack/unpack and collectives before tuning expert kernels.

## Interpreting Failures And Missing Data

Failed configs remain in the report with their exit code. Missing `stdout_stderr.log` means benchmark timing cannot support throughput observations. Missing `moe_overlay_profile.csv` means row, byte, fallback, and sparse transport observations cannot be made for that config. The analyzer should still produce Markdown for partial runs so interrupted sweeps are easy to triage.

## Final Reports

Place durable benchmark reports either beside the raw run under `benchmark_results/moe_overlay/<run>/analysis.md` or under `docs/v2/perf/` when the report is intended to be preserved with the repository. Reports in docs must include hardware inventory, model path, config list, profiler tables, timing evidence, and tuning recommendations.