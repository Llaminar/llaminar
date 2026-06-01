# Graph-Native MoE Overlay Benchmarks

These configs drive the Release benchmark lane for Qwen3.5 graph-native MoE overlay placement. They intentionally use whole-expert routed domains with `compute=replicated_experts`; none of these configs use the deleted shadow-domain or tensor-parallel-expert overlay runtime.

## Build

```bash
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel
```

The lightweight harness gate only validates config shape and command generation:

```bash
cmake --build build_v2_release --target v2_perf_moe_graph_native_overlay --parallel
ctest --test-dir build_v2_release -R "V2_Perf_MoEGraphNativeOverlay" --verbose
```

## Run The Sweep

```bash
LLAMINAR_MOE_OVERLAY_MODEL=/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  scripts/run_moe_graph_native_overlay_benchmarks.sh
```

You can also pass the model as the first argument:

```bash
scripts/run_moe_graph_native_overlay_benchmarks.sh /path/to/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

The script uses `build_v2_release/llaminar2` by default, enables `LLAMINAR_PROFILING=1`, never adds `--no-mpi-bootstrap`, and writes logs under `benchmark_results/moe_overlay/<timestamp>-<git-hash>/`.

## Hardware And Model

The curated configs assume a Qwen3.5 or Qwen3.6 MoE model with 256 routed experts. The focused ROCm config uses one MPI rank with two local ROCm participants in a `ReplicatedExperts` domain. The CUDA/ROCm all-GPU configs expect a CUDA rank and a ROCm rank. Mixed configs add one CPU fallback rank. The `benchmark.mpi_ranks` key in each YAML records the intended rank count for the sweep script and for reproducible reports.

## Expected Observations

Prefill should behave like GEMM-heavy expert batches, while decode should behave closer to GEMV because each step routes a small number of token rows. All-GPU configs should outperform mixed GPU/CPU fallback configs once the model fits the target devices. Throughput should improve as more routed experts move into GPU-resident tiers, with the all-fit variant acting as the upper-bound placement for this harness.

## MoE Profiler Metrics

With `LLAMINAR_PROFILING=1`, the benchmark prints the MoE expert overlay profiling summary at shutdown. The sweep script also sets `LLAMINAR_MOE_EP_PROFILE_CSV` per config, producing a CSV with dispatch, transfer, fallback, and graph-native sparse-stage metrics such as `selected_rows`, `transfer_bytes`, `compute_ms`, `domain_reduce_ms`, `cross_domain_reduce_ms`, `cpu_fallback_rows`, `gpu_cached_rows`, `compact_dispatch_bytes`, and `compact_return_bytes`.

Inspect one run with:

```bash
column -s, -t benchmark_results/moe_overlay/<run>/<config>/moe_overlay_profile.csv | less -S
```

## Transfer Tracing Guardrail

Graph-native overlay decode should move compact sparse payloads, not full dense hidden-state tensors. For a targeted large-transfer check, use D2H-only tracing with a minimum byte threshold:

```bash
LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1 \
LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1048576 \
./build_v2_release/llaminar2 oneshot \
  --config configs/moe_overlay/cuda_hot_rocm_warm_cpu_cold_static.yaml \
  -m /path/to/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p "Sanity check" -n 4 -t 0
```

Compact sparse dispatch/return bytes are expected and should also show up in the MoE profiler CSV. Repeated large D2H transfers during one-token decode usually mean a debug path or stage boundary is calling `TensorBase::data()` on a device-authoritative dense activation. More production hardening guidance lives in `docs/v2/MOE_GRAPH_NATIVE_OVERLAY_PRODUCTION_HARDENING.md`.

## Analyze A Sweep

Phase 20 adds a dependency-free analyzer that reads the run directory produced by the sweep script and emits a conservative Markdown report:

```bash
python3 scripts/analyze_moe_overlay_benchmarks.py \
  benchmark_results/moe_overlay/<timestamp>-<git-hash> \
  --output benchmark_results/moe_overlay/<timestamp>-<git-hash>/analysis.md
```

The analyzer compares all-GPU and mixed GPU/CPU configs only when benchmark timing exists, reports GPU budget trends only when multiple timed capacity buckets are available, and labels missing logs or profiler CSVs as `insufficient data`. Full metric definitions and report guidance live in `docs/v2/perf/MOE_GRAPH_NATIVE_OVERLAY_BENCHMARK_ANALYSIS.md`.
