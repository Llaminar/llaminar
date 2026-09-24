# ISA-Aware Benchmark CI Plan

Llaminar ships both AVX2 and AVX512 runtime images. Benchmark results and
regression checks must therefore treat direct CPU results as ISA-scoped data
instead of comparing every CPU run against one global baseline.

## Benchmark Driver

- Keep `.githooks/run_benchmark_check.sh` as the single benchmark driver for
  local pre-commit and CI.
- Local mode builds and benchmarks `build_v2_release/llaminar2`, deriving the
  CPU ISA from `build_v2_release/CMakeCache.txt`.
- Container mode runs the same benchmark matrix against a runtime image with:
  `--container-image IMAGE --cpu-isa AVX2|AVX512`.
- Container mode skips local rebuild, mounts `/opt/llaminar-models` read-only,
  exposes CUDA and ROCm devices, and runs `/usr/local/bin/llaminar2 benchmark`
  through the image entrypoint.

## Baselines

- GPU baselines remain device-scoped.
- Direct CPU baselines are ISA-scoped:
  `devices.cpu.isa.AVX2` and `devices.cpu.isa.AVX512`.
- Existing flat CPU baselines remain readable for backward compatibility, but
  new CPU rows should be recorded under the `isa` object.

## Results

- Benchmark JSON uses schema `llaminar.benchmark.v2`.
- Every result row records `source`, `image`, and `container_cpu_isa`.
- Direct CPU result rows also record `cpu_isa`.
- CI merges AVX2 and AVX512 full-image benchmark artifacts into one committed
  `benchmark_results/<short-sha>/benchmark_results.json` and CSV.
- The merged CI record keeps CPU rows from both ISAs and keeps one preferred
  non-CPU row set to avoid duplicate GPU trend lines.

## CI

- After the `full AVX2` runtime image passes E2E, run the benchmark hook in
  container mode and upload the result artifact.
- After the `full AVX512` runtime image passes E2E, do the same.
- A follow-up job downloads both artifacts, merges them, summarizes them, and
  on successful pushes to `develop` commits the benchmark result files back with
  `[skip ci]`.
