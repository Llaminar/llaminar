---
name: cuda-kernel-profiling
description: Profile and tune Llaminar V2 CUDA kernels using LLAMINAR_PROFILING (per-kernel + per-stage timing), Nsight Systems (nsys), and Nsight Compute (ncu), then validate with the GEMM perf-test harness, the benchmark subcommand, and parity tests. Use when asked to find the slowest CUDA kernel, diagnose occupancy / register pressure / memory-bound stalls, A/B two kernel variants, or close a prefill/decode throughput gap while keeping PyTorch parity.
applyTo: "src/v2/kernels/cuda/**,src/v2/backends/cuda/**,src/v2/execution/**,tests/v2/performance/kernels/cuda/**,tests/v2/integration/kernels/cuda/**"
---

# CUDA Kernel Profiling & Tuning (Llaminar V2)

## Purpose

A repeatable, evidence-driven workflow for making Llaminar V2 CUDA kernels faster
without breaking correctness. It chains four tools:

1. **`LLAMINAR_PROFILING`** — coarse per-kernel + per-stage GPU timing to find the hotspot.
2. **`nsys`** — timeline / launch-count view to understand kernel ordering and CPU↔GPU overlap.
3. **`ncu`** — per-kernel hardware-counter deep dive (occupancy, registers, spills, stalls).
4. **Perf-test harness + benchmark + parity** — isolate, A/B, and validate the change end-to-end.

The golden rule of this codebase: **isolated kernel speedups do not always translate
to full-model speedups.** Always confirm a win on the real `benchmark` subcommand, and
always re-run parity before trusting it.

---

## Step 0: Establish the target and baseline

Before touching anything, capture a baseline so every change is measured against it.

```bash
# Release build of the engine (always Ninja, never limit parallelism)
ninja -C build_v2_release llaminar2

# Baseline throughput (1st "Throughput" line = prefill, 2nd = decode)
./build_v2_release/llaminar2 benchmark -m <model>.gguf -d cuda:0 2>/dev/null | grep -iE "Throughput"
```

Record the prefill/decode tok/s and the noise band (run 2-3×; typical noise is a few
tok/s on prefill, <0.5 tok/s on decode). A change inside the noise band is **not** a win.

> **Parity gate (must stay PASS the whole time):**
> ```bash
> ctest --test-dir build_v2_integration -R "<Model>ParityTest_(Prefill|Decode)Parity.*CUDA" --output-on-failure
> ```

---

## Step 1: Find the hotspot with `LLAMINAR_PROFILING`

`LLAMINAR_PROFILING=1` enables **per-kernel timing + executor overhead + GPU stage timing**
in a single flag. It needs no special build (works on Release).

```bash
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark -m <model>.gguf -d cuda:0
```

Read the GPU Stage Timeline / kernel tables and rank stages by total GPU time. Profiled
operations include `GEMM_Q8`, `ATTENTION`, `FFN_*`, `LM_HEAD`, `RMS_NORM`, `SWIGLU`,
`ROPE`, MoE routing/expert kernels, etc.

> ⚠️ **CRITICAL CAVEAT — profiling changes execution mode.**
> `LLAMINAR_PROFILING=1` sets `executor_profiling=true`, which **disables decode GPU
> graph capture** (`buildDecodeCapturePolicy` gates `allow_segmented_capture` on
> `!executor_profiling`). So *profiled* decode runs eager (per-stage events) and is
> **slower than the real benchmark**. Use profiling to find *where* time goes
> (relative ranking), never to quote an absolute decode tok/s number. Measure absolute
> throughput with profiling **off**.

> ⚠️ **Never leave `LLAMINAR_PROFILING` exported.** Verify a clean shell before
> quoting any benchmark number:
> ```bash
> env | grep -i LLAMINAR_PROFIL   # must print nothing
> ```

Pick the single highest-time stage that is plausibly tunable (skip stages that are
already compute-bound at the math limit, e.g. the big dense GEMMs, unless that's the
explicit target).

---

## Step 2: Timeline sanity check with `nsys`

Use `nsys` to confirm the kernel of interest dominates and to see launch counts / sync stalls.

```bash
# --no-mpi-bootstrap is REQUIRED so the profiler attaches to llaminar2, not the mpirun wrapper
sudo /usr/local/cuda/bin/nsys profile -t cuda --stats=true -o /tmp/trace -f true \
  ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d cuda:0 \
  -m <model>.gguf -p "test" -n 10

# Per-kernel summary (gives the launch order needed for ncu --launch-skip)
/usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum /tmp/trace.nsys-rep
```

`nsys` also tells you the kernel's mangled name and how many times it launches — both
needed to target `ncu` precisely.

> `--no-mpi-bootstrap` is ONLY for profiler/debugger attach (`ncu`/`nsys`/`gdb`/`perf`).
> It disables NUMA-aware thread pinning, so it gives misleading *performance* numbers —
> never use it for benchmarks or production.

---

## Step 3: Deep-dive a kernel with `ncu`

`ncu` replays each kernel 4-8× through hardware counters, so always **target one kernel**
and **skip warmups**.

```bash
# sudo -E preserves env (LLAMINAR_*); ncu needs sudo for counter access
sudo -E /usr/local/cuda/bin/ncu \
  --kernel-name "<mangled_or_substr>" \
  --launch-skip 1 --launch-count 1 \
  --section SpeedOfLight \
  --section Occupancy \
  --section LaunchStats \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  --target-processes all \
  -o /tmp/k_ncu -f \
  ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d cuda:0 \
  -m <model>.gguf -p "test" -n 1

# Read it back
sudo /usr/local/cuda/bin/ncu -i /tmp/k_ncu.ncu-rep --page details
```

> 🧹 **MANDATORY CLEANUP after every ncu run** (ncu leaves zombie processes that hold
> the GPU and corrupt the next run):
> ```bash
> sudo pkill -9 -f "llaminar2 oneshot"; sudo pkill -9 -f "ncu --kernel"; sleep 1; nvidia-smi
> ```

### What to read first

| Metric | Healthy | Red flag → likely cause |
|--------|---------|-------------------------|
| `Local Memory Spilling` | 0 | >0 → register pressure spilling to DRAM |
| `Registers Per Thread` | matches `__launch_bounds__` | unexpected → check launch config / occupancy ceiling |
| `Compute (SM) Throughput` | >60% (compute-bound) | <30% → latency-bound |
| `DRAM Throughput` | <30% (compute-bound) | >60% w/ low compute → spilling or bad access pattern |
| `Achieved Occupancy` | near theoretical | low + low compute → not enough warps to hide latency |
| `Warp Cycles Per Issued Instruction` | <15 | >30 → severe stalls (check WarpStateStats breakdown) |

| Warp stall dominated by | Cause | Fix direction |
|-------------------------|-------|---------------|
| `Barrier` | uneven work across `__syncthreads()` | fewer sync points, warp-level sync |
| `L1/TEX` + spilling | spilled regs reloaded from local mem | reduce live registers / relax `__launch_bounds__` MIN_BLOCKS |
| `Long Scoreboard` (memory) | global-load latency | improve coalescing / prefetch / more warps |

---

## Step 4: Isolate & A/B with the GEMM perf-test harness

For GEMM/MoE-expert kernels, the perf test gives a fast, correctness-gated A/B loop that
is far quicker than full-model rebuilds. It shares the production decode/codebook helpers,
so it is a valid proxy for those kernels.

```bash
# Build the perf test (Release)
ninja -C build_v2_release tests/v2/v2_perf_cuda_native_vnni_gemm

# Correctness gate is cosine >= 0.9990 vs cuBLAS. The full-shape sweep is SLOW
# (single-threaded CPU reference, several minutes) — ALWAYS scope to the shapes you care about.
LLAMINAR_CUDA_NATIVE_GEMM_SHAPES="35BMoE_Expert_GateUp,35BMoE_Expert_Down" \
  ./build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm --gtest_filter="*"
```

Useful env filters (comma-separated, case-insensitive): `LLAMINAR_CUDA_NATIVE_GEMM_FORMATS`,
`LLAMINAR_CUDA_NATIVE_GEMM_SHAPES`, `LLAMINAR_CUDA_NATIVE_GEMM_PREFILL_M`,
`LLAMINAR_CUDA_NATIVE_GEMM_BENCH`, `LLAMINAR_CUDA_NATIVE_GEMM_WORKERS`.

Add new production shapes to `kQwenShapes` in
`tests/v2/performance/kernels/cuda/gemm/CUDANativeVNNIGemmPerfCommon.h` so the harness
covers the exact GEMMs a model actually runs.

### Generated GEMM/GEMV dispatch training pipeline

Do not land source-level "one shape gets this tile" overrides for CUDA NativeVNNI
unless the user explicitly asks for a temporary experiment. The durable path is:

1. Add or confirm the production shapes in
   `tests/v2/performance/kernels/cuda/gemm/CUDANativeVNNIGemmPerfCommon.h`.
2. For decode/GEMV dispatch tables, prefer the turnkey refresh wrapper:
   `scripts/refresh_native_vnni_dispatch_tables.sh --backend cuda --profile qwen36`.
   It sweeps canonical `M={1,2,3,4}` verifier buckets, trains the tree model,
   overlays exact known-shape winners, validates the generated include, and can
   install it with `--install`.
   The CUDA decode sweep trainer uses deterministic structurally-valid packed
   payloads, not per-element random quantization, so giant LM-head refreshes do
   not stall in host fixture generation. It still constructs the real tensor
   classes and production VRAM-pool preparation path.
   The default CUDA decode family set is `wide,kpar,direct`.  The perf harness
   uses the production VRAM-pool preparation path, which does not own ROWPAR's
   optional row-major auxiliary weight view; do not add `rowpar` back to the
   standard refresh unless the trainer has an explicit row-major-owner mode and
   model-level parity proves the generated table.
   If the full qwen36 inventory is too large for one pass, use the same staged
   profiles as ROCm: `--profile qwen36-core` for FFN/GDN projections and
   `--profile qwen36-lm-head` for the LM-head shape. Do not install either
   staged artifact until the combined model-level parity and benchmark gates
   have passed.
3. Use `--profile family-smoke` for a bounded representative training pass before
   a full acceptance refresh. This profile is stratified by format: it runs one
   small sweep per codebook/family, writes per-format partial CSVs, combines them,
   then runs the normal train/generate/validate flow. This avoids a capped sweep
   accidentally covering only the first format in the list.

   ```bash
   scripts/refresh_native_vnni_dispatch_tables.sh --backend cuda \
     --profile family-smoke \
     --cuda-formats Q4_0,IQ4_XS \
     --m-values 1,2
   ```

   `family-smoke` is a workflow/proxy gate, not production acceptance. Do not
   replace broad checked-in tables from this profile alone; run `qwen36` or
   `all`, then model-level parity and benchmark gates, before `--install`.
   The wrapper intentionally uses proxy hit-rate thresholds for `family-smoke`;
   production fallback-family/exact thresholds apply only to `qwen36` and `all`.

4. Generate prefill tables with
   `tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemm_dispatch.py`.
   This reads `TileSweep_AllStrategies` CSVs, validates codebook ids through the
   shared `tests/v2/performance/kernels/native_vnni_codebooks.py` map, skips
   off-policy `M` rows by default, can merge an existing generated include via
   `--base-include`, and emits
   `src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchGenerated.inc`.
5. When debugging the decode trainer itself, the lower-level flow is:
   `v2_perf_cuda_blockwise_tensorcore_gemm_sweep` emits the sweep CSV,
   `infer_gemv_dispatch_heuristic.py` trains the M-aware fallback tree, and
   `analyze_cuda_tc_gemv_dispatch.py --base-include <tree.inc>` overlays exact
   `(M,N,K)` known-shape winners. Both the fallback tree and overlay fallback
   must stay M-aware; Qwen3.6 LM-head can legitimately prefer WIDE/DIRECT at
   M=1 and KPAR at M=2..4 for the same `(N,K)`.
   The trained fallback is the real policy: it must generalize by aspect ratio
   and work-size/log-shape features. Exact `(M,N,K)` rows are overlays only.
   Do not replace the broad fallback with a table that only recognizes today’s
   model dimensions.
   The overlay must score the runtime dispatch surface, not raw source-format
   rows. CUDA NativeVNNI GEMV dispatch is keyed by `(codebook,M,N,K)`, and
   aliases such as `Q4_1/Q4_K`, `Q5_1/Q5_K`, and `IQ4_NL/IQ4_XS` cannot receive
   separate runtime tunings for the same key. Collapse alias rows to one
   aggregate codebook-level winner before enforcing exact-hit thresholds.
   The trainer must bind an explicit non-blocking CUDA stream to the GEMM kernel
   and record CUDA events on that stream. A `stream=0` trainer log or
   `cudaEventRecord(start)` without a stream argument is a bug, not a benign perf
   detail.
   The trainer must prepare/upload/repack each weight once per format+shape
   before candidate timing and size its `DeviceWorkspaceManager` from
   `IWorkspaceConsumer::getWorkspaceRequirements()`; fixed 512 MiB budgets fail
   on giant LM-head small-M partial buffers.
6. Validate generated artifacts with
   `tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py`
   and the CUDA generator alias tests:
   `V2_Unit_CUDAPrefillDispatchGeneratorAliases`,
   `V2_Unit_CUDAGemvDispatchGeneratorAliases`, and
   `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`. For decode trainer changes,
   include `V2_Unit_GpuWorkspaceAllocationPolicy` so the explicit-stream
   trainer contract is checked.

Example prefill-generation shape:

```bash
python3 tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemm_dispatch.py \
  --input benchmark_results/cuda_dense_mtp/.../Q4_K_sweep.csv \
  --base-include src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchGenerated.inc \
  --output /tmp/CUDANativeVNNIPrefillDispatchGenerated.inc \
  --summary /tmp/cuda_prefill_dispatch_summary.txt

python3 tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py \
  /tmp/CUDANativeVNNIPrefillDispatchGenerated.inc
```

After updating a checked-in generated include, rerun the focused CUDA GEMM route
regression for the affected shape and the relevant Qwen3.6 CUDA parity cells.

### MTP verifier dispatch mode

Grouped MTP verifier rows are stricter than ordinary fast decode: rows may be
published to live state, so they must be reproducible against rowwise serial
decode under strict L2/cos/KLD/max-abs gates. CUDA exposes this through
`ITensorGemm::beginVerifierDecodeEquivalentScope()`, which selects the canonical
small-M NativeVNNI dispatch/reduction policy and disables prefill/concurrent
decode reordering without enabling global `LLAMINAR_DETERMINISTIC`. Stage code
must use that shared RAII interface, never call CUDA `extern "C"` mode toggles
or set environment variables directly.

> **Build gotcha:** the MoE expert kernel `#include`s the decode header
> `src/v2/kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh`. After editing that header you
> MUST `touch src/v2/kernels/cuda/moe/CUDAMoEKernels.cu` before `ninja`, or the change
> won't be picked up. CUDA compiles are slow (~min, cicc-bound); ccache hits are fast.

---

## Step 5: Validate on the real model (the only verdict that counts)

A perf-test win is necessary but **not sufficient**. Confirm on the full model and keep parity.

```bash
# 1. Rebuild engine
touch src/v2/kernels/cuda/moe/CUDAMoEKernels.cu   # if a .cuh include changed
ninja -C build_v2_release llaminar2

# 2. Same-session A/B (build baseline binary, measure; build variant, measure)
./build_v2_release/llaminar2 benchmark -m <model>.gguf -d cuda:0 2>/dev/null | grep -iE "Throughput"

# 3. Parity MUST stay PASS
ninja -C build_v2_integration llaminar2_core llaminar2
ctest --test-dir build_v2_integration -R "<Model>ParityTest_.*CUDA" --output-on-failure
```

Accept the change only if: full-model throughput improves **beyond the noise band** AND
parity stays PASS. Reject (and `git stash`/revert) otherwise.

### Hard-won lesson (why isolated wins can regress the model)

Register-ceiling-bound fused kernels (e.g. the MoE prefill expert kernel: ~96 regs/thread,
0 spills, ~92% of the register ceiling) trade throughput for occupancy. A decode/codebook
micro-opt that is faster in isolation but **adds even a few temp registers** can drop
occupancy in the fused kernel and *regress* full-model prefill, even though the standalone
GEMM got faster. When the kernel is at the register ceiling, only **register-neutral** or
**algorithmic** changes help — instruction-count tricks that cost registers will not.

---

## Step 6: Record the result

- Update the relevant baseline (e.g. `.githooks/benchmark_baseline.json`) only with
  human-approved numbers.
- Note rejected experiments and *why* (regression cause) so they aren't retried blindly.
- Do **not** create ad-hoc markdown reports unless asked; use `changelog/` (ISO-date prefix)
  for durable write-ups.

---

## Quick reference — one-liners

```bash
# Hotspot ranking (relative only; decode runs eager under profiling)
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark -m M.gguf -d cuda:0

# Timeline + launch order
sudo /usr/local/cuda/bin/nsys profile -t cuda --stats=true -o /tmp/t -f true \
  ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d cuda:0 -m M.gguf -p "x" -n 10

# Per-kernel counters (skip warmup, 1 launch)
sudo -E /usr/local/cuda/bin/ncu --kernel-name "K" --launch-skip 1 --launch-count 1 \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --section MemoryWorkloadAnalysis --section WarpStateStats --target-processes all \
  -o /tmp/k -f ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d cuda:0 -m M.gguf -p "x" -n 1

# Cleanup (ALWAYS after ncu)
sudo pkill -9 -f "llaminar2 oneshot"; sudo pkill -9 -f "ncu --kernel"; sleep 1; nvidia-smi

# Scoped, correctness-gated GEMM A/B
LLAMINAR_CUDA_NATIVE_GEMM_SHAPES="ShapeA,ShapeB" \
  ./build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm

# Clean-shell benchmark for the real number
env | grep -i LLAMINAR_PROFIL   # expect empty
./build_v2_release/llaminar2 benchmark -m M.gguf -d cuda:0 2>/dev/null | grep -iE "Throughput"
```
