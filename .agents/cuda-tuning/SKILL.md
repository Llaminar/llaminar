---
name: cuda-kernel-profiling
description: Profile and tune Llaminar V2 CUDA kernels using graph-safe PerfStats replay timing, Nsight Systems (nsys), and Nsight Compute (ncu), then validate with the GEMM perf-test harness, the benchmark subcommand, and parity tests. Use when asked to find the slowest CUDA kernel, diagnose occupancy / register pressure / memory-bound stalls, A/B two kernel variants, or close a prefill/decode throughput gap while keeping PyTorch parity.
---

# CUDA Kernel Profiling & Tuning (Llaminar V2)

## Purpose

A repeatable, evidence-driven workflow for making Llaminar V2 CUDA kernels faster
without breaking correctness. It chains four tools:

1. **PerfStats GPU event timing** — production-graph replay timing and structured route/host counters.
2. **`nsys`** — timeline / launch-count view to understand kernel ordering and CPU↔GPU overlap.
3. **`ncu`** — per-kernel hardware-counter deep dive (occupancy, registers, spills, stalls).
4. **Perf-test harness + benchmark + parity** — isolate, A/B, and validate the change end-to-end.

The golden rule of this codebase: **isolated kernel speedups do not always translate
to full-model speedups.** Always confirm a win on the real `benchmark` subcommand, and
always re-run parity before trusting it.

## Production hot-path invariants

CUDA kernel tuning must preserve the production execution architecture, not
trade architecture for an isolated microbenchmark win:

- **No dynamic allocation or deallocation.** Hot-path kernels and launch
  bridges consume persistent workspace bindings. `cudaMalloc`,
  `cudaMallocAsync`, `cudaFree`, container growth, or hidden scratch allocation
  belongs in explicit initialization/workspace infrastructure only.
- **No host transfer.** Do not insert H2D/D2H copies, host-visible mirrors, or
  host polling into execution. Deliberate RAM/disk KV-cache tier movement is a
  transfer-service operation at an explicit lifecycle boundary, not kernel
  plumbing.
- **No stream or device synchronization.** `cudaStreamSynchronize`,
  `cudaDeviceSynchronize`, blocking copies, and event synchronization on the
  host are forbidden in the hot path. They are permitted only in explicitly
  scoped test/profiler result collection or terminal result surfacing.
- **No order-dependent atomics in batch-invariant paths.** Grouped verifier,
  deterministic decode, and batch-invariant prefill kernels must give every
  output a unique deterministic writer. An atomic is acceptable only when its
  value is provably independent of inter-thread order, bitwise equivalence is
  covered by the all-format/M-totality sweep, and profiling proves it is
  economical.
- **Use explicit producer/consumer events.** A producer launches on an explicit
  non-null stream and publishes completion by recording an event on that exact
  stream. A consumer on another stream waits for that event before reading.
  Never publish a write on a guessed/default stream, transition coherence
  manually after a blocking sync, or replace the dependency with a global
  barrier.
- **Remain graph-capturable end to end.** Persistent buffers, launches, event
  record/wait edges, and collectives must be capturable. Homogeneous GPU
  inference requires one complete captured graph; segmented execution is not a
  tuning fallback.

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

## Step 1: Find the hotspot with graph-safe PerfStats

PerfStats measures the same production graph topology used for ordinary
inference. Enable structured output and explicit GPU-event timing around graph
replay:

```bash
LLAMINAR_PERF_STATS_JSON=/tmp/cuda-profile.json \
LLAMINAR_PERF_STATS_SUMMARY=1 \
LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1 \
./build_v2_release/llaminar2 benchmark -m <model>.gguf -d cuda:0
```

Rank `stage_gpu.graph_replay.total` and graph/segment rows alongside
`forward_graph`, `forward_pass`, `kernel`, `mtp`, and model-specific route
counters. A monolithic captured graph intentionally does not pretend to have
per-stage event attribution inside replay. Use `nsys` to identify kernels
inside that graph, then target a single launch with `ncu`.

`LLAMINAR_PROFILING=1` is deprecated. It aliases the PerfStats summary and GPU
replay-event request for compatibility, emits a warning, and must never disable
capture or select eager execution. New commands and automation must use the
explicit `LLAMINAR_PERF_STATS_*` variables.

Pick the single highest-time stage that is plausibly tunable (skip stages that are
already compute-bound at the math limit, e.g. the big dense GEMMs, unless that's the
explicit target).

---

## Profiler attachment and privilege rules

- Resolve Nsight Systems with `command -v nsys`; the devcontainer package
  installs it as `/usr/local/bin/nsys`, independently of the CUDA toolkit.
  Nsight Compute remains `/usr/local/cuda/bin/ncu`.
- Run both tools through `sudo -E` on hosts where
  `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`. An
  unprivileged `nsys` launch can appear successful while collecting only CUDA
  graph-construction metadata and no executed kernels.
- Preserve `LLAMINAR_*` variables across privilege escalation with `sudo -E`,
  or pass the exact variables after `sudo`. Ordinary `sudo` strips them.
- Pass `--no-mpi-bootstrap` only when profiling or debugging `llaminar2`
  directly; otherwise Nsight attaches to the `mpirun` wrapper. Never use the
  flag for production or canonical benchmark measurements.
- Do not pass `--no-mpi-bootstrap` to standalone test/performance binaries;
  they do not auto-bootstrap MPI.
- Write `.nsys-rep` and `.ncu-rep` artifacts under `/tmp` or another explicit
  result directory, not in the repository.

### Prove profiler attachment before a model run

Do not spend a model load on an unproven profiler setup. First profile a focused
integration test that executes one eager launch and one captured replay:

```bash
sudo -E "$(command -v nsys)" profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true \
  --output=/tmp/llaminar-nsys-cuda-smoke \
  ./build_v2_integration/tests/v2/v2_integration_cuda_moe_kernel \
  --gtest_filter=Test__CUDAMoEKernel.RouteWithTensorsTiledPrefillCapturesAfterWarmup

"$(command -v nsys)" stats --report cuda_gpu_kern_sum,cuda_api_sum \
  /tmp/llaminar-nsys-cuda-smoke.nsys-rep
```

The smoke report must contain a `CUDA GPU Kernel Summary`, including both
launch instances from the focused test. A report containing
`CUDA_GRAPH_NODE_EVENTS` or `CUDA_GRAPH_EVENTS` but no
`CUPTI_ACTIVITY_KIND_KERNEL` is **not** an execution trace. Its diagnostics
normally say `CUDA profiling might have not been started correctly` or report
only graph-node creation. Check privilege first.

CUDA 13 conditional/device-loop graphs expose a second tool boundary on Ampere:
the tested Nsight Systems 2025.3 and 2026.1 releases can collect ordinary eager
and simple captured graphs yet omit all activities from Llaminar's complete
production conditional graph. Confirm this by comparing the smoke test with the
real graph report; never disable production graph execution and present the
eager timeline as production evidence. Use targeted Nsight Compute node
profiling below for the real graph.

Do not defer `ncu` attachment with `--profile-from-start off` or a late
`cudaProfilerStart()` range for this graph topology. That technique profiles a
simple already-instantiated graph, but the tested Llaminar conditional graph
exposes no inner kernel nodes when profiling begins after graph construction.
Keep NCU active from process start and use its graph-node kernel filter to
select one production launch.

---

## Step 2: Timeline sanity check with `nsys`

Use `nsys` to confirm the kernel of interest dominates and to see launch counts / sync stalls.

```bash
# --no-mpi-bootstrap is REQUIRED so the profiler attaches to llaminar2, not the mpirun wrapper
sudo -E "$(command -v nsys)" profile -t cuda --stats=true -o /tmp/trace -f true \
  ./build_v2_release/llaminar2 oneshot --no-mpi-bootstrap -d cuda:0 \
  -m <model>.gguf -p "test" -n 10

# Per-kernel summary (gives the launch order needed for ncu --launch-skip)
"$(command -v nsys)" stats --report cuda_gpu_kern_sum /tmp/trace.nsys-rep
```

`nsys` also tells you the kernel's mangled name and how many times it launches — both
needed to target `ncu` precisely.

> `--no-mpi-bootstrap` is ONLY for profiler/debugger attach (`ncu`/`nsys`/`gdb`/`perf`).
> It disables NUMA-aware thread pinning, so it gives misleading *performance* numbers —
> never use it for benchmarks or production.

### Inventory durations inside the complete production graph with NCU

Conditional/device-loop graphs on Ampere may be invisible to Nsight Systems
even when a simple graph smoke is visible. Use this proven one-counter NCU
transaction to enumerate the real production graph before selecting a deep
profile. The critical details are: attach to `llaminar2` directly with
`--no-mpi-bootstrap`, keep profiling active from process start, request graph
**node** profiling, use application replay, and filter out model-load kernels
so the launch budget reaches captured inference:

```bash
RUNTIME_KERNELS='regex:(buildGrouped|build_active|count_per|cuda_attention|cuda_derive_attention|cuda_gated|cuda_gdn|cuda_q_gate|cuda_short_conv1d|exclusive_scan|float_to_int|fp32_|grouped_|groupedImma|quantize_activations|router_gate|scatter_tokens|shared_expert|softmax_topk|embedding_lookup|fused_|mtpConcat|requestTerminal|cuda_kv|residual_add|rmsnorm|rope_|flash_attention|nativeVnniTC|route_logits|shiftedMTP|ring_append|ring_gather)'

sudo -E /usr/local/cuda/bin/ncu \
  --target-processes all \
  --replay-mode application \
  --graph-profiling node \
  --kernel-name-base demangled \
  --kernel-name "$RUNTIME_KERNELS" \
  --launch-count 1200 \
  --metrics gpu__time_duration.sum \
  --clock-control none \
  --force-overwrite \
  --export /tmp/llaminar-production-node-duration \
  ./build_v2_release/llaminar2 benchmark --no-mpi-bootstrap \
    -m <model.gguf> -d cuda:0 \
    --prompt-file <fixed-prompt.txt> \
    --seed <seed> --temperature <temperature> \
    --mtp --mtp-draft-tokens <depth> \
    --mtp-depth-policy fixed \
    --mtp-verify-mode speculative-sampling \
    <remaining-exact-release-arguments>

/usr/local/cuda/bin/ncu \
  --import /tmp/llaminar-production-node-duration.ncu-rep \
  --page raw --csv \
  > /tmp/llaminar-production-node-duration.csv

# Recover the exact profiler command and target provenance from any report.
/usr/local/cuda/bin/ncu \
  --import /tmp/llaminar-production-node-duration.ncu-rep \
  --page session --csv
```

Choose `--launch-count` above the expected runtime-node count, then verify the
CSV reached the terminal graph kernels rather than silently truncating at the
budget. Sum `gpu__time_duration.sum` by exact demangled kernel name and retain
call counts plus grid/block geometry; aggregate names without geometry can
hide one pathological shape behind many cheap launches. Inspect the CSV units
row before summing: NCU auto-scales `gpu__time_duration.sum` and may emit `us`
for one report and `ms` for another. Normalize every row to one unit rather than
assuming a fixed scale. This first pass uses one metric and is for attribution
only. It does not replace warmed unprofiled Release timing, and it does not
certify occupancy or spills.

If NCU prints `No kernels were profiled`, first run the profiler-attachment
smoke above, then inspect the report's `Available Kernels`. Do not switch to
eager execution, segmented graphs, late `cudaProfilerStart()`, or a host-side
surrogate to make the profiler easier to use; those are different execution
paths.

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

# Inspect the sections supported by the installed Nsight version
/usr/local/cuda/bin/ncu --list-sections
```

### Profile a node inside the real captured graph

`ncu` has a graph-node profiler independent of the Nsight Systems timeline.
Use it when the Systems version cannot expand Llaminar's conditional graph:

```bash
sudo -E /usr/local/cuda/bin/ncu \
  --target-processes all --graph-profiling node \
  --kernel-name-base demangled \
  --kernel-name 'regex:<unique-production-kernel-substring>' \
  --launch-count 1 \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --clock-control none --force-overwrite \
  --export /tmp/llaminar-real-node \
  ./build_v2_release/llaminar2 benchmark --no-mpi-bootstrap <exact-release-args>

sudo /usr/local/cuda/bin/ncu \
  --import /tmp/llaminar-real-node.ncu-rep \
  --page details --print-details all
```

Match one unique kernel family and set `--launch-count 1`. Default kernel replay
backs up device-written memory before each metric pass; with a resident 35B
model this can take roughly a minute even for one node. Keep the section set
minimal, or collect a one-pass custom metric list when only spill/vectorization
proof is needed. Never interpret the profiled duration as canonical latency;
use the unprofiled Release benchmark for timing.

Kernel families and template winners can change after rebuilding generated
dispatch. If a selector produces `No kernels were profiled`, read NCU's
`Available Kernels` list and select the exact current production family. A
deliberately impossible selector with a one-pass section such as `LaunchStats`
is a quick inventory probe. Do not carry yesterday's selector into today's
profile without checking it.

For the Qwen 3.6 35B grouped-prefill path, a proven real-graph selector is
`regex:groupedImmaProjectionKernel`. `MemoryWorkloadAnalysis` must report zero
`derived__local_spilling_requests`; `LaunchStats`, `Occupancy`, and
`SpeedOfLight` supply registers, launch geometry, achieved occupancy, and the
compute/memory throughput balance.

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

### Target standalone and A/B launches

Run `nsys` first and count only launches matching the eventual `--kernel-name`
filter. `--launch-skip` counts matching launches, not all launches in the
process. Profile variant A and variant B in separate, otherwise-identical
invocations with `--launch-count 1`; do not compare replay duration from the
reports. Use the saved reports only for counters and resource diagnostics, and
use the unprofiled harness for canonical latency.

For standalone performance binaries, invoke `nsys`/`ncu` on the binary
directly. Keep `--target-processes all` when the harness can fork, preserve its
shape/format/M filters, and recalculate launch skip whenever the warmup or
candidate order changes.

### Prove NativeVNNI IMMA and vectorized memory execution

Do not infer tensor-core use from a source-level `mma.sync` spelling or a
candidate-family name. Profile the actual production-selected candidate and
record the GA102 IMMA and memory-width counters on one isolated extra launch:

```bash
sudo -E /usr/local/cuda/bin/ncu \
  --kernel-name "nativeVnniTC" --launch-skip 1 --launch-count 1 \
  --metrics \
sm__inst_executed_pipe_tensor_op_imma.sum,\
smsp__average_inst_executed_pipe_tensor_op_imma_per_warp,\
smsp__sass_inst_executed_op_memory_128b.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio \
  --target-processes all -o /tmp/native_vnni_instruction_proof -f \
  ./build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm \
  --gtest_filter="CUDANativeVNNIGemmPerf.Performance_AllFormats_AllShapes"
```

The command MUST use the harness's normal shape/format/M filters so Nsight sees
the candidate selected by production dispatch. A nonzero IMMA count proves the
compiled integer tensor-core path. The 128-bit instruction count and bytes per
sector diagnose vectorization and coalescing independently. Nsight replay time
is never canonical latency and must not enter dispatch labels.

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

Read `.agents/nativevnni-gemm-tuning/SKILL.md` for the shared corpus, learner,
certification, Git LFS, and installation workflow. This section owns only
CUDA-specific kernel/profiler constraints.

Do not land source-level "one shape gets this tile" overrides for CUDA NativeVNNI
unless the user explicitly asks for a temporary experiment. The durable path is:

1. Add or confirm the production shapes in
   `tests/v2/performance/kernels/cuda/gemm/CUDANativeVNNIGemmPerfCommon.h`.
2. For decode/GEMV dispatch tables, use the Git-LFS-aware turnkey transaction:
   `scripts/train_native_vnni_dispatch.sh --backend cuda --install`.
   The lower-level `refresh_native_vnni_dispatch_tables.sh` remains the phase
   executor used by the driver and for focused diagnostic collection.
   It first sweeps and freezes `Fast M=1`, then certifies grouped verifier
   `M=2..16,31` against that exact staged serial policy. Fifteen drafts require
   verifier M16 because the target pass includes one bonus row; M31 is a deeper
   sentinel, not a runtime maximum. Installation requires this complete matrix.
   The M1 broad sweep runs complete sample-interleaved candidate rounds. Failed
   development-CV cells are converted by
   `tests/v2/performance/kernels/native_vnni_dispatch/paired_requests.py` into
   typed, concrete candidate-pair
   manifests; the CUDA trainer batches those requests, the compiler absorbs the
   dimensionless tournament ratios, and CV repeats before sealed evaluation.
   The CUDA decode sweep trainer uses deterministic structurally-valid packed
   payloads, not per-element random quantization, so giant LM-head refreshes do
   not stall in host fixture generation. It still constructs the real tensor
   classes and production VRAM-pool preparation path.
   Policy fitting itself may use every CUDA and ROCm device through the exact
   leaf-primary scorer DSOs. The wrapper auto-discovers them; use
   `--policy-accelerators` and `--policy-lanes` to make ownership explicit.
   These launches score authenticated regret matrices, not candidate kernels,
   and cannot enter canonical timing or Nsight evidence. CUDA/HIP workers are
   process-isolated, and an explicitly requested accelerator failure is fatal.
   A production refresh also emits a provenance-bound profiler request for
   every physical candidate observation. Nsight Compute starts with collection
   disabled; packing, graph capture, route proof, correctness, and warmup remain
   outside profiling, and `cuProfilerStart` brackets one extra production
   launch per pass. Never run `ncu` around canonical timing or import
   replay-inflated profiler duration as the dispatch latency. Multi-kernel
   candidates retain one ordered evidence record per physical dispatch.
   Missing `ncu`, a required metric, or one supported candidate record blocks
   a production install. The final `cuda_profiler_features.csv` is generated
   only by authenticating the original canonical observation corpus against
   the request and evidence manifests; it contains untouched pipeline timing
   plus separate per-dispatch Nsight metrics.
   The default CUDA decode family set is `wide,kpar,direct`.  The perf harness
   uses the production VRAM-pool preparation path, which does not own ROWPAR's
   optional row-major auxiliary weight view; do not add `rowpar` back to the
   standard refresh unless the trainer has an explicit row-major-owner mode and
   model-level parity proves the generated table.
   If the full qwen36 inventory is too large for one pass, use the same staged
   profiles as ROCm: `--profile qwen36-core` for FFN/GDN projections and
   `--profile qwen-mtp-head` for the complete MTP-specific matrix inventory.
   The latter is not merely an LM-head alias: it resolves every release hidden
   width into both the hidden/embedding projection `H x 2H` and terminal
   `248320 x H` family, then covers M=1, grouped M=2..16, and the M31 sentinel
   across every supported source format. Do not install either staged artifact
   until the combined model-level parity and benchmark gates have passed.
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

4. Collect and install dense prefill exact overlays through the resumable
   `native_vnni_dispatch.production_dense_prefill_sweep` transaction. It owns
   the all-format Qwen geometry/M inventory, one process per physical GPU,
   per-candidate native-event timing sidecars, byte certificates, atomic cell
   publication, resume validation, and deterministic combination. CUDA
   candidates are only AUTO, direct BK64 output tiles, BK256, and the exact
   public-M=1 K-partition tree; an independently chosen reduction tree is not a
   launchable or benchmarkable candidate. Generate the include only after
   `combine` authenticates every planned cell, using
   `native_vnni_dispatch.dense_production_overlay`.
5. When debugging the decode trainer itself, the lower-level flow is
   `Perf__CUDANativeVNNIDecodeTrainer.cpp` ->
   `analyze_cuda_native_vnni_decode_trainer.py` -> the common compiler under
   `native_vnni_dispatch/`. The retired `infer_gemv_dispatch_heuristic.py` and
   `analyze_cuda_tc_gemv_dispatch.py` pipeline is not an authority and must not
   be resurrected. Public M1 may legitimately prefer WIDE, DIRECT, or an exact
   ordered KPAR schedule for the same `(N,K)` in eager versus captured mode.
   Grouped M2+ never chooses an independent schedule; it inherits the complete
   frozen M1 arithmetic identity and proves byte equality at every required M.
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
6. Validate generated artifacts with the transaction unit suite,
   `V2_Unit_CUDAGemvDispatchGeneratorAliases`, and
   `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`. For decode trainer changes,
   include `V2_Unit_GpuWorkspaceAllocationPolicy` so the explicit-stream
   trainer contract is checked. CUDA prefill installation additionally requires
   `V2_Integration_CUDA_NativeVNNI_PrefillBucketMInvariantAllFormats` and the
   backend GEMM source-policy gate; a timing winner without byte equality is an
   invalid corpus row, not an installable exception.

Example resumable CUDA prefill transaction:

```bash
# Preflight the complete compiler-resource surface before any long timing run.
CUDA_VISIBLE_DEVICES=0 \
LLAMINAR_TILE_RESOURCE_CSV=/tmp/cuda-dense-prefill-resources.csv \
build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm \
  --gtest_filter='CUDANativeVNNIGemmPerf.CompilerResources_AllCodebooksAndCandidates'

PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_vnni_dispatch.production_dense_prefill_sweep run \
  --backend cuda \
  --output-dir benchmark_results/native_vnni_dispatch/work/cuda-dense-prefill \
  --binary build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm \
  --devices 0,1

PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_vnni_dispatch.production_dense_prefill_sweep combine \
  --backend cuda \
  --output-dir benchmark_results/native_vnni_dispatch/work/cuda-dense-prefill

PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_vnni_dispatch.dense_production_overlay \
  --backend cuda \
  --work-dir benchmark_results/native_vnni_dispatch/work/cuda-dense-prefill \
  --output src/v2/kernels/cuda/gemm/CUDADenseProductionPrefillOverlayGenerated.inc \
  --summary-csv benchmark_results/native_vnni_dispatch/work/cuda-dense-prefill/overlay.csv
```

The resource preflight enumerates every runtime codebook, all six BK64 tiles,
direct and canonical reduction, and both Q4_0 BK256 arithmetic forms. Its exact
spill set is a regression. The scorer performs one untimed production-route
probe for AUTO, queries that exact primary/reducer symbol with
`cudaFuncGetAttributes`, and rejects nonzero local memory before recording any
timing event. Aggregate rows carry registers, static/dynamic shared memory,
actual threads, compiler thread bounds, and active blocks/SM; corpus validation
authenticates them. Never restore a spilling specialization to the candidate
inventory merely because an older timing row was fast.

After updating a checked-in generated include, rerun the focused CUDA GEMM route
regression for the affected shape and the relevant Qwen3.6 CUDA parity cells.

### MTP verifier dispatch mode

Grouped MTP verifier rows are stricter than ordinary fast decode: rows may be
published to live state, so every native FP32 output byte must equal rowwise
serial M1 decode. L2/cos/KLD/max-abs are diagnostics only. CUDA exposes this through
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
LLAMINAR_PERF_STATS_JSON=/tmp/cuda-profile.json \
LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1 \
./build_v2_release/llaminar2 benchmark -m M.gguf -d cuda:0

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
