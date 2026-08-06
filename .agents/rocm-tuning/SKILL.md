---
name: rocm-tuning
description: Profile and tune Llaminar V2 ROCm/HIP kernels on AMD Instinct GPUs (gfx906 MI50/MI60), including INT8 VNNI GEMM/GEMV, sampling, grouped verification, and graph-captured inference. Use rocprof per-dispatch timing/counters plus LLVM ISA analysis to diagnose occupancy, vectorization, VGPR pressure, spills, divergence, LDS conflicts, and throughput while retaining byte-exact production behavior.
---

# ROCm / HIP Kernel Profiling & Tuning (Llaminar V2, gfx906)

## Purpose

A repeatable, evidence-driven workflow for tuning Llaminar V2 HIP INT8 GEMM/GEMV kernels
on AMD Instinct MI50/MI60 (gfx906), distilled from the multi-session effort that produced
the V7 native kernel matching/beating AMD's Composable Kernel (CK) on N-heavy LLM shapes.

It chains three tools:

1. **PerfStats GPU event timing** — production-graph replay timing and structured route/host counters.
2. **`rocprof`** — per-dispatch GPU kernel timing (the *only* trustworthy latency number).
3. **LLVM ISA toolchain** (`llvm-objcopy` / `llvm-readelf` / `llvm-objdump`) — extract code
   objects, read VGPR/SGPR/LDS metadata, disassemble, and census instructions.
4. **The dispatch-comparison perf test + parity** — A/B variants and validate correctness.

The golden rule of gfx906 tuning: **wallclock lies.** On PCIe-bottlenecked topologies,
>90% of GEMM wallclock is memory transfer. Always quote `rocprof` per-dispatch GPU time,
not wallclock, when comparing kernels.

## Production hot-path invariants

ROCm kernel tuning must preserve the production execution architecture, not
trade architecture for an isolated microbenchmark win:

- **No dynamic allocation or deallocation.** Hot-path kernels and launch
  bridges consume persistent workspace bindings. `hipMalloc`,
  `hipMallocAsync`, `hipFree`, container growth, or hidden scratch allocation
  belongs in explicit initialization/workspace infrastructure only.
- **No host transfer.** Do not insert H2D/D2H copies, host-visible mirrors, or
  host polling into execution. Deliberate RAM/disk KV-cache tier movement is a
  transfer-service operation at an explicit lifecycle boundary, not kernel
  plumbing.
- **No stream or device synchronization.** `hipStreamSynchronize`,
  `hipDeviceSynchronize`, blocking copies, and event synchronization on the
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

The full reference write-up lives at
`src/v2/kernels/rocm/gemm/README.vnni-gemm-tuning.md` (and the companion
`README.native-vnni-isa-analysis.md` for GEMV). Read those for the V1-V7 history and the
CK ISA census; this skill is the operational checklist.

For whole-model diagnosis, first measure the production captured graph:

```bash
LLAMINAR_PERF_STATS_JSON=/tmp/rocm-profile.json \
LLAMINAR_PERF_STATS_SUMMARY=1 \
LLAMINAR_PERF_STATS_GPU_STAGE_TIMING=1 \
./build_v2_release/llaminar2 benchmark -m <model>.gguf -d rocm:0
```

The `stage_gpu` graph-replay rows are GPU-event measurements around the real
captured graph. They do not fabricate per-stage attribution inside a monolithic
graph; use rocprof for that dispatch-level view. `LLAMINAR_PROFILING=1` is
deprecated and exists only as a warning compatibility alias to these PerfStats
requests. It must never disable graph capture or select eager execution.

---

## Profiler attachment and evidence rules

- Pass `--no-mpi-bootstrap` only when profiling or debugging `llaminar2`
  directly so the profiler attaches to the compute process rather than the
  `mpirun` wrapper. Never use it for production or canonical benchmark timing;
  it disables the ordinary placement bootstrap.
- Do not pass `--no-mpi-bootstrap` to standalone test/performance binaries;
  they do not auto-bootstrap MPI.
- Preserve `LLAMINAR_*`, `HSA_*`, and `ROCR_*` variables with `sudo -E` only on
  hosts that require privilege escalation. Ordinary `sudo` strips them.
- Write rocprof CSVs, traces, extracted code objects, and disassembly under an
  explicit result directory outside the repository.
- Use one exact kernel/ISA/shape/M candidate per profiler launch. Treat
  rocprof timing and counters as diagnostic evidence beside the unprofiled
  canonical timing corpus; never train dispatch on profiler overhead or replay
  duration.
- Keep the profiler generation and metric groups required by the repository's
  evidence collector. Do not silently substitute `rocprofv3` output for a
  `rocprof`/rocprofiler schema expected by the trainer.

---

## Step 0: Architecture facts you must keep in mind (gfx906)

| Parameter | Value | Why it matters |
|-----------|-------|----------------|
| CUs | 60 | Grid sizing across CUs |
| SIMDs/CU | 4 | One wavefront (64 lanes) per SIMD |
| VGPRs/SIMD | 256 | **Determines occupancy** (waves/SIMD = floor(256/VGPRs-per-wave)) |
| LDS/CU | 64 KB | Shared across workgroups on a CU |
| L1 I-cache | 16 KB/CU | **Constrains loop-body ISA size** (branch bloat hurts) |
| L2 | 4 MB shared | Caches the A tile (128×K fits for K≤8192) |
| Wavefront | 64 threads | One instruction = 64 lanes |
| `v_dot4_i32_i8` | 1/cycle/lane | gfx906's VNNI (`vpdpbusd` equivalent) |

**Occupancy ladder:** 1-64 VGPRs → 4 waves; 65-84 → 3; 85-128 → **2 (the sweet spot for
M128×N128 INT8 GEMM, matches CK)**; 129-256 → 1 (catastrophic — no latency hiding).
Below 2 waves nothing else matters; going from 2 waves to "perfect scheduling" gave
**zero** measurable gain on this workload.

---

## Step 1: Build the dispatch-comparison benchmark

```bash
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_ROCM=ON
cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel
```

This binary instantiates **all native variants (V1-V7) and CK in one executable**, which
is exactly what makes comparative disassembly and A/B benchmarking fast.

---

## Step 2: Get the truth with `rocprof` (NOT wallclock)

```bash
rocprof --stats --timestamp on \
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*"

# Aggregate per-kernel GPU time
cat results.stats.csv | column -t -s,
# Per-dispatch rows are in results.csv
```

`results.stats.csv` gives the real GPU kernel time, excluding host overhead and PCIe.
This is the metric that exposed the "19% wallclock gap" as a mere **6.6% real kernel gap**
vs CK. Rank kernels by per-dispatch GPU time; that is your optimization target list.

> ⚠️ If two variants have nearly identical wallclock but you suspect a kernel difference,
> trust `rocprof`. Wallclock convergence (all V1-V6 hit ~8.72 ms) was pure PCIe transfer,
> not kernel behavior.

Force a specific variant with env vars when isolating:

| Variable | Effect |
|----------|--------|
| `LLAMINAR_ROCM_WIDE_TILE_V7=1` | Force V7 |
| `LLAMINAR_ROCM_WIDE_TILE_V6=1` … `_V2=1` | Force V6 … V2 |

### Isolate one production-shaped launch

Whole-graph PerfStats identifies the expensive captured fragment. Attribute
individual kernels with a standalone performance harness that invokes the same
backend entrypoint, geometry, persistent scratch contract, and tensor format as
production. Never disable graph capture or select eager inference merely to
make a profiler attach: that measures a different engine. Some rocprofiler
versions cannot trace retained HIP graphs reliably; retain the outer graph
event measurement and profile the isolated real kernel instead.

Keep canonical timing unprofiled, then gather timing, counters, and ISA in a
separate deterministic profiler invocation for that exact candidate. Read
[`references/rocprof-isolated-kernel.md`](references/rocprof-isolated-kernel.md)
for the tested rocprof v1 dispatch-range/counter commands, metric
interpretation, wave64 vectorization checks, and ROCm 7 `.hip_fatbin`
extraction workflow.

---

## Step 3: ISA deep-dive (occupancy + scheduling)

`rocprof` tells you *which* kernel is slow; the LLVM toolchain tells you *why*. HIP
executables embed gfx906 code objects as ELF bundle sections.

```bash
BINARY=build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison

# 1. Extract the gfx906 code object
llvm-objcopy \
  --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' \
  "$BINARY" /dev/null

# 2. Read per-kernel resource metadata (VGPR / SGPR / LDS → occupancy)
llvm-readelf --notes /tmp/co.elf 2>/dev/null | awk '
  /\.name:/{name=$2} /\.vgpr_count:/{vgpr=$2} /\.sgpr_count:/{sgpr=$2}
  /\.group_segment_fixed_size:/{lds=$2}
  /\.wavefront_size:/{printf "%-60s VGPR=%-4s SGPR=%-4s LDS=%s\n",substr(name,1,60),vgpr,sgpr,lds}'

# 3. Full disassembly (~45k lines for the full binary)
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/disasm.txt

# 4. Locate kernels (CK symbols are >1000 chars due to C++ templates)
grep -n '<_Z' /tmp/disasm.txt | head -40
```

For ROCm 7 objects that use `.hip_fatbin`, use the extraction workflow in the
linked isolated-kernel reference. The unprofiled benchmark, rocprof evidence,
and static ISA must always describe the same compiled candidate.

### Read the metadata FIRST

The VGPR/SGPR/LDS line decides occupancy before you read a single instruction:

| Symptom in metadata | Diagnosis | Fix |
|---------------------|-----------|-----|
| VGPR ≥ 129 (1 wave) | register explosion (e.g. V4: 256 VGPRs + 118 spills) | add `__launch_bounds__(256, 2)`; stop full-unrolling the inner loop |
| VGPR 85-128 (2 waves) | the gfx906 sweet spot — good | only algorithmic/structural wins left |
| Spills > 0 | inner loop unrolled with too many live accumulators | `#pragma clang loop unroll(disable)` or `unroll_count(2)` |
| LDS > 32 KB | limits workgroups/CU | shrink KT or move A to L2 instead of LDS |

### Locate a kernel by distinctive symbol fragment

| Grep pattern | Kernel |
|--------------|--------|
| `qgemm_int8_vnni_wide_tile_v5` | native V5 |
| `Li256E.*Lb1ELb0E.*PassThrough` | CK 128×128, HasMainK=true, no padding (hot path) |
| `Li64E` / `Li32E` | CK 64×64 / 32×32 (medium / decode) |

### Instruction census within a kernel's line range

```bash
START=18369; END=22092  # from the grep above
for pat in v_dot4_i32_i8 ds_read_b128 ds_read_b32 ds_write s_barrier \
           s_waitcnt s_nop s_cbranch buffer_load s_swappc; do
  printf "%s: " "$pat"
  awk "NR>=$START && NR<=$END" /tmp/disasm.txt | grep -c "$pat"
done
```

What to read for:
- `s_barrier` count → loop structure. Double-buffered (CK, V3-V7) ≈ 3 barriers;
  single-buffered (V1) = 2 per iteration.
- `s_cbranch` count → boundary-check bloat. High branch count in the hot loop wastes
  the 16 KB L1 I-cache (V5 had 15 fast-path + 84 boundary-path = 99 branches/iter).
- `s_nop` → wasted cycles (fewer is better; CK had 2 in the entire kernel).
- `s_swappc` → subroutine calls (only V1; inline everything in the hot loop).
- `s_waitcnt lgkmcnt(N)` with N>0 between `ds_read`/`v_dot4` → **progressive waits =
  good pipelining** (CK keeps ~5 LDS reads in flight at all times).

---

## Step 4: The seven latency-hiding levers (in priority order)

Apply these to the targeted kernel; the first two are mandatory, the rest are refinements.

1. **`__launch_bounds__(threads, min_waves)` — non-negotiable.** The 2nd arg caps VGPRs
   (256/min_waves). Omitting it let V4 use 256 VGPRs + spills → 1 wave → slower than V1.
   1→2 waves is often a 3-5× swing, not 2×.
2. **LDS double-buffering.** Ping-pong even/odd buffers: compute buf[cur] while loading
   buf[nxt], one `__syncthreads()` between. Kills the serial load→barrier→compute cycle.
3. **Inner-loop unroll control.** `#pragma unroll` → VGPR explosion (V4). Prefer
   `unroll(disable)` (V5: clean 117 VGPRs) or `unroll_count(2)` (V6). On gfx906 these two
   gave **identical** performance — don't over-invest here once you have 2 waves.
4. **Progressive `s_waitcnt`** — let the compiler keep 5-6 LDS reads in flight; don't force
   `lgkmcnt(0)`.
5. **Register recycling** — reuse a `v_dot4` source VGPR as an accumulator/load target once
   dead; this is how CK fits 64 accumulators in 128 VGPRs.
6. **16-byte vectorized load/store** — `uint4` reinterpret_cast → `global_load_dwordx4` /
   `ds_write_b128` / `ds_read_b128`. 4× scalar throughput.
7. **Safe-tile split (the V7 winner)** — split the tile loop into a *safe* phase (loads
   provably in-bounds → unconditional vectorized loads, zero branches) and a *boundary*
   phase (full checks + scalar fallback, runs 0-2 tiles or never for aligned shapes). For
   aligned LLM shapes (e.g. N=18944 % 128 == 0) the boundary loop is dead code and the hot
   loop's ISA is maximally tight in the 16 KB I-cache. This was the structural change that
   closed the last 6.6% — **where you branch matters more than how you compute.**

---

## Step 5: Validate (correctness + the right benchmark)

```bash
# Re-run the dispatch comparison and confirm rocprof GPU time improved
rocprof --stats --timestamp on \
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*"
cat results.stats.csv | column -t -s,

# Full-model ROCm parity must stay PASS
ctest --test-dir build_v2_integration -R ".*Parity.*ROCM" --output-on-failure
```

Accept a change only if **`rocprof` per-dispatch GPU time** improves beyond noise AND
parity stays PASS. A wallclock-only improvement on a PCIe-bound rig is not evidence.

Remember shape-dependence: V7/KT16 wins N-heavy shapes (FFN_Up, Gate, LM_Head), but
K-heavy shapes (FFN_Down, AttnOut) still favor V1/V3's simpler/tall-K geometry. Tune and
select per shape; don't assume one kernel wins everywhere.

---

## Step 6: Train generated NativeVNNI dispatch tables

Read `.agents/nativevnni-gemm-tuning/SKILL.md` for the shared corpus, learner,
certification, Git LFS, and installation workflow. This section owns only
ROCm-specific kernel/profiler constraints.

ROCm NativeVNNI dispatch should follow the same automatic sweep/generate/validate
pipeline as CUDA. Avoid hand-coded per-shape overrides except as throwaway
experiments.

1. Keep the shape inventory in the perf harness aligned with real model routes:
   `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Sweep.cpp` for prefill
   and `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Throughput.cpp` for
   decode/GEMV.
2. For decode/GEMV dispatch tables, use the Git-LFS-aware turnkey transaction:
   `scripts/train_native_vnni_dispatch.sh --backend rocm --install`.
   The lower-level `refresh_native_vnni_dispatch_tables.sh` remains the phase
   executor used by the driver and for focused diagnostic collection.
   It runs with `LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1`, sweeps canonical
   serial `M=1` first and grouped verifier `M=2..16,31` against that exact frozen
   dependency, generates the include, validates it, and can install it with
   `--install`. M31 is a deeper sentinel rather than a production maximum.
   Keep public M1 decisions keyed by every runtime-visible discriminator.
   Grouped verifier buckets must inherit the frozen M1 K partition and ordered
   reduction tree; selecting a different M2+ family is incompatible with the
   byte-exact serial-row contract.
   The durable decode selector must generalize by aspect ratio plus work-size
   segments. Exact `(M,N,K)` winners are allowed only as overlays above that
   broad fallback; do not land an exact-shape-only table that would require
   retraining for every nearby model dimension.
   Score generated tables on the runtime dispatch surface, not raw
   source-format rows. If several GGUF formats share a NativeVNNI codebook, the
   runtime can only choose one `(codebook,M,N,K)` tuning; analyzers should
   collapse aliases before enforcing exact-hit thresholds.
   Use staged strict profiles when the full shape inventory is too expensive in
   one pass: `--profile qwen36-core` covers FFN/GDN projections and
   `--profile qwen36-lm-head` covers the giant LM-head shape. The LM-head
   profile defaults to `LLAMINAR_ROCM_NVNNI_DECODE_REFERENCE=native-auto`, which
   compares candidates with a reset-AUTO native output instead of building a
   multi-GB FP32 hipBLAS mirror. Treat that as a dispatch-equivalence trainer
   proxy; model parity and benchmarks still gate any `--install`.
   Production collection derives immutable profiler requests only after the
   canonical timing CSV and raw timing sidecar are sealed. Each request starts
   a fresh process. `roctxProfilerResume(0)`/`roctxProfilerPause(0)` bracket one
   extra production launch, while a request-specific ROCTx range identifies the
   exact pipeline in PMC passes. The trace pass keeps physical kernel names;
   renamed PMC rows join back by contiguous dispatch order. On gfx906, collect
   `FetchSize`, `WriteSize`, `Wavefronts`, `VALUUtilization`, and
   `LDSBankConflict` as reviewed singleton passes. Combining the TCC traffic
   metrics exceeds hardware profile capabilities and can hang rocprofiler's
   abort path. Never let profiler replay or overhead enter canonical timing.
   The final `rocm_profiler_features.csv` must be the authenticated export of
   the canonical observation, request, and evidence manifests, with one row per
   physical dispatch rather than an ad hoc join of rocprof CSVs.
   The common policy fit may separately use every ROCm and CUDA device through
   exact leaf-primary scorer DSOs. The refresh wrapper auto-discovers devices;
   `--policy-accelerators` fixes the inventory and `--policy-lanes` controls
   independent CPU orchestration lanes per device. Those scorer launches do not
   time candidate kernels and never enter rocprof evidence. Keep CUDA and HIP
   in separate worker processes and hard-fail any requested accelerator error.
3. Use `--profile family-smoke` for a bounded representative training pass before
   a full acceptance refresh. This profile is stratified by format: it runs one
   small sweep per codebook/family, writes per-format partial CSVs, combines them,
   then runs the normal train/generate/validate flow. This avoids a capped sweep
   accidentally covering only the first format in the list.

   ```bash
   scripts/refresh_native_vnni_dispatch_tables.sh --backend rocm \
     --profile family-smoke \
     --rocm-formats Q4_0,IQ4_XS \
     --m-values 1,2
   ```

   `family-smoke` is a workflow/proxy gate, not production acceptance. Do not
   replace broad checked-in tables from this profile alone; run `qwen36` or
   `all`, then model-level parity and benchmark gates, before `--install`.
   Treat the trainer's FP32 cosine as a health filter, not the exact dispatch
   oracle: packed K-quant formats such as Q4_K/Q2_K can sit just below the
   generic FP32 gate while still matching the native packed contract. Keep
   `V2_Integration_ROCm_NativeVNNI_GEMV`,
   `V2_Integration_ROCmQuantisedGemmSmallM`, and model parity as the promotion
   gates.
4. If you are debugging the ROCm trainer directly, run sweeps with
   `LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1`. This is important:
   AUTO normally consumes the checked-in generated tables, so training without
   this guard can benchmark the previous table and emit no explicit winner.
5. Validate CSVs with
   `tests/v2/performance/kernels/rocm/validate_rocm_native_vnni_trainer_csv.py`.
   It checks decode/prefill schema, shared codebook ids, and canonical prefill
   bucket policy from `src/v2/utils/PrefillGraphBucketDefaults.h`.
6. Generate prefill tables with
   `tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_trainer.py` and
   decode tables with
   `tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py`.
   The checked-in outputs are
   `src/v2/kernels/rocm/gemm/ROCmNativeVNNIPrefillDispatchGenerated.inc` and
   `src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc`.
7. Validate generated codebook references with
   `tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py`
   and run the focused units:
   `V2_Unit_ROCmNativeVNNITrainerCsvValidator`,
   `V2_Unit_ROCmNativeVNNITrainerGenerator`,
   `V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator`, and
   `V2_Unit_NativeVNNIGeneratedDispatchCodebooks`.

### MTP verifier dispatch mode

ROCm grouped verifier kernels can be fast only if their generated small-M policy
also matches rowwise serial decode. The shared convention is
`ITensorGemm::beginVerifierDecodeEquivalentScope()`: stages keep the returned
RAII scope alive while running serial verifier replay or grouped verifier
publication. ROCm implements the scope by selecting the M=1 decode-equivalent
NativeVNNI policy. Do not call ROCm mode toggles directly from stage code, and
do not hide unsupported verifier modes behind quiet rowwise fallbacks.
For shared-expert MoE verifier rows, compare grouped publication against the
canonical M=1 GEMM/SwiGLU/down replay oracle unless a grouped decode shortcut
has already passed the same strict all-codebook gates.

Compact refresh example:

```bash
LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1 \
LLAMINAR_ROCM_NVNNI_DECODE_FORMATS=Q4_K \
LLAMINAR_ROCM_NVNNI_DECODE_SHAPES=Qwen36_GDN_TimeProjection \
LLAMINAR_ROCM_NVNNI_DECODE_CSV=/tmp/decode.csv \
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
./build_v2_release/tests/v2/v2_perf_native_vnni_throughput \
  --gtest_filter='NativeVNNIPerfTest.TrainerCsv_CodebookTagged'

LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1 \
LLAMINAR_ROCM_NVNNI_SWEEP_FORMATS=Q4_K \
LLAMINAR_ROCM_NVNNI_SWEEP_SHAPES=Qwen36_GDN_TimeProjection \
LLAMINAR_ROCM_NVNNI_SWEEP_M=600 \
LLAMINAR_ROCM_NVNNI_SWEEP_CSV=/tmp/prefill.csv \
HSA_OVERRIDE_GFX_VERSION=9.0.6 \
./build_v2_release/tests/v2/v2_perf_native_vnni_sweep \
  --gtest_filter='NativeVNNISweepTest.TrainerCsv_CodebookTagged'

python3 tests/v2/performance/kernels/rocm/validate_rocm_native_vnni_trainer_csv.py \
  --require-policy-prefill-m --require-prefill-m 600 /tmp/prefill.csv /tmp/decode.csv
python3 tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_trainer.py \
  --input /tmp/prefill.csv --output /tmp/prefill.inc --summary /tmp/prefill.summary
python3 tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py \
  --input /tmp/decode.csv --output /tmp/decode.inc --summary /tmp/decode.summary
python3 tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py \
  /tmp/prefill.inc /tmp/decode.inc
```

After updating checked-in tables, rerun focused ROCm coverage:
`V2_Integration_ROCm_NativeVNNI_GEMM` and
`V2_Integration_ROCm_NativeVNNI_GEMV`.

---

## Key lessons (hard-won)

1. **Profile before optimizing.** `rocprof` revealed the real 6.6% gap hidden under a
   misleading 19% wallclock gap — without it you optimize the wrong thing for days.
2. **Occupancy is table stakes, not a differentiator.** Below 2 waves nothing matters;
   above it, scheduling perfection bought nothing here.
3. **The compiler is very good.** `unroll(disable)` got within 6.6% of hand-tuned CK;
   forcing cross-iteration visibility (V6) added nothing.
4. **CK's real edge is compile-time specialization** (`HasMainK`/`HasDoubleTail`/
   `MNPadding` variants → zero boundary code in the hot path). V7 approximates this at
   runtime with safe-tile splitting.
5. **Wallclock on PCIe-bottlenecked systems is misleading** — always use `rocprof`.

---

## Quick reference — one-liners

```bash
# Build the all-variants benchmark
cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel

# Truth: per-dispatch GPU kernel time
rocprof --stats --timestamp on \
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*" && cat results.stats.csv | column -t -s,

# ISA: extract → metadata (occupancy) → disassemble → find kernels
BINARY=build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison
llvm-objcopy --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' "$BINARY" /dev/null
llvm-readelf --notes /tmp/co.elf 2>/dev/null | awk '/\.name:/{n=$2}/\.vgpr_count:/{v=$2}/\.sgpr_count:/{s=$2}/\.group_segment_fixed_size:/{l=$2}/\.wavefront_size:/{printf "%-60s VGPR=%-4s SGPR=%-4s LDS=%s\n",substr(n,1,60),v,s,l}'
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/disasm.txt
grep -n '<_Z' /tmp/disasm.txt | head -40

# Force a variant
LLAMINAR_ROCM_WIDE_TILE_V7=1 ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison --gtest_filter="*WideTileVariantComparison*"

# Parity gate
ctest --test-dir build_v2_integration -R ".*Parity.*ROCM" --output-on-failure
```
