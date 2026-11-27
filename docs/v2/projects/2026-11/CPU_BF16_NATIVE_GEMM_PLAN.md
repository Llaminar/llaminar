# Native CPU BF16 GEMM: AMX-BF16 and AVX512-BF16 implementation plan

Date: 2026-11. Branch: `feat/cpu-amx-bf16-gemm` (base
`origin/feat/native-quantised-file-density-storage` @ `2348e483f`).
Host qualification target: Intel Sapphire Rapids (Xeon Gold 6530,
AVX512F/BW/DQ/VL/VNNI + AVX512-BF16 + AMX-BF16/INT8).

This is the **design and first-slice plan**, not a performance or correctness
certificate. Native BF16 GEMM replaces a library route (oneDNN) and therefore
needs the same evidence discipline as every other production kernel: exact
contract, typed interface, focused regressions, PerfStats route evidence and
isolated profiling before any default flip.

## Why

The CPU BF16 homogeneous route (`BF16 × BF16 → FP32`) today is:

- `M == 1`: `run_bf16_skinny_matmul()` — serial decode-exact scalar FMA loop.
- `M > 1`: `run_onednn_bf16_matmul()` — oneDNN library matmul; the library picks
  its own internal reduction order and implementation per shape.

Sapphire Rapids-class hosts carry two first-class hardware dot-product engines
that the engine does not use:

- **AMX-BF16** (`_tile_dpbf16ps`, 16×16 tiles, one instruction per 16×16×16
  tile product, FP32 accumulation) — the largest GEMM throughput on this host.
- **AVX512-BF16** (`_mm512_dpbf16_ps`, 2 BF16 products per FP32 lane, FP32
  accumulation) — Cooper Lake+, including all AVX512-VNNI Xeons.

Both are deterministic hardware reductions with a fixed lane/partition
structure, so a native route can publish a single fixed contract per route
instead of relying on library reproducibility.

## Scope of this slice

1. **Build envelope**: `LLAMINAR_CPU_ISA=AVX512BF16` and `AMXBF16` x86 values in
   `cmake/CPUInstructionSet.cmake`, with configure-time compiler probes (the
   NEON_I8MM pattern) and startup admission when the compiled envelope requires
   a hardware extension the machine lacks.
2. **Native kernels** (`src/v2/kernels/cpu/gemm/CPUBF16Gemm.h`):
   - AVX512-BF16 GEMM: K ascending in 32-element chunks; lane `i` accumulates
     pair `(2i, 2i+1)` of each chunk; final fixed pairwise 16→1 reduction; then
     the shared alpha/beta/bias-free epilogue rounding contract.
   - AMX-BF16 GEMM: 16×16 row/col tiles, K in 16-element tiles, palette-1 tile
     config, per-thread `tileloadconfig`/`tilerelease` RAII scope.
   - Both routes are **route-deterministic** (fixed partitions, fixed reduction
     tree, byte-identical across thread counts): they are not serial-fma
     byte-identical (hardware fused pair/tile sums), so serial decode rows keep
     their exact paths.
3. **Route policy**: `LLAMINAR_CPU_BF16_GEMM_ENGINE=auto|library|avx512bf16|amxbf16`.
   Automatic dispatch stays on the existing library/serial routes until the
   native routes are certified and measured; explicit engine selection throws
   (never silently degrades) when the compiled envelope, the hardware or the
   geometry does not admit the requested engine.
4. **Gate wiring**: BF16 homogeneous `multiply_tensor` branches prefer the
   automatic native route when it resolves; the library route remains the
   fall-through only when automatic resolution selects `Library` (the current
   production route — not a hidden substitute).
5. **Evidence**: atomic per-route invocation counters with an opt-in probe
   (the FMLAL-leaf pattern), an Integration suite with determinism/thread
   invariance, tolerance vs the serial oracle, route-probe discrimination and
   explicit-selection rejection.

## Explicit boundaries (deliberate, not fallbacks)

- `M == 1` automatic policy resolves `Library`: decode rows already own
  byte-exact serial arithmetic (`run_bf16_skinny_matmul`,
  `run_fp32xbf16_skinny_matmul`). Native routes can be forced explicitly.
- `GPUAlignedExpert` BF16/FP16 experts keep the exact cross-tier reduction
  (`run_gpu_aligned_expert_fp32_matmul`, FMLAL/AVX-512 lanes) — native BF16
  GEMM never replaces placement-invariant arithmetic.
- AVX512 engine admits `transpose_B=true` only (weights `[N,K]`); AMX engine
  admits both layouts through strided tile loads.
- Geometry gates are explicit: AVX512 requires `K % 32 == 0`; AMX requires
  `M % 16 == 0 && N % 16 == 0 && K % 16 == 0`. Non-admitted geometry with an
  explicit engine raises `std::invalid_argument`.
- The new envelope values are opt-in images (`AVX512BF16`, `AMXBF16`); the
  default `AVX512`/`AVX2` images contain no BF16 instructions and are
  unchanged.

## Follow-ups (not in this slice)

- Flip the automatic default after certification and isolated profiling
  (Release, AMX vs AVX512-BF16 vs oneDNN on the same shapes);
- fused-SwiGLU / LM-head / explicit-dimension BF16 gate sites and `M=1` native
  GEMV with a documented non-serial contract where allowed;
- registered `PerfStats` counters and the `cpu-tuning` skill evidence for the
  new kernels (occupancy is N/A; report IPC/throughput and memory traffic);
- image matrix / README / `AGENTS.md` prose and the x86 SIMD inventory refresh
  in `ARM_CPU_X86_SIMD_INVENTORY.md`;
- byte-determinism proof between AMX and AVX512-BF16 once hardware confirms the
  pair-sum rounding structure (a machine probe is required before claiming it).

## Validation plan (on this host)

1. Configure `AMXBF16` and `AVX512BF16` builds (`Integration` with snapshots).
2. Run the new `V2_Integration_CPU_BF16GemmNativeRoutes` suite on the host,
   then the complete Unit + `ProductionTestPreflight` gate under the new
   envelope.
3. `perf` the isolated AMX and AVX512-BF16 kernels (one candidate per launch)
   and record IPC / cache evidence in the plan's sibling evidence doc.
4. Re-run serial-byte-exact CPU suites (floating skinny, MTP verifier, native
   quantised) to prove the default route and exact lanes are untouched.

## Validation evidence (2026-11, Intel Sapphire Rapids-class VM)

Build trees (Integration, `AMXBF16` and `AVX512BF16`, ccache):

| Check | Result |
|---|---|
| Configure probes (AVX512-BF16, AMX tile) with GCC 15.3 | PASS |
| `V2_Integration_CPU_BF16GemmNativeRoutes` on `AVX512BF16` tree | 6/6 PASS |
| Same suite on `AMXBF16` tree | 6 PASS + 3 AMX tests SKIP |

Key findings folded into the code:

- **AMX TILECFG layout**: the 64-byte tilecfg image is header (16 bytes) +
  16 uint16 col words at offset 16 + 16 uint8 row counts at offset 48 (the
  oneDNN packed layout). The earlier paired {rows, colsb} descriptor table
  caused #GP in `ldtilecfg`. Fixed with a `static_assert(sizeof == 64)`.
- **Functional AMX admission**: this VM backfills the AMX feature leaf and
  executes `ldtilecfg`/`tilerelease` but the tile data opcodes
  (`tilezero`/`tileloadd`/`tdpbf16ps`/`tilestored`) raise #UD. CPUID+XCR0
  alone admitted it; a one-time functional probe (real `tilezero` under a
  temporary SIGILL handler) now gates the AMXBF16 envelope, the BF16 engine
  admission and the AMX tests, so the AMXBF16 binary fails fast on such
  hosts while AVX512-BF16 remains first-class.
- **GCC 15 base-branch defect**: `nativeQuantisedFP16ScaleToFP32` fell off
  the end for exponent-31 words under `__F16C__` (undefined NaN/Inf scales);
  fixed with the scalar bit expansion and an exhaustive 65,536-word
  regression in `Test__CPUCodedCompactUnits`.
- **Environment**: the box needed user-space OpenMPI 4.1.6 compiled from
  source with `--with-devel-headers` (conda's package strips the ORTE
  private headers `orte_config.h` etc. that `NativeMPISessionSchizo.cpp`
  needs), ccache, and a raised memory limit; all recorded in
  `README-BUILD-BF16.md` on the validation worktree.

## Microbenchmark evidence (2026-11, AVX512BF16 tree, this VM)

Same-buffer comparison, oneDNN BF16×BF16→FP32 (`run_onednn_bf16_matmul`
mirror, the production library path) vs the native AVX512-BF16 route,
8 worker threads, median of 30-40 reps, `nice 15`, pinned CPUs:

| M | N | K | oneDNN | native AVX512-BF16 | native / oneDNN |
|---|---|---|---|---|---|
| 16 | 4096 | 4096 | 0.266–0.444 ms (1.3–2.0 TF/s)* | 0.573 ms (0.94 TF/s) | 1.3–2.1×* |
| 32 | 4096 | 4096 | 0.338–0.535 ms (2.0–3.2 TF/s) | 1.121 ms (0.96 TF/s) | 2.1–3.3× |
| 96 | 4096 | 4096 | 0.777–1.359 ms (2.4–4.1 TF/s) | 3.333 ms (0.97 TF/s) | 2.4–4.2× |
| 32 | 8192 | 8192 | 1.401–2.266 ms (1.9–3.1 TF/s) | 4.729 ms (0.91 TF/s) | 2.1–3.3× |

*Run-to-run ranges reflect the shared node (production vLLM on the host).

### Kernel bring-up history within this slice

- Intrinsic 4×4 (committed fce10b6ed): 0.57 TF/s, load-issue limited
  (~0.45 vdpbf16/cycle per thread).
- Hand-scheduled 4×4 software pipeline (`avx512Bf16TileKernel`): 0.96 TF/s
  (1.65×). One chunk of latency overlap, exact 32-ZMM allocation
  (16 accumulators + 2×8 double-buffer; callee-saved zmm16-31 wrapped),
  countdown loop with byte-unit counter (the element-vs-byte unit bug was
  the reason the first asm build seemed to hit 1.9-2.0 TF/s — it silently
  processed only half the chunks), base+offset addressing, accumulator
  publish + restore inside the same asm block.
- Correctness: byte-exact vs the documented lane-contract oracle for the
  K sweep (64..4096) and tail shapes (16x24x128, 9x17x320); the
  `V2_Integration_CPU_BF16GemmNativeRoutes` suite stays 6/6.

### Remaining gap analysis (next work package)

**Exhaustive structural verification (continuation #2)**: six structurally
distinct designs now all plateau at 0.85-0.96 TF/s on this box, while
oneDNN's memory-fed kernel runs at ~2 vdpbf16/cycle under the same
measurement conditions:

| Design | Result |
|---|---|
| intrinsic 4×4 + cache panels (committed fce10b6ed) | 0.57 TF/s |
| asm software-pipelined 4×4 (committed 204bd2cce/a477e8973) | **0.96 TF/s** |
| spill tiles (8×4, 4×8) | 0.70-0.80 TF/s |
| register-broadcast packed 16×24 | 0.93 TF/s |
| .S hardware-broadcast packed 16×24 (Intel-syntax `{1to16}`) | 0.85 TF/s |
| .S oneDNN-shape broadcast-reuse 32×12 (0.67 load-slots/FMA) | 0.85-0.91 TF/s |

oneDNN's reference microkernel (jit_avx512_core_gemm_bf16bf16f32_kern.cpp)
uses the same instruction family with an equivalent load:FMA ratio
(~0.58 load-slots/FMA with a 96-row×8-col tile and broadcast B), which
means the doable source/asm designs are architecturally equivalent to it
and still 2-4× slower. No source-level design reached its per-core FMA
rate; the remaining delta is the JIT's register-exact scheduling and
prefetch structure, which a static .S file also failed to reproduce (the
hardware-broadcast form is accepted by binutils 2.46.1 in Intel syntax,
verified, and still plateaus).

Per-thread profile for the record: 0.96 TF/s = 0.47 vdpbf16/cycle; the
pure-FMA chain (no loads) measures 1.85/cycle (1 thread) and every
memory-fed loop — regardless of shape, packing, broadcast form or
pipelining — measures 0.45-0.48. Conservative next steps for a future
engineer with a JIT-capable toolchain, in order:

1. **4×6 microtile** (24 accumulators + 4 A regs + 4 B regs, two B chunks
   as memory operands with a dedicated base pointer; 30-32 ZMM): 24 FMA
   per phase (~12 cycles) improves the overlap and the load:FMA ratio to
   1:2.4. Measured with register broadcast: 0.72-0.83 TF/s (LSU pressure
   from replicated memory operands).
2. **3×8 oneDNN-style tile with register-broadcast B** — implemented as
   the 32×12 .S variant (24 C + 2 A + 1 B): 0.85-0.91 TF/s.
3. Only then: automatic default flip after isolation benchmarks.

**Conclusion reached from the evidence**: the achievable ceiling for the
native route with source/static-asm toolchains on this host is ~0.96 TF/s;
reaching oneDNN's 2-4 TF/s requires its JIT scheduler (or AMX hardware,
unavailable appliance-wide). The library oneDNN route remains the
production BF16 GEMM engine; the native route stays available with its
fixed deterministic contract and value as the AMX carrier.
## Remaining follow-ups

Default-engine flip after isolated
`perf` measurements on true AMX hardware (this VM cannot execute tile data
instructions), `PerfStats` counter registration, `M=1` native GEMV,
fused-SwiGLU/LM-head gate sites, codegen-witness registration, image matrix
and README synchronization.

## Remaining follow-ups

Default-engine flip after isolated
`perf` measurements on true AMX hardware (this VM cannot execute tile data
instructions), `PerfStats` counter registration, `M=1` native GEMV,
fused-SwiGLU/LM-head gate sites, codegen-witness registration, image matrix
and README synchronization.
