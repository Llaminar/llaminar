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

Remaining follow-ups are unchanged: default-engine flip after isolated
`perf` measurements on true AMX hardware (this VM cannot execute tile data
instructions), `PerfStats` counter registration, `M=1` native GEMV,
fused-SwiGLU/LM-head gate sites, codegen-witness registration, image matrix
and README synchronization.
