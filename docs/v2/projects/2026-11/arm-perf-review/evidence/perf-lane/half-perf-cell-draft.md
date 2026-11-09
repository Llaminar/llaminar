# Draft: independently owned production CPU half-operand perf cell

2026-11-09. Base authenticated: 777f7a52a. Plan only; no source/CMake edits.
Replaces the earlier proposed modification of Perf__CPUFloatingMoEExpert
(ownership outside this lane; R2 confirms its panel tournament is Q8-only).

## Owner / registration
- New source: tests/v2/performance/kernels/cpu/gemm/Perf__CPUHalfOperand
  Projection.cpp (proposed; not created).
- New target: v2_perf_cpu_half_operand_projection.
- CTest smoke: V2_Perf_CPUHalfOperandProjectionSmoke, labeled
  performance/model-free CPU (not Unit); add_v2_test (normal MPI retirement).
- Distinct from the recorded-input correctness replay in Preflight.
- Do NOT modify Perf__CPUFloatingMoEExpert; reuse stable measurement
  conventions. No model downloads / GPU runtime / private asm microkernel /
  intrinsic-only proxy.

## Production route / request identity
- Instantiate the ACTUAL engine:
  gemm::FloatingPointGemmKernel(weight, NumericalPolicy::BackendNative,
      FloatingActivationOperandPrecision::FP16RNE) (proven by
  Test__CPUHalfOperands.cpp).
- Primary timed route: multiply_tensor (real M/N/K, FP32 output); grouped:
  multiply_fused_verifier_rows_decode_equivalent (real bundle/row machinery);
  physical-half A = separate input-storage cell. Include completion
  (alpha/beta/bias), conversion + fixed-tree reduction — not dotRows alone.
- A full gate/up/down MoE cycle is NOT the alpha/beta proxy (unrelated work,
  different operand policy).
- ONE process = ONE exact candidate/config (M,N,K, storage, entrypoint,
  workers, seed/input identity, samples/warmup/repeats, candidate/build
  identity). No sweep in profiler. Baseline/candidate = separate fresh
  processes, rotating order. Baseline runs ONLY the current production impl
  (no second scalar fallback created/preserved). Later candidate = compiled
  explicit impl; unavailable selection fails precisely; comparing baseline vs
  candidate binaries is valid.

## Oracle standard (before timing)
1. Independent binary16-RNE activation oracle + original half weights (NOT
   the candidate's conversion). Immutable expected fixtures (bit patterns,
   dyadic inputs) + nontrivial recorded-input fixture where available.
2. Reference every output with canonical 256 lanes: FMA at k mod 256,
   stride256 updates, add tree 128->1, ordered completion; raw FP32 bytes.
3. Baseline/scalar results = additional check only. Grouped rows = independent
   serial M1. Verify untouched output capacity, input/weight immutability,
   dirty-buffer overwrite.
4. FP32-A reference = equality oracle ONLY for fixtures proven half-
   representable under the contract; else compare against its own rounded-half
   contract oracle + discriminating NON-half values proving the policy ran.
   No FP32-equality-everywhere demand; no tolerance.
5. Reject byte mismatches before timing. No GTEST_SKIP / seed retry /
   fallback / clipping / tolerance. Benchmark pass != correctness gate.

## Timing / evidence
- Prepare once (addresses + operand policy retained); no per-sample
  generation/conversion bank/alloc/team rebinding; production conversion
  timed. Warm same geometry; explicit repeat counts; bounded workers;
  profiler-free canonical records.
- Emit samples + median/CV/quantiles + hashes + compiler/source/binary
  identity + compiled envelope/runtime route + worker count + M/N/K + policy
  + storage + entrypoint + live extents + checksum. Counter cycles only when
  permitted; never label ticks as PMU cycles.
- Separate one-candidate profiler process: intended leaf counts, scalar
  conversion elimination, no K-loop spills, cycles/instructions/cache. No
  unrelated kernels in the interval.
- .80 authoritative for economy. First baselines: admitted real GDN
  alpha/beta source shapes from canonical definitions/fixtures + K255/256/257
  diagnostics (no invented shape table). One worker then physical-core
  scaling. No GPU/GLM touch.
- PerfStats route assert if exposed; else typed route witness/counter with
  the leaf. Never infer route from output/compiler flags alone.

## Checkpoint / holds
- BEFORE test-only source: reviewer confirms constructor/public entry
  ownership, independent oracle, one-candidate process contract.
- Source changes only on an isolated branch off 777f after arm-perf confirms
  this first slice. Scratch intrinsic candidates HELD until .80 disassembly
  confirms the gap + arm-perf authorizes. Production leaf/feature-admission
  changes = subsequent separate approval.
- Test__CPUHalfOperands.cpp already proves the engine constructor, FP32-RNE +
  stored-half entries, rounding words, serial/grouped, worker policy; the new
  file adds performance ownership only.

## Reviewer #143 contract corrections (2026-11-09) — MUST reflect before any source

(1) ORACLE: half-representable FP32 inputs = necessary, NOT sufficient for an
FP32-A byte oracle; the reference must use the SAME 256-lane tree AND the same
completion program. ORDER-DISCRIMINATING fixture (add separately from
conversion-discriminating fixtures): K=257 with products p[0]=2^24, p[1]=1,
p[256]=-2^24, all others zero (every A/B exactly binary16: +-4096x4096, 1x1)
-> increasing-K serial FP32 gives 0, canonical 256-lane tree gives 1.
A generic native/oneDNN FP32 reference CANNOT certify by representability.
(2) TIMING STATE: repeated timing with beta!=0 needs a NAMED initial-C/state
protocol: restore the same C0 outside each measured interval OR specify/replay
the same bounded trajectory in both processes (warmups/samples must not
consume changing states). GROUPED bundle is NOT an arbitrary alpha/beta API:
executeHalfBundle (FloatingPointGemmKernel.h:4728) calls executeHalfOperands
(...,1.f,0.f,...,0) -> REJECT unsupported grouped alpha/beta/offset requests
(no silent ignore). Record alpha/beta, bias/initial-output bytes, offsets/
selections and bundle geometry in the case identity.
- Confirmed by reviewer: constructor/public-entry ownership; separate new perf
  owner; smoke-vs-recorded-preflight separation. HOLD retained until the
  oracle/state/API distinctions are explicit; no test-only source authorized.
