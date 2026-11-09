# Draft v2: independently owned production CPU half-operand perf cell

2026-11-09. Base authenticated: 777f7a52a. PLAN ONLY; reviewer HOLD active.
No source/CMake edits or grant assumed. Replaces v1 + the earlier
Perf__CPUFloatingMoEExpert modification proposal (ownership outside this
lane; R2 panel tournament = Q8 only). Incorporates reviewer#143 oracle,
timing-state and case-identity corrections.

## Owner / registration
- New source: tests/v2/performance/kernels/cpu/gemm/Perf__CPUHalfOperand
  Projection.cpp (proposed). New target: v2_perf_cpu_half_operand_projection.
- CTest smoke: V2_Perf_CPUHalfOperandProjectionSmoke (performance/model-free
  CPU, not Unit; add_v2_test; distinct from the recorded-input correctness
  replay in ProductionTestPreflight).
- Do NOT modify Perf__CPUFloatingMoEExpert; reuse stable measurement
  conventions; inspect final CMake deps; no model downloads/GPU runtime/
  private asm microkernel/intrinsic-only proxy.

## Production route / supported requests
- Engine: FloatingPointGemmKernel(weight, NumericalPolicy::BackendNative,
  FloatingActivationOperandPrecision::FP16RNE). Primary: multiply_tensor
  (real M/N/K, FP32 output); physical-half A = separate input-storage case;
  include production conversion, canonical reduction, supported completion +
  publication (not dotRows alone). Gate/up/down MoE cycle is NOT a proxy.
- Single-projection API: preserve exactly the supported alpha/beta/bias/
  activation_row_offset args; validate geometry/type/transpose before
  timing; production dense FP32 destination only; unsupported shapes fail
  before writes/timing.
- GROUPED half API (multiply_fused_verifier_rows_decode_equivalent ->
  executeHalfBundle) at this base calls:
  executeHalfOperands(input, output, m, projection.n, k, true, 1.f, 0.f,
                      projection.bias, 0)
  => NOT arbitrary alpha/beta/offset/sparse-row-selection: supports
  alpha=1, beta=0, activation row offset=0, dense first-M rows, transposed
  stored weights, optional per-projection bias. REJECT grouped alpha!=1,
  beta!=0, nonzero source/output offsets, arbitrary selections/holes/prefix
  scopes, original_output_slice BEFORE any output write. Never silently
  ignore fields or borrow native row-selection semantics (the half branch
  returns before those paths). Record the fixed values even when implicit;
  bundle order, per-N weight/bias/output, shared M/K + publication semantics
  are part of the case.
- ONE process = ONE exact candidate/config, sealed before timing. No sweep in
  profiler; baseline/candidate = separate fresh processes, rotating order;
  baseline = current production only (no second scalar fallback); explicit
  candidate must compile+exist or fail precisely; separate binaries compared.

## Versioned case identity (before comparison)
Readable canonical case descriptor + digest over the complete schema.
Record: case-schema/oracle/rounding/completion-program versions; fixture/
trace identity, seed + exact operand/output-state hashes; M/N/K, live +
guard extents, transpose, storage types/layouts/strides; FP16RNE operand
policy + engine policy + fixed256-lane contract; entry point, single vs
ordered bundle, per-projection M/N/K/weight bytes/bias presence + bytes/
output extents, shared input identity, bundle order/geometry; alpha/beta RAW
FP32 words; bias presence separate from zero bias + exact bytes; initial-C/
C0 bytes incl. guards; named state/reset protocol; signed zeros preserved;
absent bias != explicit +0; source row offset/output offset/slice/
selections/ordered selected-row list (grouped: zeros/dense/1-4 rows, reject
nontrivial); worker count/team/affinity; warmup/sample/repeat + timer/
profiler mode; runtime route/compiled ISA envelope + FP environment;
source/compiler/binary identity bound to each record.
Baseline/candidate admission requires equality of ALL workload/state/contract
fields (not just M/N/K); unsupported grouped fields = validation errors, not
normalization.

## Independent oracle standard (before timing)
1. Binary16-RNE activation words built independently of the candidate
   conversion routine; immutable explicit bit-patterns/dyadic fixtures +
   nontrivial recorded-input fixture where available; original half weights
   authoritative.
2. Independently reproduce all 256 logical FP32 lanes (+0 init; lane l
   consumes k=l,l+256,... via FP32 FMA; reduce by EXPLICITLY rounded adds at
   strides 128..1). NO increasing-K serial dot, partition-dependent tree,
   oneDNN reduction or tolerance as an equality oracle.
3. Independently reproduce the HalfRNE COMPLETION program (not generic
   native-FP32): round alpha*dot to FP32; bias = separately rounded add (not
   fused); beta!=0 = single FP32 FMA(beta,C0,value); beta==0 suppresses C
   read; absent bias = no extra +0 (signed-zero preserved). Never call the
   candidate's completion helper for expected words; same FP environment;
   no oracle reassociation/contraction across required boundaries.
4. Raw output bytes vs independent half-contract oracle; baseline/scalar =
   additional check only; grouped rows also equal independent serial M1 under
   the SAME supported completion; verify guards/inactive capacity + input/
   weight immutability + dirty-buffer overwrite.
5. Original-FP32-A reference equality requires BOTH (a) every relevant A
   exactly half-representable under the conversion contract AND (b) the
   reference executes the SAME 256-lane tree + identical completion.
   Representability is necessary, NOT sufficient; generic native/oneDNN FP32
   cannot certify. General FP32 A -> rounded-half oracle + separate
   conversion-discriminating fixtures.
6. Reject mismatches before timing; no GTEST_SKIP/seed retry/clipping/
   fallback/relaxed tolerance; benchmark correctness != totality/replay/
   preflight gates.

### Order-discriminating fixture (separate from conversion fixtures)
K=257, M=N=1, alpha=1 beta=0 bias absent, dense row0. All A/B = exact +0
except: k0 A=4096 B=4096 (2^24); k1 A=1 B=1 (1); k256 A=-4096 B=4096
(-2^24). All exactly binary16 (0x6c00/0xec00/0x3c00).
- Increasing-K FP32 FMA: +2^24, +1 rounds back to 2^24 (RNE tie), -2^24 -> +0
  (word 0x00000000).
- Canonical lane oracle: lane0 k0+k256 -> +0; lane1 retains 1; tree returns 1
  (word 0x3f800000).
Require production = 0x3f800000; the deliberately wrong increasing-K
reference = 0x00000000 (proves order discrimination). NOT a conversion/FP32-
equality fixture. Other fixtures discriminate RNE conversion + completion
(nontrivial alpha, beta, present/absent bias, C0 where supported).

## Timing-state protocol — C0-per-invocation-v1 (named)
- beta!=0 single: before EVERY warmup + measured invocation restore the
  SAME C0 into the retained output buffer OUTSIDE the interval; timer starts
  after restore, one production invocation, stop before checksum/validation/
  reset; repeats-per-interval = 1; aggregate independently reset intervals as
  samples. Warmups never determine next C.
- beta==0: API must not read prior C; repeated same-input overwrites may be
  amortized in one interval with recorded repeats; same C0/guards restored
  before each warmup/sample interval outside its timer; beta-zero no-read
  contract validated with separate poisoned/dirty-output correctness
  fixtures (not a changing measurement policy).
- Grouped: beta fixed 0 (API); beta-zero protocol; every per-projection
  output/C0/guard recorded; REJECT requested nonzero beta (no fictitious
  grouped reset).
- Baseline/candidate identical C0, reset frequency, worker policy, warmup/
  sample counts, outside-interval reset/cache treatment; timer overhead
  recorded; no profiler on canonical timing; explicit-reset cache effects
  belong to this named protocol (not hidden).
- No bounded-state-trajectory alternative silently available (would need a
  new protocol identity + identical recorded trajectory + proof in both
  processes).
Prep once before samples (tensors/engine/workspaces retained); no per-sample
weights/conversion banks/alloc/rebinding/team changes; production conversion
timed; C0 restoration = bounded CPU data reset (not topology/arithmetic
replacement); validation hashes outside intervals.

## Timing / production evidence
Samples + median/CV/quantiles + exact case identity + output hashes +
source/compiler/binary identity + route/envelope + worker count + live
extents; measured counters only when permitted (never timer ticks as PMU).
Separate one-candidate profiler process: leaf counts, conversion removal,
full admitted arithmetic spill closure, instr/cycle/cache evidence; no
unrelated kernels. .80 authoritative; GDN alpha/beta geometries from
canonical definitions/fixtures + K255/256/257; one worker then physical-core
scaling; no GPU/GLM touch. PerfStats route assertion: executeHalfOperands
exposes cpu_fp16_operand_compute32_calls (fixed_256_lane_tree + activation
storage tags) — assert in correctness/route validation OUTSIDE clean timing;
never infer route from output/flags; no allocating stats tags inside
canonical timing.

## Checkpoint / HOLD
Reviewer#143 retains HOLD until EXPLICIT release of this v2
oracle/state/case-identity checkpoint. Ownership/constructor + registration
split confirmations = not permission. Sequence: revised checkpoint release ->
actual .80 shipped-object disassembly -> explicit first-slice confirmation ->
isolated branch off 777f -> test-only cell/smoke -> .80 baseline/economy ->
separately approved candidate/leaf work. Scratch intrinsics held pending
disasm + own approval; production ISA/leaf/admission changes not bundled.

Source basis: Test__CPUHalfOperands.cpp; FloatingPointGemmKernel.h
(completeSkinnyMatmulValue + HalfRNE completion branch; multiply_tensor;
grouped verifier branch; executeHalfOperands/executeHalfBundle incl.
fixed 1/0/0). No arithmetic/API contract created/altered; existing owners
untouched.
