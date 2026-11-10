# Plan first: native NEON FP16 projection, exact 256-lane contract
(MASTER — 2026-11-10 evidence-archive revision; v2.1 contract/holds retained;
scope-qualified .80 receipt added. Design approved by arm-perf; no production
code authorized. Base 777f7a52a; diff 39c->777f for the FP16 chain = EMPTY;
final-base gate PASSED. Implementation + scratch candidates HELD until .80
hot-loop/closure evidence + reviewer v2.1 release + canonical .80 census +
explicit first-slice confirmation.)

## Findings / scope
- No vfmlal/fmlalq under src/v2 at this checkout; fresh GCC14.2 TU compile
  (production Integration -O3 -g NEON_I8MM flags; armv9 diagnostic variant)
  emits FMLAL/FMLAL2=0 in RMSNorm/Softmax/KernelFactory TUs.
- Only half x half candidate = explicit FP16RNE floating projection (GDN
  selective alpha/beta), instantiations in KernelFactory.cpp from
  FloatingPointGemmKernel.h. Exact-tree dispatcher = Scalar/AVX2/AVX512 only
  (NO NEON case); active NEON -> default scalar program. Two FP16RNE
  dotRows<4> bodies (FP32-A HalfRNE; physical-half A) each emit one scalar
  FMADD, no vector FMLA/FCVT; half conversion = integer bit arithmetic.
  Missing native vector implementation, NOT missed compiler fusion.
- Closed non-candidates: attention dot (mixed FP32 query x half key; FP32
  prob x half V); T5 RMSNorm (FP32-original t5SquaredSums; normalize rejects
  non-FP32 T5; FP16 uses FP64 squaredSums); softmax (FP32 poly after half
  decode).

## Gate-1 SPARKS receipt (2026-11-10) — scope-qualified
- .80 = VERIFIED WRAPPER/DISPATCH ONLY; hot-loop dotRows<4> bodies +
  relocations + 777f-bound identity = OUTSTANDING. All 4 relay payloads
  SHA-verified (d1 bd644e4a.., d2 7a3c17ff.., d3 71e59fc0.., fp16 ef4f1387..).
  Producer: .80, rel-f9f190059-neon_i8mm, KernelFactory.cpp.o, g++-14 14.2.0
  Release NEON_I8MM; exact argv/source/object receipt + final-binary identity
  NOT supplied; f9f190059 not local, not diff-authenticated vs 777f.
- CORRECTION: d1/d2 = OpenMP wrapper only (8 scalar FMUL + 4 FADD + 8 FMADD =
  completion after BL; NOT K-loop); dotRows = empty headings; no relocations
  (bl 0 = placeholder, not self-calls/absence); d3 = dispatch/publication (no
  FP math). Do NOT import '0 FMA'/'serial partials/no 256-lane'/'per-K'.
  Local source/object analysis still establishes missing native exact-tree.
  Requested via arm-perf: hot-leaf/closure extraction + identity binding
  (prefer rel-777f7a52a from r2gate2's Release build; else checked
  39c/777f/f9f arithmetic diff receipt).

## Existing exact contract
- FloatingExpertNumericalContract.h (dotRows, reducePartials);
  DeviceFP32NumericalContract.h (fusedMultiplyAdd, add, persistRounded);
  FloatingPointGemmKernel.h (roundFloatingOperandHalfRNE,
  executeHalfOperands, run_gpu_aligned_expert_fp32_matmul_isa,
  completeSkinnyMatmulValue).
- Per output+row: 256 logical FP32 lane accumulators +0; lane l consumes
  l,l+256,... via FP32 FMA; reduce strides 128..1 rounded adds; HalfRNE
  completion = rounded alpha*dot, bias add separately when present, FMA
  (beta,C0,value) only when beta!=0 (else no C read); only entrypoint's
  supported offsets/selection; grouped half = alpha1/beta0/offset0/dense
  first-M -> REJECT other completion/offset/selection (see half-perf v2.1).
  Current scalar implements this contract; new NEON must equal BOTH an
  independent contract-ordered oracle AND current scalar (no order change
  for SIMD).

## Variant domain
HalfRNE/HalfStorage (T/F): (T,F) FP32 rounded bin16 -> FMLAL candidate after
exact rounding; (T,T) stored bin16 -> FMLAL candidate; (F,T) stored bin16 ->
half-product candidate (distinct epilogue); (F,F) unrestricted FP32 -> NOT
FMLAL (native FP32 x widened-FP16 FMA). Other weight types first-class; no
BF16/FP32 reinterpretation; no broaden to quantized/attention/RMSNorm.

## Identity proof + limits
Finite bin16 a,b: <=11 significand bits; product <=22 bits exact in FP32
(min 2^-48 normal); same per-lane fused op + 256-lane tree => finite-domain
induction (no associative claim). Limits: NaN/payload, Inf x 0, signed zero,
FPCR rounding/flush, half-subnormal treatment must match runtime contract +
oracle; no global FPCR change, no finite-only shrink, no canonicalization, no
scalar retry. Conversion must match roundFloatingOperandHalfRNE (subnormal
ties + NaN); neon_fp16::narrow NOT interchangeable (truncates subnormals);
HW FCVTN only after proving rounding/FPCR/exception words; else the RNE bit
program in SIMD once per fragment. (Reviewer: post-conversion domain open —
finite FP32 like 65520 -> half INF; geometry admission :4660-4673 doesn't
exclude NaN/Inf; full exceptional/payload/conversion + LTO isolation gates =
REAL. TAIL invariant: inactive K lanes get NO arithmetic update (no dummy
fma(+0,+0,-0) changing -0 -> +0); preserve untouched lanes/rows + beta==0
nonconsumption; signed-zero + partial-stripe discriminating controls.)

## Proposed native schedule (subject to .80 measurement approval)
Explicit NEON exact-tree impl (not a special-case in scalar default);
16-lane tile (4 FP32 vectors/row), stride-256 K visits, <=4 rows = 16 acc
regs; FMLAL/FMLAL2 only half x half; mixed = widened B + FP32 FMA;
materialize each lane once -> fixed-256 bank -> unchanged binary tree NEON
adds (bounded per-invocation storage; no heap/repack/host shadow); tiles
8/16 measured separately; K<256/K%256 tails bounded, +0 inactive lanes;
FMLAL NOT authorized by the NEON_I8MM envelope -> minimal function/TU-local
FP16FML target with native ASIMDFHM authentication via ARMCPUCapabilities
(FP16Arithmetic FPHP+ASIMDHP insufficient; Darwin separate OS key);
qualified widen+FMA = separately certified impl; explicit dispatch, never
fallback; no broad armv9 envelope/M1 change. (Reviewer: complete admission
design OPEN — name exact FPCR profile + stability/admission owner on EVERY
worker incl. retained teams; FPSR flags/traps contract role.)

## Gates
1. .80 wrapper received/SHA-verified; AWAIT actual dotRows instruction/
   relocation closure + compiler/source/object binding (never substitute
   wrapper completion counts for a leaf census). Owner discovery: no
   exact-half CPU perf mode (Perf__CPUFloatingMoEExpert + CPUFloatingVerifier
   = FP32 activations; RecordedGDNProjection = GPU adapter; R2 = none) ->
   NEW separately owned production-shaped half perf file/registration
   (scratch-half-perf-cell-draft.md; reviewer v2.1 HOLD). Initial .80 scope:
   GCC14, exact envelope/route, M/N/K, threads, persistent policy, candidate
   cycling + fresh-process medians.
2. Approve contract/variant design + scratch target-local candidate BEFORE
   production edits; require repeated latency win beyond noise incl.
   conversion + tree + completion; reject uneconomical.
3. Byte oracle: M1..supported grouped; row tiles 1-4; K 1/3/4/7/8/15/16/255/
   256/257/511/512/513 + GDN sizes; N tails; entry-supported offsets/
   selection/alpha/beta/bias; stored-half + FP32-RNE; cancellation, signed
   zero, subnormal/normal/ties/overflow + exceptions; grouped rows vs
   independent serial M1 + fixed-tree/completion. Half representability NEVER
   certifies a native FP32 oracle. ORDER fixture K257 (p0=2^24,p1=1,p256=
   -2^24): canonical 1 vs increasing-K 0. Conversion fixtures separate.
   Nonzero-beta timing: same C0 restored before every measured invocation.
4. Instruction census per emitted variant/envelope (FMLAL in half leaf, FMA
   in mixed, no prohibited in baseline, admission negative controls, complete
   identity).
5. Register witness extension: no FP/SIMD spills in accumulation; typed256-
   lane bank vs stack/ABI/reduction I/O; per row count + lane tile.
6. Reuse V2_Unit/Integration_CPUHalfOperands + production floating-expert
   gates; add focused recorded-input production replay (reset/dirty/partial/
   repeated) in ProductionTestPreflight; extend Unit policy/rounding +
   positive worker budgets (no canonical-name weakening).
7. Full Unit + ProductionTestPreflight after a slice; .80 Release
   correctness/economy + GDN E2E/PerfStats route; no deterministic
   substitution.

## Provenance / hold
Evidence: artifacts/fp16-emission/ (manifest + production/armv9 asm +
projection-inner JSON + symbols); relay audit scratch-fp16-sparks-relay-audit.md.
No production/CMake/capability/test changes. Design approved; HOLD until:
reviewer v2.1 release, canonical .80 census, explicit first-slice
confirmation. Next possible slice = the independently owned test-only
perf cell/smoke; native half rounding/schedule/admission = later separate
slice after .80 baseline. Full gated family ~3-5 focused engineer-days.

## Gate-1 BOUND evidence — CORRECTED wording (2026-11-10, lane read of the
## actual bodies; supersedes the earlier wrapper-only + 'no 256-lane' claims)
- rel-777f7a52a Release NEON_I8MM, obj SHA e01c7c1a..., source
  KernelFactory.cpp SHA a9dc843f... (verified locally by the lane), g++-14
  14.2.0; full -dr dump of E2 dotRows<4> bodies (relocations included).
- ACTUAL HalfRNE true/false + true/true dotRows bodies: EACH = ONE static
  scalar FMADD + ONE reduction FADD (the '6 per m4' aggregate included the
  unrelated Native false/false body's 4). They contain EXPLICIT k&0xff lane
  indexing -> the 256-lane structure IS present (implemented SCALARLY).
  'No 256-lane structure' = WITHDRAWN (not in the plan).
- GAP (supported): scalar y - vectorized FMLAL potential at the SAME exact
  256-lane contract (k&0xff indexing); replacement of scalar FMADD/FADD with
  vector FMLAL ops while preserving lanes/order. Opportunity stands.
- Final evidence piece: exact compile argv (build.ninja CXX entry) pending
  relay (non-blocking; idea31 in progress).
