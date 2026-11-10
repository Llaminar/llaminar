# Plan first: native NEON FP16 projection, exact 256-lane contract

2026-11-10 — PERFORMANCE lane, SPARKS/X925 target. Native leaf remains a
later unapproved slice. Reviewer #150 released v2.1; the independent
test-only half cell/smoke is explicitly authorized second after idea31.
Source analysis + fresh GCC14 TU compilation on detached 39c24d8b4. Design +
gates approved by arm-perf. Authenticated final implementation base:
**777f7a52a** (R2/C4 on main f69446e3c). git diff 39c->777f for the FP16
chain = EMPTY; R2 seams verified; final-base gate PASSED. **FP16 gate1
archival COMPLETE** (corrected .80 arithmetic bodies/relocations +
argv/literal-includes/compiler/source-object binding verified; local include
receipt hash c1fbadb2... matches; dedicated 777f worktrees preserve the
source basis while another worker owns the station's main checkout).

## Findings / scope
- No vfmlal/fmlalq under src/v2 at this checkout. Fresh GCC14.2 TU compile
  (production Integration -O3 -g NEON_I8MM flags + armv9 diagnostic variant)
  emits FMLAL/FMLAL2=0 in RMSNorm/Softmax/KernelFactory TUs.
- Only half x half candidate = explicit FP16RNE floating projection (GDN
  selective alpha/beta), instantiated in KernelFactory.cpp from
  FloatingPointGemmKernel.h. Exact-tree dispatcher = Scalar/AVX2/AVX512 ONLY
  (NO NEON case); active NEON -> default scalar program. Two FP16RNE
  dotRows<4> bodies each emit a scalar FMADD, no vector FMLA/FCVT: half
  conversion = integer bit arithmetic. Missing native vector implementation,
  NOT missed compiler fusion.

### Gate-1 SPARKS receipt (2026-11-10): COMPLETE
- Actual .80 dotRows<4> bodies/relocations + full argv+l literal include
  transcript verified (rel-777f7a52a GCC14.2 Release NEON_I8MM): each REAL
  HalfRNE dotRows body = ONE static scalar FMADD + ONE reduction FADD, no
  vector FMLA/FMLAL/FMLAL2/FCVT; conversion = integer bit arithmetic; both
  retain explicit k&0xff canonical-256-lane indexing. The unrelated native
  false/false body has 4 static FMADD sites; the aggregate 6 FMADD must
  never become a per-m4/per-K cost claim. Corroborates the missing native
  exact-tree implementation (not merely a wrapper).
- Relay: fp16-gate1-bound-777f.txt, fp16-dotrows-E2-dr-777f.txt,
  fp16-gate1-argv-777f.txt (+ includes file, 981 B, SHA c1fbadb2...).
  Object SHA e01c7c1a..., source KernelFactory.cpp SHA a9dc843f... matches
  immutable local 777f source. Final-binary/runtime identity = separate
  later gate; no performance result or native-slice authorization follows.
- SUPERSEDED f9f190059 d1/d2/d3 (wrapper/dispatch only; empty dotRows
  headings; bl0-without-relocations != recursion; 8 FMUL+4 FADD+8 FMADD =
  completion sites, not K-loop). Withdrawn claims stay withdrawn: '0 FMA',
  '8 FMUL+4 FADD per hot m4', 'no 256-lane structure'. No arithmetic change/
  native experiment/perf conclusion from gate1.

Closed non-candidates: attention dot (FP32 query x half key; FP32 prob x
half V); T5 RMSNorm (FP32-original t5SquaredSums; normalize rejects non-FP32
T5; FP16 uses FP64 squaredSums); softmax (FP32 poly after half decode).

## Existing exact contract — no changed arithmetic
FloatingExpertNumericalContract.h (dotRows, reducePartials);
DeviceFP32NumericalContract.h (fusedMultiplyAdd, add, persistRounded);
FloatingPointGemmKernel.h (roundFloatingOperandHalfRNE,
executeHalfOperands, run_gpu_aligned_expert_fp32_matmul_isa,
completeSkinnyMatmulValue). Per output+row: 256 logical FP32 lanes +0; lane
l consumes l,l+256,... via FP32 FMA; reduce strides 128..1 rounded adds;
HalfRNE completion (rounded alpha*dot; bias separately when present; FMA
(beta,C0,value) only when beta!=0; grouped = alpha1/beta0/offset0/dense
first-M -> reject other completion/offset/selection; see reviewer-released
v2.1 half-perf plan for constructor/scope/oracle/C0 details; requested width
!= observed; PerfStats isa = pre-dispatch). Current scalar implements this;
new NEON must equal BOTH an independent contract-ordered oracle AND current
scalar (no order change for SIMD).

## Variant domain
HalfRNE/HalfStorage (T,F): FP32 rounded bin16 -> FMLAL after exact rounding;
(T,T): stored bin16 -> FMLAL; (F,T): stored bin16 -> half-product candidate
(distinct epilogue); (F,F): unrestricted FP32 -> NOT FMLAL (native FP32 x
widened-FP16 FMA). Other weight types first-class; no reinterpret; no
broadening.

## Identity proof + limits
Finite bin16 a,b: <=11 bits; product <=22 bits exact in FP32 (min 2^-48
normal); same per-lane fused op + 256-lane tree => finite-domain induction
(no associative claim). Limits: NaN/payload, Inf x 0, signed zero, FPCR
rounding/flush, half-subnormal = must match runtime contract + oracle; no
global FPCR change, no finite-only shrink, no canonicalization, no scalar
retry. Conversion must match roundFloatingOperandHalfRNE (subnormal ties +
NaN); neon_fp16::narrow NOT interchangeable (truncates); HW FCVTN only after
proving rounding/FPCR/exception words; else the RNE bit program in SIMD.
(Reviewer items: post-conversion domain open [65520 -> half INF]; geometry
admission :4660-4673 doesn't exclude NaN/Inf; full exceptional/payload/
conversion + LTO isolation gates = REAL. TAIL invariant: inactive K lanes get
NO arithmetic update (no dummy fma(+0,+0,-0) changing -0 -> +0); preserve
untouched lanes/rows + beta==0 nonconsumption; signed-zero + partial-stripe
controls. Complete admission design OPEN: FPCR profile + stability/admission
owner on EVERY worker incl. retained teams; FPSR flags/traps role.)

## Proposed native schedule (subject to .80 measurement approval)
Explicit NEON exact-tree impl (not a special case in scalar default); 16-lane
tile (4 FP32 vectors/row), stride-256 K visits, <=4 rows = 16 acc regs;
FMLAL/FMLAL2 only half x half; mixed = widened B + FP32 FMA; materialize
each lane once -> fixed-256 bank -> unchanged binary tree NEON adds (bounded
storage; no heap/repack/host shadow); tiles 8/16 measured separately; K<256/
K%256 tails bounded +0 inactive; FMLAL NOT authorized by NEON_I8MM envelope
-> minimal TU-local FP16FML target with native ASIMDFHM authentication via
ARMCPUCapabilities (FPHP+ASIMDHP insufficient; Darwin separate OS key);
qualified widen+FMA = separately certified impl; explicit dispatch, never
fallback; no broad armv9 envelope/M1 change.

## Gates
1. Gate1 COMPLETE (bound bodies/relocations/transcript; never substitute
   wrapper counts). Owner: NO exact-half CPU perf mode at 39c (FloatingMoE
   Expert + CPUFloatingVerifier = FP32 activations; RecordedGDNProjection =
   GPU adapter; R2 none) -> NEW separately owned production-shaped half perf
   file/registration (half-perf-cell-draft v2.1; reviewer #150 released).
   Initial .80 scope: GCC14 exact route M/N/K threads persistent policy one
   candidate per launch fresh-process medians.
2. Approve contract/variant design + scratch target-local candidate BEFORE
   production edits; repeated latency win beyond noise incl. conversion +
   tree + completion; reject uneconomical.
3. Byte oracle: M1..grouped; rows 1-4; K list + GDN sizes; N tails; entry-
   supported offsets/selection/alpha/beta/bias; stored-half + FP32-RNE;
   exceptional inputs; grouped rows vs independent serial M1 + fixed-tree/
   completion. Representability NEVER certifies a native FP32 oracle.
   ORDER fixture K257 (2^24/1/-2^24): canonical 1 vs increasing-K 0.
   Conversion fixtures separate. Nonzero-beta C0 reset per invocation.
4. Instruction census per emitted variant/envelope + admission negative
   controls + complete identity.
5. Register witness extension (no spills in accumulation; typed256-lane bank
   vs stack/ABI/reduction; per row count + tile).
6. Reuse V2 Unit/Integration CPUHalfOperands + floating-expert gates; add
   focused recorded-input production replay (reset/dirty/partial/repeated)
   in ProductionTestPreflight; extend Unit policy/rounding + positive worker
   budgets.
7. Full Unit + Preflight after a slice; .80 Release correctness/economy +
   GDN E2E/PerfStats; no deterministic substitution.

## Provenance and hold
Evidence: artifacts/fp16-emission/ + relay audit + all receipts. No FP16
production/CMake/capability/test changes. Reviewer #150 released v2.1
(oracle/API/state/observation); canonical .80 receipts + explicit coordinator
GO landed. SEQUENCE: idea31 nibble-lifetime correctness/register/economy +
explicit disposition FIRST, then the independently owned test-only half
cell/smoke on a separate 777f branch. Every actual public invocation must
return successful bool (incl. warmup/timed/postcheck); preserve immutable/
guard bytes; stats outside clean timing. Existing floating expert performance
file/inventory untouched. Obtain the half cell's .80 baseline/economy before
separately approving native half rounding/schedule/feature admission or
scratch candidates. Gate1 closed; no further FP16 evidence collection before
the half-cell work. Fully gated native family ~3-5 focused engineer-days;
NOT native implementation authorization.
