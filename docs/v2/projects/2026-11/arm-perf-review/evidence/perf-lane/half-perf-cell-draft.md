# Draft v2.1: independently owned production CPU half-operand perf cell

2026-11-09/10. Base authenticated: 777f7a52a. PLAN ONLY; reviewer HOLD active.
Replaces v2/v1 + the earlier Perf__CPUFloatingMoEExpert modification proposal
(ownership outside this lane; R2 panel tournament = Q8 only). Incorporates
reviewer#143's oracle, timing-state, case-identity corrections PLUS the
residual addendum (public-API specifics, fully test-owned oracle, state
validation, requested-vs-observed identity).

## Owner / registration
- New source: tests/v2/performance/kernels/cpu/gemm/Perf__CPUHalfOperand
  Projection.cpp (proposed, not created). New target:
  v2_perf_cpu_half_operand_projection. CTest smoke:
  V2_Perf_CPUHalfOperandProjectionSmoke (performance/model-free CPU, not
  Unit; add_v2_test; distinct from the recorded-input correctness replay in
  ProductionTestPreflight). Do NOT modify Perf__CPUFloatingMoEExpert; reuse
  stable measurement conventions; inspect final CMake deps; no model
  downloads/GPU runtime/private asm microkernel/intrinsic-only proxy.

## Production route / supported requests
- Prepare original FP16 weights + FP32 activations. Use the ACTUAL
  BORROWED-WEIGHT constructor with a fixture owning the tensor + backing
  storage for the entire engine/call lifetime (destroy engines before their
  fixture weights):
    using Engine = gemm::FloatingPointGemmKernel;
    Engine engine(weight.get(), Engine::NumericalPolicy::BackendNative,
                  FloatingActivationOperandPrecision::FP16RNE);
  (fixture-retained owning FP16Tensor handle, not a temporary/raw allocation).
  OR the existing OWNING overload constructed entirely outside timing with the
  actual argument order:
    Engine engine(shared_const_weight, Engine::NumericalPolicy::BackendNative,
                  CPUWeightStoragePlacement::local(),
                  FloatingActivationOperandPrecision::FP16RNE);
  (copies native bytes into engine-owned storage during preparation; its third
  argument = placement, not operand precision). Record the chosen ownership/
  preparation contract in the case identity.
- Primary timed route = EXPLICIT-DIMENSION public multiply_tensor overload:
    engine.multiply_tensor(input, output, m, n, k, true, alpha, beta,
                           bias, nullptr, -1, nullptr, activation_row_offset);
  Physical-half A = separate storage case using the same public entry.
  Include production activation conversion, canonical reduction, supported
  completion + output publication. NEVER call private executeHalfOperands,
  bypass visibility, or time dotRows/helper as a proxy. A full gate/up/down
  MoE cycle is not a GDN alpha/beta proxy.
- Single-projection: preserve exactly the supported alpha/beta/bias +
  activation_row_offset; validate geometry/type/transpose, completion
  scalars, initial-C + supported row/offset policy before warmup/timing/
  postcheck; production dense FP32 destination only; unsupported shapes fail
  before writes/timing.
- GROUPED half API: multiply_fused_verifier_rows_decode_equivalent enters
  executeHalfBundle directly for FP16RNE; at this base executeHalfBundle calls
  executeHalfOperands(input, output, m, projection.n, k, true, 1.f, 0.f,
                      projection.bias, 0). Invoke the real public homogeneous
  verifier bundle with its real scope (per the existing public regression):
    ITensorGemm::VerifierKernelModeScope verifier_scope;
    engine.multiply_fused_verifier_rows_decode_equivalent(input, projections,
                                                          m, k);
  Every ordered TensorProjectionDesc retains its real FP16RNE engine, output,
  N, optional bias; fixtures own every referenced object; same scope/entry
  across warmup/measured/postcheck (per-thread scope constructed on the actual
  participating thread). NO private helper bypass. Therefore grouped is NOT
  arbitrary alpha/beta/offset/sparse-row-selection: supports alpha=1, beta=0,
  activation offset=0, dense first-M rows, transposed stored weights, optional
  per-projection bias. REJECT grouped alpha!=1, beta!=0, nonzero source/output
  offsets, arbitrary selections/holes/prefix scopes, original_output_slice
  before any output write; never silently ignore or borrow native row-selection
  semantics (the half branch returns before those paths). Record the fixed
  values even when implicit; bundle order, per-N weight/bias/output, shared
  M/K + publication semantics are part of the case.
- ONE process = ONE exact candidate/config sealed before timing; no sweep in a
  profiler; baseline/candidate = separate fresh processes, rotating external
  order; baseline = current production only (no second scalar fallback);
  explicit candidate must compile+exist or fail precisely; separate binaries
  compared.

## Versioned case identity
Readable canonical case descriptor + digest over the complete schema; candidate/
build identity retained alongside shared workload identity. Record:
- Case-schema + oracle/rounding/completion-program versions; fixture/trace
  identity, seed + exact operand/output-state hashes (seed insufficient).
- M/N/K, live + guard extents, transpose, storage types/layouts/strides;
  FP16RNE operand policy, engine numerical policy, fixed-256-lane contract.
- EXACT public overload, verifier scope, weight ownership/preparation contract;
  single vs ordered bundle; per-projection M/N/K, weight bytes, bias presence +
  bytes, output extents; shared input identity; bundle order/geometry.
- Alpha/beta RAW FP32 words; bias presence separate from a zero bias + exact
  bytes; initial-C/C0 bytes incl. guards; named state/reset protocol; signed
  zeros preserved; absent bias != explicit +0. Source activation-row offset,
  output offset/slice, selections/request-row scope (grouped: zeros/dense/
  first-M + REJECT nontrivial).
- REQUESTED vs OBSERVED execution/worker identity as SEPARATE fields
  (requested OpenMP width/dynamic/nesting/team/affinity; observed width/
  placement ONLY with evidence from the actual executing team + observation
  method/scope); warmup/sample/repeat counts; timer/profiler mode; FP
  environment; compiled envelope; PRE-DISPATCH active-ISA/requested-route tag
  vs resolved-leaf identity/status SEPARATE (unknown = explicitly unknown,
  never copied from requested).
Admission requires equality of ALL workload/state/contract fields; unsupported
grouped fields = validation errors.

## Independent oracle standard (ALL STAGES TEST-OWNED)
1. Independent binary16-RNE conversion/half-word decode, lane traversal,
   reduction AND completion = TEST-OWNED implementations of the contract (NOT
   calls to production roundFloatingOperandHalfRNE / half-conversion helpers /
   dotRows/reducePartials / completeSkinnyMatmulValue / device_fp32_contract
   wrappers). Test-owned code may use correctly-rounded platform arithmetic /
   std::fma with explicit FP32 boundaries but must independently implement the
   prescribed walk/order/conversion. Immutable explicit bit-patterns/dyadic
   fixtures + nontrivial recorded-input fixture where available; original half
   weights authoritative; serial production outputs = second check only.
2. Independently reproduce all 256 logical FP32 lanes (+0 init; lane l
   consumes k=l,l+256,... via FP32 FMA; reduce by explicitly rounded adds
   strides 128..1). No increasing-K serial dot / partition-dependent tree /
   oneDNN / tolerance as equality oracle.
3. Independently reproduce the HalfRNE COMPLETION program (rounded alpha*dot;
   bias added separately when present, not fused; FMA(beta,C0,value) ONLY when
   beta!=0; beta==0 suppresses C read; absent bias = no extra +0). Same FP
   environment; no oracle reassociation/contraction across required boundaries.
4. Raw output bytes vs the independent half-contract oracle; baseline/scalar =
   additional check only; grouped rows = independent serial M1 under the same
   supported completion; verify guards/inactive capacity, input/weight
   immutability, dirty-buffer overwrite.
5. FP32-A reference equality requires BOTH (a) every A exactly half-
   representable under the conversion contract AND (b) the reference executes
   the SAME 256-lane tree + identical completion. Representability is
   necessary, NOT sufficient; generic native/oneDNN cannot certify. General
   FP32 A -> rounded-half oracle + separate conversion-discriminating fixtures.
6. Reject mismatches before timing; no GTEST_SKIP/seed retry/clipping/
   fallback/tolerance; benchmark correctness != totality/replay/preflight
   gates.

### Order-discriminating fixture (separate from conversion fixtures)
K=257, M=N=1; alpha=1, beta=0, bias absent, dense row0. All A/B exact +0
except k0 A=4096 B=4096 (2^24), k1 A=1 B=1 (1), k256 A=-4096 B=4096 (-2^24)
(all exactly binary16: 0x6c00/0xec00/0x3c00).
- Increasing-K FP32 FMA: 2^24 -> +1 rounds back to 2^24 (RNE tie) -> -2^24 = +0
  (word 0x00000000).
- Canonical lane: lane0 = +0, lane1 = 1; tree returns 1 (word 0x3f800000).
Require production 0x3f800000 + the deliberately wrong increasing-K reference
0x00000000 (proves order discrimination). Not a conversion/FP32-equality case;
other fixtures discriminate RNE conversion + completion (nontrivial alpha,
beta, present/absent bias, C0 where supported).

## Timing-state protocol — C0-per-invocation-v1 (named)
- beta!=0 single: restore the SAME C0 into the retained output buffer OUTSIDE
  every warmup + measured invocation; timer after restore, one production
  invocation, stop before checksum/validation/reset; repeats-per-interval = 1;
  aggregate independently reset intervals; warmups never determine next C.
- beta==0: API must not read prior C; repeated same-input overwrites may be
  amortized with recorded repeats; same C0/guards before each warmup/sample
  interval outside its timer; beta-zero no-read validated with separate
  poisoned/dirty-output fixtures.
- Grouped: beta fixed 0 (API); beta-zero protocol; every per-projection output/
  C0/guard recorded; REJECT requested nonzero beta.
- WARMUP/TIMING/POSTCHECK use the SAME named beta-state protocol + the same
  validated public entry/completion args; final correctness-only invocation
  resets from the same C0 (or validates against that interval's recorded
  C0/result); per-entry validation = alpha/beta words, bias/C0/guards, offsets/
  selections, bundle geometry, verifier scope.
- Baseline/candidate identical C0, reset frequency, worker policy, warmup/
  sample counts, outside-interval treatment; timer overhead recorded; no
  profiler attached to canonical timing; explicit-reset cache effects belong to
  this named protocol (not hidden). No bounded-state-trajectory alternative
  silently available (would need a new protocol identity + identical recorded
  trajectory + proof).
Prep once before samples; no per-sample weights/conversion banks/alloc/
rebinding/team changes; production conversion timed; C0 restoration = bounded
CPU data reset; validation hashes outside intervals.

## Timing / production evidence
Samples + median/CV/quantiles + exact case identity + output hashes +
source/compiler/binary identity + distinct requested/observed route + worker
fields + live extents. The Workers guard only REQUESTS a team
(omp_set_num_threads/dynamic); omp_get_max_threads/env/affinity-size/probe-region
are NOT proof of execution width; observe from the real executing team where a
valid non-perturbing mechanism exists, else observed_width=unknown (no
actual-width scaling claim; never wrap a standalone call in an outer retained
team just for a count — that's a distinct case needing its own validation).
Measured counters only when permitted (never timer ticks as PMU). Separate
one-candidate profiler process: leaf counts, conversion removal, full admitted
arithmetic spill closure, instruction/cache/cycle evidence; no unrelated
kernels. Target .80 authoritative; first baselines from canonical GDN
alpha/beta geometries + K255/256/257 diagnostics (no invented shape table);
one worker then physical-core scaling; no GPU/GLM touch.
PerfStats counter cpu_fp16_operand_compute32_calls (fixed_256_lane_tree +
activation-storage tags) = invocation/policy/geometry validation OUTSIDE clean
timing ONLY, NOT resolved-implementation evidence; its isa tag =
activeISALevel() BEFORE dispatch (:4697-4707) = pre-dispatch requested-route
only; resolved_leaf=unobserved unless an actual valid observation exists;
typed resolved-leaf witness belongs to separately approved leaf work; never
infer resolved route from output/flags/counter; no stats tags inside canonical
timing.

## Checkpoint / HOLD
Reviewer#143 retains HOLD until EXPLICIT release of this v2.1
public-API/test-owned-oracle/state/observation checkpoint. Sequence: revised
checkpoint release -> actual .80 shipped-object disassembly -> explicit
first-slice confirmation -> isolated branch off authenticated 777f -> test-only
cell/smoke -> .80 baseline/economy -> separately approved candidate/leaf work.
Scratch intrinsics held pending disasm + own approval; production ISA/leaf/
admission changes not bundled. Source basis: Test__CPUHalfOperands.cpp;
FloatingPointGemmKernel.h (completeSkinnyMatmulValue + HalfRNE completion;
multiply_tensor; grouped verifier branch; executeHalfOperands/executeHalfBundle
incl. fixed 1/0/0). No arithmetic/API contract created/altered; existing owners
untouched.

## Synchronized status (2026-11-10; replaces the v2/v2.1 header lines)
- Reviewer #150 RELEASED the v2.1 public-API/test-owned-oracle/state/
  observation checkpoint (checked at 69ad36d6f vs 777f). Coordinator GO
  covers this independently owned TEST-ONLY cell/smoke, SECOND after the
  idea31 nibble-lifetime slice; it does NOT cover native leaf/ISA/admission/
  scratch intrinsic work. FP16 gate1 archival COMPLETE (bodies/relocations +
  argv/literal-includes + SHAs verified; includes receipt SHA c1fbadb2...).
- CONTRACT TEXT = the v2.1 body above (unchanged): borrowed/owning
  constructor (placement-before-precision), explicit-dimension multiply_tensor,
  real homogeneous verifier bundle/scope (1/0/0 + rejection), HalfRNE
  completion order, all-oracle-stages test-owned, K257 order fixture
  (0x3f800000 vs 0x00000000), C0-per-invocation-v1, requested-vs-observed
  worker fields, pre-dispatch ISA vs resolved-leaf separation.
- OBLIGATIONS (release requirements): every ACTUAL public call returns
  successful bool (warmup/timed/postcheck; false rejects the run even if old
  bytes match); immutable/guard bytes preserved; stats outside clean timing;
  no private helper timing; proved in the actual test source.
- Sequence: idea31 .80 economy + explicit disposition → separate 777f branch
  → this cell/smoke → .80 production baseline/economy → only then separate
  native candidate/leaf slice. No arithmetic/API contract created/altered.

### Reviewer initial scope finding (2026-11-11 ~09:51Z) — economy NOT ADMITTED (insufficient evidence)

At exact b512e25a: CPUHalfOperandProjection.md explicitly states no native
leaf candidate is IMPLEMENTED; resolved-leaf/observed-team remain
unobserved; K255/256/257 diagnostics are NOT canonical GDN evidence; and
authentic GDN trace + Linux economy are still OUTSTANDING. Therefore the
source-complete cell + 5/5 correctness can neither justify nor cancel
FMLAL on economic grounds. v2.1 requires matched fresh-process samples +
authentic geometry + external identity/FP/team/profiler receipts.
Provisional disposition: **economy NOT ADMITTED / insufficient evidence**
— NOT a measured FMLAL rejection. Existing source ACK stays scoped.
Review of supplied records (read-only) proceeds once exact artifact paths
arrive from arm-sparks.

### PROVENANCE CORRECTIONS (reviewer-verified + coordinator-confirmed, 2026-11-11 ~09:52Z)

1. **a847db3286 is NOT an ancestor of b512e25a** (merge-base --is-ancestor
   exits 1; verified). b512e25a = exactly adb1c98e9 → 1ea08211 → b512e25a
   (#173 docs; only Perf__CPUHalfOperandProjection.cpp +166/−13).
   a847db3286 lives on the witness/mainline lineage (contained in
   origin/codex/arm-s4-nibble-witness AND feat/arm64-native-kernels;
   parent 0d6f09919). The earlier framing 'fixture defects FIXED at
   a847db3286 within the b512 gate' was a coordinator mislabel: any
   fixture-repair overlay used in b512-era gate builds is a separate tree
   whose exact identity was never recorded — the arm-sparks artifact
   bundle must state what tree the b512-era builds actually used. The
   b512 failed gates (Unit 990/16, Preflight 2419/23) stand UNCHANGED as
   failures.
2. **half-perf-cell-draft.md (v2.1) is NOT at b512e25a** — it is
   tracker-only: branch codex/arm-perf-review, blob
   69bb3800b05de4e7d9c18efa1d6536bfd3d30a0a (this file). The coordinator's
   pointer claiming it was on the half-perf branch was wrong.
Disposition unchanged: economy NOT ADMITTED / insufficient evidence;
reviewer proceeds read-only once artifact paths arrive.

### Lane-2 step 1: carry-verification GREEN at bb893cd79 (arm-addon, ~11:16Z)

Detached /tmp/lane2 at bb893cd79 (clean); sanctioned macOS primitive path
(cmake -S tests/v2/portability, CMake 4.3.1, AppleClang-21, Integration,
NEON): 121/121 targets built. **18/18 focused cells PASS** (4.08 s):
MetadataSeal (U+Int Coverage+Lifecycle), WeightView (U+I), NEONExecution
(U+I), ProjectionBatch (U+I), NumericalContract (U+I), PreparedProjection
(U+I), NEONAttentionMath (U+I), NEONAttentionQ16Int32Bound,
CPUAttentionInvocationAdmission, PanelTailsMatchSerial. BONUS: the real
register-gate probe executed during the build and PASSED ("64 native NEON
tile programs: single-row spill nothing, grouped within budget") — the
descoped tooling still functions at bb. X86 syntax sweep: MetadataSeal +
AttentionInvocationAdmission pass -mavx512; PreparedProjection/
NEONAttentionMath correctly #error under non-AArch64 (by-design ARM-only;
x86 shared path = separate VNNI files). Per-file SHA-256s recorded
(PublishedBytes 3ed210b8…, WeightView 4883f771…, PreparedProjection
ad00ac6a…, ProjectionBatch 6fc68dad…, NEONAttentionMath ccd6e1db…,
AttentionInvocationAdmission 664d31bb…). Raw logs:
/tmp/lane2-carry-verification.log, /tmp/lane2-pt-build.log,
/tmp/lane2-pt-cfg.log. ATTRIBUTION: owner evidence only (M1/
AppleClang-21, portability project) — NOT the full Unit/preflight
inventory; the .80/container gate remains Main's.
STEP-2 WINDOW SPEC ready for the booking queue (after campaign/A2):
M=1 decode economy window on .80 — encodings Q8_K/Q8_0/Q4_K (+Q6_K
known-slow control), M=1, shapes 512×2048/2048×4096/17408×5120, threads
1/2/4/8/10/12/16/20, ABBA 31 iters raw, OMP_PLACES=cores
OMP_PROC_BIND=close, bb893cd79 Release NEON. Decision gate: the Q8_K
10-thread regression signal + per-strip activation touch counts decide
the load-once-per-block + lane-indexed-dot successor.

### Step-2 window script staged + queued (arm-addon, ~11:25Z)

Script: /Users/floriankinder/repo/arm-addon-notes/window-m1-activation.sh
(window-r1.sh protocol). Design approved as-is: SINGLE bb893cd79 Release
NEON_I8MM build; TWO reps per cell (no candidate yet — implementation
gated ON this window; successor gets its own ABBA); harness
v2_perf_cpu_native_quantised_thread_scaling with FORMATS=all (unknown
labels fatal); three baseline shapes isolated + one all-shapes sweep each
r1/r2; THREADS=1,2,4,8,10,12,16,20; N=1; WARMUP=5 ITERS=31;
OMP_PLACES=cores OMP_PROC_BIND=close. DECISION METRICS (script header):
(a) Q8_K 10-thread regression (2.96×) present/gone; (b) per-encoding
thread-saturation knee at M=1; (c) per-width ms/iter curves sizing the
activation-reuse win (4-strip panel = 4× touch amplification + per-strip
metadata recompute; load-once-per-block + vdotq_laneq removes both). Est.
20-40 min; outputs /work/build/results/w-m1act-<ts>/. QUEUED with
arm-sparks BEHIND the Main campaign — preferred slot .80 post-campaign
(disk-safe); alternative .79 post-trace only if the Release build fits
the 66G budget (sparks' disk call). Model-inference window: protected
from interruption once started.

### HALF-PERF ARTIFACT BUNDLE (arm-sparks read-only, ~15:19Z) — FMLAL leaf does NOT exist

(1) **MATCHED TIMING SAMPLES: NONE exist** — no per-cell matched baseline/
candidate timing samples were ever captured (the b512-era validations =
5/5 ctest cell PASS 0.16-0.55s each = REGISTRATION/protocol checks, not
matched economy samples). The reviewer was right.
(2) BINARY/BUILD IDENTITIES (all at .80 build/int-b512e25a, tree
b512e25a): llaminar2 = 33804e49…, libllaminar2_core.so = de9c1225…,
v2_perf_cpu_half_operand_projection = 83e1efab…; hpb-* logs: hpb-cells.log
(1824 B) + hpb-unit.log (10.6 MB) + hpb-pre.log (1.2 MB) + hp-b512.status;
register-gate receipts build/rg-717b2bfe6/{build,receipt}/ (9 cells) +
rg-717b.log.
(3) **FMLAL NATIVE LEAF: DOES NOT EXIST** — grep of the b512/1ea082118/
a847db3286 history for fmlal/vfmlal in src/v2/kernels/cpu/gemm/ = **0
hits** (CPUNativeNEONPreparedProjection.h at 1ea082118 = 0 fmlal). Beyond
the half-cell oracle/FP tests, a NATIVE FMLAL leaf candidate was NEVER
IMPLEMENTED → **the half-perf economy disposition = IMPLEMENTATION-FIRST**
(the reviewer's "implement the leaf first" branch), not evidence-first.
(4) Command/FP/team records: the 5/5 cells = the ctest invocations in
hpb-cells.log; FP path = FP16-codegen tests; no separate team record files.

### DISPOSITION UPDATE (coordinator, ~15:20Z): half-perf = IMPLEMENTATION-FIRST

The bundle settles the reviewer's gate question: **no FMLAL leaf candidate
exists in the branch history** and no matched economy samples exist → the
half-perf item moves from "insufficient evidence" to **IMPLEMENTATION-FIRST:
the FMLAL leaf must be implemented (per the half-perf plan v2.1 contract)
before any economy claim is possible.** Route: the implementation belongs
to the half-perf lane (addon2's half-perf lineage or the addon lane —
coordinator to assign with the plan v2.1 contract + the M=1 window's
diagnostic outcome as input). NOT urgent: queued behind the active
campaign + the measurement cascade; the item stays in the ledger with its
status updated. The reviewer's disposition request is answered: the bundle
is the verdict input — implementation-first, with the tracked artifact
identities (llaminar2 33804e49…, core de9c1225…, perf binary 83e1efab…)
as the historical context.

### REVIEWER CORRECTION (reviewer, ~15:20Z) — attribution + sequencing fixed

Half-perf inventory received as owner evidence; verdict remains **NOT
ADMITTED / candidate unimplemented** — not a measured FMLAL cancellation.
**ATTRIBUTION CORRECTION: the reviewer did NOT issue an "implement leaf
first" direction or approval.** The released v2.1 sequence still places
authentic production/GDN baseline evidence BEFORE separately authorized
native leaf work; the missing candidate is NOT permission to skip the
baseline, arithmetic/resource review, or the operator fence. The
coordinator may QUEUE a future design/implementation REQUEST; no
implementation/economy grant follows from the inventory. Preserved b512
source ACK + failed-gate provenance remain separate.
COORDINATOR NOTE (mine to fix): my disposition entry above said
"IMPLEMENTATION-FIRST (the reviewer's branch)" — WRONG ATTRIBUTION; the
correct statement: the inventory ANSWERS the reviewer's question (the
leaf never existed), the verdict stays NOT ADMITTED, and any future
leaf work = a coordinator-QUEUED REQUEST through the v2.1 sequence
(baseline evidence first), not a reviewer-authorized path. Headline
corrected to: half-perf = NOT ADMITTED, candidate unimplemented, future
leaf work = queued request only.

### OPERATOR GRANT: FMLAL leaf implementation authorized (coordinator accepted, ~15:30Z)

The operator granted implementation ("I allow implementation if you want
to"); the coordinator accepted and assigned the addon lane (plan-first).
RECONCILIATION with the reviewer's sequence correction: code
implementation proceeds NOW (no host needed — syntax-only until a
window), while the EVIDENCE sequence stays intact — authentic baseline +
economy measurement still precede any ADMISSION claim (v2.1 order
preserved for evidence, not for code). Reviewer verdicts the leaf when
it lands. Tracker state: candidate implementation AUTHORIZED + in
flight; economy admission FENCED on evidence. Owner: arm-addon (plan
first: FMLAL decode design, register budget, byte-auth oracle,
integration point in CPUNativeNEONPreparedProjection; base = rebase
half-perf branch onto bb893cd79 or fresh branch — lane's call, stated
in the plan).

### OPERATOR AUTHORIZATION RECORD (coordinator relay to Main, ~15:30Z demand satisfied)

Main demanded the literal operator instruction (text/timestamp/scope) for
the FMLAL leaf implementation. THE LITERAL TEXT, verbatim: **"But I allow
implementation if you want to"** — received from the operator
approximately 15:25Z (between the reviewer's 15:20:36Z attribution
correction and the coordinator's 15:30Z acceptance dispatches; operator
messages carry no broker id in this interface — the position in the
sequence is the record). SCOPE: the half-perf FMLAL leaf implementation —
the operator's grant responds directly to the coordinator's disposition
explanation (NOT ADMITTED / candidate unimplemented / queued request
only) and the addon lane's signal-gated idleness. RECONCILIATION with
reviewer204's v2.1 sequence (both hold, different axes): CODE
implementation proceeds now (addon lane, plan-first, syntax-only until a
window, NO economy claims); the BASELINE/EVIDENCE step precedes ADMISSION
— the implementation carries no economy grant and the v2.1 order is
intact for evidence. Reviewer verdicts the leaf when it lands.

### FMLAL VALIDATION #1: FAIL — wrong ACLE macro guard (Main on .80, ~21:27Z)

Container main-fmlal exited. Result: **FAIL — the leaf was never
compiled: the guard tests `__ARM_FEATURE_FP16FML` instead of the correct
ACLE macro `__ARM_FEATURE_FP16_FML`** — caught by the addon's own
selection test on .80. Main sent addon the required fixes (not applied by
Main; addon owns the source). .80 handed back to the coordinator; Main
re-books when addon sends the new tip. The plan/structure otherwise
validated (the build ran to the guard point — the failure is the
compile-time macro, caught exactly as the selection test was designed).

### FMLAL VALIDATION #3: FAIL — first real execution, byte mismatch on specials (Main on .80, ~22:26Z)

Tip 7175eca8c (container main-fmlal3, started 21:59:45Z, exited). BUILD OK;
**the leaf EXECUTES for the first time.** Result: byte mismatch on EVERY
special-operand case (±0, subnormals, Inf, NaN, ties) while ORDINARY data
matches. Separately: the ISA-profile contract test needs updating for the
new FP16FML envelope bit. Sent to addon (not applied by Main); .80 idle,
handed back to the queue.
INTERPRETATION (coordinator): ordinary-data match + specials-mismatch = a
denormal/zero/Inf handling difference between FMLAL and the scalar FMA
path (FMLAL's FP16→FP32 widening may flush subnormals differently, or the
RNE tie handling diverges at the specials) — a REAL arithmetic semantics
finding, not a harness defect. The v2.1 byte-identity requirement (the
leaf must be byte-identical to Scalar per the exactness argument) is NOT
met for specials. The addon diagnoses: which specials diverge and why
(FMLAL2's denormal behavior vs the widened-scalar path; possibly a
FZ/flush-to-zero mode difference — FPCR.FZ16 was stated explicitly in the
plan). The ISA-profile test update is a separate mechanical fix.

### **CORRECTION of the coordinator's #3 interpretation (reviewer F2):** the
earlier note "ordinary data matches + specials mismatch = REAL arithmetic
semantics finding (FMLAL denormal behavior)" is **NOT ESTABLISHED — the
specials ORACLE itself is defective** (canonicalHalfTree writes
mantissa<<13 directly into an exponent-0 FP32 word instead of normalizing
half subnormals; hand-table ties wrong; NaN/Inf cycled through full rows
so no special is isolated). The validation-#3 specials mismatch does NOT
evidence a native FMLAL bug. My attribution error = the same class the
coordinator discipline warns about (interpreting a red gate before the
harness is verified).

### FMLAL LANDING/PLAN SOURCE VERDICT (reviewer seq 213): HOLD — F1-F5

Range a1913240..311c78403 (+ the 311→7175eca8c counter/include fixes so
already-fixed findings aren't recycled). No execution by reviewer; #4
unresolved there.
- **F1 (BLOCKER): NEW production arithmetic regression** — leaf selected
  for ALL FP16 weights (:1566-1569) with NO HalfRNE||HalfStorage
  restriction; full 16-blocks always narrow FP32 A even for (F,F);
  real mixed callers (:3127-3136, :5118-5127) use (F,F); tail leaves A
  unrounded → precision rules differ by block size in ONE invocation.
  Witness: M=N=1,K=16, A[0]=1+2^-12, B[0]=half1 → prescribed 1+2^-12,
  new full block gives 1. Violates the plan's (F,F)=NOT FMLAL clause.
  Fix: restore mixed-policy arithmetic; limit the half-product to its
  operand domain; focused production regression incl. the 16-boundary.
- **F2 (BLOCKER): specials oracle wrong** (canonicalHalfTree subnormal
  widening defect; hand-table ties wrong; NaN/Inf not isolated). Fix the
  oracle + isolate finite/signed-zero/subnormal/Inf/NaN cases; do NOT
  waive exceptional bytes or rebaseline to the candidate.
- F3: FP environment/conversion proof open (FCVTN ambient rounding vs
  explicit RNE helper; no FPCR-profile admission contract). Preserve the
  exact conversion contract.
- F4: landing broadens the whole NEON_I8MM envelope (+fp16fml global +
  FEAT_FHM admission) vs the plan's TU-local proposal — reconcile
  scope explicitly; Test__CPUISAProfile.cpp:148-156 asserts the OLD mask
  (contradiction, update); don't reopen cancelled register work.
- F5: route counter ≠ actual-FMLAL-work evidence (counts FP32/BF16
  weights, invalid work, K<16 scalar tails; unconditional in the timed
  invocation). Distinguish compiled capability / route entry / executed
  work; keep optional evidence out of clean timing.
POSITIVE: bounded half/half schedule acknowledged (16 lanes/tile,
stride 256, rows 1-4, 128..1 tree); comment fix +4*b not +16*b; the
256-lane partial bank = intentional storage, not auto-spill.
DISPOSITION: implementation grant INTACT; not a withdrawal or measured
economy rejection; F1/F2 = real source blockers; Main's campaign +
evidence separate and unverified-at-lane.

### FMLAL SUCCESSOR TIP (~23:02Z): 0b53fa7a4 — F1-F5 disposition, validation #4 GRANTED

addon's repair tip on codex/arm-fmlal-half-leaf: F1 domain restriction +
the K=16 witness regression as a focused production regression; F2 oracle
repaired (proper half-subnormal normalization + corrected ties) with
ISOLATED per-class specials; F3 FPCR-profile admission at leaf entry
(fatal on non-RNE/FZ); F4 scope reconciled (I8MM-global declaration;
ISAProfile mask updated); F5 route probe counts only RESOLVED FMLAL
dispatches (FP16 domain + K>=16 at the calling thread). Local: GCC14/
AppleClang/x86 syntax 0 errors, identity PASS.
VALIDATION #4: GRANTED on .80 (Main, ~35 min, Integration NEON_I8MM,
fresh source/build, no models). STATION CO-VALIDATION dispatched in
parallel (P1 focused cells natively, cross-host byte-identity agreement
= the deliverable).

### Station FMLAL co-validation: OneDNN lock CORRECTION + worktree-local fix (r5 in flight)

CORRECTION of the earlier OneDNN note: the lock blocker DID reproduce
(r3 failed at cmake/OneDNNBuild.cmake:160 via llaminar_prepare_onednn,
fresh configure, GPU languages OFF). Root cause of the r2 confusion:
r2's log was a CACHE-REUSE reconfigure on the r1-poisoned build dir —
its cache carried USE_ONEDNN=OFF from r1's fallback, so OneDNN was
SKIPPED, never satisfied. Structural for station containers:
FloatingPointGemmKernel.h hard-requires OneDNN (#ifndef HAVE_ONEDNN
#error → USE_ONDNN=OFF cannot build this tree at all); the canonical
helper builds OneDNN INSIDE external/onednn under a file lock — the
SHARED external mount is read-only with no completed install, so the
lock can never be created there.
FIX (script v5, watcher r5 in flight): the container mounts the
WORKTREE'S OWN external/ instead of the shared one — provisioned
host-side with a pinned shallow clone of the helper's exact ref (oneDNN
v3.11.3 = 74d04752d9eaefff6a9ff62466c4d20b155e5bca, origin URL = the
helper's ONEDNN_GIT_URL; vendor/ tracked content present; shared
external/ untouched, no longer mounted). The canonical helper still
performs its own origin check, schema=3 identity, lock, build,
install-surface checks and digest — entirely inside the writable
worktree-local checkout. CMAKE_BUILD_PARALLEL_LEVEL=2 bounds the helper
build on the one-CPU quota. Provision inventory (all hash-verified):
pcre2 10.49 + utf8proc 2.11.3 (tarball SHA256s == pinned URL_HASH), ftxui
v5.0.0 = cdf28903a7781f97ba94d30b79c3a4b0c97ccce7, libfort v0.4.2 =
b1c32b67511f4612996b287a1ef4a9df012521d2, onednn v3.11.3 = 74d04752d.
Dependency versions unchanged — local transport only, same approved
pattern. Attempt-3 receipts archived (attempt3-onednn-lock/ incl. the
exact v3 script). Expect a longer run (full OneDNN + core build, ccache
cold, one CPU); incremental resume supported via the helper's
configured/identity markers. Exit + JUnit + SHAs when it reports.

### Station r5→r6: origin exact-string check resolved (transport detail)

r5 failed on the helper's origin check = EXACT string compare; git
normalized the clone's origin to 'https://github.com/uxlfoundation/
oneDNN' (no .git) → llaminar_require_onednn_origin rejected. Fixed via
git remote set-url with the .git suffix — objects/commit untouched
(74d04752d @ v3.11.3); the helper's own origin check now authenticates
directly. r5 receipts archived (attempt5-origin-spelling/). r6 in
flight, PAST the origin check, helper configure/build underway in the
writable worktree-local checkout. OneDNN + core build + two JUnit
receipts on the one-CPU quota; incremental resume via the helper's
markers if the watcher expires.

### FMLAL VALIDATION #4 (~23:30Z): specials CLEAR, one F1 residue

Main on .80 (main-fmlal4, exited; host handed back): **special values
match byte for byte with the repaired, isolated oracle** — the F2 oracle
repair + per-class isolation cleared the native FMLAL specials
semantics. **REMAINING: the F1 mixed-policy K=16 witness still returns 1
instead of 1+2^-12** — either the restriction isn't reached, or base
arithmetic already narrows; addon dispatched to determine WHICH with
evidence (route-trace vs shared-block-narrowing; the witness
distinguishes the two). Not applied by Main.

Note: reviewer-220 = the same spec ACK I received as seq-219
(SPEC-FAITHFUL design-only @ 10137aa3); delivery-ordering across
channels again. Diagnostic HOLD A1-A5 with 688b0c14b pending
re-review.

### Station r6→r7: OneDNN SOLVED structurally; legacy-path symlinks approved

r6: the canonical helper completed the FULL OneDNN v3.11.3 build+install
inside the worktree-local checkout (identity schema=3 committed:
.llaminar-onednn-commit=b6c4c39ba5481aa9f625f132cac089ab3d7b57283b7befb
d70c4279a7e69517f; installed-artifacts digest
7e93a6a78805284976a9da966cf8be0d01443ff1ca894c96ca74eb3b74c94ff2;
install_manifest present; origin authenticated; lock/surface checks
passed; worktree git-clean — external/ ignored). Self-caught measurement
error: the earlier 'build dir missing' peeks used the wrong dirname
(build-neon_i8mm vs its .llaminar.lock suffix).
r6 then failed at TESTS level: tests/v2/CMakeLists.txt:5299/:24699
reference two PRE-src/v2 layout paths at the repo root (backends/
NativeGraphDependencyReduction.cpp; kernels/rocm/gemm/
ROCmDenseProductionPrefillOverlay.cpp) which a fresh worktree does not
track (top-level backends//kernels/ only exist as leftovers in older
checkouts). FIX (script v6, APPROVED with boundary): symlinks those two
exact paths to the IDENTICAL in-tree files so generate resolves the real
production sources — nothing fabricated; targets under test untouched;
symlinks removed at run end (tree ends clean). Boundary check imposed:
byte-identity of the symlink targets confirmed by SHA before linking;
if NOT identical, STOP and report.
r7 launched with OneDNN reuse: configure + core build + canonical
V2_Unit_CPUHalfOperands + the four P1 cells with JUnit.

### Station r7→r8: legacy-path audit + generalized transport APPROVED

r7: both symlinked targets resolved, configure ran to completion (4.1s),
generate failed on ONE more legacy path (tests/v2/CMakeLists.txt:4821 →
app/MPIProcessSession.cpp). FULL AUDIT: 27 ${CMAKE_SOURCE_DIR} source
references in tests/v2/CMakeLists.txt, 25 missing in a fresh worktree
(app/, backends/, kernels/, collective/, execution/, models/ legacy
roots) — EVERY one has an identical tracked counterpart at
src/v2/<same path> (audit receipt kept).
v7 transport (APPROVED): for each missing reference, symlink to the
identical tracked in-tree file; REFUSES (exit 91) any reference without
an existing counterpart; exit 92 on any dirty target; SHA256-records
every target in legacy-transport.txt; removes all symlinks + created
dirs on EVERY exit path. No CMakeLists edit; nothing fabricated.
SCOPE HONESTY: generate-only plumbing — the P1 target's TUs are all
current src/v2 tracked sources; the legacy-referenced targets are never
built/run in the station worktree. Cross-host caveat archived: if .80's
long-lived checkout holds STALE pre-migration files at those legacy
paths, its legacy-referenced targets would compile different bytes —
irrelevant to the P1 cells, noted for future cross-host audits.
r8 in flight with OneDNN reuse: configure + core build + canonical
V2_Unit_CPUHalfOperands + four P1 cells with JUnit.

### Station r8→r9: awk field bug in the v7 check (self-caught, transport correct)

r8 exited 92 because the tracked-clean check read the WRONG awk field:
$2 (the LINK path, an untracked symlink) instead of $4 (the TARGET
path) — git status ran on the symlink and tripped the refusal. The
transport itself was CORRECT: all 27 references linked with
SHA256-recorded targets (legacy-transport.txt in the r8 archive); every
TARGET verifies tracked-clean on the host (empty git status, ls-files
confirms tracked). One-field fix applied ($2→$4, message names the
actual target); cleanup had already removed all symlinks (tree clean).
r8 receipts archived (attempt8-check-field-bug/). r9 relaunched with
the corrected check — next report = the real one: exit + JUnit + SHAs.
NOTE: second self-caught harness defect tonight (dirname peek, awk
field) — both caught by the station's own verification before they
could contaminate a receipt.

### OPERATOR DECISION (~00:20Z): "FMLAL accept" — FP16RNE narrowing-by-contract ACCEPTED as baseline

The FMLAL leaf's contract = **byte-identical to base on EVERY route**:
- FP16RNE route: the base scalar's narrowing of FP32 A at every k is the
  documented HalfRNE contract — pre-existing behavior, NOT the leaf's
  doing; the leaf matches base there (no candidate regression).
- (F,F) mixed route: the F1 narrowing defect lived HERE (GPUAlignedExpert
  FP32-A/FP16-B); fixed by the domain restriction; new tip 67e8856c9
  targets this route with the 1+2^-12 witness expectation preserved.
**Validation #5 = the final gate** (byte-identity both routes + the K=16
witness + isolated per-class specials). Sequencing: addon rebases the
FMLAL branch onto dd8f2bd96 BEFORE #5 (validation + the reviewer's
landing verdict pin the rebased tip in ONE cycle — verdicts do not
transfer across the rebase), then #5 books after campaign #2 + W3.
Collection candidate: rebase → #5 PASS → handback → Main applies.

### FP16RNE SCOPE RESOLVED (addon code-reading, ~00:23Z): policy-level, X86-AND-ARM

The operator asked whether FP16RNE is ARM-specific or also x86. Answer
with file/line evidence:
- The intercept is ARCHITECTURE-BLIND: `if (operand_precision_ ==
  FloatingActivationOperandPrecision::FP16RNE) return
  executeHalfOperands(...)` (FloatingPointGemmKernel.h:2722 at base,
  ~:3102 at the current tip) routes EVERY FP16RNE engine before any ISA
  switch.
- x86 instantiates the SAME narrowing: AVX-512
  `_mm512_cvtph_ps(_mm512_cvtps_ph(activation,
  _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC))` (:908, F16C, explicit
  RNE) inside the AVX-512 stripe schedule; analogous AVX2 form; scalar
  fallback via `HalfRNE ? roundFloatingOperandHalfRNE(value)` (:1655).
- Identical rounding semantics across architectures (FP32→binary16 RNE);
  products run as FP32 FMAs of exactly-widened halves — the same
  arithmetic class FMLAL implements on ARM.
**CONCLUSION: the operator's "FMLAL accept" narrowing-by-contract
ruling is X86-AND-ARM (policy-scoped, FloatingPointGemmKernel-wide).
Any FP16RNE engine on either architecture narrows A to binary16 RNE by
contract; any future change to that contract = a cross-architecture
policy decision.**

### STATION FMLAL CO-VALIDATION COMPLETE (~00:30Z): 3/4 PASS, cross-host agreement on the K=16 defect

Station receipt at tip 0b53fa7a4 (verified vs origin; Grace aarch64,
GCC 14.2.0, envelope -march=armv8-a+simd+dotprod+i8mm+fp16fml,
FEAT_FHM admission PROVEN by the selection test's invocation
assertion):
- KTailBoundariesAllRowsMatchSerial: PASS (K 255..767 × m 1-4 × workers
  1,3; leaf increments exactly 36/proveLinear; repeated 2,257,3
  byte-identical).
- FmlalSpecialOperandsMatchOwnedTree: PASS (isolated per-class
  specials incl. NaN payloads; bytes = the test-owned fixed-256-lane
  tree).
- FmlalLeafSelectionMatchesEnvelope: PASS (compiled-envelope ↔ FEAT_FHM
  admission; K=65 probe = exactly 1 leaf invocation; FP16-without-FML
  admits, FML refuses without FHM).
- **MixedPolicyKeepsUnroundedFP32AtK16: FAIL — output exactly 1.0f,
  expected 1+2^-12; deterministic across two runs.**
**CROSS-HOST: .80's #4 at the SAME tip failed the SAME way (Main's
verbatim finding) → NO divergence; the K=16 defect is real and
host-independent; the repair tip 67e8856c9 targets exactly it.**
Validation #5 at the rebased tip should clear it on both hosts; a
station P1-cell re-run at the rebased tip is queued for cross-host #5
agreement.
CTEST-ENTRY LIMITATION (honest): the canonical V2_Unit_CPUHalfOperands
entry is registered but the focused build cannot execute the session
wrapper; direct binary runs = same binary + registered filter.
Binary SHA 1290f6d0…; JUnit + artifact SHA256s archived under
/home/tdsynnex/repo/llaminar-perf-fmlal/artifacts/fmlal-copilot/.
Station reusable assets: worktree-local OneDNN v3.11.3 install
(identity b6c4c39b…, digest 7e93a6a7…) + legacy-transport v7 (27 refs,
refusal semantics armed).

### FMLAL PRE-REBASED ONTO ebd66f0cb (addon, ~00:50Z): new local tip 45e9b6690, HOLDING THE PUSH

Addon had NOT rebased onto dd8 (reset to 67e8856c9 per the correction);
on Main's ebd66f0cb announcement they rebased directly:
`git rebase --onto ebd66f0cb 12675c956 codex/arm-fmlal-half-leaf` — all
8 FMLAL commits replayed CLEANLY (no conflicts; the endif fix is outside
their touched files). New local tip **45e9b6690** on ebd66f0cb; chain
7a225437e → 20ba683ab → c805dd372 → a34adb6df → 313dd4bce → 1d37d3f5d →
cf739375b → 45e9b6690 (old-SHA equivalents a1913240d..67e8856c9 replayed
1:1). LOCAL VERIFICATION on the rebased tree: GCC14 syntax 0 errors,
ISA identity contract PASS, NEON portability configure/build + 18/18
focused cells green; worktree clean. HOLDING THE PUSH until Main
announces PR #43's force-push (naming ebd66f0cb or its successor); on
that: push → report tip → validation #5 booking → landing verdict pins
the rebased tip (one cycle). Station cross-host #5 re-run follows.

### FMLAL PUSHED ON THE PR BASE (~02:21Z): tip b5e928e40 on 74797687f

The 8-commit chain rebased onto 74797687f CLEANLY (no
REQUIRES_BACKEND/registration conflicts; the deferred gate tooling
correctly absent from this line). Local verification on the rebased
tip: GCC14/AppleClang 0 errors, identity PASS, NEON 18/18.
**VALIDATION #5 BOOKED: top of the .80 queue after the in-flight B1
export** — Integration NEON_I8MM build of b5e928e40; the full
v2_test_cpu_half_operands binary (byte-identity BOTH routes incl. the
GPU-aligned K=16 witness + isolated specials + the selection/invocation
probe) + V2_Unit/Integration_CPUInstructionSetContract; raw JUnit to
Main + me. Main's locked intent: #5 PASS → reviewer re-pins the landing
verdict → apply (FF — b5e928e40's parent IS 74797687f) → push + PR body,
no economy claim. Station cross-host #5 re-run follows on the same tip.

### FMLAL VALIDATION #5 (~02:47Z): FAIL — K=16 witness throws on an INVALID MODE COMBINATION

Main on .80 (main-fmlal5, exited; host handed back → B1 export can go):
- The K=16 regression **throws before multiply: it uses the invalid
  Mode::FP32+FP16 weight combination — the validator rejects it, as
  reviewer 232 predicted.** The witness was constructed with an
  unreachable mode combination, not the (F,F) GPUAlignedExpert path the
  production callers use.
- The specials, selection and ALL OTHER half-operand tests PASS.
- Reviewer 233 findings: (1) **x86 compile break — unguarded
  ARMCPUFeatures.h include** (ARM-only header; needs a per-arch guard —
  the D6 arch-split class); (2) **wrong FZ16 bit — bit 26 is AHP; FZ16
  is bit 19** (the FPCR-profile admission tests the wrong bit).
- Sent to addon (not applied); #6 after the fix tip on 74797687f.

### FMLAL LANDING REVIEW (~02:49Z): SOURCE HOLD at b5e928e40 — L1-L5

Fresh full-range static verdict (74797687f..b5e928e40, six files
+628/-11; no execution by the reviewer):
- L1: x86 compilation regression — Test__CPUHalfOperands.cpp:15
  unconditionally includes ARMCPUFeatures.h (#error on non-AArch64);
  ARM-only types unguarded at :411-445; the target registers on x86
  (:30367). Guard include/body; retain portable half correctness.
- L2: the F1 regression cannot reach the repaired route — :402-403
  constructs FP16 weights + Mode::FP32 (rejected at base :5208-5210);
  the real (F,F) route = GPUAlignedExpert + Mode::Native; descriptor
  [N,K]={1,16} not {16,1}; don't weaken the storage validator.
  (= Main's #5 throw finding, confirmed at source.)
- L3: kFZ16=1ULL<<26 is WRONG — FPCR.FZ16 is bit 19; bit 26 = AHP; an
  actual FZ16-only word (0x80000) passes the FZ16-off admission. Fix
  the bit; preserve AHP rejection under its correct name; focused
  profile controls with this contract.
- L4: F4 process admission incomplete — CPUFeatures.h:208-245 admits
  only DotProd/I8MM before overrides; native()/observe() only require
  ASIMD; requireCompiledEnvelope() not wired into the ordinary ISA
  resolver; FHM checked only at the FMLAL dispatch :1836 → not complete
  startup admission for a globally-enabled feature that may execute
  outside this leaf. Bind the envelope authority into ordinary
  admission incl. explicit scalar selection.
- L5: F5 partial — the counter still increments before pointer/
  publication validation (:1459-1469) and live-span checks (:1483-1487);
  unconditional relaxed atomic in every eligible call; tests derive
  expectations from the COMPILED envelope (resolved-Scalar legitimately
  executes 0). Separate capability/attempt/admitted-execution; resolved-
  route expectations; evidence out of clean timing; stale comments.
ACKNOWLEDGED: F1 guard, F2 oracle+ties+specials, 256-lane chains, '+w'
constraint, FHM observations, I8MM-global scope + mask — source
acknowledgments, NOT a blanket byte/runtime certificate (the finite-255
background rounds away the smallest-subnormal contribution in some
special cases).
DISPOSITION: implementation authorization intact; economy NOT ADMITTED;
focused controls = lane obligations; broader checks = Main's campaign.

### FMLAL STATE (~02:50Z): 6c7a1a477 covers L1-L3; ONE combined tip for L4+L5 required; #6 HELD

Main: reviewer 234's HOLD at b5e928e40 (L1-L5) — addon's 6c7a1a477
ALREADY ADDRESSES L1-L3 (pushed quickly after Main's #5 + reviewer 233).
L4 (FHM not in ordinary startup ISA admission) + L5 (counter counts
unexecuted dispatches; unconditional atomic; tests key on the compiled
envelope not the resolved route) remain open — Main requires ONE tip
covering both and HOLDS #6 until then. .80 shows container tender_morse
(presumably the B1 export; sparks pressed for confirmation + the record
answer). #6 books when the combined tip arrives.

### FMLAL #6 RUNNING (~02:58Z): main-fmlal6 at f987501f9, focused + full canonical, ~45 min

tender_morse exited; .80 empty → Main started #6 (campaign-src-fmlal6,
tree 9d9f020b3). Scope: focused cells PLUS the complete canonical
Unit+preflight (L4 changes startup ISA admission). .80 held by Main
until exit. Post-#6 .80 queue: W3 (B4 Integration 52fec643f) → 27B
rider. B1 export done pending the seal.

### FMLAL #6 PRE-VERDICT (~03:05Z): SOURCE HOLD at f987501f9 — H1-H4 (H1/H2/H3 = source-predicted #6 failures)

Fresh exact-tip verdict (full landing identity 74797687f..f987501f9, 7
files +755/-11 + the b5 successor delta; no execution):
- H1: the selection fixture never constructs FmlalProbe (defaults false;
  KTail/Specials restore false) → counts 0 vs :495 expecting 1 on the
  native route. Source-predicted #6 failure; give the positive control
  its own evidence scope.
- H2: the FPCR predicate compiles real only under __ARM_FEATURE_FP16_FML
  (else returns true for every word) while :512-516 requires five false
  results under __aarch64__ → ordinary NEON/DOTPROD builds fail all
  five. Pure integer policy must stay truthful; no always-admit stub;
  include the RMode=10 discriminator.
- H3: :1486 admits N=0 (evidence enabled → live-row found, increment 1,
  column loop zero = no FMLAL work) → N>0 in the predicate; the
  live-span scan runs BEFORE the enabled check → move it behind.
- H4: requireCompiledEnvelope() outside the cached initializer (:274
  caches only resolveCPUISALevel()) → every activeISALevel() call
  repeats full admission (all ISA_DISPATCH macros use this hot getter).
  Bind into the cached path, retaining Scalar coverage.
CLOSED: L1 guarded; L2 the real route + [N,K]={1,16}; L3 FZ16=19/AHP=26
on the executing worker; the 2^-24 weight-side case pins 0x33800000;
L4 functionally repaired (subject to H4); the L5 improvements accepted.
NONBLOCKING: x86 arm wording → "no FMLAL" (x86 can run AVX2/AVX512).
Main owns #6; implementation authorized; economy NOT ADMITTED.

### FMLAL #6 RESULT (~03:39Z): 9 failures = 6 inherited + EXACTLY the 3 H1 aliases — predictions confirmed

#6 at f987501f9 (focused + full canonical): L4 caused no other
regressions (the startup-admission change is clean beyond the
predicted fixture failure). **#7 at 14888fef3 (H1-H4 combined)
running** — container main-fmlal7, started 03:38:40Z, ~45 min; .80
held by Main. Reviewer re-review requested at 14888fef3. Landing
cycle: #7 PASS → verdict re-pin → FF-apply → PR commits.

### FMLAL #7 PRE-VERDICT (~03:48Z): SOURCE HOLD at 14888fef3 — H1-H4 VERIFIED FIXED; J1-J3 in the new fixture

H1-H4 source-verified: FmlalProbe owned (:493); the FPCR predicate
covers every AArch64 build incl. RMode=10; evidence gated by
K>=16,N>0,M>0 + enable BEFORE the live-span walk (:1514-1527);
complete-envelope admission INSIDE cached initialization incl. Scalar
(CPUFeatures.h:276-288); x86 wording fixed; the 2^-24 case executes
both calls. No further arithmetic/routing change requested.
REMAINING (FmlalRouteProbeDiscrimination fixture only):
- J1: probe lifetime wrong (alive for the whole test → off-arms
  enabled, each native multiply counts 1 vs 0) — scope the enable
  between the off arms.
- J2: zero-dimension controls use the wrong API (zero-N returns false
  via validHalfOperands; m=0 THROWS through currentProjectionRowWork(0))
  — keep public admission; discriminate the primitive counter at the
  production boundary that permits zero M/N; separate public-rejection
  from primitive-counter controls; positive-capacity/all-inactive
  control still absent.
- J3: resolved-Scalar in I8MM → after_on=0 → unconditional
  EXPECT_GT(after_on,0) fails; GTEST_SKIP ≠ coverage — build a real
  no-FMLAL control via the no-op probe + zero-count interface.
EXPECTED #7 from source: :538 fails, then :553 fails fatally.
DISPOSITION: Main owns #7; implementation authorized; economy NOT
ADMITTED; B4 pending; B9 closed.

### FMLAL J-FIX TIP + RUNTIME GATE (~04:07Z): 67afc62ea → station compile+run

Addon's J1/J2 successor tip **67afc62ea** (probe scoping, zero-N/null-A
public-rejection checks, resolved-route keyed positive-live assertion,
the all-inactive-span control via a real ProjectionRowWorkScope) maps
1:1 to Main's #7 findings. **RUNTIME GATE dispatched to the station**
(Grace GCC 14.2): incremental rebuild (warm OneDNN + legacy-transport
machinery) → the full v2_test_cpu_half_operands binary (all cells incl.
FmlalRouteProbeDiscrimination) → JUnit + exit + SHAs to coordinator +
addon. Functional truth only; same envelope/discipline. Cross-host PASS
at the J-fix tip completes the FMLAL evidence chain for the landing
cycle. Idea-04 M=1 window queues behind it.

### FMLAL #7 RESULT (~04:21Z): J1/J2 predictions confirmed — 6 inherited + 3 fixture aliases

#7 at 14888fef3: the 6 inherited + 3 CPUHalfOperands aliases, ALL from
the new FmlalRouteProbeDiscrimination fixture (probe scope wrong —
off-cases ran enabled; public zero-N asserted success where admission
rightly rejects). Everything else passes: specials, K=16 witness,
selection, H4 caching, no regressions. **Main's directive to addon:
RUN the half binary natively on ARM Linux before the next push**
(build-only checks keep missing runtime fixture bugs). The station's
runtime gate at 67afc62ea is in flight; #8 after the runtime evidence.
.80 free → W3 + 27B rider next in the queue.

### FMLAL J2-SUCCESSOR STATE (~04:25Z): 2af1e08cf adds primitive controls; P5 (positive-capacity all-inactive span at the primitive) still missing — native runs held

Main: 2af1e08cf adds the primitive controls but one 242 item is
missing — the positive-capacity all-inactive span control at the
primitive (P5, fixture-only). Both native runs (station I8MM, .80 NEON)
held for the P5 tip. Sequence: P5 fix → push → both runs pin → runtime
handoff → reviewer #209 → landing cycle.

### FMLAL GATE TIP RELEASED (~04:27Z): e8c5bd505 — both native runs pin to it

P5 landed (fixture-only: the positive-capacity all-inactive-span
control runs AT the primitive via a real ProjectionRowWorkScope with a
zero-count device range; selection spans empty; live-span scan finds
nothing; count 0; output sentinel-pinned). Gate tip = e8c5bd505de86bc
710e2839a3e9a82cb5017fe64 (13 commits on 74797687f). RELEASED:
- Station: I8MM build + the v2_test_cpu_half_operands binary (the
  three-run script re-run at the new tip = the ACTUAL gate).
- .80: NEON-envelope build (-DLLAMINAR_CPU_ISA=NEON, Integration
  GCC14) + the same binary (Main/sparks).
Raw gtest XML/logs + exact tree + build exit + binary exit to Main +
addon + the reviewer. Green both → Main's #8 (full I8MM canonical) +
the reviewer verdict → application → PR commits. Idea-04 M=1 queues
behind the landing.

### FMLAL GATE-TIP PRELIMINARY FINDING (~04:28Z): P1-P5 still bypass the FMLAL counter — two fixture bypasses

Reviewer static source at exact e8c5bd505: all five P-calls instantiate
run_gpu_aligned_expert_fp32_matmul<FP16> with DEFAULT HalfRNE=false/
HalfStorage=false (production correctly excludes (F,F) from FMLAL) AND
the prior public-controls FmlalProbe has destructed (no probe in the
primitive block despite its comment). **Two independent bypasses: the
intended <FP16,true> domain is not instantiated, and no enable scope
spans P1-P5.** P4 predicts 0 vs expected 1 on native I8MM/FMLAL; plain
NEON can pass while the negatives stay nondiscriminating. P5 reaches
the row selection but not the FMLAL evidence scan. FIX = fixture-only;
DO NOT relax the production domain guard. The in-flight native runs may
confirm empirically (P4 failing on I8MM); the reviewer's final verdict
waits for the fixed fixture + runtime handoffs.

### FMLAL LOOP BROKEN (~04:28Z): direct .80 iteration granted to addon

Reviewer 249: P1–P5 use the (F,F) template domain (excluded from FMLAL
by production) + no probe enabled → P4 will fail on I8MM; the gate
re-pins to the next tip. **Main's loop-breaking suggestion adopted: a
short .80 slot is GRANTED to the addon for DIRECT iteration** — build
v2_test_cpu_half_operands at their fix tip(s) on aarch64 Linux, run,
fix, rebuild, rerun until GREEN under BOTH envelopes (NEON + I8MM),
then push the green tip + hand back. No seals for the iterations; the
formal gate runs (Main I8MM-canonical #8 + the station re-run) pin to
the green tip after handback. The fixture-only constraint stands (the
production domain guard stays). This kills the push→validate round-trip
latency that five fixture rounds have paid.

### FMLAL GATE RE-PIN x2 (~04:32Z): 532fec60f also misses K1 (P5 still on (F,F)) — pin to the next tip

Main's check: P5 still instantiates the default (F,F) <FP16> domain
(K1 requires ALL of P1-P5 on <FP16,true>) — a one-token addon fix.
BOTH native runs (station I8MM + .80 NEON) pin to the tip AFTER that
fix; the .80 NEON slot stays held; #209 waits for the corrected tip's
handoff. The direct-iteration grant covers the fix (addon builds+runs
the half binary itself until green under both envelopes).

### FMLAL GATE SCHEDULE FLIP (~04:34Z): Main #8 on .80 (I8MM) + station NEON in parallel

Gate tip 0ad09600b: Main verified ALL 5 primitive calls <FP16,true>
with per-arm probe scopes (K1 finally closed after two review rounds
missed P5). Schedule flip (Main's proposal): **Main runs #8 NOW on .80
(I8MM focused + full canonical = the native I8MM run, Main-
authenticated); the STATION runs the NEON-envelope build + binary at
0ad09600b in parallel** (different host, no .80 overlap) — replacing
the previous station-I8MM plan. W3 (B4) yields .80: not-started →
holds; mid-build → finishes then #8. Green both → the reviewer's #209
+ application → PR commits.

### FMLAL #8 RUNNING (~04:34Z): main-fmlal8 at 0ad09600b (tree ffa6a156a), started 04:33:26Z

.80 was empty → #8 running (I8MM focused + canonical, ~45 min); .80
held by Main; W3 held until handback. Station NEON gate at 0ad09600b
in parallel. Post-#8 .80 queue: W3 → 27B rider → B1ATTR2 state check.

### STATION INTEGRITY EVENT (~04:35Z): foreign container strange_euclid on the FMLAL worktree

A container (llaminar-dev, network-none, RW-bound to the FMLAL
worktree) appeared running an src/v2-direct diagnostic configure with
an Error-grep — matching the addon's direct-iteration grant pattern.
The station did NOT interfere (correct): let it self-terminate (stuck
in the pcre2 network-none retry loop, dies in minutes), verify the
scratch dir gone, then launch the NEON gate (one-CPU sequencing).
Addon ownership query dispatched. PROCESS NOTE: the addon's
direct-iteration grant was .80-scoped — station usage must be
coordinated with the station's one-CPU slot discipline explicitly.

### FOREIGN CONTAINER UNATTRIBUTED (~04:35Z): addon disowns; CB lead queried; NEON gate proceeds

strange_euclid = UNATTRIBUTED: not the addon (no station launches; Mac-
only verification; .80 grant unexercised), not the station peer (all
their containers llaminar-* named). Candidates: a CB/batch-inference
session probing the OneDNN failure pattern (queried — read-only actor
reconciliation), or other station-side automation. The station
proceeds with the NEON gate once the host is clean; the no-other-
containers-on-my-worktree request is granted while gate receipts are
produced. Escalation to Main/operator if it reappears after removal.
No evidence impact (worktree clean; scratch untracked; self-terminated).

### FOREIGN CONTAINER RECONCILIATION (~04:36Z): CB lead disowns; c527 queried; forensic ask

CB lead: their session did NOT launch it (recent station work = read-only
SSH/SCP in CB c1; no automation retained); CB authorization explicitly
excludes ARM/FMLAL worktrees; they cannot attest for the DGX worker
without checking (check in flight). c527 queried directly (read-only).
The station asked to PRESERVE docker inspect output (ID, times, command,
mounts, labels) if the container record still exists — the forensic
record for the origin question. If no CB origin: escalates to Main/the
operator as a station-integrity question. No evidence damage; the NEON
gate proceeds.

### FOREIGN CONTAINER ORIGIN RESOLVED (~04:37Z): the CB DGX worker (c527), on a cross-lane request

The actor = the CB DGX worker (c527), acting on a request received
through their intercom channel PURPORTING to be from the arm-addon
lane ("fetch edccdec00, reset the fmlal worktree branch, run
v2_test_cpu_half_operands in my bounded container"). Full accounting:
their own scratch rm -rf only; the stash + branch reset (04:28-04:29)
preserved the pre-reset state; the diagnostic configure self-terminated
on the pcre2 network-none exhaustion; no source files touched. c527
committed to no further fmlal worktree touches without explicit
in-scope direction. **THE REAL FINDING: a cross-lane worktree-mutation
request reached the WRONG worker through the shared broker —
impersonation/misattribution, cross-wired delivery, or a forwarded
conversation. The addon disowned the request. This is a
messaging-layer routing/attribution failure, not a worker-discipline
failure — flagged to Main + the operator.** The station's NEON gate
proceeds (worktree clean).

### NEON GATE LAUNCHED (~04:38Z): host cleaned, watcher audit rules out the station orchestrator

strange_euclid self-terminated; its scratch removed; tree verified
clean at 0ad09600b; the station's watcher state audited ("No watchers"
→ their own orchestrator ruled out as the foreign actor). New
integrity observation: `thirsty_benz` in Created state (random name,
never started, origin unknown) — possibly the same actor's failed
first attempt; logged for the CB-lead/Main reconciliation (the CB
worker's session log would confirm or refute). NEON-envelope gate
expectations: FP16FML=0 → the leaf-less control — probe-enabled-but-
route-not-FMLAL counts of 0 = PASS (the no-FMLAL control the reviewer
demanded). A separate OneDNN build-neon identity via the canonical
helper (flags-bound, disclosed). Three-run receipts + exit + tree SHAs
to coordinator + Main + addon when it lands.

### INTEGRITY FORENSICS + GATE CORRECTION (~04:38Z)

(1) c527's branch reset to edccdec00 landed 04:29:30Z DURING the
aborted r3 build; the 67→edcc diff is test-file-only (production
sources identical → core objects not contaminated), but r3's
incremental state = UNTRUSTED (any future I8MM run there
reconfigures/rebuilds). **c527's stash preserved as incident evidence
(NOT dropped) for the operator's review.** Forensics:
station-integrity/strange_euclid-forensics.txt (SHA 816d725f…) with
inspect/top/network-none facts + the thirsty_benz observation.
(2) First NEON launch failed instantly on a wrapper path slip
(exit 127; archived attempt15-wrapper-path); the NEON gate NOW RUNNING
as watcher fmlal-neon-gate-r2 (22ea42db0): reconfigure → NEON build
(separate OneDNN build-neon identity) → three JUnit runs at 0ad09600b.

### FMLAL SOURCE CLOSURE (~04:39Z): K1/K2 source-closed at 0ad09600b — no remaining blocking source finding

The reviewer's exact-tip source pass: ALL five primitive calls
explicitly <FP16,true>; FmlalProbe :578 covers P1-P5 and retires before
the restored-off arm; P1/P2 legal zero-M/zero-N handling; P3 null-
rejection before instrumentation; P4 = live positive control keyed to
ACTUAL RESOLVED FMLAL eligibility; P5 = positive dims + real zero-count
ProjectionRowWorkScope + enabled evidence reaching the live-span scan;
off/on/off + public-rejection + no-FMLAL/Scalar separated; no
GTEST_SKIP. Successor = fixture-only (+12/-8 from e8); production/
admission/arithmetic/CMake unchanged from 14888. **Static source
closure — NOT a claim #8/station passed; no authentication; no landing
instruction; economy NOT ADMITTED; #209 open for the native handoff.**
Chain: #8 (running) + station NEON gate (running) → greens → #209 →
application → PR commits.

### FMLAL #8 GREEN (~05:26Z): the I8MM gate CLOSED — one gate remains (station NEON)

#8 at 0ad09600b: half binary 12/12, focused 19/19; canonical 3532
tests, 0 skipped, 6 failed = the established inherited/topology set
(identical to campaign 747). **ZERO FMLAL-introduced failures.** .80
handed back → W3 GO (B4 Integration at e09903875). Remaining FMLAL
gate: the station NEON-envelope run at 0ad09600b (running) → route its
XML/exit/tree to Main + the reviewer → reviewer #209's verdict →
Main FF-applies → PR commits.

### #8 I8MM HANDOFF RECORDED (~05:32Z): bounded handoff accepted; NEON-compiled receipt still required

The reviewer read Main's fmlal8 records without execution/authentication:
half 12/12, focused 19/19, canonical 3532/6-failures (names matching
Main's list; the inherited/topology cause remains Main's comparison).
**Bounded handoff accepted at the owner-reported 0ad09600b/main-fmlal8b
identity — NOT a complete green prerequisite receipt, economy evidence,
or an application grant. Source closure stands. #209 still awaits the
station's ACTUAL NEON-COMPILED handoff — the NEON-labelled runtime
alias inside Main's I8MM build does not substitute for that compiler
envelope.** The station's NEON gate receipts (raw XML + exit + tree,
NEON-compiled) close #209 → landing.

### FMLAL NEON GATE GREEN (~05:47Z): the cross-envelope gate pair COMPLETE at 0ad09600b

Station NEON gate: FMLAL_NEON_GATE_EXIT=0; all three direct binary
runs GREEN:
1. Canonical filter 11/11 (FmlalRouteProbeDiscrimination's zero-N arm
   fired the expected public-admission rejection at :1488; selection/
   probe = ZERO counts with the probe enabled = the correct leaf-less
   control outcome; FPCR negatives passed).
2. P1 cells 4/4 (K-tail, specials oracle, the K=16 mixed-policy witness
   now HOLDING the unrounded FP32 result, selection/envelope).
3. Complete unfiltered 12/12.
ENVELOPE: -march=armv8-a+simd exactly; fp16fml in 0 compile commands
(= leaf NOT compiled) — an ACTUAL NEON-compiled receipt (the reviewer's
compiler-envelope requirement satisfied). OneDNN build-neon identity
separate + flags-bound + origin-authenticated. Receipt SHAs: binary
0572753c…; JUnit b5ebc1de…/b23ebc7d…/de37c2d6…; the cosmetic grep flaw
disclosed + corrected extract alongside.
**Cross-envelope gate pair COMPLETE: Main's native I8MM (#8 GREEN) +
the station's native NEON (GREEN, NEON-compiled). The FMLAL collection
candidate's evidence chain is CLOSED pending the reviewer's #209
verdict → Main FF-applies → PR commits.**

### NEON GATE ORIGIN CONFIRMED (~05:48Z): the ARM station session; raw files relayed for #209

Main's attribution caution (post-c527) answered: the NEON gate report =
OUR station session (the same peer behind every FMLAL station artifact
tonight — r1-r3, the provision chain, the co-validation; consistent
watcher IDs + artifact paths + container patterns). The c527 incident
was a different failure mode (a cross-lane REQUEST routed to the wrong
worker), not a report-authenticity question. Origin CONFIRMED; the
station relays the raw full-junit.xml + result.txt + neon-gate-exit.txt
directly to Main + the reviewer for the #209 authentication. On the
verdict: Main FF-applies 0ad09600b → PR commits.
