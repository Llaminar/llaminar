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
