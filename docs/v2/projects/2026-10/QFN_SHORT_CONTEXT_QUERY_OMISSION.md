# Retained QFN graphs without unnecessary indexer launches

Status: scoped accuracy, lifecycle and matched Release gates green,
2026-10-25. The no-ranking program physically omits empty indexer launches.
Final combined-source performance is 819.673 prefill / 43.453 decode tok/s,
against 820.881 / 43.268 for its control. Decode improves 0.426% in median,
with overlapping samples; prefill changes -0.147%. No MTP speedup is claimed.

## Intended behavior

At generation admission, choose one complete retained decode program:

- **Ranking possible:** preserve the complete indexer program, including a
  request that starts below the boundary and can cross it later.
- **Proven no ranking:** omit the indexer query projection, query preparation
  and pool scoring for the entire admitted response window. Keep raw keys,
  pooled-key updates, selection/count publication, attention and state commits.

Use the existing QSA geometry constant: ranking first occurs at visible token
count 2,052. For each request, let P be its authenticated initial target-cache
extent and B its existing response budget. Conservative maximum visibility is
P+B ordinarily, and P+B+Dmax+1 for MTP. Dmax is maximum executed draft depth,
including dynamic/observe modes. Verifier execution can overshoot the remaining
response budget even when publication is clipped. Every request must be bounded
by 2,051 before choosing omission. Check arithmetic and reject invalid input.
An intentional one-row forced transaction can instead prove P+1 from supplied
authenticated initial-count metadata. Its zero response budget alone proves
nothing. This lets the canonical teacher-forced parity panel exercise the
optimized graph without a diagnostic arithmetic override.

The current 512-prompt/256-response ordinary workload has bound 768. Its physical
context capacity remains 4,096; changing capacity is not this optimization.

## Ownership and implementation

1. Extend existing `DeviceGenerationAdmissionRequest` with typed immutable
   initial-visibility metadata, issued only at the runner's completed-prefill or
   authenticated terminal-continuation boundary. Full/partial prefix restores
   contribute the complete restored-plus-new history. Reuse existing response
   budget and depth policy in one bound resolver. No extra live counter or
   physical-memory ledger is introduced. A new response window gets a new bound.
2. Carry a closed query program through existing `ForwardInput`,
   `ForwardGraphSignature`, graph configuration and the declarative QFN builder.
   Request-specific numeric budgets remain admission data; they must not create
   a new topology key for each request. Embedded pointers, geometry and immutable
   program still belong in capture identity.
3. Retain both forward programs through the existing forward cache. Extend the
   existing generation-parent owner to retain both corresponding complete parent
   families, including admitted MTP geometries. Select only at idle admission.
   Parent ownership now has a closed slot for each program; each slot authenticates
   its exact child signature. Changing only a stage signature would have
   invalidated the singleton parent and caused recapture.
   Setup seals both model-forward programs. The complete generation parents
   require the actual sampling law, so idle sampling-law admission composes
   both before the first launch and retains them afterward. Do not invent a
   setup sampling law or fake an active admission. Reset and teardown visit
   both. The shared predictor uses dense attention and keeps its existing
   sidecar; only its paired main-verifier identity needs a program-indexed slot.
   Native PP also retains both existing forward-only families when an original
   proof selects the shorter program. Raw forwards without proof stay complete.
4. Price additional native graph owners through the existing typed captured-graph
   BOM and PMA. Mutually exclusive families share existing activation/workspace
   storage. Do not add independent reserves, duplicate banks or a graph framework.
5. The QFN builder physically omits the indexer query projection and its exact
   dependency. The indexer stage omits query-only input waits, queryNorm weight
   binding, prepared-query/score output publication and associated snapshots.
   The core attention query is a different value and remains required.
6. CPU, CUDA and ROCm consume the same explicit no-ranking program. Complete
   grouped GPU preparation fuses required pool work with query work: select the
   existing pool-only kernel, then the existing selector. Skip query and scoring
   dispatches. Preserve required pool arithmetic, raw-tail retention, acceptance,
   counts and state updates. Validate actual device replay against the selected
   no-ranking invariant before writes; never recover by executing another path.
7. Initial scope is **decode**. Prefill retains its current AllLive diagnostic
   authorship. Setup/PerfStats records identify program, phase and participant;
   actual graph inspection proves physical node omission. No per-token host
   decision, count download, eager execution or threshold-triggered recapture.

CPU and composed host responses use a lexical immutable admission scope; nested
decode calls borrow it and every exit expires the selection. A CPU MPI follower
may admit each existing public decode command using its authenticated host
extent and supplied remaining budget. It does not maintain another progress
counter. Expert-only participants have no sparse consumer; dense predictor
sidecars are shared. Neither is assigned unnecessary duplicate graphs.

The current source seams are the runner's completed-prefill initialization and
generation admissions, DGO resident-forward/parent materialization, the existing
forward signature/cache, `Qwen4ExpSparseAttentionBuilder`, `QwenSparseAttentionStage`
and the three native indexer launchers. This is a scoped graph-family extension,
not an admission-accounting refactor.

## Required proof

- Device-free bounds at 2,050/2,051/2,052, overflow, missing/negative metadata,
  remaining-budget-one MTP overshoot, all fixed/dynamic/observe depth bounds,
  pending/already-emitted leading rows, restored prefixes, continuation extension
  and unequal request origins/holes. Existing lifecycle rejects stale admission.
- Builder/contracts tests prove query-only nodes, inputs, outputs and dependencies
  are absent; required key/pool/core-attention/state contracts remain identical.
- CPU and native CUDA/ROCm compare selected IDs/counts, pooled values, prior tails,
  attention and accepted state bytewise against the complete program. Cover pool
  modulo boundaries, grouped rows 1–16, empty/partial/full and dirty reuse. Unused
  query banks remain unauthored, not falsely published as valid outputs.
- Retained short→long→short requests preserve both executable identities without
  recapture, allocations or workspace rebinding. A boundary-crossing request
  selects the complete program once. Prove ordinary/MTP parent and ticket
  ownership, TP agreement and existing pipeline/composed ownership explicitly.
- Register focused regressions in `ProductionTestPreflight`. Qualify captured
  node census and complete-operation timing, then the canonical model accuracy
  panel and matched unprofiled Release workload. Old profile sums are opportunity
  bounds, never predicted additive token-latency gains.

Local source audits: [admission/parent seams](/tmp/qfn-shortquery-admission-current-private/REPORT.md),
[bound derivation](/tmp/qfn-qsa-request-bound-audit-private/REPORT.md), and
[backend/contract changes](/tmp/qfn-shortquery-omission-backend-private/PLAN.md).
These are historical design evidence, not substitutes for executed tests.

## Qualification and adjacent defects

The first accepted real-model panel has 185 final-logit rows across 26 files,
byte-identical to all five retained references. Each ROCm participant recorded
154 successful no-ranking and five complete scalar forward submissions. This
includes visibility 2,048 through 2,056 and reset/replay. The ordinary and
teacher-forced TP4 short→long→short lifecycle tests also passed. Their stable
capture census and source-owned construction paths prove retained reuse; the
native API does not expose opaque executable IDs, so the test does not claim
to have compared those handles directly.

Qualification exposed lifecycle defects, each receiving an explicit
production-preflight regression:

- Forced-token publication left a consumed logits stream pending. Publication
  now retires that handoff only after success; failure preserves ownership.
  Native CUDA/ROCm tests cross streams and repeat dirty request resets.
- Canonical TP reduction rejected a single live request bank. Constructor and
  rebind now normalize it into the existing canonical live-row implementation.
  CUDA2/ROCm2/ROCm4 tests prove rank-order arithmetic, inactive bytes and exact
  communicated extents. Its bank also survives prefill phase filtering when
  the predictor requires canonical arithmetic there; the real allocator/PMA
  regression checks exact capacity and a zero-scratch native-sum control.
  Multi-request canonical banks still reject precisely.
- Weight materialization adopted a prepared binding before marking its
  store-owned copy immutable. It now seals the finalized declaration before
  publication. A real WeightManager regression covers predictor projection
  identities across CPU/CUDA/ROCm without creating a device context.
- Floating routers omitted the original-FP32 Q8 activation publication needed
  by quantized grouped shared experts. CUDA and ROCm now publish it only when
  that consumer is declared. Native tests prove exact Q8 bytes, dirty/live-row
  replay and no extra packing for ordinary scalar decode.
- One serving-program setup iteration left verifier logits owners bound for
  its sibling. A lexical scope restores ordinary full/sharded logits owners
  on entry and every exit; the strict output validator remains unchanged.

The final Integration-r14 / Release-r15 source passes all 778 Unit
registrations, 13 omission aliases and 17 defect aliases. The existing 95 inner
Unit skips are unchanged. Actual TP4 and singleton-stage PP4 ordinary,
teacher-forced and MTP short→long→short lifecycles all pass. Within each
topology, ordinary and MTP prompts and all 12 emitted IDs match exactly.
Parent/forward and publication-child census stays flat across resets; each MTP
participant retains one publication child per program. TP forced mode proves
retained forward submissions and census, without claiming an unobserved
separate generation-parent ID. Driver and retirement checks pass.

Accepted QSA/PLE writes are declared exactly by the outer publication contract,
and each program retains its own authenticated publication child. A further
preexisting defect surfaced in PP MTP: RoPE's replay-preserving reset cleared
the captured device-position pointer and row extent. Executing a graph first
restamped the fields and hid the fault; exporting its inactive sibling exposed
it. Reset now preserves this device binding; host-position data and hard-reset
behavior remain unchanged. No capture validator or arithmetic was relaxed.
`V2_Integration_RoPERetainedDevicePositionIdentity` reproduced the original
pointer/extent/policy failure with a passing host control, then passed with the
repair. The real PP MTP lifecycle also passes after the repair.

The fresh canonical panel has all 185 final-logit rows byte-identical to five
retained references; required short/heldout KL and agreement groups pass.
The threshold history remains a diagnostic group. Final-linked allocation
inspection covers CUDA SM80/86/89/90 and HIP gfx906 with no memory spills.
These scoped results do not certify the complete production-preflight suite,
continuous stochastic MTP matrix, broader topology matrix or a release image.

Current local evidence: `shortquery-omission-unit-r11`, `-focused-r14`,
`-defect-regressions-r9`, `-lifecycle-ordinary-r5`, `-lifecycle-forced-r3`,
`-lifecycle-mtp-r8`, `-lifecycle-pp-ordinary-r3`, `-lifecycle-pp-forced-r3`,
`-qfn-pp4-mtp-r3`, and `-model-fp16-r4`, all under
`parity-results/qwen-flash-next/` with the common `shortquery-omission` prefix.
Independent lifecycle review: `/tmp/qfn-shortquery-lifecycle-final-review-r14-private/`.
The successful final timing packet is `/tmp/qfn-shortquery-omission-model-benchmark-r5/`.

## Final matched ordinary Release result

The control is the immutable `cbdaf88c` grid checkpoint. Control→candidate→control
used one warmup and three measurements per arm, without profiling. All nine
natural 256-token trajectories were identical. All three owned runs, actual
runtime/environment checks, driver windows, quiet-window checks and retirement
passed. PerfStats program selection is proved by the same-source model/lifecycle
tests; the timed runs were uninstrumented.

| Median tok/s | Control, six samples | Omission, three samples | Change |
|---|---:|---:|---:|
| Prefill | 820.881 | 819.673 | -0.147% |
| Decode | 43.268 | 43.453 | +0.426% |

Decode ranges were 43.212–43.529 for control and 43.425–43.527 for omission.
Prefill ranges were 820.077–821.091 and 818.870–820.075, respectively.
The decode samples overlap, so this is a small median improvement rather than
proof of a statistically resolved gain. Both phases retain the cumulative
target: roughly 815 prefill tok/s and more than the recorded Castagna 42.959
decode tok/s. All nine 256-token trajectories match exactly.

Settings match the [grid workload](QFN_SCOPED_QUERY_COLUMN_GRID.md): four MI50s,
one MPI rank, uniform full TP with gate/up-owned and down-column routed work,
all experts on GPU, vocabulary sharded, RAM ngrams, FP16 KV and FP32 activation
storage. The model is Qwen3.8-Flash-Next UD-Q4_K_XL. The prompt is 512 repetitions
of ` test`, authenticated as exactly 512 tokens; response256, context4096,
prefill bucket512, greedy sampling with seed42, 28 physical workers on NUMA1
(CPUs28–55). MTP, prefix cache and residency maintenance are disabled; arithmetic
and kernel debug overrides are absent.

Local evidence: `parity-results/qwen-flash-next/shortquery-omission-model-fp16-r4/`
and `shortquery-omission-model-host-window-r5/`. The frozen A/B/A packet and all
samples are in `/tmp/qfn-shortquery-omission-model-benchmark-r5/summary.json`.
The candidate runtime is `/tmp/qfn-shortquery-omission-qualified-runtime-r4/`,
prepared from Release-r15 and the same source as Integration-r14. These
receipts bind the measured pre-commit source and frozen runtime, including all
the lifecycle repairs above. Host, driver, native worker, loaded-library and
retirement checks pass for every arm.

Earlier r3 timing remains separate historical evidence. The r4 wrapper failed
before native inference because it reused an existing output path. Packet r5
uses independent arm directories and authenticates its own host collector;
the benchmark workload and acceptance checks are unchanged.
