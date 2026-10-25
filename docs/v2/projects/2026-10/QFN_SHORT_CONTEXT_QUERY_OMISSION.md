# Retained QFN graphs without unnecessary indexer launches

Status: implementation plan, 2026-10-25. The user requested complete omission
of empty launches. This is the next priority after the qualified attention
rescale change. The separate column-grid candidate improves active projection
work as well, but does not implement omission.

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
   Current parent ownership is a singleton and invalidates on a changed child;
   merely changing the stage signature would recapture and is insufficient.
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
6. CPU, CUDA and ROCm consume the same explicit no-ranking program. Grouped GPU
   preparation currently fuses required pool work with query work: select the
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
These are design evidence; none claims implemented or certified omission.
