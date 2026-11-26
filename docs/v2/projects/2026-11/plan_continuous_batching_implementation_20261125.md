# Tiered, pooled-KV continuous batching: implementation plan

**Revision 5 implementation plan, 2026-11-25.** Implement the owner-approved
[cache-centric design](design_continuous_batching_20261105.md): maximal
PMA-admitted shared GPU KV residency, bounded configurable RAM/disk tiers,
configurable disk location, UMA-aware physical accounting, and retain/restore-
first serving with complete suspension and bounded pressure progress.

This supersedes revision 4's independent resident-maximum policy and deferral of
live-request suspension to P10. Suspension/tiering now belong to the main path
before cutover. The architectural decision is selected; it is not an implemented
capability, accepted milestone, numerical default table or gate exemption.

This detailed plan is not a size-limited dashboard. Keep implementation status
and evidence here/in the migration inventory, contracts in the design, and
historical reviews, RED attempts, corrections and seals unchanged. No capacity,
PR-readiness, merge, release or image-publication authority follows from this
rewrite.

**Amendment, 2026-11-26 (owner-requested).** Adds standard client retention
intent and its tier ordering (design §8.4) to P0, P3, P5 and P8. No exit, gate or
numeric default changes beyond those additions.

## 1. Direction and operating constraints

1. Use the existing `feat/continuous-batching` branch and canonical worktree.
   Preserve ongoing source work, later dirty native append/chunk work and the
   three suspended MTP edits. Coordinate document writers; do not create another
   design/validation/handoff branch or agent.
2. Retain one runner-owned `ServingSession`, PMA physical memory/page ownership
   and `TransferEngine` movement/coherence authority. Implement deep ownership
   modules with small typed interfaces, not a collection of caller-managed
   counters, transitions and event edits.
3. Replace fixed full-context request KV partitions with native pooled pages,
   actual immutable sharing and private/COW writable tails. Capacity one and
   multi-request serving use one implementation per backend.
4. Deliver complete continuation preservation and the progress policy together
   with tiered serving. Prefix archival is not complete suspension; retaining
   token IDs is not retaining KV, recurrent, logits or controller state.
5. Keep GPU execution/control device-owned under adopted G1 interfaces. Tier I/O
   services immutable storage commands; it cannot become a host scheduler or
   downloaded occupancy/table mirror.
6. Qualify coherent private implementation slices before installed cutover.
   Existing behavior remains the pre-cutover baseline, not a selectable fixed-
   slot/scalar fallback in the finished engine. No private full-history gather
   adapter, silent context/output reduction or automatic MTP/prefix disablement.
7. Refactor interfaces, graphs, estimators and tests as deeply as needed. Old
   source acceptance does not freeze an unsuitable interface. Preserve useful
   behavioral/numerical oracles, not retired physical layout assumptions.

## 2. Settled decisions and intentionally unselected details

| Selected decision | Implementation consequence |
|---|---|
| Cache-centric retain/restore-first serving | Protect valuable computation and complete live continuation; idle cache may be discarded by value/quota policy |
| Maximum useful GPU KV pool | PMA resolves `auto` after the complete execution/non-pool BOM; an explicit cap bounds the same pool |
| One common resident pool | Active history and retained idle prefix pages compete under one owner; no fixed active/cache or per-request split |
| Independent RAM and disk controls | RAM off/auto/cap; disk off or configured finite cap and directory; actual admitted routes and staging are explicit |
| UMA single physical budget | Observe real physical-domain identity, omit redundant retained RAM copies, account distinct allocations and required I/O staging |
| Bounded optimistic execution admission | Protect immediate finite operations plus a concrete all-resource completion witness; do not sum independent maximum reservations in GPU or simply move them to RAM |
| Complete suspension is main-path scope | Preserve KV, auxiliary/recurrent/MTP/sampler/handoff/grant identity, then reclaim only after preservation and actual retirement |
| Cache-aware scheduling with fairness | Bounded lookahead, coordinated prefix work, restore preparation, intrinsic prefill chunks, aging, bounded bypass/drain and thrash protection |
| No unqualified recomputation escape | Restore economics depend on avoided compute and matched end-to-end critical-path cost, not cache size alone |
| G1 adopted 2026-11-19 | Immutable ingress, destination-backed windows and authenticated terminal/ticket/storage receipts; no new permission for mutable host GPU state |
| Existing runner and DecodeWindow seam | Keep runner/session ownership; refactor `ServingWindowDecode`/DecodeWindow identity as required, without reviving `GroupedServingEngine` or parallel proof machinery |
| Preserve portability and request contracts | Keep cached/injectable NUMA-less handling and agreeing observations; preserve sampler/window/reasoning semantics or version and qualify explicit changes |
| Standard client retention intent (2026-11-26) | Accept only served-API standard fields (`prompt_cache_key`, `prompt_cache_retention`; Responses/Anthropic fields only if those APIs are served); unexpired class outranks value within a tier; minimum lifetime, never a pin; tail-first, deficit-only demotion |

The owner did **not** choose numeric RAM/disk capacities, page sizes, staging
counts, worker counts, watermarks, retention-class durations or defaults, or a
universal disk directory. P0 must map the
existing canonical configuration into the new typed intents and explicitly
resolve any proposed default change. Do not invent a new public option table or
silently reinterpret a prefix-only VRAM cap as the entire live KV pool cap.
These implementation details do not reopen the selected architectural direction.

The pressure guarantee is precise: admitted physical ownership is never
oversubscribed; future residency may exceed the resident pool only when the
complete preservation/restore/completion plan remains valid. Estimates guide
ranking, not safety. Tiers disabled/full use a proved resident completion order
or queue/refuse before exhaustion. A single request must still fit its required
resident execution working set; disk caching does not implement nonresident
streaming attention.

## 3. Baseline, pending reviews and carried obligations

### 3.1 Source and evidence boundary at this rewrite

The original migration baseline is
`b4fe53a96f257d1911574265b1b554e77c2bd289`; revision-4 documentation was adopted
on the canonical feature branch at
`a0b69f633e2cdc666292f246adf79285c8b4e4d4`. Neither is whole-tree acceptance.
The observed feature HEAD before this rewrite is
`ef1c3a012b0035882584ab21a38329f30e07b71c`.

The [migration inventory](paged_kv_migration_inventory_20261125.md) and
[authority transition draft](paged_kv_authority_transitions_20261125.md) remain
implementer-owned. Reconcile their old “owner unselected”/independent-completion
language with revision 5, preserving the exact scope of their existing prototype
and tests. Do not rename prior full-maximum grant tests as evidence for the new
pressure policy.

A combined independent review remains pending for foundation
`a0b69f633..36c1e3351` and the CPU native-table/private-frame-return milestone
`ef1c3a012`. The submitter reports PMA-backed payload/metadata, finite private
ownership, generations, all-class rollback, registered CPU access, versioned
native table publication and actual-retirement return to the original eligible
grant. The table milestone reportedly exercises native CPU attention against
those views. These are reported narrow implementation results, not accepted
P0/P1/P2 exits, serving integration, all-format qualification or GPU authority.
Later dirty append/chunk work and the suspended MTP files are outside that pin.

Retained intake packets under `CB-checkpoints/` are
`paged-foundation-review.Tp2tg9`, `paged-table-review-candidate.9flaH3` and
`paged-table-checkpoint-publication.2FrSsy`. The submitter reports PMA Unit147
for the foundation and Unit151/CPU-attention60/lifecycle2/contract7/replay4 for
the table candidate. Counts identify reported focused selections, not independent
execution or the complete configured suite. Preserve the earlier PMA 139/140
failure and its later claimed clearance until independently reconciled.

The full-gate wrapper reportedly exited timeout143 after 1800 seconds while its
original container build continued. This is neither PASS nor proof of process
retirement. Do not source-sync into that build, start an overlapping gate, kill
unrelated process groups or infer a final result from the wrapper alone.

### 3.2 Reuse without carrying the old physical model

- `OrchestrationRunner`/`ServingSession`, `IServingRowExecutor`, inference-thread
  submission and request-local delivery provide the existing ownership seam.
- Response lifetimes, execution claims, output grants, first-terminal arbitration
  and fail-closed session sealing remain behavioral requirements.
- Existing dense/hybrid CPU execution and selective prefix hooks are migration
  inputs, not proof of tiered continuous serving.
- Narrow accepted live-ID, full-vocabulary trace integrity and lazy SSE
  fragmentation/order contracts survive the refactor. Their acceptance does
  not close aggregate memory, native numerical or capacity qualification.
- CPU MTP sidecars and slot-map work contain reusable request isolation and
  oracles; replace their physical slot assumptions with versioned page roles.
- Preserve every supported authored frontend/model/policy/topology/capacity cell,
  including existing multi-row CPU and benchmark behavior. New target features
  are separate from regressions in the existing supported baseline.

### 3.3 Open obligations

| Obligation | Required closure in this plan |
|---|---|
| M5 R1: complete streaming physical envelope | P5: producer/queued/consumer overlap, framing/metadata/intake transients and final sink custody; focused regression plus actual accounting |
| M5 R2: actual prefix observations | P3/P5/P8: original lookup, usable boundary, matched/restored coverage, tier/checkpoint and harvest receipts through HTTP; inferred block counts are insufficient |
| F5 / old G4: non-streaming and host allocations | P0/P1 input geometry and P5 ownership: checked complete-response/cap semantics, parse/tokenizer and final serialization/body overlap |
| E1: production replay and policies | P1–P8: recorded initial page/continuation state, active DRY, prefix restore/harvest, cancellation, failure/seal, hybrid and MTP |
| E2/E3: complete gates and numerical matrix | P8: complete Unit/preflight and canonical receipts, AVX2/AVX512/applicable ARM, native captures and canonical frozen-history panels |
| E4: economy | P9: matched dedicated-host Release useful-throughput/latency under real pressure and tier routes, plus isolated native profiling |
| Runtime custody | Exact source/build/runtime/loaded-code identities and exits; hashes and registration alone do not establish execution origin or retirement |

The last independently reviewed complete AVX512 gate remained FAIL; complete
AVX2 and accepted complete preflight evidence remained open. Reconcile later
results explicitly rather than deleting the RED record. No current full-gate,
canonical numerical, response-envelope or economy exit is asserted here.

## 4. Ownership map and design-review reconciliation

Paths below locate existing owners, not required new class names or claims of
installed interfaces. Inspect the exact implementation head before modifying.

| Area | Existing source owner | Target transformation |
|---|---|---|
| Configuration / admission | `src/v2/config/OrchestrationConfigParser.cpp`, `OrchestrationConfigDocument.*`; `execution/config/ServingCapacity.h`; planning `PhysicalMemoryAuthority.*`, `PhysicalMemoryBOM.h`, `KVCacheMemoryEstimator.*`, `RankMemoryPlanInputs.*`, `MoEOverlayMemoryPlanInputs.*` | Typed tier intents, S/Q/R versus residency, physical-domain identities, complete BOM and maximal legal pool |
| Page authority | `src/v2/planning/PhysicalPagePool.*` and PMA | Finite grants, exact conservation and original-incarnation returns, three visibility interfaces and bounded retirement; no second live ledger |
| Kernel/cache seam | `src/v2/kernels/IKVCache.h`, CPU ring/hybrid implementations, CUDA/ROCm KV implementations and `KernelFactory` | Native append/attention/COW/truncate/import/export for real layouts; no full-history gather or fixed-stride payload owner |
| Native auxiliary history | `kernels/cpu/CPURingKVCacheAnchored.inl`, `kernels/qwen4exp/QwenSparseAttention.h`, `kernels/kvcache/CacheAuxiliaryState.h`, `execution/mtp/MTPShiftedCacheOrigin.h` | AQ8 basis, QFN compressed/selected history and partial tails, model-owned shifted coordinates and portable provenance |
| Session / continuation | `src/v2/execution/serving/ServingSession.*`, `ServingWindowDecode*`; runner/device/rank/global orchestration | Request lifetime independent of residency; complete continuation manifests and authority-owned progress selection |
| Graph / movement | `src/v2/memory/BufferArena.*`, stages/model graph builders, `TransferEngine` and retained transfer channels | Stable native bindings, exact live work, registered access frontiers, admitted pack/scatter and immutable storage completion |
| Prefix / tiers | `src/v2/execution/prefix_cache/`, host arena and disk persistence implementations | Actual sharing, coherent archive sections, protected live references, replica directory, effective-use ordering and finite storage quota |
| MTP / hybrid | `src/v2/execution/mtp/` and model-owned recurrent/checkpoint policies | Main/shifted/speculative state, exactly-once commit/correction, coherent suspension and request-local numerical policy |
| Frontends / response | `src/v2/app/modes/ServingBatchDriver.h`, `ServingResponseLeasePool.h`, `RequestGenerationLane.h`, `SSEPublishChannel.h`, `ChatCompletionHandler.cpp`, `ServerMode.cpp` | Intake through final HTTP borrower ownership; actual prefix summaries; detached bounded delivery; strict parsing of standard retention fields into a typed intent; standard cached-token usage |
| Qualification | `tests/v2/CMakeLists.txt`, `tests/v2/utils/ServingStepTrace.h`, production controller/kernel/prefix/serving suites | Focused preflight registrations, recorded production replay, native resource/capture/numerical and complete gates |

The independent revision-4 review remains a historical review of its original
pin. The owner now explicitly supersedes its recommendation to leave live
suspension outside first cutover. The following maps its technical findings to
revision-5 contracts and required evidence; it is **not** blanket closure:

| Finding | Design contract | Implementation/evidence owner |
|---|---|---|
| PKV-01 pressure-policy selection | §§1, 5: selected bounded optimistic retain/restore policy and completion witness | P0 records decision; P5 proves actual policy, including tiers off/full |
| PKV-02 conservation | §5.2: O/G/T conservation, sharing/COW debit, original-grant return and true retirement | P1; integrated pressure/MTP traces in P4–P8 |
| PKV-03 write/read/commit | §6: three typed objects, registered-before-enqueue accesses and same-frame content versions | P1/P2; native GPU and transfer proofs P6 |
| PKV-04 early GPU feasibility | §7: CUDA and ROCm live-shape/captured-copy vertical witnesses before irreversible freeze | P2 native feasibility track; complete P6 remains mandatory |
| PKV-05 queue and fairness | §§4.1, 5.1, 5.4, 13.1: direct/queued/refused outcomes, original response incarnation, bounded bypass/wakes | P5, with early P0/P1 interface bounds |
| PKV-06 native family/provenance | §§4.4, 8.1, 9: AQ8/QFN/shifted origins and neutral authenticated archive sections | P0/P1 geometry, P2/P3 formats, P4 continuation |
| PKV-07 effective-use ordering | §8.2: successful attachment, replica-bound receipts, retention watermark and no-copy refresh | P3/P6, integrated P8 eviction after pins release |
| PKV-08 topology agreement | §13.2: common usable state, exact decision/participants, prepare/commit/abort or seal | P0/P1 protocol model; P7 native composition |
| PKV-09 final HTTP borrower | §12: bounded intake, complete simultaneous allocations, transferred response lease and cancellable delivery | P5/P8 production HTTP custody |

## 5. Main dependency sequence

**All P0–P9 qualification exits remain open at this rewrite.** Documentation
completion, owner policy selection and reported focused passes are distinct
from those exits. Revision-4 P10 is absorbed into P3–P6/P7 and the final gates;
it is no longer a deferred escape-path extension.

```text
P0 inventory + policy/config/numerical contracts
  -> P1 parameterized authority and resource/visibility contracts
  -> P2 native CPU vertical path + early CUDA/ROCm feasibility
  -> P3 real sharing + tier storage/config/UMA
  -> P4 complete request continuation + hybrid/MTP suspend/restore
  -> P5 pressure scheduler + complete admission/response lifetimes
  -> P6 full CUDA/ROCm integrated native execution
  -> P7 real topology composition
  -> P8 complete feature/numerical/lifecycle qualification
  -> P9 matched economy + one qualified cutover
```

P0 brings host overlap, GPU catalog and PP concurrency **interface/feasibility**
questions forward; their full implementations retain the later exits. P1 proves
its contract for supplied bounds, not the completeness of unimplemented P5/P6/P7
bounds. Disjoint scoped work can overlap only with explicit writer ownership;
this dependency graph itself authorizes no new delegation. No installed elastic
admission before the actual continuation, progress and corresponding native
contracts exist.

### P0 — Reconcile inventory and freeze explicit contracts

**Do:** pin the source and coexisting dirty work; classify every slot-address
assumption, global/request mutable field, auxiliary history, raw-pointer reader,
prefix/archive section, sidecar/rollback reference, estimator, CLI/config input
and test as keep/refactor/retire. Record exact concurrency and retirement owners.
Preserve suspended MTP work rather than finishing its old physical-slot path.

Map existing prefix RAM/VRAM/disk budget and directory semantics to the new pool
and tier controls without silently changing authored intent. Define actual UMA
physical-domain evidence, per-host/NUMA disk/RAM ownership, complete BOM inputs,
S/Q/R suspension semantics, immutable storage messages and response-intake bounds.
Bind the recorded G1 decision to the exact exchange table in design §10 and
reconcile canonical architecture/source-policy declarations before installation;
a date or stub alone grants no host-control permission. Price explicit output
bounds against remaining legal context with the original terminal-token/feed
and shifted-append rules; preserve the visible context/length terminal.
Record unresolved numeric/default choices as choices—not invented constants.
Record how the current frontends treat unknown and cache-related request fields,
then propose the design §8.4 value-to-class map, minimum lifetimes, default
class and any operator ceiling as explicit choices for owner decision.

Specify a bounded completion-witness algorithm and its all-resource/topology
inputs; queue/direct/refused outcomes; finite bypass/drain and protected progress;
worst-case speculative/copy peaks; common usable prefix/terminal boundary;
trust source and versioned neutral archive sections. Freeze the baseline matrix
of existing supported frontend × model/policy × topology × capacity cells and
its request/window/RNG/numerical version.

**Exit:** source-linked complete migration inventory; no unexplained fixed-stride
or auxiliary full-history owner; agreed typed contracts and explicit default
migration proposal; no unresolved pressure-policy selection. The inventory is
investigation, not a source scan pretending to prove runtime correctness. P0
acceptance does not qualify implementation or increase capacity.

### P1 — Implement parameterized conservation and visibility

**Depends on:** P0 contract, with native feasibility questions still explicit.

**Do:** implement one PMA-owned page authority over materialized backing, bounded
metadata and typed class/participant geometry. Implement finite grants, monotonic
frame/content/table/request/receipt identities, exact O/G/T transitions, immutable
sharing, COW, actual-retirement return and all-resource prepare/refusal. Separate
exclusive write plans, operation-readable snapshots and committed publications.
Registered-before-enqueue accesses own all payload/basis/auxiliary/checkpoint
referents until exact completion or proved abandonment.

The private supplied-maximum prototype can be a reusable finite-grant mechanism;
it is not the selected scheduling policy. Reconcile its return semantics with
cancelled/reused request records. Derive simultaneous storage from declared
concurrency, never by adding mutually exclusive peaks or omitting overlapping
borrowers. Define device-resident canonical metadata, not a temporary host mirror.

**Tests:** actual authority contract/lifecycle tests for exact fit, overflow,
invalid geometry, all-class rollback, exception-safe partial construction,
sharing and partial-tail COW, tentative accept/reject, abort/version burn,
registered-not-enqueued access, late borrower refusal, same-frame reappend,
old/new native referents, grant destruction and stale/duplicate return. Protect
bounded remaining version/nonce issuance for accepted work; test stopping new
admission and draining before exhaustion, including burned aborted versions. Include
the nine-token shared-tail example and negative controls; label constructed
examples contract tests. Record actual production controller inputs/initial
state for replay and register it explicitly in `ProductionTestPreflight`.

**Exit:** conservation and exception safety hold for the supplied resource
vectors; refusal leaves original owners intact; retired storage returns once
and to the correct incarnation; no early payload/table/metadata reuse. This is
not proof that all model/response/GPU/PP envelopes are already supplied correctly.

### P2 — Prove native access and early accelerator feasibility

**Depends on:** P1; feeds final layout/catalog decisions rather than waiting for
all CPU features to be finished.

**CPU vertical path:** native paged append and attention through actual runner
and graph views, then truncate/import/export. No contiguous full-history adapter
or private full-context backing. Use capacity one and several/odd request counts,
permuted physical frames, page-edge/partial extents, long-over-equal-share and
mixed lengths, cancellation/replacement and exact original borrower retirement.
Compare retained native content and committed outputs, not just sampled text.

**Early CUDA and ROCm track:** prove device-selected grants/tables/live shapes,
noncontiguous native append/attention, registered-reader overwrite ordering,
large-to-small captured reuse and selective storage-completion readiness.
Change live shapes **within** a complete transaction: one row reaches EOS while
a peer continues, verifier width changes and routing produces late-known expert
work. Specify each vendor's native lowering/factoring; entry selection alone is
not the implementation. Bind tickets to exact operation ordinals and reject
replay across two prefill chunks under the same grant. Record internal submitted
grids, branch/catalog cardinality, retained bytes and every host ticket/receipt.
No per-layer host tickets, empty capacity grids, live-count downloads, recapture,
default streams or host-step substitute. A CPU protocol model cannot pass this
track. Unavailable native hardware is explicitly unqualified.

**Format expansion:** native supported floating/quantized/TurboQuant formats,
AQ8 second-page basis and asymmetric codec behavior, QFN selected/compressed
history and raw tails, shifted origins, all supported live M and representative
N/K, CPU ISA and positive thread counts. Preserve arithmetic order across frame
permutation and compaction. Every changed kernel has production recorded replay
and explicit preflight registration; record geometry and precision.

**Exit:** the vertical path genuinely consumes pooled pages and the early
native feasibility claims have actual backend evidence before irreversible
layout choices. Full kernel-family and serving qualifications remain P6/P8/P9;
a small witness is not a whole-backend exit.

### P3 — Deliver shared prefixes and first-class tier storage

**Depends on:** P1 ownership and P2 native interfaces/feasibility.

**Do:** attach compatible prefixes to actual immutable frames; implement partial
COW, collision-resolving ancestry/trust checks, coherent terminal/recurrent
boundaries and deduplicated nonresident import. Define typed portable archive
sections; reject incompatible format/numerical provenance before payload access.
New prefix attachment must not restore another request's sampler or domain-wide
expert placement state.

Implement configured GPU/RAM/disk routes and capacity/directory controls through
one parser/planner path. Resolve maximal legal GPU pools from complete BOM inputs.
Observe real UMA physical domains, omit redundant retained RAM copies and retain
bounded I/O staging. Give disk extents one finite owner, including rounded quota,
metadata, pending/old replacement extents, durable completion and namespace-safe
cleanup. Pre-admit all transfer/persistence records; use exact source and prior-
access destination leases through `TransferEngine` and device-indexed staging.

Implement the design §8.4 tier ordering in the same retention owner: typed
intent (namespace, class, expiry) in bounded admitted metadata; unexpired class,
then value, then stable tie-break; expiry refreshed only by effective use;
shared pages at the highest referencing class and moved once; tail-first,
deficit-only demotion that cascades only on real overflow and discards only from
the lowest enabled tier. Live state and protected continuations outrank every
class.

Implement effective-use ordering at successful attachment: content/replica-bound
immutable receipts, retention watermark/acknowledgment and bounded temporary
pin/fence. A GPU no-copy hit refreshes valid backing usefulness; duplicates and
obsolete receipts cannot affect a replacement. Recency pins retire after the
ordering point rather than making “hotness” tests pass by permanent protection.

**Tests:** actual shared IDs with private divergence; concurrent imports and
lookup/eviction races; source/export/destination custody; delayed/failed/cancelled
I/O; partial/corrupt archives; full quotas and staging; wrong disk directory and
engine-only cleanup; UMA single accounting versus genuine duplicate allocations;
RAM-off disk staging. Force competing backing eviction after a no-copy hit and
pin release, including reordered receipts. Real multi-block/clamped/partial and
mixed-participant lookup observations must reach runtime summaries and HTTP.
Replay the design §8.4 worked trace through the real authority, plus class
expiry before and after a refresh, a shared page under mixed classes, a partial
hit from a tail-demoted prefix (including a hybrid boundary without a
checkpoint), deficit-only cascade with a free lower tier, tiers off/full, and
live demand displacing the longest class.

**Exit:** implemented tier routes, exact replica/quota/memory ownership and
actual reuse are demonstrated. A RAM experiment and a disk experiment are not
proof of a composed route. Prefix tiering alone is not complete suspension;
corresponding GPU-native qualification remains P6.

### P4 — Preserve complete request continuation, hybrid and MTP

**Depends on:** P2/P3.

**Do:** separate request lifetime from residency without expanding S/Q/R.
Implement transaction-boundary suspension and coherent resume of the original
request generation. Manifests include native KV/auxiliary history, basis/roles,
GDN/conv, main/shifted MTP, learner, sampler/RNG/penalty/DRY history, stop/think,
terminal logits/hidden/condition, pending legal handoff/correction, committed
output and outstanding grant/publication identity. No unresolved speculative
model transaction is discarded at suspension.

Freeze and preserve before releasing the resident owner; reuse an unchanged
valid backing replica without a new write. Protect every live continuation from
idle retention eviction. Restore its complete working set and next-operation
peak, publish all required readiness, then resume without reseeding, resampling
an old condition or duplicate output. A new prefix-attaching request instead
initializes its own policy/history. Keep domain placement separate from both.

**Tests:** interleaved A/B/C laws/seeds/depths, full/partial hits, page-boundary
proposals, zero/some/all acceptance, COW then reject, correction exactly once,
sidecar cancellation, reused frame/record identities and model-specific shifted
origins. Suspend/restore between complete transactions with recurrence/DRY and
outstanding output credit, repeat across actual tiers, then compare complete
native state, logits, decisions, learner and outputs against the same no-movement
control. Include live peers, failed preservation and cancellation during I/O.

**Exit:** complete original-request continuation survives residency changes for
the implemented cells with bounded storage and no lost/duplicated state. No
claim follows for unsupported models/formats/backends; never call a prefix-only
archive or retained token list a continuation implementation.

### P5 — Integrate pressure progress, scheduling and response custody

**Depends on:** P1 supplied resource contracts and P3/P4 preservation; response
interface work starts in P0/P1 but its complete closure is required here.

**Pressure/scheduling:** implement atomic execute-admit/queue/refuse outcomes
and the bounded canonical-authority completion witness in design §5.3. Bind
all live generations, topology, versions, actual tier capacities and outstanding
frontiers. Prove each phase's simultaneous resource vector, including suspension,
restore, speculative/COW, checkpoint, table/event/staging/control and response
peaks. Do not count shared victims twice or rely on unowned future filesystem
space. Revalidate worst-case successor state before spending a new grant; retain
and fairly execute a protected progress step when ordinary work cannot proceed.

Use cache-aware bounded lookahead, coordinated prefix work, useful restore
preparation and intrinsic chunked prefill. Score only legal choices; estimates
never authorize allocations. Respect request windows and authored MTP depth.
Implement saturating age, stable ties, finite bypass/drain and suspension-thrash
protection. Disabled/full tiers queue/refuse or use a proved resident completion
order; no hidden recompute or normal mid-response OOM after lost escape capacity.

**Response/custody:** complete intake/body/parse admission before copies;
validate immutable preparation concurrency; price tokenizer/input/metadata and
producer/queued/consumer/framing overlap. Transfer a detached response lease
through the HTTP provider, result body, callbacks and final socket borrower.
Keep byte credit through actual `sink.write` completion, not pop or producer exit.
Price optional tracing and non-streaming final serialization plus body together.
Define a maximum grant that really fits, pre-success refusal and explicit checked
non-streaming cap failure. Reserve independent control/terminal progress and
cancellable delivery deadlines; a deadline does not free borrowed storage.

**Tests:** all live requests need growth; tiers off/full/unavailable; uneven
shared victims; insufficient staging or table/event records; pressure with spare
S, Q=0, queue/response exhaustion and early/late cancellation incarnation; bursts
of hot arrivals versus an old cold/long request; repeated suspension and concrete
wakes. A prewarm (only on a served API that defines one) is an ordinary admitted
request without sampling; it gains no priority over live work and cannot reserve
capacity for later requests. Exercise both analytical safe-order cases with real
resource geometry, and a lower-tier displacement case without independent full
maxima in RAM.

Hold an oversized semantic event plus a prior consumer payload; verify exact
bytes/framing and simultaneous ownership. Block the last write after producer
exit while refilling R; run concurrent rejected parses and a connected sink that
stops draining. Cover all-output-blocked cancel/shutdown, non-streaming cap,
failed shared forward/reset with completed peers, stats reset and no duplicate
terminal. Preserve fragments → finish → usage → DONE order and borrowed/lazy
allocation witnesses. Replace sleep-based readiness with explicit observations.

**Exit:** integrated policy makes bounded progress under admitted assumptions,
conserves physical resources and preserves numerical policy. R1/R2/F5 have
focused production regressions explicitly in preflight and complete custody/
materialization evidence. A scheduling simulator or fragment-helper bound alone
cannot close this exit; all-resource witness and actual integration are required.

### P6 — Complete CUDA and ROCm tiered captured serving

**Depends on:** P1–P5 semantics and P2's early native feasibility.

**Do:** implement the same finite grants, witness/policy, request/residency state,
sharing and continuation semantics on device. Stable pool/table/arena bindings
are established before capture. CPU tier workers consume immutable I/O commands
and return exact completion; GPU-selected frame addresses use native pack/scatter
without host enumeration. Conditional execution retains its complete parent;
HIP uses authenticated retained complete transactions. Enumerate and price every
selector/native-MTP/storage receipt and catalog family.

**Native tests on both vendors:**

- Capture/reset/repeated large-to-small production replay with mixed generations,
  frame permutations, partial/empty work and exact live grids/communication.
- Registered-before-enqueue and queued readers on multiple streams while another
  stream attempts restore/overwrite; both old and new byte-correct consumers.
- Main/shifted verifier/sidecar access, COW/reject/reappend, delayed/obsolete/failing
  receipts and cancellation with actual native frame/table reuse.
- Actual configured tier chains and complete suspension/restore while unrelated
  ready work progresses, plus no-copy-hit recency under forced backing eviction.
- Negative controls removing each required event edge must fail the intended
  assertion under controlled ordering, not rely on slow storage to expose races.
- GPU ingress/publication source-policy checks, no live count/phase readbacks,
  host model-state copies, default streams, recapture or hot-path native allocation.

**Exit:** each backend has production capture, native lifetime, numerical and
exact live-work evidence for its claimed cells. Compile every shipping GPU
architecture with spill guards; isolate profiling for occupancy, registers,
spills, throughput and memory efficiency. CPU models and one vendor do not
qualify the other. Missing hardware remains unqualified, not silently skipped.

### P7 — Compose topology-wide progress and continuation

**Depends on:** single-participant contracts and corresponding native adapters;
protocol/geometry design and device-free models begin in P0/P1.

**Do:** thread exact logical decisions, content/table/continuation epochs and
local resource leases through existing rank/global TP/PP/MoE and MPI owners.
Agree one actually usable prefix/terminal boundary, not merely matching epochs
or successful local fits. Prepare all required owners, commit coherently before
consumption or abort unpublished grants. Indeterminate/post-publication failure
seals/drains; no local retry can undo a peer's consumed state.

Apply this to admission, growth, tentative accept/reject, preservation and resume.
Respect original ranks/NUMA, genuine homogeneous native GPU domains and existing
CPU/cross-rank transports. Empty/expert-only followers participate in necessary
progress without dummy KV. Declare PP microbatch concurrency and exclusive arena
lifetimes; include them in every progress/retirement resource vector.

**Tests:** successful-but-incompatible proposals; refusal/failure on each side
of commit; late/duplicated messages; empty followers; asymmetric memory pressure
and backing routes; actual live transport extents; shared-prefix shards;
singleton, odd/non-power-of-two and larger memberships, unequal stage widths,
more than two PP stages and valid CPU/GPU compositions. Device-free counts exceed
local hardware; native unavailable cells are unqualified. Compare numerical
reduction order, coherent state and actual ownership as well as outputs.

**Exit:** the same admitted topology owns planning through retirement, and a
resource-valid progress witness spans all its real owners. No hidden global
slot map, remote GPU alias, padded-capacity communication or arbitrary count
whitelist remains. Evidence states each actually qualified configuration.

### P8 — Complete feature, numerical and lifecycle qualification

**Depends on:** P2–P7 implemented cells, including P5 response closure.

Migrate every supported frontend, including benchmark, through the same paged
serving seam. Preserve the baseline matrix, not only capacity one. Explicitly
version any changed window/reasoning/prefix/RNG policy and retain old numerical
controls externally. Separately qualify genuinely new features such as absent
GPU DRY; neither their absence nor new policy is an excuse for a regression.

Required evidence:

1. Complete Unit and complete configured model-free `ProductionTestPreflight`,
   canonical prerequisite receipt, exact source/build/test-inventory identity
   and real supported hardware qualification. Selected passes are insufficient.
2. Strict recorded-input/initial-state replay through actual production controllers
   and kernels; versioned geometry, native content/table identities, grants,
   completions, tier/continuation transitions and expected decisions/results.
3. Format/codebook/backend sweeps, representative N/K and supported live M,
   CPU ISA/positive-thread geometry, dense/MoE, hybrid, ordinary/MTP, prefix and
   enabled/disabled tier combinations, stochastic defaults and dynamic maintenance.
4. Canonical multi-prompt frozen-history final-logit panels plus applicable
   byte-equivalence under the same request/prefix/control/placement trace.
   Diagnostic top-k or a single full-vocabulary capture is not that panel.
5. Actual HTTP/frontend lifecycle, streaming/non-streaming, tools/reasoning,
   summary observations, passive stats, final borrower custody, slow consumers,
   cancellation/reset and source-to-loaded-code provenance through retirement.
6. Standard retention fields over HTTP: accepted values, precise rejection of
   unsupported values, key isolation across trust domains, actual
   `cached_tokens` usage in streaming and non-streaming responses, and a
   recorded real client session (for example pi with long retention) whose
   requests round-trip unchanged.

**Exit:** all claimed cells have current evidence; no previously supported cell
is silently refused or redirected to a legacy implementation. Historical failures
are fixed with focused regressions or remain blockers. Large certified corpora
publish only in the optional `Llaminar/corpora` repository, not in source.

### P9 — Measure useful throughput and perform one qualified cutover

**Depends on:** P8 correctness plus native resource evidence.

Use dedicated-host Release binaries and matched production workloads, with
defaults intact and separately measured authored controls. Cover capacity one,
varied context/output/concurrency, prefix fanout, misses, long/short coexistence,
bursty and sustained offered load, cancellation/refill, non-streaming/streaming
and slow delivery. Include actual discrete GPU→RAM→disk and UMA memory→disk
routes where admitted, tier-off controls, all-full pressure, no-copy backing hits
and repeated suspension. Do not infer economy from Docker VM timings.

Report useful output tokens/s and SLO goodput alongside queue-inclusive TTFT,
inter-token/completion/tail latency, unfinished work and starvation. Attribute
actual avoided prefill, restore critical-path time, transfer/persistence and
contention costs, COW/page-lookup overhead, suspension/thrash and full physical
resource high water. Large cache size alone proves nothing about restore versus
recompute; compare the actual model/backend/format/context. External engine
comparisons must match policies and disclose differing optional modes; there is
no assumed superiority.

Keep CPU physical-core/NUMA scaling, isolated GPU profiling, all-target spill
builds and actual submitted launch/communication evidence separate from canonical
unprofiled timing. A fast completed subset with a growing queue is not success.

**Exit:** correctness and economy pass together. Cut over planner/BOM, native
kernels, graph/cache/auxiliary bindings, prefix serializers, controller, frontends
and benchmark at the same qualified candidate. Retire fixed-slot allocation/
estimation, copy-per-resident-hit restoration, full-history adapters, obsolete
scalar/cohort launchers, names, capabilities, source-policy allowances and docs.
Keep the old binary as an external comparison control, not an installed mode.

Synchronize CLI/help/config, README, architecture instructions, tests and docs.
Ordinary PR and release gates still apply; this plan grants no bypass or release
publication authority.

## 6. Evidence and coherent-milestone review protocol

For each coherent bounded slice, retain and submit:

- Exact base/head or immutable candidate identity and scoped diff, including new
  files; identify coexisting unrelated changes and explicit scope exclusions.
- Changed production contract, ownership map and focused registrations, correctly
  distinguishing recorded replay from constructed lifecycle/geometry tests.
- Raw selections/results/exits, compiler/backend/ISA/model/config identity and
  source-to-loaded-code custody for claimed execution; final process retirement
  when lifecycle is part of the claim.
- What was independently executed, submitter-reported or source-inspected only;
  focused versus complete gates; device-free versus actual native evidence;
  unavailable/unqualified cells and remaining defects.
- Memory/launch/communication/movement evidence for economy claims; original RED
  attempts, corrections and seals preserved rather than overwritten.

Registration is not execution; a setup fixture is not a substantive native test.
Hashes establish retained-byte identity, not runtime origin or loaded-code
custody. Reject evidence symlinks; seal regular retained files with
`REVIEW-SHA256SUMS`, excluding the seal itself. Preserve explicitly documented
historical seal exclusions rather than silently rewriting old packets.

The pending foundation/table review is one bounded milestone, not a replacement
agent or new per-fix review loop. Use its immutable packet/published pin, not later
dirty source. Documentation direction and narrow source acceptance do not clear
complete gates or grant capacity. Observe the original continuing build to actual
retirement/final outcome before launching overlapping work; wrapper timeout alone
is not either outcome.

Use the canonical [testing skill](../../../../.agents/llaminar-testing/SKILL.md),
[architecture map](../../../../.github/instructions/llaminar-v2-architecture.instructions.md)
and [production CI policy](../../../production-ci.md). Documentation authority is
separate from implementation qualification; any exceptional skipped gate needs
its operation-specific authorization. The earlier documentation-only bypass was
for its original completed commit, not this rewrite or future changes.

## 7. Next coherent implementer deliverable

After the explicit document-writer handoff:

1. Reconcile the implementer-owned inventory and authority-transition draft with
   revision 5, including S-held suspension, physical tier controls, UMA and the
   selected completion witness. Preserve historical prototype/evidence claims.
2. Present the foundation/table milestone at its immutable pin for the already
   pending conservation/lifetime/exception/boundedness review. Keep later dirty
   append work and suspended MTP edits separately identified; no implied acceptance.
3. Continue the P1/P2 native vertical path and bring CUDA/ROCm feasibility and the
   P3–P5 tier/continuation/progress interfaces forward. Specify the bounded witness
   algorithm and actual all-resource inputs before enabling optimistic admission.
4. Report precise dependency gaps and evidence in coherent slices. Do not finish
   the old fixed-slot MTP path, rename slots “pages,” install independent maximum
   reservations as the product default, or defer full suspension to a new P10.

The approved destination is tiered retain/restore-first serving. Its implementation,
complete qualification and any increased serving capacity remain unaccepted until
their own evidence closes the stated gates.
