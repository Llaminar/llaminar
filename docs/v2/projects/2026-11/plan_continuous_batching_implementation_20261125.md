# Pooled paged-KV continuous batching: implementation plan

**Replacement plan, 2026-11-25.** Implement
[design revision 4](design_continuous_batching_20261105.md), not the former
fixed-slot/full-context roadmap. The operator authorized a complete design and
plan rewrite, documentation commit/push, and an implementer handoff with full
refactoring allowed. The documentation change does not modify the engine.

This is a detailed implementation plan, not a size-limited tuning dashboard.
Current facts below are a migration baseline, not qualification of the new
design. Update status/evidence as work lands; keep the design's contracts
separate from progress. Historical reviews, RED attempts and sealed evidence
remain unchanged.

## 1. Read first and change direction

1. Read the replacement design in full, especially §3–§8 (authority, pool,
   admission, page views and sharing) and §11 (transfer/offload lifetimes).
2. Stop extending dedicated per-request full-context KV as the target. Preserve
   in-progress changes, then classify each as reusable request-state logic,
   an oracle, or machinery to replace. Slot-to-KV-address mappings are not the
   foundation of the replacement.
3. Implement pooled pages and native access before continuing the old
   “per-slot MTP, then GPU, then distributed” sequence. Request-local MTP and
   recurrent ownership remain necessary, but must bind page views rather than
   fixed KV partitions.
4. Retain one runner/session authority. Refactor interfaces, graph builders,
   estimators, cache implementations and tests as deeply as needed; old source
   acceptance is not a requirement to preserve an unsuitable interface.
5. Keep each landing slice explicit about implemented, tested and unqualified
   combinations. Do not advertise capacity or enable a substitute path to
   conceal a missing paged implementation.

The finished engine has **one paged KV implementation per backend**, shared by
capacity-one and multi-request serving. There is no fixed-slot compatibility
mode, automatic scalar replay, private full-history gather adapter, or second
memory ledger. Legacy code is the pre-cutover baseline until replacement is
qualified, not a user-selectable fallback after cutover.

## 2. Settled decisions and their limits

| Decision | Consequence |
|---|---|
| Pooled, paged KV replaces the physical lane design | Logical request records remain; physical pages are assigned from common pools, not S full-context partitions |
| PMA remains the single physical authority | Add a real typed logical-page grant implementation under PMA; do not rename allocation claims or add private free-byte arithmetic |
| Shared immutable prefixes and COW are core scope | Resident hits borrow actual pages; partial shared tails copy only their valid extent before mutation |
| Resident-only admission initially guarantees declared completion bounds | Protect fungible worst-case growth commitments; assign frames on demand; disclose that default full-context output can still limit simultaneous execution |
| Future live-request offload is not enabled by this rewrite | Build identity, overwrite fences, replica/recency and transfer seams now; elastic admission needs its own complete progress and native proofs |
| G1 serving-boundary decisions were adopted on 2026-11-19 | Preserve typed immutable ingress, destination-backed window grants, terminal/ticket receipts and closed host/device authority; paging adds no host state-readback permission |
| `GroupedServingEngine` and its parallel proof machinery were retired | Do not resurrect them; production state remains runner-owned `ServingSession` |
| DecodeWindow is the retained GPU serving seam | Refactor its ABI as required for typed page views and complete identity; an unimplemented override remains unimplemented |
| Hardened NUMA-less-kernel handling stays on the feature branch | Preserve its cached/injectable, agreeing-observation contract; paging does not authorize unrelated portability changes |
| Refactoring is unrestricted within this target | Production changes still use required testing/review/branch gates; this documentation publication grants no blanket hook bypass or release authority |

Do not confuse page pooling with overcommit. If the product later requires
admitting requests whose combined maximum retained histories cannot fit, bring
forward the complete live-offload pressure policy rather than silently dropping
the resident completion guarantee or parking every request on an empty pool.
Changing that policy requires an explicit design update and tests, not a local
allocator heuristic.

## 3. Baseline and obligations carried forward

Migration inspection starts from `feat/continuous-batching` commit
`b4fe53a96f257d1911574265b1b554e77c2bd289`. It is not an independently accepted
whole-tree milestone. Concurrent implementation work must be preserved and
inventoried separately before integration.

### Current private implementation checkpoint

The [P0 migration inventory](paged_kv_migration_inventory_20261125.md) and
[authority transition draft](paged_kv_authority_transitions_20261125.md) track
new source work and independent PKV-01–09 findings. P0 and P1 exits remain open.
In particular, the proposed pressure policy is not unconditionally adopted;
the owner referred the choice to the existing reviewer, whose bounded-pressure
safe-order recommendation remains a recommendation pending owner resolution.
No default output cap or overcommit policy has been installed.

A private PMA-issued CPU prototype materializes page backing/metadata, protects
fungible explicitly supplied commitments, issues frame incarnations, rolls back
multi-class prepares, and pins retirement through exclusive CPU byte writes,
completed byte publications and registered CPU consumers. It is not wired to
KV kernels, page tables, serving admission, prefix sharing or GPU execution.
These interfaces are an implementation checkpoint, not the frozen P1 ABI.

The complete PMA Unit registration (147 tests) and three focused preflight
registrations pass in `CB-checkpoints/paged-p1-foundation-checkpoint.ZmLTUE`.
Small native-recorded ownership/append/registered-reader traces are retained and
embedded into the test binary for compiler/source-free runners. Earlier failures
remain preserved. This is not complete Unit/preflight or independent P1
acceptance, numerical model qualification or measured serving economy.

### 3.1 Useful existing structure

- `OrchestrationRunner` owns `ServingSession`; `IServingRowExecutor` connects
  the session to participant execution. `ServingBatchDriver` owns inference
  submission while request-local lanes handle delivery.
- Separate response lifetimes, execution claims, output grants, exactly-once
  terminal handling and session sealing provide reusable behavioral contracts.
- CPU dense/hybrid row execution and selective prefix hooks are migration
  inputs, not evidence that a pooled page allocator already exists.
- Live-ID rejection, lossless full-vocabulary trace parsing/admission checking,
  and lazy SSE allocation/order regressions have narrow accepted reviews.
  Preserve their invariants while changing the storage model.
- The latest CPU MTP groundwork binds sidecars to resident KV slots and keys
  their caches by those maps. It does not enable continuous MTP serving.
  Preserve required request isolation and useful oracles, replacing physical
  slot assumptions with versioned paged views.

### 3.2 Open findings do not disappear

| Carried obligation | Replacement work and required evidence |
|---|---|
| M5 R1: total streaming envelope | P5: include producer, queued and consumer-held payloads; remove arbitrary oversize admission and early credit release; slow-consumer plus oversized-event regression |
| M5 R2: prefix `matched_blocks` | P3/P5: preserve actual per-request block/tier/checkpoint observations through terminal summaries and HTTP; multi-block regression |
| F5 / old G4: non-streaming and other host allocations | P1/P5: geometrical complete-response bounds, materialization and lifetime accounting, including parser/metadata/framing overlap |
| E1: broader replay and native policies | P2–P8: page state, resident cancellation, active DRY, prefix restore/harvest, failure/seal, hybrid and MTP traces through production implementations |
| E2/E3: complete gates and x86 numerical coverage | P8: fresh complete Unit/preflight receipts, AVX2/AVX512 and applicable ARM coverage; focused old passes are not substitutes |
| E4: CPU economy and all-backend economy | P9: dedicated-host Release serving workloads plus isolated CPU/GPU profiling and live-work evidence |
| Historical source-to-loaded-code custody gaps | Evidence protocol below: exact sources, build/runtime identities, exits and retained original results; do not infer custody from hashes alone |

The last reviewed complete AVX512 gate remained FAIL; complete AVX2
qualification and a complete preflight receipt remained open. Reconcile any
newer evidence explicitly. Reported focused 4/4 streaming, 3/3 replay/identity,
aarch64 native serving, or MTP slot-map tests do not clear those gates and do
not qualify the new physical model.

## 4. Implementation map

These paths locate existing owners, not mandatory names for new classes.
Confirm exact call sites against the implementation head before editing.

| Area | Starting owners | Required transformation |
|---|---|---|
| Capacity/config/planning | `src/v2/execution/config/ServingCapacity.h`; `src/v2/planning/KVCacheMemoryEstimator.*`, `PhysicalMemoryAuthority.*`, `RankMemoryPlanInputs.*`, `MoEOverlayMemoryPlanInputs.*` | Separate request-record capacity, page-pool geometry and completion commitments; price native formats and every owner once |
| Cache/kernel seam | `src/v2/kernels/IKVCache.h`; CPU `CPURingKVCache*`/`CPUHybridRingKVCache*`; CUDA/ROCm `kvcache/` implementations; `KernelFactory` | Native typed paged views for append/attention/COW/truncate/import/export; retire slot-stride payload ownership |
| Runner/session | `src/v2/execution/serving/ServingSession.*`, `ServingWindowDecode*`; `execution/runner/`; device/rank/global orchestrators | Generation-qualified request lifecycle and atomic page-view publication; no second controller |
| Graph/arena | `src/v2/memory/BufferArena.*`; graph storage layouts, compute stages and declarative model builders | Stable page/table bindings, exact live work, explicit handoff and concurrency BOM |
| Prefix/state | `src/v2/execution/prefix_cache/`; cache-family and model-runtime checkpoint interfaces | Actual shared resident pages, coherent hybrid/terminal snapshots, COW, identity/recency and event-aware retirement |
| MTP | `src/v2/execution/mtp/`; main/shifted predictor caches and sidecar graph policy scopes | Bounded tentative page ownership, atomic commit/rollback and request-local conditions/learner/history |
| Transport/frontends | `src/v2/app/modes/ServingBatchDriver.h`, `ServingResponseLeasePool.h`, `RequestGenerationLane.h`, `SSEPublishChannel.h`, `ChatCompletionHandler.cpp`, `ServerMode.cpp` | Complete host-memory envelope, detached delivery and actual summaries; no inference work on writers |
| Qualification | `tests/v2/CMakeLists.txt`; `tests/v2/utils/ServingStepTrace.h`; serving/kernel/prefix integration and Unit suites | Explicit focused preflight registrations, new paged traces, native capture/arithmetic/economy and complete gates |

Preserve code-owned CLI/config authority. Existing S/R/Q spellings describe
logical capacities, not equal physical partitions. Add pool/page policy intent
through the same parser, lossless documents, MPI/startup publication, planner
and validation; avoid a second frontend-only capacity table or old aliases.

## 5. Dependency sequence and exit criteria

**All P0–P9 qualification exits are open at this rewrite.** Documentation
completion is not P0 implementation acceptance. P10 is a later extension, not
a hidden prerequisite for claiming the resident-only guarantee. Focused
implementation slices may overlap only with explicit ownership and disjoint
writers; this plan does not authorize a new multi-agent run by itself.

### P0 — Inventory the replacement surface

**Do:** freeze a source baseline; enumerate every slot-to-KV address assumption,
request/global mutable field, prefix import/export, sidecar/rollback reference,
raw-pointer consumer, estimator, public capacity input and test that encodes
fixed backing. Record concurrency and exact retirement ownership for each.
Classify current in-progress MTP changes rather than discard them.

**Exit:** a source-linked migration inventory maps every identified owner to
keep/refactor/retire and its new contract. The selected initial pressure policy,
request/control numerical version and cutover scope are explicit. No unexplained
fixed-stride consumer remains outside the inventory. This is investigation and
design evidence, not a source sanitizer pretending to prove native correctness.

### P1 — Implement the canonical page-pool contract

**Depends on:** P0.

**Do:** extend PMA with typed pool geometry and logical grants over materialized
backing. Implement CPU-authority state transitions with a GPU-compatible data
contract: frame/table generations, immutable references, private append, COW,
completion commitments, atomic multi-class prepare/commit/refusal, deferred
retirement and transfer-access leases. Price all metadata, request history,
recurrent/terminal state, transaction work and response storage inputs.

Define how GPU-resident suballocation remains the one PMA-owned authority;
do not implement a host counter mirror as a temporary substitute. Resolve
pool size independently of S and per-request context limits. Freeze page
formats and quantization alignment through native layout factories.

**Tests:** actual production authority via fast device-free contract/lifecycle
and recorded-input controller tests; exact fit/one-page refusal, arithmetic
overflow, zero/invalid geometry, partial construction rollback, stale handles,
frame/table exhaustion, shared-reference accounting, pending retirement,
full-output-default growth, non-overlapping multi-pool admission, and reset.
Register focused changed-controller replay explicitly in production preflight.

**Exit:** each physical allocation has one retained PMA claim; page grants are
real ownership operations rather than another byte ledger; refused operations
leave all original owners unchanged; admitted completion cannot deadlock on
unreserved growth. Metadata and retirement capacity are bounded.

### P2 — Replace CPU KV access and wire live serving

**Depends on:** P1.

**Do:** implement native paged append/attention/truncate/import/export for the
supported CPU formats; wire the existing runner/session and graph stages to
paged request views. Compact work rows independently of request records and
physical frames. Keep stable table storage and persistent terminal logits.
Remove fixed full-context allocation from the replaced CPU path rather than
wrap it with a paged-looking index.

**Tests:** S = 1 and larger/odd capacities; noncontiguous and permuted frames;
lengths immediately before/at/after a page boundary; one request larger than
an equal-share partition; mixed short/long requests; finish/cancel/replacement
while another continues; context/output limits; all actual supported formats,
live M and CPU ISA/thread geometries. Compare actual retained page content,
committed bytes, launch/work extents and PMA ownership—not just final text.
Use recorded production inputs for controller/kernel replay.

**Exit:** dense CPU serving reaches attention directly through pooled pages,
with no full-history contiguous copy or per-request serial execution substitute.
The paged and blessed numerical controls agree under their declared contracts.
A focused native result establishes only its actual configuration, not default
serving cutover or all-backend support.

### P3 — Implement live prefix sharing and safe copy publication

**Depends on:** P2; lifecycle primitives from P1.

**Do:** join compatible requests to actual immutable resident pages. Implement
partial-page COW, canonical prefix keys/trust policy, cache-reference eviction,
nonresident import into exclusive pages, deduplicated concurrent imports and
terminal checkpoint/logit coherence. Extend existing TransferEngine storage
interfaces with source-ready and destination-prior-access ownership; no raw
coherence edits. Add effective-use recency to valid backing replicas even when
the hit is GPU-resident, with a protocol that does not poll mutable GPU state.

**Tests:** two/three readers share the same physical identities; divergent tails;
COW on a partial page; original prefix bytes unchanged; cancellation and cache
eviction with surviving readers; missing/incompatible key; provisional lookup
race; duplicate import; delayed restore completion after cancellation; source
retained through export; full-hit terminal policy. Multi-block observations
must reach both runtime summaries and HTTP.

**Exit:** resident hits do not clone whole prefixes into private request KV.
References, valid extents and pool/replica physical costs reconcile. CPU/task
ordering and device-free protocol replay are proved; CUDA/ROCm event/capture
qualification remains P6, not inferred from those tests.

### P4 — Complete request-local hybrid and MTP state

**Depends on:** P2/P3.

**Do:** bind main and shifted predictor histories to page views; retain terminal
hidden/logit/condition state by request generation. Implement speculative page
reservation, accept/reject/correction and rollback with coherent visible-length
publication. Separate mutable GDN/conv and model-runtime state from immutable
prefix snapshots and domain placement. Keep request-local sampler laws,
penalties/DRY, depth decisions and forced-reasoning policy.

**Tests:** interleaved A/B/C prompts and decode; mixed laws/seeds/depths; proposal
crossing a page boundary; zero/some/all accepted drafts; COW followed by reject;
correction exactly once; cancellation during sidecar work; reused request-record
and frame generations; hybrid checkpoint restore/harvest with a live peer;
full-hit/partial-hit terminal handoff. Record initial page/recurrent/controller
state and compare the real implementation to the applicable serial-row oracle.
The oracle must never become a production row-replay path.

**Exit:** continuous MTP and hybrid-prefix behavior operate on pooled pages with
independent request history and no global mutable handoff. Native coverage is
reported per backend/model/format; no default MTP/prefix disablement is accepted
as a completed feature.

### P5 — Close host-memory, publication and lifecycle obligations

**Depends on:** P1 geometry; integrates with P2–P4. May be developed as a
separate scoped slice, but must complete before capacity/cutover claims.

**Do:** replace nominal response pricing with checked demand/caps covering
preparation/parser/tokenizer storage, arbitrary metadata and compound fields,
producer/queue/in-flight-consumer overlap, framing and non-streaming results.
Keep credit owned through `sink.write` completion. Reserve terminal/control
capacity separately from ordinary output. Preserve lazy allocation witnesses,
UTF-8/tool raw-byte reconstruction and fragments -> finish -> usage -> DONE
ordering, including immediate stop on rejected delivery.

**Tests:** oversized event above the old channel cap while a slow consumer still
retains the preceding payload; exact bytes/framing and measured simultaneous
ownership bound; non-streaming cap; rejection before success; all consumers
blocked plus cancel/shutdown; failed shared forward/reset with already completed
peers; stats epoch reset and no double terminal. Replace timing sleeps with
readiness observations and scoped close/join.

**Exit:** R1/R2/F5 have focused production regressions explicitly in preflight
and reconciled materialization evidence. A fragment-helper allocation bound
alone is not closure of aggregate response memory.

### P6 — Implement and qualify CUDA and ROCm paged execution

**Depends on:** P1–P4 semantics; P5 publication contract.

**Do:** implement device-owned pool grants, page tables, reference/retirement
state and native paged kernels. Bind stable pool/table addresses before capture.
Complete mixed-admission, per-request laws/history, ordinary/MTP, lifecycle and
selective prefix-copy operations. Enumerate exact retained graph/branch families
and all HIP selector/ticket observations with their physical BOM.

Preserve the full dependency chain for every reuse/restore/COW:
prior readers/writers -> exclusive overwrite -> completion publication -> new
consumers. Independent roots/streams join only the pages they actually need.
Host code never selects work by downloaded occupancy, position or depth.

**Native tests on both backends:**

- Capture/reset/repeated large-to-small replay with frame permutations, partial
  pages, empty work and mixed request generations; no recapture/rebinding.
- Deliberately hold an old compute reader on one stream while another would
  restore/overwrite its destination; release it and verify both old and new
  consumers. Cover multiple reader streams and queued MTP/sidecar work.
- Retire/cancel while copies remain pending; obsolete/failing completion and
  actual frame reuse; both source and destination byte guards/identities.
- Observe submitted grids, useful work and real transfer/collective bytes;
  correct outputs plus early exits do not prove economy.
- Negative controls removing each required ordering edge must fail the intended
  assertion under controlled synchronization, not depend on storage latency.

**Exit:** each backend passes actual production capture/kernel/lifetime proofs,
including explicitly registered replay. All shipping GPU architecture targets
compile with spill guards; isolated profiling records occupancy, registers,
spills, throughput and memory efficiency. Host models and one GPU vendor do not
qualify the other. Missing hardware remains unqualified.

### P7 — Compose real topology-wide ownership

**Depends on:** single-participant contract and corresponding native adapters.

**Do:** thread logical block identity and coherent table/commit epochs through
rank/global orchestration, TP/PP/MoE continuation and MPI commands. Each real
owner grants its own physical shard; coordinate all-or-nothing admission and
rollback without remote aliases or independently reordered rank decisions.
Specify pipeline microbatch scheduling and exclusive simultaneous arenas.

**Tests:** singleton/odd/larger membership, unequal stage widths, more than two
PP stages, CPU/GPU mixtures and real NUMA ownership; participant refusal and
retirement; prefix sharing across actual local shards; skewed/empty/partial
work and large-to-small replay. Device-free tests exceed locally available
hardware counts; native tests report availability honestly. Assert live wire
extents and numerical reduction order as well as outputs.

**Exit:** admitted topology—not a fixed count—owns planning through retirement;
no padded capacity communication or hidden global slot/table authority remains.
Qualified capacity is recorded per real topology and feature combination.

### P8 — Complete numerical, lifecycle and feature qualification

**Depends on:** implemented cells from P2–P7 and P5; required before cutover.

**Do:** migrate every supported frontend, including benchmark, through the same
paged serving seam at capacity one. Explicitly version any changed window,
reasoning, prefix or RNG contract. Preserve existing supported laws and
restrictions, then separately qualify new features such as currently absent
GPU DRY. Retire obsolete fixed-cohort benchmark and scalar serving entrypoints
once their replacement is proven; do not leave a runtime alternative.

**Evidence:**

1. Fast complete Unit and complete configured model-free
   `ProductionTestPreflight`, with the canonical prerequisite receipt and exact
   source/build/test-inventory binding.
2. Recorded-input production replay, strict trace validation and native capture
   suites; new traces include page geometry/initial identities, table versions,
   valid extents, commitments, ownership/completion ordering and expected results.
3. Supported format/codebook/backend sweeps, representative N/K and every
   supported live M, CPU ISA/thread geometry, dense/MoE, hybrid, MTP/non-MTP,
   prefix/tier policies, stochastic/default sampling and dynamic maintenance.
4. Canonical multi-prompt frozen-history final-logit panels and byte-equivalence
   comparisons under the design's request/prefix/placement trace. Diagnostic
   top-k fixtures or one native full-vocabulary recording are not the panel.
5. Real HTTP/frontends: lifecycle, streaming/non-streaming, tools/reasoning,
   summary observations, passive stats, slow consumers, cancellation and reset.

**Exit:** every claimed cell has current evidence; no previously supported cell
is silently rejected or rerouted to legacy execution at cutover. Legacy failures
are fixed with regressions or remain blockers, not dismissed as unrelated.
Large certified corpora publish only through the optional corpus repository.

### P9 — Prove economy and retire the replaced machinery

**Depends on:** P8 correctness and native resource evidence.

**Do:** run matched Release offered-load and closed-concurrency serving workloads
on dedicated hosts, with defaults intact and explicit separately measured
controls. Include S = 1, varied lengths/output bounds, shared-prefix fanout,
miss-heavy traffic, long/short coexistence, cancellation/refill and slow delivery.
Compare the pooled implementation's memory and useful work to the retained
baseline and relevant external engines without enabling that baseline in the
new runtime.

Report physical pool/metadata bytes, live versus committed growth, sharing/COW
savings, attention/page-lookup overhead, copy contention, queue-inclusive TTFT,
output gaps, E2E, SLO goodput and unfinished work. CPU physical-core/NUMA scaling,
GPU profiling, actual launch geometry and transport extents are all required.
A completed-only percentile or faster kernels with growing queues is not a win.

**Exit:** correctness and economy pass simultaneously; all obsolete fixed-slot
allocation/estimation paths, full-prefix-copy resident hits, launcher/API names,
source-policy allowances and stale public claims retire together. Check code,
help/config, README/architecture instructions, tests and docs for consistent
names and capabilities. Use the repository's ordinary PR and release gates;
this plan grants no readiness, merge or image-publication authority.

### P10 — Later live-request KV offload and elastic admission

**Not enabled by P0–P9.** Design §11 is the required seam now; implementing this
extension is a separately scoped follow-up, not an emergency pressure escape.
Existing prefix-tier I/O still receives complete qualification in P3/P6/P8.

Before changing admission to allow overcommit, implement actual supported tier
chains, full request suspend/resume (paged KV plus recurrent/MTP/handoff state),
backing/staging capacity, clean/dirty replica versions, durable receipts,
effective-use recency and pressure-victim/fairness policy. Prove progress when
all active requests need another page; prove bounded waits or explicit outcomes
when every backing tier is full or unavailable.

Native evidence must include concurrent old readers and restore, queued/speculative
work, failed/cancelled I/O, delayed completions, hot GPU prefix/backing-tier
recency, repeated eviction/rehydration and capture reuse. RAM and SSD alternative
experiments do not prove a multi-tier promotion chain. External upstream fixes
are review leads, not Llaminar findings or patches to import without inspection.

## 6. Evidence and review protocol

At each coherent logical slice, submit:

- Exact base/head and scoped diff; newly added files and coexisting unrelated
  changes explicitly identified. Preserve old attempts and reviewer corrections.
- The production contract changed, source ownership map and focused regression
  registrations, distinguishing replay from lifecycle/geometry tests.
- Exact test selection, raw results/exits, compiler/backend/ISA/model/config
  identity and source-to-loaded-binary evidence where execution is claimed.
- What ran versus what is source-inspected only; focused versus complete gates;
  real native hardware versus device-free protocol models; unqualified cells.
- Actual memory/launch/transfer evidence for any economy claim, plus remaining
  obligations and the next dependency-unblocked slice.

Registration is not execution. File hashes preserve bytes but do not alone
establish loaded-code custody. Setup fixtures do not count as substantive native
tests. A focused accepted slice does not accept a coexisting whole working tree,
resolve complete-gate failures, or authorize increased production capacity.

Use the canonical [testing skill](../../../../.agents/llaminar-testing/SKILL.md),
[architecture map](../../../../.github/instructions/llaminar-v2-architecture.instructions.md)
and [production CI policy](../../../production-ci.md) rather than copying a
second build/test/release command matrix here. Documentation commit authority
is separate from production-code qualification. Any exceptional skipped gate
requires the operation-specific authorization required by repository policy.

## 7. First implementer deliverable

After adopting the documentation commit, return **P0's migration inventory and
the P1 implementation slice**: real PMA-owned page grants/commitments, typed
views/access leases, complete geometry, and focused production contract/replay
tests. Explain how the CPU implementation transfers to device-owned CUDA/ROCm
metadata without a host shadow. Include the source locations that will lose
full-context-per-slot allocation in P2.

Do not spend the next milestone finishing the old physical-slot MTP path or
renaming fixed slots “pages.” Preserve useful request-local work, but make the
pooled page authority and native cache-access replacement the new critical
path. Overall capacity and publication qualification remain open.
