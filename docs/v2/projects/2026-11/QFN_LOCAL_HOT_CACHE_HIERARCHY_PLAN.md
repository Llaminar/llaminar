# QFN participant-local hot caches ahead of shared ngram capacity

Implementation and qualification plan, 2026-11-06. This is a project plan, not a
tuning dashboard. The scope extends the existing pooled lookup implementation;
it does not replace the cache policy, graph executor, transport or memory
authority. The base checkpoint is `f0d4ba9a8`; the focused current-demand extension is
recorded in [its qualification note](QFN_POOLED_CURRENT_READAHEAD_FOCUSED_20261106.md).

## Outcome and current boundary

The requested hierarchy is **local VRAM cache → shared GPU pool → RAM → SSD**.
Each actual PLE consumer can retain immutable original-format rows locally,
while the shared pool owns miss service and its independently distributed cache
capacity. Expert-only followers may own shared capacity without becoming PLE
consumers. An immutable replica owns its own physical bytes, not another
authority over the source table.

Today `ConsumerLocal` and `DeclaredRankLocal` are alternative arrangements.
The latter already uses addressed native transports, but its replicated-reader
contract delivers expanded FP32 lookup output. Device-counted transport alone
does not implement independent local miss sets. Existing owner/source/MTP passes
do not qualify the combined hierarchy.

| Slice | Implementation | Functional qualification | Performance qualification |
|---|---|---|---|
| Existing ConsumerLocal | Installed | Existing scoped evidence | Existing measurements; remeasure matched controls |
| Existing shared pool | Installed; hosted verifier and ordinary retained identity repairs | ROCm TP2 fixed3/fixed1 exact replay plus current-demand fixed1 and ordinary reset/reuse pass; see focused note | No new matched comparison in this slice |
| Combined local hot cache and shared pool | In progress | Pending | Pending; no default promotion |
| Native compact responses and unequal consumer miss sets | In progress | Pending | Pending |

Keep these columns separate as implementation lands. A policy/parser, source
admission, CPU replay, native component test, real-model run and Release benchmark
prove different things. None can be substituted for a later qualification.

## Configuration and admission

Add an optional `AdaptiveGPULocalHotCacheRequest local_hot_cache` to the existing
`AdaptiveVRAMRAMSSDRequest`. Its common row and physical-byte ceilings apply to
each actual consumer; exact typed consumer overrides may replace those ceilings.
It reuses `CpuImmutableRowCachePolicy`. Unknown, duplicate or foreign endpoints
reject before materialization. Only `DeclaredRankLocal` accepts this component:
absence means existing shared-only, and an engaged zero-capacity component means
the combined path with no optional local slots. Existing `ConsumerLocal` remains
a first-class comparison, without stacking a second local cache on it.

Use distinct `--ngram-local-cache-*` controls and lossless document/wire fields.
Existing source/pool and RAM cache controls retain their meaning. Resolve one
checked local geometry per actual consumer alongside the unchanged pool layout;
do not reinterpret resident ownership, a replica-group declaration, or expert
placement as permission to allocate local cache bytes.

`PhysicalMemoryAuthority` remains the sole admission/materialization ledger:

1. Freeze all existing higher-priority model, expert, KV and runtime obligations.
2. Include the full mandatory local transaction, pool and source floors. Price
   actual cache metadata, compact miss/scatter arrays and transport controls,
   including C0, with the existing exact geometry/root BOM machinery.
3. Admit optional local native slots up to their explicit ceilings, then shared
   GPU slots, then the existing optional RAM row cache. This order is authored,
   not inferred from observed free memory or a duplicated reserve calculation.
4. Materialize each root once through its existing PMA claim and allocator.
   A cache ceiling includes its payload and slot-dependent metadata; mandatory
   frames remain independently visible. Compare total physical cost in timing.

All ngram capacity remains lower priority than frozen model/expert/KV resources.
Changing cache contents does not release its admitted pool or reconstruct a
graph. A RAM-backed source does not acquire a redundant RAM row cache. Full
original source identity and storage generation bind local roots at setup;
request reset preserves valid immutable rows, while model/table replacement
requires a new binding. No payload checksum or hash is needed for that identity.

## Captured lookup transaction

Extend participant-local stages and existing prepared roots; model graph files
continue to declare the policy. The native local cache uses the existing
`AdaptiveImmutableRowCache` operations, radix directory, slot generations,
pin-before-reserve discipline, aged frequency and protected/probation streams.
Do not copy those algorithms into a second cache implementation.

1. **Probe locally.** Validate the original query/count through its existing PLE
   authority, deduplicate keys and pin usable local rows. Publish local readiness
   immediately for an all-hit frame. Record a compact miss roster and stable
   inverse/scatter map on device.
   Original sixteen-head token extent and arbitrary compact miss extent are
   distinct typed values; never round the latter up to a token or capacity.
2. **Issue only misses.** Address compact row keys to the shared authority through
   the existing counted NCCL/RCCL or cross-vendor TransferEngine edges. All-hit
   participants publish only necessary bounded progress metadata. Local output
   readiness must have no dependency on an owner result or acknowledgement;
   an empty request/receipt round trip is not a local-hit implementation.
3. **Coalesce shared work.** Form a device-owned stable union where consumers
   request the same source rows. Preserve per-consumer inverse maps. Reuse the
   existing pool coordinator and RAM/SSD source rather than starting an extra
   host service for each local cache. Native response delivery may repeat a row
   for its actual consumers; source lookup/fetch need not repeat it.
4. **Deliver native rows.** Return exactly each consumer's required original row
   bytes, authenticated by table/generation/interval, format, ordered roster and
   byte extent. One main-stream LocalRead stage acquires the frame's ready
   frontier, decodes pinned local slots or native responses with the existing
   format implementation, and writes all original output rows. There is one
   model-embedding producer, with no expanded hit staging or cross-stream
   partial tensor publication. Shared-only remains an unchanged control until
   a separately measured change is justified.
5. **Publish and retire.** Keep Reserved slots unreadable until their payload is
   present and validated; Ready publication follows the actual producer. Demand
   consumption applies the existing promotion policy once, and pins retire
   only after actual readers. Cancellation/error drains original
   accepted work through existing lifetime authorities. Immutable eviction has
   no writeback or coherence exchange.

Static capture retains admitted worst-case storage but communicates live extents
only. Device counts and cache decisions stay on device. There is no host count
readback, per-invocation allocation, host callback, eager path or recapture.
Exact streams/events and original captured transport ownership remain mandatory.
An all-hit consumer still participates in required protocol progress without
forcing the model consumer to wait for unrelated owner service. Prove this
distinction with a delayed owner, not only a zero-payload counter.

Ordinary decode, prefill and fixed/dynamic MTP use the same typed transaction.
Grouped verifier and scalar MTP-condition graph identities/count semantics stay
distinct. Cache frequency reflects consumed demand, not unconsumed speculative
read-ahead. Request reset cannot invalidate stable topology or rebind captured
roots. Dynamic expert movement may contend for bandwidth but cannot invalidate
immutable ngram rows or silently reclaim their PMA-owned physical pools.

## Read-ahead and bounded local retention

Read-ahead and caching share the same lookup identity and native row lifecycle.
Distinguish early issue of already consumed/current demand from speculative
future work with a typed access intent. A later demand for an outstanding exact
table/generation/row request adopts that result once; it must not issue a second
owner/source fetch or credit the same access twice. Probe Ready local rows before
transport and share exact in-flight requests where their lifetime permits it.

Use existing bounded native response/request storage for speculative staging.
An unconsumed prefetched row does not earn demand-frequency/protection credit,
and prefetch does not reserve the whole local cache or evict protected demand
rows. Consumption applies the existing cache policy; a response can supply its
consumer without acquiring an optional cache slot. Prefill scan length therefore
does not dictate local-cache occupancy or retained pin count. A future read may
be omitted when the admitted lookahead/staging window is full; required demand
still has its first-class counted transaction. No background unbounded queue or
second RAM/SSD provider is introduced. Cold prefill favors compact batching and
useful chunk overlap; decode favors low fixed overhead and the local-hit path.
The same fixed lookahead depth is not presumed economical for both phases.

Known prompt chunks and already published MTP candidates are valid inputs;
unknown future sampled tokens are not. The existing producer frontier determines
how early work can start. Rejecting an MTP proposal or cancelling/resetting a
request retires speculative staging through the original transaction, without
invalidating immutable hot rows or leaving reserved slots pinned. Short decode
and all-hit workloads must not pay an owner round trip merely to run a prefetch
controller. The capture-safe local-ready frontier belongs to the actual cache
frame; independent auxiliary progress joins at required retirement, not before
the model's local-hit consumer.

Measure read-ahead disabled and enabled at the same total admitted bytes and
cache partition. Include local-hot decode, shared-hit/RAM/SSD misses, cold long
prefill, skewed consumers, rejected MTP candidates and sudden hot-set changes.
Report useful/wasted native fetches, duplicate suppression, consumer wait saved,
time-to-use, staging/pin high-water marks, protected-row survival and full-model
throughput. Earlier submission alone is not a performance win. Keep any adaptive
lookahead decisions in the existing device controller and test recorded inputs;
do not add a host policy that reads live counts or cache contents.

## Functional qualification

Extend existing cache/provider/pooled graph fixtures and the canonical typed QFN
matrix; do not create a second topology list or a synthetic serving path.
Register focused production regressions explicitly in `ProductionTestPreflight`.
During this iteration run those focused registrations, not the full preflight.

| Level | Required cases and evidence |
|---|---|
| Fast device-free | Real cache operations and root geometry: C0/positive/unequal capacities; all-hit/all-miss/mixed; duplicate/disjoint/overlapping consumer rosters; missing/foreign identities; exact fit and one-byte-short admission; every supported native row format and 32/64-bit keys. |
| Lifecycle replay | Pin/eviction interleavings, scan pollution, hot-set changes, stale/reserved slots, partial and empty frames, large-to-small replay, reset, table replacement, error/cancel and repeated execution. Test source, shared and local generations separately. |
| Native CUDA/ROCm | Actual captured producer/consumer kernels and transports, poisoned inactive tails, output byte agreement, exact per-edge request/native-result/metadata bytes, repeated counts including zero. Hold owner service and prove all-hit output is independently ready. Compile shipped CUDA SM80/86/89/90 and HIP gfx906/gfx1100 with the normal spill guard. |
| Placement | Original RAM and bounded mmap SSD sources; RAM cache on/off; shared owners colocated with continuation, follower expert, or both roles. Empty owners, non-owner coordinator, reader-only participants and unequal local caches remain valid. |
| Real model | Project the existing six hardware rows: MI50×1/2/4, RTX3090×1/2, and RTX3090×2 continuation plus MI50×4 expert followers. Use original GGUF, admitted source and immutable table lifetime. Run actual ordinary generation and Off/fixed1/fixed3/Dynamic MTP witnesses with repeated requests/reset/prefix restoration. |
| Accuracy and experts | Existing frozen-history numerical panels report prefill/decode mean KL and top-1/top-5; diagnose drift with canonical layer/stage PyTorch CSVs. Preserve serial/grouped byte equivalence. Dynamic balancing must demonstrate actual published expert migration and live payload while ngram caches remain correct. |

Use cheap metadata/ops coverage for the full policy product, with complementary
native/model projections for real transports, lifetimes and arithmetic. Each
advertised topology/mode needs its actual execution witness; unavailable or
unfinished cells remain unqualified, not successful skips. CPU-backed sources
are required for all GPU configurations; existing CPU lookup/cache tests remain
the source-format and host-policy controls.

## Matched performance qualification

Compare **ConsumerLocal**, **shared-only**, and **combined** at matched total
physical VRAM cache budgets, counting local plus shared metadata and payload.
Keep the RAM budget, original source, expert residency, arithmetic, MTP intent,
token IDs and context fixed. Record admitted bytes per participant and tier;
nominal row count alone does not establish a fair comparison. Use several
explicit local/shared splits rather than spending unlimited capacity on replicas.

Extend the existing lookup measurement and canonical Release benchmark drivers.
Record realistic QFN token/ngram traces from cold prefill, sustained decode,
repeated coding requests, partial-prefix reuse and changed conversations. Retain
original request/phase boundaries and speculative/consumed status. Include a
held-out trace, sequential scan pollution and working sets above each tier's
capacity. Synthetic IDs remain adversarial controls, not the sole economy proof.

For every policy report:

- End-to-end prefill and committed decode tokens/s with MTP work/acceptance, plus
  complete lookup consumer-stall latency p50/p95/p99 and separate submit/event
  scopes. Timing runs are unprofiled and interleaved with matched controls.
- Local hits, shared hits, RAM service and SSD service, by original demand and
  unique native rows. Specify denominators; a window mapping hit is not an SSD
  cache hit. Preserve per-participant and total views.
- Actual request/native-payload/control bytes per edge, dedup savings, necessary
  empty progress metadata and useful/wasted read-ahead. Do not infer wire bytes
  from admitted capacity or count all replicas as unique source rows.
- Promotions, evictions, protected/probation retention, pin occupancy, turnover
  per useful row, resource peaks and bounded steady-state allocations. OS page
  cache residency/fault/read-byte evidence remains separate from explicit RAM.

Collect cold-source evidence using only fixture-owned files and the existing
page-cache observation procedure; never clear unrelated host cache. Include warm
repeated and warm changed-hot-set trajectories. Profile new kernels with the
backend skill's isolated shape/ISA procedure, documenting registers, spills,
occupancy and bandwidth separately from canonical timing. Profile transport
waits to explain gains or regressions; no direct P2P is assumed on this machine.
NCCL/RCCL may use P2P when available without a new application-level transport.

Promotion is per proven topology/workload policy. Correctness and economical
native traffic are prerequisites; an isolated hit-rate increase does not justify
a default if complete throughput or tail latency regresses. Preserve explicit
capacity-oriented modes even where another mode is faster. Dashboard rows link
the exact source/settings/evidence and keep unqualified columns visible.

## Delivery sequence

1. Add typed policy/geometry/PMA, production local cache operations and compact
   request/response contracts in parallel, with focused host regressions.
2. Join their native roots, participant-local stages and original model binding;
   prove coherent builds and actual captured all-hit/miss/mixed transactions.
3. Qualify the real model on representative CUDA, ROCm and hybrid slices with
   ordinary/MTP and dynamic experts, then cover the remaining canonical rows.
4. Run matched Release trace and model benchmarks. Tune measured bottlenecks,
   preserve numerical gates and promote only the combinations with evidence.

Checkpoint meaningful green slices. Implementation progress never changes a
pending functional or performance column to green automatically.
