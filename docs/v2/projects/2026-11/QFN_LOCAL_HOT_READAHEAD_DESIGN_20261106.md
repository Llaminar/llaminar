# Smallest genuine future-prefetch extension — integration proposal

The independent CPU cache lookup-only operation is implemented in the current
integration tree. The normal owning build and both focused registrations pass with clean
driver observation: the complete existing cache suite and the explicit new
lookup-only regression (`cpu-lookup-only-owning-focused-r1`, 2.06 s). The closed GPU-source protocol, native
adoption and future producer/consumer graph attachment below remain unimplemented.
No model future-prefetch mode is enabled. The first implementation should retain one admitted L1 frame/native bank and the existing source/owner/transport authorities. Do not add another cache, source, free list, budget rule, stream waiter or generic capability flag.

## 1. Make speculative fetch genuinely nonpromoting at every traversed owner

Add a closed `FutureLookupOnly` access contract consumed by the original shared coordinator/native source and CPU subordinate source read. It is not a new replacement policy: it performs directory lookup/pin/native read, but never calls frequency/recency/reserve/victim/admit/invalidate operations. Existing Current behavior stays unchanged.

The shared coordinator reuses the same stable coalescer/inverse map and authentic actual owner pin/read grants. Only missing unique rows reach the same RAM/SSD service. It needs explicit lookup-only phase completion to reach original source-readable/retirement states without fabricated zero allocation receipts. This must not enter `planFreeImpl`'s demand-observation loop or `applyQueues`. Source native pack still copies only each actual consumer's compact misses; zero rows send protocol metadata only.

CPU `Read` is not safe to relabel: it touches/admit rows. The independent
`CpuAdaptiveImmutableRowCache::gather(..., FutureLookupOnly)` operation now
provides the typed nonpromoting read through the SAME cache/original source: preserve actual native copy, duplicate handling, pinned-hit and failure lifetime, but omit clock/frequency/touch/admit/invalidate. The existing demand ABI remains unchanged, and subordinate GPU processor
requests explicitly reject this independent entry; an authenticated future
processor grant still needs implementation. The focused twin-cache regression
covers the canonical native format inventory, C0/1/4 slots, repeated mixed and
duplicate scans, pinned reads, invalid/empty input and retirement. Ordinary
RAM/SSD source identity and exact request/response extents remain authoritative. The source's bounded physical mmap-window lifecycle is still an I/O mechanism, not hot-cache priority.

For this first slice, **future fetches do not earn lower-tier admission/priority**. Actual consumption promotes through the existing L1 `consumeDemand` policy once. This is an explicit read-only prefetch operation, not a silently weakened ordinary demand policy. If the requirement instead includes retroactive shared/RAM admission for consumed future fetches, that is additional closure (below), not a flag we can advertise in this slice.

## 2. Authenticate actual demand before adoption; retain one real bank

A future request must originate from an actual native token/history producer—e.g. original MTP proposal output plus its genuine prior-state snapshot—not guessed future IDs or a host mirror. Bind producer identity, original request/table/storage generation, live row range and access intent into capture identity.

`adopt(query)` alone is not sufficient for model integration: it checks the retained frame identity, but does not prove that actual demanded rows equal the prefetched prediction. Add an adoption boundary that compares the actual demand's complete original ordered IDs/count against the retained `unique[original_to_unique[i]].row` (native parallel O(N), no second source lookup/sort/scratch). Same table/source/root/request identity and bytes must match. Accepted demand adopts the same frame; rejected MTP work disposes it through the original reject branch. A mismatch is an explicit declared prediction-miss/discard result, never success or an implicit execution fallback.

Initial hits remain pinned; fetched bytes stay only in the existing response bank until actual LocalRead. Neither DataReady nor arrival ACK grants policy credit. LocalRead alone writes the model output; read-complete→consume→fill→Ready→retire stays ordered exactly as now. Request termination/reset must physically discard/drain any unused bank before publishing clean terminal state. Normal discard is reusable; source/ABI failure remains sticky.

## 3. Avoid a cross-parent wait deadlock

Current local aux includes ready2→consume/fill/retire in the same parent. A future **producer** parent cannot wait for LocalRead in a later parent: its terminal would prevent that later parent from launching.

Introduce a narrow typed future-prefetch range (distinct from current complete-demand range): Probe→native result acquisition→actual source/transfer retirement, retaining the bounded local future bank. It has no model output writer and makes no LocalRead/consumption claim. A later original consumer/reject branch owns Adopt+LocalRead+maintenance or Discard+retire. Reuse existing parent fork/import and original generation/ticket ordering; no standalone aux executable, host split, recapture or new raw wait. Preserve existing strict current-range validators; never accept a body-less current parent.

The missing producer/consumer binding must be added to the original model/PLE/generation declaration. Merely changing Current to Future in `appendLocalMain` is incorrect. Ordinary sampled-next-input, fixed MTP and dynamic MTP require their own actual producer/reject edges; absent lineage remains a precise unsupported candidate. Start with one explicitly bound adjacent producer/consumer transaction, not a matrix or multiprefetch redesign.

## Cost / ownership expected before implementation

- Reuse existing per-consumer cache/frame/scratch/native-response roots and mapped ready page; no second payload bank or additional hot slots. Current max-inflight remains1.
- Request88/Result96 already contain intent. Any new disposition/target fields must be laid out/priced by original Geometry/BOM; do not assume padding is free. Existing data-arrival ACK remains data-arrival evidence, not consumption proof.
- No per-lookup host count/state transfer. Native adoption validation adds one O(original IDs) comparison pass; existing static parent/child dependencies carry producer/consumer ordering. No extra native event is assumed: confirm actual cold creators/executables and let existing graph-memory/PMA projection price any growth.
- Source transport technology/cohort and original lookup ordinals are unchanged. The read-only branch requires actual source/owner completion, so all original participants still follow the same collective order.
- Any changed CUDA/HIP kernel requires symmetric captured replay, exact format/key/bucket coverage and all shipped architectures/spill resources. Actual timing/profiling then proves useful/on-time/wasted transfers and lookup wait avoidance separately from source/parent retirement.

Safe ownership split: local owner handles adoption/read/discard and private bank lifecycle; source owner handles typed lookup-only coordinator/CPU source and wire validation; root owns actual PLE/token-producer and retained parent/generation attachment; planner owner verifies unchanged/new cold BOM from actual geometry, never new admission arithmetic.

## If retroactive shared/RAM demand credit is required

It cannot be added as a boolean after the old arrival ACK. The original policy owner needs an authenticated consumed/discarded disposition tied to the exact retained query, and the original native/host byte provenance must remain valid until it applies the same demand policy. A bounded semantic receipt can travel on the existing directed channel, but the single shared/source frame must not be reused while deferred policy work owns it. A producer parent must still finish without waiting for a future consumer. Holding source/owner leases across parents changes original SourcePlan retirement and CPU response-bank lifetimes and needs explicit typed closure; it is not honest to claim zero extra state/cost or simply delay a callback. Decide this contract before implementation. No such consumer receipt exists in current code.
