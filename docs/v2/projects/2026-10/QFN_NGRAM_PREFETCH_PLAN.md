# QFN ngram prefetch plan across VRAM, RAM, SSD and remote GPUs

Design and source review, 2026-10-31. Early-lookup candidates are implemented
in the isolated tuning tree; native qualification and default promotion remain
pending. The status below distinguishes completed host checks from native proof.
This is a regular project plan, not a size-bounded tuning dashboard. The review
used no native jobs, builds or GPU access. Read-only remote main resolved to `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`; the prior
baseline remains `011c9e600a93bbc9fd451c563501b5831c32bbca`. Both objects were
examined in the new isolated `Strata/` checkout; older baselines were untouched.
This plan complements [the lookup performance plan](QFN_NGRAM_LOOKUP_PERFORMANCE_PLAN.md)
and [tiered-storage admission](QFN_NGRAM_TIERED_STORAGE_PLAN.md). Early lookup
is an all-tier operation; SSD is only one miss source. Existing benchmark pins
remain unchanged. Review provenance is retained in
`/tmp/qfn-strata-prefetch-review-r1/manifest.json`.

## What Strata actually implements

The links below are immutable source references at current main.

| Mechanism | Source evidence and boundary |
|---|---|
| Earliest IDs | [MTP early hash](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/core/mtp.cpp#L1657-L1671) copies host `ple_prev`, hashes each available token and advances a local candidate history before issuing its rows. This is host token knowledge, not GPU-only execution. |
| Verify window | [Verifier request issue](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/core/verify.cpp#L3089-L3099) hashes and prefetches every known input token before launching the verifier. |
| Prefetch slots | [16 token frames](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/kernels/ngram.cpp#L240-L244), [issue/reuse](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/kernels/ngram.cpp#L470-L500): compare complete sixteen-ID keys, deduplicate matching frames, drain all when full, and keep original packed bytes. Windows mmap uses PrefetchVirtualMemory; this branch is absent on non-Windows. |
| Match by identity | [Gather reuse](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/kernels/ngram.cpp#L603-L645) matches each requested frame against any retained frame, allowing reordered/subset/duplicate requests. Exact matches decode directly from retained raw bytes, without issuing another ticket or routing through the row cache. It drains every outstanding frame, including unused ones, and preserves the first collection error. |
| Prompt lookahead | [Prompt gather](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/prefill/prefill.cpp#L3058-L3077) derives correct chunk predecessors from prompt tokens. [Pipeline](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/prefill/prefill.cpp#L3229-L3284) starts first-chunk reads beside layer zero, then gathers the next chunk into the other buffer. Raw native bytes can upload on a side stream; copy/free events protect device raw storage. It also uses host `future.get`, `std::async` and `cudaEventSynchronize` to guard reuse. |
| Direct source grouping | [Reader issue](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader.cpp#L422-L489) checks row cache, groups rows sharing a starting 4096-byte page, extends to two pages for straddling rows, then sorts jobs by file offset. Grouping is per issue, not a global deduplication of all outstanding tickets. Overlapping two-page jobs with different starting pages can still duplicate physical reads. |
| Bounded I/O buffers | [Setup](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader.cpp#L337-L375) allocates a fixed aligned `max_inflight * 2 * PAGE` slab and retains a reader worker. Dynamic maps/vectors/queued jobs and tickets are separate from this bounded slab; a fixed hardware queue does not prove total metadata admission. |
| Linux implementation | [DirectFile](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/platform/direct_file.cpp#L264-L326) uses a thread pool doing blocking O_DIRECT preads, with pool size tied to queue depth. This is not one free asynchronous syscall or a GPU DMA implementation. Windows uses unbuffered overlapped reads. |
| Cache admission | [RowCache](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader.cpp#L35-L74) is eight-way, hashed-set, round-robin replacement. [Completion](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader.cpp#L228-L249) inserts every successfully read row, including unused speculative rows. There is no demand-versus-prefetch protection in this cache. |
| Lifetime/failure | [Close/drain](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader.cpp#L380-L418) drops unsubmitted jobs, joins worker and drains actual in-flight reads before freeing aligned buffers. [Sticky-error guard](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader.cpp#L432-L437) prevents queueing tickets behind a parked failed worker. `PleTable::wait_prefetches` drains with ignored errors; Llaminar must preserve fatal publication evidence instead. No equivalent captured device invocation/reset-generation authority was established in this host-ticket implementation. |
| Regression value | [Prefetched-key tests](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/src/ngram/ple_reader_test.cpp#L250-L326) cover reordered, duplicated, subset and extra frames; later tests add concurrent prompt/verify usage and discarded drafts. Port these invariants, not a positional-slot assumption. |

At old `011c9e6`, the 16-frame prefetch machinery, identity-based gather reuse and
MTP token-time issue already existed (`ngram.cpp` lines 238,425,543;
`mtp.cpp` lines 1611-1619). Current `ple_reader.cpp` adds the important sticky
failure issue guard and ticket erasure on failed collect. This is a meaningful
failure-lifetime lesson beyond the old baseline. Current prompt and verifier
files have substantial additional changes; this review does not certify all
of them or conflate all speed changes with prefetch.

## What is measured, and what is only asserted

Strata's [Intel performance table](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/docs/INTEL.md#L474)
reports first-chunk PLE beside layer zero changing a 2,184-token prompt result
from 784 to 825 tok/s. Its [combined optimization table](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/docs/DETAILS.md#L1546)
reports +2.2% to +2.6% prompt throughput for four switches together: compact PLE,
PLE copy stream, resident embedding and side-stream dequantization. That cannot
be attributed solely to PLE prefetch. Source comments mention ~0.4 seconds of
32K-prompt first-read idle and 1.8ms/round on Strix; these are author observations,
not independently authenticated benchmark receipts in this review. Source
mechanisms and correctness tests are present; no Strata or Llaminar benchmark
was executed here. No quantified all-tier/remote-GPU gain was established.

## All-tier Llaminar design

Compute exact IDs on the current authority as soon as token/history/count inputs
are published, not when PLE trunk activations arrive. Existing QFN hashing uses
only current+two preceding tokens, reset semantics and frozen head constants.
Today PLEBuilder attaches hash to the incoming layer trunk. Move the *data-only*
producer edge earlier while retaining the same native hash/count and immutable
history bank. Prefill rows are known only when their retained prompt slice is
published; decode rows only after their token is sampled/accepted; MTP rows only
after candidate-token publication, with rejection counted as waste. The next
ordinary unsampled decode token is not known.

Separate `prepare/read-ahead` from `consume`, and preserve identity through a
bounded retained frame keyed by source/table/version, invocation/generation,
sequence, exact IDs, row precision and placement-policy generation. IDs are
GPU-owned for GPU inference. Host receives only explicit immutable source-I/O
requests, never host hash/history/sampler/verifier shadows. All launch geometry,
workspace generations and embedded pointers enter capture identity.

| Actual tier | Economical early action |
|---|---|
| Local resident VRAM | Hash early; optionally perform the existing native gather/decode into the original eventual PLE embedding buffer while trunk work runs. Keep immutable source authority. Do not copy rows into another VRAM cache just to call it prefetch, and do not claim controllable hardware-cache residency. Measure contention: early memory work may slow trunk kernels. |
| Local pooled VRAM hit | Move the existing complete pin/read transaction earlier where its controller dependencies permit. Retain a read lease or original output until consume; never bypass pin/victim/source seals or pin an entire allocation. Long-lived speculative pins need an explicit bounded quota so they cannot block useful demand. |
| Whole RAM / adaptive RAM hit | Early ordinary Publish; existing host worker gathers exact useful native rows once. Retain its response until the original GPU consume. Copy once through the admitted channel, then native GPU decode; do not create an extra CPU cache mirror. Adaptive-cache hits must be observed by its sole existing admission policy. |
| SSD miss | Deduplicate rows/pages, order/coalesce actual required aligned extents, use explicit DirectSSD for strict page-cache isolation, and read into PMA-admitted persistent staging. Existing mmap remains a separate explicit user choice whose cache advice is not a hard bound. Keep response order by a bounded reorder map. |
| Remote same-vendor GPU | Send exact compact owner-needed IDs and acquire immutable native rows through the existing explicit peer/source graph transport. Read in place where existing authority/transport truly permits it; otherwise move each necessary native row once into its actual final consumer frame. No broadcast of unused heads, rows or capacity. |
| Remote cross-vendor GPU | Use the existing named cross-vendor source boundary and admitted staging. Keep authoritative owner admission/pins on the source device, explicit transport receipts/events and reader retirement. Stage only missing consumer-needed native bytes; no host model-state echo or invented native P2P. Start early only after the complete source seal. |

A collective is not required merely because prefetch exists. Use existing
owner routing and movement; add a collective only if a declared ownership/query
contract actually needs it. First-slice local success does not qualify remote
cohorts. For pooled paths, early Pin->Plan->Seal->Publish->AcquireResponse and
retirement must be moved as one lawful transaction prefix; moving source
Publish alone ahead of victim/source seal is invalid.

Protect demand: exact current-invocation early lookup is known demand and can
retain the ordinary policy's admission semantics. Future prompt chunks and MTP
candidates are lookahead. Mark them explicitly so an unused prefetch neither
bumps demand frequency nor displaces protected hot demand. Retain speculative
bytes in bounded staging/probation rather than silently adding a second cache;
only a real consuming observation may advance existing admission policy. If
probation has no budget, decline speculative issue and record it; ordinary
demand remains the same first-class source path, not an alternate fallback.
Deduplicate against resident and already-issued identities before movement.
Match subset/reordered/duplicate consumers by identity, preserving original
response order and reference-counted reader retirement. Avoid a full-frame copy
when the actual consumer can directly read the retained immutable result.

Budget every physical buffer and metadata capacity with PMA: lookahead depth,
rows, owner fanout, aligned extent bytes, reorder entries, response banks and
pinned leases. Cap total queued descriptors as well as hardware in-flight I/O.
Start at depth one, current invocation only. Later prompt lookahead may use two
exclusive persistent banks, protected by exact-stream completion edges. No
std::async per chunk, blocking host event reuse, new unbudgeted I/O pool,
hot allocation, new host count readback or conditional recapture. Preserve one
complete captured homogeneous GPU parent (or the already declared authenticated
HIP MTP transaction policy).

Slots need typed Free->Issued->Ready->Consuming->Retiring->Free transitions,
plus Cancelled/Failed retirement. Reset changes data generations, not topology.
Cancellation suppresses further issue, drains actual reads/transfers, and waits
for all actual readers before storage reuse. Dropped MTP drafts do not commit
history, cache demand statistics or model state. A late old-generation response
cannot satisfy a new request. Source failure remains sticky/fatal with original
completion evidence; do not copy Strata's ignored-prefetch-error convenience.

## Capture scheduling constraint

The [capture design](QFN_NGRAM_PREFETCH_CAPTURE_DESIGN.md) names exact source
functions, input/history ownership, stage contracts and causal overlap tests.

The established executor assigns all stages in a captured program the same native
capture stream. Removing the trunk dependency alone neither establishes an
earlier issue order nor overlaps a GPU gather with trunk kernels. Kahn-ready
root ordering must not become the scheduling authority. RAM/mmap service can
still overlap through its existing worker: explicitly order exact input/history
publication -> hash -> Publish -> embedding/trunk, then Consume at the PLE
consumer. The GPU continues useful work while the existing source worker reads.

Local VRAM gather needs an executor-owned retained stream fork/join to overlap
GPU computation. A per-stage stream override is overwritten by current capture
and is not a solution. Qualify early issue with the current stream semantics
first; add declarative side-stream scheduling, exact producer events and
concurrency-aware arena lifetimes before claiming VRAM overlap. Remote movement
likewise needs its actual complete captured collective boundary. The all-tier
API must describe these distinctions without advertising unimplemented overlap.

## Small first slice and proof

First split ordinary immutable GPU lookup Publish/Consume for one current
invocation, hash immediately after exact token/prior/count publication and
consume at PLE. Qualify local resident VRAM and ordinary whole-RAM paths first
using their existing storage/output banks; this tests the general overlap
mechanism without requiring a new SSD policy. Include ordinary mmap as its
explicit existing policy, labeling warm-page versus verified disk activity.
Then implement/qualify DirectSSD, adaptive pooled and remote paths separately
under the same typed producer/consumer contract. Do not advertise unimplemented
combinations. Speculative next-chunk/MTP lookahead comes after exact-demand
identity, reset and admission regressions are green.

Measure terminal used/wasted native bytes by tier, unique rows/pages, duplicate
suppression, source physical bytes, staging occupancy/high-water, pin duration,
cache admissions/evictions, demand displacement, queued/in-flight depth and
cancellation/rejection waste. Count actual movement once; label identity reuse
separately from a new hit. Record GPU Publish->Consume and intervening useful
compute, consumer wait, full request latency/tails and worker CPU duty. Report
remote transport volume and actual owner skew, including empty owners and
large-to-small replay. Cold SSD proof needs actual residency/read evidence;
local VRAM proof needs evidence early gather did not merely contend with trunk.

Replay identity reorder/subset/duplicates, partial counts, reset, cancellation
and failures at every transition; prove no stale completion, leak, hot-set
pollution or extra transfer. Add explicit production-preflight replay membership
and actual CUDA/ROCm captured integration coverage. All-format byte parity,
controller arithmetic/ordering and existing terminal volume gates remain intact.

## Direct SSD policy details

Use the exact immutable source descriptor and validate filesystem offset, length
and buffer alignment before admission. The explicit source policy owns PMA
reservations for persistent aligned staging, extent descriptors and response
reordering. Bound total queued work as well as in-flight reads. Translate native
rows to necessary aligned extents, deduplicate overlaps (including rows that
straddle pages), merge only within a declared read-amplification budget and
scatter original bytes back into response order. Preserve all supported native
formats; do not requantize the table.

A depth-one implementation may perform synchronous direct reads in the existing
retained I/O worker while the GPU performs useful independent work. Measure
whether concurrent I/O is needed before adding a bounded implementation. Short
reads, EINTR, EOF, unsupported alignment and read failures need exact handling;
an inconvenient tail must not silently select buffered I/O. Keep mmap as its
own first-class user-selectable policy. NOREUSE/DONTNEED advice is not a hard
OS-page-cache bound. File-local cold controls and actual physical-read evidence
remain required; never drop the host's global page cache.

## Implementation completion criteria

1. Complete the producer/consumer dependency and alias-lifetime audit. Prove the
   captured scheduler can overlap early source publication with the trunk and
   that prior history stays immutable until hashing retires. No graph-root
   assumption may replace the exact token/count/history producer edges.
2. Implement ordinary single-invocation early lookup using original banks and
   native GPU-owned inputs. Cover local VRAM, whole RAM and explicit mmap on
   CUDA/ROCm; CPU preserves its own host authority. Do not claim performance
   benefit before paired Release runs and actual overlap evidence.
3. Extend the same contract to pooled/local and declared remote cohorts only
   after the complete pin/seal/source/read-retirement protocol is preserved.
   Same-vendor and cross-vendor paths each require their native transport gates.
4. Add depth-bounded prompt/MTP lookahead, exact-ID reuse and demand-protected
   speculation only after the current-invocation gates pass. Test reordered,
   subset, duplicate and rejected candidate consumption, including reset and
   large-to-small replays. Expose used/wasted bytes by tier and capacity peaks.
5. Introduce the named direct-I/O policy with its admission, all-format, failure,
   cancellation and OS-cache tests; qualify cold SSD and mixed-tier economics.
   Maintain device-free controller replays explicitly in production preflight
   and native capture/communication regressions in CUDA/ROCm integration.

## Current implementation and qualification

The ordinary consumer-scheduled policy remains the default. The independent
NgramHash capture-identity repair now seals token/history/count/output bindings,
geometry and hash policy without changing integer arithmetic. Its coherent
Release rebuild and 30 affected CPU tests pass. Explicit CUDA/ROCm cold-hash
preflight regressions now pass natively: actual executor output
allocation, changing tokens/history/live count, partial/empty work, stable
addresses and retained graph identity. Evidence:
`parity-results/qwen-flash-next/ngram-lookup-perf-20261031/hash-hosted-native-r1/`.

The hosted early-current candidate is integrated into the isolated Release tree.
The generic builder owns one lexical frame, and the real PLE consumer consumes
that exact frame after trunk readiness. Existing hosted Publish/Consume kernels
retain source/response ownership; there is no new native stream or response bank.
The coherent Release core builds for CUDA80/86/89/90 and HIPgfx906/gfx1100
with the spill guard enabled. The affected metadata, model-hook, hash/lookup
and resource ownership tests pass 183/183; process/session cleanup is empty.
Both native hosted registrations pass and replay the actual graph-local frame,
Hash, Publish, independent residual expansion and delayed Consume. It checks
twenty changed request epochs per RAM/mmap tier, partial/empty work, exact
bytes, untouched tails, live communication and final PMA retirement. The test
uses the production executor and dependency ledger to record internal producers;
its original direct-stage recording failure is retained in the first run.
Driver observation and owned-session retirement pass on both backends.
Actual-model eligibility, causal overlap and matched throughput
remain separate unqualified gates. Evidence:
`parity-results/qwen-flash-next/ngram-lookup-perf-20261031/read-ahead-host-validation-r1/`
and `parity-results/qwen-flash-next/ngram-lookup-perf-20261031/hash-hosted-native-r2/`.

The separate resident candidate declares consumer-scheduled Prefill and early
side-lane Decode explicitly. Its generic event-DAG/phase replay passes 15 tests
with 20 repeats in the private agent fixture. Three strict diagnostic-selector
tests and three snapshot-ordering tests also pass 20 repeats. Original resident
gather kernels are reused. The integrated snapshot path records the side
producer's immutable copy before its join event; later manual/ticket stages
retain their original copy owner. Native overlap and snapshot tests still need
qualification. The production resource catalog is empty and fails before creation. The
model-free native test has a separately admitted parent-terminal event. Whole
parent lifetime, exact resource BOM consumption, safe failed-test retirement,
native causal replay and side-stream numerical snapshots must qualify before
promotion. A concrete ExpertOverlay composition joins at the first declared
native-unit boundary before a CPU ticket and imports that same completed
embedding at the later PLE consumer. This preserves the existing heterogeneous
parent and expert maintenance. The combined Release source builds successfully
for all shipped CUDA/HIP architectures, with spill guards enabled; native
qualification remains pending. No native overlap or full-model speedup has been
measured for this candidate.

Fresh linked device-free checks against that Release core pass 221 tests plus
the isolated original-pool provenance replay. The workspace declaration check
uses real selector admission for RAM, mmap and VRAM on both GPU backend types,
at 1, 17 and 128 rows, and proves the same buffer capacities as the corresponding
consumer-scheduled policy. The original compile failure from a stale PLE builder
signature check and the original missing-placement fixture failure are retained
in the focused receipts. This is a focused linked replay, not the complete Unit
or preflight gate. Evidence:
`/tmp/qfn-read-ahead-root-integration-r1/linked-cpu-r6/result.json`.

The lifetime integration consumes the original successful terminal publication
through a typed capability carrying its event owner, producer stream and
submission epoch. It no longer records the same terminal event twice. The
native fixture separately reserves its declared graph obligations, charges the
native eight-byte ROCm timeline signal, and uses the existing TransferEngine
retirement transaction to verify reset and release of the original PMA tokens.
These latter changes still await the combined native qualification run.

The explicit resource calibration executable builds as
`v2_perf_read_ahead_resource_calibration`. Its driver has twelve device-free
replays, including owned-process TERM/KILL/reaping, partial output and continuous
non-checkpoint output. The first native probe rejected missing process-entry
ownership before GPU context creation; its clean retirement and original red
receipt are retained. The probe now uses the existing capture-safe copy API.
Its passive library observer inspects only regular files: opening mapped driver
device nodes had generated NVIDIA ioctl warnings. The focused negative control
fails on that original observer and passes with the fix.

Both backends now pass 24 process lifetimes each: 20 cold processes, one process
with 20 repeated family lifetimes, and three partial-creation controls. Each
backend completes 800 captured replays with producer/consumer poison before
every launch, exact output checks, real context-reset receipts, and clean driver
windows. Evidence:
`parity-results/qwen-flash-next/ngram-lookup-perf-20261031/read-ahead-resource-calibration-repeat20-r2.json`.
These resource observations do not certify model overlap or throughput.

Reviewed family limits are 1 MiB host plus 2 MiB CUDA or 4 MiB ROCm device
backing; the separate single Ordering-event test parent uses 16 KiB host and
zero additional device contribution. The largest observed retained family host
delta was 261,552 bytes; the conservative creator-failure observation reached
655,808 bytes. ROCm retained a 2 MiB side-stream slab after handle destruction.
Its separate graph-instantiation allocation remains in the existing graph
contract. The original context must retain the same PMA obligations until reset;
handle destruction cannot release that capacity. Installing these limits awaits
that lifetime implementation and actual resource-owner replay qualification.
The intended contract
is reviewed conservative, version/population-bound tested opaque limits with
headroom and drift rejection, as approved by the user. Explicit buffers stay
exact. Backing retained by a driver pool must retain the same original context
obligation through actual context retirement; handle destruction alone is not
evidence of reclaimed physical memory.

Continuation and read-ahead qualification must also preserve the existing Qwen
MoE dynamic expert balancing behavior. Ngram storage owners are independent of
expert owners. Pair fast source admission with separate inference/replay tests
that witness real expert movement, compact live transfers, ordered publication
and numerical correctness, including supported mixed-vendor compute domains.
Multiple ngram storage owners alone do not certify multi-device continuation
compute. Future lookahead and direct SSD I/O remain separate implementation steps.
