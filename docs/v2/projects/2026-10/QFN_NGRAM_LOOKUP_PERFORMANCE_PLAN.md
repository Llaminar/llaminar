# QFN ngram lookup performance and tier-churn qualification

This plan measures the production lookup operation consumed by a GPU continuation.
It complements source admission and numerical model parity; a component timing
is not a full-model, topology or serving certificate. This is a regular project
plan, not a tuning dashboard. Detailed samples stay in ignored result artifacts.

## Measurement contract

Use one source-consistent Release core and the existing production provider,
lookup service, graph builders, captured participant programs and TransferEngine.
Synthetic original IQ4_NL blocks use QFN's actual width160,90native bytes/row and
16 UInt64 IDs/token. Record the seed, source extent, original head/ID pattern and
cache history; do not hash an entire model/table/cache payload. Retained real
model ID traces, when available, are a separate labeled workload.

Compare fixed continuations independently: CUDA RTX3090 and ROCm MI50 first;
RX7900XTX is a distinct architecture row. Never substitute a different card for
a named endpoint. A multi-card lookup names the continuation, every physical
storage owner and their exact native UUID/PCI identity; it does not imply full
model tensor/pipeline parallelism was exercised.

Measure M1,8,64,512 first, then neighboring/partial/empty and larger admitted
buckets. Cover distinct and duplicate-heavy IDs, uniform and skewed ownership,
small/large cache capacities and stable versus changing hot sets. Report actual
unique IDs, tier hits/misses, wire payload and metadata bytes, fills, evictions,
RAM admissions/invalidations and source-window work. Allocation capacity must
never be mistaken for useful or communicated data.

Validate exact decoded output and inactive tails before/after timing. Time the
complete required transaction, including publication, reads and retirement.
Report source-device event time and host submit-to-completion time separately;
remote wait and host-provider service are part of the critical path. No
allocation, upload, result readback, oracle or per-stage synchronization belongs
inside a timed operation. A necessary terminal event/worker join is result
collection, and its scope is reported. Do not subtract event or host overhead
using an invented constant. Keep raw samples and label fixed-input cache-hot
replay. Cold/hot state transitions are prescribed trajectories, not repeated
mutation assumed to perform equal work.

## Placement matrix

| Continuation/storage | Scope | Qualification boundary |
|---|---|---|
| CUDA/ROCm local resident VRAM | Original two-kernel validation/native decode floor | Separately labeled; omits pooled/service protocol |
| CUDA/ROCm whole RAM | Real provider/service/captured lookup | Same original bytes/IDs and ordinary terminal scope |
| CUDA/ROCm mmap SSD source | Real bounded-window provider/service | Separate warm-page and verified disk-read observations |
| CUDA/ROCm adaptive RAM+SSD | Real host cache policy and original source | Hot-set A→B→A; working set exceeds admitted RAM |
| CUDA/ROCm local pooled VRAM+RAM+SSD | Complete local pooled operation | Matched GPU/RAM budgets and actual promotions/victims |
| CUDA continuation,1/2/4 ROCm storage owners | Complete cross-vendor pooled operation | Native component capture/bytes/retirement gates pass; full-model binding remains unqualified |
| ROCm continuation,1/2 CUDA storage owners | Complete cross-vendor pooled operation | Native component capture/bytes/retirement gates pass; full-model binding remains unqualified |
| Same-vendor multi-card pooled storage | Actual native peer factory and complete operation | Unmeasured until the required source-owned native factory qualifies; no mapped-peer substitute |
| Mixed local+remote owner cohorts | Real declared local and remote participant paths | Add after shared multi-owner fixture proves exact extents/retirement |

The source matrix remains code-owned. Canonical performance declarations should
select one exact geometry/placement/trajectory per process; the report projects
those declarations and recorded samples instead of keeping another executable
list. Unsupported cells are explicit unqualified entries, never skipped passes.

## Dynamic placement and pollution workloads

Run stable hot-set reuse, cold-source misses, mixed hot/cold traffic, sequential
scan pollution and abrupt hot-set A→B→A changes. Use working sets below/at/above
VRAM and RAM budgets. Include empty owners, uneven owner capacities and highly
skewed requests. Preserve the original policy parameters while comparing
placements. Require observed useful promotions and bounded resident counts;
show time-to-recover hit rate, latency distribution and churn per useful row.
Immutable eviction does not imply a copied demotion: report actual copied bytes
and logical tier-policy transitions independently.

A Linux mapping/window hit is not an OS page-cache hit or an SSD read. Before and
after measurement, collect exact requested-page residency with mincore, process
major/minor faults and physical read-byte counters. Also record the provider's
window hits/misses/evictions and active mapping bytes. Keep these observations
outside timing. A cold attempt may advise only the fixture-owned file; no
host-wide cache clearing or unrelated-file eviction is allowed. Reject a claim
of disk-cold performance when residency/I/O evidence does not support it.

## SSD cache-control strategy

The current ImmutableRowTable uses a fixed LRU mapping directory and MADV_RANDOM.
That bounds mapped storage and discourages readahead, but unmapping does not
establish a hard bound on retained shared OS file-cache pages. Measure this
separately from process RSS and PMA-owned RAM. Source-window capacity alone must
not be advertised as a physical OS-cache bound.

First establish the current path under repeatable above-budget working sets.
Retain hot rows in the explicit VRAM/RAM tiers; deduplicate native rows and group
source work by file pages/windows while preserving original response order.
Measure physical-read amplification against useful native bytes, fault/IOPS rate,
and recovery after the working set changes. CPU policy and mapping overhead are
included, so an SSD optimization cannot hide a new serialized host bottleneck.

Then compare a named Linux cache-advice candidate against that baseline.
NOREUSE can reduce promotion of streaming pages on Linux6.3+; DONTNEED can request
retirement of page-aligned cold ranges after their last reader. Both are advice,
not a hard capacity guarantee. Do not evict overlapping live ranges or mistake
success from the API for observed page retirement. Source identity and shared
file ownership matter: do not indiscriminately evict data another active reader
still needs. Cache advice must not become an untested automatic fallback.

If measurements require a strict bounded SSD path, implement it as an explicit
first-class source-I/O policy with PMA-admitted aligned persistent staging,
source-owned lifetimes, bounded in-flight reads and coalesced page/sector access.
A direct-I/O policy must validate actual filesystem alignment/capabilities and
fail precisely when unavailable. It must preserve mmap as its own supported
user choice, never silently replace it. Such a path requires all-format native
byte parity, partial/EOF/unaligned-row/error/retirement tests and the same GPU
continuation/multi-owner gates before promotion.

Linux interface sources:
[posix_fadvise](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html),
[madvise](https://man7.org/linux/man-pages/man2/madvise.2.html).

## Early lookup and prefetch across every tier

[QFN_NGRAM_PREFETCH_PLAN.md](QFN_NGRAM_PREFETCH_PLAN.md) owns the implementation
sequence and pinned Strata source review. Prefetch applies to local/remote VRAM,
RAM and SSD. Exact IDs become available from token/history/count publication;
they do not depend on the trunk activation arriving at PLE. Start necessary
lookup work then and retain its original result until its consumer. Local VRAM
gather can target the existing eventual output buffer; an extra row copy/cache
is not necessary. Hosted/remote tiers retain their existing publication,
transfer, lease and retirement authorities.

Qualify current-invocation known demand first. Future prompt chunks and published
MTP candidates add bounded lookahead; unknown future sampled tokens cannot be
prefetched exactly. Speculative reads must not increment demand frequency or
displace protected hot rows before consumption. Record useful/wasted native and
physical bytes, duplicate suppression, consumer wait, retained pin/staging
capacity, cache churn and full-request performance. Compare identical inputs
and placement with and without early issue. Actual intervening useful compute
and a shorter complete critical path establish overlap; earlier issue alone
does not. Keep late/error/reset/rejection completion replays in preflight.

## Current component evidence and next priorities

The evidence root is
`parity-results/qwen-flash-next/ngram-lookup-perf-20261031/`. This is synthetic
original-format component evidence, not a full-QFN throughput certificate.

* `index-paired-r1`: the indexed mmap directory preserved native bytes and
  cache-work counts. M512 all-hit/2048 windows improved 4844 to418us;
  default1024 windows improved56963 to45999us. The tiny2-window case was slower
  in this sequential pair (50501 to56884us); retain that result and repeat
  interleaved/reversed controls before claiming an unconditional win.
* `resident-first-r2`: direct resident gather M1/M512 measured6.656/20.512us
  on CUDA and26.560/59.680us on MI50, with byte/cleanup/driver checks green.
* `pooled-first-r3`: all8 local/cross-vendor warm-hit cells passed byte,
  retirement and driver gates, but economy remains red. Local M1 is812us CUDA
  and1922us MI50; local M512 is235/539ms. Remote M1 is about15ms. All rows hit
  VRAM; SSD read latency cannot explain these cells.
* `channel-floor-r1`: actual captured bidirectional channel protocol with1440B
  each way is41.9/45.8us source-event time;737280B each way is301/299us.
  This is a protocol reference, not a physical hardware minimum or a one-way
  latency obtained by division.
* `events-local-cuda-m1-r1` and `events-local-cuda-m512-r1`: native graph-event
  diagnostics pass output, retirement and driver checks. Spans include event
  and scheduling overhead and describe only the final replay. They isolate
  serial pin validation, repeated immutable root validation and sorting old
  pins during empty allocation waves as the next economy targets. The earlier
  Nsight attachment in `nsys-local-cuda-m1-r1` produced driver warnings/assertion
  and remains failed; no further CUPTI attachment is qualified.

The first scoped coordinator change validates its immutable copied layout once
per operation object while retaining all live state checks. `layout-paired-r1`
contains12 matched original/candidate cells with exact loader identity: local
CUDA M1 improved812 to677us and M512234 to160ms. MI50 remained approximately
1.92ms/534–538ms. Remote M1 is still about14–15ms; do not claim a remote speedup.
All12 pass byte/retirement/driver checks. `layout-transition-r1` adds16 passing
CUDA/ROCm cases over alternating/churn/empty/duplicate traffic, with identical
cross-backend promotion/eviction counts. Shipped CUDA80/86/89/90 and HIPgfx906/
gfx1100 compilation and spill guard pass. Device-free coordinator tests pass20
cases, including20 repeats of three new lifetime/error regressions; they have
explicit preflight registration. Complete Unit/preflight and full-model/source
admission remain pending for the combined tree.

The bounded exact-identity successor uses pairwise equality through64 combined
identities and preserves radix above that point. `small-identity-paired-r1`
contains12 matched native cases, all byte/retirement/driver green: local M1
improves677 to520us on CUDA and1931 to1415us on MI50 versus the layout candidate.
M512 remains approximately160ms/536ms (within1% of its matched predecessor).
Remote M1 differences remain too small relative to prior process variation for
a strong speed claim. CPU production-operation replay passes107 tests plus20
repetitions of six focused regressions. All shipped GPU targets compile with
the spill guard. `events-local-cuda-m1-small-identity-r1` provides clean native
event attribution, not hardware counters or an uninstrumented benchmark.

The source-consistent Integration refresh completed in226seconds. Its new
multi-owner and threshold tests exposed fixture assumptions about TinyLFU:
source misses need not acquire GPU grants, and repeated requests need not evict
hotter rows. Exact boundary replays now use enough explicitly admitted free
slots for six disjoint cohorts; they still require actual32 initial pins plus
31/32/33 grants and exact original release receipts. Multi-owner warm queries
use a closed set including the high-bit row. These isolate required identity,
transfer and retirement proofs from unrelated victim-policy choices. Original
red results are retained in `integration-slice-r2` and
`integration-trajectories-r3`. The corrected `integration-native-r4` run passes
all21 cases: twelve CUDA/ROCm exact-identity boundary replays and nine multi-owner
replays over CUDA consumers with2/4 MI50 owners and a ROCm consumer with2 CUDA
owners. Original output bytes, inactive tails, live communication extents,
retirement and driver observations pass. This certifies those actual component
paths; production full-model continuation binding, native same-vendor factories,
victim/rejection and all-format gates remain separate obligations.

Continuation qualification also requires the existing Qwen MoE dynamic expert
balancing behavior. Keep ngram storage owners distinct from routed-expert owners
and reuse the existing runtime table, controller, publication and movement
journal. Source admission proves maintenance-family allocation and authored
policy; a separate inference/replay gate must witness real expert movement with
correct compact payloads, ordering, reset and numerical results in supported
same-vendor and mixed-vendor domains. Static admission alone cannot qualify this
requirement. Pair lookup timings with stable and changing expert placement so
overlap measurements include actual contention and do not assume fixed routing.

Continue reducing measured pooled software work while preserving all original
phase/failure semantics and communication extents.
Then measure early issue against the improved baseline: overlap must not
disguise a needlessly expensive operation. The all-tier capture design records
why RAM/mmap overlap is possible with existing infrastructure and why concurrent
VRAM gather needs additional executor-owned stream support.

## Detecting uneconomical paths

Compare latency and bytes across M,unique IDs,owners and cache budgets. Preserve
slow cells. Investigate discontinuities, superlinear work, oversized payloads,
serial owner service, repeated page faults, excessive policy work and duplicated
rows. Profile one exact slow production operation after unprofiled measurements;
keep profiler overhead out of the baseline. Structural volume/capacity invariants
are hard regression assertions. Performance ratchets follow measured noise bands
on named hardware; no arbitrary cross-machine wall-time threshold is installed.

Every discovered production defect gets a focused canonical regression and an
explicit ProductionTestPreflight entry. Unit remains device-free. Benchmark
cases remain explicit Release performance work and do not turn Unit/preflight
into a timing tournament. Keep native driver monitoring and process/file
retirement receipts; an otherwise correct sample with a driver fault is red.
