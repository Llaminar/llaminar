# QFN local-cache coordination and fusion investigation

This experiment record separates measured current-demand behavior from focused
kernel candidates and unfinished model read-ahead. The main implementation and
qualification plan is [the local-cache hierarchy plan](QFN_LOCAL_HOT_CACHE_HIERARCHY_PLAN.md).

## Matched short natural-history result

The Release comparison uses the same Qwen3.8-Flash-Next UD-Q4_K_XL GGUF,
ROCm TP2 continuation, CPU capacity tier, eight GPU-resident experts per layer,
RAM ngram backing, disabled prefix cache, MTP off with retained graph capacity
three, and seed 17. Four natural histories are each repeated once. Every policy
generates the same eight 64-token sequences. Native retirement and driver checks
pass. Rates below are total tokens divided by total phase time, rather than an
average of individual rates.

| Policy | Prefill tokens/s | Decode after prefill tokens/s |
|---|---:|---:|
| Shared pool | 42.7608 | 9.0760 |
| ConsumerLocal | 49.1197 | 11.0610 |
| Local cache plus shared pool | 42.0256 | 8.6267 |

The combined policy loses 4.95% decode throughput against shared-only and 22.01%
against ConsumerLocal. These are short, CPU-capacity-heavy workloads, not a
general ranking of the policies. No default is promoted. The comparison covers
212 prefill tokens and 504 timed decode-after-prefill tokens per policy; it is
neither cold-SSD evidence nor a cache-pressure workload.

Evidence is retained in
`/tmp/qfn-declared-benchmark-command-prep-r785/comparison.{json,md}` and
`final-varied4-analysis/{REPORT.md,measurements.json}` beneath that directory.
The original shared run's passive validation failure remains preserved; its
separate postvalidation receipt uses the repaired actual expert-bank contract.

## Measured traffic and next attribution

Shared-only and combined each read 4,768 source rows (429,120 native bytes).
ConsumerLocal reads that amount independently on each consumer. The combined
decode unions coalesce 9,248 miss IDs into 4,624 unique misses. There are no shared
cache hits or evictions in this short cohort.

All four repeated combined histories have entirely local demand hits on both
consumers. They send zero native key/result payload, but retain 64 source
invocations and the required request/result/ACK progress per 64-token request.
The respective metadata totals are 5,632, 6,144 and 6,144 bytes. This makes empty
auxiliary progress and parent retirement useful attribution targets. It does not
establish their critical-path duration. Setup-stage counts are declarations,
not executed launch counts or lookup-stall measurements.

## Separate candidates

1. **Bound the shared coordinator by the captured query.** The local probe
   already consumes its actual query's immutable row bound. The compact shared
   union and coordinator instead used the persistent prefill capacity. A private
   candidate binds each consumer's miss ceiling to its original canonical query
   and declared root, then checks their sum. For an M1 query with two consumers,
   the bound is 32 original IDs and selects the existing single-block coalescer.
   It changes no stored capacity, wire layout, memory budget or live count
   authority. Host and full-architecture compilation checks pass. The integrated
   bound-only candidate passes all 11 focused host/native checks on CUDA and
   MI50, including a real source replay with 1,024-ID banks and a 32-ID captured
   query, then empty and shorter reuse. The combined Release measurement below
   includes this repair; no isolated speedup is attributed to it. Evidence:
   `/tmp/qfn-compact-query-launch-bound-r790/freeze/`.

2. **Fuse the local small-query probe.** For at most 512 original IDs, combine
   opening, stable coalescing, independent probes and ordered pin/miss
   publication into one 256-thread block, replacing four launches. Reuse the
   same arithmetic and policy bodies, with block barriers replacing internal
   kernel boundaries. Larger queries retain their existing parallel bundle.
   This candidate now passes focused native, numerical and Release qualification.
   It compiles
   for CUDA sm80/sm86/sm89/sm90 and ROCm gfx906/gfx1100 without memory spills.
   CUDA uses 55–56 registers; HIP uses 27 VGPRs, 18,056 bytes of LDS and no private
   scratch. HIP scalar-to-vector register moves are not memory spills. The
   initial 26 native cases pass, followed by all eight expanded boundary cases:
   both key widths, zero/four local slots, every token count from zero through
   32, dirty shorter reuse, adoption and reset on CUDA/ROCm. Splitting the
   expanded cells into separate processes corrected a fixture ownership error;
   the original memory-authority rejection remains intact. Matched component
   timing and model qualification are separate steps.
   Evidence: `/tmp/qfn-local-probe-fusion-r792/`.

3. **Fuse a small shared union separately.** Preparing union offsets, copying
   disjoint live misses and sealing their publication can use one block when
   the checked query bound permits it. The candidate must preserve canonical
   membership checks and failure publication. Request-header acceptance and key
   receipt cannot be fused across the actual transfer dependency. This candidate
   is now integrated separately after the bound-only native pass. All ten
   selected host/CUDA/MI50 cases pass; a further 14-case native gfx1100 sweep
   passes local all-format, key-width, boundary, next-window, source/adoption
   and union regressions. All driver windows are clean with no skips. It is not
   part of the bound-only patch. The combined Release measurement below includes
   it, without attributing a separate speedup to this fusion.
   Evidence:
   `/tmp/qfn-small-compact-union-fusion-r791/freeze/`.

The two kernel candidates and the bound repair compile under the canonical
spill guards for CUDA sm80/sm86/sm89/sm90 and HIP gfx906/gfx1100. Native tests use
the RTX 3090, MI50 and RX 7900 XTX; compilation does not qualify runtime behavior
on the other CUDA architectures.

The native local-cache replay now includes the 512-ID dispatch boundary,
496/512-to-small/empty reuse, demand adoption, reset and poisoned inactive output.
Existing tests cover every supported source format, both key widths, zero local
capacity, cache pressure and the large-query path. They must pass on the actual
candidate before performance claims. Native-event measurements bracket the
complete retained local graph with recorded source responses, excluding input
uploads, reset/admission and result oracles. They include graph launch/scheduling
latency and possible host submission gaps; they are neither isolated kernel
timings, source-transport measurements nor model throughput. The timing roster
also includes the actual QFN IQ4_NL160 row width and 320,001,536-row table extent.

The matched second pair (`paired-timing-r2.json`) gives these median complete
current-lookup graph latencies in microseconds. Each warm/empty group has 38
samples after excluding the first full request; each mixed group has 19.

| QFN IQ4_NL160 workload | CUDA original → fused | ROCm original → fused |
|---|---:|---:|
| Empty | 50.368 → 46.496 | 85.040 → 76.320 |
| Warm, 16 original IDs | 102.912 → 99.056 | 250.080 → 246.320 |
| Mixed, 16 IDs / 12 misses | 139.200 → 131.808 | 319.200 → 315.840 |
| 512 IDs / 60 misses | 1,435.140 → 1,426.210 | 3,860.640 → 3,856.960 |

CUDA shows a repeatable modest improvement. ROCm warm and large-query changes
are close to run variation, so this does not establish a material ROCm speedup.
All profile/shape groups are retained, including the 496-ID ROCm case that is
0.87% slower in this pair. `component-qualification.json` records 32 distinct passing native
cases without promoting that result to full preflight or model qualification.

One baseline NCU attachment triggered new NVIDIA driver assertions and late
profiler-module admission failures. Its failed receipt is retained under
`/tmp/qfn-local-probe-fusion-r792/cuda-baseline-ncu/`. Further CUDA profiler
attachment is stopped in this environment. Subsequent unprofiled native tests
have clean driver windows. Static compiler/ISA resources and native event
measurements remain available; achieved occupancy and hardware counters are
unavailable, not inferred from those alternatives.

## Next adjacent candidate

The read bundle is the strongest remaining local candidate: validate all native
rows, decode, and publish read completion currently require three launches. A
bounded small-output CTA could retain those barriers and the exact decoder,
while recovering each checked row pointer once instead of repeating its
metadata checks for every output element. Larger outputs must keep parallel
work. This candidate remains private and unqualified. It must not cross Ready1
or publish Ready2 before every model-output write completes. Cache maintenance
remains a separate auxiliary consumer after Ready2; it cannot be fused across
that ownership edge merely to reduce launch count.

## Model qualification of the probe/bound/union slice

Fresh CUDA TP2 and MI50 TP2 numerical panels each pass all required groups and
retain exactly the previous 178 per-row metric records, including mean KL,
top-1 and top-5. The six focused prerequisites pass; complete Unit/preflight
was intentionally not rerun. The initial prerequisite attempt overlapped the
Release build and exceeded two Python-test timeouts; both pass with the same
timeouts after the build retires. No correctness limit was changed.

The matching Release campaign completes eight requests with identical prompt
and generated token IDs, clean driver state and complete native retirement.

| Combined hierarchy | Prefill tokens/s | Decode after prefill tokens/s |
|---|---:|---:|
| Before probe/bound/union | 42.0256 | 8.6267 |
| After probe/bound/union | 42.5867 | 8.7000 |
| Observed change | +1.34% | +0.85% |

This historical paired run has no interleaved timing repeats, so those small
changes do not establish a firm throughput win. ConsumerLocal remains faster
on the earlier matched short cohort. Future scheduling is still disabled and
no policy default changes. Evidence is under
`/tmp/qfn-fused-release-comparison-r793/`: both `numerical-*-combined-fusions-r1`
reports, `combined-varied4-execute-r2/report.json`, and
`combined-before-after-r1.json`.

The first Release invocation omitted `ROCR_VISIBLE_DEVICES`, exposing the
7900 XTX as ordinal zero. It completed on a different pair, and the original
resolved-placement validator correctly rejected it. That run remains excluded.
The private launcher now checks exact visibility selectors against the accepted
panel before execution, with absent/empty/reordered selector negative tests.
The corrected launch uses the original four-MI50 UUID selection; CPU placement
was unchanged. No engine NUMA or default-policy change was necessary.

## Read-ahead integration boundary

The integrated source has lookup-only future purpose, complete-demand adoption,
borrowed next-prefill windows, and finite-child dependency joins. Its first
coherent build found a CUDA trap-intrinsic error and an omitted test include;
both are corrected. The 41 selected component cases now pass, combining original
passing cases with focused reruns of repaired fixture boundaries. Their receipt
is `/tmp/qfn-future-component-integration-r789/component-qualification.json`;
this is not a full preflight or model-read-ahead certificate.

Model future scheduling remains disabled. Ordered auxiliary bodies and a single
source plan must cover the actual one- or two-occurrence source transaction.
The two-occurrence owner must also drain a first-occurrence failure without
waiting for a phantom successful second request. Future Hash/Probe joins both
accepted PLE history and the preceding local read's retirement. Component
correctness, real model overlap, numerical parity and throughput remain separate
qualification steps.
