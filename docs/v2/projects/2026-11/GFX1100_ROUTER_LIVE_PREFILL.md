# Router WMMA retained live-prefix qualification

Sixteen cooperating K waves improve the native router's full-bucket performance
and remove the measured small-prefix loss for K2560/E512. All measured live
counts at that geometry beat Dot4 in both input bases. The other widths retain
one-row regressions, so Automatic still uses Dot4 pending narrow, ownership-aware
installation and actual-Auto qualification.

## Complete-operation timing

The Release fixture retains explicit Dot4 and native WMMA graphs at capacity512.
Each graph contains 16 complete operations, including input quantization, logits
and restoration of the original expert-input Q8 bank for SignedHadamard32.
Parent-owned ordered-timeline events bracket that chain inside the retained
graph. An external event only joins completion; host submission gaps outside
the graph cannot supply the timing interval. Native scheduling inside the graph
remains included, and no constant overhead is subtracted.

All three wave caps, four model width/expert geometries, both input bases and
live counts 0,1,16,256,511,512 are measured: 24 cases,144 cells and 31 alternating
paired samples per cell. Each exact case has an independent producer/overlap
receipt. Clean cases are retained when unrelated builds interrupt another case;
all final cases have unchanged producers and no compiler overlap.

An immutable full-live Dot4 snapshot owns expected logits, Q8 codes and scales.
Both banks receive fresh poison before every timed pair. Every active byte must
match the snapshot, and every inactive/guard byte must remain poisoned. Setup,
transfers and observations stay outside the measured interval. The terminal
guard drains before host staging, source fragments, parents or tensors retire.

Ratios below are Dot4 median time divided by native median time; larger is better.

| K / experts | Basis | Four waves, live512 | Eight waves, live512 | Sixteen waves, live512 | Sixteen waves, live1 |
|---|---|---:|---:|---:|---:|
| 2048 / 256 | Original | 1.424 | 1.502 | 1.522 | 0.956 |
| 2048 / 256 | Hadamard32 | 1.378 | 1.413 | 1.418 | 0.956 |
| 2560 / 512 | Original | 2.549 | 2.646 | 2.742 | 1.010 |
| 2560 / 512 | Hadamard32 | 2.326 | 2.412 | 2.474 | 1.012 |
| 3072 / 256 | Original | 1.854 | 2.075 | 2.210 | 0.938 |
| 3072 / 256 | Hadamard32 | 1.718 | 1.891 | 1.951 | 0.953 |
| 8192 / 512 | Original | 0.914 | 1.213 | 1.649 | 0.876 |
| 8192 / 512 | Hadamard32 | 0.921 | 1.192 | 1.559 | 0.902 |

At K2560/E512, the sixteen-wave ratios across the retained live prefix are:

| Live rows | Original | Hadamard32 |
|---:|---:|---:|
| 0 | 1.073 | 1.062 |
| 1 | 1.010 | 1.012 |
| 16 | 1.324 | 1.282 |
| 256 | 2.759 | 2.526 |
| 511 | 2.733 | 2.474 |
| 512 | 2.742 | 2.474 |

All 12 K2560/E512 cells have paired-median bootstrap 95% intervals above 1. At live1,
the intervals are 1.0075–1.0137 (Original) and1.0089–1.0140 (Hadamard32), from 4096
paired resamples with seed 20261114. These observations justify investigating an
exact normal-M512 admission key; they do not qualify compact owned rows or
unmeasured shapes. The other widths' live1 regressions remain material.

Earlier `router-live-prefix-*` and `router-wave-external-*` clocks bracketed host
graph submission. They remain archived component observations, but the current
in-graph records supersede them for this admission decision. The initial combined
wave tournament also overlapped an unrelated compiler and was rejected entirely
as timing evidence; its successful byte checks are not a performance certificate.

## Exact arithmetic, replay and resources

Each wave owns an adjacent complete subtree of the original 128-owner reduction.
Four/eight/sixteen-wave variants retain the original per-leaf FMA chain and use a
balanced adjacent-subtree fold through LDS. Tiny K uses only nonempty subtrees.
No extra projection launch, global partial bank, atomic, host live-count readback
or replay-time allocation is introduced. The recording scope changes a retained
graph's concrete kernel/geometry; it does not alter an already captured graph.

The existing native replay passes in 29.33s. Two new normal/compact-owned wave
replays pass in 39.30s, covering both bases, tiny/ragged K, K>4096, row boundaries,
partial/empty/reset work, source preservation and original Q8 publication.
They authenticate the requested template cap, actual launch threads and LDS.
The new timing replay alternates separate owners for 20 dirty replays per cap,
requires one fresh named sample, and rejects stale or disabled timing. It passes
in 3.064s in Release and 4.95s through its Integration registration.
`V2_Integration_ROCmRouterQ8WmmaReplay`,
`V2_Integration_ROCmRouterQ8WaveTilesReplay` and
`V2_Integration_ROCmRouterQ8TimingReplay` are explicit production-preflight entries.
The full preflight suite and model E2E were not run for this slice.

Release and Integration builds cover both shipping targets, gfx906 and gfx1100,
with the mandatory spill guard. Six isolated native profiles bind exactly the
same executable, core and source files as the final in-graph timing receipts.
Each counter pass contains one matching dispatch on actual gfx1100 PCI
0000:b5:00.0. Profiling parents omit timing instrumentation. All profiles have
positive GPU/wave activity, two INT8 WMMA instruction sites, wave64, zero scratch,
zero register spills and zero measured LDS bank conflicts. AUTO power is restored
and authenticated after every profile; profiler durations are not latency labels.

| K / experts | Waves | Logical / allocated VGPRs | Logical SGPRs | Static LDS | Threads | L2 hit | Memory-unit busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048 / 256 | 4 | 67 / 68 | 19 | 4KiB | 256 | 95.94% | 57.74% |
| 2048 / 256 | 8 | 63 / 64 | 19 | 8KiB | 512 | 97.11% | 58.65% |
| 2048 / 256 | 16 | 74 / 76 | 20 | 16KiB | 1024 | 96.40% | 61.16% |
| 8192 / 512 | 8 | 83 / 84 | 19 | 8KiB | 512 | 99.16% | 87.35% |
| 8192 / 512 | 16 | 79 / 80 | 19 | 16KiB | 1024 | 99.10% | 84.47% |
| 2560 / 512 | 16 | 74 / 76 | 19 | 16KiB | 1024 | 95.98% | 70.06% |

Runtime resource queries report potential block residency, not achieved
occupancy. gfx1100's unavailable VALU-utilization counter is not inferred from
GPU busy or memory-unit busy. Full loaded code objects, descriptor metadata and
selected-symbol ISA remain beside each profile summary.

Local evidence is under `/tmp/llaminar-gfx1100-wmma-prefill-evidence/`:
`router-wave-receipt.json`, `router-wave-summary.json`, `router-wave-cases/`,
`router-wave-producer-receipt.json`, `router-wave*-profile/summary.json`,
`router-wave*-power-receipt.json`, `router-wave-registered-replay.log` and
`router-wave-internal-registered-timing-replay.log`. The original and wave-cap
performance cases are individually filterable in `v2_perf_moe_verifier_prefill`.

## Production bucket domain

The default ladder has 448 and512, so ordinary non-owned M512 work has 449–512
live rows. `PrefillBucketUtils` selects the smallest fitting remainder; prefix
restoration and chunk tails are reselected. TP forwards the policy, and PP
preserves the selected physical width and live extent.

This is not a universal kernel invariant. Both
`LLAMINAR_PREFILL_GRAPH_MIN_SEQ=512` and
`LLAMINAR_PREFILL_GRAPH_BUCKET_SIZES=512` are supported. With resident cap 512, a
prompt of 512+r tokens executes two M512 graphs, the second with live=r. The
ROCm row materializer computes `min(bucket,total-source_start)`. The serial
decode exception for a complete one-token suffix does not apply to a one-token
tail inside a longer prefill.

Code owners are `utils/PrefillBucketUtils.cpp`, `utils/PrefillGraphBuckets.def`,
`execution/runner/OrchestrationRunner.cpp`, `utils/DebugEnv.h` and
`kernels/rocm/ops/ROCmRowSelectKernels.hip`. Capture identity intentionally omits
live scalar values and does not carry a minimum-live/predecessor-bucket proof.
Capacity-only router admission cannot infer that default-ladder lower bound.
