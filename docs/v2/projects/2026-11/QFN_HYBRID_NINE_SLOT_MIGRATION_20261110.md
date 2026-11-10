# QFN hybrid migration-wave capacity and throughput

The existing nine-slot/nine-cycle configuration measures **335.999 prefill /
40.798 after-prefill decode tok/s** on two MI50s plus two CPU socket ranks.
It improves both rates in the matched same-capacity comparison below. It is
the working configuration for the next hybrid tuning measurements; global
migration defaults remain unchanged. The 500/40 target is still open.

## Fixed workload and existing policy

Source checkpoint 733270d57; Release SDK 10.1 build with GNU OpenMP,
28 physical workers per CPU socket, and the application's original automatic
MPI bootstrap. GPU work remains on its local NUMA rank. Model is
Qwen3.8-Flash-Next UD-Q4_K_XL with the shared Q8_0 MTP sidecar. Context is
262144; each run has one warmup and three measured 512-input/256-output
requests, seed42. Decode uses the same 255 after-prefill steps. Dynamic MTP,
expert balancing, RAM ngrams, local GPU row caches and 16/32 GiB RAM/disk
prefix budgets stay enabled.

Only the existing typed migration settings change:

```text
--moe-migration-transfer-slots 9
--moe-migration-execution-streams 4
--moe-migration-cycles-per-wave 9
```

The measured runs use those fields in the existing lossless configuration
document, not an additional controller or alternate launcher. One cycle
remains available for participant placement; the global tier-exchange limit
increases from four to eight. Every selected transaction still passes the
original shadow, coherent objective, publication, acknowledgement and reader
retirement rules. Projected isolated candidate gains are never summed.

## Capacity is admitted, not assumed

Both nine-slot arms start with empty automatic quota vectors and placements.
Fresh PMA admission accepts the full262144 context. Actual admission and
model configuration agree on49 planned layers: GPU capacity164 in the first
layer and162 in each remaining layer, totaling7940 whole experts,3970/card.
The CPU tensor cohort covers17148 logical experts. The five-slot configuration
holds8066 GPU experts,4033/card; the additional storage costs63 experts/card
(1.56%). This is not a same-capacity comparison against five-slot storage.

Source-derived transfer staging grows192 MiB/card,576 MiB on execution-rank0
host and192 MiB on rank1 host. Original grouped shadow arenas and control
storage also grow. Those staging increments are not the complete memory BOM;
PMA retains sole authority over the full admitted allocation. No independent
reserve, overcommit or inferred free-VRAM allowance is introduced.

The complete existing capacity unit (90 tests, including physical-piece
cases) and the dependent-cycle integration owner (six tests) pass in1.873s.
The separately named physical-piece alias is absent from this selected
Release registration; those seven cases execute inside the full unit owner.
No complete production-preflight pass is claimed.

## Matched measurements

| Physical slots / active cycle cap | GPU experts | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
|5 /5, first current control|8066|328.794|39.609|
|5 /5, repeated current control|8066|331.278|39.794|
|9 /5, same-storage control|7940|330.527|39.668|
|9 /9|7940|335.999|40.798|

The9/5 and9/9 runs resolve identical initial expert plans. Apart from output
and configuration paths, only the active cycle cap differs. All768 output
IDs and complete MTP records match in every comparison. Wider waves improve
prefill1.656% and decode2.849% against their same-storage control; versus the
five-slot repeat the differences are1.425% and2.525%. These are individual
fresh-lifetime comparisons, not a large-sample noise or universal-topology
certificate.

The9/9 cohort completes12 transactions and96 promotions, moving941,056,000
physical bytes. The9/5 cohort completes15 transactions and60 promotions,
with618,905,600 bytes. More movement alone is not the success criterion.
Against the preceding five-slot repeat, the observed root CPU prefill route
counter falls449911→424309 and grouped-verifier routes224187→203196. The
component call counts also differ; these are actual counter totals, not
per-call averages or an additive latency model. Follower lifetime counters
include preparation/warmup and are neither added nor normalized as another
three-request cohort.

## Resource lifetime and cleanup

Peak individual-process RSS in9/9 is36.472/77.798 GiB;9/5 is36.473/77.724 GiB
(rounding upward). Selected per-client VRAM remains below30.311/30.079 GiB
in both. All resource observers report zero errors. These short runs do not
certify absence of long-duration leaks. Native, driver and original process
retirement pass in251.477s and252.893s respectively; admissions take9.298s
and8.389s. After retirement, owned prefix files reclaim727,912,498 and
727,912,490 bytes using metadata only, with zero payload reads or hashes.

Local evidence is under `/tmp/qfn-prefill500-next-source-review/`: exact
prepared configs/commands in `wave-inputs/`, original native owners,
`model-comparison.json`, `same-capacity-comparison.json`, route counters,
resource and cleanup receipts. Result payloads and host-specific config copies
remain outside the repository.

## CPU candidate screened out before promotion

A separate AVX512 paired-dot candidate split low/high16 integer accumulation
chains and merged exact INT32 values before unchanged FP arithmetic. Its
unchanged baseline complete-FFN owner passes, measuring6.663ms persistent FFN
on the authored Q4_K/Q5_1 E84/R3208 workload. No candidate timing or numerical
pass is claimed.

Actual compiled inspection finds a64-byte integer-partial spill/reload in one
emitted identity/symmetric/owned-storage body, once per K32 block. The current
canonical Q4_1/Q4_K asymmetric paths do not spill; neither do the other
canonical nibble bodies. The candidate nevertheless fails its deliberately
broader emitted-body spill criterion and is parked without a rescue sweep.
Original source and coherent binaries are restored. This is not evidence that
the current QFN hot path spills or that the candidate was measured slower.

The frozen patch, integer proof, original timing and actual baseline/candidate
assembly are under `/tmp/qfn-cpu-pair-dot-ilp-next/`, with the independent
classification in `isa-review/CANDIDATE.md`. Next investigation is native
compact route transport: establish actual reduction arithmetic first, then
resolve capture-safe live work before changing production communication.
