# QFN HC reduction and activation publication

Private experiment182 passes its focused Release byte and timing checks.
It is archived while fresh checkpoint180 stage attribution determines the next
production target. No production selector, default or model score changed.

The candidate retains the original Native19 HC-down producer, its sixteen
K partitions and ascending FP32 reduction. One kernel combines that reduction
with the existing division by four, SiLU and TwoTermQ8 publication. The
compiled short-K up projection and residual-read kernels remain unchanged.
The complete captured chain has six kernels instead of seven; its input
normalization, activation quantization, matrix products and final read are
included in both timing arms.

| Complete M1 chain, synthetic H2560/R320 | Median µs | Range µs |
|---|---:|---:|
| Current production control | 42.9599 | 42.9149–43.3349 |
| Fused reduction/publication | 41.5649 | 41.4999–41.6599 |

The time reduction is **3.247%**, across 31 alternating pairs after five warmup
pairs, with 32 complete operations per captured replay. These are component
times, not measured model throughput. The source-format weights are synthetic
Q8_0; no real-model accuracy or speed claim follows from this screen.

The independent functional run passes twenty dirty replays for each of two
layouts: separate raw/activated destinations and the production exact alias.
It checks the original partial bank, raw values where observable, activated
values, both Q8 terms and scales, up/read outputs and inactive guards. Empty
up work still preserves the original ordinary-down publication. Timing repeats
the dirty checks, verifies full work before events, and verifies full/empty/full
reuse afterward. Invalid ticket, stream, source, prefix and storage-alias
bindings reject before submission.

Offline bounded code inspection proves the original producer, reducer,
HC publisher and short-K up bodies/resources exactly match frozen180.
Both new fold specializations use 14 VGPRs (16 allocated), zero static LDS,
zero private storage and zero spills; SGPR usage is 34 for ordinary work and 36
for counted work. Only ordinary work was executed by this bounded screen.
Isolated hardware-counter collection is prepared but has not been run.

Production integration remains open: the reusable HC builder must retain
both existing prepared sources and workspace leases in one typed composition.
Observed raw-down snapshots need a cold, admitted diagnostic destination when
ordinary raw/activated storage aliases; the same optimized kernel must remain
under test. This prototype does not implement that facility or broader
format, grouped-row, backend or real-model qualification.

Evidence is retained under `parity-results/qwen-flash-next/` as
`fold182-release-build-r1`, `fold182-functional-r1` and `fold182-events-r1`.
Private source and frozen Release assets are at
`/tmp/qfn-hc-fold182-component-green-r1`; bounded ISA evidence is at
`/tmp/qfn-fold182-isa-private`. The candidate source was removed from the
working tree before profiling the qualified180 production binary.
