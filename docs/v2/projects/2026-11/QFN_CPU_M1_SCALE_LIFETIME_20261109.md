# CPU M1 scale lifetime and final-add fusion qualification

The retained production change moves the existing single-row activation-half
conversion before integer dots and weight-scale vectors become live. It changes
no conversion, FP operation, reduction order, storage, format, dispatch or team.
The separate final-add/preweight fusion was tested and withdrawn: its additional
implementation did not establish a throughput benefit.

## Actual compiled owner and numerical gates

The coherent Release owner is `/tmp/qfn-native-quantised-full-release-r1063/build`.
The actual `v2_perf_cpu_moe_expert_verifier_rows` object proves, for the selected
Q4_K/asymmetric M1 first and steady loops in both owned and workspace banks:

| Per-K observation | Prior checkpoint | Earlier activation conversion |
|---|---:|---:|
| Vector stack references | 16 | 0 |
| Outlined calls | 0 | 0 |
| VNNI updates | 32 | 32 |
| INT-to-FP conversions | 4 | 4 |
| FP multiplies / adds | 16 / 8 | 16 / 8 |

The selected worker shrinks 35,604 to 34,575 bytes and its frame shrinks by
256 bytes. Ordinary identity dispatch remains unchanged. This is a selected
loop result, not an all-format or whole-worker zero-stack claim. The fixed
partition tree and separately rounded FP terms remain unchanged.

The existing 14 all-format NativeQuantised aliases pass on AVX2 and AVX512
runtime routes. Six shard-reduction aliases also pass. The existing final-add
contract test now covers every width 1 through 65, plus 127/128/129/2560,
degrees 2 through 8, one/three workers, dirty replay, and distinct quiet/signaling
NaNs, signed zero, infinities, subnormals and cancellation. All 20 aliases are
already explicitly registered in production preflight. The final coherent
closure passes 20/20 in 22.70 seconds, with zero skips. This does not certify a
separate AVX2 compiler build or the deferred complete preflight.

Evidence: `/tmp/qfn-cpu-final-add-weight-next/qualification/scale-only.{json,xml,log}`,
`build-scale-only.{json,log}`, and `../isa-review/NOTE.md` with frozen actual
objects and original/candidate instruction excerpts.

## Economy and matched model

The existing authored 28-worker decode/MTP owner remains approximately flat in
complete FFN latency: all four measured formats are within about one percent
of the previous conversion-order candidate. Individual phase timings vary more;
there is no universal format-specific speedup claim. Raw samples and exact
workload/dispatch evidence are retained at
`/tmp/qfn-cpu-unsigned-nibble-one-row-next/m1-mtp-economy/activation-scale-early1`.

Model observations use the original Release driver: QFN UD-Q4_K_XL with the same
Q8 MTP sidecar, two MI50s plus two real CPU socket ranks, 28 physical workers per
socket, 262144 context, dynamic MTP and balancing, RAM ngrams/local GPU caches,
16/32 GiB prefix tiers, 512 input/256 generated tokens, one warmup and three
measurements. Passive observation filters and sampling seed42 are unchanged.

| Retained lifetime | Prefill tok/s | Benchmark decode tok/s | After-prefill decode tok/s |
|---|---:|---:|---:|
| Earlier fresh control | 296.123 | 40.189 | 40.032 |
| Prior pushed checkpoint | 303.365 | 39.901 | 39.745 |
| Fusion plus scale lifetime | 296.851 | 39.183 | 39.030 |
| Scale lifetime only | 298.175 | 39.891 | 39.735 |

The small scale-lifetime change removes proved temporary spill traffic, but
these model observations do **not** establish a throughput gain. The final run
matches all 768 measured output IDs, MTP work, effective configuration and
147/49 initial root/follower placement records against the original control.
Native/driver exits and resource retirement pass. Run-owned prefix archives
were metadata-audited and reclaimed without payload reads.

Final evidence: `/tmp/qfn-pair-ngram-model-r1151/scale-early-only-passive`,
`/tmp/qfn-cpu-final-add-weight-next/qualification/scale-only-model-comparison.json`.
The 500 prefill /40 decode goal remains open.

## Withdrawn final-add fusion

The first SIMD/scalar candidate failed distinct-NaN byte comparisons at width31
on scalar and AVX2 dispatch. Actual assembly showed compiler-reversed operand
precedence in the vectorized scalar remainder. Reusing the original public add
for scalar work and bounded tails corrected this; complete SIMD groups retained
the rounded sum for raw and weighted stores without reload, FMA or hot spills.
Expanded original regressions passed without changing their reference.

However, the cached-PMA publication timing moved with its unchanged two-pass
control rather than showing a clear independent gain. The full-model result
also did not justify retaining the extra implementation. The three fusion
production files were restored to the preceding checkpoint. The wider numerical
regression remains. The qualified-but-unpromoted patch, initial failure, corrected
ISA and all measurements are retained under `/tmp/qfn-cpu-final-add-weight-next`.
No mapped-ticket or transport benefit is inferred from the cached-PMA fixture.

## NUMA ownership discovered during the next experiment

These retained model lifetimes assigned the socket-1 GPUs to the socket-0 MPI
rank. Existing GPU context/TP workers inherit the rank's OpenMP partition; they
do not independently move onto the GPU's NUMA node. User direction now requires
rank-local GPU ownership: the GPU-local rank performs GPU work and its CPU expert
shard; the other rank participates only in CPU expert work. Planning/admission
and corrected-topology qualification are separate pending work. The measurements
above do not certify that new ownership invariant.

The public saved-rank-order diagnostic currently rejects CPU ownership even
with explicit identity order. Its verbose native report confirms MPI binds the
two discovery ranks to their respective sockets, so a different declaration
must not be invented to bypass the error. The active procfs NUMA detector has a
confirmed substring/range parsing defect and arbitrary eight-node search bound;
its relationship to the failed late inventory observation still needs native
confirmation. Evidence is `/tmp/qfn-socket-order-next/{current,v2/current}/results`.
