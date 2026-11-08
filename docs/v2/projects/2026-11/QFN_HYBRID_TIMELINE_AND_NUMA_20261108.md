# QFN hybrid timeline, NUMA ownership and CPU gather measurements

This investigation follows `QFN_HYBRID_ROOT_TIMING_20261108.md`. It records
diagnostics for the 2×MI50 continuation plus two physical CPU socket expert
configuration. The target remains 500 prefill and 40 decode tok/s; none of the
measurements below achieves it. No transport or NUMA ownership default changes
are justified by this slice.

## Matched model workload

The Release AVX512 application uses Qwen3.8-Flash-Next UD-Q4_K_XL and its shared
Q8_0 MTP GGUF, context 262144, M512 activation capacity, dynamic MTP with maximum
depth 15, and 28 physical workers on each CPU socket. ROCm devices 1 and 2 are
MI50s. The GPU expert tier uses whole-expert ownership; each CPU expert is
tensor-sharded across the two original socket ranks. Dynamic expert balancing
remains enabled. Ngrams use adaptive RAM backing and ConsumerLocal GPU caches.
Prefix-cache budgets are 16 GiB RAM and 32 GiB disk.

The unprofiled workload is 512 prompt tokens, 256 generated tokens, seed 42,
one warmup and three measured requests. Actual prefill graphs have 448 and 64
rows. Decode-after-prefill throughput uses 255 incremental tokens per request.
Default sampling and arithmetic remain unchanged.

| Original ownership | Prefill tok/s | Decode-after-prefill tok/s |
|---|---:|---:|
| GPU owner 0; CPU cohort `cpu:0,cpu:1`, ranks `0,1` | 184.3104 | 30.8388 |
| GPU owner 1; CPU cohort `cpu:1,cpu:0`, ranks `1,0` | 181.6488 | 30.2673 |

Both GPUs are attached to NUMA node 1, but moving the continuation owner and
CPU joint root there did not produce a measured win. The second configuration
is valid: all three 256-token outputs match, MTP work matches, original native
processes retire, and the driver observer passes. CPU tensor halves follow
the declared cohort order; arithmetic reduction still follows the canonical
shard intervals. This short comparison does not establish a statistical
regression or justify changing owner 0.

CPU root gather time was about 490.65 versus 493.86 ms per prefill. That timer
includes producer readiness and native progress; dividing payload bytes by it
does not measure link bandwidth. Root-local preweight/publication rose from
36.50 to 58.28 ms. Ordinary CPU roots have certified NUMA placement; the exact
physical placement of GPU-mapped return pages was not established here.

The owner-1 run retained 4037 GPU expert slots per card and 17014 CPU-domain
slots across 49 layers. Per-request work counters are not capacity inventories.
Post-warmup free VRAM remained at 1659.0/1630.5 MiB. Per-process RSS increased
about 41.6/42.1 MiB over the short measured window; this is not a leak certificate.
The 728 MB owner-specific disk prefix-cache payload was removed after retirement.

## Captured ROCm timeline

A separate diagnostic uses the same admitted configuration with 512 prompt
tokens and 16 output tokens. Output IDs equal the first 16 baseline IDs. It
records one preparation prefill, one warmup and one measured request. All 320
native graph launches account for 96868 correlated dispatches. Another 58979
dispatches without recorded graph identity remain unclassified rather than
being assumed to be setup.

The pinned ROCm 10.0 rocprofiler SDK lazily sets
`HSA_TOOLS_ROCPROFILER_V1_TOOLS=0` during HSA API-table registration. Setting that
same value before diagnostic process startup avoids a real post-freeze
environment mutation. An original absent-key control reproduces the metadata
guard failure, and the preconfigured native probe passes. The immutable runtime
environment guard is unchanged. The procedure belongs in the ROCm tuning skill.

The SDK replaces native device-memory queues with system-memory intercept
queues and drops their priority/CU-mask attributes. These timings attribute
work; they do not certify unprofiled scheduling or throughput.

| Measured phase | Root GPU interval union ms | Pure CPU-ticket wait union ms |
|---|---:|---:|
| Prefill, 448 + 64 rows | 2916.326 | 1824.369 |
| Four main-verifier bodies | 415.491 | 235.281 |
| Twelve root MTP draft bodies | 33.671 | 6.969 |
| Four MTP commit policies | 6.871 | 0 |

Intervals on different devices overlap and must not be summed into request
latency. Peer RCCL durations include opaque waiting; they are not communication
bandwidth or pure computation. The 139-node commit graphs are distinct from
the main verifier bodies. Measured root bodies contain 3223, 3277, 3219 and
3277 actual dispatches, although static service metadata says 3277. Each native
launch count closes; the cause of that variation remains unassigned.

QSA scoring is avoidable in this short prefill. `scorePools` is launched over
the admitted maximum block capacity, although every block returns when there
are at most 512 pooled keys. For context 262144 and M448 that is 8192 workgroups
per row, or 3670016 workgroups. The M448 scoring kernels sum to 128.268 ms per
GPU in measured prefill; the M64 kernels add further work. The existing
`ProvenNoRanking` implementation omits indexer-query projection, query
preparation and scoring, but the original production selection restricted it to decode. The qualified
candidate below extends that retained topology to proven short prefill chunks
while preserving key/pool history and the 2051/2052 visible-token boundary.

Native profiler grid dimensions are work-items, not necessarily workgroups.
For example, `materializeCanonicalRouteTicket` reports 65536 work-items with
256 threads per group: 256 workgroups, not 65536. Its launch already has a
bounded live-row loop. Avoid inferring a capacity-sized block explosion from
the raw grid field alone.

## Omit unnecessary short-prefill launches

The candidate carries an immutable proof of the accepted contiguous prefill
range from request ingress to each chunk. A checked visible end at or below
2051 selects the retained `ProvenNoRanking` program; a crossing chunk or larger
end keeps ranking. Bucket capacity, RoPE origin and maximum context are not
proofs. Complete-family and exact-chunk validation precede execution/cache
lookup, and the native guard rejects an incompatible actual visible range.
CPU-owned single-request prefill derives the same proof from its own cursor.
Generic GPU entrypoints without the accepted proof retain the complete program.

The existing builder and native implementation then omit the indexer-query
projection, query preparation and scoring. Required key/pool history, attention
and commit work remain. No kernel body, precision or reduction order changed.
Both ordinary prefill programs are prepared before serving and separately
identified, retained and priced through the existing graph/BOM machinery.
Request proofs are data, not additional cache keys. Prefix rehydration retains
its pre-existing first-use materialization lifecycle; this slice does not claim
to remove that initial lazy capture. Restored and ordinary signatures retain
both programs distinctly without evicting one another on repeat.

A matched original-selector control keeps the **same new paired graphs and
expert placement**, but selects complete ranking work for GPU prefill. Its
source change was reverted and the candidate Release closure rebuilt after the
comparison. All 768 measured output IDs and the MTP work match. Each iteration
accepts 192 draft tokens with no rejects and verifies 256 rows in 64 transactions.

| Release 512/256 workload | Prefill tok/s | Decode-after-prefill tok/s |
|---|---:|---:|
| Original selector, original retained owners | 184.3104 | 30.8388 |
| Original selector, paired retained owners | 183.2827 | 30.1790 |
| Short-prefill omission, paired retained owners | **196.2793** | **30.2303** |

The isolated selector gain is **7.091% prefill** and **0.170% decode**. Candidate
prefill samples are 206.488, 176.266 and 209.722 tok/s; control samples are
190.831, 167.332 and 194.109. This is a short matched workload, not a confidence
interval or the 500/40 target. Decode is effectively unchanged in this control.
The difference from the older owner's decode rate is not attributed solely to
placement or graph retention by these measurements.

Actual ready-graph counts fall by **36 native nodes per GPU per prefill chunk**:
root M64/M448 3913/4154 → 3877/4118, peer 3096/3337 → 3060/3301.
Both runs retain 24 graph entries with zero evictions, compared with the original
12. The paired owners cost eight GPU expert slots in total: 8074 → 8066
(4037 → 4033 per card); CPU-domain slots rise 17014 → 17022. Apart from that
original-owner comparison, the paired selector control has identical admitted
configuration. Dynamic balancing remains enabled. Both runs retire cleanly and
pass their driver brackets; both retired disk-cache payloads were removed.

The broad Integration build compiles CUDA SM80/86/89/90 and ROCm gfx906/gfx1100
with the spill guard enabled. Its three affected Unit registrations run 1253
cases. Together with six focused host aliases, all nine registrations pass.
Eight focused native CUDA/ROCm registrations pass without skips, covering the
new prefill path, existing no-ranking/ranking work and grouped rollback.
New native prefill fixtures sweep M64/448/512/1024/2048 and FP32/FP16/BF16,
poison omitted workspaces, compare full and omitted required bytes, and exercise
partial/empty/reset/repeated capture and the 2051/2052 boundary. An independent
builder test proves projection omission; the native graph inspector proves the
remaining four-node attention program. The original broad production builder
fails the added public-entry regression with its decode-only omission error.
New proof-API tests are positive interface coverage, not that original failure.

These GPU and model receipts belong to the broader local development source.
The feature adaptation preserves its existing CurrentOnly roster and excludes
uncheckpointed KnownNext changes. Exact-feature CPU validation is recorded
separately; broad native/model measurements do not certify an exact feature
binary or a release image. Full preflight and OpenCode remain open.

### Feature checkpoint and fixture repairs

The exact feature's CPU-only Integration companion passes all three new
prefill admission/retention/BOM aliases, the existing no-ranking builder and
CPU aliases, the new topology/locality alias, the existing shared-reader model
authority alias (20 repetitions), and the two-socket gather fixture. Its complete
Qwen4Exp primitives (676 cases) and ForwardExecutionEngine (102) Unit owners
pass. The first run exposed stale rank/node declarations in two planning
fixtures and a locality corruption test that wrote an obsolete byte offset.
The fixes provide the fixture's explicit topology and derive the corruptible
locality words from two real serializer outputs; production guards are intact.

The shared-reader builder also resolved a native hash kernel before rejecting
missing original reader authority. Both ordinary append and pooled read-ahead
now reject that malformed metadata before constructing a stage. The original
append test fails in the CPU-only companion; the expanded two-entry regression
now passes for CUDA and ROCm metadata without native code or graph mutation.
The existing explicit preflight owner runs it 20 times on both build closures.

The final exact-feature run has nine passing and two failing registrations.
**It is not a full CPU-only Unit pass.** All 27 remaining MemoryPlanner case
failures (one also selected by the older combined BOM alias) request exact
compiled GPU graph-object geometry from a build configured with CUDA and ROCm
OFF. The new paired-prefill BOM case itself passes. No guessed GPU object sizes,
skips or relaxed guards were introduced. The broad build with both adapters
compiled passes all ten final selected registrations, including its complete
695-case primitives and 453-case MemoryPlanner owners. Together with the
previously passed 106-case FEE owner, those broad affected Unit owners contain
1254 cases. GPU ABI-dependent cases and a real model run of the exact extracted
feature remain explicitly unqualified; this checkpoint is a focused slice.

## Original two-socket gather comparison

`Perf__CPURootedPublicationTransport.OriginalNUMAVariableRecordGatherReplay`
compares the retained SharedSpin gather with the actual
`GlobalTPContext::gatherVariableFloatRecordsToRoot` MPI path. Both use the same
PMA-owned NUMA-local packed buffers, native width 2561 (2560 FP32 values and
one route slot), root-local receive alias and original socket/domain lifetime.
The narrow test-only friend invokes the already retained shared backend; it
does not introduce a public transport override or another backend owner.

This measurement uses root rank 1 on NUMA node 1, remote rank 0 on node 0.
Each arm has three warmup and twelve measured sequences, covering ten live,
empty, unequal and large-to-small shapes. A readiness barrier is outside the
timed native operation. Source-tagged bytes, received ranges, untouched tails,
corruption controls and retirement checks all pass. The focused CTest completes
in 5.60 seconds. This is ready-data service, not model sender-lateness evidence.

| Remote live payload | SharedSpin slowest-rank median µs | Production MPI median µs | MPI effective GB/s |
|---|---:|---:|---:|
| 1044888 bytes | 344.911 | 232.693 | 4.490 |
| 4189796 bytes | 1336.087 | 941.182 | 4.452 |
| 16769428 bytes | 5447.560 | 3745.913 | 4.477 |
| 71708 bytes after a large request | 29.917 | 28.450 | 2.520 |
| Both participants empty | 3.779 | 3.160 | — |

MPI is 1.42–1.48× faster for these large ready payloads. SharedSpin runs first;
this is not an order-independent crossover certification. Both arms use the
existing `btl_vader_single_copy_mechanism=none` session setting. No result here
qualifies a different MPI mechanism. Keep the production transport policy.

## Next transport candidate: existing OpenMPI CMA

A separate invocation-only diagnostic selects OpenMPI's existing CMA mechanism
on the same original two-socket gather fixture. No host security setting,
transport implementation or inference default changes. An untimed syscall trace
identifies the exact two MPI worker PIDs: rank 1 completes 120 positive
`process_vm_readv` operations against rank 0, transferring 709448220 bytes in
the fixture's exact positive-payload shapes. Both per-process ptracer admissions
succeed, both ranks select OB1/vader, and post-init configuration remains CMA.
There are no permission failures or mechanism substitutions. Both ranks pass,
retire, and leave the owned session base empty. Trace timings are excluded.

Untraced fresh lifetimes run both orders with the same binary/core/helper
metadata, PMA roots, root1/NUMA1 and remote0/NUMA0. Each retains three warmup and
twelve measured sequences, every empty/skewed/repeated shape and exact-byte
checks. The original missing session-directory setup failure is preserved;
it occurred before MPI started and is not a transport failure.

| Remote live bytes | `none` median µs, first/reverse | CMA median µs, first/reverse | Speedup, first/reverse |
|---|---:|---:|---:|
| 1044888 | 236.296 / 236.712 | 179.171 / 183.162 | 1.319 / 1.292 |
| 4189796 | 954.276 / 930.006 | 679.328 / 677.980 | 1.405 / 1.372 |
| 16769428 | 3748.458 / 3724.668 | 2826.588 / 2730.531 | 1.326 / 1.364 |
| 71708 after large | 26.868 / 27.824 | 26.935 / 32.099 | 0.998 / 0.867 |

Large ready-data gathers consistently improve; small and empty shapes have
mixed results. This justifies a matched model diagnostic next, not multiplying
whole-model throughput by the component ratio or changing a global default.
The current default still uses `btl_vader_single_copy_mechanism=none`.

## Retained local evidence

Detailed artifacts are local ignored results, not repository payloads:

- `/tmp/qfn-cpu-root-timing-qualification-r1033/`: baseline model and root timings.
- `/tmp/qfn-hybrid-numa-owner-r1039/`: accepted admission, matched output/work
  comparison, resource observations, driver bracket and cleanup receipt.
- `/tmp/qfn-rocprof-env-replay-r1037/`: original-red/fixed-green probe and
  corrected schema-based `revalidated-summary.json`.
- `/tmp/qfn-hybrid-rocm-timeline-r1038/`: native captured-model trace and
  workload comparison.
- `/tmp/qfn-rocm-timeline-analysis-r1038/REPORT.md` and
  `phase-qualified-summary/`: source-bound phase attribution. The earlier
  marker-only `phase-summary/` is superseded.
- `/tmp/qfn-cpu-gather-perf-r1040/`: candidate patch, build/native logs,
  `ctest.xml`, `results.csv` and `results.json`.
- `/tmp/qfn-no-ranking-prefill-r1041/`: original-red builder, Integration
  compilation, affected Unit/native regressions, Release model and
  `paired-owner-comparison.json`/`.log`.
- `/tmp/qfn-no-ranking-prefill-selector-control-r1044/`: matched complete-selector
  control, source-restoration/native-rebuild receipt and retired-cache cleanup.
- `/tmp/qfn-no-ranking-prefill-checkpoint-r1042/`: adapted feature patch and
  exact-feature companion validation; its source scope is distinct from r1041.
- `/tmp/qfn-ple-reader-admission-order-r1043/`: reader validation-order fix,
  original failure and expanded existing preflight regression.
- `/tmp/qfn-checkpoint-planning-fixture-repair-r1046/`: test-only fixture repairs
  and source/capability boundaries; final validation is in each r1041/r1042
  `fixture-fix-tests.xml`, `fixture-fix-full.log` and `fixture-fix-summary.json`.
- `/tmp/qfn-cma-owned-diagnostic-r1045/`: positive activation receipt, four
  untraced lifetimes and `ab-ba-comparison.json`; native artifacts stay unchanged
  and all owned session bases retire empty.
