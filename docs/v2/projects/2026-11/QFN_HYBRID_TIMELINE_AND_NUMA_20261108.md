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
this is not an order-independent crossover certification. Both initial arms
used `btl_vader_single_copy_mechanism=none`. The separate CMA qualification below
does not change the existing SharedSpin-versus-MPI collective selection policy.

## Native MPI selection and OpenMPI CMA qualification

A separate invocation-only diagnostic selected OpenMPI's existing CMA mechanism
on the same original two-socket gather fixture, without changing host security
or implementing a transport. An untimed syscall trace
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
mixed results. The matched complete-model comparison uses the original public
process owner, two real socket ranks, 28 physical workers per socket and the
same Release closure. Each arm has one warmup and three measured 512-token
prefills /256-token completions with seed42. Context remains262144, dynamic
MTP and balancing stay enabled, and the model, prompt, placement, RAM ngrams,
GPU cache and16/32GiB RAM/disk prefix budgets match. Only the explicitly
selected MPI mechanism differs.

| Mechanism | Prefill tok/s | Decode-after-prefill tok/s |
|---|---:|---:|
| Explicit `none` | 197.507752 | 30.737784 |
| Explicit `cma` | **199.674087** | **30.770366** |

This is **+1.0968% prefill /+0.1060% decode** in this short paired run. The later
staging investigation below found156 native read failures in this CMA arm;
these are diagnostic timings, not a clean transport qualification. All768
measured output token IDs and MTP records
match:192 accepted drafts,0 rejects,64 verifiers and256 verified tokens per
request. Native/driver exits are zero, all processes retire, session bases
are empty, and the executable/core/decoder/helper/module metadata is unchanged
throughout. Owned prefix files are removed only after retirement. The user
accepted locking in this modest observed gain; the500/40 target remains open.
The discovered staging defect must be fixed before promotion, and complete
model timing must be repeated with clean native transport diagnostics.

The implementation removes forced `none` from self-bootstrap, frontend runtime
initialization and canonical CMake/parity/admission/KL launch defaults. Native
MPI owns available mechanism selection; no new transport, mechanism enum,
security change or universal forced-CMA setting is introduced. Explicit
operator settings remain intact. Frozen-history launch validation now permits
an absent or explicitly authored mechanism while retaining required lifecycle
controls and rejecting duplicates, unknown controls and malformed argv.

Original-code regressions fail for both the bootstrap argv and the frontend's
environment mutation. The installed candidate passes all18 selected CTest
registrations, including the complete six affected Unit registrations, native
default/explicit-none two-rank startup and eight newly explicit preflight
aliases. The frozen-history follow-up passes all15 tests and its original
validator fails the new valid absent/non-none cases. Full preflight is not run
in this focused iteration.

An additional default-selection activation trace removes the mechanism from
argv/environment and explicitly unsets inherited intent. MPI reports
`cma (default)` and the actual two workers perform the same120 positive
cross-rank copies and709448220 live bytes. Both ranks pass, retire and clean
their session state, without permission failure or mechanism substitution.
These traced timings are excluded from the throughput table.

### GPU migration staging must support kernel cross-process access

The subsequent ordinary self-bootstrap model run (`r1050`) completed with
identical768 tokens/MTP at198.638 prefill /30.605 decode tok/s, but printed156
OpenMPI `Read -1, expected ..., errno = 14` diagnostics. The earlier explicit-CMA
model had the same156 failures; the explicit-none control had none. Normal
process exits and byte-correct eventual output did not prove successful native
CMA transport. The original timing/driver receipts remain intact, accompanied
by an explicit failed transport qualification.

The focused captured-publication regression reproduces the fault on all five
ROCm devices: actual GPU staging maps have Linux `io`/`pf` flags and
`process_vm_readv` returns EFAULT, despite CPU memcpy and captured GPU roundtrip
being byte-correct. Original CUDA backing passes. The existing two-rank ROCm
remote-projection fixture independently reproduces native CMA read failures;
it now rejects those diagnostics even when MPI eventually completes the
byte-correct transfer. No whole-model rerun is needed to reproduce this defect.

`HostTransferAccess::CrossProcessReadable` selects ordinary host pages registered
for the existing GPU aliases. Only remote expert-migration slabs request it;
local GPU slabs retain their native backing. The same slab, exact live-byte
extent, PMA/BOM geometry, NUMA first-touch policy and event lifetime remain in
use. There is no extra payload buffer, added copy, host readback, global HSA
change or second transport implementation. The immutable requirement propagates
to slices; remote endpoints reject local-only staging before native setup.
An arbitrary external registration cannot manufacture this capability.

All17 focused registrations pass: complete mapped-host/transfer Unit owners,
the explicit capability/lifetime regression, CUDA/ROCm captured publication and
process-vm reads, existing registration lifetimes/native ROCm backing, real
two-rank migration, and the affected packed/rectangular all-format, mapped
progress and heterogeneous GPU transfer suites. Both driver windows are clean.
The exact-feature CPU companion also passes all 21 selected registrations,
including the eight complete affected Unit registrations. The new captured
staging aliases are labeled lifecycle tests, rather than recorded-input replay;
final registration metadata validates that distinction. Full preflight remains
deferred for this focused iteration. Native/model evidence applies the same
scoped patch to the broad optimization tree; exact-feature GPU/model
certification remains open.

The repaired actual two-rank migration trace proves 36,864 successful cross-worker
CMA reads totaling 169,869,312 live bytes: 18,432 reads/84,934,656 bytes per
direction, with 4608 bytes returned per call. MPI selects `cma (default)`, both
ptracer authorizations succeed, and no read/permission/substitution error occurs.
All 18 traced tasks retire and the session base is empty. Its 22.53-second traced
lifetime is not a throughput measurement.

The fresh clean Release model (`r1054`) passes admission and normal self-bootstrap
with dynamic MTP and expert migration, zero native read errors, clean driver
observation and complete retirement. The same 512/256 workload gives
**197.140 prefill /30.199 after-prefill decode tok/s**, versus the clean
explicit-none control's 197.508/30.738: **-0.19%/-1.75%**. This is approximately
flat whole-model performance, not evidence of a new end-to-end speedup. All 768
output IDs, prompt/configuration and MTP work match. Dynamic migration changes
the routed-work distribution, so these are not identical per-expert execution
traces. The large-gather component benefit remains separately measured above.
Both GPU canonical allocation totals return to zero at retirement. Individual
inference-process peak RSS is 77.281/36.035 GiB; shared mappings are not summed.
The post-warmup measurement window adds 42.73/41.87 MiB RSS, with stable VRAM
and about 1.6 GiB free per card. All owned processes and MPI session state
retire; only then are the run's 18 prefix-cache files (727,912,505 bytes) removed.
This short run does not certify leak freedom or the 500/40 performance target.

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
- `/tmp/qfn-cma-model-r1047/`: matched explicit-none/CMA model lifetimes,
  `comparison.json`, resource/driver observations and owned-cache cleanup.
- `/tmp/qfn-mpi-native-transport-default-r1048/`: original-red bootstrap/runtime,
  final builds and18-registration focused green, plus `activation-default/`.
- `/tmp/qfn-history-native-mpi-followup-r1049/`: original-red/fixed-green
  frozen-history validator follow-up; these are model-free metadata tests.
- `/tmp/qfn-mpi-default-model-r1050/`: self-bootstrap model, matched-token
  comparison, failed transport qualification and retired-cache cleanup.
- `/tmp/qfn-mpi-readable-staging-r1051/`: original-red captured-syscall and
  actual-MPI proofs, candidate builds, focused/format green and native traces.
- `/tmp/qfn-cma-cross-process-staging-r1052/`: frozen typed backing patch,
  source/ownership analysis and native activation protocol.
- `/tmp/qfn-remote-staging-test-adaptations-r1053/`: existing remote fixture
  adaptations; local-only cases retain their original backing selection.
- `/tmp/qfn-mpi-clean-model-r1054/`: clean admission/model, matched comparison,
  resource/driver and retirement evidence, native closure metadata and cleanup.
- `/tmp/qfn-mpi-native-payload-owner-audit-r1054/REPORT.md`: read-only audit of
  remaining MPI payload owners; no other concrete GPU-native mapping found.
