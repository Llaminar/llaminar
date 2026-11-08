# QFN CPU NativeVNNI spill audit and bounded nibble dispatch

This is a dated investigation and qualification boundary, not a tuning dashboard.
The applied candidate passed its new focused Integration replays. Exact Release
Q4_K pairwise ISA and corrected 28-core component measurements support the M8
change. The matched Release hybrid run below improves throughput while retaining
all generated tokens. A separately linked pure-AVX2 runtime remains unqualified.

The feature checkpoint extracts the qualified CPU kernel, Q8 publication,
profiler, counter, affinity and benchmark-fixture changes onto the remote
feature branch. Its focused CPU build/tests are recorded separately below.
The matched hybrid measurements come from retained Release binaries of the
broader local tuning tree; they do not certify the whole extracted checkpoint
as a new model/image release. Prefix/known-next protocol changes
and the unmeasured four-row candidate remain separate work.

## Latest matched slice: unique-source Q8 publication

The transported CPU expert stage previously quantized each expert-major route
independently, including repeated copies of the same hidden row. The new typed
schedule derives representatives from the original live route map, retires an
existing scratch permutation after its last reader, quantizes each distinct
source once, and copies exact native Q8 blocks to the remaining destinations.
No activation arithmetic, payload allocation, route selection or physical-memory
ledger is added. Existing-team execution keeps its worksharing and root-to-copy
barrier; ordinary small verifier calls avoid opening a team for duplicate work.

The original-body control fails the focused work regression with 32 quantized
rows instead of 4. The candidate passes all 12 selected tests, including AVX2 and
AVX512 runtime routes, all 21 source formats in the actual stage, alias/malformed
schedule rejection, repeated/partial/reset input and profiler-view lifecycle.
Both new unique-source integration aliases and both profiler lifecycle aliases
are explicitly registered in production preflight. This is a focused receipt,
not a complete Unit/preflight pass:
[qualification](/tmp/qfn-cpu-unique-q8-qualified-r1020/state.json).

The same slice removes per-packet expert-set construction and repeated sorting
from the CPU profiler by borrowing the completed stage's retained sorted view.
New keys and genuine aggregate growth may still allocate. Three scalar child
timings distinguish local execution, its nested post-service observation, and
joint publication; observations are published through the existing terminal
flush. Parent timing semantics remain unchanged.

The coherent Release run retained 2 MI50s, two actual CPU sockets, dynamic MTP
and balancing, RAM-backed ngrams, identical cache budgets and admitted 262144
context. One warmup and all three 512/256 measurements were retained. Median
prefill rose 150.781608→**161.706403 tok/s**; after-prefill decode rose
27.298050→**29.483932 tok/s**. All 768 generated tokens and all MTP counters match
the preceding C run. Config/prompt differences are zero apart from artifact
paths; both runs perform full-miss prefills. Dynamic placement timing and live
CPU route counts remain observations, not normalized inputs. This is the combined
Q8/profiler effect, not isolated attribution to either change.

[Comparison](/tmp/qfn-hybrid-q8-release-r1021/model-c-q8-comparison.json),
[build/admission/lifetime](/tmp/qfn-hybrid-q8-release-r1021/state.json), and
[frozen closure](/tmp/qfn-hybrid-q8-release-r1021/snapshot/manifest.json) retain
the evidence. Native and driver exit 0, processes retire, and the full native
artifact metadata remains unchanged across admission and measurement. The short
repeated prompt, 100% MTP acceptance and three requests do not certify arbitrary
OpenCode throughput or long-lifetime leak freedom. The 500/40 target remains open.

### Matched workload configuration

Both measurements use the four-shard Qwen3.8-Flash-Next UD-Q4_K_XL GGUF
and `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, with Release AVX512 CPU codegen.
The MI50 continuation domain is `rocm:1,rocm:2`, tensor parallel on rank 0.
The same GPUs own the priority-0 apportioned expert tier. The priority-10 CPU
capacity tier tensor-shards experts across two physical Xeon 6238R sockets,
one MPI rank per socket, 28 physical cores each, using the existing node-local
UPI transport. Expert phase is uniform and both phase assignments use
`static-owner`; physical-memory admission chooses expert quotas.

Ngram policy is `gpu-adaptive-vram-ram-ssd`, source backing `ram`, placement
`consumer-local`, with 1,048,576 maximum rows and 35,216,192 cache bytes per
declared consumer. Dynamic MTP and dynamic expert balancing remain enabled.
Admitted context is 262,144; prefix budgets are 16,384 MiB RAM and 32,768 MiB
disk. The exact 512-token prompt is `" test"` repeated 512 times. Each request
generates 256 tokens, with seed 42 and native sampling/precision defaults.
One warmup precedes three measured requests; decode throughput uses the 255
tokens generated after the first token. The retained comparison above owns
the complete commands and unmodified per-request records.

The latest root verifier Q8 publication falls **15.1904 to 2.89294 ms** per
four-token transaction; root endpoint service falls **57.5038 to 46.1518 ms**.
Root FFN time is 33.4592 to 34.3626 ms, so this comparison establishes no
additional GEMM kernel win. Follower joint publication includes waiting and
bookkeeping; its time cannot establish a root combine/preweight bottleneck.
[Attribution and bounded resource observations](/tmp/qfn-cpu-endpoint-profiler-r1017/q8-model-attribution/ATTRIBUTION.md).

### Extracted checkpoint validation

The exact 41-file CPU/documentation checkpoint was configured and built from
`feat/qfn-ngram-continuation` based on `a5d3ee32a`, independently of the broader
tuning tree. Its CPU-only Integration build uses AVX512 codegen and the pinned
existing dependencies. **All 36 selected CTest registrations pass, with zero
failures or skips, in 49.85 seconds.** They cover the AVX2/AVX512 runtime Q8
and NativeVNNI paths, all-format byte replay, partial/reset/lifetime behavior,
counter multiplicity, profiler views, aligned benchmark layouts, MPI and HTTP
affinity, actual OpenMP loader binding, and Release module selection.

All 16 new Integration aliases are present in the generated preflight
registration, and the generated preflight build target depends on their actual
compiled owners. The final inventory is retained after CMake's main preflight
assignment so earlier registrations cannot be accidentally erased.
[Focused build/test receipt](/tmp/qfn-cpu-perf-checkpoint-r1024/state.json),
[JUnit](/tmp/qfn-cpu-perf-checkpoint-r1024/focused-tests.xml), and
[generated membership proof](/tmp/qfn-cpu-perf-checkpoint-r1024/preflight-membership.json).
This is a focused green checkpoint, not a full Unit/preflight or model/image
certificate. No GPU kernel changes or unmeasured four-row implementation are
included. Full preflight and a new model/OpenCode run of this exact extracted
checkpoint remain open.

## Actual formats and compiled families

The [GGUF header inventory](/tmp/qfn-cpu-fused-perf-measure-r1010/expert-tensor-header-inventory.json)
authenticates tensor descriptors without reading payloads. Primary gate/up banks
are **Q4_K**, giving TP2 shards N=320/K=2560; primary down banks are literal
**Q5_1**, N=2560/K=320. Five down banks are Q8_0; two gate/up banks are Q5_K.
Prepared codebook 5 alone cannot distinguish Q4_1 from Q4_K, nor codebook 7 Q5_1
from Q5_K. Source superblock identity and actual prepared encoding are required.

The [source catalog](../../../../src/v2/tensors/NativeVnniFormatInfo.h),
[CPU preparation](../../../../src/v2/kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h)
and [canonical test registry](../../../../tests/v2/utils/QuantizedVerifierFormats.h)
map all **21** supported quantized source tuples to these shared CPU families:

| Prepared family | Source formats | Arithmetic/layout distinction |
|---|---|---|
| NibbleLUT | Q4_0, IQ4_NL, Q4_1, Q4_K | Linear, non-linear LUT or asymmetric identity decode |
| NibbleLUTSourceScale | IQ4_XS | Original base/coefficient factors reconstructed in FP32 |
| ExpandedInt8 | Q5_0, Q5_1, Q5_K, IQ3_XXS, IQ2_XXS, IQ1_S, Q8_0, Q8_1, Q8_K | Shared signed-byte body; source asymmetry selects correction |
| IQ3SExpandedSourceScale | IQ3_S | Expanded signed grid with original source coefficient |
| Q6KNativeDualScale | Q6_K | Native bitplanes and original dual coefficients |
| CompactMultiScale | Q3_K, Q2_K, IQ2_S, IQ2_XS, IQ1_M | Codebook-specialized payload and scale/min equations |

Q8_K's FP32 source scale does not imply a separate FP32-scale CPU hot body;
its existing CPU preparation boundary belongs to ExpandedInt8. Explicit rotation
also selects ExpandedInt8 while preserving source identity. F32/F16/BF16 use
other kernels; source IDs 1/2/3/18 have no tuple in this NativeVNNI catalog.
The [21-format/call-site completeness table](/tmp/qfn-cpu-moe-isa-audit-r1011/FORMAT_CALLSITE_COMPLETENESS.md)
separates the complete baseline compiled inventory, focused candidate Release
observations, Integration receipts and remaining pure-AVX2 qualification. The complete [per-format inventory](/tmp/qfn-cpu-moe-isa-audit-r1011/retained-core-all-families/source-tuples.json)
records source codebook, superblock, asymmetry and branch-specific findings.

## What the Release control proves

The frozen [retained baseline](/tmp/qfn-cpu-fused-perf-measure-r1010/expert-retained-cardinality-only-snapshot/manifest.json)
binds executable, core, critical sources and owner closure. Benchmark-owned
inline code and model-core bodies were inspected separately: 106 and 245 initial
ranges, followed by complete ordered and single-row policy dispatchers. Both
BackendNative and GPUAlignedExpert branches are retained where applicable;
AVX512 rows 1/2/3/4 and runtime AVX2 rows 1/2 map to their actual bodies.
Grouped AVX2 wider work composes the same pair/single bodies. These are offline
artifact observations, with no production source-hash or probe machinery.

* Actual Q4_K GPUAlignedExpert pairwise/local execution enters the generic
  64-column two-row body. It spills five live FP vectors through stack memory
  per K block, about 640 bytes of store/load traffic, in addition to the wrapper's
  intentional 512-byte per-partition materialization. The same pattern is
  independently present in the benchmark executable and core.
* Generic IQ4 pair bodies materialize three widened scale vectors per K block.
  Ordinary asymmetric expanded pairs spill one activation-sum scalar; wide4
  spills three. The separately retained SumPairs variant has an FP accumulator
  slot. Ordinary Q5_1 false-pair execution must not inherit that variant's count.
* Q6's hot stack references primarily hold decoder constants. All 40 retained
  compact Codebook/Lanes/Rows/Tree classes were mapped in both artifacts; their
  own bodies have no FP accumulator spills. IQ2_XS runtime AVX2 pair materializes
  one decode constant. Pointer metadata, constants and scale temporaries remain
  distinct findings from live accumulator spills.
* The complete bounded ordered pair dispatcher already contains every decoder,
  asymmetry and original-scale branch, with no FP accumulator stack operands.
  Its integer K-bound metadata accounts for ten vector stack references. The
  complete ordered single-row and policy-specialized single-row owners have no
  vector FP stack traffic. The exact partial reducer's 256-byte stack temporary
  is solely a source-declared column-tail branch; full 64 stores directly.

The fused owner's deliberate 4x16x64-float partial tree is 16 KiB, not proof of
compiler spilling. Native wide-row kernel cleanliness does not establish that
this materialization is economical. Exact offsets, call edges and the corrected
LUT=0/Q4_0_LINEAR=1/Q4_1_IDENTITY=2 enum interpretation are in
[the detailed findings](/tmp/qfn-cpu-moe-isa-audit-r1011/confirmed-findings.md).

## Scoped change and remaining qualification

The [projection implementation](../../../../src/v2/kernels/cpu/gemm/CPUNativeVNNIGemv.h)
now admits every actual nibble encoding to the existing bounded ordered pair
dispatcher in the GPUAlignedExpert/AVX512/OutputTileLocal multi-block-partition
branch. It preserves the one-block shortcut, integer dots, original scale/min
edges, positive-zero partial initialization, first-partition assignment and
increasing K-partition fold. AVX2 and BackendNative dispatch are unchanged.
The independent down K=320 four-row idea remains a
[separate arithmetic-proof note](/tmp/qfn-cpu-ordered-nibble-r1012/down-one-block-partition-note.md).
A potential four-row/one-segment ordered tile is recorded in a
[deferred feasibility note](/tmp/qfn-cpu-moe-isa-audit-r1011/future-four-row-ordered-nibble.md);
its shared-body design and measurements follow the current A/C result.

The focused all-format replay uses actual prepared kernels, explicit GPU-aligned
activation quantization and independent public serial rows. It covers
M=1/2/3/8/64, Pairwise/Auto, workers 1/4, real N=320/K=2560, tails, empty terminal
partitions, partial descriptors, dirty buffers and repeated large-to-small work.
Small Decode and signed-zero controls cover every eligible nibble source.
It removes the diagnostic K-tile override and asserts actual local-route
selection. The existing owning fused-completion target has explicit
`V2_Integration_CPU_NativeVNNIOrderedNibbleReplay_AVX2/AVX512` production-preflight
entries. This arithmetic control is expected to pass original and candidate;
the original economy defect is the compiled spill traffic.

The [original arithmetic controls](/tmp/qfn-hybrid-prefill-cpu-focused-r1013/original-kernel-and-counter-controls.json)
passed all nine registrations, including the new nibble replay on both runtime
ISAs. This is expected: the kernel defect is compiled spill/economy, not an
observed arithmetic mismatch. The separate
[original Runner controls](/tmp/qfn-hybrid-prefill-cpu-focused-r1013/original-negative.json)
record two genuine storage/order behavioral REDs. The
[candidate focused selection](/tmp/qfn-hybrid-prefill-cpu-focused-r1013/candidate-focused.json)
passed 16 of 17 registrations; its sole existing prefix-flow fixture needed the
new generation binding. The
[focused correction rerun](/tmp/qfn-hybrid-prefill-cpu-focused-r1013/transition-fixture-focused.json)
then passed both selected registrations. The initial red receipt remains; no
replacement full-suite or Release performance pass is implied.

## Corrected Release comparison and observed execution

The first [baseline prefill](/tmp/qfn-cpu-fused-perf-measure-r1010/expert-retained-cardinality-only-prefill/report.json)
and [decode M3](/tmp/qfn-cpu-fused-perf-measure-r1010/expert-retained-cardinality-only-decode-m3/report.json)
used real GPUAlignedExpert preparation, fixed 16/10 partitions and policy
N=320/2560. They requested 28 workers, but later process evidence exposed a
launcher defect: direct frontend startup loaded libgomp before the existing
managed-owner re-exec, collapsing the inherited affinity to CPU 0. The initial
one-core A/C loss and IPC reduction are not full-socket evidence. The earlier
generic BackendNative FullK fixture is also outside the production expert regime.

The [corrected public-owner comparison](/tmp/qfn-cpu-owner-affinity-r1016/matched-results)
invokes the immutable session helper before the OpenMP-linked executable.
All 32 cells authenticate actual 28-worker placement on distinct physical cores
0–27. Frozen binary/core identities are unchanged. Disabled-counter Q4_K M8
improves from A891.43 to C821.75 microseconds in forward order and A897.82 to
C811.21 microseconds in reverse order: **1.085× / 1.107×**. M64 is approximately
flat; unchanged Q5_1 down is also approximately flat. These observations do not
establish a universal per-mode overhead or real-model throughput gain.

The separate [assembly and profiler investigation](/tmp/qfn-cpu-moe-isa-audit-r1011/A_C_Q4K_ASSEMBLY_COMPARISON.md)
profiles the same exact M8/Q4_K shape under that public owner, using 15000
existing bundle iterations and fresh per-candidate PC/stat runs. The actual
28-thread worker is observed across CPUs 0–27 in all four runs. Setup, serial
oracle, planning and repeated clearing remain visible; profiler durations are
not speed measurements. Normalized to two rows ×64 columns ×32 K, the executed
Q4_K path has 455 static instructions in A versus 410 in C, with 64 VNNI updates
in either. C eliminates A's five FP-vector spill pairs and wrapper partial
materialization, but has four rather than eight independent dot chains and
doubles scalar Q8 activation reconstruction. The note annotates the exact
sampled executable addresses and confirms matching core instructions.

Fresh whole-tree counters show A/C instructions859.687/778.093 billion,
cycles858.999/771.511 billion and IPC1.00/1.01. Task-clock falls429.193→387.653
seconds, with effective CPUs26.382/26.290 and near-equal cycles/ref-cycles
0.921/0.918. Thus C reduces instructions9.49% and cycles10.19% in this workload;
no port-saturation inference is made. Complete projection work accounts for
about84% of weighted user PCs in both variants (A includes its ordered wrapper).
Libgomp user PCs account for5.51%/5.10%; that is not wall waiting time. Mixed
memset PCs account for5.53%/5.44%, including potentially repeated capacity-sized
decoder-bank clears, and cannot all be called setup without call stacks.

C's actual Q4_K ordered pair body has no FP accumulator stack traffic. The
candidate's complete ordered dispatcher and selected unchanged generic helpers
were inspected in both executable and production core; this does not replace
the baseline all-family audit with an unperformed all-family C audit. Final
application throughput requires the exact Release real-model run. The private
metadata-lifetime candidate remains an unmeasured, paused hypothesis.

Runtime AVX2 inside this AVX512-codegen artifact is audited only as that
compiled regime. Existing pure-AVX2 build directories are historical until their
source provenance is proved. A [proportional diagnostic plan](/tmp/qfn-cpu-moe-isa-audit-r1011/PURE_AVX2_TARGET_PLAN.md)
compiles the actual benchmark and expert-stage TUs with pure-AVX2 codegen,
retaining current production includes/configuration and private outputs. The
object-only compilation and disassembly are recorded below. Separately linked
pure-AVX2 correctness/performance and complete manual spill classification remain
open; this note makes no all-ISA zero-spill claim.


## Production-layout correction and whole-model result

The initial component fixture used separately allocated `N+7` output strides.
Those guarded rows are appropriate for detecting overruns, but differ from the
actual aligned expert output layout. The fixture now times logical `ldc=N`
rows in production `AlignedVector<float>` backing: two interleaved gate/up roots
and one down root, with live cumulative expert offsets. Independent padded
replays before/after timing still check guard bytes; all timed live rows must
match the serial oracle. No timed setup/copy or extra counted invocation was
introduced. Old padded rates above remain identified as component diagnostics.

The original padded fixture completed correctly and then failed the new layout
observer for missing actual aligned layout identity. All six selected layout
and semantic-counter registrations now pass on runtime AVX2/AVX512. Separately,
the original process-affinity regression failed with both workers on CPU0;
the existing admitted OpenMP partition now survives managed process creation
and MPI exec. All eleven focused owner/affinity/runtime/byte checks pass.
These are scoped regressions, not a complete Unit/preflight verdict.
[Layout/build receipts](/tmp/qfn-hybrid-layout-release-r1019/state.json);
[affinity receipts](/tmp/qfn-cpu-affinity-qualified-r1017/state.json).

The coherent Release server/core/decoder/session-owner closure was frozen
through fresh admission and the following actual model run. Same four-shard
Qwen3.8-Flash-Next UD-Q4_K_XL and Q8_0 MTP sidecar; ROCm1/2 MI50 continuation,
real MPI ranks on both CPU sockets with tensor-sharded CPU experts; RAM ngrams
with ConsumerLocal GPU caches; dynamic MTP and dynamic expert balancing.
Full262144 context fits, prefix budgets remain16GiB RAM and32GiB disk. Exact
512-token prefill/256-token output, seed42, one warmup/three measured requests,
normal numerical/sampling policy; no profiler overlaps timing.

| Median actual throughput | Earlier baseline | Current candidate | Change |
|---|---:|---:|---:|
| Prefill tokens/s |140.4554 |150.7816 |+7.35% |
| Decode after prefill tokens/s |20.9153 |27.2980 |+30.52% |

Current prefill samples are170.6973/149.5406/150.7816; decode samples are
27.0971/27.2980/27.4194. All three generated256-token arrays match the baseline
exactly. Each measured request has64 verifier transactions,192 accepted drafts,
zero rejection and256 verified tokens. Decode denominator is255 after-prefill
tokens. This repetitive prompt's100% acceptance is not an OpenCode acceptance
prediction. No configuration or prompt differences remain after excluding
artifact output paths. Prefix work remains full misses, without bypass.

Measured root prefill CPU routes changed577029→576436 (-0.103%) under the
normal dynamic controller; both final movement epochs are8. This is the combined
kernel/counter/owner/lifecycle result, not an isolated claim for the nibble
kernel. Historical admission did not record a core identity; the preserved
pre-C closure cannot retrospectively repair that evidence limitation.
[Complete comparison](/tmp/qfn-hybrid-layout-release-r1019/model-a-c-comparison.json),
[current original artifacts](/tmp/qfn-hybrid-quick-tuning-r1007/candidate-c).

Passive observation retains127 per-TID samples without errors; both actual
compute process trees show activity across28 physical cores on their respective
sockets. Per-TID CPU samples do not give exact per-core ticks or full-time team
occupancy, and historical placement observations are absent. Collection totals
4.12seconds over263seconds native lifetime (maximum51.6ms/sample); it is a
background observer, not a zero-overhead claim. Resource samples have no errors
and at most1.09second gaps. Both original native processes retire cleanly,
with successful driver observations and final canonical device allocations0.
The short run does not certify absence of a long-session leak.

At this earlier C measurement, the next identified costs were duplicate
transported FP32→Q8 publication (about31 expert rows per verifier call from at
most4 distinct live inputs) and post-service profiler allocations/sorting.
The latest slice at the top of this note implements and measures both changes
using retained route/output storage. The active target remains **500 prefill
/40 decode tokens/s**. The broad changed-format economy matrix and a coherent
old-kernel/new-layout control remain pending; byte equality alone does not
certify their speed. Full Unit/preflight and resumed OpenCode remain open.


### Remaining measured costs

The [paired attribution](/tmp/qfn-decode-attribution-r1018/A_C_ATTRIBUTION.md)
uses matched populations, not sums of overlapping rank timers. Root measured
FFN falls77.359→33.459ms per four-token verifier transaction; Q8 publication
remains15.389→15.190ms. Main-verifier parent falls172.782→129.415ms; mean decode
wall falls190.397→146.102ms. Follower dispatch wait remains50.040ms, and its
outer expert/joint scope exceeds endpoint service by27.750ms; that residual
includes instrumentation and cannot be called pure transfer.

Mean prefill wall falls3596.208→3272.973ms/request, while root CPU FFN falls
1456.018→1127.857ms and Q8 stays92.304ms. Follower dispatch remains1077.400ms;
its expert/joint minus endpoint residual is808.199ms, from a warmup-inclusive
population. These phases overlap across ranks. The500tok/s target allows only
1024ms total per512-token request, so Q8 deduplication alone cannot achieve it.

Current actual Q4_K M64 Auto prefill selects Pairwise with a physical row tile2,
not the generic four-row body. A bounded four-row candidate must first prove
its explicit WideRows path, then justify any Auto selection by measured geometry
and task parallelism; simply changing an unused kernel is not a model speedup.
No arithmetic-reassociation, smaller context or extra warmup is authorized by
these measurements. Final CPU publication also has distinct combine/preweight
passes; inspect dedicated timing before attributing its residual to that work.


### Pure-AVX2 object and short resource observations

Both actual owning translation units compiled successfully with the configured
Release recipe, changing only AVX512 feature flags/compiled-ISA definition and
private object/dependency paths. This is a codegen experiment, not a separately
linked AVX2 target. The complete actual object disassembly retains38 selected
NativeVNNI symbols in the expert-stage object and60 in the benchmark object;
no ZMM or high XMM/YMM registers occur. Family/loop stack operands are retained
for manual classification, not declared spills automatically. Full pure-AVX2
runtime/image qualification remains open.
[Recipes, objects and assembly](/tmp/qfn-cpu-moe-isa-audit-r1011/pure-avx2-objects-current-c/manifest.json).

Across the three measured requests the two compute processes grew60.27MiB and
44.69MiB RSS respectively. One large GPU client moved25160.83→25230.22MiB by the
first decode and remained at that level; the other stayed24996.01MiB through
all measured windows. Teardown released both. These are short sampled process
and per-client observations, not summed physical totals or proof of leak freedom.
[Window definitions and raw limits](/tmp/qfn-hybrid3-terminal-resources-r1005/C_MODEL_RESOURCE_WINDOWS.md).
