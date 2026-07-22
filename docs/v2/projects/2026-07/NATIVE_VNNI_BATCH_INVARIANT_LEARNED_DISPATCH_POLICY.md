# Cross-Backend Batch-Invariant NativeVNNI Learned Dispatch Policy

- **Date**: 2026-07-18
- **Status**: Implementation in progress. Exact-point profiler evidence, resumable collection, accelerator-backed fitting, generic totality checks, and authenticated promotion criteria are implemented. The shared overlay inventory covers every declared production geometry and all 21 runtime formats on CPU, CUDA, and ROCm. Decode is measured at M=1 and grouped verification at M=2..16 plus M=31. GPU ordinary prefill retains M=64,256,1024,2048,4096,8192,16384. CPU v11 bounds new evidence to M=64,256,512,1024,2048 below 14B and M=64,256,512 at 14B and above; generic dispatch remains total beyond those measured ceilings. The immediate milestone is a best-effort installation of CPU, CUDA, and ROCm decode/grouped/prefill policies using explicit recorded promotion overrides, followed by production smoke and unit gates. No performance override may waive byte equality, runtime totality, evidence completeness, or route correctness.
- **Scope**: CPU, CUDA, and ROCm NativeVNNI GEMV/GEMM dispatch for `Fast M=1`, a frozen serial-M1 oracle, bitwise MTP verifier rows `M=2..16` plus the deeper M31 sentinel, and ordinary full-K prefill/large-M GEMM policy generation
- **Parent project**: [vLLM-Style MTP Project Plan](../2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md)
- **Evidence dashboard**: [vLLM-Style MTP Tuning Dashboard](../2026-06/MTP_VLLM_STYLE_TUNING_DASHBOARD.md)
- **Normative language**: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY` are requirements levels.

---

## 1. Executive Decision

Llaminar will use one shared, numerics-constrained dispatch compiler for CPU,
CUDA, and ROCm NativeVNNI small-M GEMV/GEMM policies. Each backend will retain
its own candidate kernels and generated C++ include, but all three backends will
use the same:

- semantic dispatch contracts;
- observation schema;
- candidate eligibility model;
- alias-robust runtime exact-winner construction;
- generic aspect/work learner;
- shape-grouped development/CV and sealed-certification protocol;
- strict per-domain measured-p95 and conservative-p95-UCB regret gates below
  5%, with at least 95% of domains required to pass and global p95/maximum
  regret retained as diagnostics;
- generated-policy intermediate representation;
- provenance and stale-policy checks; and
- staged validation and installation transaction.

The compiler pipeline is:

```text
backend candidate measurements
              |
              v
contract-specific correctness eligibility
              |
              v
complete candidate-by-shape cost matrix
              |
              +------> alias-robust runtime exact winner per key
              |
              v
common bounded aspect/work learner minimizing measured latency regret
              |
              v
grouped development CV, then fit/freeze final generic policy on development
              |
              v
sealed shape certification by generic domain, exact overlays disabled, no refit
              |
              v
100% structural coverage + at least 95% of domains with both p95 gates < 5.0%
              |
              v
exact overlays + certified generic rules + policy manifest
              |
              v
CUDA / ROCm / CPU emitters
              |
              v
staged compile + route tests + byte parity + model parity
              |
              v
atomic bundle publish
```

There are two independent dispatch contracts:

1. `Fast`: ordinary performance-oriented execution using the backend's normal
   numerical correctness requirements.
2. `VerifierSerialM1Bitwise`: grouped MTP verifier execution where every
   `M=2..16` and `M=31` FP32 output row MUST be byte-for-byte identical to the same row
   produced by the backend's production serial `M=1` decode path.

The coverage matrix is:

```text
Fast                         M=1
FrozenSerialM1Oracle         M=1 dependency selected and staged first
VerifierSerialM1Bitwise      M=2..16,31 certified against that frozen dependency
```

`Fast M=1` receives first-class exact overlays and generic aspect/work rules.
Grouped verifier rows do not choose an independent arithmetic schedule: their
only production candidate inherits the complete frozen serial-M1 family,
exact K partition, and ordered reduction tree for the same runtime key. The
verifier sweep certifies that inherited grouped implementation for every
required M and format; it cannot retune around byte equality.

### 1.1 Implementation status on 2026-07-13

The shared schema, candidate registry, exact oracle, segmented-regret learner,
common corpus, and CPU/ROCm verifier adapters are implemented. The CPU verifier
path is the first fully trained production consumer of the common compiler.
Its dispatch key includes all runtime-visible performance dimensions:

- compile-time ISA (`AVX2` or `AVX512`);
- effective runtime ISA (`AVX2` or `AVX512`);
- OpenMP thread count;
- normalized runtime codebook;
- verifier depth `M`; and
- exact `N` and `K`, followed by certified generic aspect/work rules.

CPU training MUST collect three independent regimes. An AVX512-capable host
does not make the first two interchangeable:

1. AVX2 build with AVX2 runtime dispatch;
2. AVX512 build with runtime dispatch forced to AVX2; and
3. AVX512 build with native AVX512 runtime dispatch.

The AVX2 and AVX512 builds use independent oneDNN build/install directories
(`external/onednn/build-avx2` and `external/onednn/build-avx512`) so configuring
one build cannot silently replace the linked ISA regime of the other.

The first, now superseded CPU corpus used 28 threads on an Intel Xeon Gold 6238R
and retained 4,536 strong aggregate observations plus 84,924 raw timing rows.
It covered every registered source-format alias, 18 runtime codebooks,
verifier `M=2..4`, twelve Qwen 3.6 dense/MoE production shapes, and all three
ISA regimes. Every accepted candidate had zero output-byte mismatches and zero
repeat-byte mismatches. The compiler emitted 1,944 exact policies plus the
cross-validated generic policy into:

```text
src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc
```

That artifact does **not** certify the expanded speculative-depth envelope and
must be replaced before the batch-invariant project is considered complete.
The replacement corpus includes every integer M=2..16 plus M31. Fifteen draft
tokens require sixteen target-verifier rows because the grouped target pass
contains the draft rows and one bonus row; M31 remains a deliberate deeper
sentinel and is not a runtime maximum.

The reproducible production refresh command is:

```bash
scripts/refresh_native_vnni_dispatch_tables.sh \
  --backend cpu \
  --profile qwen36 \
  --cpu-threads 28 \
  --cpu-avx2-sweep-bin build_v2_release/tests/v2/v2_perf_cpu_native_vnni_gemv \
  --cpu-avx512-sweep-bin build_v2_release_avx512/tests/v2/v2_perf_cpu_native_vnni_gemv \
  --output-dir benchmark_results/native_vnni_dispatch/<run-id> \
  --install
```

The wrapper rejects CPU installation from smoke or partial profiles, rejects
an incomplete ISA matrix, and validates generated codebook references before
publishing. Production resolution has no serial-row or untrained-policy
fallback: an uncertified build/runtime/thread domain fails closed.

CPU ordinary prefill is now also a trained production consumer. A zero-kernel
C++ route probe calls the production `computeTileConfig()` implementation for
every supported execution codebook, canonical shape, and ISA regime. Its
authenticated manifests expose the actual thread count, `k_tiles`, and
full-K/serial-K-part arithmetic bundle to the covering planner and analyzer.
This prevents Python planning logic from silently approximating the
cache-aware production route.

The route-certified v26 corpus contains 16,856 strong observations and covers
all 18 CPU execution codebooks across all three build/runtime ISA regimes. It
emitted 1,820 exact production policies plus a provisional generic table into:

```text
src/v2/kernels/cpu/native_vnni/CPUNativeVNNIPrefillPolicyGenerated.inc
```

That generic table is **not certified** and must not be treated as installable.
The first installed one-pass policy escaped without the mandatory regret gate;
offline evaluation later measured 6.762298% maximum verifier regret and
8.178757% maximum prefill regret. The defect was procedural as well as
statistical: installation accepted a generated artifact without proving the
then-required 3% worst-surface and simultaneous-UCB economy thresholds. The
current v15 transaction supersedes that maximum gate with strict measured-p95
and conservative-p95 gates while retaining worst-cell values as diagnostics.

The first CPU prefill sealed transaction is also burned. Once the certification
projection was corrected to preserve the development corpus's collapsed-aspect
domain mapping, the six-shape v1 holdout covered only 123 of 266 sealed cells,
left 344 rules unexercised, and reported 87.1861% measured maximum regret with a
90.7268% simultaneous 95% upper bound. Those rows have now been opened and may
only be incorporated as development evidence in a later attempt; they can never
certify another freeze.

The common learner now promotes a generic domain only when grouped development
CV has complete required coverage and measured p95 regret strictly below 5%.
Worst-cell regret remains visible in every diagnostic but cannot veto a
p95-passing domain. Failed or
sparse domains remain explicit `performance_unpromoted` refinement obligations
and make the entire policy non-installable; exact overlays cannot satisfy or
hide that obligation. Profiler penalties may steer tree selection, but the
published development regret is always recomputed from canonical timing. The
profiler-informed CPU prefill v27 planning freeze contained 1,820 exact entries,
promoted 73 generic leaves, kept 461 domains exact-only, and had a measured
maximum promoted development regret of 2.962859%. Its frozen generic digest was
`sha256:37f24bc4b970a538285d7d98f103fae898ed5eb29ca2bf72b65cf2eab3726543`.
That digest is historical planning evidence only and MUST NOT be certified.

The v2 sealed collector opened `V4FastSealed_Balanced_608x704` and found that
the economical two-row full-K kernel published a physical 64-column tail into
a 32-column logical row tail. This corrupted the next row, produced 480 bit
mismatches at `M=64`, and eventually overran the final output allocation. The
repair stages only partial physical chunks and copies the logical prefix while
leaving full chunks on the direct grouped path. A focused production-route
integration regression now sweeps every format, even and odd M, the exact
failure cell, all five N-block schedules, and output guard regions. The exact
trainer cell passes with zero bit mismatches in AVX2-build/AVX2-runtime,
AVX512-build/AVX2-runtime, and AVX512-build/AVX512-runtime regimes.

Because v2 measurements were opened and the kernel source hash changed, the
entire v27 freeze and v2 seal are permanently ineligible for certification.
The subsequent v3 transaction completed all 504 process jobs and retained
4,032 strong observations, but its sealed matrix contained only `M=64`. It
independently rejected the frozen policy at 35.8377% maximum measured regret
and left 39 generic rules unexercised. Those rows are now opened development
evidence; v3 can never certify another freeze.

CPU prefill v4 repairs both evidence gaps without recollecting the immutable
baseline. Its development corpus contains 32,984 strong observations: the
16,856-observation route-certified baseline, 4,032 opened v3 observations, and
12,096 new observations around four focused CV-transition geometries. Each
focused geometry crosses all 18 CPU runtime codebooks, all three ISA regimes,
and `M={64,256,1024,2048,4096,8192,16384}`. Source aliases that normalize to
the same CPU codebook share one physical refinement launch; the baseline still
co-measures every source alias, and the fresh sealed holdout remains fully
Cartesian over all 21 source formats.

The historical frozen v4 development policy contains 3,764 exact production overlays, 70
generic leaves, and 486 explicitly unpromoted domains. Every emitted
generic leaf passed grouped development CV, with 2.934441% maximum measured
regret. Its frozen generic digest is
`sha256:aeb6dcc2eefcb3f739b5c0cb1d620469a9bc99b7caf3eda18077c758a74ecdb0`.
A timing-free pre-seal validator proves every frozen leaf is reachable through
the committed C++ serial route and sealed geometry before collection can begin.
That check exposed one very-wide leaf just beyond the original synthetic
`16608x512` witness, so the still-unopened inventory was amended to
`17216x512` and refrozen before any v4 sealed timing existed. All 70 leaves are
now structurally reachable. The fresh v4 holdout consists of eight geometries,
all 21 formats, all seven M buckets, and all three ISA regimes; collection is
open, but no v4 result is installable: generic dispatch was not total.

The later v7 development fit contains 33,944 observations, 3,858 exact
overlays, 22 generic leaves across only 17 promoted domains, 510 unpromoted
domains, and 378 grouped-CV reports. Its frozen generic digest is
`sha256:941674d2b3f2b9aa2a7efaf274b4b560478a3057bbf102563ad3ca85609cb515`.
All selected sealed launches passed byte equality, but certification correctly
failed because exact overlays had allowed the sparse generic policy to reach
the seal. The v7 seal is spent and the artifact is historical refinement input
only. The replacement transaction requires all 756 CPU prefill domains (`3`
ISA surfaces times `18` codebooks times `7` M buckets times `2` authenticated
arithmetic bundles) to own one total generic partition before freeze.

Failed-fit diagnostics are now explicitly non-installable. Each diagnostic is
converted into a digest-bound refinement plan that selects fresh non-overlay
geometry near every failed CV boundary, authenticates the C++ serial route,
and appends only the missing timing. The isolated profiler corpus is already
one launch per physical candidate variant and is reused without recollection;
a changed candidate registry requires a new profiler transaction. A
later fit may consume a lineage of those plans; old timing and profiler rows
remain immutable and are never recollected merely because feature engineering
or fitting changes.

AVX2 serial-K-part execution has only the economical Pairwise route. Native
AVX512 measured WideRows as the winner in all 252 forceable serial-K-part cells;
the generated ABI selects Pairwise explicitly for `M=2`, where WideRows is
physically unavailable, and WideRows from `M=3` onward. Real production-Auto
all-format integration sweeps pass byte equality against independent serial-M1
rows in both AVX2 and AVX512 builds, while explicit perf-only forcing continues
to exercise every candidate as a diagnostic. Correctness is established;
generic economy installation remains gated on the fresh sealed certificate.

CUDA now has a production-path M1 trainer with a sample-interleaved broad
protocol, mode-aware policy ABI v2, shape-resolved K-partition formulas, and a
strict paired-tournament protocol. The first Q4 development corpus retained
49,600 strong observations across eager and graph-captured execution. A typed
planner now converts every over-budget held-out CV decision into concrete
selected/reference requests, resolves formulas to forceable exact-KB IDs,
batches same-cell tournament edges with request IDs, refits, and stops only at
green, a measured policy failure, or an evidence conflict. The canonical
refresh wrapper performs this development loop before opening sealed data or
staging grouped verifier M=2..16,31.

The complete all-format CUDA retune, paired sealed confirmation, ROCm
batched-projection migration, GPU ordinary-prefill training, and final
cross-backend release transaction remain open.
Therefore this document is not yet a claim that the cross-backend transaction
or sealed-corpus release gate is complete.

The profiler sidecar transaction is now implemented across all three backends.
CPU Linux-perf control was proven on real AVX2 and AVX512 grouped launches.
ROCm was proven on gfx906 with an isolated three-dispatch production pipeline
and eight physical profile passes. CUDA isolated-launch control and report
normalization are proven with Nsight Compute 2025.3.1, installed through the
CUDA-13.0-matched `cuda-nsight-compute-13-0` package; the development image
enables that package explicitly through `INSTALL_CUDA_PROFILERS=1`.

The bounded retained CUDA development corpus completed all 932 physical-candidate
Nsight requests with zero missing, failed, or unsupported records. The bounded
ROCm development corpus completed all 64 physical KB requests with zero missing
or failed records. Each request launches exactly one candidate transaction
independently of canonical timing and retains the trainer row, profiler-native
report/counter files, exported normalized metrics, tool and binary provenance,
and immutable digests. Collection was lane-partitioned over two RTX 3090s and
four MI50s, respectively, with no duplicated request assignment. These
sidecars are raw reusable corpus inputs: refitting or feature engineering does
not authorize recollecting canonical timing or profiler evidence.

### 1.2 Historical CPU-prefill diagnosis superseded on 2026-07-18

The current anchor-aware geometry-coherent cross-M replay emitted 1,430 rules
and passed 556/756 required domains, or 73.5450%. Its 200 failures are
exclusively full-K domains; every serial K-part domain passes. Failure p95
regret ranges from 5.0062% to 50.2571%, with an 8.3936% median. The policy is
therefore structurally useful but not remotely promotable under the
99%-of-domains gate, and no generated include has been installed from it.

Direct ExtraTrees, per-candidate experts, per-M models, histogram boosting, and
pairwise ranking were evaluated as learner diagnostics. Balanced-fold variants
reached at most 677/756 domains, but those figures are optimistic because they
do not use the canonical coherent spatial KD folds. Their intersection is more
useful: 41 full-K domains failed every tested model family, while per-M models
underperformed cross-M sharing. A canonical 315-surface diagnostic was stopped
before completion because it would have recomputed every forest on each
process restart. Prediction surfaces are now content-addressed and resumable;
no timing or profiler launch was repeated.

The diagnosis also invalidated the old assumption that normalized profiler
counters transfer unchanged away from the isolated profiler shape. Dynamic
CPU/GPU counters are now tied to their exact `(M,N,K)` anchor and masked when
that N/K lies inside a coherent held-out region. Static launch/resource data
remain transferable. This correction changes the profiler model identity, so
the next canonical fit must rebuild its surfaces once before subsequent replay
becomes a cache hit.

The first direct-teacher canonical replay trained 1,549 expanded held-point
surfaces in 1,697.080 seconds on 56 physical-core workers, then completed
432,576 exact CV tasks in 1,242.0 seconds across two RTX 3090 and four MI50
devices. All six devices sustained production load. The teacher increased the
passing count from 556 to 619 of 756 domains and reduced failures from 200 to
137, but did not meet the 99% quota; no policy was installed. It was selected
for 178 domains, of which 106 passed held p95. The residual set contains 136 CV
p95 failures and one publication-distillation p95 failure.

Residual teacher misses expose a specific missing model feature rather than a
need for broader timing evidence. Twenty-eight select the one-block pair grid
where the one-block N-major two-row schedule is exact, while another cluster
selects oversized pair grids where row-chunk-grid is exact. The old model sees
`M`, `N`, `K`, candidate configuration, and normalized counters, but not the
candidate-specific production task grid. Learner v9 therefore derives exact
parallel task count, worker waves, final-wave utilization, rows/output
values/MACs per task, and N/M tail utilization from the production launch
formulas. These are runtime-visible analytical features with no timing or
winner label. A one-leaf non-installable replay evaluates the teacher before a
second full publication tournament is authorized.

The grouped-CV accelerator has now moved from host-published temporary trees to
scorer ABI v8's fused device path. One captured replay performs complete beam
search, precomputes held-out threshold predicates, finds exact held-out winners,
routes every fitted budget tree, and publishes only compact fold metrics. The
pipeline owns persistent device buffers and an explicit non-default stream; it
has no intermediate or full-device synchronization and performs exactly one
terminal stream synchronization after the final compact D2H. The old
tree-returning API remains only for final policy publication, where stable
generic IR is the requested result.

ABI v8 uses eight 64-bit point-mask words, lifting graph-resident training and
held-out capacity to 512 points. CUDA and ROCm integration tests now exercise
the actual 444-point collapsed-aspect CPU decode domain that exceeded ABI v7.

Focused CUDA and ROCm integration regressions prove byte-identical agreement
with the serial Python evaluator, including heldout-only exact winners, every
budget, production feature predicates, graph replay, compact D2H accounting,
and the no-intermediate-sync contract. Both scorer DSOs are explicitly compiled
with device `-O3` in Integration builds. CUDA ncu reports zero local-memory
spills, 19-36 registers per thread, and 66.67-100% theoretical occupancy across
the three kernels. Final gfx906 code objects report zero private segment,
dynamic stack, SGPR spills, and VGPR spills; rocprofiler records 97.72% VALU
utilization for threshold precomputation, with low memory-unit pressure and
44-68% L2 hit rates in the intentionally tiny regression geometry. Exact-winner
and routing utilization are grid-limited in that five-point test and scale to
64 and 128 blocks at the supported corpus maxima. Tensor cores are deliberately
absent because these kernels are integer control/reduction work plus exact FP64
comparisons, not matrix multiplication.

### 1.2.1 CUDA grouped verifier checkpoint on 2026-07-20

CUDA grouped verification now exposes two real production candidates: trained
DP4A row-reuse tiles and a dedicated Ampere `m16n8k32` signed-INT8 tensor-core
kernel. Reusing the ordinary prefill tensor-core epilogue was rejected after
the first Q8_1 M=4 byte gate produced 12,288 mismatched values out of 16,384.
The accepted kernel instead feeds exact INT32 MMA dots into the same explicit
round-to-nearest FP32 scale/min/delta helper and ascending K-partition reducer
as serial M1 decode.

The dedicated tensor-core candidate first passed 672/672 production-route
cells: all 21 formats, eager and graph-captured execution, and every canonical
verifier depth M=2..16 plus M=31. The follow-up complete tournament exercised
every launchable DP4A row tile and the tensor-core candidate in 2,940 rows over
that same format/depth/mode matrix. Every row had zero output-byte mismatches,
zero repeat mismatches, and valid route, workspace, explicit-stream,
serial-route, and graph-capture counters. This wider proof also removed the old
WIDE/DIRECT per-row kernel: those cells now execute the same true grouped DP4A
weight-decode reuse kernel as KPAR while forcing the serial WIDE/DIRECT KB=1
tree and publishing partition zero without an extra signed-zero-changing add.

The corrected low-sample M={16,31} diagnostic selected tensor cores in 16 of
84 format/M/mode domains, with DP4A r2/r4/r8/r16 owning the remaining domains.
For representative Q8-family M=31 cells, tensor cores are approximately
1.10-1.12x faster. These timings prove candidate diversity and justify keeping
both physical families; the full corpus transaction remains authoritative for
installed exact overlays and generic rules.

Nsight identified and closed a one-warp-block occupancy limit. Packing four
independent private-shared-memory warp tiles per CTA increased theoretical
occupancy from 33.33% to 83.33% and achieved occupancy from 31.29% to 75.12%.
The final Q8_1 M=31 profile records 46 registers per thread, zero local-memory
spills, 65,536 IMMA instructions, and 90.40% memory-pipe utilization. The
DP4A path now similarly packs independent N tiles for low-register row depths.
The Q4_0 M=16 direct r2 profile records 32 registers per thread, zero spills,
100% theoretical occupancy, and 25.97% achieved occupancy. The latter is
finite-grid limited: byte-exact KB=1 work exposes only 32,768 useful threads,
and packing changes their block ownership rather than inventing arithmetic.
Nsight replay duration improved from 73.22 to 72.38 microseconds. The generated
grouped policy ABI is typed: exact overlays and then total geometry/M rules
choose `Dp4aRows` or `TensorCoreMma16`; an absent decision is a hard failure and
cannot invoke serial row replay. Full CUDA corpus fitting, sealed
certification, and installed-policy smoke remain open.

### 1.3 Three-backend symmetry acceptance matrix

Cross-backend symmetry means an equal semantic contract and equal quality of
evidence, not identical kernel families. CUDA and ROCm grouped verifier kernels
must inherit the frozen serial-M1 K partition and ordered reduction tree; their
economical tuning axes may change row work sharing, occupancy, and weight reuse
only when those choices cannot change FP32 parenthesization.

| Production surface | CPU | CUDA | ROCm | Work required before symmetric installation |
|---|---|---|---|---|
| Serial decode / `Fast M=1` GEMV | Five-candidate NBC registry, all-format trainer, exact/generic compiler, sealed certification, and fail-closed include implemented; production corpus collection pending | Common registry/trainer implemented; profiler corpus complete; certification pending | Common registry/trainer implemented; profiler corpus complete; certification pending | Collect/profile/certify the CPU M1 corpus and certify both GPU policies |
| Grouped verifier projection | Pairwise/WideRows common trainer exists; expanded-depth replacement certificate pending | DP4A row-reuse and integer-tensor-core candidates are byte-proven for every format/depth/mode; typed exact/generic compiler implemented; full corpus certificate pending | `INHERIT_SERIAL_M1` common surface exists; grouped proof and final certificate pending | Complete CUDA fitting/certification, then prove the same all-format/depth route and economy contract for ROCm against each frozen M1 policy |
| Ordinary dense prefill / GEMM | Partial-N corruption fixed; v4 development frozen at 2.934441% maximum promoted CV regret; fresh all-format/all-M sealed collection open | Legacy backend-specific generator | Legacy backend-specific generator | Complete CPU v4 certification, migrate both GPU generators to common profiler-informed freeze plus fresh sealed certification, and install only after each backend's sealed gate passes |
| Fused dense bundles | Correctness sweeps exist, but QKV, gate/up, fused SwiGLU/down, and GDN bundles are not first-class common learned domains | Same | Same | Add honest projection vectors and bundle signatures to all three registries, corpora, learners, emitters, and sealed partitions |
| Routed/shared grouped MoE | Production grouped implementation and byte sweeps exist; no common learned grouped-MoE policy | Production grouped implementation and byte sweeps exist; no common learned grouped-MoE policy | Common adapter and 12-candidate registry exist, but the emitter is exact-anchor only and lacks profiler-informed freeze/sealed certification | Build one symmetric grouped-MoE transaction for CPU/CUDA/ROCm, including router-to-expert Q8 reuse counters and fused gate/up/down roles |
| Certified-artifact hard fail | Present only for the new CPU prefill include | Missing from decode and prefill includes | Missing from decode, prefill, and grouped-MoE includes | Emit and require a provenance-bound certified marker for every production generated table; an uncertified table must not compile into a usable route |
| Whole verifier-pass integration | 13 discoverable grouped lanes | 17 discoverable grouped lanes | 21 discoverable grouped lanes | Close semantic gaps for stochastic resident sampling, request-batched recurrent state, explicit all-format/runtime-M GEMM discovery, attention formats, and MoE codegroups; require production route counters in every applicable cell |

The common refresh wrapper and CI/precommit integration gate MUST enumerate this
matrix by surface and backend. A green all-format arithmetic test does not
substitute for a profiler-backed economy certificate, and a certified GEMM
table does not substitute for byte equality of the remaining verifier stages.

### 1.4 CPU M1 dependency and collection checkpoint on 2026-07-18

Exact CPU-prefill profiler completion exposed a legitimate dependency rather
than a timeout problem. The profiler harness launches the production `Auto`
decode oracle before the target prefill launch. The checked-in CPU M1 include
is deliberately uncertified and fails closed, so profiler requests added after
the earlier exact-v4 transaction could not reach their target. Replacing that
route with `FrozenSerialOracle` would profile a diagnostic path and is not an
acceptable workaround.

CPU M1 is now a first-class common-policy surface. Its trainer measures five
forceable N-chunk-grid candidates (`NBC=1,2,4,8,16`) for all 21 source formats,
three build/runtime ISA regimes, serial-full-K and serial-K-part arithmetic
bundles, and shared/multi-input production launchers. Every admitted row must
match the frozen serial oracle byte-for-byte, repeat byte-for-byte, and publish
the requested route. The compiler emits exact overlays before generic
aspect/work rules and certifies totality across every runtime codebook, bundle,
ISA/thread surface, and legal geometry.

The refresh wrapper now supports `--stop-after-cpu-decode --install`. A complete
transaction collects development and sealed evidence, gathers one isolated
Linux-perf record per physical candidate launch, certifies and installs the M1
include, rebuilds both CPU trainers, and stops before grouped collection. All
M1 timing cells are atomic and resumable; `--cpu-batch-limit N` checkpoints
after N pending two-socket MPMD batches.

A production-size smoke found that canonical kernels were correctly bound to
all 28 physical cores per socket, but raw test-weight synthesis spent most of
its setup time in one scalar normal-distribution stream. Source formats now
prepare concurrently before measurement while retaining each factory's exact
seed and byte stream. Packing and candidate timing remain sequential by format,
so this shortens corpus setup without changing or contending canonical timing.

After M1 installation, the existing CPU-prefill profiler transaction is
extended with `profiler_evidence emit-missing-requests`, resumed with failure
retry, and composed by exact physical launch identity. Canonical timing is not
recollected. The profiler-informed/ablated diagnostic then reruns from the same
immutable corpus before grouped-verifier timing and profiling begin.

An isolated decode calibration on 2026-07-18 also closed two profiler-harness
gaps: CPU M1 observations now resolve through their first-class candidate
registry, and the M1 trainer publishes the same exact one-launch attestation as
the grouped and prefill trainers. Five Q4_0 `152064x5120` M1 candidates were
profiled in three fresh processes. Canonical latency and isolated wall/cycle
rank agreed (`Spearman rho=0.9` over the five-point diagnostic), while
instructions repeated with 0.16% mean CV and correctly ranked the slower
coarse-NBC schedules.

The grouped diagnostic profiled Pairwise and WideRows at `M={2,4,8,16}` on
both `17408x5120` and `152064x5120`, with three fresh processes per geometry.
All 42 grouped target launches were byte-equal to the immutable serial oracle,
repeat-equal, route-authenticated, and isolated one launch per perf report. The
large geometry showed the Pairwise M=16 launch doing 7.9 times the M=2
instructions in 2.0 times the wall duration while LLC misses remained roughly
constant. At equal M, WideRows used about 6% more instructions and 47% more L1
loads and was 8-10% slower despite fewer cache misses. Therefore profiler-model
v12 keeps both per-expected-MiB and per-million-MAC decode features, adds
runtime decode arithmetic intensity, and gives GPU decode both GB/s and GOP/s
views. This is a feature-only refit change; the authenticated raw counter corpus
remains reusable.

Because the production M1 policy was still deliberately uncertified during
this diagnostic, only the out-of-band serial comparison used
`FrozenSerialOracle`. The option is rejected unless an exact isolated profiler
transaction is active, never enters the measured grouped interval, and cannot
produce installable evidence. Production collection continues to require the
certified `Auto` M1 route described above.

### 1.5 CPU M1 fit and scorer checkpoint on 2026-07-19

The CPU M1 evidence surface is now complete: 119,448 canonical observations
have one-to-one exact profiler requests, evidence records, and compact timing
witnesses, with zero failed, unsupported, or missing physical launch IDs. The
immutable timing and profiler payloads live under
`benchmark_results/native_vnni_dispatch/20260718-cpu-m1-full-v2/`; fit retries
reuse those artifacts and MUST NOT launch candidate kernels or Linux `perf`
again.

GPU policy fitting now uses ABI v8 with eight 64-bit point-mask words (512
training and 512 held-out points), compact virtual expanded candidates, and one
persistent scorer lane per physical GPU. Exact positioned-token hashes are
computed by 32-lane subgroups, but every collision still receives a complete
canonical token comparison. The ordered FP64 mean remains one left-to-right
thread; order-only p95 work runs in subgroup finishing kernels with reversible
integer order keys. CUDA and ROCm all-depth 444-point integration suites match
the serial Python oracle byte-for-byte through the real fused production path.

Captured replay owns persistent matrices, streams, graph executables, and
grow-only scratch. Focused CUDA and ROCm traces show no allocation,
deallocation, scratch growth, recapture, or intermediate synchronization in a
stable replay. Nsight reports zero local-memory spilling; rocprofiler reports
zero scratch/private bytes for the child scorer, compact objective finisher,
deduplication, and top-k kernels. A shared-atomic byte-radix p95 experiment was
rejected after exact tests passed because it slowed the representative CUDA
child scorer by about 24%; the retained bounded subgroup upper-tail selector is
both exact and faster.

### 1.6 Comprehensive overlays and best-effort promotion on 2026-07-20

Exact overlays are additive to generic learned dispatch, never a substitute for
it. The shared shape manifest currently resolves 86 production decode/grouped
geometries. Every backend measurement plan forms the Cartesian product of those
geometries, all 21 runtime formats, eager and graph-captured execution, and the
canonical decode/grouped M inventory. Ordinary prefill resolves the 75
non-LM-head production geometries that the prefill graph can actually invoke
and measures each at all seven canonical prefill M buckets. Generic rules remain
mandatory and prove total dispatch for every positive M and every supported
N/K geometry outside those finite exact anchors.

The turnkey driver accepts `--maximum-p95-regret-percent` and
`--minimum-passing-domain-percent`. Both values are frozen into development
metadata, the sealed certificate, and the installable policy artifact, so a
policy cannot be certified under one criterion and installed under another.
The normal defaults remain p95 regret below 5% in at least 95% of domains. An
operator may explicitly set the passing-domain percentage lower, including zero
for a best-effort installation, while every miss remains visible as a typed
performance exception. This affects only the performance quota: missing generic
coverage, empty or unexercised trees, incomplete profiler evidence, ambiguous
dispatch, byte mismatches, and M/N*K/codebook totality failures remain hard
installation failures.

`scripts/train_native_vnni_dispatch.sh --backend all` runs the CPU decode and
grouped transaction, CPU prefill transaction, CUDA transaction, and ROCm
transaction as independent resumable corpus generations. One backend failure
does not discard another backend's evidence, but the aggregate command exits
nonzero until every requested transaction succeeds.

## 2. Authority, Applicability, and Non-Goals

### 2.1 Authority

This policy is normative for NativeVNNI small-M dispatch training, generation,
runtime selection, and table installation after it is accepted. It refines the
grouped-verifier requirements in Phase 9.8 of the parent MTP plan.

Where older NativeVNNI documents accept cosine similarity, relative L2, KL
divergence, or nonzero maximum absolute error for a grouped verifier kernel,
this policy takes precedence. Those metrics remain useful diagnostics and
model-level checks, but they cannot waive byte equality at the grouped-kernel
publication boundary.

Historical background, not normative for verifier eligibility:

- [NativeVNNI Project Plan](../2026-02/NATIVE_VNNI_PROJECT_PLAN.md)
- [NativeVNNI GEMV Expansion](../2026-03/NATIVE_VNNI_GEMV_EXPANSION.md)
- [NativeVNNI GEMM Design](../2026-03/NATIVE_VNNI_GEMM_DESIGN.md)
- [CUDA Dual-Format Quantised GEMM Plan](../2026-03/CUDA_DUAL_FORMAT_QUANTISED_GEMM_PROJECT_PLAN.md)
- [ROCm Dual-Format Prefill GEMM Plan](../2026-02/ROCM_DUAL_FORMAT_PREFILL_GEMM_PROJECT_PLAN.md)

### 2.2 Initial applicability

The first implementation covers:

- CPU, CUDA, and ROCm;
- NativeVNNI prepared quantized weight families and every source-format alias
  advertised by the backend;
- `Fast M=1`, a frozen production serial-M1 oracle, and verifier
  `M=2..16,31` projections;
- single projection, fused projection groups, fused gate/up, fused
  SwiGLU/down, dense, and routed/shared MoE paths that use the same prepared
  kernels;
- exact production shapes and generic nearby shapes; and
- eager and captured execution where graph capture is a production route.

The same framework MUST generate ordinary large-M prefill policies. Prefill
candidates that claim batch invariance MUST use a per-row reduction tree proven
byte-identical to production M1 decode; an atomic or K-tiled schedule with a
different FP32 parenthesization is ineligible even when its approximate error
is small. Prefill uses an independently certified operation surface and cannot
borrow a verifier or GEMV winner without measurements.

The CPU ordinary-prefill kernel implementation is complete. The production M>1 launcher emits
its physical route, N blocking, K tile count, ISA, thread count, and codebook in
perfstats. Eight explicit schedules are registered: one row-chunk-grid formula,
five two-row full-K output-tile N granularities, and Pairwise/Wide grouped
K-part schedules. The latter inherit production M1's K partition and reduce
independent row partials in the same tile order. Separate all-format integration
sweeps force full-K and serial-K-part geometries over runtime `M`, requiring all
FP32 bytes to match independent M1 rows. The production-Auto sweep uses a real
Qwen 1.5B FFN-down serial-K-part shape and independently forces each candidate,
so both generated dispatch and candidate reachability are covered. The
three-regime AVX2/AVX512 development corpus is complete. The fresh v2 sealed
policy and production-table installation are still pending.

Production collection now uses deterministic covering-plan v7 instead of the
full source-format/shape/M/ISA Cartesian product. Before production-route
closure, the generic covering array retains 1,583 runtime-codebook cells and
1,829 source-alias cells. The three C++ route manifests expand that to 1,820
runtime cells, 2,107 source-alias cells, and 653 grouped process jobs: 529 for
AVX2-build/AVX2-runtime, 525 for AVX512-build/AVX2-runtime, and 766 runtime cells
for AVX512-build/AVX512-runtime. This is a 75.8% reduction from 8,694 Cartesian
source cells while still measuring every runtime codebook/M/aspect dispatch
domain in every ISA regime, every production geometry in every ISA regime,
every legal shape/M tier, and every source alias on exactly the same cells as
its normalized runtime codebook.

Version 7 partitions the 653 jobs into contiguous groups whose members have
identical ordered M inventories. Each group is paired only within itself; an
odd final member runs in a one-rank MPI world. This is a correctness property
of the measurement protocol, not merely a scheduling heuristic, because
per-round MPI coordination would deadlock or compare different load phases if
ranks visited different M values.

Every codebook/M/aspect/ISA domain also includes its minimum-N*K production
geometry. The v4 feature-interaction cover could select every witness from one
larger model-size group; grouped cross-validation then correctly refused to
certify extrapolation toward an unmeasured smaller shape. The production total-
policy gate exposed this as three missing AVX2 Q4_0 decisions for the 896x896
attention projection at M=64, 2048, and 4096. The systematic lower-boundary
anchors add 161 runtime cells but only 2.65% estimated GEMM work, and they keep
the same 323 two-socket launch count after source-record consolidation.

The focused serial-K-part gate covers every one of the 21 source formats. AVX2
admits only the physical Pairwise candidate. The route-aware production plan
exact-closes the finite AVX512 serial-K-part family because generic three-shape
cross-validation is not meaningful for its two production geometries. It
measured both Pairwise and WideRows for every forceable cell with zero byte
mismatches; WideRows won all 252 cells. Alias collapse, dispatch-ID validation,
standalone C++20 compilation, and real production-Auto route tests all pass.

CPU corpus collection must use the machine's physical sockets rather than
leaving standalone OpenMP placement implicit. The refresh driver batches
distinct jobs with identical M inventories into one MPMD MPI launch and maps
one rank to each socket. Adjacent jobs are ordered by calibrated work so the two
ranks have similar phase durations. Each rank owns different CSV and raw-timing
paths. This halves collection time on the blessed two-socket host without
duplicating cells or allowing candidates from one cell to run concurrently on
the same memory domain. `OMP_PLACES=cores`, close binding, fixed thread count,
disabled dynamic/nested teams, and cross-rank complete-round coordination are
part of the authenticated measurement contract. GPU acceleration starts only
after this CPU timing corpus is complete, when the common learner scores
policy-tree leaves.

GPU corpus timing uses disjoint format shards across homogeneous devices.
`--cuda-measurement-lanes N` and `--rocm-measurement-lanes N` validate that the
first `N` visible devices share one architecture, partition the source-format
inventory exactly once by round robin, bind one process to each physical GPU,
and merge per-lane aggregate and raw-timing CSVs only after every lane exits
successfully. A lane failure leaves the partials diagnostic and prevents policy
generation. CUDA and ROCm backend transactions may run concurrently when they
own physically separate GPU sets; during such an overlap, each transaction
MUST set `--policy-accelerators` to its own backend so policy fitting cannot
consume or perturb the other backend's canonical timing devices. Isolated
profiler replay may overlap the other backend, but never canonical timing on
the same physical GPU.

The broad shape manifest is a runtime support and feature inventory, not a
license to time every development refinement on every backend and format.
`native_vnni_gpu_measurement_plan_v1.json` names a reviewed 64-shape common
base and declaratively appends every canonical production geometry that the
base does not already contain. The current resolved plan therefore contains
111 common development shapes, including all 86 production entries and a
deterministic feature-diverse certification cover. Those shapes are measured
for both Fast M1 and grouped verifier surfaces over all 21 source formats. The 141
`fast-m1-cv-refinement-v5` geometries are measured only for CUDA, `Q4_0`, Fast
M1, because that is the physical surface whose held-out misses motivated them;
they are shape-sharded across both CUDA devices. Fast and verifier sealed
partitions remain physically separate. The typed plan enumerates the exact
`(contract, M, shape, source format, execution mode)` installation inventory,
and both backend analyzers reject either missing or unexpected surfaces.

A stopped 10m24s calibration completed 154 Q4 Fast-M1 shapes on each RTX 3090
and 217 Q4 Fast-M1 shapes on each MI50. Projecting those measured rates onto the
bounded plan gives approximately 70-85 minutes for CUDA and 25-40 minutes for
ROCm, including common, scoped, sealed, and grouped-verifier timing phases but
excluding unusually expensive profiler replay. These estimates are evidence
for starting the replacement transaction, not completion claims.

Long CPU collections must be format-sharded and checkpointed rather than run as
one fragile transaction. `--cpu-format-shards` gives each shape/format/ISA cell
its own process and output pair. The refresh driver writes aggregate and raw
timing CSVs to `.inprogress` paths and publishes both with atomic renames only
after the complete MPMD launch succeeds. `--cpu-batch-limit N` stops cleanly
after at most `N` pending two-socket launches; repeat the identical command with
`--resume-cpu-partials` and the same output directory to skip already published
cells. A bounded checkpoint never invokes the learner or installs a policy;
generation begins only after the entire required corpus is present.

The complete timing, isolated-profiler, policy-fit, validation, and optional
installation transaction has a reported target of two wall-clock hours per
backend. This is a planning and observability target, not a destructive timeout:
an overrun is printed as `TARGET MISSED`, while atomic partials remain resumable
and a useful run is allowed to finish. Corpus cardinality, adaptive timing
budgets, and profiler request selection MUST be designed to meet the target on
the blessed machine; repeatedly exceeding it requires redesign rather than a
larger nominal budget.

Installable CPU collection also requires every cpufreq policy to use the
`performance` governor. The wrapper fails before collection if any policy uses
`schedutil`, `ondemand`, or another dynamic governor; it does not silently alter
host state. Turbo limits remain unchanged so the corpus represents sustained
production behavior rather than a forced base-clock regime.

The original v3 work-only projection estimated 65 minutes for 312 timing
batches. Canonical v17 reached 92 published batches in 66 minutes and then
correctly aborted on an unstable `IQ1_M`, `M=256`, AVX512-build/AVX2-runtime
cell. That run demonstrated that work-only timing did not model adaptive
stability cost and is retained as diagnostic evidence, not as a valid duration
claim. Route-certified v26 migrated 611 still-valid atomic jobs from v24 and
collected the remaining 42 jobs in 21 paired MPMD launches in approximately
20-25 minutes. This is useful incremental evidence, not a clean end-to-end
duration measurement. A from-scratch v7 collection remains governed by the
two-hour backend target and must be timed before this section claims collection
headroom.

The historical isolated-profiler transaction selected 192 bounded
representative launches from the complete timing corpus. Seventy-four required
requests completed, 118 were explicitly unsupported or nonforceable, and none
failed or went missing. That transaction remains authenticated historical
evidence, but exact-point schema v4 makes it non-admissible for a current fit:
bounded representative selection cannot characterize unsampled M/N/K points.

The first learner-v17 round-11 fit is retained as a non-installable diagnostic.
Its 48-request profiler sidecar covered only `NativeVNNI_cpu_CB0`; the old
physical-candidate key then incorrectly lent those Q4_0 counters to every other
prepared codebook. The corrected request-v3 identity includes prepared family,
packing ABI, and runtime codebook. Re-emitting from the unchanged 94,208-row
round-11 timing corpus produced 864 obligations across 18 prepared codebooks
and three build/runtime regimes: 396 required Linux `perf` launches completed,
468 unreachable candidates were explicitly terminal, and none failed or went
missing. The two socket lanes retained exactly 198 unique required requests
each on disjoint CPU masks. This discrete profiler enrichment took minutes and
did not recollect or mutate canonical timing.

The request manifest, raw reports, evidence manifest, and source-observation
digest are fit inputs, not disposable intermediates. A later fit may rebind an
authenticated descriptor only when prepared family, packing ABI, runtime
codebook, architecture, operation, bundle, and effective physical candidate
all match. Missing codebook-local evidence is a hard failure.

The corrected six-accelerator refit emits 1,380 provisional generic rules and
leaves 31 domains unpromoted. Its matched timing-only ablation emits 1,295
rules and leaves 46 domains unpromoted. The corrected profiler features promote
15 domains and introduce zero failures relative to that admissible baseline;
98 of 756 domains select a profiler-weighted model. This proves the profiler
path is active and useful without granting installation: all 31 residual
domains fail both the timing-only and profiler-informed tournaments.

A full 32-leaf capacity experiment produces byte-identical exact entries,
generic rules, unpromoted domains, selected CV models, and promotion diagnostics
to the normal 16-leaf search. Only seven planner-only competitive-edge unions
grow. Tree capacity is therefore not the residual limitation, and 16 leaves
remains the production default. The reviewed 32-leaf CLI remains available for
explicit diagnostics and records its capacity in policy metadata.

The shared shape inventory is bounded by the largest projection we intend to
serve, not by an unbounded synthetic lattice. Manifest v5 keeps the runtime
support ceiling at `1,271,398,400` weight elements, matching the real
`248320x5120` `Qwen36_LM_Head`. Its separate canonical CPU measurement ceiling
is `778,567,680`, exactly Qwen2.5 32B's `152064x5120` LM head. CPU production
collection includes explicit 32B attention, QKV, FFN-up, FFN-down, and LM-head
overlays but excludes larger timing jobs; larger runtime geometries remain in
the manifest and use the certified generic policy. Six synthetic shapes above
the runtime ceiling were removed, and the loaders reject future entries beyond
that supported bound.

Grouped verifier collection remains `M=2..16,31`, the speculative-depth
surface. CPU ordinary prefill v12 measures
`M={64,128,256,512}` for below-14B geometries and `M={64,128}` for geometries
owned by 14B-or-larger models. CUDA and ROCm
retain `M={64,256,1024,2048,4096,8192,16384}`. These are measurement ceilings,
not runtime limits: generic CPU policy maps every larger positive M to the
deepest trained bucket, and totality regressions exercise values beyond 512.
Grouped verification and ordinary prefill remain intentionally different
semantic surfaces.

The ordinary CPU prefill measurement surface is executable in
`native_vnni_dispatch.prefill_matrix`. It currently contains 75 real,
non-LM-head production projections and 224 shape/depth cells. The strong
`TrainerCsv_StrongPrefill_AllFormats` test measures all thirteen explicit registry
requests through `gemm_native_vnni_preq`, records normalized physical routes,
and compares selected output rows from two launches with serial M1 rows.
It accepts arbitrary `M>1`; the checked-in matrix, rather than the kernel or
adapter, supplies the economical model-tier limits. Its isolated Linux `perf`
launch uses the same FIFO acknowledgement protocol as verifier training and is
outside canonical timing.

Canonical performance collection uses the deterministic
`boundary-quartile-sentinel-v1` serial oracle: up to sixteen first, quartile,
and final rows are replayed through production M1, while every grouped candidate
still executes at the real requested M. Sentinel native bytes must equal M1 and
the adapter recomputes the expected sentinel count. This limits oracle cost
without weakening the exhaustive all-format grouped integration sweep, which
remains the authority that checks every output byte for every format and depth.

Ordinary CPU prefill uses elapsed-and-convergence-controlled timing rather than
blindly replaying every expensive GEMM 30 times. Warmup and canonical timing
both use independently shuffled complete rounds. Every forceable candidate
launches in every round until the final active candidate stops. Timing protocol
v13 retains the v12 source-preconditioning rule: each source-format process's
first M cell gives every
forceable candidate at least five warm launches and one second of measured
complete-round preconditioning. Later M cells retain the same two correctness
launches and derive a candidate-specific transition budget from the first
production-route latency probe:
`max(100 ms, min(2 s, probe_latency * 60))`. This preserves at least sixty
launch-equivalents of transition warmup without charging a blanket two seconds
to cheap candidates. Source policy, M, probe, floor, multiplier, ceiling,
effective budget, and achieved candidate duration are authenticated on every
row. During timing, a retired
candidate similarly becomes an unmeasured pacing launch, while each active
candidate receives one sample before any active candidate receives its next
sample.

This complete-round load envelope is mandatory. Independently retiring timing
candidates changed cache and CPU frequency cadence mid-series and manufactured
7-10% drift in a Q4_0
`1.5B_AttnOut`, `M=4096` row-chunk route that was independently stable on both
sockets. Holding the cadence fixed reduced the reproduced socket-1 drift to
0.46% under the normal adaptive protocol. Independently retiring warmup
candidates later produced a 2.011% hard-ceiling failure for Q4_0
`3B_AttnOut`, `M=4096`, `nbc1`; equal complete warmup rounds reduced the same
socket's drift to 1.075% and convergence at 30 samples. Protocol v11 therefore
requires complete-round pacing in both phases and source-level preconditioning.
The additional per-M transition budget is a regression for an IQ1_M
`0.5B_FFN_Up`, `M=256`, AVX512-build/AVX2-runtime cell whose first 15 samples
remained about 5% above its later steady windows after only three transition
launches; it reached the old 90-sample ceiling at 2.435% median drift.
Process-local pacing is still insufficient for a two-socket MPMD collector. In
v17 one rank advanced through larger M phases while its peer extended an
unstable M256 series. Both forceable peer candidates showed the same mid-run
latency excursion; `nbc1` reached 180 samples with 9.552% half-history drift,
even though its final 30 and 60 samples were stable. Protocol v11 fixes the
source rather than discarding a convenient tail: every warmup and timing round
ends in an MPI active-rank reduction, and a locally complete rank continues
unmeasured pacing rounds until every rank completes the same M phase.

The aggregate CSV authenticates the coordination policy, MPI world size/rank,
global warmup-round count, and global timing-round count. The CPU prefill
adapter independently groups forceable rows by measurement cell, requires
candidate-local and global round provenance to agree, and rejects installable
multi-rank evidence without `mpi-complete-round-v1`; a protocol label alone is
not accepted as proof of this invariant.

Canonical v18 then published 79 coordinated jobs before the first M64-only
source group exposed a different control-plane bug: two sub-100-microsecond
candidates needed more than the fixed 10,000-round safety ceiling to accumulate
one second of measured warmup. Both ranks failed at the count guard with no
numeric or stability failure. Protocol v11 removes that obsolete unit mismatch.
Measured per-candidate duration remains the warmup requirement. The exact
failed `IQ1_M`/`IQ2_XS` AVX512-native pair subsequently completed 19,381
coordinated rounds: every candidate accumulated at least one measured second,
the complete phase took 3.082 seconds, all native bytes matched, and the
production adapter accepted the resulting v11 diagnostic evidence.

The first v19 7B/M16384 job then proved that a 30-second complete-*phase*
watchdog was also the wrong scope: one legitimate complete round took about
15.8 seconds and the minimum three rounds naturally exceeded 30 seconds.
Protocol v11 applies the 30-second watchdog independently to each complete
round, requires every measured candidate launch to return finite positive
duration, and records total warmup wall time separately. The CSV authenticates
total wall duration, maximum round duration, and round-watchdog budget; the
adapter rejects an elapsed round or an over-permissive production watchdog.
The exact 7B FFN-up regression completed three M16384 warmup rounds in 47.079
seconds total; its slowest complete round was 15.719 seconds, every candidate
converged byte-exactly, and the production adapter accepted the v11 evidence.

Canonical v20 passed the prior coordinated failures, then exposed a longer
same-cell transition in `IQ3_S`, `0.5B_FFN_Up`, AVX2-build/AVX2-runtime,
`M=4096`, `nbc4`. The fixed 100 ms transition floor produced only four
complete rounds. Its first roughly 45 timing launches retained the prior-M
latency state, so the full 180-sample history failed at 2.437% median drift even
though its tail was stable. Protocol v12 fixes the warmup source instead of
discarding the transient from timing. The exact two-rank v20 progression through
`M={256,1024,2048,4096}` was then replayed 20 times. The target received 59-61
complete warmup rounds and a two-second effective budget on every run, remained
byte/repeat exact, and converged with worst drift 1.796% and at most 36 samples.
The production adapter accepted the resulting v12 formula and timing evidence.

Canonical v21 passed that IQ3 transition and then found a different cheap-
candidate settling failure at `IQ2_XS`, `1.5B_FFN_Dn`, M=64, AVX512 native.
The pairwise route reached the old 180-sample guard after only 202.9 ms of
measured kernel time, while the complete acquisition history was still settling
from about 1,269 us toward 1,077 us. Protocol v13 makes the hard rejection guard
conjunctive: an unstable candidate must have both at least 180 samples and at
least two seconds of measured kernel time. Stable candidates still retire at
the ordinary 30-sample floor. A forced zero-drift regression ran 1,797 samples,
accumulated 2,000,527 us, then rejected collectively with byte/repeat equality
intact; this proves that sample count alone can no longer terminate timing.

Timing rejection is also a coordinated MPI decision. Aggregate and sidecar CSVs
are flushed first, every rank participates in one failure reduction, and any
rejected peer aborts the MPMD communicator before another rank can enter the
next-M barrier or MPI finalization. An asymmetric regression with an impossible
zero-drift gate on one rank and a normal gate on its peer exited with the
expected diagnostic in under one second rather than hanging.

A candidate receives at least five timing samples and may stop before the
30-sample stable target only after accumulating at least two seconds of measured
kernel time and passing a 2% first-half versus second-half median-drift gate.
Cheap candidates still reach 30 samples quickly; a candidate still drifting at
30 continues adaptively until it either converges or reaches both the
180-sample guard and two-second measured-time guard, and it is not promotable
unless the drift gate passes. The sample count may therefore exceed 180. After
the v17 failure, the exact two-rank
`IQ1_M`/`IQ1_S`, AVX512-build/AVX2-runtime versus AVX2-build/AVX2-runtime pair
was replayed 20 times over `M={256,1024,2048,4096,8192,16384}`. All 480
forceable rows converged with zero byte/repeat mismatches, worst drift 1.997%,
and at most 69 samples. Aggregate rows and acquisition-order sidecars are flushed
before the hard stability assertion. Failed checkpoint files remain marked
`.inprogress`, preserving diagnostics without making them publishable.
Warmup rounds/duration, deterministic ordering seed, minimum/maximum sample
counts, elapsed budget, stability limit, observed drift, stop reason, and
measured duration are part of the authenticated collection contract. The
retained sidecar preserves acquisition order so the adapter independently
recomputes drift, while robust aggregates and their digest continue to use a
sorted copy.

### 2.3 Non-goals

This policy does not require:

- the same kernel candidate IDs on different backends;
- identical performance choices across different hardware architectures;
- one universal policy trained across CUDA SM, ROCm GFX, and CPU
  microarchitectures;
- replacing backend-specific kernel implementations;
- accepting serial row replay as an economical grouped verifier
  implementation; or
- making a generic performance claim for a bucket with insufficient independent
  sealed evidence.

The commonality is the contract, data, learner, evaluation, and promotion
mechanism. Candidate enumeration and C++ encoding remain backend adapters.

## 3. Baseline Audit Before Migration

The table below records the implementation at project start. Section 1.1 is the
authoritative current progress snapshot; this baseline is retained to explain
why each migration is required.

| Backend | Existing behavior | Policy gap |
|---|---|---|
| CUDA | A 3%-set-cover plus `log2(M,N,K)` decision tree is followed by a separate exact-overlay/aspect analyzer. | The tree evaluates its training corpus, not held-out shapes. The overlay analyzer scores newly inferred aspect/work rules while `--base-include` preserves a different tree fallback, so the scored generic policy can differ from the emitted policy. Trainer correctness is not a verifier byte certificate. |
| ROCm | Exact winners are emitted above aspect/work rules. | The trainer/generator eligibility collapses the candidate matrix to exact winners, then optimizes modal winner labels. Its reported metric is in-sample exact-label hit rate, not latency regret, and its trainer eligibility permits approximate output despite stronger byte-exact integration tests. |
| CPU | A generated exact `(codebook,M,N,K)` table selects Pairwise or WideRows. | There is no generic learner. `Auto` measurements can be labeled as a policy they did not execute, fused/MoE paths do not all consult the same selector, and trainer/generator eligibility is tolerance-based despite stronger byte-exact integration tests. |

The refresh wrapper currently validates generated codebook IDs and then copies
each `.inc` independently. It does not prove byte eligibility, candidate
reachability, held-out generic regret, M1-oracle freshness, or transactional
all-backend installation.

Primary audited surfaces are listed in Appendix D.

## 4. Terminology

**Semantic contract**

The numerical and execution contract under which a candidate is selected.
This policy defines `Fast` and `VerifierSerialM1Bitwise`.

**Source format**

The externally loaded tensor format, such as a GGUF Q/K/IQ format.

**Prepared family**

The concrete runtime weight representation consumed by a kernel. Multiple
source formats may map to one prepared family or dispatch codebook.

**Runtime alias group**

All source formats that become indistinguishable at the runtime dispatch key.
A policy cannot choose different candidates for members of the same alias
group.

**Candidate ID**

A stable backend-specific name for a forceable kernel configuration.
`Auto` is not a candidate ID.

**Effective candidate ID**

The configuration actually launched after contract normalization. Nominal
candidate IDs that normalize to the same launch share one effective ID for
training purposes.

**Arithmetic fingerprint**

A stable description of activation quantization, weight decode, accumulator
type, K partition, partial-sum layout, reduction order, epilogue order, and
atomic/non-atomic behavior for an effective launch.

**Serial M1 oracle**

The actual production `M=1` decode route for the backend, prepared family,
operation kind, and shape. It is not a separate reference implementation.

**Alias-robust runtime exact winner**

The eligible runtime-representable candidate selected for a measured exact key
by minimizing worst-alias and required-execution-mode normalized latency under
one semantic contract. It need not be the fastest candidate for every alias.

**Exact overlay**

A generated entry keyed by the full exact runtime shape and placed above the
generic policy.

**Generic rule**

A generated candidate selection keyed by prepared family/codebook, semantic
contract, `M`, aspect bucket, and a bounded work-size segment.

**Certified floor**

A genuinely grouped, contract-eligible candidate proven safe for the declared
domain. It is not an unrestricted fast fallback and is not hidden row replay.

**Performance regret**

The relative latency loss of a selected candidate against the alias-robust
runtime exact winner for the same semantic contract and runtime key.

**Policy ABI**

The versioned meaning of dispatch keys, candidate IDs, arithmetic
fingerprints, generated fields, and lookup order.

## 5. Non-Negotiable Invariants

### 5.1 Verifier byte equality

For every supported backend, prepared family, operation kind, source-format
alias, shape, deterministic payload seed, and `M in {2..16,31}`:

```text
grouped_output[M,N]
    == bytewise_concat(
           production_serial_M1(row_0),
           ...,
           production_serial_M1(row_M-1))
```

The comparison MUST cover every logical `M x N` FP32 result element row by row
using `memcmp` or an equivalent exact byte comparison. Leading-dimension
padding is compared only when it is initialized and explicitly part of the
operation contract. A one-ULP or signed-zero difference is failure.
Correctness fixtures MUST reject unexpected NaNs rather than treating matching
NaN payload bytes as useful proof.

Cosine, relative L2, symmetric KL, maximum absolute error, and sampled-token
equality MUST NOT be used as candidate eligibility substitutes.

The proof has two layers:

1. a primitive kernel proof starting from identical prepared weights and
   identical prequantized activation bytes; and
2. a public-operation proof starting from the same input tensor and covering
   activation quantization, alpha/beta behavior, bias, and every fused
   epilogue inside the declared `operation_kind`.

Both layers are required where the production operation owns those steps.

### 5.2 Stable oracle

The production serial M1 oracle MUST be byte-stable across repeated hot runs
and the supported first-use/cold preparation route. An unstable oracle makes
the affected verifier domain uncertifiable.

### 5.3 Contract isolation

`Fast` and `VerifierSerialM1Bitwise` MUST resolve independently. An environment
override, atomic reduction option, ordinary-prefill policy, or ambient tuning
flag MUST NOT influence grouped verifier resolution unless that exact effective
candidate is separately verifier-certified. There is no relaxed `Fast M>1`
publication contract: grouped decode rows are verifier rows.

### 5.4 Exact and generic parity of treatment

Every backend MUST emit both exact overlays and generic rules through the same
common policy IR. CPU MUST NOT remain an exact-only special case.

### 5.5 Exact precedence

Runtime lookup MUST test an exact overlay before a generic rule. Sealed
certification MUST bypass exact overlays so exact entries cannot mask a broken
generic policy.

Exact overlays are additive optimizations for explicitly swept runtime keys.
They MUST NOT satisfy generic coverage, rescue an unpromoted domain, or make a
partial generic policy installable. Every backend, format/codebook, semantic
contract, arithmetic bundle, ISA/device class, execution mode, and canonical M
bucket MUST own a certified generic rule partition even when every known model
shape also has an exact overlay.

The generated hot selector MUST remain model-independent and mechanically
simple. Backend, architecture/ISA, semantic contract, arithmetic bundle, and
execution mode select the outer policy table. Within that table runtime lookup
uses only `(runtime codebook, N, K, M)`: a packed exact-overlay lookup runs
first, followed by generic geometry/aspect/work predicates on a miss. Model and
projection names are evidence provenance and generated comments only; they
MUST NOT become runtime predicates.

The canonical production catalog includes every officially released Qwen 3.5
and Qwen 3.6 geometry: Qwen 3.5 dense 0.8B, 2B, 4B, 9B, and 27B; Qwen 3.5 MoE
35B-A3B, 122B-A10B, and 397B-A17B; Qwen 3.6 dense 27B; and Qwen 3.6 MoE
35B-A3B. Releases with identical tensor dimensions share geometry evidence
rather than creating model-name dispatch aliases. The derived exact-overlay
inventory covers attention, GDN, dense/expert FFN, mirrored MTP-head, and LM
head projections; FP32 router tensors remain outside NativeVNNI dispatch.

The generic partition MUST be structurally total over every legal positive
`(N,K)` geometry. Flattened tree leaves must reconstruct one root, each
internal threshold must have complementary `<=` and `>` branches, and every
input must reach exactly one leaf. Threshold features include N, K, N*K,
exact-rational aspect ratio, tile counts, K groups, tail utilization, and
backend occupancy/wave quantities where applicable. Terminal leaves therefore
dispatch unseen shapes below, between, and above the fitted geometries; a
missing rule for any unseen legal shape or work size is a publication failure.
Canonical runtime M bucketing must likewise map every legal unseen M to a
certified generic domain rather than rejecting it or consulting an exact table.

### 5.6 Held-out p95 below 5%

Every promoted generic bucket MUST have 100% required sealed coverage and a
nearest-rank p95 sealed worst-surface regret strictly below 5.0%. The p95 of
the conservative simultaneous per-cell 95% upper regret bounds MUST also be
strictly below 5.0%. Mean regret is insufficient. Maximum observed regret and
maximum simultaneous upper regret MUST be reported as diagnostics, but an
isolated tail cell above 5% is not by itself an installation failure.

### 5.7 Runtime-surface aliasing

Candidates MUST be evaluated on the dispatch surface the runtime can express.
If several source formats share one codebook/prepared-family key, a candidate
must pass correctness for every alias and performance selection must use the
worst alias-relative result.

### 5.8 No quiet fallback

If no grouped candidate satisfies the verifier contract, the capability is
unsupported and MUST fail closed before publication. The runtime MUST NOT fall
through to `Fast`, silently replay rows, or claim generic performance coverage.

### 5.9 Measured route equals emitted route

Every candidate selected by a generated table MUST be forceable in the trainer
and reachable in the production launcher. Generator acceptance MUST fail if a
candidate cannot round-trip through the backend emitter and resolver.

### 5.10 Reproducibility

A generated artifact MUST identify its corpus, candidate registry, learner,
feature schema, serial-M1 oracle policies, development folds, and sealed split
by stable digest.
A stale or incomplete certificate MUST fail generation or runtime validation.

### 5.11 Profiler isolation and per-candidate coverage

Hardware profiling MUST be a physically separate transaction from canonical
timing. Nsight Compute replay, rocprofiler counter passes, Linux `perf`, static
resource inspection, or any other instrumentation MUST NOT wrap a launch whose
latency enters the common observation corpus. The canonical timing corpus is
sealed first and emits an immutable profiler-request manifest; collectors then
run each requested effective candidate in a separate process with exactly one
target production launch per profiler pass. Tool-managed or collector-managed
counter replay is permitted, but every pass must isolate that same launch.

Every supported candidate observation MUST have its own profiler evidence.
Candidates implemented as a physical pipeline retain one ordered record per
kernel dispatch, such as a K-part producer and ordered reducer; metrics from
different variants or dispatches MUST NOT be averaged into one anonymous row.
Unsupported candidates receive an explicit `candidate_unsupported` record.
Missing tools, failed launches, failed parsing, and unavailable optional
counters are typed states with reasons, never omitted fields. Missing required
metrics or evidence for one supported candidate blocks production evidence
completeness.

## 6. Shared Architecture

### 6.1 Components

The target implementation has eight layers:

1. **Backend candidate registry**: declares forceable candidates, static
   reachability, normalization, arithmetic fingerprinting, and C++ encoding.
2. **Backend trainer adapter**: runs the production prepared-weight path and
   emits common observations for every candidate.
3. **Common corpus validator**: verifies schema, completeness, aliases,
   numerical evidence, timing quality, and oracle freshness.
4. **Common policy compiler**: constructs alias-robust runtime exact winners, grouped development
   CV, frozen generic aspect/work rules, and sealed certification.
5. **Backend emitter**: maps common IR candidate IDs to backend-specific `.inc`
   declarations without changing policy decisions.
6. **Profiler request/evidence layer**: derives immutable per-candidate requests
   from canonical observations, gathers static resources and dynamic counters
   in isolated launches, and validates complete provenance-bound coverage.
7. **Backend profiler collectors**: drive Nsight Compute, rocprofiler-sdk, and
   Linux `perf`, retaining raw tool outputs and one ordered record per physical
   dispatch in a candidate pipeline.
8. **Acceptance driver**: compiles staged artifacts, compares Python/C++
   resolution, runs parity/performance gates, and installs atomically.

The recommended common Python package is:

```text
tests/v2/performance/kernels/native_vnni_dispatch/
    schema.py
    candidate_observation.py
    corpus.py
    exact_oracle.py
    segmented_policy.py
    splits.py
    certification.py
    policy_ir.py
    profiler_evidence.py
    validation.py
    adapters/
        cuda.py
        rocm.py
        cpu.py
```

Existing trainers MAY remain in their backend directories, but their CSVs
must conform to the common schema and their analyzers must delegate policy
fitting and evaluation to this package.

### 6.2 Dispatch key

The logical dispatch key is:

```text
backend
architecture_class
semantic_contract
operation_kind
bundle_signature and ordered projection-N vector
prepared-family/runtime-codebook vector
M, K
```

`source_format` is retained in the corpus for alias correctness and
performance aggregation, but it is not a runtime discriminator unless the
backend actually exposes it in its production dispatch API.

`architecture_class` MUST prevent measurements from incompatible devices from
being pooled. Initial classes SHOULD include:

- CUDA compute capability plus any kernel ABI-relevant device class;
- ROCm GFX target plus wave/kernel ABI-relevant class; and
- CPU ISA tier plus a declared microarchitecture tuning class.

For CPU, thread count, affinity/binding policy, and NUMA placement are part of
the architecture/topology key unless evidence explicitly certifies that one
policy is portable across those values. A table's 5% claim applies only to the
architecture/topology key recorded in its manifest.

Unknown architectures use only an explicitly certified portable policy or are
unsupported; they do not consume a table trained for another class by default.

### 6.3 Explicit runtime contract

The target C++ API carries semantics in the dispatch request, not in ambient
backend state:

```cpp
enum class GemmSemanticContract : uint8_t
{
    Fast,
    VerifierSerialM1Bitwise,
};

struct GemmDispatchRequest
{
    GemmSemanticContract contract;
    GemmOperationKind operation;
    int m;
    int k;
    std::span<const GemmProjectionDescriptor> projections;
};

struct GemmProjectionDescriptor
{
    GemmProjectionRole role;
    int n;
    PreparedFamilyId prepared_family;
    NativeVNNICodebookId codebook;
    PackingABI packing;
    GemmEpilogueSignature epilogue;
};

struct ResolvedGemmDispatch
{
    CandidateId candidate;
    EffectiveCandidateId effective_candidate;
    DispatchMatchKind match; // Exact, Generic, CertifiedFloor
    PolicyVersion policy;
    ArithmeticFingerprint arithmetic;
};
```

`bundle_signature` is derived from the ordered projection roles, prepared
families, codebooks, packing ABIs, and epilogues. The exact key also carries
the ordered projection-N vector.

Version 1 generic bundle learning is restricted to homogeneous bundles: every
projection shares K, source format, source codebook, prepared family, runtime
codebook, packing ABI, and a candidate support domain. Its generic features
are:

```text
aggregate_n = sum(N_i)
aspect_ratio = aggregate_n / K
work_items = aggregate_n * K
```

The candidate must be eligible for every projection and for the ordered fused
epilogue as a whole. Alias coverage substitutes the same tested alias across
the complete homogeneous bundle. A bundle mixing source aliases or any other
listed identity is heterogeneous in v1 and must be decomposed into supported
homogeneous sub-bundles or assigned a separately versioned operation policy;
it MUST NOT consume a single-projection or homogeneous generic certificate.

`ITensorGemm::beginVerifierDecodeEquivalentScope()` is the current common
bridge and MAY remain while callers migrate. The final implementation SHOULD
pass `GemmSemanticContract` through the call/plan object so concurrent host
threads and streams cannot observe process-global mode changes.

### 6.4 Canonical format and prepared-family registry

One versioned registry MUST define:

```text
source format and source codebook
backend execution codebook
prepared family and packing ABI
source-format aliases
block size and divisibility constraints
supported semantic contracts and operation kinds
```

The trainer inventory, all-format verifier inventory, backend traits, and
generated-policy validator MUST be derived from or checked against this
registry. A source alias may share a runtime entry only after every alias
passes byte eligibility for the same prepared-family packing ABI.

The audit found a concrete inventory divergence that this registry must close:
`tests/v2/utils/QuantizedVerifierFormats.h` advertises source `Q8_K`/codebook
21 and maps GPU execution to codebook 19, while
`tests/v2/performance/kernels/native_vnni_codebooks.py` does not currently list
`Q8_K`. Inventory-consistency tests MUST fail on this class of omission.

## 7. Common Measurement Corpus

### 7.1 Observation granularity

The corpus MUST retain every measured candidate, not just winners. One
aggregate observation represents one:

```text
(run, backend, architecture, contract, operation, bundle signature,
 ordered projection-N vector, execution mode, source format,
 prepared family, M, K, candidate, payload trial set)
```

Raw timing samples MUST be retained as an auditable sidecar artifact for the
lifetime of every accepted bundle. The aggregate row contains robust timing
statistics and a digest of the sorted sample copy. Acquisition order is
normative whenever an adaptive protocol claims clock or thermal convergence;
fixed-count producers may retain their existing sorted sidecars. Whether those
samples live in the repository or durable CI/artifact storage remains an
implementation choice.

### 7.2 Required schema

| Group | Required fields |
|---|---|
| Schema/provenance | `schema_version`, `run_id`, `corpus_id`, `git_revision`, `build_id`, `compiler_id`, `policy_abi`, `learner_version` |
| Hardware | `backend`, `architecture_class`, `device_name`, `driver_runtime`, `threading_or_stream_mode` |
| Runtime key | `semantic_contract`, `operation_kind`, `bundle_signature`, `projection_n_vector`, `source_format`, `source_codebook_id`, `prepared_family_id`, `packing_abi`, `runtime_codebook_id`, `shape_group_id`, `shape_name`, `execution_mode`, `m`, `aggregate_n`, `k` |
| Features | `aspect_ratio`, `aspect_bucket`, `work_items`, `n_tail_class`, `k_tail_class`, `alignment_class` |
| Candidate | `candidate_id`, `effective_candidate_id`, `candidate_family`, `config_json`, `supported`, `graph_capture_ok`, `generic_eligible` |
| Arithmetic | `arithmetic_fingerprint`, `serial_m1_policy_id`, `serial_m1_policy_hash`, `candidate_policy_hash`, `ordered_reduction`, `uses_atomic_reduction` |
| Correctness | `trial_set_hash`, `bitwise_equal`, `repeat_equal`, `mismatch_count`, `first_mismatch_index`, `grouped_output_digest`, `serial_output_digest`, `max_abs`, `relative_l2`, `cosine`, `symmetric_kld` |
| Timing | `warmup_count`, `sample_count`, `min_us`, `median_us`, `p95_us`, `mad_us`, `cv`, `timing_sample_hash`, `effective_bandwidth_gbs`; adaptive raw producers additionally require `warmup_round_count`, `warmup_budget_policy`, `warmup_probe_latency_us`, `warmup_budget_floor_us`, `warmup_latency_multiplier`, `warmup_budget_ceiling_us`, `warmup_budget_us`, `warmup_duration_us`, `global_warmup_round_count`, `warmup_wall_duration_us`, `warmup_max_round_duration_us`, `warmup_round_timeout_us`, `timing_order_seed`, `global_timing_round_count`, `timing_protocol`, `min_sample_count`, `stable_sample_count`, `max_sample_count`, `timing_budget_us`, `timed_duration_us`, `median_stability_limit`, `median_relative_drift`, `timing_converged`, and `timing_stop_reason` |
| Route proof | `forced_route_ok`, `observed_candidate_id`, `route_counter_ok`, `workspace_ok`, `explicit_stream_ok` |

Missing required fields are errors. Empty correctness values are not interpreted
as passing values.

### 7.3 Candidate enumeration rules

- Every measured candidate MUST have a stable explicit ID.
- `Auto`, a previous generated policy, or a heuristic fallback MUST NOT be
  recorded as though it were a concrete candidate.
- Each registry entry MUST declare an arithmetic signature, schedule
  signature, support predicate, required prepared resources, workspace
  requirements, graph-capture status, and runtime launch mapping.
- The trainer MUST force the candidate and confirm the observed route ID.
- Nominal candidates that normalize to one effective launch MUST be collapsed
  before learning.
- A candidate unsupported for a shape remains an explicit unsupported matrix
  cell; it is not silently omitted.
- A candidate missing for a required alias is unavailable for that runtime key.

### 7.4 Shape inventory

The production corpus MUST include:

- every current exact production shape;
- every advertised prepared family and source-format alias;
- `Fast M=1`, plus verifier `M=2..16,31` against the frozen serial-M1
  policy, together for each logical shape;
- nearby unseen dimensions on both sides of aspect boundaries `0.75`, `2.0`,
  and `16.0`;
- log-spaced work sizes within each aspect bucket;
- N and K tile-tail/alignment boundaries;
- fused and unfused operation kinds that have different launcher
  reachability; and
- graph-captured cells for every captured production route.

The inventory SHOULD live in one backend-neutral versioned manifest. Backend
adapters MAY mark a shape/candidate unsupported but MUST NOT silently remove a
required family or M value.

### 7.5 Eager and captured execution modes

`execution_mode` is a required corpus and dispatch axis in policy ABI v2.
Eager and graph-captured routes receive independent exact and generic
decisions because launch gaps and graph replay can change the economical
schedule. Each mode MUST independently pass correctness, reachability,
workspace, repeatability, and regret gates. If an operation has only one
production mode, only that mode is required.

### 7.6 Measurement protocol

Production acceptance measurements MUST:

- use Release builds and the production prepared-weight path;
- prepare/upload/repack weights outside the timed candidate loop;
- use an explicit non-default CUDA/HIP stream and record timing events on that
  stream;
- use the production CPU affinity/OpenMP configuration;
- perform at least 5 warmups and either 30 stable timed samples or an
  operation-specific adaptive floor of at least 5 samples, sufficient measured
  duration, and authenticated median convergence;
- run a broad first-pass candidate sweep in complete sample-interleaved rounds,
  independently shuffling every candidate permutation and retaining its seed;
- use median latency for fitting;
- retain variance and raw samples; and
- run correctness on multiple deterministic payload seeds, including
  structural edge payloads and the production packed representation.

Quick and family-smoke profiles MAY use fewer timings for workflow validation,
but they cannot produce installable artifacts.

#### 7.6.1 GEMV HBM roofline signal

For `Fast M=1` GEMV, the headline efficiency diagnostic is effective prepared-
weight bandwidth divided by an empirically measured sustainable HBM bandwidth
for the same device, clocks, build, and profiling regime. Marketing peak
bandwidth is not an acceptable denominator. Reports MUST retain both useful
prepared-weight bytes and profiler-observed physical DRAM bytes so extra KPAR
partial/workspace traffic, cache effects, and unpack work remain visible rather
than being credited as useful bandwidth.

This roofline fraction is a diagnostic and training feature, not the winner
label. Canonical median latency remains the dispatch objective because small
shapes can be launch-limited, compressed formats perform nontrivial unpack
work, and deterministic K partitioning may trade extra traffic for enough
parallelism to reduce latency. As M increases, weight reuse raises arithmetic
intensity; grouped GEMM reports MUST therefore add effective integer GOPS,
compute speed-of-light, and a byte-accurate arithmetic-intensity estimate
instead of assuming every verifier depth remains HBM-bound.

During development, grouped CV retains every held-out selected/exact decision.
Only a domain whose p95 regret is not below 5% opens paired refinement; within
that domain, each over-budget cell emits a typed request for each source alias.
Shape-resolved formulas MUST become concrete, forceable schedules before the
request reaches a backend trainer. The trainer batches those requests, assigns
each tournament edge a stable request ID, and interleaves one timing window per
candidate in every shuffled sample round. Validated dimensionless paired ratios
replace broad ratios only on the represented development surface, after which
CV is rerun. This loop stops at green, a directly measured over-budget policy,
an evidence conflict, or its explicit iteration limit; it never consults a
sealed row.

After the generic policy is frozen, a separate paired confirmation pass
interleaves its selected candidate and exact reference for each sealed
certification cell without refitting. Confirmation
sampling continues until the one-sided simultaneous 95% confidence bounds are
precise enough for the required p95 regret decision. The p95 of those
conservative per-cell bounds must remain below 5%; maximum regret remains a
reported diagnostic. The bound estimator targets no more than 0.5
percentage-point uncertainty. The implementation may use a bootstrap maximum
statistic, familywise correction, or another recorded method with equivalent
simultaneous coverage. Independent per-cell 95% intervals are insufficient.

A cell that cannot reach the required precision is inconclusive and blocks
promotion; it is not rounded down to a pass.

### 7.7 Isolated hardware-profiler sidecars

After canonical aggregate observations and raw timing samples are immutable,
`profiler_evidence.py` emits a request manifest bound to:

```text
complete observation digest and timing-sample digest
backend, architecture, build, compiler, device, and runtime
contract, operation, source format, prepared family, mode, M, N, and K
candidate ID, effective candidate ID, configuration, schedule, and workspace
candidate-registry digest and profiler metric-set version
```

Current request schema `native-vnni-profiler-request-v4-exact-point` emits one
request for every launchable measured production point. Its physical identity
contains backend, architecture/ISA, operation, bundle, prepared family,
packing ABI, runtime codebook, effective candidate, execution mode, `M`, the
ordered projection-`N` vector, aggregate `N`, and `K`. Two source observations
may share one profiler launch only when all of those fields are identical true
aliases. A shape-resolved policy formula is not another launch and cannot
create a profiler request; its directly measured concrete observation does.

Dynamic profiler values never transfer from a representative anchor to another
work point. Occupancy, IPC, cache behavior, traffic, cycles, and duration from
one `(M,N,K,candidate)` invocation belong only to that invocation. This avoids
both candidate/geometry confounding and the subtler error of treating one
candidate's counters as shape-independent constants. `compose-evidence` and
feature-catalog merge hard-fail on mixed request-schema generations, missing
records, conflicting counters, duplicate non-alias records, or any profiled
launch whose complete physical identity differs from its request.

Historical v3 representative-anchor catalogs remain immutable and readable so
their provenance and old policy decisions can be reproduced. They are not
admissible current fit evidence. The CPU-prefill replay recipe therefore names
one authenticated exact-point primary transaction; additive transactions must
use the same exact-point schema. Replacement requests are derived from existing
canonical timing witnesses, so profiling does not repeat or perturb canonical
latency measurement and never relabels old counters.

The collector launches a dedicated backend trainer process for each request.
All setup, allocation, packing, graph capture, correctness checks, and warmup
occur while profiling is disabled. The profiler is enabled only around one
production candidate launch (or one graph replay) and the required completion
synchronization. CUDA uses `cudaProfilerStart`/`cudaProfilerStop` with Nsight
Compute start-disabled collection and requests only the reviewed metric set,
rather than replaying hundreds of unrelated section metrics. When NVIDIA's
`RmProfilingAdminOnly` is enabled, the authenticated tool command is prefixed
with non-interactive `sudo -E`. CUDA and ROCm profiler requests are coordinated
concurrently across the same disjoint physical-device lanes used for canonical
timing, with at most one profiler process per GPU. ROCm uses
`roctxProfilerResume(0)`/`roctxProfilerPause(0)` with rocprofiler selected
regions plus a request-specific ROCTx range. The unrenamed trace retains
physical pipeline symbols; renamed PMC rows are selected by range and joined
to those symbols by contiguous dispatch order. CPU uses trainer-owned Linux
`perf_event_open` groups attached directly to the persistent OpenMP worker TIDs
captured after warmup. Each worker opens a simultaneously scheduled event group;
short concurrent control regions arm and disarm every group around exactly one
production candidate launch. The transaction rejects multiplexed counters.
External system-wide `perf stat` FIFO control is forbidden because its polling
delay leaves counters enabled after the target returns and admits unrelated
machine work. Multi-socket collection uses
one profiler worker per physical socket with disjoint semicolon-delimited CPU
masks; the coordinator partitions request IDs across lanes and alone publishes
the resumable evidence manifest, so the sockets never duplicate work.

CPU process startup is amortized without amortizing evidence. Exact requests
with identical build/ISA/threading, operation, contract/bundle, execution mode,
and N/K geometry may share one trainer process; format, M, and candidate remain
independent TSV batch members. The process reuses fixture and persistent OpenMP
team setup, then rebinds request/output identity, resets and arms every per-TID
group, launches one exact candidate, disarms, reads, and atomically publishes
before the next member. Request-local raw directories contain only that exact
perf report, binary provenance, and batch-member record. Shared process logs and
trainer diagnostics are excluded from request hardware-evidence digests. The
coordinator verifies the complete returned ID set and checkpoints once per
batch to an authenticated append-only `*.inprogress.jsonl` journal rather than
rewriting a growing evidence manifest after every member. Each append is
flushed and fsynced. Resume preserves every newline-complete record, truncates
only a torn final JSON object, and materializes the full canonical manifest once
when the collector invocation exits.

On gfx906, `FetchSize`, `WriteSize`, `Wavefronts`, `VALUUtilization`, and
`LDSBankConflict` use separate profiler passes. In particular, combining the
two TCC traffic metrics exceeds the hardware's one-pass capabilities and can
leave rocprofiler handling a fatal capability error for tens of seconds. The
reviewed singleton schedule avoids that failure path while retaining both
signals.

The v1 evidence sidecar contains one ordered dispatch list per request and the
complete backend metric inventory for every dispatch. Initial signal includes:

- CUDA registers/thread, shared and local memory, theoretical and achieved
  occupancy, compute/DRAM/L1/L2 speed-of-light, IPC, warp issue cost, and local
  spill requests;
- ROCm VGPR/SGPR/LDS/scratch resources, occupancy where available, GPU/VALU and
  memory-unit busy, VALU utilization, L2 hit rate, traffic, wavefront count,
  stalls, and LDS conflicts; and
- CPU cycles, reference cycles, instructions, task clock, exact wall clock,
  L1D loads/misses, and LLC misses under each AVX2/AVX512
  build/runtime-dispatch regime. Events unavailable in one simultaneous group
  remain explicit unsupported records rather than being multiplexed.

After evidence coverage is complete, `export-features` performs an authenticated
three-way join of the original common-observation CSV, profiler request
manifest, and evidence manifest. It writes one `*_profiler_features.csv` row
per ordered physical dispatch. Every row retains the complete canonical timing
observation (`min`, `median`, `p95`, MAD, CV, effective bandwidth, and raw-sample
hash), candidate schedule/workspace identity, pipeline dispatch count and
ordinal, physical kernel identity and launch geometry, and explicit
value/availability columns for every backend profiler metric. The export fails
if the timing corpus digest differs, a request is not the exact derivative of
its observation, or profiler coverage is incomplete. A multi-kernel candidate
therefore repeats its pipeline-level canonical latency on each dispatch row,
while each dispatch's profiler duration and hardware counters remain distinct.

These features are mandatory inputs to the current performance-model fit, as
well as supporting candidate diagnostics, principled candidate pruning, and
active measurement. Raw evidence remains immutable and complete. The model
descriptor excludes raw duration, raw cycle/cache/traffic counts, and raw grid
size. It binds every dynamic normalized feature to the complete exact launch
identity. Without that ownership, the first profiler-informed learner treated
normalized counters measured at one shape as shape-independent candidate
attributes and leaked held geometry back into its own cross-validation fold.

The reviewed model schema instead derives work-normalized, regime-specific
features:

| Regime | Primary economic signal | Supporting normalized signals |
|--------|-------------------------|-------------------------------|
| Decode/GEMV/grouped verifier | effective bytes per second and duration/cycles/instructions per expected payload MiB | cache/branch miss fractions, occupancy, register/shared/local-memory pressure, spill rate per output, physical-fetch amplification, and output-write amplification |
| Prefill/GEMM/grouped MoE | effective operations per second and duration/cycles/instructions per logical MAC | arithmetic intensity, compute/memory speed-of-light fractions, occupancy, resource pressure, cache behavior, spill rate, and wavefront density |

Expected bytes are recomputed from each timing row's actual source-format
payload width. Physical profiler descriptors are additionally codebook-local,
so a LUT-decoded kernel can never inherit the instruction, cache, or traffic
behavior of Q4_0 merely because both use the same scheduling candidate ID.
Static launch configuration and resource descriptors remain candidate
attributes. Dynamic normalized counters, including throughput, IPC, achieved
occupancy, cache behavior, traffic, and duration-derived rates, remain
exact-point evidence. Only normalized metrics and reviewed candidate
configuration interact with runtime work, aspect ratio, row count, and tile
geometry; raw counter magnitudes never enter a fit record.

CPU duration-sensitive features have an additional reliability gate. The
profile wall interval MUST be at least 500 microseconds and
`task_clock / wall_clock` MUST NOT exceed the physical worker count by more than
5%. Below that boundary, control-region and OpenMP runtime tails are too large a
fraction of the launch. Cycles, reference cycles, task clock, wall-derived
rates, and IPC are masked, while stable instructions-per-work and L1 behavior
remain available. The focused Q4_K probe found instruction-count coefficients
of variation below 0.6% across three repeats for all five tested cells.
Instructions, L1 loads/misses, and LLC misses are normalized by expected payload
MiB for decode or million logical MACs for prefill. Sparse LLC misses receive a
lower auxiliary reliability weight than instructions and L1 behavior. One-shot
wall/task clock values remain diagnostics beside canonical repeated timing and
do not become auxiliary timing labels.

The pre-collection smoke used AVX512 Q4_1 at `N=6144,K=2048`. Three candidates
in one process each emitted one complete report: row-chunk retired about 131.2M
instructions, while pair-grid NBC1 and NBC16 each retired about 89.8M. In a
second process, the same row-chunk candidate at M64 and M256 retired 134.4M and
533.4M instructions respectively; L1 loads and misses also scaled almost
exactly 4x. Those coherent candidate and work-size differences establish useful
signal before broad exact-point collection; they are diagnostics, not latency
labels or promotion evidence.

Inside each grouped-CV fold, a deterministic multi-output ExtraTrees surrogate is trained
over a cross-M transfer domain: architecture, arithmetic contract, operation,
bundle, and execution mode remain fixed, while tensor formats and prefill M
values share evidence. M is a first-class numeric runtime feature, so the model
learns candidate scaling across work sizes instead of fitting disconnected
forests for each sampled M. Codebook, prepared family, packing ABI, candidate
configuration, analytical schedule geometry, genuinely static resources, and
runtime geometry are model inputs. Exact dynamic profiler metrics are not
inputs: they are centered against competing candidates at the same work point,
standardized, and used as auxiliary targets beside log-regret. Their combined
variance weight is 0.5 relative to the normalized timing target's weight of 1.0.
Within that fixed budget, repeatable instructions-per-work receives the largest
CPU reliability weight, L1 evidence is secondary, and sparse LLC evidence is
diagnostic-weighted. A metric is centered only when every candidate in that
exact work-point contest published it; partial optional-event availability is
neutral rather than a synthetic zero-cost candidate. This lets execution
evidence shape partitions without requiring counters for a held or unseen
runtime point. Every occurrence of the fold's held-out N/K geometries is removed
from the training pool across all formats and all M values before fitting. Thus
neither a Q4 row nor another prefill row count can leak the answer for a held-out
Q5 row at the same geometry.

The same exclusion applies to profiler targets. Every dynamic record at held
N/K geometry is absent across all M and formats. Genuine static resources and
configuration remain visible as inputs. If the remaining training rows have no
varying candidate-relative dynamic targets, the profiler-influence variant is
measured-only; configuration columns cannot masquerade as profiler evidence.

The non-negative prediction supports two additive tournament members; neither
is another timing oracle. Bounded-prior variants reshape training costs before
an ordinary generic tree is fitted. The cross-fitted teacher instead chooses a
held candidate directly from the surrogate's predicted regret, then exposes
that choice to held canonical timing for the first time. Measured timing retains
the hard five-percent pass/fail topology in both cases. If the teacher wins, its
out-of-fold decisions are distilled into the same bounded, total generic C++
publication tree as every other model; sklearn is never part of runtime
dispatch. Exact winners, paired confirmation, and sealed certification also
score only canonical measured latency.

Profiler-model construction is a distinct host-parallel phase before generic
tree scoring. The learner groups all required prediction surfaces by cross-M
transfer domain, forks up to
`LLAMINAR_NATIVE_VNNI_PROFILER_MODEL_WORKERS` workers (all affinity-visible
physical cores by default). The default counts physical `(socket, core)`
identities, not SMT siblings; an explicit environment value remains available
for controlled A/B work. The parent materializes every pool's candidate-point
feature index once, then workers inherit those immutable records copy-on-write
and schedule one independent held-out ExtraTrees surface per task. Individual
estimators remain deterministic and single-threaded, while one large cross-M
pool can use the complete two-socket host instead of collapsing to one process. Only
prediction maps return to the parent. CV task assembly is a strict read-only
cache consumer and hard-fails if any requested surface is absent, preventing
an accidental serial fit from returning inside a domain loop. All
profiler-model workers exit before CUDA/ROCm leaf-scorer workers are created,
so the two CPU-intensive phases cannot oversubscribe each other.

Paid surfaces also survive process restarts in the content-addressed policy fit
cache. A `profiler-prediction-surface` key binds the complete cross-M
candidate-cost pool, model-visible descriptor digest, held N/K set, and exact
prediction-point inventory. The payload contains only domain-local training
points consumed by bounded-prior fitting and held points consumed by teacher
scoring, with hexadecimal floats for byte-stable replay; unrelated transfer-
pool rows are excluded. Missing, partial, duplicate, or foreign point maps are
hard failures. Parallel workers publish in completion order, so one slow early
forest cannot strand completed surfaces in an ordered executor buffer. An
identical fit with an empty in-memory cache therefore performs zero ExtraTrees
work after the first complete surface publication.

The first complete parallel CPU-prefill fit exercised 42 fixed-M transfer
pools, 1,145 held-out CV surfaces, and 42 full-development surfaces: 1,187
forests total. Profiler models completed in
82.487 seconds; the superseded serial loop was still building the same models
after eight minutes when stopped. The subsequent 48-lane CUDA/ROCm exact CV
phase took 1,091.631 seconds, canonical reduction took 29.204 seconds, and final
tree fitting took 5.372 seconds. These phase timings are emitted separately so
future runs cannot hide single-core model construction inside generic “task
construction” time.

The complete CPU-prefill round-6 corpus is also the mandatory profiler-feature
ablation fixture. With the old one-sided penalty, the authenticated profiler-on
fit produced 1,099 rules and 90 unpromoted domains, while an otherwise identical
fit with the profiler catalog deliberately withheld produced 1,077 rules and 96
unpromoted domains. Profiler evidence fixed eight domains and harmed two, but it
also worsened mean domain regret from 2.105% to 2.491% and increased the maximum
from 50.998% to 162.518%. Fold-held-out feature experiments showed that the
normalized profiler columns improve candidate economics on average and at p95.
A follow-up per-domain bounded Ridge experiment was also rejected: it produced
only 944 rules and 133 unpromoted domains, fixing six profiler-off failures but
introducing 43. The problem was forcing a weak 31-shape linear surrogate into
every domain, not missing counter coverage.

The replacement fixed-M ExtraTrees ablation is materially stronger. With
geometry and candidate configuration but profiler columns removed, strict
held-out-geometry prediction selected the exact candidate 87.12% of the time,
left 663 direct candidate cells over budget, and had 5.62% p95 selected regret.
Adding normalized profiler columns raised exact selection to 87.50%, reduced
failures to 636, and lowered p95 to 5.23%; profiler-derived columns contributed
44.37% of aggregate feature importance. These are development diagnostics, not
promotion claims. The generated generic tree must still pass its complete
grouped CV and sealed timing gates.

The first complete fixed-M ExtraTrees policy fit was therefore diagnostic and
was rejected. It produced 1,000 generic rules and 118 unpromoted domains, with
2 profiler-off failures fixed but 24 previously passing domains harmed. Mean
domain regret rose from 2.105% to 2.553%, p95 rose from 14.20% to 16.67%, and
maximum regret rose from 50.998% to 67.143%. The surrogate contained useful
candidate signal, but requiring its bounded prior in every domain overrode
better measured-only tree structures.

Learner v14 makes profiler influence an explicit grouped-CV hyperparameter.
For every feature family, threshold placement, and leaf budget, held-out CV
compares a `measured_only` tree with a `bounded_prior` tree and scores both
using raw canonical held-out latency. Final full-development fitting uses the
mode selected for that domain, and exact objective ties retain deterministic
enum ordering. Because the complete measured-only model frontier remains in
the tournament, profiler evidence is strictly additive: it can promote a
better-generalizing tree but cannot force a domain to accept worse held-out
economics. The selected influence is frozen in policy IR, fit-cache records,
and paired-request summaries.

The complete round-6 v14 fit selected `measured_only` for 724 domains and
`bounded_prior` for 32. Relative to profiler-off it harmed zero passing
domains, promoted two previously failing domains, reduced mean domain regret
from 2.105% to 2.022%, and reduced p95 from 14.197% to 14.020%. The maximum
remained 50.998%, and 94 domains remained unpromoted, so the fit is useful but
still non-installable. Its exact phase timings were 83.367 seconds for 1,187
profiler forests, 2,255.133 seconds for the doubled 48-lane CV frontier, 56.246
seconds for canonical reduction, and 5.948 seconds for final trees. The next
development-only boundary plan contains 94 obligations and 346 targeted
neighbor records across all 18 formats and all three CPU ISA regimes.

Learner v15 changes the installation objective from maximum regret to measured
p95 regret and conservative p95 UCB, both strictly below 5%. The authenticated
v8-to-v9 additive lineage added the four Qwen 3.6 35B MoE production
geometries without recollecting the historical corpus. It produced 88,288
strong observations and 1,011,556 raw timing samples. The first complete v15
fit emitted 1,245 provisional generic rules and reduced the unpromoted set from
94 to 55 domains. Forty-eight were genuine held-out-CV p95 failures; seven
passed CV but failed the final full-development tree gate. This artifact is a
development diagnostic only and is not installable.

Round 8 measured exactly those 55 obligations. Its 211 process jobs completed
in 842 seconds and appended 1,760 strong observations plus 20,939 raw timing
samples, yielding an immutable 90,048-observation/1,032,495-sample development
corpus. The aggregate and timing SHA-256 digests are respectively
`e39f5627ac14e513b0cb581018b7829b223e80a18073a9f3404d7037aa12d3c5`
and
`a194b1e2ba733cc5e685211ebfd1c3a3f312f4b0f7a28f8c2b3f4356d50f18d3`.
Profiler counters were not relaunched: the refit consumes the original
per-physical-candidate profiler corpus as an immutable feature source.

Rounds 9 through 11 appended only their authenticated failed-domain boundary
plans. The current round-11 corpus contains 94,208 strong observations and
1,081,579 retained timing samples. Aggregate and timing file SHA-256 digests
are respectively
`45c0a43ff401bdfda51c66e83b980bb768e79237d8813fb6d872985a011b9964`
and
`900a894d7f0db460b22ea9ddceb2e2e72443281b2ddc7e01e810a39e4ff8891e`.
The additive learner-v17 tournament retained every prior feature family while
adding row-grid launch geometry; it emitted 1,384 provisional generic rules
and left 32 domains unpromoted. It fixed the AVX2 CB8 `M=2048` failure from v15
and the AVX512 CB16 `M=2048` regression from v16 without introducing a new
failure relative to either parent.

That v17 result is still diagnostic. Its inherited 48-request sidecar profiled
only CB0 and aliased those counters across all formats, so its apparently
profiler-informed decisions are not admissible installation evidence. The
corrected request/evidence files have SHA-256 digests
`c7d0698ab671a6ab8d8c9b52f17d5afe210b0146797ce317d752fca92111581d`
and
`d97172e13260ab04dd4a82b8c6fb3de886ed3075e77a0688ca096548caba3f23`.
Refitting from those sidecars reuses the exact round-11 timing corpus.

The corrected codebook-local round-11 fit emitted 1,380 generic rules and left
31 domains unpromoted. Its matched timing-only ablation emitted 1,295 rules and
left 46 domains unpromoted. Profiler-informed selection fixed 15 of those 46
failures and introduced none; 98 of the 756 domains selected a bounded profiler
prior while every published regret remained a canonical timing measurement.
This is the admissible profiler A/B result. The earlier aliased-CB0 comparison
is retained only as a diagnostic of the catalog bug.

Increasing the generic-tree leaf ceiling from 16 to 32 changed no exact entry,
generic rule, selected grouped-CV model, promotion diagnostic, or unpromoted
domain. It enlarged only seven planner-only competitive unions. Capacity is
therefore not the remaining limitation, and 16 remains the production default;
`--generic-max-leaves` exists as an explicit bounded diagnostic rather than an
implicit policy change.

Production fitting no longer materializes every losing tournament model's
millions of repeated held-out cell objects. It ranks the complete model
frontier from immutable fold results, retains the selected model, and then
hydrates the selected validation's complete cell evidence. Paired planning
continues to retain its full competitive frontier. Fit-cache keys distinguish
those two selection modes, so a planning cache entry cannot truncate a later
production fit. On the identical round-11 corpus this reduced canonical CV
reduction from 413.539 seconds in the 32-leaf diagnostic to 21.959 seconds;
all 756 candidate-cost domains were cache hits and policy semantics were
unchanged.

Focused round 12 collected 124 requested M cells in 120 socket-local jobs and
completed in 385 seconds. It appended 992 strong observations and 13,299 raw
timing samples, producing a 95,200-observation/1,094,878-sample corpus. The
aggregate and timing SHA-256 digests are respectively
`9c7b8396cd49d45e1ddf27ef8d19b34a3da0c874e8578f3266e3cc8b7cb786e1`
and
`adb89034dbca412f7f43d738cf4130300ae9d969d0a1a628495654fc9b0e2a96`.
The refit emitted 1,393 generic rules and reduced the unpromoted set from 31 to
29: six round-11 failures passed, four newly exposed held-out failures became
visible, and no domain was silently promoted. All 29 residuals use the full-K
bundle; 27 fail grouped held-out CV and two pass CV but fail final-tree p95.
The AVX512-build/AVX2-runtime surface is now fully promoted. Round 13 contains
29 obligations, 114 process jobs, and 116 fresh cells across only the remaining
native AVX2 and native AVX512 domains. These artifacts remain development-only
until the unpromoted set reaches zero and a fresh sealed certificate passes.

Rounds 13 through 16 remained strictly additive and produced an immutable
98,112-observation development corpus. Learner v18 replaced the rejected
shape-hash fold assignment with a stratified, cross-fitted publication pass.
The admissible round-16 fit emitted 1,551 generic rules and left 16 domains
unpromoted. Every final full-development tree passed; all 16 residuals were
held-out-CV p95 failures. The 740 publication domains had a maximum publication
p95 regret of 2.819%, so publication fitting itself was below the three-percent
budget without weakening the held-out gate.

The residual pattern exposed a physical schedule gap rather than another
learner-capacity problem. Ordinary full-K two-row prefill scheduled only
`ceil(N_chunks / n_block_chunks)` OpenMP tasks and made each task visit every
row pair serially. At larger M, especially on AVX2, otherwise useful two-row
microkernels therefore starved most CPU workers. The grouped verifier helper
already owned the economical shape: a two-dimensional `(row_pair, N_block)`
task grid using the same ordered two-row arithmetic. That schedule is now a
first-class production candidate family with `nbc={1,2,4,8,16}`. The old
N-major family remains a separately measured physical candidate until evidence
proves it dominated; the generated ABI does not conflate the two schedules.

The new pair-grid family is byte-equal to serial decode across all 21 formats
at `M={2,3,15}` in native AVX-512 and AVX-512-build/forced-AVX2 runs. The
all-format production trainer reported 315/315 correct route/candidate cases in
each regime, with zero output or repeat-byte mismatches. At the representative
Q2_K `N=K=5120`, `M=2048` point, pair-grid `nbc1` improved the AVX2-only build
from 89.714 ms to 87.158 ms, improved the AVX-512 build with forced AVX2 from
95.121 ms to 87.705 ms, and tied the already favorable native AVX-512 path at
34.760 ms versus 34.789 ms. Coarser N blocking benefits more because pair-grid
continues to expose M parallelism.

Candidate expansion is an authenticated additive evidence transaction. The
round-16 source aggregate contains 12,264 immutable historical measurement
cells and 4,683 grouped process records, but historical exploratory geometries
are not an admissible substitute for a homogeneous candidate matrix. The
reviewed expansion cohort therefore selects exactly 6,394 cells and 1,855
records: every production overlay plus every systematic generic-fit geometry
witness named by the development split. Every selected cell must expose the
complete current registry after adaptation; a candidate family measured on
only the smaller production subset is a failed matrix, not partial generic
evidence.

Collection launches only the five new pair-grid candidates, the existing
row-grid timing anchor, and the retained pairwise k-part route diagnostic on two
disjoint socket-local MPI ranks. It does not recollect the other old timing
candidates. Absolute clocks from separate runs are never compared directly:
the adapter scales each new candidate's timing distribution by
`source_anchor_median / expansion_anchor_median` on the exact source alias,
shape, M, thread count, build ISA, and runtime ISA surface. Raw aggregates and
timing sidecars remain unchanged, and every derived observation records the
plan digest, both anchor medians, and the exact scale in adaptive-timing
provenance. Missing anchors, changed source bytes, incomplete source matrices,
or attempts to recollect an existing candidate fail closed.

Each shape/format/ISA result is still published as an atomic aggregate/timing
pair. A timing-policy change creates a new content-addressed plan, revalidates
compatible completed pairs through the current adapter, and copies their exact
bytes into the new checkpoint; only absent or rejected records are measured
again. Candidate-expansion MPMD batches receive at most three fresh timing
epochs. Every rank's temporary pair must pass byte equality, route provenance,
and stationary timing validation before either rank is published. A transient
frequency or host-interruption window therefore retries the complete paired
batch without relaxing the 2% drift gate, and an interrupted or exhausted run
retains every final pair collected before it failed.

New anchored records retain at least five stationary samples and a 100 ms
elapsed floor. Cheap kernels therefore contribute thousands of observations;
long 60-75 ms GEMMs contribute at least the explicit minimum without spending
seconds repeating one already steady candidate. Ordinary production collection
uses a 50 ms fixed warmup and the same 100 ms timing floor, plus the independent
latency-scaled transition warmup. The 2% median-drift decision, 180-sample
ceiling, global MPI complete-round protocol, and paired retry behavior remain
hard gates. Historical five-sample suffixes remain admissible only when their
retained raw sidecar independently passes that same 2% gate.

Incremental fit-cache identity is now dependency-accurate. Candidate-cost
matrices bind only the observation rows, paired evidence, and serial-M1 hashes
owned by their generic domain. Profiler-informed CV additionally binds the
complete set of domain-cost identities in its cross-M, cross-format transfer
pool, because changing one format or work size can change predictions for every
peer in that pool. The cache binds normalized model-visible profiler
descriptors, not the complete timing-corpus provenance digest: adding timing
evidence in one domain must not invalidate unrelated ISA/contract/operation/
bundle pools when their profiler features are unchanged. Full request,
evidence, and corpus provenance remains recorded in frozen policy metadata.
Existing global-serial-hash and full-catalog-digest cache records are
authenticated and promoted through read-through migrations.
Failed-fit artifacts separately record CV and final-tree rejection stages,
including the actual worst final leaf geometry; refinement plans use that
geometry and cryptographically bind the diagnostic list. Historical v1
refinement plans retain byte-identical canonical mappings, while new plans use
the diagnostic-bound v2 schema.

The learner rejects a genuinely competitive transfer pool if no profiler metric
has nonzero tree importance. A one-candidate domain or an exact all-candidate
tie is explicitly trivial: complete profiler coverage is still required, but
there is no dispatch choice for counters to explain. The model-visible
descriptor digest is part of the CV cache key, while the complete profiler
catalog digest remains in frozen policy metadata. Changing a normalized
counter feature or candidate descriptor therefore forces the dependent pool to
refit; changing authenticated provenance without changing model inputs does
not manufacture a global cache miss.

Profiler values are not runtime dispatch features and cannot replace measured
latency, bitwise correctness, sealed p95 regret, or the conservative p95
simultaneous-upper-bound promotion gate. Maximum regret and maximum UCB remain
diagnostics. Sealed profiler evidence remains inaccessible until the generic
policy digest is frozen and cannot trigger a refit, preserving the sealed
holdout boundary.

### 7.8 Exact GPU acceleration of policy fitting

Policy fitting may use CUDA and ROCm as compute accelerators without changing
the learner or its evidence. ABI v8 runs the complete deterministic 64-wide
bounded beam on the selected accelerator and carries 512-point training and
held-out masks. Grow-only fitting and canonical measured-regret matrices, legal
feature-value masks, ping-pong frontiers, compact virtual expanded candidates,
exact-signature hash storage, and compact selection indices remain
device-resident across fold transactions.
CUDA and ROCm enumerate every legal adjacent selected-value split, score both
new leaves, hash exact preorder signatures cooperatively, compare every hash
collision token-for-token, remove structurally identical trees, and retain the
canonical top-64 frontier for every leaf budget. Fused CV publishes only compact
fold results; the standalone final-publication entry point returns winning
trees only when Python must reconstruct backend-neutral policy IR.

ABI v8 removes the per-depth control loop from the host. Device counters feed
each captured depth directly into the next; there is no expanded-count download,
device-wide synchronization, allocator call, or whole-tree payload sort between
leaf budgets. An exact open-addressed set uses a deterministic hash for probing
but compares every colliding canonical signature in full. Hierarchical bitonic
top-64 selection moves only uint32 indices and is exact because an item outside
its partition's best 64 already has 64 globally better competitors. One final
result/status download and one session-stream synchronization terminate the
complete search. Persistent runtime diagnostics regression-test graph replay,
high-water reuse, retry counts, and the zero-intermediate-sync invariant.
With `LLAMINAR_NATIVE_VNNI_POLICY_TIMING=1`, the parent scheduler also publishes
weighted task completion, elapsed time, and ETA every 30 seconds. The weighting
uses the same point-count-by-leaf-budget estimate that orders dynamic work, so a
few large folds cannot make a raw task count falsely imply near-completion.

ABI v8 also retains the cooperative replacement for the long single-threaded
device sections found in the first
captured implementation. One cooperative pipeline now:

- enumerates legal split descriptors across the grid;
- hashes and compacts exact child masks before scoring duplicate leaves;
- scores eight candidate groups per block with adjacent matrix loads;
- performs bounded exact nearest-rank p95 selection across each 32-lane group;
- stores virtual expanded candidates instead of copying complete leaf arrays;
- hashes positioned signature tokens cooperatively before exact collision
  comparison; and
- computes ordered FP64 means separately from subgroup order-only p95.

The only serial arithmetic retained is each candidate owner's ascending-point
FP64 mean. That order is part of the byte-exact learner contract and cannot be
reassociated into a parallel reduction. Tensor cores are likewise inapplicable
to the irregular FP64 order statistic: GA102 has no FP64 tensor MMA path, and
integer or reduced-precision MMA would change policy bytes. These control
kernels use coalesced scalar loads and compact index/descriptor records instead.

The measured per-leaf nearest-rank p95 pass/fail bit owns the hard first key.
The bounded fitting nearest-rank p95, fixed-order mean, worst measured leaf
p95, measured point-failure count, diagnostic measured maximum, leaf count,
and exact structural signature follow in canonical order. P95 ranks use exact
integer `ceil(95*n/100)` arithmetic, and means use one left-to-right FP64
addition order on all three implementations. CUDA compiles this DSO with FMA
contraction and flush-to-zero disabled; ROCm compiles it without fast math or
FP contraction. The accelerator MUST reproduce Python policy bytes exactly.
Installation independently requires both measured and conservative p95 regret
to be strictly below 5%.

CUDA and HIP runtimes MUST remain process-isolated. One policy worker loads one
vendor DSO, binds one backend-local device ordinal, and never changes backend or
ordinal during its lifetime. Multiple independent CPU orchestration lanes MAY
share a physical device to fill independent hardware queues; each lane is still
a separate one-vendor process. Each search session uploads one immutable regret
matrix exactly once, retains its non-default stream, and reuses device
allocations across the full beam transaction. Moving to another matrix destroys
that session before preparing its replacement; the stateless
allocate/upload/free-per-call ABI and host-owned tree-expansion loop are obsolete
and MUST NOT return.

Fold tasks use an accelerator-worker dynamic queue rather than static
partitions. The parent offers longest estimated tasks first, immediately
refills the first free lane, permits only one in-flight task per lane, and
restores exact task-index order before canonical reduction. This preserves
deterministic policy bytes while preventing a few expensive folds from leaving
most devices idle at the end of a run. A requested DSO, device, or worker
failure is fatal. It MUST NOT fall back to CPU fitting after an accelerator run
has begun.

Round-12 release-geometry expansion exposed that the v5 tree-search ABI's
128-point mask was no longer real headroom, and the collapsed all-format CPU
decode domains subsequently reached 444 points. ABI v8 raises the reviewed
bound to 512 points, retains 32 leaves and 64 feature axes, and represents every
point and shape-group mask as eight little-word-first uint64 words.
Intersection, partition, hashing, exact collision comparison, group counting,
signature construction, and result publication all consume the complete
eight-word value. The exact nearest-rank p95 subgroup scan covers the complete
reviewed upper tail. An explicit
request beyond the reviewed bound remains a hard error; it cannot invoke a
hidden CPU path. Canonical multiword masks and exact preorder structure tokens
prove disjoint leaves, complete point coverage, and byte-identical
reconstruction.

The accelerator implementation is accepted only when CUDA and ROCm match an
independent serial oracle for the diagnostic leaf primitive and every complete
tree budget. Coverage includes exact 5% neighbors, conservative p95 rank
boundaries, duplicate candidate columns, missing candidate cells, irregular
N/K geometry, every feature-policy family and boundary placement,
profiler-adjusted fitting costs, exact structural ties, a natural 40-shape case
that catches pooled-p95 versus per-leaf-p95 mistakes, and an 81-shape case that
crosses the old mask boundary. Complete planner
report and paired-request JSON must also be byte-identical to canonical CPU
fitting. These tests belong to Integration; Unit tests use CPU fakes and pure
oracles without doing GPU work. A scheduler-only fork regression MUST also
prove that a free lane consumes pending tail work while a slower lane remains
occupied.

The production refresh wrapper defaults to every auto-discovered CUDA and ROCm
device with one persistent CPU orchestration lane per device. More lanes
duplicate the grow-only device high-water allocation and are reserved for
reviewed small-domain diagnostics. Ownership can be explicit:

```bash
scripts/refresh_native_vnni_dispatch_tables.sh \
  --backend all \
  --profile all \
  --policy-accelerators cuda:0,cuda:1,rocm:0,rocm:1,rocm:2,rocm:3 \
  --policy-lanes 1 \
  --output-dir benchmark_results/native_vnni_dispatch/<run-id>
```

`--policy-accelerators cpu` deliberately selects the canonical CPU process
pool. This is a requested training mode, not a runtime dispatch fallback.
Policy acceleration does not profile candidate kernels and cannot perturb,
replace, or become a label in canonical timing or isolated profiler evidence.

The first leaf-primary Q4 proof on 2026-07-12 used two RTX 3090s and four gfx906
MI50/MI60 devices, with eight lanes per device. It processed 71,298 strong
observations and reproduced the prior CPU planner artifacts byte-for-byte:

```text
report   2da21aae9ec1fa762cc4d9b91c87aadd47859c6287bd6e6708c0f923c0a42591
requests 62065bea6cbe935d752b8d3f2b83ca11e92d2cb464879ce2c3ed821884ddd7da
```

The prior exhaustive CPU pass took about 23 minutes. The cross-backend GPU run
took 195.559 seconds end to end, of which approximately 127 seconds preceded
worker launch and 69 seconds covered accelerated CV and final planning. A
smaller transition-neighborhood corpus remained fixed-overhead-bound (20.605
seconds GPU versus 20.727 seconds CPU) while still reproducing report digest
`6ea87dd5...` and request digest `d938e0ed...`. These measurements justify GPU
acceleration for full development/all-format jobs, not as ceremony on tiny
smoke corpora.

A cold six-domain Q4 ABI-v2 transaction after persistent sessions and dynamic refill
completed fold fitting in 124.115 seconds and the full planner in 171.758
seconds. Replacing tuple subset expansion with exact integer masks reduced
those figures to 113.051 seconds and 161.576 seconds respectively, while
reproducing request SHA-256 `4a28c709...` and report SHA-256 `e41af083...`
byte-for-byte.

ABI v3 first removed the remaining host-owned beam loop. On the largest round-9
production domain (43 points, 215 candidate-point rows, six candidates, all 16
budgets), the canonical Python beam takes 13.221 seconds. Complete device search
takes 1.416 seconds on an RTX 3090 and 6.537 seconds on a gfx906 MI50, producing
the exact same tree sequence. Small domains can remain process/launch-overhead
bound; dynamic scheduling assigns independent domains across all requested
devices rather than duplicating work. An identical cache replay bypasses all
cost matrices and CV fits and launches no accelerator process.

ABI v4 then replaced monolithic expansion and serial leaf scoring with the
cooperative pipeline above. On gfx906, a focused complete-search rocprof trace
dropped aggregate device time from 528.176 ms to 169.983 ms while preserving
the exact CUDA, ROCm, and host-oracle result bytes. The remaining largest
contributors were compact top-64 selection at 96.970 ms, cooperative
materialization at 27.932 ms, and candidate scoring at 22.651 ms. On an RTX
3090, candidate-aware launch sizing reduced a representative scoring launch
from 35.20 us to 26.85 us. Nsight Compute reported 40 registers per thread,
12.19 KiB static shared memory, and zero local-memory spills.

The subsequent subgroup reduction specialized candidate counts `1..8` and
replaced shared-memory p95 reduction with warp/subgroup shuffles. On the RTX
3090, the representative scoring launch fell from 39.94 us to 35.36 us,
theoretical occupancy increased from 43.75% to 100%, achieved occupancy rose
from 15.22% to 19.04%, and static shared memory fell from 12.19 KiB to 3.42
KiB. Nsight reported 39 registers per thread and zero local-memory spills. A
complete CUDA/ROCm/host tree-sequence regression proves that the optimized
scorer preserves the exact integer primary-objective ordering.

The ABI v8 compact-frontier pass removed complete expanded-tree copies and
parallelized exact positioned-token signature hashing. On the representative
production domain, CUDA deduplication fell from 2.162 ms to 0.011 ms and gfx906
deduplication from 13.654 ms to 0.026 ms. Separating the required serial FP64
mean from subgroup order-only p95 reduced objective-plus-finishing time from
about 2.115 ms to 1.134 ms on GA102 and from about 4.492 ms to 2.232 ms on
gfx906. Nsight reports zero local-memory spills; rocprofiler reports zero
scratch/private bytes. Grow-only graph scratch now peaks near 0.5-0.9 GiB per
worker instead of the earlier multi-GiB materialized-expanded-tree footprint.
Allocation traces and runtime counters prove that stable replay performs no
allocation, free, graph recapture, scratch growth, or intermediate sync.

Production candidate kernels remain a separate concern from policy-tree
control. A focused Qwen 3.6 MoE GDN Q4_0 prefill launch
(`M=128`, `N=8192`, `K=2048`) selected the real `nativeVnniTC_BK64` path and
timed at 72.704 us outside the profiler. A separate Nsight launch observed
524,288 `sm__inst_executed_pipe_tensor_op_imma` instructions, 512 IMMA
instructions per warp, and 524,288 128-bit memory instructions. Thus Ampere
INT8 tensor-core execution and vectorized memory operations are measured facts,
not source-level assumptions. Nsight replay inflated that launch to roughly
one second; profiler duration remains explanatory evidence and MUST NOT replace
canonical timing.

### 7.9 Incremental paired-fit transaction

Paired refinement MUST NOT rebuild every unchanged domain after each new
selected/reference measurement. The planner maintains content-addressed,
strictly validated artifacts for:

1. a compact development-corpus identity keyed by source-file, shape-manifest,
   semantic-surface, and projection-arithmetic digests;
2. each independent domain candidate-cost matrix, additionally keyed by only
   the paired tournament edges and serial-M1 dependency capable of affecting
   that domain; and
3. each domain CV result, keyed by its cost matrix, feature schema, beam width,
   fold seed, leaf budget, and minimum leaf support; and
4. each stable all-development publication tree alongside its exact CV entry,
   so an unchanged production domain does not rerun final GPU fitting.

A malformed cache entry or identity mismatch is fatal. A missing entry is a
first-use cache miss, not permission to weaken the learner. Cache reuse MUST be
artifact-identical to a clean fit and receives a regression that makes any
unexpected fold evaluation, candidate-cost load, or final-tree fit fail.

The projected formula corpus is not serialized. The planner retains compact
direct observations and lazily projects reviewed shape-resolved formulas only
for a domain whose costs or CV must be recomputed. Request publication resolves
a nominal formula directly from the candidate registry and authenticates the
resulting concrete schedule against its measured source row. This prevents the
108k-row source corpus from becoming a gigabyte-scale repeated-metadata cache.

Large common-observation checkpoint replay uses deterministic byte-range
process parsing. Workers own disjoint complete CSV records and the parent
reduces partitions in original range order before running the unchanged corpus
and provenance validators. On the 578 MiB, 215,748-row CPU prefill checkpoint,
8 workers reduced parse wall time from 48.70 seconds to 37.21 seconds while
producing the identical corpus digest; 16 workers showed no further gain. Set
`LLAMINAR_NATIVE_VNNI_ADAPTATION_WORKERS` to override the measured default, or
`LLAMINAR_NATIVE_VNNI_DISABLE_PARALLEL_ADAPTATION=1` for diagnostic serial
replay.

Before reducing CV models to one selected tree, the learner retains every
evaluated feature-policy, boundary-placement, and leaf-budget model and unions
their unresolved over-budget selected/exact edges. The paired trainer measures
that complete competitive frontier in the current batch. Directly measured
edges are never requested again merely because they belong to an alternate
model, and only the
currently selected model can create a terminal development failure. This
frontier closure plus domain-local invalidation turns late refinement into one
measurement batch and one affected-domain refit instead of a sequence of full
corpus passes.

## 8. Candidate Eligibility and Oracle Versioning

### 8.1 Fast eligibility

A `Fast` candidate is eligible only if it:

- is statically supported and production-reachable;
- passes the backend's normal numerical correctness gate;
- observes the forced route;
- satisfies stream, workspace, and graph-capture contracts; and
- is deterministic to the degree required by the ordinary operation.

Fast correctness is separate from verifier correctness.

### 8.2 Verifier eligibility

A `VerifierSerialM1Bitwise` candidate is eligible only if all of the following
are true:

```text
supported
&& forced_route_ok
&& route_counter_ok
&& bitwise_equal
&& repeat_equal
&& serial_m1_policy_hash == current_serial_m1_policy_hash
&& no_uncertified_atomic_reduction
&& stream/workspace/capture contracts satisfied
```

Eligibility is evaluated for every source-format alias. One failing alias
rejects the candidate for the shared runtime key.

### 8.3 Exact-only versus generic-eligible

Empirical equality at one exact shape is sufficient only for an exact entry.
A candidate may enter a generic rule only when:

1. its arithmetic fingerprint is structurally compatible with the serial M1
   policy for the declared shape domain; and
2. it passes byte equality on all development and sealed tail/alignment classes
   in that domain.

This prevents isolated lucky equality from being extrapolated across unseen K
partitions or reduction trees.

### 8.4 Serial M1 policy identity

The serial M1 policy ID MUST resolve from the real production launcher. Its
hash includes at least:

- policy ABI;
- backend and architecture class;
- operation and prepared family;
- effective candidate ID;
- arithmetic fingerprint;
- kernel-math ABI; and
- shape-dependent K partition/reduction parameters.

Changing the M1 resolver, kernel math, quantization order, reduction order,
epilogue, or relevant compiler specialization invalidates dependent grouped
certificates. The generator MUST reject a base artifact with a different M1
hash rather than retaining its verifier entries.

M1 selection and verifier certification form an ordered two-stage
transaction:

1. train the `Fast M=1` policy;
2. generate, build, and stage that production M1 artifact;
3. resolve and freeze its policy/arithmetic hashes;
4. collect every `M=2..16,31` verifier observation against that exact staged M1
   artifact; and
5. compile verifier exact and generic policies without changing M1.

Any subsequent change to the staged M1 decision, code, packing ABI, or hash
restarts verifier collection and certification from step 3. A transaction MUST
NOT retune M1 after collecting dependent verifier rows.

## 9. Alias-Robust Runtime Exact Winner Construction

Exact winners are constructed after correctness filtering and before generic
learning.

For shape `s`, alias `f`, required execution mode `x`, candidate `c`, and
contract `q`, let:

```text
L(c,s,f,x,q) = median measured latency
Lbest(s,f,x,q) = min L(e,s,f,x,q) over eligible candidates e
surface_regret(c,s,q) = max_(f,x) (L(c,s,f,x,q) / Lbest(s,f,x,q) - 1)
```

Because policy ABI v1 cannot distinguish aliases or required execution modes,
the alias-robust runtime exact winner is selected lexicographically by:

1. minimum `surface_regret`;
2. minimum p95 alias/mode regret;
3. minimum mean normalized latency;
4. lower measured variance;
5. stable candidate ID as deterministic tie-breaker.

Candidates whose measured difference is smaller than the declared timing
uncertainty form a co-winner set. Tie-breaking within that set uses stability,
lower policy complexity, and stable candidate ID rather than claiming a noisy
ordering as fact.

For a non-aliased, single-mode key this reduces to the fastest stable eligible
candidate.

Exact overlays MUST be emitted for required production keys. Training-only
synthetic shapes MAY remain generic-only unless they expose a demonstrated
generic failure that the runtime domain must support.

No exact entry may reference a candidate missing from the compiled backend
registry.

## 10. Generic Aspect/Work Learner

### 10.1 Common v5 feature policy

Feature schema v5 uses the common aspect definition:

```text
aspect_ratio = aggregate_N / K

very_wide: aspect_ratio >= 16.0
wide:       aspect_ratio >= 2.0
balanced:   aspect_ratio >= 0.75
tall:       otherwise

work_items = aggregate_N * K
```

Rules are learned independently for:

```text
(backend, architecture_class, semantic_contract, operation_kind,
 bundle_signature, prepared_family/runtime_codebook, execution_mode, M,
 aspect_bucket)
```

Every tree may use exact rational predicates over `N`, `K`, `N*K`, and `N/K`.
Cross-validation also chooses one reviewed launch-geometry feature family:

```text
ceil(N / tile_N)
K / (32 * ceil(N / tile_N))
((N - 1) mod tile_N) + 1
N / (tile_N * ceil(N / tile_N))
```

The final two features expose active values in the final N tile and aggregate
N-tile utilization. They are required because a `tn64`/`tn128` candidate switch
can be invisible to monotonic N and tile-count cuts. Reviewed tile widths are
32, 64, 128, 256, and 512. A model may use one width, continuous features only,
or the combined 64+128 family; grouped CV selects the family before the final
development fit. All predicates use exact integer/rational comparisons in
Python and generated C++.

Each bucket defaults to at most 16 bounded tree leaves. The v5 feature schema
permits an explicit ceiling of 32 only for development experiments where
grouped CV proves that the 16-leaf budget is binding; a larger policy is not
promotable merely because its training fit improves. Candidate support must
cover every tail/alignment class assigned to a leaf. Learning aspect boundaries
themselves MAY be introduced in a future feature-schema version, but all
backends must move together.

A generic leaf MUST contain at least two distinct development-training shape
groups and must be exercised by at least one sealed certification shape group;
the production corpus SHOULD target four or more distinct development shapes
per leaf. A rule that merely encodes one measured shape is an exact entry
disguised as a generic policy and is not promotable.

### 10.2 Required cost matrix

The learner consumes the complete eligible candidate latency matrix. It MUST
NOT consume only exact-winner labels. A missing candidate measurement makes
that candidate invalid for any segment containing the missing shape.

### 10.3 Segment objective

For each proposed segment and candidate, compute regret against the
alias-robust runtime exact winner for each development-training shape, alias,
and required execution mode. Choose cuts and candidates using this
deterministic lexicographic objective. Sealed certification rows MUST NOT
participate in any choice:

1. complete required coverage;
2. fewest emitted leaves whose canonical measured nearest-rank p95 is not
   strictly below 5.0%;
3. lowest bounded-fitting nearest-rank p95, measured-only unless grouped CV
   explicitly selects the bounded profiler prior;
4. lowest fixed-order bounded-fitting mean;
5. lowest worst measured leaf p95;
6. fewest measured development points at or above 5.0%;
7. lowest maximum measured regret as a diagnostic tie-breaker;
8. fewest segments;
9. stable candidate and cut ordering.

The hard per-leaf measured-p95 key MUST precede every profiler-informed or
pooled-tree statistic. A tree whose emitted generic leaves all pass p95 is
categorically better than one that would be rejected by final validation, even
when the rejected tree has a lower pooled maximum or mean. Grouped CV then
scores each complete held-out prediction against raw canonical timings; fitted
profiler features never become installation evidence.

Modal exact-winner count, candidate-label accuracy, and family hit rate MAY be
reported as diagnostics but MUST NOT drive promotion.

A deterministic 64-wide beam evaluates every legal threshold for each retained
tree and every exact leaf budget from 1 through 16. The policy remains bounded,
reviewable, and maps identically to all backends. GPU acceleration executes the
same complete objective and structural ordering; it cannot approximate a
threshold, percentile, floating-point tie-break, or signature.

### 10.4 Domain promotion quota and failure to meet 5%

If a bucket cannot meet the generic gate:

- missing production-domain coverage or an empty generic tree always blocks
  promotion as a structural failure;
- an otherwise complete domain is marked as a typed performance exception;
- a non-installable diagnostic records its worst CV cell and complete policy
  provenance;
- fresh route-compatible non-overlay geometries are appended to the immutable
  timing corpus while existing per-variant profiler evidence is reused; and
- the policy is refit until at least 95% of required domains have both measured
  p95 and conservative p95 UCB strictly below 5%.

When the 95% quota passes, the at-most-5% performance-exception set remains
explicit in policy diagnostics and each exception still publishes its best
measured generic tree. The allowance applies only to performance. Missing
domains, uncovered points, empty trees, ambiguous dispatch, unexercised sealed
leaves, byte mismatches, and M/N*K/codebook totality failures remain hard
failures regardless of the passing fraction.

Known-shape exact overlays do not change this result. The learner MUST NOT
relax numerical eligibility or install a partial generic policy to recover
performance.

## 11. Shape-Grouped Development and Sealed Holdout Protocol

### 11.1 Split unit

The split unit is `shape_group_id`, representing one logical model role and
`(N,K)` neighborhood. All rows for that group remain together:

- every candidate;
- every source-format alias;
- every M value governed by that certification surface;
- eager and captured variants; and
- repeated timing/correctness trials.

Random candidate-row splitting is forbidden because it leaks the held-out
shape into training.

### 11.2 Development and sealed certification partitions

Before feature, cut, candidate, or complexity choices are made, logical shape
groups MUST be assigned to:

1. a development partition used for fitting and grouped cross-validation; and
2. a sealed certification partition used exactly once to certify the final
   emitted generic policy.

The assignment is deterministic from a versioned split manifest and seed.
Every candidate, alias, M value, execution mode, and trial for a shape follows
that shape into the same partition. The sealed measurements MUST remain
unavailable to learner selection code until the final generic policy IR and
digest are frozen. Sealed feature rows likewise remain hidden from selection
code; only the predeclared coverage obligations are visible before the policy
is frozen.

The split manifest defines separate certification surfaces for frozen
`Fast M=1` and `VerifierSerialM1Bitwise M=2..16,31`. Their sealed shape
groups SHOULD be disjoint; the verifier sealed groups MUST not have had
verifier candidate performance exposed while M1 was selected.

The sealed partition MUST exercise every generic leaf, aspect/work
boundary neighborhood, required tail/divisibility class, and relevant model
family. A domain with insufficient independent development and sealed shape
groups cannot claim generic certification and blocks the entire publication.

Once a sealed shape has been inspected for a certification decision, it cannot
later be added to development under the same certificate. Doing so requires a
new untouched sealed partition and a new manifest digest.

### 11.3 Development cross-validation and final fit

Development folds MUST be deterministic from the corpus and split-manifest
version.

- Use grouped 5-fold cross-validation when a domain has enough development
  shape groups.
- Use leave-one-shape-group-out validation for sparse development domains.
- Stratification SHOULD preserve prepared family, M coverage, aspect bucket,
  work scale, bundle signature, and model-family diversity without breaking
  group integrity.
- Where several shapes come from one model family, at least one development
  evaluation SHOULD leave the whole model family out.

Development cross-validation selects the feature-schema version, fold-local
rule complexity, work cuts, and candidates. After those choices are frozen,
the learner distills the complete cross-fitted decision surface exactly once
on all development groups. The publication tree may use the reviewed global
leaf bound because the union of independently selected fold boundaries can
require more leaves than any one fold-local tree. This does not relax economy:
every emitted leaf must still pass measured p95, and the resulting generic IR
and digest are frozen before untouched sealed data is opened.

### 11.4 Sealed generic-only evaluation

The exact generic IR that will be emitted is evaluated on the sealed
certification partition:

1. disable every exact overlay;
2. resolve the frozen generic candidate for each sealed runtime key;
3. find that candidate's sealed measured latency in every required execution
   mode;
4. construct the sealed alias-robust runtime exact winner under the same
   semantic contract;
5. run the paired confirmation timing protocol; and
6. aggregate the worst source-format alias and required execution mode.

A candidate not measured on a sealed point is uncovered, not zero-regret. For
generic candidate `g` and runtime-representable sealed exact winner `cexact`:

```text
generic_regret(g,s,q) =
    max_(f,x) (L(g,s,f,x,q) / L(cexact,s,f,x,q) - 1)
```

The denominator is the exact candidate the runtime can actually dispatch for
the shared alias/mode key, not a different unattainable per-format candidate.

No learner parameter, segment cut, or candidate may change after sealed
evaluation. A failure requires a new policy attempt and a new untouched sealed
partition; the failed sealed shapes become development evidence only in that
later attempt.

The sealed certificate MUST record the pre-open
`frozen_generic_policy_digest` and the validator MUST prove it matches the
generic section of the emitted IR byte-for-byte.

Exact production overlays are constructed separately and MAY use all accepted
exact-shape measurements, including sealed shapes, because exact lookup is
bypassed during generic certification. They MUST NOT alter or retrain the
frozen generic IR.

### 11.5 Acceptance statistic

For every promoted generic bucket:

```text
sealed_coverage == 100%
sealed_p95_observed_worst_surface_regret < 0.05
sealed_p95_simultaneous_95pct_upper_regret < 0.05
bitwise_failures == 0 for VerifierSerialM1Bitwise
```

The summary MUST report development-CV metrics separately from sealed
certification metrics, including median, p95, maximum, simultaneous confidence
bound, shape count, alias count, execution-mode count, fold count, and every
near-threshold cell.

After sealed certification, the common IR is emitted without refitting. The
C++ resolver round-trip proves that the compiled selector emits the exact
frozen decisions; the sealed timing certificate proves their performance.

## 12. Generated Artifacts and Provenance

### 12.1 Common policy IR

The learner emits backend-neutral policy IR before C++ generation:

```cpp
struct ExactDispatchEntry
{
    SemanticContract contract;
    OperationKind operation;
    BundleSignature bundle;
    ProjectionNVector projection_n;
    PreparedFamilyId prepared_family;
    uint8_t codebook;
    uint8_t m;
    int aggregate_n;
    int k;
    CandidateId candidate;
    ArithmeticFingerprint arithmetic;
};

struct GenericDispatchRule
{
    SemanticContract contract;
    OperationKind operation;
    BundleSignature bundle;
    PreparedFamilyId prepared_family;
    uint8_t codebook;
    uint8_t m;
    AspectBucket aspect;
    int64_t max_work_items;
    CandidateId candidate;
    ArithmeticFingerprint arithmetic;
};
```

Backends MAY encode these as constexpr arrays, switches, or generated helper
functions, but emitter round-trip tests must resolve the same candidate as the
IR for every test key.

The common IR metadata MUST carry a per-contract
`frozen_generic_policy_digest`. The sealed certificate binds to that digest,
and the digest covers only generic decisions and their feature schema. It is
distinct from the final policy/bundle digest, which also covers sealed-derived
exact overlays and other artifact metadata.

### 12.2 Installable artifact set

Each backend policy installation consists of:

1. the generated `.inc` consumed by production;
2. a machine-readable `.policy.json` containing exact and generic IR;
3. an acceptance `.manifest.json`; and
4. immutable profiler request/evidence sidecars plus retained raw profiler
   artifacts for every supported candidate observation;
5. an authenticated dispatch-level `*_profiler_features.csv` joining canonical
   timing with the separately collected profiler evidence; and
6. a human-readable summary.

These files live inside a versioned immutable bundle. A bundle index records
all backend/architecture artifacts it contains and is the target of the single
atomic active-pointer switch defined in Section 15.

The manifest MUST contain:

```text
schema_version
policy_abi
learner_version
feature_schema_version
backend and architecture_class
git/build/compiler identifiers
candidate_registry_hash
format_registry_hash and packing_abi_hash
corpus_hash and trial-set hash
profiler request/evidence/metric-set hashes and complete coverage counts
serial_m1_policy_hashes
per-contract frozen_generic_policy_digest
exact-entry count
generic-rule count
required-key coverage
development-CV and sealed-split manifest hashes
development-CV metrics
sealed median/p95/max/simultaneous-UCB regret
rejected, uncovered, and performance-unpromoted buckets
bitwise certificate counts
emitter round-trip result
staged test results
generation timestamp
```

The `.inc`, policy IR, and manifest MUST share one policy digest. Every
generated `.inc` MUST embed constexpr metadata for the policy ABI,
architecture class, policy digest, candidate-registry hash, format-registry
hash, packing-ABI hash, per-contract frozen generic policy digests, and
relevant serial-M1 oracle hashes. The compiled resolver MUST compare those
values with the compiled/runtime identities before using the table. A JSON
manifest alone is not runtime validation.

Production builds MUST expose the active digest in version output or debug
diagnostics.

### 12.3 Base artifacts

Partial refreshes merge measurement corpora and retrain the policy. They MUST
NOT layer new exact entries over an old opaque generic `.inc` and then report
metrics for a different newly inferred fallback.

If an unchanged base generic policy is intentionally retained, evaluation must
load and score the actual retained IR. A base include without matching IR,
manifest, and compatible oracle hashes is not mergeable.

## 13. Runtime Resolution, Failure Policy, and Telemetry

### 13.1 Lookup order

Runtime resolution is:

```text
1. validate policy ABI/backend/architecture/contract
2. exact overlay lookup
3. certified generic aspect/work lookup
4. certified floor for the declared domain
5. fail closed
```

Verifier resolution MUST never call the unrestricted fast resolver as step 4.

### 13.2 Launch validation

Before launching, the backend MUST validate:

- candidate support for the concrete operation, bundle signature, ordered
  projection-N vector, aggregate N, M, K, and execution mode;
- workspace availability;
- stream ownership;
- graph-capture compatibility;
- arithmetic fingerprint compatibility for verifier calls; and
- absence of an unapproved runtime override.

An unsupported emitted candidate is a generator/runtime contract failure, not
an invitation to choose another heuristic silently.

### 13.3 Concurrency

Semantic mode MUST be per request/call/plan. Process-global verifier flags are
not permitted in the target architecture. Nested and concurrent verifier/fast
calls on different host threads and streams MUST resolve independently.

### 13.4 Route telemetry

Every grouped verifier dispatch SHOULD record, through the route-record design
in [Perf Stats Collector](../2026-06/PERF_STATS_COLLECTOR.md):

```text
backend
semantic_contract
operation_kind
bundle_signature
projection_n_vector
aggregate_n
prepared_family/codebook
M,K
candidate_id and effective_candidate_id
match_kind = exact | generic | certified_floor
policy_digest
frozen_generic_policy_digest
serial_m1_policy_hash
execution_mode = graph | eager
```

Tests MUST assert route telemetry so byte-equal output from an accidental row
loop or different kernel does not masquerade as a grouped-policy proof.

## 14. Backend Migration Requirements

### 14.1 CUDA

CUDA migration MUST:

1. Replace the independent decision-tree and aspect/overlay fitting paths with
   the common learner.
2. Retain every candidate's timing row rather than relying only on `is_best`.
3. Emit explicit byte and serial-M1-policy evidence for verifier candidates.
4. Normalize candidate settings under the verifier contract before assigning
   `effective_candidate_id`; nominal target-wave, KB, or two-phase settings
   that launch identically must not become different labels through timing
   noise.
5. Prove that every emitted tile/CPT/family tuple is supported by the grouped
   production dispatcher. Current support includes the repaired `256x4` KPAR
   route; regression coverage must prevent future generated/runtime drift.
6. Encode optional ROWPAR preparation as an explicit candidate resource
   predicate. The current verifier path suppresses lazy ROWPAR promotion by
   passing a null row-major slot; that behavior must remain covered unless a
   future ROWPAR verifier candidate uses the same immutable prepared
   representation as M1 and passes the complete byte gate.
7. Preserve explicit non-default stream timing, prepared-weight reuse, and
   workspace sizing through `IWorkspaceConsumer`.

Primary migration surfaces:

- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp`
- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativeVNNIDecodeTrainer.cpp`
- `tests/v2/performance/kernels/cuda/gemm/analyze_cuda_native_vnni_decode_trainer.py`
- `tests/v2/performance/kernels/native_vnni_dispatch/candidate_registry.py`
- `tests/v2/performance/kernels/native_vnni_dispatch/compiler.py`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc`

### 14.2 ROCm

ROCm migration MUST:

1. Preserve the full candidate-by-shape latency matrix instead of collapsing
   immediately to the fastest exact row.
2. Replace modal KB/target-wave selection and label-hit metrics with the common
   regret learner and sealed-certification gate.
3. Replace cosine/tolerance eligibility with byte equality against production
   serial M1 for verifier candidates.
4. Make the real serial M1 launcher and grouped resolver consume one canonical
   policy source; duplicated fallback heuristics are prohibited.
5. Replace the process-global verifier mode with the explicit semantic
   contract.
6. Reject atomic reduction and arithmetic-changing environment overrides for
   verifier calls unless the exact effective route is separately certified.
7. Include Q8 and IQ special routes in the same candidate registry and policy
   surface rather than retaining handwritten dispatch exceptions.

Primary migration surfaces:

- `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Throughput.cpp`
- `tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`

### 14.3 CPU

CPU migration MUST:

1. Replace exact-only generation with the same exact-plus-generic common IR.
2. Benchmark explicit forceable candidates instead of inferring the executed
   policy from an `Auto` phase name.
3. Give Pairwise, Wide3, Wide4, K-parallel, fused, and any other selectable
   implementation stable candidate IDs and route counters.
4. Compare every verifier candidate with production serial M1 using bytes and
   emit the serial-M1 policy hash.
5. Route single projection, fused projection, fused gate/up/down, and routed or
   shared MoE paths through the same semantic resolver. A fused path may not
   bypass the learned policy.
6. Add generic aspect/work rules and grouped development/sealed data, including CPU
   topology and tail-sensitive shapes.
7. Preserve Pairwise only as a certified floor where it is genuinely grouped
   and performance/correctness evidence covers the domain.

Primary migration surfaces:

- `tests/v2/performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_GEMV.cpp`
- `tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc`
- CPU fused and MoE callers in `CPUNativeVNNIGemmKernel.h`

## 15. Transactional Retune and Installation

### 15.1 Profile classes

Profiles are classified as:

- `quick`: script/schema smoke only;
- `family-smoke`: bounded candidate/family workflow proof;
- `partial-production`: complete evidence for a declared subset, merged into a
  complete compatible corpus; or
- `production`: complete required inventory and installable evidence.

`quick` and `family-smoke` MUST reject `--install`.

### 15.2 Required transaction

An installable refresh performs these steps in order:

1. **Preflight**
   - identify backend/architecture targets;
   - freeze schema, learner, feature, policy ABI, and candidate registry;
   - validate required shape/format/M inventory;
   - freeze the development/sealed split manifest without exposing sealed
     measurements to the learner;
   - create an isolated staging directory.
2. **Collect Fast M1 development timing**
   - run explicit `Fast M=1` candidates with generated policy disabled;
   - retain immutable canonical timing and raw samples for development shapes;
   - adapt and validate the development common-observation corpus.
3. **Collect isolated development profiler evidence**
   - derive immutable requests from the validated development observations;
   - emit one request for every exact launchable physical point, deduplicating
     only aliases with identical backend/ISA/operation/bundle/prepared-family/
     packing-ABI/codebook/effective-candidate/mode/M/ordered-N/K identity;
   - launch each requested physical effective candidate separately with profiling
     disabled during setup/warmup and enabled for one target launch per pass;
   - retain complete static/dynamic metrics per ordered physical dispatch;
   - export the authenticated dispatch-level feature table only after complete
     evidence validates against the immutable canonical timing corpus;
   - reject missing supported-candidate records or required metrics;
   - keep all canonical timing aggregates and raw samples unchanged.
4. **Fit, freeze, and certify Fast M1**
   - fit the profiler-informed generic policy with grouped development CV;
   - freeze and publish its IR/digest before opening sealed timing;
   - evaluate that exact policy on the sealed partition without refitting;
   - compute observed p95 regret and p95 simultaneous-UCB regret per domain,
     require both strictly below 5% in at least 95% of domains, and retain all
     exception/global/maximum values as diagnostics;
   - compile, emit, and build the staged production M1 policy;
   - freeze its policy, packing, and arithmetic hashes.
5. **Collect dependent policies**
   - run verifier `M=2..16,31` candidates against the exact frozen staged M1
     artifact;
   - collect common-schema observations and raw timing artifacts;
   - verify route, stream, workspace, and correctness fields.
   - allow architecture corpora to be collected on separate hosts while
     preserving one schema, manifest, and artifact transaction.
   - restart dependent collection if any M1 input or hash changes.
6. **Validate corpus**
   - reject missing required keys, aliases, candidates, M values, or trials;
   - reject stale M1 hashes and unstable measurements.
7. **Compile and freeze dependent generic policies**
   - filter eligibility;
   - construct alias/mode-robust runtime exact winners;
   - use grouped development CV to select learner complexity;
   - fit the final generic policy on development shapes only;
   - freeze its IR/digest before opening sealed measurements;
   - evaluate that exact policy on the sealed partition with exact overlays
     disabled;
   - require complete structural coverage and the strict per-domain p95 `<5%`
     sealed gate in at least 95% of required domains;
   - write final IR/manifest without refitting the generic policy.
8. **Collect sealed profiler evidence without refitting**
   - open sealed profiler requests only after the generic IR/digest is frozen;
   - require complete per-candidate evidence, but do not alter policy decisions
     from sealed profiler counters.
9. **Emit immutable bundle**
   - generate every selected backend include from the accepted IR;
   - embed the required runtime metadata;
   - validate IDs, sorting, reachability, and deterministic output;
   - assemble a complete immutable bundle identified by policy digest.
10. **Stage build**
   - build Llaminar against the staged includes without modifying checked-in
     artifacts;
   - run Python-to-C++ resolver round-trip tests.
11. **Correctness acceptance**
   - enumerate exact entries and generic leaves/boundaries;
   - run all-format grouped byte parity and model operation equivalence.
12. **Performance acceptance**
   - validate the emitted generic policy against the sealed certificate;
   - run relevant full-model benchmarks outside profiling mode.
13. **Atomic publish**
    - under one install lock, rename the completed staged directory into a
      versioned immutable bundle location;
    - atomically rename one `active` symlink or pointer file from the old
      complete bundle to the new complete bundle;
    - record old/new digests in the transaction journal.

A failure before the single active-pointer rename leaves the active production
bundle unchanged. A crash after it exposes either the complete old bundle or
the complete new bundle, never a mixture. Rollback atomically repoints
`active` to the prior immutable bundle. The build include path must resolve
through this active bundle; sequential copying of several destination files is
not an atomic install and is forbidden.

`--skip-sweep` is acceptable only when the supplied corpus passes schema,
build, hardware, trial-set, and provenance digest checks. It is not a way to
reuse an unverifiable CSV.

### 15.2.1 Turnkey Git LFS corpus workflow

The supported operator entry point is:

```bash
scripts/train_native_vnni_dispatch.sh --backend cuda --install
scripts/train_native_vnni_dispatch.sh --backend rocm --install
scripts/train_native_vnni_dispatch.sh --backend cpu --install -- \
  --cpu-format-shards
scripts/train_native_vnni_dispatch.sh --backend cpu-prefill --install -- \
  --cpu-format-shards
```

CPU measurement lanes default to every detected physical socket. Each lane
owns a distinct resumable cell and uses the production thread count for that
socket; the MPMD launcher never duplicates one timing cell across ranks. Use an
explicit `--cpu-measurement-lanes N` only for controlled isolation or A/B work.
CPU-prefill sealed collection also coalesces source formats that share one
geometry, ISA regime, and ordered M inventory. The trainer still packs and
times every format independently, but one process reuses byte-identical Q8_1
activation fixtures and avoids repeated MPI/process startup. MPMD pairing is
partitioned by both M inventory and format-phase count, with an independent
runtime handshake before the first collective.

The driver rebuilds both CUDA and ROCm policy scorer DSOs, rebuilds the selected
backend trainer, runs scorer equivalence integration tests, and computes the
resolved shared shape-inventory digest. It selects a corpus by backend,
architecture, and that digest. If no matching generation exists, it resumes an
ignored staging collection, performs fitting and sealed certification, and only
then atomically moves the sealed generation under
`benchmark_results/native_vnni_dispatch/corpora/`. If the generation already
exists, it pulls LFS objects, verifies every payload SHA-256, materializes a
disposable copy-on-write fit workspace, and runs `--skip-sweep
--reuse-profiler-evidence`; this path launches neither candidate kernels nor
profilers.

The corpus path also includes a digest of forwarded collection options, so
different format inventories, lane counts, or profiler tools cannot collide.
The turnkey command rejects `--shapes` and `--shape-partition`: production exact
overlays are added once to the shared inventory, never as backend-local command
line subsets.

`corpus.manifest.json` remains ordinary reviewable Git JSON. Corpus CSVs,
profiler JSON, raw Nsight/rocprof/perf reports, and compressed archives are
tracked by Git LFS through `.gitattributes`. A pointer-only checkout, symlink,
`.inprogress` file, changed inventory source, missing payload, or digest mismatch
is a hard failure. Fit caches and failed staging directories are not corpus
evidence and remain ignored.

For `--profile all`, immutability alone is insufficient. The corpus sealer and
verifier require the backend's complete development and timing sidecars,
frozen/sealed evidence, per-candidate development and final profiler manifests,
certified policy JSON, and generated include. A bounded collection checkpoint
therefore cannot be published or reused as a production corpus even when every
file it happens to contain has a valid digest.

The exact-overlay inventory is backend-independent. Python policy compilers and
all three C++ trainer targets consume the one resolved inventory returned by
`native_vnni_dispatch.shape_manifest`; CMake centralizes its checked-in shape
manifest and Qwen release catalog paths. Backend target blocks must not define
private model geometry lists. Adding a release geometry changes the resolved
inventory digest, so the turnkey command selects a new corpus generation and
cannot silently reuse evidence that lacks the overlay.

CPU prefill additive migrations retain their complete historical provenance in
fit-only mode. Invoke the refresh executor with
`--cpu-prefill-fit-development-lineage-plan` and provide the immutable source
aggregate/timing, source route manifests, ordered source refinement plans, and
every distinct source split named by those plans. The analyzer matches each
refinement plan to its embedded split digest before authenticating it; a mixed
v8/v9 history is never reinterpreted as though every round belonged to v9.
These lineage arguments are forwarded through adaptation, development fitting,
freeze, and sealed certification, with the stable combined lineage CSV and
timing sidecar serving as the v10 development input.

The complete pure-Python transaction toolchain is registered in the canonical
CTest unit suite. This includes corpus bundling, turnkey execution, refresh
orchestration, shape geometry, compatible-CSV merging, measurement planning,
CPU lineage/refinement/route/checkpoint handling, backend adapters, profiler
evidence, fitting, and policy generation. Run it by contract labels rather than
maintaining a test-name allowlist:

```bash
ctest --test-dir build_v2_integration \
  -L Unit -L NativeVNNI \
  --output-on-failure --parallel
```

Every new NativeVNNI Python unit module MUST be registered with both labels.
CUDA and ROCm scorer execution remains in the explicit integration gate; no
Python unit test launches GPU work.

Each isolated profiler transaction stores an exact compact
`*_profiler_observation_witnesses.csv` beside its requests, evidence, features,
and raw reports. Feature re-export authenticates against this witness rather
than requiring the much larger source aggregate. For older transactions that
predate the witness artifact, `profiler_evidence compact-witnesses
--feature-table ...` recovers the strict common-observation columns embedded in
the feature table, verifies each recorded observation digest, deduplicates
multi-dispatch rows, and re-authenticates the result against request/evidence
manifests. It never trusts the feature table's derived metric values.

#### Turnkey replay defect journal

The July CPU-prefill installation replay exposed the following workflow
defects. These are pipeline bugs, not operator errors, and remain tracked here
until the one-command path owns a focused regression for each item.

| Defect | Observed failure | Required automatic behavior |
|---|---|---|
| Print-only dry run | `--dry-run` printed a plausible transaction that later failed after route probes because no plan-to-source relationship had been authenticated. | A real preflight must load and authenticate every recipe artifact before any route probe, scorer fit, timing launch, or profiler launch. Dry-run output must say whether it is merely rendering commands or running authenticated preflight. |
| Historical/current plan role conflation | A historical `full-expansion-r17.5pct.round1` plan was accidentally passed as a current v10 generic-refinement plan and changed the expected route manifest. | The recipe schema must store historical lineage plans separately from current additive refinement plans. Cross-role duplicates are invalid. |
| Duplicate split provenance | The v9 source split was supplied both as the primary source split and as an additional historical split. | Preflight must reject duplicate split digests and report both recipe fields before fitting. |
| Candidate-expansion source ambiguity | The v4 expansion plan was initially paired with the later r17 aggregate instead of its authenticated r16 source. | The recipe must bind the expansion plan, source aggregate/timing pair, and expansion aggregate/timing pair as one typed transaction and verify their embedded digests. |
| Profiler anchor relabeling | Regenerating one combined request manifest selected different anchor geometries for existing candidates and would have mislabeled old counters. | Final profiler publication retains every physical launch's actual anchor geometry. It must never regenerate canonical timings or relabel old measurements. |
| Representative profiler points poisoned exact work features | The legacy full-K family was profiled at `M64,N8960,K1536`, while the later pair-grid family was profiled at `M4096,N896,K896`. Matching candidate anchor sets would remove family/geometry correlation but would still broadcast one launch's dynamic behavior across unrelated M/N/K points. | Request schema v4 profiles every exact launchable `(backend, ISA, operation, bundle, prepared family, packing ABI, runtime codebook, effective candidate, mode, M, ordered N, K)` point. Only true aliases deduplicate. Evidence composition rejects mixed schema generations and incompatible identities; historical representative-anchor catalogs remain provenance-only. |
| System-wide CPU `perf` control admitted unrelated work | A nominal 69-microsecond Q4_K launch reported roughly 642 million cycles, 124 million instructions, and 311 milliseconds of task clock. The target launched once, but external FIFO control polling left system-wide counters enabled after it returned. | The trainer owns per-TID `perf_event_open` groups for the exact persistent OpenMP team, arms and disarms them concurrently around one launch, and rejects multiplexing. Focused regressions and A/B probes validate `launches=1`, exact request identity, scheduled-time coverage, repeat stability, and duration-feature reliability before broad collection. |
| Completed checkpoint treated as a prefix | A final round-12 common-observation checkpoint was treated as an intermediate prefix merely because current refinement plans were present, then failed because it was not the source of a nonexistent round 13. | Checkpoint state and represented boundary must be explicit in the recipe. `complete` checkpoints are authenticated against the complete transaction; only `prefix` checkpoints may be extended, and their exact input count/source-plan boundary is mandatory. |
| Historical expansion reinterpreted under current code | Rebuilding a complete checkpoint from raw candidate-expansion files failed the implementation-digest guard after the adapter evolved. Weakening that guard would silently reinterpret historical timing. | A content-authenticated complete checkpoint must be rebased by validating its candidate normalization and full plan partition, then changing transaction provenance only. Raw historical expansion is never re-adapted under a changed implementation. |
| Checkpoint/expansion generation mismatch | The complete checkpoint embedded normalization plan `376391...`, while the first recipe draft named another v4 plan with the same source corpus but different reviewed rebase identity. | Preflight must scan the checkpoint's embedded expansion provenance and require its exact plan digest before loading the full checkpoint. Similar filenames or equal source hashes are not identity. |
| Hand-assembled replay command | Final replay required manually ordering dozens of aggregate, timing, route, split, plan, profiler, and checkpoint arguments. Several individually reasonable combinations were semantically wrong. | One machine-readable replay recipe must own the ordered lineage. `train_native_vnni_dispatch.sh` and the refresh executor must consume it directly so collection, fit, certification, and installation are one command. |
| Authentication after expensive setup | Some invalid combinations reached repeated route probes and scorer setup before failing. | Recipe and artifact preflight must be the first backend-specific action. Expensive work starts only after a single success summary identifies the authenticated source, additive rounds, profiler transactions, and checkpoint state. |
| Serial canonical identity and cache setup | A cached 216,632-row replay used roughly one host core for corpus JSON canonicalization and 756 independent domain cache identities; the coarse `candidate_costs` timer hid almost 100 seconds of setup. | Corpus row serialization/sorting and domain cache identities run in deterministic process partitions, reduce in canonical order, preserve historical SHA-256 bytes, and are covered by serial/parallel equality tests. Timing reports identity, key, profiler, cache lookup, and actual cost phases separately. |
| Duplicate raw evidence hashing | Recipe preflight authenticated multi-gigabyte aggregate/timing files, then every fit/freeze/certify process serially streamed the same raw transaction again (13.4 seconds per invocation). | Preflight emits the authenticated raw corpus identity and analyzer development contexts consume it directly. Prefix and sealed contexts retain their independent identities and are never relabeled. |
| Synthetic refinement exact overlays | `CPUPrefillAutoRefine_N512_K288` correctly trained a generic boundary but was also emitted as an exact overlay, so freeze demanded a production serial-route record for a diagnostic-only geometry. | Generic learning consumes every authenticated refinement row. Exact publication is independently filtered to declared production shape/M cells and uses canonical production shape names only. Recipe preflight separately proves the target route manifests cover the current production matrix. |
| Fixed sealed geometry under-covered expanded trees | The original eight sealed shapes could not exercise 1,489 of the 2,579 generic leaves after candidate and feature expansion, including every serial-K partition. | Freeze selects one untimed physical route witness per frozen leaf from authenticated target routes plus diagnostic-only completion probes. The self-contained plan embeds the exact C++ route, rejects any generated geometry visible to development, and is replayed semantically before collection and certification. The current plan covers all 2,579 leaves with 1,859 records over 641 selected geometries. |
| CPU decode static holdout could not certify learned leaves | The CPU M=1 V4 holdout exercised only 39 of 108 domains and missed 143 of 352 frozen leaves. Its separately timed candidate blocks also made a global unpaired Bonferroni UCB far wider than the small schedule differences under test. Grouped decode inherited the same fixed-partition weakness. | CPU M=1 and grouped decode now commit to fresh post-freeze reserves. M=1 generates unseen geometry candidates per leaf, probes the real C++ full-K/K-part route on all three ISA regimes, then measures the selected schedule against every forceable challenger and source alias in interleaved pairs. Grouped decode does the same for every frozen M leaf with the production Pairwise/WideRows launcher and complete `M*N` byte equality against production serial M=1. Static CPU sealed CSV inputs are rejected; request shards are architecture-homogeneous, resumable, and reduced in parallel at physical-core width. |
| Sealed cost discovered only at installation | Development fitting optimized grouped CV without continuously projecting the independent certification transaction. Only after the final freeze did the adaptive witness planner reveal 641 geometries, 1,859 source-format records, 2,941 format/M cells, and 2,579 leaves requiring fresh measurements. The exact witnesses correctly depend on the frozen tree, but their scale and cost should not have been an installation surprise. | Before final refinement, authenticated preflight must report provisional leaf count, untouched-reserve reachability, projected witness geometries/records/cells/process jobs, and calibrated wall time for every ISA regime. The reserve commitment and coverage obligations are fixed before fitting, while timings remain unopened until the generic digest freezes. Promotion cannot proceed without a complete reserve and an explicit collection-budget acknowledgement. |
| Interpolation-only CV certified boundary failures | The 216,632-row development fit passed 99.07% of domains, but its first complete untouched seal passed only 405/756 domains: 53.57%, with 40.23% global p95 observed regret. Coverage was 2,579/2,579 leaves and bitwise failures were zero. Failures were broad across ISA, M, and codebook but concentrated at K=32/64 below the development minimum K=192. Feature-sorted round-robin folds also allowed another shape name at the same N/K to remain in training. | Cross-validation treats exact N/K geometry as the atomic group, not the shape label. Larger domains are partitioned into coherent balanced spatial regions, so local neighborhoods and support boundaries leave development together; small domains use leave-one-geometry-out validation. The fold schema is part of cache identity. Interpolation-friendly neighbor spreading is not accepted as evidence for generic unseen-shape dispatch. |
| Burned seal could only be reused by provenance relabeling | After the failed seal was inspected, its 38,233 adapted candidate rows became valuable development evidence, but ordinary additive inputs would have assigned the historical development build/runtime identity to the fresh sealed measurements. Re-adapting the 1.9 GiB timing sidecar for every fit would also make experimentation unnecessarily expensive. | A content-addressed burned-seal development transaction binds aggregate, timing sidecar, old witness plan, and the exact measuring git/build/compiler/host/runtime/serial-policy provenance. Replay validates that old sealed matrix independently, composes original corpus-ID partitions without relabeling, materializes a reusable mixed checkpoint before fitting, and requires every changed policy to use a new untouched seal. The replay recipe retains these transactions as a typed v4 field. |
| Paired CPU seal ratios had no generic-only replay path | A complete M=1 seal failed promotion but retained 1,286 byte-exact selected/challenger edges over 82 shards. Treating those rows as ordinary observations could create exact overlays at opened holdout geometry; discarding them would force the generic learner to repeat known boundary mistakes. | CPU M=1 and grouped analyzers authenticate each old plan plus complete paired directory, replay every request field, and convert ratios only into supplemental candidate costs for existing generic domains/candidates. Burned geometry cannot enter exact overlays or the next reserve, and unseen-point profiler descriptors retain static resources while masking request-local dynamic counters. Repeatable positionally paired wrapper options carry every burned generation through freeze, fresh planning, and certification. |
| Historical paired refinement omitted from a manual refit | A burned-seal replay was initially launched with the broad corpus and burned ratios but without the 173 retained paired-development shards that produced the 99.07% baseline. The command was syntactically valid and would have spent accelerator time fitting a weaker, non-comparable policy. | The turnkey transaction owns the complete paired-refinement inventory and appends it to freeze, fresh-plan generation, and certification together with every burned generation. Focused command-graph regressions require plan/directory pairing and prove all evidence arguments reach every generation step. Direct diagnostics must explicitly enumerate retained paired shards. |
| Burned paired seal reached freeze but not refinement | The first corrected CPU M=1 replay let the paired planner declare the old broad corpus green, then introduced the failed seal only during freeze. Freeze therefore solved a different cost graph and failed after another full fit. | Burned plan/directory pairs are authenticated before refinement, contribute supplemental generic costs and digest identity to every planner iteration, and are never redundantly requested. The exact same burned transactions reach planner, freeze, fresh planning, and certification. |
| Additive geometry invalidated burned-seal replay | Adding the V7 CPU M=1 geometry correctly changed the current development digest, but burned M=1 and grouped loaders required it to equal the older seal plan's source digest. The valid historical transaction then failed before fitting even though its candidate registry, format registry, requests, paired rows, and target generic domains were unchanged. | A burned plan is self-authenticated against current registry generations and every paired row is replayed against its immutable request. Witnesses must resolve to exactly one compatible generic domain in the enlarged corpus. Whole-corpus digest equality is required for a live seal, never for generic-only burned evidence consumed by a later additive generation. Focused M=1 and grouped regressions use changed development digests. |
| Minimax exact candidate hid the failing source alias | One CPU M=1 cell selected `nbc1`, reported 11.29% broad regret, and named minimax compromise `nbc8` as its exact candidate. The planner repeatedly measured `nbc1` versus `nbc8`, although the Q4_1 alias's actual regret witness was `nbc4`; six completed paired runs could not change the reported failure. | Paired refinement reconstructs corrected effective latency independently for every source alias and requests the candidate with the greatest advantage over the selected route. A completed direct edge that still contradicts the fitted graph is a terminal evidence conflict, not a corpus-digest-dependent request for the same launch pair. Focused regressions cover both minimax/exact disagreement and conflict termination. |
| Pairwise refinement composed incompatible process histories | The exhausted CPU M=1 generation reached iteration 39 with one impossible cycle at AVX512-build/AVX2-runtime Q4_0 `3072x6144`: separately gathered edges implied both `nbc1 ~= nbc2` and an indirect 23% advantage for `nbc2`. Focused replays found the same pair stable within each socket but changed its old ranking, while a complete four-edge cycle gathered in one invocation closed within 2.5% on both sockets. After the repair, the next refit emitted 145 requests over 44 runtime cells; an explicit identity audit found zero overlap with the 991 cells in the first complete-star generation, proving these were newly exposed surfaces rather than replacement anchors. | Every failing runtime cell now requests a complete direct star from the selected launch to every forceable physical candidate in its first refinement generation. Content-addressed sharding keeps that cell indivisible in one producer process; the request limit is a packing target, never permission to split a tournament. The first complete directional star is immutable cell evidence: later fits derive every candidate ratio from it and cannot create another star. Indirect connectivity across sessions is not accepted as a substitute. CPU M=1 and grouped decode share this rule, and focused planner/sharder regressions lock it in. The inconsistent cross-session generation is diagnostic-only and cannot be installed. |
| Coarse K-part geometry skipped a 32-wide route boundary | A complete all-format 5-by-5 neighborhood around the 3B FFN-down geometry stepped N and K by 64. Remaining failures switched among `nbc1`, `nbc4`, and `nbc8` inside those cells, so the learner had no observations at the 32-value alignment and tail boundaries exposed by its own features. An isolated `3072x4096` miss had the same local-supervision problem. | Generic development geometry includes dense all-format, all-ISA neighborhoods at every route-relevant alignment scale. The V7 CPU M=1 family adds the 16 missing 32-step inner-lattice points around `2048x11008` plus eight non-overlay neighbors around `3072x4096`; manifest regressions lock the exact geometry, partition, and aspect inventory. |
| Grouped decode skipped paired development | The grouped workflow adapted broad all-format timings and went directly to generic freeze plus a fresh seal, while M=1 already used iterative isolated selected/challenger ratios. A failed grouped seal would have been the first reliable indication that the broad winner graph was wrong. | `paired_requests --surface grouped-verifier` is a mandatory resumable gate before grouped freeze. It uses the production byte-exact grouped launcher, retains content-addressed per-ISA shards, and passes the complete paired-development set to both freeze and certification. |
| Suspected CPU tree-capacity miss was not capacity-bound | After the paired graph was nearly exhausted, six domains remained above budget. A 32-leaf diagnostic kept all six accelerators busy for 56 minutes, yet every failing domain selected only 1-7 leaves and the failed-cell count rose from 17 to 20. | Do not raise the production ceiling unless diagnostics show the selected model exhausted it. Preserve this negative experiment in the fit cache, return production to 16 leaves, and investigate candidate-edge evidence, fold geometry, and feature/model selection for these domains. |
| Global leaf-cap expansion recomputed passing domains | Raising CPU M=1 from 16 to 32 leaves treated the larger ceiling as an unrelated tournament, including domains already promotable at the smaller horizon. | If a future domain truly exhausts the bound, persist complexity-prefix CV state and escalate only failing domains. A mixed-complexity result must retain deterministic global ordering and one policy digest, and a cold full search must produce byte-identical rules and diagnostics to the incremental path. |
| Grouped freeze repeated raw adaptation | The grouped transaction first published an authenticated common-observation checkpoint, then freeze and certification each converted the large aggregate/timing corpus and rebuilt normalized profiler descriptors again. M=1 already reused both caches, leaving grouped decode slower and operationally asymmetric. | Grouped freeze and certification use the same `--reuse-development-common` contract and persistent `profiler_feature_catalog_v1.json` as M=1. Every common row must match the requested raw corpus digest and full git/build/compiler/host/runtime/serial-policy provenance, including its ISA-specific build suffix; successful reuse does not rewrite the checkpoint. |
| Grouped adaptation remained serial | The grouped transaction shared the accelerator-backed learner with M=1, but exact timing-sidecar parsing and common-observation construction still ran on one host core. Large grouped corpora therefore entered the fast learner through a serial front end. | Large grouped timing files are split at byte-safe record boundaries across affinity-visible physical cores, reduced in file/range order with global sample-index validation, and aggregate observations are adapted into private canonical CSV shards before atomic ordered assembly. Small corpora remain serial below the reviewed threshold. A focused regression proves serial and parallel timing maps, observations, and common CSV bytes are identical. |
| Profiler scaling signal partitioned by exact M | The first geometry-coherent fit had complete coverage but only 554/756 domains passed; all 202 failures were multi-candidate full-K domains. The profiler surrogate already carried `log2(M)` and arithmetic-intensity features, yet exact M was part of its transfer key, so seven disconnected forests could not learn how the same physical candidate scales from M=64 through 16384. | Prefill profiler transfer is cross-M and cross-format while retaining hard architecture, arithmetic-contract, operation, bundle, and execution-mode boundaries. Held-out N/K geometry is removed from every M and format in the pool before fitting. M remains an explicit numeric feature, pool-local cache identity covers every dependent M domain, and regressions prove both cross-M sharing and cross-M leakage exclusion. |
| Cross-M pooling collapsed host occupancy | The first cross-M replay reduced 42 pools to six but scheduled one worker per pool. Only three expensive full-K processes remained active while most of the 48-core host sat idle. | Build each pool's immutable normalized feature index once, share it copy-on-write, and schedule each independent held-out forest as its own deterministic process task. A focused regression proves one pool occupies multiple configured workers and serial/parallel prediction bytes remain equal. |
| Profiler model preprocessing left every accelerator idle | Before the 216 cross-M forest surfaces could run, one parent eagerly hashed singleton observation aliases and materialized 132,939 runtime/profiler feature records. The latter stage took more than four minutes on one core. | Alias selection compares cheap source/mode/candidate fields first and hashes only true ties. Large model-record pools use deterministic contiguous physical-core shards inherited through `fork`, publish private result files, and merge in source order. Serial/parallel record order and values are regression-equal. On the 158,760-row CPU M=1 corpus, observation indexing is 2.55 seconds and all six record pools complete in 19.8 seconds instead of more than four minutes. The same implementation is mandatory for grouped decode. |
| Profiler workers defaulted to SMT width | Surface-level scheduling initially launched 112 ExtraTrees processes on a 56-physical-core host, doubling process state and run-queue contention for CPU-bound estimators. | Every policy CPU process pool defaults to unique affinity-visible `(physical_package_id, core_id)` identities. Incomplete topology falls back conservatively to visible logical CPUs, explicit worker variables remain available for A/B diagnosis, and a synthetic two-socket SMT regression locks in the physical-core count. |
| Dynamic profiler counters treated as shape-independent inputs | Normalized IPC, cache, throughput, achieved occupancy, traffic, and duration-derived rates from one representative launch were attached to a candidate at every runtime geometry. Even with exact profiling, training on dynamic values while masking them for held points creates a missing-feature distribution shift. | Dynamic descriptors are keyed by complete exact launch identity and never transfer to another M/N/K point. Runtime geometry, candidate/schedule configuration, and static resources are inputs. Candidate-relative exact dynamic metrics are bounded auxiliary targets beside log-regret; held geometry is removed across all M and formats before target construction. No varying comparative target makes the profiler variant measured-only. Focused regressions prove exact ownership, input/target separation, static retention, and held-geometry exclusion. |
| Optional profiler metric absence became numeric zero | Auxiliary target assembly previously used zero for a missing optional metric while retaining measured values from competing candidates. That could make unsupported evidence look like an exceptionally economical kernel. LLC misses were also collected without LLC loads and then discarded because no fraction could be formed. | A dynamic metric is centered only when every candidate in the exact runtime/shape contest publishes it; unavailable contests contribute the neutral centered value for that target. L1 and LLC counts are normalized by regime work even when a matching denominator event is unavailable. One-shot duration labels are excluded, and reviewed reliability weights keep instructions above L1 and sparse LLC evidence. Raw counters remain immutable for future re-featurization without recollection. |
| Paid profiler surfaces were process-local | The canonical cross-M diagnostic required 315 coherent spatial surfaces, but every restart rebuilt all ExtraTrees forests before CV could resume. Interrupting a long fit therefore discarded paid deterministic model work even though timing and profiler corpora were immutable. | `profiler-prediction-surface` entries content-address the candidate-cost pool, model descriptor digest, held geometry set, and exact consumed point inventory. Only CV training points are stored, with byte-stable float encoding. A cold in-memory replay loads the persisted map with zero model workers, while partial or foreign maps hard-fail. |
| Parallel profiler forests serialized cache publication | Surface fitting used all physical cores, but every fitted prediction map returned to the coordinator for expensive canonical sorting, JSON encoding, and multi-megabyte atomic writes. Accelerators remained idle while the parent published surfaces one at a time. | The worker that fits each independent surface also owns its deterministic sort, lossless encoding, and content-addressed atomic publication before returning the in-memory result. Completion-order reduction remains deterministic, the coordinator performs no parallel-miss publication, and a focused fork regression rejects parent-owned writes. |
| Monolithic profiler catalogs made additive composition CPU-bound | Extending the CPU M=1 catalog by 6,480 isolated launches required composing 130,248 request/evidence/witness records. The bounded digest used physical-core workers, but standard-library JSON decode, dataclass construction, ordered merge, and final emission still held one host core for most of a 189-second transaction and rewrote roughly 1.3 GiB. | Published profiler corpora retain authenticated content-addressed source shards and a small ordered index. Coverage, feature export, and fit consume the indexed union directly; compaction into a legacy monolith is optional publication work, never an installation prerequisite. Any parallel or streaming replacement must preserve canonical request/evidence/witness digests and have byte-equality plus interrupted-resume regressions. |
| Serial common-observation publication | Rewriting the 216,632-row checkpoint spent tens of seconds formatting roughly 580 MiB through one Python `csv.writer`. | Large observation CSVs are split into deterministic row ranges, formatted in fork workers, assembled in source order, and atomically published. A byte-equality regression proves parallel output is identical to canonical serial output. |
| Idle second socket during sealed collection | The sealed queue launched `mpirun -np 1` batches because the refresh default was one CPU lane, despite two physical sockets and already-disjoint resumable jobs. | CPU measurement lanes default to all detected sockets. Each MPMD rank receives a different `(format, shape, ISA regime)` output cell, and regressions prove both automatic socket use and absence of duplicate paths. |
| Per-format sealed process explosion | The 1,859-record post-freeze witness plan launched one MPI process job per format even when several formats shared geometry, ISA, and the complete M inventory. This repeated process startup and regenerated identical format-seed activation fixtures. | The process serializer coalesces only geometry/ISA/M-compatible formats, producing 1,337 jobs for the current plan while retaining all 1,859 format records. It emits a content-derived path token, keeps equal format/M phase inventories on paired ranks, and the trainer caches byte-identical legacy Q8_1 fixtures. Regressions prove no format is duplicated or lost, phase groups are contiguous, and cached bytes equal the old generator. |
| Long prefill failure discarded completed M phases | A largest-envelope CPU prefill shard completed M=64,256,1024,2048, then its M=4096 complete-candidate warmup round legitimately ran longer than the fixed 30-second watchdog. The process aborted and the wrapper would have deleted every earlier row before retrying the full seven-M inventory. | Collection contract v17 flushes each complete M phase, authenticates an interrupted aggregate/timing pair with the production adapter, requires paired socket ranks to own the identical ordered prefix, and appends only the missing suffix. The trainer records the summed candidate probe duration and derives a bounded 4x-plus-one-second round watchdog from it. Legacy headers, partial rounds, non-prefix inventories, and cross-rank mismatches cannot append; final publication requires the exact planned M inventory. |
| Fresh seal mislabeled as development build | Fit-only replay correctly restored the immutable development build ID, but fresh post-freeze timing reused that ID in its resume contract and final adapter context even after the trainer binaries changed. | Development provenance remains active through fit/freeze only. A live seal contract binds the current AVX2/AVX512 binary digest, and current git/compiler/host/build provenance is restored before launching and certifying sealed rows. Collection hard-fails if the production serial-policy hash differs from development, so this split cannot excuse an arithmetic change. |
| Sealed installation replay overwrote final shards | Replaying the one-command transaction after a certification failure omitted `--resume-cpu-partials` and immediately relaunched the two most expensive already-final shards into new `.inprogress` files. The final evidence survived, but preservation depended on an easy-to-miss operator flag. | Complete sealed aggregate/timing pairs are immutable under the active collection contract and are always reused. Only an incomplete `.inprogress` pair may be replaced. Focused regression coverage proves post-freeze sealed replay has no dependency on the general CPU partial-resume flag. |
| Grouped continuation re-entered completed CPU decode | After the sealed M=1 policy had been certified and installed with `--stop-after-cpu-decode`, rerunning the production CPU transaction without that stop flag re-adapted decode inputs, launched a fresh isolated profiler transaction, and consequently opened a new expensive decode fit generation before grouped collection. The fit cache was correctly responding to changed profiler evidence; the phase orchestration was wrong. | `--resume-after-cpu-decode --install` is an explicit production continuation boundary. It validates the sealed policy/include in the selected output directory, requires that include to match the installed CPU decode table byte-for-byte, bypasses all decode timing/profiling/fitting, and enters grouped collection directly. Focused CTest coverage proves the continuation emits no decode trainer command and refuses an uninstalled prerequisite. |
| CPU grouped corpus omitted its serial M=1 dependency | The immutable CPU bundle treated grouped verifier timing, profiler evidence, policy, and include as complete even when the exact serial M=1 policy used as its arithmetic oracle was absent. A grouped policy could therefore be published without the production dependency that defines byte equality and supplies the row implementation. | CPU corpus assembly requires the certified M=1 policy/include, development and sealed timing transaction, common observations, and profiler request/evidence alongside the grouped artifacts. A focused bundle regression rejects grouped publication when any serial dependency is missing, while preserving backend-specific payload rules. |

The journal is complete only when recipe preflight has negative tests for every
row above and the supported CPU-prefill invocation no longer requires
individual lineage flags. New workflow failures discovered during CUDA or ROCm
promotion must be added here before they are fixed.

The implemented CPU-prefill recipe is
`cpu_prefill_replay_recipe.v1.json`. It content-addresses the candidate
expansion, ordered additive timing transactions, historical lineage, current
refinement rounds, target routes/split, profiler additions, and checkpoint
lifecycle. `refresh_native_vnni_dispatch_tables.sh` expands it without shell
evaluation and runs semantic preflight before route probes. The turnkey driver
automatically discovers the recipe in a materialized corpus. A checkpoint
copied to a new fit workspace is content-authenticated, validated against the
complete plan partition, and rebased because the historical adapter identity
contains source paths. Rebase changes transaction provenance only; it neither
reinterprets historical timing under current adapter code nor collects new
evidence. Complete replay records also carry the raw corpus identity already
proved by preflight, eliminating repeated full-file hashing in development
fit, freeze, and certification.

The sealed witness plan is intentionally derived only after generic freeze.
Candidate route probes contain launchability and exact C++ routing, never
timing or correctness labels. Completion geometries use the
`CPUPrefillAutoRefine_*` namespace, are forbidden from the development corpus,
and are embedded directly in the plan so certification does not depend on a
discarded candidate pool. The compact candidate-route manifest and plan are
authenticated replay-recipe artifacts.

### 15.3 Cross-backend atomicity

A change to the common schema, learner, feature policy, policy ABI, or IR
requires CPU, CUDA, and ROCm regeneration and acceptance in one transaction.

A hardware-specific corpus refresh MAY update one backend if the common
versions are unchanged and the other installed manifests remain compatible.
It still creates a complete immutable bundle containing the accepted unchanged
artifacts for the other backends and publishes that bundle through the one
active-pointer update.

### 15.4 Base merge policy

Staged profiles merge compatible corpora and retrain. Direct `.inc` copying or
`--base-include` overlay installation is forbidden unless the base has
machine-readable IR and the validator proves that the scored and emitted
generic policies are identical.

## 16. Required Tests

### 16.1 Common schema and learner units

The common learner requires synthetic regression tests proving that:

- a one-ULP verifier candidate is rejected even with perfect-looking cosine;
- missing `bitwise_equal`, oracle hash, or route evidence is rejection;
- an unsafe lowest-latency candidate is filtered before exact selection;
- a policy with 97% winner-label accuracy and one 40% performance miss fails;
- a policy with low winner-label accuracy but every regret below 5% passes;
- exact overlays cannot mask a broken generic policy during sealed evaluation;
- candidate/format/M/mode rows from one shape cannot leak across development
  folds or into the sealed partition;
- sealed observations are unavailable until the final generic digest is frozen;
- the emitted generic IR is not refit after sealed certification;
- opposing alias/mode winners choose the deterministic worst-surface robust
  candidate;
- missing alias coverage rejects a candidate;
- a stale M1 oracle hash invalidates verifier entries and requires dependent
  evidence to be recollected;
- a candidate unsupported by the production launcher fails generation;
- segment fitting and artifact digests are deterministic;
- profiler requests change when any bound observation/timing identity changes;
- a shape-resolved formula cannot masquerade as a physical profiler launch;
- every supported request has exactly one complete evidence record and every
  unsupported request has one explicit non-profile state;
- multi-kernel candidates preserve contiguous physical dispatch order;
- missing tools, required metrics, request records, or raw artifact digests
  fail profiler completeness, while unsupported optional counters remain
  explicit; and
- profiler evidence cannot mutate or substitute canonical timing evidence;
- base-policy metrics describe the exact emitted generic rules; and
- a domain with one shape cannot claim generic certification;
- `Auto` cannot be relabeled as an explicit CPU candidate;
- canonical-format inventory divergence, including a missing `Q8_K` entry, is
  rejected;
- quick/family-smoke installation is rejected; and
- a failed bundle transaction leaves the active pointer unchanged.

### 16.2 Backend emitter and resolver units

Each backend MUST test:

- exact precedence over generic rules;
- every aspect boundary and work-segment boundary;
- Python IR to generated C++ round-trip candidate equality;
- every emitted candidate's launcher reachability;
- homogeneous bundle-signature and aggregate-feature resolution;
- rejection of an uncertified heterogeneous bundle;
- worst-mode aggregation across eager and captured observations;
- separation of `Fast` and verifier policies with a deliberately poisoned fast
  table;
- nested and concurrent contract resolution;
- embedded policy/architecture/registry/oracle metadata mismatch failure;
- telemetry match kind and candidate identity; and
- deterministic fail-closed behavior for missing coverage.

Existing generated-dispatch units remain necessary but are not sufficient:

```bash
ctest --test-dir build_v2_integration \
  -R 'V2_Unit_(NativeVNNIDispatchRefreshScript|CUDAGemvDispatchGeneratorAliases|CUDAGemvDispatchBaseMerge|ROCmNativeVNNIDecodeTrainerGenerator|ROCmNativeVNNITrainerCsvValidator|NativeVNNIGeneratedDispatchCodebooks)' \
  --output-on-failure --parallel
```

These tests remain during migration. Once opaque `--base-include` merging is
removed, `V2_Unit_CUDAGemvDispatchBaseMerge` SHOULD be retired and replaced by
common corpus-merge and emitted-policy identity tests.

### 16.3 Kernel byte-equivalence matrix

Generated-table acceptance MUST enumerate:

- every exact verifier entry;
- at least one interior point in every generic leaf;
- both sides of every aspect and work boundary;
- every prepared family/codebook and source-format alias;
- every integer `M=2..16`, plus `M=31`;
- eager and graph-captured production routes;
- cold first use, warmed execution, and repeated execution;
- fused and unfused paths;
- alpha/beta, existing-output, and bias/epilogue variants where supported;
- dense and routed/shared MoE operation kinds; and
- every emit-able reduction/K-partition family.

For each cell, force the resolved candidate, run grouped rows once, run the
production M1 oracle for each row, compare every logical FP32 result element,
and assert the grouped route counter. The public-operation layer must
additionally start from the same input tensor so activation quantization and
preparation are inside the proof boundary.

The canonical cross-backend prefix remains mandatory:

```bash
ctest --test-dir build_v2_integration -N \
  -R '^V2_Integration_GroupedVerifierRows_'
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_GroupedVerifierRows_' \
  --output-on-failure --parallel
```

The prefix is authoritative; a hand-maintained test list is not.

### 16.4 Model-level gates

Kernel equality is necessary but not sufficient. A promoted table MUST also
pass the affected Qwen dense and MoE verifier-operation equivalence,
continuation, graph-capture, prefix-cache/MTP, and stochastic tests. These
model-level gates keep cosine/L2/KLD and token checks as layered diagnostics,
but they do not replace kernel `memcmp`.

The current six backend/model operation-equivalence lanes are mandatory:

```text
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_Dense_CPU
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_Dense_CUDA
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_Dense_ROCm
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_MoE_CPU
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_MoE_CUDA
V2_Integration_Qwen36_MTPForwardVerifierOperationEquivalence_MoE_ROCm
```

### 16.5 Performance gate

The install test MUST parse the staged manifest and fail unless every required
generic bucket reports:

- complete sealed shape/alias/execution-mode coverage;
- zero missing candidate measurements;
- p95 worst-surface regret strictly below 5.0%;
- p95 of simultaneous per-cell 95% upper regret bounds strictly below 5.0%;
- a generic policy digest frozen before sealed evaluation and unchanged in the
  emitted IR; and
- zero verifier byte failures.

Full-model throughput must then remain within its declared regression budget;
an isolated generic-policy pass does not authorize a model-level regression.

## 17. Promotion Criteria

A backend/architecture/contract domain is promotable only when all rows below
are green.

| Gate | Required evidence |
|---|---|
| Corpus completeness | Every required operation, bundle, family, alias, M, exact shape, development/sealed group, execution mode, and explicit candidate is represented. |
| Oracle stability | Production M1 route is repeat-byte-stable and its current policy hash is recorded. |
| Candidate correctness | Every selected verifier exact/generic/floor candidate is byte-equal for all required trials. |
| Exact policy | All required production keys resolve to eligible reachable candidates. |
| Generic policy | The exact frozen generic IR has 100% sealed coverage and observed p95 regret/p95 simultaneous UCB `<5.0%`, with no post-certification refit; maxima remain diagnostics. |
| Profiler evidence | Every supported canonical candidate observation has provenance-bound isolated profile evidence with one target launch per pass, complete required metrics, ordered physical dispatches, and retained raw artifacts. |
| Emitter/runtime | Common IR and C++ resolver agree; exact precedence, boundaries, digests, and telemetry pass. |
| Grouped implementation | Route counters prove one genuine grouped implementation, not hidden row replay. |
| Integration | Canonical grouped prefix and affected Qwen operation/continuation gates pass. |
| Performance | Relevant kernel and model benchmarks are non-regressing beyond the accepted budget. |
| Installation | One immutable complete bundle is published by an atomic active-pointer rename and can roll back by digest. |

No aggregate green percentage can override a red required cell.

## 18. Phased Implementation Plan

### Phase 0: Freeze unsafe promotion paths

- Reject `--install` for quick and family-smoke profiles.
- Prevent partial copy-on-success before all selected backends validate.
- Require staged outputs and a transaction directory.
- Add source-policy tests preventing new hand-coded exact-shape exceptions.

Exit: current tables remain usable, but no retune can bypass the future
certificate boundary.

### Phase 1: Common schema, inventory, and candidate registries

- Add the shared observation schema and validator.
- Add the backend-neutral shape, development-fold, and sealed-split manifests.
- Create explicit CUDA, ROCm, and CPU candidate registries.
- Add effective-candidate normalization and launcher reachability checks.
- Add serial-M1 policy and arithmetic fingerprints.

Exit: all three trainers can emit complete, validated common rows without
changing production dispatch.

### Phase 2: Common exact and generic compiler

- Implement eligibility filtering and alias aggregation.
- Implement exact robust-winner construction.
- Implement bounded aspect/work segmentation over the full cost matrix.
- Implement deterministic grouped development CV, a frozen final development
  fit, and sealed generic-only certification with no later refit.
- Implement policy IR, manifests, and synthetic learner tests.

Exit: one compiler can consume fixture corpora for every backend and prove or
reject the 5% sealed-certification claim.

### Phase 3: Backend trainer and emitter migration

- Migrate CUDA away from the two mismatched generic learners.
- Migrate ROCm away from modal winner rows and approximate verifier gates.
- Add explicit CPU candidates and first-class generic output.
- Add emitter/resolver round-trip tests for all three.

Exit: all backend `.inc` files are decisions from the same common compiler.

### Phase 4: Explicit runtime semantic contract

- Add `GemmSemanticContract` to the common dispatch request/plan.
- Route every single, fused, dense, and MoE verifier caller through it.
- Remove process-global backend mode state and arithmetic-changing verifier
  overrides.
- Add policy digest validation and route telemetry.

Exit: ordinary fast tables cannot affect grouped verifier execution, including
under concurrent calls.

### Phase 5: Staged acceptance and atomic publish

- Extend the refresh wrapper to produce versioned immutable
  include/IR/manifest bundles and one atomic active pointer.
- Build against staged artifacts.
- Enumerate exact/generic runtime decisions and run the canonical grouped
  matrix.
- Require model parity, sealed performance certification, and atomic bundle
  publication.

Exit: an installed table is reproducible, byte-certified, sealed-certified,
and rollbackable.

### Phase 6: Production retune and default policy

- Collect complete blessed hardware corpora for CPU, CUDA, and ROCm.
- Train, stage, and freeze `Fast M=1` first; then recollect and certify
  verifier `M=2..16,31` against that exact staged artifact.
- Run the full MTP benchmark matrix.
- Update the parent plan and evidence dashboard with artifact digests,
  correctness counts, development-CV and sealed regret, and model throughput.

Exit: all three backends have first-class exact and generic learned tables
under this policy.

## 19. Failure Handling and Rollback

- Corpus or schema failure: do not fit or emit.
- No eligible verifier candidate: mark the domain unsupported; do not fall
  back to fast or row replay.
- Development CV p95 not below 5% or a missing generic domain: emit a non-installable
  refinement diagnostic, gather only the named fresh timing/profiler cells,
  and refit; exact overlays cannot waive this failure.
- Sealed certification p95 not below 5%: reject the entire policy. A later policy
  attempt may use those revealed shapes as development evidence only after
  reserving a new untouched sealed partition.
- Stale M1 hash: invalidate and recollect, then retrain, every dependent
  verifier entry.
- Emitter/runtime mismatch: reject the artifact as corrupt.
- Integration or model parity failure: retain the prior installed transaction.
- Model performance regression: retain the prior table even if microbench and
  sealed gates pass.
- Post-install regression: atomically repoint `active` to the previous
  immutable bundle digest; never restore only one file from the set.

## 20. Open Implementation Decisions

These choices must be resolved before Phase 1 exits and recorded in the policy
ABI or manifest as appropriate:

1. The initial blessed `architecture_class` inventory for CPU model families,
   CUDA compute capabilities, and ROCm GFX targets.
2. The repository versus CI-artifact retention policy for raw timing samples,
   full corpora, and raw profiler reports.
3. The exact simultaneous/familywise bootstrap or correction method used for
   the p95 of conservative per-cell 95% regret upper bounds.
4. The initial kernel-math ABI fields and which compiler changes invalidate an
   arithmetic certificate.
5. Whether operation kinds with identical candidate reachability may share a
   policy bucket, with proof, or remain separate in v1.
6. The required model-level throughput regression budget after the kernel
   sealed certification gate passes.

None of these decisions may weaken verifier byte equality or the strict p95
`<5%` sealed-certification regret requirement.

## 21. 2026-07-19 CPU Decode Checkpoint

The CPU decode transaction now includes the 16-shape Cartesian K-part
neighborhood around `N=2048, K=11008`. Canonical timing contains 151,200 common
observations. Its profiler corpus is an authenticated additive composition of
119,448 retained exact launches and 4,320 new all-format/ISA launches, for
123,768 requests and evidence records total. The extension launched no repeated
timing cells and the completed profiler delta was reused without another
`perf_event_open` collection.

Profiler resume is now based on complete physical launch identity rather than
timestamped request IDs. Compatible raw-profiler feature-schema generations
normalize to the current exporter during composition. Completed but unpublished
content-addressed deltas are recovered automatically; interrupted collector
journals resume in place; changed arithmetic, candidate policy, schedule,
workspace, execution mode, or geometry remains a fresh profiling obligation.
CPU decode also checkpoints a stable development run ID per output directory.

For the first uncached enlarged-corpus planner fit:

- profiler-model fitting: 78.674 seconds across 30 physical-core workers;
- accelerator CV frontier: 481.162 seconds across CUDA and ROCm scorers;
- compact domain-parallel CV reduction: 13.578 seconds across 56 workers;
- total paired planner fit: 624.240 seconds;
- generic domains assessed: 108;
- focused paired requests produced: 152 across 20 unresolved cells;
- focused requests measured and retained: 152 across 11 socket-local shards.

The next step is a resumable iteration-9 fit over those retained paired rows,
followed by domain-quota inspection. No CPU policy is promotable or installed
at this checkpoint; installation still requires at least 95% of required
domains to satisfy both strict p95 regret gates below 5%.

## Appendix A: Example Observation

```json
{
  "schema_version": 1,
  "run_id": "2026-07-11-rocm-gfx906-production-01",
  "corpus_id": "sha256:corpus...",
  "git_revision": "0123456789abcdef",
  "build_id": "rocm-release-01234567",
  "compiler_id": "clang-19.0.0-rocm",
  "policy_abi": 1,
  "learner_version": "native-vnni-segmented-regret-v1",
  "backend": "rocm",
  "architecture_class": "gfx906-native-vnni-v1",
  "device_name": "AMD Instinct gfx906",
  "driver_runtime": "rocm-example",
  "threading_or_stream_mode": "explicit_non_default_stream",
  "semantic_contract": "VerifierSerialM1Bitwise",
  "operation_kind": "FusedGateUp",
  "bundle_signature": "gate_up:homo:cb5:fp32-epilogue:v1",
  "projection_n_vector": [17408, 17408],
  "source_format": "Q4_K",
  "source_codebook_id": 5,
  "prepared_family_id": "NativeVNNI_CB5",
  "packing_abi": "native-vnni-cb5-v1",
  "runtime_codebook_id": 5,
  "shape_group_id": "qwen36-ffn-gate-up-neighborhood",
  "shape_name": "Qwen36_FFN_GateUp",
  "execution_mode": "graph_captured",
  "m": 3,
  "aggregate_n": 34816,
  "k": 5120,
  "aspect_ratio": 6.8,
  "aspect_bucket": "wide",
  "work_items": 178257920,
  "n_tail_class": "aggregate_n_mod_256=0",
  "k_tail_class": "k_mod_256=0",
  "alignment_class": "prepared_16b_aligned",
  "candidate_id": "rocm.kb4.tw8.ordered",
  "effective_candidate_id": "rocm.kb4.tw8.ordered",
  "candidate_family": "shared_rows",
  "config_json": {"kb": 4, "target_waves": 8, "reduction": "ordered"},
  "supported": true,
  "graph_capture_ok": true,
  "generic_eligible": true,
  "arithmetic_fingerprint": "sha256:...",
  "serial_m1_policy_id": "rocm.cb5.m1.kb4.tw8.ordered",
  "serial_m1_policy_hash": "sha256:...",
  "candidate_policy_hash": "sha256:...",
  "ordered_reduction": true,
  "uses_atomic_reduction": false,
  "trial_set_hash": "sha256:trials...",
  "bitwise_equal": true,
  "repeat_equal": true,
  "mismatch_count": 0,
  "first_mismatch_index": null,
  "grouped_output_digest": "sha256:...",
  "serial_output_digest": "sha256:...",
  "max_abs": 0.0,
  "relative_l2": 0.0,
  "cosine": 1.0,
  "symmetric_kld": 0.0,
  "warmup_count": 5,
  "sample_count": 30,
  "min_us": 40.98,
  "median_us": 41.25,
  "p95_us": 41.91,
  "mad_us": 0.19,
  "cv": 0.006,
  "timing_sample_hash": "sha256:samples...",
  "effective_bandwidth_gbs": 216.1,
  "forced_route_ok": true,
  "observed_candidate_id": "rocm.kb4.tw8.ordered",
  "route_counter_ok": true,
  "workspace_ok": true,
  "explicit_stream_ok": true
}
```

## Appendix B: Learner Pseudocode

```python
def compile_policy(development_observations, sealed_partition_handle,
                   sealed_commitment, required_inventory,
                   frozen_m1_manifest, split_manifest):
    development = validate_common_schema(development_observations)
    development = normalize_effective_candidates(development)
    assert_development_surface(
        development, required_inventory, frozen_m1_manifest, split_manifest
    )
    assert_opaque_sealed_commitment(sealed_commitment, split_manifest)

    development = filter_by_semantic_contract(development)
    development_exact = build_alias_mode_robust_exact_winners(development)

    # CV chooses complexity using development shapes only.
    cv_results = grouped_development_cv(development, split_manifest)
    learner_spec = select_learner_spec(cv_results)

    # This is the final generic policy. Freeze it before opening sealed data.
    final_generic = fit_bounded_aspect_work_rules(
        development, development_exact, learner_spec
    )
    final_generic_digest = freeze_policy_digest(final_generic)

    # No sealed row or feature has been loaded before this point.
    sealed_observations = open_sealed_partition(
        sealed_partition_handle,
        sealed_commitment,
        final_generic_digest,
    )
    sealed = validate_common_schema(sealed_observations)
    sealed = normalize_effective_candidates(sealed)
    assert_disjoint_shape_groups(development, sealed, split_manifest)
    assert_sealed_surface(
        sealed, required_inventory, frozen_m1_manifest, sealed_commitment
    )
    sealed_exact = build_alias_mode_robust_exact_winners(sealed)
    sealed_results = evaluate_generic_only(
        final_generic, sealed, sealed_exact
    )
    sealed_results = paired_confirmation_timing(
        final_generic, sealed_exact, sealed_results
    )

    require_complete_structural_coverage(sealed_results)
    sealed_domains = aggregate_by_generic_domain(sealed_results)
    require_domain_promotion_quota(
        sealed_domains,
        passing_fraction=0.95,
        observed_p95_strict_limit=0.05,
        simultaneous_ucb_p95_strict_limit=0.05,
    )
    require_total_generic_tree_for_every_domain(final_generic)
    record_max_regret_diagnostics(sealed_results)

    # Exact overlays are separate and do not retrain final_generic.
    all_eligible = development.concat(sealed)
    all_exact = build_alias_mode_robust_exact_winners(all_eligible)
    policy_ir = emit_common_ir(
        exact_overlays=all_exact.required_production_entries(),
        generic_rules=final_generic,
        development_cv=cv_results,
        sealed_certificate=sealed_results,
        frozen_generic_policy_digest=final_generic_digest,
        frozen_m1_manifest=frozen_m1_manifest,
    )
    return policy_ir
```

## Appendix C: Example Acceptance Manifest Fragment

```json
{
  "policy_abi": 1,
  "learner_version": "native-vnni-segmented-regret-v1",
  "feature_schema_version": "aspect-work-v1",
  "backend": "cuda",
  "architecture_class": "sm90-native-vnni-v1",
  "policy_digest": "sha256:...",
  "candidate_registry_hash": "sha256:...",
  "format_registry_hash": "sha256:...",
  "packing_abi_hash": "sha256:...",
  "corpus_hash": "sha256:...",
  "development_fold_manifest_hash": "sha256:...",
  "sealed_split_manifest_hash": "sha256:...",
  "frozen_serial_m1_policy_digest": "sha256:...",
  "serial_m1_policy_hashes": ["sha256:..."],
  "contracts": {
    "Fast": {
      "frozen_generic_policy_digest": "sha256:fast-generic...",
      "exact_entries": 612,
      "generic_rules": 104,
      "development_cv_p95_regret_pct": 1.6,
      "sealed_coverage_pct": 100.0,
      "sealed_passing_domain_fraction": 0.997,
      "sealed_p95_regret_pct": 1.7,
      "sealed_max_regret_pct": 2.6,
      "sealed_simultaneous_max_ucb_regret_pct": 2.9
    },
    "VerifierSerialM1Bitwise": {
      "frozen_generic_policy_digest": "sha256:verifier-generic...",
      "exact_entries": 612,
      "generic_rules": 96,
      "bitwise_cells_passed": 7344,
      "bitwise_cells_failed": 0,
      "development_cv_p95_regret_pct": 1.8,
      "sealed_coverage_pct": 100.0,
      "sealed_passing_domain_fraction": 1.0,
      "sealed_p95_regret_pct": 1.9,
      "sealed_max_regret_pct": 2.5,
      "sealed_simultaneous_max_ucb_regret_pct": 2.8
    }
  },
  "emitter_round_trip": "pass",
  "staged_grouped_verifier_gate": "pass"
}
```

## Appendix D: Current Implementation Surface Inventory

Common runtime and refresh:

- `src/v2/tensors/TensorKernels.h`
- `scripts/refresh_native_vnni_dispatch_tables.sh`
- `tests/v2/performance/kernels/native_vnni_codebooks.py`
- `tests/v2/performance/kernels/validate_native_vnni_generated_dispatch_ids.py`

CUDA:

- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp`
- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativeVNNIDecodeTrainer.cpp`
- `tests/v2/performance/kernels/cuda/gemm/analyze_cuda_native_vnni_decode_trainer.py`
- `tests/v2/performance/kernels/native_vnni_dispatch/CUDANativeVNNILeafPrimaryScorer.cu`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu`
- `src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp`
- `src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc`

ROCm:

- `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Throughput.cpp`
- `tests/v2/performance/kernels/rocm/Perf__NativeVNNI_GEMM.cpp`
- `tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py`
- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip`
- `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp`
- `src/v2/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc`

CPU:

- `tests/v2/performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_GEMV.cpp`
- `tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_decode_trainer.py`
- `tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py`
- `tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_prefill_trainer.py`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIDecodePolicyGenerated.inc`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIPrefillPolicyGenerated.inc`
- `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIVerifierRowsPolicyGenerated.inc`

Canonical integration coverage:

- `tests/v2/utils/QuantizedVerifierFormats.h`
- `tests/v2/integration/kernels/cuda/Test__CUDAGemmParity.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmQuantisedGemmSmallM.cpp`
- `tests/v2/integration/kernels/cpu/native_vnni/Test__CPUNativeVNNI_GEMV.cpp`
- `tests/v2/integration/parity/qwen36/Test__Qwen36_MTPForwardVerifierOperationEquivalence.cpp`
