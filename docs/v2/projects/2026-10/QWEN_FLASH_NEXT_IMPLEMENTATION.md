# Qwen Flash Next implementation and qualification

Started 2026-10-18 on `feat/qwen-flash-next`, from `9a0e26bcc`.
The architectural research is in
[QWEN38_FLASH_NEXT_INFERENCE_RESEARCH.md](QWEN38_FLASH_NEXT_INFERENCE_RESEARCH.md).
This document records implementation evidence; it does not replace live source contracts.

The next feature-branch checkpoint freezes the installed five-profile HC-up
TwoTerm accuracy candidate, after fresh complete Unit/production preflight and
the combined broad, held-out and diagnostic-boundary corpus. The 769-test
Unit/native preparation pass and subsequent 759-test Unit/paired-adapter pass
are green. This checkpoint does not promote arithmetic defaults or qualify
performance. The unfinished all-expert intermediate sharding and additional
pipeline cases remain private drafts for the next slice. Local receipts and
corpus payloads stay outside the source commit.

The distributed ngram and runtime hot-row cache design is recorded in
[QFN distributed and adaptive ngram storage](QFN_NGRAM_TIERED_STORAGE_PLAN.md).
It covers simultaneous VRAM/RAM/SSD residency, explicit replication/sharding,
and promotion/eviction during serving. Its
[qualification strategy](QFN_NGRAM_TIERED_STORAGE_PLAN.md#functional-and-performance-qualification-strategy)
defines the typed configuration inventory, exhaustive component and admission
coverage, native 1/2/4/8-device obligations, same-GGUF numerical/HTTP tests,
Release performance experiments and integration with the existing CI gates.

## Required outcome

Correct, performant text inference on CPU-only, ROCm+CPU, CUDA+CPU, and
CUDA+ROCm+CPU. Each configuration must offer native n-grams in RAM and a
bounded SSD mmap mode. GPU placement must support explicit replication and
sharded ownership across cards, including TP, PP and expert-parallel domains.
Automatic planning prefers VRAM, then RAM, then SSD; bulk ngram storage remains
the lowest-priority VRAM obligation. Storage owners and lookup consumers are
distinct identities. Sharded lookup should share existing collective exchanges
where their dependencies and participants permit, otherwise declare its own
graph communication. Every exchange carries only live requested rows and
required protocol metadata, with device-owned counts on GPU paths.
GPU execution, recurrent state, sampling and acceptance
remain device-owned. Only the explicit immutable-row storage service crosses
into CPU storage ownership. Quantized source weights are never requantized to
implement the new model.

The model is `qwen4exp`, with a 2560-wide compute path, four-branch 10240-wide
residual trunk, GDN/QSA mixers, 512 routed experts, and PLE before decoder block
1. Registering it as an existing Qwen architecture would be incorrect.

## Pinned model and reference

- GGUF repository: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
  `766911a6b7369840a91dbcd95f9f997acaab6cd6`.
- Main export: `UD-Q4_K_XL`, four shards. Its PLE table is IQ4_NL: 320001536
  rows, 160 elements/90 bytes per row, 28800138240 bytes total.
- Additional predictor export: `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.
  This is a separate source, not evidence of predictor heads in the main file.
- Official reference and converter revisions are authenticated in the research
  document. Numerical diagnosis must use the canonical PyTorch parity workflow
  against these exact quantized GGUF payloads.
- Local downloads and evidence are ignored under `models/` and
  `parity-results/qwen-flash-next/`; do not commit their payloads.

## Optional FP16 GDN and router comparison — 2026-10-20

Keep GDN alpha/beta in FP16 in every comparison arm. Reuse the existing
setup-only arithmetic policies; do not add a second policy mechanism or
promote a default before measuring accuracy and performance:

| Arm | `LLAMINAR_GDN_PROJECTION_ARITHMETIC` | `LLAMINAR_MOE_ROUTER_ARITHMETIC` |
|---|---|---|
| Current QFN baseline | `fp16_moe_alpha_beta` | `native` |
| FP16 router | `fp16_moe_alpha_beta` | `fp16` |
| Full FP16 GDN | `fp16` | `native` |
| Full FP16 GDN and router | `fp16` | `fp16` |

Full FP16 GDN additionally prepares QKV, Z and output projections with FP16
weights and operands, using FP32 accumulation. Preparation decodes the
original GGUF values directly; it does not requantize them into another
integer format. Router FP16 means FP16 weights and hidden operands with
FP32 accumulation and routing reductions. Expert GEMMs retain their original
weights and Q8 activation policy, independently quantizing the original hidden
state when the router no longer publishes Q8 operands. GDN recurrence and its
FP32 state arithmetic remain unchanged.

Use SingleQ8 for the independent residual-correction selectors in these four
arms. An explicitly incompatible TwoTermQ8 request must fail during admission,
not silently select another implementation or fail after allocation.
Focused regressions must prove original-source preparation, exact capacity,
effective stage dispatch and grouped/serial equivalence on CPU, CUDA and ROCm.
These tests supplement the existing generic FP16 operand tests.

Measure full-vocabulary prefill and teacher-forced decode KL through the
canonical same-GGUF PyTorch framework. Separately time snapshot-free Release
production inference with matched prompt tokens, topology, placement,
context and output budget; retain actual token counts, generated token IDs,
arithmetic-path counters, memory costs and repeated request measurements.
Natural-generation timing is explicitly distinguished from teacher-forced
accuracy. GPU overlay comparisons must use canonical cells that actually
declare the intended expert capacity; the hybrid PP cell has its own declared
full-expert placement and is not an overlay control.

The existing implementations are under QFN-specific audit. No new model KL
or Release speed result for this four-arm comparison is claimed yet.

## Focused build46 qualification — 2026-10-20

The new bounded sparse-selector candidates and CPU direct-emit MTP checkpoint
fix compile for CUDA SM80/86/89/90 and ROCm gfx906 with the spill guards enabled.
The first focused run completed all 947 selected registrations: 945 passed,
with two aliases of one CPU test failing because the fixture poisoned the
first **unused** score rather than the last live score. The corrected fixture
now tests both boundaries separately.

After that test-only repair, all 747 Unit tests and its explicit CPU preflight
alias pass (748/748, no skips, clean driver window). The CUDA/ROCm selector
and CPU MTP regressions passed in the preceding focused run. Evidence is in
`build46-repair2/`, `focused46/` and `focused46-repair2/` beneath the ignored
QFN results directory. These are focused results, not a current complete
production-preflight receipt or a real-model MTP/performance qualification.
The selector production default remains unchanged.

## vLLM precision comparison and Q8 acceptance — 2026-10-20

The [pinned vLLM review](VLLM_QFN_PRECISION_REVIEW.md) finds a material
comparison difference: its official QFN FP8 evaluation checkpoint excludes
all main-layer GDN/QSA projections, hyperconnection projections, routers and
shared-expert projections from FP8 conversion. QFN's router is constructed
unquantized; specialized FP32/BF16 multi-term router kernels are conditional
implementations, not a universal QFN default on Ampere or MI50.

The reviewed code protects selected FP32 reductions and explicit BF16
rounding boundaries. It contains no general mechanism that recovers
FP32-reference expert choices after upstream activation quantization.
Its FP8 sparse-attention tests allow tied-score index differences, but that
does not establish a permissible MoE error rate or a same-GGUF KL threshold.

As requested, accept the possibility that small expert-choice differences
are an ordinary Q8 trade-off once correctness is established. Preserve the
FP32 same-GGUF model oracle while separating same-input operator error,
propagated hidden-state error, ranking margins and final model quality.
The archived original-F32 router controls reproduce their changed expert
choices from upstream inputs with approximately 4e-7 local probability
relative L2 error. That is positive localized evidence, not total coverage.
Continue scoped diagnostics and the ordinary-Q8 versus selective-correction
quality/performance comparison; do not chase identical FP32 routes by
silently replacing ordinary Q8 GEMMs. Broader prompt/context evidence is
needed before changing a canary. Production defaults and the strict
grouped/serial MTP arithmetic contract remain unchanged.

## Strata precision comparison — 2026-10-20

The [pinned Strata source review](STRATA_QFN_PRECISION_REVIEW.md) identifies
FP32 decode inputs for sensitive floating projections, an optional two-term
BF16 activation correction in prefill, and overflow saturation for FP16
intermediates. Its ordinary quantized QKV decode still uses Q8 activations;
its default dense prefill instead decodes weights to FP16. There is no single
precision rule to copy across all operations or backends.

For Llaminar, prioritize selective activation residual correction using the
original GGUF weights. Do not import Strata's optional BF16 conversion of
already-quantized sensitive weights or its approximate expert skipping.
The reviewed top-64 quant-quality metric and internal engine comparisons are
not replacements for our same-GGUF, full-vocabulary PyTorch gate.

Build45's complete prerequisite gate passes all 747 Unit and 1,576 preflight
registrations, with only explicit unavailable-hardware skips and a clean
driver window (`prerequisites-main-45/`, `preflight-driver45/`). The six
matched build45b QKV controls complete with valid artifacts and clean driver
windows. QKV-only TwoTermQ8 lowers worst decode KL from 1.07722 to 0.487234
on CPU, 1.57389 to 0.361550 on CUDA+CPU, and 0.461583 to 0.311559 on
ROCm+CPU, but all six remain numerical red. Production defaults and numerical
thresholds are unchanged. The review records the next combined-correction
comparison and its functional/performance obligations.

## DeepInfer-informed optimization slice — 2026-10-20

The [DeepInfer research](DEEPINFER_INFERENCE_RESEARCH.md) is now part of this
implementation plan. Its pinned DeepSelect, FlashMLA, DeepGEMM and DeepEP
revisions identify algorithm sources; they do not establish performance or
numerical qualification for QFN. The immediate target remains CPU, RTX 3090
(SM86) and MI50 (gfx906), with every shipped GPU target compiled for spill
qualification. The build45 prerequisite and model windows have retired. The
new selection and expert scheduling candidates remain separate, unqualified
changes until their focused native tests and Release measurements pass.

### First: exact bounded-candidate sparse selection

Adapt [DeepSelect's bounded-candidate algorithm](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/docs/DeepSelect-deep-dive.md)
through the existing `QwenSparseIndexer` contract on all three backends. Filter
successive score blocks against a rising threshold and reduce a bounded local
candidate set, instead of repeatedly scanning the whole row for threshold bits.
Use a deterministic traversal and a total `(score, original index)` ordering;
block scheduling must not alter tie selection. Cluster-specific SM90 operations
are not an implementation for Ampere, CPU or MI50.

The QFN operation still selects 512 complete four-token pools, expands them in
ascending token order, and appends the zero-to-three incomplete tail tokens.
FP32 score calculation, non-finite-input handling, replay visibility and
transaction-prior ownership remain unchanged. Local/shared scratch is bounded;
any persistent workspace is declared through the existing arena BOM and PMA.
Selection starts only when more than 512 complete pools are visible. The
existing direct all-visible path remains an intentional geometry specialization.

Required evidence before dispatch installation:

- Compare every selected index and count against an independent ordered top-k
  oracle, including all-equal scores, ties at the cutoff, monotone and adversarial
  orderings, the established zero/non-finite contract, and all pool-tail lengths.
- Cover M1 through the supported grouped range, partial and empty buckets,
  M512/M1024/M2048, threshold boundaries around 2048 visible tokens, and context
  growth through 64K. Prove dispatch totality outside measured geometries.
- Replay full/small/empty/full captured transactions with poisoned inactive
  rows; compare downstream attention bytes, serial/grouped MTP, accepted-prefix
  rollback and restored prefixes. Reuse the sparse-attention native suites and
  add an explicit focused `ProductionTestPreflight` registration for each new
  invariant or discovered defect. Device-free Unit cases must remain short.
- Measure selection alone, complete indexer/QSA, and matched Release model
  prefill/decode/MTP. Record scratch, registers/VGPRs, spills, occupancy, launch
  geometry and memory traffic using the applicable tuning skill. Thirty-one
  source-level scans do not imply thirty-one HBM reads. Keep profiling separate
  from timing; use supported resource/ISA inspection while profiler attachment
  remains unqualified on this host.
- Authenticate actual candidate execution, live rows and matched placement via
  PerfStats. Exact index selection should introduce no numerical change;
  confirm this in the existing same-GGUF PyTorch and prefix lifecycle campaigns.
  Retire replaced selector machinery only after correctness and economy pass.

### Next: expert scheduling and intermediate traffic

[DeepGEMM's grouped interfaces](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/README.md)
motivate device-counted work scheduling and reducing expert intermediates.
Its published SM90/SM100 precision/layout contracts are not a drop-in kernel
for the current hardware. QFN keeps its original quantized weights, native
codebooks, Q8 activation policy and fixed reduction order. Existing fused
gate/up and SwiGLU/requantization must be included in the baseline.

Audit the actual 512-expert path for idle tiles, repeated activation packing,
unused workspace traffic and projection handoff costs. Implement the smallest
measurable improvement through the current grouped-kernel interfaces. Tile
selection must follow geometry/aspect ratio and live expert demand, not a list
of benchmark M values. Test every supported source format, skewed and empty
experts, M1/grouped equivalence and large-to-small replay on CPU/CUDA/ROCm.
Measure both expert-kernel time and complete MoE-layer/request critical paths;
lower standalone GEMM time is insufficient if packing or communication grows.

The first source candidate is a bijective ordering of the existing compact
directory/column tile grid: column-first baseline versus groups of four/eight
adjacent directory entries. The final short group uses its actual extent;
grid cardinality, row ownership, K reductions and persistent storage are
unchanged. CUDA and ROCm drafts use the same typed mapping and retain an
identity path without extra integer division. Native candidate identities and
PerfStats must include the chosen group. Qualification includes an independent
domain-coverage oracle, original-format full-buffer byte comparisons, large
grid/padding boundaries and uniform/skewed QFN512 demand. These are private,
uncompiled candidates; no measured benefit or dispatch promotion is recorded.

Any compute/communication overlap remains an explicit graph collective beside
participant-local compute. Counters must prove exact live payloads, actual
overlap, admitted storage and lower critical-path time. DeepEP's dependency
upgrade and a monolithic remote-progress Mega MoE port are separate work,
not prerequisites for these local optimizations.

### Attention fusion and integration order

Profile remaining normalization/rotary/cache-store intermediate writes before
expanding existing fusion. Preserve each intermediate rounding boundary, native
KV encoding, prepared-weight identity and exact producer event. Fusion must
retain retained-graph reset and grouped serial-row equivalence.

Private source candidates may proceed alongside the current unchanged gate.
After its model controls complete, integrate a coherent candidate with focused
Unit/native/architecture checks, then refresh complete prerequisites once for
that source. Qualify numerical behavior before unprofiled Release comparisons.
Do not mix changes in selector, arithmetic, expert placement or ngram capacity
within a comparison arm. Shared archives, phase-disaggregated serving and
confidence-based MTP allocation remain later designs; approximate SWA replay
does not enter exact prefix restoration.

## Current evidence

The reviewed QKV diagnostic is now installed in
[install43](../../../../parity-results/qwen-flash-next/install43/result.json),
with ten explicit production-preflight regressions. Its opt-in two-term Q8
operand uses the original main-layer native Q8_0 QKV source; Z, alpha and beta
retain their existing arithmetic. Planning uses the existing named workspace
BOM, while request and prefix identities distinguish the selected arithmetic.
The default remains SingleQ8. Full-model KL/performance measurement of this
combined slice is pending.

The complete build43 attempt collects two new fixture compilation failures in
1,139.10 seconds: a missing `DevicePlanConfig` declaration and a retained-identity
test appended outside its namespace. No production compilation failure is
recorded. [Install44](../../../../parity-results/qwen-flash-next/install44/result.json)
repairs those two fixtures and adds exact `kernel.gdn_projection_route` collection
to the QFN campaigns. Each cell saves its raw route CSV before validating an
explicit TwoTerm selection against original source metadata. These are native
submission/capture records; `production_path.csv` separately proves replay.
No timing domain, numerical gate or default is changed. The
[incremental build44](../../../../parity-results/qwen-flash-next/build44/result.json)
passes all eleven actions in 107.09 seconds, with a no-work repeat. Its
[focused run](../../../../parity-results/qwen-flash-next/focused44/result.json)
collects all **891 registrations** in 446.79 seconds: all 747 Unit registrations
pass, with no skips or driver faults. Only the two ROCm mixed-source GDN bundle
registrations fail, both reporting a stale `hipErrorInvalidSymbol` after native
graph metadata inspection.

[Install45](../../../../parity-results/qwen-flash-next/install45/result.json)
repairs that inspection error ownership. A library/module kernel legitimately
misses HIP's host-function registry; successful module-resource queries do not
clear that query's pending error. The next corrected-Q8 launch was therefore
reporting the inspection error. Inspection now rejects and preserves an incoming
error, and consumes only its own immediately verified registry miss. The actual
kernel launch checker and arithmetic remain unchanged. The new explicit
`V2_Integration_HIPGraphInspectionErrorOwnership` regression fails on the original
implementation for both ordinary and same-code incoming errors, with clean
driver retirement. The original mixed-module failure is preserved in focused44.
The [complete build45](../../../../parity-results/qwen-flash-next/build45/result.json)
passes 737 actions in 57.21 seconds, with a no-work repeat and the complete
CUDA 80/86/89/90 and HIP gfx906 target set. The
[two focused native regressions](../../../../parity-results/qwen-flash-next/hip-inspection-proof45/result.json)
now pass with clean driver retirement. The
[broader focused45 run](../../../../parity-results/qwen-flash-next/focused45/result.json)
passes **899/899 registrations** in 463.17 seconds, including complete Unit and
the affected HIP graph, GDN, verifier and adaptive lifecycle families, with no
skips or driver faults. The fresh complete prerequisite gate is running before
the model reruns; focused success does not substitute for that receipt.

Build42 installs the prepared CPU M1 verifier repair, the adaptive fixture's
public retirement/generation observation boundary, and an expert-ID-aligned
diagnostic contribution comparison. The
[complete build](../../../../parity-results/qwen-flash-next/build42/result.json)
passes all 863 actions in 406.21 seconds, including CUDA 80/86/89/90 and HIP
gfx906, with a no-work repeat. Its
[focused gate](../../../../parity-results/qwen-flash-next/focused42/result.json)
passes **821/821 registrations** in 156.69 seconds: complete Unit plus affected
CPU/CUDA/ROCm lifecycle and verifier regressions, with no skips or driver faults.
All seven new focused aliases are explicitly registered in production preflight.
The aligned diagnostic leaves every raw numerical row and acceptance gate intact.
The [retained-source repetition](../../../../parity-results/qwen-flash-next/adaptive-retirement-repeat42/result.json)
then passes 20 CUDA and 20 ROCm executions with clean driver retirement.
Full prerequisites and real-model reruns for this source are still pending;
the next full gate will include the separately reviewed QKV diagnostic so that
the combined changes receive one fresh prerequisite receipt.

Build41 installs the two runtime repairs described below. The
[complete rebuild](../../../../parity-results/qwen-flash-next/build41/result.json)
passes all 819 build actions in 197.59 seconds and verifies a no-work repeat.
The [focused run](../../../../parity-results/qwen-flash-next/focused41/result.json)
passes all 814 registrations, including complete Unit and the affected native
lifecycle tests, in 150.60 seconds with no skips or driver faults. Both cold
capture tests and the predictor-hidden zero-norm test have explicit preflight
registrations. [Original-code controls](../../../../parity-results/qwen-flash-next/regression41-negative-fixed-fixture/result.json)
reproduce the zero rejection, premature frontier acquisition and independent
admitted-only gather rejection. A fixture pointer-type compile correction is
retained separately; it changes no production behavior.

The completed build41 prerequisite gate (2026-10-20) is green: **747 Unit
registrations and 1,528 production-preflight registrations pass**, with 29
explicit hardware skips and no failures. The
[canonical receipt](../../../../parity-results/qwen-flash-next/prerequisites-main-41/prerequisites.json)
and [observer result](../../../../parity-results/qwen-flash-next/preflight-driver41/result.json)
retain the exact source/binary identity and clean driver retirement. Elapsed
time was 3,912.81 seconds across 2,304 registrations. This is model-free qualification, not a numerical or
serving certificate. Subsequent source repairs require a fresh receipt.

The [five model41 reruns](../../../../parity-results/qwen-flash-next/model41-runtime-repairs-summary.json)
pass the repaired zero-hidden and cold-capture boundaries and expose two later
failures. CPU Depth1 rejects the valid M1/N320/K10240 prepared HC projection;
both its outer guard and its inner grouped scheduler require correction through
the existing prepared M1 primitive. CPU numerical artifacts remain incomplete,
so their placeholder zero fields are not KL measurements. CUDA and ROCm C0/C64
each complete all eight canonical numerical/prefix/path artifacts, then the
fixture requests retained source ownership while the suffix-prefill worker is
still active. The worker's idle/joined guard is correct. The fixture must retire
that public request and exercise actual public generation before validating
the independently published GPU/source activity counters. All five driver
windows retire cleanly; no cache-activity or generation qualification is claimed.

Both cache capacities give the same reported numerical metrics within each
backend. CUDA prefill KL is 0.00760721, with decode KL
0.141026 / 0.116757 / 1.57389. ROCm prefill KL is 0.0194998, with decode KL
0.160465 / 0.0435705 / 0.461583. Native full/partial prefix state equivalence
passes; external-reference numerical gates remain red. C64 means 64 cache
slots per tier with a 1 MiB physical budget, not 64 MiB.

The [eight model38 controls](../../../../parity-results/qwen-flash-next/model38-controls-direct/matched-controls-summary.json)
and [repaired hybrid case](../../../../parity-results/qwen-flash-next/model38-expert-ready-sequential/main38-hybrid-mmap-single/report.json)
complete prefill, three decode steps and the canonical artifact contract, but
remain numerical red under unchanged gates. Their settings and detailed KL
table are retained in the dated record below. Two further
[matched CPU alpha/beta controls](../../../../parity-results/qwen-flash-next/model40-cpu-alpha-beta-comparison.json)
also complete and remain red. With HC/GDN/FA output projections using the
two-term Q8 diagnostic, selected FP16 alpha/beta gives prefill KL 0.007848 and
decode KL 0.0825384 / 0.0154821 / 0.136449; selecting original FP32 alpha/beta
gives 0.0053157 and 0.0318103 / 0.0325557 / 0.157937. This does not justify a
default change; the record distinguishes selected policies from independently
exported native path counters.

Two qualified same-input decompositions narrow the numerical investigation:

- [Original-F32 router replay](../../../../parity-results/qwen-flash-next/router-same-input39/summary.json)
  reproduces the complete gold probabilities and the observed expert-set changes
  from saved native inputs. Local probability relative L2 is about 4e-7 at the
  examined CUDA/ROCm cutoffs; incoming activation drift explains those changes.
- [Layer0 GDN QKV replay](../../../../parity-results/qwen-flash-next/gdn-qkv-same-input40/results.json)
  reproduces the entire gold 9-by-10240 output byte-for-byte from the original
  Q8_0 matrix. Across CPU, CUDA and ROCm snapshots, local projection relative
  L2 is about 0.00221, versus 0.00036 propagated from the input. All 27 rows are
  retained. The observed-operand experiment below resolves the local error's
  main contributor without identifying a kernel arithmetic defect.

The [build41 observed-operand experiment](../../../../parity-results/qwen-flash-next/qkv-observed-operands41/results.json)
uses the original Q8_0 QKV bytes and actual native activation codes/scales.
CPU, CUDA and ROCm each reproduce all nine archived output rows byte-for-byte.
Local relative L2 is 0.00221536 / 0.00221230 / 0.00221251; canonical FP32
projection of the observed quantized operands explains essentially all of it.
The remaining native-versus-reference reduction difference is only
1.79e-7 / 1.23e-7 / 1.29e-7. It is not evidence of a dot-product defect.

Calling the existing two-term Q8 operation on the same original prepared source
reduces local relative L2 to 8.62e-6 / 8.22e-6 / 8.66e-6, a 256–269-fold
reduction, with every row improved. All three backends pass both the M9
component and the separate M15 finite-tail control: original primary operands
remain byte-identical, retained replays agree, and inactive candidate output
and workspace suffixes remain poisoned. All driver intervals are clean. These
are component accuracy measurements, not final-model KL or performance results.
The next diagnostic will use the existing typed projection policy to test QKV
correction in the complete model; defaults and numerical gates stay unchanged.

Build39's declaration/binding repair passes its
[807-registration focused run](../../../../parity-results/qwen-flash-next/adaptive-focused39/result.json).
Its lexical cache-control optimization preserves exact ordered policy in all
[28 native component cases](../../../../parity-results/qwen-flash-next/adaptive-component-timing39-a/result.json)
and improves warm-VRAM prefill time about 25% on ROCm and 6–8% on CUDA.
Warm VRAM still costs more than source-only, so cache economy remains open.
These are Integration component measurements, not Release throughput.

No numerical threshold, default precision, distributed/adaptive model
eligibility, MTP serving qualification, HTTP lifecycle or Release-performance
qualification is granted by these results.

### Historical foundation and earlier gate collection

The following records retain their original build/date scope. Later dated
sections preserve the detailed findings and superseded failures.

Historical admission status, build29 (2026-10-19): all 32 registered setup cases
pass: 20 main-model RAM/mmap topologies, nine paired-source MTP cases and three
serving-tier cases. This closes the focused admission sprint. The subsequent
build30 prerequisite transaction passes all 746 Unit registrations and 1,365
preflight registrations, with 26 explicit skips for unavailable eight-device
hardware. Fresh CUDA/ROCm numerical attempts expose a retained-storage reset
defect after prefill; their decode and complete artifact contracts remain
incomplete. Numerical parity and Release performance remain open.

The integrated build31 focused run completes all 65 selected registrations:
50 pass and 15 fail, with no skips and clean driver diagnostics. The actual
captured RAM/mmap prefill-reset retirement tests pass on both CUDA and ROCm,
as do fixed-head native lookup primitives and direct GDN correction kernels.
The failures are being repaired together: host request bounds were incorrectly
treated as GPU directory capacity; GPU stage fixtures bypassed the real prepared
weight loader; a workspace fixture omitted required GDN metadata; and the reset
terminal event needed its exact lifecycle entry in the blocking-sync policy.
Build32 contains those repairs and their focused regressions. Its 69-registration
run passes 65; the remaining four registrations expose two CPU fixtures missing
explicit worker/cache/ISA observations. Both fixtures are corrected and all four
registrations pass the focused follow-up, including their complete Unit targets.
Both driver windows are clean, with no skips. Thus every selected registration
has passing current-source evidence across the run and follow-up, not one new
aggregate certificate. Exact typed ngram selection for future canonical E2E
cells also passes, without granting QFN E2E eligibility. The next unchanged-source
transaction is `prerequisites-main-32/`, followed by complete model reruns.

- The four main GGUF shards and separate MTP source are present on the model
  SSD. The main file advertises 48 ordinary layers and no predictor. Paired-source
  setup now admits the separate predictor in all nine registered admission
  cases; predictor numerical execution remains a separate qualification task.
- Typed GGUF arrays retain exact signedness and all 64 bits. Configuration
  separates compute/head width 2560 from residual width 10240 and excludes
  predictor blocks from ordinary layer intervals.
- The 512-expert runtime ABI, native CUDA/HIP mirrors, directory/tag capacity,
  and router paths have been expanded. Fresh captured CPU/CUDA/ROCm focused
  regressions pass, including expert 511 and the shared-expert slot. CUDA
  sm80/86/89/90 and HIP gfx906 compile without spills.
- Ten residual/PLE primitives and native sparse attention have CPU/CUDA/ROCm
  implementations. Backend-neutral stages and reusable hyperconnection/QSA/PLE
  builders use existing prepared projection identities and explicit arena views.
- The coherent foundation gate passed all 41 suites in 58.05 seconds. This
  includes captured sparse attention and immutable-row lookup on both GPUs,
  native integer arena types, all 12 sparse-state banks, prefix capacity and
  archive lifecycles. A fresh CUDA test exposed production fast-math rotary
  differences missed by a standalone build; the corrected native/HF rotary
  policies pass the actual production compiler flags on both backends.
- Native table storage supports RAM and bounded read-only SSD mmap. Focused
  storage tests cover native codecs, cross-window rows, eviction, concurrent
  clients, file mutation and PMA release. No payload checksum or whole-table
  format conversion is used.
- User CLI and lossless configuration select RAM/mmap and mapping-window
  geometry. Native table demand has one CPU owner; selected GPU bytes are
  separate admitted scratch. One declaration inventory feeds QFN arena
  allocation and admission, including device/PP placement filtering.
- An explicit GPU/CPU immutable-row protocol now has generation, table,
  sequence, live extent, response acknowledgment and complete-parent terminal
  authentication. RAM and mmap captured lookup tests passed on both backends.
  Complete-parent termination, poisoned first admission and absorbing failure
  regressions pass in focused native runs. CPU completion and GPU terminal
  statistics publish exact live lookup traffic without a new inference transfer.
- The canonical PyTorch reference consumes the same quantized GGUF payloads,
  including IQ4_NL table rows and expert 511. Fifteen reference math regressions
  pass, including independent sigmoid GDN norm, PLE EOS/chunk history and QSA
  selection. Typed INT64 diagnostic transport and canonical NPY/CSV comparisons
  now preserve all identity bits. The fresh snapshot/NPY and semantic GR/QSA/PLE
  publication checks pass. The exact main-model reference pack is generated:
  1123 checkpoints for prefill and each of three decode steps.
- The registered whole-model graph compiles with production flags. It composes
  shared GDN/MoE with branch mixers, PLE and sparse attention. CPU mmap smoke
  completed prefill and generated four tokens, beginning `\n\n<think>\nThe`.
  Native lookup statistics report 224 selected rows, 20,160 native bytes and
  zero failures. This proves execution, not complete numerical qualification.
  Setup exposed early arena binding, INT32 schema materialization,
  terminal-weight inventory/name errors and an incorrect extension lookup for
  the core residual trunk. These fixes have fresh passing regressions and a
  coherent build. The terminal contract now requires native `output_hc_*`
  operands. Prefill exposed ranking scratch incorrectly published as a
  diagnostic output even when all visible blocks are selected without ranking;
  its scoped fix preserves real output validation. Decode then exposed a
  16-projection CPU wrapper limit: top-10 expert gate/up needs 20 descriptors.
  One capacity now follows the admitted expert ABI, with AVX2/AVX512 exact
  regressions through 1,026 descriptors and oversized rejection. CPU semantic
  snapshots now recognize valid host-owned tensors without a GPU-owner header.
  Multi-request and accepted-prefix MTP wiring remain open.
- The first complete CPU mmap parity campaign executed every layer and three
  teacher-forced decode steps. Prefill logits have cosine 0.996144 / KL 0.0113508;
  decode KL is 0.235456, 0.0455317 and 1.07722, so the numerical gate remains
  red. A diagnostic defect applied HF head-order conversion to already-native
  GGUF GDN references; the corrected typed reference-layout policy and five
  focused regressions pass. Raw snapshot analysis independently reproduces
  layer-zero branch-read differences with the actual Q8 activation policy;
  its low-rank up projection dominates that early error. Later routing-related
  divergence is under investigation. Complete and partial prefix restoration
  reproduce the corresponding engine terminal-logit hashes exactly; that does
  not certify their agreement with the external reference.
- CUDA/ROCm auxiliary first-use regressions now prove that the existing arena
  owner materializes all four state banks before initialization, with exact
  producer/consumer events and stable pointers. The first full CUDA+CPU model
  setup then exposed host-only token bindings in GPU workspace declarations
  and external serial decode. The correction binds the permanent native token
  bank during declaration and external decode. A typed position policy retains
  the canonical captured KV-position producer, without uploading invented
  positions or unused decode token capacity. Its four device-free regressions
  pass; the complete Unit/preflight rebuild precedes the next model attempt.
  GPU model execution is not yet qualified.
- An independent original-GGUF calculation reproduces the layer-zero HC read
  to approximately 1e-7 relative L2 with the native Q8 activation rule. The HC
  up projection is the principal early contributor. Keeping all other equations
  canonical, quantizing only those up-projection inputs reproduces worst decode
  KL 1.178. A two-term Q8 activation correction reduces that isolated worst KL
  to 2.2e-7 without changing any weight bytes. With the other native activation
  policies enabled, its simulated worst decode KL is 0.170. This remains an
  off-tree numerical/performance candidate, not a production result.
- On that corrected simulation, retaining reference expert IDs while using the
  candidate's own router probabilities lowers last-step KL from 0.169731 to
  0.002602. Retaining reference probabilities on the candidate's selected IDs
  gives 0.080125. The F32 router itself agrees with its equation for the actual
  incoming hidden values: discrete expert membership amplifies upstream error.
  These controls isolate the cause; they are never production routing modes.
- A pinned llama.cpp CPU Release diagnostic at converter commit `1b70350c9`
  used the same four GGUF shards, exact nine prompt tokens, FP16 KV, context
  4096 and teacher-forced continuation `13,271,248045`. Its reference-to-engine
  KL was 0.019943 / 0.192094 / 0.052714 / 0.635448 for prefill and three decode
  steps; all four top tokens matched the reference. This supplies external
  accuracy context, not a timing comparison or permission to weaken gates.
  Raw logits, tokenizer identity checks and build logs remain under
  `parity-results/qwen-flash-next/llama-cpu-logits*`.
- Cache-owned auxiliary state now has exact setup publication, request/layer
  reset, compact rollback tails and separate live block/endpoint archive
  sections. The fresh RAM/SSD archive format and lifecycle tests pass.
  No per-token copy of the full indexer pool or payload checksum is introduced.

## Current gate repair slice

The complete initial production-preflight diagnostic collected all 1,196
registrations: 1,132 passed, 38 failed and 26 skipped for unavailable device
counts. The interrupted ROCm lane was resumed for its remaining 116 cases;
all passed. This is a complete failure inventory, not a passing prerequisite
receipt. The preceding complete Unit run passed 736 of 744 registrations.

The preflight failures have three root causes: the archive-version expectation,
a 512-thread CUDA controller launch exceeding the register budget, and the
hybrid pipeline classifying an admitted request token as a controller-mailbox
offset. Focused repairs preserve the actual archive version, the fixed
512-expert reduction tree and one-token communication extent. The same batch
addresses the eight Unit failures without weakening their invariants, CPU
sparse inactive-row writes, owned-RAM NUMA placement, and asynchronous archive
completion on section exceptions. New regressions are registered explicitly in
preflight. The coherent build15 closure completed successfully and an immediate
repeat reported no work. All 20 focused native/infrastructure registrations
passed. Ten of 11 focused Unit registrations passed; the remaining suite has
two stale archive source guards, corrected in the next reviewed slice.
A fresh complete prerequisite receipt is still required.

The next slice also fixes residual-width and live-extent errors at direct PP
boundaries. QFN's four-branch residual must cross as 10240 columns, while ordinary
compute remains 2560. Retained graph replay must resolve each invocation's real
rows rather than cache the first copy size; metadata-only graph construction
must never copy an unadmitted payload. Focused regressions include large-to-small
and empty replays and preserve canonical native receive-bank aliases.

CUDA controller timing stopped before reporting a sample because the native
node reported 32 bytes of local storage. Final linked SASS and all four shipped CUDA
architecture images confirm the indexed eight-element per-source maximum
array. The equivalent scalar maximum eliminates that storage while preserving
validity and owner masks. Both CUDA and ROCm controller sites use the same
production aggregation helper and an independent per-source reference test;
final coherent-build/native/profiling evidence remains required.

The complete build16 closure and immediate no-work repeat passed. All 15 focused
Unit registrations passed; 25 of 28 focused integrations passed. The three
failing registrations share one new observability regression: direct activation
handoff submits the correct live DMA extent but bypasses the existing physical
H2D/D2H counters. The repair records each successful physical leg once, including
the completed first leg of a failed staged transfer. Native byte assertions stay
intact; a separate device-free regression covers successful, empty, rejected and
partially completed handoffs. A fresh complete prerequisite receipt remains
pending. Linked CUDA controller images for sm80/86/89/90 now show zero stack and
local storage, with no local load/store instructions; native timing remains open.

The transfer-counter repair passed all six affected registrations. The subsequent
complete Unit run passed 744 of 745 registrations; the remaining two assertions
in the workspace source-policy suite also assumed inline archive publication.
Their test-only repair follows the typed callback's retained owners and all
three successful restore commits, with negative controls for early, missing,
duplicated and detached publication. Production code and the native binary are
unchanged by this follow-up; complete prerequisite qualification is still open.

The subsequent complete build16c failure inventory passed all 745 Unit
registrations. Across all 1,213 preflight registrations, 1,185 passed, 26 skipped
for unavailable device counts and two failed: the same retained-child fixture
on CUDA and ROCm. The interrupted ROCm lane and the exclusive lane were collected
separately; these diagnostic continuations are not a passing prerequisite receipt.
The fixture uploaded a distinct upstream tensor but relied on setup-only graph
materialization to populate canonical input. The repair explicitly publishes
one admitted live row through TransferEngine and joins its receiving event;
it preserves setup-only behavior, child identity and reset/replay assertions.
All three affected fixtures passed 20 repetitions and the full 23-test fixture
passed on each native backend against the unchanged production core.

The ROCm diagnostic observer also exposed an invalid timestamp assumption:
concurrent kernel messages can be published with decreasing timestamps. The
repair authenticates publication ordinals and exact cursor continuity without
sorting or discarding records. Negative controls still reject missing/reordered
records and retain earlier-stamped driver faults. All 19 observer tests pass.
Build16d integrates these test/observer repairs and three explicit focused
preflight registrations. Its complete canonical prerequisite run passed all
745 Unit registrations and 1,190 preflight registrations; 26 preflight cases
skipped for unavailable device counts. The driver observer completed with no
new faults. QFN arithmetic and production defaults are unchanged in this repair
slice; the passing receipt is retained in the ignored
`prerequisites-main-16d/prerequisites.json` evidence directory.

The first exact CUDA+CPU and ROCm+CPU main-model cells then reached model
admission but failed before inference: the layer3 QSA indexer's
`prepared_query` snapshot descriptor had no stable GPU source during graph
snapshot preparation. Both runs ended with clean driver observations. These
are setup failures, not numerical parity or performance results. The shared
snapshot ownership contract needs a focused regression before those model
comparisons can progress.

The matching mixed-vendor cell also stopped before inference: its expert-overlay
declaration put a ROCm expert participant inside a CUDA-owned continuation graph.
The execution invariant correctly rejected that ownership. The next matrix uses
explicit participant-local pipeline stages on CUDA, ROCm and CPU, with a
initial 8/16/24 main-layer partition. Four device-free tests, including the
public CLI round trip, passed 20 repetitions. Native admission and inference for
that partition remain unqualified.

The cold snapshot failure was traced to write-only QSA scratch declared as
preallocated. A diagnostic-only output has no downstream reader to trigger its
ordinary arena materialization. The pending repair declares those outputs for
the existing allocation pass and checks original native owner, base pointer and
subview extent before recording a snapshot copy. Five focused CPU tests passed
20 repetitions; the old stage fails the new allocation-contract regression.
The actual cold-allocation/capture/full-partial-empty replay fixture compiles for
both GPU backends, with native execution pending the coherent rebuild.

Separate ROCm controller replay checks passed 3,200 exact expert-511 plans per
mode, with no errors and unchanged poisoned capacity tails. Timing events inside
the captured graph were rejected by HIP; those runs prove correctness only.
Explicit timing outside capture measured complete transaction medians of
1.490 ms for the three-participant static case and 1.003 ms for the two-participant
LLEP case, including reset and validation. These different geometries are not a
cross-mode speed comparison or isolated controller performance qualification.

### Complete build16f failure collection, 2026-10-19

The coherent build16f closure completed for CUDA sm80/86/89/90 and HIP gfx906,
with the compiler spill guards enabled and an immediate no-work repeat. Its
complete Unit run passed all 746 registrations. The unchanged preflight
diagnostic collection completed all 1,306 registrations: 1,273 passed, seven
failed and 26 skipped for unavailable device counts. The interrupted ROCm and
exclusive lanes were completed separately to gather every failure. All three
driver observation windows were clean. These continuations are diagnostic
evidence, not a passing canonical prerequisite receipt.

The seven failures have three causes. The CUDA and ROCm ordinary sparse-tail
fixtures expected an untouched, distinct destination to contain prior state;
the corrected oracle poisons the destination and verifies exact accepted-prefix
writes and no-op preservation. The mapped MoE transport fixture attempted to
replace its GGUF source after WeightManager had sealed it; it now loads the
real tiny source through the production factory before sealing. Both fixture
repairs are installed with explicit focused preflight registrations. Their
device-free reproductions pass 20 repetitions; native reruns remain pending.

The four mixed-vendor fixed/dynamic MTP lifecycle failures expose a production
graph defect. An authenticated predictor source-layer number was also used to
select a main-decoder deferred-residual fusion. The sidecar consequently read
`MTP_ATTN_PROJ`, which has no preceding producer, instead of normalizing the
complete hidden state from `mtp0_fc`. TransferEngine correctly rejected that
unprepared input. The installed correction uses the explicit graph workload to
select this operation. Actual graph-builder regressions cover full and KV-only
sidecars across source ordinals, cache-local layer zero, the PP boundary and
unchanged main-decoder fusion. Production-flag syntax checks pass; coherent
runtime qualification remains pending.

Separate predictor parity work is still private and unqualified. It covers
the reference pack, exact native snapshot publication, diagnostic storage BOM,
logical FC rows and CPU shifted-prefill interval collection. Independent review
caught stale contextual/result aliases across repeated CPU collections and a
failure-path statistics ordering defect; those proposals will not be installed
until the focused freshness regressions pass. Whole-model GPU numerical
execution, predictor qualification, real SSD mmap economics and Release
throughput remain open. The next model attempts require a fresh complete green
Unit/preflight transaction after the narrow repairs above.

The complete build16g repair closure passed in 202.62 seconds with an immediate
no-work repeat. All seven focused fixture/input-ownership registrations passed
20 repetitions each. All six real mixed-vendor request-lifecycle cases then
passed: MTP disabled, fixed and dynamic in both ROCm2-to-CUDA2 and
CUDA2-to-ROCm2 order, including reset and prefix reuse. Both driver windows were
clean. The fresh complete gate now contains 746 Unit and 1,310 preflight
registrations; its canonical qualification remains pending.

### Green build16g prerequisites and unchanged model diagnostics

The complete build16g prerequisite receipt passed all 746 Unit registrations
and 1,284 production-preflight registrations; 26 additional preflight cases
skipped for unavailable hardware. There were no failures or new driver faults.
The canonical run covered 2,056 registrations in 3,578.9 seconds. Both orders
of the two-ROCm/two-CUDA MTP lifecycle regressions passed. The receipt is retained
at `prerequisites-main-16g/prerequisites.json`; later source changes require a
new receipt rather than relabeling this one.

The unchanged CUDA+CPU and ROCm+CPU mmap cells now prepare the cold QSA output
correctly, but both stop before inference while admitting diagnostic snapshots.
The first arena uses 273,529,792 bytes and serial decode requests another
36,868,160 bytes, exceeding the 301,965,504-byte declared envelope. QSA pooled-key
snapshots retain native context capacity, whereas the reference files contain
only live pooled rows. A scoped typed native-state inventory is being prepared;
neither a larger anonymous reserve nor a relaxed physical-memory check is a fix.

The CUDA0/ROCm0/CPU0 pipeline cell also stops during setup. Its migration
admission requires multi-participant native GPU controller domains, so it
rejects even the singleton GPU stages before reaching the CPU stage. The
repair must preserve independent pipeline-stage authorities and authenticate
the no-peer movement identity. The existing host/native pipeline boundaries
remain the intended execution path. This topology is not yet qualified.

Current native CPU accuracy, with unchanged original-GGUF reference and gates:

| HC-up activation policy | Prefill KL | Decode 0 KL | Decode 1 KL | Decode 2 KL |
|---|---:|---:|---:|---:|
| Default single Q8 | 0.0113508 | 0.235456 | 0.0455317 | 1.07722 |
| Explicit two-term Q8 | 0.00225386 | 0.132576 | 0.0315883 | 0.20669 |

The candidate improves every measured KL but still fails the full gate and
its last top token is 271 rather than reference token 1787. It remains an
explicit experiment; the default and thresholds are unchanged. The corrected
CSV layout removes the earlier false GDN head-order divergence. Expert
membership first differs at prefill layer 0 for the default and layer 1 for
the candidate. These are observed comparison boundaries, not a claim that
the router kernel caused the preceding activation error.

Candidate RAM and mmap runs produce identical recorded metrics and all 4,392
exported FP32 snapshot payloads match byte-for-byte (235,625,152 bytes). Each
run records six completed lookup calls, 352 native rows, 31,680 native bytes
and zero failures; all 52 exact integer checkpoint comparisons pass. Full and
partial prefix restores retain engine-state equivalence, while their external
numerical qualification remains red. Canonical model staging is tmpfs, so
these results do not establish actual SSD throughput.

The observability source slice installs strict typed-stage CSV schema synchronization,
per-successful-gather mapping-window observations, page-rounded mmap virtual
extent reporting, and lossless benchmark RAM/mmap argument export. All four
defects have explicit focused preflight registrations. The coherent build and
all 746 Unit registrations pass. A test fixture initially selected two storage
cells where its E2E exporter requires one; each storage policy is now checked
separately, retaining the existing two-cell axis regression. The four focused
aliases pass 20 repetitions. This is not a new full-preflight certificate.

The user selected a focused admission sprint before another broad gate run.
The reviewed main-QSA snapshot fix replaces live reference pool rows with the
actual native context-capacity state and prices authored metadata/alignment in
the existing snapshot BOM. The reviewed hybrid fix authenticates stage-local
CPU/GPU ownership and complete no-peer residency without changing Dynamic
policy or inventing migration work. The focused loop also corrected the stale
host-demand validation for no-peer stages and preserved authored singleton TP
backend/workshare declarations when translating a rank plan. The affected
targets build coherently; all seven focused device-free aliases pass 20
repetitions, including real planner/PMA snapshot claims. A separate
`SnapshotAdmission` case uses each of the existing 20 real-model configurations
and performs actual weight preparation and serving-graph materialization, then
stops before numerical inference. Its artifacts cannot certify numerical parity
or stand in for full preflight. The separate MTP diagnostic and grouped-HC
performance stacks remain uninstalled during this sprint.

Real CPU admission passes for both RAM and mmap (about 71 and 65 seconds,
respectively) using the original GGUF path on SSD. These cases materialize all
48 layers and retain the admitted memory report, then tear down without
numerical inference. The typed snapshot inventory now prices the shared wide
lane and both dedicated scalar lanes; completed route snapshots no longer
collide with continuation-local route diagnostics. The combined build6 passes
and actual CUDA/ROCm model captures no longer exceed the snapshot allocation.
Eleven of twelve focused device-free registrations pass; the remaining new
fixture incorrectly requested authored counted-copy metadata for a legacy raw
snapshot. Its correction preserves the production payload-only contract.

The full CUDA/ROCm/CPU pipeline now passes real-model mmap admission in about
44 seconds, including teardown and a clean driver window. It authors an
8/12/28 layer split: the initial 8/16/24 split legitimately exceeded the MI50's
capacity and was rejected before expert materialization. The matrix and CLI
regression retain the new exact layer boundaries; the memory authority and
allocation limits are unchanged. The matching RAM admission also passes in
about 81 seconds, including teardown and a clean driver window.

CUDA+CPU and ROCm+CPU dynamic/ordinal mmap now pass real-model admission in
about 47 and 50 seconds. The hybrid mmap representative also passes again in
about 45 seconds. The setup caller previously asked for a native-composable
child even though ExpertOverlay owned a host-serviced retained transaction.
The implementation now selects the explicit hosted/native ownership contract
before materialization and authenticates the complete retained graph identity
at replay; the strict native exporter still rejects hosted substitution. Local
HIP ticket scheduling remains distinct from hosted ExpertOverlay execution.
All three windows are driver-clean; no numerical inference ran.

Singleton native lowering authenticates the original sealed owner map and
uses its prepared local expert graph/runtime bank. CUDA1-to-CUDA1 and
ROCm1-to-ROCm1 preparation pass. The new explicit native no-peer sampler role
also passes materialization; request tests now expose a separate missing native
activation-receive edge during the one-token prefix-restore continuation.
Explicit activation ownership now distinguishes an ordinary history forward,
which encloses its native receive/send edges, from a resident generation child
whose exact parent owns those edges. Both CUDA and ROCm singleton pipeline
request/reset/prefix regressions pass with this fix.

The four host-tail fixtures now use valid finite source payloads without
disabling the verifier. They exposed a production CPU expert bounds check that
compared a global model-layer index against a local layer count. The check now
uses the prepared bank's actual global interval, with positive and negative
execution regressions. The CPU-tail fixture now follows the actual scalar
decode contract: two calls, each retired into maintenance, with exactly one
history activation transfer. This exposed a second production defect: the
host-driven PP scalar loop omitted token ingress for middle GPUs. Each stage
now receives the original immutable token input, matching the prefill scheduler;
native parent-owned decode remains on its existing device path. All four native
CPU-tail cases pass, including both three-stage CUDA/ROCm orders, with exact
per-device token, position, padding and activation-byte assertions.
Both singleton lowering controls and all four selected weight-preparation
topologies pass, including both CUDA/ROCm orders. The catalog fixture now obeys
the production Dynamic-only economy-catalog contract.

The coherent build10 passes in about 66 seconds and its immediate repeat has
no work. All 16 focused device-free aliases and all eight affected native
registrations pass. Hosted/native owner controls each exercise 20 request resets
on CUDA and ROCm. Their counters prove exactly two actual forwards per
three-token response and stable graph materialization. Hosted replay uses its
existing successful-retained-replay publisher; native replay uses its existing
authenticated-terminal publisher. The control fixture now selects ordinary
single-device execution rather than an invalid one-tier ExpertOverlay domain.

The complete 20-cell main-model admission matrix passes on build10:
CPU-only, both GPU+CPU ExpertOverlay backends, and CUDA/ROCm/CPU PP, with ten
RAM and ten original-SSD mmap cells. Overlay cases cover static/dynamic
maintenance and ordinal/random placement. Every case initialized the full
48-layer main model and completed teardown; every GPU driver window was clean.
SSD-mmap cases took 30–52 seconds and RAM cases 56–88 seconds. The evidence is
in `parity-results/qwen-flash-next/focused-admission-main20-build10/` (ignored).
These focused initialization proofs do not certify numerical inference, the
deferred MTP stack, or full preflight.

Ngram placement now also has a user-required VRAM target. The implementation
in progress separates requested `auto`/VRAM/RAM/mmap policy from a concrete,
admitted placement. Automatic planning prefers VRAM, then RAM, then SSD mmap,
after fixing all higher-priority weight, expert, KV and workspace owners. A
candidate may not shrink those owners to fit the table. Each actual GPU PLE
consumer needs its own exact native-byte copy, shared across its graph families;
CPU expert-only participants do not count as PLE consumers. Explicit VRAM on
a topology with a CPU PLE consumer must fail admission. Materialization will
use bounded existing weight-upload staging, exact streams/events and PMA
leases, without requantization or a full host mirror. VRAM lookup will gather
and decode directly on device, with no retained host I/O channel. The actual
table occupies about 26.8 GiB, so the 24 GiB CUDA devices cannot admit it; their
automatic choice must resolve to a host tier. These VRAM changes and tests are
not part of the green build10 evidence above.

The reverse four-GPU selected-weight preparation interrupted in native6 was
rerun successfully in native7, including its complete clean driver receipt.
No full Unit/preflight run has been repeated during this admission sprint.

The resident lookup kernel slice is now installed and focused-native green on
CUDA and ROCm. Build13 compiled the affected original embedding and lookup
closure for CUDA80/86/89/90 and HIPgfx906 with fatal spill guards, followed by an
immediate no-work repeat. All ten native registrations passed: both resident
all-format cases, both binary16 payload regressions, both original-source
embedding preservation cases, both grouped embedding verifier cases, and both
hosted QFN primitive suites. Every driver diagnostic window was clean. Evidence
is in `parity-results/qwen-flash-next/native-half-source-payload-{build,native}13/`.

The resident tests exposed CUDA binary16-to-FP32 conversion canonicalizing NaN
payload bits, contrary to the shared native decoder's source-preservation
contract. Diagnostic12 proved that finite values and GPU grouped/serial bytes
were unchanged; only NaN payloads differed from CPU. The canonical helper now
widens exponent31 bits explicitly on GPU, preserving source sign and payload.
Finite values retain the existing intrinsic conversion. Signed quiet/signaling
NaNs, infinities, signed zero and subnormal boundaries have explicit CUDA/ROCm
preflight regressions. This focused correction does not certify the still-pending
VRAM loader/planner/graph integration or numerical QFN inference.

The combined VRAM admission implementation is installed and build17 passes
for CUDA80/86/89/90 and HIPgfx906, including fatal spill checks and a no-work
repeat. Requested policy remains lossless; concrete placement is sealed
only after graph rows, expert quotas and other owners are frozen. Ordinary and
composed-overlay planning share that final admission step. Each actual GPU PLE
consumer retains one original-format table, loaded through a released ordinary
weight-upload slot, then acquired on the exact capture stream. Host tiers retain
their existing explicit I/O boundary. Device-free graph fixtures now obtain a
real metadata-only admission rather than bypassing the placement contract.

The native resident path exposed a second issue: cold prefill legitimately starts
with controller health/error both zero. Lookup now gates on first-error status;
an explicit graph policy preserves prefill errors through response admission and
clears them only at ordered request reset. New tests cover the actual DGO reset
and response initializer, plus the complete loader/ModelContext/arena/stage path.
All 12 focused native registrations pass on build17 with clean driver windows:
both backends' resident owner/stage, prefill failure lifecycle, all-format lookup,
binary16 payload, hosted QFN primitives, and auxiliary admission cases. Evidence
is in `parity-results/qwen-flash-next/resident-table-native17/`. A failed
ordinary-loader drain also now stops before freeing staging or issuing a
released-slot capability.

The first combined device-free run completed all 25 selected registrations;
16 passed and nine failed. The failures expose an incomplete CLI allowed-value
list, a graph fixture without a local-head placement, and planner/overlay
assertions that need diagnosis after tier selection moved to final admission.
These are being fixed as a batch before the real-model matrix is repeated.
Native lookup success does not certify the complete real-model placement path.

Build20 passes with an immediate no-work repeat. The expanded focused run
passes 27 of 28 registrations; the remaining MemoryPlanner executable exposes
21 failures. The complete real-model main20 rerun then fails all 20 cases
before model payload preparation, with clean GPU driver windows. The new VRAM
admission code incorrectly uses rank NUMA affinity as a CPU allocator ordinal:
the supported whole-host observation is `-1`, while runtime allocations use
the rank-local `DeviceId::cpu()` owner. Bound nonzero NUMA affinity is likewise
separate from this allocator identity. The correction restores that existing
contract and derives final ngram ownership from the frozen PLE consumer inputs;
it does not reinterpret an unknown affinity as physical NUMA0. Focused tests
cover unbound and NUMA3 inventories and missing/conflicting consumer host owners.
The previous build10 main20 result remains historical evidence, not a green
certificate for the changed tree.

Additional setup tests are prepared for CPU/CUDA automatic host-tier selection
and explicit resident storage on ROCm0/ROCm1/CPU0 PP with layer counts 1/1/46.
They disable diagnostic snapshots explicitly to test the serving allocation
envelope. Metadata-only planning leaves roughly 718 MiB on the PLE MI50 under
an optimistic full-32-GiB observation; actual admission and materialization are
still required. Separate gold-free tests cover eight full-source active MTP
admissions plus an inactive-sidecar control. These tests use the original four
main shards and explicit predictor source, and do not claim numerical inference.

Build22 resolves all 21 previous MemoryPlanner failures and passes 28 of 32
focused registrations. The remaining four registration failures reduce to two
test defects: a stale exception type expectation and an explicit cross-rank TP
fixture incorrectly delegated to the rank-local automatic search. Both fixes
are installed with stronger frozen-quota and per-rank table-owner assertions.
The complete main20 matrix now passes six cases: CPU RAM/mmap and both ROCm
Dynamic placement orders in RAM/mmap. The other 14 cases expose mandatory
lookup execution costs being added after expert quotas, or a missing resolved
placement handoff from pipeline admission to the runner runtime. All driver
windows were clean. Evidence is in `resident-table-devicefree22/` and
`focused-admission-main20-build22/` under the ignored QFN result root.

The three serving-table admissions pass CPU Auto to RAM. CUDA Auto reaches
the same late lookup-budget defect. The two-ROCm/CPU VRAM case is rejected by
a parity prerequisite that incorrectly requires both GPU vendors for every
heterogeneous topology; it has not reached model admission yet. The nine
full-source MTP admissions pass only the explicit-disabled control. Active CPU
finds a mistaken GPU prepared-embedding requirement; active GPU/PP finds a
main-only GGUF-directory lookup for real external predictor expert tensors.
Dense0 describes the predictor attention layout, not an absence of routed FFN
experts. Neither source ownership nor predictor capacity may be dropped to make
admission pass. These failures are being repaired separately with focused tests.

The next coherent build includes native CPU embedding borrowing, complete PP
runtime-capacity publication, and a typed RAM locality policy. CPU0 remains the
canonical allocator; whole-host locality follows inherited Linux first-touch,
while an explicit NUMA node is certified before source reads. Locality survives
placement, setup wire and ModelContext reuse checks. Pending lookup provisioning
must price mandatory execution buffers before expert quotas while keeping only
the bulk native table as the lowest-priority VRAM owner. These are focused
admission repairs, not a numerical or full-preflight certification.

Build23 completes in 191 seconds with a no-work repeat. Of 36 focused
registrations, 34 pass; the only remaining unique failure is a multi-rank
fixture whose second rank has no PLE placement. All new CPU-native embedding,
PP runtime publication, real NUMA placement and locality reuse tests pass.
The actual CUDA/ROCm/CPU main-model mmap PP case now passes full initialization
and teardown in 46 seconds with clean drivers. CPU active MTP also initializes
successfully; its test still fails a prepared-expert record assertion, whose
CPU ownership source is under investigation. Evidence is in
`resident-table-devicefree23/`, `pp-main-admission-build23/` and
`mtp-cpu-admission-build23/` under the ignored QFN result root.

The four affected native loader/lookup registrations also pass build23 with
clean driver windows: resident owner/stage and hosted QFN primitives on CUDA
and ROCm (`resident-host-owner-native23/`). CPU MTP setup logs confirm reuse of
1,536 predictor expert engines; the test's overlay-only registry count did not
observe that CPU prepared authority. The corrected assertion now inspects
all three source48 CPU slabs, their engine lifetimes and original source format,
and requires exactly 1,536 engines on the whole-source CPU case. It awaits the
next coherent model admission run.

Build24 completes in 193 seconds with a no-work repeat, and all 41 focused
registrations pass. Mandatory lookup execution now belongs to the fixed BOM
before expert quotas; only bulk table storage is selected afterward. Auto uses
the existing mutually-exclusive PMA execution envelope, without a second byte
ledger. Source geometry, RAM locality and the prepaid execution bound survive
the versioned rank wire. Tight-capacity tests prove first-pass quota selection
and reject omitted or duplicate execution charges.

Actual CPU and CUDA automatic placement both pass, selecting RAM. The formerly
failing main-model CUDA/ROCm static-ordinal mmap admissions also pass in 48/49
seconds. All nine MTP cases ran: CPU active RAM/mmap and the inactive control
pass; the six GPU/PP cases expose one shared main/total-layer interpretation
error in overlay input validation. The source has 48 main layers plus predictor
source48; consumers must use the canonical 48/49 accessors. The explicit VRAM
pipeline reaches graph construction but exposes an unconditional nonempty QFN
auxiliary-state owner on a stage with neither PLE nor sparse attention. Both
defects are under focused repair. Every driver window is clean. Evidence lives
in `resident-table-devicefree24/`, `ngram-serving-admission-build24/`,
`mtp-serving-admission-build24/` and `main-overlay-mmap-admission-build24/`.
The full main20 matrix has not been requalified on this revision.

Build25 passes in 97 seconds with a no-work repeat; all 43 focused registrations
pass. Overlay inputs now use canonical main/total layer counts and retain the
predictor's global zero-residency slot before quotas. A source-derived GDN-only
PP interval keeps no QFN auxiliary owner, while its ordinary recurrent cache
and strict geometry checks remain intact. The actual two-ROCm/CPU VRAM case
now passes in 76 seconds: the complete 28,800,138,240-byte original table is
resident only on ROCm1, with an exact PMA claim and no host table mirror. Full
48-layer initialization and teardown succeed with clean driver diagnostics.
The serving-tier fixture no longer resolves or generates numerical references.
All three serving-tier cases pass, including CPU and CUDA automatic RAM
placement. Existing native auxiliary admissions pass on both backends. The new
absence fixtures now admit their exact KV/frontier and recurrent-cache BOM;
CPU, CUDA and ROCm checks pass, including lease retirement and zero unrelated
owner charges. This fixture-only repair rebuilt two test translation units.

The complete build25 nine-case MTP admission run passes the three CPU cases and
fails the six GPU/PP cases at one shared prefix-capacity check. The Dense0
predictor owns one dense attention KV layer but no sparse/PLE auxiliary state;
the helper incorrectly requires those counts to match. Both fixed-BOM planning
and PP runtime materialization reach this helper. The focused repair must retain
the dense KV charge and strict nonempty sparse ownership, with production RAM
and disk prefix tiers enabled in the fast full-BOM regression. The earlier
whole-source fixed-BOM regression disabled prefix caching and missed this path.
All observed driver windows remain clean. Evidence is in
`ngram-serving-admission-build25/`, `auxiliary-participant-native25a/` and
`mtp-serving-admission-build25/`. The full main20 matrix has not yet been
requalified on this revision.

Build26 repairs that shared prefix validation without reducing dense predictor
KV or staging charges. All 45 focused registrations pass after the coherent
30-second rebuild and no-work repeat. New explicit preflight aliases cover
empty shifted auxiliary ownership through the real CPU cache publisher, an
independent compact-record maximum, and 84 full-BOM configurations with RAM
and disk prefix tiers enabled. Those configurations cover CPU/CUDA/ROCm,
RAM/mmap tables, static/dynamic overlay policies, the whole model, and every
stage interval used by the two registered pipeline shapes. Restoring the old
helper fails all three new regression controls. The real-model MTP matrix now
passes five of nine cases: all three CPU cases and both mixed-backend PP cases.
The four CUDA/ROCm continuation cases reach graph construction and expose an
independent shifted-prefill binding defect. That read-only path incorrectly
authenticates draft residual write targets at the full prefill row count. Its
typed read-only repair is being prepared without widening buffers or weakening
draft checks. Every driver window is clean. The full main20 matrix remains
pending admission qualification on the repaired revision.

Resident gather timing with a bounded 2048-CTA grid improved the M1024/live15
case from 13.55 to 4.36 microseconds on CUDA and 51.39 to 20.14 on ROCm.
M512/full improved from 12.54 to 12.00 and 43.16 to 40.52 microseconds;
M1024/full measured 24.19 versus 23.64 and 71.60 versus 69.79 respectively.
All ten timing processes passed byte checks and driver-health checks. These are
fixed-ID captured lookup timings including submission gaps, not cold-DRAM
bandwidth or model-throughput claims. Build16 compiled the complete configured
CUDA80/86/89/90 and HIPgfx906 set without memory spills. The timing artifacts
are `parity-results/qwen-flash-next/resident-lookup-{measure-matrix14,cap2048-matrix16}/`.

Execution qualification still needs VRAM work-counter publication at existing
terminal boundaries. Hosted RAM/mmap statistics cannot stand in for resident
work. The intended observation must retain prefill counts across initial
response admission, include failed terminals, and cover CPU-tail and grouped PP
paths without per-forward readback. This remains separate from admission proof.

### Build29 admission closure and native layout foundation

The shifted-prefill repair distinguishes read-only prefill bindings from draft
residual writes. Full M512 prefill no longer requires the compact draft output
to hold M512 rows; real draft capacity checks remain strict. Its focused alias
is explicitly in production preflight. The subsequent overlay failures were
fixture identity errors: original authored domains, legacy named-domain
projection and effective typed domains are now checked separately. Legacy AUTO
cannot represent canonical SINGLE. The regression binds a real inventory before
checking resolved rank ownership, rejects changed scope intent and checks that
failed publication leaves the original plan intact.

Build29 compiles the coherent production/test target closure in 184.65 seconds
and its immediate repeat reports no work. All 50 focused registrations have
passing evidence after correcting the new singleton fixture's missing inventory
binding: 48 passed in the first run; the corrected full 30-case overlay Unit
suite and both domain-identity aliases passed in the follow-up. This is not a
claim that the original 50-test invocation passed or that full preflight ran.

The frozen binary then passes all 32 real-source admission cases, including
clean teardown and all observed driver windows. The 20 main cases cover CPU,
CUDA+CPU, ROCm+CPU and mixed CUDA/ROCm/CPU PP with RAM/mmap, plus the declared
static/dynamic and ordinal/random overlay combinations. The nine MTP cases
retain the full original source set and separate predictor, including explicit
MTP-disabled admission. Serving coverage proves automatic CPU/CUDA RAM selection
and the 1/1/46-layer ROCm0/ROCm1/CPU0 pipeline with the original
28,800,138,240-byte table resident only on ROCm1. Setup success does not prove
MTP numerical execution, arbitrary depths, model quality or serving throughput.
Evidence is under the ignored `admission-closure-build29/`,
`mtp-serving-admission-build29/`, `focused-admission-main20-build29/` and
`ngram-serving-admission-build29/` result directories.

`ImmutableRowTableLayoutGeometry` now provides checked source-native metadata
for fixed ranges, replicas, adaptive slots, allowed tiers, separate owners and
consumers, and explicit SSD service. Its 19-case regression is explicitly in
preflight. It has no materialization authority or independent capacity ledger;
PMA remains the physical authority. Directory capacity follows admitted GPU
slots plus live requests, not the source table's 320 million rows. Permitted
zero-capacity tiers stay representable. Adaptive operation without SSD is
rejected until its last-readable-copy migration protocol exists. Existing
whole-table serving only uses this geometry for construction validation; mixed
and distributed runtime modes are not yet installed.

Fixed-native-input numerical controls reproduce the saved CPU HC reads and
QKV/Z words exactly. Independent GDN convolution, recurrence and sigmoid gating
agree at about 2e-7 relative L2. Ordinary GDN output-projection Q8 explains the
larger local discrepancy; reconstructing the router on its actual native input
reproduces the first expert cutoff swap. A further control covers all 36 original
Q8_0 output matrices and all 144 saved prefill/decode boundaries. Existing
TwoTerm Q8 reduces median same-input projection error from 0.00904812 to
0.0000480761, improving every local case. Eight gold comparisons worsen because
local quantization noise had cancelled earlier trajectory error; all cases are
retained. This motivates an opt-in candidate, not a new default or an end-to-end
KL/performance claim. Current CPU full-model numerical qualification stays red.
No original quantized weights were converted or table/model payloads hashed.

### Complete build29 gate collection and repair

The complete Unit run reports 743 passing registrations and three failed suites.
The subsequent diagnostic collects all 1,387 Integration preflight registrations
without stopping at ordinary failures: 1,359 pass, two fail and 26 explicitly skip
for unavailable eight-device hardware. All host, accelerator and exclusive driver
windows are clean. This diagnostic does not issue a green prerequisite receipt.
The exact inventory and results remain in
`parity-results/qwen-flash-next/preflight-main29-complete-diagnostic/`.

The three Unit repairs correct an incomplete synthetic host-RCU authority fixture,
source guards that did not follow the current committed-maintenance/snapshot
ownership paths, and a function-body scan that included a neighboring reset
method. Their negative controls retain the ownership and publication requirements.
The two CUDA failures are native pipeline MTP evidence errors: successful native
follower materialization lacked its counter, and the fixture required a hosted
transaction counter even for the complete CUDA conditional parent. The repair
records successful native materialization and requires an actual speculative
native-parent launch for the exact participant, backend and execution policy.
Ordinary depth-zero work cannot satisfy that proof. Hosted transaction checks,
one materialization, repeated graph reuse and real reset/prefix execution remain.

All four reviewed repair packages are installed, with four explicit preflight
aliases. The coherent Unit/preflight target rebuild passes in 107.12 seconds;
its immediate repeat reports no work. All 12 focused request-lifecycle cells
pass: disabled/fixed/dynamic MTP on CUDA1/CUDA1, ROCm1/ROCm1 and both orders of
CUDA2/ROCm2. The first 19-registration run passes 17, with one new negative
source-control fixture failing in its Unit suite and preflight alias. Its mutation
targeted the earlier embedded acknowledgement instead of the native launch gate.
The corrected branch-scoped mutation, full source-guard suite and alias both pass
after a coherent two-action rebuild. All focused driver windows are clean. Evidence
is in `gate29-focused-repairs/` and `gate29-source-control-recheck/`; these are
focused results, not a complete prerequisite receipt. The fresh canonical full
transaction is `prerequisites-main-30/`, before native full-model numerical cells.
This repair changes telemetry and tests, not QFN arithmetic or its remaining KL.

The private CPU adaptive-cache/provider closure passes 24 tests repeated 20 times.
Four deliberate source mutants compile and link, then fail their intended test
assertions. The exact fresh 11-TU closure and both earlier build failures are
retained under `/tmp/qfn-cpu-row-component-closure-run3/`. Two fixture-only brace
corrections are separately recorded; original component packages are preserved.
All original table/ModelContext/contiguous-alias cases still need canonical
qualification. A separately reviewed owner-observation extension reports actual
retained PMA storage/staging leases; its five new cases and serving integration
remain pending. No adaptive serving mode or performance result is claimed yet.

The optional HC correction also has a separate, uninstalled ordinary-query count
repair under review. Main block reads, read-only attention and terminal HC must
retain the original live-row owner for the corrected up projection without changing
ordinary down/injection arithmetic. The private GDN correction has independently
reviewed ordinary-prefill, grouped-M1 and compact-CPU row contracts. Source review
and parameter fixtures do not certify native projection replay or performance.

### Build30 qualification and fixed-input expert attribution

`prerequisites-main-30/prerequisites.json` records a complete successful canonical
transaction: 746 Unit passes, 1,365 preflight passes and 26 hardware skips, in
3,715.99 seconds. The host, CUDA, ROCm and exclusive driver windows contain no
findings. This qualifies that exact unchanged source/build, not later changes.

The following single-GPU/CPU ExpertOverlay mmap campaigns use that receipt:
`main30-cuda-mmap-single/` and `main30-rocm-mmap-single/`. Prefill final-logit KL
is 0.00760721 on CUDA and 0.0194998 on ROCm, with strict layer checks passing
3/48 and 2/48 respectively. Both then fail with an unretired immutable lookup
storage interval. Standalone prefill opens the service interval, but the old
request-reset path advances the session without the terminal close/join normally
performed by a complete generation parent. The admission guard correctly rejects
the replacement epoch. Both driver windows are clean. These are partial numerical
attempts, not completed or passing model campaigns. The scoped repair is now
installed; build31 actual captured RAM/mmap request-lifecycle regressions pass
on both backends. Current-source full prerequisites and model reruns remain required.

A bounded current-core CPU control now reproduces the saved HC-TwoTerm run's
layer 0/row 3 and layer 1/row 3 routed expert outputs byte for byte. It uses the real
prepared NativeVNNI transaction, original quantized expert planes, saved native
inputs/IDs/weights and the production route-slot FMA fold. Changing only the
layer 1 cutoff expert from 143 to 369 in the same slot, with unchanged input and
weights, reduces relative L2 against the independent reference from 0.15189409
to 0.01342266. The unchanged layer 0 membership control is byte-identical. With
membership held fixed, native-vs-independent FP32 equation error is 0.00793633
and 0.00918634 for the two rows. Thus the first large routed-output jump is chiefly
the expert-selection difference, while continuous activation arithmetic error
remains. This does not measure full-model KL or qualify a new numerical default.

The control reads only 21 original expert triples (63 planes), totaling 64,512,000
unique native bytes; reference decode and native bundle extraction together read
129,024,000 selected bytes. No model/table/activation payload is hashed. Evidence
is under `/tmp/qfn-fixed-native-expert-control-prepared-main30/` and
`/tmp/qfn-fixed-native-expert-control-results-main30/`, with the coherent current
ABI and genuine prerequisite receipt recorded by the deferred adapter.

### Build32 full-model diagnostics and next coherent repair

The complete `prerequisites-main-32/prerequisites.json` transaction passes
746 Unit registrations and 1,413 production-preflight registrations, with
26 explicit unavailable-hardware skips and no failures, in 3,771.42 seconds.
Its complete driver window is clean. This receipt authenticates the unchanged
build32 source and binaries only; the following fixes require a fresh receipt.

The unchanged-build diagnostics retain all 48 layers, three decode checkpoints,
and the same-GGUF CPU/FP32 reference. Completed CPU, CUDA+CPU ExpertOverlay and
ROCm+CPU ExpertOverlay cells emit all eight canonical CSVs. Their full and
partial prefix restores reproduce fresh native state exactly; their numerical
parity checks remain red. No threshold or arithmetic default was changed.

| Configuration | Prefill final-logit KL | Decode KL, three checkpoints |
|---|---:|---|
| CPU, mmap, existing single-term Q8 | 0.0113508 | 0.235456 / 0.0455317 / 1.07722 |
| CUDA+CPU, mmap, existing single-term Q8 | 0.00760721 | 0.141026 / 0.116757 / 1.57389 |
| ROCm+CPU, mmap, existing single-term Q8 | 0.0194998 | 0.160465 / 0.0435705 / 0.461583 |
| CPU, mmap, experimental HC-up and GDN-output two-term Q8 | 0.00768937 | 0.0453223 / 0.0269688 / 0.053912 |
| CPU, RAM, same two-term candidate | 0.00768937 | 0.0453223 / 0.0269688 / 0.053912 |

The RAM/mmap candidate pair has identical numerical observations in all six
numerical CSVs; only the explicit cell identity differs. The candidate passes
17/48 strict prefill layer checks and all three top-1 comparisons, but still
fails stage/cosine requirements. It remains experimental and needs matched
Release timing, GPU numerical runs and further accuracy work. These are short
teacher-forced mathematical diagnostics, not HTTP generation, MTP or throughput
certificates. Evidence is under `parity-results/qwen-flash-next/main32-*/`.

The model runs expose three separate contract defects:

- CUDA two-term construction rejects workspace-only graph declarations as if
  they were executable requests. Ordinary nonchunked GPU prefill also needs an
  authenticated borrow of the original request token, position and geometry
  owners. Explicit nonexecutable declaration intent and original-request
  binding address both cases; executable graphs must still reject unowned counts.
- CUDA+ROCm+CPU reaches the first prefill frontier, then rejects
  `QFN_NGRAM_LOOKUP_STATE` as host-authoritative. Its admission kernel writes
  the original GPU protocol state without publishing that owner through
  TransferEngine. Retaining the original Tensor and publishing on the exact
  producer stream repairs ownership; no state mirror or engine reordering is
  needed. Native regression uses the same strict graph frontier.
- The model fixture assumes every immutable-lookup counter describes storage
  work. Named request-reset close/wait counters are lifecycle observations and
  intentionally lack storage/transport tags. A typed classifier accepts only
  those exact lifecycle records while preserving strict lookup-work validation.

The coherent fixed-head TP slice additionally binds actual native collective
receipts to the original generation controller. The pinned ring implementation
accumulates outgoing send bytes: logical contribution bytes multiplied by P-1.
Terminal projection must validate that exact relation before publishing useful
lookup bytes. Native tests initialize and accumulate the real receipts, assert
individual owners, and prove reset windows; summing peers cannot hide missing
or duplicated traffic. The new source and regressions require coherent build,
focused CPU/CUDA/ROCm gates, then full prerequisites before further model runs.

### Build33 focused results and build34 repair, 2026-10-19

The coherent build33 closure and two scoped compile/fixture repairs build for
CUDA sm80/86/89/90 and HIP gfx906 with the spill guard intact. The final build
also verifies a no-work repeat. Complete Unit observes 746 registrations:
744 pass and two fail. The CLI-domain fixture repair subsequently passes its
complete Unit target; the other failure exposes a real TP8 head-assignment bug.

The completed focused native run observes all 232 selected registrations:
221 pass, eight fail and three skip for unavailable devices, with a clean
driver window. Captured fixed-head lookup passes on CUDA TP2, ROCm TP2 and
ROCm TP4, including exact collective receipts, peer-failure cleanup and replay.
CUDA TP4/TP8 and ROCm TP8 remain unqualified on this host. Passing a local TP8
mapping test does not establish an eight-device inference pass.

The eight failed registrations share three causes addressed in build34:

- The replicated-KV splitter rounds local query counts to multiples of two.
  For QFN's 24 query heads, TP8 therefore produces 4/4/4/4/2/2/2/2 instead of
  eight three-head intervals. Those boundaries cannot own integral GDN key
  heads. The repair retains the original global GQA origin/ratio, removes the
  artificial local alignment, and preserves exact GDN boundary rejection.
  CPU and native CUDA/ROCm tests exercise every original interval, malformed
  mappings, captured attention and sparse-attention byte equivalence. Nine
  explicit production-preflight aliases cover the defect.
- A hosted-lookup fixture reuses its first admission event when recording a
  second child graph. It now performs the same current-frontier preparation
  as production before each recording; the executor's strict check is unchanged.
- A repeated host-poison negative test fails to publish its deliberate overwrite.
  It now uses `TransferEngine::publishHostWrite` for each overwrite, keeping
  repeated host-authority rejection meaningful without uploading the poison.

Build34 compiles the complete gate/model target closure in 418.41 seconds,
including all shipped CUDA targets and HIP gfx906, with a verified no-work
repeat. Its complete focused run observes all 986 registrations: 982 pass,
one fails and three skip for unavailable hardware, with a clean driver window.
All nine new head-mapping aliases and all eight previously failing registrations
pass. The sole failure is an older Unit fixture declaring 8Q/2KV across TP8
impossible; original global mapping makes that geometry valid. A fixture-only
repair adds exact one-query ownership and full KV/PMA capacity assertions across
CPU/CUDA/ROCm and reversed participant orders. A globally integral 4Q/2KV TP8
case still rejects its necessarily empty query participants. The capacity oracle
sums independent logical requirements per physical owner: one rank-host allocator
owns eight CPU cache banks; eight GPU allocators each own one. Both cases receive
an explicit preflight registration. The final fixture build verifies no remaining
work, and the complete MemoryPlanner target plus its exact preflight alias pass
in 20.85 seconds with clean driver health (`build34-repair2/` and
`focused34-repair2/`). A fresh complete Unit/preflight transaction still precedes
current-source model reruns. Evidence remains in ignored `build34/`, `focused34/`
and the follow-up result directories.

Offline original-GGUF same-input controls now isolate the layer-3 router cutoff
change. PyTorch reproduces every gold router output exactly and all nine native
expert selections from native inputs; no top-k sorting defect or tie explains
the change. The two-term candidate's row-1 expert 190/374 reversal is already
present in the incoming HC trunk. The HC local contribution opposes that swap,
while the router-local pair contribution is zero. At the first full-attention
output projection, still using SingleQ8, local same-input relative L2 is 0.0103301
versus 0.00618216 propagated input contribution. Original source is Q8_0;
independent FP32 projection exactly reproduces all nine gold rows. RAM/mmap
observations agree. This identifies the next bounded accuracy experiment but
does not prove an equation bug or justify another default. FA-output TwoTerm
wiring remains held pending current GPU/hybrid controls. Numerical defaults,
the original reference and thresholds remain unchanged.

## Current qualification update — 2026-10-20

The complete repaired build34 prerequisite transaction passes 746 Unit and
1,489 production-preflight registrations, with a clean driver window. Its
receipt is now historical: build35 adds the changes below and requires a new
complete transaction before more full-model runs.

Build34 single-GPU mmap diagnostics completed all canonical artifacts
with the experimental two-term HC/GDN operands. CUDA prefill KL is 0.0040319;
three teacher-forced decode KL values are 0.203043, 0.0416161 and 0.260471.
ROCm prefill KL is 0.00387029; decode values are 0.138771, 0.0799268 and
0.298530. Only three and seven of the 48 decoder layers respectively pass every
strict stage comparison. All three top-1 predictions match; full and partial
prefix restores reproduce fresh native state. These are numerical failures,
not a model qualification or throughput result.

The CUDA/ROCm/CPU PP diagnostic completes prefill (KL approximately 0.0090)
but traps on the first ROCm decode. Metadata-only GPU/host inspection identifies
the cause: valid source descriptors for logical experts above 255 are absent
from the runtime scratch table. Both backends launched 256 publication lanes
and visited one expert per lane, although QFN has 512 experts. The same omission
affects floating descriptor-only publication. Safe old-core regressions now
reproduce both omissions on CUDA and ROCm while the four-expert control passes;
all driver windows are clean. The repair visits ascending fixed-width chunks,
retains stable compaction across chunks and pads once. Focused captured tests
cover full/partial/empty lists, partial final chunks, all three floating formats
and actual upper-half native expert arithmetic. Corrected-core qualification
and the hybrid rerun are still pending.

An exact ordinary-Q8 quantizer control isolates the first full-attention output
projection's approximately 1.03% local relative-L2 error. Projecting the engine's exact Q0 operand against the original weights
agrees with the saved engine output within 5e-7 relative L2. Adding a second
quantized residual operand reduces the isolated error to approximately
4.4e-5 against the same original Q8_0 source, without changing weight bits.
These saved-input controls are not end-to-end KL or speed evidence.
Build35 adds an independent opt-in FA-output two-term selector, complete
policy/capture/prefix/capacity identities, actual CPU/CUDA/ROCm stage regressions
and grouped correction kernels. Existing defaults and numerical gates remain
unchanged; matched model and Release economy results are required before any
promotion.

The coherent build35 set also contains the bounded native GPU ngram-cache
component and CPU source-grant protocol, with explicit Unit/native preflight
regressions. The public adaptive GPU serving route remains unadvertised while
request/resolved admission, original-root leases, stage/boundary wiring and
terminal lifecycle are completed. Component correctness alone does not certify
mixed-tier serving, distributed hot-row ownership or MTP.

## Component tuning evidence

These are isolated kernel results, not model throughput certification. The
current production-flag QSA evidence and revision-qualified timing matrix live
in the ignored `parity-results/qwen-flash-next/qsa-kernel-evidence.md` report;
earlier standalone builds missing production flags do not qualify performance.

- ROCm QSA M1/64K fell from 3.78 ms to 0.849 ms after small-M selection tuning.
- CUDA FP16 QSA M1/16K core attention measured 93.7 us, versus an initial
  1.83 ms candidate. The tuned launch used 192 CTAs, 256 threads, 56 registers,
  about 38.8% occupancy and zero spills.
- CPU QSA uses runtime AVX2/AVX512 dispatch and nested-safe physical-core
  work sharing. At 28 cores, M1 measured 2.339/2.276/3.053 ms at 4K/16K/64K;
  M512 measured 136.343/163.353/310.961 ms. Profiling and exact worker/ISA
  parity evidence remain in ignored local artifacts.
- Residual/PLE CUDA M1024 primitives measured 86–95% occupancy, with 22–28
  registers and zero spills. Native immutable-row kernels compile without
  CUDA spills for sm80/86/89/90 and HIP gfx906 without scratch/spills.
- Native IQ4_NL160 CPU lookup measured about 473 us on one physical core and
  48.5 us on 28 at M1024. M15 measured best at four workers (3.976 us); higher
  thread counts are not a small-request speed claim. AVX2 and AVX512 match exact native values
  for every tested positive worker count through 28.
- Sigmoid GDN CUDA M15/M1024 measured 2.523/99.451 us unprofiled. Nsight reports
  32 registers, no local traffic and 65.5/91.1% achieved occupancy. ROCm reports
  20 VGPRs, 32 SGPRs, 512 B LDS and no scratch/spills, with 96.4% VALU utilization.

The experimental two-term Q8 HC-up correction retains native Q8 weights and
improves the numerical simulation, but its first large-prefill implementation is
not economical. Clean paired measurements show approximately 5.8–6.1 times the
single-term latency on CUDA and 2.8–2.9 times on ROCm at M512/M1024. It remains
experimental while grouped operand reuse is implemented. These ratios describe
that projection, not total model throughput. The clean timing/ROCm profile
window has no driver findings. An earlier CUDA profiler window produced NVIDIA
driver assertions, so its reports are diagnostic only and further CUPTI
attachment is paused pending a tooling resolution or an explicitly authorized
isolated investigation.

## Remaining implementation

1. Execute and qualify the registered `qwen4exp` main graph, then complete
   multi-request and accepted-prefix MTP bindings. Compilation and component
   passes do not certify model inference support.
2. Finish graph-owned immutable lookup admission, failure publication, retained
   parent termination and teardown. One exclusive channel serves each concurrent
   GPU continuation for RAM/mmap; all hosted channels share the rank-owned
   native table. Add explicit replicated and sharded VRAM layouts. Sharded
   placement must account each native source range once, support separate
   storage owners and lookup consumers in TP/PP/EP, and prove compact transfer
   extents, empty owners and large-to-small replay. The current resident path
   still stores the complete table at each GPU lookup consumer.
3. Attach typed auxiliary state to the canonical cache lifecycle. MTP rollback
   retains small lexical/indexer tails; prefix archives retain only live pooled
   indexer rows. Rejected future pools stay unreachable behind the restored
   canonical frontier. Do not copy full-context pools on every decode step.
4. Complete residual-width propagation through pipeline transport, terminal
   hidden storage and the separate predictor. Keep transferred row counts live,
   independent of allocation/bucket capacity.
5. Run the integrated exact integer snapshot/CSV comparisons and first-divergence parity
   against the same GGUF for prefill and serial/grouped decode.
6. Qualify CPU-only, ROCm+CPU, CUDA+CPU and CUDA+ROCm+CPU with RAM and mmap,
   request reset, prefix hits and dynamic MTP where an exported predictor is
   admitted. Keep full Unit, focused backend tests and preflight green.
7. Measure Release end-to-end inference, memory bounds, lookup behavior and
   context scaling, then tune the remaining production bottlenecks.

The current QSA storage contract covers execution within the admitted context,
including nonzero physical ring origins. True sliding eviction would regroup
four-token pools and needs additional explicitly admitted indexer state; it must
not be silently enabled by wrapping the existing pool indices.

## Qualification matrix

| Topology | RAM n-grams | SSD mmap n-grams | Numerical parity | Release performance |
|---|---|---|---|---|
| CPU-only | Pending | Pending | Pending | Pending |
| ROCm+CPU | Pending | Pending | Pending | Pending |
| CUDA+CPU | Pending | Pending | Pending | Pending |
| CUDA+ROCm+CPU | Pending | Pending | Pending | Pending |

## Build35 focused results and build36 integration (2026-10-20)

The complete Unit plus affected native/ownership run in `focused35/` executes
1,028 CTest entries in 388.6 seconds: 1,017 pass, eight registrations fail and
three explicitly skip unavailable CUDA TP4/TP8 and ROCm TP8 hardware. Driver
diagnostics are clean. The original selection report is retained; its separate
expanded-inventory audit accounts for one implicitly added model setup fixture.
The two model-free MoE compaction registrations now explicitly declare
`NO_MODELS`, removing that unrelated fixture from subsequent focused runs.

The 512-expert descriptor publication and actual captured decode regressions
pass on both CUDA and ROCm, including upper-half expert IDs. Existing adjacent
MoE checks, grouped two-term Q8 kernels, FA-output stages/composers, CUDA/ROCm
TP2 and ROCm TP4 fixed-head lookup checks also pass. Full hybrid-model decode
remains to be rerun after current-source prerequisites.

The eight red registrations reduce to two fixture defects: the FA workspace
oracle omitted compact-family storage and misattributed an HC family, while
all six adaptive native registrations used the independent CPU cache factory
for a subordinate GPU source. Build36 includes exact family-envelope tests and
the canonical subordinate provider factory. No numerical thresholds or
production workspace accounting change in these repairs. Their execution is
pending; fixture setup failure did not exercise the adaptive native arithmetic.

Build36 also integrates the explicit single-GPU adaptive VRAM/RAM/SSD request,
source-owned mandatory BOM, lowest-priority GPU-then-RAM cache fit, retained
PMA-backed roots, actual model-context and participant-local stage wiring. One
model-owned source/cache boundary is shared across serialized graph families.
Setup and admission retain their exact completion events through exceptional
retirement. Nine owner-authored work counters use the existing canonical
terminal controller copy, including real hits, misses, promotions and evictions.

These changes have source review and focused regressions authored; compilation,
native admission/stage/retirement tests, current-source complete prerequisites,
full-model parity and Release economy/performance qualification are pending.
The explicit mixed GPU mode is not selected automatically or advertised by a
new CLI/E2E profile. Distributed adaptive ownership and final MTP/HTTP coverage
remain open parts of the plan.

## Build36 component results and build37 model-cell integration (2026-10-20)

The coherent build36 and collected fixture-API repair compile successfully for
CUDA SM80/86/89/90 and HIP gfx906 with the spill guard enabled. The complete Unit
plus affected preflight/native run in `focused36/` accounts for all 1,057 exact
CTest registrations: 1,043 pass, 11 fail, and three explicitly skip unavailable
CUDA TP4/TP8 and ROCm TP8 hardware. Elapsed time is 415.6 seconds; driver
diagnostics are clean. This focused run is not a full prerequisite certificate.

Every new adaptive admission, source ownership, stage, exact-original-root,
submission retirement and terminal counter regression passes on both CUDA and
ROCm. The existing adaptive native all-format, live-row/reset/U0 and failure
isolation tests also pass. The prior FA workspace and subordinate-provider
fixture failures are resolved. The 11 red registrations reduce to three stale
test assertions: total controller extent still expected 72 rather than 90 words,
and two shared readmission fixtures expected a lifecycle exception instead of
the retained fatal source diagnostic. Build37 updates those assertions without
changing production arithmetic, admission policy or numerical thresholds.

Build37 adds four explicit real-GGUF canonical adaptive GPU cases: CUDA+CPU and
ROCm+CPU, each with zero optional cache slots or bounded positive VRAM and RAM
cache ceilings over the mandatory SSD source. The new suite prefix is short
enough for the canonical flat artifact directory; regressions retain complete
configuration identities and reject the original overlong positive names. They use the existing main 48-layer
prefill/decode/prefix comparisons, not an alternative runner. Ordinary CLI/YAML
projection and strict actual-owner/device-terminal/CPU-source evidence accompany
the cases. Exact executable-family PMA admission is also added to native test
fixtures, including their additional warm graph. These additions still require
compilation and execution; four passing model admissions or numerical gates are
not yet claimed. Automatic selection, distributed adaptive topology support and
E2E eligibility are unchanged.

Next qualification is focused current-source regressions, then one fresh full
Unit/production-preflight transaction before the canonical real-model campaign.
Numerical diagnosis, distributed adaptive placement, MTP/HTTP coverage and matched
Release performance remain open work toward the full QFN goal.

## Build37 focused qualification and cancellation control (2026-10-20)

The complete Unit plus affected registered native/preflight run in `focused37/`
accounts for all 1,063 selected registrations: 1,060 pass, none fail, and three
explicitly skip unavailable CUDA TP4/TP8 and ROCm TP8 hardware. It completes in
410.1 seconds with a clean driver window. All eleven build36 fixture failures
are resolved. The new adaptive model-inventory, CLI, evidence and artifact-name
checks pass; actual real-model adaptive inference is still unqualified.

Both native `AdaptiveACKReadyCancellation` registrations pass all zero/bounded
cache, physical-row and retained replay cases. A separate controlled negative
removes only the two Ready-publication cancellation guards. Both backends then
fail first at the intended promotion counter, followed by the extra Ready row
and row-directory assertions. This proves that the regression detects publication
after cancellation, rather than merely failing during setup. Driver diagnostics
are clean in both negative and restored positive windows.

The exact guarded source is restored, the complete build closure succeeds for
CUDA SM80/86/89/90 and HIP gfx906 with spill guards, a repeated build reports no
work, and both cancellation aliases pass again. Evidence is retained in
`ready-cancellation-negative37/`. No negative source or numerical-policy change
is installed. A fresh complete Unit/production-preflight transaction is next;
the historical build34 receipt and this focused result cannot certify the
current real-model campaign.

## Build37 full-gate failure and worker repair (2026-10-20)

`prerequisites-main-37/` collected all 2,284 Unit/preflight registrations in
4,006.8 seconds. Its sole failing registration was the ROCm adaptive lookup
original-format test; the receipt is failed and cannot authorize model runs.
The driver window was clean. Two focused unchanged-production runs reproduced
the timeout on their fifth invocation, across different codecs and optional
cache sizes. The diagnostic run retained a valid first request, protocol status
9 (`TimedOut`) and a completely untouched CPU response before the first close.

The cause was construction of `RetainedParentTicketServiceWorker`: its thread
started in a member initializer before the lifecycle members declared after it
were initialized. Reused storage could still contain the previous lifetime's
`Stopping` state, allowing the new thread to exit before any request was served.
Starting the thread in the constructor body establishes initialization before
thread entry. No timeout, inference arithmetic or cache policy changed.

An isolated device-free regression forces the actual worker to run before its
thread constructor returns and reconstructs it in the same untouched storage.
The original code fails with the intended early-return diagnostic on the first
reused lifetime in 0.52 seconds. The fixed code passes 20 consecutive runs;
existing worker/graph lifecycle tests pass, and CUDA and ROCm each pass 20
consecutive original-format runs. Both backend loops cover 23 native formats,
zero/one optional GPU slot and retained readmission. Driver diagnostics are clean.
Evidence is in `worker-construction-negative37/`,
`worker-construction-positive37/` and `adaptive-format-status37/`. The new test
is registered explicitly in Unit and `ProductionTestPreflight`.

The format fixture now preserves the first protocol/controller observation
before attempting close and cannot continue a success replay after failed
observations. Full current-source prerequisites and real-model numerical
qualification remain pending. The next coherent batch also strengthens complete
48-layer prefill acceptance and ordinary decode witnesses, and prepares the
canonical paired-source MTP reference/snapshot path; its model tests have not run.

## Build38 numerical evidence and predictor qualification (2026-10-20)

The source batch is installed and its complete build closure passes. Complete QFN prefill now
requires embedding, logits and all 48 distinct decoder layers to pass; an
embedding record cannot replace a missing or failed decoder layer. Ordinary
decode retains its existing comparable-stage averages and thresholds while
also requiring finite raw data and exact integer witnesses. The shared tensor
comparator rejects nonfinite operands and overflowing FP32 errors before error
histogram indexing. These are evidence-integrity repairs, not relaxed numerical
budgets or new production precision defaults.

The same batch connects the original shared Dense0 predictor to the canonical
same-GGUF reference generator and adds a separate numerical MTP matrix. Its
four topology families each declare Off, fixed depths 1/2/3/15, and Dynamic.
The generator reuses the authenticated main reference pack and publishes the
predictor's original 4H hidden input and recursive state; normalized H-wide
head output cannot substitute for that state. The graph exposes a read-only
snapshot of the original wide input without adding an inference copy or a
second state owner. These declarations remain unqualified candidates and do
not add E2E eligibility.

Focused model-free regressions have explicit production-preflight aliases.
Qualification proceeds through the complete build, focused current-source
tests and one fresh full prerequisite transaction, then hybrid main inference,
the four CUDA/ROCm adaptive-cache cells and the first CPU predictor numerical
cell. Historical failed build37 prerequisites cannot authorize those runs.
The separate native variable-size collective packages remain private pending
their dependency rebuild, native resource and transport qualification.

The initial focused run completes all 1,071 registrations: 1,066 pass, three
skip for unavailable hardware, and two registrations expose the same CLI test
fixture error. The server-argument projection intentionally excludes the main
model, which belongs to its outer invocation; the fixture now tests that
contract and supplies the outer model explicitly before parsing. Both affected
registrations pass in `fixture-repair38/`. The original failed record in
`focused38/` is retained. No production parser change was needed.

QFN MTP now has its missing independent selected-expert reference. It reads
only the authenticated original selected GGUF expert matrices and canonical
FP32 reference operands. QFN's four-branch residual suffix remains canonical:
expert evidence cannot rewrite combined-output, HC residual, final-norm or
logit verdicts. The ordinary-family RMS reconstruction is an explicit separate
contract and retains its prior arithmetic. The nine-path reviewed package and
corrected cnpy/ZLIB and Python worker registration are recorded in
`mtp-expert-reference-install38/`. Its complete build takes 126.60 seconds and
the immediate repeated build does no work (`build38-repair3/`). All eight
focused registrations pass in 14.08 seconds, including the complete affected
Unit targets and 22 independent Python equation/schema cases
(`expert-reference-focused38/`). Actual full-model predictor execution remains
unqualified.

The native adaptive component timing sweep completes 28/28 cases with exact
decoded bytes and clean driver windows (`adaptive-component-timing38/`). It
uses original IQ4_NL rows, CUDA and ROCm, physical M1/M512/M2048, source/RAM/
mixed/VRAM tiers and partial/empty replay. At M512, warm source versus warm
VRAM medians are 13.63 versus 28.02 ms on CUDA and 21.07 versus 88.85 ms on
ROCm. All-hit VRAM cells prove zero CPU requests and zero source sequence
advance. These are Integration component measurements with uncontrolled OS
page-cache residency, not cold SSD throughput or Release model performance.
The serial policy, validation and pin-retirement loops require attribution
before optimization. A proposed bulk-hit transformation is under independent
state-equivalence review; no cache policy or kernel optimization from that
proposal is installed.

Four isolated native CUDA graph-event observations locate the M512 all-hit
cost: policy takes 21.16 ms of a 28.02 ms last replay; a separate observation
places pin retirement at 5.42 ms of 28.03 ms. C0 source observations report
4.33 ms in policy and 1.66 ms in retirement. Each process brackets one kernel,
preserves the full graph, matches unobserved live/source counters and exact
output, and has clean driver diagnostics. Observed transaction medians are
within 0.8% of the unobserved controls. These last-replay event intervals include
scheduling/event overhead and are not hardware-counter or Release evidence
(`adaptive-native-events38-separated/`). The first attempt's combined stdout/
stderr interleaved a counter line; its parser failure and clean driver record
are retained separately in `adaptive-native-events38/`. The corrected runner
keeps both streams separate. No CUPTI attachment was used.

## Full prerequisite pass and matched numerical controls (2026-10-20)

`prerequisites-main-38/` completes all 747 Unit and 1,547 preflight
registrations in 3,890.96 seconds, with no failures and a clean driver window.
The 29 skips require hardware absent from this host, chiefly four/eight CUDA
devices or eight ROCm devices. This qualifies the build38 model-free closure;
it does not qualify QFN model accuracy or performance.

The hybrid CUDA/ROCm/CPU main control now completes all three decode steps.
The earlier 512-expert descriptor failure is resolved. Its prefill final KL is
0.0090393 and decode KL is 0.115569/0.0547513/0.491987, so it remains numerical
red. Full and partial native prefix state restoration are exact. The local
queue wrapper misclassified its colored GTest completion banner; a separately
reviewed SGR-only parser fixes that classification while retaining the original
raw logs and failed queue receipt. No engine run was repeated for this wrapper
issue (`model38-expert-ready-sequential/`).

Eight matched controls in `model38-controls-direct/` complete with all required
artifacts and clean driver windows. All retain the original GGUF, reference,
prompt, three decode inputs, mmap storage, optimized expert execution and prefix
cache. HC+GDN means TwoTermQ8 for those two projections; all-three additionally
uses TwoTermQ8 for the full-attention output projection. These are diagnostic
selectors, not changed production defaults or performance measurements.

| Backend | Projection control | Router control | Prefill KL | Decode KL, steps 0/1/2 |
|---|---|---|---:|---|
| CPU | HC+GDN | Native | 0.00768937 | 0.0453223 / 0.0269688 / 0.053912 |
| CPU | All three | Native | 0.007848 | 0.0825384 / 0.0154821 / 0.136449 |
| CUDA | HC+GDN | Optimized | 0.0040319 | 0.203043 / 0.0416161 / 0.260471 |
| CUDA | All three | Optimized | 0.00675852 | 0.0162174 / 0.0612932 / 0.164837 |
| CUDA | All three | FP32 | 0.00675852 | 0.0231793 / 0.021689 / 0.134988 |
| ROCm | HC+GDN | Optimized | 0.00387029 | 0.138771 / 0.0799268 / 0.29853 |
| ROCm | All three | Optimized | 0.0223242 | 0.11058 / 0.0880011 / 0.24505 |
| ROCm | All three | FP32 | 0.00610789 | 0.0100147 / 0.0377377 / 1.20033 |

Every complete model control remains red under unchanged stage/logit gates.
Current HC+GDN controls reproduce the archived scalar results. Improved local
projection accuracy does not yield monotonic final KL. Early positional expert
contribution mismatches must be distinguished from changed expert membership;
complete expert and downstream failures remain real. At decode step 2, the
FP32-router ROCm arm first changes expert membership at layer 10, while CUDA
first changes it at layer 25. All three inputs are teacher-forced IDs
13/271/248045, and the first two output tokens agree, ruling out earlier token
feedback as the cause. `matched-controls-summary.json` retains the original
artifact paths and observations; no threshold or default is promoted.

The first CPU fixed-depth-1 MTP control generates the authenticated original
predictor sidecar reference, then fails before successful prefill:
`MTP sidecar could not lease a current persistent terminal-hidden publication`.
Its missing numerical artifacts are correctly rejected; the independent expert
reference has therefore not yet been exercised by this real-model cell.
This runtime publication defect is a separate focused repair obligation
(`main38-cpu-mtp-depth1-mmap-single-expert-ready/`).

The adaptive zero-slot CUDA main control also exposes a setup-order defect:
`setArena` acquired the runtime source boundary before the factory materialized
the source after ordinary weights. The build39 repair keeps arena declaration
free of source acquisition and preserves strict placement/source/root checks
when actual PLE stages are built. Unit and real ModelContext native regressions
cover missing and changed owners, original roots, PMA claims and retirement,
with explicit preflight aliases. Build39 also installs only the reviewed
lexical control-access candidate for adaptive-cache measurement; the ordered
algorithm, persistent arrays and C0 policy remain unchanged. Compilation,
focused correctness and matched component timing must precede any economy
claim. The separate cooperative C0 and native ragged-collective packages remain
uninstalled. Full model prerequisites must be refreshed after this source batch.

## Build39 adaptive-cache qualification and next runtime repair (2026-10-20)

Build39 compiles the complete Unit/preflight and QFN model-test closure for
CUDA SM80/86/89/90 and HIP gfx906 with spill enforcement intact; the repeated
build is a verified no-op. `adaptive-focused39/` records 807/807 passing
registrations (all 747 Unit tests plus 60 affected integration registrations),
zero skips/failures and clean driver diagnostics. The actual ModelContext
declaration/source-binding regressions pass on both GPU backends. A fresh full
prerequisite transaction is still required before another real-model run.

The matched native component experiment `adaptive-component-timing39-a/`
completes all 28 cases with byte-exact output, matching source/traffic counters
and clean drivers. Only lexical policy/control access changed; original cache
operations, order, persistent capacities and formats remain fixed. Warm-VRAM
complete transaction medians are:

| Backend | Live rows | Previous build, ms | Lexical-control A, ms |
|---|---:|---:|---:|
| CUDA | 512 | 28.0172 | 25.8651 |
| CUDA | 2048 | 136.8504 | 128.3959 |
| ROCm | 512 | 88.8483 | 66.5455 |
| ROCm | 2048 | 355.2201 | 265.9689 |

These are Integration component timings for original IQ4_NL rows of width160,
not Release model throughput or cold-SSD measurements. The OS page cache is
uncontrolled. Source-only, host-cache, mixed-tier, partial-live and empty frames
are also retained in the artifact. Warm VRAM causes no source-worker requests,
but remains slower than the source-only component: cache economy is unfinished.

Separate source-bound native CUDA event observations identify the remaining
M512 warm-VRAM policy at 19.410ms of a 25.914ms replay, and finish at 5.024ms
of a separate 25.890ms replay. Each observation instruments one named kernel;
whole-transaction medians stay within 0.5% of the uninstrumented measurement.
`adaptive-native-events39-a-separated/` retains exact live-work counters and
driver/source identity. No CUPTI/Nsight attachment was used. Static object
evidence in `adaptive-object-resources39/` shows zero spills/private storage:
CUDA policy48 registers and finish54 (52 on SM90), and gfx906 policy34/35
VGPRs and finish35/36 VGPRs with 128B LDS. Static resource evidence does not
claim hardware-counter utilization on unavailable architectures.

The next installed repair is limited to the reached CPU MTP bootstrap defect.
The original FP32 4H arena mailbox is validated, its old read lease invalidated,
its first row zeroed, and host coherence published before the distinct typed
`VisibleZeroHiddenBootstrap` publication. GPU bootstrap and strict sidecar
read-lease checks remain unchanged. Six real-producer/device-free regressions
and an explicit preflight registration are authored; build40 qualification and
the same CPU Depth1 real-model rerun remain pending.

## QFN router default and final-output qualification (2026-10-20)

The user selected FP16 router weights and hidden operands with FP32
accumulation as the QFN default across CPU, CUDA and ROCm. Main GDN QKV,
Z and output projections remain native; alpha/beta retain the existing FP16
MoE policy. Expert GEMM continues to use its original quantized weights and
Q8 activation arithmetic. Explicit native, full-FP16 GDN and other diagnostic
selectors remain independent. This selection does not change other model
families' defaults.

The complete build47 model-free prerequisite transaction is green:
2,318 passes, 29 skips requiring unavailable hardware, and no failures among
2,347 registered tests. The driver window is clean and source/binary identities
remain unchanged throughout that transaction. `prerequisites-main-47/` and
`preflight-driver47/` retain the evidence. This receipt covers build47, not the
subsequent default or acceptance-policy changes.

Current build47 explicit router-FP16 controls complete on all three backends,
using the same original GGUF and CPU/FP32 reference, native GDN plus FP16
alpha/beta, and unchanged teacher-forced inputs. GPU controls use the declared
static eight-experts-per-layer GPU tier with remaining experts on CPU.
All required numerical artifacts and driver windows are complete. These are
Integration numerical measurements, not Release performance measurements.

| Backend | Position | Mean final-logit KL, nats | Top-1 agreement | Top-5 overlap |
|---|---|---:|---:|---:|
| CPU | Prefill final position | 0.0122444 | 100% | 100% |
| CPU | Decode 0 | 0.188283 | 100% | 80% |
| CPU | Decode 1 | 0.0665847 | 100% | 80% |
| CUDA + CPU | Prefill final position | 0.00977154 | 100% | 80% |
| CUDA + CPU | Decode 0 | 0.0462046 | 100% | 80% |
| CUDA + CPU | Decode 1 | 0.0745501 | 100% | 80% |
| ROCm + CPU | Prefill final position | 0.0152396 | 100% | 100% |
| ROCm + CPU | Decode 0 | 0.305130 | 100% | 80% |
| ROCm + CPU | Decode 1 | 0.0616232 | 100% | 80% |

KL is KL(reference || native), summed across vocabulary and averaged across
final-logit positions separately at each step. Top-1 agreement is the fraction
of positions with the same highest-ranked token. Top-5 overlap is the mean
intersection size of the two top-five token sets divided by five. It is not
the separate test that the reference top token occurs in the native top five.
These cells expose one final-logit position per phase. No maximum across
steps or layers is used to rank their output accuracy.

The original three-step observations and strict failing verdicts remain
unaltered in `fp16-menu47/accuracy/`; the combined report is
`fp16-menu47/router-policy-three-backend-summary.json`. In particular, the
third decode step remains available even though the user has now selected
prefill plus two teacher-forced decode steps as the QFN regression horizon.
This shorter diagnostic is not a long-generation quality certificate.

The next source batch installs the family-specific default and a typed QFN
final-logit acceptance contract, retaining complete layer diagnostics,
non-finite/shape/integer validity checks, production-path and prefix-restore
proofs. The provisional conservative limits are KL below 0.05 for prefill and
below 0.10 independently for each of the two decode steps. CPU and ROCm
decode-0 observations above remain red against that proposed limit; limits
will not be fitted to make every backend green. MTP exactness and other model
families' numerical contracts remain separate.

After rebuilding and qualifying the new source once, run the remaining
canonical QFN backend/topology cells with these defaults and report all three
output metrics, failures and unavailable hardware explicitly. Release
throughput comparison and real-model MTP bootstrap requalification remain
outstanding; the build47 Release binary has been built and its loader closure
checked, but no matched Release menu timings have completed.

The default-policy source is now installed. Its first Integration rebuild
completed successfully for CUDA SM80/86/89/90 and HIP gfx906, including the
production application. All eight selected device-free suites passed: the four
affected Unit targets and explicit default selection, prepared-source,
planning-wire and prefix-identity preflight regressions. These focused results
are in `build48-default-focused/` and `focused48-default/`; they do not replace
the next complete prerequisite transaction or fresh full-model observations.

The paired MTP matrix now requests the same exact arithmetic counter families
as the main matrix, so the final comparison can inspect actual router operands
and prepared sources. The focused registration regression failed against the
old filter and passed after the correction. This changes observation only,
with no MTP execution, depth, numerical threshold or benchmark-policy change.

## Accuracy corpus and retained numerical diagnosis — 2026-10-20

The user authorized adapting QFN's accuracy regime while retaining the ability
to isolate numerical failures at every layer and stage. The nine-token prompt
and two teacher-forced decode observations are useful execution and diagnostic
cases, but are insufficient evidence for selecting a model-wide precision
policy. KL is a per-position distribution difference, not accumulated error:
different positions can have very different sensitivity to the same arithmetic.
The current ROCm observations are reproducible; their approximately 0.305 and
0.062 decode KL values reflect real differences in token probabilities, not a
metric implementation discrepancy. Recomputing the same saved rows with full
FP64 normalization and no probability cutoff changes KL by less than 1.2e-7.

The qualification regime will have two connected parts:

1. Keep the canonical all-layer prefill/decode fixture, exact integer/shape/
   finite checks, prefix restoration, captured-path evidence and all six
   numerical CSVs. Preserve routing IDs, weights and source-row identities.
   Same-input operator tests distinguish a local arithmetic defect from
   upstream drift and changed expert selection. Neither an aggregate output
   score nor a successful external engine can waive a local correctness,
   state, communication, or serial/grouped byte-equivalence failure.
2. Extend the existing full-logit canary/corpus framework with fixed QFN
   histories and explicit topology/configuration identity. Reuse one admitted
   production runner across cases. Save full FP32 vocabulary rows after final
   prefill and each observed teacher-forced decode, and compare against the
   same original-GGUF CPU/FP32 oracle. Token histories, checkpoint positions,
   reference metadata and case inventory are frozen before candidate timing.
   Both engines consume the same history; independently generated text is
   separate functional evidence.

Start with a fast panel of multiple independent short histories covering
prose, Python, structured tool/JSON content, multilingual text and Unicode,
including the existing failing history. Retain two observed decode steps per
fast case. Add a separate extended panel with longer continuations and
contexts on either side of actual prefill-bucket, four-token pooling and QSA
selection boundaries, plus long-context workloads used in performance tests.
The current QSA oracle selects at most 512 complete four-token blocks and
then appends the live incomplete tail; tests must cover selection actually
discarding a completed block, not merely a large requested context capacity.
Panel sizes and run costs are measured before CI scheduling is finalized.
Repeating one prompt does not increase independent content coverage.

Use one shared full-tail, temperature-one, FP64 metric implementation for the
corpus: KL(reference || candidate), total variation, exact top-1 agreement,
top-five set intersection divided by five, reference entropy/margins and
probability error at the reference top token. Keep prefill and decode,
context groups, backend/topology and arithmetic policy separate. Report mean
KL, quantiles and worst positions; weight histories explicitly so a long
sample cannot conceal regressions in shorter ones. Compare a candidate to a
reviewed baseline at the exact same row identities and retain the paired
differences. Report uncertainty by history, not by treating neighboring
positions as independent samples. Uncertainty reporting does not replace
deterministic regression checks on the frozen panel.

Acceptance needs both an absolute fidelity envelope and a paired
non-regression envelope, with explicit tail/catastrophic-row limits and
top-1/top-five/probability-drift requirements. No automatic threshold fitting
or baseline regeneration is allowed. Initial corpus collection is explicitly
uncalibrated and cannot produce a green accuracy certificate. Calibrate and
review limits using complete CPU, ROCm+CPU, CUDA+CPU and hybrid observations,
plus the matched Castagna Veloce measurements for context; another engine's
result does not prove our reference or implementation correct. The historical
0.05/0.10 short-case limits remain recorded as provisional and are not a
calibrated corpus standard.

Each corpus row must identify an exact replayable prompt/teacher history,
reference pack, canonical topology and original checkpoint. A failing case
can then run through the existing complete layer/stage fixture without
changing arithmetic or replacing it with eager execution. Full stage
snapshots need not be saved for every long-corpus position on every tuning
iteration; they remain mandatory for the canonical detailed diagnostic and
must be reproducible for the selected failing history. Large corpus payloads
remain ignored locally and belong in `Llaminar/corpora` when published.

Focused device-free regressions must reject missing, duplicated, reordered,
truncated and nonfinite observations, mismatched histories/configurations,
unapproved calibration, and incomplete phase/context groups. Metric controls
include constant-logit shifts, near-tie argmax changes with tiny distribution
drift, identical argmax with large probability drift, and a corrupted single
row hidden by an otherwise small mean. Register those regressions explicitly
in production preflight. Native qualification separately proves captured
execution, actual placement, repeated-run/reset equivalence and complete
retirement on every declared backend. The existing native canary's
single-device guard must be replaced by explicit supported topology contracts,
not simply removed; unsupported forced-token transactions must fail precisely.

The first implemented panel fixes 16 short histories and 48 scored rows. All
16 share the `short` context group: prose, Python and Unicode categories are
content coverage, not separate context-length acceptance buckets. Prefill,
decode offset zero and decode offset one each retain their own 16-history
mean, with the combined decode mean reported separately. A final replay of the
original history checks exact reset equivalence but contributes no extra vote
to the accuracy averages.

While tracing the native observer, a separate LocalTP diagnostic defect was
reproduced before repair: resident condition forwards updated child logits
without retiring or refreshing the rank's preceding host result. Pending token
publication is not a forward and must preserve that result. The repair uses a
typed distinction in the shared participant dispatch, invalidates before a
real forward, and republishes exactly one row only when host decode observation
is explicitly enabled. Failed or partially successful fan-out cannot expose
old output. The device-only path does not gather. The focused
`V2_Integration_ResidentMainConditionLogitsFreshness` preflight registration
exercises CUDA/ROCm interfaces, TP2/TP4, sharded/replicated vocabulary and
ordinary/MTP condition entrypoints with device-free participant doubles.
Native verification remains separate from this interface regression. This
defect does not explain the preserved single-ROCm-plus-CPU high-KL history.

The existing GPU-only PP ordinary forced-condition API also rejects that
transaction explicitly. Such a topology cannot be certified by the new
full-logit observer until its production transaction exists; the test must not
substitute a host-driven forward or report a skipped configuration as passed.

#### Same-input routing attribution

The preserved ROCm+CPU default history now has an executed six-case router
control: decode offset zero at layer 10, offset one at layer 3, and both
offsets at layers 43 and 47. These include the first changed expert sets and
the late divergence. The control reads the actual saved FFN input,
probabilities and expert IDs, plus only four original FP32 router matrices
(20 MiB total). It evaluates FP32 and FP16-rounded operands on both the
reference and native inputs, always accumulating in FP32. No model policy or
acceptance threshold changes.

FP32 arithmetic on the native input reproduces the native **ordered** top ten
experts in all six cases. FP16 arithmetic on the reference input preserves the
reference order in all six. FP16 operands on the native input reproduce the
native probabilities with relative L2 error of 2.80e-7 to 4.67e-7. That last
residual includes the CPU versus ROCm reduction and exponential order; this
experiment does not claim byte-identical native arithmetic.

| First changed decision | Reference pair margin | Incoming-input contribution | FP16-rounding contribution | Native pair margin |
|---|---:|---:|---:|---:|
| Decode offset 0, layer 10, experts 206−325 | +0.000563740 | −0.000721282 | +0.000001247 | −0.000156304 |
| Decode offset 1, layer 3, experts 88−235 | +0.000199558 | −0.000355956 | −0.000000449 | −0.000156846 |

Incoming drift already crosses both boundaries. The local FP16 router is not
the source of those selection changes. This conclusion is limited to the
observed inputs and selected stages; it does not certify the whole model or
prove the accumulated drift acceptable. One-thread CPU replay also differs
slightly from the original reference reduction, so its measured residual is
retained separately rather than treated as exact.

The next upstream boundary is the first hyperconnection read. Its normalized
input is equal in the current CSV, while read-output relative L2 is about
0.0164 and 0.0121 at the two decode positions. Earlier CPU controls show that
this operation is sensitive to Q8 activation rounding, but their old operands
and policies cannot establish current ROCm correctness. Reuse the prepared
HC component and original source weights for a controlled comparison; do not
substitute the older Python quantizer for the current native one. The full
same-input evidence is retained locally in
`parity-results/qwen-flash-next/accuracy49-router-same-input/`.

#### Executed HC decomposition and broader output panel

The same-input HC control is now executed on the prepared ROCm down and up
projections. Both use untouched original Q8_0 tensor slices. The normalized
inputs come from the saved build47 FP16-router policy and are byte-identical
to the reference. The CPU reconstruction uses the canonical reference's SiLU,
sigmoid and branch mean, with the actual native linear outputs substituted.
It reproduces the saved native read to relative L2 4.62e-8 / 5.16e-8.
This is a historical-input component attribution, not a complete current-build
capture of the intermediate nonlinear GPU stages.

| Decode offset | SingleQ8 up versus same-input FP32 | TwoTerm up versus same-input FP32 | Down-only propagated read error | Single chain read error | Up-only TwoTerm read error |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00849237 | 0.0000278085 | 0.000425906 | 0.0163747 | 0.000397707 |
| 1 | 0.00623574 | 0.0000333298 | 0.000279518 | 0.0120868 | 0.000260856 |

Each value is relative L2, not final-logit KL. The up-only control retains the
observed SingleQ8 down output and improves reconstructed read accuracy about
41x / 46x. The exact vector decomposition closes to zero, with reference
reduction and nonlinear reconstruction residuals reported separately. It
identifies the up projection as the dominant local contributor; it does not
prove a defective Q8 kernel or monotonic improvement through subsequent MoE
selection. Full-model paired controls change only the existing HC-up selector.
No default or tolerance has been promoted. Evidence is retained in
`accuracy49-hc-decomposition/`, with actual native outputs and clean driver
windows in `accuracy49-hc-down/` and `accuracy49-hc-up/`.

The new observer completes all 48 rows plus an independent byte-identical
reset replay on both static-eight-expert GPU+CPU configurations with adaptive
ngram storage and zero cache slots. Both runs authenticate actual prepared
owners, captured execution, source metadata, and clean retirement.

| Backend | Phase | Mean final-logit KL | Top-1 agreement | Mean top-5 intersection/5 |
|---|---|---:|---:|---:|
| ROCm+CPU | Prefill (16 rows) | 0.0263587 | 87.5% | 88.75% |
| ROCm+CPU | Decode (32 rows) | 0.0225455 | 100% | 90.625% |
| CUDA+CPU | Prefill (16 rows) | 0.0564827 | 81.25% | 88.75% |
| CUDA+CPU | Decode (32 rows) | 0.0218477 | 100% | 91.875% |

These are uncalibrated diagnostic measurements, not a new pass threshold.
The original-nine anchor differs from the older canonical run despite
reference-logit differences below 5e-12 KL. Both use the same adaptive-C0,
static-eight-expert placement: the old fixture's physical prefill bucket is
9, whereas the production observer replays bucket 64. The logits already
differ after prefill, before decode publication. Detailed capture must reproduce
the production bucket before attributing that cross-run difference to an
arithmetic change. The CUDA accented-Unicode prefill row reaches
0.498031 KL and remains visible alongside the mean. Evidence is in
`accuracy49-native-rocm/` and `accuracy49-native-cuda/`.

At the user's invitation, extend every one of the same 16 prompts to six
decode steps (112 rows total); retain the completed two-step corpus unchanged.
The reference writer now derives D+1 teacher positions, files and extents from
the authored depth, rejects a mismatched CLI before admission, and enforces
both existing per-history and total-row bounds. Its 17 device-free regressions
pass. The six-step CPU reference completed in `reference49-panel-six/`; all
48 rows overlapping the preserved two-step reference are byte-identical, with
the same original prompt and teacher prefixes.

The complete Unit suite passed 752/752 before that small Python depth extension.
It initially found two observer source-policy violations: redundant broad
telemetry gates and an undeclared passive snapshot boundary. Exact domain
gates replace the broad checks; the source sanitizer confines diagnostic
snapshot access to the two observation functions, keeping the inference loop
and production core forbidden. Full production preflight still needs refresh
after the coherent diagnostic changes; the earlier build48 receipt is stale.

#### Paired HC-up control and validation repairs

The whole-model control changes only `hc_up` from `single_q8` to
`two_term_q8`. The original model, reference report, runtime configuration and
every teacher position match. The observer executable was rebuilt to permit
explicit existing arithmetic variants; both executable identities are retained.
This is an accuracy observation, not a same-executable performance comparison.

| Backend | Phase | Default mean KL | HC-up TwoTerm mean KL | Default / control top-1 | Default / control top-5 | Rows with lower / higher KL |
|---|---|---:|---:|---:|---:|---:|
| ROCm+CPU | Prefill | 0.02635873 | 0.02462766 | 87.5% / 87.5% | 88.75% / 95% | 13 / 3 |
| ROCm+CPU | Decode | 0.02254546 | 0.02030540 | 100% / 100% | 90.625% / 91.25% | 23 / 9 |
| CUDA+CPU | Prefill | 0.05648269 | 0.02341030 | 81.25% / 81.25% | 88.75% / 93.75% | 13 / 3 |
| CUDA+CPU | Decode | 0.02184769 | 0.01536689 | 100% / 96.875% | 91.875% / 93.75% | 27 / 5 |

The local HC improvement is not monotonic through the full MoE model. The CUDA
Arabic first decode loses top-1 while KL improves from 0.00598473 to 0.00238389;
the reference's leading two probabilities differ by only 0.00130755. The
ROCm accented-Unicode prefill instead worsens from 0.129376 to 0.268846 KL.
Every row and probability/margin observation remains in
`accuracy49-paired-hc-summary/paired-summary.json`. No default, numerical limit
or pass verdict has changed.

The new synthetic original-Q8 HC-up/read composition regression passes on CPU,
CUDA and ROCm. It preserves a nonzero cancellation-sensitive read signal and
tests captured live counts 15→9→1→0→15 on both GPU backends. Existing
all-format residual-correction tests also pass, with a clean driver window.
All six focused registrations are in ProductionTestPreflight. The complete
Unit suite subsequently passed 752/752 in 130.6 seconds.

Two additional diagnostic execution defects are being fixed before collecting
CPU/hybrid results. CPU-only observation skipped the canonical startup
bootstrap and requested all 56 host cores for an explicit 28-core NUMA
participant. The fix reuses `MPIBootstrapPhase` with exact original wrapper
arguments and canonical automatic thread selection; it preserves the affinity
verifier. The hybrid run reached a valid empty QSA tail after accepting eight
tokens, but strict validation incorrectly required a nonzero partial bank.
`CommitPartial` writes only `(prefix + accepted) % 4` raw keys. Its typed
zero-state contract now permits that legitimate empty state while retaining
finite/storage checks. A focused real-stage regression covers all four
remainders and rejects NaN/Inf. Both repairs built and passed their focused
regressions. The hybrid retry completed all histories but exposed a separate
observer defect: it required profiling-only stage totals even with profiling
disabled. The failed original runs remain retained. The subsequent full Unit
run passed 752/753 registrations; its sole failure was the Python test's
importlib sibling-module scope, now fixed with CLI and isolated-import controls.

Explicit full-stage history replay is also installed for focused validation.
It derives the serving bucket ladder from the original public configuration,
requires actual native replay geometry, and preserves the existing all-layer
checkpoint inventory and numerical limits. The first M64/live9 replay can
reuse the preserved original-nine full-stage reference, without another model
reference generation. Ordinary canonical cells retain their existing behavior
until the broader regime is qualified.

#### Next canonical accuracy and performance checkpoint

The user requires the following order:

1. Complete the fixed-history numerical diagnosis and obtain green accuracy.
   Retain six decode observations on all 16 prompts, phase/step mean
   full-vocabulary KL, literal top-5 intersection, top-1 agreement and
   probability/margin diagnostics. Preserve selective replay through every
   layer and stage, with the actual production physical bucket.
2. Install the qualified workflow as the canonical QFN PyTorch parity regime,
   including its CI registrations and focused model-free contract regressions.
   A completed uncalibrated collection is not an accuracy pass; define and
   record the accepted baseline and limits before claiming qualification.
3. Derive one topology inventory with paired accuracy and Release benchmark
   projections per cell. Both projections share original model/quantization,
   placement, arithmetic, ngram tier policy and public runtime configuration;
   phase lengths and intrusive diagnostic observations remain explicitly
   separate from timing. Do not maintain a second hand-copied topology list.
4. Use `benpeterson40/castagna-veloce` as the performance target. Inventory the
   pinned fork's actual supported topologies from source, and admit every one
   as a prerequisite to performance comparisons. Record implementation gaps,
   admission failures and unavailable hardware explicitly rather than counting
   them as coverage. Map each supported baseline topology to its corresponding
   Llaminar accuracy/admission/benchmark cell.
5. Begin matched, unprofiled Release measurements only after those prerequisites
   are green. Use the llama.cpp comparison workflow, then backend profiling to
   attribute the gaps. Each retained optimization must keep the corresponding
   accuracy cell green. Do not time the intrusive numerical observer or reuse
   the stale build48 preflight receipt.

The source-audited topology families, paired workload design and functional/
performance obligations are in
[the qualification plan](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md). Neither that
plan nor a completed uncalibrated collection grants numerical qualification.

#### Six-step GPU panel

Both default GPU+CPU cells completed all 112 rows with source, production-path,
publication, teardown, driver and byte-identical reset checks. All 48 rows
overlapping each backend's earlier two-step native observation are byte-identical,
including histories executed after the longer intervening continuations.
Evidence: `accuracy49-six-gpu/`, with the original per-row metrics and
`d2-prefix-comparison.json` in each cell.

| Backend | Phase | Mean final-logit KL | Top-1 | Top-5 |
|---|---|---:|---:|---:|
| ROCm+CPU | Prefill, 16 rows | 0.02635873 | 87.5% | 88.75% |
| ROCm+CPU | Decode, 96 rows | 0.03762123 | 95.8333% | 88.3333% |
| CUDA+CPU | Prefill, 16 rows | 0.05648269 | 81.25% | 88.75% |
| CUDA+CPU | Decode, 96 rows | 0.03802909 | 97.9167% | 89.1667% |

| Decode observation | ROCm mean KL | CUDA mean KL |
|---|---:|---:|
| First | 0.02246961 | 0.01992013 |
| Second | 0.02262132 | 0.02377526 |
| Third | 0.07807667 | 0.04614830 |
| Fourth | 0.04691041 | 0.07003763 |
| Fifth | 0.03243962 | 0.01966156 |
| Sixth | 0.02320974 | 0.04863167 |

These remain `UNCALIBRATED`; no numerical limits or defaults were changed.
The depth extension reveals higher intermediate step means than the two-step
panel. CPU and mixed CPU-tail observations are still pending the observer's
profiling-independent completion proof.

#### Completed CPU and mixed six-step observations

The fresh CPU and mixed CPU-tail retry completed all 112 rows per cell, with
source/publication/path authentication, byte-identical reset replay, complete
retirement and clean driver windows. Evidence:
`accuracy49-six-host-hybrid-r2/`; combined phase and per-offset results:
`accuracy49-six-summary/REPORT.md` and `summary.json`.

| Configuration | Phase | Mean final-logit KL | Top-1 | Top-5 |
|---|---|---:|---:|---:|
| CPU | Prefill, 16 rows | 0.04230495 | 93.75% | 88.75% |
| CPU | Decode, 96 rows | 0.03596756 | 94.7917% | 87.5% |
| CUDA+ROCm+CPU | Prefill, 16 rows | 0.04593095 | 93.75% | 90% |
| CUDA+ROCm+CPU | Decode, 96 rows | 0.03443501 | 97.9167% | 90.8333% |

The observer now scopes real runner calls with the existing prefill/decode
phase metadata and proves successful CPU runtime execution independently of
profiling. The previous failed CPU record remains unchanged: its numerical
work completed, but every completion had incorrectly been tagged `combined`.
The new strict proof is not satisfied by relabeling those old artifacts.

Stage validation now follows each output's authored live extent, with the
original native count acquired on the exact producing stream. Legitimately
empty sparse pools do not fail a nonzero heuristic; observed rows still reject
zeros, NaN and Inf. The same repair preserves tensor subview offsets and
rejects malformed owners/counts before treating an output as empty. It adds
no kernel launch or count-only readback. All 16 focused registrations passed
in 14.4 seconds with clean driver evidence in `accuracy49-validation-focused/`.
The complete build covers CUDA 80/86/89/90 and HIP gfx906. Full Unit and
production preflight still require the coherent final prerequisite transaction.

All four panel collections remain uncalibrated. The third/fourth original-nine
decode observations expose substantial individual-row errors even though the
phase means are close. Extend the existing full-stage reference helper with
the exact frozen teachers and replay that history before numerical approval;
do not substitute a newly sampled continuation or hide those rows in a mean.

#### Frozen six-step full-stage reference and complete TP4 source declaration

The original-nine full-stage CPU reference completed in 117.8 seconds, retaining
7,861 snapshot files across prefill and all six decode observations. All seven
final-vocabulary rows are byte-identical to the existing frozen panel reference:
6,952,960 compared bytes with no difference. Evidence lives under
`reference49-full-stage/`, including `frozen-row-equivalence.json`. This extends
stage visibility without changing the numerical oracle or continuation.

The optional frozen-history adapter passed all 24 Python tests, including
non-greedy teacher inputs, pending-token accounting, source replacement and
close-before-metadata failures. Four focused CTest registrations also passed.
The coherent Integration rebuild completed in 116.8 seconds; this is still
focused validation, not a refreshed complete prerequisite gate.

The new declared ROCm TP4 cell requires all 512 routed experts in its sole GPU
tier and keeps PLE/ngrams in RAM. Its five device-free tests passed. Independent
review found the declared inactive predictor missing from discovery's source
manifest. The correction uses existing ROCm-scoped model registration, with an
actual generated-discovery/inventory regression; CPU-only and CUDA-only source
closures remain four main shards. Native TP4 admission remains unproved.

Before native detailed replay, align CPU execution with the observed run:
the corpus used 56 actual router workers, while ordinary canonical campaign
launches bind one socket and 28 workers. Matching the M64 bucket alone does not
establish the same execution. Numerical limits and arithmetic defaults remain
unchanged.

The complete current Unit inventory finished in 117.5 seconds: 751/753 passed,
with no skips. Both failures were source-policy expectations left behind by
the semantic validator refactor. The exact one-event diagnostic allowances
now name `validateObserved`, and the StageVerifier guard follows the combined
owner check before dtype dispatch. Four focused registrations then passed in
10.3 seconds, including both failed suites and the synchronization sanitizer's
negative controls. The final complete prerequisite transaction remains pending.

The explicit replay now installs the original positive worker request before
admission and verifies actual OpenMP settings, current unbound physical-core
affinity and CPU router worker counters. Its launch projection preserves the
registered absolute MPI executable, transport arguments and environment exports;
stale `-x` resource/filter values cannot override the requested policy. Ordinary
campaign launches retain their original resources. Eight Python tests and all
five generated campaign launch projections passed in private validation; the
installed source still needs its coherent build and complete prerequisite gate.
Two device-free C++ regressions are explicitly registered in preflight.

The coherent 51-action Integration rebuild then passed in 133.0 seconds, followed
by all seven focused registrations in 7.0 seconds. These include the two worker
checks, complete frozen-history replay contract, Python driver, frozen full-stage
reference and both TP4 declaration/source-closure regressions. The installed
Python driver also passed its eight checks and all five real generated launch
projections. Evidence is in `accuracy49-workers-build/` and
`accuracy49-workers-focused/`. The current source is now held fixed for the
complete `prerequisites-main-49` transaction and subsequent detailed replays.

#### Complete prerequisites and the next admission repairs

`prerequisites-main-49` completed all 2,397 registrations: 2,368 passed,
29 required unavailable hardware, and none failed. Unit passed 753/753.
The canonical transaction and its independent driver window completed with
unchanged source metadata and no new driver findings. This receipt qualifies
that source state; subsequent source changes require a fresh transaction.

The original-nine, six-step ROCm replay then stopped in setup before model
admission or numerical output. The public canonical configuration left the
prefill cap unset, while its frozen observer recorded the equivalent default,
512. All other retained intent fields matched. The repair compares the
effective cap and bucket ladder; genuine changes, such as 1024, still reject.
It preserves the original frozen descriptors, reference, M64/live-nine workload
and numerical limits. The corresponding CUDA replay was not launched against
the same known setup defect.

The standalone all-GPU ROCm TP4 snapshot-admission case separately reached
fixed-BOM construction and rejected the supplied predictor's down projection.
Planning retains 48 main layers plus the predictor's layer 48, but the geometry
validator had bounded the complete source inventory by the main-layer count.
It now uses the canonical total-layer count. Focused metadata tests distinguish
inline/shared predictor planning from the main-only graph directory and retain
strict missing, duplicate and malformed geometry rejection.

Singleton homogeneous GPU PP now has a complete retained ForwardOnly parent
for forced-teacher observations. Its device controller remains authoritative;
terminal publication contains fixed protocol metadata only. Dedicated CUDA
and ROCm regressions exercise reset/pending teachers and a synthetic API error
after a real successful terminal-copy enqueue, checking that accepted copies
retire before their destinations. These changes do not implement nested TP
groups or repeated-physical virtual PP. This is model-free execution coverage;
no QFN numerical or performance pass follows from it.

The native regressions exposed two ownership defects. ForwardOnly initially
opened a logical-row writer before joining the preceding observations, then
failed to close follower writes because only the tail owns a complete logical
outcome. The repaired path joins observations before acquiring writers. The
tail publishes its complete mailbox; followers publish only their actual
physical-write event through the existing access epoch and TransferEngine.

Injected terminal-copy failures then exposed teardown ordering: the rank's
child DGOs were destroyed before the composed parent owner waited on their
events. Rank destruction now retires those parents while the children are
alive, preserving transport and edge owners until the child graphs retire.
Unused by-value rank moves are forbidden; moving its unique owner remains
supported. Existing event-completion assertions and the exact synchronization
budget remain in place.

The coherent Integration rebuild passed in 235.4 seconds. All 15 focused
registrations then passed in 29.9 seconds, without skips or driver findings:
CUDA PP2, ROCm PP2/PP4, reversed order, repeated teacher/reset execution,
accepted-copy error retirement, physical BOM, frozen bucket intent and
canonical projection source layers. Evidence is in `scoped-fixes50-build-r6/`
and `scoped-fixes50-focused-r4/`. The complete synchronization sanitizer and
its 17 unit checks also passed. Source is held for the fresh
`prerequisites-main-50` transaction before real-model replay.

#### Prefix journal repair after complete failure collection

`prerequisites-main-50` passed all 753 Unit registrations, then reported one
host failure: `PrefixArchiveBackgroundPublication` recovered two committed
entries instead of 53. The failed canonical receipt is preserved. The
supplemental `remaining-preflight50` collection completed every remaining
registration with the same source: CUDA 213 passes; ROCm 271 passes and two
unavailable-eight-device skips; exclusive 270 passes and 27 unavailable-hardware
skips. Combined coverage is 2,407 registrations, 2,377 passes, one failure and
29 skips. Both native driver windows were clean. This collection diagnoses the
failure; it does not turn the failed canonical receipt into a pass.

The original concurrent-writer case passed 20 unchanged repeats, so its exact
lost-record interleaving remains unproven. Review identified two concrete
journal defects. First, cached offsets retained only raw device/inode numbers.
An idle backend could miss peer compactions and mistake a recycled inode for
its cached generation. The existing descriptor RAII now retains exactly one
metadata inode per cached index until refresh or shutdown, with no payload
retention, hashing or second index. Second, failed native reads of stat-bounded
record framing were treated as an incomplete append and authorized truncation.
Those reads now fail the operation without truncating committed bytes. Existing
incomplete-tail recovery and archive format remain unchanged.

The new regressions were built and executed against unchanged production first.
Both focused registrations failed for their intended invariants. The inode
case observed real inode-number reuse across two peer compactions without
externally pinning the old journal. The injected read-error cases truncated a
5,085-byte committed journal to 80 bytes during initialization and to 1,750
bytes during refresh; the latter retained one of three keys. These are
negative-control observations, not evidence of a hardware I/O error in the
original gate. The focused aliases are `PrefixArchiveCachedIndexInode` and
`PrefixArchiveJournalReadFailure`; the existing payload-hydration read-failure
alias remains separate.

After installing both fixes, the coherent Integration rebuild passed in
188.0 seconds, retaining CUDA 80/86/89/90 and HIP gfx906. All eight focused
registrations passed, followed by 20 successful repetitions of each new alias
and the unchanged original background-publication alias: 60 completed repeated
executions, no failures or skips. Evidence is in `prefix51-negative-r2/`,
`prefix51-build/` and `prefix51-focused/`. The earlier duplicate-registration
configure failure is retained separately in `prefix51-negative/`.

Source is held for the fresh complete `prerequisites-main-51` transaction.
The original-nine six-decode full-stage ROCm/CUDA replays and the actual
all-512-expert ROCm TP4 snapshot admission remain pending that gate. Arithmetic,
reference inputs, numerical limits and performance acceptance are unchanged.

#### 2026-10-21: green preflight, focused admission defects and external logits

`prerequisites-main-51` completed green on unchanged source: 753 Unit, 873 host,
213 CUDA, 271 ROCm and 270 exclusive passes, plus 29 explicit unavailable-hardware
skips. That is 2,409 registrations, 2,380 passes and zero failures. The complete
driver window was clean; elapsed time was 4,075.4 seconds. This receipt precedes
the focused repairs below and cannot certify their rebuilt source.

Both original-nine six-step full-stage replays then failed before producing
logits. Their serving M512 graph requested a 12,637,047,456-byte snapshot arena
against a declaration derived from nine reference rows. The memory authority
correctly rejected the undersized declaration. These are setup failures, not
numerical red results. The correction prices the existing wide snapshot owner
from the effective production bucket inventory and retained MTP capacity.
It preserves the selected checkpoints, arena-owner counts and native QSA state
pricing. It also considers every authored participant, including CPU-first
pipelines with later GPU stages. This diagnostic envelope is conservative when
production fits a smaller resident bucket; no separate allocation ledger or
anonymous reserve is introduced.

The independent full-ROCm TP4 snapshot-admission case reached model
materialization, then rejected QFN's replicated terminal weights for lacking
`output_norm.weight`. QFN declares `output_hc_norm.weight`,
`output_hc_down.weight` and `output_hc_up.weight` instead. Planning already uses
that registered schema; the auxiliary installer still hardcoded the conventional
norm. The focused repair makes installation validate the same model-owned
requirements, retaining independent vocabulary-head and PP ownership checks.
Both detailed replays and this admission attempt retired with clean drivers and
unchanged inputs. Evidence remains in `full-stage51-original-nine-six-{rocm,cuda}/`
and `full-rocm-tp4-snapshot-admission51/`.

The pinned Castagna fork completed an **untimed** original-nine plus six-step
forced-history diagnostic in `castagna-logits51-nopeer-tp4-r2/`. The private build
adds only passive static-placement accessors; inference arithmetic is unchanged.
Its completion parser rejects the server-only `--spec-type` option, so the
corrected invocation uses its existing NONE default, which the compiled probe
asserts before loading. The rejected first invocation is preserved separately.

All 144 original expert banks match the GGUF directory's names, types and
dimensions. Their physical children are on four distinct MI50s, with no placement
violations. For this fork topology, every child retains all 512 expert IDs;
gate/up intermediate rows and down input columns split as **128/128/128/256**
out of 640. This is not Llaminar's gate/up expert-ID ownership with down output
columns. The actual ownership evidence, not CLI labels, defines future matched
comparisons. Only directory metadata and observed buffer extents were joined;
model payloads were not hashed.

For this single diagnostic history, full-vocabulary final-logit metrics are:

| Engine/configuration | Prefill KL | Mean decode KL, six rows | Decode top-1 | Decode top-5 overlap |
|---|---:|---:|---:|---:|
| Fork, four-MI50 staged TP | 0.020118 | 0.157903 | 66.67% | 73.33% |
| Llaminar ROCm + CPU, frozen panel49 | 0.011093 | 0.362366 | 66.67% | 66.67% |
| Llaminar CUDA + CPU, frozen panel49 | 0.018850 | 0.409473 | 83.33% | 63.33% |

The source GGUF and forced token history match, but topology, native prefill
geometry and engine arithmetic differ. These rows establish numerical
perspective, not a paired performance result or calibrated acceptance policy.
The fork also varies substantially by decode position (KL 0.039–0.284), while
Llaminar's larger spikes still require the full-stage diagnosis. No threshold,
precision default, canonical accuracy approval or performance claim changes.

#### 2026-10-21: complete full-stage replays and remaining terminal ownership seams

The coherent `admission52-build` passed with the configured CUDA 80/86/89/90
and HIP gfx906 targets. `admission52-focused` then passed all 753 Unit
registrations and six focused regressions, including both serving-snapshot
envelopes and replicated auxiliary-terminal installation. The new installer
regression was first run against the old implementation: it rejected a valid
QFN head and accepted missing HC operands when an unrelated ordinary norm was
present. Both failures are retained in `admission52-negative`.

The user-authorized focused admission iteration deliberately does not claim
that these checks refresh the full preflight receipt. Actual TP4 admission
now passes installation and reaches graph construction, where conventional
final-norm assumptions remain in mirrored-head validation and final-projection
resolution. The terminal HC builder also reads primary bindings rather than
the selected replica. The next repair covers this complete selection chain,
with real graph-construction tests distinguishing primary and replica tensor
and prepared-weight identities.

`full-stage52-focused-original-nine-six-{rocm,cuda}` both completed all 48
layers, prefill and six forced decode steps. All seven full-vocabulary FP32
logit rows on **each backend are byte-identical** to the corresponding frozen
panel49 observations above. Thus the snapshot-capacity correction restores
diagnostic execution without changing these numerical results. All six
canonical numerical CSVs and raw stage exports are retained. The runs remain
red at their unchanged numerical limits; the full-history per-step results
must not be replaced by a selected favorable decode position.

Both runs additionally expose a diagnostic artifact-ordering defect:
`applyModelOverrides` writes `history_replay_binding.json` before base setup
clears the per-cell directory. Later physical-row/worker evidence survives,
but the original binding does not. The reports preserve this artifact failure.
The passive observations explicitly record the missing binding and do not
reconstruct it or certify either run. The lifecycle repair will publish the
validated binding after rank-zero artifact initialization and before model
admission, with symmetric failure publication to peers.

#### 2026-10-21: terminal selection repaired; four-MI50 prefill reaches reset

The terminal-selection repair now covers installation, mirrored-head
validation, projection resolution and HC construction. The vocabulary head
and its three HC operands select one complete primary or replica binding set;
QFN no longer evaluates an unused ordinary-norm lookup. Six real graph
construction regressions distinguish the selected tensor and prepared-weight
owners. Against the old production implementation, four of those six tests
failed; the corrected implementation passes all six. The frozen-history
binding now publishes after artifact-directory reset, before admission, with
owner-only I/O and readiness publication to peers on failure. Two device-free
lifecycle tests exercise the real reset/publisher ordering and failure paths.
Both defects have explicit production-preflight registrations.

`admission53-build` completed the coherent 822-action Integration rebuild in
214.0 seconds, retaining CUDA 80/86/89/90 and HIP gfx906. `admission53-focused`
passed all 753 Unit registrations and nine focused registrations, with no
skips and unchanged source. Full production preflight still needs refreshing;
the older green receipt does not certify these changes.

`full-rocm-tp4-snapshot-admission53-focused` passed in 84.3 seconds with clean
drivers and unchanged source and model inputs. All four MI50 participants
materialized the prefill family and two retained main decode graphs without
launching executables or mutating request state during setup. Its
`numerical_inference_executed=0` evidence deliberately limits this result to
admission and graph materialization.

Both `full-stage53-focused-original-nine-six-{rocm,cuda}` replays then completed
all 48 layers, prefill and six forced decode steps. Their actual history
bindings survive and both artifact contracts pass. All seven final-logit rows
on each backend remain byte-identical to frozen panel49. The existing numerical
failures are unchanged; the artifact repair does not improve or weaken
numerical acceptance. Driver retirement and source identity were clean.

The actual TP4 numerical sibling, owned by the canonical ROCm campaign,
completed prefill with KL **0.00728955**, top-1 agreement **100%** and top-5
overlap **100%**. It then aborted during the first request reset with an
immutable-lookup kernel reporting an invalid GPU resource handle. Decode,
prefix restoration and complete path evidence are absent, so this is a failed
cell, not numerical qualification. Evidence is preserved in
`full-rocm-tp4-production53-focused-r2`. The first local wrapper attempt had
incorrectly treated the aggregate-owned numerical case as a standalone CTest
registration; discovery rejected it before any native launch. The corrected
attempt uses the existing campaign discovery and exact-cell runner.

Source inspection points to caller device selection at reset: rank reset
walks participants on one host thread, while the immutable terminal kernel
uses the participant stream directly. Earlier event APIs preserve the caller's
ambient device rather than establishing the reset owner's device. A focused
CUDA/ROCm regression will deliberately select another GPU before the real
DGO reset, followed by an explicit checked activation of the reset owner.
This remains a diagnosis pending its negative control and native verification.

The separate numerical investigation found exact embeddings and near-exact
layer-zero normalization, followed by a measurable HC read difference. The
first changed expert sets at decode2/layer2 and decode3/layer3 occur after
continuous FFN-input drift on both backends. Selected contribution folds and
routed-plus-shared sums agree with their native components; no new equation,
source or folding defect has been demonstrated. Current same-input HC and
physical-FP16 router controls are being prepared to separate propagated
inputs, operand precision and local implementation residuals. No arithmetic
policy, acceptance threshold or performance claim changes.

The two-device negative control in `reset54-negative` reproduced the exact
ROCm invalid-resource-handle failure in 0.49 seconds using real captured
lookup work and the real DGO reset. The CUDA counterpart already passed;
both use two actual GPU owners and retain the existing single-device tests.
The driver window was clean. The production repair now checks and activates
the participant's declared backend/device after validating the reset request
and before resolving its stream or entering any reset phase. It adds no
allocation, synchronization, device readback or alternative execution path.
Both new native aliases are explicitly registered in production preflight.
The coherent rebuild and positive verification are next.

#### 2026-10-21: reset repaired; complete TP4 numerics and same-input attribution

`admission54-build` completed 822 actions in 219.4 seconds with CUDA
80/86/89/90 and HIP gfx906 unchanged. `admission54-focused` passed all 753 Unit
registrations and all 15 focused registrations, with no skips. Both new
foreign-current-device reset tests pass, alongside the original one-device
reset and resident-prefill cases. The driver window was clean and source
metadata stayed unchanged. A fresh complete production preflight is next;
these focused checks do not refresh the older full receipt.

`full-rocm-tp4-production54-focused` now completes the real four-MI50
all-512-expert cell, including decode, full/partial prefix restoration and all
canonical artifacts. The native cell took 90.1 seconds, with clean drivers and
unchanged source. Its artifact contract passes; its numerical result remains
red at the existing pointwise decode limit:

| Phase | Final-logit KL | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|
| Prefill | 0.00728955 | 100% | 100% |
| Decode 0 | 0.10125006 | 100% | 80% |
| Decode 1 | 0.0409678 | 100% | 100% |

Both prefix probes report equivalent restored state. The full hit retains all
3,973,120 terminal-logit bytes across the four participants. Its checkpoint
cosine, 0.973338, is the mean of the two **post-restore decode** cosines,
not a second prefill-head comparison. Its numerical failure inherits the
decode failure above. The partial-hit checkpoint intentionally compares its
one-token suffix to decode 0. No new prefix or terminal-replica defect is
indicated by these observations.

The physical FP16 router controls in `router54-component-r2` use the original
F32 router weights, production weight preparation and the actual native
captured scalar routing kernels. On both backends, replaying native inputs at
decode2/layer2 and decode3/layer3 reproduces every saved probability, selected
expert ID and selected weight byte-for-byte. Reference inputs select the
reference ordered expert set. FP32, half-operand FP64 and native arithmetic
all retain the changed native expert set when given the native inputs.
At these cutoffs, the propagated input effect is 0.0199–0.0260 in relative
probability-vector L2, versus 0.000062–0.000104 from rounding operands to half.
Native arithmetic agrees with the exact half-operand oracle to roughly
2e-7 relative L2. This attributes the selected expert switches to incoming
activation drift, not FP16 router rounding or a demonstrated router defect.

Use the separately normalized FP64 probability oracles in
`{rocm,cuda}-normalized-oracle/oracle.json`. The first private helper had
compared raw FP32 probability masses and could report a tiny negative KL.
Its corrected form normalizes both distributions independently and has six
passing guard tests. Canonical full-model final-logit KL was already normalized
in FP64 and is unaffected.

The companion `hc-cutoff54-component` diagnosis uses original Q8_0 HC weights,
native down/up projections and the canonical CPU nonlinear continuations.
The reconstructed current-policy chain matches saved FFN inputs to
4.6e-8–8.3e-8 relative L2. Across the same four backend/cutoff pairs, the
HC-up single-Q8 contribution accounts for 0.0146–0.0159 local relative L2,
while its existing two-term control reduces that local contribution to
0.000027–0.000039. These are isolated projection/continuation observations,
not a captured whole-HC or full-model certificate, and their minimal timing
samples are not performance results.

The earlier 16-history paired HC-up panel remains the broader evidence:
two-term improves aggregate KL but worsens selected histories and one CUDA
top-1 result. A local improvement therefore does not justify promotion or a
duplicate two-step panel run. Current arithmetic defaults, numerical limits
and the uncalibrated acceptance status remain unchanged. Long-context accuracy,
the complete topology matrix and matched Release performance still remain.

### October 21: complete preflight and current six-step controls

The coherent build54 prerequisite transaction completed with **2,387 passes,
29 hardware skips and zero failures** across 2,416 registrations. This includes
all 753 Unit tests and the available CUDA, ROCm and hybrid production-preflight
cases. Skips require unavailable device counts; they are not passes.
`prerequisites-main-54/prerequisites.json` and `preflight-driver54/result.json`
retain the source/build identity and clean driver retirement.

The original 16 histories and six teacher-forced decode steps were then
collected on all five current configurations with that unchanged binary:
112 metric rows and seven separate reset-proof rows per configuration.
Every native process returned zero, reset replay was byte-identical and
driver retirement passed. The four existing configurations reproduce their
build49 scores; the new TP4 panel uses actual serving buckets, not the earlier
exact-M9 three-row diagnostic. All results remain uncalibrated observations.

The fork's complete panel also finished on its explicit staged-copy TP4
profile. Castagna prefill/decode mean KL is **0.033583 / 0.027522**,
top-1 **87.5% / 94.79%**, and top-5 overlap **90% / 89.375%**.
It has a large individual error too: prose-cause decode4 KL 0.707655 and
TV 0.563256. Its native reset and complete expert-placement inventory pass.
This is numerical context, not a throughput comparison: the fork's
within-expert F slicing differs from Llaminar's current
GateUpOwnedDownColumns ownership. The original scoring-only Python failure
and its successful successor receipt are both preserved; inference was not
repeated to repair reporting.

One explicit existing candidate changes only HC-up from SingleQ8 to TwoTermQ8,
retaining the same native weights, FP16 router, alpha/beta policy, reference
rows, worker configuration and executable. The complete six-step comparison
is in `accuracy54-hc-up-comparison/comparison.json`:

| Configuration | Default prefill / decode mean KL | HC-up TwoTerm prefill / decode mean KL | Candidate top-1 prefill / decode | Candidate top-5 prefill / decode |
|---|---:|---:|---:|---:|
| ROCm TP4, all experts on GPUs, RAM ngrams | 0.034704 / 0.049832 | 0.033536 / 0.032070 | 93.75% / 97.92% | 87.5% / 92.29% |
| ROCm + CPU, static8, adaptive ngrams | 0.026359 / 0.037621 | 0.024628 / 0.032013 | 87.5% / 98.96% | 95% / 92.08% |
| CUDA + CPU, static8, adaptive ngrams | 0.056483 / 0.038029 | 0.023410 / 0.020687 | 81.25% / 97.92% | 93.75% / 91.88% |
| CPU, mmap ngrams | 0.042305 / 0.035968 | 0.016063 / 0.022886 | 93.75% / 100% | 92.5% / 90.21% |
| CUDA + ROCm + CPU, mmap ngrams | 0.045931 / 0.034435 | 0.032444 / 0.024587 | 93.75% / 97.92% | 90% / 90.21% |

TP4's largest decode KL falls from 1.672362 to 0.688284; its worst
decode-offset mean falls from 0.117292 to 0.049369. Improvement is not uniform:
TP4 prefill top-5 and hybrid decode top-5 decrease slightly, and some row KLs
increase. All row deltas remain available. No production default has changed
and intrusive observation elapsed times are not speed measurements.

A short-context pilot policy was frozen before any heldout reference/native
measurement. It retains 0.05 prefill and 0.10 decode as prompt-mean budgets,
also bounds every decode-offset mean, and adds explicit row/tail, TV and
agreement checks. This is an explicit proposed aggregation change; earlier
pointwise diagnostic reds remain reds. All five candidate panels satisfy its
absolute pilot checks. That does not grant baseline approval, a canonical
acceptance pass, long-context coverage or a performance certificate.

The reference-harness extension now accepts bounded authored subsets with exact
tokenizer IDs and independent decode depths, preserving the original FP32
same-GGUF equations and cached execution. Its eight new device-free regressions
cover full pre-forward validation and teacher/row identity; the complete script
passes 32 tests, and the focused subset is explicitly registered in preflight.
After installation, all 753 Unit registrations and four relevant preflight
registrations pass (757/757, no skips). Reconfiguration needed no native object
rebuild. The original tokenizer freezes the eight heldouts at 33–63 tokens and
the boundary at exactly 2047; their combined reference has 66 observed rows.
The next collection uses the eight previously frozen unseen prompts plus
N2047/D9, crossing the first sparse-pool discard at consumed position 2052.
Full-stage multichunk replay remains a separate uninstalled diagnostic patch.
The build54 full gate predates this reference/registration-only extension;
do not describe its old receipt as qualifying the added tests.

### 2026-10-21: heldout pilot completed; native projection preparation integration

The original CPU/FP32 reference completed eight unseen short histories with six
decode observations each and a separate N2047/D9 sparse-boundary diagnostic.
The same native observer binary then collected defaults and HC-up TwoTerm on
all five configurations, 66 metric rows per cell. All ten cells retired cleanly,
retained clean driver windows and passed seven-row byte-identical reset replay.
Original source, input and local binary/DSO metadata remained unchanged during
collection. There is no external runtime-closure or timing claim.

The canonical scorer applied the previously frozen absolute pilot independently
to every heldout-short context/phase group. All ten policy/configuration pairs
pass. This leaves HC-up passing both broad and heldout short panels on all five
configurations; defaults still fail two broad groups and cannot borrow the
heldout means to erase those failures.

| Configuration | Default mean KL prefill / decode | HC-up mean KL prefill / decode | HC-up top-1 prefill / decode | HC-up top-5 prefill / decode |
|---|---|---|---|---|
| ROCm + CPU, static eight GPU experts | 0.026785 / 0.009954 | 0.012405 / 0.005529 | 100% / 97.92% | 95% / 92.92% |
| CUDA + CPU, static eight GPU experts | 0.026492 / 0.009625 | 0.009881 / 0.005724 | 100% / 100% | 100% / 95% |
| CPU, mmap ngrams | 0.027738 / 0.008658 | 0.011671 / 0.004002 | 100% / 100% | 97.5% / 95.42% |
| CUDA + ROCm + CPU-tail PP | 0.027684 / 0.007479 | 0.010880 / 0.004979 | 100% / 100% | 100% / 94.17% |
| Four ROCm GPUs, existing TP ownership | 0.024534 / 0.007840 | 0.009691 / 0.003838 | 100% / 100% | 95% / 94.17% |

These means weight prompts equally; top-5 is literal set overlap divided by
five. The N2047/D9 repeated-token history crosses the first QSA pool discard
and retains 100% top-1 on both policies and every configuration. It remains a
diagnostic group with no borrowed short-context pass or long-context certificate.
No same-policy numerical baseline has been approved. Defaults are unchanged
pending measured performance.

Local evidence: `accuracy55-heldout-boundary/result.json` and its original native
summaries, plus separate `absolute-pilot55-heldout/comparison.json` and `REPORT.md`.
Two report-wrapper representation defects—JSON lists versus tuples and authored
symlink paths versus resolved identity paths—were corrected while preserving all
14,512 recorded source identities. Ten pure checks include actual receipt
validation and symlink/replacement negative controls. Inference was not repeated
for those reporting corrections.

The reviewed source components for exact native K/N expert slices, immutable
intermediate partitions, CPU materialization and participant-scoped GPU
preparation are now installed, with five explicit preflight registrations.
Their first complete Unit and focused regression pass is green: **762/762**
registrations passed with no skips (`expert-projection56-validation/corrected`).
The subsequent complete Unit plus fifteen focused pass is **769/769**, with
no skips and clean driver checks (`canonical-native56-validation/tests-r6`).
It includes actual CPU projection execution and CUDA/ROCm preparation,
source-byte, lifetime and execution checks. The changes preserve original
quantized blocks and all expert IDs.

The native checks distinguish immutable `SourceNative` storage from the
larger `MigrationReusable` layout. Their shared layout selector now drives
the preparation and geometry paths; all catalogued codebooks and both layouts
are compared with the actual pool planner. Existing movement defaults,
CPU allocation and arrival storage are unchanged. Lowering a complete frozen
model also preserves ordinary bindings and its original physical-memory owner
while expanding only marked static expert parents.

The public within-expert TP mode stays guarded until participant-local compute,
live-output reduction, admission and native correctness/economy are complete.
The paired-workload document contract now distinguishes an initial absolute
pilot from a calibrated revision with an explicit baseline locator. Its local
canonical consumer and exact case bindings are installed. Actual discovery
retains 109 production cases and attaches exactly five optional profiles;
all five pass original-document/source checks and both public parser intent
projections. This is model-free evidence, not a fresh numerical or timing run.

The actual-export check caught a harness defect that synthetic metadata had
missed: diagnostic benchmark arguments contain policy only, with the model and
context declared separately. The parser projection now supplies both declared
fields and has a focused preflight regression. The adapter's 27 tests pass;
the original failed check remains preserved beside the successful
`local-paired-integration/metadata-parser-r7` result.
The installed checkpoint passed complete Unit and production preflight:
2,403 passed, 29 unavailable-hardware skips, no failures, with clean driver
diagnostics and unchanged source metadata (`prerequisites-checkpoint-56`).
The first paired accuracy invocation then stopped before inference because
exact-cell discovery enumerated an unrelated stale model executable. No
numerical result was produced by that invocation.

Discovery now chooses binaries from registered exact-cell membership before
launching them, while auditing the complete registration of every chosen
binary and rejecting duplicate identities globally. The 28 device-free
inventory tests pass; the original implementation fails the focused unrelated
binary regression. All five real paired discovery and public-parser checks
also pass (`local-paired-integration/exact-discovery-parser-r8`).
`V2_Integration_ModelParityExactCellDiscovery` explicitly covers the defect
in production preflight.

Refresh the complete prerequisite receipt for this tooling change, then run
the five installed HC-up TwoTerm paired accuracy profiles and checkpoint the
green slice. Additional within-expert sharding and PP work remains in separate
uninstalled drafts and must not delay that checkpoint. HC-up and other
production defaults remain unchanged; matched Release timing is pending.
