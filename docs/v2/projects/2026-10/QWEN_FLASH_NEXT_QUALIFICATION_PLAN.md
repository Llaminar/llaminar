# QFN numerical qualification and matched performance plan

Status: short-context absolute pilot passed for HC-up TwoTerm; canonical
integration and matched performance pending, 2026-10-21.
This plan records obligations, not additional advertised runtime support.
Measurements remain in [the implementation project](QWEN_FLASH_NEXT_IMPLEMENTATION.md).

The requested order is numerical qualification, installation of the canonical
QFN parity regime, admission of the baseline's supported topology families,
then matched Release tuning. Missing hardware, capacity rejection, missing
implementation and numerical failure are separate outcomes. A local diagnostic
is not an image certificate.

The build54 complete prerequisite transaction and five current-source
112-row native controls are complete, with clean retirement and exact reset
replay. The complete Castagna TP4 panel is also available. One six-step
HC-up TwoTerm control improved mean prefill and decode KL on all five native
configurations. It remains optional pending canonical acceptance installation
and matched performance. The subsequent unseen-prompt panel also passes the
same frozen short-context limits on all five configurations.
The reference-only subset extension has its focused regression; the earlier
full prerequisite receipt does not certify that later addition.

The canonical scorer now supports an explicit initial absolute pilot without
inventing an approved baseline. It requires a complete role inventory for all
observed context/phase groups; diagnostic groups retain unavailable verdicts.
The unchanged broad-panel observations pass the frozen short budgets on all
five HC-up candidate configurations. Current defaults fail CUDA+CPU prefill
mean KL and the ROCm TP4 decode row/offset limits. These are short-panel
numerical results only; performance remains a prerequisite to choosing a default.
The scorer, optional canary CLI
and adapter's 33 pure tests pass. The refreshed complete Unit suite and four
focused scoring registrations pass: 757 tests, no failures or skips. This is
not a refresh of the complete production preflight.

The original PyTorch reference for all eight heldout histories and N2047/D9
is complete, with unchanged source/input metadata and clean process retirement.
The completed native collection compares defaults and HC-up on the same
original observer binary across five source-owned configurations: 660 metric
rows, with exact seven-row reset replay, clean driver windows and retired
processes in every cell. All ten heldout-short policy/configuration pairs pass
the frozen limits. The HC-up candidate therefore passes broad and heldout
short-context panels separately; current defaults still fail the two broad
groups described above. Initial pilot acceptance is not an approved same-policy
baseline or a complete-model certificate.

HC-up heldout mean KL ranges are 0.00969–0.01241 for prefill and
0.00384–0.00572 for decode. Top-1 agreement is 100% for prefill and
97.92–100% for decode; mean top-5 overlap is 95–100% and 92.92–95.42%.
The N2047/D9 boundary remains explicitly diagnostic. It crosses the first
sparse-pool discard with 100% top-1 agreement in every cell, but this single
repeated-token history does not qualify long-context accuracy.

Evidence is retained under local `parity-results/qwen-flash-next/`:
`accuracy55-heldout-boundary` preserves original uncalibrated observations;
`absolute-pilot55-heldout` contains separate canonical-scoring results.
The report wrapper's JSON tuple/list and authored-symlink/resolved-path
validation defects were corrected without changing any original source field
or rerunning inference. The whole-source freeze ended after successful scoring.

## Numerical acceptance and diagnosis

Use the exact original four GGUF shards and the existing CPU/FP32 PyTorch
loader with incremental cached execution. Preserve source identity, token IDs,
physical prefill buckets, arithmetic selectors and forced continuation.
Q8 GEMM activation quantization remains enabled. Current defaults use FP16
router operands with FP32 accumulation, FP16 MoE alpha/beta, and native wide
projections. HC-up TwoTerm remains a diagnostic control.

The authored panel has 16 fixed histories, each with a final prefill row and
six decode rows: 112 full-vocabulary observations per configuration. Preserve
the completed two-step corpus. Its 48 overlapping reference rows and teacher
prefixes are byte-identical to the extended reference. Report:

- Prompt-balanced mean KL(reference || candidate) separately for prefill,
  pooled decode and every decode offset. Compute full-tail metrics in FP64
  from original FP32 logits at temperature 1.
- Exact top-1 agreement and literal top-5 intersection divided by five.
  Retain every row, total variation, entropy, reference margins, maximum
  probability error and paired differences from the approved baseline.
- Structural failures independently of numerical scores: missing/nonfinite
  rows, stale source/configuration, incomplete execution/publication, changed
  teacher tokens and failed byte-identical reset replay.
- Explicit initial absolute limits, followed by baseline approval and calibrated
  paired bounds for subsequent revisions. `UNCALIBRATED` collection cannot
  become `GREEN` merely because the command returned zero. Initial acceptance
  covers only the policy's required groups; it cannot qualify diagnostic groups,
  a complete model or an image. Do not choose favorable prompts or fit thresholds
  to conceal regressions.

The broad panel is the acceptance workload. The production PyTorch framework
remains the detailed diagnostic: select a frozen failing history, generate its
full-stage reference, and run the exact registered cell with `--history-replay`.
Keep every layer/stage and all canonical CSVs. Replay binds actual serving
geometry: a nine-row diagnostic and a production 64-row bucket are different
executions. Match worker affinity too; an MPI resource banner does not prove
the CPU kernel thread count.

Short histories alone cannot certify a long-context benchmark. Freeze accuracy
workloads at the timed contexts and bucket boundaries, partial four-token QSA
pools, and continuation crossing 2052 tokens, where the completed pool first
exceeds 512 rows. MTP retains its independent serial/grouped byte-equivalence
obligations.

### Frozen short-context pilot and unseen prompts

The following engineering budgets were fixed before reference or native
measurement of the eight prespecified heldout prompts. They apply equally
to the unchanged defaults and the HC-up candidate. They are proposed numerical
quality limits, not statistical confidence intervals or a new approval step:

| Criterion | Prefill | Decode |
|---|---:|---:|
| Prompt-balanced mean KL | ≤ 0.05 | ≤ 0.10 |
| Every decode-offset mean KL | — | ≤ 0.10 |
| Maximum individual row KL | ≤ 1.0 | ≤ 1.0 |
| Prompt-balanced fraction of rows with KL > 0.10 | ≤ 12.5% | ≤ 12.5% |
| Mean total variation | ≤ 0.10 | ≤ 0.075 |
| Top-1 agreement | ≥ 75% | ≥ 87.5% |
| Mean top-5 overlap | ≥ 80% | ≥ 80% |

These bounds retain the original mean KL budgets, expose tails that a pooled
mean hides, and use agreement as a coarse check because a close argmax can
change at very small probability error. Retain entropy, margins and every raw
row alongside the verdict. The maximum-row limit is not a claim that an error
just below it is desirable. Both engines have demonstrated material individual
errors; the external engine provides context, not an automatic quality waiver.

Once a policy is accepted on the broad and heldout panels, future same-policy
revisions additionally permit at most +0.005 prompt-mean and +0.10 individual-row
KL relative to its immutable observations. Use the existing corpus calibration
and baseline binding; cross-policy diagnostics cannot impersonate that baseline.
Do not revise these pilot budgets or replace heldout prompts to make a failed
heldout result pass. Preserve any failure for diagnosis and an explicit next
policy decision.

The eight heldouts have six decode steps each. A separate context group adds
N2047/D9 and observes consumed positions 2047–2056, including the first QSA
pool discard at 2052. That boundary group is initially diagnostic; the short
budgets do not silently qualify it or a later timed long-context workload.
Actual tokenizer IDs, teacher histories, public runtime configurations, and
reset replay remain mandatory on both groups.

## One topology authority, paired workloads

Keep `ModelParityDefinition` → `ModelParityCase` as the sole inventory.
Add a typed qualification profile to the existing runtime exporter, joining
the approved accuracy corpus and benchmark workload to the same case.
Both projections share sources, topology, device order, layer ranges, expert
ownership, ngram placement, activation/KV formats and arithmetic policy.
The native public parser supplies the lossless effective configuration.

Reuse the existing accuracy and benchmark drivers. Extend explicit declared-cell
projections without manufacturing E2E eligibility or a second Python topology
registry. Do not benchmark both a cell and its identical snapshot-admission
companion. Unit tests prove pair completeness, configuration round-trip and
rejection of mismatched sources, placements, arithmetic or workloads.

## Baseline topology requirements

The baseline is Castagna Veloce at
[`3483d462715d61f21ed5877836be35a724446359`](https://github.com/benpeterson40/castagna-veloce/tree/3483d462715d61f21ed5877836be35a724446359).
Its [QFN guide](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/docs/models/qwen3.8-flash-next.md)
uses TP2×PP2 for decode and layer splitting for prefill. The
[CLI](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/common/arg.cpp)
also exposes CPU, single-device, tensor, weighted-split and offload modes.
Although its help advertises legacy `row`, the pinned CUDA/HIP registry does
not export `ggml_backend_split_buffer_type`; the model loader rejects that
mode before allocation. Legacy row is therefore outside the applicable
MI50/CUDA comparison at this pin. Inherited SYCL/Hexagon support is a separate
backend question. Source reachability alone does not qualify a QFN workload.

Each applicable instance below requires **cold admission, accuracy and a Release
benchmark**. Derive instances in the existing typed inventory; this table is a
family plan, not another executable registry.

| Family | Initial instances and boundaries | Remaining obligation |
|---|---|---|
| CPU | One NUMA participant; RAM/mmap; canonical physical-core workers | Six-step native panel and startup/affinity proof complete; numerical acceptance remains uncalibrated. |
| Single GPU with CPU | CUDA and ROCm, expert/layer offload, RAM/mmap and bounded adaptive ngrams | Match actual fork ownership; GPU8/CPU504 controls cannot represent full GPU residency. |
| Homogeneous TP | TP2/TP4 locally, TP8 when available; legal even/skewed/zero slices, reversed order | Prove actual segments, live communication and capacity; extend observer beyond exact ROCm TP4. |
| Contiguous layer PP | PP2/PP4 with equal/unequal ranges | Complete ordinary GPU pipeline resident control and observer proof. |
| TP groups in PP | Four MI50s as `PP(TP2, TP2)`; singleton/full-cohort boundaries; eight-card TP2×PP4 and TP4×PP2 later | Native TP-domain composition; TP4 cannot substitute for TP2×PP2. |
| Interleaved layer PP | Four MI50s with twelve logical stages, order 0,1,2,3 repeated three times; explicit one-stage-per-card control | Typed repeated physical ownership and concurrency-aware PMA admission. |
| Partial-layer/expert offload | CPU-prefix/GPU-suffix and whole-expert-tensor overrides with applicable splits | Match source ownership and scheduling; whole-layer offload is not expert-ID EP. |
| Mixed CUDA/ROCm/CPU | Existing 8/12/28-layer CPU-tail cell, then applicable composed domains | Required by Llaminar's goal; fork mixed-vendor execution remains unproved. |

The fork's gfx906 layer mapping defaults to multiple virtual stages per card:
four cards become twelve stages. Its invalid TP group selections can resolve
silently to another topology; reject such a comparison mismatch. Its pair-only
`HCP_INJ` optimization must not be applied to TP4. See the pinned
[device mapping](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/ggml-cuda.cu)
and [HC kernel](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/hc-persist.cu).

The workspace currently exposes four MI50s and two RTX 3090s. Availability does
not prove a full-model placement fits. Eight-card and remote cases remain
pending hardware. Optional fork RPC and inherited backend plugins do not
establish QFN runtime support; retain those prerequisites separately. No
expert-ID EP selector was found in the fork; Llaminar EP remains a separate
implementation obligation.

The row-mode correction follows the actual loader check in
`src/llama-model.cpp` and complete CUDA/HIP procedure registry in
`ggml/src/ggml-cuda/ggml-cuda.cu`, verified against the pinned Git objects.
The older help's claim that `--main-gpu` owns row intermediates/KV is not
execution evidence: this source consumes `main_gpu` only for `NONE` selection.
Do not add a Llaminar topology requirement solely from that stale help.

Ngram residency is an applicability axis: RAM, SSD mmap, bounded RAM/SSD,
replicated VRAM where it fits, fixed sharded VRAM, and pooled VRAM/RAM/SSD.
Preserve VRAM > RAM > SSD priority and lowest-priority ngram VRAM admission.
A zero-cache control does not prove positive-capacity promotion or eviction.
Verify actual grants, traffic, hot-row movement and reset for each supported
placement; avoid an invalid Cartesian product of mode names and topologies.

## Functional and performance execution

1. CPU/hybrid observer and semantic validation fixes passed focused regressions
   and the six-step panel. Refresh complete Unit and ProductionTestPreflight
   for the coherent source/build, including frozen full-stage references and
   the explicit all-GPU ROCm TP4 declaration.
2. Complete the extended panel, matched full-stage diagnosis, reset/prefix and
   source/path/driver checks. Review baseline and limits before installing
   canonical numerical acceptance and CI projections.
3. Implement/admit each required topology. Typed model-free tests cover degrees,
   boundaries, empty participants and live communication extents; native tests
   prove original-model admission, capture, placement and numerics. Neither
   replaces the other.
4. Compare unprofiled Release with exact prompt IDs, bucket geometry, context,
   KV precision, worker affinity, cache state, warmups and generation work.
   Keep canonical 512-prefill/256-generation distinct from the fork's separately
   reset pp2048/tg128 publication. Reproduce either workload explicitly.
5. Attribute exposed phase/kernel gaps before tuning. Profile exact CUDA/HIP
   candidates separately, preserve zero-spill/arithmetic obligations, and rerun
   the corresponding accuracy/performance pair. MTP additionally matches
   attempted/accepted work and predictor placement. Intrusive observer timings
   never become throughput results.

The earlier fork observation failed in warmup and produced no usable logits
or timing baseline. Its command used TP2 pairs, not TP4; diagnose that actual
launch before claiming a baseline comparison.

### Explicit comparison profiles without GPU peer access

Source review of the pinned fork distinguishes its published fused-HCP profile
from its staged transports. `HCP_AR=1` launches direct writes into other GPUs'
inboxes and requires real enabled peer access. Disabling producer-side push,
absorb fusion or the separate ordinary pair reduction's P2P option does not
remove that requirement. This is consistent with the earlier warmup fault,
but source inspection alone does not prove its cause.

The following are separately selected comparison profiles. The TP4 profile
has completed the short frozen-history numerical diagnostic described below;
their matched performance qualification remains pending:

- TP2×PP2: local `HC_PERSIST=4`, pair-compatible `HCP_INJ=1`,
  `HCP_AR=0`, `ALLREDUCE=internal`, `AR_P2P=0`, and staged boundary copies.
- TP4: `HC_PERSIST=4`, `HCP_INJ=0`, `HCP_AR=0`, `ALLREDUCE=none`,
  `ARN=0`, and `STAGED_COPY=1`, selecting the existing generic butterfly.
- Four physical singleton PP stages: layer split, one virtual stage per GPU,
  `HCP_AR=0`, and `STAGED_COPY=1`.

The TP4 original-nine/six-decode diagnostic completed with clean retirement and
all 144 original expert banks verified on four physical MI50s. Its intermediate
dimension splits 128/128/128/256, with all 512 expert IDs on every child; down
projections split input columns by the same widths. Preserve those observed
semantics when implementing the corresponding Llaminar cell. Current Llaminar
TP4 gate/up expert-ID ownership and down output-column sharding is a different
cell. Neither source reachability nor this short diagnostic certifies the
fork's published throughput, long-context behavior or MTP policy.

These abbreviations identify the fork's corresponding `GGML_CUDA_*` controls;
the deferred command records retain their complete environment names. Clear
inherited timing-only and device-virtualization overrides. The retained fork
build has RCCL disabled; none of these candidates claims RCCL execution.
Actual preparation must prove topology, transport, full GPU expert placement,
CPU PLE placement, numerical accuracy and clean retirement. A 4K smoke does
not admit the matched 32K workload. These candidates must never be reported
as reproductions of the published peer-dependent optimized result.
