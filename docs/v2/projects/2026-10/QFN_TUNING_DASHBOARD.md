# QFN tuning dashboard

Updated 2026-10-30. **Keep this file at or below 15,000 UTF-8 bytes.** Replace
status entries in place; put experiment histories, commands, samples and failure
analysis in the linked tuning notes, not in this dashboard.

## Current status

**Checkpoint `2cf99995d`:** fresh finite-trajectory MTP comparison and full OpenCode
session complete on accepted ROCm TP4. [Contract and live MTP table](QFN_MTP_TP4_TUNING.md).
The A1 rates below are historical **MTP-off** results.

**QFN ROCm TP4: accepted Dynamic; fresh finite-trajectory timing complete.**
512/256 repetition: Dynamic **101.367 D**, Off **43.512**, fresh Castagna fixed6
**100.098**: **+1.267%** Dynamic/fork. All nine measured streams match using the
same 255-token decode denominator. Native ignores stop termination; the fork
permits EOS. This qualifies the observed finite trajectory, not identical EOS
semantics or general performance superiority. Speed 6/10 remains diagnostic.

That checkpoint passed **2,981 checks / 29 hardware skips**, including 786
Unit; Release passed. Its receipt does not certify subsequent changes.
Off/Serving-Dynamic HTTP passed with exact serial IDs, prefix restoration,
ticket/branch proof and clean retirement. The host note retains the evidence
and limitations.

Full OpenCode **passes**: ten phases, 110 tools, zero errors, 85 requests,
one client context compaction, 1,950.81 seconds; independent acceptance and
45 application tests. Native audit joins 85 responses/110 calls unchanged.
Runtime/resources/archive/driver and normal unforced retirement pass. Prefix
token reuse is **95.1%**; all owned disk-cache payloads are reclaimed after a
metadata-only audit. Earlier failed receipts remain unchanged.
[Host diagnosis and receipts](QFN_HOST_REGISTRATION_INVALIDATION_20261028.md);
[controller plan](QFN_MTP_TARGET_CONFIRMATION_20261028.md).

Other component work retains 185 exact logit rows and neutral model timing:
[query omission](QFN_SHORT_CONTEXT_QUERY_OMISSION.md), [QSA][qsa], [GDN][gdn]
and [online rescale][online-rescale]. Component savings do not establish model gain.

| Next action | State / decision |
|---|---|
| Admission | [Matrix](QFN_SOURCE_ADMISSION_MATRIX.md): `6f4c742e7` prerequisites green (3,048 passes/29 hardware skips); CUDA15/CPU33 16/16 source passes; 484 eligible. [Native init fix](QFN_NCCL_GIN_INITIALIZATION.md). |
| CUDA QSA timing | First M1/512 component saves 22.685%; remaining shapes pending, no model claim. |
| Broader topologies | Tune admitted A rows, then qualify B–D with matched references. |

QSA temporal/page certificates did not repay their costs through **64K/128K**;
no pruning policy is installed. [Study][certificates]. CUDA hardware profiling
is paused after a profiler/driver assertion; native events remain available.

## Topology and performance matrix

Rates are **tok/s**: prefill (**P**) and after-prefill decode (**D**).
🟢 = accuracy passed; 🔴 = numerical failure; ⚪ = pending.
**— means unmeasured/unavailable, never zero.** Accuracy green does not certify
speed or an image. G60 is the last complete topology gate; later A1 gates are scoped.

### A — implemented and admitted

| ID | Topology and ngrams | Accuracy | Llaminar P | Reference P | Strata P | Llaminar D | Reference D | Strata D | Next / limitation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 | ROCm TP4; all GPU experts; RAM ngrams | 🟢 serial panel + MTP HTTP | 821.373 | 466.222 C | — | 43.360 | 42.668 C | — | Fresh Dynamic101.367 D / C100.098; EOS caveat above. |
| A2 | ROCm1 + CPU1; 8 GPU / 504 CPU experts per layer; adaptive tiers, zero-cache control | [🟢 G60][a2] | 156.707 | 159.177 C | 45.270≈ | 15.164 | 11.600 C | 24.024≈ | Prefill trails Castagna; decode trails Strata. |
| A3 | CUDA1 + CPU1; 8 GPU / 504 CPU experts per layer; adaptive tiers, zero-cache control | [🟢 G60][a3] | 180.920 | 129.020 U⚠ | 74.301≈ | 18.231 | 13.059 U⚠ | 31.253≈ | Resolve upstream variance; decode trails Strata. |
| A4 | CPU1, 28 physical workers; SSD mmap ngrams | [🟢 G60][a4] | 98.466 | 48.408 C | n/a | 5.797 | 4.611 C | n/a | Both recorded phases lead Castagna. |
| A5 | PP(CUDA1, ROCm1, CPU1), 8/12/28 layers; SSD mmap ngrams | [🟢 G60][a5] | — | — | — | — | — | — | Exact GPU→GPU→CPU-tail reference unavailable at current pins. |

**A1 policy:** `TensorParallel` dense compute + `GateUpOwnedDownColumns` experts,
`VocabularySharded` terminal head, Automatic-resolved Four-worker owner gate/up,
shape-selected Q5 down publication, ValueMajor GDN and FP16 Prefill8/Decode33
attention. The feature branch selects that attention policy through admission
instead of diagnostic overrides. Do not mix it with the policy pairs
below or substitute their throughput.

| A1 policy variant | Dense compute | Experts | Accuracy | Llaminar P | Castagna P | Llaminar D | Castagna D | Decision |
|---|---|---|---|---:|---:|---:|---:|---|
| A1.1 | `TensorParallel` | `Apportioned` | 🟢 L64 | 561.867 | 467.821 | 30.515 | 42.959 | Slower than A1. |
| A1.2 | `PrefillTensorParallelDecodeReplicated` | `GateUpOwnedDownColumns` | 🔴 L65 | — | 467.821 | — | 42.959 | One forced decode row KL 1.119 exceeds the unchanged 1.0 ceiling; diagnose before timing. |
| A1.3 | `PrefillTensorParallelDecodeReplicated` | `Apportioned` | 🟢 L65 | 382.757 | 467.821 | 30.761 | 42.959 | Ornith arrangement; slower than A1. |

**Reference labels:** C = Castagna Veloce; U = separately labeled upstream
llama.cpp for CUDA because the pinned Castagna fork does not build for SM86.
⚠ Upstream A3 decode samples span 12.293–17.417 tok/s; tensor placement and
repeatability still need closure. A2/A3 expert partition and mmap policies
differ between engines, so these are disclosed engine comparisons, not matched
kernel measurements. [Baseline details and ranges][history].

**Strata ≈** uses the approved BF16 compatibility pack and default tail7 expert
skipping (about 29.5% fewer decode expert contributions). It has no inherited
accuracy badge. Full top-10 controls are **44.235 P / 19.750 D** on MI50 and
**73.508 P / 24.791 D** on RTX 3090. Preserve these arithmetic/routing distinctions;
the faster approximation is a labeled comparison, not equivalent work.
[Strata variants and staging analysis][strata-history].

### B — explicitly requested compositions

Two CPUs mean two physical NUMA/socket participants. These expert-domain
compositions do **not** imply mixed-vendor TP. B2/B3 use a ROCm TP4 continuation
domain; a PP composition of ROCm TP4 and CUDA TP2 is a different configuration.

| ID | Topology | Ngrams | Accuracy | Llaminar P | Ref P | Llaminar D | Ref D | Reference / prerequisite |
|---|---|---|---|---:|---:|---:|---:|---|
| B1 | CUDA2 + CPU2; CUDA TP2 continuation, both CPUs own experts | RAM | ⚪ | — | — | — | — | Upstream; MPI2 admission and real owner/path proof. |
| B2 | CUDA2 + ROCm4; CUDA expert domain | **Fully GPU**, sharded on ROCm lookup cohort | ⚪ | — | — | — | — | Mixed-vendor reference unproved; source capacity/MPI2 proof. No CPU tier substitution. |
| B3 | CUDA2 + ROCm4 + CPU2; CUDA and both CPUs own experts | **RAM** | ⚪ | — | — | — | — | Mixed-vendor reference unproved; positive work on every requested domain. |

[Exact composition intent and tier caps][compositions] remain planning inputs,
not admitted memory grants. Strata measurements for these rows are pending.

### C — remaining local topology coverage

RAM ngrams are the initial control. Admit two-card capacity explicitly; any
necessary CPU offload changes the row identity and must be disclosed and matched.
Never switch GGUF or placement silently. Strata baselines remain pending.

| ID | Topology / storage variant | Accuracy | Llaminar P | Ref P | Llaminar D | Ref D | Reference / next prerequisite |
|---|---|---|---:|---:|---:|---:|---|
| C1 | CPU1; RAM ngrams | ⚪ | — | — | — | — | C; qualify independently of SSD row A4. |
| C2 | ROCm TP2 | ⚪ | — | — | — | — | C; source partition/capacity. |
| C3 | CUDA TP2 | ⚪ | — | — | — | — | U; source partition/capacity. |
| C4 | ROCm TP4; Castagna within-expert F split | ⚪ | — | — | — | — | C; exact F slices 128/128/128/256; distinct from A1. |
| C5 | ROCm PP2; equal contiguous layer ranges | ⚪ | — | — | — | — | C; pipeline admission and ranges. |
| C6 | ROCm PP2; unequal contiguous layer ranges | ⚪ | — | — | — | — | C; pipeline admission and ranges. |
| C7 | CUDA PP2; equal contiguous layer ranges | ⚪ | — | — | — | — | U; pipeline admission and ranges. |
| C8 | CUDA PP2; unequal contiguous layer ranges | ⚪ | — | — | — | — | U; pipeline admission and ranges. |
| C9 | ROCm PP4; equal ranges, one stage/card | ⚪ | — | — | — | — | C; lifecycle green; accuracy/perf pending. |
| C10 | ROCm PP4; unequal layer ranges | ⚪ | — | — | — | — | C; source owners/capacity. |
| C11 | ROCm PP(TP2, TP2) | ⚪ | — | — | — | — | C; TP cohorts and pipeline boundary. |
| C12 | ROCm interleaved PP; 12 stages, 0/1/2/3 repeated ×3 | ⚪ | — | — | — | — | C; repeated ownership/concurrency-aware admission. |
| C13 | CPU prefix + CUDA suffix; partial-layer offload | ⚪ | — | — | — | — | U; exact layer/source placement. |
| C14 | CPU prefix + ROCm suffix; partial-layer offload | ⚪ | — | — | — | — | C; exact layer/source placement. |
| C15 | CUDA + CPU; whole-expert-tensor offload | ⚪ | — | — | — | — | U; tensor overrides, distinct from A3 expert-ID ownership. |
| C16 | ROCm + CPU; whole-expert-tensor offload | ⚪ | — | — | — | — | C; tensor overrides, distinct from A2 expert-ID ownership. |

### D — additional hardware required

This host has four MI50s and two RTX 3090s. Six mixed GPUs cannot satisfy TP8.
Initial ngram placement is RAM; bind the eight-card backend before admission.

| ID | Topology | Accuracy | Llaminar P | Ref P | Llaminar D | Ref D | Block / reference |
|---|---|---|---:|---:|---:|---:|---|
| D1 | CUDA TP4 | ⚪ | — | — | — | — | Need four CUDA cards; U. |
| D2 | CUDA PP4; equal ranges | ⚪ | — | — | — | — | Need four CUDA cards; U. |
| D3 | CUDA PP4; unequal ranges | ⚪ | — | — | — | — | Need four CUDA cards; U. |
| D4 | CUDA PP(TP2, TP2) | ⚪ | — | — | — | — | Need four CUDA cards; U. |
| D5 | Homogeneous TP8 | ⚪ | — | — | — | — | Need eight same-backend cards. |
| D6 | Homogeneous TP2 × PP4 | ⚪ | — | — | — | — | Need eight same-backend cards. |
| D7 | Homogeneous TP4 × PP2 | ⚪ | — | — | — | — | Need eight same-backend cards. |

## Accuracy and measurement contract

Mean final-logit KL is in nats; top-5 is mean set overlap divided by five.
Values below are **prefill / decode**. A1's current required short panel uses
16 frozen histories plus six forced decode steps; eight heldout histories and
reset replay also pass. All 185 rows, including diagnostic/reset rows, match
the accepted references. N2047/D9 remains diagnostic, not long-context certification.

| Accuracy slice | Mean KL | Top-1 | Top-5 |
|---|---:|---:|---:|
| A1 current / checkpoint196, short | 0.025614 / 0.028276 | 93.75% / 95.83% | 92.50% / 92.29% |
| A1 current, heldout-short | 0.014911 / 0.004553 | 100% / 100% | 95.00% / 91.67% |
| A2 G60 | 0.024628 / 0.032013 | 87.50% / 98.96% | 95.00% / 92.08% |
| A3 G60 | 0.023410 / 0.020687 | 81.25% / 97.92% | 93.75% / 91.88% |
| A4 G60 | 0.016063 / 0.022886 | 93.75% / 100% | 92.50% / 90.21% |
| A5 G60 | 0.032444 / 0.024587 | 93.75% / 97.92% | 90.00% / 90.21% |

**Matched A1 workload:** original four-shard `Qwen3.8-Flash-Next-UD-Q4_K_XL`,
512 prompt tokens, 256 emitted tokens (255 after-prefill steps), context4096,
FP16 KV, M512, greedy seed42, MTP/prefix cache off, unprofiled Release,
28 NUMA1 physical workers, one warmup and three measured iterations per lifecycle.
Arithmetic: HC-up TwoTermQ8, FP16 router operands with FP32 accumulation,
FP16 MoE alpha/beta and ordinary Q8 projection activations.

Every changed topology, precision, storage policy or context needs its own
accuracy/performance pair. Keep all samples and failed runs in linked evidence;
observe quiet-host, driver, retirement and token-equality checks. Use the
[canonical numerical gates][accuracy-plan] and layer/stage PyTorch CSVs for
diagnosis. A pooled mean or a component gain is insufficient for promotion.

Storage coverage includes RAM, SSD mmap, bounded/pool tiers and replicated/
sharded VRAM, with hot-row movement/reset. Priority is VRAM → RAM → SSD;
ngrams have lowest VRAM admission priority. The [qualification plan][plan]
owns full geometry, ownership and live-communication coverage.

## Detail and evidence

- [Current QSA/prepared-GDN qualification][qsa]; [GDN resources and tuning][gdn].
- [Expert kernels and checkpoint196][checkpoint196]; [stage comparison][stages];
  [attention tuning][attention]; [TP compute scaling][scaling].
- [Qualification plan][plan], [implementation plan][implementation] and
  [ngram tier plan][ngrams] own the broader requirements.
- [Dated history][history] preserves the former dashboard's measurements,
  failed attempts, reference pins and sample ranges. **Do not append new status there.**
  The live [paired-case registry][cases] owns test definitions; this dashboard is a view.

[checkpoint196]: QFN_ROUTE_OWNED_MOE_TUNING.md#canonical-q5-partition-workers196--qualified-checkpoint-2026-10-25
[qsa]: QFN_QSA_PREPARATION_FUSION.md#selected-production-candidate
[gdn]: QFN_GDN_RECURRENCE_TUNING.md
[certificates]: QFN_QSA_CERTIFIED_PRUNING_STUDY.md
[history]: QFN_TUNING_HISTORY_2026-10-25.md
[strata-history]: QFN_TUNING_HISTORY_2026-10-25.md#additional-strata-reference--full-routing-and-default-approximation
[compositions]: QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md#additional-requested-device-and-ngram-cells
[accuracy-plan]: QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md#frozen-short-context-pilot-and-unseen-prompts
[plan]: QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md
[implementation]: QWEN_FLASH_NEXT_IMPLEMENTATION.md
[ngrams]: QFN_NGRAM_TIERED_STORAGE_PLAN.md
[stages]: QFN_CASTAGNA_STAGE_COMPARISON.md
[attention]: QFN_PACKED_ATTENTION_TUNING.md
[online-rescale]: QFN_ATTENTION_ONLINE_RESCALE.md
[column-grid]: QFN_SCOPED_QUERY_COLUMN_GRID.md
[scaling]: QFN_TP_COMPUTE_SCALING.md
[cases]: ../../../../tests/v2/integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h
[a2]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/rocm-cpu-static8-adaptive-hc-up-two-term/report.json
[a3]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/cuda-cpu-static8-adaptive-hc-up-two-term/report.json
[a4]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/cpu-mmap-hc-up-two-term/report.json
[a5]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/hybrid-cpu-tail-mmap-hc-up-two-term/report.json
