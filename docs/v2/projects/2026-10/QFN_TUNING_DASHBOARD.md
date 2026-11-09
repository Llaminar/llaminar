# QFN tuning dashboard

Updated 2026-11-09. **Keep this file at or below 15,000 UTF-8 bytes.** Replace
status entries in place; put experiment histories, commands, samples and failure
analysis in the linked tuning notes, not in this dashboard.

## Current status

**Active hybrid target: 500 prefill /40 decode tok/s on 2×MI50 +2×CPU TP.**
Provisional NUMA-correct ROCm 10.1 Release closes at **300.092 P /38.357
after-prefill D**; all 768 IDs/MTP/initial placement match. Historical remote-GPU
rates are numerical controls, **not a placement speedup baseline**. Passive
PID masks are local, but benchmark rank→PID publication is missing; nonzero DRM
discovery bytes do not prove ownership. Native inventory/ownership gates pass.
[SDK status](../2026-11/QFN_ROCM_101_SDK_UPGRADE_20261109.md).
Short-prefill indexer elision saves **36 kernels/chunk/GPU**; its matched
control showed **+7.09% P**, with unchanged output IDs/MTP.
Paired retained graphs cost eight GPU expert slots: **8066 total, 4033/card**.
Context stays 262144 with dynamic MTP/balancing, RAM ngrams, local GPU caches
and 16/32 GiB RAM/disk prefix budgets.

**MPI/CMA fix is pushed in `2eda003d6`.** Typed registered migration backing
passes 17 native/transfer and 21 exact-feature checks; clean model output/MTP
matches with zero read errors. Throughput is approximately flat versus the
explicit-none control. Full preflight, exact-feature GPU/model and OpenCode
qualification remain open; previous broad-tree passes are not that certificate.

**Current tuning:** modern scope retains105 geometries,315 completed M1 jobs
and11 grouped cells; no broad resweep. M3 overlay: **11% component gain**, model flat.
CPU Auto: all21 byte-exact; mixed FFN improves **2.5–14%** in16 formats,
five within **+2.5%**. QFN-shaped FFN saves **26–31%** in both row orders.
Cache/work block sizes1–4 pass; model uses Auto. CPU-result readiness remains next.
[All-format evidence](../2026-11/QFN_CPU_ALL_FORMAT_JOB_OWNERSHIP_20261109.md).
**Latest CPU slice:** selected M1 loops have zero vector stack traffic;
20 focused tests pass, all 768 IDs/MTP/placement match. No model speedup proved;
final-add fusion withdrawn; eight-row candidate rejected: bytes/spills pass,
component +0.9%/−4.5% throughput in reversed orders. [Decision](../2026-11/QFN_CPU_EIGHT_ROW_REGISTER_TILE_20261109.md). Two complete P/D profile
pairs: exact GNU libgomp uses **59–68% P /76–82% D user cycles**, nearly all in
two pause loops—not wall overhead or saved time. Original requested-SIGINT
receipt stays red; separate actual-recorder control retires/flushes correctly.
**Keep GNU wait policy unchanged:** fresh unset 299.560P/38.994D; spin0
199.393/29.381 and spin1000 203.445/32.726 regress. All 768/MTP/config/initial
placement match; actual inputs verified, dynamic movement can differ.
[OpenMP result/limits](../2026-11/QFN_CPU_OPENMP_WAIT_ATTRIBUTION_20261109.md).
[Evidence](../2026-11/QFN_CPU_M1_SCALE_LIFETIME_20261109.md).
[Overlay scope](../2026-11/QFN_NATIVE_QUANTISED_EXACT_OVERLAY_20261108.md),
[timeline](../2026-11/QFN_HYBRID_TIMELINE_AND_NUMA_20261108.md),
[root phases](../2026-11/QFN_HYBRID_ROOT_TIMING_20261108.md),
[admission](../2026-11/QFN_SELECTED_ATTENTION_WORKSPACE_ADMISSION_20261108.md),
[CPU assembly](../2026-11/QFN_CPU_NATIVEQUANTISED_SPILL_AUDIT_20261108.md).

Earlier ROCm TP4 Dynamic/OpenCode checkpoint: `2cf99995d`.
[Contract and MTP table](QFN_MTP_TP4_TUNING.md). A1 controls below use MTP off.

**QFN ROCm TP4: accepted Dynamic; fresh finite-trajectory timing complete.**
512/256 repetition: Dynamic **101.367 D**, Off **43.512**, fresh Castagna fixed6
**100.098**: **+1.267%** Dynamic/fork. All nine measured streams match using the
same 255-token decode denominator. Native ignores stop termination; the fork
permits EOS. This qualifies the observed finite trajectory, not identical EOS
semantics or general performance superiority. Speed 6/10 remains diagnostic.

That checkpoint records 110 tools, zero errors, 95.1% prefix-token reuse and
clean retirement. It does not certify later changes or its29 unavailable cases.
[Host diagnosis and receipts](QFN_HOST_REGISTRATION_INVALIDATION_20261028.md);
[controller plan](QFN_MTP_TARGET_CONFIRMATION_20261028.md).

[QSA][qsa], [GDN][gdn] and [online rescale][online-rescale] retain exact
logits; component savings do not establish model gain.

| Next action | State / decision |
|---|---|
| Admission / SDK10.1 | **Scoped final gates green; full preflight deferred.** Host 53/53 (5.105s), repaired RCCL 9/9 (36.549s, all four original storage cases), HIP 19/19 on each ISA, NUMA 13/13 and full-visibility inventory 4/4. Actual P2P proof 11 policies/2,640 checks: native 0 failures/original 96. lto-build8 GREEN 37.282 s; configure14/build5 exit0 (1216.682/179.489s). Receipt-prefix and probe-logging follow-ups pass. [SDK evidence](../2026-11/QFN_ROCM_101_SDK_UPGRADE_20261109.md). |
| Six-GPU moving MTP | Prior CUDA2 continuation + MI50×4 follower fixed1/fixed3/dynamic slices pass: 57 host +16 native, 40 cancellation lifetimes, 2,416 byte comparisons, four prefix histories and eight migrations. All contexts retire. This earlier slice is not a complete current SDK/model certificate. [Focused qualification](../2026-11/QFN_RECURRENT_SIX_GPU_MTP_FOCUSED_QUALIFICATION_20261106.md). |
| Native lifetime | [Captured component proof](../2026-11/QFN_READ_AHEAD_NATIVE_CAPTURE_QUALIFICATION.md): CUDA/ROCm each pass 40 lifetimes and 800 replays. [Early-current RAM/SSD proof](../2026-11/QFN_EARLY_CURRENT_CAUSAL_FOCUSED_20261105.md) and [ROCm TP2 fixed1/shared-owner proof](../2026-11/QFN_POOLED_CURRENT_READAHEAD_FOCUSED_20261106.md) pass. These establish scoped correctness and causal progress; combined-cache/dynamic-MTP Release overlap remains unqualified. Current hybrid benchmarks use ConsumerScheduled and warm all-hit requests. [Policy/overlap audit](../2026-11/QFN_HYBRID_NGRAM_POLICY_AND_OVERLAP_20261109.md). |
| Continuation / ngrams | Prior ROCm TP2 shared-owner RAM/fixed3 and SSD/fixed1 slices pass 2,416/1,208 byte comparisons, four prefix histories, public generation and retirement. Current hybrid uses ConsumerScheduled; combined-cache/dynamic-MTP Release overlap remains unqualified. [Shared-owner evidence](../2026-11/QFN_SHARED_OWNER_MTP_IDENTITY_FOCUSED_20261106.md), [policy/overlap](../2026-11/QFN_HYBRID_NGRAM_POLICY_AND_OVERLAP_20261109.md). |

QSA temporal/page certificates did not repay their costs through **64K/128K**;
no pruning policy is installed. [Study][certificates]. CUDA hardware profiling
is paused after a profiler/driver assertion; native events remain available.

## Canonical functional and performance matrix

The current target is these **six hardware rows**, across RAM/SSD ngram backing
and bounded RAM/VRAM row caches with promotion/demotion. [Coverage plan][target-matrix]
owns the pairing strategy; the existing typed test definitions own commands.
Q6 tests ngram ownership on continuation GPUs, follower GPUs, and both. Its four
MI50s must execute real expert work. Any required CPU expert backing is explicit.

Rates are tok/s: prefill **P**, after-prefill decode **D**. ⚪ means the revised
cell is pending; **— means unmeasured, never zero**. Earlier measurements below
remain controls and do not qualify the new tier/ownership combinations.

| Row | Topology | Accuracy | Llaminar P | Reference P | Strata P | Llaminar D | Reference D | Strata D |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1× MI50 | ⚪ | — | — | — | — | — | — |
| Q2 | 2× MI50 | 🟢 RAM/C0 + SSD/C64 Static/Off slices; rest ⚪ | — | — | — | — | — | — |
| Q3 | 4× MI50 | ⚪ | — | — | — | — | — | — |
| Q4 | 1× RTX 3090 | ⚪ | — | — | — | — | — | — |
| Q5 | 2× RTX 3090 | ⚪ | — | — | — | — | — | — |
| Q6 | 2× RTX 3090 continuation + 4× MI50 followers | ⚪ | — | — | — | — | — | — |

Functional coverage retains MTP off/1/3/dynamic and dynamic expert balancing.
Small C0/C64 row-cache fixtures force movement; performance uses useful admitted
capacities and reports exact tier bytes, hit rates and working set. Earlier pre-10.1
native transfer floor: retained Release 22/28 green. Fresh original-owner
Integration runs pass all six RCCL cases and all six CUDA cases after native
stack provisioning; the original CUDA failure is reproduced by an isolated
control. Release timing and complete lookup/inference remain unqualified; no
transport default is promoted. Retained RCCL small-message event round trips
are about 48 µs versus mapped 65–66 µs.
[Timings, scope and failures](QFN_NGRAM_NATIVE_TRANSFER_FLOORS_2026-11-02.md).

### Earlier measured controls

| Control | Original policy | Accuracy | Llaminar P / D | Reference P / D | Strata P / D |
|---|---|---|---:|---:|---:|
| A1 | MI50 TP4; all GPU experts; RAM ngrams, MTP off | 🟢 short panel | 821.373 / 43.360 | 466.222 / 42.668 C | — |
| A2 | MI50×1 + CPU; 8/504 GPU/CPU experts, adaptive zero cache | [🟢 G60][a2] | 156.707 / 15.164 | 159.177 / 11.600 C | 45.270 / 24.024≈ |
| A3 | RTX 3090×1 + CPU; same 8/504 expert split | [🟢 G60][a3] | 180.920 / 18.231 | 129.020 / 13.059 U⚠ | 74.301 / 31.253≈ |
| C2a | MI50×2 + CPU, independent ConsumerLocal C64 | 🟢 four panels | 111.211 / 14.611 | — | — |
| C3a | RTX 3090×2 + CPU, independent ConsumerLocal C64 | 🟢 four panels | 116.941 / 17.988 | — | — |

C = Castagna Veloce; U = upstream llama.cpp because the pinned Castagna fork
does not build for SM86. ⚠ Upstream A3 decode spans12.293–17.417tok/s; placement
and repeatability still need closure. A2/A3 partition/mmap policies differ
between engines. Strata≈ uses the approved BF16 pack and default tail7 skipping
(about29.5% fewer expert contributions); it has no inherited accuracy badge.
Full top10 Strata controls are44.235P/19.750D on MI50 and73.508P/24.791D on3090.
[Baseline details][history]; [Strata distinctions][strata-history].

A1 uses `TensorParallel`, `GateUpOwnedDownColumns`, `VocabularySharded`,
Automatic Four-worker gate/up, shape-selected Q5 down, ValueMajor GDN and
FP16 Prefill8/Decode33 attention. Its fresh accepted Dynamic result is above.
Earlier alternative policies, CPU/PP controls and the superseded topology list
are preserved in the [target note][target-matrix], not expanded here.

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
28 NUMA 1 physical workers, one warmup and three measured iterations per lifecycle.
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

[target-matrix]: QFN_FUNCTIONAL_PERFORMANCE_TARGET_2026-11-02.md
