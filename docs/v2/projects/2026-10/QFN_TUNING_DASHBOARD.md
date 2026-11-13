# QFN tuning dashboard

Updated 2026-11-13. **Keep this file at or below 15,000 UTF-8 bytes.** Replace
status entries in place; put experiment histories, commands, samples and failure
analysis in the linked tuning notes, not in this dashboard.

## Current status

**Current milestone: Auto CUDA2 continuation + MI50×4 whole-expert EP, RAM
ngrams, dynamic MTP1–15, 262,144 context.**
[Complete OpenCode (`q702`)](../2026-11/QFN_CUDA2_MI50X4_RAM_OPENCODE_20261113.md):
**10/10 phases**, 98 requests, **107 tools/zero errors**, 56 app tests pass.
All app/runtime/resource/driver/retirement gates green. **98.04% prefix reuse**,
**2.81 s TTFT**, **76.64 decode tok/s**; short greedy 512/256 **766.34/120.00 P/D**.
Restored-prefix MTP role admission repaired; 571 MemoryPlanner tests and four
focused aliases pass. Auto **30.41 s**; selected MI50s span switches, no optimal
fabric claim. Full preflight not repeated.

**Retained CPU EP:** [CUDA2+CPU2 `q686`](../2026-11/QFN_CUDA2_CPU2_EP_OPENCODE_20261113.md)
10/10 phases, 1/105 tool errors, 53 tests; stress green, strict task flag false.
98.14% reuse, 4.73 s TTFT, 37.82 decode tok/s; short **389.86/59.66 P/D**.

**Retained GPU milestones:** PP8 `q607` full ten-phase pass, 124 tools/zero errors,
98.18% reuse. TP4 `q639` user-accepted **8/10 partial**, 1/131 errors, 92.34% reuse;
engineering closure passes, full app/native audit incomplete. TP4 context 47,168,
approved 16K reply reserve; PP8 context 262,144. Short P/D: TP4 **616.19/93.41**,
PP8 **1082.63/58.41**. Both use GateUpOwnedDownColumns; PP8 ngrams GPU-sharded,
TP4 RAM. [Auto placement](../2026-11/QFN_BOUNDED_INTERCONNECT_PLANNING_20261112.md)
selects intact switches; 22.1/23.8 s planning remains above the 5 s target.
**Prior checkpoint:** **5,256** selected Unit/preflight tests passed, zero skips;
39 absent-hardware cases unqualified. No whole-run receipt; the user accepted
this checkpoint and waived repetition.
**Current GateUp PP8 accuracy (`q587`):** 16 histories, prefill + six decode
steps, mean KL **0.01851/0.02042**, top-1 **93.75/96.88%**, top-5 overlap
**96.25/92.08%**. Reset byte-exact; native retirement/driver clean. Local
absolute-pilot pass at context 1024, MTP off with capacity 15 retained.
Campaign closed at the user's accepted TP4 stopping point; no rerun scheduled.
[Post-checkpoint inference](../2026-11/QFN_EP4_EP4_POST_CHECKPOINT_INFERENCE_20261112.md).
[Prerequisite repairs](../2026-11/QFN_PREREQUISITE_REPAIRS_20261111.md).
Earlier [prefill tuning](../2026-11/QFN_EP4_EP4_PREFILL_HC_UP_KPAIR_20261112.md)
reached **817.65 P tok/s**, +11.26% on its matched512/1 Static/MTP-off control.
Six byte/reset cells and model health pass; the broader arithmetic case's
lost driver-log cursor leaves those health windows unqualified.
[Decode study](../2026-11/QFN_EP4_EP4_DECODE_SUBMISSION_TUNING_20261112.md): submission/HC fusion unpromoted.
[Current repairs and evidence](QFN_CROSS_RANK_PREFLIGHT_2026-11-10.md),
[fixed VRAM ngram plan](QFN_FIXED_VRAM_PIPELINE_PLAN_2026-11-10.md).

Former hybrid control: 2×MI50 +2×CPU TP **339.306/40.566 P/D**. IDs/MTP
matched, but movement differed; no causal model speedup was claimed.
[Wave results](../2026-11/QFN_HYBRID_NINE_SLOT_MIGRATION_20261110.md),
[SDK status](../2026-11/QFN_ROCM_101_SDK_UPGRADE_20261109.md).

**Retained CPU:** byte-exact [21-format scheduling](../2026-11/QFN_CPU_ALL_FORMAT_JOB_OWNERSHIP_20261109.md)
and [ILP](../2026-11/QFN_CPU_PAIR_DOT_ILP_20261110.md); unchanged
[GNU wait policy](../2026-11/QFN_CPU_OPENMP_WAIT_ATTRIBUTION_20261109.md).
**Native Q8:** measured M448 tiles installed on gfx906/gfx1100; M64 unchanged.
No transport change installed: [compact-route boundary](../2026-11/QFN_NATIVE_ROUTE_COMPACTION_BOUNDARY_20261110.md).
[Scope/results](../2026-11/QFN_ROCM_NATIVE_Q8_PREFILL_20261109.md).
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
| Admission / SDK10.1 | **Focused fixes pass; full rerun waived.** Release accuracy and cleanup match. No whole-run certificate. [Evidence](../2026-11/QFN_EP4_EP4_POST_CHECKPOINT_INFERENCE_20261112.md). |
| Six-GPU moving MTP | **Current RAM/dynamic OpenCode green (`q702`).** Prior fixed1/fixed3/dynamic [migration/reset slices](../2026-11/QFN_RECURRENT_SIX_GPU_MTP_FOCUSED_QUALIFICATION_20261106.md) remain scoped; broader numerical/topology matrix open. |
| Native lifetime | CUDA/ROCm [captured component](../2026-11/QFN_READ_AHEAD_NATIVE_CAPTURE_QUALIFICATION.md), [early-current RAM/SSD](../2026-11/QFN_EARLY_CURRENT_CAUSAL_FOCUSED_20261105.md), and [TP2 shared-owner](../2026-11/QFN_POOLED_CURRENT_READAHEAD_FOCUSED_20261106.md) proofs pass. Combined-cache/dynamic-MTP Release overlap remains unqualified. |
| Continuation / ngrams | TP2 shared-owner RAM/fixed3 and SSD/fixed1 [focused slices pass](../2026-11/QFN_SHARED_OWNER_MTP_IDENTITY_FOCUSED_20261106.md). Hybrid timing uses ConsumerScheduled/warm hits; combined-cache overlap remains open. [Policy audit](../2026-11/QFN_HYBRID_NGRAM_POLICY_AND_OVERLAP_20261109.md). |

QSA temporal/page certificates did not repay their costs through **64K/128K**;
no pruning policy is installed. [Study][certificates]. CUDA hardware profiling
is paused after a profiler/driver assertion; native events remain available.

## Canonical functional and performance matrix

The matrix retains **six hardware rows** across RAM/SSD ngram backing and bounded
RAM/VRAM row caches, plus the new **Q7 active eight-GPU target** with GPU-sharded
ngrams. [Coverage plan][target-matrix]
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
| Q3 | 4× MI50; GateUpOwnedDownColumns, RAM ngrams | 🟢 short panel, accepted one-row limit; app partial 8/10 accepted | 616.19† | — | — | 93.41† | — | — |
| Q4 | 1× RTX 3090 | ⚪ | — | — | — | — | — | — |
| Q5 | 2× RTX 3090 | ⚪ | — | — | — | — | — | — |
| Q6 | 2× RTX 3090 continuation + 4× MI50 followers | ⚪ parity; 🟢 RAM/dynamic app 10/10 | 766.34† | — | — | 120.00† | — | — |
| Q7 | PP(EP(4×MI50), EP(4×MI50)); ngrams sharded across all8 | 🟢 short panel; P/D KL0.01851/0.02042; full app green | 1082.63† | — | — | 58.41† | — | — |

† Current selected-Auto diagnostics: 512/256, greedy dynamic MTP, prefix off,
one warmup + three samples; Q3 context 47,168, Q6/Q7 context 262,144. No matched
external reference or general throughput claim. Accuracy uses short contexts
with MTP off; full stochastic apps qualify the serving configuration separately.

Functional coverage retains MTP off/1/3/dynamic and dynamic expert balancing.
Small C0/C64 fixtures force movement; performance reports useful admitted tier
bytes, hit rates and working sets. Original-owner Integration passes all six
RCCL and six CUDA transfer cases after native stack provisioning. Complete
lookup/inference and Release transport timings remain unqualified; no transport
default is promoted. Earlier retained RCCL event round trips were about48 µs
versus mapped65–66 µs.
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
[a2]: QFN_TUNING_HISTORY_2026-10-25.md#accuracy-detail-for-checkpoint60-broad-panel-observations
[a3]: QFN_TUNING_HISTORY_2026-10-25.md#accuracy-detail-for-checkpoint60-broad-panel-observations

[target-matrix]: QFN_FUNCTIONAL_PERFORMANCE_TARGET_2026-11-02.md
