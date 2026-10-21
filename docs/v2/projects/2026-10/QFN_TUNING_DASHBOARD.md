# QFN tuning dashboard

Updated: 2026-10-21 14:00 UTC.

Goal: beat **Castagna Veloce in both prefill and decode**, while keeping each row's accuracy gate green. This is our working scorecard. Runtime configurations remain owned by the typed parity inventory; a planned row does not advertise implemented support.

**Priority:** obtain Castagna baselines for the implemented/admitted rows in A, then measure and tune Llaminar. Finish that pass before implementing the novel compositions and remaining topology families in B–D. Preserve the private implementation drafts for that later work.

Current evidence: all five admitted rows passed checkpoint60 accuracy; three retain checkpoint58 Llaminar policy timings. **A1 now has uninstrumented Llaminar and Castagna observations; decode remains behind.** Complete checkpoint60 Unit/production preflight and all five model accuracy cells passed, with clean drivers, unchanged runtime metadata, exact reset replay and normal native retirement. The qualified source is committed and pushed to `feat/qwen-flash-next` as `6c06e8efe`; its commit hook authenticated all 2,454 prerequisite registrations without rerunning them. R58 timings still need an uninstrumented refresh.

Accuracy legend: **🟢 GREEN** = all required numerical groups and execution/retirement/reset checks passed; **🔴 RED** = an observed gate failure; **⚪ PENDING** = not tested or prerequisites missing. Hardware, capacity and implementation blocks are recorded separately from numerical failure. `—` means unmeasured. All throughput cells are **tokens/second**.

## A. Implemented and admitted — baseline and tune first

`G60` names the last completed accuracy checkpoint. `R58†` names the earlier instrumented Release policy comparison, which needs a fresh uninstrumented measurement for the fork comparison. No row has yet demonstrated a speed win over Castagna.

| ID | Topology / actual work placement | Ngrams | Accuracy | Llaminar prefill | Castagna prefill | Llaminar decode | Castagna decode | Next action |
|---|---|---|---|---:|---:|---:|---:|---|
| A1 | 4×ROCm: TP4, all GPU experts | RAM | [🟢 GREEN G60][a1] | 779.815 L61‡ | 460.320 F61 | 29.696 L61‡ | 41.264 F61 | Align effective host placement, then attribute the decode gap. |
| A2 | 1×ROCm + 1×CPU: expert overlay, 8 GPU / 504 CPU experts per layer | Adaptive VRAM/RAM/SSD, zero-cache control | [🟢 GREEN G60][a2] | 131.796 R58† | — | 11.885 R58† | — | Match actual expert offload and ngram residency. |
| A3 | 1×CUDA + 1×CPU: expert overlay, 8 GPU / 504 CPU experts per layer | Adaptive VRAM/RAM/SSD, zero-cache control | [🟢 GREEN G60][a3] | 145.968 R58† | — | 13.823 R58† | — | Match actual expert offload and ngram residency. |
| A4 | CPU only: one NUMA participant, 28 physical-core workers | SSD mmap | [🟢 GREEN G60][a4] | 82.806 R58† | — | 5.127 R58† | — | Match CPU ISA, affinity, workers and mmap behavior. |
| A5 | PP(1×CUDA, 1×ROCm, 1×CPU): 8 / 12 / 28 layers | SSD mmap | [🟢 GREEN G60][a5] | — | — | — | — | Prove fork mixed-vendor execution before admitting its baseline. |

A2/A3 retain most experts on CPU: these are not all-GPU performance results. Their zero-cache adaptive controls do not qualify positive-capacity promotion/eviction. A1 uses Llaminar's existing `GateUpOwnedDownColumns` partition. Castagna TP4 shards each expert's intermediate dimension instead. A same-hardware/model end-to-end comparison is useful now, with that difference recorded; implementing C4 must not delay tuning A1.

`F61` is the graphs-enabled Castagna profile, chosen as the main A1 comparison because disabling graphs substantially reduced decode speed. Both profiles remain visible; the dashboard does not combine different profiles' best phase numbers.

| A1 Castagna profile | Median prefill | Median decode | Measured prefill range | Measured decode range | Evidence |
|---|---:|---:|---:|---:|---|
| Graphs enabled | 460.320 | 41.264 | 457.272–461.473 | 41.247–41.526 | [F61 enabled](../../../../parity-results/qwen-flash-next/tp4-baseline61-castagna-graphs-default/result.json) |
| Graphs explicitly disabled | 463.898 | 33.393 | 462.698–465.584 | 33.065–33.470 | [F61 disabled](../../../../parity-results/qwen-flash-next/tp4-baseline61-castagna-graphs-disabled/result.json) |

Each profile ran one full warmup and three measured requests with 512 prompt tokens, zero cached prompt tokens, 256 output tokens and 255 after-prefill decode steps. Source/runtime identities, driver windows and requested graceful native retirement passed. The synthetic repeated-token prompt produced the same output sequence on every request. These establish throughput, not numerical model qualification. The fork's 320+192 physical prefill split and within-expert partition differ from Llaminar's M512/expert ownership.

`L61‡` ([result](../../../../parity-results/qwen-flash-next/tp4-baseline61-llaminar/result.json), [native benchmark](../../../../parity-results/qwen-flash-next/tp4-baseline61-llaminar/benchmark.json)) is an uninstrumented Release observation: prefill 779.352–780.204 and decode 29.527–29.817 tok/s across three measurements, retained M512 graphs, MTP/cache disabled and the same 256 output token IDs as the fork. Clean runtime/driver/retirement checks passed. **Host placement still needs alignment:** both launchers began with physical CPU0–55, but Llaminar's ordinary MPI bootstrap selected the GPUs' NUMA1 physical CPU28–55 and OMP cores/close. Bootstrap exports OMP28, but the explicit `--threads 56` is applied again during runtime initialization; this is a 56-requested-worker/28-place profile. The fork used 56 workers without that rebinding. These observations identify a decode deficit but do not establish a controlled speedup ratio. Preserve both before repeating a declared common host-resource profile.

## B. Explicitly requested compositions — after A is tuned

Two CPU participants mean the two physical NUMA/socket participants, with distinct physical-core worker sets. These proposed MPI2 expert-domain compositions do not create a mixed-vendor TP group. The requested ngram tier is part of each row's identity.

| ID | Topology / proposed work placement | Ngrams | Accuracy | Llaminar prefill | Castagna prefill | Llaminar decode | Castagna decode | Remaining prerequisite |
|---|---|---|---|---:|---:|---:|---:|---|
| B1 | 2×CUDA + 2×CPU: CUDA TP2 continuation; both CPUs own experts | RAM | ⚪ PENDING | — | — | — | — | Actual MPI2 admission, owner/path proof and paired accuracy; fork equivalence unproved. |
| B2 | 2×CUDA + 4×ROCm: ROCm TP4 continuation; CUDA expert domain | **Fully GPU VRAM**, sharded on the actual ROCm lookup cohort | ⚪ PENDING | — | — | — | — | Original-source capacity and MPI2 proof; no CPU expert or ngram tier substitution. |
| B3 | 2×CUDA + 4×ROCm + 2×CPU: ROCm TP4 continuation; CUDA and both CPUs own experts | **RAM** | ⚪ PENDING | — | — | — | — | Actual MPI2 admission and positive ownership/work on every requested domain. |

Rank ownership and initial expert tier caps remain in the [qualification plan](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md#additional-requested-device-and-ngram-cells). They are benchmark intent, not admitted memory grants. The alternative PP composition of ROCm TP4 and CUDA TP2 is distinct and cannot stand in for B2/B3.

## C. Remaining local baseline topology coverage — queued

These rows preserve the agreed baseline-family coverage; they are not yet qualified full-model configurations. RAM ngrams are the initial control unless stated otherwise. Two-card capacity needs explicit admission: record any necessary CPU offload in the exact configuration and match it in both engines. Never silently change the GGUF or offload policy to make a row fit.

| ID | Topology / configuration | Ngrams | Accuracy | Llaminar prefill | Castagna prefill | Llaminar decode | Castagna decode | Remaining prerequisite |
|---|---|---|---|---:|---:|---:|---:|---|
| C1 | CPU only: one NUMA participant, RAM control | RAM | ⚪ PENDING | — | — | — | — | Qualify this storage variant independently of A4. |
| C2 | 2×ROCm: homogeneous TP2 | RAM | ⚪ PENDING | — | — | — | — | Bind source partition and any declared CPU offload; admit capacity. |
| C3 | 2×CUDA: homogeneous TP2 | RAM | ⚪ PENDING | — | — | — | — | Bind source partition and any declared CPU offload; admit capacity. |
| C4 | 4×ROCm: TP4 with the fork's within-expert F partition | RAM | ⚪ PENDING | — | — | — | — | Exact native F slices 128/128/128/256 and live-output reduction; distinct from A1. |
| C5 | 2×ROCm: contiguous PP2, equal layer ranges | RAM | ⚪ PENDING | — | — | — | — | Full-model admission, GPU pipeline and exact range proof. |
| C6 | 2×ROCm: contiguous PP2, unequal layer ranges | RAM | ⚪ PENDING | — | — | — | — | Bind ranges and capacity before timing. |
| C7 | 2×CUDA: contiguous PP2, equal layer ranges | RAM | ⚪ PENDING | — | — | — | — | Full-model admission, GPU pipeline and exact range proof. |
| C8 | 2×CUDA: contiguous PP2, unequal layer ranges | RAM | ⚪ PENDING | — | — | — | — | Bind ranges and capacity before timing. |
| C9 | 4×ROCm: contiguous PP4, equal ranges, one stage per card | RAM | ⚪ PENDING | — | — | — | — | Singleton-stage control for C12; full-model admission. |
| C10 | 4×ROCm: contiguous PP4, unequal layer ranges | RAM | ⚪ PENDING | — | — | — | — | Bind ranges, source owners and capacity. |
| C11 | 4×ROCm: PP(TP2, TP2) | RAM | ⚪ PENDING | — | — | — | — | Native homogeneous TP cohorts and pipeline boundary. |
| C12 | 4×ROCm: interleaved PP, 12 logical stages, card order 0/1/2/3 repeated three times | RAM | ⚪ PENDING | — | — | — | — | Repeated physical ownership and concurrency-aware admission. |
| C13 | CPU prefix + CUDA suffix: partial-layer offload | RAM | ⚪ PENDING | — | — | — | — | Exact layer ranges and original source placement. |
| C14 | CPU prefix + ROCm suffix: partial-layer offload | RAM | ⚪ PENDING | — | — | — | — | Exact layer ranges and original source placement. |
| C15 | CUDA + CPU: whole-expert-tensor offload | RAM | ⚪ PENDING | — | — | — | — | Match actual fork tensor overrides; differs from A3's expert-ID ownership. |
| C16 | ROCm + CPU: whole-expert-tensor offload | RAM | ⚪ PENDING | — | — | — | — | Match actual fork tensor overrides; differs from A2's expert-ID ownership. |

## D. Additional hardware — queued

This host has four MI50s and two RTX 3090s. Six mixed-vendor GPUs do not satisfy an eight-card homogeneous case. The CUDA four-card rows retain backend symmetry for the topology families.

| ID | Topology / configuration | Ngrams | Accuracy | Llaminar prefill | Castagna prefill | Llaminar decode | Castagna decode | Block |
|---|---|---|---|---:|---:|---:|---:|---|
| D1 | 4×CUDA: homogeneous TP4 | RAM initially | ⚪ PENDING | — | — | — | — | Four CUDA devices unavailable. |
| D2 | 4×CUDA: contiguous PP4, equal layer ranges | RAM initially | ⚪ PENDING | — | — | — | — | Four CUDA devices unavailable. |
| D3 | 4×CUDA: contiguous PP4, unequal layer ranges | RAM initially | ⚪ PENDING | — | — | — | — | Four CUDA devices unavailable. |
| D4 | 4×CUDA: PP(TP2, TP2) | RAM initially | ⚪ PENDING | — | — | — | — | Four CUDA devices unavailable. |
| D5 | 8 homogeneous GPUs: TP8 | RAM initially | ⚪ PENDING | — | — | — | — | Eight same-backend devices unavailable; bind backend before execution. |
| D6 | 8 homogeneous GPUs: TP2 × PP4 | RAM initially | ⚪ PENDING | — | — | — | — | Eight same-backend devices unavailable. |
| D7 | 8 homogeneous GPUs: TP4 × PP2 | RAM initially | ⚪ PENDING | — | — | — | — | Eight same-backend devices unavailable. |

## Accuracy detail for the current green observations

Checkpoint60 broad-panel prompt-balanced final-logit metrics: 16 frozen histories, prefill plus six forced decode steps. Eight separate heldout histories also passed on every row. Values below are **prefill / decode**. KL is in nats; top-1 is exact agreement; top-5 is mean set intersection divided by five. N2047/D9 remains diagnostic, not long-context certification.

| Row | Mean KL | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|
| A1 | 0.033536 / 0.032070 | 93.75% / 97.92% | 87.50% / 92.29% |
| A2 | 0.024628 / 0.032013 | 87.50% / 98.96% | 95.00% / 92.08% |
| A3 | 0.023410 / 0.020687 | 81.25% / 97.92% | 93.75% / 91.88% |
| A4 | 0.016063 / 0.022886 | 93.75% / 100.00% | 92.50% / 90.21% |
| A5 | 0.032444 / 0.024587 | 93.75% / 97.92% | 90.00% / 90.21% |

Use the frozen [numerical gates](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md#frozen-short-context-pilot-and-unseen-prompts), including every decode offset, tails and structural checks. A pooled mean alone cannot make a row green. Retain full layer/stage PyTorch parity and CSV diagnostics for failures. Accuracy must bind the same original quantized GGUF, arithmetic policy and topology as the timed Llaminar candidate.

## Measurement contract

- **Model:** the same four original `Qwen3.8-Flash-Next-UD-Q4_K_XL` shards in both engines. No requantization or smaller-model substitution.
- **Initial workload:** 512 actual prompt tokens, 256 generated tokens, context capacity 4096, FP16 KV, greedy sampling with seed 42, MTP off, prompt-cache misses. Freeze ` test` repeated 512 times and authenticate its token IDs in both engines. Longer contexts receive their own accuracy/performance entries.
- **Llaminar arithmetic:** HC-up TwoTermQ8, FP16 router operands with FP32 accumulation, FP16 MoE alpha/beta; ordinary wide-projection Q8 activation math retained. Record all six selectors. Precision changes cannot inherit another policy's green result.
- **Timing:** unprofiled Release, initialization separate. Explicitly match warmup/repetitions and retained graph state. Compare Llaminar `decode_after_prefill` to the corresponding fork interval: 256 emitted tokens contain 255 after-prefill decode steps. Retain every sample, dispersion and token sequence; a noise-sized median lead is not a demonstrated win.
- **Placement:** match hardware, CPU ISA/affinity/workers, expert/layer placement, ngram residency and cache policy. Different internal partitions must be disclosed as a same-hardware/model end-to-end comparison; they cannot establish a kernel-only speed advantage.
- **Fork pin:** Castagna Veloce `3483d462715d61f21ed5877836be35a724446359`. Qualify explicit graph-enabled and graph-disabled candidates separately where applicable; never recover by silently changing the topology or execution policy. Mixed-vendor/expert-ID support remains unproved until exercised. Unsupported fork configurations need evidence, not invented scores or automatic wins.
- **Row completion:** current green accuracy and repeatable wins in **both** phases on the declared workload. Record prefill/decode verdicts separately when only one wins. Preserve failures and earlier measurements instead of selecting favorable samples.

`R58†` values are TwoTerm medians of six measurements across two process lifecycles in an ABBA policy experiment, with one warmup and three measurements per lifecycle and 28 physical-core workers. Native records show `perf_stats.enabled=true`: these are prior policy measurements, not the final uninstrumented cross-engine baseline. Fresh baseline work also reauthenticates per-request cache misses; process-wide cache counters can include setup/warmup. Replace these numbers only after the fork baselines are collected.

## Remaining qualification work

The optional all-format performance sweep stopped after 19 completed CUDA M1 formats because its Q8_1 fixture called the generic tensor factory instead of the existing typed raw-byte Q8_1 factory. Production Q8_1 correctness coverage passed; this is a separate performance-fixture defect. A narrow saved-source construction repair and all-format exact-byte regression are prepared for a subsequent test-only change. Preserve the completed component observations and resume the missing formats after the repair. The full preflight/accuracy checkpoint above does not imply that this optional performance sweep completed.

## Evidence and update rules

- [Accuracy checkpoint60](../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/result.json). Each green cell links its original report and exact canonical case identity.
- [Release checkpoint58 comparison](../../../../parity-results/qwen-flash-next/hc-default58-model-abba-r2/comparison.json): all samples, token equality, resource observations and native retirement. Raw results stay ignored, outside versioned source payloads.
- [Qualification plan](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md), [implementation record](QWEN_FLASH_NEXT_IMPLEMENTATION.md), [ngram tier plan](QFN_NGRAM_TIERED_STORAGE_PLAN.md).
- Canonical topology/pair authority: [Qwen4ExpLocalPairedCases.h](../../../../tests/v2/integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h). This dashboard is a reporting view, not a runner registry.
- Stable procedure: [llama.cpp comparison skill](../../../../.agents/llama-cpp-comparison/SKILL.md).

Keep applicable RAM, SSD mmap, bounded RAM/SSD, replicated VRAM, sharded VRAM and pooled VRAM/RAM/SSD variants visible. VRAM has highest placement priority; ngrams have lowest VRAM admission priority. Test positive-capacity hot-row promotion/eviction/reset. Add a separate variant row before measuring a different storage policy, context, ISA or arithmetic policy; do not combine them in one timing cell.

TP/PP coverage retains even/skewed/legal-empty ownership, reversed device order, singleton/full-cohort boundaries, equal/unequal ranges and actual live communication checks. Unsupported combinations reject admission explicitly; family coverage does not authorize an invalid Cartesian product.

After every complete baseline, accuracy run and tuning experiment, update the row with the exact case ID, model/build revisions, effective configuration, corpus/workload, grants, commands and artifact links. Replace historical `G60`/`R58†` only with identified newer evidence. If a candidate fails accuracy, mark it red immediately rather than keeping an older green badge next to its new speed.

[a1]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/rocm-tp4-allgpu-ram-hc-up-two-term/report.json
[a2]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/rocm-cpu-static8-adaptive-hc-up-two-term/report.json
[a3]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/cuda-cpu-static8-adaptive-hc-up-two-term/report.json
[a4]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/cpu-mmap-hc-up-two-term/report.json
[a5]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/hybrid-cpu-tail-mmap-hc-up-two-term/report.json
