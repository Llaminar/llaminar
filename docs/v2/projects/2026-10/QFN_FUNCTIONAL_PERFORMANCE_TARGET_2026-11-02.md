# QFN functional and performance target — 2026-11-02

This is the current user-selected hardware campaign. It supersedes the broader
aspirational tuning rows as the immediate functional/performance target; it does
not remove existing regressions or claim unsupported combinations now work.
The existing typed Qwen4Exp model/topology definitions remain the configuration
authority. This document explains coverage; it is not another test inventory.

## Six hardware rows

| Row | Continuation GPUs | GPU follower experts |
|---|---|---|
| Q1 | 1 × MI50 | None |
| Q2 | 2 × MI50, homogeneous domain | None |
| Q3 | 4 × MI50, homogeneous domain | None |
| Q4 | 1 × RTX 3090 | None |
| Q5 | 2 × RTX 3090, homogeneous domain | None |
| Q6 | 2 × RTX 3090, continuation domain | All 4 MI50s in a real expert follower domain |

Any CPU expert backing needed to fit the original GGUF is explicit in the cell
identity and kept fixed in matched comparisons. A RAM/SSD source service is not
CPU expert execution. Q6 must prove expert work on every declared follower;
ngram service alone cannot count as that work. Test ngram owners on continuation
GPUs, follower GPUs, and both in Q6. Pure rows use the declared compute GPUs as
ngram owners. Dedicated ngram-only GPUs remain diagnostic controls.

## Storage and movement coverage

Distinguish full-table RAM backing from SSD mmap backing plus a bounded RAM row
cache. Cover each backing where admitted, cache-disabled controls, bounded RAM
rows, bounded VRAM rows, and VRAM+RAM+SSD hierarchy. Ngrams retain lowest-priority
VRAM admission. Owner quotas are sharded, not silently replicated. All physical
owners remain declared when their optional cache capacity is zero.

Functional small-cache controls deliberately force promotion, eviction/demotion,
working-set overflow, duplicate demand, skewed ownership, empty owners and
large-to-small/reset replay. C0/C64 denote slots/rows, not MiB and not a tuned
production performance budget. Require actual movement and byte/hit/occupancy
receipts, not merely an enabled policy. For SSD, distinguish cold reads from OS
page-cache hits and measure page-cache churn and read-ahead effects.

Performance uses useful PMA-admitted cache sizes and reports exact tier bytes,
resident working set, hits, misses, promotion/demotion and I/O. Keep available
memory, CPU worker budget, expert quotas, source file, token histories and
arithmetic matched when comparing transport or placement. Include warm-cache
steady state and a bounded churn workload. Do not infer useful production cache
behavior from tiny pressure fixtures alone.

## Functional and numerical projections

Keep MTP Off, fixed1, fixed3 and Dynamic, plus static controls and dynamic expert
balancing. Across the campaign, every hardware row exercises every required MTP
mode and real rebalancing; representative storage pairings cover every tier and
mixed-owner relation. Cheap declaration/admission/state-machine checks may
expand the full product. Expensive model runs use the existing typed projections
and selected complementary pairings instead of an indiscriminate Cartesian run.
A mode is qualified only after its selected real execution and numerical checks.

Each qualified cell needs real source admission, retained native execution,
correct original rows/results, serial/grouped verifier equivalence, repeated
request/reset/prefix behavior, bounded resources and complete process/driver
retirement. Use canonical frozen-history PyTorch panels with separate prefill
and decode mean KL, top-1 and top-5; preserve exact-history layer/stage replay.
Source-only admission, metadata passes and earlier arithmetic badges are not
substitutes. Missing hardware or implementation stays explicitly unqualified.

## Performance projections and transport

Start with NCCL for homogeneous CUDA transfers, RCCL for homogeneous ROCm, and
the existing optimized cross-vendor transport between them. This host has no
direct P2P; native libraries choose their supported physical route. Measure
alternatives on the same endpoint pair, payload, stream/capture policy and
workload before selecting an optimization. Native transport floors with fixed
live sizes do not certify variable-count ngram traffic.

For every hardware row, retain Release prefill/decode and complete-request
throughput alongside completed lookup latency (p50/p95/p99), sustained useful
payload rate, live wire bytes and cache movement costs. Pair each performance
configuration with its accuracy cell. Report normal default behavior and matched
MTP-off controls; observe dynamic acceptance/depth and expert movement rather
than assuming equal speculative work. Compare Castagna on supported ROCm rows,
upstream llama.cpp on CUDA, and labeled Strata arithmetic where supported. A
mixed-vendor external reference remains unavailable until actually reproduced;
use matched internal controls without inventing a baseline.

## Superseded dashboard view

The following records the earlier hardware list and measurements before the
six-row scope was set. Its green badges and timings apply only to those original
policies and revisions; they do not qualify the new storage/owner combinations.

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
| C2a | ROCm2 + CPU1 experts; ConsumerLocal C64 | 🟢 four-panel slice | 111.211 | — | 14.611 | — | 8 GPU experts/layer; reference pending. |
| C3a | CUDA2 + CPU1 experts; ConsumerLocal C64 | 🟢 four-panel slice | 116.941 | — | 17.988 | — | Same source/policy; reference pending. |
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
