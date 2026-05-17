# MoE ROCm Decode Bottleneck Analysis

Date: 2026-05-17

Baseline: Qwen3.5-35B-A3B UD-Q4_K_XL, ROCm MI60, single device, grouped decode + runtime-table device-routed routing enabled.

This document reflects the current post-runtime-table implementation state:

- `decodeRouteSelect()` writes decode top-k ids/weights into `DeviceMoELayerRuntime`.
- `MoEExpertComputeStage::executeSingleToken()` uses the runtime-table grouped decode path when eligible.
- ROCm grouped expert decode consumes `runtime->topk_expert_ids` and `runtime->topk_weights` directly.
- Runtime decode pointer arrays are cached by descriptor table/top-k/device pointer values, avoiding repeated per-token H2D pointer uploads on cache hits.
- The experimental grouped decode router is present but default-off behind `LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER=1` because it benchmarked flat/slightly slower.

## Current Performance

Current unprofiled benchmark (`benchmark_final_default.log`, 128 decode tokens):

| Phase | Throughput |
|-------|------------|
| Prefill | 268.47 tok/s |
| Decode | 27.18 tok/s |
| Overall | 104.39 tok/s |

Optimization progression during the runtime-table cleanup:

| State | Prefill | Decode | Overall |
|-------|---------|--------|---------|
| Restored device-routed decode | 268.37 tok/s | 26.12 tok/s | 101.59 tok/s |
| Runtime top-k expert decode | 268.40 tok/s | 26.96 tok/s | 103.82 tok/s |
| Runtime pointer-array cache | 268.74 tok/s | 27.25 tok/s | 104.62 tok/s |
| Final default run | 268.47 tok/s | 27.18 tok/s | 104.39 tok/s |

Current profiled decode (`profile_final_current.log`, 384 decode steps):

| Stage | % of executor decode time | Total (ms) | Per-token (ms) | Per-MoE-layer (ms) |
|-------|---------------------------|------------|----------------|--------------------|
| MOE_ROUTER | 37.1% | 6515.88 | 16.97 | 0.424 |
| MOE_EXPERT_FFN | 19.6% | 3451.98 | 8.99 | 0.225 |
| MOE_SHARED_EXPERT_FFN | 6.6% | 1166.48 | 3.04 | 0.076 |
| MOE_SHARED_EXPERT_GATE | 2.6% | 454.98 | 1.19 | 0.030 |
| **Total MoE subset above** | **65.9%** | **11589.32** | **30.18** | |

GPU decode timeline from the same profile:

| Metric | Value |
|--------|-------|
| GPU avg | 46.75 ms/token |
| Wall avg | 49.40 ms/token |
| GPU-limited | 21.4 tok/s |
| Kernel efficiency | 99.8% |
| Executor overhead | 0.026 ms/token |

The key point: the remaining gap is kernel work, not framework overhead or coherence overhead.

Model dimensions (Qwen3.5-35B-A3B):

| Parameter | Value |
|-----------|-------|
| d_model | 2048 |
| num_experts | 64 |
| top_k | 6 |
| expert_intermediate | 2560 |
| MoE layers | 40 |

## Completed Since The Previous Analysis

### Runtime Route Selection

`decodeRouteSelect()` now uses a decode-specific path:

1. `hipMoE_gate_logits_single_token(...)`
2. `hipMoE_softmax_topk_decode_runtime(...)`

The old extra runtime-copy launch (`hipMoE_decode_route_select_runtime`) is no longer on the decode path. Softmax/top-k writes `DeviceMoELayerRuntime::topk_expert_ids`, `topk_weights`, optional legacy routing tensors, and the decode histogram in one launch.

### Runtime Top-K Expert Decode

The ROCm grouped expert fast path now calls:

- `groupedExpertGateUpDecodeFromRuntime(...)`
- `groupedExpertDownDecodeFromRuntime(...)`

These consume `DeviceMoELayerRuntime::topk_expert_ids` and `topk_weights` directly. The active runtime fast path no longer reads legacy FP32 routing tensors and no longer launches `hipMoE_float_to_int` for the expert stage.

The legacy routing variants still exist for fallback/testing and still do FP32 routing-index conversion; they are not the default eligible runtime path.

### Runtime Pointer-Array Cache

The runtime grouped expert path previously uploaded tiny gate/up scratch pointer arrays every decode call. It now caches device-side pointer arrays keyed by:

- descriptor table id
- `top_k`
- exact gate/up device pointer values

This removes repeated H2D pointer-array staging on cache hits. The cache is conservative and layer-safe because decode interleaves many MoE layers and the key includes the actual scratch tensor device pointers.

This is not as clean as storing scratch pointers inside `DeviceMoELayerRuntime`, but it removes most repeated pointer-copy overhead while preserving the existing kernel ABI.

### Grouped Router Experiment

An experimental grouped decode-router logits kernel was added behind:

```bash
LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER=1
```

It loads hidden state into LDS and computes multiple experts per CTA. It passed the focused ROCm MoE tests but benchmarked flat/slightly slower:

| Router mode | Prefill | Decode | Overall |
|-------------|---------|--------|---------|
| Default one-expert-per-CTA router | 268.41 tok/s | 27.31 tok/s | 104.73 tok/s |
| Opt-in grouped router | 269.66 tok/s | 27.24 tok/s | 104.71 tok/s |

It remains default-off.

## Problem 1: Router Gate Logits Still Dominate

**Current impact: 37.1% of profiled decode executor time, 16.97 ms/token.**

### What Runs

Two sequential launches per MoE layer:

1. `rocm_moe_gate_logits_single_token_kernel` - FP32 GEMV: `[1, 2048] x [64, 2048]^T -> [64]`
2. `rocm_moe_softmax_topk_decode_runtime_kernel` - softmax + top-k + runtime write

The second launch is now doing useful runtime-table work. The dominant cost is still the first launch: FP32 router gate logits.

### Gate Logits Kernel Analysis

```text
Data read per layer:  64 x 2048 x 4 bytes = 512 KB (FP32 gate weights)
MI60 peak bandwidth:  ~900 GB/s
Theoretical minimum:  512 KB / 900 GB/s ~= 0.6 us
Actual:               ~424 us per MoE layer
Efficiency:           ~0.14%
```

Launch configuration: `dim3(64), dim3(128)` - one CTA per expert, 128 threads per CTA.

Occupancy problems:

- 64 blocks on 60 CUs is only ~1.07 blocks/CU.
- 128 threads is 2 wavefronts/block.
- MI60 supports far more resident wavefronts/CU, so this shape leaves most latency-hiding capacity unused.
- Every expert CTA rereads the same 2048-float hidden vector.

The grouped-router experiment reduced hidden rereads but did not improve throughput at the tested shape, likely because fewer CTAs further reduced parallelism and LDS traffic did not compensate.

### Recommended Router Work

**R1.1: FP16 or quantized router gate weights.**

The router gate matrix is tiny by model standards (`64 x 2048`) but hot on every MoE layer and currently FP32. FP16 would halve the weight bandwidth. Q8-style storage would reduce it further, but FP16 is the lower-risk first step because top-k routing should tolerate FP16 logits well.

**R1.2: Fused dot + top-k kernel, not just fused runtime copy.**

The already-completed work fused softmax/top-k with runtime writes. The remaining fusion opportunity is bigger: avoid materializing all logits to global memory by computing logits, reduction/top-k state, and runtime writes inside a single decode-router kernel.

The tricky part is preserving enough parallelism. A one-block router would underuse the GPU. A useful design likely needs several CTAs cooperating over expert and/or K partitions, then a small reduction/top-k finalization step.

**R1.3: K-partitioned router logits.**

Partition K across multiple CTAs per expert or across expert groups. This increases block count and occupancy while reducing each thread's serial loop length. It costs a small reduction over partial logits, but the current kernel is so under-occupied that this may be worthwhile.

**R1.4: Keep the grouped-router experiment default-off until tuned.**

The current `LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER=1` path is correctness-safe but not faster. Treat it as a scaffold for future tuning, not a performance win.

## Problem 2: Expert FFN Improved, But Still Has Low-Occupancy Kernels

**Current impact: 19.6% of profiled decode executor time, 8.99 ms/token.**

The runtime-topk and pointer-cache changes moved this down from roughly 10.5 ms/token to roughly 9.0 ms/token in profiled decode. The remaining cost is the actual grouped native-VNNI work.

### What Runs Now

The runtime-table grouped decode path still has four logical launches per MoE layer:

| # | Kernel | Grid | Purpose |
|---|--------|------|---------|
| 1 | `moe_grouped_hidden_quantize_blockwise_kernel` | `(64)` | Quantize hidden to INT8 blockwise |
| 2 | `moe_grouped_native_vnni_gate_up_kernel_t` | `(40, 6)` | Gate+up GEMV for top-k experts |
| 3 | `moe_grouped_swiglu_quantize_blockwise_kernel` | `(6)` | SwiGLU activation + quantize for down |
| 4 | `moe_grouped_native_vnni_down_kernel_t` | `(32)` | Down GEMV + weighted accumulation |

Inputs are now runtime-table top-k ids/weights, not legacy FP32 routing tensors.

### Gate/Up Kernel Analysis

Grid: `((2560 + 63) / 64, 6)` = `(40, 6)` = 240 blocks x 64 threads.

Each thread computes a full dot product over `K / 32 = 64` quantization blocks and computes both gate and up outputs. That is useful data reuse, but it doubles per-thread serial work and register pressure.

Occupancy is still modest: 240 one-wave blocks on 60 CUs gives about 4 waves/CU before resource limits, well below what MI60 can hide.

### Down Kernel Analysis

Grid: `((2048 + 63) / 64)` = 32 blocks x 64 threads.

Each thread serializes over all six active experts. Per expert it loops over `K / 32 = 80` quantization blocks, so each output column thread does roughly 480 block decode+sdot4 groups.

This is still the weakest expert-kernel shape:

- Only 32 CTAs for the whole down projection.
- Top-k experts are serialized inside each thread.
- Most CUs are idle during the down phase.

### Already Fixed In This Area

- Runtime top-k ids/weights are consumed directly.
- The fast path avoids `hipMoE_float_to_int`.
- Runtime pointer arrays are cached and not repeatedly H2D-copied on cache hits.

### Remaining Expert FFN Work

**R2.1: K-partition gate/up.**

Use a grid like `(N_tiles, K_partitions, top_k)` and reduce K partitions. This mirrors the scatter+reduce pattern used by the regular native-VNNI GEMV path and should improve occupancy.

**R2.2: Parallelize down across experts.**

Current down kernel serializes top-k experts inside each output thread. A parallel version would grid over `(N_tiles, top_k)` and accumulate into output with `atomicAdd` or a deterministic reduction buffer.

```text
Current:  32 blocks, each thread loops over 6 experts x 80 K-blocks
Proposed: 32 x 6 blocks, each block handles one expert x 80 K-blocks
```

This increases block count roughly 6x and removes the deepest per-thread serial loop.

For parity/deterministic modes, keep the current serial accumulation path available.

**R2.3: Fuse SwiGLU into down.**

Current path materializes gate/up FP32, launches a SwiGLU+quantize kernel, then launches down GEMV. Fusing the activation/quantization into the down path could remove a launch and reduce intermediate bandwidth, but it is a larger kernel rewrite.

**R2.4: Move scratch pointer arrays into `DeviceMoELayerRuntime`.**

The pointer cache removes repeated H2D copies, but graph capture would be cleaner if the runtime table owned stable gate/up scratch pointers. Then runtime expert kernels would accept only the runtime pointer and descriptor table id.

## Problem 3: Shared Expert Gate Still Uses Two Kernels

**Current impact: 2.6% of profiled decode executor time, 1.19 ms/token.**

`hipMoE_shared_expert_gate` still launches:

1. `rocm_moe_shared_expert_gate_dot_kernel` - dot product + sigmoid into scratch
2. `rocm_moe_shared_expert_gate_scale_kernel` - scale shared expert output

For decode (`seq_len=1`), this is a single 2048-float dot product plus a 2048-float scale. A fused decode-only kernel could compute the scalar gate and scale output in one launch.

Expected impact is modest but low-risk: roughly 0.3-0.5 ms/token if the second launch and scratch round-trip are avoided.

## Problem 4: Descriptor Validation Still Runs In Hot Runtime Methods

The runtime grouped expert methods still validate descriptor tables on every call, for example iterating over all expert descriptors before launching gate/up or down.

That validation should be moved to descriptor-table upload time and cached as a table-valid bit. The hot path should only check table id, shape, codebook, and cached-valid status.

Expected impact is smaller than the router/expert kernel work, but it is low-risk and makes the runtime path more graph-capture friendly.

## Priority-Ordered Roadmap From Current State

| # | Optimization | Target | Status | Expected Savings | Effort |
|---|--------------|--------|--------|------------------|--------|
| 1 | FP16 or quantized router gate weights | MOE_ROUTER | Not started | 4-8 ms/token if bandwidth-bound | Medium |
| 2 | K-partition/fused router logits + top-k | MOE_ROUTER | Prototype grouped router exists but default-off | 4-8 ms/token | Medium/High |
| 3 | Parallelize down across active experts | MOE_EXPERT_FFN | Not started | 1.5-3 ms/token | Medium |
| 4 | K-partition grouped gate/up | MOE_EXPERT_FFN | Not started | 1-2 ms/token | Medium |
| 5 | Fuse shared expert gate decode kernels | MOE_SHARED_EXPERT_GATE | Not started | 0.3-0.5 ms/token | Low |
| 6 | Cache descriptor validation at upload time | MOE_EXPERT_FFN/runtime overhead | Not started | Small, graph-capture hygiene | Low |
| 7 | Move scratch pointers into runtime table | Graph capture / MOE_EXPERT_FFN | Pointer cache exists | Small runtime win, cleaner capture | Medium |
| 8 | Fuse SwiGLU into down projection | MOE_EXPERT_FFN | Not started | 0.5-1 ms/token | High |

Completed items that should no longer be counted as future savings:

| Completed item | Effect |
|----------------|--------|
| Device-routed runtime decode path restored | Decode recovered from disabled-path baseline |
| Runtime top-k expert consumption | Removed runtime fast-path routing tensor reads and `hipMoE_float_to_int` launches |
| Runtime pointer-array cache | Removed repeated pointer-array H2D copies on cache hits |
| Softmax/top-k + runtime write fusion | Removed the extra decode runtime-copy kernel |

Projected decode after the remaining medium-risk work is roughly 32-38 ms/token GPU time, depending on how much router bandwidth and down-kernel occupancy improve. The current profile is 46.75 ms/token GPU time and the current unprofiled benchmark is about 27 tok/s.

The important shift from the previous analysis: the remaining gap is no longer host routing materialization or pointer staging. It is now mostly two kernel families: router gate logits and low-occupancy grouped expert projections.