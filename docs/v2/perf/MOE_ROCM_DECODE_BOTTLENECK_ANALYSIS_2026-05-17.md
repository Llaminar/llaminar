# MoE ROCm Decode Bottleneck Action Items

Date: 2026-05-17

Model: Qwen3.5-35B-A3B UD-Q4_K_XL, ROCm MI60/gfx906, single device.

This file records the MoE ROCm decode optimization sprint that followed the initial bottleneck analysis. The original profile showed default decode around `27 tok/s`, with most GPU time in MoE routing and expert FFN. The current no-env Release benchmark now reaches `41.65 tok/s` decode.

## Final Current State

Command:

```bash
LLAMINAR_LOG_LEVEL=WARN LLAMINAR_GPU_GRAPHS=1 \
  ./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Final no-env benchmark:

| Phase | Throughput |
|-------|------------|
| Prefill | 264.20 tok/s |
| Decode | 41.65 tok/s |
| Overall | 135.77 tok/s |

Focused validation:

```bash
ctest --test-dir build_v2_integration \
  -R '^(V2_Unit_DecodeExpertHistogram|V2_Unit_MoERuntimeTable|V2_Unit_MoERoutingStage|V2_Integration_ROCmMoEKernel)$' \
  --output-on-failure --parallel
```

Result: passed, 5/5 tests.

## Accepted Defaults

These optimizations are enabled by default in Release unless deterministic mode disables the non-deterministic/approximate variants.

| Item | Status | Default | Result |
|------|--------|---------|--------|
| Runtime-table device-routed decode | Done | On | Removes host routing materialization from eligible decode |
| Runtime top-k expert consumption | Done | On | Expert decode reads `DeviceMoELayerRuntime::topk_expert_ids/topk_weights` directly |
| Runtime pointer-array cache | Done | On | Avoids repeated gate/up scratch pointer H2D staging on cache hits |
| Descriptor-table validation cache | Done | On | Validates grouped descriptors once at upload time, not in decode hot methods |
| Fused shared expert gate decode | Done | On | Uses one decode launch for `seq_len == 1`; prefill path unchanged |
| Parallel expert-down decode | Done | On, off under `LLAMINAR_DETERMINISTIC=1` | Moves down projection from serial top-k loop to `(N_tiles, top_k)` grid with atomic accumulation |
| K-partition grouped gate/up decode | Done | On, off under `LLAMINAR_DETERMINISTIC=1` | Uses 8 K partitions by default for better gate/up occupancy |
| Runtime histogram bridge | Done | On | Allows rebalance histograms without falling back to host-routed decode |
| Shared-memory router top-k | Done | On, off under `LLAMINAR_DETERMINISTIC=1` | Replaces slow global-memory runtime top-k with shared-memory wave path for <=256 experts |

## Guarded Experiments

These paths were implemented and tested but are default-off because they were flat or slower on the target benchmark.

| Experiment | Env Flag | Result |
|------------|----------|--------|
| Grouped LDS router logits | `LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER=1` | Correct but flat/slightly slower |
| FP16 cached router gate | `LLAMINAR_ROCM_MOE_ROUTER_FP16=1` | Top-k stable, not faster |
| Q8 cached router gate | `LLAMINAR_ROCM_MOE_ROUTER_Q8=1` | Top-k stable across deterministic cases, not faster |
| K-partition router logits | `LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE=1` | Correct but noise-level improvement |
| Fused K-part router reduce/top-k | `LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE=1` | Reduces K-part router launch count, still flat versus default |

## Dispatch Results

| Step | Subagent Scope | Acceptance Result |
|------|----------------|-------------------|
| 1 | Cache descriptor validation | Accepted; hot all-expert validation loops removed, focused tests passed |
| 2 | Fuse shared expert gate decode | Accepted; decode uses one launch, prefill preserved, focused tests passed |
| 3 | Parallelize expert down | Accepted; decode moved from ~27.2 to ~28.9 tok/s before later changes |
| 4 | K-partition expert gate/up | Accepted; decode moved to ~31.4 tok/s default |
| 5 | FP16 router weights | Implemented but default-off; stable, not faster |
| 6 | K-partition router logits | Implemented but default-off; stable, flat |
| 7 | Q8 router weights | Implemented but default-off; stable, not faster |
| 8 | Profile router + wave top-k | Accepted; profiling showed softmax/top-k, not gate logits, dominated runtime routing |
| 9 | Runtime histogram bridge | Accepted; default rebalance path now uses runtime routing and wave top-k |

## Key Profiling Correction

The original analysis assumed router gate logits were the dominant router cost and described the model as using 64 routed experts. The live Qwen3.5-35B-A3B path uses 256 router experts.

Profiling showed:

| Kernel | Avg Time Before Wave Top-K |
|--------|-----------------------------|
| `rocm_moe_gate_logits_single_token_kernel` | ~10.6 us |
| `rocm_moe_softmax_topk_decode_runtime_kernel` | ~407.7 us |

The real default-path problem was twofold:

1. Runtime routing was bypassed when host rebalance histograms were enabled.
2. The runtime softmax/top-k kernel materialized/scanned global probabilities and selected top-k serially.

The fix was to keep decode histograms in `DeviceMoELayerRuntime`, sync them lazily to host `DecodeExpertHistogram` for rebalance, and enable the shared-memory wave top-k runtime kernel by default.

## Current Residual Profile

With profiling enabled, absolute times are slower than unprofiled benchmark numbers, but the relative decode profile is useful:

| Stage | Profiled Decode Per Token | Share |
|-------|---------------------------|-------|
| MOE_ROUTER | ~8.94 ms | 27.8% |
| MOE_EXPERT_FFN | ~4.32 ms | 13.4% |
| MOE_SHARED_EXPERT_FFN | ~2.74 ms | 8.5% |
| MOE_SHARED_EXPERT_GATE | ~1.17 ms | 3.6% |

The remaining meaningful work is no longer host routing materialization or descriptor/pointer overhead. It is mostly residual router top-k cost under profiling and shared/expert FFN kernel work.

## Follow-Up Ideas

These are not required for the current parity target but remain useful future work.

| Follow-Up | Notes |
|-----------|-------|
| Store gate/up scratch pointers in `DeviceMoELayerRuntime` | Cleaner graph capture than pointer-array cache |
| Deterministic parallel down reduction | Replace atomic path when `LLAMINAR_DETERMINISTIC=1` needs speed too |
| Fuse SwiGLU into down projection | Larger rewrite, may remove one launch and intermediate bandwidth |
| Shared expert FFN tuning | Now comparable to routed expert FFN in residual profile |
| Re-run llama.cpp same-host benchmark | No local llama.cpp binary was present during this sprint; repo historical same-model baseline was ~27.8 tok/s |

## Outcome

The current no-env decode benchmark is `41.65 tok/s`, up from the pre-sprint `~27 tok/s` range and above the historical same-model baseline recorded in `benchmark_results/*` (`~27.8 tok/s`). The practical parity target for this workspace is met pending a fresh same-host llama.cpp run.