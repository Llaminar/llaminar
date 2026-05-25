# Handover: Qwen3.5 MoE Long-Context Cross-Prompt State Reset

Date: 2026-05-25

## Goal

Continue debugging Qwen3.5 MoE CPU-vs-ROCm long-context parity when multiple independent prompt lengths are executed through the same runners. The current failure is specifically request-boundary state surviving `clear_cache()`: an isolated length-2 prompt passes, but running length 1 first and then length 2 causes the length-2 decode row to fail.

Once the remaining state is found and reset, the follow-up design idea is to thread a request/session context object through stages/kernels so request-scoped data is centralized instead of scattered through stages, graph caches, kernel registries, KV cache state, and MoE runtime tables.

## Current Status

The active patch has narrowed the surviving state substantially, but the cross-prompt sweep still fails. The most useful signal is from selected snapshot comparison:

- Isolated length 2 with snapshots passes.
- Cross-prompt `1,2` with snapshots fails.
- The first cross-only regression is `layer3_ATTENTION_OUTPUT`, before MoE routing/expert drift.

This means the next likely surface is ROCm attention/KV/RoPE state, not MoE runtime placement.

## Files Changed So Far

Current dirty files:

- `src/v2/execution/compute_stages/stages/MoERoutingStage.h`
- `src/v2/execution/compute_stages/stages/RoPEStage.h`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
- `src/v2/execution/moe/MoERuntimeTable.cpp`
- `src/v2/execution/moe/MoERuntimeTable.h`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.h`

Patch size at handoff: about 146 insertions, 14 deletions across 10 files.

## State Found And Reset So Far

These changes are currently in the worktree. They compile, but they do not yet fix the cross-prompt failure.

1. `MoERoutingStage` request-local routing data

   File: `src/v2/execution/compute_stages/stages/MoERoutingStage.h`

   Added `resetSessionState()` to clear:

   - `routing_indices_f32_`
   - `routing_weights_`
   - `router_logits_`
   - `cached_routing_`

2. `RoPEStage` stale per-token position IDs

   File: `src/v2/execution/compute_stages/stages/RoPEStage.h`

   `updateDynamicParams()` now clears:

   - `params_.position_ids`
   - `position_ids_cache_`

   This prevents a cached RoPE stage from keeping an old position-id array when replaying decode with a dynamic `pos_offset`.

3. Forward graph signature split for one-token calls

   Files:

   - `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
   - `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`

   Added `ForwardGraphSignature::decode_has_history`, derived from the first position ID or `position_offset`. This distinguishes an empty-cache single-token prompt from true decode with history. This narrowed the hypothesis but did not solve the failure by itself.

4. Full forward graph cache request boundary

   Files:

   - `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
   - `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

   `clear_cache()` / `clearInferenceState()` now drop full forward graph cache entries via `forward_engine_->clearCache()` at prompt boundaries.

5. Per-layer decode graph cache request boundary

   Files:

   - `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
   - `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

   `clear_cache()` invalidates `layer_graph_cache_` entries again, rather than preserving them with only `resetSessionState()`. This removed another suspected survivor but did not fix the failure.

6. Kernel dynamic state reset

   File: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`

   `clear_cache()` still calls `resetKernelDynamicState()`, which maps to `KernelFactory::resetAllDynamicState()`. That resets dynamic pointers/streams on cached generic kernels such as embedding, attention, RoPE, MoE, etc. It does not clear all device kernel registries or all ROCm-specific kernel-owned state.

7. Qwen3.5 MoE graph reset hook

   Files:

   - `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
   - `src/v2/models/qwen35moe/Qwen35MoEGraph.h`

   Added `Qwen35MoEGraph::resetState()` override. It currently calls `Qwen35Graph::resetState()` (no-op today), resets the decode histogram window, and resets MoE runtime tables.

8. MoE runtime table reset

   Files:

   - `src/v2/execution/moe/MoERuntimeTable.cpp`
   - `src/v2/execution/moe/MoERuntimeTable.h`

   Added:

   - `resetDecodeHistogramCounts()` to zero host/device decode histograms.
   - `resetDecodeRuntimeState()` to reset active decode placement bank metadata, top-k state, and histograms while preserving prefill route scratch pointers/capacity.

   This did not change the cross-prompt failure, which is one reason the current next target moved away from MoE.

## Test Commands

Build target:

```bash
cmake --build build_v2_integration --target v2_integration_parity_qwen35moe_long_context --parallel
```

Minimal failing cross-prompt sweep:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=1 \
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=1,2 \
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=1 \
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

Same sweep with selected decode snapshots:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=1 \
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=1,2 \
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=1 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOTS=1 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_FAIL=0 \
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

Isolated length-2 control with selected snapshots:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=1 \
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=2 \
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=1 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOTS=1 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_FAIL=0 \
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

Snapshot filters default to:

- `GDN_DELTA_RULE_OUTPUT`
- `GDN_NORM_GATE_OUTPUT`
- `ATTENTION_OUTPUT`
- `MOE_ROUTING_INDICES`
- `MOE_ROUTING_WEIGHTS`
- `MOE_EXPERT_OUTPUT`
- `MOE_COMBINED_OUTPUT`
- `FINAL_NORM`
- `LM_HEAD`

Override with `LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_STAGES=a,b,c`.

## Latest Results

### Saved failing cross-prompt `1,2` run

Saved artifacts:

- `/tmp/llaminar_qwen35_debug/cross_1_2_logits.csv`
- `/tmp/llaminar_qwen35_debug/cross_1_2_snapshots.csv`

Key rows:

- length 1 prefill: cosine `0.999688327`, KL `0.00229971809`, passed.
- length 1 decode: cosine `0.999823332`, KL `0.00036218774`, passed.
- length 2 prefill: cosine `0.999712467`, KL `0.00336686429`, passed.
- length 2 decode: cosine `0.980509102`, KL `0.161159873`, top5 overlap `0.600000024`, failed.

### Latest isolated length-2 snapshot control

Current result path:

- `tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_sweep.csv`
- `tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_snapshot_sweep.csv`

Key rows:

- length 2 prefill: cosine `0.999371231`, KL `0.00404753257`, passed.
- length 2 decode: cosine `0.999621272`, KL `0.00191987271`, top5 overlap `0.800000012`, passed.

### Cross-vs-isolated snapshot diff

First state-dependent regression in execution order:

```text
layer3_ATTENTION_OUTPUT:
  isolated: cosine=0.999784827 rel_l2=0.0207771193 passed=1
  cross:    cosine=0.920170128 rel_l2=0.396363199  passed=0
  delta:    dcos=-0.079615 drel=+0.375586
```

Subsequent drift appears in MoE expert/combined outputs and routing, but that is downstream of the first bad attention output.

## Important Observations

- `length=2` by itself passes strongly, including snapshots.
- `length=1,2` fails only on the second prompt's decode.
- Full forward graph cache invalidation did not fix it.
- Per-layer decode graph invalidation did not fix it.
- MoE routing-stage cache reset did not fix it.
- MoE runtime active-bank reset did not fix it.
- The first cross-only snapshot regression is attention output at layer 3, so the most likely remaining survivor is in attention/KV/RoPE state.

## Next Investigation Target

Start with ROCm hybrid/ring KV cache and attention dynamic state:

- `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.*`
- `src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.*`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.*`
- `src/v2/models/qwen/QwenGraphBase.cpp` around `addKVCacheAndAttention()`

Questions to answer:

- Does `IKVCache::clear()` reset all ROCm GPU-side token counters/head pointers, not just host-side counters?
- Do captured append kernels or attention kernels retain a device pointer or scalar for previous cached-token count?
- Does `KVCacheAppendStage::resetSessionState()` or `updateDynamicParams()` fully reset append position state on ROCm?
- Does `AttentionComputeStage::updateDynamicParams()` correctly update the effective `kv_len` from the freshly cleared/refilled cache, and does the ROCm attention kernel consume that updated value?
- Does `RoPEStage` or the RoPE-on-read attention path keep a stale position buffer after request reset despite the current `position_ids_cache_` clear?

Useful quick checks:

- Compare `ROCmRingKVCache::clear()` against CPU/CUDA hybrid cache clear behavior.
- Add temporary debug logging in ROCm cache `clear()`, `append()`, and `get_cached_tokens()` for layers 0-4 across the `1,2` sweep.
- Specifically trace layer 3: prefill append count, decode attention effective `kv_len`, and any GPU-side append offset/cached-token value.
- Consider enabling transfer/coherence traces only around the ROCm length-2 decode if stale host/device cache metadata is suspected.

## Suggested Continue Commands

Rebuild after small changes:

```bash
cmake --build build_v2_integration --target v2_integration_parity_qwen35moe_long_context --parallel
```

Run the fast proof loop:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=1 \
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=1,2 \
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=1 \
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

Run the localizer only when needed:

```bash
mkdir -p /tmp/llaminar_qwen35_debug
cp tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_sweep.csv /tmp/llaminar_qwen35_debug/cross_1_2_logits.csv
cp tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_snapshot_sweep.csv /tmp/llaminar_qwen35_debug/cross_1_2_snapshots.csv
```

Then run the isolated length-2 snapshot control and diff the CSVs.

## Caveats

- The current worktree includes experimental state-reset changes that compile but are not a complete fix.
- The cross-prompt failure can move numerically between runs, but the stable pattern is: isolated length 2 passes, cross `1,2` fails, and the first material cross-only regression is in early attention output.
- Do not spend more time on MoE placement/runtime banks until attention/KV/RoPE has been ruled out; the current evidence points upstream of MoE.
