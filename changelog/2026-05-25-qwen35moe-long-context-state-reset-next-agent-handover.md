# Next-Agent Handover: Qwen3.5 MoE Long-Context Cross-Prompt State Reset

Date: 2026-05-25

## Mission

Continue debugging Qwen3.5 MoE CPU-vs-ROCm long-context parity when multiple independent prompt lengths are executed through the same runners.

The remaining bug is request-boundary state surviving `clear_cache()` / prompt reset:

- An isolated length-2 prompt passes.
- A cross-prompt sweep that runs length 1 first, then length 2, fails on the length-2 decode row.
- The most recent evidence localizes the survivor to persistent ROCm KV-cache contents, specifically the second prefill token row in layer 3 as read by decode attention.

Once the remaining state is found and reset, the follow-up design goal is to thread a request/session context object through stages/kernels so request-scoped data is centralized instead of scattered through stages, graph caches, kernel registries, KV cache state, and MoE runtime tables.

## Important Repository / Workflow Context

- Workspace: `/workspaces/llaminar`
- Active build tree: `build_v2_integration`
- Build target:

```bash
cmake --build build_v2_integration --target v2_integration_parity_qwen35moe_long_context --parallel
```

- Main failing test:

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

- The current worktree is intentionally dirty with both candidate fixes and diagnostics. Do not blindly revert without checking intent.
- There is an earlier handover at:
  - `changelog/2026-05-25-qwen35moe-long-context-state-reset-handover.md`
- This document supersedes that handover by preserving its context and adding the latest evidence.

## Current Dirty Files

At this handover, `git status --short` shows:

```text
 M src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp
 M src/v2/execution/compute_stages/stages/AttentionComputeStage.h
 M src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp
 M src/v2/execution/compute_stages/stages/KVCacheAppendStage.h
 M src/v2/execution/compute_stages/stages/MoERoutingStage.h
 M src/v2/execution/compute_stages/stages/RoPEStage.h
 M src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp
 M src/v2/execution/local_execution/engine/ForwardGraphTypes.h
 M src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp
 M src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h
 M src/v2/execution/moe/MoERuntimeTable.cpp
 M src/v2/execution/moe/MoERuntimeTable.h
 M src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp
 M src/v2/models/qwen35moe/Qwen35MoEGraph.cpp
 M src/v2/models/qwen35moe/Qwen35MoEGraph.h
 M src/v2/snapshots/SnapshotCapture.cpp
?? changelog/2026-05-25-qwen35moe-long-context-state-reset-handover.md
?? changelog/2026-05-25-qwen35moe-long-context-state-reset-next-agent-handover.md
```

Diff summary before this document was added was about `553 insertions(+), 19 deletions(-)` across 16 files.

## Original Handover Context Preserved

The original handover had already established:

1. The active patch narrowed the survivor substantially but did not fix the cross-prompt sweep.
2. Isolated length 2 with snapshots passed.
3. Cross-prompt `1,2` with snapshots failed.
4. The first initial cross-only regression was attention output at layer 3, before MoE routing/expert drift.
5. Therefore, the likely surface moved away from MoE runtime placement toward ROCm attention/KV/RoPE state.

It also documented the candidate reset/fix changes below.

## Candidate Reset/Fix Changes Already In Worktree

These compile, but **do not yet solve** the cross-prompt failure.

### 1. `MoERoutingStage` request-local routing reset

File: `src/v2/execution/compute_stages/stages/MoERoutingStage.h`

Added `resetSessionState()` clearing:

- `routing_indices_f32_`
- `routing_weights_`
- `router_logits_`
- `cached_routing_`

### 2. `RoPEStage` stale position-id reset

File: `src/v2/execution/compute_stages/stages/RoPEStage.h`

`updateDynamicParams()` now clears:

- `params_.position_ids`
- `position_ids_cache_`

Purpose: prevent cached RoPE stages from retaining an old position-id array when replaying decode with dynamic `pos_offset`.

### 3. Forward graph signature split for single-token calls

Files:

- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`

Added `ForwardGraphSignature::decode_has_history`, derived from first position ID / `position_offset`, to distinguish:

- empty-cache single-token prompt
- true decode with KV/GDN history

This narrowed the hypothesis but did not solve the failure by itself.

### 4. Full forward graph cache request boundary

Files:

- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

`clear_cache()` / `clearInferenceState()` now drop full forward graph cache entries via `forward_engine_->clearCache()` at prompt boundaries.

### 5. Per-layer decode graph cache request boundary

Files:

- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

`clear_cache()` invalidates `layer_graph_cache_` entries again rather than preserving them with only `resetSessionState()`.

### 6. Kernel dynamic state reset

File: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`

`clear_cache()` still calls `resetKernelDynamicState()`, which maps to `KernelFactory::resetAllDynamicState()`. This resets dynamic pointers/streams on cached generic kernels such as embedding, attention, RoPE, MoE, etc. It does not necessarily clear every ROCm-specific kernel-owned device registry/state.

### 7. Qwen3.5 MoE graph reset hook

Files:

- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.h`

Added `Qwen35MoEGraph::resetState()` override. It calls `Qwen35Graph::resetState()` currently, resets decode histogram/window state, and resets MoE runtime tables.

### 8. MoE runtime table reset

Files:

- `src/v2/execution/moe/MoERuntimeTable.cpp`
- `src/v2/execution/moe/MoERuntimeTable.h`

Added:

- `resetDecodeHistogramCounts()`
- `resetDecodeRuntimeState()`

This resets active decode placement bank metadata, top-k state, and histograms while preserving prefill route scratch pointers/capacity.

This did not change the cross-prompt failure, which is one reason the current next target moved away from MoE.

## Diagnostic Instrumentation Added In This Session

### 1. Raw KV-cache post-append snapshot diagnostics

Files:

- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.h`
- `src/v2/snapshots/SnapshotCapture.cpp`

Added debug snapshot vectors and snapshot-capture handling controlled by:

- `LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT=1`
- `LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT_LAYER=<layer>`

**Important warning:** this raw KV-cache snapshot path caused a segmentation fault when used with:

```bash
LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT=1
LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT_LAYER=3
```

The segfault stack decoded to:

```text
llaminar2::StageDumpInfo::addInput(...)
/workspaces/llaminar/src/v2/execution/compute_stages/ComputeStageBase.cpp:512
llaminar2::AttentionComputeStage::buildDumpInfoImpl() const
/workspaces/llaminar/src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:811
```

Likely reason: calling raw `get_kv()` during dump rebuilt/changed cache tensor view wrappers and invalidated pointers shared with attention dump info.

Recommendation for next agent:

- Do **not** use `LLAMINAR_DEBUG_KV_CACHE_SNAPSHOT` as-is.
- Either remove this crash-prone instrumentation or keep it disabled and replace with a safer copy-after-append debug path that does not expose cache tensor view wrappers.

A harmless related change remains: `KVCacheAppendStage::execute()` calls `invalidateDumpInfoCache()` after successful append so post-exec dumps can rebuild.

### 2. Effective K/V entering attention snapshots

Files:

- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.h`
- `src/v2/snapshots/SnapshotCapture.cpp`

Added safer diagnostics controlled by:

```bash
LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT=1
LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT_LAYER=3
```

These snapshot the exact `effective_K` / `effective_V` tensors already used by attention, avoiding raw cache `get_kv()` side effects.

Shape fix included:

- CPU `get_kv_converted()` shadows may be flat 1-D tensors.
- Diagnostics now compare logical shape `[effective_kv_len, n_kv_heads * head_dim]`.

### 3. Row-wise effective K/V snapshot keys

File: `src/v2/snapshots/SnapshotCapture.cpp`

Added row-wise output keys for small effective K/V tensors:

- `layer3_ATTENTION_EFFECTIVE_K_ROW0`
- `layer3_ATTENTION_EFFECTIVE_K_ROW1`
- `layer3_ATTENTION_EFFECTIVE_K_ROW2`
- `layer3_ATTENTION_EFFECTIVE_V_ROW0`
- `layer3_ATTENTION_EFFECTIVE_V_ROW1`
- `layer3_ATTENTION_EFFECTIVE_V_ROW2`

These were decisive for localizing the bad row.

## Attempted Fix In This Session That Did NOT Solve It

File: `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp`

Changed `ROCmRingKVCache<Precision>::append_typed()` to use the dynamic-head append path only during graph capture:

```cpp
if (capture_active && d_head_params_ && h_head_params_ && stream)
```

instead of:

```cpp
if (d_head_params_ && h_head_params_ && stream)
```

Rationale: normal execution should not need a pinned-host H2D dynamic-head dependency; graph capture still does.

Result: **not fixed**. Latest cross-prompt sweep after this patch still failed:

```text
length=2 decode:
  cosine = 0.974061906
  KL     = 0.235601485
  top5   = 0.6
  failed
```

Recommendation:

- Treat this patch as experimental and not proven useful.
- It can be reverted if it gets in the way.
- The failure mode remained similar, so the bug is probably not just stale dynamic head H2D in normal append.

## Key Test Commands

### Standard cross-prompt repro

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

Typical result: length-2 decode fails with cosine around `0.97-0.98`, KL around `0.15-0.28`.

### Cross-prompt repro with fixed decode FA launch knobs

Used to rule out launch autotuning/split-shape variance:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_ROCM_FA_DECODE_NUM_SPLITS=4 \
LLAMINAR_ROCM_FA_DECODE_TPB=128 \
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=1 \
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=1,2 \
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=1 \
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

Latest result after scalar-head experiment:

```text
length 2 prefill:
  cosine=0.999409914
  KL=0.00323742861
  passed=1

length 2 decode:
  cosine=0.974061906
  KL=0.235601485
  top5_overlap=0.600000024
  passed=0
```

### Effective K/V row-wise snapshot repro

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT=1 \
LLAMINAR_DEBUG_EFFECTIVE_KV_SNAPSHOT_LAYER=3 \
LLAMINAR_ROCM_FA_DECODE_NUM_SPLITS=4 \
LLAMINAR_ROCM_FA_DECODE_TPB=128 \
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=1 \
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=1,2 \
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=1 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOTS=1 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_FAIL=0 \
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOT_STAGES=layer3_ATTENTION_EFFECTIVE_K_ROW,layer3_ATTENTION_EFFECTIVE_V_ROW,layer3_ATTENTION_CONTEXT \
ctest --test-dir build_v2_integration \
  -R "V2_Integration_Parity_Qwen35MoE_LongContext_Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep" \
  --output-on-failure --parallel
```

Parse helper:

```bash
python3 - <<'PY'
import csv, os, re, time
p='tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_snapshot_sweep.csv'
print('exists', os.path.exists(p), p)
if os.path.exists(p): print('mtime', time.ctime(os.path.getmtime(p)))
with open(p, newline='') as f:
    rows=[r for r in csv.DictReader(f) if r.get('length')=='2' and r.get('phase')=='decode']
def sort_key(r):
    s=r['stage_key']; m=re.search(r'_ROW(\d+)', s)
    idx=int(m.group(1)) if m else 999
    group=0 if '_EFFECTIVE_K' in s else 1 if '_EFFECTIVE_V' in s else 2
    return (group, idx, s)
for r in sorted(rows, key=sort_key):
    print(f"{r['stage_key']:55s} rows={r['rows']:>2s} cols={r['cols']:>5s} elems={r['elements']:>6s} mode={r['compare_mode']:>28s} cos={r['cosine']:>12s} rel={r['rel_l2']:>12s} max={r['max_abs_diff']:>12s} pass={r['passed']} miss={r['missing_reason']}")
PY
```

## Most Important Evidence So Far

### 1. Original cross-vs-isolated snapshot evidence

Initial selected snapshot comparison found first state-dependent regression in layer 3 attention output:

```text
layer3_ATTENTION_OUTPUT:
  isolated: cosine=0.999784827 rel_l2=0.0207771193 passed=1
  cross:    cosine=0.920170128 rel_l2=0.396363199  passed=0
  delta:    dcos=-0.079615 drel=+0.375586
```

Subsequent MoE expert/combined/routing drift appears downstream.

### 2. Layer3 full attention/MoE snapshot evidence

A later detailed layer3 snapshot showed everything immediately before attention was good:

```text
layer3_ATTENTION_NORM   cos=0.999989688 rel=0.00453949627 pass=1
layer3_Q_PROJECTION     cos=0.999997914 rel=0.00205422053 pass=1
layer3_K_PROJECTION     cos=0.999940753 rel=0.0108864345  pass=1
layer3_V_PROJECTION     cos=0.999920487 rel=0.0126314498  pass=1
layer3_Q_NORM           cos=0.999950647 rel=0.00993648916 pass=1
layer3_K_NORM           cos=0.999937177 rel=0.0112065123  pass=1
layer3_Q_ROPE           cos=0.999950647 rel=0.0099364901  pass=1
layer3_K_ROPE           cos=0.999937177 rel=0.0112065123  pass=1
layer3_FA_GATE          cos=0.999998927 rel=0.00148953171 pass=1
```

First failures in that view:

```text
layer3_ATTENTION_CONTEXT        cos≈0.9646 rel≈0.268 pass=0
layer3_ATTENTION_CONTEXT_GATED  cos≈0.9062 rel≈0.426 pass=0
layer3_ATTENTION_OUTPUT         cos≈0.9191 rel≈0.399 pass=0
```

Conclusion: current decode Q/K/V path is good; the failure is in cached history K/V content/read, not the current-token projections.

### 3. Effective K/V entering attention evidence

With corrected effective K/V shape:

```text
layer3_ATTENTION_EFFECTIVE_K rows=3 cols=512 cos=0.878852248 rel=0.493832111 pass=0
layer3_ATTENTION_EFFECTIVE_V rows=3 cols=512 cos=0.764179111 rel=0.706713378 pass=0
layer3_ATTENTION_CONTEXT     rows=1 cols=4096 cos=0.96463716  rel=0.268413872 pass=0
```

Conclusion: cache content is already wrong before attention math runs.

### 4. Row-wise effective K/V evidence — decisive localization

For length-2 decode, effective K/V rows are logically:

```text
row0 = prefill token 0
row1 = prefill token 1
row2 = current decode token 0
```

Latest row-wise snapshot (mtime Mon May 25 14:37) showed:

```text
layer3_ATTENTION_EFFECTIVE_K_ROW0 cos=0.999972045 rel=0.00747535005 pass=1
layer3_ATTENTION_EFFECTIVE_K_ROW1 cos=0.634431183 rel=0.863483131   pass=0
layer3_ATTENTION_EFFECTIVE_K_ROW2 cos=0.999910593 rel=0.0133753186  pass=1

layer3_ATTENTION_EFFECTIVE_V_ROW0 cos=0.999962270 rel=0.00869033858 pass=1
layer3_ATTENTION_EFFECTIVE_V_ROW1 cos=0.386197716 rel=1.19487441    pass=0
layer3_ATTENTION_EFFECTIVE_V_ROW2 cos=0.999888718 rel=0.014918006   pass=1

layer3_ATTENTION_CONTEXT          cos=0.964499354 rel=0.269008756   pass=0
```

This is the most important current signal:

> The second prefill-token cache row (`row1`) is corrupt/stale for layer 3 when length 2 follows length 1. Row0 and the decode row2 are good.

Because length-2 prefill logits pass, and prefill attention uses direct `buffers.K/V` when `cached_tokens == 0`, the bad row only shows up later when decode reads from the cache.

## Hypotheses Ruled Out By Test Evidence

### MoE routing cache as primary root

Ruled out / deprioritized because:

- MoE routing state reset was added.
- MoE runtime decode bank/hist reset was added.
- First failure appears before MoE in layer3 attention context/output.

MoE differences downstream are likely consequence, not cause.

### Full forward graph cache survivor

Ruled out / deprioritized because:

- `forward_engine_->clearCache()` was added at request boundary.
- Cross-prompt failure persisted.

### Per-layer decode graph cache survivor

Ruled out / deprioritized because:

- `layer_graph_cache_` invalidation was restored at request boundary.
- Cross-prompt failure persisted.

### Stale RoPE position IDs in cached RoPE stage

Ruled out / deprioritized because:

- `RoPEStage::updateDynamicParams()` now clears `params_.position_ids` and `position_ids_cache_`.
- Layer3 `Q_ROPE` and `K_ROPE` snapshots pass strongly before attention.
- Running with `LLAMINAR_ROPE_ON_READ=0` did not fix the failure.

### Decode FA launch autotuning / split shape

Ruled out / deprioritized because:

- Repro persists with fixed knobs:
  - `LLAMINAR_ROCM_FA_DECODE_NUM_SPLITS=4`
  - `LLAMINAR_ROCM_FA_DECODE_TPB=128`

### ROCm concurrent prefill streams as primary root

Ruled out / deprioritized because:

- Running with `LLAMINAR_ROCM_CONCURRENT_PREFILL=0` still failed.
- Latest no-concurrent-prefill failure:
  - cosine `0.981066108`
  - KL `0.152855337`

### Dynamic-head append path in normal non-capture execution as sole root

Ruled out / deprioritized because:

- Experimental patch to use scalar-head append outside graph capture did not fix it.
- Latest failure after that patch:
  - cosine `0.974061906`
  - KL `0.235601485`

### Native KV / RoPE-on-read as sole root

Not fully ruled out as contributing, but not sufficient:

- `LLAMINAR_ROCM_FA_DISABLE_NATIVE_KV=1` improved cosine in one run but KL still failed.
- `LLAMINAR_ROPE_ON_READ=0` did not fix.

## Active Hypotheses To Pursue

### Hypothesis A: Prefill append source row1 is already wrong when appended

What would explain it:

- Length-2 prefill attention/logits pass because attention consumes direct `buffers.K/V` for prefill, but the tensor passed to `KVCacheAppendStage` row1 may be stale/wrong by the time append executes, or append may read from a stale device pointer/coherence state.
- Row0 and row2 being good makes this weird but still possible if row1 comes from a stale buffer segment after cross-prompt reuse.

How to test:

- Add a safe debug path that snapshots/copies **append input K/V rows** for layer3 length2 prefill, especially row1.
- Compare CPU vs ROCm append-source rows before they enter the cache.
- Avoid raw `get_kv()` view-based snapshots; copy bytes directly from source tensor device memory to host under debug env.

Suggested env name:

```text
LLAMINAR_DEBUG_KV_APPEND_SOURCE_SNAPSHOT=1
LLAMINAR_DEBUG_KV_APPEND_SOURCE_LAYER=3
```

Suggested output keys:

```text
layer3_KV_APPEND_SOURCE_K_ROW0/ROW1
layer3_KV_APPEND_SOURCE_V_ROW0/ROW1
```

Potential issue: current snapshot harness mainly emits decode rows. If it does not capture prefill snapshots for this stage, add temporary logs/checks in the append path instead.

### Hypothesis B: Prefill append writes row1 incorrectly or incompletely

What would explain it:

- Source rows are correct, but the append kernel or conversion scratch path writes wrong data for token index 1.
- Row0 good, row1 bad, row2 good is consistent with a prefill append-token-index-1 problem.

How to test:

- In `ROCmRingKVCache::append_typed()` or the conversion wrapper, under a layer3/num_tokens==2 debug env:
  1. synchronize the append stream after append,
  2. copy source row1 and physical cache row1 to host,
  3. compute/log max diff/checksum/first few values.
- This should not call `get_kv()` or create tensor views.

Likely file:

- `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp`

### Hypothesis C: FP16/Q8 conversion scratch is reused or overwritten before append consumes row1

What would explain it:

- `appendWithStream()` converts non-FP16 K/V into `conv_scratch_k_` / `conv_scratch_v_` then launches append on the same stream.
- If another stream or stage reuses the shared scratch before append consumes row1, row0 might survive and row1 might be overwritten.
- Disabling concurrent prefill did not fix, but decode/prefill graph/cache streams may still interact.

How to test:

- Log whether layer3 append uses conversion scratch and which stream pointer.
- Under debug env, use dedicated temporary scratch for one append or synchronize after conversion before append to see if row1 recovers.
- Check whether `conv_scratch_k_` is shared per cache object across layers/requests and whether multiple appends can overlap on different streams.

### Hypothesis D: Cached prefill graph replay/stage pointers reuse stale row or scratch buffer

What would explain it:

- The length1 prompt warms/captures/reuses graph/cache paths that survive into length2 despite clears.
- The next length2 prefill may hit a cached forward/prefill path with stale tensor binding or replay parameters for one row.

Evidence against total graph-cache root exists, but not every prefill graph replay path is fully ruled out.

How to test:

- Disable prefill graph/cache features if there is an env knob, or force CPU/ROCm length2 prefill path to rebuild without graph cache.
- Inspect `ForwardExecutionEngine::executeCacheHit()` for prefill replay and how it updates stable token/position buffers and `PrefillReplayParams`.
- Inspect `ForwardGraphTypes::ForwardGraphCache::resetSessionState()` and `clearAllReplayState()` logic.

Potential envs to search/try:

```bash
rg "PREFILL_GRAPH|GPU_GRAPHS|LLAMINAR_.*GRAPH" src/v2
```

### Hypothesis E: ROCm cache clear is incomplete for data rows or shadows in a specific path

What would explain it:

- Clear appears strong, but if some cache rows are in sharded/hybrid/rope-shadow paths or a later stale async write occurs after clear, row1 could retain prior-request data.

Evidence:

- `ROCmRingKVCache::clear()` zeros `h_head_params_`, `d_head_params_`, clears RoPE shadows/views, zeros pool with `hipMemsetAsync`, synchronizes, and resets host entries.
- That makes a simple missed clear less likely.

How to test:

- Add debug after `clear()` to copy layer3 row1 and verify it is zero.
- Add debug immediately after length2 prefill append to verify row1 contents.
- Check all streams are synchronized before clear and whether any captured graph/async append from prior prompt can still write after clear.

## Key Code Locations

### Graph construction around KV/attention

File: `src/v2/models/qwen/QwenGraphBase.cpp`

Function: `QwenGraphBase::addKVCacheAndAttention()`

Important behavior:

- Always adds `kv_append` before attention.
- For prefill when `cached_tokens == 0`, attention uses direct `buffers.K/V`, not cache.
- For decode, attention reads from cache via `read_kv_from_cache` / `get_kv_converted` / native KV path.

### KV append stage

Files:

- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.h`

Important behavior:

- Determines `total_tokens` from `params_.num_tokens` or `K->shape()[0]`.
- If not graph capture and `replay_advance_tokens_ > 0`, narrows to real prefix length.
- GPU path calls:

```cpp
params_.kv_cache->appendWithStream(layer, seq_idx, K, V, num_tokens, stream)
```

- `onGraphReplayed()` advances host count/head by `replay_advance_tokens_` or `params_.num_tokens`.

### ROCm append wrapper / conversion path

File: `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp`

Important functions:

- `IROCmRingKVCache::appendWithStream()`
- `ROCmRingKVCache<Precision>::append_typed()`
- `ROCmRingKVCache<Precision>::launch_append_kernel()`
- `ROCmRingKVCache<Precision>::launch_append_kernel_dynamic()`
- `ROCmRingKVCache<Precision>::clear()`
- `ROCmRingKVCache<Precision>::get_kv_converted()`

### ROCm append kernels

File: `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip`

Important kernels:

- `ring_append_kernel`
- `ring_append_kernel_dynamic`

Both use:

```cpp
const int token_idx = blockIdx.x;
const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;
const int dst_pos = (head + token_idx) % max_seq_len;
const int dst_offset = dst_pos * kv_dim + elem_idx;
const int src_offset = token_idx * kv_dim + elem_idx;

d_K_cache[dst_offset] = d_K_new[src_offset];
d_V_cache[dst_offset] = d_V_new[src_offset];
```

The kernel itself looks straightforward. The bug is more likely in inputs, scratch/coherence, stream ordering, or graph replay metadata than the index math.

## Recommended Next Steps

1. **Do not spend more time on MoE first.** Attention/KV has stronger evidence.
2. **Remove or avoid the crash-prone raw KV snapshot path.** Use direct byte-copy diagnostics instead.
3. Add a debug check around layer3 length2 prefill append:
   - source K/V row0 and row1 checksum/first values,
   - cache physical row0 and row1 after append,
   - stream pointer and whether conversion scratch was used.
4. Run the short cross-prompt repro with fixed decode knobs.
5. If source row1 is good but cache row1 is bad, focus on append/conversion/scratch/stream ordering.
6. If source row1 is already bad, focus on tensor binding/coherence/replay before `KVCacheAppendStage`.
7. If row1 is good right after append but bad by decode, focus on a later overwrite after prefill append and before decode attention.

## Handy CSV Parsers

### Sweep rows

```bash
python3 - <<'PY'
import csv, os, time
p='tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_sweep.csv'
print('exists', os.path.exists(p), p)
if os.path.exists(p): print('mtime', time.ctime(os.path.getmtime(p)))
with open(p, newline='') as f:
    for r in csv.DictReader(f):
        if r.get('length') == '2':
            print(r)
PY
```

### Snapshot rows for length2 decode

```bash
python3 - <<'PY'
import csv, os, re, time
p='tests/v2/integration/parity/results/db9ec6ee/Qwen35MoELongContextParityTest_CPUVsROCmLogitsSweep/cpu_vs_rocm_long_context_snapshot_sweep.csv'
print('exists', os.path.exists(p), p)
if os.path.exists(p): print('mtime', time.ctime(os.path.getmtime(p)))
with open(p, newline='') as f:
    rows=[r for r in csv.DictReader(f) if r.get('length')=='2' and r.get('phase')=='decode']
def sort_key(r):
    s=r['stage_key']
    m=re.search(r'_ROW(\d+)', s)
    idx=int(m.group(1)) if m else 999
    group=0 if '_EFFECTIVE_K' in s else 1 if '_EFFECTIVE_V' in s else 2
    return (group, idx, s)
for r in sorted(rows, key=sort_key):
    print(f"{r['stage_key']:55s} rows={r['rows']:>2s} cols={r['cols']:>5s} elems={r['elements']:>6s} mode={r['compare_mode']:>28s} cos={r['cosine']:>12s} rel={r['rel_l2']:>12s} max={r['max_abs_diff']:>12s} pass={r['passed']} miss={r['missing_reason']}")
PY
```

## Bottom Line

The remaining survivor is no longer just “attention somewhere.” It is localized to layer3 effective K/V cache history during length2 decode after a prior length1 request:

- current-token decode K/V row is good,
- first prefill cache row is good,
- second prefill cache row is bad,
- prefill logits still pass because prefill attention does not consume the cache for `cached_tokens == 0`.

The next best proof is to instrument layer3 length2 prefill append source and post-append cache row1 without using raw cache tensor views.
