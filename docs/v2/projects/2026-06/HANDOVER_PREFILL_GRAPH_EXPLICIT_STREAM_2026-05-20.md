# Handover: Prefill Graph Explicit Stream Capture

Date: 2026-05-20
Workspace: `/workspaces/llaminar`
Branch/HEAD: `feat/qwen35-moe` at `ac6a32c5`
Primary topic: prefill GPU graph capture stream ownership and replay lifecycle

This handover covers the work from the request:

```text
Build in capture stream management for prefill just like decode, and always capture on an explicit stream.
```

It also records the remaining Qwen3.5 35B ROCm server-suite corruption issue that was observed while validating this change.

## Current Status

The prefill explicit-stream work is implemented and locally verified.

Key behavior now in place:

- Prefill graph capture uses a dedicated explicit stream owned by `ForwardGraphCache`.
- Cold prefill warmup, capture, and replay all bind the same explicit stream to cached stages.
- `PrefillGraphCache::beginCapture()` rejects null streams and calls `createGraphCapture(stream)`.
- The capture request now launches the executable graph immediately after `endCaptureAndInstantiate()`, because HIP/CUDA stream capture records work but does not execute it.
- Session `clear_cache()` preserves prefill graph entries so a warmed/captured prefill graph can amortize across independent requests of the same shape.
- Full forward-cache invalidation still drops prefill graph entries and destroys the owned stream.

Focused unit tests pass, the focused `llaminar2` target builds, and a manual ROCm server smoke confirmed warmup -> capture -> replay for the same prefill shape.

## Files Changed By This Work

### Prefill graph stream management

```text
src/v2/execution/local_execution/engine/ForwardGraphTypes.h
src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp
src/v2/execution/local_execution/engine/PrefillGraphCache.cpp
src/v2/execution/local_execution/engine/PrefillGraphCache.h
```

Important changes:

- Added `CachedGraphStream`, a move-only RAII owner for worker GPU streams.
- Added `ForwardGraphCache::prefill_capture_stream`.
- Added `ensurePrefillCaptureStream()` and stage stream binding in `executePrefillWithGraphCache()`.
- Added `GraphCaptureGuard` around prefill capture execution to avoid timeline/coherence event capture hazards.
- Added immediate graph launch after capture/instantiate.
- Kept prefill graph cache entries alive across `resetReplayState()`.
- Kept full `ForwardGraphCache::invalidate()` responsible for invalidating prefill entries and destroying the stream.

### Session replay reset

```text
src/v2/execution/local_execution/engine/ForwardExecutionEngine.h
src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h
```

Important changes:

- Added `ForwardExecutionEngine::resetSessionReplayState()`.
- `DeviceGraphOrchestrator::clear_cache()` now preserves cached `ComputeGraph` objects but resets captured decode graph replay state before clearing model recurrent/KV state.
- Prefill graph entries are preserved during this reset; decode segmented captures are reset because they encode a request-specific replay lifecycle.

### ROCm MoE dynamic-state reset support

```text
src/v2/kernels/KernelFactory.cpp
src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp
src/v2/kernels/rocm/moe/ROCmMoEKernel.h
```

Important changes:

- `KernelFactory::resetAllDynamicState()` now also visits `moe_cache_`.
- `ROCmMoEKernel::resetDynamicState()` clears request-scoped host grouping metadata and resets histogram counts.
- ROCm MoE reset intentionally preserves device scratch, runtime pointer arrays, descriptor tables, and router weight conversion caches because captured prefill graphs can reference those stable device addresses.

### Tests

```text
tests/v2/unit/stages/Test__PrefillGraphCache.cpp
```

Important additions:

- `BeginCapture_FailsWithNullExplicitStream`
- `BeginCapture_UsesExplicitStreamOverload`

The tests use a fake `IWorkerGPUContext` and fake `IGPUGraphCapture` to prove that prefill capture refuses null streams and dispatches through `createGraphCapture(stream)`, not the default-stream overload.

## Verification Already Run

### Build

```bash
cmake --build build_v2_integration --target llaminar2 --parallel
```

Result:

```text
Linked `llaminar2` successfully.
```

Note: ROCm compilation prints existing unchecked `hipFree`/`hipMalloc` warnings in `ROCmMoEKernel.cpp`. They predate this handover and did not block the build.

### Focused prefill graph unit tests

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Unit_PrefillGraphCache(|Integration)$" \
  --output-on-failure --parallel
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

### Forced small ROCm server smoke

Command:

```bash
rm -rf /tmp/llaminar_e2e_prefill_stream_smoke2 && \
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_PREFILL_GRAPH_MIN_SEQ=1 \
LLAMINAR_PREFILL_GRAPH_TRACE=1 \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_prefill_stream_smoke2 \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --suite "models/qwen2.5-0.5b-instruct-q4_0.gguf|rocm:0|32" \
  --port 19690
```

Result:

```text
ALL PASSED: 14/14 tests passed
```

Trace evidence:

```text
Prefill graph ARMED for capture: seq_len=35
```

The E2E harness did not repeat the exact same prefill shape, so it proved arming but not capture/replay.

### Manual repeated-shape ROCm smoke

Server command:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_PREFILL_GRAPH_MIN_SEQ=1 \
LLAMINAR_PREFILL_GRAPH_TRACE=1 \
./build_v2_integration/llaminar2 serve \
  --port 19710 \
  -d rocm:0 \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  > /tmp/llaminar_prefill_stream_manual.log 2>&1
```

Then send the same request multiple times:

```bash
curl -sS --fail --max-time 120 \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:19710/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Answer with only the number."}],"max_tokens":4,"temperature":0}'
```

Observed answers were `4`.

Trace evidence from `/tmp/llaminar_prefill_stream_manual.log`:

```text
Prefill graph ARMED for capture: seq_len=42
Captured prefill graph: seq_len=42, nodes=749, device=ROCm:0
Prefill graph CAPTURED seq_len=42 nodes=749
Prefill graph REPLAY seq_len=42 replay_count=2
```

The manual server was stopped after verification.

### Editor diagnostics

`get_errors` reported no errors for:

```text
src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp
src/v2/execution/local_execution/engine/ForwardGraphTypes.h
src/v2/execution/local_execution/engine/PrefillGraphCache.cpp
src/v2/execution/local_execution/engine/PrefillGraphCache.h
src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp
src/v2/kernels/rocm/moe/ROCmMoEKernel.h
src/v2/kernels/KernelFactory.cpp
tests/v2/unit/stages/Test__PrefillGraphCache.cpp
```

## Key Lessons / Root Cause Notes

1. Prefill capture needed a decode-like stream owner.

   Using a raw/default stream was fragile because cached stages store stream pointers and replay/capture lifecycle can cross request boundaries. `CachedGraphStream` gives the prefill path explicit ownership and lets full invalidation safely destroy the stream.

2. Captured stream work must be launched immediately after instantiate.

   During HIP/CUDA capture, kernels are recorded into a graph and are not executed. Without a launch after `endCaptureAndInstantiate()`, the request that performed capture would not actually produce prefill outputs.

3. Session reset and full invalidation are different operations.

   Session reset should clear KV/GDN state and decode replay segments while preserving cached `ComputeGraph` objects and prefill graph entries. Full invalidation should still clear graph entries and destroy owned streams.

4. Captured graphs require stable device pointers.

   ROCm MoE reset cannot free scratch buffers that captured prefill graphs may reference. It should clear request-derived metadata and histogram contents without reallocating device scratch on every session boundary.

## Remaining Issue: Qwen3.5 35B ROCm Server Corruption

While validating this work, the focused Qwen3.5 35B ROCm server suite still failed after long thinking/multi-turn activity. This appears to be an existing issue, not caused by the prefill stream work.

Representative command:

```bash
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_rocm35_moe_reset \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
  --port 19650
```

Representative failure:

```text
Cache-clear request expected 8 but generated garbage / invalid UTF-8.
Later SSE checks failed because responses lacked valid finish metadata.
Server log included nlohmann JSON invalid UTF-8 exceptions.
```

Isolation attempts that did not fix it:

```text
LLAMINAR_ROCM_MOE_DEVICE_ROUTED_DECODE=0
LLAMINAR_MOE_REBALANCE=off
LLAMINAR_DETERMINISTIC=1 LLAMINAR_MOE_REBALANCE=off
LLAMINAR_GPU_GRAPHS=0 LLAMINAR_MOE_REBALANCE=off
```

The `LLAMINAR_GPU_GRAPHS=0` failure is especially important: the corruption is not only graph replay/capture.

The MoE dynamic-state reset addition was kept because `KernelFactory::resetAllDynamicState()` previously skipped cached MoE kernels, but that alone did not resolve the long-suite Qwen3.5 35B ROCm failure.

Related handovers for that thread:

```text
docs/v2/projects/2026-06/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_2026-05-20.md
docs/v2/projects/2026-06/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_CONTINUATION_2026-05-20.md
```

## Suggested Next Agent Resume Plan

1. Do not restart the prefill explicit-stream implementation from scratch.

   It builds, unit tests pass, and manual ROCm trace proves warmup/capture/replay on an explicit stream.

2. Run a focused full build or broader unit sweep if preparing to commit.

   Good next command:

   ```bash
   cmake --build build_v2_integration --parallel
   ctest --test-dir build_v2_integration \
     -R "^V2_Unit_(PrefillGraphCache|PrefillGraphCacheIntegration|StageRunPolicy|StageTimeline)$" \
     --output-on-failure --parallel
   ```

3. For the unresolved Qwen3.5 35B ROCm corruption, treat it as a separate bug.

   Since it reproduces with GPU graphs disabled, investigate persistent decode/session state outside graph replay first. The most suspicious areas are long-running Qwen3.5 GDN/short-conv/MoE state interaction after thinking decode, not the new prefill stream code.

4. If adding more validation, create a tiny repeated-shape prefill graph regression.

   The manual smoke demonstrated the behavior, but a test that can assert warmup -> capture -> replay without a full server would make this more durable. The fake context tests currently prove the stream overload; they do not execute real HIP capture.

## Repo Memory

I also recorded a short repo memory note:

```text
/memories/repo/prefill_graph_explicit_stream.md
```

It captures the main prefill graph lifecycle lessons for future agents.