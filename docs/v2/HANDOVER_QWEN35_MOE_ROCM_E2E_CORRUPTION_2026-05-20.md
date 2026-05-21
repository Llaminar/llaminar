# Handover: Qwen3.5 MoE ROCm E2E Corruption

Date: 2026-05-20
Workspace: `/workspaces/llaminar`
Branch/HEAD: `feat/qwen35-moe` at `ac6a32c5`
Primary target: Qwen3.5 35B MoE ROCm server E2E corruption
Model: `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Main test harness: `tests/v2/e2e/server/test_server_e2e.sh`

This is the primary handover for the next agent. The prefill explicit-stream work is implemented and separately documented in:

```text
docs/v2/HANDOVER_PREFILL_GRAPH_EXPLICIT_STREAM_2026-05-20.md
```

The remaining priority is to find and fix the Qwen3.5 MoE ROCm E2E corruption in the server suite.

## Current Status

The focused Qwen3.5 35B MoE ROCm server E2E suite fails after earlier requests pass. The model initially answers simple prompts correctly, then a later independent request produces arbitrary text / invalid UTF-8 instead of the expected short numeric answer.

Typical progression:

```text
Server starts: pass
GET /health: pass
Single-turn non-thinking: pass, answer 4
Single-turn thinking: pass, answer 4
Multi-turn non-thinking: pass, answer 42
Multi-turn thinking: pass, answer 42
Cache-clear / independent request: corrupt output, expected 8
Later response-format and SSE checks: fail due malformed/invalid UTF-8 output
```

This looks like a persistent state or memory corruption issue that is triggered by a longer thinking/multi-turn decode and then leaks into the next independent request. The invalid UTF-8 exceptions are a symptom: the tokenizer is receiving nonsensical token IDs/streams and nlohmann rejects the resulting string when serializing JSON.

## Reproduction Command

Use the focused ROCm suite, not the full default suite, when iterating:

```bash
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_rocm35_repro \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
  --port 19650
```

Representative failure output from a recent run:

```text
Cache-clear (non-thinking): expected answer 8, got '7 answer<|box_end|>一ったTERS大头然...'
Cache-clear (thinking): expected answer 8, got 'PARSE_ERROR'
Response format: missing/invalid usage or finish_reason
SSE streaming: no finish_reason chunk found / metadata missing
```

Representative server log errors:

```text
[ChatCompletion] non-streaming request failed with unhandled exception:
  [json.exception.type_error.316] invalid UTF-8 byte at index 512: 0xD1

[ChatCompletion] streaming request failed with unhandled exception:
  [json.exception.type_error.316] incomplete UTF-8 string; last byte: 0xAA

[ChatCompletion] streaming request failed with unhandled exception:
  [json.exception.type_error.316] incomplete UTF-8 string; last byte: 0xB6
```

Recent relevant log directories:

```text
/tmp/llaminar_e2e_rocm35_moe_reset/
/tmp/llaminar_e2e_rocm35_gpu_graphs_off_clean/
/tmp/llaminar_e2e_rocm35_no_device_routed_decode/
/tmp/llaminar_e2e_rocm35_rebalance_off_clean/
/tmp/llaminar_e2e_rocm35_deterministic/
```

## What Has Been Ruled Out

These toggles were tried and did not eliminate the corruption:

```text
LLAMINAR_ROCM_MOE_DEVICE_ROUTED_DECODE=0
LLAMINAR_MOE_REBALANCE=off
LLAMINAR_DETERMINISTIC=1 LLAMINAR_MOE_REBALANCE=off
LLAMINAR_GPU_GRAPHS=0 LLAMINAR_MOE_REBALANCE=off
```

Important implication:

```text
LLAMINAR_GPU_GRAPHS=0 still fails, so this is not only HIP graph replay/capture.
```

The `LLAMINAR_ROCM_MOE_DEVICE_ROUTED_DECODE=0` run still failed, so the device-routed decode route-table path is not the sole cause.

`LLAMINAR_MOE_REBALANCE=off` still failed, so dynamic expert rebalancing is not the sole cause.

`LLAMINAR_DETERMINISTIC=1` still failed, so the optional default-on fast paths disabled by deterministic mode are not sufficient to explain it.

## Current Code Changes Related To This Thread

### Session replay reset

Files:

```text
src/v2/execution/local_execution/engine/ForwardExecutionEngine.h
src/v2/execution/local_execution/engine/ForwardGraphTypes.h
src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h
```

Current behavior:

- `DeviceGraphOrchestrator::clear_cache()` preserves cached `ComputeGraph` objects.
- It calls `forward_engine_->resetSessionReplayState()` before clearing state.
- Decode graph captures are reset at session boundaries.
- Prefill graph entries are preserved because their explicit stream and stable arena/device pointers can survive across independent requests.

This change was necessary but did not fix the Qwen3.5 35B ROCm E2E corruption.

### MoE dynamic-state reset

Files:

```text
src/v2/kernels/KernelFactory.cpp
src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp
src/v2/kernels/rocm/moe/ROCmMoEKernel.h
```

Current behavior:

- `KernelFactory::resetAllDynamicState()` now visits cached MoE kernels.
- `ROCmMoEKernel::resetDynamicState()` resets histogram counts and clears host-side grouping metadata.
- It intentionally does not free device scratch or runtime pointer arrays because captured prefill graphs can reference those stable device addresses.

This was added because MoE kernels were previously skipped by global dynamic-state reset. It is still useful hygiene, but the focused ROCm 35B E2E suite still failed afterward.

## Related Parity Status

Related docs:

```text
docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_2026-05-20.md
docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_CONTINUATION_2026-05-20.md
```

Key takeaway from parity work:

- The Qwen3.5 MoE ROCm decode parity target passes token-level checks, but stage metrics show real drift.
- Prefill parity is clean.
- Decode step 0 shows the first meaningful drift at `GDN_DELTA_RULE_OUTPUT`, then `ATTENTION_OUTPUT` amplifies it, and MoE routing diverges downstream.

Important layer-0 decode-stage excerpt:

```text
ATTENTION_NORM           cos=1.000000 drop=0.000000
QKV_PROJECTION           cos=0.999997 drop=0.000003
GDN_Z_PROJECTION         cos=0.999997 drop=-0.000000
GDN_DELTA_RULE_OUTPUT    cos=0.983706 drop=0.016292
GDN_NORM_GATE_OUTPUT     cos=0.981886 drop=0.001819
ATTENTION_OUTPUT         cos=0.918513 drop=0.063374
MOE_ROUTING_INDICES      routing_overlap=0.875000 top1=0.000000
MOE_EXPERT_OUTPUT        cos=0.619589
MOE_COMBINED_OUTPUT      cos=0.702070 drop=0.129929
```

Interpretation:

- Do not start by chasing standalone GEMV projection math; QKV and Z projections are already high cosine in the parity run.
- Treat MoE expert divergence as downstream until routing agreement is stable.
- The most relevant parity lead is full-model decode state/context around GDN/attention, not isolated GDN projection kernels.

## Likely Investigation Direction

The E2E corruption appears after long thinking decode and then affects a later independent request. That points to one of these classes of bugs:

1. Persistent state not fully reset at request cleanup.

   Check recurrent and cache state that survives across requests:

   ```text
   src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h
   src/v2/execution/runner/OrchestrationRunner.cpp
   src/v2/app/modes/ChatCompletionHandler.cpp
   src/v2/kernels/KernelFactory.cpp
   src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h
   src/v2/kernels/rocm/conv/ROCmShortConvolution.h
   src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp
   src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp
   ```

2. Decode path writes past scratch bounds during long thinking output.

   This would explain why the following independent request produces garbage even after cache clear. Focus on kernels whose scratch capacity depends on token count, top-k, active experts, route table size, or decode length.

3. State reset happens on the wrong stream or races with outstanding ROCm work.

   The runner generally synchronizes, but verify the exact order at request cleanup:

   ```text
   ChatCompletionHandler::RequestCacheCleanup
   OrchestrationRunner::clearCache
   DeviceGraphOrchestrator::clear_cache
   state_.clear()
   KernelFactory::resetAllDynamicState()
   GDN/ShortConv reset calls
   ```

4. Full-model GDN/attention decode state differs from the isolated GDN test.

   The isolated `ROCmPrefillMatchesSequentialDecodeQwen35Shape` regression passed, but the full-model parity drift starts at GDN delta rule output. That suggests a lifecycle/context mismatch rather than a simple kernel arithmetic bug.

## Recommended Next Steps

### 1. Minimize the server reproducer

Create a small script that starts one server and runs only the smallest request sequence that corrupts:

1. Single-turn non-thinking numeric prompt.
2. Single-turn or multi-turn thinking prompt with `max_tokens=200`.
3. Independent numeric prompt expected to answer `8`.

Then binary-search:

```text
max_tokens: 32, 64, 96, 128, 160, 200
thinking vs non-thinking
single long request vs multi-turn long request
streaming disabled until root cause is found
```

Goal: determine whether corruption requires long decode length, thinking mode, multi-turn history, or a specific request order.

### 2. Add reset-state checksums

At request cleanup, log or assert device-side checksums before and after reset for:

```text
KV cache heads / offsets
GDN recurrence state
ShortConv state
MoE route/group scratch metadata
MoE histogram / expert counts
```

Useful temporary env flag name:

```text
LLAMINAR_QWEN35_RESET_TRACE=1
```

The goal is not to permanently add verbose logging; it is to identify which state remains nonzero or changes unexpectedly after `clear_cache()`.

### 3. Add sentinel/capacity checks around ROCm MoE scratch

If reset checks look clean, suspect a memory scribble during long decode. Add temporary bounds/sentinel checks around ROCm MoE scratch allocations that scale with active experts or route table capacity.

Likely files:

```text
src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp
src/v2/kernels/rocm/moe/ROCmMoEGroupedDecodeKernels.hip
src/v2/kernels/rocm/moe/ROCmMoEGroupedPrefillKernels.hip
src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp
```

### 4. Compare CPU vs ROCm reset behavior under the same request sequence

Use the minimized sequence, not the full E2E suite. Compare:

```text
cpu
rocm:0
rocm:0 with LLAMINAR_GPU_GRAPHS=0
rocm:0 with LLAMINAR_MOE_REBALANCE=off
rocm:0 with LLAMINAR_DETERMINISTIC=1
```

Do not spend more time on toggles already listed above unless the minimized reproducer changes the result.

### 5. Use parity dumps only after a minimized E2E reproducer exists

The parity tests are useful, but the E2E corruption is request-sequence dependent. First make the failing server sequence small. Then add targeted stage/state dump around the transition between the long thinking request and the next independent prompt.

## Commands Worth Keeping Handy

Focused E2E repro:

```bash
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_rocm35_repro \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
  --port 19650
```

GPU graphs disabled control:

```bash
LLAMINAR_GPU_GRAPHS=0 \
LLAMINAR_MOE_REBALANCE=off \
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_rocm35_gpu_graphs_off \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
  --port 19630
```

Focused decode parity target:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure
```

Focused prefill parity target:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_PrefillParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure
```

## Do Not Confuse With VRAM-Pressure Test Failures

A recent `v2_test_rocm_moe_kernel` ctest snippet showed failures like:

```text
LoadOrchestrator: VRAM budget preflight failed for device 0
required=64 MiB available_after_margin=0 MiB free=1174 MiB total=32752 MiB
```

Those are capacity/preflight failures under low free VRAM, not the server E2E corruption described here.

## Worktree Caution

The worktree is dirty with active WIP. Do not reset, checkout, or revert unrelated changes. Work with existing changes unless the user explicitly asks otherwise.

At the time of this handover, docs status included:

```text
M  docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_CONTINUATION_2026-05-20.md
?? docs/v2/HANDOVER_PREFILL_GRAPH_EXPLICIT_STREAM_2026-05-20.md
?? docs/v2/HANDOVER_QWEN35_MOE_ROCM_E2E_CORRUPTION_2026-05-20.md
```

## Bottom Line

The next agent should focus on reproducing and minimizing the Qwen3.5 35B MoE ROCm server corruption, then instrument request-boundary state reset and long-decode scratch bounds. The evidence so far says this is not solely GPU graph replay, device-routed decode, dynamic MoE rebalance, or standalone projection/GEMV math.