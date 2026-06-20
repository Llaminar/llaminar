# Handover: CUDA/ROCm Active Workspace Server E2E

Date: 2026-05-26
Workspace: `/workspaces/llaminar`
Branch: `feat/qwen35-moe`
Primary goal: align CUDA and ROCm around active/bucket workspace sizing, fix the CUDA short-to-long server E2E regression exposed by that policy, and add symmetric CUDA/ROCm regression coverage in the appropriate test suite.

## Executive Summary

The user wants CUDA and ROCm to use the same workspace sizing behavior. We agreed the desirable behavior is the ROCm policy: size workspace scratch by the active prompt/bucket length when available, instead of reserving the full configured context up front.

The previous CUDA-only full-context workaround made the precommit-style server E2E pass, but it is not the desired endpoint because it wastes VRAM and undermines bucketed prefill. The current worktree has already removed that CUDA/ROCm asymmetry in `DeviceGraphOrchestrator`: both backends now use `workspace_seq_len > 0 ? workspace_seq_len : default_workspace_seq_len`.

The unresolved issue is that qwen2.5 CUDA server smoke fails under active sizing: a short single-turn request succeeds, then a longer multi-turn/cache/SSE request returns EOS/empty output. Full-context workspace sizing masked the bug. Several quick discriminators have been tried and did not fix it, so the next agent should continue the CUDA root-cause investigation rather than preserving the full-context workaround.

## Current Worktree Snapshot

At handover, `git status --short` shows these modified files:

```text
M .githooks/pre-commit
M src/v2/execution/local_execution/device/DeviceWorkspaceManager.cpp
M src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp
M src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp
M src/v2/kernels/cuda/gemm/CUDAQuantisedGemmKernel.h
M src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h
M src/v2/kernels/cuda/kvcache/CUDARingKVCache.cu
M src/v2/kernels/cuda/kvcache/CUDARingKVCache.h
M src/v2/loaders/PreparedWeightStore.cpp
M src/v2/loaders/PreparedWeightStore.h
```

Current diff stat before adding this handover file:

```text
10 files changed, 280 insertions(+), 28 deletions(-)
```

`cmake --build build_v2_integration --target llaminar2 --parallel` currently reports `ninja: no work to do`, and `git diff --check` passes.

Important: `DeviceWorkspaceManager.cpp` has only formatting changes around `LOG_DEBUG` indentation from user/formatter/another automated tool. Read it before editing and avoid reverting it unless explicitly asked.

## What Is Already Done

### Precommit Server E2E Wiring

`.githooks/pre-commit` now enables long-context checks in the existing server E2E harness rather than adding standalone server E2E CTests:

```bash
LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER="${LLAMINAR_E2E_LONG_CONTEXT_TIER:-lite}" \
LLAMINAR_E2E_CONTEXT_LENGTH="${LLAMINAR_E2E_CONTEXT_LENGTH:-4096}" \
LLAMINAR_E2E_LONG_MAX_TOKENS="${LLAMINAR_E2E_LONG_MAX_TOKENS:-512}" \
LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS="${LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS:-900}" \
LLAMINAR_E2E_LONG_REQUEST_TIMEOUT="${LLAMINAR_E2E_LONG_REQUEST_TIMEOUT:-1800}" \
LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B="${LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B:-30}" \
tests/v2/e2e/server/test_server_e2e.sh \
    --binary "$BUILD_V2_INTEGRATION/llaminar2" \
    --backends "cpu,cuda:0,rocm:0"
```

The 30B gate keeps qwen2.5 1.5B and Qwen3.5 4B short, while Qwen3.5 35B CPU/ROCm run long-context checks.

Earlier validation with the CUDA full-context workaround passed:

- qwen2.5 CPU/CUDA/ROCm server smoke.
- Qwen3.5 4B CPU server smoke.
- Qwen3.5 35B CPU long-context 4k/512.
- Qwen3.5 35B ROCm long-context 4k/512.
- Final summary: `116/116` passed, with no server WARN/ERROR entries.

That pass is useful evidence that the long-context harness and ROCm long-context path are healthy, but it depended on the temporary CUDA full-context workaround and is not the target final policy.

### Symmetric Active Workspace Policy

`src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` currently has the desired symmetric policy:

```cpp
hints.max_seq_len = workspace_seq_len > 0 ? workspace_seq_len : default_workspace_seq_len;
```

Do not reintroduce a `state_.device_id.is_cuda() ? default_workspace_seq_len : ...` branch. That branch was only a masking workaround.

### Kernel/Cache Reset Hardening

These changes are believed useful and should probably stay unless the next investigation proves otherwise:

- `DeviceGraphOrchestrator::resetKernelDynamicState()` calls `prepared_weight_store_->resetDynamicState()`.
- `PreparedWeightStore::resetDynamicState()` resets prepared GEMM kernels, fused/sliced adapters, and expert slab engines while preserving packed weights.
- `CUDAQuantisedGemmKernel::resetDynamicState()` destroys per-request GEMV, prefill, and cuBLAS contexts and clears the stream binding.
- Dense `CUDARingKVCache<Precision>::clear()` now scrubs CUDA request-sidecars and persistent K/V storage while keeping stable device pointers for graph caches.
- `CUDAHybridRingKVCache::clear()` now delegates to the dense base clear, then resets GDN state.

These changes did not by themselves fix the active-sizing CUDA failure, but they address real reset-state holes uncovered during the investigation.

## Current Provisional / Failed Experiments

The worktree still contains two CUDA GEMM changes that were tried as discriminators and reportedly did not fix the qwen2.5 CUDA active-sizing failure:

1. `CUDAQuantisedGemmKernel::bindWorkspace()` resets dynamic state when the workspace pointer changes.
2. `CUDAQuantisedGemmKernel::getWorkspaceRequirements()` pads GEMM workspace rows up to 128 for prefill (`workspace_m = (m > 1) ? ((m + 127) & ~127) : m`).

Both should be treated as provisional. The next agent should decide whether to keep, revert, or replace them after reproducing the current failure. Do not assume they are the final fix.

Other failed discriminators from the previous investigation:

- Reset-on-rebind alone did not fix the CUDA failure.
- Zeroing fresh workspace in `DeviceWorkspaceManager` did not fix.
- GEMM tile-row padding did not fix.
- Full active workspace per request still failed, which suggests the first short active-sized CUDA request may corrupt persistent state or trigger a CUDA kernel path that assumes more scratch than declared.
- Sampling was ruled out: DEBUG logs showed CPU and GPU greedy sampling agreed on EOS after the bad second prefill, so logits were already wrong.
- KV metadata looked reset in logs (`position_offset=0`, attention `seq_len=64 kv_len=64`), so the obvious KV length metadata path was not the root.

## Reproduction Commands

### Focused CUDA Failure Target

Use this first after verifying the current source is symmetric active sizing:

```bash
cmake --build build_v2_integration --target llaminar2 --parallel

LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_cuda_qwen25_active_policy_validation \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "models/qwen2.5-1.5b-instruct-q8_0.gguf|cuda:0|200" \
  --port 20600
```

Expected before the real fix: CUDA qwen2.5 fails after the first short request, typically with 3 failures:

```text
Multi-turn (non-thinking): expected answer 42, got ''
Cache-clear (non-thinking): expected answer 8, got ''
SSE streaming (non-thinking): expected answer 2, got ''
```

### Compare ROCm Symmetry

Run the same short-to-long server sequence on ROCm to ensure the regression test is symmetric:

```bash
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_rocm_qwen25_active_policy_validation \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "models/qwen2.5-1.5b-instruct-q8_0.gguf|rocm:0|200" \
  --port 20620
```

ROCm has been passing this smoke path.

### Final Hook-Equivalent Validation

After the CUDA root cause is fixed:

```bash
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_precommit_e2e_long_context_symmetric_validation \
LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER="${LLAMINAR_E2E_LONG_CONTEXT_TIER:-lite}" \
LLAMINAR_E2E_CONTEXT_LENGTH="${LLAMINAR_E2E_CONTEXT_LENGTH:-4096}" \
LLAMINAR_E2E_LONG_MAX_TOKENS="${LLAMINAR_E2E_LONG_MAX_TOKENS:-512}" \
LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS="${LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS:-900}" \
LLAMINAR_E2E_LONG_REQUEST_TIMEOUT="${LLAMINAR_E2E_LONG_REQUEST_TIMEOUT:-1800}" \
LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B="${LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B:-30}" \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --backends "cpu,cuda:0,rocm:0"
```

Target result: `116/116` passed with no server WARN/ERROR entries.

## Investigation Leads

The active/bucket workspace allocator is designed to support growth safely:

- `WorkspaceAllocator` checks both missing buffer names and undersized buffer sizes.
- On growth it merges existing + new requirements, reallocates, rebinds consumers, and bumps a per-device workspace generation.
- `ForwardExecutionEngine` tracks workspace generation and invalidates captured replay state after generation changes.

Because full-context sizing masks the bug, but active sizing exposes it, the likely root remains one of these:

1. A CUDA kernel writes past its declared workspace requirement during the first short request.
2. A CUDA workspace consumer does not fully declare all scratch it uses.
3. A CUDA stage or adapter does not bind all child consumers on workspace growth.
4. A persistent CUDA object caches a workspace pointer or workspace-derived pointer outside `IWorkspaceConsumer::bindWorkspace()`.
5. A CUDA attention/KV workspace requirement is based on heads/head_dim/batch instead of the active sequence length.

Previously checked but not conclusively eliminated:

- Fused QKV and fused gate/up stages appeared to merge/bind child consumers correctly.
- Dense qwen2.5 CUDA KV cache is `CUDARingKVCache<FP16>`, not hybrid/TQ.
- Attention logs showed sane visible KV metadata on the bad second request.

Suggested next checks:

- Inspect CUDA attention/KV workspace requirements carefully. Confirm `m` or explicit sequence length is what the allocator passes for attention/KV consumers.
- Add assert/guard logging that compares actual kernel launch dimensions against `workspace_->getBufferSize()` for every CUDA workspace buffer that can be indexed by sequence length. Keep this behind assertions or DEBUG and remove temporary logs before final.
- Look for static/global CUDA scratch pools or cached pointer tables not represented by `IWorkspaceConsumer`.
- If possible, run the focused CUDA repro under `compute-sanitizer` or with guard canaries around workspace suballocations. Use a minimal direct server repro if the full harness is too slow.

## Test Coverage Requirement

The user explicitly asked to lock the fix in with symmetrical integration tests for CUDA and ROCm.

Do not add standalone long-context server CTest registrations. The precommit long-context path should remain integrated through `.githooks/pre-commit` and `tests/v2/e2e/server/test_server_e2e.sh`.

Good options for the regression test:

1. Extend the existing server E2E harness with an explicit short-to-long cache-reset/workspace-growth regression that runs for both `cuda:0` and `rocm:0` when those backends are selected. The existing qwen2.5 sequence already catches the CUDA failure, but a named check would make the intent clearer.
2. Add a C++ integration test under `tests/v2/integration/...` that drives two consecutive prefills on the same runner/orchestrator: short prompt first, longer prompt second, active workspace sizing enabled, then validates deterministic greedy output or logits. Register both CUDA and ROCm variants using existing test naming conventions.

Whichever route is chosen, keep it symmetric: same model, same request order, same expectations for `cuda:0` and `rocm:0`.

## Validation Already Run

Before the handover file was created:

```bash
cmake --build build_v2_integration --target llaminar2 --parallel
# Result: ninja: no work to do

git --no-pager diff --check
# Result: diff check ok
```

Temporary sampling diagnostics are not present. A grep for the old temporary strings only found the pre-existing Qwen35 long-context parity CTest registration:

```text
tests/v2/CMakeLists.txt: integration/parity/qwen35moe/Test__Qwen35MoE_LongContext_Parity.cpp
tests/v2/CMakeLists.txt: TEST_PREFIX V2_Integration_Parity_Qwen35MoE_LongContext
```

That parity CTest is unrelated to the server E2E hook wiring and should not be removed as part of this task.

## Subagent Note

An attempt was made to dispatch this to a default coding subagent, but the launch failed twice due model/tool errors. No useful subagent implementation progress should be assumed.

## Acceptance Criteria

Minimum acceptable continuation result:

- CUDA and ROCm both use active/bucket workspace sizing.
- Focused qwen2.5 CUDA server E2E passes under active sizing.
- Equivalent qwen2.5 ROCm server E2E still passes.
- Symmetric CUDA/ROCm regression coverage is added in the appropriate suite.
- Final hook-equivalent server E2E passes `116/116`, including Qwen3.5 35B CPU and ROCm 4k/512 long-context checks.
- No server WARN/ERROR entries in the successful final runs.
- Temporary diagnostic logs/probes are removed or gated appropriately.

Stretch result:

- Root cause is documented in the final response and, if useful, in a short repo memory update.
- Any retained reset hardening is justified as required for correctness, not just left because it was tried during debugging.