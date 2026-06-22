# Handover: Long-Context E2E CPU/ROCm Continuation

Date: 2026-05-24
Workspace: `/workspaces/llaminar`
Branch: `feat/qwen35-moe`
HEAD at handover: `e93915ea`
Primary model: `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Primary goal: finish integrating and accepting long-context server E2E coverage for `cpu` and `rocm:0`.

## Executive Summary

The long-context server E2E harness exists and is already wired into
`tests/v2/e2e/server/test_server_e2e.sh` behind `LLAMINAR_E2E_LONG_CONTEXT=1`.
The default server E2E suites already include Qwen3.5 35B MoE on both `cpu` and
`rocm:0`.

The user reports that a separate subagent has fixed the ROCm long-prefill issue.
Do not assume it is accepted yet. The current worktree contains a large staged
fix set plus a smaller unstaged test/update set. The next agent should audit
those changes, run focused validation, then run the CPU and ROCm long-context
server E2E suites.

Previous known state from 2026-05-22/23:

- CPU Qwen3.5 35B MoE long-context E2E passed at 4k context with a 512-token
  long-generation budget: `25/25`.
- ROCm short smoke passed, but ROCm long-context recall previously produced
  token soup at roughly 900+ prompt tokens, even with GPU graphs and buckets
  disabled.
- Bucketed ROCm at 4k previously reached execution after the GDN preflight fix
  but OOMed on a 32GB `rocm:0` card for 1280/1536/4096 buckets.
- The current staged changes appear to address the later ROCm long-prefill and
  workspace/memory issues. Validate them rather than relying on the old failure.

## Current Worktree Snapshot

At handover, `git status --short` shows staged and unstaged changes. Do not
revert user/subagent changes.

Staged files include:

```text
M  src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp
M  src/v2/execution/compute_stages/stages/AttentionComputeStage.h
M  src/v2/execution/local_execution/device/WorkspaceAllocator.cpp
M  src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp
M  src/v2/execution/local_execution/engine/ForwardExecutionEngine.h
M  src/v2/execution/local_execution/engine/ForwardGraphTypes.h
M  src/v2/execution/local_execution/engine/PrefillGraphCache.h
M  src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp
M  src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h
M  src/v2/kernels/IKVCache.h
M  src/v2/kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp
M  src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip
M  src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h
M  src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp
M  src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip
M  tests/v2/CMakeLists.txt
M  tests/v2/integration/kernels/rocm/Test__ROCmFlashAttentionParity.cpp
A  tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_LongContext_Parity.cpp
M  tests/v2/unit/execution/compute_stages/stages/Test__AttentionComputeStage.cpp
A  tests/v2/unit/execution/local_execution/device/Test__WorkspaceAllocator.cpp
M  tests/v2/unit/models/qwen/Test__QwenStandardGraphSchema.cpp
MM tests/v2/unit/models/qwen3/Test__Qwen3BufferSizes.cpp
MM tests/v2/unit/models/qwen35/Test__Qwen35BufferSizes.cpp
```

Unstaged files include:

```text
M tests/v2/integration/execution/graph/Test__PrefillGraphCacheExecutionCommon.h
M tests/v2/unit/execution/local_execution/engine/Test__ForwardExecutionEngine.cpp
M tests/v2/unit/execution/local_execution/engine/Test__ForwardExecutionEngineAdvanced.cpp
M tests/v2/unit/models/qwen3/Test__Qwen3BufferSizes.cpp
M tests/v2/unit/models/qwen35/Test__Qwen35BufferSizes.cpp
M tests/v2/unit/stages/Test__PrefillGraphCacheIntegration.cpp
```

Cached diff stat at handover time:

```text
23 files changed, 2048 insertions(+), 96 deletions(-)
```

Unstaged diff stat at handover time:

```text
6 files changed, 23 insertions(+), 10 deletions(-)
```

## Long-Context E2E Harness State

Primary files:

- `tests/v2/e2e/server/test_server_e2e.sh`
- `tests/v2/e2e/server/long_context_checks.py`

The shell harness has optional long-context controls:

```text
LLAMINAR_E2E_LONG_CONTEXT=1
LLAMINAR_E2E_LONG_CONTEXT_TIER=lite|full
LLAMINAR_E2E_CONTEXT_LENGTH=4096
LLAMINAR_E2E_LONG_MAX_TOKENS=512
LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900
LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=1800
LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B=4
```

The 4B+ gate is intentional. It prevents smaller smoke models from failing
semantic long-context checks they are unlikely to satisfy.

Default Qwen3.5 35B MoE suites already present:

```text
models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|cpu|200
models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200
```

The helper currently checks:

- Beginning/middle/end sentinel recall using `LCJSON-ALPHA-314159`,
  `LCJSON-MIDDLE-271828`, and `LCJSON-OMEGA-161803`.
- Multi-needle strict JSON recall.
- Structured long generation and degeneration metrics.
- Cache reset after long prompt.
- Valid near-boundary context.
- Oversized context rejection.

Avoid shrinking `LLAMINAR_E2E_LONG_MAX_TOKENS` too aggressively. Earlier 64/128
token budgets made recall checks brittle. The accepted CPU run used 512.

## Separate ROCm Fix Context

The user says the ROCm long-prefill issue has been fixed by another subagent.
Current staged changes suggest the fix touches:

- Active/bucket-aware workspace sizing and reuse checks.
- Forward graph/cache metadata for active lengths.
- ROCm FlashAttention, RingKVCache, TQ cache, and GDN kernels.
- Qwen/Qwen3/Qwen3.5 buffer size expectations.
- A new CPU-vs-ROCm long-context parity sweep:
  `tests/v2/integration/parity/qwen35moe/Test__Qwen35MoE_LongContext_Parity.cpp`.

The new parity test is gated. CMake registers it with prefix:

```text
V2_Integration_Parity_Qwen35MoE_LongContext
```

The test file documents these envs:

```text
LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1
LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=...
LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=...
LLAMINAR_QWEN35MOE_SWEEP_CONTINUE_ON_FAILURE=...
LLAMINAR_QWEN35MOE_SWEEP_SNAPSHOTS=...
```

Note: the parity test hardcodes model path
`/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`. The workspace also has
`models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`. Verify the expected path exists before
running the parity sweep.

## Already Known Validations

These passed before the May 24 separate fix:

```bash
python3 -m py_compile tests/v2/e2e/server/long_context_checks.py \
  && python3 tests/v2/e2e/server/long_context_checks.py --self-test \
  && bash -n tests/v2/e2e/server/test_server_e2e.sh
```

```bash
cmake --build build_v2_integration --target v2_test_prefill_graph_capturability --parallel \
  && ctest --test-dir build_v2_integration \
       -R "V2_Unit_.*PrefillGraphCapturability|V2_Unit_PrefillGraphCapturability" \
       --output-on-failure --parallel
```

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Unit_.*Qwen35MoEGraph|Qwen35MoEGraph" \
  --output-on-failure --parallel
```

CPU E2E acceptance command that previously passed:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
LLAMINAR_E2E_CONTEXT_LENGTH=4096 \
LLAMINAR_E2E_LONG_MAX_TOKENS=512 \
LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=1800 \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_long_context_qwen35_moe_cpu_final \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|cpu|200" \
  --port 20030
```

Result at the time: `25/25` passed.

## Recommended Continuation Plan

1. Audit the current staged and unstaged changes.

   Start with:

   ```bash
   git status --short
   git --no-pager diff --cached --stat
   git --no-pager diff --stat
   ```

   Then inspect the staged fix areas before editing anything. Do not revert
   changes you did not make.

2. Run cheap script validation.

   ```bash
   python3 -m py_compile tests/v2/e2e/server/long_context_checks.py \
     && python3 tests/v2/e2e/server/long_context_checks.py --self-test \
     && bash -n tests/v2/e2e/server/test_server_e2e.sh
   ```

3. Build the relevant binaries.

   ```bash
   cmake --build build_v2_integration --target llaminar2 --parallel
   cmake --build build_v2_release --target llaminar2 --parallel
   ```

4. Run focused tests for the staged fix.

   Suggested filters:

   ```bash
   ctest --test-dir build_v2_integration \
     -R "V2_Unit_.*(WorkspaceAllocator|ForwardExecutionEngine|AttentionComputeStage|Qwen3BufferSizes|Qwen35BufferSizes|QwenStandardGraphSchema|PrefillGraphCache|PrefillGraphCapturability|Qwen35MoEGraph)" \
     --output-on-failure --parallel
   ```

   Run ROCm integration tests touched by the fix as feasible:

   ```bash
   ctest --test-dir build_v2_integration \
     -R "V2_Integration_.*ROCm.*(FlashAttention|RingKV|RoPE)" \
     --output-on-failure --parallel
   ```

5. Run the gated Qwen3.5 MoE CPU-vs-ROCm parity sweep.

   Start small to avoid wasting time:

   ```bash
   LLAMINAR_QWEN35MOE_LONG_CONTEXT_SWEEP=1 \
   LLAMINAR_QWEN35MOE_SWEEP_LENGTHS=256,512,1024 \
   LLAMINAR_QWEN35MOE_SWEEP_DECODE_STEPS=2 \
   ctest --test-dir build_v2_integration \
     -R "V2_Integration_Parity_Qwen35MoE_LongContext" \
     --output-on-failure --parallel
   ```

   If this passes, try longer lengths such as `1536,2048` if memory permits.

6. Re-run CPU long-context server E2E.

   Use the 4k / 512-token command above. It should still pass.

7. Re-run ROCm long-context server E2E without graphs first.

   This is the primary correctness check because it removes graph capture and
   bucket padding from the equation.

   ```bash
   LLAMINAR_LOG_LEVEL=INFO \
   LLAMINAR_MOE_REBALANCE=off \
   LLAMINAR_GPU_GRAPHS=0 \
   LLAMINAR_PREFILL_GRAPH_BUCKETS=0 \
   LLAMINAR_E2E_LONG_CONTEXT=1 \
   LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
   LLAMINAR_E2E_CONTEXT_LENGTH=2048 \
   LLAMINAR_E2E_LONG_MAX_TOKENS=512 \
   LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
   LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=1800 \
   LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_long_context_qwen35_moe_rocm_release_nographs_2k_afterfix \
   bash tests/v2/e2e/server/test_server_e2e.sh \
     --binary build_v2_release/llaminar2 \
     --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
     --port 20210
   ```

   Expected after the user-reported fix: long recall should pass and the server
   log should contain no WARN/ERROR entries.

8. Re-run ROCm with bucketed prefill enabled.

   Start with 2k context on a 32GB card:

   ```bash
   LLAMINAR_LOG_LEVEL=INFO \
   LLAMINAR_MOE_REBALANCE=off \
   LLAMINAR_GPU_GRAPHS=1 \
   LLAMINAR_PREFILL_GRAPH_BUCKETS=1 \
   LLAMINAR_PREFILL_GRAPH_TRACE=1 \
   LLAMINAR_E2E_LONG_CONTEXT=1 \
   LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
   LLAMINAR_E2E_CONTEXT_LENGTH=2048 \
   LLAMINAR_E2E_LONG_MAX_TOKENS=512 \
   LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
   LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=1800 \
   LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_long_context_qwen35_moe_rocm_release_buckets_2k_afterfix \
   bash tests/v2/e2e/server/test_server_e2e.sh \
     --binary build_v2_release/llaminar2 \
     --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
     --port 20230
   ```

   If 2k passes, try 4k. Previous 4k bucketed runs OOMed on 32GB, but the current
   staged `WorkspaceAllocator` and active-length changes may have improved this.

## Acceptance Criteria

Minimum acceptance for this continuation:

- Script validation passes.
- Focused unit/integration tests for changed files pass.
- CPU Qwen3.5 35B MoE long-context server E2E passes at 4k / 512.
- ROCm Qwen3.5 35B MoE no-graph/no-bucket long-context server E2E passes at a
  feasible context, preferably 2k or higher.
- No server WARN/ERROR entries in the successful ROCm correctness run.
- Any remaining bucketed ROCm limitation is documented as memory/configuration,
  not confused with raw long-prefill correctness.

Stretch acceptance:

- ROCm bucketed prefill long-context E2E passes at 2k and logs prefill graph
  warmup/capture/replay with `LLAMINAR_PREFILL_GRAPH_TRACE=1`.
- ROCm bucketed 4k passes on available hardware, or the log clearly shows the
  remaining limit is unavoidable VRAM pressure on the local card.

## Pitfalls

- Integration builds have snapshots enabled and may not exercise the same ROCm
  runtime-table decode path as Release. Use Release for final ROCm server E2E.
- Do not use `--no-mpi-bootstrap` for the server E2E or benchmark acceptance
  path; it is for profiling/debugging only.
- Do not lower the long helper max token budget to make runs faster unless you
  also revalidate CPU and ROCm. Low budgets caused false oracle failures earlier.
- Do not update `.githooks/benchmark_baseline.json` without explicit human
  approval.
