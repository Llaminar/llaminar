# Handover: Qwen3.5 35B MoE Long-Context E2E and ROCm Corruption

Date: 2026-05-23
Workspace: `/workspaces/llaminar`
Branch context: current worktree has staged long-context / ROCm investigation changes
Primary model: `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Primary unresolved backend: `-d rocm:0`

## Executive Summary

Long-context server E2E coverage was added and exercised against Qwen3.5 35B MoE.
The CPU backend is accepted for the new long-context lite suite. ROCm is not
accepted yet.

What is proven:

- The default server E2E suite already includes Qwen3.5 35B MoE on both `cpu`
  and `rocm:0`.
- CPU Qwen3.5 35B MoE long-context E2E passes at 4k context with a 512-token
  long-generation budget: `25/25` passed.
- The original ROCm bucketed prefill `GDN_RECURRENCE` cold preflight blocker was
  fixed at the support/readiness contract level.
- A real Qwen3.5 FA graph-ordering bug was fixed: `kv_append` now depends on
  `rope` even in `rope_on_read` mode, preserving Q/K norm ordering before K is
  appended to the cache.

What remains open:

- ROCm Qwen3.5 35B MoE long-context recall still corrupts after medium/long
  prefill, even with GPU graphs and prefill buckets disabled.
- The failure appears around 900+ prompt tokens. Short smoke, arithmetic,
  streaming, and cache-reset probes pass on ROCm.
- The no-graph ROCm failure produces token soup / malformed JSON without server
  WARN/ERROR lines, which points to silent full-pipeline ROCm long-prefill
  corruption rather than an E2E oracle issue.
- Bucketed ROCm at 4k context reaches execution after the GDN preflight fix but
  OOMs on this 32GB `rocm:0` card for 1280/1536/4096 buckets. Treat this as a
  separate VRAM pressure issue from the raw no-graph correctness corruption.

## Current Staged Files

These files are currently staged in the worktree:

- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp`
- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h`
- `src/v2/models/qwen/QwenGraphBase.cpp`
- `tests/v2/e2e/server/long_context_checks.py`
- `tests/v2/integration/kernels/rocm/Test__ROCmFlashAttentionParity.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmRingKVCache.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmRoPEParity.cpp`
- `tests/v2/unit/models/qwen35moe/Test__Qwen35MoEGraph.cpp`
- `tests/v2/unit/stages/Test__PrefillGraphCapturability.cpp`

Cached diff stat at handover time:

```text
 .../compute_stages/stages/GDNRecurrenceStage.cpp   |  15 +
 .../compute_stages/stages/GDNRecurrenceStage.h     |   2 +
 src/v2/models/qwen/QwenGraphBase.cpp               |   8 +-
 tests/v2/e2e/server/long_context_checks.py         |  32 +-
 .../rocm/Test__ROCmFlashAttentionParity.cpp        | 392 ++++++++++++++++++++-
 .../kernels/rocm/Test__ROCmRingKVCache.cpp         | 109 ++++++
 .../kernels/rocm/Test__ROCmRoPEParity.cpp          |  73 ++++
 .../unit/models/qwen35moe/Test__Qwen35MoEGraph.cpp |  75 ++++
 .../stages/Test__PrefillGraphCapturability.cpp     |  73 +++-
 9 files changed, 762 insertions(+), 17 deletions(-)
```

Important: `tests/v2/integration/kernels/rocm/Test__ROCmRingKVCache.cpp` was
modified after the prior response by the user or automation. Re-read it before
making additional edits.

## Implemented Changes

### Long-Context Server E2E

File: `tests/v2/e2e/server/long_context_checks.py`

- Adds objective long-context checks against a live `/v1/chat/completions`
  server.
- Covers beginning/middle/end sentinel recall, strict multi-needle JSON recall,
  structured long generation, cache reset after a long request, valid near-boundary
  context, and oversized context rejection.
- Uses a 4B+ model gate through the shell harness so small smoke models are not
  asked to pass long-context semantic checks.
- The current helper uses a 512-token lite budget for the 35B MoE validation;
  avoid reintroducing very small budgets because they created brittle recall
  failures unrelated to backend correctness.

File: `tests/v2/e2e/server/test_server_e2e.sh`

- The default suites already include Qwen3.5 35B MoE on both `cpu` and `rocm:0`:
  `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|cpu|200` and
  `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200`.

### GDN Cold Padded-Prefill Preflight

Files:

- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h`
- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp`
- `tests/v2/unit/stages/Test__PrefillGraphCapturability.cpp`

Change:

- Added `GDNRecurrenceStage::supportsPaddedPrefillGraphCapturePreflight()`.
- Cold padded preflight now validates ROCm GDN support using the backend real-length
  contract instead of requiring already-warmed GPU recurrence state.
- `isGraphCapturable()` remains strict and still checks actual capture-time
  readiness.
- Runtime `PrefillGraphCache` snapshot checks still reject active snapshot
  callbacks before stage support checks.

Why it matters:

- Before this fix, bucketed ROCm MoE long prompts failed before execution with:
  `Stage 'layer0_gdn_recurrence' does not support cold padded-prefill graph preflight`.
- After this fix, that GDN blocker is gone. Integration builds can still hit a
  separate MoE routing snapshot-compiled blocker; Release reaches execution.

### Qwen3.5 FA Graph Ordering

Files:

- `src/v2/models/qwen/QwenGraphBase.cpp`
- `tests/v2/unit/models/qwen35moe/Test__Qwen35MoEGraph.cpp`

Change:

- `kv_append` now always depends on the RoPE node.
- In `rope_on_read` mode RoPE skips mutating K, but the RoPE node still carries
  Q/K norm dependencies. Appending directly after QKV projection could cache
  pre-normalized K in Qwen3.5 FA layers.
- Added a Qwen3.5 MoE graph dependency regression test covering:
  `layer0_kv_append -> layer0_rope -> layer0_k_norm`.

Status:

- This is a real fix and should be kept.
- It did not fix the full ROCm long-prefill corruption by itself.

### Additional Focused ROCm Tests

Files:

- `tests/v2/integration/kernels/rocm/Test__ROCmFlashAttentionParity.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmRingKVCache.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmRoPEParity.cpp`

Coverage added during investigation:

- Long Qwen3.5-shape ROCm FlashAttention FP32 prefill.
- FP16 KV prefill and long FP16 KV decode coverage.
- Ring KV cache RoPE-on-read at long Qwen3.5 shape.
- Long partial-RoPE head-dim 256 parity.

Subagent-reported result:

- Focused ROCm attention, decode, RoPE, KV-cache RoPE-on-read, and Qwen3.5 graph
  dependency tests passed during the investigation.
- Re-run these before accepting any further ROCm fix.

## Commands Already Run

### Script and Unit Validation

```bash
python3 -m py_compile tests/v2/e2e/server/long_context_checks.py \
  && python3 tests/v2/e2e/server/long_context_checks.py --self-test \
  && bash -n tests/v2/e2e/server/test_server_e2e.sh
```

Result: passed.

```bash
cmake --build build_v2_integration --target v2_test_prefill_graph_capturability --parallel \
  && ctest --test-dir build_v2_integration \
       -R "V2_Unit_.*PrefillGraphCapturability|V2_Unit_PrefillGraphCapturability" \
       --output-on-failure --parallel
```

Result: passed.

```bash
ctest --test-dir build_v2_integration \
  -R "V2_Unit_.*Qwen35MoEGraph|Qwen35MoEGraph" \
  --output-on-failure --parallel
```

Result: passed.

```bash
git --no-pager diff --check -- \
  src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h \
  src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp \
  src/v2/models/qwen/QwenGraphBase.cpp \
  tests/v2/unit/stages/Test__PrefillGraphCapturability.cpp \
  tests/v2/unit/models/qwen35moe/Test__Qwen35MoEGraph.cpp \
  tests/v2/e2e/server/long_context_checks.py \
  tests/v2/integration/kernels/rocm/Test__ROCmFlashAttentionParity.cpp \
  tests/v2/integration/kernels/rocm/Test__ROCmRingKVCache.cpp \
  tests/v2/integration/kernels/rocm/Test__ROCmRoPEParity.cpp
```

Result: passed.

### CPU Long-Context Acceptance Case

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

Result: passed, `25/25`.

Log path:

```text
/tmp/llaminar_e2e_long_context_qwen35_moe_cpu_final/20260522_192110_Qwen3.5-35B-A3B-UD-Q4_K_XL_cpu_port20031.log
```

### ROCm Bucketed 4k Case After GDN Fix

```bash
cmake --build build_v2_release --target llaminar2 --parallel

LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_MOE_REBALANCE=off \
LLAMINAR_GPU_GRAPHS=1 \
LLAMINAR_PREFILL_GRAPH_BUCKETS=1 \
LLAMINAR_PREFILL_GRAPH_TRACE=1 \
LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
LLAMINAR_E2E_CONTEXT_LENGTH=4096 \
LLAMINAR_E2E_LONG_MAX_TOKENS=512 \
LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=1800 \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_long_context_qwen35_moe_rocm_release_gdnfix \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_release/llaminar2 \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
  --port 20110
```

Result: failed from VRAM pressure, not `StageNotCapturable`.

Representative errors:

```text
[ROCmBackend] Insufficient GPU memory on device 0: requested 3.0 MB but only 30.0 MB free
Prefill failed: [ConcurrentPrefill] Projection 0 output has no GPU data (shape=1536x512, coherence=0)
[ROCmGatedDeltaNet] deinterleave scratch malloc failed
```

Log path:

```text
/tmp/llaminar_e2e_long_context_qwen35_moe_rocm_release_gdnfix/20260522_204354_Qwen3.5-35B-A3B-UD-Q4_K_XL_rocm_0_port20111.log
```

### ROCm No-Bucket / No-Graph 2k Repro

This is the most useful correctness repro because it removes graph capture and
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
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_long_context_qwen35_moe_rocm_release_nographs_2k \
bash tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_release/llaminar2 \
  --suite "models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|rocm:0|200" \
  --port 20150
```

Result: failed long-context checks, no server WARN/ERROR entries.

Representative failures:

```text
long needle recall beginning: target code LCJSON-ALPHA-314159 missing from content: {token soup...}
long needle recall middle: target code LCJSON-MIDDLE-271828 missing from content: {token soup...}
long needle recall end: target code LCJSON-OMEGA-161803 missing from content: {token soup...}
multi-needle strict JSON recall: assistant content is not strict JSON
structured long generation: only 27 numbered lines; expected at least 32
valid near-boundary context: valid boundary response missing BOUNDARY_OK
Server log: no WARN/ERROR entries
```

Log path:

```text
/tmp/llaminar_e2e_long_context_qwen35_moe_rocm_release_nographs_2k/20260522_205252_Qwen3.5-35B-A3B-UD-Q4_K_XL_rocm_0_port20151.log
```

### Recent Manual Probe

The user recently attempted:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_MOE_REBALANCE=off \
LLAMINAR_GPU_GRAPHS=0 \
LLAMINAR_PREFILL_GRAPH_BUCKETS=0 \
build_v2_release/llaminar2 serve \
  --port 20190 \
  -c 2048 \
  -d rocm:0 \
  --kv-cache-precision q16_1 \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

It exited with code 1. Re-run and capture the log before drawing conclusions;
the terminal output was not available in the current context.

## Suggested Next Steps

1. Treat the raw no-graph 2k ROCm command above as the primary correctness repro.
   Do not start with bucketed 4k; it is currently confounded by VRAM pressure.

2. Build a faster targeted repro around the first failing long recall prompt.
   The helper request is deterministic; use `long_context_checks.py` or extract
   its generated messages to send one chat completion instead of the full E2E
   suite.

3. Bisect model stages using existing debug tools:

   - `LLAMINAR_STAGE_OUTPUT_PRINT=1`
   - `LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=...`
   - `LLAMINAR_STAGE_DUMP_ENABLED=1`
   - parity CSV tooling under `tests/v2/integration/parity/` if practical

   Compare CPU versus ROCm at the first layer where long-prefill activations
   diverge. Start with Qwen3.5-specific layers:

   - FA K/Q norm and RoPE-on-read interaction
   - GDN projection / short-conv / recurrence
   - MoE router and grouped expert prefill
   - concurrent prefill projection scratch allocation and coherence

4. Re-check env toggles that were already tried by the previous subagent before
   repeating them:

   - `LLAMINAR_ROCM_CONCURRENT_PREFILL=0` still failed
   - `LLAMINAR_SYNC_AFTER_STAGE=1` still failed
   - `LLAMINAR_ROPE_ON_READ=0` still failed
   - conservative VNNI prefill variant still failed
   - grouped-MoE fallback experiment still failed
   - first-four-layers-on-CPU placement still failed

5. Keep the Qwen3.5 FA graph-ordering fix unless a better dependency model is
   found. It is independently correct even though it did not fix the full ROCm
   failure.

6. After fixing raw ROCm long-prefill correctness, revisit bucketed ROCm 4k:

   - Expect VRAM pressure on a 32GB `rocm:0` card.
   - Consider a smaller context, a smaller bucket list, streaming/eviction, or
     a larger/multi-GPU target for proving bucketed 35B MoE at 4k.
   - Re-run with `LLAMINAR_PREFILL_GRAPH_TRACE=1` to prove graph lifecycle.

## Acceptance Criteria For Next Agent

Minimum acceptance:

- CPU Qwen3.5 35B MoE long-context E2E remains green.
- ROCm Qwen3.5 35B MoE no-graph/no-bucket 2k long-context E2E passes, or a
  narrower equivalent deterministic long-prefill recall test passes and the full
  E2E failure mode is explained.
- No server WARN/ERROR entries in the successful ROCm correctness run.
- Focused tests for any touched ROCm kernel/stage path are added and passing.

Stretch acceptance:

- Bucketed ROCm Qwen3.5 35B MoE long-context run proves graph capture/replay on
  a feasible context/bucket/memory configuration.
```
