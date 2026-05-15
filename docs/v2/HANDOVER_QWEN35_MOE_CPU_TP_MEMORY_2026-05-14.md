# Handover: Qwen3.5 35B MoE CPU TP Memory and PreparedWeightStore Fix

Date: 2026-05-14
Branch context: `feat/qwen35-moe`
Primary goal for next agent: continue from the verified packed-engine ownership fix and move on to proper MoE expert-parallel startup for CPU TP. The immediate duplicate packed-engine retention across cached prefill/decode graphs is fixed and verified, but the 35B MoE `-d cpu` server still exceeds the memory target because default CPU TP still prepares all routed experts on every rank.

## Executive Summary

We fixed the immediate lifetime bug where freshly prepared CPU MoE expert GEMM engines were retained by cached graphs even after the `PreparedWeightStore` had registered them. The fix makes `PreparedWeightStore` the sole shared owner for store-backed CPU initial expert engines; cached graphs keep raw engine pointers only.

We also fixed `PreparedWeightStore::clear()` and `releaseAllPreparedState()` so expert slabs are actually released. Before this, store-owned expert engines could survive explicit store cleanup/reset paths.

Focused unit coverage passes, and a real Qwen3.5 35B MoE CPU TP server run confirms:

- The first prefill graph prepares and registers expert engines.
- Graph-local shared ownership is released after registration.
- The cached decode graph reuses all expert engines from `PreparedWeightStore`.
- No second packed-engine preparation occurs for cached decode graph construction.

However, the full server E2E memory check still fails:

```text
Memory: RAM 122910 MiB exceeds limit 88944 MiB
```

This remaining failure is not duplicate prefill/decode packed-engine retention. It is the larger architecture issue: `-d cpu` still uses the old full-expert CPU TP shape, where both CPU ranks prepare all 256 routed experts for all 40 MoE layers. Proper expert-parallel startup is still needed.

## Current Dirty Tree Notes

There are many dirty files from this and previous work. Do not revert unrelated changes.

Intentional files touched by the packed-engine lifetime fix in this slice:

- `src/v2/execution/moe/MoEExpertWeightService.cpp`
- `src/v2/execution/moe/MoEExpertWeightService.h`
- `src/v2/loaders/PreparedWeightStore.cpp`
- `tests/v2/unit/loaders/Test__PreparedWeightStore_ExpertSlab.cpp`
- `tests/v2/unit/moe/Test__MoEExpertWeightService.cpp`

Relevant already-dirty files from adjacent/previous work include:

- `src/v2/loaders/PreparedWeightStore.h`
- `tests/v2/unit/moe/Test__MoEPhaseC_StoreResolution.cpp`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/models/qwen35/Qwen35Graph.*`
- `src/v2/models/qwen35/Qwen35Schema.h`
- server/chat-template files from the CPU TP regression fix

New/important docs currently present:

- `docs/v2/HANDOVER_QWEN35_CPU_TP_REGRESSION_2026-05-14.md`
- `docs/v2/cleanup/QWEN35_MOE_EXPERT_PARALLEL_CLI_PROJECT_PLAN.md`
- This file: `docs/v2/HANDOVER_QWEN35_MOE_CPU_TP_MEMORY_2026-05-14.md`

## Implemented Fixes In This Slice

### 1. PreparedWeightStore Releases Expert Slabs

File:

- `src/v2/loaders/PreparedWeightStore.cpp`

Changes:

- `PreparedWeightStore::clear()` now clears `expert_slabs_`.
- `PreparedWeightStore::releaseAllPreparedState()` now includes `expert_slabs_` in the empty check and clears it.
- Comment updated to document that expert slabs own MoE expert GEMM engines through `ExpertEntry::engine_lifetime`.

Why it matters:

- Store-owned MoE engines must be released on model/store teardown.
- Without this, explicit store cleanup did not actually release MoE packed expert engines.

### 2. Store-Backed CPU MoE Prep Hands Ownership To PreparedWeightStore

Files:

- `src/v2/execution/moe/MoEExpertWeightService.cpp`
- `src/v2/execution/moe/MoEExpertWeightService.h`

Changes:

- After CPU initial expert GEMM engines are registered in `PreparedWeightStore`, `ctx.moe_owned_kernels` is cleared and shrunk when every registered engine has a shared lifetime in the store.
- Raw `ctx.prepared_*_gemm[e]` pointers remain valid because the store owns `engine_lifetime`.
- If any registered engine lacks a shared lifetime, the code keeps graph-local ownership and logs a warning instead of risking dangling pointers.
- Header comments now describe store-backed CPU ownership handoff instead of stale dual-path ownership language.

Why it matters:

- Cached graphs no longer retain an extra shared owner for store-backed CPU MoE packed engines.
- Decode graph caching can keep stage params without extending the engine lifetime independently of the store.

### 3. Regression Tests

Files:

- `tests/v2/unit/loaders/Test__PreparedWeightStore_ExpertSlab.cpp`
- `tests/v2/unit/moe/Test__MoEExpertWeightService.cpp`

New coverage:

- `Clear_ReleasesExpertSlabs`
- `ReleaseAllPreparedState_ReleasesExpertSlabs`
- `PrepareGemmEngines_CPU_StoreOwnsPreparedEngines`

The MoE weight-service test verifies that after store-backed CPU prepare:

- `owner.moe_owned_kernels.empty()` is true.
- `PreparedWeightStore` has 3 expert slabs.
- All gate/up/down raw engine pointers resolve from the store.

## Verification Completed

### Focused Build

```bash
cmake --build build_v2_integration \
  --target v2_unit_prepared_weight_store_expert_slab v2_test_moe_expert_weight_service \
  --parallel
```

Result: passed. Build emitted existing warnings in shared MoE/attention code, but no failure from this slice.

### Focused Unit Tests

```bash
ctest --test-dir build_v2_integration \
  -R "^(V2_Unit_PreparedWeightStoreExpertSlab|V2_Unit_MoEExpertWeightService)$" \
  --output-on-failure --parallel
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

The third test is the model fetch fixture pulled in by CTest dependencies.

### Integration Server Binary Rebuild

```bash
cmake --build build_v2_integration --target llaminar2 --parallel
```

Result: passed.

## Real 35B MoE Server Test

Model used:

```text
models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Model size:

```text
21212 MiB
```

Command:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_35b_moe_tp_mem \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "$PWD/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|cpu|200" \
  --port 19350
```

This uses `-d cpu`, which bootstraps two CPU ranks / CPU shorthand GLOBAL TP across local NUMA nodes.

Functional result:

- Server started.
- `/health` passed.
- Single-turn arithmetic passed in non-thinking and thinking modes.
- Multi-turn recall passed in both modes.
- Cache-clear request passed in both modes.
- Response format passed.
- SSE streaming passed in both modes.
- Invalid JSON and missing messages error handling passed.

Memory/log result:

```text
FAILED: 3/24 tests failed

Memory: RAM 122910 MiB exceeds limit 88944 MiB
GPU memory: CPU backend changed GPU usage (process 512 MiB, global delta 526 MiB)
Server log: 1 WARN/ERROR entry
```

Server-ready RSS from log, before request-triggered MoE expert prep:

```text
[ServerReady] VmRSS:        46778200 kB
[ServerReady] RssAnon:      24802136 kB
```

Server log:

```text
/tmp/llaminar_e2e_35b_moe_tp_mem/20260514_140539_Qwen3.5-35B-A3B-UD-Q4_K_XL_cpu_port19351.log
```

Only WARN line in that log:

```text
[StageVerifier] Stage 'embedding' output 'embeddings' appears to be all zeros (likely uninitialized)
```

No server processes remained after the run:

```bash
ps -eo pid,ppid,rss,vsz,cmd | rg 'llaminar2|mpirun|orted|prte|orterun' || true
```

Result: only the `rg` process itself.

## DEBUG Evidence For Packed-Engine Fix

### Startup-Only DEBUG Run

Command started the same 35B MoE CPU TP server with DEBUG logging, waited for `/health`, then stopped it.

Log:

```text
/tmp/llaminar_debug_35b_moe_startup.log
```

Finding:

- No `MoEExpertWeightService` activity before `/health`.
- Qwen3.5 35B MoE expert prep is lazy and happens on the first request, not at server readiness.

Startup server-ready RSS was consistent with the INFO run:

```text
[ServerReady] VmRSS:       46780412 kB
[ServerReady] RssAnon:     24802456 kB
```

### One-Token DEBUG Request

Log:

```text
/tmp/llaminar_debug_35b_moe_request.log
```

Response:

```json
{"content":"4","finish_reason":"length"}
```

Key observation:

- A one-token request performs prefill but does not force decode graph caching.
- It showed initial expert preparation and store registration, but not decode graph reuse.

### Two-Token DEBUG Request

Log:

```text
/tmp/llaminar_debug_35b_moe_decode.log
```

Response:

```json
{"content":"4","finish_reason":"stop"}
```

Counts from the log:

```text
Preparing:      80
Registered:     80
Released local: 80
Reused:         80
Cached:          2
```

Interpretation:

- `Preparing: 80`: 40 MoE layers x 2 CPU ranks prepare once during prefill.
- `Registered: 80`: all prefill expert engine sets registered in `PreparedWeightStore`.
- `Released local: 80`: graph-local shared ownership was cleared for every initial store-backed CPU expert prep.
- `Reused: 80`: decode graph construction reused every MoE layer/rank expert slab from `PreparedWeightStore`.
- `Cached: 2`: decode graph cached once per rank.

Representative reuse lines:

```text
[MoEWeightService] Reused 768 CPU expert GEMM engines from PreparedWeightStore for layer 0 (slabs=120)
...
[MoEWeightService] Reused 768 CPU expert GEMM engines from PreparedWeightStore for layer 39 (slabs=120)
[ForwardExecutionEngine] Cached forward graph for signature [seq_len=1, batch_size=1, device=CPU, decode=1] (606 stages)
```

This is the main evidence that the duplicate packed-engine retention across cached prefill/decode graphs is fixed.

## Remaining Problem

The remaining memory failure is architectural/default-placement, not decode graph duplication.

Current default behavior in `-d cpu` for Qwen3.5 35B MoE:

- CPU shorthand launches 2 ranks.
- `Qwen35MoEGraph` default legacy routed expert path calls `makeExpertParams(..., expert_mask={}, device=CPU)`.
- `local_expert_count` remains `-1`.
- `MoEExpertWeightService::prepareGemmEngines()` therefore prepares all experts in the local range, which resolves to all 256 experts.
- Both ranks prepare all 256 experts for all 40 layers.

The DEBUG logs prove this current behavior:

```text
[MoEWeightService] Preparing GEMM engines for 256/256 experts (3 weights each = 768 total)...
```

That happens once per layer per rank during prefill.

Expected expert-parallel behavior for `-d cpu` / 2 CPU ranks should become roughly:

```text
rank 0: experts [0, 128)
rank 1: experts [128, 256)
```

Then the logs should show something like:

```text
Preparing GEMM engines for 128/256 experts ...
```

and MoE output should be marked partial so the existing allreduce path combines routed expert contributions.

The current graph already has this partial-output trigger:

```cpp
const bool routed_expert_output_is_partial =
    expert_params.local_expert_count >= 0 || !expert_params.expert_mask.empty();
```

So setting a proper expert range should naturally enable the MoE allreduce stage for the routed expert output.

## Next Recommended Work

Start from:

```text
docs/v2/cleanup/QWEN35_MOE_EXPERT_PARALLEL_CLI_PROJECT_PLAN.md
```

Immediate implementation direction:

1. Add/plumb CLI configuration for MoE routed expert parallelism mode.
   - Default: expert parallelism.
   - Allowed initial values: expert-parallel, tensor-parallel.
   - If tensor-parallel is selected for MoE routed experts, parse it and then fail fast with `Not Implemented` for now.

2. For expert-parallel mode, assign each TP participant a contiguous expert range at startup.
   - For 256 experts and 2 CPU ranks: 128 experts per rank.
   - Handle remainders deterministically.
   - Set `MoEExpertComputeStage::Params::local_expert_start` and `local_expert_count` before `prepareExpertParams()`.

3. Ensure weight loading uses only the local expert range when possible.
   - Existing `MoEExpertWeightService::extractExpertViews()` already has pre-sliced tensor handling:
     - detects `tensor_expert_count != n_experts`
     - maps global expert id to local tensor index with `e - local_start`
   - Existing loader APIs include `loadTensorExpertSlice()`.
   - The next agent should verify whether Qwen35 MoE weight materialization currently loads full 3D expert tensors for each rank, and switch the EP path to load only the local slice.

4. Keep decode graph store reuse intact.
   - After EP range setup, DEBUG evidence should still show decode reuse from `PreparedWeightStore`.
   - Expected count remains 80 reuse events for 40 layers x 2 ranks, but each event should be for the rank-local expert count.

5. Wire bounded hot expert replication.
   - User wants a configurable number of hottest experts replicated per rank.
   - Default should be about 10% of routed expert count.
   - Current env config has `LLAMINAR_MOE_REBALANCE_REPLICAS`; default today is effectively auto/0 and comments mention `2 * top_k`, not the desired 10% cap.
   - CPU-device hot expert cache should be explicitly set and bounded by the dynamic expert rebalancer.

6. Retest the same server command.
   - Functional checks should still pass.
   - Memory target should drop materially below the old full-replication 122910 MiB process-tree PSS.
   - DEBUG logs should no longer show `256/256` on both ranks.

## Useful Commands For Next Agent

Focused unit test rebuild:

```bash
cmake --build build_v2_integration \
  --target v2_unit_prepared_weight_store_expert_slab v2_test_moe_expert_weight_service \
  --parallel

ctest --test-dir build_v2_integration \
  -R "^(V2_Unit_PreparedWeightStoreExpertSlab|V2_Unit_MoEExpertWeightService)$" \
  --output-on-failure --parallel
```

Build server binary:

```bash
cmake --build build_v2_integration --target llaminar2 --parallel
```

Full 35B MoE CPU TP E2E:

```bash
LLAMINAR_LOG_LEVEL=INFO \
LLAMINAR_E2E_LOG_DIR=/tmp/llaminar_e2e_35b_moe_tp_mem \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite "$PWD/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|cpu|200" \
  --port 19350
```

Two-token DEBUG smoke to prove decode reuse:

```bash
rm -f /tmp/llaminar_debug_35b_moe_decode.log /tmp/llaminar_debug_35b_moe_decode.json
LLAMINAR_LOG_LEVEL=DEBUG \
build_v2_integration/llaminar2 serve \
  --port 19363 \
  -d cpu \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  > /tmp/llaminar_debug_35b_moe_decode.log 2>&1 &
server_pid=$!

curl -s --retry 300 --retry-connrefused --retry-delay 1 --max-time 2 \
  http://127.0.0.1:19363/health >/tmp/llaminar_debug_35b_moe_health.json

curl -s --max-time 240 \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"You are a calculator. Reply with only the numeric answer."},{"role":"user","content":"What is 2+2?"}],"max_tokens":2,"enable_thinking":false,"temperature":0.0}' \
  http://127.0.0.1:19363/v1/chat/completions \
  >/tmp/llaminar_debug_35b_moe_decode.json

kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true

printf 'Preparing: '; rg -c "Preparing GEMM engines" /tmp/llaminar_debug_35b_moe_decode.log
printf 'Registered: '; rg -c "Phase B: Registered" /tmp/llaminar_debug_35b_moe_decode.log
printf 'Released local: '; rg -c "Released graph-local" /tmp/llaminar_debug_35b_moe_decode.log
printf 'Reused: '; rg -c "Reused .*PreparedWeightStore" /tmp/llaminar_debug_35b_moe_decode.log
printf 'Cached: '; rg -c "Cached forward graph" /tmp/llaminar_debug_35b_moe_decode.log
```

Clean up check:

```bash
ps -eo pid,ppid,rss,vsz,cmd | rg 'llaminar2|mpirun|orted|prte|orterun' || true
```

## Important Cautions

- Do not use `--no-mpi-bootstrap` for the server memory test. The `-d cpu` shorthand behavior depends on MPI bootstrap creating the two CPU ranks and topology-aware placement.
- Do not revert unrelated dirty files. Several files were already dirty before the packed-engine fix.
- Keep using full build/test parallelism per project instructions.
- The `GPU memory changed` failure in the CPU server E2E may be a separate harness/runtime issue because this integration build has GPU backends enabled. The main CPU memory problem is the 122910 MiB host PSS.
- The StageVerifier embedding all-zero warning appears during the 35B server run, but functional checks pass. Treat it as a separate log-hygiene issue unless it becomes actionable.
