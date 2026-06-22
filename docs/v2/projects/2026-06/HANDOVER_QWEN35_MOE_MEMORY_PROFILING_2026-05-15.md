# Handover: Qwen3.5 35B MoE CPU Expert-Parallel Memory Profiling

Date: 2026-05-15
Workspace: `/workspaces/llaminar`
Primary model: `models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
Primary command mode: `llaminar2 serve -d cpu --moe-expert-mode expert-parallel`

## Executive Summary

The Qwen3.5 35B MoE CPU server now runs in expert-parallel mode across two CPU MPI ranks, serves sane responses, coordinates dynamic MoE hot-replica rebalancing in `serve`, and no longer re-sends the same 50 hot replicas every rebalance window. The remaining problem is memory: the process still reports far too much per-rank RSS/PSS.

Observed memory from the latest release server runs:

- At `[ServerReady]`: about 65.4 GB RSS per rank, about 54.4 GB PSS per rank, about 43.4 GB anonymous per rank.
- After requests and hot-replica activity: about 47.7 GB RSS/PSS per rank, about 47.5 GB anonymous per rank.

The target mental model is much lower:

- 11 GB per rank base weights for a 22 GB model split across two ranks.
- About 1.1 GB for 10 percent hot replicated experts.
- 3 to 4 GB for buffers.
- About 3 GB for heap/runtime.

So the next question is not whether expert-parallel routing works. It does. The question is where the remaining 25 to 30 GB per rank of mostly anonymous memory is coming from, and which raw-weight or prepared-weight references are still keeping data alive.

## Follow-up Result: Raw Cache Leak Fixed, Eager VNNI Is Now The Main Bucket

Follow-up profiling added aggregate memory accounting for `PreparedWeightStore` and `WeightManager`, removed raw serialization fallbacks, made CPU VNNI expert engines fully eager, and released both stage-owned and `WeightManager`-cached MoE `_exps.weight` parents immediately after eager graph-build packing when `--moe-release-raw-expert-weights` is enabled.

Key result: the missing duplicated raw expert cache was found and removed. Before the fix, `WeightManager` retained about 9.4 GB in `cache_` and another 9.4 GB in `per_device_cache_` per rank after base preparation. After eager graph build, the patched run logs:

```text
[WeightManager] Released cached MoE expert raw data: 120 tensors (9329 MB) released, 120 already released, 0 borrowed views skipped, 0 errors
[WeightManager] Host memory summary (after eager graph build raw release): heap=161 MB mmap=0 MB released=240 borrowed_views=744 | cache=80/0 MB | per_device=80/0 MB
[DGO] Released raw expert weights after eager graph build across 40 MoE stages: 9329 MB stage-owned + 9329 MB WeightManager-cached freed
```

New representative memory, release build, expert-parallel, `--moe-release-raw-expert-weights`:

```text
Hot cache 10 percent, after long prompt and hot replicas:
rank0 Rss 28691156 kB, Pss 28591841 kB, Private_Dirty 28437404 kB, Anonymous 28425040 kB
rank1 Rss 28746100 kB, Pss 28646853 kB, Private_Dirty 28496028 kB, Anonymous 28483668 kB

Hot cache disabled (--moe-hot-expert-cache 0), after first request:
rank0 Rss 25018972 kB, Pss 24922374 kB, Private_Dirty 24773588 kB, Anonymous 24761216 kB
rank1 Rss 25013008 kB, Pss 24916521 kB, Private_Dirty 24766864 kB, Anonymous 24754496 kB
```

Current allocation breakdown from logs and `/proc/<pid>/smaps`:

- `PreparedWeightStore` after eager graph build, no hot replicas: about 19.4 GB per rank (`gemm=1170 MB`, `experts=18240 MB`).
- `PreparedWeightStore` after 10 percent hot replica activity: about 22.97 GB per rank (`experts=21802 MB`).
- `WeightManager` host tensor cache after raw release: about 161 MB per rank.
- No-hot-cache post-request `smaps`: about 21.4 GB anonymous `rw-p`, 2.7 GB `[heap]`, no GGUF file pages.
- 10 percent hot cache adds about 3.7 GB PSS per rank over no-hot-cache.

Conclusion: the dominant remaining memory is no longer raw GGUF or raw expert slice retention. It is the eager CPU VNNI packed expert representation. For this Q4_K model, eager packed expert engines roughly double routed expert storage: local raw expert slices were about 9.3 GB per rank, while local eager VNNI expert engines are about 18.2 GB per rank before hot replicas. Reaching around 20 GB per rank with fully eager VNNI likely requires reducing the packed representation itself, reducing or disabling hot replicas, or accepting less runtime/allocator headroom.

Latest follow-up artifacts:

- Hot-cache patched log: `/tmp/llaminar_serve_qwen35_cache_rawrelease_20260515_103402.log`
- Hot-cache ready memory: `/tmp/llaminar_serve_qwen35_cache_rawrelease_ready_20260515_103657.txt`
- Hot-cache post-prompt memory: `/tmp/llaminar_serve_qwen35_cache_rawrelease_after_20260515_103842.txt`
- No-hot-cache patched log: `/tmp/llaminar_serve_qwen35_nohot_cache_rawrelease_20260515_105314.log`
- No-hot-cache ready memory: `/tmp/llaminar_serve_qwen35_nohot_ready_20260515_105501.txt`
- No-hot-cache post-request memory: `/tmp/llaminar_serve_qwen35_nohot_after_20260515_105644.txt`

Tests/builds rerun after the follow-up patch:

```bash
cmake --build build_v2_integration \
  --target \
    v2_test_moe_rebalance_controller \
    v2_test_moe_expert_compute_stage \
    v2_test_moe_expert_weight_service \
    v2_test_moe_phase_c_store_resolution \
    v2_test_qwen35moe_graph \
  --parallel

ctest --test-dir build_v2_integration \
  -R '^(V2_Unit_MoERebalanceController|V2_Unit_MoEExpertComputeStage|V2_Unit_MoEExpertWeightService|V2_Unit_MoEPhaseCStoreResolution|V2_Unit_Qwen35MoEGraph)$' \
  --output-on-failure --parallel

cmake --build build_v2_release --target llaminar2 --parallel
```

Result: focused tests passed; release `llaminar2` linked. Existing warnings remain the known `StageBufferContract` `-Wstringop-overflow` and KV-cache deprecation warnings.

## Current State To Preserve

Do not revert the dirty worktree. It contains several related fixes from this thread and adjacent work.

Important recent changes already made:

- CLI fail-fast: `--moe-expert-mode tensor-parallel` is rejected before model load.
- CPU expert-parallel startup: rank 0 owns experts `[0,128)`, rank 1 owns `[128,256)`.
- Server path now calls the MoE rebalance hook after direct `decodeStep()` calls.
- MPI worker loop has an `APPLY_MOE_REBALANCE` command so both ranks enter replica transfer together.
- CPU rebalance no longer falls back to raw repacking for missing experts. It requires store-owned engines or transferred packed blobs.
- Hot-replica transfer now sends only newly arrived replicas. Live smoke showed `50 -> 24 -> 16`, not `50 -> 50 -> 50`.
- `tests/v2/unit/moe/Test__MoEPhaseC_StoreResolution.cpp` is now registered in `tests/v2/CMakeLists.txt`.
- `MoEExpertComputeStage` now fails explicitly when active CPU experts lack prepared GEMM engines.

Relevant files touched in this area:

- `src/v2/execution/moe/MoEExpertWeightService.cpp`
- `src/v2/execution/moe/MoEExpertWeightService.h`
- `src/v2/execution/moe/MoERebalanceController.cpp`
- `src/v2/execution/moe/MoERebalanceController.h`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h`
- `src/v2/execution/runner/IOrchestrationRunner.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `tests/v2/unit/moe/Test__MoEExpertWeightService.cpp`
- `tests/v2/unit/moe/Test__MoEPhaseC_StoreResolution.cpp`
- `tests/v2/unit/moe/Test__MoERebalanceController.cpp`
- `tests/v2/unit/stages/Test__MoEExpertComputeStage.cpp`
- `tests/v2/unit/app/modes/Test__ChatCompletionHandler.cpp`
- `tests/v2/unit/app/Test__Commands.cpp`

There are also broader dirty files in config, commands, tokenizer, server E2E, and Qwen35 schema/graph code. Check `git status --short` before editing.

## Build And Test Commands Already Used

Use full parallelism. The project instructions explicitly say not to limit build or test parallelism.

Focused integration build/test gate:

```bash
cmake --build build_v2_integration \
  --target \
    v2_test_moe_rebalance_controller \
    v2_test_moe_expert_compute_stage \
    v2_test_moe_expert_weight_service \
    v2_test_moe_phase_c_store_resolution \
    v2_test_qwen35moe_graph \
    v2_test_chat_completion_handler \
    v2_test_commands \
  --parallel

ctest --test-dir build_v2_integration \
  -R '^(V2_Unit_MoERebalanceController|V2_Unit_MoEExpertComputeStage|V2_Unit_MoEExpertWeightService|V2_Unit_MoEPhaseCStoreResolution|V2_Unit_Qwen35MoEGraph|V2_Unit_ChatCompletionHandler|V2_Unit_Commands)$' \
  --output-on-failure --parallel
```

Last result: all 8 tests passed. Build warnings still include the existing `StageBufferContract` `-Wstringop-overflow` warning from `MoEExpertComputeStage::bufferContract()` and existing KV-cache deprecation warnings.

Release binary build:

```bash
cmake --build build_v2_release --target llaminar2 --parallel
```

## How To Run The 35B MoE Server

Recommended baseline command:

```bash
serve_log="/tmp/llaminar_serve_qwen35_mem_$(date +%Y%m%d_%H%M%S).log"
printf '%s\n' "$serve_log" > /tmp/llaminar_serve_qwen35_latest_log

LLAMINAR_LOG_LEVEL=INFO \
./build_v2_release/llaminar2 serve \
  -d cpu \
  --moe-expert-mode expert-parallel \
  --moe-hot-expert-cache 10% \
  --moe-rebalance dynamic \
  --moe-rebalance-window 32 \
  --host 127.0.0.1 \
  --port 18080 \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  > "$serve_log" 2>&1
```

This command intentionally launches through the MPI bootstrap wrapper. Do not use `--no-mpi-bootstrap` for memory/performance measurements unless you are deliberately debugging a single process; it changes binding/NUMA behavior.

Wait for readiness:

```bash
serve_log=$(cat /tmp/llaminar_serve_qwen35_latest_log)
rg -n 'ServerReady|expert_range|max_replicas|ERROR|FATAL' "$serve_log" | tail -80
curl -sS --max-time 5 http://127.0.0.1:18080/health
```

Expected startup log lines:

```text
[InferenceRunner] MoE expert mode=expert-parallel participant=0/2 expert_range=[0, 128) count=128/256
[InferenceRunner] MoE expert mode=expert-parallel participant=1/2 expert_range=[128, 256) count=128/256
[InferenceRunner] MoE rebalance controller: mode=dynamic max_replicas=25 hot_cache=10% window=32 experts=256
[ServerReady] VmRSS:       65377412 kB
[ServerReady] RssAnon:     43397320 kB
```

Sanity prompt:

```bash
curl -sS --max-time 240 http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35moe","messages":[{"role":"user","content":"Answer in one short sentence: what is 12 + 30?"}],"max_tokens":48,"temperature":0,"enable_thinking":false}' \
  | jq -r '.choices[0].message.content // .error.message // .'
```

Expected output:

```text
12 plus 30 equals 42.
```

Long prompt that crosses multiple rebalance windows:

```bash
curl -sS --max-time 700 http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35moe","messages":[{"role":"user","content":"Count from 1 to 100, separated by commas. Do not add commentary."}],"max_tokens":190,"temperature":0,"enable_thinking":false}' \
  -o /tmp/llaminar_serve_qwen35_long_response.json
```

Check hot replica behavior:

```bash
serve_log=$(cat /tmp/llaminar_serve_qwen35_latest_log)
rg -n 'Expert replication|Hot expert replica set unchanged|newly-arrived hot replicas|Transferring .*replicated experts|Applied expert masks|Set expert replica|ERROR|FATAL' "$serve_log" | tail -220
```

Latest verified behavior:

```text
[DGO] Transferring 50 replicated experts x 40 layers via MPI
[MoE] Transferring 24 newly-arrived hot replicas; 26 already resident
[DGO] Transferring 24 replicated experts x 40 layers via MPI
[MoE] Transferring 16 newly-arrived hot replicas; 34 already resident
[DGO] Transferring 16 replicated experts x 40 layers via MPI
```

## Memory Snapshot Commands

Use `smaps_rollup`; RSS alone is misleading because mmap file-backed pages drop after `DONTNEED`.

```bash
mem_stamp="/tmp/llaminar_serve_qwen35_memory_$(date +%Y%m%d_%H%M%S).txt"
printf 'snapshot=manual\n' > "$mem_stamp"
ps -eo pid,ppid,rss,vsz,cmd | rg 'llaminar2|mpirun' >> "$mem_stamp"
printf '\nsmaps_rollup\n' >> "$mem_stamp"
for pid in $(pgrep -f 'build_v2_release/llaminar2 serve|llaminar2 serve' | sort -n); do
  if [ -r "/proc/$pid/smaps_rollup" ]; then
    printf 'pid=%s\n' "$pid" >> "$mem_stamp"
    awk '/^(Rss|Pss|Private_Dirty|Anonymous|RssFile|Shared_Clean|Private_Clean):/ {print}' "/proc/$pid/smaps_rollup" >> "$mem_stamp"
  fi
done
cat "$mem_stamp"
```

Useful extra maps breakdown:

```bash
pid=$(pgrep -f 'build_v2_release/llaminar2 serve' | sort -n | head -1)
awk '
  /^[0-9a-f]+-/ {name=$0}
  /^Size:/ {size=$2}
  /^Rss:/ {rss=$2}
  /^Pss:/ {pss=$2}
  /^Anonymous:/ {anon=$2}
  /^VmFlags:/ {
    if (rss > 1024*128) printf "%10d KB rss %10d KB pss %10d KB anon  %s\n", rss, pss, anon, name
    size=rss=pss=anon=0
  }' /proc/$pid/smaps | sort -nr | head -40
```

That command should help separate large anonymous heaps from file-backed GGUF mappings, OpenMP stacks, GPU/driver mappings, and allocator arenas.

## Latest Artifact Paths

The latest artifacts from this handoff session are still in `/tmp`:

- Latest no-retransfer server log: `/tmp/llaminar_serve_qwen35_noretransfer_20260515_064235.log`
- Latest no-retransfer response: `/tmp/llaminar_serve_qwen35_noretransfer_response_20260515_064436.json`
- Latest no-retransfer post-request memory: `/tmp/llaminar_serve_qwen35_noretransfer_after_20260515_064654.txt`
- Earlier ready memory: `/tmp/llaminar_serve_qwen35_memory_ready_retry_20260515_060227.txt`
- Earlier post-prompt memory: `/tmp/llaminar_serve_qwen35_memory_after_retry_20260515_060755.txt`
- Tensor-parallel fail-fast log: `/tmp/llaminar_phase0_tensor_parallel_failfast_retry_20260515_061143.log`

Representative snapshots:

```text
READY, release build, expert-parallel, hot-cache 10 percent:
rank0 Rss 65375044 kB, Pss 54419109 kB, Private_Dirty 43409816 kB, Anonymous 43397360 kB
rank1 Rss 65476536 kB, Pss 54520625 kB, Private_Dirty 43508688 kB, Anonymous 43496304 kB

AFTER prompts and hot replicas:
rank0 Rss 47745320 kB, Pss 47645071 kB, Private_Dirty 47491624 kB, Anonymous 47479260 kB
rank1 Rss 47789824 kB, Pss 47689877 kB, Private_Dirty 47536136 kB, Anonymous 47523776 kB
```

Interpretation:

- `ServerReady` still has about 20 GB of file-backed/mmap RSS per rank in addition to about 43 GB anonymous.
- After first real request, the file-backed mmap pages are mostly gone and RSS drops to about 47.7 GB.
- The remaining problem is almost entirely anonymous/private memory.

## What We Know About Release And MADV

Whole-model mmap reclaim is being called. Latest log includes:

```text
[ModelLoader] Advised DONTNEED on mmap regions (21211 MB) - pages reclaimable by OS
```

This line appears during the first request, not at `[ServerReady]`. That explains why ready RSS is about 65 GB and later RSS drops to about 47 GB.

MoE expert parent release is separate and is not enabled in the baseline smoke. There was no log line like:

```text
[MoE] Released ... MB raw expert weights
```

The control points are:

- CLI: `--moe-release-raw-expert-weights`
- Env: `LLAMINAR_MOE_RELEASE_RAW_WEIGHTS=1`
- Runtime config field: `moe_rebalance.release_raw_expert_weights`
- Runner gate: `OrchestrationRunner::applyMoERebalanceWithReplicas()`
- DGO method: `DeviceGraphOrchestrator::releaseRawExpertWeights()`
- Stage method: `MoEExpertComputeStage::releaseRawExpertWeights()`
- Service method: `MoEExpertWeightService::releaseRawWeights()`

Important caveat: the optional expert raw release currently happens inside the rebalance path after masks/replicas are applied. It may not happen immediately after initial graph materialization unless a rebalance window is hit. If the goal is to lower memory before or during the first request, this timing probably needs attention.

## Code Paths To Inspect First

### 1. Expert raw tensor retention

Start in `src/v2/execution/moe/MoEExpertWeightService.cpp`:

- `prepareGemmEngines()` registers CPU expert slabs in `PreparedWeightStore`.
- For store-backed CPU prep, it clears `ctx.moe_owned_kernels` when the store owns all registered engines.
- It calls `MmapRegion::adviseDontneedRange()` only when `ctx.gate_exps/up_exps/down_exps->is_mmap_data()` is true.
- Expert-parallel sliced tensors are likely heap-allocated copies, not mmap data, so this DONTNEED path may not free them.
- `releaseRawWeights()` frees heap-backed 3D parent data by calling `tensor->release_raw_data()`, then nulls the parent pointers in the stage context.

Hypothesis: expert-parallel per-rank sliced 3D expert tensors are heap data held by each `MoEExpertComputeStage` through `params_.gate_exps`, `params_.up_exps`, and `params_.down_exps`. Unless `--moe-release-raw-expert-weights` is enabled and actually runs after prep, these heap copies stay anonymous.

### 2. Tensor view references keeping parent tensors alive

`MoEWeightContext` and `MoEExpertComputeStage::Params` hold:

- 3D parents: `gate_exps`, `up_exps`, `down_exps`
- 2D views: `expert_gate_views`, `expert_up_views`, `expert_down_views`
- store entries: `ExpertArrival::view_lifetime`

Check whether 2D views retain the 3D parent even after `params_.gate_exps` is nulled. `TensorSlice::release_raw_data()` forwards to the inner tensor, so this may be safe, but the next agent should verify the lifetime graph. If `ExpertArrival::view_lifetime` keeps a view that keeps the parent tensor object alive, that is okay only if raw storage was released.

Relevant files:

- `src/v2/tensors/TensorSlice.h`
- `src/v2/tensors/TensorClasses.h`
- `src/v2/loaders/PreparedWeightStore.h`
- `src/v2/loaders/PreparedWeightStore.cpp`
- `src/v2/loaders/ExpertSlabTypes.h`

### 3. WeightManager host release gates

Start in `src/v2/loaders/WeightManager.cpp`:

- `releaseAllHostWeightData()` is guarded by lifecycle gates.
- It has special logic `moe_parent_ready_for_release()` for `_exps.weight` parent tensors.
- It logs released/retained counts and retained bytes, but returns `released_count`, not `released_bytes`.

Question: does this path ever release the heap-sliced expert-parallel parents for the CPU `serve` path, or only the original `cache_`/`per_device_cache_` tensors? The latest log does not show a `Released host weight data:` line in the no-retransfer run, so this may not be active.

### 4. PreparedWeightStore duplicate ownership

Earlier duplicate graph ownership was fixed, but verify with DEBUG:

```bash
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2_release/llaminar2 serve ...
rg -n 'Phase B: Registered|Released graph-local shared ownership|Reused .*PreparedWeightStore|expertSlabCount|totalPopulatedExperts' "$serve_log"
```

If graph-local ownership is retained, the service logs a warning:

```text
has no shared engine lifetime; retaining graph-local owner
```

That should not appear for the normal CPU store-backed path.

### 5. Hot replicas and delta transfer

The hot-replica transfer waste is fixed, but hot replicas still increase memory. Latest live result:

- Initial transfer: 50 replicated experts x 40 layers.
- Later windows: only 24 then 16 newly arrived replicas.

The post-hot-replica memory was still about 47.7 GB per rank. Replicas are not the full explanation.

## Suggested Next Experiments

### Experiment A: Run with raw expert release enabled

Use the same command but add the CLI flag:

```bash
LLAMINAR_LOG_LEVEL=INFO \
./build_v2_release/llaminar2 serve \
  -d cpu \
  --moe-expert-mode expert-parallel \
  --moe-hot-expert-cache 10% \
  --moe-rebalance dynamic \
  --moe-rebalance-window 32 \
  --moe-release-raw-expert-weights \
  --host 127.0.0.1 \
  --port 18080 \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  > "$serve_log" 2>&1
```

Then run the long prompt. Check:

```bash
rg -n 'Released .*raw expert|releaseRawExpertWeights|released .*heap data|DONTNEED|ERROR|FATAL' "$serve_log"
```

Expected useful outcome:

- If memory drops significantly, make raw expert release safe/default for this CPU expert-parallel path after initial materialization and before/after hot-replica prep.
- If it crashes, the failure identifies a stale raw-data consumer that must be converted to store/blob/prepared access.
- If it logs `0 MB`, inspect whether the release is running too late, stage contexts do not carry heap parents anymore, or the parent tensors are not recognized/releasable.

### Experiment B: DEBUG run to quantify retained tensors

Use DEBUG only for startup/one request because the logs are large:

```bash
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2_release/llaminar2 serve ...
```

Look for:

```bash
rg -n 'RETAINED host data|Released host weight data|released .*heap data|Advised .*DONTNEED|Phase B: Registered|Released graph-local shared ownership|Reused .*PreparedWeightStore|has no shared engine lifetime' "$serve_log"
```

If the log is too noisy, add temporary INFO-level aggregate counters rather than printing every tensor.

### Experiment C: Add allocator/category accounting around MoE prep

Useful counters to add temporarily:

- Bytes of `gate_exps/up_exps/down_exps` parent tensors per layer/rank.
- Whether each parent is mmap-backed or heap-backed.
- Bytes of `expert_gate_views/up/down` view lifetimes retained in store slabs.
- Bytes of CPU packed engines registered in `PreparedWeightStore` per layer/role.
- `ctx.moe_owned_kernels.size()` before and after store ownership handoff.
- `PreparedWeightStore::expertSlabCount()` and `totalPopulatedExperts()` after initial prep and after replica arrivals.

Good insertion points:

- `MoEExpertWeightService::prepareGemmEngines()` after `experts_to_prep` is built.
- `PreparedWeightStore::registerArrivedExperts()`.
- `DeviceGraphOrchestrator::applyExpertMasks()` before and after phase 1/2.
- `WeightManager::releaseAllHostWeightData()` retained/released aggregate summaries.

### Experiment D: Maps-level classification

After first request, use `/proc/<pid>/smaps` to find the largest anonymous mappings. If they are mostly heap, glibc arenas, or thread stacks, that changes the path. If they correspond to large anonymous `rw-p` mappings sized like expert slices, focus on tensor raw storage.

Suggested quick command is in the memory snapshot section above.

### Experiment E: Compare hot cache off vs on

Run once with `--moe-hot-expert-cache 0` or omit the flag if supported. If post-request RSS only drops by about 1 to 2 GB, hot replicas are not the culprit. If it drops much more, inspect replica arrival lifetime and departure release in `MoEExpertWeightService::releaseDepartedExperts()` and `PreparedWeightStore::releaseDepartedExperts()`.

## Known Good Functional Checks

After any memory trim, rerun:

```bash
ctest --test-dir build_v2_integration \
  -R '^(V2_Unit_MoERebalanceController|V2_Unit_MoEExpertComputeStage|V2_Unit_MoEExpertWeightService|V2_Unit_MoEPhaseCStoreResolution|V2_Unit_Qwen35MoEGraph|V2_Unit_ChatCompletionHandler|V2_Unit_Commands)$' \
  --output-on-failure --parallel
```

Then run the release server sanity prompt and long prompt above.

Also keep the CLI fail-fast check:

```bash
fail_log="/tmp/llaminar_phase0_tensor_parallel_failfast_$(date +%Y%m%d_%H%M%S).log"
set +e
./build_v2_release/llaminar2 oneshot --validate-only \
  -d cpu \
  --moe-expert-mode tensor-parallel \
  -m models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -p test -n 1 >"$fail_log" 2>&1
status=$?
set -e
printf 'status=%s log=%s\n' "$status" "$fail_log"
printf 'has_model_load='; rg -q 'ModelLoader|Loading model|WeightManager] Loading' "$fail_log" && printf 'yes\n' || printf 'no\n'
rg -n 'tensor-parallel|not implemented|ModelLoader|Loading model|WeightManager\] Loading|ERROR|FATAL' "$fail_log" | head -40
```

Expected:

```text
status=1
has_model_load=no
moe-tensor-parallel-experts-not-implemented
```

## Cleanup Commands

Stop a server launched in a terminal, or if needed:

```bash
ps -eo pid,ppid,rss,vsz,cmd | rg 'llaminar2 serve|mpirun -np 2.*llaminar2 serve'
pkill -f 'build_v2_release/llaminar2 serve' || true
pkill -f 'mpirun -np 2 .*llaminar2 serve' || true
```

Prefer clean terminal shutdown when possible. Verify:

```bash
ps -eo pid,ppid,rss,vsz,cmd | rg 'llaminar2 serve|mpirun -np 2.*llaminar2 serve' || true
```

## Current Best Hypotheses

1. The main remaining memory is heap-backed expert-parallel raw slices retained after CPU VNNI repack. This fits the large anonymous memory and the fact that baseline did not enable `--moe-release-raw-expert-weights`.
2. 2D expert views and `PreparedWeightStore` `view_lifetime` may keep parent tensor objects alive. That is acceptable only if raw storage is released; verify `release_raw_data()` actually clears the heap storage for the relevant tensor classes.
3. `WeightManager::releaseAllHostWeightData()` may not be reached for this server path, may be blocked by lifecycle gates, or may not see the same heap-sliced tensors held by MoE stage params.
4. Hot replicas add memory, but the delta-transfer smoke suggests they are not the dominant source. Post-request PSS stayed around 47.7 GB even after transfer optimization.
5. Some anonymous memory may be glibc arenas/OpenMP stacks/runtime overhead, but the scale is too large to assume that without maps-level classification.

## Most Useful Next Step

Run the exact server command with `--moe-release-raw-expert-weights`, cross at least one rebalance window, and compare post-request `smaps_rollup` against the latest no-release baseline. If it drops close to the expected range, make raw expert release part of the normal expert-parallel lifecycle at the earliest safe point. If it does not, add category accounting around MoE parent tensors, store slab ownership, and packed engine bytes before changing more logic.