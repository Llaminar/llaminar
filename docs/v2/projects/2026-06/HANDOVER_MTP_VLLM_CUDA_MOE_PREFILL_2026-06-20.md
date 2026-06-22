# Handover: vLLM-Style MTP And CUDA MoE Prefill State

Date: 2026-06-20
Branch: `feat/qwen35-moe`
Workspace: `/workspaces/llaminar`

## Immediate Context

The active project is driven by:

- `docs/v2/projects/2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md`
- `docs/v2/projects/2026-06/MTP_VLLM_STYLE_TUNING_DASHBOARD.md`

The older unified prefix-cache/MTP plan is background only. Keep the dashboard
compact and current after every meaningful correctness or benchmark slice.

The latest thread focused on CUDA Qwen3.6 MoE bucketed prefill graph lifecycle:
making request reset preserve replay-safe lazy resources and executable graphs
without preserving stale request-local state.

## Current Answer To The Lifecycle Question

A prefill graph system that must run a full eager warmup for every request is
unlikely to be TTFT-positive. That is not the lifecycle we want, and after the
latest patch it is not the lifecycle being exercised by the CUDA MoE E2E lane.

The intended/current lifecycle is:

1. Cold request warms lazy resources once.
2. `clear_cache()` drops request-local state but preserves lazy resource/device
   storage identity needed by capture-safe stages.
3. A same-key initialized entry can capture from `Initialized` without another
   full eager prefill warmup.
4. A `Ready` entry replays directly after request reset after dynamic metadata is
   restamped.

CUDA/HIP graph capture records the host `execute()` calls; kernels do not run
until graph launch-after-capture. So the first capture request pays graph
capture/instantiate plus one graph launch, not eager-prefill plus graph-prefill.

## Latest Root Cause And Fix

Original failing repro:

- Model: `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`
- Device: `cuda:0`
- Context: `4096`
- Long-context tier: `lite`
- Symptom: long needle middle/end generated degenerate repeated text.
- First 20-run repro root:
  `/tmp/llaminar_e2e_q36_moe_cuda_prefill_reset_map_20_20260620_125716`
- Failure: run 09.

Control run:

- `LLAMINAR_PREFILL_GRAPH_BUCKETS=0` passed all content checks.
- Only the perfstats assertion about prefill graph counters failed, as expected.
- This isolated the issue to bucketed prefill graph lifecycle.

Root cause found after the first conservative warmup-drop fix:

- `ForwardGraphTypes::resetSessionStatePreservingSegmentedReplay()` preserved
  ready prefill replay state, but many stages still used the default preserving
  hook, which called full `resetSessionState()`.
- Full reset cleared dynamic device-param storage, verifier workspace bindings,
  scalar workspace identities, or host mirrors that a preserved executable graph
  expected to remain structurally stable.
- Preserving the executable while clearing the wrong stage state could produce
  stale/corrupt dynamic metadata under CUDA MoE bucketed prefill replay.

Fix:

- Added stage-specific preserving hooks for request reset. The hooks clear
  request-local mirrors and counters while preserving lazy initialization and
  graph-stable workspace/device storage identity.
- Covered stages include `EmbeddingStage`, `RoPEStage`,
  `AttentionComputeStage`, `KVCacheAppendStage`, `GDNRecurrenceStage`,
  `ShortConv1dStage`, `MoERoutingStage`, and `SharedExpertGateStage`.
- The source-level guard
  `PrefillReplayStagesPreserveCapturedResetState` now checks that replay-critical
  stages override the preserving reset hooks.

## CUDA Attention Allocation Cleanup

The CUDA attention wrapper no longer uses ad hoc host-pinned allocation for
dynamic attention params:

- Removed `cudaMallocHost` / `cudaFreeHost` from
  `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.*`.
- FP32 decode uses fixed in-object staging:
  `std::array<attention::AttentionDeviceParams, kMaxDynamicAttentionParamRows>`.
- FP16/BF16 decode uses in-object staging plus the declared
  `AttentionWorkspaceBuffers::DEVICE_PARAMS` workspace buffer.
- H2D dynamic-param uploads are refused during graph capture.
- `CUDAFlashAttentionDecodePartialsUseWorkspace` now guards against regressing
  to host-pinned ad hoc allocation.

## Validation Run

Focused build targets:

```bash
cmake --build build_v2_integration --parallel --target \
  v2_test_gpu_workspace_allocation_policy \
  v2_test_prefill_graph_cache_execution_cuda \
  v2_test_prefill_graph_cache_execution_rocm \
  v2_integration_cuda_moe_kernel \
  v2_integration_multi_turn_session_reset
```

Result: passed.

Focused correctness CTest:

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Unit_GpuWorkspaceAllocationPolicy$|^V2_Integration_PrefillGraphCacheExecution_(CUDA|ROCm)$|^V2_Integration_CUDAMoEKernel$|^V2_Integration_MultiTurnSessionReset$" \
  --output-on-failure --parallel
```

Result: passed `6/6`.

Additional prefill/attention checks:

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Unit_(PrefillGraphCache|ForwardGraphTypes)$|^V2_Integration_CUDAFlashAttention_DecodeKVLen$|^V2_Integration_CUDAFlashAttentionParity$" \
  --output-on-failure --parallel
```

Result:

- `V2_Unit_ForwardGraphTypes`: passed.
- `V2_Unit_PrefillGraphCache`: passed.
- `V2_Integration_CUDAFlashAttention_DecodeKVLen`: passed.
- `V2_Integration_CUDAFlashAttentionParity`: one tight FP16 continuation
  relative-L2 failure remains around `1.4e-5` vs `1e-5` with cosine `1.0` and
  max error below `7e-7`. The cached-count test passed when rerun focused.

Release executable:

```bash
cmake --build build_v2_release --parallel --target llaminar2
```

Result: passed.

CUDA MoE long-context E2E loop:

- Root:
  `/tmp/llaminar_e2e_q36_moe_cuda_prefill_preserve_ready_hooks_20_20260620_161741`
- Command shape: 20 runs of
  `tests/v2/e2e/server/test_server_e2e.sh` against Qwen3.6 MoE on `cuda:0`,
  context `4096`, long-context tier `lite`, max tokens `512`.
- Result: `20/20` runs passed. Each run passed `32/32` checks, including long
  needle beginning/middle/end, multi-needle JSON recall, structured long
  generation, cache reset, near-boundary context, oversized rejection,
  perfstats artifact capture, clean server log, and VRAM release.

Perfstats lifecycle sanity from the 20-run E2E:

- Each run recorded:
  - `1280` warmup,
  - `1536` capture from `lazy_initialized_after_request_reset`,
  - `1536` ready replay,
  - a separate `1536` warmup,
  - `4096` warmup.
- This proves the clean E2E run exercised the intended initialized-capture and
  ready-replay path, not a warmup-only fallback.
- `1536` launch-after-capture averaged `691 us`; ready replay averaged `288 us`.

## Prefill Graph Performance Gauge

Benchmark mode A/B used Qwen3.6 MoE on `cuda:0`, benchmark default prompt
(`595` prefill tokens) and `128` decode tokens. Each repetition is benchmark
mode's one warmup plus three measured iterations. Root:

`/tmp/llaminar_bench_prefill_graph_ab_20260620_163052`

Graph on (`LLAMINAR_PREFILL_GRAPH_BUCKETS` default):

- Prefill latency / TTFT proxy: `192.429 ms` mean.
- Prefill throughput: `3092.044 tok/s` mean.
- Decode throughput observed in this benchmark mode: `148.122 tok/s` mean.
- Overall throughput: `684.284 tok/s` mean.
- Perfstats showed ready replay for bucket `600`.

Graph off (`LLAMINAR_PREFILL_GRAPH_BUCKETS=0`):

- Prefill latency / TTFT proxy: `226.022 ms` mean.
- Prefill throughput: `2632.499 tok/s` mean.
- Decode throughput observed in this benchmark mode: `84.664 tok/s` mean.
- Overall throughput: `416.021 tok/s` mean.
- Perfstats showed no prefill graph lifecycle or launch records.

Mean graph-on delta:

- Prefill latency: `-14.86%`.
- Prefill throughput: `+17.46%`.
- Overall throughput: `+64.48%`.

Treat prefill latency/throughput as the direct signal. The decode/overall
improvement is repeatable in this benchmark slice but needs a separate decode
isolation pass before attributing it solely to prefill graph capture.

## Important Recent Code Paths

Recent touched areas include more than this lifecycle fix. The dirty tree has
many active changes from the broader MTP/E2E work. Do not assume every modified
file belongs to the last slice.

Inspect these first:

- `src/v2/execution/local_execution/engine/PrefillGraphCache.*`
- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- `src/v2/execution/compute_stages/stages/*Stage.h`
- `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.*`
- `src/v2/kernels/IMoEKernel.*`
- `src/v2/kernels/cuda/moe/CUDAMoEKernel.*`
- `src/v2/kernels/cuda/moe/CUDAMoEKernels.cu`
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.*`
- `src/v2/kernels/rocm/moe/ROCmMoEKernels.hip`
- `tests/v2/unit/stages/Test__GpuWorkspaceAllocationPolicy.cpp`
- `tests/v2/e2e/server/test_server_e2e.sh`
- `tests/v2/e2e/server/long_context_checks.py`

## Current Dirty Tree Warning

At handover time, `git status --short` shows a large dirty tree, including docs,
MTP runner/orchestrator changes, KV cache changes, CUDA/ROCm MoE changes, E2E
scripts, parity tests, and a new untracked unit:

- `tests/v2/unit/utils/Test__MPIBootstrap.cpp`

Before committing, review scope carefully. The next commit should either be a
deliberate broad checkpoint after gates pass or a smaller staged commit if the
user asks for one. Do not revert unrelated dirty files.

## Next Recommended Work

1. Decide whether to tighten or bless the remaining CUDA FP16 attention parity
   threshold wobble (`~1.4e-5` relative L2 with cosine `1.0`).
2. Run a decode-isolation benchmark if the graph-on/off decode throughput delta
   matters for attribution.
3. Continue MoE verifier graph replay time and grouped-kernel economics only
   after strict parity gates remain green.
4. Keep Doxygen-style file/method headers and junior-friendly inline comments
   on all touched code.
5. Update the dashboard after every fresh benchmark or correctness finding.

## Guardrails For The Next Agent

- Use `.agents/mtp-tuning/SKILL.md` for this work.
- Use `.agents/cuda-tuning/SKILL.md` or `.agents/rocm-tuning/SKILL.md` for
  backend kernel profiling/tuning.
- Never use CUDA/HIP null/default streams.
- Never add GPU hot-path allocations outside declared workspace/arena patterns.
- Never capture H2D copies in GPU graphs.
- Never mutate tensor residency flags directly; use `TransferEngine` unless the
  stage buffer contract already provides resident device pointers.
- Token equality is not enough for verifier/kernel parity. Use strict relative
  L2, cosine, KLD, max_abs, and continuation checks where applicable.
- If CUDA has a deep test, ROCm and CPU should have symmetrical semantic
  coverage unless the project plan explicitly marks the lane unsupported.

## Useful Commands

Focused lifecycle checks:

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Unit_GpuWorkspaceAllocationPolicy$|^V2_Integration_PrefillGraphCacheExecution_(CUDA|ROCm)$|^V2_Integration_CUDAMoEKernel$|^V2_Integration_MultiTurnSessionReset$" \
  --output-on-failure --parallel
```

Release build:

```bash
cmake --build build_v2_release --parallel --target llaminar2
```

CUDA MoE prefill 20-run repro:

```bash
ROOT=/tmp/llaminar_e2e_q36_moe_cuda_prefill_preserve_ready_hooks_20_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"
for i in $(seq -w 1 20); do
  RUN_DIR="$ROOT/run_$i"
  mkdir -p "$RUN_DIR"
  LLAMINAR_E2E_LOG_DIR="$RUN_DIR" \
  LLAMINAR_E2E_LONG_CONTEXT=1 \
  LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
  LLAMINAR_E2E_CONTEXT_LENGTH=4096 \
  LLAMINAR_E2E_LONG_MAX_TOKENS=512 \
  LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS=900 \
  LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=1800 \
  LLAMINAR_E2E_LONG_MIN_MODEL_SIZE_B=4 \
  tests/v2/e2e/server/test_server_e2e.sh \
    --binary build_v2_release/llaminar2 \
    --suite '/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf|cuda:0|200||qwen36-moe-prefill-preserve-ready-hooks|'
done
```

Graph on/off benchmark gauge:

```bash
ROOT=/tmp/llaminar_bench_prefill_graph_ab_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT"
LLAMINAR_LOG_LEVEL=ERROR \
LLAMINAR_PERF_STATS_JSON="$ROOT/on.perfstats.json" \
build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d cuda:0 \
  --benchmark-json-output "$ROOT/on.json"

LLAMINAR_LOG_LEVEL=ERROR \
LLAMINAR_PREFILL_GRAPH_BUCKETS=0 \
LLAMINAR_PERF_STATS_JSON="$ROOT/off.perfstats.json" \
build_v2_release/llaminar2 benchmark \
  -m /opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -d cuda:0 \
  --benchmark-json-output "$ROOT/off.json"
```
