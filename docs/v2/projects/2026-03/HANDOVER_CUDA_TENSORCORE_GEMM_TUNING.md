# CUDA Tensor-Core GEMM Tuning Handover

**Date**: March 13, 2026  
**Branch**: `tensor-parallel`  
**Status**: Active tuning handoff for CUDA native-payload tensor-core prefill GEMM

---

## Executive Summary

This handover is for the CUDA native-payload **tensor-core prefill GEMM** path, with the immediate target of closing the performance gap against CUTLASS for **Q4_0** across the important inference shapes.

The active prefill runtime route is:

1. `CUDAQuantisedGemmKernel` decides whether native-payload blockwise execution is available.
2. For `m == 1`, decode goes through the tuned native-payload GEMV path.
3. For `m > 1`, native-payload prefill currently routes through `cudaTCGemm_blockwiseGemm(...)` in the tensor-core GEMM scaffold.

The recent optimization already landed in the active tensor-core path is **chunked accumulation** of multiple `K=32` partial GEMMs before a single FP32 accumulation step. That reduced launch overhead and required matching workspace logic updates. It improved performance, but it is still materially behind CUTLASS on the target sweep.

The most important constraint for the next agent is this: **prefill work should remain tensor-core-only**. There is a separate custom prefill file in the tree, but it is not the intended active direction for this handoff.

---

## Active Goal

Close the Q4_0 prefill gap with CUTLASS on the production-relevant shape sweep, without regressing correctness or blowing out workspace usage.

Practical success criteria:

- Maintain cosine correctness against CUTLASS on the perf harness.
- Improve native-payload tensor-core prefill throughput for wide, balanced, and TP-split shapes.
- Preserve the active runtime contract in `CUDAQuantisedGemmKernel` for `m > 1`.
- Avoid workspace failures on large shapes, especially LM-head-like outputs.

---

## What Is Actually Active

### Active prefill runtime path

The active prefill runtime entry is in [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](/workspaces/llaminar/src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp#L418):

- `runNativePayloadBlockwiseIfSupported(...)`
- For `m > 1`, it calls `cudaTCGemm_blockwiseGemm(...)` when native payload is supported and tc-blocked weights are available.

The active tensor-core prefill implementation is in [src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu#L342):

- `cudaTCGemm_blockwiseGemm(...)`
- `classifyGemmShape(...)`
- `selectPartialChunkBlocks(...)`
- `launchAccumulationChunk(...)`
- `accumulate_partial_blockwise_kernel<ChunkCount>`

### Important repo-state distinction

There is also a custom prefill implementation in [src/v2/kernels/cuda/CUDANativePayloadPrefillKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDANativePayloadPrefillKernels.cu), but that is **not** the path this handoff is centered on.

Reason:

- The current product direction for prefill is tensor-core-only.
- The runtime adapter still wires native-payload prefill through `cudaTCGemm_blockwiseGemm(...)`.

Do not accidentally switch the active prefill route to the custom prefill file unless that is an explicit new user request.

---

## Recent Changes Already Landed

### 1. Chunked partial accumulation in tensor-core GEMM

Previously, the tensor-core scaffold launched:

- one CUTLASS partial GEMM per `K/32` block
- one accumulation pass per block

This created a severe launch storm.

The current implementation now:

- groups multiple `K=32` blocks into a chunk
- writes each chunk's partials into stacked INT32 planes
- performs one accumulation kernel per chunk instead of per block

Relevant code:

- [src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu#L108)
- [src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu#L146)
- [src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu#L227)

### 2. Shape classification cleanup

The dispatch classification was adjusted so larger `M` shapes do not collapse too aggressively into the `SmallM` path.

Relevant code:

- [src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu#L204)

### 3. Workspace sizing coupled to chunk count

Workspace sizing now accounts for the number of partial chunk planes needed by the active tensor-core implementation.

Relevant code:

- [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](/workspaces/llaminar/src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp#L324)
- [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](/workspaces/llaminar/src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp#L2220)
- [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](/workspaces/llaminar/src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp#L2243)

### 4. Scratch budget regression fix

The chunking work initially pushed workspace usage too far on large shapes. The current logic uses a capped partial scratch budget of **256 MB**, not 512 MB.

Relevant code:

- [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](/workspaces/llaminar/src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp#L323)

---

## Current Performance Assessment

What is true right now:

- Correctness passed on the focused native-payload GEMM harness.
- Focused performance runs also passed after the scratch-budget fix.
- The chunked accumulation path reduced launch overhead.
- The implementation still does **one CUTLASS partial GEMM per `K=32` slice**; only the accumulation side was amortized.
- This means the root structural problem is only partially addressed.

Interpretation:

- The current path is better than the original per-block accumulate design.
- It is still fundamentally a decomposed pipeline rather than a more fused tensor-core-native GEMM.
- Wide shapes benefit more than balanced attention-like shapes.
- The next meaningful gains likely require reducing or restructuring the remaining per-block partial GEMM launch pattern itself.

---

## Files That Matter

### Primary implementation files

- [src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu)
- [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](/workspaces/llaminar/src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp)

### Primary validation harness

- [tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativePayloadGemm.cpp](/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativePayloadGemm.cpp)
- [tests/v2/performance/kernels/cuda/gemm/CUDANativePayloadGemmPerfCommon.h](/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/CUDANativePayloadGemmPerfCommon.h)

### Related but separate decode-path work in current repo state

- [src/v2/kernels/cuda/CUDANativePayloadGemvTuned.cu](/workspaces/llaminar/src/v2/kernels/cuda/CUDANativePayloadGemvTuned.cu)
- [src/v2/kernels/cuda/CUDANativePayloadGemvDispatchHeuristicGenerated.inc](/workspaces/llaminar/src/v2/kernels/cuda/CUDANativePayloadGemvDispatchHeuristicGenerated.inc)
- [tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp](/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp)
- [tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py](/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py)

These are worth being aware of, but they are **decode GEMV** infrastructure, not the primary prefill GEMM target for this handoff.

---

## Known Constraints And Guardrails

### Guardrail 1: Keep prefill tensor-core-only

The previous direction briefly drifted toward non-tensor-core custom prefill work. That is not the target here.

For this handoff:

- `m > 1` prefill work should stay centered on tensor-core execution.
- Do not revert to a dp4a-style prefill direction.

### Guardrail 2: Do not ignore workspace pressure

Large LM-head shapes can fail if scratch demand grows too aggressively.

The current implementation protects this by tying chunk depth to:

- output plane size
- `K/32` block count
- a 256 MB partial scratch budget

Any new design must be checked against large `M x N` shapes before considering it viable.

### Guardrail 3: Do not confuse GEMV wins with GEMM wins

There is substantial new decode GEMV tuning in the repo. That is useful context, but it does not mean prefill GEMM parity is solved.

The prefill target is still unresolved.

### Guardrail 4: Preserve runtime selection semantics

The runtime split in `CUDAQuantisedGemmKernel` is important:

- `m == 1`: GEMV/decode path
- `m > 1`: GEMM/prefill path

Do not blur those roles unless the user explicitly asks for architectural unification.

---

## Recommended Next Technical Direction

The next agent should not spend another cycle only tuning chunk counts. That lever has already been pulled.

The higher-value investigation is:

1. Measure how much time is still spent in the per-`K=32` CUTLASS partial launch sequence.
2. Determine whether the remaining gap is dominated by:
   - launch count
   - global-memory traffic into INT32 partial planes
   - accumulation bandwidth
   - poor shape specialization for balanced cases
3. Prototype a more structural tensor-core-only design that reduces the decomposition overhead further.

Promising directions:

- Fewer, larger tensor-core launches that cover multiple `K=32` groups internally.
- A fused epilogue or fused scale-accumulate strategy that reduces partial INT32 plane traffic.
- Separate strategies for:
  - wide LM-head / FFN-up shapes
  - balanced attention-like shapes
  - TP2 / TP4 narrow projections

Less promising direction for the next step:

- Another small adjustment to `selectPartialChunkBlocks(...)` without deeper structural change.

---

## Validation Entry Points

### Main perf harness

Use:

- [tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativePayloadGemm.cpp](/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativePayloadGemm.cpp)

This gives:

- correctness against CUTLASS fallback
- perf comparisons with `speedup_vs_cutlass`
- reported native payload family for the active path

### Useful environment filters

The harness supports filtering via environment variables defined in [tests/v2/performance/kernels/cuda/gemm/CUDANativePayloadGemmPerfCommon.h](/workspaces/llaminar/tests/v2/performance/kernels/cuda/gemm/CUDANativePayloadGemmPerfCommon.h):

- `LLAMINAR_CUDA_NATIVE_GEMM_FORMATS`
- `LLAMINAR_CUDA_NATIVE_GEMM_SHAPES`
- `LLAMINAR_CUDA_NATIVE_GEMM_PREFILL_M`
- `LLAMINAR_CUDA_NATIVE_GEMM_WARMUP`
- `LLAMINAR_CUDA_NATIVE_GEMM_BENCH`
- `LLAMINAR_CUDA_NATIVE_GEMM_MAX_CASES`
- `LLAMINAR_CUDA_NATIVE_GEMM_SMOKE`

### Practical focused validation pattern

For early iterations, concentrate on:

- `Q4_0`
- `M = 32,64,128`
- a small set of shapes covering:
  - balanced attention
  - FFN-up wide
  - LM-head very wide
  - TP2 / TP4 narrow projections

---

## Suggested Investigation Checklist

1. Confirm the active runtime still routes `m > 1` native payload prefill through `cudaTCGemm_blockwiseGemm(...)`.
2. Benchmark the current baseline for Q4_0 only on a narrow filtered sweep.
3. Identify whether remaining time is dominated by partial GEMM launches or partial accumulation traffic.
4. Implement one structural tensor-core-only change, not just a heuristic tweak.
5. Re-run correctness first.
6. Re-run focused perf on the same filtered shape set.
7. Only then expand to a broader sweep.

---

## Ready-To-Use Subagent Brief

Use this as a starting prompt for a follow-on coding agent:

```text
Continue the CUDA native-payload tensor-core prefill GEMM tuning work in llaminar.

Primary goal:
Close the Q4_0 performance gap with CUTLASS for m>1 prefill shapes while keeping the implementation tensor-core-only.

Important context:
- The active runtime path for native-payload prefill is CUDAQuantisedGemmKernel -> runNativePayloadBlockwiseIfSupported(...) -> cudaTCGemm_blockwiseGemm(...).
- The active implementation already has chunked accumulation of multiple K=32 partial GEMMs.
- Workspace sizing was updated to match chunk count, and the partial scratch budget is capped at 256 MB to avoid LM-head workspace failures.
- There is also a custom prefill file in the repo, but do not switch the runtime to it unless explicitly asked. This task is about the active tensor-core path.
- Decode GEMV tuning files also exist in the repo; they are related context but not the main target here.

Primary files:
- src/v2/kernels/cuda/CUDATensorCoreGemmKernels.cu
- src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp
- tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativePayloadGemm.cpp
- tests/v2/performance/kernels/cuda/gemm/CUDANativePayloadGemmPerfCommon.h

What to do:
1. Benchmark the current active tensor-core prefill path for Q4_0 on a focused set of shapes.
2. Analyze whether the remaining gap is mainly due to the per-K/32 partial GEMM launch structure.
3. Implement one structural tensor-core-only improvement beyond the existing chunked accumulation.
4. Keep changes minimal and local to the active runtime path.
5. Validate correctness against CUTLASS and report speedup_vs_cutlass deltas.

Constraints:
- Keep m>1 prefill tensor-core-only.
- Do not revert to dp4a-style prefill.
- Do not exceed workspace constraints on large shapes.
- Preserve existing runtime semantics for m==1 decode vs m>1 prefill.

Deliverables:
- code changes
- focused validation results
- short summary of whether the change materially reduces the CUTLASS gap
```

---

## Bottom Line

The current codebase has already done the first obvious cleanup: chunking the accumulation side of the tensor-core scaffold. That was necessary but not sufficient. The next agent should treat this as a **structural tensor-core GEMM problem**, not a minor heuristic problem.

If the next change does not reduce the remaining per-`K=32` decomposition cost, it is unlikely to close the parity gap.