# CUDA Native-Payload Prefill Handover

**Date**: March 14, 2026  
**Branch**: `tensor-parallel`  
**Status**: Active handoff for CUDA Q4_0 native-payload tensor-core prefill GEMM  
**Supersedes**: `docs/v2/projects/2026-03/HANDOVER_CUDA_TENSORCORE_GEMM_TUNING.md` where it claims the custom prefill file is not the active direction. That is no longer true in the current workspace state.

---

## Goal

Primary goal: reach **performance parity with CUTLASS** for the active CUDA native-payload **Q4_0 prefill** path while preserving correctness.

Success criteria:

- Match CUTLASS correctness on the native-payload perf harness.
- Close the current `speedup_vs_cutlass` gap for the important prefill shapes.
- Improve balanced attention-like and TP-split shapes without regressing wide FFN-up / LM-head shapes.
- Keep the implementation tensor-core-only for `m > 1` prefill.

---

## Active Runtime Path

The active CUDA runtime path for Q4_0 prefill is:

1. `CUDAQuantisedGemmKernel::runNativePayloadBlockwiseIfSupported(...)`
2. `cudaNativePayloadPrefillQ40_fp32(...)`
3. one of the `native_q40_tc_*` kernel families in `CUDANativePayloadPrefillKernels.cu`

Important distinction:

- `m == 1` is still the decode / GEMV path.
- `m > 1` is the prefill / GEMM path.
- Do not mix those paths unless the user explicitly asks for an architectural change.

Relevant files:

- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp`
- `src/v2/kernels/cuda/CUDANativePayloadPrefillKernels.cu`
- `src/v2/kernels/cuda/CUDANativePayloadGemvTuned.cu`
- `src/v2/kernels/cuda/CUDANativePayloadGemvDispatchHeuristicGenerated.inc`

---

## Kernel Family

The active prefill kernel family lives in:

- `src/v2/kernels/cuda/CUDANativePayloadPrefillKernels.cu`

The family names exposed by the kernel are:

- `native_q40_tc_tp_narrow`
- `native_q40_tc_balanced`
- `native_q40_tc_wide`

The main dispatch points inside that file are:

- `classifyQ40PrefillShape(...)`
- `chooseSplitK(...)`
- `launchQ40TensorCoreVariant(...)`
- `cudaNativePayloadPrefillQ40_fp32(...)`

Current shape mapping in practice:

- attention / TP-down-proj style shapes usually land in `native_q40_tc_balanced`
- FFN-up / LM-head style shapes usually land in `native_q40_tc_wide`
- smaller TP-narrow shapes land in `native_q40_tc_tp_narrow`

---

## Files Being Worked On

Primary implementation files:

- `src/v2/kernels/cuda/CUDANativePayloadPrefillKernels.cu`
- `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp`

Primary validation harness files:

- `tests/v2/performance/kernels/cuda/gemm/CUDANativePayloadGemmPerfCommon.h`
- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativePayloadGemm.cpp`
- `tests/v2/CMakeLists.txt`

Related decode-side files that are useful context but are not the main prefill target:

- `src/v2/kernels/cuda/CUDANativePayloadGemvTuned.cu`
- `src/v2/kernels/cuda/CUDANativePayloadGemvDispatchHeuristicGenerated.inc`
- `tests/v2/performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp`
- `tests/v2/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py`

Current working-tree note:

- `src/v2/kernels/cuda/CUDANativePayloadPrefillKernels.cu` is still modified in the workspace relative to git.
- That diff is pre-existing from this tuning line and currently removes `SINGLE_PASS_MATERIALIZE=true` from the balanced and TP-narrow launch variants while keeping it on the wide variants.
- Treat that as inherited workspace state; do not assume it was introduced in the most recent pass.

---

## Perf Harness And Iteration Workflow

Build only the focused benchmark target:

```bash
cmake --build /workspaces/llaminar/build_v2_release --parallel --target v2_perf_cuda_native_payload_gemm
```

Correctness gate on representative large shapes:

```bash
cd /workspaces/llaminar
timeout 180 env \
  LLAMINAR_CUDA_NATIVE_GEMM_FORMATS=Q4_0 \
  LLAMINAR_CUDA_NATIVE_GEMM_SHAPES='14B_Attn,14B_FFN_Up,14B_TP2_FFN_Down,14B_LM_Head' \
  LLAMINAR_CUDA_NATIVE_GEMM_CORRECTNESS_PREFILL_M=128 \
  LLAMINAR_CUDA_NATIVE_GEMM_MAX_CASES=4 \
  ./build_v2_release/tests/v2/v2_perf_cuda_native_payload_gemm --gtest_filter='*Correctness*'
```

Focused performance loop on the six-shape sample used in this session:

```bash
cd /workspaces/llaminar
timeout 240 env \
  LLAMINAR_CUDA_NATIVE_GEMM_FORMATS=Q4_0 \
  LLAMINAR_CUDA_NATIVE_GEMM_SHAPES='3B_FFN_Up,7B_FFN_Up,14B_Attn,14B_FFN_Up,14B_TP2_FFN_Down,14B_LM_Head' \
  LLAMINAR_CUDA_NATIVE_GEMM_PREFILL_M=128 \
  LLAMINAR_CUDA_NATIVE_GEMM_WARMUP=1 \
  LLAMINAR_CUDA_NATIVE_GEMM_BENCH=2 \
  ./build_v2_release/tests/v2/v2_perf_cuda_native_payload_gemm --gtest_filter='*Performance*'
```

Useful harness locations:

- target registration: `tests/v2/CMakeLists.txt`
- correctness test: `Perf__CUDANativePayloadGemm.cpp` `Correctness_AllFormats_KeyShapes`
- perf test: `Perf__CUDANativePayloadGemm.cpp` `Performance_AllFormats_AllShapes`
- env var parsing and `runKernel(...)`: `CUDANativePayloadGemmPerfCommon.h`

The harness reports:

- `native_payload_family`
- `cutlass_min_us`
- `native_payload_min_us`
- `speedup_vs_cutlass`
- achieved TOPS and percent of tensor-core peak

That is the primary loop for this work. Keep iterations narrow and comparable.

---

## Last Known-Good Focused Baseline

This is the last known-good focused baseline before the failed experiments in this session. It is the reference point to beat on the six-shape sample above.

Q4_0, `M=128`, `warmup=1`, `bench=2`:

- `3B_FFN_Up`: `0.496x`
- `7B_FFN_Up`: `0.326x`
- `14B_Attn`: `0.384x`
- `14B_FFN_Up`: `0.417x`
- `14B_TP2_FFN_Down`: `0.371x`
- `14B_LM_Head`: `0.424x`

Focused median: about `0.40x` versus CUTLASS.

Interpretation:

- wide shapes are still better than balanced shapes
- balanced attention-like and TP-down-proj shapes remain the main weakness
- parity with CUTLASS is still far away

---

## What Failed In This Session

These passes were tried and should not be retried blindly:

1. Enabling single-pass payload materialization for balanced and TP-narrow variants.
Result:
- correctness passed
- LM-head improved modestly
- balanced / TP-narrow shapes regressed
- focused median dropped to about `0.38x`

2. Replacing split-K output atomics with a bounded two-phase partial-buffer reduction.
Result:
- correctness passed
- balanced shapes regressed badly
- `14B_Attn` fell to about `0.285x`
- `14B_TP2_FFN_Down` fell to about `0.272x`

3. Reducing split-K eagerness for non-wide shapes by lowering the underfill threshold.
Result:
- correctness passed
- balanced shapes regressed even further
- `14B_Attn` fell to about `0.222x`
- `14B_TP2_FFN_Down` fell to about `0.214x`

Takeaway:

- the current bottleneck is unlikely to be fixed by split-K policy changes alone
- the remaining gap is more likely inside decode / staging / shared-memory materialization on the balanced path

---

## Best Current Hypothesis

The next useful work should focus on the balanced kernel internals rather than more outer heuristics.

Most likely bottleneck areas:

- payload decode cost into shared memory
- scale load / reuse pattern for balanced tiles
- shared-memory staging overhead versus actual MMA issue rate
- insufficient overlap between payload load, decode, and compute in the balanced family

What does **not** look promising right now:

- more split-K experimentation by itself
- another pass on underfill thresholds without profiling evidence
- treating wide-shape wins as evidence that balanced-shape parity is close

---

## Suggested Next Pass

1. Profile `native_q40_tc_balanced` specifically on `14B_Attn` and `14B_TP2_FFN_Down`.
2. Separate time spent in payload load + decode + scale materialization versus MMA issue.
3. If decode/staging dominates, target B-tile decode reuse or scale-access reduction inside the balanced kernel.
4. Keep the six-shape sample above as the standard comparison loop.
5. Do not claim improvement unless the focused median moves up from the current `~0.40x` baseline.

---

## Guardrails

- Keep `m > 1` prefill tensor-core-only.
- Preserve CUTLASS parity on the correctness harness.
- Do not mix decode and prefill changes in one pass unless necessary.
- Be careful with the current working-tree state in `CUDANativePayloadPrefillKernels.cu`; it already differs from git.
- If you regenerate `CUDANativePayloadGemvDispatchHeuristicGenerated.inc`, do it from `analyze_cuda_tc_gemv_dispatch.py` and a known-good CSV, not by manual editing.
