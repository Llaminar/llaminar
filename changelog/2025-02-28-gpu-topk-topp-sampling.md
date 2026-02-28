# GPU Top-K/Top-P Sampling with CUDA Parity

**Date**: 2025-02-28

## Summary

Extended GPU-side inference from greedy-only (argmax) to full top-k/top-p sampling.
Both ROCm and CUDA backends now support argmax and top-k, eliminating backend-specific
exceptions in the graph/orchestration system. The non-benchmark generation path
(`OrchestrationRunner::decodeStep`) now uses GPU top-k per device → tiny D2H → host
merge/softmax/top-p/multinomial, avoiding the full vocab-sized logits gather.

## Design Decision

**Option A chosen** (GPU top-k only, rest on CPU):
- GPU does top-k selection per device (~15-25µs on MI50)
- D2H transfer of k×8 bytes (e.g., 320 bytes for k=40) — sub-microsecond
- Host merges N_devices × k candidates, applies temperature/softmax/top-p/multinomial
- Total: ~35-45µs decode sampling overhead

**Alternative rejected** (All-GPU via RCCL AllGather):
- RCCL launch overhead alone (~20-50µs) exceeds the D2H cost for tiny data
- Would require AllGather of k candidates across devices — net slower

## Changes

### New Files
- `src/v2/kernels/rocm/ops/ROCmSamplingKernels.hip` — ROCm top-k kernel (32 threads,
  insertion sort + multi-way merge, TOPK_MAX_K=256)
- `src/v2/kernels/cuda/ops/CUDASamplingKernels.cu` — CUDA argmax + top-k kernels
  (mirrors ROCm implementations for full backend parity)

### Modified Files
- `src/v2/backends/IBackend.h` — Added `topKF32()` virtual method
- `src/v2/backends/rocm/ROCmBackend.h/.cpp` — Added `topKF32()` with lazy per-device
  buffer allocation
- `src/v2/backends/cuda/CUDABackend.h/.cu` — Added `argmaxF32()`, `topKF32()`,
  `pinHostMemory()`, `unpinHostMemory()` implementations
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h` — Added
  `sampleOnDevice(SamplingParams)`
- `src/v2/execution/runner/IOrchestrationRunner.h` — Added `sampleOnDevice()` and
  `setSamplingParams()`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp` — Implemented `sampleOnDevice()`
  forwarding, `setSamplingParams()`, rewrote `decodeStep()` with GPU-first sampling
  (greedy→argmax, else→top-k/top-p), updated `generate()` with param storage +
  skipLogitsGather lifecycle
- `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h/.cpp` —
  Full `sampleOnDevice()` implementation: per-device GPU top-k → host merge → sort →
  temperature → softmax → top-p nucleus → multinomial sampling
- `src/v2/Main.cpp` — BenchmarkRunnerAdapter forwards `sampleOnDevice()`;
  main decode loop now calls `setSamplingParams()` + `setSkipLogitsGatherDecode(true)`
- `src/v2/CMakeLists.txt` — Added new kernel files to ROCM_KERNEL_SOURCES and
  CUDA_KERNEL_SOURCES

### Sampling Pipeline (TP=2 example)
```
Device 0: GPU top-k(vocab_local=76032, k=40) → 40 (value, index) pairs
Device 1: GPU top-k(vocab_local=76032, k=40) → 40 (value, index) pairs
           ↓ D2H: 2 × 40 × 8 = 640 bytes
Host:      Merge 80 candidates → sort → keep top-40 global
           → temperature scaling → softmax → top-p nucleus
           → multinomial sampling → token ID
```

## Test Results

**Benchmark (greedy, TP=2, Qwen2.5-7B Q8_0)**:
- Decode: 87.99 tok/s (no regression from 87.91 baseline)
- Prefill: 457.15 tok/s

**Non-benchmark (top-k=40, top-p=0.9, temp=0.7, TP=2)**:
- All decode tokens use GPU top-k path — zero CPU fallbacks
- Generates coherent text ("Paris, known for its iconic Eiffel Tower...")

**Path routing verified**:
- `-t 0` → greedy → `sampleGreedyOnDevice()` → GPU argmax
- `-t 0.7` → top-k/top-p → `sampleOnDevice(params)` → GPU top-k + host finish
