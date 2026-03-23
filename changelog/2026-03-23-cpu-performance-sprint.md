# CPU Performance Sprint — FP16 KV Cache Fix + CPU Tensor Parallel

**Date**: 2026-03-23  
**Model**: Qwen2.5-7B-Instruct-Q8_0.gguf (8GB, 28 layers)  
**Hardware**: 2× Intel Xeon Gold 6238R (28 cores/socket, 56 total), 2 NUMA nodes, DDR4-2933 6ch  

## Summary

Major CPU inference performance breakthrough through fixing a per-token full KV cache re-dequantization bug, plus enabling CPU tensor parallel (TP) across NUMA nodes.

### Performance Results

| Mode | Before | After | Improvement | vs llama.cpp |
|------|--------|-------|-------------|--------------|
| **Single-socket decode** | 7.28 tok/s | 10.48 tok/s | **+44%** | **+29%** (vs 8.14) |
| **Single-socket prefill** | — | 34.51 tok/s | — | **+76%** (vs 19.64) |
| **2-socket TP decode** | N/A (broken) | 18.25 tok/s | — | **+124%** (vs 8.14) |
| **2-socket TP prefill** | N/A (broken) | 58.10 tok/s | — | **+196%** (vs 19.64) |

### Correctness

- Single-socket: RMSE=0, max_diff=0 for all logit comparisons
- CPU TP: RMSE=0, max_diff=0 across both ranks, MATCH: YES ✓
- 33/33 Qwen2 SingleDevice parity tests PASSED
- Qwen3 SingleDevice parity tests PASSED
- 322/322 unit tests PASSED

## Changes

### 1. FP16→FP32 Incremental KV Cache Conversion (AttentionComputeStage)

**Root Cause**: `CPURingKVCache<FP16>` stores KV in FP16. Every call to `append()` uses `mutable_typed_data()` which clears the dequantization cache. The attention kernel then calls `fp32_data()` which triggers a **full heap reallocation + FP16→FP32 conversion of the entire KV cache** — for every token, every layer (56 tensors per decode step).

**Fix**: Persistent FP32 buffers in `AttentionComputeStage` with incremental conversion:
- Lazy-allocate FP32 buffers once at `max_seq_len` capacity
- Track converted row count (`decode_kv_fp32_rows_`)
- Convert only newly appended rows via `simd::convert_fp16_to_fp32()`
- Detect cache clears (kv_len < tracked rows) and reset

**Files**:
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.h` — persistent buffer members
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp` — incremental conversion logic

**Impact**: Attention dropped from **31.2% → 2.3%** of decode time (19× reduction).

### 2. Decode Mask Skip

Small optimization: skip causal mask allocation for seq_len=1 decode (mask is identity for single-token queries).

**File**: `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp`

### 3. CPU Tensor Parallel Config Validation Fix

**Bug**: `-d cpu` auto-bootstrap sets `device_map`, `tp_degree`, and `tp_scope` via `RuntimeInitPhase`, but left the original `device_for_this_rank` set. `ConfigValidator` then sees conflicting `-d cpu:0` + `--device-map` + `-tp 2`.

**Fix**: Clear `device_for_this_rank` after setting `device_map` in the CPU shorthand path.

**File**: `src/v2/app/RuntimeInitPhase.cpp`

### 4. LM Head AllGather Buffer Overflow Fix

**Bug**: AllGather for LM head used `actual_seq_len = total_tokens` (prompt length during prefill), but the logits buffer is `[1, vocab_local]` because LM head always computes only the last token's logits (`lm_m = (seq_len > 1) ? 1 : seq_len`).

**Fix**: Changed all 4 LM head AllGather instances to `actual_seq_len = 1`.

**File**: `src/v2/models/qwen/Qwen2Graph.cpp`

### 5. ensureOnDevice/allocateOnDevice CPU No-Op

**Bug**: `TensorBase::ensureOnDevice()` and `allocateOnDevice()` logged ERROR for CPU devices, generating 12,656 error lines per benchmark run and adding measurable overhead.

**Fix**: Made both functions return `true` immediately for CPU devices (data is inherently on-host). Updated corresponding tests.

**Files**:
- `src/v2/tensors/TensorBase.cpp`
- `tests/v2/unit/tensors/Test__TensorAllocateOnDevice.cpp`
- `tests/v2/unit/backends/Test__MultiGPU.cpp`

### 6. Transparent Huge Pages (THP) for Large Allocations

Added `madvise(MADV_HUGEPAGE)` for allocations >2MB in `AlignedVector::allocate_raw()` to reduce TLB misses for large weight/buffer allocations.

**File**: `src/v2/tensors/AlignedVector.h`

## Profile Breakdown (Post-Fix)

### Single Socket Decode (per-token: 96.7ms)

| Kernel | % of Time |
|--------|-----------|
| GEMM_FUSED_GATE_UP | 49.9% |
| GEMM (attn_out + ffn_down) | 31.3% |
| GEMM_FUSED_QKV | 8.1% |
| LM_HEAD | 6.8% |
| ATTENTION | 2.3% |
| Kernel Efficiency | 99.1% |
| Effective BW | ~80 GB/s (83% of 96.7 peak) |

### 2-Socket TP Decode (per-token: 54.8ms)

| Kernel | % of Time |
|--------|-----------|
| GEMM_FUSED_GATE_UP | 45.2% |
| GEMM (attn_out + ffn_down) | 29.5% |
| GEMM_FUSED_QKV | 9.8% |
| ALLREDUCE (MPI) | 5.9% |
| ATTENTION | 2.7% |
| ALLGATHER (LM head) | 1.0% |
| Kernel Efficiency | 98.2% |
| TP Scaling | 1.76× (88% parallel efficiency) |

## Theoretical Analysis

| Metric | Single Socket | 2-Socket TP |
|--------|--------------|-------------|
| Weight data/token | 7.95 GB | 3.98 GB/rank |
| Measured read BW | 96.7 GB/s | 96.7 GB/s × 2 |
| Theoretical max | 12.2 tok/s | 24.3 tok/s |
| Actual | 10.48 tok/s | 18.25 tok/s |
| Efficiency | 86% | 75% |

The system is operating near the hardware memory bandwidth wall. Remaining gaps come from MPI synchronization overhead (5.9%), stage management (1.8%), and GEMV kernel inefficiency (~10%).
