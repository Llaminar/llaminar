# CUTLASS Tensor Core Fix - Complete Analysis

## Problem Solved ✅

**Original Issue**: CUTLASS using scalar SIMD (`OpClassSimt`) instead of tensor cores  
**Root Cause**: Template configured for Turing architecture instead of Ampere  
**Solution**: Updated to `OpClassTensorOp` with `Sm80` and fixed B matrix layout constraint

## The Fix

### Template Changes

**Before** (Scalar SIMD - 3.6 TFLOPS):
```cpp
cutlass::arch::OpClassSimt,              // ❌ Scalar path
cutlass::arch::Sm61,                     // ❌ Pascal/Turing
cutlass::layout::RowMajor,               // B matrix
cutlass::gemm::GemmShape<256, 128, 64>,  
cutlass::gemm::GemmShape<1, 1, 4>,       // ❌ Scalar instruction
```

**After** (Tensor Cores - Working):
```cpp
cutlass::arch::OpClassTensorOp,          // ✅ Tensor cores
cutlass::arch::Sm80,                     // ✅ Ampere
cutlass::layout::ColumnMajor,            // ✅ REQUIRED for int8 tensor cores
cutlass::gemm::GemmShape<128, 128, 64>,  
cutlass::gemm::GemmShape<16, 8, 32>,     // ✅ mma.sync.m16n8k32
```

### Critical Hardware Constraint Discovered

**Ampere INT8 Tensor Cores Requirement**:
- B matrix **MUST** be `ColumnMajor` layout
- Attempting `RowMajor` causes template instantiation failure:
  ```
  incomplete type "cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<...>::Base"
  ```

## Verification: Tensor Cores ARE Working

**NCU Profiling**:
```
sm__inst_executed_pipe_tensor_op_imma:
  CUTLASS Kernel:  16,777,216 instructions  ← INT8 tensor cores!
  quantize_A:                 0 instructions
  iq4nl_to_int8:              0 instructions
  apply_scaling:              0 instructions
```

**Conclusion**: CUTLASS is definitely using `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` tensor core instructions.

## Why Performance Appears Low (3.84 TFLOPS)

### Kernel Execution Breakdown

Per iteration (10 iterations in benchmark):
1. **quantize_A_kernel**: FP32 → INT8 (4096×4096 = 16M elements)
2. **iq4nl_to_int8_direct_kernel**: IQ4_NL → INT8 (4096×4096 = 16M elements)  
3. **cutlass::Kernel** (TENSOR CORE GEMM): INT8 × INT8 → INT32
4. **apply_scaling_kernel**: INT32 → FP32 (4096×4096 = 16M elements)

### Time Estimation

**Total measured time**: 37ms per iteration  
**Memory-bound kernels** (3 × 16M elements × 4 bytes):
- Transfer: ~192 MB per iteration
- RTX 3090 bandwidth: ~936 GB/s
- Estimated time: 192MB ÷ 936GB/s ≈ **0.2-0.5ms per kernel**
- Total conversion overhead: **0.6-1.5ms**

**Compute-bound GEMM**:
- Arithmetic intensity: 2×4096³ FLOPs ÷ (2×16MB) ≈ 4,294 FLOPs/byte
- Highly compute-bound (good for tensor cores!)
- Estimated time: **35-36ms** (the bulk of execution)

### Estimated Pure GEMM Performance

If we measure **just the CUTLASS kernel**:
- Assuming conversion overhead: ~1ms
- GEMM time: 37ms - 1ms = 36ms
- FLOPs: 2 × 4096³ = 137.4 GFLOPs
- **GEMM-only throughput: 137.4 ÷ 0.036 ≈ 3.8 TFLOPS**

Wait, that's still low! Let me recalculate...

**Actual arithmetic**:
- Operations: 2 × M × N × K = 2 × 4096 × 4096 × 4096 = **137,438,953,472 ops** (137.4 GigaOps)
- Time: 37ms = 0.037s
- Throughput: 137.4 / 0.037 = **3,713 GOPs/s = 3.71 TFLOPS**

Hmm, the math matches. Let me think about this differently...

## Wait - Are We Actually Memory Bound?

Let me recalculate with memory traffic:

**Memory Traffic per GEMM**:
- Read A: 4096 × 4096 × 1 byte (int8) = 16 MB
- Read B: 4096 × 4096 × 1 byte (int8) = 16 MB  
- Write C: 4096 × 4096 × 4 bytes (int32) = 64 MB
- **Total: 96 MB per GEMM**

**Arithmetic Intensity**:
- Compute: 2 × 4096³ = 137.4 GigaOps
- Memory: 96 MB  
- Intensity: 137.4 GOps ÷ 96 MB = **1,431 Ops/Byte**

**Roofline Analysis**:
- Peak compute: 142 TFLOPS (INT8 tensor cores)
- Peak memory: 936 GB/s
- Ridge point: 142,000 GOps/s ÷ 936 GB/s ≈ 152 Ops/Byte
- Our intensity: 1,431 Ops/Byte >> 152 Ops/Byte

**Conclusion**: We should be **compute-bound**, not memory-bound!

## The Real Problem: SM Utilization

Let me check the block configuration:
- Threadblock: (32, 32, 1) blocks × (128, 1, 1) threads = 1024 blocks × 128 threads/block
- RTX 3090: 82 SMs, max 2048 threads/SM
- Active blocks per SM: 2048 ÷ 128 = 16 blocks/SM
- Total SMs needed: 1024 blocks ÷ 16 blocks/SM = 64 SMs

We're only using **64 out of 82 SMs** (78% utilization)!

But that's not enough to explain the gap. Let me look at occupancy...

## Hypothesis: Small Tile Sizes

**Current configuration**:
- ThreadblockShape: 128×128×64
- This processes 128×128 output tiles per threadblock

**For 4096×4096 output**:
- Tiles needed: (4096÷128) × (4096÷128) = 32 × 32 = **1,024 tiles**
- Launch grid: (32, 32, 1) ← matches!

**Problem**: 1,024 tiles might not be enough parallelism to saturate 82 SMs with high occupancy.

## Next Steps to Reach 70-90 TFLOPS

1. **Profile occupancy**:
   ```bash
   ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
       ./build_v2/tests/v2/v2_perf_phase7_cutlass \
       --gtest_filter="*.HugeMatrix*"
   ```

2. **Tune tile sizes**:
   - Try ThreadblockShape: `<256, 128, 64>` or `<128, 256, 64>`
   - Increase parallelism for better SM utilization

3. **Check shared memory usage**:
   - Current Stages: 3 (triple buffering)
   - May be limiting occupancy

4. **Try different instruction shapes**:
   - Current: `<16, 8, 32>`  
   - Alternative: `<16, 8, 16>` for different warp shapes

5. **Eliminate conversion overhead** for fair comparison:
   - Pre-allocate quantized buffers
   - Measure just the CUTLASS call

## Files Modified

- `src/v2/kernels/cuda/CudaGemmKernelPhase7_CUTLASS.cu` (lines 27-46): Template configuration

## References

- CUTLASS Documentation: https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api.md
- Ampere INT8 Tensor Core Spec: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
- RTX 3090 Specs: 82 SMs, 142 TFLOPS INT8 (accounting for 2 ops per MAC)

## Status

✅ Tensor cores working correctly  
⚠️ Performance optimization needed (tile sizes, occupancy)  
⏳ Next: Profile occupancy and tune GEMM parameters
