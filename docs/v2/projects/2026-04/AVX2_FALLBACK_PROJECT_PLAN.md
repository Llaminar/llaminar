# AVX2 Fallback Support — Project Plan

**Goal**: Enable Llaminar CPU inference on systems without AVX512/AVX512-VNNI (i.e., AVX2-only CPUs like AMD Zen 2, Intel Haswell–Broadwell, most consumer CPUs before ~2017).

**Current State**: The build uses `-march=native` and all quantized GEMM/GEMV kernels require AVX512-VNNI (`vpdpbusd`). Users without AVX512 cannot run CPU quantized inference at all. Many other components (primitives, tensor dequant, etc.) already have AVX2 fallbacks.

**Status legend**: `[ ]` not started, `[~]` in progress, `[x]` done

---

## Phase 0: Build System (P0 — prerequisite for everything else)

**Timeline**: 2–3 days  
**Goal**: Produce binaries that work on AVX2-only machines.

### Task 0.1: Remove `-march=native` hardcoding
- **File**: `src/v2/CMakeLists.txt` (line 61)
- **Current**: `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")`
- **Change**: Add a `CPU_ISA` option that defaults to `native` but allows `avx2`, `avx512`, or explicit arch:
  ```cmake
  option(CPU_ISA "Target CPU ISA: native, avx2, avx512" "native")
  if(CPU_ISA STREQUAL "avx2")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell -mtune=generic")
  elseif(CPU_ISA STREQUAL "avx512")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake-avx512 -mtune=generic")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
  endif()
  ```
- **Validation**: Build with `-DCPU_ISA=avx2` succeeds with no undefined intrinsic errors.
- `[ ]` Complete

### Task 0.2: Add CI build target for AVX2
- **File**: New or existing CI config
- **Change**: Add a build matrix entry that builds with `-DCPU_ISA=avx2` to catch regressions.
- `[ ]` Complete

### Task 0.3: Runtime feature detection guard at startup
- **File**: `src/v2/utils/CPUFeatures.h` (already has `cpu_supports_avx512()`, `cpu_supports_avx2()`, `cpu_supports_avx512_vnni()`)
- **File**: Main executable entry point (likely `src/v2/app/main.cpp` or similar)
- **Change**: At startup, log detected ISA features and warn if compiled for AVX512 but running on AVX2. Add a function `log_cpu_features()` that prints detected capabilities.
- `[ ]` Complete

---

## Phase 1: AVX2 Quantized GEMM/GEMV Kernels (P0 — CRITICAL BLOCKER)

**Timeline**: 2–3 weeks  
**Goal**: All quantized weight formats (Q4_0, Q8_0, IQ4_NL, Q4_1, Q5_0, Q5_1, etc.) produce correct output on AVX2-only CPUs.

**Background**: The VNNI instruction `vpdpbusd` computes `acc += dot(u8[4], i8[4])` in one instruction. On AVX2, this must be emulated with two instructions: `_mm256_maddubs_epi16` (u8×i8→i16 pairwise) + `_mm256_madd_epi16` (i16 horizontal pair add→i32). This is the proven approach used by llama.cpp/GGML.

### Task 1.1: Create AVX2 GEMV kernel file
- **New file**: `src/v2/kernels/cpu/native_vnni/CPUNativeAVX2Gemv.h`
- **Reference**: `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h` (1630 lines, 39 `vpdpbusd` calls)
- **Approach**:
  1. Copy the structure of `CPUNativeVNNIGemv.h`
  2. Replace all `_mm512_dpbusd_epi32(acc, a, b)` with the AVX2 two-instruction equivalent:
     ```cpp
     // AVX2 emulation of vpdpbusd (u8 × i8 → i32 accumulate)
     __m256i prod16 = _mm256_maddubs_epi16(a_u8, b_i8);  // u8×i8 → i16
     __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)); // i16 pairs → i32
     acc = _mm256_add_epi32(acc, prod32);
     ```
  3. Replace all `__m512i` → `__m256i`, `_mm512_*` → `_mm256_*` equivalents
  4. Adjust loop counts: ZMM processes 64 bytes, YMM processes 32 bytes, so outer loops iterate 2× more
  5. Replace `_mm512_reduce_add_epi32()` with AVX2 horizontal reduction
  6. Handle tail differently (AVX512 mask registers don't exist in AVX2)
- **Key functions to port** (all in `CPUNativeVNNIGemv.h`):
  - `gemv_native_vnni_avx512_chunk_native()` (line ~140–400) — core VNNI GEMV microkernel
  - `gemv_native_vnni_preq()` (line ~420–650) — pre-quantized activation path
  - `gemv_native_vnni()` (line ~665–690) — top-level dispatcher
  - `q8_0_native_gemv()` (line ~650–665) — Q8_0 native bypass path
  - `gemv_native_vnni_avx512_chunk_native_scalar()` (line ~65–130) — scalar reference (keep as-is for validation)
- `[ ]` Complete

### Task 1.2: Create AVX2 weight packer
- **New file**: `src/v2/kernels/cpu/native_vnni/CPUNativeAVX2WeightPacker.h`
- **Reference**: `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h` (1132 lines)
- **Approach**: The VNNI weight packer interleaves bytes for VNNI lane order. AVX2 needs the same interleave but for 32-byte (256-bit) lanes instead of 64-byte (512-bit). The packed format may differ.
- **Decision point**: Can AVX2 use the same packed format as VNNI (just process half at a time), or does it need a different interleave? Using the same format is simpler but may leave performance on the table. Start with the same format.
- `[ ]` Complete

### Task 1.3: Create AVX2 tile config
- **New file**: `src/v2/kernels/cpu/native_vnni/CPUNativeAVX2TileConfig.h`
- **Reference**: `src/v2/kernels/cpu/native_vnni/CPUNativeVNNITileConfig.h` (~200 lines)
- **Change**: Tile sizes based on YMM (256-bit) register count and width instead of ZMM (512-bit).
- `[ ]` Complete

### Task 1.4: Add dispatch in GEMM kernel
- **File**: `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h` (586 lines)
- **Change**: At the top-level `multiply_tensor()` and `multiply_fused_tensor()` methods, add runtime dispatch:
  ```cpp
  if (cpu_supports_avx512_vnni()) {
      // existing VNNI path
  } else if (cpu_supports_avx2()) {
      // new AVX2 path
  } else {
      // scalar fallback (already exists in scalar reference)
  }
  ```
- **Alternative**: Compile-time dispatch via `#if defined(__AVX512VNNI__)` / `#elif defined(__AVX2__)`. Simpler but requires separate builds.
- **Recommendation**: Use compile-time dispatch initially (consistent with rest of codebase), runtime dispatch later if multi-ISA binaries are desired.
- `[ ]` Complete

### Task 1.5: Update KernelFactory dispatch
- **File**: `src/v2/kernels/KernelFactory.cpp` (line ~578–630)
- **Change**: When `__AVX512VNNI__` is not defined, the `createGemm()` for quantized tensors should still create a working kernel (using AVX2 GEMM). Currently it unconditionally creates `CPUNativeVNNIGemmKernel`.
- `[ ]` Complete

### Task 1.6: Test parity
- Run existing parity tests (`V2_Integration_Parity_*`) with AVX2 build
- Verify token-level match between AVX512 and AVX2 outputs for greedy sampling
- Key test: `ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_Qwen2" --output-on-failure`
- `[ ]` Complete

---

## Phase 2: AVX2 Flash Attention FP32 Helpers (P1)

**Timeline**: 1 week  
**Goal**: FP32 flash attention runs at near-optimal speed on AVX2 (currently falls to scalar).

### Task 2.1: Port core flash attention SIMD helpers
- **File**: `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h` (4378 lines)
- **Target functions** (add `#elif defined(__AVX2__)` blocks):
  - `fast_exp_avx512()` (line ~924) → `fast_exp_avx2()` using `__m256`
  - `dot_fp32_avx512()` (line ~904) → `dot_fp32_avx2()` using `_mm256_fmadd_ps`
  - `softmax_rescale_avx512()` (various) → AVX2 version
  - V-accumulation loops (lines ~2034, 2086) → AVX2 `_mm256_fmadd_ps` loops
- **There are 27 `#if defined(__AVX512F__)` blocks** in this file. Most have `#else` with scalar fallback. The priority targets are the hot-path compute functions:
  1. `fast_exp_avx512` — 5th-order polynomial exp() approximation
  2. FP32 dot product — `_mm512_fmadd_ps` loop + `_mm512_reduce_add_ps`
  3. Score computation and softmax normalization
  4. Weighted V accumulation (`_mm512_fmadd_ps` scatter-gather pattern)
- **Note**: `CPUAttentionKernelT` (1204 lines, zero AVX512) already exists as a functional fallback for non-FP32 paths. It can serve as the baseline for correctness.
- `[ ]` Complete

### Task 2.2: Port BF16/FP16 attention helpers
- **File**: `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h`
- **Lines**: 664–690 (`__AVX512F__ && __F16C__`), 767–793
- **Change**: Add `#elif defined(__AVX2__) && defined(__F16C__)` paths for FP16↔FP32 conversion in attention.
- `[ ]` Complete

### Task 2.3: Port absmax helper
- **File**: `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h` (line ~1413)
- **Change**: `_mm512_max_ps` + `_mm512_reduce_max_ps` → `_mm256_max_ps` + horizontal max.
- `[ ]` Complete

---

## Phase 3: AVX2 Q16_1 Attention (P1)

**Timeline**: 1–2 weeks  
**Goal**: Q16_1 KV cache decode/prefill attention works on AVX2 (currently VNNI-only, ~780 lines).

### Task 3.1: Port Q16_1 prefill attention
- **File**: `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h` (lines 2778–3148)
- **Current**: `#if defined(__AVX512F__) && defined(__AVX512VNNI__)` — 370-line block with no fallback
- **VNNI usage**: Int16 dot products for QK scores using Q16_1 blocks
- **AVX2 approach**: Use `_mm256_madd_epi16` for int16×int16→int32 dot products (native AVX2 instruction)
- **Key operations** to port:
  - Q16_1 block dot product (int16 Q × int16 K → float score)
  - Q16_1 V weighted accumulation (float weight × dequant(V_q16) → float accum)
  - Online softmax state maintenance
- `[ ]` Complete

### Task 3.2: Port Q16_1 decode attention
- **File**: `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h` (lines 3150–3563)
- **Current**: `#if defined(__AVX512F__) && defined(__AVX512VNNI__)` — 413-line block with no fallback
- **Same approach as 3.1** but for M=1 decode path
- `[ ]` Complete

### Task 3.3: Port Q8_1 attention VNNI helpers
- **File**: `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h`
- **Lines**: 1597–1622, 1640–1695, 1718–1766, 1801–1876, 1899–1933, 1965–2015
- **Current**: 6 blocks gated by `__AVX512F__ && __AVX512VNNI__` for Q8_1 KV cache operations
- **Each has a scalar `#else` fallback** that works but is slow
- **AVX2 approach**: Same `_mm256_maddubs_epi16` + `_mm256_madd_epi16` emulation as Phase 1
- `[ ]` Complete

---

## Phase 4: AVX2 GDN Kernels (P2 — Qwen3.5 only)

**Timeline**: 1 week  
**Goal**: Qwen3.5 GDN (Gated Delta Net) layers run efficiently on AVX2. Scalar fallback already works.

### Task 4.1: AVX2 GDN recurrence kernel
- **File**: `src/v2/kernels/cpu/gdn/CPUGatedDeltaNet.cpp` (866 lines)
- **Current**: 173 AVX512 intrinsics, 0 AVX2. Has scalar `#else` fallback.
- **Key loops to port** (heaviest AVX512 usage):
  - `compute_gates()` — exp/sigmoid via `_mm512_*` (line ~270–400)
  - `state_update()` — matrix multiply + gating (line ~400–600)
  - `fast_exp_avx512()` local copy — same polynomial as attention
- **Change**: Add `#elif defined(__AVX2__)` paths between existing AVX512 and scalar blocks
- `[ ]` Complete

### Task 4.2: AVX2 short convolution
- **File**: `src/v2/kernels/cpu/gdn/CPUShortConvolution.cpp` (374 lines)
- **Current**: 30 AVX512 intrinsics, 0 AVX2. Has scalar fallback.
- **Change**: Add `#elif defined(__AVX2__)` paths. This is primarily `_mm512_fmadd_ps` → `_mm256_fmadd_ps` conversions.
- `[ ]` Complete

### Task 4.3: AVX2 GatedRMSNorm and AttentionOutputGate stages
- **Files**:
  - `src/v2/execution/compute_stages/stages/GatedRMSNormStage.cpp` (30 AVX512 intrinsics, scalar fallback)
  - `src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp` (29 AVX512 intrinsics, scalar fallback)
  - `src/v2/execution/compute_stages/stages/FusedResidualNormStage.cpp` (0 AVX512 intrinsics but has `#if __AVX512F__` guard)
- **Change**: Add `#elif defined(__AVX2__)` intermediate paths
- `[ ]` Complete

---

## Phase 5: AVX2 TurboQuant Kernels (P2)

**Timeline**: 3–5 days  
**Goal**: TurboQuant KV cache quantization/dequantization runs at near-optimal speed on AVX2. Scalar fallback already works.

### Task 5.1: AVX2 TQ8 quantize
- **File**: `src/v2/kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h` (187 lines)
- **Current**: 17 AVX512 intrinsics, scalar fallback at line 131
- **Change**: Add `#elif defined(__AVX2__)` before the `#else` scalar block
- `[ ]` Complete

### Task 5.2: AVX2 TQ4 quantize
- **File**: `src/v2/kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h` (176 lines)
- **Current**: 14 AVX512 intrinsics, scalar fallback
- `[ ]` Complete

### Task 5.3: AVX2 TQ8 dequantize
- **File**: `src/v2/kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h` (120 lines)
- **Current**: 8 AVX512 intrinsics, scalar fallback
- `[ ]` Complete

### Task 5.4: AVX2 TQ4 dequantize
- **File**: `src/v2/kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h` (460 lines)
- **Current**: 7 AVX512 intrinsics, scalar fallback
- `[ ]` Complete

### Task 5.5: AVX2 TurboQuant rotation
- **File**: `src/v2/kernels/cpu/turboquant/TurboQuantRotation.h` (401 lines)
- **Current**: 34 AVX512 intrinsics, 11 AVX2 intrinsics (partial coverage), scalar fallback
- **Change**: Complete the AVX2 coverage for remaining functions
- `[ ]` Complete

### Task 5.6: AVX2 activation rotation (FWHT)
- **File**: `src/v2/kernels/cpu/rotation/ActivationRotation.h` (534 lines)
- **Current**: 86 AVX512 intrinsics, scalar `fwht_scalar()` fallback
- **Change**: Add AVX2 FWHT (Fast Walsh-Hadamard Transform) — straightforward since it's mostly FMA operations
- `[ ]` Complete

### Task 5.7: AVX2 TQ fused attention primitives
- **File**: `src/v2/kernels/cpu/attention/TQFusedAttentionPrimitives.h` (210 lines)
- **Current**: 12 AVX512 intrinsics, no fallback
- **Change**: Add `#elif defined(__AVX2__)` and/or scalar fallback
- `[ ]` Complete

---

## Phase 6: Cleanup & Validation (P3)

**Timeline**: 2–3 days

### Task 6.1: AVX2 CPU benchmark
- **File**: `src/v2/backends/benchmarks/CPUBenchmark.cpp`
- **Current**: AVX512 FLOPS benchmark with AVX2 fallback for FMA but not for VNNI throughput
- **Change**: Add AVX2 VNNI-emulation throughput benchmark
- `[ ]` Complete

### Task 6.2: Full regression test suite on AVX2 build
- Build with `-DCPU_ISA=avx2`
- Run: `ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel`
- Run: `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure`
- Run: `ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure`
- All must pass.
- `[ ]` Complete

### Task 6.3: Performance comparison report
- Benchmark AVX2 vs AVX512 builds on same hardware
- Document expected performance delta per operation
- Expected: ~1.5–2× slower for GEMM/GEMV (VNNI vs emulated), ~1.3× for attention, ~1× for most primitives
- `[ ]` Complete

### Task 6.4: Documentation update
- **File**: `.github/copilot-instructions.md`
- Add build instructions for AVX2 target
- Document minimum CPU requirements (AVX2 + FMA required)
- `[ ]` Complete

---

## File Index by Priority

### P0 — Must change (inference won't work without these)

| File | Lines | What | Current AVX2 | Task |
|------|-------|------|-------------|------|
| `src/v2/CMakeLists.txt` | 61 | `-march=native` hardcoded | None | 0.1 |
| `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemv.h` | 1630 | VNNI GEMV (M=1 decode) | None (39 `vpdpbusd`) | 1.1 |
| `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIWeightPacker.h` | 1132 | VNNI weight interleave | Partial (14 AVX2) | 1.2 |
| `src/v2/kernels/cpu/native_vnni/CPUNativeVNNITileConfig.h` | ~200 | ZMM tile sizing | None | 1.3 |
| `src/v2/kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h` | 586 | GEMM top-level dispatch | None | 1.4 |
| `src/v2/kernels/KernelFactory.cpp` | ~580–630 | GEMM factory dispatch | None | 1.5 |

### P1 — Important for performance (functional scalar fallback exists)

| File | Lines | What | Current AVX2 | Task |
|------|-------|------|-------------|------|
| `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h` | 4378 | Flash attention core | 27 AVX512F blocks, most have scalar `#else` | 2.1–2.3, 3.1–3.3 |

### P2 — Nice to have (scalar fallback works, model-specific)

| File | Lines | What | Current AVX2 | Task |
|------|-------|------|-------------|------|
| `src/v2/kernels/cpu/gdn/CPUGatedDeltaNet.cpp` | 866 | GDN recurrence | Scalar fallback | 4.1 |
| `src/v2/kernels/cpu/gdn/CPUShortConvolution.cpp` | 374 | GDN short conv | Scalar fallback | 4.2 |
| `src/v2/execution/compute_stages/stages/GatedRMSNormStage.cpp` | ~280 | GDN gated norm | Scalar fallback | 4.3 |
| `src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp` | ~210 | GDN gate | Scalar fallback | 4.3 |
| `src/v2/execution/compute_stages/stages/FusedResidualNormStage.cpp` | ~160 | Fused residual | Scalar fallback | 4.3 |
| `src/v2/kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h` | 187 | TQ8 quantize | Scalar fallback | 5.1 |
| `src/v2/kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h` | 176 | TQ4 quantize | Scalar fallback | 5.2 |
| `src/v2/kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h` | 120 | TQ8 dequantize | Scalar fallback | 5.3 |
| `src/v2/kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h` | 460 | TQ4 dequantize | Scalar fallback | 5.4 |
| `src/v2/kernels/cpu/turboquant/TurboQuantRotation.h` | 401 | TQ rotation | Partial AVX2 | 5.5 |
| `src/v2/kernels/cpu/rotation/ActivationRotation.h` | 534 | FWHT rotation | Scalar fallback | 5.6 |
| `src/v2/kernels/cpu/attention/TQFusedAttentionPrimitives.h` | 210 | TQ fused attention | None | 5.7 |

### Already working — no changes needed

| File | Lines | What | Status |
|------|-------|------|--------|
| `src/v2/tensors/Q4_0Tensor.cpp` | ~300 | Q4_0 dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q8_0Tensor.cpp` | ~700 | Q8_0 dequant/requant | AVX2 + scalar fallback |
| `src/v2/tensors/Q8_1Tensor.cpp` | ~1200 | Q8_1 dequant/requant | AVX2 + scalar fallback |
| `src/v2/tensors/Q16_1Tensor.cpp` | ~1200 | Q16_1 dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q4_1Tensor.cpp` | ~300 | Q4_1 dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q5_0Tensor.cpp` | ~300 | Q5_0 dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q5_1Tensor.cpp` | ~300 | Q5_1 dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q3_KTensor.cpp` | ~450 | Q3_K dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q4_KTensor.cpp` | ~550 | Q4_K dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q5_KTensor.cpp` | ~550 | Q5_K dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q6_KTensor.cpp` | ~300 | Q6_K dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q8_KTensor.cpp` | ~250 | Q8_K dequant | AVX2 + scalar fallback |
| `src/v2/tensors/Q2_KTensor.cpp` | ~300 | Q2_K dequant | AVX2 + scalar fallback |
| `src/v2/tensors/IQ*.cpp` (9 files) | ~100 each | IQ dequant | Via SIMDHelpers (AVX2) |
| `src/v2/tensors/SIMDHelpers.h` | 16304 | SIMD dispatch | 155 AVX512 / 149 AVX2 guards |
| `src/v2/tensors/QuantizationUtils.h` | ~350 | Quant utils | AVX2 + scalar |
| `src/v2/tensors/BF16Tensor.cpp` | ~200 | BF16 convert | AVX2 fallback |
| `src/v2/tensors/FP32Tensor.cpp` | ~300 | FP32 ops | AVX2 fallback |
| `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp` | ~2300 | RMS norm | 30 AVX512 / 16 AVX2 |
| `src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp` | ~2000 | RoPE | 34 AVX512 / 33 AVX2 |
| `src/v2/kernels/cpu/primitives/SoftmaxPrimitives.cpp` | ~400 | Softmax | 7 AVX512 / 7 AVX2 |
| `src/v2/kernels/cpu/primitives/SoftmaxPrimitivesImpl.h` | ~500 | Softmax impl | 3 AVX512 / 4 AVX2 |
| `src/v2/kernels/cpu/primitives/SwiGLUPrimitives.cpp` | ~400 | SwiGLU | 6 AVX512 / 6 AVX2 |
| `src/v2/kernels/cpu/primitives/SimdTraits.h` | ~100 | SIMD traits | 4 AVX512 / 3 AVX2 |
| `src/v2/kernels/cpu/attention/CPUAttentionKernelT.h` | 1204 | Non-flash attention | Zero AVX512 |

---

## Key Technical Notes

### VNNI emulation pattern (used throughout Phase 1 and 3)

The AVX512-VNNI instruction `vpdpbusd` does: `acc[i] += u8[4i..4i+3] · i8[4i..4i+3]` for 16 groups (64 bytes).

AVX2 emulation for 32 bytes (8 groups):
```cpp
// Input: a = 32 unsigned bytes, b = 32 signed bytes, acc = 8 int32 accumulators
__m256i prod16 = _mm256_maddubs_epi16(a, b);              // u8×i8 → 16 int16
__m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)); // 16 int16 → 8 int32
acc = _mm256_add_epi32(acc, prod32);                       // accumulate
```

This processes half the data per instruction vs VNNI, so the inner loop runs 2× as many iterations.

### AVX2 horizontal reduction patterns

```cpp
// AVX2: reduce 8×int32 to scalar
inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_extract_epi32(lo, 0);
}

// AVX2: reduce 8×float to scalar
inline float hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
```

### AVX512 mask register elimination

AVX512 uses `__mmask16` / `__mmask8` for predicated operations. AVX2 doesn't have mask registers. Options:
- Use `_mm256_blendv_*` with a comparison result as the mask
- Process full vectors and mask the result with `_mm256_and_si256`
- Handle tails with scalar loops (simpler, fine for non-hot paths)

### Compile guards pattern

For each function with an AVX512 path, add the AVX2 intermediate:
```cpp
#if defined(__AVX512F__)
    // AVX512 implementation (existing)
#elif defined(__AVX2__)
    // AVX2 implementation (new)
#else
    // Scalar fallback (existing or new)
#endif
```

---

## Timeline Summary

| Phase | Priority | Effort | Cumulative |
|-------|----------|--------|------------|
| 0: Build system | P0 | 2–3 days | 3 days |
| 1: AVX2 GEMM/GEMV | P0 | 2–3 weeks | ~3.5 weeks |
| 2: AVX2 Flash Attention FP32 | P1 | 1 week | ~4.5 weeks |
| 3: AVX2 Q16_1 Attention | P1 | 1–2 weeks | ~6 weeks |
| 4: AVX2 GDN kernels | P2 | 1 week | ~7 weeks |
| 5: AVX2 TurboQuant | P2 | 3–5 days | ~7.5 weeks |
| 6: Cleanup & validation | P3 | 2–3 days | ~8 weeks |

**Minimum viable (P0 only)**: ~3.5 weeks for quantized CPU inference on AVX2.  
**Full AVX2 parity**: ~8 weeks.
