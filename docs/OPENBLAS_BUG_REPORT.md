# OpenBLAS Bug Report: cblas_sbgemm NaN on Large Matrices (Software Emulation)

**Date:** October 20, 2025  
**Reported By:** David Sanftenberg (Llaminar Project)  
**Status:** Prepared for filing at <https://github.com/OpenMathLib/OpenBLAS/issues>

---

## Summary

`cblas_sbgemm` (BF16 × BF16 → FP32 GEMM) produces **all-NaN outputs** for large matrices (e.g., 64×896×896) when running on CPUs **without native AVX512_BF16 instructions**. Small matrices (e.g., 2×2) work correctly, proving the API usage is correct. The bug is in OpenBLAS's software emulation path for BF16 operations.

---

## Environment

### Hardware
- **CPU:** Intel Xeon Gold 6238R (Cascade Lake, 2nd Gen Xeon Scalable)
- **Microarchitecture:** Cascade Lake (2019)
- **Sockets:** 2
- **Cores per socket:** 28 physical (56 with hyperthreading)
- **Total cores:** 112 logical CPUs

### CPU Instruction Sets
**Present:**
- ✅ AVX512F (base AVX-512)
- ✅ AVX512DQ, AVX512BW, AVX512VL, AVX512CD
- ✅ AVX512_VNNI (INT8/INT16 VNNI instructions)
- ✅ F16C (FP16 conversion instructions)

**Missing:**
- ❌ **AVX512_BF16** (native BF16 multiply-accumulate - introduced in Cooper Lake 2020)

This is expected for Cascade Lake (2019). AVX512_BF16 was added in Cooper Lake (2020) and Ice Lake Xeon (2021).

### Software
- **OS:** Ubuntu 24.04.2 LTS
- **OpenBLAS Version:** v0.3.26+ds-1 (built with `-DBUILD_BFLOAT16=ON`)
- **Compiler:** GCC 13.3.0
- **Build Flags:** Standard OpenBLAS build with BF16 support enabled

---

## Bug Description

### Expected Behavior
On CPUs without native AVX512_BF16 instructions, `cblas_sbgemm` should use software emulation to perform BF16×BF16→FP32 matrix multiplication and produce correct (or reasonably approximate) results.

### Actual Behavior
For large matrices (≥ 64×896), `cblas_sbgemm` produces **all-NaN outputs** despite valid inputs (verified no NaN/Inf before the call).

### Test Results

| Matrix Size | Input Valid? | Output Result | Status |
|-------------|--------------|---------------|--------|
| 2×2 | ✅ Yes | ✅ Correct (values match expected) | **PASS** |
| 8×8 | ✅ Yes | ⚠️ Untested | Unknown |
| 64×64 | ✅ Yes | ⚠️ Untested | Unknown |
| 64×896 | ✅ Yes | ❌ All NaN | **FAIL** |
| 64×896×896 | ✅ Yes | ❌ All NaN | **FAIL** |

**Key Finding:** Small matrices work perfectly, proving:
1. API usage is correct
2. BF16 conversion utilities work correctly
3. The bug is in OpenBLAS's emulation path for large matrices

---

## Reproduction

### Minimal Reproducer

See attached: `openblas_sbgemm_bug_reproducer.c`

**Compile:**
```bash
# Using built OpenBLAS with BF16 support
gcc -o openblas_sbgemm_bug_reproducer openblas_sbgemm_bug_reproducer.c \
    -I./build/external/OpenBLAS/include \
    -L./build/external/OpenBLAS/lib -lopenblas -lm -lpthread -fopenmp -O2

# Run
LD_LIBRARY_PATH=./build/external/OpenBLAS/lib:$LD_LIBRARY_PATH \
    ./openblas_sbgemm_bug_reproducer
```

**Expected Output on Cascade Lake:**
- Test 1 (2×2 matrix): **PASS** ✅
- Test 2 (64×896×896 matrix): **FAIL** ❌ (all NaN outputs)

**Note:** On CPUs with native AVX512_BF16 (Cooper Lake+, Ice Lake Xeon+), both tests should pass.

### Full Test Case

Our production test suite includes `tests/TestBF16GemmParity.cpp` which originally failed on Cascade Lake but now automatically falls back to FP32 expansion to work around this bug.

---

## Root Cause Analysis

### Suspected Issue
OpenBLAS's software emulation path for BF16 operations (used when AVX512_BF16 is not available) appears to have a bug in the general-case matrix multiply kernel.

**Possible causes:**
1. **Buffer alignment issue** - Software emulation may assume certain alignment guarantees that don't hold for large matrices
2. **Accumulation overflow** - Incorrect accumulation pattern causing NaN propagation
3. **Kernel selection bug** - Wrong kernel dispatch for emulated BF16 on large dimensions
4. **Register spilling** - AVX-512 emulation path may corrupt registers for large problem sizes

### Evidence
1. ✅ Small matrices (2×2) work perfectly → API and BF16 conversion are correct
2. ❌ Large matrices (64×896×896) fail completely → Emulation kernel bug
3. ✅ Inputs validated (no NaN/Inf) → Bug happens during computation
4. ✅ FP32 path works perfectly → Not a general GEMM issue

### Likely Code Location
Based on OpenBLAS source structure:
- `kernel/x86_64/sbgemm_*.c` - AVX-512 BF16 GEMM kernels
- Software emulation fallback path for non-AVX512_BF16 CPUs
- Kernel dispatch logic in `driver/level3/sbgemm.c`

---

## Workaround

We've implemented automatic CPU feature detection to detect AVX512_BF16 availability and fall back to FP32 expansion when not available:

```cpp
// src/utils/CpuFeatures.h
bool can_use_native_bf16_gemm() {
    return CpuFeatures::instance().has_avx512_bf16();
}

// Usage in adaptive matmul
if (can_use_native_bf16_gemm()) {
    // Use cblas_sbgemm (safe on Cooper Lake+ / Ice Lake Xeon+)
    cblas_sbgemm(...);
} else {
    // Fall back to BF16→FP32 expansion + cblas_sgemm
    expand_to_fp32_and_gemm(...);
}
```

This ensures correctness on all CPU generations at the cost of slightly lower performance on older hardware (which lacks native BF16 anyway).

---

## Impact

### Severity
**Medium** - Bug affects legitimate use case (BF16 inference on Cascade Lake) but:
- Only affects CPUs without AVX512_BF16 (2019 and older)
- Workaround available (fallback to FP32)
- Small matrices work correctly (API is sound)

### Affected Users
- Anyone using `cblas_sbgemm` on Cascade Lake or older CPUs
- Machine learning inference workloads using BF16 quantization
- Users expecting software emulation to "just work" on older hardware

### Recommended OpenBLAS Action
1. **Document limitation:** Clearly state that BF16 emulation is unreliable without native AVX512_BF16
2. **Fix emulation:** Debug and fix the software emulation path for large matrices
3. **Runtime detection:** Consider adding internal CPU feature detection and automatic fallback
4. **Test coverage:** Add tests for BF16 emulation on CPUs without AVX512_BF16

---

## Additional Context

### Why This Matters
BF16 (bfloat16) is increasingly important for:
- **LLM inference** (Qwen, LLaMA, etc.) - 2× memory bandwidth reduction
- **Training workloads** - Reduced precision without FP16 gradient issues
- **Edge deployment** - Running quantized models on older server hardware

Many production systems still run on Cascade Lake (launched 2019, still widely deployed in 2025). Users expect OpenBLAS's software emulation to provide a slower-but-correct fallback.

### Related Work
- **Intel MKL:** Has working BF16 GEMM on Cascade Lake (verified in our testing)
- **COSMA:** Alternative distributed BLAS with potential BF16 support
- **oneDNN:** Intel's deep learning primitives with robust BF16 emulation

---

## References

- **Minimal Reproducer:** `openblas_sbgemm_bug_reproducer.c` (attached)
- **CPU Specifications:** Intel Xeon Gold 6238R - <https://ark.intel.com/content/www/us/en/ark/products/199342/intel-xeon-gold-6238r-processor-38-5m-cache-2-20-ghz.html>
- **AVX512_BF16 Introduction:** Cooper Lake (2020) - <https://en.wikichip.org/wiki/x86/avx512_bf16>
- **Discovery Changelog:** `changelog/2025-01-19-bf16-openblas-large-matrix-bug-discovered.md`
- **Workaround Implementation:** `changelog/2025-10-19-cpu-feature-detection-bf16-fallback.md`

---

## Proposed OpenBLAS Issue Title

```
cblas_sbgemm produces NaN for large matrices on CPUs without AVX512_BF16 (software emulation bug)
```

## Proposed OpenBLAS Issue Labels

- `bug`
- `bfloat16`
- `x86_64`
- `emulation`

---

**Contact:** David Sanftenberg  
**Project:** Llaminar LLM Inference Engine  
**Repository:** <https://github.com/dbsanfte/llaminar>
