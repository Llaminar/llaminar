# Intel MKL Integration Plan

**Date**: October 19, 2025  
**Author**: David Sanftenberg  
**Status**: Planning → Implementation

## Executive Summary

Integrate Intel oneAPI Math Kernel Library (oneMKL) as an alternative BLAS backend to unlock native BF16 GEMM support via `cblas_gemm_bf16bf16f32()`, working around the OpenBLAS `cblas_sbgemm` bug on CPUs without AVX512_BF16.

## Why MKL?

### Current Problem
- **OpenBLAS BF16 Bug**: `cblas_sbgemm` produces NaN on large matrices (64×896×896) on Cascade Lake CPUs
- **Fallback Overhead**: BF16→FP32 expansion adds ~5-10% overhead
- **No Timeline**: OpenBLAS fix timeline unknown

### MKL Solution
- **Better BF16 Emulation**: Intel MKL has superior software emulation for BF16 on non-native CPUs
- **Production-Tested**: Widely used in HPC and ML workloads
- **API Compatibility**: Similar `cblas_gemm_bf16bf16f32()` function with FP32 accumulation
- **Performance**: Competitive with or better than OpenBLAS on Intel CPUs
- **Compiler Agnostic**: **Works perfectly with GCC** - no Intel compiler needed!

## Compiler Compatibility

✅ **GCC is fully supported!** Intel MKL works with:
- GCC 4.8+ (tested with GCC 8.3, 11.x, 12.x, 13.x)
- Clang/LLVM
- Intel ICC/ICX (optional, not required)

The library is ABI-compatible and performance is excellent with GCC. Intel-specific optimizations are **inside the library**, not compiler-dependent.

## Installation Options

### Option 1: APT Repository (Recommended for Ubuntu/Debian)
```bash
# Add Intel APT repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
  https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install intel-oneapi-mkl-devel
```

**Size**: ~500MB download, ~2GB installed  
**Location**: `/opt/intel/oneapi/mkl/`

### Option 2: Standalone Installer
Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html

### Option 3: Docker/Dev Container Integration
Add to `.devcontainer/Dockerfile`:
```dockerfile
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | tee /etc/apt/sources.list.d/oneAPI.list \
 && apt-get update \
 && apt-get install -y intel-oneapi-mkl-devel \
 && apt-get clean
```

## CMake Integration Strategy

### Design Goals
1. **Coexist with OpenBLAS**: Keep OpenBLAS as default, MKL as optional backend
2. **User Choice**: `USE_MKL=ON/OFF` CMake option
3. **Runtime Selection**: `LLAMINAR_QUANT_BF16_PREFER_MKL=1` environment flag
4. **Fallback Chain**: MKL BF16 → OpenBLAS BF16 → BF16→FP32 expansion → FP32 GEMM
5. **No Breaking Changes**: Existing builds unaffected

### CMake Structure

```cmake
# Option: Enable Intel MKL backend
option(USE_MKL "Enable Intel MKL backend for BF16 GEMM" OFF)

if(USE_MKL)
    find_package(MKL REQUIRED)
    if(MKL_FOUND)
        add_compile_definitions(HAVE_MKL)
        message(STATUS "Intel MKL found: ${MKL_ROOT}")
        message(STATUS "MKL include: ${MKL_INCLUDE_DIRS}")
        message(STATUS "MKL libraries: ${MKL_LIBRARIES}")
    endif()
endif()

# Link targets
if(USE_MKL)
    target_link_libraries(llaminar_core PRIVATE ${MKL_LIBRARIES})
    target_include_directories(llaminar_core PRIVATE ${MKL_INCLUDE_DIRS})
endif()
```

### MKL Linking Options

Intel MKL offers different linking models:

**Static Linking (Recommended for deployment)**:
```cmake
set(MKL_LINK static)
set(MKL_THREADING_VENDOR gnu_omp)  # Use GNU OpenMP (GCC)
set(MKL_INTERFACE lp64)  # 32-bit integers
```

**Dynamic Linking (Better for development)**:
```cmake
set(MKL_LINK dynamic)
set(MKL_THREADING_VENDOR gnu_omp)
set(MKL_INTERFACE lp64)
```

## Code Changes Required

### 1. AdaptiveMatmul Backend Enum (src/AdaptiveMatmul.h)

```cpp
enum class BLASBackend {
    OPENBLAS,
    MKL,
    COSMA
};

BLASBackend current_backend = BLASBackend::OPENBLAS;

// Runtime backend selection based on environment + availability
BLASBackend selectBLASBackend() {
    #ifdef HAVE_MKL
    if (debugEnv().quant.bf16_prefer_mkl) {
        return BLASBackend::MKL;
    }
    #endif
    return BLASBackend::OPENBLAS;
}
```

### 2. MKL BF16 GEMM Wrapper (src/backends/MKLBackend.h - NEW FILE)

```cpp
#pragma once
#ifdef HAVE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include "utils/BFloat16.h"
#include "utils/Logging.h"

namespace llaminar {

class MKLBackend {
public:
    static bool multiplyBF16(
        const float* A,           // FP32 activations [m×k]
        const bfloat16* B_bf16,   // BF16 weights [k×n]
        float* C,                 // FP32 output [m×n]
        int m, int n, int k,
        float alpha = 1.0f,
        float beta = 0.0f,
        bool transpose_A = false,
        bool transpose_B = false
    ) {
        // Convert bfloat16* to MKL_BF16*
        const MKL_BF16* B_mkl = reinterpret_cast<const MKL_BF16*>(B_bf16);
        
        // MKL BF16 GEMM: C = alpha * A * B + beta * C
        // Note: cblas_gemm_bf16bf16f32 requires BOTH inputs as BF16
        // We need to convert A (FP32 activations) to BF16 first
        
        size_t A_size = m * k;
        std::vector<bfloat16> A_bf16(A_size);
        
        #pragma omp parallel for if (A_size > 32768) schedule(static)
        for (size_t i = 0; i < A_size; ++i) {
            A_bf16[i] = bfloat16::from_float(A[i]);
        }
        
        const MKL_BF16* A_mkl = reinterpret_cast<const MKL_BF16*>(A_bf16.data());
        
        CBLAS_TRANSPOSE transA = transpose_A ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE transB = transpose_B ? CblasTrans : CblasNoTrans;
        
        int lda = transpose_A ? m : k;
        int ldb = transpose_B ? k : n;
        int ldc = n;
        
        try {
            cblas_gemm_bf16bf16f32(
                CblasRowMajor,
                transA, transB,
                m, n, k,
                alpha,
                A_mkl, lda,
                B_mkl, ldb,
                beta,
                C, ldc
            );
            
            LOG_DEBUG("[MKL] BF16 GEMM succeeded: " << m << "×" << k << " × " 
                      << k << "×" << n << " → " << m << "×" << n);
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR("[MKL] BF16 GEMM failed: " << e.what());
            return false;
        }
    }
};

} // namespace llaminar

#endif // HAVE_MKL
```

### 3. Integrate MKL into AdaptiveMatmul (src/AdaptiveMatmul.h)

```cpp
bool multiplyBF16(...) {
    auto backend = selectBLASBackend();
    
    #ifdef HAVE_MKL
    if (backend == BLASBackend::MKL) {
        bool ok = MKLBackend::multiplyBF16(A, B_bf16, C, m, n, k, alpha, beta, 
                                           transpose_A, transpose_B);
        if (ok) return true;
        LOG_WARN("[MKL] BF16 GEMM failed, falling back to OpenBLAS");
    }
    #endif
    
    // Fallback to OpenBLAS (existing code)
    if (debugEnv().quant.bf16_gemm) {
        // Try OpenBLAS cblas_sbgemm...
    }
    
    // Final fallback: BF16→FP32 expansion
    // ...existing code...
}
```

### 4. Environment Flag (src/utils/DebugEnv.h)

```cpp
struct QuantSnapshotConfig {
    // ... existing fields ...
    
    bool bf16_prefer_mkl = false;  // LLAMINAR_QUANT_BF16_PREFER_MKL
};
```

### 5. Update Documentation (README.md, BENCHMARK_QUICK_REFERENCE.md)

```markdown
## Building with Intel MKL

### Install MKL
\`\`\`bash
# Ubuntu/Debian
sudo apt install intel-oneapi-mkl-devel

# Or download from Intel website
\`\`\`

### Build with MKL
\`\`\`bash
cmake -B build -S . -DUSE_MKL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
\`\`\`

### Runtime MKL Selection
\`\`\`bash
# Prefer MKL for BF16 GEMM
export LLAMINAR_QUANT_BF16_PREFER_MKL=1
./run_llaminar.sh -m model.gguf
\`\`\`
```

## Implementation Timeline

### Phase 1: Infrastructure (Day 1-2, ~8 hours)
- [ ] Add CMake `find_package(MKL)` with proper configuration (2h)
- [ ] Create `src/backends/MKLBackend.h` with BF16 GEMM wrapper (2h)
- [ ] Add `HAVE_MKL` preprocessor guards (1h)
- [ ] Update environment flags in DebugEnv (1h)
- [ ] Integrate MKL backend selection in AdaptiveMatmul (2h)

### Phase 2: Testing (Day 3, ~6 hours)
- [ ] Test MKL BF16 GEMM with small matrices (2×2, 64×64) (1h)
- [ ] Test production matrix sizes (64×896×896, 512×4096×4096) (2h)
- [ ] Run `TestBF16Conversion` with MKL backend (1h)
- [ ] Verify parity tests pass with MKL (2h)

### Phase 3: Performance Validation (Day 4-5, ~6 hours)
- [ ] Benchmark MKL BF16 vs OpenBLAS BF16 vs BF16→FP32 fallback (3h)
- [ ] End-to-end decode performance with MKL (2h)
- [ ] Document performance characteristics (1h)

### Phase 4: Documentation & Cleanup (Day 6, ~4 hours)
- [ ] Update README.md with MKL build instructions (1h)
- [ ] Update BENCHMARK_QUICK_REFERENCE.md (1h)
- [ ] Add MKL troubleshooting guide (1h)
- [ ] Update architecture document (1h)

**Total Estimated Time**: 24 hours (3 days)

## Performance Expectations

### MKL Advantages on Intel CPUs
- **Tuned for Intel**: Hand-optimized assembly for all Intel microarchitectures
- **Better BF16 Emulation**: Superior software emulation vs OpenBLAS
- **NUMA-Aware**: Built-in NUMA optimizations
- **Thread Scaling**: Excellent multi-socket performance

### Expected Speedups
- **Small ops (<8K elements)**: Similar to OpenBLAS (~0.9-1.1× parity)
- **Medium ops (8K-1M elements)**: 1.1-1.3× faster than OpenBLAS
- **Large ops (>1M elements)**: 1.2-1.5× faster than OpenBLAS
- **BF16 GEMM**: **Should actually work** (unlike OpenBLAS bug)

### Realistic Goals
- **Primary Goal**: Eliminate BF16→FP32 expansion overhead (~5-10% gain)
- **Secondary Goal**: Competitive or better than OpenBLAS FP32 performance
- **Stretch Goal**: 1.2× overall speedup on full decode workload

## Risk Assessment

### Low Risk ✅
- **Coexistence**: MKL and OpenBLAS can coexist, OpenBLAS remains default
- **Fallback Chain**: Multiple fallback options if MKL fails
- **Community Proven**: MKL + GCC is well-tested combination
- **Easy Rollback**: Can disable with `-DUSE_MKL=OFF`

### Medium Risk ⚠️
- **Binary Size**: Static linking adds ~50-100MB to binary
  - *Mitigation*: Use dynamic linking for development
- **Licensing**: Intel Simplified Software License (free for all uses)
  - *Mitigation*: License is permissive, no restrictions for our use case
- **Dependencies**: Requires Intel repos setup
  - *Mitigation*: Document clearly, provide Docker/devcontainer setup

### High Risk ❌
- None identified

## Alternative Considered: Custom Kernel

**Rejected** because:
- Fused dequant kernels already tested and failed (6-130× slower)
- BF16→FP32 expansion is only ~5-10% overhead
- MKL provides professional-grade implementation
- Custom kernel maintenance burden too high

## Success Metrics

### Week 1 (MKL Integration)
- ✅ MKL builds successfully with GCC
- ✅ `cblas_gemm_bf16bf16f32` works on 64×896×896 matrices
- ✅ Parity tests pass with MKL backend
- ✅ No regressions in FP32 performance

### Month 1 (Performance Validation)
- ✅ BF16 GEMM operational (no NaN, correct results)
- ✅ 5-10% speedup from eliminating BF16→FP32 expansion
- ✅ Competitive with or better than OpenBLAS on large ops
- ✅ Documentation complete

### Month 2 (Production Deployment)
- ✅ MKL backend default for BF16 workloads
- ✅ Multi-node MPI testing complete
- ✅ Performance reports published

## Next Steps

1. **Install MKL** in dev container (30 minutes)
2. **Add CMake integration** (2 hours)
3. **Create MKLBackend wrapper** (2 hours)
4. **Test small matrices** (1 hour)
5. **Test production sizes** (2 hours)
6. **Benchmark performance** (3 hours)
7. **Document** (2 hours)

**Start Date**: October 19, 2025  
**Target Completion**: October 22, 2025 (3 days)

## References

- [Intel MKL Developer Guide (2024.1)](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/overview.html)
- [cblas_gemm_bf16bf16f32 API Reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2024-0/cblas-gemm-bf16bf16f32-compute.html)
- [MKL CMake Integration](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/cmake-config-for-onemkl.html)
- [Intel Community: GCC + MKL Discussion](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/GCC-Compilers-and-MKL/m-p/1126110)
