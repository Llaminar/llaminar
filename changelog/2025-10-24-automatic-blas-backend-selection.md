# Automatic BLAS Backend Selection

**Date**: 2025-10-24  
**Status**: ✅ Implemented  
**Component**: V2 Build System  

---

## Summary

Implemented **automatic BLAS backend selection** at CMake configure time based on CPU vendor detection:
- **Intel CPUs** → Prefer Intel MKL (if available)
- **Non-Intel CPUs** (AMD, ARM, etc.) → Prefer OpenBLAS

This optimizes performance by using vendor-optimized libraries without manual configuration.

---

## Motivation

**Problem**: V2 had MKL support in code but never used it. Users had to manually enable MKL via `-DHAVE_MKL=ON`, and there was confusion about whether both backends could coexist.

**Solution**: Automatic backend selection at configure time based on CPU detection, with graceful fallback to OpenBLAS if MKL is unavailable.

---

## Implementation

### 1. CPU Vendor Detection (`src/v2/utils/CPUFeatures.h`)

Added CPUID-based CPU vendor detection:

```cpp
/**
 * @brief Detect CPU vendor
 * @return "GenuineIntel" for Intel, "AuthenticAMD" for AMD, or other vendor string
 */
inline const char* cpu_vendor() {
    static char vendor[13] = {0};
    static bool detected = false;
    
    if (!detected) {
        uint32_t regs[4];
        cpuid(0, 0, regs);
        // EBX, EDX, ECX contain 12-character vendor string
        *reinterpret_cast<uint32_t*>(vendor + 0) = regs[1]; // EBX
        *reinterpret_cast<uint32_t*>(vendor + 4) = regs[3]; // EDX
        *reinterpret_cast<uint32_t*>(vendor + 8) = regs[2]; // ECX
        vendor[12] = '\0';
        detected = true;
    }
    
    return vendor;
}

/**
 * @brief Check if CPU is Intel
 */
inline bool cpu_is_intel() {
    const char* vendor = cpu_vendor();
    return strcmp(vendor, "GenuineIntel") == 0;
}
```

### 2. CMake Backend Selection (`src/v2/CMakeLists.txt`)

Added automatic BLAS backend selection logic:

```cmake
# BLAS Backend Selection (MKL vs OpenBLAS)
set(BLAS_BACKEND "AUTO" CACHE STRING "BLAS backend: AUTO, MKL, or OPENBLAS")

if(BLAS_BACKEND STREQUAL "AUTO")
    # Detect CPU vendor at configure time
    execute_process(
        COMMAND bash -c "lscpu | grep 'Vendor ID:' | awk '{print $3}'"
        OUTPUT_VARIABLE CPU_VENDOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(CPU_VENDOR STREQUAL "GenuineIntel")
        message(STATUS "V2: Detected Intel CPU - preferring Intel MKL")
        set(BLAS_BACKEND_SELECTED "MKL")
    else()
        message(STATUS "V2: Detected non-Intel CPU (${CPU_VENDOR}) - preferring OpenBLAS")
        set(BLAS_BACKEND_SELECTED "OPENBLAS")
    endif()
else()
    set(BLAS_BACKEND_SELECTED ${BLAS_BACKEND})
    message(STATUS "V2: Manual BLAS backend selection: ${BLAS_BACKEND_SELECTED}")
endif()
```

### 3. Backend Compilation (`src/v2/backends/ComputeBackend.cpp`)

Backend type is determined at compile time via `HAVE_MKL` or `HAVE_OPENBLAS` preprocessor definitions:

```cpp
static ComputeDevice enumerate_cpu_device() {
    ComputeDevice dev;

#ifdef HAVE_MKL
    dev.type = ComputeBackendType::CPU_MKL;
    dev.name = "Intel MKL (CPU)";
#elif defined(HAVE_OPENBLAS)
    dev.type = ComputeBackendType::CPU_OPENBLAS;
    dev.name = "OpenBLAS (CPU)";
#else
    #error "No BLAS backend configured"
#endif
    // ...
}
```

### 4. GEMM Kernel (`src/v2/kernels/cpu/FP32GemmKernel.cpp`)

Uses standard CBLAS API which works with both backends:

```cpp
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

// Both MKL and OpenBLAS use the same cblas_sgemm API
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
```

---

## Usage

### Automatic Selection (Default)

```bash
# Configure - automatically detects CPU and chooses backend
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Release

# Output on Intel CPU:
# -- V2: Detected Intel CPU - preferring Intel MKL
# -- V2: Intel MKL found - using MKL backend
# -- V2: Selected BLAS backend: MKL

# Output on AMD/ARM CPU:
# -- V2: Detected non-Intel CPU (AuthenticAMD) - preferring OpenBLAS
# -- V2: Using OpenBLAS: /usr/lib/x86_64-linux-gnu/libopenblas.so
# -- V2: Selected BLAS backend: OPENBLAS
```

### Manual Override

```bash
# Force MKL (even on non-Intel CPU)
cmake -B build_v2 -S src/v2 -DBLAS_BACKEND=MKL

# Force OpenBLAS (even on Intel CPU)
cmake -B build_v2 -S src/v2 -DBLAS_BACKEND=OPENBLAS
```

### Installing MKL (for Intel CPUs)

```bash
# Ubuntu/Debian
sudo apt install intel-oneapi-mkl-devel

# Or download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
```

---

## Fallback Behavior

If the preferred backend is unavailable, CMake automatically falls back:

1. **Intel CPU + MKL not installed** → Falls back to OpenBLAS
   ```
   -- V2: Detected Intel CPU - preferring Intel MKL
   -- V2: Intel MKL requested but not found - falling back to OpenBLAS
   -- V2: Install MKL with: sudo apt install intel-oneapi-mkl-devel
   ```

2. **Non-Intel CPU + Only MKL available** → Uses MKL anyway
   ```
   -- V2: Detected non-Intel CPU (AuthenticAMD) - preferring OpenBLAS
   -- V2: OpenBLAS not found - using available MKL backend
   ```

---

## Technical Constraints

### Why Not Link Both Libraries?

**Symbol Conflicts**: MKL and OpenBLAS both export CBLAS symbols (`cblas_sgemm`, etc.). Linking both causes:
- Linker errors (duplicate symbols)
- Runtime crashes (symbol resolution ambiguity)

**Solution**: Choose **one** backend at CMake configure time based on CPU vendor.

### Why Not Runtime Selection?

**Requires Separate Binaries**: To support true runtime selection, you'd need:
1. Compile separate shared libraries for each backend
2. Load them dynamically with `dlopen()` at runtime
3. Manage function pointers and error handling

**Complexity vs Benefit**: Configure-time selection is simpler and covers 99% of use cases:
- Users rarely switch between Intel/AMD CPUs on the same system
- Docker/container deployments are CPU-homogeneous
- Performance difference is significant enough to justify recompilation

---

## Performance Characteristics

### Intel MKL (on Intel CPUs)
- ✅ Hardware-optimized for Intel microarchitecture
- ✅ AVX-512 optimizations on Skylake-X and newer
- ✅ BF16 GEMM acceleration on Ice Lake and newer
- ✅ Vectorized transcendental functions (exp, log, etc.)

### OpenBLAS (on AMD/ARM/Non-Intel)
- ✅ Generic optimizations for all x86-64 CPUs
- ✅ Good performance on AMD Zen architectures
- ✅ ARM NEON optimizations available
- ✅ Widely available and well-tested

### Benchmark Expectations (based on V1 data)
- **Intel Ice Lake + MKL**: ~15-25% faster than OpenBLAS on GEMM
- **AMD Zen 3/4 + OpenBLAS**: Better than MKL due to non-Intel optimizations
- **ARM Neoverse + OpenBLAS**: Only practical option (MKL is x86-only)

---

## Verification

### Check Active Backend

```bash
# During CMake configuration
cmake -B build_v2 -S src/v2 | grep "Selected BLAS backend"
# Output: -- V2: Selected BLAS backend: MKL  (or OPENBLAS)

# At runtime (when device enumeration is logged)
./build_v2/llaminar2 --list-devices
# Output: Device 0: Intel MKL (CPU) - 32 GB RAM
#    (or: Device 0: OpenBLAS (CPU) - 32 GB RAM)
```

### Test GEMM Performance

```bash
# Run GEMM benchmarks
cd build_v2
./tests/v2/v2_test_cpu_kernels --gtest_filter="*GEMM*"

# Compare MKL vs OpenBLAS builds on same hardware
cmake -B build_mkl -S src/v2 -DBLAS_BACKEND=MKL && cmake --build build_mkl
cmake -B build_openblas -S src/v2 -DBLAS_BACKEND=OPENBLAS && cmake --build build_openblas

./build_mkl/tests/v2/v2_test_cpu_kernels --gtest_filter="*GEMM*"
./build_openblas/tests/v2/v2_test_cpu_kernels --gtest_filter="*GEMM*"
```

---

## Files Modified

1. **`src/v2/utils/CPUFeatures.h`**: Added `cpu_vendor()` and `cpu_is_intel()` functions
2. **`src/v2/utils/BlasWrapper.h`**: Created (helper for future multi-backend support)
3. **`src/v2/CMakeLists.txt`**: Added automatic BLAS backend selection logic
4. **`src/v2/backends/ComputeBackend.cpp`**: Simplified to compile-time backend selection

---

## Future Enhancements

### Multi-Backend Runtime Selection (If Needed)

If true runtime selection becomes necessary:

1. **Separate Shared Libraries**:
   ```bash
   # Build MKL variant
   cmake -B build_mkl -S src/v2 -DBLAS_BACKEND=MKL -DBUILD_SHARED_LIBS=ON
   
   # Build OpenBLAS variant
   cmake -B build_openblas -S src/v2 -DBLAS_BACKEND=OPENBLAS -DBUILD_SHARED_LIBS=ON
   ```

2. **Dynamic Loading**:
   ```cpp
   // Load appropriate backend at runtime
   void* backend_lib = dlopen(is_intel ? "libllaminar2_mkl.so" : "libllaminar2_openblas.so", RTLD_NOW);
   auto create_kernel = (CreateKernelFunc)dlsym(backend_lib, "create_gemm_kernel");
   ```

3. **Distribution**: Package both `.so` files and select at runtime

**Current Status**: **Not implemented** - configure-time selection is sufficient for now.

---

## Conclusion

V2 now **automatically selects the optimal BLAS backend** based on CPU vendor:
- ✅ Intel CPUs use MKL (if installed) → Best performance on Intel hardware
- ✅ Non-Intel CPUs use OpenBLAS → Reliable performance everywhere
- ✅ Graceful fallback to OpenBLAS if MKL unavailable
- ✅ Manual override available via `-DBLAS_BACKEND=MKL|OPENBLAS`

This removes the need for users to manually configure backends while ensuring optimal performance across different CPU architectures.
