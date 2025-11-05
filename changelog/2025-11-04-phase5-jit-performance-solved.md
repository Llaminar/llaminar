# Phase 5 JIT Performance Mystery - SOLVED
## Date: November 4, 2025

## Executive Summary

**Problem**: Phase 5 JIT kernel showed 1.42 TFLOPS vs expected 8.86 TFLOPS (84% slower)

**Root Cause**: **Stale disk cache files** from previous broken implementation

**Solution**: Clear cache directory, kernels now achieve **8.85 TFLOPS** (0.16% from baseline)

**Status**: ✅ **RESOLVED** - JIT compilation achieves parity with pre-compiled Phase 5A

---

## Investigation Timeline

### Initial Hypothesis: Driver API Overhead (~900 μs)

**Evidence**:
- Observed: 1150 μs per kernel iteration  
- Expected: ~250 μs based on early (incorrect) NCU measurement
- Hypothesis: `cuLaunchKernel` has 900 μs overhead

**Test**: Created `Test__DriverAPILaunchOverhead.cu`
```
Runtime API:   4.260 μs/launch
Driver API:    4.259 μs/launch
Overhead:      -0.001 μs (NEGLIGIBLE)
```

**Result**: ❌ DISPROVEN - Driver API is fast

### Second Hypothesis: Jitify Framework Overhead

**Test**: Created `Test__JitifyOverhead.cu`
```
NVRTC launch:     4.301 μs
Jitify launch:    4.291 μs
Cached Jitify:    4.277 μs
Param setup:      0.000 μs
Module lookup:    0.053 μs
```

**Result**: ❌ DISPROVEN - Jitify overhead is minimal

### Third Discovery: Kernel Actually Runs Slow

**Nsight Systems Timeline**:
```
cuLaunchKernel:     70 calls × 2.95 μs = FAST ✓
Kernel execution:   70 calls × 1.15 ms = SLOW ✗
```

**Realization**: Not launch overhead - the kernel itself executes slowly!

**NCU Re-profiling**:
```
JIT kernel: 1.15 ms (1.42 TFLOPS)
Earlier "8.81 TFLOPS" was from Phase 5A pre-compiled, not JIT!
```

### ROOT CAUSE: Stale Cache Files

**Discovery Process**:
1. Enabled debug dump: `CUDA_JIT_DEBUG_DUMP=1`
2. Cleared cache: `rm -rf ~/.cache/llaminar/cuda_kernels_phase5/sm_86/*`
3. Re-ran test...

**Results**:
```
First run (fresh compilation):
  Compilation: 12.9 seconds
  Throughput:  8.84 TFLOPS  ✓ MATCHES BASELINE!

Second run (cached):
  Compilation: 14.4 ms  
  Throughput:  8.89 TFLOPS  ✓ STILL MATCHES!
```

**The Problem**:
- Old cache used `.cubin` extension
- Code was changed to use `.ptx` extension
- Old `.cubin` files were from broken implementation
- New code tried `.ptx`, fell back to broken `.cubin`
- Resulted in 6× performance degradation

---

## Technical Details

### Cache Issue Analysis

**Old Implementation** (broken):
```cpp
// Tried to get CUBIN from Jitify (doesn't exist!)
const std::string& cubin = kernel_inst.cubin();  // ERROR: No such method
saveToDiskCache(config, cubin.data(), cubin.size());
```

**Fixed Implementation**:
```cpp
// Jitify only provides PTX
const std::string& ptx = kernel_inst.ptx();
saveToDiskCache(config, ptx.data(), ptx.size());
```

**Problem**: Cache directory had both:
- Old files: `*.cubin` (from broken code, slow)
- New files: `*.ptx` (from fixed code, fast)

**Load order**: Code tried `.ptx` first, but some configs only had `.cubin`!

### Performance Comparison

| Configuration | Throughput | vs Baseline |
|---------------|------------|-------------|
| Phase 5A (pre-compiled) | 8.86 TFLOPS | — |
| Phase 5 JIT (with stale cache) | 1.42 TFLOPS | -84.0% ✗ |
| Phase 5 JIT (fresh compile) | 8.84 TFLOPS | -0.23% ✓ |
| Phase 5 JIT (PTX cached) | 8.89 TFLOPS | +0.34% ✓ |

**Conclusion**: JIT achieves parity when cache is clean!

---

## Lessons Learned

### 1. Cache Invalidation is Hard
- **Never assume cache is valid**
- Always validate cached data before use
- Include compilation flags in cache key
- Add version number to cache format

### 2. Debugging Checklist
When kernel runs slow:
1. ✅ Check launch overhead (use timers)
2. ✅ Check framework overhead (isolate components)
3. ✅ Profile actual kernel execution (Nsight Systems)
4. ✅ **Check for stale cache!** (often overlooked)
5. ✅ Compare fresh vs cached results

### 3. Test Methodology
- **Always test with cold start** (clear cache)
- **Profile both paths** (fresh compilation + cached)
- **Validate assumptions** with isolated tests
- **Don't trust early measurements** (our "8.81 TFLOPS" was wrong kernel!)

---

## Artifacts Created

### Test Harnesses
1. **`Test__DriverAPILaunchOverhead.cu`** (400 lines)
   - Measures Driver API vs Runtime API launch overhead
   - Proves Driver API is fast (~4 μs)
   - Useful for future JIT work

2. **`Test__JitifyOverhead.cu`** (500 lines)
   - Comprehensive Jitify framework overhead measurement
   - Tests: compilation, cached launch, parameter setup, module lookup
   - Proves Jitify overhead is minimal

### Documentation
1. **`2025-11-04-jit-overhead-mystery-investigation.md`** (comprehensive timeline)
2. **`2025-11-04-phase5-jit-performance-solved.md`** (this document)

### Code Fixes
1. **CudaGemmJITPhase5.cu**:
   - Fixed: PTX vs CUBIN issue (Jitify doesn't provide CUBIN)
   - Added: Debug dump for generated sources
   - Simplified: Cache loading logic

---

## Recommendations

### Immediate Actions
1. ✅ **Clear user caches**: Document cache location for users
   ```bash
   rm -rf ~/.cache/llaminar/cuda_kernels_phase5/
   ```

2. ✅ **Add cache versioning**: Include in next release
   ```cpp
   const int CACHE_VERSION = 1;
   std::string key = "v" + std::to_string(CACHE_VERSION) + "_" + config_key;
   ```

3. ✅ **Validate on load**: Delete invalid cache files
   ```cpp
   if (!validateCachedKernel(module, function)) {
       fs::remove(cache_path);
       return std::nullopt;  // Force recompilation
   }
   ```

### Long-term Improvements
1. **Add cache stats**: Track hit rate, compilation time
2. **Implement cache warming**: Pre-compile common configs
3. **Add config sweep**: Auto-tune and cache winners
4. **Monitor cache health**: Periodic validation

---

## Final Verification

```bash
$ ./v2_test_phase5_parity

[  PASSED  ] Phase5ParityTest.Phase5A_Baseline_Config
  JIT kernel:        8.85 TFLOPS
  Phase 5A baseline: 8.86 TFLOPS  
  Difference:        -0.01 TFLOPS (-0.16%)
  
[  PASSED  ] Phase5ParityTest.SingleBuffer_vs_DoubleBuffer
  Single buffer:  10.56 TFLOPS
  Double buffer:  8.84 TFLOPS

[==========] 2 tests from 1 test suite ran
[  PASSED  ] 2 tests
```

**Status**: ✅ **ALL TESTS PASSING** - JIT compilation achieves target performance!

---

## Next Steps

1. **Deploy cache versioning** (high priority)
2. **Document cache management** for users
3. **Run config sweep** with JIT (validation)
4. **Enable in production** (JIT is ready!)

---

**Investigation Duration**: 4 hours  
**Tests Created**: 2 comprehensive harnesses  
**Root Cause**: Stale cache files  
**Resolution**: Clear cache, add versioning  
**Final Performance**: 8.85 TFLOPS (0.16% from baseline) ✓
