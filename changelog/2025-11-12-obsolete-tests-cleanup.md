# Cleanup: Obsolete Template-Based Test Files

**Date**: November 12, 2025  
**Related**: `2025-11-12-autotuner-registry-fix.md`

## Changes Made

Renamed obsolete template-based test files and removed them from the build to avoid confusion and prevent accidental use.

### Files Renamed to `.old` / `.OLD`

1. **`Perf__IntegerGEMM_FullSweep.cpp`** → `Perf__IntegerGEMM_FullSweep.cpp.OLD`
   - **Reason**: Superseded by `Perf__IntegerGEMM_FullSweep_Registry.cpp`
   - **Issue**: Template explosion - tried to instantiate ~3,456+ configurations in single file
   - **Build impact**: Hours of compilation (or infinite hang)
   - **Replacement**: Registry-based version compiles in seconds

2. **`Perf__IntegerGEMM_TileSweep.cpp`** → `Perf__IntegerGEMM_TileSweep.cpp.old`
   - **Reason**: Work completed, conclusion reached (MR=8 optimal)
   - **Superseded by**: Full sweep via registry-based FullSweep test
   - **Status**: Historical artifact, no longer needed

### Build System Changes

**File**: `tests/v2/CMakeLists.txt`

Commented out obsolete test registrations:

```cmake
# BEFORE:
add_executable(v2_perf_integer_gemm_tile_sweep performance/cpu/kernels/gemm/Perf__IntegerGEMM_TileSweep.cpp)
target_link_libraries(v2_perf_integer_gemm_tile_sweep ...)

# AFTER:
# OBSOLETE: Comprehensive sweep of MR, UNROLL_K, PREFETCH_DIST, cache blocking params
# Superseded by Perf__IntegerGEMM_FullSweep_Registry.cpp (registry-based, no template explosion)
# Conclusion: MR=8 found optimal
# add_executable(v2_perf_integer_gemm_tile_sweep performance/cpu/kernels/gemm/Perf__IntegerGEMM_TileSweep.cpp.old)
# ...
```

### Files Retained (Still Active)

The following template-based tests remain in the build:

1. **`Perf__IntegerGEMM_Minimal.cpp`**
   - **Purpose**: Minimal debugging harness for isolated bug reproduction
   - **Template usage**: Very limited (2 instantiations)
   - **Build impact**: Negligible (~seconds)
   - **Status**: Keep - useful debugging tool

2. **`Perf__IntegerGEMM_QwenProfile.cpp`**
   - **Purpose**: Profiling specific Qwen 0.5B operations
   - **Template usage**: Limited (representative configs only)
   - **Build impact**: Minimal
   - **Status**: Keep - useful profiling tool

### Rationale

**Problem**: Template-based tests that sweep large configuration spaces cause:
- Hours of compilation time (or infinite hangs)
- Single-threaded template expansion (compiler bottleneck)
- Exponential memory usage during compilation
- Confusion for developers (why is cpptools using 100% CPU?)

**Solution**: Use kernel registry pattern:
- Pre-generated 64-shard system (`IntegerGemmInstantiations_00.cpp` through `_63.cpp`)
- Parallel compilation (64 files compiled simultaneously)
- Runtime dispatch via `IntegerGemmKernelRegistry`
- Test code has zero template instantiation overhead
- Build time: seconds instead of hours

**Guideline**: 
- ✅ **DO**: Use registry for sweeping large config spaces (>100 configs)
- ✅ **DO**: Use templates for minimal/debugging tests (<10 configs)
- ❌ **DON'T**: Instantiate hundreds of template combinations in test files

## Verification

CMake configuration succeeds after changes:
```bash
$ cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
-- Configuring done (10.5s)
-- Generating done (0.4s)
-- Build files have been written to: /workspaces/llaminar/build_v2_release
```

## Impact

- **Build system**: Cleaner, obsolete tests removed
- **Compilation time**: No change (tests were already superseded)
- **Developer clarity**: Reduced confusion about which tests to use
- **Maintenance**: Easier to understand current test suite

## Next Steps

If these obsolete files are confirmed unnecessary after autotuner pipeline completion, they can be deleted entirely:

```bash
# After pipeline completes and results validated:
rm tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_FullSweep.cpp.OLD
rm tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_TileSweep.cpp.old
```

## See Also

- `2025-11-12-autotuner-registry-fix.md` - Root cause analysis and registry-based solution
- `AUTOTUNER_CTEST_INTEGRATION.md` - CTest integration documentation
- `src/v2/kernels/cpu/gemm/int8/IntegerGemmKernelRegistry.h` - Registry implementation
