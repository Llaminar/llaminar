# V2 Performance Test Documentation in copilot-instructions.md

**Date**: October 24, 2025  
**Author**: GitHub Copilot + User  
**Session**: Phase 13 - Documentation Update

## Summary

Added comprehensive V2 performance testing documentation to `.github/copilot-instructions.md` and clarified V1 vs V2 performance testing approaches to prevent developer confusion.

## Changes Made

### 1. Added V2 Performance Testing Section

**Location**: After V2 Testing Conventions section (lines 805-875)

**Content Added** (70 lines):
- Complete subsection: "V2 Performance Testing"
- Running V2 performance tests with CTest
- Available benchmarks (IQ4_NL GEMM)
- Optimal settings automatically applied
- Adding new V2 performance tests
- Best practices
- Cross-references to comprehensive documentation

**Key Features Documented**:
```bash
# Run all V2 performance tests
cd build_v2
ctest -L Performance --verbose

# Run specific benchmarks
ctest -L "Performance;GEMM" --verbose
ctest -R "V2_Perf_IQ4NL_GEMM" --verbose
```

**Comparison Table** (V1 vs V2):
- Command: `./run_benchmark.sh` vs `ctest -L Performance`
- Build management: Manual vs Auto-detected
- Settings: External script vs CTest properties

### 2. Clarified V1 Performance Test References

**Updated Sections**:

1. **"Performance Testing Scripts"** → **"Performance Testing Scripts (V1)"** (line 463)
   - Clarified these are V1-specific scripts
   - `run_batch_performance.sh`, `run_performance_demo.sh`, `run_pytorch_parity_test.sh`

2. **"Generic Benchmark Runner"** (line 493)
   - Added note: "This script is primarily for V1 component benchmarks"
   - Added cross-reference: "V2 performance tests use CTest directly"
   - Linked to V2 Performance Testing section

### 3. Cross-References Added

**Documentation Links**:
- `tests/v2/performance/README.md` (500+ lines comprehensive guide)
- `changelog/2025-10-24-v2-performance-test-framework.md` (framework details)
- `.github/instructions/llaminar-v2-architecture.instructions.md` (V2 architecture)

## File Inventory

**Modified**:
- `.github/copilot-instructions.md` (+70 lines, 3 sections updated)

**Created**:
- `changelog/2025-10-24-v2-performance-docs-copilot-instructions.md` (this file)

## Documentation Structure

**V1 Performance Testing** (still valid, production):
- Line 349: Benchmark Mode (`run_llaminar.sh --benchmark`)
- Line 463: Performance Testing Scripts (V1)
- Line 493: Generic Benchmark Runner (primarily V1)

**V2 Performance Testing** (new):
- Line 805: V2 Performance Testing section
- Line 811: Running V2 Performance Tests
- Line 829: Available V2 Benchmarks
- Line 842: Adding New V2 Performance Tests

## Developer Impact

**Before**:
- No V2 performance test documentation in copilot-instructions.md
- Unclear if `run_benchmark.sh` works for V2
- Developers might miss CTest-integrated approach

**After**:
- ✅ Clear V2 performance test documentation
- ✅ V1 vs V2 distinction explicit
- ✅ CTest usage documented with examples
- ✅ Cross-references to comprehensive guides
- ✅ No confusion about which approach to use

## Usage Examples

**V1 Component Benchmarks** (still valid):
```bash
./run_benchmark.sh benchmark_iq4nl_gemm
./run_benchmark.sh test_batch_performance
```

**V2 Performance Tests** (new approach):
```bash
# From workspace root
cd build_v2
ctest -L Performance --verbose

# Or specific tests
ctest -L "Performance;IQ4_NL" --verbose
ctest -R "V2_Perf_IQ4NL_GEMM" --verbose
```

## Cross-Reference Validation

**Links Verified**:
- ✅ `tests/v2/performance/README.md` exists (500+ lines)
- ✅ `changelog/2025-10-24-v2-performance-test-framework.md` exists
- ✅ `.github/instructions/llaminar-v2-architecture.instructions.md` exists
- ✅ `tests/v2/CMakeLists.txt` (lines 1-100) referenced correctly

## Integration with Existing Documentation

**Table of Contents**:
- Performance Optimization (line 21) - unchanged
- V2 Performance Testing - new subsection under V2 Testing Conventions

**Related Sections**:
- V2 Testing Conventions (line 575) - parent section
- CTest Label Best Practices (line 679) - complements labeling discussion
- V2 Build Process (earlier in file) - referenced for Release builds

## Success Criteria

✅ V2 performance testing documented in copilot-instructions.md  
✅ Clear V1 vs V2 distinction for performance testing  
✅ `ctest -L Performance` usage documented with examples  
✅ Cross-references to comprehensive README included  
✅ No stale/confusing V1 performance test references  
✅ V1 production benchmark docs preserved (still valid)  

## Next Steps

**Completed**:
- V2 performance test framework implementation ✅
- Comprehensive README (tests/v2/performance/README.md) ✅
- Developer guide documentation (copilot-instructions.md) ✅

**Future Enhancements**:
- Add more V2 benchmarks (Attention, RoPE, etc.)
- Multi-GPU heterogeneous benchmarks (when V2 supports)
- Performance comparison dashboard (V1 vs V2)

## Related Changes

**Previous Session** (Phase 12):
- `changelog/2025-10-24-v2-performance-test-framework.md`
- Created performance test framework infrastructure
- Implemented IQ4_NL GEMM benchmark
- Created comprehensive README

**This Session** (Phase 13):
- Updated canonical developer guide
- Clarified V1 vs V2 performance approaches
- Cross-referenced comprehensive documentation

## Testing

**Documentation Validation**:
- ✅ Markdown formatting correct
- ✅ Code blocks syntax highlighted
- ✅ Cross-references valid
- ✅ Table formatting correct
- ✅ Sections numbered consistently

**Developer Experience**:
- ✅ V1 developers: Clear which scripts to use
- ✅ V2 developers: Clear CTest approach
- ✅ New contributors: Understand V1 vs V2 distinction
- ✅ No confusion about optimal settings (auto-applied in V2)

---

**Session Status**: ✅ COMPLETE - Documentation update finished successfully
