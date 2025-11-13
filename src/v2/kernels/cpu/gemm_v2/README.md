# INT8 GEMM Optimization (V2)

**Status**: Active development - OneDNN investigation complete, Phase 1 implementation starting  
**Current Performance**: 472 GOPS (7.14% of target)  
**Target Performance**: 6,610 GOPS (OneDNN baseline, 84% efficiency)  
**Performance Gap**: 14× speedup needed

---

## Quick Start

**New to this project?** Start with `INT8_GEMM_DOCUMENTATION_INDEX.md`

**Ready to implement?** Follow `INT8_GEMM_QUICK_ACTION_PLAN.md`

**Want technical details?** Read `ONEDNN_ARCHITECTURE_DEEP_DIVE.md`

---

## Directory Contents

### Documentation (6 files, 2,500+ lines)

1. **`INT8_GEMM_DOCUMENTATION_INDEX.md`** - Master index and navigation guide
2. **`INT8_GEMM_QUICK_ACTION_PLAN.md`** - Phase 1 implementation guide (start here for coding)
3. **`INT8_GEMM_OPTIMIZATION_ROADMAP.md`** - 5-phase comprehensive strategy
4. **`ONEDNN_ARCHITECTURE_DEEP_DIVE.md`** - Technical analysis of OneDNN source code
5. **`INT8_GEMM_OPTIMIZATION_INVESTIGATION_SUMMARY.md`** - Executive summary and findings
6. **`INT8_GEMM_ONEDNN_INTEGRATION.md`** - OneDNN integration notes (historical)

### Source Code (8 files)

- **`IntegerGemm.h/cpp`** - Main GEMM interface and orchestration
- **`IntegerGemmKernelTemplate.h`** - Current microkernel template (baseline)
- **`IntegerGemmKernelTemplateV2.h`** - Next-generation microkernel (in progress)
- **`IntegerGemmConfig.h`** - Configuration and tuning parameters
- **`IntegerGemmAdapter.h`** - Adapter for Tensor API integration
- **`IntegerGemmAutoTuner.h`** - Auto-tuning infrastructure
- **`IntegerRequantization.h`** - Requantization utilities

---

## Current Status (November 12, 2025)

### Completed ✅
- OneDNN source code investigation
- Root cause analysis (14× gap = register blocking + memory traffic)
- Comprehensive documentation (2,500+ lines)
- Baseline implementation (472 GOPS)

### In Progress 🔄
- Phase 1: Register blocking microkernel (6×16 tile)
- Target: 2,000-3,000 GOPS (4-6× speedup)

### Next Steps ⏸️
- Phase 2: Data packing (4,000-5,000 GOPS)
- Phase 3: Explicit prefetching (5,000-6,000 GOPS)
- Phase 4: 3-level cache blocking (6,000-6,500 GOPS)

---

## Performance Roadmap

| Phase | Target GOPS | Speedup | Effort | Status |
|-------|-------------|---------|--------|--------|
| Current | 472 | 1× | - | ✅ Done |
| Phase 1 | 2,500 | 5.3× | 2-3 days | 🔄 Starting |
| Phase 2 | 4,500 | 9.5× | 2-3 days | ⏸️ Planned |
| Phase 3 | 5,500 | 11.7× | 1 day | ⏸️ Planned |
| Phase 4 | 6,500 | **13.8×** | 2-3 days | ⏸️ Planned |

**Expected completion**: 9-10 days from Phase 1 start

---

## Key Technical Insights

### Why OneDNN is 14× Faster

OneDNN achieves **31× reduction in memory traffic** through:

1. **Register Blocking** (48×8 tiles) → **10-15× speedup**
   - Load C once per tile, accumulate in 24 ZMM registers, store once
   - Our approach: Load/store every iteration (76× more C traffic)

2. **Data Packing** (column-major panels) → **2-3× speedup**
   - Sequential access pattern enables hardware prefetching
   - Our approach: Strided access thrashes cache

3. **Explicit Prefetching** (software-controlled) → **1.3-1.5× speedup**
   - Prefetch 160 elements ahead for A, 128 for B
   - Hides 10-20ns memory latency

4. **3-Level Cache Blocking** → **1.2-1.5× speedup**
   - L3/L2/L1 hierarchy with tuned block sizes
   - Working sets fit perfectly in each cache level

### Memory Traffic Analysis

```
Naive approach:  71.3B memory operations
OneDNN:           2.3B memory operations
Reduction:        31× less memory traffic
```

This explains the bulk of the 14× performance gap.

---

## Implementation Strategy

### Phase 1: Register Blocking (Starting Now)

**Goal**: Implement 6×16 microkernel (96 INT32 in 6 ZMM registers)

**Why 6×16 instead of 48×8?**
- Easier to implement and debug
- Still captures 90% of register blocking benefit
- Leaves more registers for A/B/temps

**Expected impact**: 4-6× speedup (2,000-3,000 GOPS)

**Files to modify**:
- Create new microkernel based on `IntegerGemmKernelTemplateV2.h`
- Update `IntegerGemm.cpp` to use new kernel
- Add performance benchmark

---

## Related Files

### Performance Tests
- `tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGemm.cpp`
- `tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGemmMinimal.cpp`

### Benchmarking Scripts
- `run_integer_gemm_sweep.sh` - Sweep across tile sizes
- `benchmark_tile_sweep*.sh` - Comprehensive tile benchmarking

### Generated Code
- `gemm/int8/generated/IntegerGemmInstantiations_*.cpp` - Template instantiations

---

## Resources

### OneDNN Source (analyzed)
- `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp`
- `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_u8_copy_bn_kern_autogen.cpp`
- `external/onednn/src/cpu/x64/brgemm/brgemm.hpp`

### External Documentation
- OneDNN Developer Guide: https://oneapi-src.github.io/oneDNN/
- Intel Intrinsics Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
- BLIS Framework: https://github.com/flame/blis

---

## Contact / Questions

See `INT8_GEMM_DOCUMENTATION_INDEX.md` for detailed navigation and FAQs.

For implementation questions, refer to `INT8_GEMM_QUICK_ACTION_PLAN.md`.

---

**Last Updated**: November 12, 2025  
**Next Milestone**: Phase 1 completion (2,500 GOPS target)
