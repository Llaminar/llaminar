# IPC Improvement Attempt: 4× K-Loop Unrolling

**Date**: October 26, 2025  
**Status**: ⚠️ Implementation failed, needs redo  
**Objective**: Improve IPC from ~1.17 to 1.3-1.5+ in L1-optimized GEMM kernel

---

## Executive Summary

Attempted to implement 4× K-loop unrolling to improve IPC in `QuantizedGemmL1Opt.cpp`. The **strategy is sound** but the **implementation corrupted the file** due to brace mismatch errors. File was reverted cleanly. Ready to reimplement more carefully.

**Current Performance**:
- Throughput: 326 GFLOPS
- L1 Miss Rate: 0.43%
- IPC: ~1.17

**Target Performance**:
- Throughput: 326+ GFLOPS (maintain or improve)
- L1 Miss Rate: ≤0.43% (maintain)
- IPC: 1.3-1.5+ (33-50% improvement)

---

## Problem Analysis

### IPC Bottleneck Diagnosis

**Current Implementation** (2× K-loop unrolling):
- Main loop processes 32 floats per iteration (2 iterations of 16 each)
- Register usage: 
  - 48 ZMM registers for accumulators (c00-c77)
  - 8 ZMM registers for A temps (a0-a7)
  - 6 ZMM registers for B temps (b0-b5)
  - **Total: 62 logical ZMM registers**

**Hardware Constraint**:
- AVX512 provides only **32 physical ZMM registers**
- 62 logical → 32 physical = **register spilling to memory**
- Memory spills hurt IPC (extra load/store instructions)

**Measured Impact**:
- IPC: ~1.17 (below 1.5-2.0 target)
- Cause: Register pressure from 8×6 micro-kernel

### Strategy Evaluation

**Option 1: Sub-Blocked 4×6 Micro-Kernel** ❌ **REJECTED**

Process 8×6 block as two sequential 4×6 sub-blocks:
- **Block 1** (rows 0-3): 24 accumulators + 4 A + 6 B = **34 registers** ✅ Fits!
- **Block 2** (rows 4-7): 24 accumulators + 4 A + 6 B = **34 registers** ✅ Fits!

**Benefit**: Eliminates register spilling entirely

**Problem**: **Requires iterating K dimension twice**
- First pass: Compute rows 0-3, load B_panel data
- Second pass: Compute rows 4-7, **reload same B_panel data**

**Impact**: Hurts L1 cache efficiency
- Current: Each B element accessed once per iteration
- Sub-blocked: Each B element accessed **twice per iteration**
- L1 cache lines evicted and reloaded → higher miss rate

**Decision**: **REJECTED** - Cache downsides outweigh register benefits

---

**Option 2: 4× K-Loop Unrolling** ✅ **SELECTED**

Increase main loop from 32-element to 64-element chunks:
- **Before**: `for (; p + 32 <= k_panel; p += 32)` - 2 iterations of 16
- **After**: `for (; p + 64 <= k_panel; p += 64)` - 4 iterations of 16

**Benefits**:
1. **50% reduction in loop overhead**
   - Fewer loop counter increments, condition checks, branches
2. **More instructions per iteration**
   - 192 FMAs per iteration (vs 96 currently)
   - Better out-of-order execution opportunities
3. **Better instruction scheduling**
   - Compiler has more freedom to reorder instructions
   - Can hide load latencies behind FMAs
4. **Amortized register spilling**
   - Same 62 registers, but more work per spill event
   - Spilling cost divided over 4× more computation

**Register Pressure**: **Unchanged at 62**
- Still spills, but less frequently relative to work done

**Cache Impact**: **None**
- Same access pattern - each element accessed once
- Prefetch updated from 64 to 128 floats ahead

**Expected IPC**: **1.3-1.5+** (33-50% improvement)

---

## Implementation Design

### Loop Hierarchy Structure

```cpp
// Main loop: 4× unrolling (64-element chunks)
for (; p + 64 <= k_panel; p += 64) {
    // Iteration 1: Process p+0 to p+15
    // Iteration 2: Process p+16 to p+31
    // Iteration 3: Process p+32 to p+47
    // Iteration 4: Process p+48 to p+63
}

// Cleanup loop 1: 2× unrolling (32-element chunks)
for (; p + 32 <= k_panel; p += 32) {
    // Iteration 1: Process p+0 to p+15
    // Iteration 2: Process p+16 to p+31
}

// Cleanup loop 2: 1× unrolling (16-element chunks)
for (; p + 16 <= k_panel; p += 16) {
    // Iteration 1: Process p+0 to p+15
}

// Tail loop: Scalar code for k_panel % 16 remaining elements
int p_remainder = (k_panel / 16) * 16;
for (int p = p_remainder; p < k_panel; ++p) {
    // Scalar FMAs
}
```

### Per-Iteration Code Pattern

Each 16-element iteration follows this pattern:

```cpp
// Load A panel (8 rows × 16 floats)
__m512 a0 = _mm512_loadu_ps(A_panel + 0 * k_panel + p);
__m512 a1 = _mm512_loadu_ps(A_panel + 1 * k_panel + p);
// ... a2-a7 ...

// Load B panel (6 columns × 16 floats)
__m512 b0 = _mm512_loadu_ps(B_panel + 0 * k_panel + p);
__m512 b1 = _mm512_loadu_ps(B_panel + 1 * k_panel + p);
// ... b2-b5 ...

// 48 FMA operations (8 rows × 6 columns)
c00 = _mm512_fmadd_ps(a0, b0, c00); c01 = _mm512_fmadd_ps(a0, b1, c01);
c02 = _mm512_fmadd_ps(a0, b2, c02); c03 = _mm512_fmadd_ps(a0, b3, c03);
// ... all 48 FMAs ...
```

**Code Size Per Iteration**: ~50 lines
**Total Added**: ~150-180 lines for all loops

### Prefetch Adjustment

```cpp
// Before (2× unrolling):
_mm_prefetch(reinterpret_cast<const char*>(A_panel + 0 * k_panel + p + 64), _MM_HINT_T0);

// After (4× unrolling):
_mm_prefetch(reinterpret_cast<const char*>(A_panel + 0 * k_panel + p + 128), _MM_HINT_T0);
```

Prefetch 128 floats (2048 bytes) ahead to match new loop stride.

---

## Implementation Attempt

### Approach Used

Incremental edits via multiple `replace_string_in_file` operations:
1. Update main loop header to 64-element chunks
2. Update prefetch distance to 128 floats
3. Add iterations 3 and 4 to main loop (~140 lines)
4. Add 32-element cleanup loop (~120 lines)
5. Add 16-element cleanup loop (~60 lines)

**Total Changes**: +227 lines, -6 lines

### Build Failures

**First Error** (after adding iterations 3-4):
```
Line 546: error: 'i' does not name a type
Line 571: error: 'QuantizedGemmL1Opt' has not been declared
Line 601: error: 'QuantizedGemmL1Opt' has not been declared
Line 632: error: expected declaration before '}'
```
**Symptom**: Function scope corruption

**Second Error** (after adding 32-element loop):
```
Line 653: error: 'QuantizedGemmL1Opt' has not been declared
Line 684: error: expected declaration before '}'
```
**Symptom**: pack_B_panel function not recognized

**Final Error** (after adding 16-element loop):
```
Line 480: error: 'a3' was not declared in this scope
  (also: a4-a7, b0-b5 not in scope)
Line 501: error: expected unqualified-id before 'for'
Lines 554-584: error: 'c_scalar' does not name a type
  (all horizontal reduction code appearing at global scope)
Line 587: error: 'k_panel' was not declared in this scope
Line 598: error: writeback loop at global scope
Lines 623, 653: error: 'QuantizedGemmL1Opt' has not been declared
Line 684: error: expected declaration before '}'
```

### Root Cause Analysis

**Problem**: **32-element cleanup loop has malformed structure**

The second iteration of the 32-element loop was added with incorrect brace placement. This caused the `micro_kernel` function to **close prematurely**. All code after (horizontal reduction, tail loop, writeback) appeared at **global scope** instead of inside the function.

**Evidence**:
1. Variables (a3, b0-b5) not in scope → loop ended prematurely
2. c_scalar assignments at global scope → outside function
3. for loops at global scope → outside function
4. QuantizedGemmL1Opt not declared → namespace corruption
5. Brace mismatch at file end → structural corruption

**Expected Code Size**: ~150-180 lines  
**Actual Changes**: +227 lines  
**Conclusion**: Code duplication or structural errors present

### Resolution

```bash
git checkout -- src/v2/kernels/cpu/QuantizedGemmL1Opt.cpp
```

File reverted cleanly to clean state (commit e623b24).

---

## Next Steps

### Immediate Action Plan

**1. Reimplement Using Single Large Edit** (RECOMMENDED)

Instead of incremental edits, replace the entire AVX512 micro-kernel section (lines ~146-550) in one operation:

```cpp
// Replace entire section:
#ifdef __AVX512F__
    // Initialize accumulators
    // ...
    // Main 64-element loop (4× unrolling)
    // Cleanup 32-element loop (2× unrolling)
    // Cleanup 16-element loop (1× unrolling)
    // Horizontal reduction
    // Tail loop
    // Writeback
#else
    // Scalar fallback
#endif
```

**Benefits**:
- Single atomic change
- Easier to verify brace matching
- Clearer code structure
- Less error-prone

**2. Incremental Validation**

After large edit:
- Count opening/closing braces: `grep -o '{' | wc -l` vs `grep -o '}' | wc -l`
- Verify function closes correctly: Check last 50 lines
- Build immediately: `cmake --build build_v2_release --target v2_perf_l1_opt_comparison`

**3. Performance Testing**

```bash
# Build Release
cmake --build build_v2_release --target v2_perf_l1_opt_comparison --parallel

# Run benchmark
./build_v2_release/tests/v2/v2_perf_l1_opt_comparison

# Measure IPC
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses \
  ./build_v2_release/tests/v2/v2_perf_l1_opt_comparison
```

**Expected Results**:
- Throughput: 326+ GFLOPS (maintain or improve)
- L1 Miss: ≤0.43% (maintain)
- IPC: 1.3-1.5+ (target improvement)

### Alternative Approaches (If 4× Insufficient)

**If IPC < 1.3 after 4× unrolling:**

**Option A: Software Pipelining**

Interleave loads from iteration N+1 with FMAs from iteration N:

```cpp
// Iteration i
load a0-a7, b0-b5 at p+i*16       // Load for iteration i
FMA with a0-a7, b0-b5 from i-1   // Compute iteration i-1
```

**Benefit**: Hides load latency behind computation  
**Expected**: +10-20% additional IPC

**Option B: Manual Instruction Scheduling**

Reorder FMA instructions to alternate between accumulator groups:

```cpp
a0 = load(...);   // Load
c00 = FMA(...);   // FMA row 0
a1 = load(...);   // Load (hide latency)
c10 = FMA(...);   // FMA row 1
```

**Benefit**: Better pipeline utilization  
**Effort**: High (manual optimization)

---

## Lessons Learned

### What Worked
✅ **Rigorous strategy evaluation** - Sub-blocking rejected for good reasons  
✅ **Cache-aware design** - Preserved L1 efficiency  
✅ **Clear loop hierarchy** - Cascading cleanup structure sound  
✅ **Performance baseline** - 326 GFLOPS, 0.43% L1 miss validated  

### What Failed
❌ **Incremental file editing** - Error-prone with large code blocks  
❌ **Brace tracking** - Hard to verify across multiple edits  
❌ **No intermediate validation** - Built only after all changes  

### Best Practices for Next Attempt

1. **Use single large replacement** - Atomic change is safer
2. **Verify braces immediately** - Count before building
3. **Build after each major change** - Catch errors early
4. **Read back edited sections** - Verify structure looks correct
5. **Start with smaller test** - Maybe 2× → 3× unrolling first

---

## Technical Foundation

**Baseline (Commit e623b24)**:
- ✅ pack_B_panel stride fix committed
- ✅ All tests passing: 23/23 unit, 5/5 integration
- ✅ Performance validated: 326 GFLOPS, 0.43% L1 miss, ~1.17 IPC

**Target File**: `src/v2/kernels/cpu/QuantizedGemmL1Opt.cpp`  
**Target Function**: `QuantizedGemmL1Opt::micro_kernel` (lines ~140-621)  
**Changes Required**: ~150-180 lines (main loop, 2 cleanup loops, prefetch)

**No Changes Needed**:
- Header file: `QuantizedGemmL1Opt.h` - micro_kernel signature unchanged
- Benchmark: `Perf__L1OptimizationComparison.cpp` - uses class via interface
- Tests: All existing tests will validate correctness

---

## Performance Expectations

### Before (2× Unrolling)
- **Throughput**: 326 GFLOPS
- **L1 Miss**: 0.43%
- **IPC**: ~1.17
- **Loop Overhead**: Main loop processes 32 floats per iteration

### After (4× Unrolling)
- **Throughput**: **326+ GFLOPS** (maintain or improve)
- **L1 Miss**: **≤0.43%** (maintain cache efficiency)
- **IPC**: **1.3-1.5+** (target: 33-50% improvement)
- **Loop Overhead**: **50% reduction** (64 floats per iteration)

### Improvement Sources
1. **Reduced branch mispredictions** - Fewer loop iterations
2. **Better out-of-order execution** - More independent instructions per iteration
3. **Improved instruction scheduling** - Compiler optimization opportunities
4. **Amortized register spilling** - More work between spill events

---

## Risk Assessment

### Low Risk
✅ **Cache efficiency maintained** - Same access pattern, just longer stride  
✅ **Correctness preserved** - Same FMA operations, just reorganized  
✅ **Fallback available** - Can revert to 2× if problems arise  

### Medium Risk
⚠️ **Code size increase** - +150-180 lines, may affect i-cache  
⚠️ **Compiler behavior** - Optimization may differ with larger loop bodies  

### Mitigation
- Benchmark thoroughly before committing
- Compare perf counters (i-cache misses, branch predictions)
- Keep 2× version available for comparison

---

## Timeline Estimate

**Reimplement and Validate**: 30-60 minutes
1. Prepare single large edit: 10-15 min
2. Apply and verify braces: 5 min
3. Build and fix any errors: 5-10 min
4. Run L1OptComparison benchmark: 2 min
5. Measure IPC with perf stat: 5 min
6. Analyze results and document: 10-15 min

**If Successful**: 
- Commit with performance numbers
- Update documentation
- Total time: ~1 hour

**If IPC < Target**:
- Investigate with profiler
- Try software pipelining or instruction scheduling
- Additional time: 1-2 hours

---

## Conclusion

The **4× K-loop unrolling strategy is sound** and well-justified:
- ✅ Preserves cache efficiency (vs sub-blocking)
- ✅ Reduces loop overhead by 50%
- ✅ Provides better instruction scheduling opportunities
- ✅ Expected to improve IPC by 33-50%

The **implementation just needs careful execution**:
- Use single large file edit (not incremental)
- Verify brace matching before building
- Test incrementally and validate performance

**Ready to proceed** when next session begins.

---

## References

**Related Files**:
- `src/v2/kernels/cpu/QuantizedGemmL1Opt.cpp` - Target file (reverted cleanly)
- `src/v2/kernels/cpu/QuantizedGemmL1Opt.h` - Header (no changes needed)
- `tests/v2/performance/Perf__L1OptimizationComparison.cpp` - Benchmark
- `changelog/2025-10-26-hybrid-nc-strategy-implementation.md` - Previous work

**Performance History**:
- Commit e623b24: 326 GFLOPS, 0.43% L1 miss, ~1.17 IPC (current baseline)
- Previous: 335-451 GFLOPS range during NC tuning experiments

**Next Document**: Will be `2025-10-26-4x-unrolling-ipc-results.md` after successful implementation
