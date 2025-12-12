# JIT Microkernel Infrastructure for Q8_1 Fused Attention

**Date:** December 12, 2025

## Summary

Completed the JIT microkernel infrastructure for Phase 4 of the Q8_1 fused attention work. This establishes the foundation for generating optimized runtime-compiled attention kernels.

## Changes

### New JIT Header Files

1. **`src/v2/kernels/cpu/jit/JitMicrokernelBase.h`**
   - Base class for JIT microkernels using Xbyak
   - Defines ZMM register zones for composability:
     - Accumulator (zmm0-7): Results that persist
     - Input (zmm8-15): Loaded inputs
     - State (zmm16-19): Running algorithm state
     - Scratch (zmm20-25): Temporaries
     - Constants (zmm26-31): Preserved across calls
   - Common SIMD helpers: horizontal sum/max, broadcasts, debug emission

2. **`src/v2/kernels/cpu/jit/JitQ8DotProduct.h`**
   - μK1: Q8_1 dot product emitter
   - Uses AVX-512 VNNI (vpdpbusd) for unsigned × signed dot product
   - Handles FP16 scale dequantization and sum_qs bias correction
   - Methods: `emit_single_block_dot()`, `emit_multi_block_dot()`

3. **`src/v2/kernels/cpu/jit/JitFastExp.h`**
   - μK5: Fast vectorized exp() using polynomial approximation
   - Range reduction to [-ln(2)/2, ln(2)/2]
   - 6th-degree polynomial + vscalefps for final scaling
   - Methods: `emit_fast_exp()`, `emit_exp2_poly()`

4. **`src/v2/kernels/cpu/jit/JitOnlineSoftmax.h`**
   - μK2: Online softmax state management
   - Streaming algorithm - no need to store all scores
   - Handles max tracking and correction factors
   - Methods: `emit_init_softmax()`, `emit_update_softmax()`, `emit_finalize()`

5. **`src/v2/kernels/cpu/jit/JitVWeightedAccum.h`**
   - μK3: Weighted V accumulation
   - Accumulates context += weight * dequant(V)
   - Handles rescaling when max changes
   - Methods: `emit_init_context()`, `emit_weighted_accum()`, `emit_rescale_context()`, `emit_normalize_context()`

6. **`src/v2/kernels/cpu/jit/JitWoProjection.h`**
   - μK4: Wo output projection
   - Supports FP32, Q8_1, FP16, BF16 weight types
   - Fuses into attention without materializing context
   - Methods: `emit_project_fp32()`, `emit_project_q8_1()`, `emit_project_fp16()`, `emit_project_bf16()`

7. **`src/v2/kernels/cpu/jit/JitFusedAttentionWo.h`**
   - Composed kernel using all microkernels
   - Full fused attention: Q·K → softmax → ·V → Wo
   - Single-pass streaming design
   - Tiled execution for cache efficiency

### Test File

**`tests/v2/unit/attention/Test__JitMicrokernels.cpp`** - 13 tests:
- JIT infrastructure tests (register zones, constants)
- Register convention tests (no overlap, full coverage)
- JIT kernel generation tests (minimal kernel)
- Emitter include tests (verify headers compile)
- Test utilities (FP16 conversion)

## Test Results

All 157 Q8_1 attention tests pass:

| Test Suite | Tests | Status |
|------------|-------|--------|
| Microkernel FastExp | 13 | PASSED |
| Microkernel OnlineSoftmax | 9 | PASSED |
| Microkernel Q8DotProduct | 10 | PASSED |
| Microkernel VWeightedAccum | 8 | PASSED |
| Microkernel WoProjection | 18 | PASSED |
| FusedAttentionWo Ref | 8 | PASSED |
| FusedAttentionWo Batch | 12 | PASSED |
| FusedAttentionWo Tiled | 21 | PASSED |
| FusedAttentionWo Robustness | 28 | PASSED |
| Q8_1 FusedAttention Unit | 17 | PASSED |
| JIT Microkernels | 13 | PASSED |
| **TOTAL** | **157** | **PASSED** |

## Technical Notes

### Xbyak Register Naming

When using Xbyak, named register constants like `xmm8`, `ymm4` are in the `Xbyak::util::` namespace. For portability, use explicit construction:

```cpp
// Instead of: Ymm ymm_q = ymm4;
// Use:
Ymm ymm_q(4);
Xmm xmm_tmp(8);
```

### Register Zone Design

The ZMM register partitioning enables microkernel composition:

```
zmm0-7:   Accumulators - persist across microkernel calls
zmm8-15:  Inputs - may be clobbered by microkernels
zmm16-19: State - online algorithm state (max, sum, etc.)
zmm20-25: Scratch - freely clobbered temporaries
zmm26-31: Constants - preserved, shared across microkernels
```

### Emitter Pattern

JIT emitters take a `JitMicrokernelBase&` reference and emit code:

```cpp
class JitQ8DotProductEmitter {
public:
    void emit_single_block_dot(JitMicrokernelBase& gen, 
                               const Xbyak::Reg64& q_ptr,
                               const Xbyak::Reg64& k_ptr,
                               const Xbyak::Xmm& dst_xmm);
};
```

## Next Steps

1. **Emitter invocation tests** - Create tests that actually call emitter methods
2. **JitFusedAttentionWo composition** - Wire up all microkernels into a complete JIT kernel
3. **Numerical parity tests** - Compare JIT output against reference implementation
4. **Benchmark JIT vs reference** - Measure speedup from JIT compilation

## File Locations

```
src/v2/kernels/cpu/jit/
├── JitMicrokernelBase.h
├── JitQ8DotProduct.h
├── JitFastExp.h
├── JitOnlineSoftmax.h
├── JitVWeightedAccum.h
├── JitWoProjection.h
└── JitFusedAttentionWo.h

tests/v2/unit/attention/
└── Test__JitMicrokernels.cpp
```
