# Q16_1 JIT Residual Fusion - Complete Implementation Plan

## Executive Summary

The current Q16_1 residual fusion implementation in `JitFusedAttentionWo.h` is **incomplete**. Only one code path (`emit_wo_projection`) has Q16_1 support, but this function is **dead code** - never called in actual execution.

The actual execution paths are:
1. **DECODE**: `generate_decode_kernel` → `emit_wo_projection_batched` → NO Q16_1 support
2. **PREFILL**: `generate_prefill_kernel` → `emit_prefill_wo_projection_tile` → NO Q16_1 support

This document provides a comprehensive plan to add Q16_1 residual fusion to all code paths.

---

## Current Code Architecture Analysis

### JIT Kernel Entry Points

```
generate()
├── generate_decode_kernel()     [seq_len_q == 1]
│   └── emit_wo_projection_batched()
│       ├── Q8_1_VNNI_PACKED: llaminar2_wo_q8_1_vnni_packed_gemm() [external C]
│       └── Other formats: emit_wo_projection_with_reg_offset()
│           └── emit_wo_projection_fallback_with_reg_offset()
│
└── generate_prefill_kernel()    [seq_len_q > 1]
    └── emit_prefill_normalize_project()
        └── emit_prefill_wo_projection_tile()
            ├── Q8_1: emit_prefill_wo_q8_1_tile()
            ├── Q8_1_VNNI_PACKED: llaminar2_wo_q8_1_vnni_packed_gemm() [external C]
            └── Other formats: emit_prefill_wo_projection() [fallback loop]
```

### Q16_1 Fusion Status by Path

| Path | Function | Q16_1 Support | Notes |
|------|----------|---------------|-------|
| DECODE | `emit_wo_projection_batched` | ❌ None | Main decode entry point |
| DECODE | `emit_wo_projection_with_reg_offset` | ❌ None | Fallback for non-VNNI |
| DECODE | `emit_wo_projection_fallback_with_reg_offset` | ❌ None | Scalar fallback |
| PREFILL | `emit_prefill_wo_projection_tile` | ❌ None | Main prefill entry point |
| PREFILL | `emit_prefill_wo_q8_1_tile` | ❌ None | Optimized Q8_1 path |
| PREFILL | `emit_prefill_wo_projection` | ❌ None | Fallback loop |
| PREFILL | `emit_prefill_wo_fp32` | ❌ None | FP32 weights |
| PREFILL | `emit_prefill_wo_fp16` | ❌ None | FP16 weights |
| PREFILL | `emit_prefill_wo_bf16` | ❌ None | BF16 weights |
| PREFILL | `emit_prefill_wo_q8_1` | ❌ None | Q8_1 weights |
| UNUSED | `emit_wo_projection` | ✅ Implemented | **DEAD CODE - never called** |
| UNUSED | `emit_single_query_attention` | N/A | Contains `emit_wo_projection` call, but never called |

---

## Implementation Plan

### Phase 5a: Decode Path Q16_1 Support

**Priority: HIGH** (decode is the most common inference operation)

#### 5a.1: Modify `emit_wo_projection_batched()`

**File**: `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h`
**Lines**: ~4995-5090

**Current behavior**:
- Calculates FP32 output pointer: `output + batch_start * d_model * 4`
- Calls GEMM to produce FP32 output

**Required changes**:
1. When `config_.fuse_residual_add && config_.residual_type == Q16_1`:
   - Calculate Q16_1 output pointer: `output + batch_start * num_blocks * 72`
   - Allocate temporary FP32 buffer for GEMM output
   - After GEMM, call `emit_q16_1_residual_fusion_batched()` for all queries in batch
   - Cleanup temporary buffer

```cpp
void emit_wo_projection_batched(...) {
    // Calculate output pointer based on fusion mode
    if (config_.fuse_residual_add && config_.residual_type == Q16_1) {
        int num_blocks = (d_model + 31) / 32;
        int q16_1_row_stride = num_blocks * 72;
        // reg_out_ptr = output + batch_start * q16_1_row_stride
        mov(reg_out_ptr, reg_batch_start);
        imul(reg_out_ptr, reg_out_ptr, q16_1_row_stride);
        add(reg_out_ptr, reg_output);
        
        // Allocate temp FP32 buffer on stack
        int temp_size = batch_size * d_model * 4;  // or use context_buffer
        // ... GEMM to temp buffer ...
        // ... loop: emit_q16_1_residual_fusion for each query ...
    } else {
        // Existing FP32 path
    }
}
```

#### 5a.2: Create `emit_q16_1_residual_fusion_batched()`

**New function** to fuse residual for multiple queries in batch.

```cpp
void emit_q16_1_residual_fusion_batched(
    const Xbyak::Reg64& reg_residual_base,  // Q16_1 output buffer
    const Xbyak::Reg64& reg_wo_base,        // FP32 Wo output buffer  
    int batch_size,
    int d_model)
{
    // Loop over batch entries
    for (int b = 0; b < batch_size; ++b) {
        // Calculate offsets for this batch entry
        // Call emit_q16_1_residual_fusion() for this entry
    }
}
```

#### 5a.3: Modify `emit_wo_projection_with_reg_offset()`

**Lines**: ~5597-5640

Similar changes for the fallback path when not using VNNI packed format.

---

### Phase 5b: Prefill Path Q16_1 Support

**Priority: MEDIUM** (prefill is less frequent than decode)

#### 5b.1: Modify `emit_prefill_wo_projection_tile()`

**Lines**: ~3557-3640

**Current dispatch**:
```cpp
switch (config_.wo_format) {
    case Q8_1: emit_prefill_wo_q8_1_tile(...);
    case Q8_1_VNNI_PACKED: llaminar2_wo_q8_1_vnni_packed_gemm(...);
    default: fallback loop calling emit_prefill_wo_projection(...)
}
```

**Required changes**:
1. Add outer check for `config_.fuse_residual_add`
2. When Q16_1 fusion enabled:
   - Allocate temp FP32 buffer for tile
   - Run existing Wo projection to temp buffer
   - Loop over tile queries calling `emit_q16_1_residual_fusion()`

```cpp
void emit_prefill_wo_projection_tile(...) {
    if (config_.fuse_residual_add && config_.residual_type == Q16_1) {
        // Allocate temp buffer: tile_size * d_model * 4 bytes
        int temp_size = (tile_size * d_model * 4 + 63) & ~63;
        sub(rsp, temp_size);
        
        // Run existing Wo projection to temp buffer (adjust offsets)
        // ... dispatch by wo_format to temp ...
        
        // Q16_1 residual fusion for each query in tile
        emit_prefill_q16_1_residual_tile(
            reg_output, rsp, tile_size, d_model,
            tile_start_spill, tile_size_spill);
        
        add(rsp, temp_size);
    } else {
        // Existing path
    }
}
```

#### 5b.2: Create `emit_prefill_q16_1_residual_tile()`

**New function** for tile-based Q16_1 fusion.

```cpp
void emit_prefill_q16_1_residual_tile(
    const Xbyak::Reg64& reg_residual_base,
    const Xbyak::Reg64& reg_wo_base,
    int tile_size,
    int d_model,
    int tile_start_spill,
    int tile_size_spill)
{
    // Loop over queries in tile
    L(".q16_1_tile_loop");
    // Calculate Q16_1 residual offset: (tile_start + q_local) * num_blocks * 72
    // Calculate FP32 Wo offset: q_local * d_model * 4
    // Call emit_q16_1_residual_fusion()
    // Increment, loop
}
```

#### 5b.3: Modify `emit_prefill_wo_projection()` (Fallback)

**Lines**: ~4032-4085

Update the single-query prefill projection to support Q16_1 fusion.

---

### Phase 5c: External GEMM Functions

**Priority: LOW** (external C functions need separate handling)

The external C functions (`llaminar2_wo_q8_1_vnni_packed_gemm`, `llaminar2_wo_fp32_streaming_dequant_gemm`) always output FP32. Q16_1 fusion must happen AFTER these calls return.

Current locations that call external GEMM:
1. `emit_wo_projection_batched()` line ~5020
2. `emit_prefill_wo_projection_tile()` line ~5597

These are already addressed by phases 5a and 5b - we'll add Q16_1 fusion as a post-processing step after the GEMM call.

---

### Phase 5d: Stack Layout Updates

Several functions need updated stack layouts when Q16_1 fusion is enabled:

1. **`generate_decode_kernel()`**: 
   - Add temp FP32 buffer space for batched Wo output
   - May need to reduce `wo_batch_size` to fit

2. **`generate_prefill_kernel()`**:
   - Stack already has `context_offset` buffer which could be reused
   - Or allocate additional temp space in `emit_prefill_wo_projection_tile()`

---

## Testing Plan

### Unit Tests (Already Done)
- `Test__JitFusedAttentionWo_Q16_1Residual.cpp` - Tests `emit_q16_1_residual_fusion()` directly

### Integration Tests (New)

1. **`Test__JitFusedAttentionWo_Q16_1Fused_Decode_vs_FP32.cpp`**
   - seq_len=1 (decode path)
   - Compare fused Q16_1 output against FP32 reference
   
2. **`Test__JitFusedAttentionWo_Q16_1Fused_Prefill_vs_FP32.cpp`**
   - seq_len>1 (prefill path)
   - Compare fused Q16_1 output against FP32 reference

3. **`Test__JitFusedAttentionWo_Q16_1Fused_Batched_vs_FP32.cpp`**
   - Multiple queries with batch Wo projection
   - Test batch-size edge cases

---

## Implementation Order

1. **Phase 5a.1**: `emit_wo_projection_batched()` - enables decode path
2. **Phase 5a.2**: `emit_q16_1_residual_fusion_batched()` - helper for batch
3. **Test**: Run decode integration test
4. **Phase 5b.1**: `emit_prefill_wo_projection_tile()` - enables prefill path  
5. **Phase 5b.2**: `emit_prefill_q16_1_residual_tile()` - helper for tile
6. **Test**: Run prefill integration test
7. **Phase 5a.3**: `emit_wo_projection_with_reg_offset()` - fallback path
8. **Phase 5b.3**: `emit_prefill_wo_projection()` - fallback path
9. **Full test suite**: All paths covered

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stack overflow from temp buffers | High | Calculate max buffer size, validate against stack limit |
| Register pressure in fusion loop | Medium | Reuse scratch registers, spill if needed |
| Performance regression | Medium | Benchmark before/after, optimize hot path |
| Numerical divergence | Low | Q16_1 has high precision, should match FP32 closely |

---

## Cleanup Tasks

After Q16_1 fusion is working in all paths:

1. **Remove dead code**: 
   - `emit_single_query_attention()` - never called
   - `emit_wo_projection()` (single-query version) - only called by dead code
   
2. **Consolidate fusion logic**:
   - Consider a unified `emit_fused_residual()` that works for both decode/prefill
   
3. **Update project plan**:
   - Mark Phase 5 complete
   - Document integration points for Phase 6 (FFN fusion)
