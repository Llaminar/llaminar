# CUDA Kernel Atom Parameter Expansion

**Date**: 2025-01-XX  
**Status**: Ready for testing  
**Estimated Test Duration**: 4-8 hours (Qwen 0.5B only)

## Summary

Expanded the CUDA GEMM kernel configuration space to include atom-level tensor core parameters (`atom_type`, `atom_layout_m`, `atom_layout_n`, `atom_layout_k`). Previously these were hardcoded, limiting flexibility for different quantization types and preventing optimal tensor core utilization.

## Motivation

### Profiling Results
- **Best config**: tile_16×16×32, threads_8×8, work_2×2 → 33.50 GFLOPS
- **Problem**: Only 4.17% GPU occupancy (2 warps/SM out of 32 possible)
- **Root cause**: Small block size (64 threads) limits parallelism

### Strategic Insight
- Hardcoded atom params assume IQ4_NL forever
- Other quant types (Q6_K, Q8_0, MXFP4) will have different block sizes
- Atom layout directly affects tensor core utilization on large tiles
- Need config space to learn optimal atom configs per shape

## Changes Made

### 1. Kernel Template (`__launch_bounds__`)

**File**: `src/v2/kernels/cuda/CudaGemmKernelTemplate.h`

```cpp
// Before:
extern "C" __global__ void quantized_gemm_kernel_iq4nl(...)

// After:
extern "C" __global__ void
__launch_bounds__(${THREADS_M} * ${THREADS_N}, 4)
quantized_gemm_kernel_iq4nl(...)
```

**Effect**: Compiler optimizes for minimum 4 blocks/SM, should improve occupancy

### 2. Kernel Variant Template (Atom Parameters)

**File**: `src/v2/kernels/cuda/CudaGemmVariantsBaseline.cu`

**Added template parameters**:
```cpp
template <
    // ... existing params ...
    int VECTORIZE_LOAD,
    int ATOM_TYPE = 0,          // 0=SM80_16x8x16, 1=SM80_16x8x8
    int ATOM_LAYOUT_M = 1,      // 1, 2, 4
    int ATOM_LAYOUT_N = 1,      // 1, 2, 4
    int ATOM_LAYOUT_K = 1       // Always 1 for SM80
>
__global__ void quantized_gemm_kernel_variant(...)
```

**Added validation**:
```cpp
static_assert(ATOM_TYPE == 0 || ATOM_TYPE == 1, ...);
static_assert(ATOM_LAYOUT_M == 1 || ATOM_LAYOUT_M == 2 || ATOM_LAYOUT_M == 4, ...);
// ... etc
```

### 3. Config Space Generator

**File**: `src/v2/kernels/cuda/generate_cuda_gemm_variants.py`

**Added parameter sweeps**:
```python
ATOM_TYPE_VALUES = [0, 1]              # SM80_16x8x16 vs SM80_16x8x8
ATOM_LAYOUT_M_VALUES = [1, 2, 4]       # M dimension tiling
ATOM_LAYOUT_N_VALUES = [1, 2, 4]       # N dimension tiling
ATOM_LAYOUT_K_VALUES = [1]             # Always 1 for SM80
```

**Updated config tuple**: 10 params → 14 params
- Old: `(tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n, prefetch, transpose, vectorize)`
- New: `(..., atom_type, atom_m, atom_n, atom_k)`

**Updated validation**:
```python
def is_valid_config(..., atom_type, atom_layout_m, atom_layout_n, atom_layout_k):
    # Atom compatibility checks
    atom_m_dim = 16
    atom_n_dim = 8
    atom_k_dim = 16 if atom_type == 0 else 8
    
    # Tile must be divisible by atom layout dimensions
    if tile_m % (atom_m_dim * atom_layout_m) != 0:
        return False
    # ... etc
```

### 4. Registry Updates

**File**: `src/v2/kernels/cuda/CudaGemmKernelRegistry.h`

**Updated key type**:
```cpp
// Before: 10-element tuple
using CudaKernelKey = std::tuple<int, int, int, int, int, int, int, int, bool, int>;

// After: 14-element tuple
using CudaKernelKey = std::tuple<int, int, int, int, int, int, int, int, bool, int, 
                                 int, int, int, int>;
```

**Updated registration**:
```cpp
void register_kernel(
    int tile_m, int tile_n, int tile_k,
    int threads_m, int threads_n,
    int work_m, int work_n,
    int prefetch, bool transpose, int vectorize,
    int atom_type, int atom_layout_m, int atom_layout_n, int atom_layout_k,  // NEW
    CudaKernelLauncher launcher)
```

### 5. Generated Code Updates

**File**: `src/v2/kernels/cuda/generate_cuda_gemm_variants.py`

**Updated launcher generation**:
```cpp
quantized_gemm_kernel_variant<IQ4_NL_Decoder<IQ4_NLBlock>, 
    ${TILE_M}, ${TILE_N}, ${TILE_K}, 
    ${THREADS_M}, ${THREADS_N}, ${WORK_M}, ${WORK_N}, 
    ${PREFETCH}, ${TRANSPOSE}, ${VECTORIZE},
    ${ATOM_TYPE}, ${ATOM_M}, ${ATOM_N}, ${ATOM_K}>  // NEW
    <<<gridDim, blockDim, 0, stream>>>(A, C, m, n, k, decoder);
```

**Updated registration calls**:
```cpp
CudaGemmKernelRegistry::instance().register_kernel(
    tile_m, tile_n, tile_k, threads_m, threads_n, work_m, work_n,
    prefetch, transpose, vectorize,
    atom_type, atom_m, atom_n, atom_k, &launcher);  // NEW
```

## Config Space Impact

### Before
- **10 parameters**: tile_m/n/k, threads_m/n, work_m/n, prefetch, transpose, vectorize
- **~37,000 valid configs** (after filtering)
- **4 atom params hardcoded**: `atom_type=0, atom_layout_m/n/k=1`

### After
- **14 parameters**: All 10 previous + 4 atom params
- **~335,000 valid configs** (9× expansion, before filtering)
- **Atom diversity**: 2 types × 3 layouts × 3 layouts × 1 = 18 combinations

### Test Strategy

**Phase 1: Qwen 0.5B Only** (THIS PHASE)
- **4 shapes**: 1×896×896, 1×4864×896, 1×896×4864, 128×896×896
- **Runtime**: 4-8 hours (vs 20-30 hours full suite)
- **Goal**: Validate atom params help before expensive full sweep

**Phase 2: Full Sweep** (if Phase 1 shows >5% improvement)
- **53 shapes**: All Qwen models (0.5B-72B)
- **Runtime**: 20-30 hours
- **Output**: Complete ML training dataset

## Test Script

**Location**: `test_qwen05b_atom_sweep.sh`

**Usage**:
```bash
# Run Qwen 0.5B test
./test_qwen05b_atom_sweep.sh

# Output: cuda_atom_sweep_results/qwen05b_atom_sweep_YYYYMMDD_HHMMSS.csv
```

**What it does**:
1. Regenerates kernel variants with atom parameters
2. Rebuilds `profile_cuda_config` executable
3. Creates limited benchmark shapes file (4 shapes)
4. Runs profiler on each shape across all configs
5. Analyzes results:
   - Top 10 overall configs
   - Atom type distribution in top 100
   - Atom layout distribution in top 100

**Output Format**:
```csv
batch,m,n,k,tile_m,tile_n,tile_k,threads_m,threads_n,work_m,work_n,prefetch,transpose,vectorize,atom_type,atom_m,atom_n,atom_k,gflops,ms,label
1,1,896,896,16,16,32,8,8,2,2,0,1,1,0,1,1,1,33.50,0.063,single_token_qkv
```

## Expected Outcomes

### Success Criteria (>5% improvement on any shape)
- **Finding**: Atom layout 2×2 or 4×4 improves large tiles (128×896×896)
- **Finding**: Atom type 1 (K=8) helps specific shapes with K%16≠0
- **Action**: Keep atom params, run full benchmark suite
- **Benefit**: Better tensor core utilization, future-proof for other quants

### Neutral Results (2-5% mixed improvements)
- **Finding**: Some shapes benefit, others neutral
- **Action**: Keep atoms but limit values (e.g., only 1×1 and 2×2 layouts)
- **Trade-off**: Smaller config explosion, still get flexibility

### Failure Criteria (<2% improvement everywhere)
- **Finding**: Atom params don't significantly affect performance
- **Action**: Revert atom expansion, focus on `__launch_bounds__` only
- **Benefit**: Keep config space manageable, avoid 9× explosion

## Performance Notes

### Atom Type Difference
- **Type 0 (SM80_16x8x16)**: K dimension = 16 elements
  - Better for tiles with K%16=0
  - More FLOPs per MMA instruction
  
- **Type 1 (SM80_16x8x8)**: K dimension = 8 elements
  - Better for tiles with K%8=0 but K%16≠0
  - More flexible for odd K dimensions

### Atom Layout Impact
- **1×1 layout**: Minimal tensor core usage (16×8 per warp)
  - Low register pressure
  - Good for small tiles (16×16)
  
- **2×2 layout**: Medium tensor core usage (32×16 per warp)
  - Balanced register/utilization
  - Good for medium tiles (32×32)
  
- **4×4 layout**: Maximum tensor core usage (64×32 per warp)
  - High register pressure
  - Best for large tiles (64×64) if registers allow

### Why This Matters for Future Quants

**IQ4_NL**: block_size=32 → K dimension divisible by 32
- Atom type 0 (K=16) perfectly divides
- Current hardcoded params work well

**Q6_K**: block_size=256 → K dimension divisible by 256
- Atom type 0 (K=16) perfectly divides
- BUT: Large block size → different atom_layout_k may help

**Q8_0**: block_size=32 → Same as IQ4_NL
- Similar requirements

**MXFP4** (future): Variable block size (16-64)
- Atom type 1 (K=8) may be better for block_size=16
- Needs flexible atom params

## Files Changed

### Code
- `src/v2/kernels/cuda/CudaGemmKernelTemplate.h` - Added `__launch_bounds__`
- `src/v2/kernels/cuda/CudaGemmVariantsBaseline.cu` - Added atom template params + validation
- `src/v2/kernels/cuda/generate_cuda_gemm_variants.py` - Expanded config space
- `src/v2/kernels/cuda/CudaGemmKernelRegistry.h` - Updated key and registration

### Scripts
- `test_qwen05b_atom_sweep.sh` - Qwen 0.5B limited benchmark (NEW)

## Next Steps

1. **Run Qwen 0.5B test**: `./test_qwen05b_atom_sweep.sh`
2. **Analyze results**: Check atom type/layout distribution in top configs
3. **Decision point**:
   - If helpful (>5%): Run full sweep, retrain ML model
   - If neutral (2-5%): Limit atom values, run targeted sweep
   - If unhelpful (<2%): Revert atom expansion, focus on `__launch_bounds__`

## References

- **Profiling Session**: `ncu_profile_best_config.txt`
- **Optimization Plan**: `CUDA_KERNEL_OPTIMIZATION_PLAN.md`
- **Config Space Audit**: Config space analysis showing threads_m/n already covered
- **User Insight**: "other quant types will not have the same block size necessarily"

---

**Bottom Line**: We've made the CUDA kernel config space future-proof for multiple quantization types by allowing atom-level tensor core parameters to vary. The test-first approach (Qwen 0.5B only) validates this expansion before committing to an expensive full benchmark sweep.
