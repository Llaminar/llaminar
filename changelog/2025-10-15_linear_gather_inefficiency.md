# Fine-Grained Linear Operator Instrumentation - October 15, 2025

## Discovery: Row-by-Row Gather Inefficiency

### Overview
Added fine-grained instrumentation to `MPILinearOperator` to expose detailed timing of internal operations. **Discovered critical inefficiency**: `gatherOutput()` performs **893 separate MPI_Allgatherv calls** per linear layer during prefill (one per token), creating massive overhead.

## Instrumentation Added

### 1. Weight Distribution Breakdown
- `distribute_weight_internal` (depth 3): Top-level distribution
- `weight_memcpy` (depth 4): Actual memcpy operation

### 2. Bias Distribution Breakdown  
- `distribute_bias_internal` (depth 3): Top-level distribution
- `bias_memcpy` (depth 4): Actual memcpy operation

### 3. Gather Output Breakdown (Critical!)
- `gather_output_internal` (depth 3): Top-level gather orchestration
- `gather_setup_metadata` (depth 4): Recv counts/offsets computation
- `allgatherv_per_row_loop` (depth 4): **Row-by-row MPI_Allgatherv loop**

### 4. Bias Addition Breakdown
- `add_bias_internal` (depth 3): Top-level bias addition
- `bias_omp_parallel_loop` (depth 4): OpenMP parallelized loop

## Performance Analysis

### Prefill Benchmark (893 tokens, Q8_0, 73 linear operations)

**Total Linear Operator Time**: 1594.88ms (73.1% of execution)

**Breakdown by Component**:
1. **Weight Distribution**: 124.07ms (7.8%)
   - Pure memcpy: 124.07ms
   - Setup overhead: negligible
   
2. **Gather Output**: 43.41ms (2.7%)
   - **Row-by-row loop**: 42.73ms (98.4% of gather time!)
   - Metadata setup: 0.01ms (negligible)
   
3. **Actual Matmul**: 28.33ms (1.8%)
   
4. **Bias Addition**: <1ms (included in other operations)

### Critical Inefficiency: Row-by-Row Gather

**Per-Linear-Layer Statistics**:
- Average gather time: 0.595ms per layer
- Worst case (lm_head): **6.568ms** for 893 rows
- Number of MPI_Allgatherv calls: **893 per layer** (1 per token)

**Calculation for lm_head**:
- Total gather time: 6.568ms
- Number of rows: 893
- **Per-row MPI overhead**: 6.568ms / 893 = **7.35 μs per MPI_Allgatherv**

**Total MPI Collective Overhead** (across 73 linear layers):
- Total gather time: 43.41ms
- Estimated rows gathered: ~65,000 (73 layers × 893 tokens average)
- **65,000 individual MPI_Allgatherv operations**

### Why This Is Catastrophic

**MPI Collective Overhead**:
- Each `MPI_Allgatherv` has ~5-10μs latency overhead (rank synchronization, message passing)
- With 893 rows: 893 × 7μs = **6.25ms pure overhead** per lm_head layer
- Actual data transfer is minimal (448 floats × 2 ranks = ~3.5KB per row)

**Comparison with Optimal Approach**:
- Current: 893 separate collectives = 6.57ms
- Optimal (single collective): ~0.1-0.2ms (estimated)
- **Speedup potential**: 33-65× faster gather!

### Root Cause Analysis

**Current Implementation** (`src/operators/MPILinearOperator.cpp`, lines 314-337):
```cpp
// Row-wise Allgatherv (cannot trivially collapse into single collective without packing
// because column blocks for each rank are strided in row-major layout).
for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx)
{
    const float *local_row = local_output->data() + seq_idx * local_output_size;
    float *global_row = global_output->data() + seq_idx * output_size;
    checkMPIError(MPI_Allgatherv(local_row, static_cast<int>(local_output_size), MPI_FLOAT,
                                 global_row, recv_counts.data(), recv_offsets.data(), MPI_FLOAT,
                                 getComm()),
                  "MPI_Allgatherv in gatherOutput");
}
```

**Problem**: 
- Output is row-major: `[seq_len, output_dim]`
- Each rank owns **column slice**: `[seq_len, local_output_dim]`
- Cannot gather full matrix with single `MPI_Allgatherv` because:
  - Rank 0's columns are at offsets: `global[i, 0:447]`
  - Rank 1's columns are at offsets: `global[i, 448:895]`
  - Strided access pattern requires either row-by-row gather OR temporary packing

**Comment in Code**:
> "Future optimization: pack local_output into an interleaved buffer (seq-major) with one Allgatherv using a custom MPI datatype for the column block."

## Optimization Strategies

### Strategy 1: Packed Buffer + Single Collective (Recommended)

**Approach**:
1. Pack local output `[seq_len, local_dim]` into contiguous buffer
2. Single `MPI_Allgatherv` to gather all packed data
3. Unpack into global output `[seq_len, global_dim]` with proper interleaving

**Pseudocode**:
```cpp
// Pack: Contiguous copy (already optimal layout)
float *packed_local = local_output->data(); // No copy needed!

// Single Allgatherv
std::vector<float> packed_global(seq_len * output_size);
MPI_Allgatherv(packed_local, seq_len * local_output_size, MPI_FLOAT,
               packed_global.data(), recv_counts_packed, recv_offsets_packed, MPI_FLOAT,
               getComm());

// Unpack: Interleave columns
#pragma omp parallel for
for (size_t row = 0; row < seq_len; ++row) {
    for (int rank = 0; rank < world_size; ++rank) {
        size_t src_offset = rank_offsets[rank] + row * rank_local_dims[rank];
        size_t dst_offset = row * output_size + rank_column_offsets[rank];
        memcpy(&global_output[dst_offset], &packed_global[src_offset], 
               rank_local_dims[rank] * sizeof(float));
    }
}
```

**Expected Speedup**:
- Remove 892 extra MPI collectives (keep only 1)
- MPI overhead: 6.5ms → 0.1ms (**65× faster**)
- Add unpacking cost: ~0.5ms (OpenMP parallelized memcpy)
- **Net savings**: 6.5ms → 0.6ms per lm_head layer
- **Total savings across 73 layers**: 43ms → 6ms (**7× faster gather**)

### Strategy 2: MPI Derived Datatype (Advanced)

**Approach**:
Use `MPI_Type_vector` or `MPI_Type_indexed` to describe strided column access pattern.

**Pros**:
- Single `MPI_Allgatherv` without packing/unpacking
- MPI implementation can optimize strided transfers

**Cons**:
- Complex datatype construction
- MPI implementation may not optimize well for this pattern
- Debugging difficulty

**Estimated Speedup**: 30-50× (less than Strategy 1 due to potential strided memory overhead)

### Strategy 3: Transpose Local Output (Not Recommended)

**Approach**:
Transpose local output to `[local_dim, seq_len]` before gather, then gather is contiguous.

**Cons**:
- Adds transpose cost: ~1-2ms per layer
- Memory bandwidth intensive
- Not compatible with downstream expectations
- Only beneficial if transpose is needed elsewhere

## Immediate Action Items

1. **Implement Strategy 1** (packed buffer approach):
   - Modify `gatherOutput()` to use single collective
   - Add unpacking step with OpenMP parallelization
   - Benchmark against current implementation
   - **Expected total savings**: ~35-40ms (2.3% of total time)

2. **Verify correctness**:
   - Run parity tests to ensure numerical equivalence
   - Check edge cases (single rank, uneven distribution)

3. **Profile unpacking overhead**:
   - Measure unpacking cost separately
   - Optimize with SIMD/AVX if needed

## Impact Assessment

### Current Bottleneck Ranking:
1. Weight distribution: 124ms (7.8%)
2. Gather output: **43ms (2.7%)** ← Target for optimization
3. Actual matmul: 28ms (1.8%)

### After Gather Optimization:
1. Weight distribution: 124ms (8.0%)
2. Actual matmul: 28ms (1.8%)
3. Gather output: **6ms (0.4%)** ← 7× improvement

**Net Effect**:
- Total linear time: 1595ms → 1558ms (2.3% faster)
- Overall throughput: 42.86 tok/s → 43.85 tok/s (+2.3%)

**Caveat**: This is a modest improvement because weight distribution (124ms) remains the dominant overhead. However, it's a **low-hanging fruit** that requires minimal code changes and validates the instrumentation framework.

## Next Steps After Gather Optimization

Once gather is optimized, the remaining bottleneck is **weight distribution** (124ms):

1. **Weight Caching**: 
   - Detect identical weight shapes across layers
   - Cache distributed weights
   - Expected savings: 100-110ms (eliminate redundant distribution)

2. **lm_head Investigation**:
   - Profile why lm_head is 1.5× slower than FFN layers
   - Check backend selection (COSMA vs OpenBLAS)
   - Expected savings: 150-200ms

3. **Combined Effect**:
   - Gather optimization: -37ms
   - Weight caching: -110ms
   - lm_head fix: -150ms
   - **Total savings**: ~300ms
   - **Projected throughput**: 42.86 → 62.5 tok/s (**46% faster**)

## Related Files

- `src/operators/MPILinearOperator.cpp`: Instrumented gather implementation
- `changelog/2025-10-15_granular_performance_instrumentation.md`: Initial instrumentation
- `changelog/2025-10-15_implementation_summary.md`: Overall optimization roadmap

## References

- MPI_Allgatherv documentation: https://www.mpich.org/static/docs/v3.3/www3/MPI_Allgatherv.html
- COSMA tensor distribution patterns for inspiration
- llama.cpp's approach to output aggregation (investigate for comparison)
