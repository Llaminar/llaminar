# Qwen3-MoE 235B A22B Coverage Extension

**Date**: November 2, 2025  
**Author**: David Sanftenberg  
**Component**: CUDA GEMM ML Heuristic Training - Qwen3-MoE 235B Model Coverage

## Summary

Extended the CUDA GEMM heuristic validation benchmark suite to include comprehensive coverage of **Qwen3-MoE 235B A22B**, a 128-expert Mixture-of-Experts model with Multi-Query Attention (MQA). This brings total test coverage to **34 test cases** spanning **0.5B to 671B parameters**.

## Qwen3-MoE 235B A22B Architecture

### Model Specifications
- **Total Parameters**: 235B
- **Active Parameters**: 22B (A22B - "Activated 22B")
- **Architecture**: Mixture of Experts (MoE)
- **Number of Experts**: 128
- **d_model**: 4,096
- **d_ff (per expert)**: 1,536
- **vocab_size**: 151,936

### Unique Architectural Features

1. **Doubled Q Projection**:
   - Expands d_model 4096 → 8192
   - Increases query capacity
   - Shape: `[1, 8192, 4096]`

2. **Multi-Query Attention (MQA)**:
   - KV projection: 4096 → 512 (8× compression)
   - Reduces KV cache memory by 8×
   - Shape: `[1, 512, 4096]`

3. **128-Expert MoE FFN**:
   - **Expert Routing**: `[1, 128, 4096]` (gate_inp)
     - Determines which experts to activate
     - Very small operation (N=128)
   - **Per-Expert Gate**: `[1, 1536, 4096]`
   - **Per-Expert Up**: `[1, 1536, 4096]`
   - **Per-Expert Down**: `[1, 4096, 1536]`

4. **Attention Output**:
   - Contracts doubled dimension back to d_model
   - Shape: `[1, 4096, 8192]`

## Test Cases Added (9 Total)

### Single-Token Attention Operations

1. **Q Projection** - `[1, 8192, 4096]`
   - Doubled output dimension (4096 → 8192)
   - Memory-bound (small M=1)
   - Wide output (N=8192)

2. **KV Projection (MQA)** - `[1, 512, 4096]`
   - Extreme compression (4096 → 512, 8× reduction)
   - Very narrow output (N=512)
   - Critical for KV cache efficiency

3. **Attention Output** - `[1, 4096, 8192]`
   - Wide input (K=8192)
   - Contracts back to d_model

### MoE FFN Operations

4. **Gate Input (Expert Routing)** - `[1, 128, 4096]`
   - Very small output (N=128 experts)
   - Determines expert activation
   - Extremely narrow output

5. **Expert Gate** - `[1, 1536, 4096]`
   - Per-expert operation (×128 total)
   - Moderate output dimension

6. **Expert Up** - `[1, 1536, 4096]`
   - Per-expert operation (×128 total)
   - Identical dimensions to expert gate

7. **Expert Down** - `[1, 4096, 1536]`
   - Per-expert operation (×128 total)
   - Transpose of expert gate/up

### Batch Prefill Operations

8. **Batch-128 Q Projection** - `[128, 8192, 4096]`
   - Large batch prefill
   - Compute-bound (M=128)
   - Wide output (N=8192)

9. **Batch-128 Expert Gate** - `[128, 1536, 4096]`
   - Large batch prefill
   - Tests MoE scaling with batch size

## Coverage Statistics

### Overall Benchmark Suite
- **Total Test Cases**: 34
- **Model Range**: 0.5B → 671B parameters
- **Total Data Points**: ~132,192 rows (34 tests × 3,888 configs)
- **CSV Size**: ~40-50 MB (estimated)

### Model Breakdown
- **Qwen 0.5B-72B**: 17 tests
- **DeepSeek V3 671B**: 8 tests
- **Qwen3-MoE 235B**: 9 tests

### Unique Matrix Shapes (Qwen3-MoE)
```
[1, 8192, 4096]    - Q projection (doubled dimension)
[1, 512, 4096]     - KV projection (MQA, 8× compressed)
[1, 4096, 8192]    - Attention output
[1, 128, 4096]     - MoE gate input (expert routing)
[1, 1536, 4096]    - MoE expert gate/up
[1, 4096, 1536]    - MoE expert down
[128, 8192, 4096]  - Batch Q projection
[128, 1536, 4096]  - Batch expert gate
```

## Implementation Details

### File Modified
- **File**: `tests/v2/performance/Perf__CudaGemmHeuristicValidation.cpp`
- **Size Before**: 1,659 lines
- **Size After**: 1,941 lines (+282 lines)
- **Tests Before**: 25
- **Tests After**: 34 (+9 tests)

### Test Structure
Each test case:
1. Benchmarks all 3,888 kernel configurations
2. Measures GFLOPS and execution time
3. Ranks configurations by performance
4. Compares ML heuristic ranking vs empirical ranking
5. Computes rank correlation (target >0.3)
6. Exports to CSV for ML training

### CSV Output Format
```csv
test_name,m,n,k,tile_m,tile_n,tile_k,atom_m,atom_n,atom_k,prefetch_a,prefetch_b,gflops,time_ms
Qwen3MoE_235B_SingleToken_Q_Projection,1,8192,4096,64,64,16,8,8,16,0,1,245.3,0.274
Qwen3MoE_235B_SingleToken_KV_Projection,1,512,4096,32,32,16,4,8,16,1,0,189.7,0.022
...
```

## Performance Expectations

### Single-Token Operations (M=1)
- **Q Projection** `[1, 8192, 4096]`:
  - Expected: ~200-300 GFLOPS
  - Memory-bound due to small M
  - Wide output increases bandwidth

- **KV Projection (MQA)** `[1, 512, 4096]`:
  - Expected: ~150-250 GFLOPS
  - Extremely narrow output (N=512)
  - Critical for memory efficiency

- **MoE Gate Input** `[1, 128, 4096]`:
  - Expected: ~100-200 GFLOPS
  - Very small operation
  - Routing decision overhead

- **MoE Expert Operations** `[1, 1536, 4096]`:
  - Expected: ~180-280 GFLOPS
  - Moderate size, memory-bound
  - Executed for top-k experts only

### Batch Prefill (M=128)
- **Batch Q Projection** `[128, 8192, 4096]`:
  - Expected: ~3,000-5,000 GFLOPS
  - Compute-bound with large M
  - Better tile utilization

- **Batch Expert Gate** `[128, 1536, 4096]`:
  - Expected: ~2,500-4,000 GFLOPS
  - Tests MoE batch scaling

## ML Model Impact

### Training Data Enhancement
- **Aspect Ratio Coverage**: N from 128 to 49,152 (383× range!)
  - Narrowest: MoE gate input (N=128)
  - Widest: Qwen 72B FFN (N=49,152)
  - Qwen3-MoE adds extreme narrow case (N=512 KV)

- **MoE Patterns**: First comprehensive MoE coverage
  - Expert routing (N=128)
  - Per-expert operations (N=1536)
  - Batch MoE scaling

- **MQA Patterns**: 8× KV compression
  - Different optimization than standard projections
  - Memory vs compute tradeoff

### Expected ML Model Improvements
- Better generalization for narrow matrices (N<1000)
- MoE-specific heuristics (expert routing, per-expert ops)
- Improved batch scaling predictions
- Better handling of extreme aspect ratios

## Comparison with Other Models

| Model | Size | Architecture | Test Count | Unique Shapes |
|-------|------|--------------|------------|---------------|
| Qwen 0.5B-72B | 0.5B-72B | Dense Transformer | 17 | 11 |
| DeepSeek V3 | 671B | MoE + LoRA + MQA | 8 | 8 |
| **Qwen3-MoE** | **235B (A22B)** | **128-expert MoE + MQA** | **9** | **8** |

### Key Differences
- **Qwen**: Standard transformer, full-rank projections
- **DeepSeek V3**: LoRA-style Q (2-stage), MQA KV (576-dim), MoE FFN
- **Qwen3-MoE**: Doubled Q (8192-dim), extreme MQA (512-dim), 128-expert MoE

## Workflow Integration

### Step 1: Run Benchmark (30-60 min)
```bash
cd /workspaces/llaminar
./build_v2_release/performance/v2_perf_cuda_heuristic_validation

# Output: cuda_gemm_benchmark_data.csv (~132,192 rows)
```

### Step 2: Train ML Model (30 sec)
```bash
cmake --build build_v2_release --target train_cuda_heuristic

# Output: src/v2/kernels/cuda/cuda_heuristic_weights.h
```

### Step 3: Validate (1-2 min)
```bash
./build_v2_release/tests/v2/v2_test_gemm_autotuner_ml

# Expected:
# - R² ≈ 0.9997 (near-perfect GFLOPS prediction)
# - Top-10 accuracy ~70-80%
# - Rank correlation >0.80
```

### Step 4: Rebuild (2-3 min)
```bash
cmake --build build_v2_release --target cuda_backend
```

## Key Design Decisions

### Why 9 Test Cases?
1. **3 Attention Operations**: Q, KV, Output (unique dimensions)
2. **4 MoE Operations**: Gate input, expert gate, expert up, expert down
3. **2 Batch Operations**: Q projection, expert gate (scaling validation)

### Why These Shapes?
- **Completeness**: All unique GEMM operations in Qwen3-MoE architecture
- **Aspect Ratio Coverage**: Extreme cases (N=128 to N=8192)
- **MoE Patterns**: First comprehensive MoE operation coverage
- **MQA Validation**: 8× compression vs standard projections

### Why Not More?
- Avoids redundant shapes (expert gate/up identical, only 1 batch test each)
- Focuses on architecturally unique operations
- Balances coverage vs runtime (34 tests × 3,888 configs = ~132K runs)

## Next Steps

1. **Run Complete Benchmark** (~60 min with 34 tests)
   - Generates ~132,192 CSV rows
   - ~40-50 MB CSV file

2. **Train ML Model** (~30 sec)
   - Expected R² ≈ 0.9997
   - Should improve narrow-matrix predictions

3. **Validate Improvements**
   - Check top-10 accuracy for new shapes
   - Verify MoE-specific heuristics
   - Test MQA operation performance

4. **Test Actual Qwen3-MoE Model** (when available)
   - End-to-end inference benchmark
   - Measure speedup vs manual heuristic
   - Validate expert routing efficiency

## Files Changed

- `tests/v2/performance/Perf__CudaGemmHeuristicValidation.cpp`
  - Added 9 Qwen3-MoE test cases
  - Updated file header with architecture documentation
  - Total: 1,941 lines (was 1,659)

## Related Documentation

- `CUDA_ML_HEURISTIC_SUMMARY.md` - Quick reference guide
- `docs/ML_HEURISTIC_TRAINING.md` - Comprehensive workflow guide
- `changelog/2025-11-02-ml-heuristic-documentation-and-extension.md` - Initial 32B/72B extension
- `changelog/2025-11-02-deepseek-v3-671b-coverage.md` - DeepSeek V3 671B extension

## Conclusion

This extension completes comprehensive coverage of modern LLM architectures:
- **Dense Transformers**: Qwen 0.5B-72B
- **MoE with LoRA/MQA**: DeepSeek V3 671B
- **128-Expert MoE with MQA**: Qwen3-MoE 235B

Total benchmark suite: **34 test cases**, **0.5B-671B parameters**, **~132K data points**, covering all major architectural patterns in production LLMs as of November 2025.
