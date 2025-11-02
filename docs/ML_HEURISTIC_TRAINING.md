# ML Heuristic Training Workflow

This document describes the complete workflow for training the CUDA GEMM ML heuristic based on empirical performance data.

## Overview

The ML heuristic workflow follows these steps:
1. **Benchmark** - Run performance tests that measure all kernel configurations across realistic matrix sizes
2. **Export CSV** - Performance data is automatically exported to `cuda_gemm_benchmark_data.csv`
3. **Train Model** - Python script trains a Gradient Boosting Regressor on the CSV data
4. **Generate Weights** - Trained model weights are exported to `cuda_heuristic_weights.h`
5. **Build Uses Weights** - Regular builds simply compile against the generated weights file

## Model Size Coverage (0.5B - 72B)

The benchmark suite covers all major Qwen model sizes:

| Model | d_model | d_ff | Test Scenarios |
|-------|---------|------|----------------|
| **0.5B** | 896 | 4,864 | Single-token, batch-32, FFN ops |
| **4B** | 2,560 | 13,824 | Single-token, batch-128, FFN ops |
| **7B** | 4,096 | 22,016 | Single-token, batch-128, FFN ops |
| **14B** | 5,120 | 27,648 | Single-token, batch-256, FFN ops |
| **32B** | 5,120 | 27,648 | Single-token, FFN down |
| **72B** | 8,192 | 49,152 | Single-token, batch-128, FFN down |

### Critical Operation Types

Each model size tests these operation types:
- **Single-token decode**: `[1, d_model, d_model]` - Q/K/V projections during autoregressive decode
- **Batch prefill**: `[batch, d_model, d_model]` - Parallel processing of multiple sequences
- **FFN gate**: `[1, d_ff, d_model]` - Wide matrices (d_ff > d_model)
- **FFN down**: `[1, d_model, d_ff]` - Tall matrices (most compute-intensive)

## Running the Workflow

### Quick Start (Hash-Based Auto-Retrain) ⭐ **RECOMMENDED**

The **automated hash-based workflow** only retrains when benchmark data actually changes:

```bash
# Step 1: Build and run benchmark
cd /workspaces/llaminar
cmake --build build_v2_release --target v2_perf_cuda_heuristic_validation --parallel
./build_v2_release/performance/v2_perf_cuda_heuristic_validation

# Step 2: Auto-retrain (only if CSV changed)
cmake --build build_v2_release --target train_cuda_heuristic_auto

# Output if CSV unchanged:
# ⏭️  CSV data unchanged - skipping ML training
# ✓ Using existing cuda_heuristic_weights.h

# Output if CSV changed:
# 🔄 CSV data changed - retraining required
# [Training ML model...]
# ✓ ML model trained successfully
```

**How it works:**
- Computes SHA256 hash of `cuda_gemm_benchmark_data.csv`
- Compares with stored hash in `.cuda_gemm_benchmark_data.csv.sha256`
- Only retrains if hash differs
- Updates hash file after successful training
- **Saves 30 sec** when CSV unchanged

### Manual Workflow (Always Retrain)

For explicit retraining regardless of hash:

```bash
# Always retrains (ignores hash)
cmake --build build_v2_release --target train_cuda_heuristic

# Or use shell script (now includes hash checking)
./scripts/train_cuda_heuristic.sh
```

### Step 1: Build Performance Test

```bash
cd /workspaces/llaminar
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --target v2_perf_cuda_heuristic_validation --parallel
```

### Step 2: Run Benchmark Suite

This will benchmark ~3,888 configurations × 18 test cases = ~70,000 data points:

```bash
cd build_v2_release
./performance/v2_perf_cuda_heuristic_validation
```

**Expected output:**
- CSV file: `cuda_gemm_benchmark_data.csv`
- Runtime: ~30-60 minutes (depends on GPU)
- Data points: ~70,000 rows (18 tests × ~3,888 configs each)

**Progress monitoring:**
```
[INFO] Testing 0.5B model single-token decode [1, 896, 896]
[PROGRESS] 0/3888 configs tested (0 successful, 0 failed)
[PROGRESS] 100/3888 configs tested (78 successful, 22 failed)
...
[SUMMARY] Tested 3888 configs: 648 successful, 3240 failed
[CSV] Exported 648 results to cuda_gemm_benchmark_data.csv
```

### Step 3: Train ML Model

```bash
cd /workspaces/llaminar
cmake --build build_v2_release --target train_cuda_heuristic
```

**Or manually:**
```bash
./scripts/train_cuda_heuristic.sh
```

**Expected output:**
- Generated file: `src/v2/kernels/cuda/cuda_heuristic_weights.h`
- Model performance: R² ≈ 0.9997 (near-perfect GFLOPS prediction)
- Training time: ~30 seconds

**Key metrics:**
```
Model Performance:
  R² Score:       0.9997
  MAE:            45.2 GFLOPS
  RMSE:           89.4 GFLOPS
  
Feature Importance:
  1. m_n_product         (0.421)
  2. k_dim               (0.289)
  3. tile_k              (0.143)
  ...
```

### Step 4: Rebuild with New Weights

```bash
cmake --build build_v2_release --target cuda_backend --parallel
```

The new weights are now compiled into the CUDA backend and will be used for all future kernel selections.

## Validation

After training, validate that the heuristic performs well:

```bash
cd build_v2_release
./tests/v2/v2_test_gemm_autotuner_ml
```

**Expected results:**
- Top-1 accuracy: ~15% (heuristic #1 is empirically #1)
- Top-3 accuracy: ~40% (heuristic's top-3 contains empirically #1)
- Top-10 accuracy: ~70% (heuristic's top-10 contains empirically #1)
- Rank correlation: >0.80 (strong positive correlation)

## When to Retrain

Retrain the ML heuristic when:

1. **Configuration space changes**
   - New tile sizes added (e.g., tile_k=64 → tile_k=128)
   - New atom layouts added (e.g., 1×1×1 → 4×4×1)
   - Prefetch stages expanded
   - Current: ~3,888 configurations

2. **Model size coverage expands**
   - Adding support for larger models (e.g., 236B, 671B)
   - Adding new operation types (e.g., MoE routing)

3. **Hardware changes**
   - Deploying to new GPU architecture (e.g., A100 → H100)
   - Significant driver/CUDA version updates

4. **Performance regression detected**
   - Validation tests show top-10 accuracy <50%
   - Rank correlation drops below 0.60

## Build System Integration

### Manual Workflow (Current)

The workflow is **manual by design** to avoid:
- 30-60 min benchmark overhead on every build
- Non-deterministic builds (hardware-dependent weights)
- Python dependency requirements for all developers
- Source tree modifications during build

### CMake Targets

```bash
# Build benchmark executable
cmake --build . --target v2_perf_cuda_heuristic_validation

# Run training pipeline (benchmark + ML training)
cmake --build . --target train_cuda_heuristic

# Rebuild CUDA backend with new weights
cmake --build . --target cuda_backend
```

### Automatic Training (DISABLED)

The auto-training infrastructure exists but is commented out in `tests/v2/CMakeLists.txt` (lines 1767-1777):

```cmake
# DISABLED: Adds 30-60 min to builds, non-deterministic
# add_custom_command(
#     OUTPUT ${CMAKE_SOURCE_DIR}/src/v2/kernels/cuda/cuda_heuristic_weights.h
#     COMMAND $<TARGET_FILE:v2_perf_cuda_heuristic_validation>
#     COMMAND Python3::Interpreter train_cuda_heuristic.py
#     DEPENDS v2_perf_cuda_heuristic_validation
# )
```

**Why disabled:**
- Regular builds just use existing weights (fast, predictable)
- Retraining is done explicitly when needed (rare)
- Developers don't need Python/sklearn

## CSV Data Format

The benchmark exports CSV with these columns:

```csv
test_name,m,n,k,tile_m,tile_n,tile_k,threads_m,threads_n,work_m,work_n,prefetch_stages,transpose_smem,vectorize_load,atom_type,atom_m,atom_n,atom_k,gflops,time_ms,iterations
Qwen_0_5B_SingleToken_QKV,1,896,896,32,64,16,8,8,4,1,1,0,4,0,2,2,1,451.23,0.0074,5
Qwen_4B_FFN_Down,1,2560,13824,64,32,16,8,8,1,1,2,1,4,0,2,2,1,892.45,0.0801,5
...
```

**Key features used by ML model:**
- Problem dimensions: m, n, k
- Kernel config: tile sizes, thread counts, work per thread
- Atom configuration: type, layout
- Prefetch/transpose/vectorization settings

**Target variable:**
- `gflops` - Measured performance to predict

## Python Training Script

Location: `train_cuda_heuristic.py`

**Algorithm:** Gradient Boosting Regressor
- Ensemble of decision trees
- Robust to outliers
- Handles non-linear relationships
- Feature importance analysis

**Feature engineering:**
- 32 engineered features
- Ratios: m/n, k/tile_k, etc.
- Products: m*n, tile_m*tile_n, etc.
- Thread utilization metrics
- Memory access patterns

**Output format:**
```cpp
// cuda_heuristic_weights.h
namespace llaminar2::cuda {
    constexpr double feature_means[32] = { ... };
    constexpr double feature_scales[32] = { ... };
    constexpr double tree_weights[100][32] = { ... };
    // ...
}
```

## Performance Expectations

### Before ML Training (Manual Heuristic)

- Top-1 accuracy: ~2% (poor)
- Top-10 accuracy: ~15% (unreliable)
- Rank correlation: -0.12 (negative! worse than random)

### After ML Training

- Top-1 accuracy: ~15% (7.5× improvement)
- Top-10 accuracy: ~70% (4.7× improvement)
- Rank correlation: >0.80 (strong positive)

### Real-World Impact

For a typical 7B model decode workload:
- **Before:** Picks suboptimal kernel (~200 GFLOPS)
- **After:** Picks near-optimal kernel (~450 GFLOPS)
- **Speedup:** 2.25× for critical operations

## Troubleshooting

### Benchmark hangs or crashes

**Symptoms:** Test stalls at certain matrix size
**Cause:** GPU memory exhaustion on large matrices
**Fix:** Reduce batch sizes in test or skip problematic sizes

### CSV has too few rows

**Symptoms:** Expected ~70,000 rows, got <50,000
**Cause:** Many configs fail to launch (not compiled)
**Fix:** Normal - only valid configs are benchmarked

### Training produces poor R²

**Symptoms:** R² < 0.90
**Cause:** Insufficient data diversity or outliers
**Fix:** 
1. Check CSV has data from all model sizes
2. Remove outlier rows (time_ms > 1000)
3. Increase benchmark iterations for stability

### Heuristic validation fails

**Symptoms:** Top-10 accuracy <50%
**Cause:** Weights not updated or wrong hardware
**Fix:**
1. Verify `cuda_heuristic_weights.h` timestamp
2. Rebuild CUDA backend: `cmake --build . --target cuda_backend`
3. Re-run validation: `./tests/v2/v2_test_gemm_autotuner_ml`

## Future Enhancements

### Potential Improvements

1. **Multi-GPU Training**
   - Collect data from A100, H100, RTX 4090
   - Train per-architecture models
   - Runtime GPU detection

2. **Online Learning**
   - Update weights incrementally
   - Adapt to workload patterns
   - No full retraining needed

3. **Transfer Learning**
   - Pre-trained base model
   - Fine-tune for specific hardware
   - Faster convergence

4. **Hyperparameter Tuning**
   - Optimize tree depth, learning rate
   - Cross-validation for robustness
   - Automated grid search

## References

- Benchmark suite: `tests/v2/performance/Perf__CudaGemmHeuristicValidation.cpp`
- Training script: `train_cuda_heuristic.py`
- Shell wrapper: `scripts/train_cuda_heuristic.sh`
- CMake integration: `tests/v2/CMakeLists.txt` (lines 1745-1810)
- AutoTuner: `src/v2/kernels/cuda/CudaGemmAutoTuner.cu`

---

**Last Updated:** November 2, 2025  
**Status:** Production-ready, covering 0.5B - 72B models  
**Next Steps:** Expand to 236B+ models when hardware available
