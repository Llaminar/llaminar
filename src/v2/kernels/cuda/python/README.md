# CUDA Kernel Heuristic Training Scripts

This directory contains Python scripts for training ML models that predict optimal CUDA GEMM kernel configurations.

## Scripts

### `train_cuda_heuristic.py`
Trains a Gradient Boosting model to predict CUDA GEMM performance from kernel configuration parameters.

**Generates**:
- `../generated/cuda_heuristic_weights.h` - ML model weights (C++ header)
- `../generated/cuda_heuristic_lookup.h` - Lookup table with best configs per shape
- `../generated/cuda_heuristic_model_weights.txt` - Model coefficients (text dump)
- `../generated/cuda_heuristic_validation.png` - Performance validation plots

**Usage**:
```bash
cd src/v2/kernels/cuda/python

# Train with default paths (reads from build_v2/, outputs to ../generated/)
python train_cuda_heuristic.py

# Custom input/output
python train_cuda_heuristic.py \
  --input /path/to/cuda_gemm_benchmark_data.csv \
  --output-dir /path/to/output
```

**Requirements**:
1. Run benchmarks first to generate training data:
   ```bash
   cd build_v2
   ctest -R "V2_Perf_CudaHeuristicValidation" -V
   # Generates: build_v2/cuda_gemm_benchmark_data.csv
   ```

2. Python dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

### `train_tensorcore_heuristic.py`
Trains a model for Tensor Core GEMM heuristic selection.

**Generates**:
- `../generated/tensorcore_heuristic_weights.h` - Tensor Core heuristic (C++ header)
- `../generated/tensorcore_heuristic_validation.png` - Validation plots

**Usage**:
```bash
cd src/v2/kernels/cuda/python

# Train with default paths
python train_tensorcore_heuristic.py

# Custom paths
python train_tensorcore_heuristic.py \
  --input /path/to/tensorcore_benchmark_data.csv \
  --output-dir /path/to/output
```

**Requirements**:
1. Run Tensor Core benchmarks:
   ```bash
   cd build_v2
   ctest -R "V2_Perf_TensorCoreHeuristicValidation" -V
   # Generates: build_v2/tensorcore_benchmark_data.csv
   ```

## Workflow

### 1. Run Benchmarks
```bash
cd /workspaces/llaminar/build_v2

# CUDA GEMM benchmarks (generates ~648 configs × multiple shapes)
ctest -R "V2_Perf_CudaHeuristicValidation" -V

# Tensor Core benchmarks
ctest -R "V2_Perf_TensorCoreHeuristicValidation" -V
```

### 2. Train Models
```bash
cd /workspaces/llaminar/src/v2/kernels/cuda/python

# Train both models
python train_cuda_heuristic.py
python train_tensorcore_heuristic.py
```

### 3. Rebuild with New Heuristics
```bash
cd /workspaces/llaminar
cmake --build build_v2 --target cuda_backend -j$(nproc)
```

### 4. Validate Performance
```bash
cd build_v2

# Re-run validation with new heuristics
ctest -R "V2_Perf_CudaHeuristicValidation" -V

# Check that top-1 predictions improved
```

## Output Files

All generated files go to `../generated/` (relative to this `python/` directory):

```
src/v2/kernels/cuda/
├── python/                              ← Training scripts
│   ├── train_cuda_heuristic.py
│   ├── train_tensorcore_heuristic.py
│   └── README.md                        ← This file
└── generated/                           ← Generated outputs
    ├── cuda_heuristic_weights.h         ← ML weights (C++)
    ├── cuda_heuristic_lookup.h          ← Config lookup table
    ├── cuda_heuristic_model_weights.txt ← Model dump
    ├── cuda_heuristic_validation.png    ← Plots
    └── tensorcore_heuristic_weights.h   ← Tensor Core heuristic
```

## Model Details

### CUDA GEMM Model

**Algorithm**: Gradient Boosting Regressor
- **Training samples**: ~7,776 benchmarks (648 configs × 12 test shapes)
- **Features**: 20+ engineered features
  - Tile dimensions (TILE_M, TILE_N, TILE_K)
  - Thread block config (threads_x, threads_y)
  - Work items (work_x, work_y)
  - Prefetch stages (0-2)
  - Shared memory transpose (bool)
  - Vectorization (1/2/4)
  - Problem size (M, N, K)
  - Derived features (tile utilization, efficiency, etc.)

**Performance**: R² ≈ 0.9999 on validation set

**Outputs**:
1. **Regression model** (`cuda_heuristic_weights.h`): Predicts GFLOPS for any config
2. **Lookup table** (`cuda_heuristic_lookup.h`): Best config per (M, N, K) shape

### Tensor Core Model

**Algorithm**: Simplified heuristic (Phase 3 design)
- **Key insight**: TILE_M should match M dimension for optimal performance
- **Secondary factors**: TILE_K=16 optimal for sm_80, tile efficiency

**Status**: Pending full ML implementation (currently uses rule-based heuristic)

## Debugging

### Common Issues

**Issue**: `FileNotFoundError: cuda_gemm_benchmark_data.csv`
```bash
# Solution: Run benchmarks first
cd build_v2
ctest -R "V2_Perf_CudaHeuristicValidation" -V
```

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
```bash
# Solution: Install Python dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Issue**: Generated headers not found during build
```bash
# Solution: Check paths - outputs should go to ../generated/
ls -la ../generated/*.h

# If missing, re-run training scripts
python train_cuda_heuristic.py
```

**Issue**: Poor model performance (low R² or bad predictions)
```bash
# Solution: Check feature engineering in script
# Look for:
# - Missing features (engineered in Python but not in C++)
# - Feature scaling mismatches
# - NaN/Inf values in training data

# Validate with manual config:
grep "tile_64x64x32_threads_16x16" ../generated/cuda_heuristic_lookup.h
```

## Development

### Adding New Features

1. **Engineer feature in Python** (`train_cuda_heuristic.py`):
```python
def engineer_features(df):
    # Add new feature
    df['my_new_feature'] = df['tile_m'] * df['tile_n'] / df['k']
    return df
```

2. **Export to C++ header** (`export_cpp_heuristic()`):
```python
f.write("    double my_new_feature = tile_m * tile_n / k;\n")
```

3. **Update C++ code** (`CudaGemmAutoTuner.cu`):
```cpp
double my_new_feature = tile_m * tile_n / k;
// Use in prediction...
```

### Extending to New Operations

To add heuristics for new operations (e.g., Flash Attention, fused kernels):

1. Create benchmark test in `tests/v2/performance/`
2. Export CSV with same format (config params + GFLOPS)
3. Copy `train_cuda_heuristic.py` → `train_myop_heuristic.py`
4. Adjust features for your operation
5. Train and export to `../generated/myop_heuristic_weights.h`

## References

- **Feature importance analysis**: See `../generated/cuda_heuristic_validation.png`
- **Model validation**: Check correlation plots in validation output
- **C++ integration**: `../CudaGemmAutoTuner.cu` shows how generated headers are used

## Maintenance

**When to retrain**:
- New GPU architecture (e.g., Hopper sm_90)
- New tile sizes added to search space
- Compiler optimizations change performance characteristics
- New prefetch/pipeline strategies implemented

**Workflow**:
1. Re-run benchmarks on new hardware/config
2. Append new data to CSV (or replace entirely)
3. Retrain models
4. Validate predictions match new benchmarks
5. Commit updated `../generated/*.h` files
