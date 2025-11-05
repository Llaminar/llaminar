# CUDA GEMM AutoTuner Retraining Pipeline

**Date**: November 3, 2025  
**Status**: Production Ready

## Overview

Formal CMake-based pipeline for retraining the CUDA GEMM neural network heuristic. Profiles kernel configurations on Qwen 0.5B, 4B, and 7B models, trains a neural network, validates on unseen shapes, and deploys the new model.

## Quick Start

```bash
# Run complete pipeline (10-20 minutes)
cmake --build build_v2_release --target cuda_gemm_retrain_pipeline

# Or run individual stages
cmake --build build_v2_release --target cuda_gemm_profile    # Stage 1: Profiling
cmake --build build_v2_release --target cuda_gemm_train      # Stage 2: Training
cmake --build build_v2_release --target cuda_gemm_validate   # Stage 3: Validation
cmake --build build_v2_release --target cuda_gemm_deploy     # Stage 4: Deployment

# Check status
cmake --build build_v2_release --target cuda_gemm_pipeline_status

# Clean artifacts
cmake --build build_v2_release --target cuda_gemm_clean_pipeline
```

## Pipeline Stages

### Stage 1: Profiling (`cuda_gemm_profile`)

**Purpose**: Collect performance data for kernel configurations

**Actions**:
1. Profiles kernel configurations on Qwen 0.5B, 4B, 7B models
2. Selects top/bottom 10 configs per shape for diversity
3. Measures GFLOPS, occupancy, registers, shared memory, warp efficiency
4. Generates `cuda_gemm_profiling_data.csv`

**Output**:
- `cuda_gemm_profiling_data.csv` - Profiling data (1,404+ samples expected)

**Duration**: ~5-10 minutes

**Configuration**:
```cmake
set(CUDA_GEMM_MODEL_SIZES "0.5B" "4B" "7B")  # Model sizes to profile
set(CUDA_GEMM_TOP_N 10)                       # Top performing configs
set(CUDA_GEMM_BOTTOM_N 10)                    # Bottom performing configs
```

### Stage 2: Training (`cuda_gemm_train`)

**Purpose**: Train neural network heuristic

**Actions**:
1. Backs up existing model (timestamped `.backup` files)
2. Trains neural network on profiling data
3. Generates ONNX model and StandardScaler parameters
4. Saves training metrics (R², loss, etc.)
5. Creates `.new` files (not yet deployed)

**Inputs**:
- `cuda_gemm_profiling_data.csv` - Profiling data from Stage 1

**Outputs**:
- `cuda_heuristic_nn.onnx.new` - Trained ONNX model (not deployed)
- `cuda_heuristic_scaler.txt.new` - Scaler parameters (not deployed)
- `training_metrics.json` - Training metrics (R², loss, etc.)

**Duration**: ~3-5 minutes

**Configuration**:
```python
--epochs 10
--batch-size 32
--learning-rate 0.0001
--hidden-dims 256 128 64
--validation-split 0.2
```

### Stage 3: Validation (`cuda_gemm_validate`)

**Purpose**: Validate neural network on unseen shapes

**Actions**:
1. Temporarily installs `.new` files as `.validation` for testing
2. Runs validation on unseen problem sizes
3. Measures top-30 hit rate (must be ≥95%)
4. Generates validation results JSON

**Inputs**:
- `cuda_heuristic_nn.onnx.new` - Trained model from Stage 2
- `cuda_heuristic_scaler.txt.new` - Scaler from Stage 2

**Outputs**:
- `validation_results.json` - Validation metrics (hit rate, etc.)

**Duration**: ~2-3 minutes

**Success Criteria**:
- Top-30 hit rate ≥95%
- If validation fails, pipeline aborts (model not deployed)

### Stage 4: Deployment (`cuda_gemm_deploy`)

**Purpose**: Deploy validated model to production

**Actions**:
1. Verifies validation passed (≥95% hit rate)
2. Replaces production model files
3. Cleans up temporary `.new` and `.validation` files
4. Rebuilds CUDA backend to use new model

**Inputs**:
- `cuda_heuristic_nn.onnx.new` - Validated model from Stage 3
- `cuda_heuristic_scaler.txt.new` - Validated scaler from Stage 3
- `validation_results.json` - Validation proof

**Outputs**:
- `src/v2/kernels/cuda/cuda_heuristic_nn.onnx` - Production model (updated)
- `src/v2/kernels/cuda/cuda_heuristic_scaler.txt` - Production scaler (updated)
- Rebuilt `cuda_backend` library

**Duration**: ~1-2 minutes (depends on backend rebuild time)

## File Structure

```
/workspaces/llaminar/
├── cmake/
│   └── CudaGemmPipeline.cmake          # Pipeline definition
├── python/
│   ├── collect_profiling_data.py       # Profiling script
│   ├── train_cuda_neural_network.py    # Training script
│   ├── validate_heuristic.py           # Validation script
│   └── export_scaler_for_cpp.py        # Scaler export utility
├── src/v2/kernels/cuda/
│   ├── cuda_heuristic_nn.onnx          # Production model
│   ├── cuda_heuristic_scaler.txt       # Production scaler
│   ├── cuda_heuristic_nn.onnx.backup.* # Backup models (timestamped)
│   └── cuda_heuristic_scaler.txt.backup.* # Backup scalers
└── (workspace root)
    ├── cuda_gemm_profiling_data.csv    # Profiling data
    ├── training_metrics.json           # Training metrics
    └── validation_results.json         # Validation results
```

## Configuration

### Model Sizes

Edit `cmake/CudaGemmPipeline.cmake` to change profiled model sizes:

```cmake
set(CUDA_GEMM_MODEL_SIZES "0.5B" "4B" "7B")  # Add/remove sizes
```

### Top/Bottom Config Count

```cmake
set(CUDA_GEMM_TOP_N 10)     # Top performing configs per shape
set(CUDA_GEMM_BOTTOM_N 10)  # Bottom performing configs per shape
```

### Training Hyperparameters

Edit `python/train_cuda_neural_network.py` or override in CMake:

```python
--epochs 10              # Training epochs
--batch-size 32          # Mini-batch size
--learning-rate 0.0001   # AdamW learning rate
--hidden-dims 256 128 64 # Neural network architecture
--validation-split 0.2   # Validation set size (20%)
```

### Validation Threshold

Edit `cmake/CudaGemmPipeline.cmake` to change acceptance threshold:

```cmake
# Default: 95% top-30 hit rate required
if hit_rate < 0.95:
    print('ERROR: Hit rate below 95% threshold!')
    sys.exit(1)
```

## Environment Variables

### Optional Overrides

```bash
# Override model/scaler locations
export LLAMINAR_CUDA_HEURISTIC_MODEL=/path/to/model.onnx
export LLAMINAR_CUDA_HEURISTIC_SCALER=/path/to/scaler.txt

# Enable neural network heuristic (already enabled by pipeline)
export LLAMINAR_USE_NN_HEURISTIC=1
```

## Backup and Recovery

### Automatic Backups

Pipeline automatically creates timestamped backups before training:

```
cuda_heuristic_nn.onnx.backup.20251103_163545
cuda_heuristic_scaler.txt.backup.20251103_163545
```

### Manual Restore

```bash
# Find latest backup
ls -lt src/v2/kernels/cuda/cuda_heuristic_nn.onnx.backup.* | head -1

# Restore from backup
cp src/v2/kernels/cuda/cuda_heuristic_nn.onnx.backup.YYYYMMDD_HHMMSS \
   src/v2/kernels/cuda/cuda_heuristic_nn.onnx
cp src/v2/kernels/cuda/cuda_heuristic_scaler.txt.backup.YYYYMMDD_HHMMSS \
   src/v2/kernels/cuda/cuda_heuristic_scaler.txt

# Rebuild backend
cmake --build build_v2_release --target cuda_backend
```

## Troubleshooting

### Pipeline Fails at Profiling

**Symptom**: `ERROR: Profiling data not found!`

**Causes**:
- CUDA device not available
- Models not downloaded (0.5B, 4B, 7B)
- Insufficient GPU memory

**Solution**:
```bash
# Check CUDA availability
nvidia-smi

# Download models
./fetch_models.sh

# Check GPU memory (need ~8GB for 7B model)
```

### Pipeline Fails at Training

**Symptom**: `ERROR: ONNX model not generated!`

**Causes**:
- Insufficient profiling data (<100 samples)
- Python dependencies missing
- ONNX Runtime not installed

**Solution**:
```bash
# Check profiling data
wc -l cuda_gemm_profiling_data.csv  # Should be >100

# Install Python dependencies
pip install torch onnx onnxruntime scikit-learn pandas numpy

# Verify ONNX Runtime
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
```

### Pipeline Fails at Validation

**Symptom**: `ERROR: Hit rate below 95% threshold!`

**Causes**:
- Insufficient training data diversity
- Model architecture too simple
- Training didn't converge (check training_metrics.json)

**Solution**:
```bash
# Check training metrics
cat training_metrics.json

# Increase training epochs
# Edit cmake/CudaGemmPipeline.cmake: --epochs 20

# Add more model sizes for diversity
# Edit cmake/CudaGemmPipeline.cmake:
# set(CUDA_GEMM_MODEL_SIZES "0.5B" "1.5B" "4B" "7B" "14B")

# Increase top/bottom config count
# Edit cmake/CudaGemmPipeline.cmake:
# set(CUDA_GEMM_TOP_N 15)
# set(CUDA_GEMM_BOTTOM_N 15)
```

### CUDA Backend Fails to Rebuild

**Symptom**: Compilation errors after deployment

**Causes**:
- Model file corruption
- ONNX Runtime headers not found
- CMake cache stale

**Solution**:
```bash
# Restore from backup (see Backup and Recovery section)

# Clean CMake cache
rm -rf build_v2_release
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release \
  -DHAVE_CUDA=ON -DUSE_ONNX_HEURISTIC=ON
cmake --build build_v2_release --target cuda_backend --parallel
```

## Performance Expectations

### Profiling Stage
- **Time**: 5-10 minutes
- **GPU**: 80-95% utilization
- **Samples**: ~1,400 configurations (with 0.5B/4B/7B)

### Training Stage
- **Time**: 3-5 minutes
- **Epochs**: 10 (default)
- **R² Score**: >0.995 (expected)
- **Validation Loss**: <0.1 (expected)

### Validation Stage
- **Time**: 2-3 minutes
- **Hit Rate**: ≥95% (required)
- **Test Cases**: ~26 unseen shapes

### Deployment Stage
- **Time**: 1-2 minutes
- **Backend Rebuild**: Incremental (only recompiles changed files)

### Total Pipeline
- **End-to-End**: 10-20 minutes
- **Success Rate**: >95% (with proper setup)

## Best Practices

### 1. Run During Low-Activity Periods
- Pipeline uses 100% GPU for 10-20 minutes
- Avoid running during inference workloads

### 2. Verify Prerequisites
```bash
# Before running pipeline
cmake --build build_v2_release --target cuda_gemm_pipeline_status
nvidia-smi  # Check GPU availability
./fetch_models.sh  # Ensure models downloaded
```

### 3. Monitor Progress
```bash
# Watch profiling CSV grow
watch -n1 wc -l cuda_gemm_profiling_data.csv

# Monitor training metrics
tail -f training_metrics.json

# Check validation progress
tail -f validation_results.json
```

### 4. Archive Successful Runs
```bash
# After successful pipeline
mkdir -p model_archives/$(date +%Y%m%d)
cp cuda_gemm_profiling_data.csv model_archives/$(date +%Y%m%d)/
cp training_metrics.json model_archives/$(date +%Y%m%d)/
cp validation_results.json model_archives/$(date +%Y%m%d)/
cp src/v2/kernels/cuda/cuda_heuristic_nn.onnx model_archives/$(date +%Y%m%d)/
```

### 5. Incremental Retraining
If you want to add more data without full reprofiling:

```bash
# Profile additional model size
python3 python/collect_profiling_data.py --models 14B --append

# Train on combined data
cmake --build build_v2_release --target cuda_gemm_train
cmake --build build_v2_release --target cuda_gemm_validate
cmake --build build_v2_release --target cuda_gemm_deploy
```

## Integration with CI/CD

### Automated Nightly Retraining

```yaml
# .github/workflows/retrain_cuda_gemm.yml
name: Retrain CUDA GEMM Heuristic
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sundays at 2 AM
  workflow_dispatch:      # Manual trigger

jobs:
  retrain:
    runs-on: self-hosted  # Requires GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Run Pipeline
        run: |
          cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
          cmake --build build_v2_release --target cuda_gemm_retrain_pipeline
      - name: Commit New Model
        run: |
          git add src/v2/kernels/cuda/cuda_heuristic_nn.onnx
          git add src/v2/kernels/cuda/cuda_heuristic_scaler.txt
          git commit -m "chore: Retrained CUDA GEMM heuristic (automated)"
          git push
```

## Related Documentation

- **Training Script**: `python/train_cuda_neural_network.py`
- **Profiling Script**: `python/collect_profiling_data.py`
- **Validation Script**: `python/validate_heuristic.py`
- **AutoTuner API**: `src/v2/kernels/cuda/CudaGemmAutoTuner.h`
- **Quick Reference**: `CUDA_AUTOTUNER_QUICK_REFERENCE.md`
- **Session Summary**: `changelog/2025-11-03-cuda-gemm-autotuner-session-summary.md`

## Changelog

### November 3, 2025 - Initial Release
- ✅ Complete CMake-based pipeline
- ✅ Four-stage workflow (profile, train, validate, deploy)
- ✅ Automatic backup and recovery
- ✅ Validation gating (≥95% hit rate)
- ✅ CUDA backend auto-rebuild
- ✅ Helper targets (status, clean)

---

**Status**: Production Ready  
**Last Updated**: November 3, 2025  
**Maintainer**: David Sanftenberg
