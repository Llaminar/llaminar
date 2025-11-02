# Python Training Scripts Organization - November 2, 2025

## Summary

Moved ML heuristic training scripts from workspace root to a dedicated `python/` subfolder within the CUDA kernels directory for better organization.

**Before**: Scripts scattered in workspace root  
**After**: All Python training scripts organized in `src/v2/kernels/cuda/python/`

---

## Changes

### 1. Created `python/` Folder

```bash
src/v2/kernels/cuda/
└── python/          ← NEW: Dedicated folder for training scripts
    ├── README.md
    ├── train_cuda_heuristic.py
    └── train_tensorcore_heuristic.py
```

### 2. Moved Training Scripts

**From workspace root to `python/`:**
- `train_cuda_heuristic.py` → `src/v2/kernels/cuda/python/train_cuda_heuristic.py`
- `train_tensorcore_heuristic.py` → `src/v2/kernels/cuda/python/train_tensorcore_heuristic.py`

### 3. Updated Default Paths

Scripts now use relative paths from their new location in `python/`:

**`train_cuda_heuristic.py`**:
```python
# Input: ../../../../../build_v2/cuda_gemm_benchmark_data.csv
# Output: ../generated/
```

**`train_tensorcore_heuristic.py`**:
```python
# Input: ../../../../../build_v2/tensorcore_benchmark_data.csv
# Output: ../generated/
```

### 4. Created Documentation

**`python/README.md`** (comprehensive training guide):
- Script usage instructions
- Workflow (benchmark → train → rebuild)
- Output file descriptions
- Debugging tips
- Development guide for adding new features

---

## Directory Structure

**Complete CUDA kernels folder organization**:

```
src/v2/kernels/cuda/
├── CudaGemmAutoTuner.{cu,h}          ← C++ source (uses generated headers)
├── CudaGemmFactory.{cu,h}
├── CudaGemmVariants*.{cu,h}
├── CudaGemmKernelTensorCoreCuTe.cuh  ← BF16 CuTe kernel
├── IQ4_NL_BlockDecoder.{cu,h}        ← BF16 decoder
├── CudaGemmConfig.h
├── generate_cuda_gemm_variants.py    ← Variant generator
│
├── python/                           ← **Python training scripts**
│   ├── README.md                     ← Training guide (see below)
│   ├── train_cuda_heuristic.py       ← CUDA GEMM ML trainer
│   └── train_tensorcore_heuristic.py ← Tensor Core ML trainer
│
└── generated/                        ← **All generated outputs**
    ├── CudaGemmVariants_*.inc        ← Generated variants
    ├── cuda_heuristic_weights.h      ← ML model weights
    ├── cuda_heuristic_lookup.h       ← Best config LUT
    └── tensorcore_heuristic_weights.h ← TC heuristic
```

**Benefits**:
- ✅ **Separation of concerns**: Python scripts separate from C++ code
- ✅ **Clear organization**: All training-related files in one place
- ✅ **Easy discovery**: `python/` folder clearly indicates purpose
- ✅ **Relative paths**: Scripts work from their natural location
- ✅ **Documentation**: README explains entire training workflow

---

## Usage (NEW)

### Training Scripts

**From `python/` folder** (recommended):
```bash
cd src/v2/kernels/cuda/python

# Train CUDA GEMM heuristic
python train_cuda_heuristic.py

# Train Tensor Core heuristic
python train_tensorcore_heuristic.py

# Output goes to: ../generated/*.h
```

**From workspace root** (with explicit paths):
```bash
cd /workspaces/llaminar

# CUDA GEMM
python src/v2/kernels/cuda/python/train_cuda_heuristic.py \
  --input build_v2/cuda_gemm_benchmark_data.csv \
  --output-dir src/v2/kernels/cuda/generated

# Tensor Core
python src/v2/kernels/cuda/python/train_tensorcore_heuristic.py \
  --input build_v2/tensorcore_benchmark_data.csv \
  --output-dir src/v2/kernels/cuda/generated
```

### Complete Workflow

**1. Run benchmarks** (generates training data):
```bash
cd build_v2
ctest -R "V2_Perf_CudaHeuristicValidation" -V
ctest -R "V2_Perf_TensorCoreHeuristicValidation" -V
```

**2. Train models** (generates C++ headers):
```bash
cd ../src/v2/kernels/cuda/python
python train_cuda_heuristic.py
python train_tensorcore_heuristic.py
```

**3. Rebuild** (uses new heuristics):
```bash
cd /workspaces/llaminar
cmake --build build_v2 --target cuda_backend -j$(nproc)
```

**4. Validate** (verify improvements):
```bash
cd build_v2
ctest -R "V2_Perf_CudaHeuristicValidation" -V
```

---

## Documentation

### `python/README.md` Contents

The README provides:

1. **Script descriptions**: What each script does, what it generates
2. **Usage examples**: Default and custom paths
3. **Requirements**: Benchmark prerequisites, Python dependencies
4. **Workflow guide**: Step-by-step training process
5. **Output files**: Where generated files go, what they contain
6. **Model details**: Algorithm descriptions, feature engineering
7. **Debugging**: Common issues and solutions
8. **Development**: How to add new features, extend to new operations
9. **Maintenance**: When to retrain, how to update

**Key sections**:
- Quick start for common use cases
- Detailed explanations for developers
- Troubleshooting guide
- Reference for model internals

---

## Files Changed

### Moved Files (2)
1. `train_cuda_heuristic.py` → `src/v2/kernels/cuda/python/train_cuda_heuristic.py`
2. `train_tensorcore_heuristic.py` → `src/v2/kernels/cuda/python/train_tensorcore_heuristic.py`

### Modified Files (2)
Both training scripts updated with new default paths:
- Input: `../../../../../build_v2/[benchmark_data].csv`
- Output: `../generated/`

### Created Files (1)
3. `src/v2/kernels/cuda/python/README.md` - Comprehensive training guide

### Documentation Updates (2)
4. `changelog/2025-11-02-heuristic-files-organization.md` - Updated with python/ folder
5. `changelog/2025-11-02-cuda-kernel-folder-cleanup.md` - Updated file listing

---

## Verification

### Scripts Moved Correctly
```bash
$ find /workspaces/llaminar -name "*heuristic*.py"
src/v2/kernels/cuda/python/train_cuda_heuristic.py
src/v2/kernels/cuda/python/train_tensorcore_heuristic.py
# ✓ No files in workspace root
```

### Paths Correct
```bash
$ cd src/v2/kernels/cuda/python
$ grep "default=" train_cuda_heuristic.py | grep output-dir
parser.add_argument('--output-dir', default='../generated',
# ✓ Relative path to generated/

$ grep "default=" train_tensorcore_heuristic.py | grep output-dir
parser.add_argument('--output-dir', default='../generated',
# ✓ Relative path to generated/
```

### Directory Structure
```bash
$ ls src/v2/kernels/cuda/
CudaGemmAutoTuner.cu  CudaGemmVariants.h            IQ4_NL_BlockDecoder.h
CudaGemmAutoTuner.h   CudaGemmVariantsOptimized.cu  generate_cuda_gemm_variants.py
CudaGemmConfig.h      CudaGemmVariantsOptimized.h   generated/
CudaGemmFactory.cu    CudaGemmVariantsTensorCore.cu python/         ← NEW
CudaGemmFactory.h     CudaGemmVariantsTensorCore.h
...

$ ls src/v2/kernels/cuda/python/
README.md                      ← NEW
train_cuda_heuristic.py        ← MOVED
train_tensorcore_heuristic.py  ← MOVED
```

---

## Related Changes

This organization is part of the **November 2, 2025 cleanup session**:

1. **Dead code cleanup**: Removed orphaned kernels and artifacts
   - See: `changelog/2025-11-02-cuda-kernel-folder-cleanup.md`

2. **Generated files organization**: Moved heuristic headers to `generated/`
   - See: `changelog/2025-11-02-heuristic-files-organization.md`

3. **Python scripts organization**: This document
   - Created `python/` folder
   - Moved training scripts
   - Added comprehensive README

4. **BF16 support**: Added BF16 to CuTe kernel (earlier in session)
   - See: `changelog/2025-11-02-bf16-support-cute-kernel.md`

---

## Benefits

### Before (Scattered Organization)
```
/workspaces/llaminar/
├── train_cuda_heuristic.py           ← Root level (confusing)
├── train_tensorcore_heuristic.py     ← Root level (confusing)
├── src/v2/kernels/cuda/
│   ├── CudaGemmAutoTuner.cu          ← Uses heuristics
│   └── generated/
│       └── cuda_heuristic_*.h        ← Generated by scripts
└── build_v2/
    └── cuda_gemm_benchmark_data.csv  ← Input data
```

**Issues**:
- Training scripts far from generated outputs
- Not obvious these scripts are CUDA-specific
- Hard to find related files
- No documentation of training workflow

### After (Clean Organization)
```
src/v2/kernels/cuda/
├── python/                           ← Clear separation
│   ├── README.md                     ← Complete guide
│   ├── train_cuda_heuristic.py       ← CUDA trainer
│   └── train_tensorcore_heuristic.py ← TC trainer
├── generated/                        ← Generated outputs
│   ├── cuda_heuristic_weights.h      ← From python/train_cuda_heuristic.py
│   └── tensorcore_heuristic_*.h      ← From python/train_tensorcore_heuristic.py
└── CudaGemmAutoTuner.cu              ← Uses generated headers
```

**Benefits**:
- ✅ Scripts next to their outputs
- ✅ Clear "python/" indicates scripting code
- ✅ README provides full documentation
- ✅ Easy to discover training workflow
- ✅ Consistent organization (like `generated/` folder)

---

## Future Enhancements

### Potential Additions to `python/`

1. **Visualization scripts**: Plot heuristic performance
   ```
   python/
   ├── train_cuda_heuristic.py
   ├── visualize_heuristics.py      ← NEW
   └── compare_models.py             ← NEW
   ```

2. **Data preprocessing**: Clean/augment benchmark data
   ```
   python/
   ├── preprocess_benchmarks.py     ← NEW
   └── augment_training_data.py     ← NEW
   ```

3. **Model validation**: Automated regression testing
   ```
   python/
   └── validate_heuristic_accuracy.py ← NEW
   ```

4. **Common utilities**: Shared functions
   ```
   python/
   ├── utils/                        ← NEW
   │   ├── __init__.py
   │   ├── feature_engineering.py
   │   └── cpp_export.py
   └── train_*.py                    ← Import from utils/
   ```

---

## Conclusion

✅ **Organized Python training scripts** into dedicated `python/` folder within CUDA kernels directory.

**Key improvements**:
- Clear separation: Python scripts vs C++ code vs generated files
- Better discoverability: All training scripts in one place
- Comprehensive documentation: README explains entire workflow
- Relative paths: Scripts work from natural location
- Consistent organization: Follows `generated/` folder pattern

**Developer experience**:
- Easy to find: "Where are the training scripts?" → `python/` folder
- Self-documenting: README provides complete guide
- Natural workflow: Train from python/, output to ../generated/
- Clean root: No more scripts scattered in workspace root

---

**Date**: November 2, 2025  
**Author**: David Sanftenberg (with GitHub Copilot)  
**Related**: 
- `changelog/2025-11-02-cuda-kernel-folder-cleanup.md`
- `changelog/2025-11-02-heuristic-files-organization.md`
