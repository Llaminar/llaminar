# Heuristic File Organization Fix - November 2, 2025

## Problem

ML training scripts (`train_cuda_heuristic.py` and `train_tensorcore_heuristic.py`) were generating files directly in `src/v2/kernels/cuda/` instead of the `generated/` subfolder:

**Before** (wrong location):
```
src/v2/kernels/cuda/
├── cuda_heuristic_weights.h       ← AUTO-GENERATED (wrong place!)
├── cuda_heuristic_lookup.h        ← AUTO-GENERATED (wrong place!)
├── tensorcore_heuristic_weights.h ← AUTO-GENERATED (wrong place!)
├── CudaGemmAutoTuner.cu           ← Manual source code
├── IQ4_NL_BlockDecoder.h          ← Manual source code
└── generated/                     ← Should contain ALL generated files
    ├── CudaGemmVariants_00.inc
    └── ...
```

**Issues**:
1. Mixed manual code with generated code (confusing)
2. Training scripts could overwrite files in wrong location
3. Inconsistent with `generated/` folder pattern for `.inc` files
4. No `.gitignore` protection for artifacts

---

## Solution

### 1. Fixed Training Script Defaults

**Moved from workspace root to `python/` subfolder:**
- `train_cuda_heuristic.py` → `src/v2/kernels/cuda/python/train_cuda_heuristic.py`
- `train_tensorcore_heuristic.py` → `src/v2/kernels/cuda/python/train_tensorcore_heuristic.py`

**Updated default paths** (relative to new location):

**`python/train_cuda_heuristic.py`**:
```python
# Before (from workspace root):
parser.add_argument('--input', default='cuda_gemm_benchmark_data.csv')
parser.add_argument('--output-dir', default='src/v2/kernels/cuda/generated')

# After (from python/ subfolder):
parser.add_argument('--input', default='../../../../../build_v2/cuda_gemm_benchmark_data.csv')
parser.add_argument('--output-dir', default='../generated')
```

**`python/train_tensorcore_heuristic.py`**:
```python
# Before (from workspace root):
parser.add_argument('--input', default='build_v2/tensorcore_benchmark_data.csv')
parser.add_argument('--output-dir', default='src/v2/kernels/cuda/generated')

# After (from python/ subfolder):
parser.add_argument('--input', default='../../../../../build_v2/tensorcore_benchmark_data.csv')
parser.add_argument('--output-dir', default='../generated')
```

### 2. Moved Existing Generated Files

```bash
cd src/v2/kernels/cuda
mv cuda_heuristic_weights.h generated/
mv cuda_heuristic_lookup.h generated/
mv tensorcore_heuristic_weights.h generated/
```

### 3. Updated Include Paths

**`src/v2/kernels/cuda/CudaGemmAutoTuner.cu`**:
```cpp
// Before:
#include "cuda_heuristic_weights.h"
#include "cuda_heuristic_lookup.h"

// After:
#include "generated/cuda_heuristic_weights.h"
#include "generated/cuda_heuristic_lookup.h"
```

### 4. Added `.gitignore` Protection

**`.gitignore`**:
```gitignore
# ML/Benchmark training artifacts (should be in generated/ folder)
src/v2/kernels/cuda/*.csv
src/v2/kernels/cuda/*.png
src/v2/kernels/cuda/*.txt
src/v2/kernels/cuda/*_heuristic_*.h
```

**Rationale**: If someone accidentally runs a training script with the old default, these files won't be committed.

---

## After: Clean Organization

**Current structure**:
```
src/v2/kernels/cuda/
├── CudaGemmAutoTuner.cu              ← Manual source code
├── CudaGemmAutoTuner.h
├── CudaGemmConfig.h
├── CudaGemmFactory.cu
├── CudaGemmFactory.h
├── CudaGemmKernelTensorCoreCuTe.cuh  ← Manual source code (BF16 kernel)
├── CudaGemmVariants.cu
├── CudaGemmVariants.h
├── CudaGemmVariantsOptimized.cu
├── CudaGemmVariantsOptimized.h
├── CudaGemmVariantsTensorCore.cu
├── CudaGemmVariantsTensorCore.h
├── IQ4_NL_BlockDecoder.cu            ← Manual source code (BF16 decoder)
├── IQ4_NL_BlockDecoder.h
├── generate_cuda_gemm_variants.py    ← Code generator script
├── python/                           ← **Training scripts (NEW)**
│   ├── README.md                     ← Training guide
│   ├── train_cuda_heuristic.py       ← CUDA GEMM heuristic trainer
│   └── train_tensorcore_heuristic.py ← Tensor Core heuristic trainer
└── generated/                        ← **All generated code**
    ├── CudaGemmVariants_00.inc       ← Generated variant includes
    ├── CudaGemmVariants_01.inc
    ├── ...
    ├── CudaGemmVariants_09.inc
    ├── GeneratedConfigs.h             ← Generated config list
    ├── sources.cmake                  ← Generated CMake
    ├── cuda_heuristic_lookup.h        ← Generated ML heuristic LUT
    ├── cuda_heuristic_weights.h       ← Generated ML weights
    └── tensorcore_heuristic_weights.h ← Generated TC heuristic
```

**Benefits**:
- ✅ Clear separation: Manual code vs generated code
- ✅ Consistent pattern: All `.inc`, `.h` generated files in `generated/`
- ✅ `.gitignore` protection: Artifacts won't be accidentally committed
- ✅ Easy cleanup: `rm -rf generated/*` regenerates everything

---

## Files Changed

### Python Scripts (2 files)
1. **`train_cuda_heuristic.py`**: Changed `--output-dir` default (line 406)
2. **`train_tensorcore_heuristic.py`**: Changed `--output-dir` default (line 297)

### C++ Source (1 file)
3. **`src/v2/kernels/cuda/CudaGemmAutoTuner.cu`**: Updated include paths (lines 12-13)

### Configuration (1 file)
4. **`.gitignore`**: Added artifact protection patterns

### Generated Files Moved (3 files)
5. `cuda_heuristic_weights.h` → `generated/cuda_heuristic_weights.h`
6. `cuda_heuristic_lookup.h` → `generated/cuda_heuristic_lookup.h`
7. `tensorcore_heuristic_weights.h` → `generated/tensorcore_heuristic_weights.h`

---

## Verification

### Build Success
```bash
cmake --build build_v2 --target cuda_backend -j$(nproc)
# Result: [100%] Built target cuda_backend ✅

cmake --build build_v2 --target llaminar2_core -j$(nproc)
# Result: [100%] Built target llaminar2_core ✅
```

### Include Paths Correct
```bash
grep -r "#include.*heuristic" src/v2/kernels/cuda/*.cu
# src/v2/kernels/cuda/CudaGemmAutoTuner.cu:#include "generated/cuda_heuristic_weights.h"
# src/v2/kernels/cuda/CudaGemmAutoTuner.cu:#include "generated/cuda_heuristic_lookup.h"
# ✅ All includes point to generated/
```

### No Orphaned Files
```bash
ls src/v2/kernels/cuda/*_heuristic_*.h 2>/dev/null
# ls: cannot access: No such file or directory ✅
```

---

## Future Training Script Usage

**Correct usage** (new location):
```bash
# Navigate to python/ folder
cd src/v2/kernels/cuda/python

# Run with default paths (relative to python/ directory)
python train_cuda_heuristic.py
python train_tensorcore_heuristic.py

# Output files (relative to python/):
# - ../generated/cuda_heuristic_weights.h
# - ../generated/cuda_heuristic_lookup.h
# - ../generated/tensorcore_heuristic_weights.h
# - (plus .png, .txt, .csv artifacts - gitignored)
```

**Old usage from workspace root** (deprecated):
```bash
# DON'T DO THIS (scripts moved!)
cd /workspaces/llaminar
python train_cuda_heuristic.py  # ❌ File not found
```

**Override if needed**:
```bash
cd src/v2/kernels/cuda/python
python train_cuda_heuristic.py --output-dir /custom/path
```

**Documentation**: See `src/v2/kernels/cuda/python/README.md` for complete usage guide.

---

## Related Changes

This fix is part of the **November 2, 2025 cleanup session**:

1. **Cleanup dead code**: Removed 4 orphaned kernel files + artifacts
   - See: `changelog/2025-11-02-cuda-kernel-folder-cleanup.md`

2. **BF16 support**: Added BF16 precision to CuTe kernel
   - See: `changelog/2025-11-02-bf16-support-cute-kernel.md`

3. **File organization**: Fixed heuristic file placement (this document)

**Combined impact**: `src/v2/kernels/cuda/` now has **clear structure**:
- Manual source: 15 files (`.cu`, `.h`, `.cuh`, `.py`)
- Generated code: `generated/` folder (15 files)
- No artifacts, no orphaned code, no confusion

---

## Conclusion

✅ **Problem solved**: Generated files now live in `generated/` folder alongside other generated code.

**Key improvements**:
- Training scripts default to correct output location
- Include paths updated (`generated/` prefix)
- `.gitignore` prevents accidental commits of artifacts
- Clean separation of manual vs generated code

**Developer experience**: 
- Easier to understand folder structure
- Safe to delete `generated/*` and regenerate
- No more confusion about which files are auto-generated

---

**Date**: November 2, 2025  
**Author**: David Sanftenberg (with GitHub Copilot)  
**Related**: `changelog/2025-11-02-cuda-kernel-folder-cleanup.md`
