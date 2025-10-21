# Documentation Refresh: Matrix Multiplication Backends

**Date**: October 20, 2025  
**Author**: David Sanftenberg  
**Status**: ✅ **COMPLETED**

## Summary

Comprehensive documentation update across three major files to reflect recent Intel MKL backend integration, refresh stale backend selection information, and consolidate matrix multiplication backend documentation.

## Objectives

1. ✅ Add comprehensive "Matrix Multiplication Backends" section to README.md
2. ✅ Update copilot-instructions.md to include Intel MKL as third backend
3. ✅ Refresh llaminar-architecture.instructions.md with MKL milestone and backend details
4. ✅ Remove stale references to "OpenBLAS vs COSMA" (now three backends)
5. ✅ Document BF16 quantization support and environment variables

## Documentation Changes

### 1. README.md Updates

**Added Major Section**: "Matrix Multiplication Backends" (~300 lines)

**Content Added**:
- **Backend #1 - OpenBLAS**: Baseline CPU BLAS (always available)
  - Performance characteristics (1.04 tok/s decode, 33.60 tok/s prefill)
  - Build configuration (standard cmake)
  - Environment variables (OMP_NUM_THREADS, KMP_AFFINITY)
  - Optimal use cases (decode, small prefill)

- **Backend #2 - COSMA**: Distributed MPI matmul (large prefill)
  - Engagement criteria (≥4096 tokens, multi-rank, not disabled)
  - Build configuration (COSMA submodule)
  - Environment variables (13 documented)
  - Performance notes (3.6× faster for ≥64K tokens)

- **Backend #3 - Intel MKL**: BF16 hardware acceleration (optional)
  - Motivation (OpenBLAS NaN bug on Cascade Lake)
  - Backend selection priority (MKL → OpenBLAS → FP32)
  - Build configuration (`-DUSE_MKL=ON`)
  - Environment variables (LLAMINAR_QUANT_BF16_GEMM, MKL_NUM_THREADS)
  - Parity status (387/387 tests passing)
  - Performance characteristics (within 3% of FP32, 2× memory savings)

- **Backend Selection Decision Tree**: Visual flowchart showing routing logic
- **Comparison Matrix**: Table comparing latency, throughput, memory, best use cases
- **Testing Commands**: How to run parity tests for each backend
- **Debugging Commands**: How to force specific backends for A/B testing

**Feature List Updates**:
- Changed: "OpenBLAS (low latency) and COSMA (high throughput)" 
- To: "OpenBLAS (low latency), COSMA (distributed), and Intel MKL (BF16 acceleration)"

**Architecture Diagram Updates**:
- Changed: "OpenBLAS (small ops) and COSMA (large ops)"
- To: "OpenBLAS (baseline), COSMA (distributed), and Intel MKL (BF16 quantized)"

### 2. copilot-instructions.md Updates

**Section Renamed**: "COSMA vs OpenBLAS Integration" → "Matrix Multiplication Backends"

**Project Overview Updated**:
- Centralized Backend Selection: Now mentions "OpenBLAS, COSMA, and Intel MKL"
- Adaptive Backends: Now "OpenBLAS (baseline), COSMA (distributed), Intel MKL (BF16 acceleration)"

**Backend and Execution Section**:
- Added: `src/AdaptiveMatmul.{h,cpp}` with BF16 quantization support
- Added: `src/backends/MKLBackend.{h,cpp}` (optional with -DUSE_MKL=ON)

**Build System Section**:
- Added: Intel MKL-enabled build configuration
- Example: `cmake -B build_mkl -DUSE_MKL=ON -DCMAKE_PREFIX_PATH="/opt/intel/oneapi/mkl/latest"`

**Backend Selection Logic**:
- Updated: Now includes BF16 quantized path as highest priority
- Added: MKL vs OpenBLAS decision logic
- Added: Graceful fallback chain (MKL → OpenBLAS → FP32)

**New Subsection**: "Intel MKL BF16 Backend"
- Why Intel MKL? (OpenBLAS bug on Cascade Lake)
- Build configuration example
- Runtime activation commands
- Parity testing status (387/387 passing)
- Environment variables table
- Implementation details (source files, execution flow)

**Integration Patterns**:
- Added: `execute_bf16_matmul()` code example
- Shows: Fallback chain with error handling

**Production Ready Status**:
- Added: ✅ Intel MKL backend (BF16 quantized inference)
- Added: ✅ COSMA backend (large prefill ≥8K tokens)

**Removed Duplicate**:
- Fixed: Duplicate COSMA backend line

### 3. llaminar-architecture.instructions.md Updates

**Date Updated**: October 12, 2025 → October 20, 2025

**New Milestone Added**: "Intel MKL BF16 Backend Integration" (7 bullet points)
- Production-Quality BF16 GEMM
- OpenBLAS Bug Mitigation
- 100% PyTorch Parity (387/387 tests)
- Graceful Fallback Chain
- 2× Memory Savings
- Build Configuration
- References to implementation files and changelog

**Backend Abstraction Layer Section** (lines ~4150-4270):
- Updated Goals: Added "Enable BF16 quantization with production-quality GEMM"
- **New Table**: "Supported Backends (October 2025)" with 4 rows (OpenBLAS, COSMA, MKL, GPU future)
- **New Subsection**: "Intel MKL BF16 Backend" (50+ lines)
  - Motivation (OpenBLAS bug)
  - Key Features (5 bullet points)
  - Implementation details (source files)
  - Build configuration example
  - Runtime activation example

**Components Table**:
- Added: `src/backends/MKLBackend.{h,cpp}`
- Added: `AdaptiveMatmul` (backend selection and fallback logic)
- Updated: Decode path note about BF16 quantized support

**Invocation Pattern**:
- Added: BF16 quantized path code example
- Shows: MKL vs OpenBLAS selection with fallback

**Provider Selection Logic**:
- Added: BF16 Quantized Path as first priority
- Shows: MKL default when HAVE_MKL defined

## Key Documentation Improvements

### Comprehensive Backend Coverage

**Before**: Documentation scattered, incomplete backend information  
**After**: Centralized "Matrix Multiplication Backends" sections in all three files

**Content Structure**:
1. Overview of all three backends
2. When to use each backend (decision criteria)
3. Build configuration for each
4. Environment variables (comprehensive tables)
5. Performance characteristics (empirical data)
6. Testing commands (parity validation)
7. Debugging commands (forced backend selection)

### Consistency Across Files

All three documentation files now consistently describe:
- **Three backends**: OpenBLAS (baseline), COSMA (distributed), Intel MKL (BF16)
- **Backend selection logic**: Size-based + quantization-based routing
- **Build configurations**: Standard (build) vs MKL-enabled (build_mkl)
- **Parity testing**: 387/387 tests passing for all three backends
- **Environment variables**: Centralized reference with defaults and notes

### User-Facing Guidance

**README.md** provides:
- Decision tree for choosing backends
- Quick start commands for each backend
- Performance comparison table
- Testing and debugging commands

**copilot-instructions.md** provides:
- Developer guidelines for backend integration
- Code examples (backend selection, fallback chains)
- Build system details
- Project status tracking

**llaminar-architecture.instructions.md** provides:
- Architectural rationale for backend choices
- Implementation details (source files, line numbers)
- Provider pattern documentation
- Invocation patterns with code examples

## Environment Variable Documentation

### BF16 Quantization

| Variable | Purpose | Default | Status |
|----------|---------|---------|--------|
| `LLAMINAR_QUANT_BF16_GEMM` | Enable BF16 GEMM path | 0 (disabled) | Primary control |
| `LLAMINAR_QUANT_BF16_PREFER_MKL` | Force OpenBLAS BF16 | 1 (MKL default) | **DEPRECATED** - MKL now default |
| `MKL_NUM_THREADS` | MKL-specific threading | Inherits OMP_NUM_THREADS | MKL tuning |
| `MKL_DYNAMIC` | Dynamic thread adjustment | false (recommended) | MKL tuning |

### COSMA Prefill (13 variables documented)

All COSMA-related environment variables now documented in comprehensive table with:
- Variable name
- Purpose
- Default value
- Notes/warnings

### OpenBLAS Threading

| Variable | Purpose | Auto-Set |
|----------|---------|----------|
| `OMP_NUM_THREADS` | Thread count | Yes (physical cores per socket) |
| `OMP_PLACES` | Thread placement | Yes (sockets) |
| `OMP_PROC_BIND` | Binding policy | Yes (close) |
| `KMP_AFFINITY` | Intel runtime affinity | Yes (granularity=fine,compact) |
| `KMP_BLOCKTIME` | Spin-wait time | Yes (0 for low latency) |

## Build System Documentation

### Standard Build (OpenBLAS only)

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### MKL-Enabled Build (BF16 support)

```bash
cmake -B build_mkl -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/intel/oneapi/mkl/latest" \
  -DUSE_MKL=ON
cmake --build build_mkl --parallel
```

### Verification

```bash
ldd build_mkl/llaminar | grep mkl
# Expected: libmkl_intel_lp64.so, libmkl_core.so, libmkl_gnu_thread.so
```

## Testing Documentation

### Parity Tests (All Three Backends)

```bash
# OpenBLAS
ctest --test-dir build -R "ParityFramework.OpenBLASPrefillVsPyTorch" --verbose

# COSMA (requires MPI)
mpirun -np 2 ctest --test-dir build -R "ParityFramework.COSMAPrefillVsPyTorch" --verbose

# Intel MKL (requires build_mkl)
ctest --test-dir build_mkl -R "ParityFramework.MKLPrefillVsPyTorch" --verbose

# All three backends
GTEST_FILTER="ParityFramework.*VsPyTorch" \
  ctest --test-dir build_mkl -R "ParityFrameworkTest" --verbose
```

### Backend Forcing (Debugging)

```bash
# Force OpenBLAS (disable COSMA)
export ADAPTIVE_DISABLE_COSMA=1
./run_llaminar.sh -m model.gguf -v

# Force MKL BF16
export LLAMINAR_QUANT_BF16_GEMM=1
./run_llaminar.sh -m model.gguf -v

# Force OpenBLAS BF16 (bypass MKL)
export LLAMINAR_QUANT_BF16_GEMM=1
export LLAMINAR_QUANT_BF16_PREFER_MKL=0
./run_llaminar.sh -m model.gguf -v

# Validate COSMA tiles (debugging only)
export LLAMINAR_COSMA_VALIDATE_TILE=64
mpirun -np 2 ./run_llaminar.sh -m model.gguf -v
```

## Performance Documentation

### OpenBLAS Baseline

- Decode: ~1.04 tok/s (Q8_0, Debug build)
- Prefill (8 tokens): ~6.58 tok/s
- Prefill (512 tokens): ~33.60 tok/s
- 134× faster than COSMA for single-token decode

### COSMA Distributed

- Becomes competitive at ≥ 8K tokens
- 3.6× faster for ≥ 64K tokens
- Communication overhead dominates for small ops
- Best for document summarization, RAG

### Intel MKL BF16

- Within 3% of FP32 baseline (same FLOPs)
- 2× memory savings (BF16 vs FP32 storage)
- Hardware acceleration on Ice Lake, Sapphire Rapids
- Software fallback on Cascade Lake (robust vs OpenBLAS)

## Files Modified

| File | Lines Changed | Type |
|------|---------------|------|
| `README.md` | +319 | Major addition |
| `.github/copilot-instructions.md` | +158 | Updates + additions |
| `.github/instructions/llaminar-architecture.instructions.md` | +94 | Milestone + updates |
| **Total** | **+571 lines** | **Comprehensive refresh** |

## Verification

### Documentation Consistency Checks

✅ All three files mention three backends (OpenBLAS, COSMA, Intel MKL)  
✅ Backend selection logic consistent across files  
✅ Build instructions consistent (standard vs MKL-enabled)  
✅ Environment variables documented in all relevant places  
✅ Parity test status consistent (387/387 passing)  
✅ References to implementation files consistent  

### Stale Information Removed

✅ "OpenBLAS vs COSMA" → "Three backends" (OpenBLAS, COSMA, MKL)  
✅ "Two adaptive backends" → "Three matrix multiplication backends"  
✅ Removed references to only two backend choices  
✅ Updated date in llaminar-architecture.instructions.md (Oct 12 → Oct 20)  

### New Information Added

✅ Intel MKL backend documentation (Why, When, How)  
✅ BF16 quantization support and environment variables  
✅ Backend selection decision tree  
✅ Comparison matrix (latency, throughput, memory, use cases)  
✅ Testing commands for all three backends  
✅ Debugging commands for forced backend selection  
✅ Build configuration for MKL-enabled builds  
✅ OpenBLAS bug mitigation documentation (Cascade Lake NaN issue)  

## Impact

### For End Users

- **Clear guidance** on which backend to use for their use case
- **Step-by-step instructions** for enabling BF16 quantization
- **Performance expectations** for each backend (empirical data)
- **Debugging commands** for troubleshooting backend selection

### For Developers

- **Implementation details** (source files, line numbers)
- **Backend integration patterns** (code examples)
- **Provider abstraction architecture** (extensibility for GPU)
- **Build system configuration** (MKL, COSMA, future backends)
- **Testing framework** (parity validation for all backends)

### For AI Assistants (Copilot)

- **Up-to-date project status** (Production Ready section)
- **Recent milestones** (Intel MKL integration)
- **Architectural decisions** (why three backends, when to use each)
- **Code patterns** (fallback chains, error handling)
- **Environment variables** (comprehensive reference)

## Follow-Up Tasks

### Completed in This Session ✅

- [x] Add "Matrix Multiplication Backends" section to README.md
- [x] Update feature list to mention three backends
- [x] Update architecture diagram to include MKL
- [x] Rename "COSMA vs OpenBLAS" to "Matrix Multiplication Backends"
- [x] Add Intel MKL subsection to copilot-instructions.md
- [x] Update build system documentation
- [x] Add MKL milestone to llaminar-architecture.instructions.md
- [x] Update Backend Abstraction Layer section
- [x] Update date in architecture instructions
- [x] Create comprehensive summary document (this file)

### Future Documentation Improvements 🔄

- [ ] Add performance benchmark results for MKL BF16 (Release build)
- [ ] Create visual diagrams for backend selection flowchart
- [ ] Add GPU backend placeholder documentation
- [ ] Document activation BF16 storage (Phase 5 of quantized tensor architecture)
- [ ] Add KV cache BF16 storage documentation (future work)
- [ ] Create troubleshooting guide for backend-specific issues
- [ ] Add example inference scripts for each backend

## References

### Implementation Files

- `src/backends/MKLBackend.{h,cpp}` - Intel MKL cblas_sbgemm wrapper
- `src/AdaptiveMatmul.{h,cpp}` - Backend selection logic (lines ~315-380)
- `src/MatmulBackendSelection.{h,cpp}` - Centralized backend decision logic
- `src/CosmaPrefillManager.{h,cpp}` - COSMA distributed prefill coordination
- `src/OpenblasPrefillProvider.{h,cpp}` - OpenBLAS prefill implementation

### Related Changelogs

- `changelog/2025-10-20-mkl-default-backend.md` - MKL integration summary
- `changelog/2025-10-20-quantized-tensor-status-assessment.md` - Quantized tensor status
- `changelog/2025-10-19-mkl-integration-complete.md` - Initial MKL integration

### Documentation Files Updated

- `README.md` - Main project documentation (+319 lines)
- `.github/copilot-instructions.md` - Development guidelines (+158 lines)
- `.github/instructions/llaminar-architecture.instructions.md` - Architecture deep dive (+94 lines)

## Conclusion

This documentation refresh provides comprehensive, consistent coverage of Llaminar's three matrix multiplication backends across all major documentation files. Users, developers, and AI assistants now have clear guidance on:

1. **Which backend to use** (decision criteria, use cases)
2. **How to enable each backend** (build configuration, environment variables)
3. **What to expect** (performance characteristics, parity status)
4. **How to debug** (forced backend selection, validation commands)

The documentation accurately reflects the current state of the codebase (October 20, 2025) with all three backends production-ready and tested.

---

**Total Lines Added**: 571  
**Files Modified**: 3  
**New Sections**: 4 major sections  
**Documentation Coverage**: 100% for all three backends  
**Status**: ✅ **COMPLETE**
