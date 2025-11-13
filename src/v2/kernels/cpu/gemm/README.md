# CPU GEMM Kernels - Architecture Overview

**Location**: `src/v2/kernels/cpu/gemm/`  
**Purpose**: High-performance matrix multiplication kernels for CPU inference  
**Last Updated**: November 11, 2025

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Integer GEMM System (Q8_0)](#integer-gemm-system-q8_0)
4. [FP32 GEMM System](#fp32-gemm-system)
5. [Registry Pattern](#registry-pattern)
6. [Configuration Space](#configuration-space)
7. [Auto-Tuning System](#auto-tuning-system)
8. [File Organization](#file-organization)
9. [Adding New Kernels](#adding-new-kernels)

---

Main tests to be iterating within:

1. Perf__IntegerGEMM_Minimal.cpp -- a minimal, isolated test file for `perf` profiling.

2. Perf__IntegerGEMM_QwenProfile.cpp -- the MAIN TEST FILE. This tests a variety of realistic shapes and sizes. You should iterate on this test suite except for `perf` profiling. 


---

## System Overview

The CPU GEMM subsystem provides highly optimized matrix multiplication for various data types:

- **Integer GEMM** (`int8/`): Q8_0×Quantized→Q8_0 (INT8 VNNI computation)
- **FP32 GEMM** (`fp32/`): Float32×Float32→Float32 (FP32 FMA computation)
- **BF16 GEMM** (`bf16/`): BFloat16 computation (future)
- **FP16 GEMM** (`fp16/`): Float16 computation (future)

### Key Design Principles

1. **Template Metaprogramming**: Kernels are C++ templates specialized for ISA, tile sizes, and blocking parameters
2. **Registry Pattern**: Runtime dispatch to optimal template instantiation based on hardware and workload
3. **Static Registration**: `__attribute__((constructor))` auto-registers kernels at program startup
4. **Zero Runtime Overhead**: Template instantiation happens at compile-time, no virtual dispatch in hot paths
5. **Auto-Tuning**: Machine learning-based configuration selection from large configuration space

---

## Architecture Layers

The GEMM system has a clean layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Application Layer                                           │
│ • Tensor operations (Tensor::gemm(), Q8_0Tensor::gemm())    │
│ • Pipeline kernels (Attention, FFN, etc.)                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ Adapter Layer                                               │
│ • IntegerGemmAdapter (int8/IntegerGemmAdapter.h)            │
│ • FP32GemmAdapter (fp32/GemmAdapter.h)                      │
│ • Selects backend, validates shapes, manages buffers        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ Registry Layer                                              │
│ • IntegerGemmKernelRegistry (int8/IntegerGemmKernelRegistry.h) │
│ • GemmMicroKernelRegistry (GemmMicroKernelRegistry.h)       │
│ • Runtime dispatch: get_kernel(ISA, MR, NR, ...)           │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ Auto-Tuner Layer (Optional)                                 │
│ • GemmAutoTuner (GemmAutoTuner.h)                           │
│ • IntegerGemmAutoTuner (IntegerGemmAutoTuner.h)             │
│ • ML-based config selection, smart search, caching          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ Template Kernel Layer                                       │
│ • IntegerGemmKernel<ISA,MR,NR,...> (IntegerGemmKernelTemplate.h) │
│ • GemmKernel<ISA,MR,NR,...> (GemmKernelTemplate.h)          │
│ • Cache blocking, tiling, loop unrolling                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ Micro-Kernel Layer                                          │
│ • IntegerGemmMicroKernel (int8/IntegerGemmMicroKernel.h)    │
│ • GemmMicroKernel (GemmMicroKernel.h)                       │
│ • SIMD intrinsics, register blocking, prefetching           │
└─────────────────────────────────────────────────────────────┘
```

---

## Integer GEMM System (Q8_0)

**Purpose**: Quantized inference with INT8×INT8→INT32 computation (AVX512-VNNI)

### Component Architecture

```
IntegerGemmKernel<ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC>
                    │
                    ├─ Uses: Q8_0BlockProvider interface
                    ├─ Uses: IntegerGemmMicroKernel for computation
                    ├─ Uses: IntegerRequantization for INT32→Q8_0 conversion
                    └─ Registered in: IntegerGemmKernelRegistry
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `IntegerGemmKernelTemplate.h` | Main kernel template, cache blocking logic | ~306 |
| `int8/IntegerGemmKernelRegistry.h` | Runtime dispatch registry (singleton) | ~151 |
| `int8/IntegerGemmKernelInit.cpp` | Force-link 64 instantiation shards | ~169 |
| `int8/IntegerGemmMicroKernel.h` | AVX512-VNNI micro-kernel (register blocking) | ~600+ |
| `IntegerRequantization.h` | INT32→Q8_0 requantization with scale fusion | ~200 |
| `GemmWeightCache.h` | Q8_0BlockProvider interface, caching strategies | ~500+ |

### Data Flow

```cpp
// 1. Application calls tensor operation
Q8_0Tensor result;
result.gemm(A_q8, B_quantized);

// 2. Adapter selects backend
IntegerGemmAdapter adapter;
adapter.multiply(A_blocks, B_provider, C_blocks, m, n, k);

// 3. Registry dispatches to optimal kernel
auto& registry = IntegerGemmKernelRegistry::instance();
auto kernel_func = registry.get_kernel(
    "simd::AVX512VNNITag", 
    mr=4, nr=32, unroll_k=4, prefetch_dist=2,
    mc=256, kc=512, nc=128
);
kernel_func(A_blocks, B_provider, C_blocks, m, n, k);

// 4. Template kernel executes (cache-blocked)
IntegerGemmKernel<AVX512VNNITag, 4, 32, 4, 2, 256, 512, 128>::multiply(...) {
    // Outer loops: MC×NC×KC blocking
    for (int jc = 0; jc < n; jc += NC) {
        for (int pc = 0; pc < k; pc += KC) {
            for (int ic = 0; ic < m; ic += MC) {
                // Micro-kernel: INT8 VNNI computation
                IntegerGemmMicroKernel<...>::compute_block(...);
            }
        }
    }
}

// 5. Micro-kernel computes tile
IntegerGemmMicroKernel::compute_block() {
    // Load A_q8 into registers (MR rows)
    // Load B_q8 from provider (NR cols)
    // INT8×INT8→INT32 VNNI multiply-accumulate
    // Requantize INT32→Q8_0 with scale fusion
    // Store Q8_0 result
}
```

### Template Parameters

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| `ISA` | SIMD instruction set | `AVX512VNNITag`, `AVX2Tag` | Micro-kernel implementation |
| `MR` | Micro-kernel M (rows) | 1, 2, 4, 8, 16 | Register usage, L1 cache |
| `NR` | Micro-kernel N (cols) | 32 (fixed for Q8_0) | Q8_0 block alignment |
| `UNROLL_K` | K-loop unroll factor | 1, 2, 4, 8, 16 | Instruction-level parallelism |
| `PREFETCH_DIST` | Prefetch distance | 0, 1, 2, 3, 5 | Cache miss latency hiding |
| `MC` | M cache block size | 128, 256, 512, 1024 | L2 cache blocking |
| `KC` | K cache block size | 256, 512, 1024, 2048 | L2 cache blocking |
| `NC` | N cache block size | 64, 128, 256, 512 | L2 cache blocking |

**Configuration Space**: 8000 valid combinations (5 ISAs × 5 MR × 1 NR × 5 UNROLL_K × 5 PREFETCH_DIST × 4 MC × 4 KC × 4 NC)

### Static Registration Pattern

Each template instantiation auto-registers itself at program startup:

```cpp
// In generated/IntegerGemmInstantiations_XX.cpp
template class IntegerGemmKernel<simd::AVX512VNNITag, 4, 32, 4, 2, 256, 512, 128>;

namespace {
    __attribute__((constructor))
    void register_simd_AVX512VNNITag_4_32_4_2_256_512_128() {
        IntegerGemmKernelRegistry::instance().register_kernel(
            "simd::AVX512VNNITag", 4, 32, 4, 2, 256, 512, 128,
            [](const Q8_0Block* A, Q8_0BlockProvider& B, Q8_0Block* C, int m, int n, int k) {
                return IntegerGemmKernel<simd::AVX512VNNITag, 4, 32, 4, 2, 256, 512, 128>::multiply(A, B, C, m, n, k);
            }
        );
    }
}
```

**Force-Link Mechanism**: `IntegerGemmKernelInit.cpp` calls all 64 `forceLink_IntegerGemmInstantiations_XX()` functions to prevent linker from dropping unused object files from static library.

### Generation Scripts

**Location**: `python/generate_integer_gemm_instantiations.py`

**Purpose**: Generate 64 instantiation files (8000 kernels ÷ 125 per file)

**Usage**:
```bash
cd src/v2/kernels/cpu/gemm/python
python3 generate_integer_gemm_instantiations.py

# Output: int8/generated/IntegerGemmInstantiations_00.cpp through _63.cpp
```

**Key functions**:
- `generate_config_space()`: Enumerate valid (ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC) tuples
- `generate_instantiation()`: Produce template instantiation + static registration code
- `generate_file_header()`: Include directives, force-link extern declaration
- `distribute_to_files()`: Shard 8000 configs across 64 files for parallel compilation

---

## FP32 GEMM System

**Purpose**: Float32 inference with FP32×FP32→FP32 computation (FMA)

### Component Architecture

Similar to Integer GEMM but with FP32 types:

```
GemmKernel<ISA, MR, NR, UNROLL_K, PREFETCH_DIST, MC, KC, NC>
           │
           ├─ Uses: float* inputs/outputs
           ├─ Uses: GemmMicroKernel for FP32 FMA computation
           └─ Registered in: GemmMicroKernelRegistry
```

### Key Files

| File | Purpose |
|------|---------|
| `GemmKernelTemplate.h` | FP32 kernel template |
| `GemmMicroKernelRegistry.h` | FP32 registry |
| `fp32/GemmMicroKernelInit.cpp` | Force-link mechanism |
| `GemmMicroKernel.h` | FP32 micro-kernel base |
| `fp32/GemmMicroKernelTemplateAVX512.h` | AVX512 FP32 micro-kernel |
| `fp32/GemmMicroKernelTemplateAVX2.h` | AVX2 FP32 micro-kernel |

### Template Parameters

Same as Integer GEMM, except:
- `NR` is not fixed (typical: 8, 16, 24, 32 for FP32)
- Larger configuration space (~40,000 combinations)

---

## Registry Pattern

### Design

The registry pattern provides **runtime dispatch** to **compile-time specialized** template instantiations:

```cpp
template<typename ISA, int MR, int NR, ...>
class IntegerGemmKernel {
    static bool multiply(...);  // Template-specialized implementation
};

class IntegerGemmKernelRegistry {
    std::map<IntegerGemmKey, IntegerGemmFunc> kernels_;
    
    // Runtime lookup
    IntegerGemmFunc get_kernel(string isa, int mr, int nr, ...);
};
```

### Benefits

1. **Zero Runtime Overhead**: Template specialization happens at compile-time
2. **Optimal Code Generation**: Compiler can fully optimize each instantiation (loop unrolling, SIMD, etc.)
3. **Flexible Selection**: Runtime can choose best config based on hardware detection, profiling, or ML heuristic
4. **Easy Extension**: Add new configs by regenerating instantiation files

### Implementation Details

**Registry Key**: 8-tuple uniquely identifying a configuration:
```cpp
using IntegerGemmKey = std::tuple<
    std::string,  // ISA name ("simd::AVX512VNNITag")
    int,          // MR
    int,          // NR
    int,          // UNROLL_K
    int,          // PREFETCH_DIST
    int,          // MC
    int,          // KC
    int           // NC
>;
```

**Registry API**:
```cpp
class IntegerGemmKernelRegistry {
public:
    static IntegerGemmKernelRegistry& instance();  // Singleton
    
    void register_kernel(
        const std::string& isa_name,
        int mr, int nr, int unroll_k, int prefetch_dist,
        int mc, int kc, int nc,
        IntegerGemmFunc kernel_func
    );
    
    IntegerGemmFunc get_kernel(...) const;  // Returns nullptr if not found
    bool has_kernel(...) const;
    size_t size() const;  // Number of registered kernels
};
```

**Lookup Performance**: O(log n) with n=8000 → ~13 comparisons. Negligible vs kernel execution time (millions of ops).

---

## Configuration Space

### Integer GEMM Configuration Space

**Total**: 8000 valid combinations

**Dimensions**:
- **ISA** (5): AVX512VNNITag, AVX512FTag, AVX2Tag, SSE4Tag, ScalarTag
- **MR** (5): 1, 2, 4, 8, 16 (micro-kernel M dimension)
- **NR** (1): 32 (fixed for Q8_0 block alignment)
- **UNROLL_K** (5): 1, 2, 4, 8, 16 (K-loop unrolling)
- **PREFETCH_DIST** (5): 0, 1, 2, 3, 5 (prefetch iterations ahead)
- **MC** (4): 128, 256, 512, 1024 (M cache blocking)
- **KC** (4): 256, 512, 1024, 2048 (K cache blocking, multiples of 32)
- **NC** (4): 64, 128, 256, 512 (N cache blocking)

**Constraints**:
- `KC % 32 == 0` (Q8_0 block size alignment)
- `NR == 32` (Q8_0 block width)
- Valid ISA for target hardware (runtime check)

### FP32 GEMM Configuration Space

**Total**: ~40,000 valid combinations

**Additional flexibility**:
- **NR** (5): 8, 16, 24, 32, 48 (not fixed like Q8_0)
- No alignment constraints on KC (can use arbitrary values)

### Trade-offs

| Parameter | Small Values | Large Values |
|-----------|-------------|--------------|
| **MR** | Lower register pressure, more tiles | Fewer tiles, better amortization |
| **UNROLL_K** | Lower code size, less pressure | More ILP, better throughput |
| **PREFETCH_DIST** | Lower cache pollution | Hide more latency (if accurate) |
| **MC/KC/NC** | Fit in L1/L2 cache better | Fewer outer loop iterations |

**No Universal Best**: Optimal config depends on:
- Matrix dimensions (m, n, k)
- Hardware (cache sizes, SIMD width, memory bandwidth)
- Memory access patterns (contiguous vs strided)
- Compiler optimizations

---

## Auto-Tuning System

### Overview

The auto-tuner selects optimal configuration from 8000+ possibilities using:

1. **Heuristic Rules**: Quick decisions for common cases (small/large matrices, cache-friendly sizes)
2. **Machine Learning Model**: Neural network trained on empirical performance data
3. **Smart Search**: Binary search over sorted configuration space by predicted performance
4. **Runtime Caching**: Memoize best configs for frequently seen (m, n, k) triplets

### Components

| File | Purpose |
|------|---------|
| `GemmAutoTuner.h` | Base auto-tuner interface |
| `IntegerGemmAutoTuner.h` | Integer GEMM auto-tuner |
| `SmartGemmSearch.h` | Binary search over config space |

### Auto-Tuner Workflow

```cpp
// 1. Check cache
auto cached_config = autotuner.lookup_cache(m, n, k);
if (cached_config) {
    return registry.get_kernel(*cached_config);
}

// 2. Apply heuristic rules
if (m * n * k < SMALL_THRESHOLD) {
    return registry.get_kernel(small_matrix_heuristic());
}

// 3. Use ML model for prediction
auto predictions = ml_model.predict_all_configs(m, n, k, hardware_features);
std::sort(configs, by_prediction_desc);

// 4. Smart search (binary search over top K configs)
auto best_config = smart_search.find_best(configs.top_k(20), m, n, k);

// 5. Cache result
autotuner.cache_config(m, n, k, best_config);
return registry.get_kernel(best_config);
```

### Training Data Collection

**Script**: `python/collect_gemm_training_data.py`

**Process**:
1. Enumerate all 8000 configs
2. For each config, benchmark on representative (m, n, k) workloads
3. Record: (m, n, k, ISA, MR, NR, ..., MC, KC, NC) → throughput (GFLOPS)
4. Export CSV for ML training

**Usage**:
```bash
# Run benchmark sweep (outputs CSV)
./build_v2_release/performance/v2_perf_integer_gemm_config_sweep \
  --benchmark_mode=full \
  --output=gemm_training_data.csv

# Train ML model
python3 python/train_gemm_autotuner.py \
  --input=gemm_training_data.csv \
  --output=gemm_autotuner_weights.h
```

---

## File Organization

```
src/v2/kernels/cpu/gemm/
├── README.md                          # This file
├── README.md.old                      # Historical documentation
│
├── GemmKernelTemplate.h               # FP32 GEMM kernel template
├── IntegerGemmKernelTemplate.h        # Integer GEMM kernel template
├── IntegerRequantization.h            # INT32→Q8_0 requantization
├── GemmWeightCache.h                  # Q8_0BlockProvider interface
│
├── GemmMicroKernel.h                  # FP32 micro-kernel base
├── GemmMicroKernelRegistry.h          # FP32 registry
├── GemmMicroKernelInit.cpp            # FP32 force-link
├── GemmMicroKernelAdapter.h           # FP32 adapter
│
├── IntegerGemm.h                      # Integer GEMM public API
├── IntegerGemmAdapter.h               # Integer GEMM adapter
│
├── GemmAutoTuner.h                    # Auto-tuner base
├── IntegerGemmAutoTuner.h             # Integer auto-tuner
├── SmartGemmSearch.h                  # Config search
│
├── int8/                              # Integer GEMM specific
│   ├── IntegerGemmKernelRegistry.h    # Integer registry (8000 kernels)
│   ├── IntegerGemmKernelInit.cpp      # Force-link 64 shards
│   ├── IntegerGemmMicroKernel.h       # AVX512-VNNI micro-kernel
│   ├── GemmMicroKernelTemplateINT8.h  # INT8 micro-kernel specializations
│   └── generated/                     # Auto-generated instantiations
│       ├── IntegerGemmInstantiations_00.cpp  # Shard 0 (125 kernels)
│       ├── IntegerGemmInstantiations_01.cpp  # Shard 1 (125 kernels)
│       ├── ...
│       └── IntegerGemmInstantiations_63.cpp  # Shard 63 (125 kernels)
│
├── fp32/                              # FP32 GEMM specific
│   ├── GemmMicroKernelInit.cpp        # Force-link FP32 shards
│   ├── GemmMicroKernelTemplateAVX512.h # AVX512 FP32 micro-kernel
│   ├── GemmMicroKernelTemplateAVX2.h   # AVX2 FP32 micro-kernel
│   └── generated/                     # Auto-generated FP32 instantiations
│
├── bf16/                              # BFloat16 GEMM (future)
├── fp16/                              # Float16 GEMM (future)
│
└── python/                            # Code generation & training
    ├── generate_integer_gemm_instantiations.py  # Generate int8/generated/
    ├── generate_fp32_gemm_instantiations.py     # Generate fp32/generated/
    ├── collect_gemm_training_data.py            # Benchmark sweep
    └── train_gemm_autotuner.py                  # ML model training
```

### Generated Files (Not in Git)

**Integer GEMM**: 64 files, ~30KB each, ~2MB total
```
int8/generated/IntegerGemmInstantiations_00.cpp  # Configs 0-124
int8/generated/IntegerGemmInstantiations_01.cpp  # Configs 125-249
...
int8/generated/IntegerGemmInstantiations_63.cpp  # Configs 7875-7999
```

**FP32 GEMM**: ~320 files (40,000 configs ÷ 125 per file)

**Regeneration**:
```bash
cd python/
python3 generate_integer_gemm_instantiations.py  # Regenerate int8/generated/
python3 generate_fp32_gemm_instantiations.py     # Regenerate fp32/generated/
```

---

## Adding New Kernels

### 1. Add New ISA Support

**Example**: Add NEON support for ARM64

```cpp
// 1. Define ISA tag (in src/v2/kernels/cpu/simd/)
namespace simd {
    struct NEONTag {};
}

// 2. Implement micro-kernel specialization
template<>
class IntegerGemmMicroKernel<simd::NEONTag, MR, NR, ...> {
    // NEON intrinsics implementation
};

// 3. Add to config space (in python/generate_integer_gemm_instantiations.py)
SUPPORTED_ISAS = [
    'simd::AVX512VNNITag',
    'simd::AVX2Tag',
    'simd::NEONTag',  # NEW
]

// 4. Regenerate instantiations
python3 generate_integer_gemm_instantiations.py
```

### 2. Add New Data Type

**Example**: Add INT4 GEMM

```cpp
// 1. Create new subdirectory
mkdir int4/

// 2. Define block format (src/v2/tensors/Tensors.h)
struct INT4Block { ... };

// 3. Create kernel template (int4/IntegerGemmKernelTemplateINT4.h)
template<typename ISA, int MR, int NR, ...>
class INT4GemmKernel {
    static bool multiply(const INT4Block* A, INT4BlockProvider& B, INT4Block* C, ...);
};

// 4. Create registry (int4/INT4GemmKernelRegistry.h)
class INT4GemmKernelRegistry {
    // Similar to IntegerGemmKernelRegistry
};

// 5. Create generation script (python/generate_int4_gemm_instantiations.py)
// ... modeled after generate_integer_gemm_instantiations.py
```

### 3. Extend Configuration Space

**Example**: Add MC=2048 option

```python
# In python/generate_integer_gemm_instantiations.py
MC_VALUES = [128, 256, 512, 1024, 2048]  # Added 2048

# Regenerate (now 10,000 configs instead of 8,000)
python3 generate_integer_gemm_instantiations.py
```

**Impact**: Increases compilation time proportionally (more shards needed).

### 4. Optimize Micro-Kernel

**Location**: `int8/IntegerGemmMicroKernel.h`

**Example**: Add deeper K-loop unrolling for AVX512

```cpp
template<>
class IntegerGemmMicroKernel<simd::AVX512VNNITag, 4, 32, 16, ...> {
    // Unroll K-loop 16× instead of 4×
    // Requires more registers, may improve ILP
};
```

**Testing**:
```bash
# Build specific config
./build_v2_release/performance/v2_perf_integer_gemm_config_sweep \
  --gtest_filter="*RegistryDispatch*"

# Compare GFLOPS before/after optimization
```

---

## Performance Characteristics

### Build Performance

| Metric | Factory Pattern (Old) | Registry Pattern (New) |
|--------|----------------------|------------------------|
| **Core library build time** | Indefinite (blocked) | 1m3s real (51m30s CPU) |
| **Parallelism** | None (single 24K-line file) | 50× (64 shards compile independently) |
| **Incremental rebuild** | Full reparse required | Only changed shards rebuild |
| **Developer experience** | Terrible | Excellent |

### Runtime Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Registry lookup** | O(log 8000) = ~13 comparisons | Negligible vs kernel execution |
| **Template overhead** | Zero | Fully inlined at compile-time |
| **Optimal config selection** | Depends on auto-tuner | Heuristic: fast, ML: accurate |

### Memory Footprint

| Component | Size | Notes |
|-----------|------|-------|
| **Static library** | ~80MB | 8000 instantiated kernels |
| **Registry map** | ~500KB | 8000 entries × ~60 bytes each |
| **ML model weights** | ~50KB | Small neural network |

---

## Troubleshooting

### Build Issues

**Problem**: `undefined reference to IntegerGemmKernel<...>::multiply`

**Solution**: Regenerate instantiation files
```bash
cd python/
python3 generate_integer_gemm_instantiations.py
cmake --build build_v2_release --parallel
```

**Problem**: Compilation extremely slow

**Solution**: Check if accidentally regenerating with too many configs (should be 8000, not 80,000+)

### Runtime Issues

**Problem**: `Registry::get_kernel()` returns nullptr

**Cause**: Requested config not instantiated

**Solution**: Check config is in generated space:
```cpp
auto& registry = IntegerGemmKernelRegistry::instance();
std::cout << "Registry has " << registry.size() << " kernels" << std::endl;

bool has_config = registry.has_kernel("simd::AVX512VNNITag", 4, 32, 4, 2, 256, 512, 128);
std::cout << "Has config: " << has_config << std::endl;
```

**Problem**: Poor performance despite auto-tuner

**Cause**: May need to retrain ML model for new hardware

**Solution**: Collect fresh training data:
```bash
./build_v2_release/performance/v2_perf_integer_gemm_config_sweep \
  --benchmark_mode=full --output=new_training_data.csv
python3 python/train_gemm_autotuner.py --input=new_training_data.csv
```

---

## References

### Key Papers

- **BLIS**: "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (Van Zee & van de Geijn, 2015)
- **GEMM Optimization**: "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn, 2008)
- **INT8 GEMM**: "Quantizing deep convolutional networks for efficient inference" (Krishnamoorthi, 2018)

### Related Documentation

- **V2 Architecture**: `.github/instructions/llaminar-v2-architecture.instructions.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`
- **Changelog**: `changelog/2025-01-XX-integer-gemm-registry-pattern-migration.md`

### External Libraries

- **Intel MKL**: Reference GEMM implementation for validation
- **OpenBLAS**: Baseline comparison
- **AVX-512 Intrinsics Guide**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

---

**Last Updated**: November 11, 2025  
**Maintainer**: David Sanftenberg  
**Status**: Production-ready (Integer GEMM), Active Development (FP32/BF16/FP16)
