# Llaminar V2 Architecture - Operator-Free Design

*Last Updated: January 20, 2025*  
*Architecture Version: 2.0 (Greenfield Rewrite)*  
*Pipeline Architecture: PipelineBase inheritance with architecture-specific implementations*  
*Device Orchestration: Phase 4.3 Complete (Multi-device buffers, Device transfers, Device-aware execution)*

## Table of Contents

1. [Overview](#overview)
2. [Core Design Principles](#core-design-principles)
3. [Architecture Philosophy](#architecture-philosophy)
4. [Directory Structure](#directory-structure)
5. [Component Details](#component-details)
    - [PipelineConfig: Runtime Configuration](#pipelineconfig-runtime-configuration)
    - [WeightPlacementMap: Multi-Device Tensor Mapping](#weightplacementmap-multi-device-tensor-mapping)
    - [KV Cache Architecture](#kv-cache-architecture)
6. [Tensor System](#tensor-system)
7. [Kernel Interface Design](#kernel-interface-design)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Device Orchestration](#device-orchestration)
10. [Multi-GPU Design](#multi-gpu-design)
11. [IQ4_NL Implementation](#iq4_nl-implementation)
12. [Development Guidelines](#development-guidelines)
    - [Adding New Kernels](#adding-new-kernels)
    - [Quantized Tensor Strategy Pattern (IBlockDecoder)](#quantized-tensor-strategy-pattern-iblockdecoder)
    - [Adding New Pipelines](#adding-new-pipelines)
    - [Testing New Components](#testing-new-components)
13. [Migration from V1](#migration-from-v1)
14. [Future Roadmap](#future-roadmap)

---

## Overview

**Last Updated**: October 29, 2025

Llaminar V2 represents a **complete architectural rewrite** that eliminates the operator abstraction layer in favor of **direct kernel orchestration from pipelines**. This greenfield design addresses fundamental limitations of V1 while enabling true multi-GPU heterogeneity.

### Recent Enhancements (October 2025)

**Phase 1: KV Cache Semantic Clarification** ✅
- Renamed `layer_devices` → `attention_devices` for clarity
- Added `get_attention_device(layer_idx)` accessor method
- Per-layer device placement for KV cache tensors

**Phase 2: WeightPlacementMap Enhancement** ✅
- Block-level methods: `setAttentionDevice()`, `setFFNDevice()`
- MoE-level methods: `setSharedExpertDevice()`, `setLocalExpertDevice()`
- Device detection helpers: `detectAttentionDevices()`, `detectFFNDevices()`

**Phase 3: Pipeline Simplification** ✅
- `PipelineConfig` struct for runtime configuration (separate from GGUF metadata)
- `initializeInfrastructure()` method consolidates device/MPI/cache setup
- 93% code reduction in pipeline constructors (~15 lines → 1 line)

**Combined Impact**:
- **24/24 tests passing** (KVCache: 8/8, WeightPlacementMap: 9/9, DeviceDetection: 7/7, Factory: 12/12)
- **Simplified pipeline pattern**: 1-line initialization replaces 15-line boilerplate
- **Runtime configurability**: `--ctx-size` command-line override for context size
- **Enhanced multi-device support**: Block-level and MoE-level device placement

### Key Differences from V1

| Aspect | V1 (Operator-Based) | V2 (Kernel-Centric) |
|--------|---------------------|---------------------|
| **Abstraction** | Heavy operator layer (MPILinearOperator, MPIAttentionOperator, etc.) | Direct kernel calls from pipelines |
| **Device Model** | Per-rank MPI slicing | Per-tensor device affinity |
| **Partitioning** | Static pre-computed slices | Runtime pipeline-driven strategies |
| **Configuration** | Hardcoded parameters (max_seq_len=2048) | Runtime PipelineConfig (--ctx-size override) |
| **Initialization** | Duplicated 15-line setup per pipeline | Single `initializeInfrastructure()` call |
| **Code Complexity** | ~15,000 lines of operators | ~3,000 lines of kernels + pipelines |
| **Multi-GPU** | Single-backend per run | Heterogeneous (CUDA + ROCm + Vulkan) |
| **Extensibility** | New ops require operator boilerplate | Direct kernel registration |

### V2 Motivations

**Problems with V1 Operators**:
- ❌ Abstraction overhead (operator execute() → kernel execute())
- ❌ MPI slicing baked into operators (not flexible)
- ❌ Difficult to support multi-GPU (single backend assumption)
- ❌ Hardcoded configuration (context size, threading)
- ❌ Duplicated initialization logic across pipelines
- ❌ Large codebase (~15K lines of operator boilerplate)
- ❌ Hard to optimize end-to-end (kernel boundaries hide opportunities)

**V2 Solutions**:
- ✅ **Direct orchestration**: Pipelines call kernels directly
- ✅ **Per-tensor devices**: Runtime device selection per tensor
- ✅ **Heterogeneous backends**: Mix CUDA, ROCm, Vulkan in single run
- ✅ **Runtime configuration**: PipelineConfig separates params from model metadata
- ✅ **Simplified initialization**: Base class handles generic setup (device/MPI/cache)
- ✅ **Minimal code**: ~80% reduction via eliminated abstractions
- ✅ **Optimizable**: Pipelines own full execution flow

---

## Core Design Principles

### 1. **Operator-Free Architecture**

**V1 Pattern** (Eliminated):
```cpp
// Heavy abstraction layer
class MPILinearOperator : public MPIKernelBase {
    bool execute(inputs, outputs) override {
        // MPI slicing logic
        // Weight distribution
        // Call actual kernel
        return linear_kernel_->execute(...);
    }
};
```

**V2 Pattern** (Direct):
```cpp
// Pipeline directly orchestrates kernels
class Qwen2Pipeline {
    bool forward(...) {
        // Direct kernel calls with runtime device selection
        auto device = selectDevice(tensor_size, available_gpus);
        auto kernel = device->getGemmKernel();
        kernel->execute(input, weight, output);
    }
};
```

### 2. **Per-Tensor Device Affinity**

**V1**: MPI ranks own slices → single backend per rank  
**V2**: Tensors own device placement → heterogeneous execution

```cpp
// V2 tensor with device affinity
class TensorBase {
    virtual DeviceId device() const = 0;  // Where tensor lives
    virtual void* data() = 0;             // Device memory pointer
    virtual DataLayout layout() const = 0; // Memory layout
};

// Runtime placement decision
auto device = DeviceManager::selectOptimal(
    tensor_size,
    operation_type,
    available_devices
);
```

### 3. **Runtime Partitioning Strategy**

**V1**: Pre-computed MPI slices in operators  
**V2**: Pipelines own partitioning decisions

```cpp
class QwenPipeline {
private:
    MPIContext mpi_ctx_;  // Pipeline owns MPI coordination
    
    PartitionStrategy selectStrategy(int seq_len, int hidden_dim) {
        if (seq_len < 512) return PartitionStrategy::REPLICATED;
        if (mpi_ctx_.size == 1) return PartitionStrategy::LOCAL;
        return PartitionStrategy::TENSOR_PARALLEL;
    }
};
```

### 4. **Clean Separation of Concerns**

| Component | Responsibility | What It Does NOT Do |
|-----------|----------------|---------------------|
| **Tensors** | Data storage, layout, device placement | Computation, MPI communication |
| **Kernels** | Core computation (GEMM, attention, etc.) | Data movement, partitioning |
| **Pipelines** | Orchestration, partitioning, data flow | Low-level compute primitives |
| **DeviceManager** | Device enumeration, allocation | Kernel implementation |

---

## Architecture Philosophy

### Elimination of Operator Abstraction

**Why Operators Were Removed**:

1. **Indirection Overhead**: Every operation required 2 virtual calls (operator → kernel)
2. **Rigid MPI Model**: Operators assumed per-rank slicing (incompatible with multi-GPU)
3. **Code Duplication**: Each operator reimplemented slicing, gather, scatter logic
4. **Hidden Optimization Barriers**: Abstraction boundaries prevented kernel fusion

**V2 Direct Orchestration Benefits**:

1. **Single Virtual Call**: Pipeline → kernel (eliminate operator layer)
2. **Flexible Partitioning**: Runtime strategies owned by pipeline
3. **Code Reduction**: 12,000 lines of operator boilerplate eliminated
4. **Fusion Opportunities**: Pipelines see full computation graph

### Kernel-Centric Design

**Core Concept**: Kernels are **pure computation** with well-defined interfaces.

```cpp
// Kernel interface (no MPI, no device management)
class ITensorGemm {
public:
    virtual bool execute(
        const TensorBase* A,      // Input tensor
        const TensorBase* B,      // Weight tensor  
        TensorBase* C,            // Output tensor
        GemmParams params         // Operation parameters
    ) = 0;
};

// Concrete implementations
class OpenBLASGemm : public ITensorGemm { /* CPU implementation */ };
class CUDAGemm : public ITensorGemm { /* GPU implementation */ };
class ROCmGemm : public ITensorGemm { /* AMD GPU implementation */ };
```

**Key Properties**:
- ✅ **No MPI**: Kernels operate on local tensors only
- ✅ **No Device Selection**: Caller provides device-compatible tensors
- ✅ **Pure Computation**: Focus on algorithmic efficiency
- ✅ **Testable**: Unit testable without MPI environment

### Multi-GPU Heterogeneity

**V1 Limitation**: Single backend per MPI rank (all CUDA or all OpenBLAS)

**V2 Capability**: Mix backends in single execution:

```cpp
// Example: Use CUDA for prefill, ROCm for attention, Vulkan for FFN
auto prefill_device = DeviceManager::getCudaDevice(0);
auto attn_device = DeviceManager::getRocmDevice(1);
auto ffn_device = DeviceManager::getVulkanDevice(2);

// Different kernels from different backends
auto cuda_gemm = prefill_device->getGemmKernel();
auto rocm_attn = attn_device->getAttentionKernel();
auto vulkan_ffn = ffn_device->getFFNKernel();
```

---

## Directory Structure

```
src/v2/
├── tensors/              # Tensor types and quantization utilities
│   ├── QuantTypes.h      # Quantization type enums (69 lines)
│   ├── FP16Utils.h       # FP16 conversion utilities (105 lines)
│   ├── IQQuantTables.h   # IQ4_NL lookup tables (42 lines)
│   ├── IQ4_NLTensor.h    # IQ4_NL tensor implementation (1708 lines)
│   ├── TensorBase.h      # Base tensor interface (150 lines)
│   └── TensorKernels.h   # Kernel interfaces (ITensorGemm, etc., 210 lines)
│
├── utils/                # General utilities
│   ├── MPIContext.h      # MPI coordination (153 lines)
│   ├── CPUFeatures.h     # CPU feature detection (130 lines)
│   ├── DebugEnv.h        # Environment configuration (140 lines)
│   └── SIMDHelpers.h     # SIMD conversion helpers (210 lines with BF16)
│
├── backends/             # Device management and compute contexts
│   ├── ComputeBackend.h  # Device manager + compute contexts (280 lines)
│   ├── CudaBackend.h     # CUDA backend (future)
│   ├── RocmBackend.h     # ROCm backend (future)
│   └── VulkanBackend.h   # Vulkan backend (future)
│
├── loaders/              # Model loading and weight placement
│   ├── ArgParser.h       # CLI argument parser (150 lines)
│   ├── ArgParser.cpp     # Argument parsing implementation (350 lines)
│   ├── ModelLoader.h     # GGUF model loading (200 lines)
│   ├── ModelLoader.cpp   # Model loader implementation (800 lines)
│   ├── DeviceOrchestrator.h   # Device placement orchestration (250 lines)
│   ├── DeviceOrchestrator.cpp # Placement strategies (650 lines)
│   ├── WeightPlacementMap.h   # Weight→device mapping (120 lines)
│   └── WeightPlacementMap.cpp # Placement map implementation (180 lines)
│
├── kernels/              # Kernel implementations
│   ├── CpuGemmKernel.cpp # OpenBLAS GEMM wrapper (future)
│   ├── CudaGemmKernel.cu # CUDA GEMM (future)
│   └── RocmGemmKernel.cpp # ROCm GEMM (future)
│
├── pipelines/            # Transformer pipelines
│   ├── PipelineBase.h    # Base pipeline interface (125 lines)
│   ├── PipelineBase.cpp  # Base implementation (30 lines)
│   └── qwen/             # Qwen-specific pipeline
│       ├── Qwen2Pipeline.h    # Qwen 2.x pipeline (104 lines)
│       └── Qwen2Pipeline.cpp  # Implementation (306 lines)
│
├── tools/                # Benchmarks and utilities
│   └── benchmark/        # Performance benchmarks (future)
│
├── Main.cpp              # Entry point (future)
├── CMakeLists.txt        # Build configuration (future)
└── README.md             # V2 documentation
```

**File Organization Principles**:

1. **Tensor utilities in `tensors/`**: QuantTypes, FP16Utils, IQQuantTables (co-located with tensor types)
2. **General utilities in `utils/`**: MPIContext, CPUFeatures, DebugEnv, SIMDHelpers
3. **Backend-specific code in `backends/`**: Device management, not kernels
4. **Kernel implementations in `kernels/`**: Pure computation, no device management
5. **Pipelines in `pipelines/`**: High-level orchestration

---

## Component Details

### 5.1 PipelineConfig: Runtime Configuration

**Status**: ✅ **Complete** (October 2025)

**Purpose**: Separates runtime configuration from model architecture metadata (GGUF)

**File**: `src/v2/pipelines/PipelineConfig.h` (93 lines)

```cpp
/**
 * @brief Runtime pipeline configuration (separate from GGUF model metadata)
 * 
 * This struct contains runtime parameters that control pipeline behavior
 * but are independent of the model architecture. Separating these from
 * GGUF-derived parameters allows:
 *   - Command-line overrides (--ctx-size, --threads, --batch-size)
 *   - Dynamic reconfiguration without reloading model
 *   - Testing with different execution parameters
 */
struct PipelineConfig {
    /**
     * @brief Maximum sequence length for KV cache and activation buffers
     * 
     * Determines memory allocation for KV cache layers and intermediate
     * activations. Unlike GGUF's max_seq_len (training parameter), this
     * controls inference-time buffer sizes.
     * 
     * Default: 2048 tokens
     * Override: --ctx-size <N> or -c <N>
     */
    int max_seq_len = 2048;
    
    /**
     * @brief OpenMP thread count for CPU kernels
     * 
     * Controls parallelism for CPU GEMM, attention, and layer norm.
     * -1 = auto-detect (use OMP_NUM_THREADS or hardware concurrency)
     * 
     * Default: -1 (auto)
     * Override: -t <N>
     */
    int n_threads = -1;
    
    /**
     * @brief Batch size for multi-sequence processing (future)
     * 
     * Reserved for batch inference support. Currently unused (single-sequence).
     * 
     * Default: 1
     */
    int batch_size = 1;
    
    /**
     * @brief Enable memory-mapped file access for weights
     * 
     * When true, uses mmap() for GGUF weight loading (zero-copy).
     * When false, reads entire file into memory.
     * 
     * Default: true
     */
    bool use_mmap = true;
    
    /**
     * @brief Random seed for sampling (future)
     * 
     * -1 = time-based seed
     * ≥0 = fixed seed for reproducibility
     * 
     * Default: -1 (random)
     */
    int seed = -1;
    
    // Convenience constructor for common case
    explicit PipelineConfig(int max_seq_len_) : max_seq_len(max_seq_len_) {}
    
    // Default constructor
    PipelineConfig() = default;
};
```

**Design Rationale**:

**Why Separate from GGUF?**
- GGUF contains **architecture metadata**: `n_layers`, `n_heads`, `d_model` (immutable)
- PipelineConfig contains **runtime parameters**: buffer sizes, threading, sampling (mutable)
- Separation allows changing context size without modifying model file

**Configuration Flow**:
```
Command Line (--ctx-size 4096, -t 16)
    ↓
ArgParser → ArgContext.max_seq_len, ArgContext.n_threads
    ↓
Main.cpp → PipelineConfig{max_seq_len = 4096, n_threads = 16}
    ↓
PipelineFactory::create(..., config)
    ↓
PipelineBase(config) → config_ member
    ↓
Derived::initializeInfrastructure() → uses config_.max_seq_len
```

**Usage Example**:
```cpp
// In Main.cpp
PipelineConfig pipeline_config;
pipeline_config.max_seq_len = args.max_seq_len;  // From --ctx-size
pipeline_config.n_threads = args.n_threads;      // From -t
pipeline_config.batch_size = args.batch_size;    // From -b

// Create pipeline with config
auto pipeline = factory.create(
    args.architecture,
    loader,
    device_mgr,
    mpi_ctx,
    pipeline_config  // Runtime parameters
);

// In PipelineBase constructor
PipelineBase::PipelineBase(
    ModelLoader& loader,
    DeviceManager& device_mgr,
    MPIContext& mpi_ctx,
    const PipelineConfig& config
)
    : loader_(loader),
      device_mgr_(device_mgr),
      mpi_ctx_(mpi_ctx),
      config_(config)  // Store for use in initialization
{
    LOG_INFO("Pipeline runtime config: max_seq_len=" << config_.max_seq_len
             << ", n_threads=" << config_.n_threads);
}

// Accessing config in derived classes
void Qwen2Pipeline::initializeInfrastructure() {
    int max_seq_len = config_.max_seq_len;  // Read from config
    initializeDeviceInfrastructure(max_seq_len);
    configureMPIStrategy();
    initializeKVCache(max_seq_len);
}
```

**Command-Line Interface**:
```bash
# Default context size (2048)
./llaminar2 -m model.gguf

# Custom context size (4096 tokens)
./llaminar2 -m model.gguf --ctx-size 4096
./llaminar2 -m model.gguf -c 4096  # Short form

# Combined runtime configuration
./llaminar2 -m model.gguf -c 8192 -t 32 -b 4
#   Context: 8192 tokens
#   Threads: 32 CPU cores
#   Batch: 4 sequences (future)
```

**Future Extensions**:
- `float rope_freq_base`: Override RoPE frequency base (model-specific tuning)
- `int n_gpu_layers`: Partial offloading (memory-aware placement)
- `std::string lora_path`: LoRA adapter path for fine-tuning
- `SamplingParams sampling`: Temperature, top-p, top-k configuration

---

### 5.2 WeightPlacementMap: Multi-Device Tensor Mapping

**Status**: ✅ **Phase 2 Complete** (October 2025 - Block/MoE Methods)

**Purpose**: Maps individual model weights to devices for heterogeneous execution

**File**: `src/v2/loaders/WeightPlacementMap.h/cpp` (~300 lines)

**Core Abstraction**:
```cpp
class WeightPlacementMap {
public:
    // Tensor-level device placement
    void setDevice(const std::string& tensor_name, DeviceId device);
    DeviceId getDevice(const std::string& tensor_name) const;
    
    // Block-level device placement (NEW - Phase 2)
    void setAttentionDevice(int layer_idx, DeviceId device);
    void setFFNDevice(int layer_idx, DeviceId device);
    DeviceId getAttentionDevice(int layer_idx) const;
    DeviceId getFFNDevice(int layer_idx) const;
    
    // MoE-level device placement (NEW - Phase 2)
    void setSharedExpertDevice(int layer_idx, DeviceId device);
    void setLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device);
    DeviceId getSharedExpertDevice(int layer_idx) const;
    DeviceId getLocalExpertDevice(int layer_idx, int expert_idx) const;
    
    // Device detection helpers (NEW - Phase 3)
    std::vector<DeviceId> detectAttentionDevices(int n_layers) const;
    std::vector<DeviceId> detectFFNDevices(int n_layers) const;
    
private:
    std::unordered_map<std::string, DeviceId> tensor_to_device_;
};
```

**Key Features**:

**1. Tensor-Level Granularity**
```cpp
WeightPlacementMap map;
map.setDevice("model.layers.0.attn.wq", DeviceId::GPU_0);
map.setDevice("model.layers.0.attn.wk", DeviceId::GPU_0);
map.setDevice("model.layers.1.attn.wq", DeviceId::CPU);

DeviceId device = map.getDevice("model.layers.0.attn.wq");  // GPU_0
```

**2. Block-Level Methods** (Phase 2)
```cpp
// Assign entire attention block to GPU
map.setAttentionDevice(layer_idx, DeviceId::GPU_0);
// Internally sets devices for:
//   - blk.{layer_idx}.attn_q.weight
//   - blk.{layer_idx}.attn_k.weight
//   - blk.{layer_idx}.attn_v.weight
//   - blk.{layer_idx}.attn_output.weight

// Assign entire FFN block to CPU
map.setFFNDevice(layer_idx, DeviceId::CPU);
// Internally sets devices for:
//   - blk.{layer_idx}.ffn_gate.weight
//   - blk.{layer_idx}.ffn_up.weight
//   - blk.{layer_idx}.ffn_down.weight
```

**3. MoE-Level Methods** (Phase 2)
```cpp
// MoE architecture: Shared experts on GPU, local experts on CPU
for (int layer = 0; layer < n_layers; ++layer) {
    map.setSharedExpertDevice(layer, DeviceId::GPU_0);  // Frequent access
    
    for (int expert = 0; expert < n_experts_per_layer; ++expert) {
        map.setLocalExpertDevice(layer, expert, DeviceId::CPU);  // Sparse activation
    }
}
```

**4. Device Detection Helpers** (Phase 3)
```cpp
// Detect which devices are used for attention across all layers
std::vector<DeviceId> attn_devices = map.detectAttentionDevices(n_layers);
// Example result: {GPU_0, GPU_0, GPU_1, GPU_1, CPU, CPU, ...}
//   Layers 0-1 → GPU_0
//   Layers 2-3 → GPU_1
//   Layers 4+ → CPU

// Detect which devices are used for FFN
std::vector<DeviceId> ffn_devices = map.detectFFNDevices(n_layers);
// Used for KV cache placement (match attention device per layer)
```

**Implementation Details**:

**Block-Level Method Implementation** (~40 lines):
```cpp
void WeightPlacementMap::setAttentionDevice(int layer_idx, DeviceId device) {
    std::string prefix = "blk." + std::to_string(layer_idx) + ".attn_";
    tensor_to_device_[prefix + "q.weight"] = device;
    tensor_to_device_[prefix + "k.weight"] = device;
    tensor_to_device_[prefix + "v.weight"] = device;
    tensor_to_device_[prefix + "output.weight"] = device;
}

DeviceId WeightPlacementMap::getAttentionDevice(int layer_idx) const {
    std::string key = "blk." + std::to_string(layer_idx) + ".attn_q.weight";
    return getDevice(key);  // Query projection as representative
}

void WeightPlacementMap::setFFNDevice(int layer_idx, DeviceId device) {
    std::string prefix = "blk." + std::to_string(layer_idx) + ".ffn_";
    tensor_to_device_[prefix + "gate.weight"] = device;
    tensor_to_device_[prefix + "up.weight"] = device;
    tensor_to_device_[prefix + "down.weight"] = device;
}

DeviceId WeightPlacementMap::getFFNDevice(int layer_idx) const {
    std::string key = "blk." + std::to_string(layer_idx) + ".ffn_gate.weight";
    return getDevice(key);
}
```

**MoE Method Implementation** (~30 lines):
```cpp
void WeightPlacementMap::setSharedExpertDevice(int layer_idx, DeviceId device) {
    std::string key = "blk." + std::to_string(layer_idx) + ".ffn.shared_expert.weight";
    tensor_to_device_[key] = device;
}

void WeightPlacementMap::setLocalExpertDevice(int layer_idx, int expert_idx, DeviceId device) {
    std::string key = "blk." + std::to_string(layer_idx) 
                    + ".ffn.experts." + std::to_string(expert_idx) + ".weight";
    tensor_to_device_[key] = device;
}

DeviceId WeightPlacementMap::getSharedExpertDevice(int layer_idx) const {
    std::string key = "blk." + std::to_string(layer_idx) + ".ffn.shared_expert.weight";
    return getDevice(key);
}

DeviceId WeightPlacementMap::getLocalExpertDevice(int layer_idx, int expert_idx) const {
    std::string key = "blk." + std::to_string(layer_idx) 
                    + ".ffn.experts." + std::to_string(expert_idx) + ".weight";
    return getDevice(key);
}
```

**Device Detection Implementation** (~50 lines):
```cpp
std::vector<DeviceId> WeightPlacementMap::detectAttentionDevices(int n_layers) const {
    std::vector<DeviceId> devices;
    devices.reserve(n_layers);
    
    for (int i = 0; i < n_layers; ++i) {
        devices.push_back(getAttentionDevice(i));
    }
    return devices;
}

std::vector<DeviceId> WeightPlacementMap::detectFFNDevices(int n_layers) const {
    std::vector<DeviceId> devices;
    devices.reserve(n_layers);
    
    for (int i = 0; i < n_layers; ++i) {
        devices.push_back(getFFNDevice(i));
    }
    return devices;
}
```

**Testing**:
- **Unit Tests**: `tests/v2/unit/loaders/Test__WeightPlacementMap.cpp` (9/9 tests passing)
- **Coverage**:
  - ✅ Block-level methods (`setAttentionDevice`, `setFFNDevice`)
  - ✅ MoE methods (`setSharedExpertDevice`, `setLocalExpertDevice`)
  - ✅ Device detection (`detectAttentionDevices`, `detectFFNDevices`)
  - ✅ Edge cases (invalid layer index, missing tensors)

**Use Cases**:

**Case 1: Hybrid CPU+GPU Execution**
```cpp
// First 12 layers on GPU, rest on CPU
for (int i = 0; i < 12; ++i) {
    map.setAttentionDevice(i, DeviceId::GPU_0);
    map.setFFNDevice(i, DeviceId::GPU_0);
}
for (int i = 12; i < n_layers; ++i) {
    map.setAttentionDevice(i, DeviceId::CPU);
    map.setFFNDevice(i, DeviceId::CPU);
}
```

**Case 2: Multi-GPU Pipeline Parallelism**
```cpp
// Split 32-layer model across 2 GPUs
for (int i = 0; i < 16; ++i) {
    map.setAttentionDevice(i, DeviceId::GPU_0);
    map.setFFNDevice(i, DeviceId::GPU_0);
}
for (int i = 16; i < 32; ++i) {
    map.setAttentionDevice(i, DeviceId::GPU_1);
    map.setFFNDevice(i, DeviceId::GPU_1);
}
```

**Case 3: MoE Optimization (Mixtral)**
```cpp
// 8 experts per layer, 2 active per token
for (int layer = 0; layer < n_layers; ++layer) {
    // Shared expert: used every token → GPU
    map.setSharedExpertDevice(layer, DeviceId::GPU_0);
    
    // Local experts: sparse activation → CPU (save VRAM)
    for (int expert = 0; expert < 8; ++expert) {
        map.setLocalExpertDevice(layer, expert, DeviceId::CPU);
    }
}
```

---

### 5.3 KV Cache Architecture

**Status**: ✅ **Phase 1 Complete** (October 2025 - Semantic Clarification)

**Purpose**: Per-layer device placement for KV cache tensors in autoregressive decode

**Files**:
- `src/v2/pipelines/PipelineBase.h/cpp` (~900 lines)
- `tests/v2/unit/pipelines/Test__KVCache.cpp` (8/8 tests passing)

**Core Concept**:

KV cache stores past key/value projections for efficient autoregressive generation:
```
Prefill: Process full prompt → Generate K/V for all positions
Decode:  Process 1 new token → Append to existing K/V cache
```

**Device Placement Strategy**:
```cpp
class PipelineBase {
protected:
    // Per-layer device placement (NEW - Phase 1: renamed from layer_devices)
    std::vector<DeviceId> attention_devices_;  // Semantic clarity: "where KV cache lives"
    
    // KV cache storage (per layer)
    std::vector<std::shared_ptr<TensorBase>> k_cache_;  // [n_layers][max_seq_len, n_kv_heads, head_dim]
    std::vector<std::shared_ptr<TensorBase>> v_cache_;  // [n_layers][max_seq_len, n_kv_heads, head_dim]
    
    int cache_position_ = 0;  // Current insertion position in cache
    
    // Initialization (called from initializeInfrastructure)
    void initializeKVCache(int max_seq_len);
    
    // Access (NEW - Phase 1)
    DeviceId get_attention_device(int layer_idx) const {
        return attention_devices_.at(layer_idx);
    }
};
```

**Phase 1 Changes** (Semantic Clarification):

**Before** (Ambiguous Naming):
```cpp
std::vector<DeviceId> layer_devices_;  // What does "layer" mean?
```

**After** (Clear Intent):
```cpp
std::vector<DeviceId> attention_devices_;  // "Where KV cache lives for this layer"
```

**Rationale**:
- "layer_devices" was ambiguous (entire layer? just weights? activations?)
- "attention_devices" clarifies: **device placement for KV cache** (attention mechanism state)
- FFN has no cache → no device tracking needed (stateless computation)

**Initialization**:
```cpp
void PipelineBase::initializeKVCache(int max_seq_len) {
    k_cache_.resize(n_layers_);
    v_cache_.resize(n_layers_);
    
    for (int i = 0; i < n_layers_; ++i) {
        DeviceId device = attention_devices_[i];  // Use detected device
        
        // Allocate KV cache on device where attention weights live
        k_cache_[i] = TensorFactory::create(
            {max_seq_len, n_kv_heads_, head_dim_},  // GQA-aware shape
            device,
            DataType::FP32  // Or BF16 for memory optimization
        );
        v_cache_[i] = TensorFactory::create(
            {max_seq_len, n_kv_heads_, head_dim_},
            device,
            DataType::FP32
        );
        
        k_cache_[i]->zero();  // Initialize to zeros
        v_cache_[i]->zero();
    }
    
    cache_position_ = 0;  // Start at position 0
    LOG_INFO("KV cache initialized: " << n_layers_ << " layers, "
             << "max_seq_len=" << max_seq_len);
}
```

**Device Detection** (Phase 3 Integration):
```cpp
void PipelineBase::initializeDeviceInfrastructure(int max_seq_len) {
    // Detect devices from weight placement
    attention_devices_ = weight_placement_.detectAttentionDevices(n_layers_);
    
    // Log placement summary
    std::unordered_map<DeviceId, int> device_counts;
    for (DeviceId device : attention_devices_) {
        device_counts[device]++;
    }
    
    LOG_INFO("Attention device placement:");
    for (const auto& [device, count] : device_counts) {
        LOG_INFO("  " << deviceIdToString(device) << ": " << count << " layers");
    }
}
```

**Usage in Attention**:
```cpp
bool Qwen2Pipeline::attention_block(int layer_idx, TensorBase* input, TensorBase* output) {
    // Get device for this layer's attention
    DeviceId device = get_attention_device(layer_idx);
    
    // Ensure input is on correct device
    auto input_on_device = transferToDevice(input, device);
    
    // Compute Q/K/V projections
    auto Q = computeProjection(input_on_device, weights_.wq[layer_idx]);
    auto K = computeProjection(input_on_device, weights_.wk[layer_idx]);
    auto V = computeProjection(input_on_device, weights_.wv[layer_idx]);
    
    // Append to KV cache (on same device as attention)
    appendToCache(k_cache_[layer_idx], K, cache_position_);
    appendToCache(v_cache_[layer_idx], V, cache_position_);
    
    // Compute attention using full cache
    auto attn_output = computeAttention(Q, k_cache_[layer_idx], v_cache_[layer_idx]);
    
    return true;
}
```

**Testing**:
- **Unit Tests**: `tests/v2/unit/pipelines/Test__KVCache.cpp` (8/8 tests passing)
- **Coverage**:
  - ✅ Cache initialization with correct shapes
  - ✅ Device placement matches weight placement
  - ✅ Zero initialization
  - ✅ Per-layer device retrieval (`get_attention_device`)
  - ✅ Multi-device scenarios (hybrid CPU+GPU)
  - ✅ Edge cases (empty cache, out-of-bounds access)

**Memory Characteristics**:
```cpp
// Per-layer KV cache size (FP32)
size_t cache_size_per_layer = max_seq_len * n_kv_heads * head_dim * sizeof(float) * 2;  // K + V

// Example: Qwen 2.5 7B, max_seq_len=4096
// n_kv_heads=8, head_dim=128
size_t per_layer = 4096 * 8 * 128 * 4 * 2 = 33,554,432 bytes = 32 MB
size_t total_cache = per_layer * 28 layers = 896 MB

// With BF16 optimization: 896 MB / 2 = 448 MB
```

**Future Extensions** (Phase 4):
- ✅ BF16 cache storage (2× memory reduction)
- 🔄 Dynamic cache resizing (grow beyond initial max_seq_len)
- 🔄 Multi-sequence batching (cache per sequence)
- 🔄 Speculative decoding (tentative cache branches)

---

### 6.1 Tensor System (`tensors/`)

#### TensorBase Interface

**Purpose**: Abstract tensor storage with device placement and layout metadata

```cpp
// src/v2/tensors/TensorBase.h
class TensorBase {
public:
    virtual ~TensorBase() = default;
    
    // Core properties
    virtual std::vector<size_t> shape() const = 0;
    virtual DataType dtype() const = 0;
    virtual DataLayout layout() const = 0;
    
    // Device placement
    virtual DeviceId device() const = 0;
    virtual void* data() = 0;
    virtual const void* data() const = 0;
    
    // Operations
    virtual size_t size_bytes() const = 0;
    virtual void zero() = 0;
};

enum class DataType { FP32, FP16, BF16, INT8, IQ4_NL, IQ6_K };
enum class DataLayout { RowMajor, ColumnMajor, TileOptimized };
enum class DeviceId { CPU, CUDA_0, CUDA_1, ROCM_0, VULKAN_0 };
```

**Key Design Choices**:

- **No Virtual `at()`**: Direct memory access via `data()` pointer (avoid virtual calls in hot loops)
- **Device-Aware**: Every tensor knows where it lives
- **Layout-Explicit**: No implicit transpose assumptions
- **Type-Safe**: Strongly typed data types and layouts

#### IQ4_NL Quantized Tensor

**File**: `src/v2/tensors/IQ4_NLTensor.h` (1708 lines)

**Purpose**: 4-bit quantized tensor with fused dequant+GEMM kernel

**Key Features**:
- **64×32 Optimal Tiles**: +41% FP32 performance from sweep optimization
- **+26% BF16 Speedup**: Specialized BF16 dequant path
- **Fused Dequant+GEMM**: Implements `ITensorGemm` directly on quantized data
- **AVX512/AVX2 Paths**: SIMD-optimized decode
- **L1 Cache Tuning**: Conservative 32KB constant (no dynamic detection)

**Performance** (vs reference):
- FP32 GEMM: **+41% speedup** (64×32 tiles vs 32×32)
- BF16 GEMM: **+26% speedup** (16-bit pathway)

**Usage**:
```cpp
auto quant_tensor = std::make_shared<IQ4_NLTensor>(shape, device);
quant_tensor->populateQuantized(fp32_weights);  // Quantize on load

// Direct fused GEMM (no dequant buffer)
auto gemm = quant_tensor->getGemmKernel();
gemm->execute(input, quant_tensor, output, params);
```

#### Quantization Types (`tensors/QuantTypes.h`)

```cpp
enum class QuantizationType {
    FP32, FP16, BF16, INT8,
    IQ4_NL, IQ4_XS,  // 4-bit importance-weighted
    IQ6_K,           // 6-bit K-quants
    Q4_0, Q6_K, Q8_0 // llama.cpp formats
};
```

**Helper Functions**:
- `quantizationBytesPerElement(QuantizationType)`: Memory footprint
- `quantizationName(QuantizationType)`: Human-readable name
- `parseQuantizationType(string)`: CLI parsing

#### FP16 Utilities (`tensors/FP16Utils.h`)

**Purpose**: IEEE 754 half-precision conversion

```cpp
namespace fp16 {
    uint16_t fp32_to_fp16(float fp32);
    float fp16_to_fp32(uint16_t fp16);
    void convert_array_fp32_to_fp16(const float* in, uint16_t* out, size_t count);
}
```

**Features**:
- ✅ Correct rounding (round-to-nearest-even)
- ✅ Denormal handling
- ✅ Inf/NaN propagation
- ✅ Vectorized conversion (future: AVX512 path)

#### IQ4_NL Quantization Tables (`tensors/IQQuantTables.h`)

```cpp
namespace iq4nl {
    extern const float kvalues_iq4nl[16];  // [-0.1250, ..., 1.3125]
}
```

**Purpose**: Lookup table for IQ4_NL dequantization (importance-weighted values)

---

### 6.2 Utilities (`utils/`)

#### MPIContext

**File**: `src/v2/utils/MPIContext.h` (153 lines)

**Purpose**: MPI coordination state (no singleton, pipeline-owned)

```cpp
struct MPIContext {
    int rank;
    int world_size;
    MPI_Comm comm;
    
    bool is_root() const { return rank == 0; }
    
    // Factory methods
    static MPIContext fromMPI();
    static MPIContext singleRank();
};
```

**Key Difference from V1**:
- **No Singleton**: Each pipeline owns its context
- **Immutable**: Set at construction, never changes
- **Lightweight**: Just 3 fields (8 bytes)

#### CPUFeatures

**File**: `src/v2/utils/CPUFeatures.h` (130 lines)

**Purpose**: CPU SIMD feature detection (free functions, no singleton)

```cpp
namespace cpufeatures {
    bool cpu_supports_avx512();
    bool cpu_supports_avx2();
    bool cpu_supports_avx();
    bool cpu_supports_sse41();
    
    // No L1 cache detection in V2 (use conservative constant)
}
```

**Key Changes from V1**:
- **No Singleton**: Static free functions
- **No L1 Cache**: Removed runtime detection (use constant)
- **Simpler**: Just SIMD flags

#### DebugEnv

**File**: `src/v2/utils/DebugEnv.h` (140 lines)

**Purpose**: Centralized environment configuration (same pattern as V1)

```cpp
struct DebugEnvSnapshot {
    struct QuantConfig {
        bool bf16_gemm = false;
        bool iq4nl_fused = true;
    } quant;
    
    struct DeviceConfig {
        bool prefer_cuda = false;
        int cuda_device_id = 0;
    } device;
};

const DebugEnvSnapshot& debugEnv();  // Lazy static initialization
```

#### SIMDHelpers

**File**: `src/v2/utils/SIMDHelpers.h` (210 lines with BF16)

**Purpose**: SIMD conversion helpers for quantization

```cpp
namespace simd {
    // FP16 conversions (same as V1)
    void convert_fp16_to_fp32_avx512(const uint16_t* in, float* out, size_t count);
    void convert_fp32_to_fp16_avx2(const float* in, uint16_t* out, size_t count);
    
    // BF16 conversions (NEW in V2)
    float bf16_to_fp32(uint16_t bf16);
    uint16_t fp32_to_bf16(float fp32);
    void convert_bf16_to_fp32_avx512(const uint16_t* bf16, float* fp32, size_t count);
}
```

**BF16 Support**:
- Scalar: `bf16_to_fp32()` (16-bit left shift)
- Scalar: `fp32_to_bf16()` (round-to-nearest-even)
- Vector: `convert_bf16_to_fp32_avx512()` (32 elements at a time)

---

### 6.3 Backend Management (`backends/`)

#### DeviceManager & ComputeContext

**File**: `src/v2/backends/ComputeBackend.h` (280 lines)

**Purpose**: Device enumeration, allocation, and context management

```cpp
class DeviceManager {
public:
    static DeviceManager& instance();  // Singleton
    
    // Device enumeration
    void initialize();
    std::vector<DeviceInfo> available_devices() const;
    
    // Device selection
    ComputeContext* selectDevice(
        size_t tensor_size,
        OperationType op_type,
        const std::vector<DeviceId>& candidates
    );
    
    // Specific device getters
    ComputeContext* getCudaDevice(int device_id);
    ComputeContext* getRocmDevice(int device_id);
    ComputeContext* getCPUContext();
};

class ComputeContext {
public:
    virtual DeviceId device_id() const = 0;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    
    // Kernel factories
    virtual ITensorGemm* getGemmKernel() = 0;
    virtual ITensorAttention* getAttentionKernel() = 0;
    virtual ITensorRoPE* getRoPEKernel() = 0;
};
```

**Device Selection Strategy**:

```cpp
ComputeContext* selectDevice(size_t tensor_size, OperationType op) {
    // Large operations → GPU
    if (tensor_size > 1024 * 1024 && has_gpu_) {
        if (has_cuda_) return cuda_context_.get();
        if (has_rocm_) return rocm_context_.get();
    }
    
    // Small operations → CPU (lower latency)
    return cpu_context_.get();
}
```

**Backend Selection Strategy Explained**:

The `DeviceManager` uses heuristics to select optimal devices:

1. **Size-based selection**: Large tensors (>1MB) prefer GPU, small tensors prefer CPU (lower latency)
2. **Memory-aware**: Checks available memory before allocation
3. **Round-robin**: Distributes work across multiple GPUs
4. **Automatic BLAS backend**: CPU backend selected at compile-time based on CPU vendor
   - Intel CPUs → Intel MKL (if available), fallback to OpenBLAS
   - Non-Intel CPUs → OpenBLAS
   - Selection happens via CMake `BLAS_BACKEND` option (AUTO/MKL/OPENBLAS)

**Current Implementation Status**:

| Backend | Status | Features |
|---------|--------|----------|
| CPU (OpenBLAS) | ✅ Production | FP32, BF16, kernel factories (RoPE, Softmax, RMSNorm, SwiGLU) |
| CPU (Intel MKL) | ✅ Production | FP32, BF16 (hardware on Ice Lake+), same kernels as OpenBLAS |
| CUDA | 🚧 Stub only | Header defined, not implemented |
| ROCm | 🚧 Stub only | Header defined, not implemented |
| Vulkan | 🚧 Stub only | Header defined, not implemented |

---

#### Practical ComputeBackend Usage

**1. Device Enumeration at Startup**

```cpp
// main.cpp or pipeline initialization
DeviceManager& dm = DeviceManager::instance();
dm.initialize();  // Scans for CPU, CUDA, ROCm, Vulkan devices

// Inspect available devices
for (const auto& dev : dm.devices()) {
    std::cout << backend_type_name(dev.type) << ": " << dev.name 
              << " (" << dev.free_memory_bytes / (1024*1024) << " MB free)\n";
}

// Example output:
// CPU (OpenBLAS): OpenBLAS (CPU) (62144 MB free)
// NVIDIA CUDA: NVIDIA RTX 4090 (23040 MB free)
// AMD ROCm: AMD Radeon RX 7900 XTX (20480 MB free)
```

**2. Manual Device Selection**

```cpp
// Get specific device by type
int cuda_idx = dm.find_device(ComputeBackendType::GPU_CUDA, 0);  // CUDA device 0
if (cuda_idx >= 0) {
    auto ctx = dm.create_context(cuda_idx);
    // Use ctx for memory allocation, kernel creation
}

// Or get CPU context directly
auto cpu_ctx = dm.create_context(0);  // Index 0 is always CPU
```

**3. Automatic Device Selection**

```cpp
// Let DeviceManager choose based on available memory
size_t tensor_bytes = 1024 * 1024 * 512;  // 512 MB
size_t best_device_idx = dm.select_device(tensor_bytes);
auto ctx = dm.create_context(best_device_idx);

// Heuristics:
// - If tensor_bytes > available GPU memory, fallback to CPU
// - Prefer GPU with most free memory
// - Round-robin among GPUs with similar memory
```

**4. Kernel Creation via ComputeContext**

```cpp
// CPU context provides kernel factories
auto cpu_ctx = dm.create_context(0);
ITensorRoPE* rope = cpu_ctx->get_rope_kernel();
ITensorSoftmax* softmax = cpu_ctx->get_softmax_kernel();
ITensorRMSNorm* rmsnorm = cpu_ctx->get_rmsnorm_kernel();
ITensorSwiGLU* swiglu = cpu_ctx->get_swiglu_kernel();

// Kernels are cached - subsequent calls return same instance
auto rope2 = cpu_ctx->get_rope_kernel();  // Same pointer as rope
```

**5. Memory Management via ComputeContext**

```cpp
// Allocate device memory
void* buffer = ctx->allocate(1024 * 1024);  // 1 MB

// Copy data to device
float host_data[256] = { /* ... */ };
ctx->copy_to_device(buffer, host_data, sizeof(host_data));

// Synchronize (GPU operations are async)
ctx->synchronize();

// Copy results back
float host_result[256];
ctx->copy_from_device(host_result, buffer, sizeof(host_result));

// Free device memory
ctx->free(buffer);
```

**6. Query Backend Capabilities**

```cpp
auto ctx = dm.create_context(device_idx);

// Check precision support
if (ctx->supports_bf16()) {
    // Use BF16 kernels
} else {
    // Fallback to FP32
}

// Check backend type
ComputeBackendType type = ctx->backend_type();
switch (type) {
    case ComputeBackendType::CPU_OPENBLAS:
        // OpenBLAS-specific optimizations
        break;
    case ComputeBackendType::CPU_MKL:
        // MKL can use native BF16 GEMM on Ice Lake+
        break;
    case ComputeBackendType::GPU_CUDA:
        // Use cuBLAS
        break;
    // ...
}
```

**7. Multi-Device Heterogeneous Execution (Future)**

```cpp
// Distribute work across multiple devices
auto cpu_ctx = dm.create_context(0);
int cuda_idx = dm.find_device(ComputeBackendType::GPU_CUDA, 0);
int rocm_idx = dm.find_device(ComputeBackendType::GPU_ROCM, 0);

if (cuda_idx >= 0 && rocm_idx >= 0) {
    auto cuda_ctx = dm.create_context(cuda_idx);
    auto rocm_ctx = dm.create_context(rocm_idx);
    
    // Different operations on different devices
    auto cuda_gemm = cuda_ctx->get_gemm_kernel();  // Future
    auto rocm_attn = rocm_ctx->get_attention_kernel();  // Future
    
    // Mix backends in single inference
    cuda_gemm->execute(Q_proj_input, Q_weight, Q_output);
    rocm_attn->execute(Q, K, V, attn_output);
}
```

**8. Integration with Tensors**

```cpp
// Tensors know their device placement
class FP32Tensor : public TensorBase {
    int device_idx_ = -1;  // -1 = CPU, >=0 = GPU device index
    
    std::unique_ptr<ITensorGemm> createGemm() override {
        // Get context for this tensor's device
        auto ctx = DeviceManager::instance().create_context(device_idx_);
        
        // For CPU, we have FP32GemmKernel directly
        if (device_idx_ == -1) {
            return std::make_unique<FP32GemmKernel>(this);
        }
        
        // For GPU, context would provide device-specific kernel
        return ctx->get_gemm_kernel();  // Future
    }
};
```

**9. Compile-Time Backend Selection**

The CPU BLAS backend is selected at CMake configure time:

```bash
# Automatic selection based on CPU vendor
cmake -B build_v2 -S src/v2 -DBLAS_BACKEND=AUTO

# Manual override
cmake -B build_v2 -S src/v2 -DBLAS_BACKEND=MKL      # Force Intel MKL
cmake -B build_v2 -S src/v2 -DBLAS_BACKEND=OPENBLAS # Force OpenBLAS

# CMake logic (src/v2/CMakeLists.txt):
if(BLAS_BACKEND STREQUAL "AUTO")
    execute_process(COMMAND lscpu OUTPUT_VARIABLE LSCPU_OUTPUT)
    if(LSCPU_OUTPUT MATCHES "Vendor ID:[ ]+GenuineIntel")
        # Try MKL first, fallback to OpenBLAS
        find_package(MKL)
        if(NOT MKL_FOUND)
            find_package(BLAS)  # OpenBLAS
        endif()
    else()
        find_package(BLAS)  # OpenBLAS for non-Intel CPUs
    endif()
endif()
```

This sets either `HAVE_MKL` or `HAVE_OPENBLAS` (not both - symbol conflicts).

**10. Runtime Backend Reporting**

```cpp
// CPUComputeContext reports selected backend
ComputeBackendType CPUComputeContext::backend_type() const {
#ifdef HAVE_MKL
    return ComputeBackendType::CPU_MKL;
#else
    return ComputeBackendType::CPU_OPENBLAS;
#endif
}

// Usage
auto ctx = dm.create_context(0);  // CPU
if (ctx->backend_type() == ComputeBackendType::CPU_MKL) {
    std::cout << "Using Intel MKL for BLAS operations\n";
} else {
    std::cout << "Using OpenBLAS for BLAS operations\n";
}
```

**Implementation Files**:
- `src/v2/backends/ComputeBackend.h` - Interface definitions (285 lines)
- `src/v2/backends/ComputeBackend.cpp` - Device enumeration, context management (824 lines)
- `src/v2/CMakeLists.txt` - BLAS backend selection logic (lines 70-120)

---

### 6.4 Kernel Interfaces (`tensors/TensorKernels.h`)

#### ITensorGemm

```cpp
class ITensorGemm {
public:
    virtual ~ITensorGemm() = default;
    
    virtual bool execute(
        const TensorBase* A,        // [m, k]
        const TensorBase* B,        // [k, n] or quantized
        TensorBase* C,              // [m, n]
        const GemmParams& params    // Alpha, beta, transpose flags
    ) = 0;
};

struct GemmParams {
    float alpha = 1.0f;
    float beta = 0.0f;
    bool transpose_A = false;
    bool transpose_B = false;
};
```

**Implementations**:
- `OpenBLASGemm`: CPU BLAS wrapper
- `IQ4_NLQuantizedGemm`: Fused dequant+GEMM (in `IQ4_NLTensor.h`)
- `CudaGemm`: cuBLAS wrapper (future)
- `RocmGemm`: rocBLAS wrapper (future)

#### ITensorAttention

```cpp
class ITensorAttention {
public:
    virtual bool execute(
        const TensorBase* Q,        // [batch, seq_len, d_model]
        const TensorBase* K,        // [batch, seq_len, d_model]
        const TensorBase* V,        // [batch, seq_len, d_model]
        TensorBase* output,         // [batch, seq_len, d_model]
        const AttentionParams& params
    ) = 0;
};

struct AttentionParams {
    int num_heads;
    int head_dim;
    float scale;
    bool causal_mask;
};
```

#### ITensorRoPE

```cpp
class ITensorRoPE {
public:
    virtual bool execute(
        TensorBase* tensor,         // [batch, seq_len, n_heads, head_dim]
        const RoPEParams& params
    ) = 0;
};

struct RoPEParams {
    int seq_offset;
    float theta_base;
};
```

#### ITensorSoftmax

```cpp
class ITensorSoftmax {
public:
    virtual bool execute(
        const TensorBase* input,    // [batch, seq_len, feature_dim]
        TensorBase* output,         // Same shape
        int dim                     // Softmax dimension
    ) = 0;
};
```

**Key Design**: All interfaces are **pure computation** (no MPI, no device selection)

---

### 6.5 TensorDimensions Verification

**Phase 4.1 Complete** (January 18, 2025): Runtime dimension validation infrastructure

**Problem**: Tensor shape mismatches cause silent bugs (wrong GEMM outputs, crashes, numerical errors). Need runtime validation to catch dimension errors early.

**Solution**: `TensorSpec` struct + `VALIDATE_TENSOR()` macro + helper methods

**Implementation** (Qwen2Pipeline):

```cpp
// Specification struct
struct TensorSpec {
    std::vector<size_t> expected_shape;
    std::string name;
    
    bool matches(const TensorBase* tensor) const {
        return tensor && tensor->shape() == expected_shape;
    }
    
    std::string mismatch_message(const TensorBase* tensor) const {
        std::ostringstream oss;
        oss << "TensorSpec mismatch for '" << name << "': expected [";
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            oss << expected_shape[i];
            if (i < expected_shape.size() - 1) oss << ", ";
        }
        oss << "], got [";
        auto actual = tensor ? tensor->shape() : std::vector<size_t>{};
        for (size_t i = 0; i < actual.size(); ++i) {
            oss << actual[i];
            if (i < actual.size() - 1) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }
};

// Validation macro
#define VALIDATE_TENSOR(tensor, spec) \
    if (!(spec).matches(tensor)) { \
        LOG_ERROR((spec).mismatch_message(tensor)); \
        return false; \
    }
```

**Helper Methods** (Qwen2Pipeline):

```cpp
class Qwen2Pipeline : public PipelineBase {
private:
    // Expected dimension specifications
    TensorSpec spec_hidden() const {
        return TensorSpec{{seq_len_, d_model_}, "hidden_state"};
    }
    
    TensorSpec spec_q() const {
        return TensorSpec{{seq_len_, d_model_}, "query_projection"};
    }
    
    TensorSpec spec_kv() const {
        return TensorSpec{{seq_len_, n_kv_heads_ * head_dim_}, "key_or_value_projection"};
    }
    
    TensorSpec spec_ffn_gate_up() const {
        return TensorSpec{{seq_len_, d_ff_}, "ffn_gate_or_up"};
    }
    
    TensorSpec spec_ffn_intermediate() const {
        return TensorSpec{{seq_len_, d_ff_}, "ffn_intermediate"};
    }
};
```

**Usage in Transformer Blocks**:

```cpp
bool Qwen2Pipeline::attention_block(int layer_idx) {
    // ... get buffers, transfer activation ...
    
    // Validate Q/K/V projection outputs
    auto& buffers = getBuffersForDevice(attn_device);
    VALIDATE_TENSOR(buffers.q_buffer.get(), spec_q());
    VALIDATE_TENSOR(buffers.k_buffer.get(), spec_kv());
    VALIDATE_TENSOR(buffers.v_buffer.get(), spec_kv());
    
    // ... execute attention ...
    
    // Validate attention output
    VALIDATE_TENSOR(buffers.attn_out.get(), spec_hidden());
    
    return true;
}

bool Qwen2Pipeline::ffn_block(int layer_idx) {
    // ... get buffers, transfer activation ...
    
    // Validate FFN intermediate outputs
    auto& buffers = getBuffersForDevice(ffn_device);
    VALIDATE_TENSOR(buffers.gate_buffer.get(), spec_ffn_gate_up());
    VALIDATE_TENSOR(buffers.up_buffer.get(), spec_ffn_gate_up());
    VALIDATE_TENSOR(buffers.ffn_out.get(), spec_ffn_intermediate());
    
    return true;
}
```

**Benefits**:
- ✅ **Early Detection**: Catches dimension errors at creation time, not deep in computation
- ✅ **Clear Error Messages**: Shows expected vs actual dimensions with tensor name
- ✅ **Zero Runtime Cost** (when disabled): Macro can be compiled out in Release builds
- ✅ **Centralized Specs**: Helper methods document expected dimensions

**Example Error Output**:

```
[ERROR] TensorSpec mismatch for 'query_projection': expected [128, 896], got [128, 448]
```

**Common Validation Points**:
1. **Projection outputs**: Q/K/V after GEMM
2. **Intermediate activations**: FFN gate/up, attention context
3. **Residual connections**: Before/after addition
4. **Buffer allocation**: Ensure correct size before transfer

**Future Enhancements**:
- Compile-time dimension checking (C++20 concepts)
- Automatic shape inference (propagate through pipeline graph)
- Integration with static analysis tools

---

### 6.6 Pipeline Architecture (`pipelines/`)

**V1**: Adapter pattern wrapping pipelines in AbstractPipeline interface  
**V2**: Base class inheritance with `PipelineBase` providing common infrastructure

#### Pipeline Class Hierarchy

**Base Class** (`PipelineBase`):
- Provides common infrastructure for all model architectures
- MPI context management and device placement coordination
- Weight and activation lifecycle management
- Pure virtual methods for architecture-specific logic

**Architecture-Specific Pipelines**:
- `Qwen2Pipeline`: Qwen 2.0/2.5 models (0.5B-72B) - **implemented**
- `Qwen3Pipeline`: Qwen 3.x models (future)
- `Qwen3MoEPipeline`: Qwen MoE models (future)
- `LlamaPipeline`: LLaMA 3.x models (future)

#### PipelineBase Interface

**File**: `src/v2/pipelines/PipelineBase.h` (125 lines)

```cpp
class PipelineBase {
public:
    // Constructor
    PipelineBase(const std::string& model_path,
                 std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                 int device_idx = -1);

    virtual ~PipelineBase() = default;

    // ===== Public Interface =====
    
    /**
     * @brief Forward pass (prefill or decode)
     * @param tokens Token IDs [seq_len]
     * @param seq_len Number of tokens
     * @return true on success
     */
    virtual bool forward(const int* tokens, int seq_len) = 0;

    /**
     * @brief Get output logits (FP32)
     * @return Logits tensor [seq_len, vocab_size]
     */
    virtual const float* logits() const = 0;

    /**
     * @brief Get model architecture name
     * @return Architecture string (e.g., "qwen2", "llama")
     */
    virtual const char* architecture() const = 0;

    // ===== Context Accessors =====
    
    std::shared_ptr<MPIContext> mpi_context() const { return mpi_ctx_; }
    int device_index() const { return device_idx_; }

protected:
    // ===== Protected Interface (derived classes implement) =====
    
    /**
     * @brief Load weights from GGUF file
     * @param model_path Path to GGUF model file
     * @return true on success
     */
    virtual bool load_weights(const std::string& model_path) = 0;

    /**
     * @brief Execute one transformer layer
     * @param layer_idx Layer index (0-indexed)
     * @param seq_len Sequence length
     * @return true on success
     */
    virtual bool transformer_layer(int layer_idx, int seq_len) = 0;

    // ===== Shared State =====
    
    std::shared_ptr<MPIContext> mpi_ctx_;  // MPI coordination
    int device_idx_;                        // Default device (-1 = CPU)
    std::string model_path_;                // GGUF file path

    // Common model parameters (set by derived classes)
    int n_layers_ = 0;
    int d_model_ = 0;
    int vocab_size_ = 0;
};
```

#### Qwen2Pipeline Implementation

**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.h` (185 lines - updated Phase 4)

**Purpose**: Qwen 2.x transformer with **multi-device support** and direct kernel orchestration

```cpp
class Qwen2Pipeline : public PipelineBase {
public:
    Qwen2Pipeline(const std::string& model_path,
                  std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                  int device_idx = -1);

    ~Qwen2Pipeline() override = default;

    // PipelineBase interface
    bool forward(const int* tokens, int seq_len) override;
    const float* logits() const override;
    const char* architecture() const override { return "qwen2"; }

    // Qwen2-specific
    std::shared_ptr<TensorBase> get_layer_weight(int layer_idx, 
                                                  const std::string& weight_name);

protected:
    bool load_weights(const std::string& model_path) override;
    bool transformer_layer(int layer_idx, int seq_len) override;

private:
    // Qwen2-specific architecture parameters
    int n_heads_ = 0;
    int n_kv_heads_ = 0;  // GQA
    int head_dim_ = 0;
    int d_ff_ = 0;

    // Layer weights structure
    struct LayerWeights {
        std::shared_ptr<TensorBase> wq, wk, wv, wo;          // Attention
        std::shared_ptr<TensorBase> attn_norm;               // Pre-attention norm
        std::shared_ptr<TensorBase> gate_proj, up_proj, down_proj;  // FFN
        std::shared_ptr<TensorBase> ffn_norm;                // Pre-FFN norm
    };

    // Weights (quantized)
    std::shared_ptr<TensorBase> embedding_table_;
    std::vector<LayerWeights> layers_;
    std::shared_ptr<TensorBase> final_norm_;
    std::shared_ptr<TensorBase> lm_head_;

    // Activations (FP32)
    std::shared_ptr<FP32Tensor> current_hidden_;
    std::shared_ptr<FP32Tensor> logits_;

    // ===== Multi-Device Support (Phase 4) =====
    
    // Per-device buffer pools (Phase 4.1)
    struct ActivationBuffers {
        std::shared_ptr<FP32Tensor> q_buffer;
        std::shared_ptr<FP32Tensor> k_buffer;
        std::shared_ptr<FP32Tensor> v_buffer;
        std::shared_ptr<FP32Tensor> attn_out;
        std::shared_ptr<FP32Tensor> gate_buffer;
        std::shared_ptr<FP32Tensor> up_buffer;
        std::shared_ptr<FP32Tensor> ffn_out;
        std::shared_ptr<FP32Tensor> residual;
    };
    std::map<int, ActivationBuffers> buffers_per_device_;
    
    // Weight placement map (Phase 4.1)
    WeightPlacementMap placement_map_;
    std::set<int> active_devices_;
    
    // Helper methods (Phase 4)
    ActivationBuffers& getBuffersForDevice(int device_id);
    int getWeightDevice(const std::string& weight_name, int layer_idx) const;
    TensorBase* prepareActivationForDevice(TensorBase* activation, 
                                           int target_device,
                                           const std::string& context_name);
    
    // Transformer blocks (Phase 4.3 - device-aware)
    bool attention_block(int layer_idx);
    bool ffn_block(int layer_idx);
    
    // TensorDimensions verification helpers (Phase 4.1)
    TensorSpec spec_hidden() const;
    TensorSpec spec_q() const;
    TensorSpec spec_kv() const;
    TensorSpec spec_ffn_gate_up() const;
    TensorSpec spec_ffn_intermediate() const;
};
```

**Key Features**:

1. **Multi-Device Buffer Pools** (Phase 4.1):
   - `buffers_per_device_`: Separate Q/K/V/FFN buffers per device
   - `getBuffersForDevice()`: Lazy allocation on first use
   - Enables parallel execution across devices

2. **Weight Placement** (Phase 4.1):
   - `placement_map_`: Maps weight names to device IDs
   - `getWeightDevice()`: Query which device holds a weight
   - Supports heterogeneous placement (layer 0 → GPU:0, layer 12 → CPU, etc.)

3. **Device-Aware Execution** (Phase 4.3):
   - `prepareActivationForDevice()`: Smart transfer logic
   - `attention_block()`/`ffn_block()`: Query device, transfer, execute, track
   - Minimal transfers (only when weight device ≠ activation device)

4. **TensorDimensions Verification** (Phase 4.1):
   - `spec_*()` methods return expected dimensions
   - Runtime shape validation with `VALIDATE_TENSOR()` macro
   - Catches mismatches early

**Implementation Highlights**:

**Constructor** (initializes and loads weights):

```cpp
Qwen2Pipeline::Qwen2Pipeline(const std::string& model_path,
                             std::shared_ptr<MPIContext> mpi_ctx,
                             int device_idx)
    : PipelineBase(model_path, mpi_ctx, device_idx)
{
    // TODO: Read from GGUF metadata instead of hardcoding
    n_layers_ = 24;
    n_heads_ = 14;
    n_kv_heads_ = 2;   // Grouped-query attention
    head_dim_ = 64;
    d_model_ = 896;
    d_ff_ = 4864;
    vocab_size_ = 151936;

    if (!load_weights(model_path)) {
        throw std::runtime_error("Failed to load weights");
    }
}
```

**Weight Loading** (with placement map integration):

```cpp
bool Qwen2Pipeline::load_weights(const std::string& model_path) {
    ModelLoader loader;
    if (!loader.loadModel(model_path)) return false;

    // Validate architecture
    const GGUFModel& model = loader.getModel();
    if (model.architecture != "qwen2") return false;

    // Load embedding
    int embed_device = placement_map_.getDevice("token_embd.weight");
    embedding_table_ = loader.loadTensor("token_embd.weight", embed_device);
    active_devices_.insert(embed_device);
    
    // Load layers (each layer can be on different device)
    layers_.resize(n_layers_);
    for (int i = 0; i < n_layers_; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        
        int attn_device = placement_map_.getDevice(prefix + "attn_q.weight");
        layers_[i].wq = loader.loadTensor(prefix + "attn_q.weight", attn_device);
        layers_[i].wk = loader.loadTensor(prefix + "attn_k.weight", attn_device);
        layers_[i].wv = loader.loadTensor(prefix + "attn_v.weight", attn_device);
        layers_[i].wo = loader.loadTensor(prefix + "attn_output.weight", attn_device);
        layers_[i].attn_norm = loader.loadTensor(prefix + "attn_norm.weight", attn_device);
        
        int ffn_device = placement_map_.getDevice(prefix + "ffn_gate.weight");
        layers_[i].gate_proj = loader.loadTensor(prefix + "ffn_gate.weight", ffn_device);
        layers_[i].up_proj = loader.loadTensor(prefix + "ffn_up.weight", ffn_device);
        layers_[i].down_proj = loader.loadTensor(prefix + "ffn_down.weight", ffn_device);
        layers_[i].ffn_norm = loader.loadTensor(prefix + "ffn_norm.weight", ffn_device);
        
        active_devices_.insert(attn_device);
        active_devices_.insert(ffn_device);
    }

    int output_device = placement_map_.getDevice("output_norm.weight");
    final_norm_ = loader.loadTensor("output_norm.weight", output_device);
    lm_head_ = loader.loadTensor("output.weight", output_device);
    active_devices_.insert(output_device);

    return true;
}
```

**Execution Flow** (multi-device aware):

```cpp
bool Qwen2Pipeline::forward(const int* tokens, int seq_len) {
    // 1. Embedding
    auto embedded = allocateTensor({seq_len, d_model_});
    if (!embedding_layer(tokens, seq_len, embedded.get())) return false;
    current_hidden_ = embedded;
    current_hidden_->set_device(embedding_table_->device_index());
    
    // 2. Transformer layers (each layer may be on different device)
    for (int i = 0; i < n_layers_; ++i) {
        if (!transformer_layer(i, seq_len)) return false;
    }
    
    // 3. Output projection
    int output_device = lm_head_->device_index();
    TensorBase* output_input = prepareActivationForDevice(
        current_hidden_.get(), output_device, "OutputProjection"
    );
    if (!output_input) return false;
    
    auto gemm = getGemmKernel(output_device);
    return gemm->execute(output_input, lm_head_.get(), logits_.get(), {});
}

bool Qwen2Pipeline::transformer_layer(int layer_idx, int seq_len) {
    // Attention block (device-aware)
    if (!attention_block(layer_idx)) return false;
    
    // FFN block (device-aware)
    if (!ffn_block(layer_idx)) return false;
    
    return true;
}
```

**Attention Block** (device-aware execution - Phase 4.3):

```cpp
bool Qwen2Pipeline::attention_block(int layer_idx) {
    // 1. Query weight device
    int attn_device = getWeightDevice("attn_q", layer_idx);
    
    // 2. Transfer activation if needed
    TensorBase* input = prepareActivationForDevice(
        current_hidden_.get(), 
        attn_device, 
        "Attention-L" + std::to_string(layer_idx)
    );
    if (!input) return false;
    
    // 3. Get device-specific buffers
    auto& buffers = getBuffersForDevice(attn_device);
    
    // 4. Execute on weight device (direct kernel calls)
    auto gemm = getGemmKernel(attn_device);
    auto rope = getRoPEKernel(attn_device);
    auto attn_kernel = getAttentionKernel(attn_device);
    
    // RMSNorm
    rmsNorm(input, layers_[layer_idx].attn_norm.get(), buffers.residual.get());
    
    // Q/K/V projections
    gemm->execute(buffers.residual.get(), layers_[layer_idx].wq.get(), buffers.q_buffer.get(), {});
    gemm->execute(buffers.residual.get(), layers_[layer_idx].wk.get(), buffers.k_buffer.get(), {});
    gemm->execute(buffers.residual.get(), layers_[layer_idx].wv.get(), buffers.v_buffer.get(), {});
    
    // RoPE
    rope->execute(buffers.q_buffer.get(), {.seq_offset = 0, .theta_base = 10000.0f});
    rope->execute(buffers.k_buffer.get(), {.seq_offset = 0, .theta_base = 10000.0f});
    
    // Attention
    attn_kernel->execute(buffers.q_buffer.get(), buffers.k_buffer.get(), 
                         buffers.v_buffer.get(), buffers.attn_out.get(), {...});
    
    // Output projection
    gemm->execute(buffers.attn_out.get(), layers_[layer_idx].wo.get(), buffers.residual.get(), {});
    
    // Residual connection
    addResidual(input, buffers.residual.get(), current_hidden_.get());
    
    // 5. Track result device
    current_hidden_->set_device(attn_device);
    
    return true;
}
```

**FFN Block** (device-aware execution - Phase 4.3):

```cpp
bool Qwen2Pipeline::ffn_block(int layer_idx) {
    // 1. Query FFN weight device
    int ffn_device = getWeightDevice("ffn_gate", layer_idx);
    
    // 2. Transfer activation if needed
    TensorBase* input = prepareActivationForDevice(
        current_hidden_.get(), 
        ffn_device, 
        "FFN-L" + std::to_string(layer_idx)
    );
    if (!input) return false;
    
    // 3. Get device-specific buffers
    auto& buffers = getBuffersForDevice(ffn_device);
    
    // 4. Execute SwiGLU on FFN device
    auto gemm = getGemmKernel(ffn_device);
    
    // RMSNorm
    rmsNorm(input, layers_[layer_idx].ffn_norm.get(), buffers.residual.get());
    
    // Gate and Up projections
    gemm->execute(buffers.residual.get(), layers_[layer_idx].gate_proj.get(), 
                  buffers.gate_buffer.get(), {});
    gemm->execute(buffers.residual.get(), layers_[layer_idx].up_proj.get(), 
                  buffers.up_buffer.get(), {});
    
    // SwiGLU: gate * silu(up)
    swiGLU(buffers.gate_buffer.get(), buffers.up_buffer.get(), buffers.ffn_out.get());
    
    // Down projection
    gemm->execute(buffers.ffn_out.get(), layers_[layer_idx].down_proj.get(), 
                  buffers.residual.get(), {});
    
    // Residual connection
    addResidual(input, buffers.residual.get(), current_hidden_.get());
    
    // 5. Track result device
    current_hidden_->set_device(ffn_device);
    
    return true;
}
```

**Key Patterns**:

1. **Device-Aware Execution**: Query → Transfer → Execute → Track
2. **Minimal Transfers**: Only when activation device ≠ weight device
3. **Buffer Isolation**: Per-device buffers prevent conflicts
4. **Direct Kernel Calls**: No operator layer overhead
2. **Direct Kernel Calls**: No operator indirection
3. **Explicit Partitioning**: Pipeline owns MPI coordination
4. **Flexible Fusion**: Can combine kernels (e.g., fused QKV projection)

---

## Device Orchestration

**Status**: ✅ **Phase 4.3 Complete** (January 20, 2025)

Device orchestration is the strategic layer that determines **where weights live** (CPU, GPU0, GPU1, etc.) and **how activations flow between devices** across the model architecture. Unlike V1's static MPI slicing, V2 provides flexible placement strategies optimized for different deployment scenarios, with full support for heterogeneous multi-device execution.

### Component Overview

**Files**:
- `src/v2/loaders/DeviceOrchestrator.h/cpp` (~900 lines total)
- `src/v2/loaders/WeightPlacementMap.h/cpp` (~300 lines total)
- `src/v2/loaders/ArgParser.h/cpp` (~500 lines total)
- `tests/v2/Test__DeviceOrchestrator*.cpp` (~750 lines, 25 tests)

**Core Abstraction**:

```cpp
// Central orchestration class
class DeviceOrchestrator {
public:
    // Create placement strategy based on config
    WeightPlacementMap orchestrate(
        const GGUFModel& model,
        const OrchestrationConfig& config
    );
    
    // Parse custom device map strings
    static std::vector<DeviceMapRule> parseDeviceMapString(
        const std::string& device_map
    );
};
```

### Placement Strategies

V2 supports **4 distinct placement strategies** selected via `PlacementStrategy` enum:

#### 1. ALL_CPU (Default)

**Use Case**: Single-machine inference without GPU, development/debugging

```cpp
OrchestrationConfig config;
config.strategy = PlacementStrategy::ALL_CPU;

auto placement = orchestrator.orchestrate(model, config);
// Result: All 100% of model weights → CPU
```

**Properties**:
- ✅ Zero GPU memory required
- ✅ Simplest deployment (no device coordination)
- ✅ Portable (works everywhere)
- ❌ Slower than GPU for large models

#### 2. ALL_GPU (Offload Everything)

**Use Case**: Single GPU with sufficient VRAM (e.g., A100 80GB)

```cpp
OrchestrationConfig config;
config.strategy = PlacementStrategy::ALL_GPU;
config.primary_gpu_id = 0;  // Target GPU

auto placement = orchestrator.orchestrate(model, config);
// Result: All 100% of model weights → GPU:0
```

**Properties**:
- ✅ Maximum throughput (all ops on GPU)
- ✅ No CPU↔GPU transfers during inference
- ❌ Requires large VRAM (7B model ≈ 14GB FP16)
- ❌ Limited to models that fit single GPU

#### 3. MEMORY_AWARE (Auto-Fit Layers)

**Use Case**: GPU with limited VRAM - maximize GPU usage within memory budget

```cpp
OrchestrationConfig config;
config.strategy = PlacementStrategy::MEMORY_AWARE;
config.max_gpu_memory_mb = 8192;  // 8GB budget

auto placement = orchestrator.orchestrate(model, config);
// Result: First N layers → GPU, remainder → CPU
//   where N layers fit within 8GB including activations
```

**Algorithm**:
1. Estimate per-layer memory: `sum(tensor.size_bytes for layer)`
2. Add 20% activation overhead: `memory_per_layer *= 1.2`
3. Calculate GPU capacity: `gpu_layers = min(total_layers, available_memory / memory_per_layer)`
4. Place first `gpu_layers` on GPU, rest on CPU

**Properties**:
- ✅ Automatic resource optimization (no manual tuning)
- ✅ Graceful degradation (uses as much GPU as fits)
- ✅ Predictable memory usage (respects budget)
- ⚠️ Hybrid execution (CPU fallback for later layers)

**Implementation** (~70 lines):

```cpp
WeightPlacementMap DeviceOrchestrator::createMemoryAwareMap(
    const GGUFModel& model,
    const OrchestrationConfig& config
) {
    size_t model_memory = estimateModelMemory(model);
    size_t available_gpu = config.max_gpu_memory_mb.value_or(
        queryGPUMemory(config.primary_gpu_id)
    ) * 1024 * 1024;
    
    size_t layers = model.layers();
    size_t memory_per_layer = model_memory / layers;
    size_t gpu_layers = std::min(layers, available_gpu / memory_per_layer);
    
    WeightPlacementMap map;
    for (size_t i = 0; i < gpu_layers; ++i) {
        map.assignLayer(i, config.primary_gpu_id);
    }
    for (size_t i = gpu_layers; i < layers; ++i) {
        map.assignLayer(i, DeviceId::CPU);
    }
    return map;
}
```

#### 4. MOE_OPTIMIZED (Mixture-of-Experts)

**Use Case**: MoE architectures (Mixtral, Qwen-MoE) with sparse expert activation

```cpp
OrchestrationConfig config;
config.strategy = PlacementStrategy::MOE_OPTIMIZED;
config.moe_shared_experts_gpu = true;   // Shared → GPU (frequent)
config.moe_sparse_experts_cpu = true;   // Sparse → CPU (rare)

auto placement = orchestrator.orchestrate(model, config);
// Result: 
//   - Shared experts + gate → GPU (accessed every token)
//   - Sparse experts → CPU (only top-K activated)
```

**Pattern-Based Placement**:

```cpp
// Shared experts: accessed every forward pass → GPU
if (name.find("shared_expert") != std::string::npos ||
    name.find("gate") != std::string::npos) {
    return DeviceId::GPU_0;
}

// Sparse experts: only top-K used → CPU
if (name.find("experts.0") != std::string::npos ||
    name.find("experts.1") != std::string::npos /* ... */) {
    return DeviceId::CPU;
}
```

**Properties**:
- ✅ Optimized for MoE activation patterns (shared → GPU, sparse → CPU)
- ✅ Reduces GPU memory pressure (only frequent experts on device)
- ⚠️ Requires MoE-aware architecture (Mixtral, Qwen-MoE, etc.)

**Implementation** (~50 lines): Pattern matching on weight names

#### 5. CUSTOM (User-Defined Device Map)

**Use Case**: Advanced users with domain-specific placement requirements

```cpp
OrchestrationConfig config;
config.strategy = PlacementStrategy::CUSTOM;
config.device_map = "0-11:gpu:0,12-23:cpu,*embed*:gpu:1";

auto placement = orchestrator.orchestrate(model, config);
// Result:
//   - Layers 0-11 → GPU:0
//   - Layers 12-23 → CPU
//   - Any weight with "embed" in name → GPU:1
```

**Device Map Syntax** (3 rule types):

1. **Layer Ranges**: `"0-11:gpu:0"` → Layers 0-11 to GPU 0
2. **Percentages**: `"first_50%:gpu:0"` → First 50% of layers to GPU 0
3. **Patterns**: `"*embed*:gpu:1"` → Any weight matching `*embed*` to GPU 1

**Full Example**:

```cpp
// Complex hybrid placement
config.device_map = "first_25%:gpu:0,"       // First quarter → GPU 0
                    "last_25%:gpu:1,"        // Last quarter → GPU 1
                    "*attention*:gpu:0,"     // All attention → GPU 0
                    "*experts.0*:cpu";       // Expert 0 → CPU
```

**Properties**:
- ✅ Maximum flexibility (arbitrary placement logic)
- ✅ Supports multi-GPU load balancing
- ✅ Pattern-based overrides (e.g., keep embeddings on GPU)
- ❌ Requires expert knowledge (easy to create suboptimal maps)

**Implementation** (~200 lines):
- `parseDeviceMapString()`: Tokenize comma-separated rules
- `parseDeviceMapRule()`: Parse individual rule (type detection)
- `parseDeviceString()`: Resolve device type + ID
- `applyDeviceMapRule()`: Apply rule to placement map

### WeightPlacementMap

**Purpose**: Lightweight map from weight name → device ID

```cpp
class WeightPlacementMap {
    std::unordered_map<std::string, DeviceId> weight_to_device_;
    
public:
    void assign(const std::string& weight_name, DeviceId device);
    DeviceId getDevice(const std::string& weight_name) const;
    void assignLayer(int layer_idx, DeviceId device);  // Bulk assignment
    
    // Statistics
    size_t countWeightsOnDevice(DeviceId device) const;
    std::string summary() const;  // Human-readable report
};
```

**Integration with ModelLoader**:

```cpp
// 1. Orchestrate placement strategy
DeviceOrchestrator orchestrator;
OrchestrationConfig config = parseArgs(argc, argv);
auto placement = orchestrator.orchestrate(model, config);

// 2. Load weights with device affinity
ModelLoader loader;
for (const auto& tensor_info : model.tensors) {
    DeviceId target = placement.getDevice(tensor_info.name);
    auto tensor = loader.loadTensorToDevice(tensor_info, target);
}

// 3. Pipeline uses per-tensor device info
auto Q_kernel = weights.wq[layer]->device()->getGemmKernel();
Q_kernel->execute(...);
```

### ArgParser Integration

**Phase 1 Complete** (October 23, 2025): CLI argument parsing infrastructure

```cpp
// Example CLI usage
./llaminar2 \
  --model qwen2.5-7b-q4_0.gguf \
  --strategy memory-aware \
  --max-gpu-memory 8192 \
  --device-map "first_50%:gpu:0,last_50%:cpu"
```

**ArgParser API**:

```cpp
class ArgParser {
public:
    bool parse(int argc, char** argv);
    
    // Getters
    std::string model_path() const;
    PlacementStrategy strategy() const;
    std::optional<size_t> max_gpu_memory() const;
    std::string device_map() const;
    int gpu_id() const;
};
```

**27 Tests Passing**: Validates all argument patterns, edge cases, defaults

### Testing

**Test Coverage** (25 tests, 100% passing):

**Phase 1** (8 tests - Device selection basics):
- Strategy enum support (ALL_CPU, ALL_GPU, MEMORY_AWARE, MOE_OPTIMIZED, CUSTOM)
- WeightPlacementMap operations
- Basic orchestration workflows

**Phase 2** (17 tests - Advanced strategies):
- Device map parsing: layer ranges, percentages, patterns, mixed rules
- MEMORY_AWARE: explicit budgets, auto-detection, edge cases (0 layers, oversized)
- MOE_OPTIMIZED: shared→GPU, sparse→CPU, pattern matching
- CUSTOM: complex multi-device maps, rule application, override precedence

**Example Test**:

```cpp
TEST(Test__DeviceOrchestrator_Phase2, ParseDeviceMapMixed) {
    auto rules = DeviceOrchestrator::parseDeviceMapString(
        "0-11:gpu:0,last_25%:cpu,*embed*:gpu:1"
    );
    
    ASSERT_EQ(rules.size(), 3);
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::LAYER_RANGE);
    EXPECT_EQ(rules[1].type, DeviceMapRuleType::PERCENTAGE);
    EXPECT_EQ(rules[2].type, DeviceMapRuleType::PATTERN);
}
```

### Multi-Device Buffer Management

**Phase 4.1 Complete** (January 18, 2025): Per-device activation buffer pools

**Problem**: When weights are distributed across devices (e.g., layer 0-11 on GPU:0, layer 12-23 on GPU:1), each device needs separate activation buffers to avoid conflicts during parallel execution.

**Solution**: Pipeline maintains per-device buffer pools via `std::map<int, ActivationBuffers>`.

**Implementation** (Qwen2Pipeline):

```cpp
class Qwen2Pipeline : public PipelineBase {
private:
    // Per-device buffer pools (Phase 4.1)
    struct ActivationBuffers {
        std::shared_ptr<FP32Tensor> q_buffer;      // Query projection buffer
        std::shared_ptr<FP32Tensor> k_buffer;      // Key projection buffer
        std::shared_ptr<FP32Tensor> v_buffer;      // Value projection buffer
        std::shared_ptr<FP32Tensor> attn_out;      // Attention output buffer
        std::shared_ptr<FP32Tensor> gate_buffer;   // FFN gate buffer
        std::shared_ptr<FP32Tensor> up_buffer;     // FFN up buffer
        std::shared_ptr<FP32Tensor> ffn_out;       // FFN output buffer
        std::shared_ptr<FP32Tensor> residual;      // Residual connection buffer
    };
    
    std::map<int, ActivationBuffers> buffers_per_device_;
    
    // Helper to get/create buffers for a device
    ActivationBuffers& getBuffersForDevice(int device_id) {
        auto it = buffers_per_device_.find(device_id);
        if (it == buffers_per_device_.end()) {
            // First time using this device - allocate buffers
            ActivationBuffers buffers;
            buffers.q_buffer = std::make_shared<FP32Tensor>(
                TensorDimensions({seq_len_, d_model_})
            );
            buffers.q_buffer->set_device(device_id);
            // ... allocate all buffers ...
            
            auto [inserted_it, success] = buffers_per_device_.emplace(device_id, std::move(buffers));
            return inserted_it->second;
        }
        return it->second;
    }
};
```

**Benefits**:
- ✅ **Parallel Execution**: Different layers can execute on different devices simultaneously
- ✅ **Memory Isolation**: Each device's buffers don't conflict
- ✅ **Lazy Allocation**: Buffers only created when device is first used
- ✅ **Efficient**: Reuses buffers across layers on same device

**Usage Pattern**:

```cpp
bool Qwen2Pipeline::attention_block(int layer_idx) {
    // 1. Determine which device this layer's weights live on
    int weight_device = getWeightDevice("attn", layer_idx);
    
    // 2. Get that device's buffer pool
    auto& buffers = getBuffersForDevice(weight_device);
    
    // 3. Use device-specific buffers
    auto gemm = getGemmKernel(weight_device);
    gemm->execute(current_hidden_, weights_.wq[layer_idx], buffers.q_buffer.get());
    // ... use buffers.k_buffer, buffers.v_buffer, etc.
}
```

### Device Transfer Infrastructure

**Phase 4.2 Complete** (January 19, 2025): Cross-device tensor copying

**Problem**: When activation lives on Device A but next operation's weights are on Device B, we need to transfer the activation tensor.

**Solution**: `TensorBase::copyFrom()` virtual interface for device-to-device transfers.

**Interface** (src/v2/tensors/Tensors.h):

```cpp
class TensorBase {
public:
    /**
     * @brief Copy data from another tensor (potentially on different device)
     * @param source Source tensor to copy from
     * @return true on success, false on error
     * 
     * Handles:
     * - Shape validation (must match)
     * - Device routing (CPU↔CPU, CPU↔GPU, GPU↔GPU, etc.)
     * - Null pointer safety
     */
    virtual bool copyFrom(const TensorBase* source) = 0;
};
```

**FP32Tensor Implementation** (CPU↔CPU transfers):

```cpp
bool FP32Tensor::copyFrom(const TensorBase* source) {
    if (!source) {
        LOG_ERROR("copyFrom: source is null");
        return false;
    }
    
    // Validate shapes match
    if (source->shape() != shape_) {
        LOG_ERROR("copyFrom: shape mismatch");
        return false;
    }
    
    // Device routing
    int src_device = source->device_index();
    int dst_device = device_index();
    
    if (src_device == -1 && dst_device == -1) {
        // CPU → CPU: Direct memcpy
        const auto* src_fp32 = dynamic_cast<const FP32Tensor*>(source);
        if (!src_fp32) {
            LOG_ERROR("copyFrom: source is not FP32Tensor");
            return false;
        }
        
        std::memcpy(data_.data(), src_fp32->data_.data(), 
                    element_count() * sizeof(float));
        return true;
    }
    
    // GPU transfers (future Phase 4 CUDA)
    if (src_device >= 0 || dst_device >= 0) {
        LOG_ERROR("copyFrom: GPU transfers not yet implemented");
        return false;
    }
    
    return false;
}
```

**Quantized Tensor Stubs** (18 types):

```cpp
// IQ4_NL, Q6_K, Q8_0, F16, BF16, etc. all have:
bool IQ4_NLTensor::copyFrom(const TensorBase* source) {
    LOG_ERROR("copyFrom: quantized tensor transfers not yet implemented");
    return false;  // Stub for Phase 4 CUDA
}
```

**Future GPU Support** (Phase 4 CUDA):
- CPU → GPU: `cudaMemcpy(..., cudaMemcpyHostToDevice)`
- GPU → CPU: `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
- GPU → GPU (same device): `cudaMemcpy(..., cudaMemcpyDeviceToDevice)`
- GPU → GPU (different device): `cudaMemcpyPeer()` or staging via CPU

**Test Coverage** (7 tests, all passing):
- Basic FP32 CPU→CPU copy (same shape)
- Shape mismatch detection
- Null pointer handling
- Data integrity (values preserved exactly)
- Device index tracking (copyFrom preserves destination device)
- Multi-step transfer chains (A→B→C)
- Quantized tensor stubs (verify error handling)

### Device-Aware Execution

**Phase 4.3 Complete** (January 20, 2025): Intelligent activation transfers and device tracking

**Problem**: Activations flow through transformer layers, but weights may be on different devices. We need to:
1. Detect when activation device ≠ weight device
2. Transfer activation to weight device (if needed)
3. Execute operation on weight device
4. Track activation's current device for next layer

**Solution**: `prepareActivationForDevice()` helper + device tracking in transformer blocks.

#### prepareActivationForDevice() Helper

**Purpose**: Smart activation transfer logic with staging buffer reuse

```cpp
TensorBase* Qwen2Pipeline::prepareActivationForDevice(
    TensorBase* activation,
    int target_device,
    const std::string& context_name
) {
    int current_device = activation->device_index();
    
    // Fast path: already on target device
    if (current_device == target_device) {
        return activation;  // No transfer needed
    }
    
    // Transfer required
    LOG_INFO("[" << context_name << "] Transferring activation from device " 
             << current_device << " to device " << target_device);
    
    // Get target device's residual buffer as staging area
    auto& target_buffers = getBuffersForDevice(target_device);
    TensorBase* staging = target_buffers.residual.get();
    
    // Validate staging buffer size
    if (staging->element_count() < activation->element_count()) {
        LOG_ERROR("Staging buffer too small: " << staging->element_count() 
                  << " < " << activation->element_count());
        return nullptr;
    }
    
    // Perform transfer
    if (!staging->copyFrom(activation)) {
        LOG_ERROR("Failed to transfer activation to device " << target_device);
        return nullptr;
    }
    
    // Update staging buffer's device index
    staging->set_device(target_device);
    
    return staging;  // Return transferred activation
}
```

**Key Features**:
- ✅ **Skip unnecessary transfers**: Fast path when already on target device
- ✅ **Reuse staging buffers**: No dynamic allocation during forward pass
- ✅ **Size validation**: Catches buffer size mismatches early
- ✅ **Error handling**: Returns nullptr on failure, caller can propagate
- ✅ **Logging**: Debug visibility into cross-device traffic

#### Attention Block (Device-Aware)

**Before Phase 4.3** (device-agnostic):

```cpp
bool Qwen2Pipeline::attention_block(const LayerWeights& layer) {
    // Assumed all weights on same device, no transfers
    auto gemm = getGemmKernel();  // Which device?
    gemm->execute(current_hidden_, layer.wq, q_buffer_);
    // ...
}
```

**After Phase 4.3** (device-aware):

```cpp
bool Qwen2Pipeline::attention_block(int layer_idx) {
    // 1. Query weight device
    int attn_device = getWeightDevice("attn_q", layer_idx);
    
    // 2. Transfer activation if needed
    TensorBase* input = prepareActivationForDevice(
        current_hidden_.get(), 
        attn_device, 
        "Attention-L" + std::to_string(layer_idx)
    );
    if (!input) return false;
    
    // 3. Get device-specific buffers
    auto& buffers = getBuffersForDevice(attn_device);
    
    // 4. Execute on weight device
    auto gemm = getGemmKernel(attn_device);
    gemm->execute(input, weights_.wq[layer_idx], buffers.q_buffer.get());
    gemm->execute(input, weights_.wk[layer_idx], buffers.k_buffer.get());
    gemm->execute(input, weights_.wv[layer_idx], buffers.v_buffer.get());
    
    // ... attention computation (scores, softmax, output) ...
    
    // 5. Track result device
    current_hidden_->set_device(attn_device);
    
    return true;
}
```

#### FFN Block (Device-Aware)

**Implementation**:

```cpp
bool Qwen2Pipeline::ffn_block(int layer_idx) {
    // 1. Query FFN weight device
    int ffn_device = getWeightDevice("ffn_gate", layer_idx);
    
    // 2. Transfer activation if needed
    TensorBase* input = prepareActivationForDevice(
        current_hidden_.get(), 
        ffn_device, 
        "FFN-L" + std::to_string(layer_idx)
    );
    if (!input) return false;
    
    // 3. Get device-specific buffers
    auto& buffers = getBuffersForDevice(ffn_device);
    
    // 4. Execute SwiGLU (gate × σ(up)) on FFN device
    auto gemm = getGemmKernel(ffn_device);
    gemm->execute(input, weights_.gate_proj[layer_idx], buffers.gate_buffer.get());
    gemm->execute(input, weights_.up_proj[layer_idx], buffers.up_buffer.get());
    
    // Element-wise: gate = gate * silu(up)
    swiGLU(buffers.gate_buffer.get(), buffers.up_buffer.get(), buffers.ffn_out.get());
    
    // Down projection
    gemm->execute(buffers.ffn_out.get(), weights_.down_proj[layer_idx], buffers.residual.get());
    
    // 5. Track result device
    current_hidden_->set_device(ffn_device);
    
    return true;
}
```

#### Execution Flow Example

**Scenario**: 3-layer model with heterogeneous placement
- Layer 0 weights: GPU:0
- Layer 1 weights: GPU:1
- Layer 2 weights: CPU

**Execution Trace**:

```
[Embedding] Output on CPU → current_hidden_ device = -1

[Layer 0 Attention]
  - getWeightDevice("attn_q", 0) → GPU:0
  - prepareActivationForDevice(CPU → GPU:0) → TRANSFER
  - Execute attention on GPU:0
  - current_hidden_ device = 0

[Layer 0 FFN]
  - getWeightDevice("ffn_gate", 0) → GPU:0
  - prepareActivationForDevice(GPU:0 → GPU:0) → NO TRANSFER
  - Execute FFN on GPU:0
  - current_hidden_ device = 0

[Layer 1 Attention]
  - getWeightDevice("attn_q", 1) → GPU:1
  - prepareActivationForDevice(GPU:0 → GPU:1) → TRANSFER
  - Execute attention on GPU:1
  - current_hidden_ device = 1

[Layer 1 FFN]
  - getWeightDevice("ffn_gate", 1) → GPU:1
  - prepareActivationForDevice(GPU:1 → GPU:1) → NO TRANSFER
  - Execute FFN on GPU:1
  - current_hidden_ device = 1

[Layer 2 Attention]
  - getWeightDevice("attn_q", 2) → CPU
  - prepareActivationForDevice(GPU:1 → CPU) → TRANSFER
  - Execute attention on CPU
  - current_hidden_ device = -1

[Layer 2 FFN]
  - getWeightDevice("ffn_gate", 2) → CPU
  - prepareActivationForDevice(CPU → CPU) → NO TRANSFER
  - Execute FFN on CPU
  - current_hidden_ device = -1

[Output] Already on CPU, no transfer needed
```

**Performance Characteristics**:
- ✅ **Minimal Transfers**: Only 3 transfers for 6 blocks (2 GPU→GPU, 1 GPU→CPU)
- ✅ **Locality Optimization**: Attention + FFN on same device = no intermediate transfer
- ✅ **Explicit Tracking**: Device index always reflects true location

**Test Coverage** (6 tests, all passing):
- `TransferWhenNeeded`: Detects device mismatch, performs transfer
- `NoTransferWhenSameDevice`: Fast path when already on target
- `PlacementMapIntegration`: Uses WeightPlacementMap correctly
- `DeviceTrackingCorrect`: Verifies device index updates after operations
- `MultiLayerHeterogeneous`: 3-layer scenario with mixed GPU/CPU placement
- `BufferReuseAcrossLayers`: Validates staging buffer reuse pattern

---

## Multi-GPU Design

### Heterogeneous Execution

**V1 Limitation**: Single backend per rank (all ranks use same backend)

```cpp
// V1: All ranks must use same backend
mpirun -np 4 llaminar  # All 4 ranks use OpenBLAS OR all use CUDA
```

**V2 Capability**: Per-tensor device selection (mix backends in single run)

```cpp
// V2: Ranks can use different backends for different tensors
auto embedding_device = DeviceManager::getCPUContext();     // Rank 0: CPU
auto attn_device = DeviceManager::getCudaDevice(0);         // Rank 0: CUDA
auto ffn_device = DeviceManager::getRocmDevice(1);          // Rank 1: ROCm

// Example: CUDA for attention, ROCm for FFN, CPU for output
auto attn_kernel = attn_device->getAttentionKernel();       // cuBLAS attention
auto ffn_kernel = ffn_device->getGemmKernel();              // rocBLAS GEMM
auto output_kernel = embedding_device->getGemmKernel();     // OpenBLAS GEMM
```

### Device Manager Initialization

```cpp
void DeviceManager::initialize() {
    // Enumerate CPU
    cpu_context_ = std::make_unique<CPUComputeContext>();
    
    // Enumerate CUDA devices
    #ifdef HAVE_CUDA
    int num_cuda_devices = 0;
    cudaGetDeviceCount(&num_cuda_devices);
    for (int i = 0; i < num_cuda_devices; ++i) {
        cuda_contexts_.push_back(std::make_unique<CudaComputeContext>(i));
    }
    #endif
    
    // Enumerate ROCm devices
    #ifdef HAVE_ROCM
    int num_rocm_devices = 0;
    hipGetDeviceCount(&num_rocm_devices);
    for (int i = 0; i < num_rocm_devices; ++i) {
        rocm_contexts_.push_back(std::make_unique<RocmComputeContext>(i));
    }
    #endif
    
    // Enumerate Vulkan devices
    #ifdef HAVE_VULKAN
    auto vulkan_devices = enumerateVulkanDevices();
    for (auto& device : vulkan_devices) {
        vulkan_contexts_.push_back(std::make_unique<VulkanComputeContext>(device));
    }
    #endif
}
```

### Tensor Data Movement

```cpp
class TensorBase {
    // Move tensor to different device
    virtual void to_device(DeviceId target_device) = 0;
    
    // Check if device transfer needed
    virtual bool is_on_device(DeviceId device) const = 0;
};

// Usage in pipeline
if (!tensor->is_on_device(kernel->device_id())) {
    tensor->to_device(kernel->device_id());  // Implicit copy
}
kernel->execute(tensor, ...);
```

**Future Optimization**: Async transfers with compute overlap

---

## IQ4_NL Implementation

### Overview

**File**: `src/v2/tensors/IQ4_NLTensor.h` (1708 lines)

**Purpose**: High-performance 4-bit quantized tensor with fused dequant+GEMM

**Key Achievements**:
- **+41% FP32 GEMM Speedup**: 64×32 tile optimization (vs 32×32 baseline)
- **+26% BF16 GEMM Speedup**: Specialized 16-bit pathway
- **Zero Memory Overhead**: Fused dequant (no intermediate FP32 buffer)
- **L1 Cache Tuned**: Conservative 32KB assumption (no runtime detection)

### Quantization Format

**IQ4_NL (Importance-Weighted 4-bit)**:

```cpp
struct IQ4_NLBlock {
    uint8_t qs[QK4_NL / 2];  // 16 4-bit values packed into 8 bytes
    uint16_t d;              // Delta (FP16 scale factor)
};

// Dequantization
float dequantize(const IQ4_NLBlock& block, int index) {
    uint8_t nibble = (block.qs[index / 2] >> (4 * (index % 2))) & 0xF;
    float scale = fp16_to_fp32(block.d);
    return scale * kvalues_iq4nl[nibble];  // Importance-weighted lookup
}
```

**Properties**:
- **Compression**: 4 bits per weight (8× vs FP32)
- **Importance Weighting**: Non-uniform quantization levels
- **Block Size**: 32 values per block (QK4_NL = 32)
- **Overhead**: 10 bytes per block (8 bytes quant + 2 bytes scale)

### Fused Dequant+GEMM Kernel

**Key Innovation**: Dequantize directly into GEMM accumulator (no intermediate buffer)

```cpp
class IQ4_NLQuantizedGemm : public ITensorGemm {
public:
    bool execute(const TensorBase* A, const TensorBase* B, TensorBase* C,
                 const GemmParams& params) override {
        auto iq4nl_B = dynamic_cast<const IQ4_NLTensor*>(B);
        
        // Tile-optimized fused dequant+GEMM
        for (size_t m_tile = 0; m_tile < M; m_tile += TILE_M) {
            for (size_t n_tile = 0; n_tile < N; n_tile += TILE_N) {
                for (size_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
                    // Dequantize B tile on-the-fly
                    auto B_dequant = dequantizeTile(iq4nl_B, k_tile, n_tile);
                    
                    // Accumulate into C tile
                    gemm_tile(A_tile, B_dequant, C_tile);
                }
            }
        }
    }
};
```

**Benefits**:
- ✅ **No Dequant Buffer**: Save memory allocation overhead
- ✅ **L1 Cache Friendly**: Dequant directly into GEMM working set
- ✅ **Vectorized**: SIMD-optimized dequant in tight loop

### Tile Size Optimization

**Sweep Results** (from Oct 2025 optimization):

| Tile Size | FP32 GEMM Time | Speedup vs 32×32 | BF16 GEMM Time | Speedup vs 32×32 |
|-----------|----------------|------------------|----------------|------------------|
| 32×32 | 100% (baseline) | 1.0× | 100% (baseline) | 1.0× |
| 64×32 | **59%** | **1.69×** (41% faster) | **79%** | **1.26×** (26% faster) |
| 64×64 | 61% | 1.64× | 81% | 1.23× |
| 128×32 | 63% | 1.59× | 83% | 1.20× |

**Optimal Choice**: **64×32 tiles** (best balance of cache utilization and vectorization)

**Implementation**:

```cpp
constexpr size_t TILE_M = 64;  // Optimal from sweep
constexpr size_t TILE_N = 32;  // Optimal from sweep
constexpr size_t TILE_K = 32;  // Balance register pressure

// L1 cache constant (conservative estimate, no runtime detection)
constexpr size_t L1_CACHE_SIZE = 32 * 1024;  // 32KB
```

### BF16 Specialized Path

**Motivation**: Many activations are FP32, but can benefit from BF16 conversion

```cpp
void IQ4_NLTensor::decodeRowToBF16(size_t row_idx, uint16_t* out_bf16) {
    const IQ4_NLBlock* blocks = /* ... */;
    
    // Decode 4-bit → FP32 → BF16 in tight loop
    for (size_t i = 0; i < row_elements; ++i) {
        float dequant = dequantize_iq4nl(blocks, i);
        out_bf16[i] = simd::fp32_to_bf16(dequant);  // Round-to-nearest-even
    }
}
```

**Performance**: +26% speedup vs FP32 path (reduced memory bandwidth)

### CPU Feature Detection

**Current**: Uses free functions from `CPUFeatures.h`

```cpp
if (cpufeatures::cpu_supports_avx512()) {
    decode_iq4nl_avx512(block, output);
} else if (cpufeatures::cpu_supports_avx2()) {
    decode_iq4nl_avx2(block, output);
} else {
    decode_iq4nl_scalar(block, output);
}
```

**Removed**: V1's runtime L1 cache detection (now conservative constant)

---

## MPI Tensor Partitioning Strategy

### Overview

V2 pipelines own **runtime partitioning decisions** for optimal memory usage and communication patterns. This supports production requirements:
- ✅ **Model sizes**: 0.5B → 1T parameters (dense and MoE)
- ✅ **Long context**: Up to 128K tokens with sequence slicing
- ✅ **Batching**: Multi-user serving (32-128 concurrent users)
- ✅ **Scalability**: 2 ranks (dual-socket) → 1000+ ranks (multi-node clusters)

### Activation Partitioning Strategies

**Multi-Dimensional Partitioning**:

```cpp
enum class ActivationStrategy {
    REPLICATED,      // Full copy on each rank (decode single token)
    SEQUENCE_SLICE,  // Split by sequence dimension (long context: 64K-128K)
    BATCH_SLICE,     // Split by batch dimension (multi-user: 32-128 users)
    HYBRID_2D        // Split by both batch and sequence (extreme scale: 1T MoE)
};

class QwenPipeline {
    ActivationStrategy selectActivationStrategy(int seq_len, int batch_size) {
        size_t total_activation_size = batch_size * seq_len * config_.d_model * sizeof(float);
        size_t memory_per_rank = getAvailableMemory() / mpi_ctx_.world_size;
        
        // Single token decode: always replicate (minimize latency)
        if (seq_len == 1 && batch_size == 1) {
            return ActivationStrategy::REPLICATED;
        }
        
        // Large batch with short sequences: batch slicing
        // Example: 32 users × 512 tokens → split 16 users per rank
        if (batch_size >= 8 && seq_len < 2048) {
            return ActivationStrategy::BATCH_SLICE;
        }
        
        // Long context with small batch: sequence slicing
        // Example: 1 user × 128K tokens → split 64K per rank
        if (seq_len >= 4096 && total_activation_size > memory_per_rank * 0.3) {
            return ActivationStrategy::SEQUENCE_SLICE;
        }
        
        // Extreme scale (1T MoE models): hybrid 2D slicing
        // Example: 16 users × 64K tokens → 8 users × 32K per rank
        if (total_activation_size > memory_per_rank * 0.6) {
            return ActivationStrategy::HYBRID_2D;
        }
        
        // Default: replicated (fits in memory, simple)
        return ActivationStrategy::REPLICATED;
    }
    
    // Calculate 2D partition grid for hybrid slicing
    std::pair<int, int> compute2DGrid(int batch_size, int seq_len) {
        int batch_splits = 1;
        int seq_splits = 1;
        
        if (mpi_ctx_.world_size == 2) {
            // Dual-socket: choose dominant dimension
            if (batch_size > seq_len / 1024) {
                batch_splits = 2;  // Batch-heavy: [2, 1]
            } else {
                seq_splits = 2;    // Sequence-heavy: [1, 2]
            }
        } else if (mpi_ctx_.world_size == 4) {
            batch_splits = 2;
            seq_splits = 2;        // Balanced 2×2 grid
        } else if (mpi_ctx_.world_size == 8) {
            if (batch_size >= 16) {
                batch_splits = 4;
                seq_splits = 2;    // Batch-heavy: 4×2 grid
            } else {
                batch_splits = 2;
                seq_splits = 4;    // Sequence-heavy: 2×4 grid
            }
        }
        
        return {batch_splits, seq_splits};
    }
};
```

### Weight Partitioning

**Column/Row Slicing for Linear Projections**:

```cpp
struct TensorPartition {
    enum class Type {
        REPLICATED,      // Full tensor on all ranks
        COLUMN_SLICE,    // Split by output features
        ROW_SLICE,       // Split by input features
        EXPERT_SLICE     // MoE: split by expert assignment
    } type;
    
    int offset;          // Start index for this rank's slice
    int count;           // Number of elements in slice
    bool requires_allgather;   // Need to gather after computation
    bool requires_allreduce;   // Need to reduce-sum after computation
};

class QwenPipeline {
    TensorPartition partitionWeight(
        TensorBase* global_weight,
        const std::string& weight_name,
        int layer_idx
    ) {
        // Q/K/V projections: column slice (split by heads)
        if (weight_name == "wq" || weight_name == "wk" || weight_name == "wv") {
            int out_features = global_weight->shape()[0];
            int features_per_rank = out_features / mpi_ctx_.world_size;
            
            return TensorPartition{
                .type = TensorPartition::Type::COLUMN_SLICE,
                .offset = mpi_ctx_.rank * features_per_rank,
                .count = features_per_rank,
                .requires_allgather = true  // Gather for full Q/K/V
            };
        }
        
        // Output projection: replicated (memory efficient for small weights)
        if (weight_name == "wo") {
            return TensorPartition{
                .type = TensorPartition::Type::REPLICATED,
                .offset = 0,
                .count = global_weight->shape()[0]
            };
        }
        
        // FFN gate/up: column slice
        if (weight_name == "w_gate" || weight_name == "w_up") {
            int d_ff = global_weight->shape()[0];
            int ff_per_rank = d_ff / mpi_ctx_.world_size;
            
            return TensorPartition{
                .type = TensorPartition::Type::COLUMN_SLICE,
                .offset = mpi_ctx_.rank * ff_per_rank,
                .count = ff_per_rank,
                .requires_allgather = true
            };
        }
        
        // FFN down: row slice for efficient allreduce
        if (weight_name == "w_down") {
            int d_ff = global_weight->shape()[1];
            int ff_per_rank = d_ff / mpi_ctx_.world_size;
            
            return TensorPartition{
                .type = TensorPartition::Type::ROW_SLICE,
                .offset = mpi_ctx_.rank * ff_per_rank,
                .count = ff_per_rank,
                .requires_allreduce = true  // Sum partial results
            };
        }
        
        // Default: replicated (RMSNorm, embeddings)
        return TensorPartition{.type = TensorPartition::Type::REPLICATED};
    }
};
```

### Concrete Example: Attention with Sequence Slicing

**Large Prefill (seq_len = 128K, batch_size = 1)**:

```cpp
bool QwenPipeline::attention_block_sequence_sliced(
    int layer, TensorBase* input, TensorBase* output
) {
    // Input: [128K, 896] globally, split to [64K, 896] per rank
    const int global_seq_len = 128 * 1024;
    const int local_seq_len = global_seq_len / mpi_ctx_.world_size;  // 64K per rank
    const int d_model = config_.d_model;
    
    // === Step 1: Local Q/K/V projections ===
    auto device = device_mgr_.selectDevice(input->size_bytes(), OperationType::GEMM);
    auto gemm = device->getGemmKernel();
    
    // Each rank processes its local sequence chunk
    auto local_Q = allocateTensor({local_seq_len, d_model});
    auto local_K = allocateTensor({local_seq_len, config_.d_model_kv});
    auto local_V = allocateTensor({local_seq_len, config_.d_model_kv});
    
    // Weights are REPLICATED for sequence slicing (each rank needs full weight)
    gemm->execute(input, weights_.wq[layer], local_Q, {.transpose_B = true});
    gemm->execute(input, weights_.wk[layer], local_K, {.transpose_B = true});
    gemm->execute(input, weights_.wv[layer], local_V, {.transpose_B = true});
    
    // === Step 2: RoPE with sequence-aware offsets ===
    auto rope = device->getRoPEKernel();
    int seq_offset = mpi_ctx_.rank * local_seq_len;  // Rank 0: 0, Rank 1: 64K
    rope->execute(local_Q, {.seq_offset = seq_offset, .theta_base = 10000.0f});
    rope->execute(local_K, {.seq_offset = seq_offset, .theta_base = 10000.0f});
    
    // === Step 3: Distributed attention via ring-reduce ===
    // Each rank computes attention against its local K/V
    auto local_scores = allocateTensor({local_seq_len, local_seq_len});
    compute_local_attention_scores(local_Q, local_K, local_scores);
    
    // Ring-reduce pattern for cross-chunk attention
    // Rank 0 sends its K/V to Rank 1, receives K/V from Rank 1
    // This enables full [128K, 128K] attention matrix computation distributedly
    auto cross_chunk_scores = allocateTensor({local_seq_len, local_seq_len});
    ring_reduce_attention(local_Q, local_K, local_V, cross_chunk_scores, mpi_ctx_);
    
    // Combine local and cross-chunk attention
    auto attn_out = allocateTensor({local_seq_len, d_model});
    combine_attention_chunks(local_scores, cross_chunk_scores, local_V, attn_out);
    
    // === Step 4: Output projection and gather ===
    auto local_output = allocateTensor({local_seq_len, d_model});
    gemm->execute(attn_out, weights_.wo[layer], local_output, {.transpose_B = true});
    
    // Allgather to reconstruct full [128K, 896] output
    MPI_Allgather(
        local_output->data(), local_seq_len * d_model, MPI_FLOAT,
        output->data(), local_seq_len * d_model, MPI_FLOAT,
        mpi_ctx_.comm
    );
    
    return true;
}
```

### Concrete Example: Batched Inference

**Multi-User Serving (batch_size = 32, seq_len = 512)**:

```cpp
bool QwenPipeline::attention_block_batch_sliced(
    int layer, TensorBase* input, TensorBase* output
) {
    // Input: [32, 512, 896] globally, split to [16, 512, 896] per rank
    const int global_batch = 32;
    const int local_batch = global_batch / mpi_ctx_.world_size;  // 16 per rank
    const int seq_len = 512;
    const int d_model = config_.d_model;
    
    // === Step 1: Local Q/K/V projections for local batch ===
    auto device = device_mgr_.selectDevice(input->size_bytes(), OperationType::GEMM);
    auto gemm = device->getGemmKernel();
    
    // Shape: [16, 512, 896] for local batch
    auto local_Q = allocateTensor({local_batch, seq_len, d_model});
    auto local_K = allocateTensor({local_batch, seq_len, config_.d_model_kv});
    auto local_V = allocateTensor({local_batch, seq_len, config_.d_model_kv});
    
    // Batched GEMM: each batch element independent
    gemm->execute(input, weights_.wq[layer], local_Q, {.transpose_B = true});
    gemm->execute(input, weights_.wk[layer], local_K, {.transpose_B = true});
    gemm->execute(input, weights_.wv[layer], local_V, {.transpose_B = true});
    
    // === Step 2: RoPE (position embeddings same across batch) ===
    auto rope = device->getRoPEKernel();
    rope->execute(local_Q, {.seq_offset = 0, .theta_base = 10000.0f});
    rope->execute(local_K, {.seq_offset = 0, .theta_base = 10000.0f});
    
    // === Step 3: Independent attention per batch element ===
    // Key advantage: NO cross-rank communication for attention!
    // Each rank computes attention for its local batch elements
    auto attn_kernel = device->getAttentionKernel();
    auto attn_out = allocateTensor({local_batch, seq_len, d_model});
    
    attn_kernel->execute(local_Q, local_K, local_V, attn_out, {
        .num_heads = config_.n_head,
        .head_dim = config_.head_dim,
        .causal_mask = true
    });
    
    // === Step 4: Output projection (still local) ===
    gemm->execute(attn_out, weights_.wo[layer], output, {.transpose_B = true});
    
    // No gather needed! Each rank's output goes to its batch slice
    return true;
}
```

### Memory Footprint Analysis

**Dual-Socket Xeon (2 Ranks) - Production Scenarios**:

#### Scenario 1: Single User, Long Context (seq_len = 128K)
| Component | Memory per Rank | Strategy |
|-----------|-----------------|----------|
| Activations | 229 MB (64K × 896 × 4B) | Sequence slice |
| Weights (0.5B model) | ~1 GB (shared) | Replicated |
| KV cache | 1.8 GB (24 layers × 64K × 896 × 2 × 4B) | Sequence slice |
| **Total** | **~3 GB** | Fits easily in 64 GB NUMA node |

#### Scenario 2: Multi-User, Short Context (batch=32, seq_len=512)
| Component | Memory per Rank | Strategy |
|-----------|-----------------|----------|
| Activations | 29 MB (16 × 512 × 896 × 4B) | Batch slice |
| Weights (0.5B model) | ~1 GB (shared) | Replicated |
| KV cache | 226 MB (24 × 16 × 512 × 896 × 2 × 4B) | Batch slice |
| **Total** | **~1.3 GB** | Efficient for throughput |

#### Scenario 3: Extreme MoE (1T params, batch=16, seq_len=64K)
| Component | Memory per Rank | Strategy |
|-----------|-----------------|----------|
| Activations | 1.8 GB (8 × 32K × 896 × 4B) | Hybrid 2D [2,2] |
| Weights (experts) | ~500 GB (64 experts per rank) | Expert sharding |
| KV cache | 14 GB (layers × 8 × 32K × 896 × 2 × 4B) | Hybrid 2D |
| **Total** | **~516 GB** | Requires multi-node (256+ ranks) |

### Communication Patterns

**Sequence Slicing**:
- **Pattern**: Ring-reduce for cross-chunk attention
- **Volume**: O(seq_len² / world_size) per layer
- **Frequency**: Once per transformer layer
- **Optimization**: Overlap with next layer computation

**Batch Slicing**:
- **Pattern**: Independent computation (minimal communication)
- **Volume**: Only for weight-sliced projections (if any)
- **Frequency**: Rare (most operations batch-independent)
- **Optimization**: Best for latency-sensitive serving

**Hybrid 2D**:
- **Pattern**: Hierarchical (batch-local, then sequence-local)
- **Volume**: O((batch × seq_len) / world_size)
- **Frequency**: Per layer, staged
- **Optimization**: Requires careful choreography

### Implementation Roadmap

**Phase 1: Replicated Baseline (Week 1)**
- [x] V2 infrastructure complete
- [ ] Implement replicated activation path
- [ ] Validate single-token decode
- [ ] Benchmark small models (0.5B-7B)

**Phase 2: Batch Slicing (Weeks 2-3)**
- [ ] Implement batch dimension partitioning
- [ ] Add independent batch-element attention
- [ ] Test multi-user serving (8-32 users)
- [ ] Optimize throughput metrics

**Phase 3: Sequence Slicing (Weeks 4-5)**
- [ ] Implement sequence dimension partitioning
- [ ] Add ring-reduce attention pattern
- [ ] Test long context (64K-128K tokens)
- [ ] Validate memory reduction

**Phase 4: Hybrid 2D (Weeks 6-7)**
- [ ] Implement 2D grid partitioning
- [ ] Add hierarchical communication
- [ ] Test extreme scale (large batch + long context)
- [ ] Optimize for 1T MoE models

**Phase 5: MoE Expert Sharding (Weeks 8-9)**
- [ ] Implement expert-based partitioning
- [ ] Add dynamic expert routing
- [ ] Test Kimi K2 class models (128+ experts)
- [ ] Optimize load balancing

---

## Development Guidelines

### Adding New Kernels

**Step 1**: Define interface in `TensorKernels.h`

```cpp
class ITensorLayerNorm {
public:
    virtual bool execute(
        const TensorBase* input,
        TensorBase* output,
        const LayerNormParams& params
    ) = 0;
};
```

**Step 2**: Implement for target backend

```cpp
// kernels/CudaLayerNormKernel.cu
class CudaLayerNormKernel : public ITensorLayerNorm {
public:
    bool execute(const TensorBase* input, TensorBase* output,
                 const LayerNormParams& params) override {
        // CUDA implementation
        launch_layernorm_kernel<<<blocks, threads>>>(
            input->data(), output->data(), params.eps
        );
        return true;
    }
};
```

**Step 3**: Register in `ComputeContext`

```cpp
class CudaComputeContext : public ComputeContext {
    ITensorLayerNorm* getLayerNormKernel() override {
        if (!layernorm_kernel_) {
            layernorm_kernel_ = std::make_unique<CudaLayerNormKernel>();
        }
        return layernorm_kernel_.get();
    }
};
```

**Step 4**: Use in pipeline

```cpp
auto device = device_mgr_.selectDevice(...);
auto layernorm = device->getLayerNormKernel();
layernorm->execute(input, output, params);
```

### Quantized Tensor Strategy Pattern (IBlockDecoder)

**Problem**: Quantized tensors need format-specific decode logic (IQ4_NL, Q6_K, Q8_0, etc.), but we want a **single generic GEMM kernel** that works for all formats without code duplication.

**Solution**: **IBlockDecoder** interface + generic `QuantizedGemmKernel`

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Separation of Concerns                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  tensors/                      kernels/cpu/                │
│  ┌─────────────────┐           ┌──────────────────┐        │
│  │ IQ4_NLTensor    │◄─────────►│ QuantizedGemm    │        │
│  │ (decode logic)  │           │ (generic kernel) │        │
│  └─────────────────┘           └──────────────────┘        │
│          │ implements                    │ uses            │
│          ▼                                ▼                 │
│  ┌─────────────────┐           ┌──────────────────┐        │
│  │ IBlockDecoder   │◄─────────►│ ITensorGemm      │        │
│  │ (interface)     │           │ (interface)      │        │
│  └─────────────────┘           └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Step 1: Define IBlockDecoder Interface

**File**: `src/v2/tensors/TensorKernels.h`

```cpp
/**
 * @brief Strategy interface for block-based quantized decode
 * 
 * Decouples quantization-specific decode logic from generic GEMM kernels.
 * Each quantized tensor type (IQ4_NL, Q6_K, Q8_0) implements this interface
 * to provide its unique decode algorithm.
 * 
 * Performance: All methods marked `always_inline` for zero overhead when
 * called from GEMM hot paths (compiler devirtualizes inline calls).
 */
class IBlockDecoder {
public:
    virtual ~IBlockDecoder() = default;
    
    /**
     * @brief Decode one quantized block to FP32
     * 
     * @param row_idx Tensor row index
     * @param k_block_offset Block offset along K dimension (0-based, units of block_size())
     * @param output Destination buffer (block_size() floats)
     */
    __attribute__((always_inline))
    virtual void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const = 0;
    
    /**
     * @brief Direct access to raw quantized block (for specialized kernels like VNNI)
     * 
     * @param row_idx Tensor row index
     * @param k_block_offset Block offset along K dimension
     * @return Const pointer to raw quantized block data
     */
    __attribute__((always_inline))
    virtual const void* get_raw_block_at(size_t row_idx, size_t k_block_offset) const = 0;
    
    /**
     * @brief Number of logical rows in decoder's tensor
     */
    __attribute__((always_inline))
    virtual size_t decoder_rows() const = 0;
    
    /**
     * @brief Number of logical columns in decoder's tensor
     */
    __attribute__((always_inline))
    virtual size_t decoder_cols() const = 0;
    
    /**
     * @brief Elements per quantized block (e.g., 32 for IQ4_NL, 256 for Q6_K)
     */
    __attribute__((always_inline))
    virtual size_t block_size() const = 0;
};
```

**Key Design Choices**:

- **`__attribute__((always_inline))`**: Eliminates virtual dispatch overhead in hot paths (compiler inlines into GEMM loop)
- **Pure Virtual**: No default implementation (each format must define decode)
- **Block-Oriented**: All quantized formats use block structure (32-256 elements)
- **Zero Overhead**: Measured identical performance to direct calls when inlined

#### Step 2: Implement IBlockDecoder in Tensor Class

**File**: `src/v2/tensors/Tensors.h`

```cpp
/**
 * @brief IQ4_NL quantized tensor (4.5 bpw, 7.1× compression)
 * 
 * Implements IBlockDecoder to provide decode logic to QuantizedGemmKernel.
 */
class IQ4_NLTensor : public TensorBase, public IBlockDecoder {
public:
    // TensorBase interface...
    
    // IBlockDecoder implementation (INLINE for zero overhead)
    __attribute__((always_inline))
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const IQ4_NLBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeBlock(block, output);  // Format-specific decode
    }
    
    __attribute__((always_inline))
    const void* get_raw_block_at(size_t row_idx, size_t k_block_offset) const override {
        return &blocks_[row_idx * blocks_per_row_ + k_block_offset];
    }
    
    __attribute__((always_inline))
    size_t decoder_rows() const override { return shape_[0]; }
    
    __attribute__((always_inline))
    size_t decoder_cols() const override { return shape_[1]; }
    
    __attribute__((always_inline))
    size_t block_size() const override { return 32; }  // IQ4_NL: 32 elements/block
    
    // Factory: Create generic GEMM kernel
    std::unique_ptr<ITensorGemm> createGemm() const override {
        return std::make_unique<QuantizedGemmKernel>(this);  // Pass decoder interface
    }
    
private:
    std::vector<IQ4_NLBlock> blocks_;
    size_t blocks_per_row_;
    
    // Format-specific decode (private, called by decode_block_at)
    static void decodeBlock(const IQ4_NLBlock& block, float* output);
};
```

**Implementation Notes**:

- **Dual Inheritance**: `TensorBase` (tensor interface) + `IBlockDecoder` (decode strategy)
- **Inline Overrides**: Critical for performance (devirtualization)
- **Private Decode**: Format-specific logic stays encapsulated in tensor class
- **Factory Pattern**: `createGemm()` returns generic kernel with `this` as decoder

#### Step 3: Generic Quantized GEMM Kernel

**File**: `src/v2/kernels/cpu/QuantizedGemm.h`

```cpp
/**
 * @brief Generic quantized GEMM kernel (reusable across all formats)
 * 
 * Uses IBlockDecoder strategy to support IQ4_NL, Q6_K, Q8_0, etc.
 * without per-format kernel duplication.
 * 
 * Performance: 335-451 GFLOPS (matches format-specific fused implementations)
 */
class QuantizedGemmKernel : public ITensorGemm {
public:
    explicit QuantizedGemmKernel(const IBlockDecoder* decoder)
        : decoder_(decoder) {}
    
    bool multiply(const float* A, float* C,
                  int m, int n, int k,
                  bool transpose_B = true,
                  float alpha = 1.0f,
                  float beta = 0.0f,
                  const MPIContext* mpi_ctx = nullptr,
                  int device_idx = -1) override;
    
private:
    const IBlockDecoder* decoder_;  // Decode strategy (tensor provides)
    
    // Strategy selection
    bool multiply_cache_blocked(const float* A, float* C, int m, int n, int k, float alpha, float beta);
    bool multiply_row_wise(const float* A, float* C, int m, int n, int k, float alpha, float beta);
};
```

**File**: `src/v2/kernels/cpu/QuantizedGemm.cpp` (200 lines)

```cpp
bool QuantizedGemmKernel::multiply(...) {
    // Strategy selection based on batch size
    if (m >= 2 && m <= 16) {
        return multiply_cache_blocked(A, C, m, n, k, alpha, beta);
    } else {
        return multiply_row_wise(A, C, m, n, k, alpha, beta);
    }
}

bool QuantizedGemmKernel::multiply_cache_blocked(...) {
    const int num_k_blocks = (k + decoder_->block_size() - 1) / decoder_->block_size();
    
    #pragma omp parallel for
    for (int j = 0; j < n; ++j) {
        float acc[16] = {0};  // Max m=16
        
        for (int kb = 0; kb < num_k_blocks; ++kb) {
            alignas(64) float B_block[256];  // Max block size
            
            // CRITICAL: Inline decode call (zero overhead)
            decoder_->decode_block_at(j, kb, B_block);
            
            // Immediate reuse (hot in L1 cache)
            for (int i = 0; i < m; ++i) {
                acc[i] += dot_product_simd(A + i*k + kb*decoder_->block_size(), 
                                           B_block, 
                                           std::min(decoder_->block_size(), k - kb*decoder_->block_size()));
            }
        }
        
        // Write results
        for (int i = 0; i < m; ++i) {
            C[i*n + j] = alpha * acc[i] + beta * C[i*n + j];
        }
    }
    return true;
}
```

**Performance Pattern**:

- **Cache-Blocked** (m ∈ [2,16]): Decode 1 block → use immediately across all M rows
- **Row-Wise** (m > 16): Decode tiles of columns (64×32 optimal), amortize decode cost

#### Step 4: Extend to New Quantization Formats

**Adding Q6_K Support**:

```cpp
// tensors/Q6_KTensor.h
class Q6_KTensor : public TensorBase, public IBlockDecoder {
public:
    // Same IBlockDecoder implementation pattern
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const Q6_KBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeQ6KBlock(block, output);  // Q6_K-specific decode
    }
    
    size_t block_size() const override { return 256; }  // Q6_K: 256 elements/block
    
    // Factory: Same generic kernel!
    std::unique_ptr<ITensorGemm> createGemm() const override {
        return std::make_unique<QuantizedGemmKernel>(this);  // Reuse generic kernel
    }
    
private:
    static void decodeQ6KBlock(const Q6_KBlock& block, float* output);
};
```

**Result**: Single `QuantizedGemmKernel` works for IQ4_NL, Q6_K, Q8_0, etc. with zero code duplication.

#### Benefits Summary

| Aspect | Before (V1 Fused) | After (IBlockDecoder) |
|--------|-------------------|------------------------|
| **Code Reuse** | ~1000 lines per format | ~350 lines shared kernel |
| **Extensibility** | Reimplement GEMM for each format | Implement decode only |
| **Performance** | 335-451 GFLOPS | 335-451 GFLOPS (identical) |
| **Overhead** | N/A | Zero (inline devirtualization) |
| **Maintenance** | High (duplicated logic) | Low (single kernel) |

#### Verification

**Compiler Devirtualization Check**:
```bash
# Ensure decode_block_at is inlined (no virtual calls)
objdump -d build_v2/libllam2_core.a | grep -A 50 "multiply_cache_blocked"
# Should see: direct decode instructions, no call to vtable
```

**Performance Parity Test**:
```cpp
TEST(QuantizedGemm, IBlockDecoderZeroOverhead) {
    // Compare generic QuantizedGemmKernel vs format-specific fused kernel
    auto iq4nl = std::make_shared<IQ4_NLTensor>(shape);
    auto generic_gemm = std::make_unique<QuantizedGemmKernel>(iq4nl.get());
    auto fused_gemm = std::make_unique<IQ4_NLQuantizedGemm>(iq4nl.get());
    
    benchmark(generic_gemm);  // 357 GFLOPS
    benchmark(fused_gemm);    // 357 GFLOPS (identical)
}
```

---

### Adding New Pipelines

**Status**: ✅ **Simplified Pattern** (October 2025 - `initializeInfrastructure()`)

**Overview**: V2 pipelines follow a standardized construction pattern that separates architecture-specific setup from generic infrastructure initialization.

**Step 1**: Create pipeline class inheriting from PipelineBase

```cpp
// pipelines/llama/Llama3Pipeline.h
#include "../PipelineBase.h"
#include "../PipelineConfig.h"  // Runtime configuration

class Llama3Pipeline : public PipelineBase {
public:
    // Constructor signature (REQUIRED)
    Llama3Pipeline(
        ModelLoader& loader,
        DeviceManager& device_mgr,
        MPIContext& mpi_ctx,
        const PipelineConfig& config  // Runtime parameters
    );

    ~Llama3Pipeline() override = default;

    // PipelineBase interface
    bool forward(const int* tokens, int seq_len) override;
    const float* logits() const override;
    const char* architecture() const override { return "llama"; }

protected:
    bool transformer_layer(int layer_idx, int seq_len) override;

private:
    // LLaMA-specific architecture parameters (from GGUF metadata)
    int n_heads_ = 0;
    int head_dim_ = 0;
    int d_ff_ = 0;
    float rope_theta_ = 10000.0f;

    // Layer weights
    struct LayerWeights {
        std::shared_ptr<TensorBase> wq, wk, wv, wo;
        std::shared_ptr<TensorBase> gate_proj, up_proj, down_proj;
        std::shared_ptr<TensorBase> attn_norm, ffn_norm;
    };
    
    std::shared_ptr<TensorBase> embedding_table_;
    std::vector<LayerWeights> layers_;
    std::shared_ptr<TensorBase> final_norm_;
    std::shared_ptr<TensorBase> lm_head_;
    
    std::shared_ptr<FP32Tensor> current_hidden_;
    std::shared_ptr<FP32Tensor> logits_;
};
```

**Step 2**: Implement constructor with simplified initialization pattern

**CRITICAL**: Pipeline constructor should follow this exact pattern:
1. **Read architecture params** from GGUF metadata
2. **Allocate architecture-specific structures** (layers, weights, buffers)
3. **Call `initializeInfrastructure()`** to set up device placement, MPI, and KV cache

```cpp
// pipelines/llama/Llama3Pipeline.cpp
Llama3Pipeline::Llama3Pipeline(
    ModelLoader& loader,
    DeviceManager& device_mgr,
    MPIContext& mpi_ctx,
    const PipelineConfig& config
)
    : PipelineBase(loader, device_mgr, mpi_ctx, config)  // Initialize base (stores config_)
{
    const GGUFModel& model = loader_.getModel();
    
    // ========================================
    // STEP 1: Read Architecture Parameters
    // ========================================
    // These are model-specific constants from GGUF metadata
    // (NOT runtime configuration - that comes from config_)
    
    n_layers_ = model.block_count;
    n_heads_ = model.head_count;
    n_kv_heads_ = model.head_count_kv;  // For GQA support
    d_model_ = model.embedding_length;
    head_dim_ = d_model_ / n_heads_;
    d_ff_ = model.feed_forward_length;
    vocab_size_ = model.vocab_size;
    rope_theta_ = model.rope_freq_base;
    
    LOG_INFO("LLaMA 3 architecture: "
             << n_layers_ << " layers, "
             << n_heads_ << " heads, "
             << "d_model=" << d_model_);
    
    // ========================================
    // STEP 2: Allocate Architecture-Specific Structures
    // ========================================
    
    // Resize layer weight containers
    layers_.resize(n_layers_);
    
    // Load weights from GGUF into tensors
    embedding_table_ = loader_.getTensor("token_embd.weight");
    
    for (int i = 0; i < n_layers_; ++i) {
        std::string prefix = "blk." + std::to_string(i) + ".";
        
        layers_[i].wq = loader_.getTensor(prefix + "attn_q.weight");
        layers_[i].wk = loader_.getTensor(prefix + "attn_k.weight");
        layers_[i].wv = loader_.getTensor(prefix + "attn_v.weight");
        layers_[i].wo = loader_.getTensor(prefix + "attn_output.weight");
        
        layers_[i].gate_proj = loader_.getTensor(prefix + "ffn_gate.weight");
        layers_[i].up_proj = loader_.getTensor(prefix + "ffn_up.weight");
        layers_[i].down_proj = loader_.getTensor(prefix + "ffn_down.weight");
        
        layers_[i].attn_norm = loader_.getTensor(prefix + "attn_norm.weight");
        layers_[i].ffn_norm = loader_.getTensor(prefix + "ffn_norm.weight");
    }
    
    final_norm_ = loader_.getTensor("output_norm.weight");
    lm_head_ = loader_.getTensor("output.weight");
    
    // Allocate activation buffers
    current_hidden_ = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(config_.max_seq_len), static_cast<size_t>(d_model_)},
        DeviceId::CPU  // Will be overridden by device orchestration
    );
    
    logits_ = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(vocab_size_)},
        DeviceId::CPU
    );
    
    // ========================================
    // STEP 3: Generic Infrastructure Initialization (ONE LINE!)
    // ========================================
    // This replaces the old 15-line initialization block:
    //   - initializeDeviceInfrastructure(max_seq_len)
    //   - configureMPIStrategy()
    //   - initializeKVCache(max_seq_len)
    
    initializeInfrastructure();  // 🎯 Handles device detection, MPI, KV cache
    
    LOG_INFO("LLaMA 3 pipeline initialized successfully");
}
```

**What `initializeInfrastructure()` Does** (PipelineBase implementation):

```cpp
// src/v2/pipelines/PipelineBase.cpp
void PipelineBase::initializeInfrastructure() {
    int max_seq_len = config_.max_seq_len;  // Read from runtime config
    
    // Phase 3: Device detection from weight placement
    initializeDeviceInfrastructure(max_seq_len);
    // - Calls weight_placement_.detectAttentionDevices(n_layers_)
    // - Stores result in attention_devices_ vector
    // - Logs device placement summary
    
    // Phase 2: MPI strategy configuration
    configureMPIStrategy();
    // - Determines tensor-parallel strategy (auto, manual, disabled)
    // - Validates MPI context and rank/world_size
    // - Logs MPI configuration
    
    // Phase 1: KV cache allocation
    initializeKVCache(max_seq_len);
    // - Allocates k_cache_[i] and v_cache_[i] for each layer
    // - Uses attention_devices_[i] for device placement
    // - Initializes to zero
    // - Sets cache_position_ = 0
    
    LOG_INFO("Pipeline infrastructure initialized");
}
```

**Key Benefits of Simplified Pattern**:

1. **Code Reduction**: 93% less initialization code per pipeline (~15 lines → 1 line)
2. **Consistency**: All pipelines follow identical initialization sequence
3. **Maintainability**: Infrastructure changes propagate automatically to all pipelines
4. **Readability**: Constructor clearly shows architecture-specific vs generic setup
5. **Testing**: Base class handles complex initialization logic (single test location)

**Before (Old Pattern)**:
```cpp
Qwen2Pipeline::Qwen2Pipeline(...) : PipelineBase(...) {
    // Read architecture params...
    n_layers_ = model.block_count;
    // ...
    
    // Allocate structures...
    layers_.resize(n_layers_);
    // ...
    
    // DUPLICATED ACROSS ALL PIPELINES (15 lines):
    int max_seq_len = 2048;  // Hardcoded!
    
    // Detect devices from weight placement
    attention_devices_ = weight_placement_.detectAttentionDevices(n_layers_);
    std::unordered_map<DeviceId, int> device_counts;
    for (DeviceId device : attention_devices_) {
        device_counts[device]++;
    }
    LOG_INFO("Attention device placement:");
    for (const auto& [device, count] : device_counts) {
        LOG_INFO("  " << deviceIdToString(device) << ": " << count << " layers");
    }
    
    configureMPIStrategy();
    initializeKVCache(max_seq_len);
}
```

**After (New Pattern)**:
```cpp
Qwen2Pipeline::Qwen2Pipeline(..., const PipelineConfig& config)
    : PipelineBase(..., config)  // Store config
{
    // Read architecture params...
    n_layers_ = model.block_count;
    // ...
    
    // Allocate structures...
    layers_.resize(n_layers_);
    // ...
    
    // Generic initialization (ONE LINE):
    initializeInfrastructure();  // Uses config_.max_seq_len
}
```

**Step 3**: Register pipeline in factory

```cpp
// pipelines/PipelineFactory.cpp
#include "llama/Llama3Pipeline.h"

PipelineFactory::PipelineFactory() {
    // ... existing registrations ...
    
    registerPipeline(
        "llama",  // Architecture name from GGUF
        [](ModelLoader& loader, DeviceManager& device_mgr,
           MPIContext& mpi_ctx, const PipelineConfig& config) -> std::unique_ptr<PipelineBase> {
            return std::make_unique<Llama3Pipeline>(loader, device_mgr, mpi_ctx, config);
        }
    );
}
```

**Step 4**: Implement transformer_layer (architecture-specific logic)

```cpp
bool Llama3Pipeline::transformer_layer(int layer_idx, int seq_len) {
    // LLaMA 3 architecture:
    // 1. RMSNorm(x)
    // 2. Multi-head attention with RoPE (GQA variant)
    // 3. Residual: x = x + attn_out
    // 4. RMSNorm(x)
    // 5. SwiGLU FFN
    // 6. Residual: x = x + ffn_out
    
    const LayerWeights& layer = layers_[layer_idx];
    DeviceId device = get_attention_device(layer_idx);  // From Phase 1
    
    // === Attention Block ===
    auto normed = applyRMSNorm(current_hidden_, layer.attn_norm);
    
    auto Q = computeProjection(normed, layer.wq, device);
    auto K = computeProjection(normed, layer.wk, device);
    auto V = computeProjection(normed, layer.wv, device);
    
    applyRoPE(Q, K, layer_idx, rope_theta_);
    
    // Append to KV cache
    appendToCache(k_cache_[layer_idx], K, cache_position_);
    appendToCache(v_cache_[layer_idx], V, cache_position_);
    
    auto attn_out = computeAttention(Q, k_cache_[layer_idx], v_cache_[layer_idx]);
    auto attn_proj = computeProjection(attn_out, layer.wo, device);
    
    addResidual(current_hidden_, attn_proj);  // x = x + attn_out
    
    // === FFN Block ===
    normed = applyRMSNorm(current_hidden_, layer.ffn_norm);
    
    auto gate = computeProjection(normed, layer.gate_proj, device);
    auto up = computeProjection(normed, layer.up_proj, device);
    
    applySwiGLU(gate, up);  // gate = silu(gate) * up
    
    auto ffn_out = computeProjection(gate, layer.down_proj, device);
    
    addResidual(current_hidden_, ffn_out);  // x = x + ffn_out
    
    return true;
}
```

**Step 5**: Implement forward() for full inference

```cpp
bool Llama3Pipeline::forward(const int* tokens, int seq_len) {
    // Embedding lookup
    embedding(tokens, seq_len, current_hidden_.get());
    
    // Transformer layers
    for (int i = 0; i < n_layers_; ++i) {
        if (!transformer_layer(i, seq_len)) {
            return false;
        }
    }
    
    // Final norm + LM head
    auto normed = applyRMSNorm(current_hidden_, final_norm_);
    auto output = computeProjection(normed, lm_head_, DeviceId::CPU);
    
    // Extract logits for last token
    copyLastToken(output, logits_);
    
    // Update cache position
    cache_position_ += seq_len;
    
    return true;
}

const float* Llama3Pipeline::logits() const {
    return static_cast<const float*>(logits_->data());
}
```

**Best Practices**:

1. **Separation of Concerns**:
   - Constructor: Read metadata + allocate structures + call `initializeInfrastructure()`
   - `transformer_layer()`: Architecture-specific computation logic
   - `forward()`: Orchestrate layers + embedding + final projection

2. **Device Awareness**:
   - Use `get_attention_device(layer_idx)` for device-specific operations
   - Transfer activations between devices as needed
   - KV cache automatically placed on correct device (Phase 1)

3. **Runtime Configuration**:
   - Access via `config_.max_seq_len`, `config_.n_threads`, etc.
   - Never hardcode context size or threading parameters
   - Allow CLI overrides

4. **Weight Loading**:
   - Use ModelLoader::getTensor() for GGUF weight extraction
   - Respect device placement from WeightPlacementMap (Phase 2)
   - Validate tensor shapes match architecture expectations

5. **Error Handling**:
   - Validate architecture metadata on construction
   - Check kernel execution return values
   - Provide meaningful error messages with context

**Common Pitfalls to Avoid**:

❌ **DON'T**: Hardcode max_seq_len
```cpp
int max_seq_len = 2048;  // BAD: Ignores config
```

✅ **DO**: Read from config
```cpp
int max_seq_len = config_.max_seq_len;  // GOOD: Respects runtime setting
```

❌ **DON'T**: Manually initialize device infrastructure
```cpp
// BAD: Duplicates base class logic
attention_devices_ = weight_placement_.detectAttentionDevices(n_layers_);
configureMPIStrategy();
initializeKVCache(max_seq_len);
```

✅ **DO**: Call initializeInfrastructure()
```cpp
// GOOD: Uses base class method
initializeInfrastructure();
```

❌ **DON'T**: Forget to register in factory
```cpp
// BAD: Pipeline exists but can't be created
// (factory doesn't know about it)
```

✅ **DO**: Add factory registration
```cpp
// GOOD: Architecture name maps to creator lambda
registerPipeline("llama", [...](auto&&...) { return std::make_unique<Llama3Pipeline>(...); });
```

**Testing Checklist**:

- [ ] Factory can create pipeline with valid GGUF model
- [ ] Constructor reads architecture metadata correctly
- [ ] Device placement matches WeightPlacementMap
- [ ] KV cache allocated with correct shapes and devices
- [ ] MPI strategy configured appropriately (if using MPI)
- [ ] Forward pass produces expected logits shape
- [ ] Custom context size (`--ctx-size`) respected
- [ ] Multi-layer execution succeeds
- [ ] Memory cleanup (no leaks)

**Example Test**:
```cpp
// tests/v2/unit/pipelines/Test__Llama3Pipeline.cpp
TEST(Llama3Pipeline, ConstructionWithConfig) {
    // Create config with custom context size
    PipelineConfig config;
    config.max_seq_len = 4096;  // Override default 2048
    
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel("models/llama-3-8b-q4_0.gguf"));
    
    DeviceManager device_mgr;
    MPIContext mpi_ctx{0, 1, MPI_COMM_WORLD};
    
    // Construct pipeline
    auto pipeline = std::make_unique<Llama3Pipeline>(
        loader, device_mgr, mpi_ctx, config
    );
    
    // Verify KV cache uses custom context size
    EXPECT_EQ(pipeline->kCacheShape(0)[0], 4096);  // max_seq_len dimension
}
```

---

**Step 6 (Optional)**: Add architecture-specific optimizations

    // Load embedding, layer weights, final norm, LM head
    // (similar pattern to Qwen2Pipeline)
    
    return true;
}

bool LlamaPipeline::transformer_layer(int layer_idx, int seq_len) {
    // LLaMA-specific architecture:
    // 1. RMSNorm
    // 2. Multi-head attention with RoPE
    // 3. Residual connection
    // 4. RMSNorm
    // 5. SwiGLU FFN
    // 6. Residual connection
    
    const LayerWeights& layer = layers_[layer_idx];
    
    // Attention block
    // ... (orchestrate kernels directly)
    
    // FFN block
    // ... (orchestrate kernels directly)
    
    return true;
}

bool LlamaPipeline::forward(const int* tokens, int seq_len) {
    // 1. Embedding lookup
    // 2. For each layer: transformer_layer(i, seq_len)
    // 3. Final norm + LM head
    return true;
}

const float* LlamaPipeline::logits() const {
    return logits_ ? logits_->data() : nullptr;
}
```

**Step 3**: Register in factory (future - when factory is implemented)

```cpp
PipelineFactory::registerPipeline("llama", [](const std::string& path, auto ctx, int dev) {
    return std::make_unique<LlamaPipeline>(path, ctx, dev);
});
```

### Testing New Components

**Unit Tests** (kernel-level):

```cpp
TEST(IQ4_NLTensor, FusedGemmCorrectness) {
    auto iq4nl = std::make_shared<IQ4_NLTensor>(shape, DeviceId::CPU);
    iq4nl->populateQuantized(reference_weights);
    
    auto gemm = iq4nl->getGemmKernel();
    auto output = allocateTensor({M, N});
    
    gemm->execute(input, iq4nl, output, {});
    
    // Compare with reference OpenBLAS GEMM
    auto reference = openblasGemm(input, dequantized_weights);
    EXPECT_LT(relativeL2(output, reference), 1e-3f);  // Quantization tolerance
}
```

**Integration Tests** (pipeline-level):

```cpp
TEST(QwenPipeline, ForwardPassCorrectness) {
    QwenPipeline pipeline(config, mpi_ctx);
    auto output = allocateTensor({seq_len, vocab_size});
    
    bool success = pipeline.forward(tokens, output.get());
    EXPECT_TRUE(success);
    
    // Compare with PyTorch reference
    auto pytorch_logits = loadPyTorchReference();
    EXPECT_LT(maxAbsDiff(output, pytorch_logits), 1e-2f);
}
```

---

## Migration from V1

### Code Reduction

**V1 Codebase** (~18,000 lines):
- Operators: 5,000 lines (MPILinearOperator, MPIAttentionOperator, etc.)
- Kernels: 8,000 lines (implementations)
- Pipelines: 3,000 lines (QwenPipeline, LlamaPipelineAdapter)
- Infrastructure: 2,000 lines (factories, providers, etc.)

**V2 Codebase** (~3,200 lines):
- Kernels: 1,708 lines (IQ4_NL + interfaces)
- Pipelines: 112 lines (QwenPipeline)
- Infrastructure: 1,380 lines (DeviceManager, MPIContext, utils)

**Total Reduction**: **~82% code elimination** (14,800 lines removed)

### Key Architectural Changes

| V1 Concept | V2 Replacement | Migration Path |
|------------|----------------|----------------|
| `MPILinearOperator` | Direct `ITensorGemm` calls | Remove operator, call kernel from pipeline |
| `MPIAttentionOperator` | Direct `ITensorAttention` calls | Remove operator, orchestrate in pipeline |
| `MPIEmbeddingOperator` | Direct embedding lookup | Simple memcpy in pipeline |
| `PrefillProvider` | Pipeline `forward()` method | Merge provider logic into pipeline |
| `ModelWeightsProvider` | Simple weight map in pipeline | Remove provider abstraction |
| Per-rank MPI slicing | Runtime partitioning in pipeline | Move slicing logic to pipeline |

### Migration Example: Attention

**V1** (Operator-Based):

```cpp
// V1: Heavy operator abstraction
auto attn_operator = std::make_unique<MPIAttentionOperator>(config, mpi_ctx);
attn_operator->registerKernel("attention", attention_kernel);

std::vector<std::shared_ptr<TensorBase>> inputs = {
    input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache
};
std::vector<std::shared_ptr<TensorBase>> outputs = {attn_out, updated_k, updated_v};

bool success = attn_operator->execute(inputs, outputs);
```

**V2** (Direct Orchestration):

```cpp
// V2: Direct kernel calls
auto device = device_mgr_.selectDevice(input->size_bytes(), OperationType::ATTENTION);
auto gemm = device->getGemmKernel();
auto rope = device->getRoPEKernel();
auto attn = device->getAttentionKernel();

// Q/K/V projections
auto Q = allocateTensor({seq_len, d_model});
gemm->execute(input, wq, Q, {});
rope->execute(Q, {...});

auto K = allocateTensor({seq_len, d_model_kv});
gemm->execute(input, wk, K, {});
rope->execute(K, {...});

auto V = allocateTensor({seq_len, d_model_kv});
gemm->execute(input, wv, V, {});

// Attention
auto attn_out = allocateTensor({seq_len, d_model});
attn->execute(Q, K, V, attn_out, {...});

// Output projection
gemm->execute(attn_out, wo, output, {});
```

**Benefits**:
- ✅ **50% fewer lines** (eliminate operator boilerplate)
- ✅ **Explicit control flow** (easier to understand)
- ✅ **Fusion opportunities** (combine Q/K/V projections)
- ✅ **Runtime device selection** (heterogeneous execution)

---

## Future Roadmap

### Phase 1: Core Infrastructure

**Status**: ✅ **Complete** (October 23, 2025)

- [x] V2 folder structure and namespaces
- [x] TensorBase interface with device placement
- [x] DeviceManager and ComputeContext abstractions
- [x] IQ4_NL tensor migration (1708 lines, performance preserved)
- [x] All utility classes (MPIContext, CPUFeatures, DebugEnv, SIMDHelpers)
- [x] File reorganization (tensor utilities in tensors/)
- [x] BF16 support in SIMDHelpers

### Phase 2: Device Orchestration & Model Loading

**Status**: ✅ **Complete** (October 24, 2025)

- [x] `DeviceOrchestrator` with 5 placement strategies (ALL_CPU, ALL_GPU, MEMORY_AWARE, MOE_OPTIMIZED, CUSTOM)
- [x] `WeightPlacementMap` for weight→device mapping
- [x] `ArgParser` for CLI argument parsing (27 tests)
- [x] Device map parsing (layer ranges, percentages, patterns)
- [x] Memory-aware placement (auto-fit layers within budget)
- [x] MoE-optimized placement (shared→GPU, sparse→CPU)
- [x] Custom placement (user-defined device maps)
- [x] `ModelLoader` for GGUF loading (48 tests)
- [x] Comprehensive test suite (25 DeviceOrchestrator tests, 100% passing)

**Key Achievements**:
- ✅ Flexible device placement strategies for different deployment scenarios
- ✅ Automatic memory budget optimization (MEMORY_AWARE)
- ✅ MoE-specific optimizations (pattern-based expert placement)
- ✅ Advanced device map syntax (3 rule types: ranges, percentages, patterns)
- ✅ Full test coverage (25 tests validating all strategies)

### Phase 3: CPU Backend (Next)

**Status**: 🔄 **Next Sprint**

- [ ] Implement `CPUComputeContext`
- [ ] Create `OpenBLASGemm` kernel wrapper
- [ ] Port attention primitives (RoPE, softmax, causal mask)
- [ ] Implement `QwenPipeline::forward()` (full prefill path)
- [ ] Port benchmark suite to V2
- [ ] Validate GFLOPS match V1 baseline

**Target**: Functional CPU-only inference with parity to V1

### Phase 4: CUDA Backend (GPU Support)

**Status**: 📋 **Planned Q1 2026**

- [ ] Implement `CudaComputeContext` (device management, memory allocation)
- [ ] Create `CudaGemmKernel` (cuBLAS wrapper)
- [ ] Port IQ4_NL fused dequant to CUDA (CUDA C++ kernel)
- [ ] Implement `CudaAttentionKernel` (consider FlashAttention integration)
- [ ] Add tensor data movement (CPU ↔ CUDA)
- [ ] Benchmark CUDA vs CPU performance

**Target**: Heterogeneous CPU+CUDA execution

### Phase 4: CUDA Backend (GPU Support)

**Status**: 📋 **Planned Q1 2026**

- [ ] Implement `CudaComputeContext` (device management, memory allocation)
- [ ] Create `CudaGemmKernel` (cuBLAS wrapper)
- [ ] Port IQ4_NL fused dequant to CUDA (CUDA C++ kernel)
- [ ] Implement `CudaAttentionKernel` (consider FlashAttention integration)
- [ ] Add tensor data movement (CPU ↔ CUDA)
- [ ] Benchmark CUDA vs CPU performance

**Target**: Heterogeneous CPU+CUDA execution

### Phase 5: ROCm Backend (AMD GPU)

**Status**: 📋 **Planned Q2 2026**

- [ ] Implement `RocmComputeContext`
- [ ] Create `RocmGemmKernel` (rocBLAS wrapper)
- [ ] Port kernels to HIP (ROCm equivalent of CUDA)
- [ ] Test on AMD MI100/MI200 series
- [ ] Benchmark ROCm vs CUDA vs CPU

**Target**: Full AMD GPU support

### Phase 6: Vulkan Backend (Portable Compute)

**Status**: 📋 **Planned H2 2026**

- [ ] Implement `VulkanComputeContext`
- [ ] Write compute shaders for kernels (GLSL → SPIR-V)
- [ ] Create `VulkanGemmKernel` using compute pipelines
- [ ] Test on Intel/NVIDIA/AMD/Apple GPUs
- [ ] Validate cross-platform compatibility

**Target**: Universal GPU support (including macOS, mobile)

### Phase 7: Production Features

**Status**: 📋 **Planned Q3 2026**

- [ ] Build system integration (CMakeLists.txt for llaminar2)
- [ ] Main.cpp entry point with CLI argument parsing
- [ ] Model loading from GGUF (migrate V1 ModelLoader)
- [ ] LlamaPipeline implementation (alternative architecture)
- [ ] Comprehensive test suite (parity with V1)
- [ ] Benchmark suite (performance tracking)
- [ ] Documentation (user guide, API reference)

**Target**: Production-ready release of Llaminar V2

### Research Directions (Future)

**Kernel Fusion**:
- Fused QKV projection (single GEMM call)
- Fused attention (Q@K + softmax + @V in one kernel)
- Fused FFN (gate + up + SwiGLU + down)

**Advanced Quantization**:
- INT8 quantization for activations
- FP8 support (H100/MI300 series)
- Mixed-precision execution (FP16 activations, quantized weights)

**Distributed Execution**:
- MPI-aware tensor parallel (inter-node weight sharding)
- Pipeline parallel (layer-wise distribution)
- Hybrid TP+PP for very large models

**Performance Optimization**:
- Asynchronous execution (overlap compute + data transfer)
- Kernel auto-tuning (tile size, thread block configuration)
- Memory pooling (reduce allocation overhead)

---

## Conclusion

Llaminar V2 represents a **radical simplification** of the inference architecture:

**Key Achievements**:
- ✅ **82% code reduction** (18,000 → 3,200 lines for core architecture)
- ✅ **Operator elimination** (direct kernel orchestration)
- ✅ **Multi-GPU foundation** (per-tensor device affinity)
- ✅ **Performance preserved** (+41% IQ4_NL FP32, +26% BF16)
- ✅ **Clean abstractions** (TensorBase, ITensorGemm, DeviceManager)
- ✅ **Device orchestration** (5 placement strategies: ALL_CPU, ALL_GPU, MEMORY_AWARE, MOE_OPTIMIZED, CUSTOM)
- ✅ **Advanced placement** (memory-aware auto-fitting, MoE optimization, custom device maps)
- ✅ **Comprehensive testing** (25 orchestrator tests + 27 argparser tests + 48 loader tests = 100 tests, 100% passing)

**Phase 2 Complete** (October 24, 2025):
- ✅ DeviceOrchestrator with flexible placement strategies
- ✅ WeightPlacementMap for weight→device assignments  
- ✅ ArgParser for CLI argument parsing
- ✅ Device map parsing (layer ranges, percentages, patterns)
- ✅ ModelLoader integration with device affinity
- ✅ Full test coverage validating all strategies

**Next Steps**:
1. Implement `CPUComputeContext` (Phase 3)
2. Port `QwenPipeline::forward()` to V2 architecture
3. Validate GFLOPS parity with V1 baseline
4. CUDA backend for GPU acceleration (Phase 4)
5. Production deployment (Phase 7)

**Philosophy**: **Simplicity over abstraction. Performance over generality. Directness over indirection.**

---

**End of V2 Architecture Documentation**

*For questions or contributions, see `src/v2/README.md` or contact the development team.*  
*Last updated: October 24, 2025 - Phase 2 Device Orchestration Complete*
