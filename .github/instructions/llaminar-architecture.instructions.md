# Llaminar LLM Inference Engine - Architecture Documentation

*Last Updated: September 19, 2025*

## Overview

Llaminar is a high-performance, distributed LLM inference engine built on a modular object-oriented architecture. It features a **hybrid tensor system** that provides zero-copy COSMA optimization while maintaining full backward compatibility. The engine combines COSMA's high-performance matrix multiplication with GGUF model support, MPI distributed computing, and comprehensive system topology detection to create a scalable inference platform.

## Core Design Principles

1. **Modular Architecture**: Each component has a single responsibility and clear interfaces
2. **Hybrid Tensor System**: Zero-copy COSMA optimization with backward compatibility
3. **Distributed Computing**: Built-in MPI support for multi-node inference
4. **High Performance**: COSMA integration for optimized matrix operations
5. **System Awareness**: Comprehensive CPU, NUMA, and GPU topology detection
6. **Extensibility**: Plugin-based kernel registration system
7. **Observability**: Multi-level logging and performance profiling

## Architecture Components

### 1. Entry Point & Orchestration

**File**: `src/main.cpp` (219 lines)
- **Purpose**: Application entry point and execution orchestration
- **Key Features**:
  - MPI initialization with thread support
  - 7-stage execution pipeline
  - Exception handling and graceful shutdown
  - Performance measurement and reporting

**Execution Pipeline**:
1. Command line argument parsing
2. Logging system initialization  
3. System topology detection
4. Kernel manager initialization
5. Model loading (if specified)
6. Compute graph execution
7. Performance reporting and cleanup

### 2. Command Line Interface

**Files**: `src/argument_parser.h/cpp`
- **Class**: `ArgumentParser`
- **Structure**: `LlaminarParams`
- **Features**:
  - POSIX-style argument parsing (`-v`, `--verbose`, etc.)
  - Multi-level verbosity (`-v`, `-vv`, `-vvv`)
  - Model file specification (`-m`, `--model`)
  - System configuration flags
  - Legacy COSMA parameters
  - Comprehensive help and version information

**Supported Arguments**:
```bash
# Model and Logging
-m, --model <file>     # GGUF model file path
-v/-vv/-vvv           # Verbosity levels (INFO/DEBUG/TRACE)

# System Configuration  
--print-topology      # Display system topology
--enable-hyperthreading # Use HT cores
--detect-gpus         # Enable GPU detection

# Performance
--profile             # Enable kernel profiling
--validate            # Enable result validation
--matrix-size <size>  # Matrix dimensions for benchmarks
--repeat <num>        # Benchmark iterations
```

### 3. Logging System

**Files**: `src/logger.h`, `src/log_level.h`
- **Pattern**: Header-only singleton with macros
- **Levels**: ERROR(0), WARN(1), INFO(2), DEBUG(3), TRACE(4)
- **Features**:
  - Timestamped output with millisecond precision
  - File and line number tracking
  - Environment variable configuration
  - Thread-safe singleton pattern
  - Convenient macros (`LOG_INFO`, `LOG_DEBUG`, etc.)

**Usage Example**:
```cpp
LOG_INFO("Model loaded successfully");
LOG_DEBUG("Processing iteration " << i << "/" << total);
LOG_ERROR("Failed to load model: " << filename);
```

### 4. System Topology Detection

**Files**: `src/topology_manager.h/cpp`, `src/common.h/cpp`
- **Classes**: `TopologyManager`, `CPUTopology`, `SystemTopology`
- **Features**:
  - CPU architecture detection (sockets, cores, hyperthreading)
  - NUMA topology mapping
  - GPU device enumeration (CUDA/ROCm support planned)
  - Memory capacity reporting
  - MPI environment information

**Detection Capabilities**:
- **CPU**: Socket count, cores per socket, hyperthreading status
- **NUMA**: Node count, CPU affinity, memory distribution  
- **Memory**: Total and per-NUMA-node capacity
- **GPU**: Device enumeration (framework-agnostic detection)
- **MPI**: Rank, size, process distribution

**Example Output**:
```
=== CPU Topology ===
Total CPUs: 112, Physical cores: 56, Sockets: 2
Hyperthreading: Yes, Using: No

=== NUMA Topology ===  
Node 0: 56 CPUs, 376 GB memory
Node 1: 56 CPUs, 377 GB memory
```

### 6. Hybrid Tensor Architecture

**Files**: `src/tensors/tensor_base.h`, `src/tensors/tensor_factory.cpp`, `src/tensor.h`
- **Classes**: `TensorBase`, `SimpleTensor`, `COSMATensor`, `TensorFactory`
- **Pattern**: Abstract interface with concrete implementations
- **Purpose**: Zero-copy COSMA optimization with legacy compatibility

**Key Components**:

#### TensorBase Abstract Interface
```cpp
class TensorBase {
public:
    virtual const std::vector<int>& shape() const = 0;
    virtual float* data() = 0;
    virtual const float* data() const = 0;
    virtual std::string type_name() const = 0;
    virtual bool is_distributed() const = 0;
};
```

#### SimpleTensor (Legacy Compatibility)
- **Purpose**: Zero-copy wrapper around existing `Tensor` struct
- **Data Storage**: Standard `std::vector<float>`
- **Use Cases**: Small matrices, single-process operations, legacy code
- **Performance**: No overhead compared to original Tensor

#### COSMATensor (COSMA Optimization)
- **Purpose**: Direct integration with `cosma::CosmaMatrix<float>`
- **Data Storage**: COSMA-optimized distributed layout
- **Use Cases**: Large matrices (≥256×256), multi-process operations
- **Performance**: Zero-copy access to COSMA operations

#### TensorFactory (Smart Selection)
- **Auto-Selection Logic**:
  - **Matrix Size**: COSMATensor for matrices ≥256×256 elements
  - **MPI Context**: COSMATensor when multiple processes available
  - **Operation Type**: COSMATensor preferred for matmul/attention
  - **Fallback**: SimpleTensor for compatibility

**Usage Examples**:
```cpp
// Automatic tensor selection
auto tensor = TensorFactory::create_auto({1024, 1024});
// → COSMATensor in MPI environment

// Explicit type creation
auto simple = TensorFactory::create_simple({512, 512});
auto cosma = TensorFactory::create_cosma({2048, 2048}, "matmul_A", mpi_rank);

// Legacy compatibility
std::shared_ptr<Tensor> legacy = std::make_shared<Tensor>({256, 256});
auto upgraded = TensorFactory::from_tensor(legacy);
```

**Performance Benefits**:
- **Zero-Copy COSMA**: Direct `cosma::multiply()` calls without data copying
- **Automatic Optimization**: Large matrices automatically use COSMA layout
- **Legacy Compatibility**: No performance penalty for existing code
- **Smart Conversion**: Only copies data when necessary

### 6. Compute Graph Engine

**Files**: `src/graph_compute.h/cpp`, `src/tensor.h`
- **Classes**: `ComputeNode`, `MatMulNode`, `ComputeGraph`
- **Pattern**: Composite pattern with execution engine
- **Features**:
  - Node-based computation representation
  - **Tensor Integration**: Uses legacy `llaminar::Tensor` with hybrid bridge
  - Dependency tracking and scheduling
  - Operator overloading for graph construction
  - Performance measurement per node

**Hybrid Tensor Bridge**:
- `GraphTensorBridge::optimize_for_kernel()`: Converts legacy Tensor to optimal TensorBase
- `GraphTensorBridge::auto_upgrade()`: Zero-copy promotion for large matrices
- Seamless integration between graph system and optimized kernels

**Node Hierarchy**:
```cpp
ComputeNode (abstract base)
├── MatMulNode (matrix multiplication)
└── [Future nodes: TransformerBlock, Attention, etc.]
```

**Usage Pattern**:
```cpp
auto graph = ComputeGraph();
auto matmul = std::make_shared<MatMulNode>("benchmark", m, n, k);
graph.addNode(matmul);
graph.execute();
```

### 7. Matrix Operations

**Files**: `src/kernels/MatMulKernel.h/cpp`
- **Class**: `MatMulKernel`
- **Integration**: Hybrid tensor system with COSMA optimization
- **Features**:
  - **Zero-Copy COSMA Path**: Direct operations on COSMATensor inputs
  - **Legacy Compatibility**: Automatic tensor conversion for SimpleTensor
  - High-performance distributed matrix multiplication
  - Automatic COSMA configuration
  - MPI-aware execution
  - Performance monitoring

**Hybrid Execution Paths**:
```cpp
// Zero-copy COSMA execution (optimal)
if (canUseZeroCopyCOSMA(inputs, outputs)) {
    auto cosma_A = std::dynamic_pointer_cast<COSMATensor>(A);
    success = executeCOSMANative(cosma_A->cosma_matrix(), ...);
} else {
    // Legacy path with data copying
    success = executeCOSMA(*A, *B, *C);
}
```

**Performance Optimization**:
- **Zero-Copy Operations**: COSMATensor → `cosma::multiply()` with no data copying
- **Automatic Detection**: `canUseZeroCopyCOSMA()` checks for optimal execution path
- **Fallback Support**: Legacy path maintains compatibility with existing code
- **COSMA Integration**: Leverages COSMA's optimized PDGEMM implementation

### 8. Model Loading System

**Files**: `src/model_loader.h/cpp`
- **Class**: `ModelLoader`
- **Format**: GGUF (GPT-Generated Unified Format)
- **Features**:
  - GGUF file parsing and validation
  - Metadata extraction (architecture, parameters)
  - **Hybrid Tensor Creation**: Automatic selection of optimal tensor types
  - Tensor loading with format conversion
  - Quantization support (Q8_0 implemented)

**GGUF Support**:
- **Architectures**: Qwen2.5, LLaMA family support
- **Quantization**: Q8_0, F16, F32 (extensible)
- **Metadata**: Model parameters, tokenizer info, training details
- **Validation**: Magic number verification, version compatibility
- **Tensor Optimization**: Large weight matrices automatically use COSMATensor

### 9. Data Format Conversion

**Files**: `src/repacker.h/cpp`
- **Class**: `Repacker`
- **Purpose**: Convert between GGUF and hybrid tensor formats
- **Features**:
  - Memory layout transformation
  - Type conversion (F16 ↔ F32, quantized formats)
  - **Tensor Type Selection**: Automatic SimpleTensor vs COSMATensor choice
  - Efficient memory management
  - Distributed data placement

### 10. Build System & Dependencies

**File**: `CMakeLists.txt`
- **Pattern**: Modern CMake with submodules
- **Library Structure**: Core library + executables
- **Dependencies**:
  - **COSMA**: High-performance matrix operations
  - **GGML/LLaMA.cpp**: Model format support and inference kernels
  - **MPI**: Distributed computing (OpenMPI)
  - **OpenMP**: Shared-memory parallelism
  - **NUMA**: Memory affinity management
  - **CUDA/ROCm**: GPU acceleration (optional)

**Build Targets**:
```bash
llaminar_core    # Core library with all components
llaminar         # Main executable
test_*          # Unit test executables
```

## Data Flow Architecture

### Inference Pipeline (Planned)

1. **Model Loading**: GGUF → Parsed tensors → Distributed placement
2. **Input Processing**: Tokenization → Embeddings → Attention preparation  
3. **Forward Pass**: Transformer blocks → Matrix operations → Activations
4. **Output Generation**: Logits → Sampling → Token generation
5. **Result Collection**: Distributed gather → Response formatting

### Current State (Matrix Benchmarking)

1. **Initialization**: MPI setup → Topology detection → Kernel registration
2. **Graph Construction**: MatMul nodes → Dependency resolution
3. **Execution**: COSMA operations → Performance measurement
4. **Reporting**: Timing collection → GFLOPS calculation

## Performance Characteristics

### Scalability
- **MPI Distributed**: Multi-node execution with process-based parallelism
- **NUMA Aware**: Memory locality optimization for large systems
- **Thread Parallel**: OpenMP integration for shared-memory scaling

### Memory Management
- **Hybrid Tensor System**: Zero-copy optimization with backward compatibility
- **Distributed Tensors**: Automatic sharding across MPI ranks (COSMATensor)
- **Format Optimization**: COSMA layout for optimal cache utilization
- **Legacy Compatibility**: SimpleTensor maintains existing memory patterns
- **Smart Allocation**: TensorFactory selects optimal memory layout
- **Quantization**: Reduced precision for memory efficiency

### Compute Optimization
- **COSMA Integration**: State-of-the-art matrix multiplication algorithms
- **Kernel Registration**: Pluggable optimization for different operations
- **Topology Awareness**: Hardware-specific optimizations

## Testing Infrastructure

**Directory**: `tests/`
- **test_basic.cpp**: MPI initialization and basic functionality
- **test_numa.cpp**: NUMA topology detection and affinity
- **test_cosma.cpp**: Matrix multiplication and COSMA integration
- **test_graph.cpp**: Compute graph construction and execution

**Test Coverage**:
- Component initialization and configuration
- MPI communication and coordination
- System topology detection accuracy
- Matrix operation correctness and performance

## Development Patterns

### Error Handling
- Exception-based error propagation
- MPI-aware error coordination
- Graceful degradation for optional features
- Comprehensive logging for debugging

### Memory Management
- RAII patterns for resource management
- Smart pointers for automatic cleanup
- MPI memory coordination
- NUMA-aware allocation strategies

### Extensibility Points
- **Kernel Registration**: Add new operations via inheritance
- **Model Formats**: Extend ModelLoader for new formats
- **Topology Detection**: Platform-specific detection modules
- **Communication**: Custom MPI communication patterns

## Configuration & Environment

### Environment Variables
```bash
LLAMINAR_LOG_LEVEL    # Override default log level
MPI_THREAD_MULTIPLE   # Enable MPI threading support
OMP_NUM_THREADS       # Control OpenMP parallelism
CUDA_VISIBLE_DEVICES  # GPU device selection
```

### CMake Configuration
```bash
-DCMAKE_BUILD_TYPE=Debug|Release
-DENABLE_CUDA=ON|OFF
-DENABLE_ROCM=ON|OFF
-DCOSMA_SCALAPACK_LINK_LIBRARIES=<path>
```

## Future Architecture Enhancements

### Planned Components
1. **Attention Kernels**: Multi-head attention with hybrid tensor optimization
2. **Transformer Blocks**: Complete layer implementations using TensorBase interface
3. **Advanced Tensor Operations**: Extend hybrid system to all kernel types
4. **Communication Layer**: Optimized MPI communication patterns
5. **Inference Engine**: Complete LLM inference pipeline with zero-copy optimization
6. **Model Zoo**: Pre-trained model repository integration

### Performance Optimizations
1. **GPU Acceleration**: CUDA/ROCm kernel implementations
2. **Mixed Precision**: FP16/BF16 computation paths
3. **Pipeline Parallelism**: Layer-wise distribution
4. **Tensor Parallelism**: Within-layer distribution
5. **Memory Optimization**: KV-cache management

### Scalability Improvements
1. **Dynamic Load Balancing**: Adaptive work distribution
2. **Hierarchical Communication**: Optimized MPI topologies
3. **Asynchronous Execution**: Overlapped computation/communication
4. **Elastic Scaling**: Runtime process addition/removal

## Canonical Runtime Configuration

### Optimal Launch Settings

Llaminar includes a canonical launch script with empirically-optimized settings:

```bash
# Always use the canonical launcher
./run-llaminar.sh [arguments]

# The script automatically configures:
# - OpenMP: Auto-detected cores per socket, socket placement, close binding
# - MPI: 1 process per socket, memory pinning, NUMA-aware binding  
# - Threading: Single-threaded for small ops, multi-threaded for medium, distributed for large
# - Topology Detection: Mirrors C++ logic from src/common.cpp for consistent results
```

### Environment Configuration

**OpenMP Settings** (automatically configured):
```bash
OMP_NUM_THREADS=<detected>       # Auto-detected physical cores per socket
OMP_PLACES=sockets               # Thread placement strategy
OMP_PROC_BIND=close              # Bind threads close together
KMP_AFFINITY=granularity=fine,compact,1,0  # Intel threading
KMP_BLOCKTIME=0                  # Minimize thread blocking
```

**MPI Settings** (automatically configured):
```bash
OMPI_MCA_mpi_leave_pinned=1                     # Memory pinning
OMPI_MCA_btl_vader_single_copy_mechanism=none   # NUMA optimization
OMPI_MCA_btl_openib_allow_ib=1                  # InfiniBand support
```

### System Requirements

- **CPU**: Multi-socket x86_64 with NUMA support
- **Memory**: 16GB+ RAM, preferably balanced across NUMA nodes
- **MPI**: OpenMPI 4.0+ with thread support
- **OpenMP**: libgomp or Intel OpenMP runtime
- **Optimal**: 2-socket system, 1 MPI process per socket

## Usage Examples

### Basic Execution
```bash
# Topology detection and system info
./run-llaminar.sh -v --print-topology

# Performance benchmarking
./run-llaminar.sh -vv --matrix-size 2048

# GPU detection and profiling
./run-llaminar.sh --detect-gpus --profile --trace
```

### Model Inference
```bash
# Load and run inference with Qwen 2.5 model
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v

# Verbose inference with profiling
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -vv --profile
```

### Advanced Usage (Manual MPI)
```bash
# Manual MPI execution (if canonical script unavailable)
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 --report-bindings \
  ./build/llaminar -m models/qwen2.5-0.5b-instruct-q4_0.gguf
```

This architecture provides a solid foundation for high-performance, distributed LLM inference while maintaining modularity, extensibility, and observability throughout the system.