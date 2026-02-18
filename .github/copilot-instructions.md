# Llaminar V2 Project Development Guidelines

This document provides practical guidelines for working with the **Llaminar V2** LLM inference engine, including build processes, testing, debugging, and kernel / MPI / attention development best practices.

**Architecture Note (V2)**: The active architecture is **Llaminar V2** in `src/v2/`, an operator-free, kernel-centric design with **GraphOrchestrator** as the sole execution path.

- For a **high-level architecture map** of tensors, kernels, attention, MPI orchestration, and graph execution, see:
    - `.github/instructions/llaminar-architecture-v2.instructions.md`
- For additional V2-specific implementation details, see:
    - `.github/instructions/llaminar-v2-architecture.instructions.md`

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Build System](#build-system)
- [Canonical Runtime Configuration](#canonical-runtime-configuration)
- [Benchmark Mode](#benchmark-mode)
- [Testing Guidelines](#testing-guidelines)
- [Debugging](#debugging)
- [Stage Dump Framework](#stage-dump-framework)
- [Stage Output Print Facility](#stage-output-print-facility)
- [Parity Testing (PyTorch Reference)](#parity-testing-pytorch-reference)
- [Kernel Development](#kernel-development)
- [GPU Tensor Coherence](#gpu-tensor-coherence)
- [Weight Sharding and Tensor Parallelism](#weight-sharding-and-tensor-parallelism)
- [MPI Development Best Practices](#mpi-development-best-practices)
- [Performance Optimization](#performance-optimization)
- [Code Quality Guidelines](#code-quality-guidelines)
- [Environment Variables Reference](#environment-variables-reference)
- [Documentation and Project Resources](#documentation-and-project-resources)

---

## Architecture Overview

**Design Philosophy**: Operator-free, tensor-centric, kernel-oriented design with declarative graph execution.

**Key Characteristics**:
- **GraphOrchestrator**: Single execution path via declarative compute DAGs
- **No Operator Layer**: Graph stages orchestrate kernels directly
- **Per-Tensor Device Affinity**: Each tensor knows its device placement
- **Heterogeneous Execution**: Designed to mix CPU, CUDA, ROCm via device backends
- **Centralized Kernel Dispatch**: `KernelFactory` provides unified kernel creation with caching
- **Automatic Weight Sharding**: Megatron-style tensor parallelism across MPI ranks

**Execution Flow**:
> **GraphOrchestrator** → builds **ComputeGraph** → **DeviceGraphExecutor** runs **ComputeStages** → stages use **KernelFactory** to get kernels → kernels operate on local buffers. MPI synchronization via **AllreduceStage** / **AllGatherStage**.

See `.github/instructions/llaminar-architecture-v2.instructions.md` for a full-stack walkthrough.

---

## Build System

### Build Commands

```bash
# Debug build (for development and debugging only)
cmake -B build_v2 -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2 --parallel

# Release build (for performance)
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Integration build (for "V2_Unit*" unit tests and "V2_Integration*" integration tests with snapshots + debug symbols)
cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Integration
cmake --build build_v2_integration --parallel
```

**Build Targets**:
- `llaminar2_core`: Core library (linked by tests and tools)
- `llaminar2`: Main executable

**CMake Build Types**:
| Build Type | Optimization | Debug Symbols | Snapshots | Use Case |
|------------|-------------|---------------|-----------|----------|
| `Debug` | Off | Yes | Yes | Development, debugging |
| `Release` | Full (-O3) | No | No | Production, benchmarks |
| `Integration` | Full (-O3) | Yes | Yes | Unit tests, integration tests, parity tests |

**CMake Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `HAVE_CUDA` | OFF | Enable CUDA backend |
| `HAVE_ROCM` | OFF | Enable ROCm backend |
| `ENABLE_SNAPSHOTS` | OFF | Enable tensor snapshot capture (auto-enabled for Debug/Integration) |

**IMPORTANT: Use Ninja Generator**

Always use `-G Ninja` when configuring builds. The Ninja generator is **required** for:
- **nvlink job pool**: Limits concurrent CUDA/HIP device linking to 8 processes, preventing OOM during builds with 500+ test targets
- **Faster builds**: Ninja has better parallelism and dependency tracking than Unix Makefiles

If you see OOM errors during linking or excessive nvlink processes, verify you're using Ninja.

---

## Canonical Runtime Configuration

### Running Llaminar

The `llaminar2` executable automatically bootstraps MPI and configures the runtime environment:

```bash
# Standard inference (auto-detects topology, single device)
./build_v2_release/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50

# Debug logging
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50

# Specify device explicitly
./build_v2_release/llaminar2 -d cuda:0 -m model.gguf -p "Hello" -n 50

# Dry-run to preview configuration without execution
./build_v2_release/llaminar2 --dry-run -m models/qwen2.5-0.5b-instruct-q4_0.gguf

# Show detected topology and exit
./build_v2_release/llaminar2 --show-topology
```

The executable automatically configures:
- **CPU Topology Detection**: Parses `/proc/cpuinfo` to detect sockets, cores, NUMA nodes
- **OpenMP**: `OMP_NUM_THREADS` (cores/socket), `OMP_PLACES=sockets`, `OMP_PROC_BIND=close`
- **MPI**: `OMPI_MCA_mpi_leave_pinned=1`, socket binding, process mapping
- **BLAS**: `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`

### Orchestration CLI Options

The orchestration system provides fine-grained control over device placement and parallelism:

**Introspection Flags**:
| Option | Description |
|--------|-------------|
| `--dry-run` | Show configuration without executing |
| `--explain-placement` | Explain device placement decisions |
| `--show-topology` | Show detected topology and exit |
| `--show-numa` | Show NUMA configuration and exit |
| `--validate-only` | Validate configuration without running |

**Device Assignment**:
| Option | Description | Example |
|--------|-------------|----------|
| `-d, --device <spec>` | Device for this rank | `-d cuda:0`, `-d rocm:0`, `-d cpu` |
| `--device-mode <mode>` | Assignment mode: `auto`, `local_gpu`, `round_robin`, `explicit` | `--device-mode round_robin` |
| `--device-map <map>` | Explicit rank→device mapping | `--device-map "0=cuda:0,1=cuda:1"` |

**Tensor Parallelism**:
| Option | Description | Example |
|--------|-------------|----------|
| `-tp, --tensor-parallelism-degree <n>` | TP parallelism degree | `-tp 2` |
| `--tp-scope <scope>` | Scope: `auto`, `local`, `global`, `hybrid` | `--tp-scope local` |
| `--tp-devices <list>` | Explicit device list for LOCAL TP | `--tp-devices "cuda:0,cuda:1"` |
| `--tp-weights <list>` | Proportional weight distribution | `--tp-weights "0.73,0.27"` |
| `--tp-local <degree>` | Local TP degree (hybrid mode) | `--tp-local 2` |
| `--tp-global <degree>` | Global TP degree (hybrid mode) | `--tp-global 4` |

**Pipeline Parallelism**:
| Option | Description | Example |
|--------|-------------|----------|
| `-pp, --pipeline-parallelism-degree <n>` | PP parallelism degree | `-pp 2` |
| `--pp-split <mode>` | Layer split: `equal`, `weighted`, `manual` | `--pp-split weighted` |

**Layer Placement**:
| Option | Description | Example |
|--------|-------------|----------|
| `--cpu-layers <n>` | Number of layers on CPU | `--cpu-layers 4` |
| `--cpu-layers-first` | Put CPU layers at beginning (default: end) | `--cpu-layers 4 --cpu-layers-first` |

**Collective Backend**:
| Option | Description | Example |
|--------|-------------|----------|
| `-b, --backend <type>` | Default collective backend | `--backend nccl` |

Valid backends: `auto`, `nccl`, `rccl`, `pcie_bar`, `upi`, `mpi`, `host`

**Configuration File**:
| Option | Description | Example |
|--------|-------------|----------|
| `--config <path>` | Load configuration from YAML file | `--config orchestration.yaml` |

### Device Selection Modes

Llaminar has **five mutually exclusive device selection modes**. The `ConfigValidator` (in `src/v2/config/ConfigValidator.cpp`) enforces that only one mode is active at a time.

| Mode | Primary Flags | When to Use |
|------|---------------|-------------|
| **Single Device** | `-d cuda:0` | One GPU, no parallelism |
| **Simple TP** | `-tp 2` (no `--tp-devices`) | TP with auto-picked devices |
| **Explicit TP** | `--tp-devices "cuda:0,cuda:1"` | TP with specific device list |
| **Named Domains** | `--define-domain` + `--pp-stage` | Heterogeneous PP+TP setups |
| **Device Map** | `--device-map "0=cuda:0,1=cuda:1"` | Per-MPI-rank device assignment |

**Mutual Exclusion Rules** — the following flag combinations are **invalid** and will produce a validation error:

| Flag A | Flag B | Why |
|--------|--------|-----|
| `-d` | `--tp-devices` | TP devices list implies the primary device (first entry) |
| `-d` | `-tp N` | With `-tp N`, the system auto-picks N devices; `-d` is ambiguous |
| `-d` | `--define-domain` | Named domains fully control device assignment |
| `-d` | `--device-map` | Both specify single-device assignment differently |
| `--tp-devices` | `--define-domain` | Named domains include their own device lists |
| `--device-map` | `--tp-devices` | Device map is per-rank; TP devices are intra-rank |
| `--device-map` | `--define-domain` | Named domains fully control device assignment |

**Co-Requirement Rules** — these options depend on others:

| Option | Requires | Example |
|--------|----------|---------|
| `--tp-weights` | `--tp-devices` | `--tp-devices "cuda:0,rocm:0" --tp-weights "0.73,0.27"` |
| `--pp-stage` | `--define-domain` | `--define-domain "gpu_tp=cuda:0,cuda:1" --pp-stage "0=gpu_tp:0-13"` |
| `--device-mode explicit` | `--device-map` | `--device-mode explicit --device-map "0=cuda:0,1=cuda:1"` |

**Consistency Rules**:

| Rule | Description |
|------|-------------|
| TP device count must match degree | If both `--tp-devices` and `-tp N` are set, the device count must equal N |
| No duplicate TP devices | Each device in `--tp-devices` must be distinct |
| `--tp-scope global` incompatible with `--tp-devices` | Global TP uses MPI ranks (one device per rank); `--tp-devices` is for local multi-device |

### Named Domains (Advanced)

For complex heterogeneous setups, define named TP domains and PP stage mappings. Named domains combine tensor parallelism and pipeline parallelism in a single declaration:

```bash
# Heterogeneous PP+TP: 2 CUDA GPUs for layers 0-13, 2 ROCm GPUs for layers 14-27
./build_v2_release/llaminar2 \
  --define-domain "gpu_fast=cuda:0,cuda:1;weights=0.6,0.4;backend=nccl" \
  --define-domain "gpu_slow=rocm:0,rocm:1;backend=rccl" \
  --pp-stage "0=gpu_fast:0-13" \
  --pp-stage "1=gpu_slow:14-27" \
  -m model.gguf -p "Hello" -n 50

# Single domain (equivalent to --tp-devices but with explicit backend)
./build_v2_release/llaminar2 \
  --define-domain "tp_group=cuda:0,cuda:1;backend=nccl" \
  --pp-stage "0=tp_group:0-27" \
  -m model.gguf -p "Hello"

# Preview domain configuration without running
./build_v2_release/llaminar2 \
  --define-domain "gpu_fast=cuda:0,cuda:1;backend=nccl" \
  --define-domain "gpu_slow=rocm:0,rocm:1;backend=rccl" \
  --pp-stage "0=gpu_fast:0-13" \
  --pp-stage "1=gpu_slow:14-27" \
  --explain-placement --dry-run -m model.gguf
```

**Domain Definition Format**: `name=device1,device2[;weights=w1,w2][;backend=type]`

**PP Stage Format**: `stage_id=domain_name:first_layer-last_layer`

> **Note**: When using named domains, do NOT combine with `-d`, `-tp N`, `--tp-devices`, or `--device-map`. Named domains fully control device assignment and parallelism.

---

## Benchmark Mode

### Running Benchmarks

```bash
# Standard benchmark with auto-generated prompt
./build_v2_release/llaminar2 --benchmark -m model.gguf

# Custom prompt and decode length
./build_v2_release/llaminar2 --benchmark -m model.gguf -p "Your prompt here" -n 100

# With full profiling (kernel + executor overhead)
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50
```

**Features**:
- 1 warmup run + 3 benchmark runs averaged
- Separate prefill/decode timing
- Greedy sampling for reproducibility
- KV cache cleared between runs

**Example Output**:
```
+--------------------------------------------------------------+
|                    BENCHMARK RESULTS                         |
|              (average of 3 runs after warmup)                |
+--------------------------------------------------------------+
| PREFILL PHASE                                                |
|   Tokens:           596 tokens                               |
|   Throughput:    304.19 tok/s                                |
+--------------------------------------------------------------+
| DECODE PHASE                                                 |
|   Tokens:           128 tokens                               |
|   Throughput:     54.00 tok/s                                |
+--------------------------------------------------------------+
```

### Kernel Profiling

Enable per-kernel timing breakdown:

```bash
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50
```

**Profiled Operations**: `GEMM_Q8`, `ATTENTION`, `FFN_DOWN`, `FFN_GATE`, `FFN_UP`, `LM_HEAD`, `QUANTIZE_Q8`, `RMS_NORM`, `SWIGLU`, `ROPE`, `RESIDUAL_ADD`, `EMBEDDING`

**Note**: `LLAMINAR_PROFILING=1` enables both kernel timing and executor overhead profiling in a single flag.

---

## Testing Guidelines

### Test Organization

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/v2/unit/` | Fast, isolated component tests (no model loading) |
| Integration | `tests/v2/integration/` | Full pipeline tests with models |
| Integration/Parity | `tests/v2/integration/parity/` | PyTorch ground truth parity tests |
| Performance | `tests/v2/performance/` | Benchmark tests |

### Test File Naming Convention

- Files: `Test__ClassName.cpp` (CamelCase with double underscore)
- Suite: `TEST(Test__ClassName, TestName)`
- Example: `Test__ModelLoader.cpp` → `TEST(Test__ModelLoader, LoadsQwen2)`

### Running Tests

**CRITICAL: Test-to-Build Mapping**

| Test Type | Build Directory | CTest Filter |
|-----------|-----------------|--------------|
| Unit tests (`V2_Unit_*`) | `build_v2_integration` | `-R "^V2_Unit_"` |
| Integration tests (`V2_Integration_*`) | `build_v2_integration` | `-R "^V2_Integration_"` |
| Parity tests (`V2_Integration_Parity_*`) | `build_v2_integration` | `-R "^V2_Integration_Parity_"` |
| Performance tests (`V2_Perf_*`) | `build_v2_release` | `-R "^V2_Perf_"` |

**CRITICAL: Never Limit Parallelism**

Agents must NEVER artificially limit build or test parallelism:

```bash
# ❌ WRONG - Do NOT limit parallelism
cmake --build build_v2_integration -j8
cmake --build build_v2_integration --parallel 4
ctest --test-dir build_v2_integration -j4

# ✅ CORRECT - Use full parallelism
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration --parallel
```

**Example Commands**:

```bash
# Unit tests (Integration build - Release optimizations + debug symbols)
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests (Integration build - has snapshots + debug symbols)
ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure

# Parity tests (Llaminar vs PyTorch reference)
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure

# Performance benchmarks
ctest --test-dir build_v2_release -R "^V2_Perf_" --verbose
```

### CTest Labels

Tests use hierarchical labels for flexible filtering:
- **Tier 1 (Type)**: `Unit`, `Integration`, `Parity`, `Performance`
- **Tier 2 (Architecture)**: `V2`
- **Tier 3 (Component)**: `DeviceManagement`, `TensorOperations`, `Kernels`, `ModelLoading`
- **Tier 4 (Feature)**: `Quantization`, `GEMM`, `Attention`, `IQ4_NL`

```bash
ctest -L DeviceManagement          # All device tests
ctest -L "Quantization"            # All quantization tests
ctest -LE Performance              # Exclude performance tests
```

### TestTensorFactory Utility

Header-only utility for unit tests (`tests/v2/utils/TestTensorFactory.h`):

```cpp
#include "utils/TestTensorFactory.h"
using namespace llaminar2::test;

auto input = TestTensorFactory::createFP32Random({32, 896});
auto weights = TestTensorFactory::createQ8_0Random({1024, 896});
float mse = TestTensorFactory::computeMSE(a->data(), b->data(), count);
```

---

## Debugging

### GDB for Segfaults (MPI-aware)

```bash
# Create GDB command file
cat > /tmp/gdbcommands.txt << 'EOF'
set debuginfod enabled off
set pagination off
handle SIGSEGV stop print
run
thread apply all bt full
quit
EOF

# Set environment
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close
export LLAMINAR_LOG_LEVEL=DEBUG

# Run under GDB
timeout 120 mpirun -np 2 \
  gdb -x /tmp/gdbcommands.txt --args \
  ./build_v2/tests/v2/v2_test_my_feature 2>&1 | tee gdb_output.log

# Analyze
grep -A 50 "Program received signal" gdb_output.log
```

### ASAN for Memory Issues

```bash
cmake -B build_v2_asan -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer -fsanitize=address" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build_v2_asan --parallel

ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 \
  timeout 240 mpirun -np 2 ./build_v2_asan/tests/v2/<your_test>
```

### Valgrind for Memory Leaks

```bash
mpirun -np 2 valgrind --tool=memcheck --leak-check=full ./build_v2/tests/v2/v2_test_my_feature
```

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| MPI hangs | Use `MPI_Barrier` before/after collective operations |
| Numerical divergence | Run parity tests, check layer-by-layer snapshots |
| Performance regression | Use `--benchmark` mode, verify Release build with `-march=native` |
| Memory allocation failures | Enable NUMA verification, check first-touch allocation |
| Race conditions | Use `OMP_WORKSHARE_REGION` macro, avoid shared mutable state |
| Inference produces garbage | Use `-t 0` (greedy sampling) to eliminate randomness |

---

## Stage Dump Framework

### Overview

The Stage Dump framework captures raw input/output tensors from pipeline stages for debugging and replay testing. This is essential for isolating bugs in specific stages without running the full pipeline.

**Key Features**:
- Dumps inputs, outputs, and weights for any ComputeStage
- Flexible filtering by stage type, name, layer, iteration, and rank
- Supports substring matching for stage names
- Binary format for replay testing + human-readable metadata
- **Variable Q16_1 block size support**: Correctly handles Q16_1_32, Q16_1_64, Q16_1_128 tensors
- Zero overhead when disabled

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_STAGE_DUMP_ENABLED` | Master enable (0/1) | Disabled |
| `LLAMINAR_STAGE_DUMP_DIR` | Output directory | `/tmp/llaminar_stage_dumps` |
| `LLAMINAR_STAGE_DUMP_TYPES` | Stage types to dump (e.g., `FUSED_ATTENTION_WO,GEMM`) | `all` |
| `LLAMINAR_STAGE_DUMP_NAMES` | Stage names to dump (substring match) | `all` |
| `LLAMINAR_STAGE_DUMP_LAYERS` | Layer indices (e.g., `0,1,2`) | `all` |
| `LLAMINAR_STAGE_DUMP_ITERATION` | Decode iterations to dump | `all` |
| `LLAMINAR_STAGE_DUMP_RANK` | MPI rank to dump (-1 for all) | 0 |
| `LLAMINAR_STAGE_DUMP_MAX` | Max dumps per stage type | 100 |
| `LLAMINAR_STAGE_DUMP_INPUTS` | Dump input tensors (0/1) | 1 |
| `LLAMINAR_STAGE_DUMP_OUTPUTS` | Dump output tensors (0/1) | 1 |
| `LLAMINAR_STAGE_DUMP_WEIGHTS` | Dump weight tensors (0/1) | 1 |
| `LLAMINAR_STAGE_DUMP_ASYNC` | Use async I/O for dumps (0/1) | 1 (enabled) |
| `LLAMINAR_STAGE_DUMP_ASYNC_THREADS` | Number of async I/O threads (1-16) | 2 |

### Async Stage Dumping

By default, stage dumps use **asynchronous I/O** with a background thread pool. This significantly reduces the performance impact of dumping:

- **Sync mode (ASYNC=0)**: ~60-70ms overhead per iteration for layer 0 dumps
- **Async mode (ASYNC=1)**: ~2-5ms overhead per iteration (data copied to queue, I/O happens in background)

The async system copies tensor data to an internal queue and immediately returns. Background threads handle the actual file I/O, allowing execution to continue without blocking.

```bash
# Default: async mode with 2 I/O threads
LLAMINAR_STAGE_DUMP_ENABLED=1 ./llaminar2 -m model.gguf -p "test"

# Disable async for debugging or when memory is constrained
LLAMINAR_STAGE_DUMP_ENABLED=1 LLAMINAR_STAGE_DUMP_ASYNC=0 ./llaminar2 -m model.gguf -p "test"

# Increase I/O threads for faster writes on fast storage
LLAMINAR_STAGE_DUMP_ENABLED=1 LLAMINAR_STAGE_DUMP_ASYNC_THREADS=4 ./llaminar2 -m model.gguf -p "test"
```

### Substring Matching for Stage Names

The `LLAMINAR_STAGE_DUMP_NAMES` filter uses **substring matching** for flexibility:

```bash
# Dump all fused attention stages (matches layer0_fused_attn_wo, layer5_fused_attn_wo, etc.)
LLAMINAR_STAGE_DUMP_NAMES=fused_attn_wo

# Dump all layer 0 stages (matches layer0_attn_norm, layer0_qkv_proj, layer0_fused_attn_wo, etc.)
LLAMINAR_STAGE_DUMP_NAMES=layer0_

# Dump multiple patterns (attention and FFN norms)
LLAMINAR_STAGE_DUMP_NAMES=fused_attn_wo,ffn_norm
```

### Example Usage

```bash
# Dump fused attention stages for layers 0-2
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=fused_attn_wo \
LLAMINAR_STAGE_DUMP_LAYERS=0,1,2 \
./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline

# Dump all stages for layer 0 during decode iteration 0
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=layer0_ \
LLAMINAR_STAGE_DUMP_ITERATION=0 \
./build_v2_release/llaminar2 -m model.gguf -p "test"

# Dump only attention type stages (by type, not name)
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_TYPES=FUSED_ATTENTION_WO,ATTENTION \
./build_v2_release/llaminar2 -m model.gguf -p "test"
```

### Output Structure

Dumps are written to `LLAMINAR_STAGE_DUMP_DIR` (default: `/tmp/llaminar_stage_dumps/`):

```
/tmp/llaminar_stage_dumps/
├── stage_0000_FUSED_ATTENTION_WO_layer0_fused_attn_wo_rank0/
│   ├── inputs/
│   │   ├── Q_q16_1_64.bin    # Q tensor in native Q16_1_64 format
│   │   ├── Q_meta.txt        # Metadata for Q tensor
│   │   ├── K_q16_1_64.bin    # K tensor in native Q16_1_64 format
│   │   ├── K_meta.txt        # Metadata for K tensor
│   │   ├── V_q8_1.bin        # V tensor in native Q8_1 format
│   │   └── V_meta.txt        # Metadata for V tensor
│   ├── outputs/
│   │   ├── context.bin       # FP32 attention output
│   │   └── context_meta.txt  # Metadata for context tensor
│   └── weights/              # Stage-specific weight metadata
│       └── Wo_meta.txt
├── stage_0001_GEMM_layer0_ffn_down_rank0/
│   └── ...
```

**Directory naming convention**: `stage_<counter>_<stage_type>_<stage_name>_rank<N>`

### Metadata File Format

Each tensor dump includes a `<tensor_name>_meta.txt` file with key=value pairs:

```
name=Q
rows=126
cols=64
dtype=Q16_1_64
element_count=8064
byte_size=1904
# Block format info:
block_count=126
blocks_per_row=1
block_element_size=64
sample_min=-0.123456
sample_max=0.789012
sample_mean=0.001234
```

**Metadata Fields**:
| Field | Description |
|-------|-------------|
| `name` | Tensor name (e.g., "Q", "K", "V", "context") |
| `rows` | Number of logical rows |
| `cols` | Number of logical columns |
| `dtype` | Data type: `FP32`, `Q8_1`, `Q16_1_32`, `Q16_1_64`, `Q16_1_128`, `IQ4_NL`, etc. |
| `element_count` | Total logical elements (rows × cols) |
| `byte_size` | Actual binary file size in bytes |
| `block_count` | Total quantization blocks (for block formats) |
| `blocks_per_row` | Blocks per row (for block formats) |
| `block_element_size` | Elements per block: 32, 64, or 128 |
| `sample_min/max/mean` | Sample statistics from dequantized data |

**Q16_1 Variable Block Size Support**:

The stage dump framework correctly handles Q16_1 tensors with different block sizes:

| dtype | Block Size | Bytes/Block | Use Case |
|-------|-----------|-------------|----------|
| `Q16_1_32` | 32 elements | 72 bytes | Legacy GEMM |
| `Q16_1_64` | 64 elements | 136 bytes | Attention (head_dim=64, Qwen2.5) |
| `Q16_1_128` | 128 elements | 264 bytes | Attention (head_dim=128, Llama-3) |

### Replay Testing

Use captured dumps to create isolated replay tests:

```cpp
#include "utils/TensorDumpLoader.h"

// Load dumped tensor with metadata
auto [data, meta] = loadTensorDequantizedFP32(
    "/tmp/llaminar_stage_dumps/stage_0000_FUSED_ATTENTION_WO_layer0_fused_attn_wo_rank0",
    "Q",       // tensor name
    "inputs"); // subdir

// Access metadata
LOG_INFO("Loaded " << meta.dtype << " tensor: " 
         << meta.rows << "x" << meta.cols 
         << " (" << meta.byte_size << " bytes)");

// For Q16_1 native blocks:
auto [blocks, meta] = loadTensorAsQ16_1(dump_dir, "Q", "inputs");
```

See `tests/v2/integration/replay/Test__HybridQ16AttentionReplay.cpp` for a complete example.

---

## Stage Output Print Facility

### Overview

The Stage Output Print facility provides a lightweight way to inspect tensor values at stage boundaries during inference. Unlike the Stage Dump Framework (which writes binary files for replay testing), this facility prints tensor samples directly to the log output, making it ideal for quick debugging of buffer wiring and data flow issues.

**When to Use**:
- Debugging buffer wiring issues (e.g., wrong tensor passed to a stage)
- Verifying data is being computed correctly at each stage
- Comparing CPU vs GPU execution paths
- Quick sanity checks without writing files to disk

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_STAGE_OUTPUT_PRINT` | Master enable (0/1) | Disabled |
| `LLAMINAR_STAGE_OUTPUT_PRINT_N` | Number of elements to print per row | 8 |
| `LLAMINAR_STAGE_OUTPUT_PRINT_ROWS` | Number of rows to print (first and last) | 2 |
| `LLAMINAR_STAGE_OUTPUT_PRINT_STAGES` | Comma-separated stage name patterns (substring match) | `all` |

### Example Usage

```bash
# Print outputs for all stages (verbose!)
LLAMINAR_STAGE_OUTPUT_PRINT=1 ./build_v2_release/llaminar2 -m model.gguf -p "Hello"

# Print only FFN stages for layer 0
LLAMINAR_STAGE_OUTPUT_PRINT=1 \
LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=layer0_gate_up_proj,layer0_swiglu,layer0_down_proj \
./build_v2_release/llaminar2 -m model.gguf -p "Hello"

# Print with more elements per row
LLAMINAR_STAGE_OUTPUT_PRINT=1 \
LLAMINAR_STAGE_OUTPUT_PRINT_N=16 \
LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=layer0_swiglu \
./build_v2_release/llaminar2 -m model.gguf -p "Hello"
```

### Output Format

```
[StageOutput] layer0_gate_up_proj/output_gate [2x4864] row[0]: -0.074,-1.148,-0.085... | row[1]: 0.023,-0.439,0.652...
[StageOutput] layer0_gate_up_proj/output_up [2x4864] row[0]: -0.360,0.594,0.074...
[StageOutput] layer0_swiglu/output [2x4864] row[0]: 0.012,-0.164,-0.003...
```

The format is: `[StageOutput] <stage_name>/<tensor_name> [<rows>x<cols>] row[0]: <values>... | row[<last>]: <values>...`

### Debugging Example: Buffer Wiring Bug

This facility was instrumental in diagnosing a bug where SwiGLU was producing incorrect output on GPU:

```bash
# Compare CPU vs GPU SwiGLU outputs
LLAMINAR_STAGE_OUTPUT_PRINT=1 \
LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=layer0_swiglu \
./test_cpu_vs_gpu 2>&1 | grep StageOutput

# CPU output (correct):
# [StageOutput] layer0_swiglu/output [2x4864] row[0]: 0.012,-0.164,-0.003...

# GPU output (wrong - identical to input_up!):
# [StageOutput] layer0_swiglu/output [2x4864] row[0]: -0.360,0.594,0.074...
```

This revealed that the GPU SwiGLU output was identical to its `up` input, indicating the kernel wasn't being called. The root cause was a missing `device_id` assignment in the stage params.

### Implementation Details

- Output is printed via `LOG_INFO` after `markOutputsDirty()` completes (GPU→host sync has occurred)
- Uses `TensorBase::data()` for coherence-aware access
- Stage name filtering uses case-insensitive substring matching
- Zero overhead when disabled (no string formatting occurs)

---

## Parity Testing (PyTorch Reference)

### Overview

Parity tests validate that Llaminar's inference results match PyTorch ground truth within acceptable tolerances. Tests compare layer-by-layer activations and final logit distributions across different backends (CPU, CUDA, ROCm).

**Key Concept**: Both Llaminar and PyTorch load the **same GGUF weights** (e.g., Q4_0). PyTorch dequantizes to FP32; Llaminar uses quantized GEMM. Minor divergence is expected; token predictions should match.

### Location and Documentation

All parity tests are located in `tests/v2/integration/parity/` with a comprehensive README:

> **📖 See `tests/v2/integration/parity/README.md` for full documentation** including:
> - Declarative test architecture (three-tier inheritance)
> - `BackendThresholds` configuration
> - Writing new parity tests
> - Metrics (cosine similarity, KL divergence, Top-K overlap)
> - Troubleshooting guide

### Quick Reference

**Running Parity Tests**:
```bash
# Build integration tests (includes parity tests)
cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Integration
cmake --build build_v2_integration --parallel

# Run all parity tests
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure

# Run Qwen2 CUDA parity test
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_CUDA" -V
```

**Test Output Example**:
```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                    CUDA vs PyTorch LAYER-BY-LAYER PARITY                                 ║
╠═══════════╦═══════════════╦═══════════════╦════════════════════════════════════════╦══════╣
║   Layer   ║   Avg Cosine  ║   Min Cosine  ║            Worst Stage                 ║Status║
╠═══════════╬═══════════════╬═══════════════╬════════════════════════════════════════╬══════╣
║ EMBEDDING ║      0.999912 ║      0.999912 ║                      -                 ║  ✓  ║
║   Layer 0 ║      0.998234 ║      0.995123 ║              FFN_RESIDUAL              ║  ✓  ║
...
╚═══════════╩═══════════════╩═══════════════╩════════════════════════════════════════╩══════╝
```

### Debugging Inference Issues

**Step 1**: Use greedy sampling to eliminate randomness:
```bash
./build_v2_release/llaminar2 -m model.gguf -p "prompt" -n 10 -t 0
```

**Step 2**: Compare top-5 predictions:
```bash
LLAMINAR_LOG_LEVEL=TRACE ./build_v2_release/llaminar2 -m model.gguf -p "prompt" -n 1 -t 0 2>&1 | grep "Top-5"
```

**Step 3**: Run parity tests:
```bash
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2" -V
```

**Step 4**: Enable stage tracing for full visibility:
```bash
LLAMINAR_STAGE_DUMP_ENABLED=1 LLAMINAR_MPI_LOG_COLLECTIVES=1 \
./build_v2_release/llaminar2 -tp 2 --explain-placement -m model.gguf -p "test"
```

---

## Kernel Development

### Core Principles

1. **Tensor-centric Design**: Operations focused on Tensors
2. **Per-Tensor Device Affinity**: Tensors know their device placement
3. **Strategy Pattern**: Generic kernels + format-specific decode via `ITensorGemmTileDataProvider`
4. **ITensor Interfaces**: `ITensorGemm`, `ITensorAttention`, `ITensorRoPE`, etc.

### TypedTensorBase and `typed_data()` Pattern

The `TypedTensorBase<Derived, DataType>` CRTP base provides **zero-overhead typed access** to tensor storage:

```cpp
// CRTP base for typed tensors
template<typename Derived, typename DataType>
class TypedTensorBase {
public:
    const DataType* typed_data() const;        // Native type access (Q8_1Block*, uint16_t*, etc.)
    DataType* mutable_typed_data();            // Mutable native type access
};
```

**Usage Pattern** - After `dynamic_cast<>`, use `typed_data()` instead of type-specific accessors:

```cpp
// ✅ PREFERRED: Unified pattern with typed_data()
if (auto* q8_tensor = dynamic_cast<Q8_1Tensor*>(tensor)) {
    Q8_1Block* blocks = q8_tensor->mutable_typed_data();
    kernel->process(blocks, ...);
}

// ❌ AVOID: Type-specific accessors (legacy)
if (auto* q8_tensor = dynamic_cast<Q8_1Tensor*>(tensor)) {
    Q8_1Block* blocks = q8_tensor->mutable_q8_1_blocks();  // Deprecated
}
```

**Supported Tensor Types**:
| Tensor Class | `typed_data()` Returns |
|--------------|------------------------|
| `FP32Tensor` | `float*` |
| `BF16Tensor` | `uint16_t*` |
| `FP16Tensor` | `uint16_t*` |
| `Q8_1Tensor` | `Q8_1Block*` |
| `Q16_1Tensor` | `Q16_1Block*` |
| `Q8_0Tensor` | `Q8_0Block*` |
| `IQ4_NLTensor` | `IQ4_NLBlock*` |
| (all 27 tensor classes) | Native storage type |

### ITensorGemmTileDataProvider Strategy Pattern

Quantized tensors implement decode strategies; a single generic GEMM kernel works for all formats:

```cpp
class IQ4_NLTensor : public TensorBase, public ITensorGemmTileDataProvider {
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const IQ4_NLBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeBlock(block, output);  // Format-specific decode
    }
    
    size_t block_size() const override { return 32; }
    
    std::unique_ptr<ITensorGemm> createGemm(const MPIContext& mpi_ctx, int device_idx) const override {
        return std::make_unique<QuantizedGemmKernel>(this, mpi_ctx, device_idx);
    }
};
```

### KernelFactory

Centralized kernel creation with caching:

```cpp
// Preferred: cached kernel (pack once, use many)
ITensorGemm* gemm = KernelFactory::getOrCreateGemm(weight_tensor.get());
gemm->multiply(activations, output, m, n, k);

// Alternative: uncached (for one-off use)
auto gemm = KernelFactory::createGemm(dynamic_cast<IQ4_NLTensor*>(tensor), DeviceType::CPU);
```

### SIMD Guidelines

1. **Exploit ILP**: Unroll loops, interleaved loads/stores
2. **Vectorized Tail Handling**: AVX512 16-way → AVX2 8-way → AVX 4-way → SSE 2-way → scalar
3. **Prefetch**: Prefetch upcoming sequential reads

### OpenMP Nested-Safe Parallelism (`OMP_WORKSHARE_REGION`)

**CRITICAL**: All kernel-level OpenMP parallelization MUST use the `OMP_WORKSHARE_REGION` macro from `utils/OpenMPUtils.h`. This enables "layer-level fusion" where an outer parallel region can encompass multiple kernel calls, eliminating thread fork/join overhead.

**Why it matters**: Creating a new `#pragma omp parallel` region has ~10-50μs overhead. For decode (one token at a time), this overhead can dominate runtime. The macro checks `omp_in_parallel()` and only creates a new region if not already inside one.

**Available Macros**:

| Macro | Use Case |
|-------|----------|
| `OMP_WORKSHARE_REGION(fn)` | Standard worksharing with implicit barrier |
| `OMP_WORKSHARE_REGION_SYNC(fn)` | Explicit barrier after work |
| `OMP_WORKSHARE_REGION_IF(fn, cond)` | Conditional parallelization |
| `OMP_SINGLE(fn)` | Single-thread execution (for MPI collectives) |

**Correct Pattern**:

```cpp
#include "utils/OpenMPUtils.h"

void my_kernel(float* data, int n) {
    // Define ALL work including thread-local allocations inside the lambda
    auto do_work = [&]() {
        // Thread-local buffers (allocated per-thread automatically)
        alignas(64) float local_buffer[256];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            data[i] = process(data[i], local_buffer);
        }
    };
    OMP_WORKSHARE_REGION(do_work);
}
```

**Anti-Pattern (WRONG)**:

```cpp
// DON'T DO THIS - creates parallel region every call
void my_kernel(float* data, int n) {
    #pragma omp parallel for  // <-- BAD: always creates new parallel region
    for (int i = 0; i < n; ++i) {
        data[i] = process(data[i]);
    }
}
```

**Multi-Phase Work in Single Region**:

```cpp
auto do_attention_work = [&]() {
    // Phase 1: Parallel head processing
    #pragma omp for schedule(static) nowait
    for (int h = 0; h < num_heads; ++h) {
        process_head(h, head_outputs[h]);
    }
    
    // Implicit barrier between phases (remove 'nowait' if needed)
    
    // Phase 2: Parallel reduction
    #pragma omp for schedule(static)
    for (int i = 0; i < seq_len; ++i) {
        reduce_heads_to_output(head_outputs, output[i]);
    }
};
OMP_WORKSHARE_REGION(do_attention_work);
```

### JIT Register Guard System

The JIT kernels (in `src/v2/kernels/cpu/attention/q8_1/jit/`) use Xbyak for runtime code generation. These kernels have **complex register allocation** across ZMM registers (zmm0-31) that are partitioned into zones.

**Key Files** (all in `src/v2/kernels/cpu/jit/`):
- `RegisterAllocation.h`: Zone definitions and typed register wrappers
- `RegisterGuard.h`: RAII tracking and conflict detection
- `RegisterEnforcement.h`: Compile-time concepts for typed access
- `JitMicrokernelBase.h`: Base class with register accessors and tracking integration

**CMake Enforcement** (`cmake/EnforceTypedRegisters.cmake`):
- Build-time check that prevents bypassing the guard system
- Fails cmake configure if raw Xbyak registers (`Zmm(5)`) or untracked accessors (`scratch0().zmm()`) are used
- Enforced on all `/jit/` files except infrastructure and legacy files

#### Register Zone Layout

| Zone | Registers | Purpose |
|------|-----------|---------|
| **AccumulatorZone** | zmm0-7 | Context accumulators for attention output |
| **QVectorZone** | zmm8-15 | Q vector data (also called Input zone) |
| **StateZone** | zmm16-19 | Softmax state (max, sum, weight, corr) |
| **ScratchZone** | zmm20-25 | Temporary computation |
| **ReservedZone** | zmm26-31 | Preloaded constants (scale, log2e, etc.) |

**Critical Aliasing**: XMM/YMM/ZMM registers share the same physical register file. `xmm20` aliases `zmm20`! The **ScoreZone** (xmm20-23) overlaps **ScratchZone** (zmm20-23).

#### MANDATORY: Use borrow<>() for Register Access

**All register access MUST use `borrow<RegType>()`** for RAII-tracked lifetime management. This is enforced at build time.

```cpp
// ❌ BAD - Raw Xbyak register (build will fail)
Zmm zmm_temp(20);
gen.vmovaps(Zmm(5), src);

// ❌ BAD - Untracked accessor (build will fail)  
gen.vmovaps(scratch0().zmm(), src);
auto zmm = accum4().zmm();

// ✅ GOOD - RAII tracked with borrow<>()
auto guard = gen.borrow<Scratch0>();
gen.vmovaps(guard.zmm(), src);
// guard auto-releases at scope end
```

#### RAII Register Borrowing Pattern

```cpp
void emit_complex_operation(JitMicrokernelBase& gen) {
    // Borrow registers - tracker ensures no conflicts
    auto guard_score0 = gen.borrow<Score0>();  // xmm20
    auto guard_score1 = gen.borrow<Score1>();  // xmm21
    auto guard_scratch = gen.borrow<Scratch4>(); // zmm24 (safe, doesn't alias scores)
    
    gen.vmovss(guard_score0.xmm(), ...);
    gen.vmovaps(guard_scratch.zmm(), ...);
    
    // Release score registers before reusing as scratch
    guard_score0.release();
    guard_score1.release();
    
    // Now safe to borrow Scratch0/1 (alias xmm20/21)
    auto guard_weight0 = gen.borrow<Scratch0>(); // zmm20
    // ... use for weights ...
    
    // Guards auto-release at scope end
}
```

#### Available Register Types

| Type | Zone | Physical Register |
|------|------|-------------------|
| `Accum0`-`Accum7` | Accumulator | zmm0-7 |
| `Input0`-`Input7` | Input/Q | zmm8-15 |
| `StateMax`, `StateSum`, `StateWeight`, `StateCorr` | State | zmm16-19 |
| `Scratch0`-`Scratch5` | Scratch | zmm20-25 |
| `Score0`-`Score3` | Score (aliases Scratch) | xmm20-23 |
| `Const128`, `ConstScale`, `ConstNegInf`, etc. | Reserved | zmm26-31 |

#### Conflict Detection

If you attempt to borrow a register that's already borrowed, you get a detailed error:

```
╔══════════════════════════════════════════════════════════════════╗
║              REGISTER BORROW CONFLICT DETECTED                    ║
╠══════════════════════════════════════════════════════════════════╣
║ Physical register: zmm/ymm/xmm20
║ Currently borrowed by: 'Scratch0'
║ New borrow attempted for: 'Score0'
║
║ FIX: Either:
║   1. Release the existing borrow before this access
║   2. Use a different register that isn't borrowed
║   3. Restructure code to avoid the conflict
╚══════════════════════════════════════════════════════════════════╝
```

**Note**: Register tracking runs only during JIT compilation (not at inference runtime), so there is zero performance overhead.

**Note**: Register tracking is always enabled during JIT compilation with zero runtime overhead (the generated assembly is identical with or without tracking).

---

## GPU Tensor Coherence

### Overview

Llaminar uses a **coherence protocol** to manage tensor data movement between host (CPU) and device (GPU) memory. This system:
- Tracks whether tensor data is "dirty" on CPU or GPU
- Automatically synchronizes data when needed
- Provides both automatic (DeviceGraphExecutor) and manual (test/utility) patterns

**Key Files:**
- `src/v2/tensors/TensorClasses.h` - `TensorBase` coherence methods (`ensureOnDevice()`, `mark_device_dirty()`, `ensureOnHost()`, `data()`)
- `src/v2/execution/StageCoherence.h` - `StageCoherence` helper for DeviceGraphExecutor
- `src/v2/execution/GpuCoherence.h` - RAII utilities for tests and direct kernel calls

### Coherence Protocol

Every `TensorBase` tracks its coherence state:

| State | Meaning | `data()` Returns |
|-------|---------|------------------|
| CPU is authoritative | Data was last modified on host | Host data directly |
| GPU is authoritative | `mark_device_dirty()` was called | Syncs GPU→host first |
| Never uploaded | No GPU buffer allocated | Host data directly |

**Core Methods on TensorBase:**

```cpp
// Upload to GPU if needed (allocates buffer on first call)
bool ensureOnDevice(DeviceId device);

// Mark GPU as having authoritative data (after GPU kernel writes)
void mark_device_dirty();

// Get host data pointer (syncs from GPU if device-dirty)
const float* data();
float* mutable_data();  // Also marks CPU as authoritative
```

### Automatic Coherence (DeviceGraphExecutor)

When using `DeviceGraphExecutor` (the standard inference path), **coherence is automatic**:

1. **Stage Entry**: `StageCoherence::ensureInputsOnDevice()` uploads all stage inputs
2. **Stage Execution**: Kernel runs on GPU
3. **Stage Exit**: `StageCoherence::markOutputsDirty()` marks outputs as device-authoritative

```cpp
// DeviceGraphExecutor automatically handles this:
void DeviceGraphExecutor::executeStage(const ComputeNode& node) {
    // 1. Auto-cohere inputs to GPU
    StageCoherence::ensureInputsOnDevice(node.stage, device_);
    
    // 2. Run the stage
    node.stage->execute();
    
    // 3. Auto-mark outputs as GPU-authoritative
    StageCoherence::markOutputsDirty(node.stage);
}
```

**Stages declare their coherence policy** via `coherencePolicy()`:

| Policy | Behavior |
|--------|----------|
| `FULL` (default) | Cohere inputs on entry, mark outputs dirty on exit |
| `INPUT` | Only cohere inputs (outputs managed manually) |
| `OUTPUT` | Only mark outputs dirty (inputs already on device) |
| `NONE` | No automatic coherence (MPI stages, custom management) |

### Manual Coherence (Tests and Direct Kernel Calls)

When calling kernels **directly** (bypassing DeviceGraphExecutor), you must handle coherence manually. Use the utilities in `execution/local_execution/coherence/GpuCoherence.h`:

#### Preferred Pattern: `with_gpu_coherence()`

```cpp
#include "execution/local_execution/coherence/GpuCoherence.h"

// Clean, self-documenting pattern for kernel calls
ASSERT_TRUE(with_gpu_coherence(
    gpu_device,
    {input.get()},                              // inputs to upload
    {output_q.get(), output_k.get()},           // outputs to upload + mark dirty
    [&] {
        return kernel->multiply_fused_tensor(input.get(), projections, M, K, nullptr);
    }));

// After the lambda completes:
// - All outputs are marked device-dirty
// - Calling output->data() will sync GPU→host
```

#### Alternative: RAII Wrappers

```cpp
// For single outputs - wraps ensureOnDevice + mark_device_dirty
{
    auto output = GpuOutput<FP32Tensor>(output_tensor.get(), gpu_device);
    kernel->multiply_tensor(input.get(), output.get(), M, N, K, ...);
} // ← output automatically marked dirty when scope exits

// For read-only inputs (no dirty marking)
{
    auto weights = GpuInput<Q4_0Tensor>(weight_tensor.get(), gpu_device);
    kernel->compute(weights.get(), ...);
} // ← weights NOT marked dirty (read-only)
```

#### C++20 Concept

The utilities use a concept to ensure type safety:

```cpp
template <typename T>
concept CoherableTensor = requires(T *t, DeviceId d) {
    { t->ensureOnDevice(d) } -> std::same_as<bool>;
    { t->mark_device_dirty() } -> std::same_as<void>;
};
```

### Anti-Pattern: Manual Coherence Without RAII

Avoid manual `ensureOnDevice()` and `mark_device_dirty()` calls in tests:

```cpp
// ❌ BAD - Easy to forget mark_device_dirty, leads to stale data
input->ensureOnDevice(gpu_device);
output->ensureOnDevice(gpu_device);
kernel->compute(input.get(), output.get(), ...);
// OOPS! Forgot: output->mark_device_dirty();
const float* result = output->data();  // Returns stale host data!

// ✅ GOOD - RAII ensures correctness
ASSERT_TRUE(with_gpu_coherence(gpu_device, {input.get()}, {output.get()}, [&] {
    return kernel->compute(input.get(), output.get(), ...);
}));
const float* result = output->data();  // Correctly syncs GPU→host
```

### When to Use Which Pattern

| Context | Pattern |
|---------|--------|
| Pipeline stages (via DeviceGraphExecutor) | Automatic - do nothing special |
| Integration tests calling kernels directly | `with_gpu_coherence()` lambda wrapper |
| Simple single-output tests | `GpuOutput<T>` RAII wrapper |
| Custom pipelines bypassing DeviceGraphExecutor | `GpuCoherenceScope` for fine-grained control |

---

## Weight Sharding and Tensor Parallelism

### Overview

Weight sharding enables Megatron-style tensor parallelism by distributing weights across devices. Llaminar V2 supports two TP scopes:

- **LOCAL TP**: Multiple devices within a single MPI rank (NCCL/RCCL/PCIe BAR)
- **GLOBAL TP**: Distributed across MPI ranks (MPI collectives)
- **HYBRID TP**: Combination of local + global

### TP Scope Selection

| Scope | Use Case | Collective Backend |
|-------|----------|--------------------|
| `auto` | Let orchestrator decide (default) | Auto-detected |
| `local` | Multi-GPU single machine | NCCL (CUDA), RCCL (ROCm), PCIe BAR (mixed) |
| `global` | Multi-machine distributed | MPI |
| `hybrid` | Multi-GPU + multi-machine | Local + MPI |

### Sharding Modes

| Weight | Mode | Description |
|--------|------|-------------|
| `attn_q`, `attn_k`, `attn_v` | COLUMN_PARALLEL | Split output dim (heads) |
| `attn_output` (Wo) | ROW_PARALLEL | Split input dim, allreduce after |
| `ffn_gate`, `ffn_up` | COLUMN_PARALLEL | Split output dim |
| `ffn_down` | INPUT_PARALLEL | Split input dim, allreduce after |
| `output` (LM head) | COLUMN_PARALLEL | Split vocab, allgather logits |
| Norms, embeddings | REPLICATE | Full copy on each device |

### Memory Savings

With 2-way TP: ~50% reduction for sharded weights, ~25-30% overall.

### CLI Examples

```bash
# 2-way LOCAL TP on CUDA GPUs (auto-detects NCCL)
./build_v2_release/llaminar2 -tp 2 --tp-scope local -m model.gguf -p "Hello"

# Explicit device list for LOCAL TP
./build_v2_release/llaminar2 --tp-devices "cuda:0,cuda:1" -m model.gguf -p "Hello"

# Proportional TP (73%/27% split for heterogeneous GPUs)
./build_v2_release/llaminar2 --tp-devices "cuda:0,rocm:0" --tp-weights "0.73,0.27" -m model.gguf -p "Hello"

# Preview TP configuration without running
./build_v2_release/llaminar2 -tp 4 --tp-scope local --explain-placement --dry-run -m model.gguf
```

### YAML Configuration

For complex setups, use a YAML config file:

```yaml
# orchestration.yaml
tp_degree: 2
tp_scope: local
tp_devices: [cuda:0, cuda:1]
tp_weights: [0.6, 0.4]
backend: nccl
```

```bash
./build_v2_release/llaminar2 --config orchestration.yaml -m model.gguf -p "Hello"
```

---

## MPI Development Best Practices

### Critical Patterns

```cpp
// ALWAYS use barriers around collective operations
MPI_Barrier(MPI_COMM_WORLD);
cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, 1.0f, 0.0f);
MPI_Barrier(MPI_COMM_WORLD);

// Safe reduction pattern
float local_sum = compute_local_sum();
float global_sum;
MPI_Barrier(MPI_COMM_WORLD);
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
```

### MPI Debugging

```bash
# Log all MPI collectives with timing
LLAMINAR_MPI_LOG_COLLECTIVES=1 LLAMINAR_MPI_LOG_TIMING=1 \
mpirun -np 2 ./build_v2_release/llaminar2 -m model.gguf -p "test"

# Enable checksum verification (slow)
LLAMINAR_MPI_VERIFY_CHECKSUMS=1 mpirun -np 2 ./build_v2_release/llaminar2 -m model.gguf -p "test"
```

### Rank Comparison Testing

```bash
./scripts/compare_ranks.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello" -n 20
```

---

## Performance Optimization

### NUMA-Aware Allocation

All allocations >=128KB on hot paths MUST use NUMA first-touch initialization:

```cpp
// Parallel init for local NUMA placement
#pragma omp parallel for
for (size_t i = 0; i < buffer->numel(); ++i) {
    buffer->mutable_data()[i] = 0.0f;
}
```

**Impact**: +10-40% on multi-socket systems for large models (K/V cache access).

### Threading Strategy

```cpp
if (total_elements < 8192) {
    openblas_set_num_threads(1);  // Single-threaded
} else if (total_elements < 1048576) {
    openblas_set_num_threads(omp_get_max_threads());  // Multi-threaded
} else {
    // Use MPI distribution + local threading
}
```

---

## Code Quality Guidelines

### Assertion Framework

Llaminar uses a systematic assertion framework that is **automatically enabled** in Debug and Integration builds, and **compiled out** in Release builds.

**Build Type Behavior**:
| Build Type | `LLAMINAR_ASSERTIONS_ACTIVE` | Buffer Validation | NaN/Inf Check |
|------------|------------------------------|-------------------|---------------|
| Debug | 1 | Auto-ON | Fail by default |
| Integration | 1 | Auto-ON | Fail by default |
| Release | 0 | Compiled out | Compiled out |

**Available Assertion Macros** (see `utils/Assertions.h`):

| Macro | Purpose | Release Behavior |
|-------|---------|------------------|
| `LLAMINAR_ASSERT(cond, msg)` | Basic condition check | No-op |
| `LLAMINAR_ASSERT_NOT_NULL(ptr, name)` | Null pointer check | No-op |
| `LLAMINAR_ASSERTF(cond, msg_stream)` | Formatted message | No-op |
| `LLAMINAR_ASSERT_CAST(result, type, desc)` | `dynamic_cast` validation | No-op |
| `LLAMINAR_UNREACHABLE(msg_stream)` | Unreachable code marker | **Always active** |
| `LLAMINAR_SNAPSHOT_ASSERT*` | Snapshot-only assertions | No-op |

**Automatic Buffer Validation** (DeviceGraphExecutor):

When assertions are active, the DeviceGraphExecutor automatically validates stage outputs after each execution:

- **NaN/Inf detection**: Fails by default (catches numerical bugs early)
- **Zero-tensor detection**: Warns but doesn't fail (set `LLAMINAR_FAIL_ON_ZERO=1` for strict mode)
- **No per-stage code needed**: Validation is done by the framework using `getDumpInfo()`

**Override at Runtime**:
```bash
# Disable validation even in Debug builds
LLAMINAR_VALIDATE_BUFFERS=0 ./build_v2/llaminar2 ...

# Enable strict zero-tensor checking
LLAMINAR_FAIL_ON_ZERO=1 ./build_v2/llaminar2 ...

# Disable NaN/Inf failure (just warn)
LLAMINAR_FAIL_ON_NAN=0 ./build_v2/llaminar2 ...
```

**Usage in Stages**:

For type-safe tensor access, use `dynamic_cast` + `typed_data()` with assertion macros:

```cpp
auto* q_q16 = dynamic_cast<Q16_1Tensor*>(params_.Q);
LLAMINAR_ASSERT_CAST(q_q16, "Q16_1Tensor", "Q tensor");
const Q16_1Block* data = q_q16->typed_data();  // Type-safe access
```

### Tensor Verification System (TensorVerification.h)

The tensor verification system provides **automatic stage boundary validation** in Debug and Integration builds. It validates inputs BEFORE and outputs AFTER each stage execution, throwing `VerificationFailure` exceptions with full context on failure.

**Automatic Behavior** (Debug/Integration builds):
- `validate_buffers = true` - Validates stage outputs after execution
- `validate_inputs = true` - Validates stage inputs before execution  
- `fail_on_nan = true` - Throws exception immediately on NaN/Inf detection
- `dump_on_failure = true` - Automatically dumps all stage buffers to disk

**When a VerificationFailure Exception is Thrown**:

The exception includes full context and a formatted error message:
```
╔══════════════════════════════════════════════════════════════════╗
║               TENSOR VERIFICATION FAILED                          ║
╠══════════════════════════════════════════════════════════════════╣
║ Layer:  3
║ Stage:  FusedAttentionWoStage
║ Phase:  EXIT
║ Tensor: attention_output
║ Reason: Contains 5 NaN values in first 8 rows
║
║ Dump:   /tmp/llaminar_verification_dump/20260101_143022_456_layer3_FusedAttentionWoStage_EXIT
╚══════════════════════════════════════════════════════════════════╝
```

**Finding and Analyzing Buffer Dumps**:

When verification fails, buffers are dumped to `/tmp/llaminar_verification_dump/<timestamp>_layer<N>_<stage>_<phase>/`:

```bash
# List recent verification dumps
ls -lt /tmp/llaminar_verification_dump/ | head

# Examine dump contents
cd /tmp/llaminar_verification_dump/20260101_143022_456_layer3_FusedAttentionWoStage_EXIT/
cat manifest.json          # Lists all dumped tensors with shapes/types
cat input_Q_metadata.txt   # Metadata for specific tensor
xxd input_Q.bin | head     # Raw binary data

# Analyze in Python
python3 -c "
import numpy as np
data = np.fromfile('output_attention_output.bin', dtype=np.float32)
print(f'Shape hint from manifest, check manifest.json')
print(f'NaN count: {np.isnan(data).sum()}')
print(f'Inf count: {np.isinf(data).sum()}')
print(f'Min/Max: {data[~np.isnan(data)].min():.6f} / {data[~np.isnan(data)].max():.6f}')
"
```

**Dump Directory Contents**:
- `manifest.json` - JSON with all tensors, shapes, types, and verification results
- `input_<name>.bin` - Raw binary data for each input tensor
- `input_<name>_metadata.txt` - Shape, dtype, and element count
- `output_<name>.bin` - Raw binary data for each output tensor
- `output_<name>_metadata.txt` - Shape, dtype, and element count

**Key Files**:
- `src/v2/tensors/TensorVerification.h` - Verification system implementation
- `src/v2/execution/DeviceGraphExecutor.cpp` - Integration with `verifyStageEntry()`/`verifyStageExit()`
- `src/v2/utils/DebugEnv.h` - `ValidationConfig` struct with environment variable parsing

### Logging Standards

```cpp
LOG_ERROR("Critical failures that prevent operation");
LOG_WARN("Concerning conditions");
LOG_INFO("Important runtime information");
LOG_DEBUG("Detailed debugging");
LOG_TRACE("Verbose execution tracing");
```

### Centralized Environment Access (debugEnv)

All hot-path code MUST use `debugEnv()` instead of `std::getenv`:

```cpp
// BEFORE (bad)
if (std::getenv("LLAMINAR_ATTN_MICRO_TRACE")) { ... }

// AFTER (good)
const auto& env = debugEnv();
if (env.attention.micro_trace) { ... }
```

### Testing New Features

```cpp
TEST(Test__MyNewKernel, BasicFunctionality) {
    auto input = TestTensorFactory::createFP32Random({32, 32});
    auto output = TestTensorFactory::createFP32({32, 32});
    
    auto kernel = KernelFactory::createMyKernel(input.get());
    ASSERT_TRUE(kernel->execute(output.get()));
    
    EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(output.get()));
}
```

### Formatted ASCII Tables (libfort)

**MANDATORY**: All formatted ASCII/Unicode tables MUST use the **libfort** library instead of manual `std::cout` + `std::setw` formatting. This applies to:
- Test result tables (parity results, benchmark summaries)
- Diagnostic output tables (topology, device info)
- Any tabular data displayed to stdout

**Why libfort**:
- Automatic column width sizing based on content
- Consistent Unicode box-drawing borders
- No manual padding calculations or `setw()` juggling
- Cleaner, more maintainable code

**Reference Implementation**: See `tests/v2/integration/parity/ParityTestBase.h` for complete examples:
- `renderParityTable()` - Layer-by-layer parity results
- `renderTPParityTable()` - Multi-device tensor parallel parity
- `renderDecodeParityTable()` - Incremental decode parity

**Basic Usage Pattern**:

```cpp
#include "fort.hpp"

void renderMyTable(const std::vector<Result>& results) {
    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    
    // Header row
    table << fort::header << "Name" << "Value" << "Status" << fort::endr;
    
    // Set column alignments
    table.column(0).set_cell_text_align(fort::text_align::left);
    table.column(1).set_cell_text_align(fort::text_align::right);
    table.column(2).set_cell_text_align(fort::text_align::center);
    
    // Data rows
    for (const auto& r : results) {
        table << r.name << r.value << (r.passed ? "✓" : "✗") << fort::endr;
    }
    
    // Optional separator before summary
    table << fort::separator;
    table << "TOTAL" << total_value << "" << fort::endr;
    
    std::cout << table.to_string();
}
```

**Anti-Pattern (DO NOT USE)**:

```cpp
// ❌ BAD - Manual formatting is error-prone and hard to maintain
std::cout << "╔═══════════╦═══════════════╦══════╗\n";
std::cout << "║" << std::setw(10) << "Name" << " ║"
          << std::setw(13) << "Value" << " ║"
          << std::setw(5) << "OK" << "║\n";
// ... more manual box-drawing ...
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_LOG_LEVEL` | Logging verbosity (ERROR/WARN/INFO/DEBUG/TRACE) | INFO |
| `LLAMINAR_PROFILING` | Enable all profiling (kernel timing + executor overhead) | Disabled |
| `LLAMINAR_PROFILE_KERNELS` | (Legacy) Enable per-kernel timing in benchmark mode | Disabled |
| `LLAMINAR_EXECUTOR_PROFILING` | (Legacy) Enable per-stage profiling in DeviceGraphExecutor | Disabled |
| `LLAMINAR_VALIDATE_BUFFERS` | Enable buffer validation after stage execution | Auto-ON in Debug/Integration |
| `LLAMINAR_VALIDATE_INPUTS` | Enable input validation before stage execution | Auto-ON in Debug/Integration |
| `LLAMINAR_FAIL_ON_ZERO` | Fail on zero tensors during validation | Disabled |
| `LLAMINAR_FAIL_ON_NAN` | Fail on NaN/Inf during validation | Auto-ON in Debug/Integration |
| `LLAMINAR_DUMP_ON_FAILURE` | Dump stage buffers to disk when verification fails | Enabled |
| `LLAMINAR_STAGE_DUMP_ENABLED` | Master enable for stage dumping (0/1) | Disabled |
| `LLAMINAR_STAGE_DUMP_DIR` | Output directory for stage dumps | `/tmp/llaminar_stage_dumps` |
| `LLAMINAR_STAGE_DUMP_TYPES` | Stage types to dump (e.g., `FUSED_ATTENTION_WO,GEMM`) | `all` |
| `LLAMINAR_STAGE_DUMP_NAMES` | Stage names to dump (substring match) | `all` |
| `LLAMINAR_STAGE_DUMP_LAYERS` | Comma-separated layer indices to dump | `all` |
| `LLAMINAR_STAGE_DUMP_ITERATION` | Decode iterations to dump | `all` |
| `LLAMINAR_STAGE_DUMP_RANK` | MPI rank to dump (-1 for all) | 0 |
| `LLAMINAR_STAGE_DUMP_MAX` | Max dumps per stage type | 100 |
| `LLAMINAR_STAGE_DUMP_INPUTS` | Dump input tensors (0/1) | 1 |
| `LLAMINAR_STAGE_DUMP_OUTPUTS` | Dump output tensors (0/1) | 1 |
| `LLAMINAR_STAGE_DUMP_WEIGHTS` | Dump weight tensors (0/1) | 1 |
| `LLAMINAR_STAGE_DUMP_ASYNC` | Use async I/O for dumps (0/1) | 1 (enabled) |
| `LLAMINAR_STAGE_DUMP_ASYNC_THREADS` | Number of async I/O threads (1-16) | 2 |
| `LLAMINAR_STAGE_OUTPUT_PRINT` | Print stage outputs to log (0/1) | Disabled |
| `LLAMINAR_STAGE_OUTPUT_PRINT_N` | Elements per row in stage output print | 8 |
| `LLAMINAR_STAGE_OUTPUT_PRINT_ROWS` | Rows to print (first and last) | 2 |
| `LLAMINAR_STAGE_OUTPUT_PRINT_STAGES` | Stage names to print (substring match) | `all` |
| `LLAMINAR_DETERMINISTIC` | Force deterministic execution | Disabled |
| `LLAMINAR_CPU_PREFILL_PARTICIPATE` | Enable CPU participation in PREFILL phase (Option C fallback for memory-constrained systems) | Disabled |
| `LLAMINAR_WEIGHT_STREAMING` | Enable weight streaming for VRAM-constrained systems (Option B) | Disabled |
| `LLAMINAR_STREAM_MEMORY_MB` | GPU memory budget for weight streaming cache (0 = auto) | 0 |
| `LLAMINAR_STREAM_PREFETCH_DEPTH` | Layers to prefetch ahead during streaming | 1 |
| `LLAMINAR_STREAM_EVICTION_POLICY` | Cache eviction policy: lru, fifo, none | lru |
| `LLAMINAR_STREAM_VERBOSE` | Verbose logging for weight streaming operations | Disabled |
| `LLAMINAR_SNAPSHOT_TENSOR_DUMP` | Enable raw tensor dump to disk | Disabled |
| `LLAMINAR_SNAPSHOT_DUMP_DIR` | Output directory for tensor dumps | `/tmp/llaminar_tensor_dumps` |
| `LLAMINAR_SNAPSHOT_DUMP_LAYERS` | Comma-separated layer indices to dump | `all` |
| `LLAMINAR_SNAPSHOT_DUMP_STAGES` | Comma-separated stage names to dump | `all` |
| `LLAMINAR_MPI_LOG_COLLECTIVES` | Log MPI collective operations | Disabled |
| `LLAMINAR_MPI_LOG_TIMING` | Log timing of MPI operations | Disabled |
| `LLAMINAR_MPI_VERIFY_CHECKSUMS` | Verify checksums before/after MPI ops | Disabled |
| `LLAMINAR_DEQUANT_STATS` | Log per-tensor dequant stats | Disabled |
| `LLAMINAR_TRACE_TRANSFERS` | Enable H2D/D2H transfer tracing (0/1) | Disabled |
| `LLAMINAR_TRACE_TRANSFERS_STACKTRACE` | Include C++23 stacktrace in transfer logs | Disabled |
| `LLAMINAR_TRACE_TRANSFERS_THROW` | Throw exception on transfer (for debugging) | Disabled |
| `LLAMINAR_TRACE_TRANSFERS_MIN_BYTES` | Minimum transfer size to trace (bytes) | 0 (all) |
| `LLAMINAR_TRACE_TRANSFERS_ONLY_D2H` | Only trace Device-to-Host transfers | Disabled |

For the full list, see `src/v2/utils/DebugEnv.h`.

### Transfer Tracing for Coherence Debugging

The transfer tracing system helps identify unnecessary or wasteful memory transfers between host and device. This is critical for GPU performance optimization.

**Example Usage**:
```bash
# Trace all large D2H transfers (often indicates bugs like data() called in hot path)
LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1 \
LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1000000 \
./build_v2_release/llaminar2 -m model.gguf -p "test" -n 10

# Throw exception on first large transfer (for stack trace debugging)
LLAMINAR_TRACE_TRANSFERS=1 \
LLAMINAR_TRACE_TRANSFERS_THROW=1 \
LLAMINAR_TRACE_TRANSFERS_MIN_BYTES=1000000 \
./build_v2_release/llaminar2 -m model.gguf -p "test"
```

**Common Issues Detected**:
- `LOG_DEBUG` or `LOG_TRACE` statements calling `tensor->data()` (triggers D2H)
- Debug code unconditionally calling `data()` even when logging is disabled
- Forgotten `ensureOnDevice()` calls causing re-uploads every iteration

---

## Documentation and Project Resources

### Key Documentation

| File | Purpose |
|------|---------|
| `.github/copilot-instructions.md` | This file - development guidelines |
| `.github/instructions/llaminar-architecture-v2.instructions.md` | High-level architecture map |
| `.github/instructions/llaminar-v2-architecture.instructions.md` | Detailed implementation notes |
| `docs/v2/DISTRIBUTED_ARCHITECTURE_IMPLEMENTATION.md` | MPI tensor parallelism design |
| `docs/v2/SNAPSHOT_FRAMEWORK_DESIGN.md` | Snapshot system documentation |

### Key Source Directories

| Directory | Purpose |
|-----------|---------|
| `src/v2/inference/` | IInferenceRunner interface and factory |
| `src/v2/execution/` | ComputeGraph, DeviceGraphExecutor, ComputeStages |
| `src/v2/pipelines/qwen/` | GraphOrchestrator, Qwen2Graph, buffer specs |
| `src/v2/config/` | OrchestrationConfig, OrchestrationConfigParser (CLI/YAML parsing) |
| `src/v2/kernels/cpu/` | CPU kernels (GEMM, attention, primitives) |
| `src/v2/kernels/cpu/jit/` | JIT infrastructure (RegisterGuard, RegisterAllocation, JitMicrokernelBase) |
| `src/v2/kernels/cpu/attention/q8_1/jit/` | JIT attention microkernels (Q8DotProduct, OnlineSoftmax, etc.) |
| `src/v2/tensors/` | Tensor types (FP32, BF16, quantized) |
| `src/v2/loaders/` | GGUF loading, WeightManager |
| `src/v2/utils/` | MPIContext, MPITopology, Tokenizer, Sampler, logging |
| `src/v2/cmake/` | CMake modules (EnforceTypedRegisters, etc.) |

### Writing Changelogs

Place in `changelog/` with ISO date prefix:
```bash
# Example: changelog/2025-12-22-feature-description.md
```

Include: summary, code changes with paths, test results, performance metrics, next steps.

---

## Conclusion

This project emphasizes production reliability over peak theoretical performance:

1. **Empirical testing beats theoretical assumptions** - Always measure real performance
2. **MPI barriers are critical** - Prevent hangs with proper synchronization
3. **Adaptive approaches work** - Different backends for different problem sizes
4. **Graceful degradation** - Always have reliable fallback paths
5. **Threading is complex** - Start simple, optimize incrementally

When in doubt, prioritize correctness and reliability over raw performance.
