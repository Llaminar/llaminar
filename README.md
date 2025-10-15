# Llaminar

A high-performance, MPI-first LLM inference engine with multi-architecture pipeline support, adaptive backend selection, and distributed execution capabilities.

**Key Features:**
- ✨ **Multi-Architecture Pipeline**: Factory-based system supporting Qwen, LLaMA, and extensible to new model families
- 🚀 **Adaptive Backend Selection**: Intelligent routing between OpenBLAS (low latency) and COSMA (high throughput)
- 🔧 **Distributed Execution**: MPI-aware kernels with 1D tensor sharding and NUMA optimization
- 📊 **Comprehensive Observability**: Structured environment controls, performance counters, and validation

## Quick Start

### Building
```bash
# Configure and build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

## Debug & Instrumentation Environment Variables

The following environment variables enable additional logging, validation, and instrumentation when developing or debugging Llaminar:

| Variable | Purpose | Values / Default | Notes |
|----------|---------|------------------|-------|
| `LLAMINAR_DEQUANT_STATS` | Emit per-tensor dequantization statistics (min, max, mean, sample values). | `0` (disabled) or `1` (enabled); default: disabled | Affects all quantized tensor formats. Uses `logDequantStats` in `model_loader.cpp`. |
| `LLAMINAR_DEQUANT_ANOMALIES` | Log anomalies during Q6_K (and future) dequant: NaN, Inf, or extreme magnitudes. | Any non-empty value enables | Currently instrumented for `Q6_K`. Falls back to `LLAMINAR_DEQUANT_STATS` if that is enabled. |
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Minimum sequence length to engage COSMA-prefill path. | Integer; default: `4096` | See COSMA Prefill plan docs. |
| `ADAPTIVE_DISABLE_COSMA` | Force-disable COSMA path regardless of length. | Any non-empty value disables | Helpful for A/B comparisons. |
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Soft cap on resident COSMA working set in MiB. | Integer; default: `2048` | Prefill manager may fallback if estimate exceeds budget. |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Enable small tile correctness spot-check (relative L2) vs OpenBLAS. | Integer (tile size) or `0` to disable; default: `0` | Debug / validation only. |
| `LLAMINAR_COSMA_LOG_LEVEL` | Verbosity for COSMA prefill instrumentation. | `trace|debug|info|warn|error`; default: `info` | Independent of global logging level. |
| `OMP_NUM_THREADS` / `OMP_*` | Controls OpenMP threading behavior. | See `run_llaminar.sh` | Script sets optimal defaults (NUMA-aware binding). |
| `KMP_AFFINITY` / `KMP_BLOCKTIME` | Fine-grain thread affinity & spin-wait tuning. | See script defaults | Only relevant for Intel OpenMP runtime. |
| `LLAMINAR_EMBED_TRACE` | Enable detailed per-rank embedding buffer diagnostics (min/max, NaN/Inf counts, token & value preview). | Any non-empty (not `0`) enables; default: disabled | Produces TRACE logs labelled `MPIEmbeddingKernel[local_pre_gather|global_post_gather]`. Combine with `LLAMINAR_EMBED_TRACE_TOKENS` / `LLAMINAR_EMBED_TRACE_DIMS`. |
| `LLAMINAR_EMBED_TRACE_TOKENS` | Limit number of tokens shown in embedding diagnostics. | Positive integer; default: `2` | Ignored unless `LLAMINAR_EMBED_TRACE` enabled. |
| `LLAMINAR_EMBED_TRACE_DIMS` | Limit number of embedding dims per token shown in diagnostics. | Positive integer; default: `8` | Ignored unless tracing enabled. |
| `LLAMINAR_EMBED_FAIL_FAST` | Abort immediately if a non-finite (NaN/Inf) value is detected after copying an embedding row. | Any non-empty (not `0`) enables; default: disabled | Provides detailed token/dim context then `abort()`. Without this set a WARN is logged and execution continues. |

### Quick Usage Examples

```bash
# Enable dequant statistics and anomaly logging
export LLAMINAR_DEQUANT_STATS=1
export LLAMINAR_DEQUANT_ANOMALIES=1

# Force OpenBLAS-only path for comparison
export ADAPTIVE_DISABLE_COSMA=1

# Lower threshold to exercise COSMA prefill with small sequences
export LLAMINAR_COSMA_PREFILL_THRESHOLD=512

# Enable embedding diagnostics
export LLAMINAR_EMBED_TRACE=1
export LLAMINAR_EMBED_TRACE_TOKENS=4
export LLAMINAR_EMBED_TRACE_DIMS=16
# Fail fast on non-finite embedding values
export LLAMINAR_EMBED_FAIL_FAST=1
```

> Tip: Keep instrumentation disabled for performance benchmarking; some options add measurable overhead (e.g., validation tiles, verbose stats).

```

### Running (Canonical Method)
```bash
# Always use the canonical launcher for optimal performance
./run_llaminar.sh [arguments]


# Examples:
./run_llaminar.sh -v --print-topology                    # System info
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v  # Model inference
```

The `run_llaminar.sh` script automatically configures optimal MPI and OpenMP settings:
- **OpenMP**: 28 threads per socket, socket placement, close binding
- **MPI**: 1 process per socket with memory pinning and NUMA awareness
- **Threading**: Adaptive backend selection (single/multi/distributed)

### Manual MPI Execution (Advanced)
```bash
# If canonical script is unavailable
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  ./build/llaminar [arguments]
```

## Benchmark Mode

Llaminar includes a dedicated `--benchmark` mode for clean performance measurement with separate prefill and decode phase timing.

### Running Benchmarks

```bash
# Standard benchmark (both phases with intelligent defaults)
# Automatically generates ~512 token prompt if -p not provided
# Default: 128 decode tokens
./run_llaminar.sh --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf

# Custom prompt and decode length
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Your custom prompt here" \
  -n 50

# Direct MPI execution (2 processes)
mpirun -np 2 --bind-to socket --map-by socket \
  ./build/llaminar --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Explain machine learning in simple terms." \
  -n 100
```

### Phase-Specific Benchmarking

**Prefill-Only** (skip decode, useful for testing large context processing):
```bash
./run_llaminar.sh --benchmark -m model.gguf -p "Long prompt..." -n 0
```

**Decode-Only** (skip prefill, useful for testing autoregressive generation):
```bash
./run_llaminar.sh --benchmark -m model.gguf -p "" -n 128
```

### Benchmark Defaults

- **Prefill**: Automatically generates ~512 token prompt with mixed technical/narrative content if `-p` not provided
- **Decode**: Default 128 tokens (`-n 128`)
- **Sampling**: Greedy (argmax) for deterministic results
- **Logging**: ERROR level only for clean output
- **Phase Control**: Set `-n 0` for prefill-only or `-p ""` for decode-only

### Benchmark Output

The benchmark mode provides:
- **Clean formatted output** with box-drawing characters
- **Separate metrics** for prefill and decode phases
- **Tokens/second throughput** for each phase and total
- **Generated text preview** to verify model functionality (decode only)
- **Minimal logging** (ERROR level only) for accurate timing
- **Phase skipping** - Shows "(SKIPPED)" for omitted phases, omits TOTAL when only one phase runs

**Standard Benchmark (both phases):**
```
Tokenizing prompt... done (8 tokens)
Tokens: [840, 20772, 5662, 6832, 304, 4285, 3793, 13]
Running prefill... done (1216.49 ms, 6.58 tok/s)
Running decode... done (48095.52 ms, 1.04 tok/s)

Generated text:
Machine learning is a type of artificial intelligence...

╔══════════════════════════════════════════════════════════════╗
║                    INFERENCE BENCHMARK                       ║
╠══════════════════════════════════════════════════════════════╣
║ Model: models/qwen2.5-0.5b-instruct-q8_0.gguf              ║
║ Backend: OpenBLAS                                          ║
╠══════════════════════════════════════════════════════════════╣
║ PREFILL PHASE                                                ║
║   Tokens:              8 tokens                              ║
║   Time:          1216.49 ms                                 ║
║   Throughput:       6.58 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:             50 tokens                              ║
║   Time:         48095.52 ms                                 ║
║   Throughput:       1.04 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ TOTAL                                                        ║
║   Tokens:             58 tokens                              ║
║   Time:         49312.01 ms                                 ║
║   Throughput:       1.18 tok/s                             ║
╚══════════════════════════════════════════════════════════════╝
```

**Prefill-Only (`-n 0`):**
```
╔══════════════════════════════════════════════════════════════╗
║ PREFILL PHASE                                                ║
║   Tokens:            512 tokens                              ║
║   Time:         15234.78 ms                                 ║
║   Throughput:      33.60 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                (SKIPPED)        ║
╚══════════════════════════════════════════════════════════════╝
```

**Decode-Only (`-p ""`):**
```
╔══════════════════════════════════════════════════════════════╗
║ PREFILL PHASE                                   (SKIPPED)    ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:            128 tokens                              ║
║   Time:        123456.78 ms                                 ║
║   Throughput:       1.04 tok/s                             ║
╚══════════════════════════════════════════════════════════════╝
```

### Benchmark Tips

- **Use Release builds** for accurate performance measurement (Debug builds are 5-10x slower)
- **Vary prompt length** to test prefill scaling (prefill throughput improves with more tokens)
- **Control decode length** with `-n` flag to measure sustained decode performance
- **Disable instrumentation** - Ensure environment variables like `LLAMINAR_COSMA_VALIDATE_TILE` are unset
- **Greedy sampling** - Benchmark uses greedy sampling for deterministic, reproducible results
- **Phase-specific tests** - Use `-n 0` or `-p ""` to isolate prefill or decode performance

## Performance Optimization

### NUMA-Aware Memory Allocation

Llaminar automatically optimizes memory allocation for NUMA (Non-Uniform Memory Access) systems to ensure tensor data is local to the CPU cores that will access it, significantly improving performance for large models.

**Optimized Allocations:**
- **Model Weights** (ModelLoader): Multi-GB weight tensors use parallel first-touch initialization
- **K/V Cache** (SimpleTensor): 96MB-4GB cache tensors allocated with NUMA locality
- **Activation Tensors** (SimpleTensor): Large intermediate activations (≥128KB) use parallel initialization

**Configuration:**
```bash
# NUMA optimization enabled by default
# Disable if needed:
export LLAMINAR_NUMA_FIRST_TOUCH=0

# Enable diagnostic logging to verify NUMA locality:
export LLAMINAR_NUMA_VERIFY_LOCALITY=1
```

**How It Works:**
- Allocations ≥128KB trigger parallel initialization using OpenMP
- Each thread initializes its portion of the tensor ("first-touch" policy)
- Operating system places memory pages on the NUMA node where they're first accessed
- Ensures data locality for subsequent access by the same threads

**Performance Impact:**
- Small models (≤1B): +1-3% improvement
- Large models (7B-13B): +10-40% improvement on multi-socket systems
- Most significant for K/V cache access patterns during decode

**Implementation Details:**
- Threshold: 128KB (matches ModelLoader for consistency)
- Method: Parallel `std::fill` in `SimpleTensor::resize()` and constructor
- Thread count: Uses `omp_get_max_threads()` (auto-detected from system)
- Files: `src/tensors/SimpleTensor.h`, `src/ModelLoader.cpp`

## Development Profiling (Advanced)

For developers analyzing parallelization efficiency and conducting detailed prefill performance studies, Llaminar provides a separate GTest-based profiling suite:

```bash
# Run prefill performance benchmarks with parallelization analysis
./run_performance_bench.sh

# Run specific test suites
./run_performance_bench.sh --filter "OpenBLAS_StrongScaling*"
./run_performance_bench.sh --filter "COSMA_ModelShapes*"
```

**Note:** This is distinct from `--benchmark` mode:
- **`run_llaminar.sh --benchmark`**: Production inference benchmarking with real model execution
- **`run_performance_bench.sh`**: Development profiling with GTest-based analysis of parallelization efficiency

The performance bench script provides detailed efficiency metrics (>90% excellent, <50% poor) for tuning threading and backend selection strategies.

## Test Organization

Llaminar has a comprehensive test suite organized into 5 categories for different development workflows:

| Category | Tests | Runtime | Pass Rate | Use Case |
|----------|-------|---------|-----------|----------|
| **Smoke** | 14 | 1.16s | 100% | Every build, pre-commit hooks |
| **Unit** | 99 | 2m30s | 89% | Pull requests, development |
| **Parity** | 4 | 24.5s | 75% | Cross-implementation validation |
| **Integration** | 26 | 3m0s | 81% | Major features, releases |
| **Full** | 123 | 3m0s | 86% | Pre-release, comprehensive |

### Quick Test Commands

```bash
# Smoke tests - Ultra-fast validation (1.16s)
# VSCode: Ctrl+Shift+P → "Tasks: Run Task" → "test: smoke tests"
ctest --test-dir build --output-on-failure --parallel \
  -R "^(BasicTest|NumaTest|PipelineFactoryTest|DequantTest|TPPartitionSpecTest|LargeMatmulPlanTest|WeightRoleClassification|MPILinearKernelTest|MPIRMSNormKernelTest|MPIAttentionKernelTest|MPISoftmaxCorrectnessTest|RMSNormCoreCorrectness|SoftmaxCoreCorrectness|LinearOrientationCorrectnessTest)$"

# Unit tests - Comprehensive kernel validation (2m30s)
# VSCode: Ctrl+Shift+P → "Tasks: Run Task" → "test: unit tests"
ctest --test-dir build --output-on-failure --parallel \
  -E "(Integration|ParityFramework|Incremental|Qwen|Prefill|.*Stress.*)"

# Full test suite (3m0s)
# VSCode: Ctrl+Shift+P → "Tasks: Run Task" → "test: run all tests"
ctest --test-dir build --output-on-failure --parallel
```

### Test Naming Conventions

**"Parity" Tests** - Reserved for cross-implementation validation:
- `test_parity_framework.cpp` - Validates against llama.cpp and PyTorch
- `test_abstract_pipeline_parity.cpp` - Pipeline interface consistency

**"Correctness" Tests** - Internal validation and unit tests:
- `test_*_correctness.cpp` - Self-contained correctness checks
- Mathematical properties, component contracts, internal consistency

### Smoke Test Coverage (1.16s)

The smoke test suite provides maximum coverage with minimal runtime:

- ✅ **Infrastructure**: MPI initialization, NUMA topology, pipeline factory
- ✅ **Core Kernels**: Linear, RMSNorm, Attention, Softmax (distributed & local)
- ✅ **Tensor Operations**: Partition specs, quantization/dequantization
- ✅ **Configuration**: Weight role classification, COSMA planning

**Recommended Usage**: Run smoke tests on every build during development for instant feedback.

### Detailed Documentation

- **[Test Categories Reference](docs/TEST_CATEGORIES.md)** - Complete test organization guide
- **[Task Profile Validation](TASK_PROFILE_VALIDATION.md)** - Detailed validation report
- **[Smoke Test Documentation](SMOKE_TEST_COMPLETE.md)** - Comprehensive smoke test guide

## Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Llaminar Architecture                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│ Model Config │ (GGUF metadata: arch, dims, layers)
└──────┬───────┘
       │
       v
┌────────────────────┐
│ PipelineFactory    │ ──> Registered creators: {"qwen", "llama", ...}
└────────┬───────────┘
         │ create(config)
         v
┌──────────────────────────────────────────────────────────────┐
│              AbstractPipeline (interface)                    │
│  • prefill(tokens, weights, ctx)                            │
│  • decode(token, weights, ctx)                              │
│  • logits(out_logits)                                       │
│  • loadWeights(path) -> IModelWeights                       │
└──────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┐
         v                 v                 v
  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
  │QwenPipeline │   │LlamaPipeline│   │Future Models │
  │  Adapter    │   │  Adapter    │   │   (plugin)   │
  └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
         │                 │                  │
         └────────┬────────┴──────────────────┘
                  │
                  v
         ┌────────────────────┐
         │ Per-Layer Execution│
         └────────┬───────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    v             v             v
┌─────────┐  ┌───────┐  ┌─────────┐
│Attention│  │  MLP  │  │ RMSNorm │
└────┬────┘  └───┬───┘  └────┬────┘
     │           │           │
     └───────┬───┴───┬───────┘
             │       │
             v       v
    ┌────────────────────────┐
    │MatMulBackendSelector  │ (Centralized policy)
    └────────┬───────────────┘
             │
      ┌──────┴──────┐
      │             │
      v             v
 ┌─────────┐  ┌──────────┐
 │OpenBLAS │  │  COSMA   │
 │ (local) │  │ (distrib)│
 └─────────┘  └──────────┘

Execution Flow:
1. Factory creates architecture-specific pipeline
2. Pipeline loads weights via IModelWeights interface
3. Prefill: Large batch processing with COSMA path (if threshold met)
4. Decode: Single-token autoregressive with local OpenBLAS
5. Backend selection: Automatic based on operation size and stage
```


### Core Architecture Principles

**Multi-Architecture Pipeline System:**
- All model families implement `AbstractPipeline` interface
- `PipelineFactory` provides registration and creation
- Architecture-specific adapters (Qwen, LLaMA) handle model differences
- Clean separation: pipeline orchestrates, kernels compute, selector chooses backend

**MPI-First Design:**
- All kernels derive from `MPIKernelBase`
- Handles both single-rank (np=1) and multi-rank execution
- NUMA-aware topology detection and binding
- Legacy non-MPI kernels removed (LinearKernel, AttentionKernel, RMSNormKernel, MatMulKernel)

**Centralized Backend Selection:**
- `MatMulBackendSelector` is the single decision point
- Stage-aware: different policies for prefill vs decode
- Size-aware: small ops → single-thread, medium → multi-thread, large → distributed
- Environment-driven: `ADAPTIVE_DISABLE_COSMA`, `LLAMINAR_COSMA_PREFILL_THRESHOLD`

**Graceful Degradation:**
- COSMA failures automatically fall back to OpenBLAS
- Validation and diagnostics available via environment flags
- Performance counters track backend decisions and execution time

### Attention Output Contract (Post-Gather Removal)
The MPI attention kernel now always emits a per-rank *partial* output (only the contribution from
that rank's owned heads after the output projection). There is no internal gather or reduction.

Caller / pipeline responsibilities:
1. Determine reconstruction semantics: for the current row-sharded `Wo` distribution the per-rank
  outputs are additive and should be summed: `MPI_Allreduce` (SUM) over the full `[seq_len, d_model]` buffer.
2. Perform the reduction before any consumer treats the tensor as a fully replicated activation
  (e.g., before residual add or feeding into RMSNorm/MLP expecting full hidden state).
3. Avoid accidental direct use of the partial result as replicated. An optional safety canary can be
  enabled with `LLAMINAR_ASSERT_REPLICATED_MISUSE=1`, which injects a microscopic per-rank marker in
  the final element to aid detection during debugging.

Removed flags: `LLAMINAR_DISABLE_ATTENTION_GATHER` and `LLAMINAR_LEGACY_ATTENTION_GATHER`—they are no
longer recognized. All documentation or scripts referencing them should be updated accordingly.

Rationale: Eliminating the hidden gather makes communication explicit, prevents double reductions,
and simplifies extending the sharded pathway (e.g., future head-concat vs additive strategies).

### Supported Quantization Formats (Current)

The loader and dequant pipeline currently support the following GGUF tensor types:

- Q4_0
- Q5_0
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q4_K_M (alias layout of Q4_K with fused min variant)
- Q5_K
- Q6_K
- Q8_K
- F16 / F32 (unquantized)

Deprecated / removed (no longer recognized and will trigger an error if encountered in a model):

- Q4_1 (id 3)
- Q5_1 (id 7)
- Q8_1 (id 9)

Reason for removal: These legacy *_1 formats added maintenance complexity (especially Q5_1 with divergent 22 vs 24 byte block variants) without providing clear performance or accuracy wins over the maintained set. Models should be re-converted using any of the supported formats above. The loader now emits a clear diagnostic if a deprecated numeric id is found.

Implications:
1. Existing GGUF files using Q4_1 / Q5_1 / Q8_1 must be re-exported (e.g., via llama.cpp quantization tools) to one of the supported formats.
2. Tests referencing the removed formats have been pruned (e.g., Q5_1 layout micro test).
3. Numeric ids for removed formats are reserved; they will not be repurposed to avoid silent misinterpretation.

## COSMA Prefill (Phase 1)

Phase 1 introduces an optional COSMA-backed prefill path focused on large context construction (long sequence length matrix multiplications). Autoregressive decoding and small matrix products stay on the adaptive OpenBLAS path to avoid communication overheads.

### Engagement Criteria
- Enabled only when: (a) `seq_len >= LLAMINAR_COSMA_PREFILL_THRESHOLD`, (b) world size > 1, and (c) `ADAPTIVE_DISABLE_COSMA` is NOT set.
- Below a conservative volume (`m * n * k < LLAMINAR_COSMA_FAST_PATH_THRESHOLD`), a multi-rank "fast path" executes local OpenBLAS GEMMs plus a broadcast to avoid COSMA overhead.

### Data Flow (Simplified)
1. Row-major activation is converted (or reused if budget denied) into a temporary COSMA layout.
2. Weights are streamed into COSMA-distributed buffers (Phase 1: float32 only; quant fusion planned).
3. `cosma::multiply` executes with MPI barriers before/after to avoid collective hazards.
4. Optional validation tile spot-check (OpenBLAS reference on a small top-left tile) computes relative L2 error.
5. Outputs remain in COSMA layout until needed by elementwise kernels; copy back to row-major as required.

### Environment Variables (Prefill Specific)
| Variable | Purpose | Notes |
|---------|---------|-------|
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Sequence length required to enable COSMA prefill (default 4096). | Set lower to force early testing. |
| `LLAMINAR_COSMA_FAST_PATH_THRESHOLD` | Volume (`m*n*k`) below which a replicated local GEMM path is used. | Avoids COSMA startup & comm for small ops. |
| `ADAPTIVE_DISABLE_COSMA` | Forces all ops onto OpenBLAS adaptive paths. | A/B performance & debugging. |
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Soft upper bound for any single COSMA allocation (Phase 1). | Denies activation/weight conversion if size exceeds budget. |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Tile size (tokens) for relative L2 correctness check. | Rank 0 warns if `rel_l2 > 1e-3`. |
| `LLAMINAR_COSMA_LOG_LEVEL` | Prefill log verbosity (`trace,debug,info,warn,error`). | Independent of global logger. |
| `LLAMINAR_COSMA_DUMP_STATS` | Emit aggregate counters at shutdown. | Use `1` or `true`. |
| `LLAMINAR_COSMA_DUMP_STATS_PATH` | Override destination for JSON stats when dump enabled. | Defaults to `cosma_prefill_stats.json`. |
| `LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT` | Disable fused quantized weight dequant + COSMA population (revert to two-step path). | Safety fallback for new fused path. |

## Tensor Parallel (TP) Simulation (Output Projection)

The TP simulation path allows exercising row- or column-partitioned attention output projection logic
without requiring an actual multi-process tensor parallel deployment. It reconstructs the full
projection result from per-partition matmuls performed serially inside a single process. This is
primarily a correctness and heuristic development tool.

### Goals
1. Validate reconstruction correctness of row vs column partitioning strategies.
2. Prototype auto heuristic (choose column if hidden dimension divisible by partitions; else row if sequence length divisible).
3. Provide a stable harness for future performance modeling before integrating real multi-rank TP.

### Environment Flags
| Variable | Purpose | Values / Default | Notes |
|----------|---------|------------------|-------|
| `LLAMINAR_TP_WO_SIM_ENABLE` | Enable TP simulation path in the attention output projection. | `0` or unset = disabled, any non-zero enables | When disabled, baseline single GEMM is used. |
| `LLAMINAR_TP_WO_SIM_PARTITIONS` | Number of simulated partitions. | Integer ≥2; default: 2 | Drives row/col slicing. |
| `LLAMINAR_TP_WO_SIM_MODE` | Force partition axis. | `row`, `col`, `auto` (default) | `auto`: column if `d_model % parts == 0`, else row if `seq_len % parts == 0`, else column fallback. |

### How It Works
1. Baseline path: single matmul `Y = Attended * W_O` (shape: `[seq_len, d_model] = [M,K] * [K,K]`).
2. Simulation path:
  - Column mode: Slice output columns (N) into partitions. Pack each weight sub-block (K x N_p) and compute per-part GEMM, then stitch column regions.
  - Row mode: Slice input rows (M). For each partition perform GEMM on its row slice; stitch via row memcpy.
3. Reconstruction yields a buffer bitwise equivalent (within floating rounding) to the baseline GEMM.

### Guarantees & Tolerances
Current test tolerances: `max_abs < 1e-5`, `relative L2 < 1e-6` across random inputs.

### Test Coverage
| Test | Scope | Notes |
|------|-------|-------|
| `AttentionTPSimParityTest` | Direct executor parity (row, col, auto) | Verifies manual reconstruction logic. |
| `AttentionTPSimIntegrationTest` | Environment-driven kernel path | Ensures env toggles route through simulation branch producing identical output. |
| `ParityFramework.*` | Cross-implementation validation | Snapshot capture and comparison framework for correctness testing. |

### Parity Testing Framework ✨

**Automatic Pipeline Validation**: The parity framework enables comprehensive correctness validation by capturing and comparing intermediate tensor states across different implementations or execution paths.

**Key Features**:
- **Zero Overhead**: Completely disabled by default (compile-time elimination)
- **Strategic Capture Points**: 8 key stages in QwenPipeline (embedding, attention, FFN, logits)
- **Automatic Activation**: Set `LLAMINAR_PARITY_CAPTURE=1` to enable snapshot capture
- **Configurable Tolerances**: Per-stage absolute and relative error thresholds
- **Cross-Implementation**: Compare Llaminar vs llama.cpp or incremental vs prefill paths

**Captured Stages**:
- **EMBEDDING**: Token embedding output (input to first layer)
- **ATTENTION_NORM**: Pre-attention RMSNorm (QKV input)
- **ATTENTION_OUTPUT**: Attention output projection
- **ATTENTION_RESIDUAL**: Post-attention residual connection
- **FFN_NORM**: Pre-FFN RMSNorm
- **FFN_DOWN**: FFN down projection output
- **FFN_RESIDUAL**: Post-FFN residual (layer output)
- **FINAL_NORM**: Final RMSNorm before LM head
- **LM_HEAD**: Language model head logits

**Usage Example**:
```bash
# Enable parity capture during inference
export LLAMINAR_PARITY_CAPTURE=1
mpirun -np 2 ./build/llaminar -m model.gguf -v

# Run parity tests
./build/test_parity_framework
```

**Comparison Metrics**:
- **Max Absolute**: `max(|expected - actual|)`
- **Mean Absolute**: `mean(|expected - actual|)`
- **Relative L2**: `||expected - actual||₂ / ||expected||₂`

**Architecture**:
- `src/pipeline_stages.h`: Standardized 22-stage enumeration
- `src/parity_hooks.h/cpp`: Production-safe capture interface
- `tests/parity_test_framework.h/cpp`: Full snapshot registry and comparison
- Integration via `AbstractPipeline` and `PipelineBase` helpers

See `.github/instructions/llaminar-architecture.instructions.md` §15 for detailed documentation.

### Usage Example
```bash
export LLAMINAR_TP_WO_SIM_ENABLE=1
export LLAMINAR_TP_WO_SIM_PARTITIONS=4
export LLAMINAR_TP_WO_SIM_MODE=auto
ctest --test-dir build -R AttentionTPSimIntegration -V
```

### Limitations / Future Work
| Area | Planned Enhancement |
|------|---------------------|
| Performance Modeling | Introduce timing instrumentation and FLOP accounting per partition. |
| Mixed Partition Heuristics | Combine head-wise distribution with TP simulation for hybrid scenarios. |
| Real Multi-Rank TP | Replace serial loop with per-rank execution + collectives. |
| Overlap | Explore streaming of column weight tiles with compute. |

Reference implementation lives in `MPIAttentionKernel::computeLocalOutputProjection` guarded by the
`debugEnv().tp_sim.*` snapshot fields. A dedicated executor (`tp_output_projection_executor`) isolates
the partitioned GEMM loops for reuse.

### Instrumentation Counters
Exposed via `CosmaPrefillManager::stats()`:
- `single_rank_calls`, `fast_path_calls`, `cosma_path_calls`
- `bytes_streamed_weights`, `bytes_converted_activations`
- `matmul_invocations`, `validation_tile_checks`
- Accumulated microseconds: `us_stream_weights`, `us_convert_activation`, `us_matmul`

Strategy cache performance: `strategy_hits`, `strategy_misses` from `StrategyCache`.

### Memory Budget (Phase 1b Behavior)
Allocations are now tracked cumulatively. A request is denied when either the single allocation or the projected resident total would exceed `LLAMINAR_COSMA_MAX_RESIDENT_MB`. Releasing COSMA matrices updates the running total, ensuring large prefill sequences respect the soft budget.

### Validation Tile
When enabled, rank 0 recomputes a top-left `(T x T)` GEMM using the original row-major operands with OpenBLAS and compares against the distributed result (after gathering), logging a warning if relative L2 > 1e-3 or NaN. This provides an inexpensive early corruption detector without full matrix duplication.

### Known Limitations (Phase 1)
- No fused dequant + layout yet (weights assumed float32 during streaming).
- No transpose support for weight matrices (requests ignored with warning).
- No cumulative resident memory tracking (single-allocation guard only).
- Elementwise ops (e.g., softmax, RMSNorm) may trigger layout conversions.

### Roadmap (Planned Improvements)
- Fused quantized weight dequant + COSMA block population.
- Block-wise elementwise ops directly in distributed layout.
- Overlapping next weight stream with current GEMM (double buffering).
- Cumulative memory accounting & adaptive tiling when over budget.
- FlashAttention-style attention kernel integration.

### Debug Tips
- Force COSMA path for shorter sequences by lowering `LLAMINAR_COSMA_PREFILL_THRESHOLD`.
- Disable COSMA quickly with `export ADAPTIVE_DISABLE_COSMA=1` for parity/perf baselines.
- Use `LLAMINAR_COSMA_VALIDATE_TILE=64` (or similar) only for debugging—adds conversion overhead.
- Capture shutdown stats: `export LLAMINAR_COSMA_DUMP_STATS=1`.

### Reference
Design details and acceptance criteria are tracked in `.github/instructions/cosma-prefill-plan.instructions.md` (Phase 1).

## Llaminar

High-performance, distributed LLM inference engine.

## Key Documentation

- [Canonical Launch & Runtime Summary](docs/canonical-launch-summary.md)
- [MPI Barrier Guidelines](docs/mpi_barrier_guidelines.md)
- [Partial Attention Output Details](docs/attention-partial-output.md)
- [Tensor Parallel Architecture](docs/tensor_parallel_architecture.md)  <!-- Newly added -->

## Build

See `./run_llaminar.sh -h` for runtime options or consult the development guidelines in `.github/copilot-instructions.md`.
