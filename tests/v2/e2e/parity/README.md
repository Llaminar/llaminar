# E2E Parity Tests

End-to-end parity tests that validate Llaminar inference against various reference implementations.

## Directory Structure

```
e2e/parity/
├── README.md                          # This file
├── gtest_mpi_main.cpp                 # MPI-aware GTest main for multi-rank tests
│
├── pytorch/                           # Against PyTorch ground truth
│   └── qwen2/
│       └── Test__FP32Pipeline_vs_PyTorch.cpp
│
├── internal/                          # Llaminar modes compared against each other
│   ├── q8_1_vs_fp32/                 # Quantized vs FP32 path
│   │   └── Test__Q8_1Pipeline_vs_FP32Pipeline_LayerByLayer.cpp
│   └── mpi_vs_single_rank/           # Tensor parallel vs single rank
│       └── Test__MPIPipeline_vs_SingleRank.cpp
│
└── kernels/                           # Individual kernel comparisons
    └── attention/
        ├── Test__JitFusedAttentionWo_vs_FP32Attention.cpp      # FA2 JIT kernel
        └── Test__SimpleAttentionQ8_1_vs_FP32Attention.cpp      # Simple Q8_1 kernel
```

## Naming Convention

Test files follow the pattern:
```
Test__<TestedComponent>_vs_<Reference>.cpp
```

Where:
- **TestedComponent**: What is being tested (e.g., `FP32Pipeline`, `Q8_1Pipeline`, `JitFusedAttentionWo`)
- **Reference**: What it's compared against (e.g., `PyTorch`, `FP32Pipeline`, `SingleRank`)

## Test Categories

### PyTorch Reference (`pytorch/`)

Ground truth validation against PyTorch reference implementation.
- **Tested**: Llaminar FP32 inference pipeline
- **Reference**: PyTorch snapshots from `python/reference/generate_qwen2_pipeline_snapshots.py`
- **Purpose**: Ensure Llaminar produces identical results to PyTorch

### Internal Comparisons (`internal/`)

Compare different Llaminar execution modes against each other.

#### Q8_1 vs FP32 (`internal/q8_1_vs_fp32/`)
- **Tested**: Q8_1 quantized pipeline
- **Reference**: FP32 pipeline (same model, same inputs)
- **Purpose**: Identify where quantization divergence occurs

#### MPI vs Single Rank (`internal/mpi_vs_single_rank/`)
- **Tested**: Multi-rank tensor-parallel execution
- **Reference**: Single-rank execution
- **Purpose**: Validate tensor parallelism correctness

### Kernel Parity (`kernels/`)

Individual kernel comparisons in isolation (no full model loading).

#### Attention Kernels (`kernels/attention/`)
- **JitFusedAttentionWo**: FA2-style 7469-line JIT kernel vs FP32 decomposed attention
- **SimpleAttentionQ8_1**: 1082-line simple JIT kernel vs FP32 decomposed attention
- **Purpose**: Isolate kernel-level divergence from pipeline-level effects

## CTest Labels

All tests use hierarchical labels for filtering:
- `V2;E2E;Parity` - All parity tests
- `V2;E2E;Parity;PyTorch` - PyTorch ground truth tests
- `V2;E2E;Parity;Internal` - Internal comparison tests
- `V2;E2E;Parity;Kernels` - Kernel-level tests

Examples:
```bash
# Run all E2E parity tests
ctest --test-dir build_v2 -R "V2_E2E_Parity"

# Run only PyTorch parity tests
ctest --test-dir build_v2 -L "PyTorch"

# Run only kernel parity tests
ctest --test-dir build_v2 -L "Kernels" -L "Parity"
```

## Requirements

- **PyTorch tests**: Require pre-generated snapshots and ZLIB for .npz loading
- **MPI tests**: Require ENABLE_PIPELINE_SNAPSHOTS (Debug or E2ERelease build)
- **Kernel tests**: Run standalone, no model or special build flags required

## Key Findings (2025-12-25)

1. **Attention kernels are mathematically correct**: Both JitFusedAttentionWo and SimpleAttentionQ8_1 achieve ≥0.999 cosine similarity against FP32 attention with realistic data magnitudes (±10).

2. **E2E divergence is cumulative**: The ~0.85-0.89 cosine similarity at ATTENTION_CONTEXT is caused by Q/K/V projection quantization error (cos ~0.999) being amplified by softmax, not by attention kernel bugs.

3. **Improvement requires higher precision inputs**: See `changelog/2025-01-17-q8_1-attention-precision-analysis.md` for recommended approaches (Q16 activations, per-channel quantization, mixed precision).
