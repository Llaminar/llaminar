## Llaminar Test Agent & MPI Utilities Guide

This document explains how to write reliable, deadlock‑resistant tests for the Llaminar
inference engine using the shared utilities under `tests/`. It focuses on:

1. Canonical MPI initialization / finalization
2. Watchdog timeouts & hang diagnostics
3. Rank / world helpers and root-only sections

---

### 13. Parity Test Framework ✨

The parity test framework provides comprehensive snapshot capture and comparison capabilities for validating pipeline execution correctness. It's deeply integrated into the core pipeline architecture for automatic, zero-overhead parity checking.

---

### 14. PyTorch Reference Implementation for Stage-by-Stage Parity 🐍

The PyTorch reference implementation provides **ground-truth validation** for Llaminar's transformer pipeline by capturing intermediate activations from HuggingFace models at **21 granular stages** (vs llama.cpp's limited 2-stage capture).

#### Overview

**Purpose**: Validate Llaminar's correctness against the original PyTorch model implementations from HuggingFace, enabling fine-grained debugging of quantization errors, numerical drift, and operator implementation issues.

**Key Features**:
- **21 Capture Stages**: Granular snapshots covering embedding → attention → FFN → logits
- **Model Family Support**: Qwen (production), LLaMA (prototype), extensible to DeepSeek/Mistral
- **Quantization Testing**: Q4_0, Q6_K snapshot generation for quantization validation
- **Cross-Language Bridge**: Python .npz snapshots → C++ .npy loader → parity comparison
- **Zero External Deps**: Header-only .npy parser (no cnpy, no zlib)
- **Comprehensive Docs**: 850+ lines of guides, examples, troubleshooting

#### Architecture

**Directory Structure**:
```
python/reference/              # PyTorch reference implementation
├── README.md                 # 300+ line comprehensive guide
├── requirements.txt          # Python dependencies (torch, transformers, numpy)
├── run_reference.py          # CLI tool for snapshot generation
├── models/                   # Model family implementations
│   ├── base.py              # AbstractReferenceModel base class
│   ├── qwen.py              # QwenReferenceModel (production)
│   └── llama.py             # LlamaReferenceModel (prototype)
└── tests/                    # Python unit tests
    ├── test_qwen.py         # Qwen reference validation
    └── test_llama.py        # LLaMA reference validation

tests/                         # C++ integration
├── PYTORCH_INTEGRATION.md    # 450+ line integration guide
├── npz_loader.h             # Header-only .npy parser (303 lines)
├── npz_to_npy.py            # Python .npz extractor (100 lines)
└── test_parity_framework.cpp # Parity test with PyTorch comparison (DISABLED)
```

#### 21 Capture Stages

| Stage | Description | Layer-Specific? | Key Validations |
|-------|-------------|-----------------|-----------------|
| `EMBEDDING` | Token embedding lookup | No | Vocabulary alignment, embedding matrix correctness |
| `POSITIONAL_ENCODING` | Position encoding (if used) | No | RoPE, ALiBi, learned embeddings |
| `ATTENTION_NORM` | Pre-attention RMSNorm/LayerNorm | Yes | Normalization correctness, epsilon handling |
| `QKV_PROJECTION` | Query/Key/Value projections | Yes | Weight sharding, quantization errors |
| `ROPE_APPLICATION` | Rotary position embeddings | Yes | RoPE implementation, frequency bands |
| `ATTENTION_SCORES` | Q·K^T / √d_k scores | Yes | Numerical precision, softmax input |
| `ATTENTION_PROBS` | Softmax(scores) | Yes | Softmax correctness, temperature |
| `ATTENTION_CONTEXT` | Probs·V context vectors | Yes | Attention output, dropout |
| `ATTENTION_OUTPUT` | W_o projection | Yes | Output projection correctness |
| `ATTENTION_RESIDUAL` | Residual add after attention | Yes | Numerical stability |
| `FFN_NORM` | Pre-FFN RMSNorm/LayerNorm | Yes | Normalization consistency |
| `FFN_GATE` | Gate projection (SwiGLU) | Yes | Activation function correctness |
| `FFN_UP` | Up projection | Yes | Weight accuracy |
| `FFN_ACTIVATION` | Gate * SiLU(Up) | Yes | Fused activation correctness |
| `FFN_DOWN` | Down projection | Yes | Output projection |
| `FFN_RESIDUAL` | Residual add after FFN | Yes | Residual stream integrity |
| `LAYER_OUTPUT` | Final layer output | Yes | Per-layer validation |
| `FINAL_NORM` | Final RMSNorm before LM head | No | Output normalization |
| `LM_HEAD` | Language model head projection | No | Logit generation |
| `FINAL_LOGITS` | Raw logits before sampling | No | Pre-softmax validation |
| `PROBABILITIES` | Softmax(logits) for sampling | No | Probability distribution |

**Note**: Llaminar currently captures **8 strategic stages** in `QwenPipeline`. The PyTorch reference provides all 21 for comprehensive validation.

#### Quick Start

**1. Generate PyTorch Snapshots**:
```bash
cd /workspaces/llaminar

# Install dependencies (if not in devcontainer)
pip install -r python/reference/requirements.txt

# Generate snapshots for Qwen model with specific tokens
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --output pytorch_snapshots.npz \
    --verbose

# For quantized model validation
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --quantization q4_0 \
    --output pytorch_q4_snapshots.npz
```

**2. Extract Snapshots to .npy Files**:
```bash
# Extract .npz archive to individual .npy files
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/

# Directory structure after extraction:
# pytorch_snapshots/
#   EMBEDDING_-1.npy
#   ATTENTION_OUTPUT_0.npy
#   ATTENTION_OUTPUT_1.npy
#   ...
```

**3. Run C++ Parity Tests**:
```bash
# Set environment variables
export PYTORCH_SNAPSHOT_DIR=pytorch_snapshots/
export PYTORCH_SNAPSHOT_TOKENS=1,2,3,4,5

# Run parity test (currently DISABLED, needs API refactor)
./build/test_parity_framework --gtest_filter="*PyTorchReference*"
```

#### C++ Integration Components

**1. npz_loader.h (Header-Only .npy Parser)**

Core classes for loading PyTorch snapshots in C++:

```cpp
#include "npz_loader.h"
using namespace llaminar::parity;

// High-level API: Load snapshot by stage name and layer
PyTorchSnapshotLoader loader("pytorch_snapshots/");
NpyArray embedding;

if (loader.load_snapshot("EMBEDDING", -1, embedding)) {
    // embedding.shape = {1, 5, 512}  (batch, seq_len, hidden_dim)
    // embedding.data = std::vector<float> with embedding.total_elements() floats
    std::cout << "Loaded embedding: " << embedding.shape_str() << "\n";
}

// Load layer-specific snapshot
NpyArray attention_output;
if (loader.load_snapshot("ATTENTION_OUTPUT", 0, attention_output)) {
    // Layer 0 attention output
}

// Low-level API: Load .npy file directly
NpyArray array;
std::string error_msg;
if (NpzLoader::load_npy("pytorch_snapshots/EMBEDDING_-1.npy", array, error_msg)) {
    // Process array...
}
```

**Key Features**:
- **Zero Dependencies**: No cnpy, no zlib, pure C++11
- **Float32 Only**: Sufficient for ML parity testing
- **Shape Parsing**: Extracts dimensions from NumPy header dict
- **Error Handling**: Descriptive error messages for debugging

**2. npz_to_npy.py (Extraction Helper)**

Python script to convert .npz archives to individual .npy files:

```bash
# Basic usage
python tests/npz_to_npy.py input.npz output_dir/

# With verbose output
python tests/npz_to_npy.py input.npz output_dir/ --verbose

# Output directory auto-created, defaults to input.stem
python tests/npz_to_npy.py snapshots.npz  # Creates snapshots/ directory
```

**3. DISABLED Test Case in test_parity_framework.cpp**

Template for PyTorch parity testing (needs API update):

```cpp
TEST(ParityFramework, DISABLED_DistributedPipelineVsPyTorchReference) {
    // Environment: PYTORCH_SNAPSHOT_DIR, PYTORCH_SNAPSHOT_TOKENS
    PyTorchSnapshotLoader pytorch_loader(snapshot_dir);
    
    // Run Llaminar inference
    // ... (needs ModelConfig-based API refactoring)
    
    // Compare snapshots stage-by-stage
    for (auto& stage : {"EMBEDDING", "ATTENTION_OUTPUT_0", ...}) {
        NpyArray pytorch_snapshot;
        pytorch_loader.load_snapshot(stage, layer_idx, pytorch_snapshot);
        
        auto llaminar_snapshot = /* get from SnapshotRegistry */;
        auto metrics = SnapshotComparator::compare(pytorch_snapshot, llaminar_snapshot);
        
        EXPECT_LT(metrics.max_abs_diff, 1e-3f) << "Stage: " << stage;
        EXPECT_LT(metrics.rel_l2, 1e-2f) << "Stage: " << stage;
    }
}
```

**Why DISABLED**: Test uses old `GGUFContext` API, needs refactoring to use new `ModelConfig`-based `AbstractPipeline` API (~1 hour effort).

#### File Format Details

**NumPy .npy Format**:
```
[Magic: \x93NUMPY]  # 6 bytes
[Version: 1-3]      # 2 bytes (e.g., \x01\x00)
[Header Length]     # 2 bytes (little-endian)
[Header Dict]       # Python dict as string, null-padded to 64-byte boundary
                    # Example: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 10, 512), }\n"
[Data Blob]         # Raw float32 data (little-endian)
```

**Naming Convention**:
```
{STAGE_NAME}_{LAYER_INDEX}.npy

Examples:
  EMBEDDING_-1.npy           # Non-layer stage (layer index = -1)
  ATTENTION_OUTPUT_0.npy     # Layer 0 attention output
  FFN_DOWN_5.npy             # Layer 5 FFN down projection
  FINAL_NORM_-1.npy          # Final normalization (no layer)
```

#### Comparison Metrics & Tolerances

**Standard Metrics**:
- **max_abs_diff**: Maximum absolute difference across all elements
- **mean_abs_diff**: Mean absolute difference
- **rel_l2**: Relative L2 norm (||A - B||₂ / ||A||₂)

**Recommended Tolerances**:

| Model Precision | max_abs_diff | rel_l2 | Notes |
|-----------------|--------------|--------|-------|
| FP32 (float32) | 1e-4 | 1e-5 | Numerical precision only |
| FP16 (float16) | 1e-3 | 1e-3 | Half precision errors |
| Q6_K (6-bit) | 5e-3 | 1e-2 | High-quality quantization |
| Q4_0 (4-bit) | 1e-2 | 5e-2 | Significant quantization loss |

**Adaptive Tolerance Strategy**:
```cpp
float get_tolerance(const std::string& stage, QuantizationType quant) {
    // Stricter for early stages (embedding, norms)
    if (stage.find("NORM") != std::string::npos || 
        stage.find("EMBEDDING") != std::string::npos) {
        return quant == Q4_0 ? 5e-3f : 1e-4f;
    }
    
    // Relaxed for late stages (accumulated errors)
    if (stage.find("LOGITS") != std::string::npos) {
        return quant == Q4_0 ? 5e-2f : 1e-3f;
    }
    
    // Standard tolerance
    return quant == Q4_0 ? 1e-2f : 5e-4f;
}
```

#### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYTORCH_SNAPSHOT_DIR` | Directory containing extracted .npy files | `pytorch_snapshots/` |
| `PYTORCH_SNAPSHOT_TOKENS` | Comma-separated token IDs used for generation | `1,2,3,4,5` |
| `PYTORCH_MODEL_PATH` | Path to HuggingFace checkpoint (for regeneration) | `Qwen/Qwen2-0.5B-Instruct` |
| `PYTORCH_QUANTIZATION` | Quantization format (q4_0, q6_k) | `q4_0` |

#### Troubleshooting

**1. FileNotFoundError: snapshot file not found**
```bash
# Check directory structure
ls -la pytorch_snapshots/

# Verify extraction worked
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/ -v

# Check environment variable
echo $PYTORCH_SNAPSHOT_DIR
```

**2. Shape mismatch between PyTorch and Llaminar**
- **Cause**: Different sequence lengths or batch sizes
- **Fix**: Ensure `PYTORCH_SNAPSHOT_TOKENS` matches Llaminar input tokens
- **Debug**: Print shapes before comparison:
  ```cpp
  LOG_DEBUG("PyTorch shape: " << pytorch_array.shape_str());
  LOG_DEBUG("Llaminar shape: " << llaminar_snapshot.shape_str());
  ```

**3. High numeric drift (rel_l2 > 0.1)**
- **Cause**: Quantization errors accumulate across layers
- **Fix**: Use quantization-aware tolerances
- **Debug**: Check intermediate stages to isolate error source

**4. Parser error: invalid .npy header**
- **Cause**: Corrupted file or unsupported .npy version
- **Fix**: Regenerate snapshots with updated NumPy version
- **Debug**: Hex dump first 100 bytes:
  ```bash
  xxd pytorch_snapshots/EMBEDDING_-1.npy | head -20
  ```

**5. Test disabled / won't run**
- **Status**: Test template exists but needs API refactoring
- **Short-term**: Use manual integration (load snapshots in standalone tool)
- **Long-term**: Update test to use `ModelConfig`-based `AbstractPipeline` API

#### Advanced Usage

**Selective Stage Capture**:
```python
# Only capture specific stages for faster iteration
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --stages EMBEDDING ATTENTION_OUTPUT FINAL_NORM \
    --output selective_snapshots.npz
```

**Custom Token Sequences**:
```python
# Test with specific prompt patterns
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5,6,7,8,9,10 \
    --output long_sequence.npz

# Test edge cases (BOS, EOS, padding)
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 151643,1,2,151645 \  # BOS + tokens + EOS
    --output edge_cases.npz
```

**Quantization Comparison**:
```bash
# Generate FP32 reference
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3,4,5 \
    --output fp32_ref.npz

# Generate Q4_0 quantized version
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3,4,5 \
    --quantization q4_0 --output q4_ref.npz

# Compare in C++ test to quantify quantization error
```

#### CI/CD Integration (Future)

**GitHub Actions Workflow** (example):
```yaml
name: PyTorch Parity Tests

on: [push, pull_request]

jobs:
  parity-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Python dependencies
        run: pip install -r python/reference/requirements.txt
      
      - name: Cache HuggingFace models
        uses: actions/cache@v3
        with:
          path: ~/.cache/huggingface
          key: hf-models-${{ hashFiles('python/reference/requirements.txt') }}
      
      - name: Generate PyTorch snapshots
        run: |
          python python/reference/run_reference.py \
            --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
            --tokens 1,2,3,4,5 --output pytorch_snapshots.npz
      
      - name: Extract snapshots
        run: python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/
      
      - name: Build Llaminar
        run: |
          cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
          cmake --build build --parallel
      
      - name: Run parity tests
        env:
          PYTORCH_SNAPSHOT_DIR: pytorch_snapshots/
          PYTORCH_SNAPSHOT_TOKENS: 1,2,3,4,5
        run: |
          ctest --test-dir build -R PyTorchReference --output-on-failure
```

#### Documentation References

- **Python Reference README**: `python/reference/README.md` (300+ lines)
- **C++ Integration Guide**: `tests/PYTORCH_INTEGRATION.md` (450+ lines)
- **Architecture Documentation**: `.github/instructions/llaminar-architecture.instructions.md` §15
- **Parity Framework**: This document §13 (Llaminar snapshot capture)
- **Status Report**: `PYTORCH_INTEGRATION_STATUS.md` (implementation summary)

#### Best Practices

1. **Always regenerate snapshots** after changing model implementation or quantization
2. **Use version control** for snapshot .npz files (or document HuggingFace checkpoint versions)
3. **Document tolerances** in test comments with justification
4. **Test incrementally**: Start with embedding, then layer-by-layer
5. **Isolate quantization errors**: Compare FP32 → Q6_K → Q4_0 progressively
6. **Validate shapes first**: Shape mismatch indicates fundamental API issues
7. **Use verbose logging**: Enable `--verbose` and `-vvv` for debugging
8. **Cache HuggingFace models**: Avoid re-downloading multi-GB checkpoints

#### Migration Path for DISABLED Test

**Current State**: Test infrastructure complete, awaiting API alignment

**Steps to Enable** (~1 hour effort):

1. **Update model loading**:
   ```cpp
   // OLD: auto gguf_ctx = loader.load(model_path);
   // NEW:
   ModelConfig config = createConfigFromGGUF(model_path);
   ```

2. **Update pipeline creation**:
   ```cpp
   // OLD: auto pipeline = std::make_unique<QwenPipelineAdapter>(gguf_ctx);
   // NEW:
   auto pipeline = PipelineFactory::create(config);
   ```

3. **Update prefill call**:
   ```cpp
   // OLD: auto result = pipeline->prefill(token_ids);
   // NEW:
   IModelWeights* weights = /* ... */;
   StageContext ctx = /* ... */;
   bool success = pipeline->prefill(token_ids, weights, ctx);
   ```

4. **Remove DISABLED prefix**:
   ```cpp
   // TEST(ParityFramework, DISABLED_DistributedPipelineVsPyTorchReference)
   TEST(ParityFramework, DistributedPipelineVsPyTorchReference)
   ```

5. **Run and validate**:
   ```bash
   ./build/test_parity_framework --gtest_filter="*PyTorchReference*"
   ```

#### Future Enhancements

1. **Full .npz Support**: Eliminate extraction step by adding ZIP parsing to npz_loader.h
2. **Model Coverage**: Add DeepSeek, Mistral, GPT-2 reference implementations
3. **Automatic Tolerance Tuning**: Learn optimal tolerances from successful runs
4. **Differential Testing**: Compare consecutive Llaminar commits for regressions
5. **GPU Validation**: Extend to compare CUDA/ROCm kernels against PyTorch GPU
6. **Streaming Snapshots**: Handle very long sequences with disk-backed snapshots
7. **Visual Diff Tools**: HTML report showing per-stage drift with heatmaps

---



#### Overview

**Purpose**: Validate Llaminar pipeline execution against reference implementations (e.g., llama.cpp) or compare different execution paths (prefill vs incremental decode).

**Key Features**:
- **Zero Overhead**: Completely disabled by default (compile-time elimination)
- **Strategic Capture Points**: 8 key stages in QwenPipeline (embedding, attention, FFN, logits)
- **Automatic Activation**: Set `LLAMINAR_PARITY_CAPTURE=1` to enable snapshot capture
- **Configurable Tolerances**: Per-stage absolute and relative error thresholds
- **Cross-Implementation**: Compare Llaminar vs llama.cpp or incremental vs prefill paths
- **Production-Safe**: Default no-op implementation, tests provide real capture logic

#### Architecture Integration

**Core Components**:

1. **PipelineStage Enum** (`src/pipeline_stages.h`)
   - Standardized 22-stage enumeration covering all transformer operations
   - Shared between production code and tests
   - Inline conversion utilities: `stage_to_string()`, `string_to_stage()`

2. **Parity Hooks** (`src/parity_hooks.h/cpp`)
   - Production-safe interface with default no-op implementation
   - Environment-driven activation via `LLAMINAR_PARITY_CAPTURE`
   - Tests provide real implementation via `parity_test_framework.cpp`

3. **Pipeline Integration**
   - **AbstractPipeline**: Virtual `captureStageSnapshot()` and `isParityEnabled()` methods
   - **PipelineBase**: Convenience `captureIfEnabled()` helpers with rank filtering
   - **QwenPipeline**: 8 strategic capture points at key computation stages

#### Capture Points in QwenPipeline

| Stage | Location | Description |
|-------|----------|-------------|
| `EMBEDDING` | After token embedding | Input to first transformer layer |
| `ATTENTION_NORM` | Pre-attention RMSNorm | Input to QKV projection |
| `ATTENTION_OUTPUT` | After W_o projection | Attention block output |
| `ATTENTION_RESIDUAL` | After attention residual add | Input to FFN block |
| `FFN_NORM` | Pre-FFN RMSNorm | Input to gate/up projections |
| `FFN_DOWN` | After down projection | FFN block output |
| `FFN_RESIDUAL` | After FFN residual add | Output of transformer layer |
| `FINAL_NORM` | After final RMSNorm | Input to LM head |
| `LM_HEAD` | Language model head output | Final logits |

**Design Features**:
- **Rank 0 Only**: Captures happen on rank 0 to avoid MPI duplication
- **Automatic Shape Extraction**: Captures sequence length and feature dimension from tensors
- **Layer-Aware**: Each capture includes layer index (or -1 for non-layer stages)

#### Quick Start

**Building**:
```bash
# Build the parity framework test
cmake --build build --target test_parity_framework
```

**Running Basic Tests** (no MPI required):
```bash
./build/test_parity_framework --gtest_filter="ParityFramework.Basic*"
```

**Running Full Pipeline Comparison** (requires MPI + model):
```bash
# Enable parity capture
export LLAMINAR_PARITY_CAPTURE=1

# Run distributed inference (captures snapshots automatically)
mpirun -np 2 ./build/llaminar -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v

# Run parity comparison test
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.DistributedPipelineVsLlamaCpp"
```

#### Test Framework Files

**Framework Core** (`tests/`):
- `parity_test_framework.h`: Core framework API (SnapshotRegistry, SnapshotComparator, LlaminarSnapshotHook)
- `parity_test_framework.cpp`: Framework implementation
- `test_parity_framework.cpp`: Test suite demonstrating framework usage

**Production Integration** (`src/`):
- `pipeline_stages.h`: Shared PipelineStage enum
- `parity_hooks.h/cpp`: Production-safe hook interface (default no-op)
- `abstract_pipeline.h`: Virtual parity methods
- `pipeline_base.h/cpp`: Convenience helpers
- `qwen_pipeline.cpp`: 8 capture points

#### Usage Patterns

**Test Integration**:
```cpp
#include "parity_test_framework.h"

TEST(ParityTest, QwenPrefillVsReference) {
    // Clear previous captures
    parity::SnapshotRegistry::instance().clear();
    
    // Enable parity capture
    parity::LlaminarSnapshotHook::set_enabled(true);
    
    // Run Llaminar pipeline (automatic capture via captureIfEnabled calls)
    auto pipeline = PipelineFactory::create(config);
    pipeline->prefill(tokens, weights, ctx);
    
    // Run reference implementation and capture
    // ... reference execution ...
    
    // Compare snapshots
    auto tolerance = parity::ComparisonTolerance(1e-3f, 1e-4);
    auto& registry = parity::SnapshotRegistry::instance();
    
    for (int layer = 0; layer < num_layers; ++layer) {
        auto key_llama = registry.make_key("llaminar", PipelineStage::ATTENTION_OUTPUT, layer);
        auto key_ref = registry.make_key("reference", PipelineStage::ATTENTION_OUTPUT, layer);
        
        TensorSnapshot snap_llama, snap_ref;
        ASSERT_TRUE(registry.get_snapshot(key_llama, snap_llama));
        ASSERT_TRUE(registry.get_snapshot(key_ref, snap_ref));
        
        auto result = SnapshotComparator::compare(snap_ref, snap_llama, tolerance);
        EXPECT_TRUE(result.passed()) << "Layer " << layer << " failed parity";
    }
}
```

**Custom Capture Points** (extending to new architectures):
```cpp
class MyCustomPipeline : public PipelineBase, public AbstractPipeline {
    bool execute(...) override {
        // ... computation ...
        
        // Capture at custom stage
        captureIfEnabled(PipelineStage::CUSTOM, layer_idx, my_tensor);
        
        // ... more computation ...
        return true;
    }
};
```

#### Comparison Metrics

**Supported Metrics**:
- **Max Absolute Difference**: `max(|expected - actual|)`
- **Mean Absolute Difference**: `mean(|expected - actual|)`
- **Relative L2 Norm**: `||expected - actual||₂ / ||expected||₂`
- **Worst Element Tracking**: Index and values of maximum difference

**Configurable Tolerances**:
```cpp
// Strict tolerance for early layers
auto strict = ComparisonTolerance(1e-4f, 1e-5);

// Relaxed tolerance for final logits (quantized models)
auto relaxed = ComparisonTolerance(1e-2f, 1e-3);
```

#### Environment Variables

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LLAMINAR_PARITY_CAPTURE` | Enable automatic snapshot capture | Disabled (0) | Set to `1` to enable |
| `LLAMINAR_PARITY_COMPARE` | Enable automatic comparison | Disabled (0) | Reserved for future use |

**Note**: The parity framework (`LLAMINAR_PARITY_CAPTURE`) is complementary to the legacy layer token diff system (`LLAMINAR_LAYER_TOKEN_DIFF`):
- **Parity Framework**: Cross-implementation validation (Llaminar vs llama.cpp)
- **Layer Token Diff**: Incremental decode vs replay validation within Llaminar

#### Example Test Cases

**Test Case 1: BasicSnapshotCapture**
```cpp
TEST(ParityFramework, BasicSnapshotCapture) {
    auto& registry = parity::SnapshotRegistry::instance();
    registry.clear();
    
    // Capture a snapshot
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    parity::LlaminarSnapshotHook::set_enabled(true);
    parity::LlaminarSnapshotHook::capture(
        PipelineStage::EMBEDDING, -1, data.data(), 2, 2);
    
    // Retrieve and verify
    auto key = registry.make_key("llaminar", PipelineStage::EMBEDDING);
    EXPECT_TRUE(registry.has_snapshot(key));
}
```

**Test Case 2: SnapshotComparison**
```cpp
TEST(ParityFramework, SnapshotComparison) {
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> actual   = {1.001f, 2.001f, 2.999f, 3.999f};
    
    TensorSnapshot snap_expected({}, expected.data(), expected.size());
    TensorSnapshot snap_actual({}, actual.data(), actual.size());
    
    auto tolerance = ComparisonTolerance(1e-2f, 1e-3);
    auto result = SnapshotComparator::compare(snap_expected, snap_actual, tolerance);
    
    EXPECT_TRUE(result.passed());
    EXPECT_LT(result.metrics.max_abs_diff, 1e-2f);
}
```

**Test Case 3: DistributedPipelineVsLlamaCpp**
```cpp
TEST(ParityFramework, DistributedPipelineVsLlamaCpp) {
    // Run llama.cpp reference
    auto llama_result = run_llama_cpp_inference(model_path, tokens);
    
    // Run Llaminar with parity capture enabled
    parity::LlaminarSnapshotHook::set_enabled(true);
    auto llaminar_result = run_llaminar_inference(model_path, tokens);
    
    // Compare final outputs
    auto tolerance = ComparisonTolerance(1e-3f, 1e-4);
    auto result = SnapshotComparator::compare(llama_result, llaminar_result, tolerance);
    EXPECT_TRUE(result.passed()) << "Parity check failed";
}
```

#### Example Output

```
[PARITY_TEST] Using model: models/qwen2.5-0.5b-instruct-q4_0.gguf
[PARITY_TEST] Running llama.cpp reference...
[PARITY_TEST] Running Llaminar pipeline...
[PARITY_TEST] Comparing results...
[PARITY_LOGITS] max_abs=0.00123 mean_abs=0.00045 rel_l2=0.000234
[PARITY_FINAL_HIDDEN] max_abs=0.00089 mean_abs=0.00032 rel_l2=0.000156
[PARITY_TEST] ✓ All stages passed parity checks
```

#### Adding New Parity Tests

**To add parity checking for a new pipeline stage**:

1. **Add capture hook in pipeline code** (production):
```cpp
// In your pipeline's execute method
captureIfEnabled(PipelineStage::YOUR_STAGE, layer_idx, tensor);
```

2. **Extract corresponding data from reference** (test):
```cpp
// Capture reference implementation output
parity::LlaminarSnapshotHook::capture(
    PipelineStage::YOUR_STAGE,
    layer_idx,
    reference_data,
    seq_len,
    feature_dim
);
```

3. **Add test case comparing snapshots**:
```cpp
TEST(YourTest, StageXParity) {
    auto& registry = parity::SnapshotRegistry::instance();
    auto key_actual = registry.make_key("llaminar", PipelineStage::YOUR_STAGE, layer);
    auto key_ref = registry.make_key("reference", PipelineStage::YOUR_STAGE, layer);
    
    TensorSnapshot snap_actual, snap_ref;
    ASSERT_TRUE(registry.get_snapshot(key_actual, snap_actual));
    ASSERT_TRUE(registry.get_snapshot(key_ref, snap_ref));
    
    auto result = SnapshotComparator::compare(snap_ref, snap_actual, tolerance);
    EXPECT_TRUE(result.passed());
}
```

#### Known Limitations & Workarounds

1. **llama.cpp Intermediate States**
   - **Issue**: Public llama.cpp API doesn't expose intermediate layer states
   - **Workaround**: Framework validates final hidden state (via embeddings API) and final logits
   - **Future**: Automatic llama.cpp hooking or creative embeddings API usage

2. **MPI Distribution**
   - **Issue**: Current implementation captures snapshots on rank 0 only
   - **Workaround**: For multi-rank validation, gather snapshots explicitly before comparison
   - **Future**: Multi-rank snapshot validation with automatic gathering

3. **Quantization Precision**
   - **Issue**: Quantized models (Q4, Q6) have inherent precision loss
   - **Workaround**: Use relaxed tolerances for quantized models (e.g., `1e-2` instead of `1e-4`)
   - **Best Practice**: Document expected tolerance ranges for each quantization format

4. **Memory Overhead**
   - **Issue**: Storing full tensor snapshots can be large for long sequences
   - **Workaround**: Selective stage capture, capture only last token row
   - **Future**: Streaming snapshots to disk, differential snapshots for incremental decode

#### Performance Considerations

**Overhead When Disabled** (production default):
- **Zero Overhead**: `isParityEnabled()` compiles to constant `false`
- Capture calls eliminated by compiler dead code elimination
- No runtime checks in hot paths

**Overhead When Enabled** (testing only):
- **Memory**: Stores full tensor snapshots (~MBs for typical sequences)
- **Compute**: Tensor copy per capture point (~microseconds for typical sizes)
- **Recommendation**: Use only for validation, not performance benchmarks

#### Troubleshooting

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| No snapshots captured | `LLAMINAR_PARITY_CAPTURE` not set | Export the environment variable before running |
| Capture points not hit | Execution taking fast path | Check logs, may be using replicated small-seq path |
| High numeric drift | Quantization or numerical instability | Use relaxed tolerances, check for NaN/Inf |
| Test hangs during comparison | MPI mismatch or barrier issue | Add TestTimeoutGuard, check rank coordination |
| Snapshot key not found | Wrong stage name or layer index | Verify key construction matches capture call |

#### Future Enhancements

**Planned Features**:
1. **Selective Stage Capture**: Environment variable to capture only specific stages
2. **Streaming Snapshots**: Write captures to disk instead of memory for long sequences
3. **Differential Snapshots**: Only capture changed regions for incremental decode
4. **Automatic Tolerance Tuning**: Learn tolerances from successful runs
5. **Cross-Rank Comparison**: Validate distributed state consistency across MPI ranks
6. **CI/CD Integration**: Automated parity regression tests on every commit
7. **llama.cpp Auto-Hooking**: Automatic intermediate state extraction from llama.cpp

**Plugin Architecture** (future):
```cpp
class ParityPlugin {
    virtual bool shouldCapture(PipelineStage stage, int layer) = 0;
    virtual void onCapture(const TensorSnapshot& snapshot) = 0;
};

// Register custom parity validation logic
parity::registerPlugin(std::make_unique<MyCustomValidator>());
```

#### Documentation References

- **Architecture Details**: `.github/instructions/llaminar-architecture.instructions.md` §15
- **Main README**: `README.md` (Parity Testing Framework section)
- **Example Tests**: `test_parity_framework.cpp`, `test_prefill_attention_golden.cpp`
- **Integration Patterns**: `src/qwen_pipeline.cpp` (8 capture points)

#### Best Practices

1. **Always use TestTimeoutGuard** for parity tests involving MPI or heavy computation
2. **Clear registry before each test** to avoid cross-contamination
3. **Use deterministic seeds** for reproducible comparisons
4. **Document expected tolerances** for each quantization format in test comments
5. **Prefer stage-by-stage comparison** over final-output-only for better diagnostics
6. **Test both fast paths and distributed paths** to ensure comprehensive coverage
7. **Add parity tests alongside new kernels** to catch regressions early

#### Integration Checklist

When adding parity testing to a new component:

- [ ] Add PipelineStage enum value (if needed)
- [ ] Add `captureIfEnabled()` call at strategic point
- [ ] Create reference implementation or extract from llama.cpp
- [ ] Write test case with appropriate tolerances
- [ ] Document expected numeric behavior
- [ ] Add to CI pipeline (if applicable)
- [ ] Update AGENTS.md if introducing new patterns

---

root‑only sections
4. Patterns for distributed numeric correctness tests
5. Environment flag scoping and cleanup

---

### 1. MPI Environment (`test_mpi_utils.h`)

Include the header in any test that touches distributed code:

```cpp
#include "test_mpi_utils.h"
```

Add the canonical main at the bottom of the translation unit **once per test binary**:

```cpp
LLAMINAR_DEFINE_GTEST_MPI_MAIN();
```

This guarantees:
- Idempotent `MPI_Init_thread` with `MPI_THREAD_FUNNELED`
- A world barrier before `MPI_Finalize()` (reduces OpenMPI noisy warnings)
- Automatic cleanup if a test aborts early (RAII at‑exit hook)

Available helpers (namespace `llaminar::test_util::MPIEnvironment`):

```cpp
int rank();              // MPI rank (0-based)
int world();             // world size
bool is_root();          // rank()==0 convenience
void barrier();          // MPI_Barrier(MPI_COMM_WORLD)
template<class Fn> void root_only(Fn&& fn); // executes fn on rank 0 only
void skip_unless_world(int expected);       // GTEST_SKIP if world size mismatch
void skip_unless_world_at_least(int min);   // GTEST_SKIP if world smaller
```

Example usage in a fixture:

```cpp
class DistFixture : public ::testing::Test {
  void SetUp() override {
    using Env = llaminar::test_util::MPIEnvironment;
    world_ = Env::world();
    rank_  = Env::rank();
  }
  int world_ = 1;
  int rank_  = 0;
};
```

If you need a custom `main()` (rare), call:

```cpp
MPIEnvironment::init(&argc, &argv);
// RUN_ALL_TESTS
MPIEnvironment::finalize();
```

---

### 2. Timeout / Hang Watchdog (`test_timeout_guard.h`)

Long‑running or collective‑heavy tests should guard against silent hangs:

```cpp
#include "test_timeout_guard.h"
using llaminar::test_util::TestTimeoutGuard;

TEST(DistributedOp, CompletesUnderBudget) {
  TestTimeoutGuard guard("DistributedOp", 
      TestTimeoutGuard::ResolveTimeout({"LLAMINAR_TEST_TIMEOUT_MS"}, std::chrono::milliseconds(60000)));
  // ... test body ...
}
```

If the timeout elapses you get:
- Rank & world size
- Elapsed ms vs budget
- Symbolized stack trace (where supported)
- Process abort (fast feedback in CI)

Environment override example:

```bash
export LLAMINAR_TEST_TIMEOUT_MS=15000
```

---

### 3. Root‑Only Assertions & Reference Paths

To avoid redundant heavy reference computations:

```cpp
auto &Env = llaminar::test_util::MPIEnvironment;
std::vector<float> reference;
if (Env::is_root()) {
  reference = build_reference();
}

// Broadcast reference size then data if needed.
size_t sz = reference.size();
MPI_Bcast(&sz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
reference.resize(sz);
MPI_Bcast(reference.data(), (int)sz, MPI_FLOAT, 0, MPI_COMM_WORLD);

if (Env::is_root()) {
  EXPECT_LT(rel_l2(out, reference), 1e-5) << "drift detected";
}
```

When only rank 0 asserts, failures are still reported normally (other ranks exit after barrier/finalize).

---

### 4. Mixed Zero‑Tile & Replicated Fallback Testing

Some distributed strategies yield a "mixed zero‑tile" distribution (one or more ranks hold no tiles).
The fused RMSNorm+QKV path now auto‑falls back to a replicated host implementation in that case.

Testing pattern:

```cpp
auto fused = mgr.fused_rmsnorm_qkv(...);
bool ok_norm = fused.normalized.host_owned || fused.normalized.original_row_major;
if (!ok_norm) {
  GTEST_SKIP() << "Fallback not materialized"; // defensive skip rather than crash
}
```

Add counters (already exported): `mixed_zero_tile_fallbacks` for telemetry.

---

### 5. Environment Flag Hygiene

Use the built‑in RAII helper `MPIEnvironment::ScopedEnvVar` to apply and automatically
restore environment variable overrides:

```cpp
using Env = llaminar::test_util::MPIEnvironment;

TEST(PathSelect, ForcesCOSMA) {
  Env::ScopedEnvVar force_cosma("LLAMINAR_COSMA_FORCE", "1");
  // test body with COSMA forced
}

TEST(DisableLogTemporarily, UnsetLogLevel) {
  // Temporarily unset an existing variable
  Env::ScopedEnvVar unset_log("LLAMINAR_COSMA_LOG_LEVEL");
  // ... operations with variable removed ...
}
```

Key behaviors:
- Previous value (or absence) is restored on destruction.
- Passing nullptr (or omitting the second argument) unsets the variable for the scope.
- Move operations transfer ownership; copying is disabled to prevent double restores.

---

### 6. Choosing Sequence Lengths & Dimensions

For small sequence lengths + multi‑rank runs you are more likely to trigger zero‑tile distributions.
If your test specifically wants distributed ownership across all ranks, increase `seq_len` until each
rank reports a non‑zero local tile (see logs from `CosmaPrefillManager`). Conversely, to explicitly
exercise fallback paths choose minimal lengths (e.g. 16–64) known to produce mixed layouts.

---

### 7. Pattern Reference: Fused RMSNorm+QKV Test

See `test_cosma_fused_rmsnorm_qkv.cpp` for a complete example combining:
- MPI main macro
- Root‑only assertions
- Replicated fallback tolerance
- Deterministic random seeds

---

### 8. Adding a New Distributed Test (Checklist)

1. `#include "test_mpi_utils.h"` (and `test_timeout_guard.h` if needed)
2. Define your fixture (cache rank/world in `SetUp()` if you use them repeatedly)
3. (Optional) Install `TestTimeoutGuard` first in the test body
4. Set deterministic seeds for reproducibility
5. Materialize / broadcast any reference data once (root)
6. Execute distributed path
7. Reconstruct row‑major outputs (if needed) via manager utilities
8. Root rank: perform numeric assertions (rel L2, max abs)
9. `LLAMINAR_DEFINE_GTEST_MPI_MAIN();` at file end

---

### 9. Troubleshooting

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| OpenMPI warning about missing finalize | Missing barrier/finalize ordering | Use macro main or call `MPIEnvironment::finalize()` explicitly |
| Hang during collective | Mismatched ranks entering collective | Add `TestTimeoutGuard`, insert diagnostic barriers, ensure all ranks pass conditionals |
| Mixed zero‑tile fallback unexpected | Sequence length too small | Increase seq_len or adjust distribution thresholds |
| Sporadic numeric drift | Non-deterministic seed use | Fix seed (`std::mt19937 gen(42)`) across ranks + broadcast weights |

---

### 10. Future Extensions

Planned shared utilities (not yet implemented):
- Scoped environment variable helper (if adoption grows)
- Reference broadcast wrappers
- Automatic per‑rank log prefix injection

Contributions welcome—follow existing patterns in `test_mpi_utils.h` and submit a PR.

---

Maintainers: Keep this guide updated whenever test infrastructure changes (e.g. new counters,
new fallback modes, or new distributed kernels).

---

### 11. Case Study: Resolving a Persistent Hang in `test_attention_shard_parity`

We encountered a hard hang (no stdout/stderr, Ctrl+C ineffective until forced) while developing the `test_attention_shard_parity` harness. Root cause narrowed to interaction between:

1. Linking both a custom `main()` and `gtest_main` (duplicate entry / conflicting startup order)
2. Static initialization inside shared test utilities (`test_mpi_utils.h`) touching MPI state before `MPI_Init_thread`
3. Invoking weight sharding logic prior to confirmed MPI init (amplifying undefined ordering)

#### Symptoms
* `mpirun -np 2 ./build/test_attention_shard_parity` produced zero expected early logs.
* Single-process run also produced no early main prints.
* Strace showed normal dynamic loader activity, then apparent stall before our user-space diagnostics.

#### Investigation Timeline
| Step | Action | Observation |
|------|--------|-------------|
| 1 | Added static ctor probe printing before `main()` | No output → stall before probe executed or output suppressed |
| 2 | Replaced macro main with explicit custom main | Hang persisted |
| 3 | Removed `gtest_main` from linkage | Hang persisted |
| 4 | Created `minimal_mpi_probe` binary | Probe ran fine (MPI itself OK) |
| 5 | Strace on hanging binary | Loaded libs; no forward progress into our code |
| 6 | Rewrote test w/out any GTest/utilities | Hang disappeared |

#### Implemented Fix
Replaced the test with a standalone minimal MPI program:
* No GTest, no `test_mpi_utils.h`, no watchdog utilities
* Explicit `main()` performs: early print → `MPI_Init_thread` → build tensors + run kernel → aggregate + compare → `MPI_Finalize`
* Linked only core library + MPI; excluded `gtest_main`

#### Why It Worked
Eliminating layered static initialization removed hidden ordering dependencies and duplicate main symbol ambiguity. MPI now initializes deterministically before any higher-level constructs run.

#### Guidelines to Prevent Recurrence
1. If writing a custom main, never link against `gtest_main`
2. When a hang occurs pre-output:
  * Add a static ctor probe printing to `stderr`
  * Spin up a minimal probe binary (MPI init/finalize only)
  * Strip test down to raw MPI + simplest reproduction, then reintroduce utilities incrementally
3. Avoid indirect MPI calls from static/global objects in test files
4. Prefer first validating new distributed logic in a minimal harness before integrating shared utilities

#### Parity Harness Enhancements (Post-Fix)
After stabilization we added:
* Deterministic weight init + per-rank head sharding
* Rank 0 naive reference: QKV matmuls → RoPE → causal masked softmax → output projection
* Distributed partial sum reconstruction exploiting row-partitioned `Wo`
* Parity metrics (max_abs, rel_l2) with tight thresholds (1e-5) achieving ~1e-9 max_abs

#### Future Hardening Ideas
* CI stage running distributed tests once in "raw" mode (no utilities)
* Optional build flag to instrument static initializers with timestamp + rank
* Environment toggle to disable utilities (`LLAMINAR_NO_TEST_UTILS`) for isolation A/B

Use this case study as a playbook for future unexplained pre-main hangs.

---

### 12. Lesson Learned: Stale Test Binary + Misaligned GGUF Data Region (Float Tensor Parity Regression)

While adding IQ quant formats we hit a dramatic failure in the golden loader test:

* All F32 tensors (layer norms, output norm, biases) showed tiny denormal values (~5–20e-39)
* Relative L2 errors exploded (1e+36 – 1e+39) vs llama.cpp reference
* Quantized tensors appeared structurally fine, masking the root cause

#### Root Cause
Two interacting issues:
1. The loader previously recorded the raw file position immediately after parsing tensor metadata, without aligning to the GGUF-mandated 32-byte boundary before the tensor data blob. Reads began a few bytes early inside padding / preceding quant payload bytes, so F32 tensors deserialized junk that still formed valid (but meaningless) subnormal floats.
2. The golden test executable had not been relinked after updating `model_loader.cpp`; incremental build rebuilt `libllaminar_core.a` only. CTest kept executing an older binary lacking the alignment fix, prolonging the investigation.

#### Diagnostic Signals That Solved It
| Signal | Insight |
|--------|---------|
| Added first-10 tensor dump (name, offset, size, type) | Confirmed offset chain looked internally consistent but needed upstream comparison |
| Upstream gguf comparison block (mismatches=0) | Proved header + tensor metadata parsing was correct (enum realignment OK) |
| Logging aligned vs raw `data_offset` (raw_pos=... aligned_pos=...) | Showed a 3-byte padding adjustment was required (alignment=32) |
| Direct test binary run (bypassing CTest) after full rebuild | Passed immediately → stale binary suspicion confirmed |

#### Fix
* Seek to the next 32-byte boundary after metadata (`(pos + align - 1)/align*align`) and use that as `data_offset` (optionally recompute if first tensor offset is non-zero)
* Force full rebuild (not just target library) so test executables relink with updated loader

#### Preventive Invariants Added / Recommended
* Assert (in Debug) that: first tensor offset == 0; each subsequent offset equals previous offset + padded(previous_size)
* Log alignment decision only when `LLAMINAR_MODEL_LOAD_DEBUG` is set
* (Planned) Lightweight regression test verifying alignment + offset chain for a small known model

#### Takeaways
1. Always rebuild dependent test binaries after changing a static library with parsing logic
2. Add a structural comparison (offsets/types) against a known-good upstream parser early—this isolates logic vs data issues
3. Denormal floods across all floats usually mean misaligned base pointer or element-size mismatch, not numeric drift
4. Gate verbose diagnostics behind an env var to allow rapid deep dives without polluting normal test output

Use this pattern for future mysterious “all floats are tiny” failures in loaders or deserializers.

