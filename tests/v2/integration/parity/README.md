# Parity Test Framework

This directory contains parity tests that compare Llaminar inference against PyTorch ground truth. These tests validate that our implementations produce numerically correct results across different backends (CPU, CUDA, ROCm) and model architectures.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [ParityTestBase Class](#paritytestbase-class)
- [Configuration Options](#configuration-options)
- [Metrics Computed](#metrics-computed)
- [Test Coverage](#test-coverage)
- [Writing a New Parity Test](#writing-a-new-parity-test)
- [Running Tests](#running-tests)
- [Adding Support for New Models](#adding-support-for-new-models)
- [Troubleshooting](#troubleshooting)

---

## Overview

Parity tests ensure that Llaminar's inference results match PyTorch's ground truth within acceptable tolerances. The key insight is that while quantization introduces numerical differences, the **direction** of activations (measured by cosine similarity) and **prediction quality** (measured by KL divergence and Top-K accuracy) should remain stable.

**Test Philosophy:**
- **Layer-by-layer comparison**: Compare intermediate activations at each transformer layer
- **Final output validation**: Validate logit distribution similarity (KL divergence, Top-K overlap)
- **Backend-specific thresholds**: Different quantization schemes need different tolerances
  - CPU (Q8_1): Tighter thresholds (~0.999 cosine) due to per-block quantization
  - CUDA (INT8): Relaxed thresholds (~0.99 cosine) due to per-row symmetric quantization

---

## Directory Structure

```
tests/v2/integration/parity/
├── README.md                          # This file
├── ParityTestBase.h                   # Generic base class (metrics, rendering, orchestration)
├── qwen2/                             # Qwen2 model family tests
│   ├── Qwen2ParityTestBase.h          # Model-specific base class + macros
│   ├── Test__Qwen2_CUDA_vs_PyTorch.cpp
│   └── Test__Qwen2_CPU_vs_PyTorch.cpp
├── llama3/                            # (Future) Llama 3 model tests
│   ├── Llama3ParityTestBase.h
│   └── ...
└── mistral/                           # (Future) Mistral model tests
```

**Naming Convention:**
- Directory: `parity/<model_family>/`
- Base class: `<Model>ParityTestBase.h`
- Test files: `Test__<Model>_<Backend>_vs_PyTorch.cpp`
- Test Fixture: `Test__<Model>_<Backend>_vs_PyTorch`
- CTest Name: `V2_Integration_Parity_<Model>_<Backend>_vs_PyTorch`
- CMake Target: `v2_integration_parity_<model>_<backend>_vs_pytorch`

---

## Declarative Test Architecture

Parity tests use a **three-tier inheritance hierarchy** for maximum code reuse:

```
ParityTestBase (utils/)
    ↓
Qwen2ParityTestBase (qwen2/)     # Model-specific base
    ↓
Test__Qwen2_CPU_vs_PyTorch       # Backend-specific test (pure configuration)
```

### Tier 1: ParityTestBase (Generic)

Located in `tests/v2/integration/parity/ParityTestBase.h`. Provides:
- Metric computation (cosine similarity, KL divergence, Top-K overlap)
- PyTorch snapshot loading and regeneration
- Unicode table rendering
- Generic `runSingleDevicePrefillParity()` and `assertParity()` methods for single-device tests
- TP-aware `runTPPrefillParity()` and `assertTPParity()` methods for multi-device tests

### Tier 2: Model-Specific Base (e.g., Qwen2ParityTestBase)

Located in `parity/<model>/`. Provides:
- Model-specific configuration (model path, snapshot dir, token IDs)
- `BackendThresholds` struct for declarative threshold configuration
- `INSTANTIATE_<MODEL>_PARITY_TESTS` macro to generate test cases

### Tier 3: Backend-Specific Tests (Pure Configuration)

Each backend test file is **purely declarative** (~30-70 lines):

```cpp
class Test__Qwen2_CPU_vs_PyTorch : public Qwen2ParityTestBase {
    BackendThresholds getBackendThresholds() override {
        return {.cosine_threshold=0.999f, .early_layers_count=4, ...};
    }
    DeviceId getDevice() override { return DeviceId::cpu(); }
    std::string getBackendName() override { return "CPU"; }
};
INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_CPU_vs_PyTorch);
```

---

## ParityTestBase Class

All parity tests ultimately inherit from `ParityTestBase` (located in `tests/v2/integration/parity/ParityTestBase.h`). This base class provides:

### Core Methods

| Method | Description |
|--------|-------------|
| `getDevice()` | **Required** - Returns `DeviceId` for inference (e.g., `DeviceId::cpu()`, `DeviceId::cuda(0)`) |
| `getBackendName()` | **Required** - Returns display name (e.g., `"CUDA"`, `"CPU"`, `"ROCm"`) |
| `setupDeviceSpecific()` | Optional - Device availability checks, initialization |
| `runSingleDevicePrefillParity()` | Main test driver for single-device tests - runs inference and compares against PyTorch |
| `runTPPrefillParity()` | TP test driver for multi-device tests - compares per-device outputs against PyTorch |
| `assertParity()` | Standard assertion helper with threshold checks |

### Model-Specific Base Classes

Each model family has its own base class (e.g., `Qwen2ParityTestBase`) that provides:

| Method | Description |
|--------|-------------|
| `getBackendThresholds()` | **Required** - Returns `BackendThresholds` struct |

```cpp
struct BackendThresholds {
    float cosine_threshold = 0.99f;    // Minimum cosine similarity
    int early_layers_count = 6;        // How many early layers to check
    int min_early_layers_passed = 6;   // How many must pass
    float kl_threshold = 0.15f;        // Maximum KL divergence
};
```

### Automatic Snapshot Management

The base class automatically:
1. **Regenerates PyTorch snapshots** on each test run via `python/reference/generate_qwen2_pipeline_snapshots.py`
2. **Loads snapshots** from `.npy` files using the cnpy library
3. **Caches snapshots** in memory for efficient multi-snapshot comparisons

### Result Structures

```cpp
// Per-layer statistics
struct LayerStats {
    int layer_idx;
    float avg_cosine_sim;     // Average across all stage comparisons
    float min_cosine_sim;     // Minimum (worst) stage similarity
    std::string worst_stage;  // Name of the worst-performing stage
    int stages_compared;      // Number of stages compared
    bool passed;              // Met threshold criteria
};

// Overall test summary
struct ParityTestSummary {
    float embedding_cosine;
    std::vector<LayerStats> layer_stats;  // Per-layer results
    float lm_head_cosine, lm_head_kl;     // LM head metrics
    float lm_head_top1, lm_head_top5;     // Top-K accuracy
    int early_layers_passed;              // Layers meeting threshold
    bool overall_passed;
};
```

---

## Configuration Options

Configure thresholds in your test's `SetUp()` method via `config_`:

```cpp
struct ParityConfig {
    // Model and test setup
    std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    std::string snapshot_dir = "pytorch_qwen2_snapshots";
    std::string prompt = "The quick brown fox jumps over the lazy dog";
    std::vector<int> token_ids = {785, 3974, ...};  // Pre-tokenized prompt
    int decode_steps = 5;

    // Layer-by-layer thresholds
    float cosine_threshold = 0.99f;    // Minimum cosine similarity
    bool use_avg_cosine = true;        // Use average (true) or min (false)
    int early_layers_count = 6;        // How many early layers to check strictly
    int min_early_layers_passed = 6;   // How many must pass

    // LM_HEAD thresholds
    float kl_threshold = 0.15f;        // Maximum KL divergence (nats)
    float min_top1_accuracy = 60.0f;   // Minimum Top-1 accuracy %
};
```

**Recommended Thresholds by Backend:**

| Backend | `cosine_threshold` | `early_layers_count` | `kl_threshold` | Rationale |
|---------|-------------------|---------------------|----------------|-----------|
| CPU (Q8_1) | 0.999 | 4 | 0.15 | Per-block quantization preserves accuracy |
| CUDA (INT8) | 0.99 | 6 | 0.15 | Per-row symmetric has more error |
| ROCm | 0.99 | 6 | 0.15 | Similar to CUDA |
| FP32 | 0.9999 | 8 | 0.05 | Near-perfect match expected |

---

## Metrics Computed

### Cosine Similarity

```cpp
float computeCosineSimilarity(const float* a, const float* b, size_t size);
```

Measures directional alignment, ignoring magnitude. Range: `[-1, 1]`, where 1 = identical direction.

**Why cosine?** Quantization affects magnitude but preserves direction. High cosine similarity indicates the model "understands" the same features even if numerical values differ.

### KL Divergence

```cpp
float computeKLDivergence(const float* actual_logits, const float* expected_logits, 
                          size_t size, size_t vocab_size);
```

Measures how the actual probability distribution diverges from expected. Lower is better.

**Interpretation:**
- `< 0.05`: Excellent match
- `< 0.15`: Good match (acceptable for quantized models)
- `> 0.30`: Poor match, investigate

### Top-K Overlap

```cpp
float computeTopKOverlap(const float* actual, const float* expected, 
                         size_t size, size_t vocab_size, int k);
```

Checks if the Top-K predicted tokens overlap. Range: `[0, 1]`, where 1 = 100% overlap.

**Why Top-K?** This is the "smoke test" - even if logits differ numerically, the model should predict similar tokens.

---

## Test Coverage

Parity tests cover both **prefill** and **incremental decode** phases:

### Prefill Parity (`PrefillParity_LayerByLayer`)

Compares Llaminar's prefill output against PyTorch layer-by-layer:
- Processes the full input prompt in a single batch
- Compares activations at each transformer layer
- Validates embedding, attention, FFN, and LM_HEAD stages
- Pass criteria: average cosine similarity >= threshold per layer

### Decode Parity (`DecodeParity_Incremental`)

Compares incremental token generation (autoregressive decode):
- First runs prefill to initialize KV cache
- Generates tokens one at a time using the sampled token from PyTorch
- Compares LM_HEAD logit distribution at each decode step
- Pass criteria:
  - Each step: cosine >= threshold OR KL < threshold
  - Overall: min_decode_pass_rate (default 80%)
  - Top-1 accuracy >= min_top1_accuracy (default 60%)

**Example Decode Parity Output:**

```
╔════════════════════════════════════════════════════════════════════════════════════╗
║                    CPU INCREMENTAL DECODE PARITY                                   ║
║                    (Threshold: cosine >= 0.990 OR KL < 0.150)                      ║
╠═════════╦═══════════════╦═══════════════╦═══════════════╦═══════════════╦══════════╣
║  Step   ║    Cosine     ║      KL       ║   Llaminar    ║    PyTorch    ║  Status  ║
╠═════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════╬══════════╣
║      0  ║     0.998922  ║     0.001664  ║       323 ✓   ║       323     ║    ✓     ║
║      1  ║     0.999334  ║     0.005113  ║      1221 ✗   ║       279     ║    ✓     ║
║      2  ║     0.999353  ║     0.008595  ║      5562 ✗   ║      3974     ║    ✓     ║
║      3  ║     0.999493  ║     0.000528  ║     13876 ✓   ║     13876     ║    ✓     ║
║      4  ║     0.999577  ║     0.000179  ║     38835 ✓   ║     38835     ║    ✓     ║
╠═════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════╩══════════╣
║  SUMMARY:  Steps=5/5  AvgCosine=0.9993  Top1=60.0%  ✓ PASSED                       ║
╚════════════════════════════════════════════════════════════════════════════════════╝
```

**Note:** Token mismatches (✗ in Llaminar column) are expected when quantization shifts probability mass slightly. The key metric is cosine similarity of the logit distributions.

### Snapshot Infrastructure (`SnapshotInfrastructure`)

Verifies the PyTorch snapshot generation and loading works correctly:
- Checks that snapshots can be generated
- Validates embedding snapshot loads successfully

---

## Writing a New Parity Test

### Adding a Backend to an Existing Model (e.g., ROCm for Qwen2)

Create `tests/v2/integration/parity/qwen2/Test__Qwen2_ROCm_vs_PyTorch.cpp`:

```cpp
/**
 * @file Test__Qwen2_ROCm_vs_PyTorch.cpp
 * @brief Integration: Qwen2 ROCm Pipeline vs PyTorch Reference
 */

#include <gtest/gtest.h>
#include "Qwen2ParityTestBase.h"
#include "backends/ComputeBackend.h"

#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test::parity::qwen2;

class Test__Qwen2_ROCm_vs_PyTorch : public Qwen2ParityTestBase
{
protected:
    DeviceId rocm_device_ = DeviceId::cpu();

    BackendThresholds getBackendThresholds() override
    {
        return {
            .cosine_threshold = 0.99f,
            .early_layers_count = 6,
            .min_early_layers_passed = 6,
            .kl_threshold = 0.15f
        };
    }

    void setupDeviceSpecific() override
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "Built without ROCm support";
#else
        auto &dm = DeviceManager::instance();
        dm.initialize(-1);

        int gpu_idx = dm.find_device(ComputeBackendType::GPU_ROCM);
        if (gpu_idx < 0) {
            GTEST_SKIP() << "No ROCm device found";
        }

        rocm_device_ = DeviceId::rocm(gpu_idx - 1);
#endif
    }

    DeviceId getDevice() override { return rocm_device_; }
    std::string getBackendName() override { return "ROCm"; }
};

INSTANTIATE_QWEN2_PARITY_TESTS(Test__Qwen2_ROCm_vs_PyTorch);
```

That's it! The `INSTANTIATE_QWEN2_PARITY_TESTS` macro generates:
- `PrefillParity_LayerByLayer` test
- `SnapshotInfrastructure` test

### Adding a New Model Family (e.g., Llama3)

#### Step 1: Create Model Directory and Base Class

Create `tests/v2/integration/parity/llama3/Llama3ParityTestBase.h`:

```cpp
#pragma once
#include "../ParityTestBase.h"

namespace llaminar2::test::parity::llama3
{

struct BackendThresholds {
    float cosine_threshold = 0.99f;
    int early_layers_count = 8;      // Llama3 has 32 layers
    int min_early_layers_passed = 7;
    float kl_threshold = 0.15f;
};

class Llama3ParityTestBase : public ParityTestBase
{
protected:
    virtual BackendThresholds getBackendThresholds() = 0;

    void SetUp() override
    {
        auto thresholds = getBackendThresholds();
        config_.cosine_threshold = thresholds.cosine_threshold;
        config_.use_avg_cosine = true;
        config_.early_layers_count = thresholds.early_layers_count;
        config_.min_early_layers_passed = thresholds.min_early_layers_passed;
        config_.kl_threshold = thresholds.kl_threshold;

        // Llama3-specific model configuration
        config_.model_path = "models/llama-3-8b-instruct-q4_0.gguf";
        config_.snapshot_dir = "pytorch_llama3_snapshots";
        config_.prompt = "Hello, my name is";
        config_.token_ids = {/* tokenized prompt */};

        ParityTestBase::SetUp();
    }
};

} // namespace

#define INSTANTIATE_LLAMA3_PARITY_TESTS(TestFixture)                                   \
    TEST_F(TestFixture, PrefillParity_LayerByLayer) {                                  \
        auto summary = runSingleDevicePrefillParity();                                 \
        assertParity(summary);                                                         \
    }                                                                                  \
    TEST_F(TestFixture, SnapshotInfrastructure) {                                      \
        ASSERT_TRUE(setupPipeline()) << "Pipeline setup failed";                       \
        auto embedding = loadPyTorchSnapshot("EMBEDDING");                             \
        ASSERT_FALSE(embedding.empty()) << "Failed to load EMBEDDING snapshot";        \
    }
```

#### Step 2: Create Backend Test (CPU example)

Create `tests/v2/integration/parity/llama3/Test__Llama3_CPU_vs_PyTorch.cpp`:

```cpp
#include <gtest/gtest.h>
#include "Llama3ParityTestBase.h"

using namespace llaminar2;
using namespace llaminar2::test::parity::llama3;

class Test__Llama3_CPU_vs_PyTorch : public Llama3ParityTestBase
{
protected:
    BackendThresholds getBackendThresholds() override {
        return {.cosine_threshold=0.999f, .early_layers_count=8, .min_early_layers_passed=7};
    }
    DeviceId getDevice() override { return DeviceId::cpu(); }
    std::string getBackendName() override { return "CPU"; }
};

INSTANTIATE_LLAMA3_PARITY_TESTS(Test__Llama3_CPU_vs_PyTorch);
```

#### Step 3: Add CMake Target

In `tests/v2/CMakeLists.txt`, add under the appropriate model section:

```cmake
# Llama3 CUDA vs PyTorch Parity Test
add_executable(v2_integration_parity_llama3_cuda_vs_pytorch 
    integration/parity/llama3/Test__Llama3_CUDA_vs_PyTorch.cpp
    ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
)
target_link_libraries(v2_integration_parity_llama3_cuda_vs_pytorch 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
    ZLIB::ZLIB
)
target_include_directories(v2_integration_parity_llama3_cuda_vs_pytorch PRIVATE 
    ${WORKSPACE_ROOT_PARITY}/external/cnpy
)
add_v2_test(V2_Integration_Parity_Llama3_CUDA_vs_PyTorch
    COMMAND $<TARGET_FILE:v2_integration_parity_llama3_cuda_vs_pytorch>
    LABELS "V2;Integration;Parity;Llama3;CUDA;PyTorch;GoldenReference;FullModel;Pipeline"
    MPI_PROCS 1
)
```

### Step 3: Create PyTorch Snapshot Generator (If New Model)

For new model architectures, create a snapshot generator in `python/reference/`:

```python
# python/reference/generate_llama3_pipeline_snapshots.py
# (Similar to generate_qwen2_pipeline_snapshots.py)
```

Update `ParityTestBase::regeneratePyTorchSnapshots()` if the script name differs.

---

## Running Tests

```bash
# Build integration tests
cmake --build build_v2_integration --parallel

# Run all Qwen2 parity tests
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2" --output-on-failure

# Run all CUDA parity tests (across all models)
ctest --test-dir build_v2_integration -L "CUDA" -L "Parity" --output-on-failure

# Run a specific test
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_CUDA_vs_PyTorch" -V

# Run all parity tests
ctest --test-dir build_v2_integration -L "Parity" --output-on-failure
```

### Test Output

Tests produce a formatted Unicode table showing layer-by-layer results:

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                    CUDA vs PyTorch LAYER-BY-LAYER PARITY                                 ║
║                    (Threshold: avg cosine similarity >= 0.990)                           ║
╠═══════════╦═══════════════╦═══════════════╦════════════════════════════════════════╦══════╣
║   Layer   ║   Avg Cosine  ║   Min Cosine  ║            Worst Stage                 ║Status║
╠═══════════╬═══════════════╬═══════════════╬════════════════════════════════════════╬══════╣
║ EMBEDDING ║      0.999912 ║      0.999912 ║                      -                 ║  ✓  ║
║   Layer 0 ║      0.998234 ║      0.995123 ║              FFN_RESIDUAL              ║  ✓  ║
║   Layer 1 ║      0.997856 ║      0.993456 ║           ATTENTION_OUTPUT             ║  ✓  ║
...
╠═══════════╬═══════════════╬═══════════════╬════════════════════════════════════════╬══════╣
║  LM_HEAD  ║      0.982345 ║      0.982345 ║    KL=  0.0234 Top1=100.0%             ║  ✓  ║
╚═══════════╩═══════════════╩═══════════════╩════════════════════════════════════════╩══════╝
```

---

## Adding Support for New Models

### 1. Create Model Directory

```bash
mkdir -p tests/v2/integration/parity/<model_name>
```

### 2. Create PyTorch Reference Script

Copy and adapt `python/reference/generate_qwen2_pipeline_snapshots.py`:
- Update model loading logic
- Adjust layer naming to match model architecture
- Update snapshot key names if layer structure differs

### 3. Verify Stage Names

Check what stages Llaminar captures for the model:

```cpp
// In a test, print available snapshot keys
auto keys = runner_->getSnapshotKeys();
for (const auto& key : keys) {
    std::cout << key << std::endl;
}
```

Ensure the PyTorch generator uses matching key names.

### 4. Adjust Thresholds

Different architectures may need different thresholds:
- More layers = more error accumulation → may need relaxed thresholds
- Different attention mechanisms → check attention-specific stages
- MoE models → additional stages to compare

---

## Troubleshooting

### Test Fails with "PyTorch snapshot generation failed"

1. Check Python environment: `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Verify model exists: `ls models/`
4. Run generator manually:
   ```bash
   python python/reference/generate_qwen2_pipeline_snapshots.py \
       --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
       --prompt "Test" --output pytorch_test_snapshots
   ```

### Layer Cosine Similarity Too Low

1. Check the "Worst Stage" column - identifies where divergence occurs
2. Run with `LLAMINAR_LOG_LEVEL=DEBUG` for detailed execution trace
3. Compare specific tensors manually using stage dumps:
   ```bash
   LLAMINAR_STAGE_DUMP_ENABLED=1 LLAMINAR_STAGE_DUMP_LAYERS=0 ./test_binary
   ```

### KL Divergence High but Top-1 Accuracy Good

This is often acceptable - it means the model predicts the same tokens but with different confidence levels. Consider:
- Relaxing `kl_threshold` slightly
- Focusing on Top-K accuracy as the primary quality metric

### CUDA Test Skipped

1. Verify CUDA is available: `nvidia-smi`
2. Check build has CUDA: `cmake -B build_v2_integration -S src/v2 -DHAVE_CUDA=ON`
3. Ensure GPU is detected: Run with `LLAMINAR_LOG_LEVEL=INFO`

---

## Dependencies

Parity tests require:
- **ZLIB** - For cnpy .npy file loading
- **cnpy** - NumPy file format library (in `external/cnpy/`)
- **Python** - For PyTorch snapshot generation
- **PyTorch** - Ground truth implementation (in Python environment)

If ZLIB is not found, parity tests are disabled at CMake configure time.
