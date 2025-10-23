# Integration Test Framework

**Automated Layer-by-Layer Correctness Verification for Llaminar Models**

This directory contains a comprehensive integration test framework that verifies layer-by-layer correctness between Llaminar and PyTorch reference implementations for every model across both prefill and decode stages.

## 📁 Directory Structure

```
tests/integration/
├── model_integration_test_base.h        # Base test framework class
├── model_integration_test_base.cpp      # Framework implementation
├── qwen_integration_test.cpp            # Qwen-specific test suite
├── llama_integration_test.cpp           # [Future] LLaMA test suite
├── generate_golden_references.py        # Helper script for golden reference generation
└── README.md                            # This file

tests/golden_references/
├── qwen2.5-0.5b-instruct-q4_0/
│   ├── prefill_tokens_1639_266_285_17_10_17_30.npz
│   ├── prefill_tokens_1639_266_285.npz
│   ├── decode_tokens_1639_266_285_17_10_17_30.npz
│   └── ...
├── qwen2.5-0.5b-instruct-q6_k/
│   └── ...
└── ...
```

## 🎯 Purpose

The integration test framework provides:

1. **Automated Verification**: Runs with every `ctest` invocation
2. **Layer-by-Layer Comparison**: Identifies exact divergence point
3. **Stage Coverage**: Tests both prefill and decode independently
4. **Precision Coverage**: Tests Q4_0, Q6_K, and optionally FP32
5. **Regression Detection**: Catches numerical drift from code changes
6. **Model-Specific Suites**: Extensible to new models (Qwen, LLaMA, etc.)

## 🚀 Quick Start

### 1. Generate Golden References

First-time setup requires generating PyTorch reference snapshots:

```bash
# Generate all golden references for Qwen
python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct

# Generate only Q4_0 references (faster)
python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct --precision q4_0

# Dry run (show commands without executing)
python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct --dry-run

# List existing references
python3 tests/integration/generate_golden_references.py --list
```

### 2. Build Tests

```bash
cmake --build build --target test_qwen_integration --parallel
```

### 3. Run Integration Tests

```bash
# Run all Qwen integration tests
ctest --test-dir build -R QwenIntegrationTests --output-on-failure

# Run with verbose output
ctest --test-dir build -R QwenIntegrationTests --output-on-failure --verbose

# Run specific test filter
./build/test_qwen_integration --gtest_filter="*QwenPrefillQ4_0*"
```

## 📊 Test Coverage

### Current: Qwen 2.5 0.5B Instruct

| Test Name | Precision | Stage | Tokens | Description |
|-----------|-----------|-------|--------|-------------|
| `QwenPrefillQ4_0_StandardPrompt` | Q4_0 | Prefill | 7 | Standard "1+1=" prompt |
| `QwenPrefillQ4_0_ShortPrompt` | Q4_0 | Prefill | 3 | Short prompt |
| `QwenPrefillQ4_0_SingleToken` | Q4_0 | Prefill | 1 | Minimal single token |
| `QwenDecodeQ4_0_AfterStandardPrefill` | Q4_0 | Decode | 7 | Decode after prefill |
| `QwenDecodeQ4_0_AfterShortPrefill` | Q4_0 | Decode | 3 | Decode after short prefill |
| `QwenPrefillQ6_K_StandardPrompt` | Q6_K | Prefill | 7 | Higher precision prefill |
| `QwenDecodeQ6_K_AfterStandardPrefill` | Q6_K | Decode | 7 | Higher precision decode |
| `QwenPrefillFP32_StandardPrompt` | FP32 | Prefill | 7 | Gold standard (optional) |
| `QwenDecodeFP32_AfterStandardPrefill` | FP32 | Decode | 7 | Gold standard (optional) |

**Parameterized Tests**: `QwenComprehensiveIntegration` (2 × 2 × 3 = 12 configurations)
- Precisions: Q4_0, Q6_K
- Stages: Prefill, Decode
- Token Sets: Standard (7), Short (3), Single (1)

**Edge Cases**:
- `QwenEdgeCase_EmptyTokens`: Graceful handling of empty input
- `QwenEdgeCase_LongContext`: Stress test with 128 tokens

### Future: LLaMA, Mistral, etc.

The framework is designed to be model-agnostic. Adding a new model requires:

1. Create `<model>_integration_test.cpp`
2. Define token sets and tolerances
3. Add to CMakeLists.txt
4. Generate golden references

## 🔍 How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration Test                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Load PyTorch Golden Reference (NPZ)                      │
│    └─> Extract .npy files → TensorSnapshot → Registry      │
│                                                              │
│ 2. Run Llaminar Pipeline with Capture                       │
│    └─> Execute prefill/decode → Capture snapshots          │
│                                                              │
│ 3. Compare Layer-by-Layer                                   │
│    └─> For each layer:                                      │
│        - Match PyTorch key with Llaminar key                │
│        - Compute max_abs and rel_l2 errors                  │
│        - ASSERT within tolerances                           │
│        - Report FIRST divergence if any                     │
└─────────────────────────────────────────────────────────────┘
```

### Tolerances by Precision

| Precision | max_abs | rel_l2 | Rationale |
|-----------|---------|--------|-----------|
| FP32 | 0.001 | 0.0001 | Tight (gold standard) |
| Q6_K | 0.01 | 0.001 | Moderate (high-quality quant) |
| Q4_0 | 0.1 | 0.01 | Relaxed (aggressive quant) |

### Key Naming Convention

**PyTorch Keys** (from NPZ):
- `pytorch:embeddings` - Embedding layer output
- `pytorch:layer_N_attn_out` - Layer N attention output
- `pytorch:layer_N_mlp_out` - Layer N MLP output
- `pytorch:final_norm_out` - Final normalization output

**Llaminar Keys** (from capture):
- `llaminar:embeddings:global` - Embedding layer
- `llaminar:attn_out:layer_N` - Attention output
- `llaminar:mlp_out:layer_N` - MLP output
- `llaminar:final_norm_out:global` - Final norm

The framework automatically translates between these formats.

## 🛠️ Customization

### Adding a New Model

**Example: Adding LLaMA 3.1 8B**

1. **Create test file**: `tests/integration/llama_integration_test.cpp`

```cpp
#include "model_integration_test_base.h"

const std::vector<int> LLAMA_STANDARD_TOKENS = {128000, 9906, 1917};

TEST_F(ModelIntegrationTestBase, LLaMAPrefillQ4_0_StandardPrompt) {
    TestConfig config;
    config.model_name = "llama3.1-8b";
    config.precision = "q4_0";
    config.stage = "prefill";
    config.tokens = LLAMA_STANDARD_TOKENS;
    config.tolerances = ToleranceConfig::for_precision("q4_0");
    
    ASSERT_TRUE(run_integration_test(config));
}
```

2. **Add to CMakeLists.txt**:

```cmake
add_executable(test_llama_integration
    tests/integration/llama_integration_test.cpp
    tests/integration/model_integration_test_base.cpp
    tests/parity_test_framework.cpp)
target_link_libraries(test_llama_integration llaminar_core GTest::gtest MPI::MPI_CXX)
add_llaminar_mpi_test(LLaMAIntegrationTests 2 test_llama_integration)
```

3. **Update generation script**: Add to `TEST_CONFIGS` in `generate_golden_references.py`

4. **Generate golden references**:

```bash
python3 tests/integration/generate_golden_references.py --model llama3.1-8b
```

### Custom Token Sequences

To test specific prompts:

```cpp
const std::vector<int> MY_CUSTOM_TOKENS = {/* your tokens */};

TEST_F(ModelIntegrationTestBase, CustomPromptTest) {
    TestConfig config;
    config.model_name = "qwen2.5-0.5b-instruct";
    config.precision = "q4_0";
    config.stage = "prefill";
    config.tokens = MY_CUSTOM_TOKENS;
    config.golden_reference = "tests/golden_references/custom_prompt.npz";
    config.tolerances = ToleranceConfig::for_precision("q4_0");
    
    ASSERT_TRUE(run_integration_test(config));
}
```

Generate the golden reference:

```bash
python3 python/reference/capture_pytorch_layers.py \
    -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --tokens 1639,266,285,... \
    --output tests/golden_references/custom_prompt.npz
```

## 📈 Continuous Integration

### CTest Integration

All integration tests are registered with CTest and run automatically:

```bash
# Run all tests (including integration)
ctest --test-dir build --output-on-failure --parallel

# Run only integration tests
ctest --test-dir build -L integration --output-on-failure

# Run specific model
ctest --test-dir build -R QwenIntegration --output-on-failure
```

### Skipping Behavior

Tests automatically skip if:
- Golden reference NPZ file not found
- Model file not found
- Environment variable `SKIP_INTEGRATION_TESTS=1`

This allows CI/CD to run tests only when golden references are available.

### CI/CD Recommendations

```yaml
# .github/workflows/ci.yml example
- name: Generate Golden References (Cache)
  uses: actions/cache@v3
  with:
    path: tests/golden_references/
    key: golden-refs-${{ hashFiles('models/*.gguf') }}

- name: Run Integration Tests
  run: |
    ctest --test-dir build -L integration --output-on-failure
```

## 🐛 Debugging

### Test Failures

When a test fails, you'll see:

```
[ RUN      ] ModelIntegrationTestBase.QwenPrefillQ4_0_StandardPrompt
=== Running Integration Test ===
Model: qwen2.5-0.5b-instruct-q4_0
Stage: prefill
Tokens: 7 tokens
Loading PyTorch golden reference: tests/golden_references/...
Loaded 24 PyTorch snapshots
Running Llaminar pipeline: models/qwen2.5-0.5b-instruct-q4_0.gguf
Captured 24 Llaminar snapshots
Comparing snapshots layer-by-layer...
✓ pytorch:embeddings matches (max_abs=0.002, rel_l2=0.0001)
✓ pytorch:layer_0_attn_out matches (max_abs=0.015, rel_l2=0.0008)
✗ pytorch:layer_1_attn_out diverges!
  max_abs=0.25 (threshold=0.1)
  rel_l2=0.08 (threshold=0.01)

=== DIVERGENCE REPORT ===
Layer: pytorch:layer_1_attn_out
...
[  FAILED  ] ModelIntegrationTestBase.QwenPrefillQ4_0_StandardPrompt
```

**First diverging layer**: `layer_1_attn_out` → Investigate attention kernel in layer 1

### Manual Debugging

```bash
# Run single test with verbose logging
./build/test_qwen_integration --gtest_filter="*StandardPrompt" --gtest_also_run_disabled_tests

# Compare snapshots manually
python3 -c "
import numpy as np
npz = np.load('tests/golden_references/qwen2.5-0.5b-instruct-q4_0/prefill_tokens_1639_266_285_17_10_17_30.npz')
print(npz.files)
print(npz['layer_0_attn_out'].shape)
"
```

### Environment Variables

Control test behavior:

```bash
# Enable detailed capture logging
export LLAMINAR_PARITY_CAPTURE=1
export LLAMINAR_PARITY_VERBOSE=1

# Skip integration tests
export SKIP_INTEGRATION_TESTS=1

# Adjust tolerances (for debugging)
export LLAMINAR_INTEGRATION_MAX_ABS=1.0
export LLAMINAR_INTEGRATION_REL_L2=0.1
```

## 📚 Related Documentation

- **PyTorch Parity Test**: `docs/ENABLING_PYTORCH_PARITY_TEST.md`
- **Agent Instructions**: `tests/AGENTS.md` § 15
- **Parity Framework**: `tests/parity_test_framework.h`
- **Golden Reference Generation**: `python/reference/capture_pytorch_layers.py`

## 🔮 Future Enhancements

Planned improvements:

1. **Automatic Reference Updates**: Regenerate on model changes
2. **Differential Testing**: Compare against previous commits
3. **Performance Regression**: Track execution time per layer
4. **GPU Support**: CUDA/ROCm golden references
5. **Multi-Node**: Distributed execution tests (4+ ranks)
6. **Visualization**: Plot divergence heatmaps
7. **Bisection**: Automatically find diverging commit

## 👥 Contributing

When adding new tests:

1. Follow naming convention: `<Model><Stage><Precision>_<Description>`
2. Use appropriate tolerances for precision
3. Document token sequences
4. Generate golden references before committing test
5. Update this README with new test coverage

## ✅ Checklist

Before committing integration tests:

- [ ] Test compiles successfully
- [ ] Golden references generated
- [ ] Golden references committed (if small) OR documented how to generate
- [ ] Test added to CMakeLists.txt
- [ ] Test runs successfully with `ctest`
- [ ] Documentation updated (this README)
- [ ] CI/CD updated (if needed)

---

**Last Updated**: October 5, 2025  
**Maintainer**: David Sanftenberg  
**Status**: ✅ Production Ready (Qwen 2.5 0.5B)
