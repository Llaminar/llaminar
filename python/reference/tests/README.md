# Python Reference Tests

This directory contains unit tests for the Python reference implementation of the Llaminar GGUF loader.

## Test Files

### Core GGUF Tests

- **`test_gguf_dimension_order.py`** ⭐ **CRITICAL**
  - Tests for correct GGUF dimension handling (GGUF stores dims in reverse order)
  - Regression prevention for dimension reversal fix
  - End-to-end inference validation
  - **Run this first** - If these fail, dimension handling is broken

- **`test_gguf_parser.py`**
  - Unit tests for GGUF file format parsing
  - Header, metadata, and tensor info extraction
  - Real file integration tests

- **`test_dequantize.py`**
  - Quantization/dequantization correctness
  - Tests for Q4_0, Q8_0, Q6_K, F16, F32
  - Nibble unpacking validation

- **`test_gguf_inference.py`**
  - Full inference pipeline tests
  - Model loading and forward pass
  - Token prediction accuracy

### Other Tests

- **`test_reference.py`** - General reference implementation tests

## Running Tests

### All Tests
```bash
cd /workspaces/llaminar
python3 -m pytest python/reference/tests/ -v
```

### Specific Test File
```bash
python3 -m pytest python/reference/tests/test_gguf_dimension_order.py -v
```

### Single Test
```bash
python3 -m pytest python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_inference_correctness -v
```

### With Output
```bash
python3 -m pytest python/reference/tests/test_gguf_dimension_order.py -v -s
```

## Test Requirements

### Required Files
- `models/qwen2.5-0.5b-instruct-q4_0.gguf` - Q4_0 quantized model
- `models/qwen2.5-0.5b-instruct-fp16.gguf` - FP16 model (optional)
- `models/Llama-3.2-1B-Instruct-Q4_0.gguf` - LLaMA model (optional)

### Required Packages
```bash
pip install torch transformers numpy pytest
```

## Critical Tests (Must Pass)

### Dimension Order Tests ⚠️
These tests prevent regression of the GGUF dimension reversal fix:

1. ✅ `test_parser_reverses_dimensions` - Parser reverses dims from GGUF
2. ✅ `test_loader_no_additional_transpose` - Loader doesn't double-transpose
3. ✅ `test_inference_correctness` - Model predicts "1+1=2" correctly
4. ✅ `test_parser_has_dimension_reversal` - Code contains dimension reversal
5. ✅ `test_loader_no_transpose_logic` - Old transpose logic removed

**If any of these fail**, the GGUF dimension handling has regressed!

## Expected Results

```
====================================== test session starts =======================================
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_parser_reverses_dimensions PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_weight_matrix_dimensions PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_loader_no_additional_transpose PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_pytorch_model_compatibility PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_fp16_dimension_consistency PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_embedding_values_sanity PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionOrder::test_inference_correctness PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionRegressionPrevention::test_parser_has_dimension_reversal PASSED
python/reference/tests/test_gguf_dimension_order.py::TestGGUFDimensionRegressionPrevention::test_loader_no_transpose_logic PASSED

============================== 9 passed, 1 skipped ===============================================
```

## Debugging Failed Tests

### Test fails with "Shape mismatch"
- Check that `gguf_parser.py` contains `dimensions.reverse()`
- Verify no transpose logic in `gguf_loader.py`
- Ensure you're using the correct GGUF file

### Test fails with "Model predicts wrong token"
- This likely means dimension handling is broken
- Check that inference test predicts "2" for "1+1="
- If it predicts garbage, dimensions are wrong

### Test skipped
- Missing GGUF model file - download required model
- Missing dependencies - install torch/transformers
- Missing reference model - optional test, safe to skip

## Test Philosophy

These tests follow the principle: **"If it's wrong, tests should fail loudly."**

The dimension order fix is critical - without it, ALL inference produces garbage. The regression tests (`test_parser_has_dimension_reversal`, `test_loader_no_transpose_logic`) check the actual source code to prevent someone from accidentally removing the fix.

## Contributing

When adding new GGUF-related features:

1. Add unit tests to appropriate test file
2. Ensure dimension order is preserved correctly
3. Add integration test if it affects inference
4. Update this README

---

**Last Updated**: October 5, 2025  
**Author**: David Sanftenberg
