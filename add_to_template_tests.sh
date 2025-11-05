#!/bin/bash
# Script to add to<T>() template method tests to all tensor test files
# This adds comprehensive tests for to<float>(), to<uint16_t>(), to<int8_t>(), to<int32_t>()

set -e

TENSOR_TEST_FILES=(
    "tests/v2/unit/tensors/Test__FP16Tensor.cpp"
    "tests/v2/unit/tensors/Test__BF16Tensor.cpp"
    "tests/v2/unit/tensors/Test__Q8_0Tensor.cpp"
    "tests/v2/unit/tensors/Test__Q4_0Tensor.cpp"
    "tests/v2/unit/tensors/Test__Q4_1Tensor.cpp"
    "tests/v2/unit/tensors/Test__Q5_0Tensor.cpp"
    "tests/v2/unit/tensors/Test__Q5_1Tensor.cpp"
    "tests/v2/unit/tensors/Test__Q6_KTensor.cpp"
    "tests/v2/unit/tensors/Test__Q2_KTensor.cpp"
    "tests/v2/unit/tensors/Test__Q5_KTensor.cpp"
    "tests/v2/unit/tensors/Test__Q3_KTensor.cpp"
    "tests/v2/unit/tensors/Test__Q4_KTensor.cpp"
    "tests/v2/unit/tensors/Test__Q8_KTensor.cpp"
    "tests/v2/unit/tensors/Test__IQ4_XSTensor.cpp"
    "tests/v2/unit/tensors/Test__IQ2_XXSTensor.cpp"
    "tests/v2/unit/tensors/Test__IQ2_XSTensor.cpp"
    "tests/v2/unit/tensors/Test__IQ3_XXSTensor.cpp"
    "tests/v2/unit/tensors/Test__IQ2_STensor.cpp"
    "tests/v2/unit/tensors/Test__IQ1_MTensor.cpp"
    "tests/v2/unit/tensors/Test__IQ1_STensor.cpp"
    "tests/v2/unit/tensors/Test__IQ3_STensor.cpp"
)

echo "This script will add to<T>() template method tests to all tensor test files."
echo "Files already modified: Test__FP32Tensor.cpp, Test__IQ4_NLTensor.cpp"
echo ""
echo "Remaining files to modify: ${#TENSOR_TEST_FILES[@]}"
echo ""

for file in "${TENSOR_TEST_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "⚠️  File not found: $file (skipping)"
        continue
    fi
    
    # Check if file already has to<T>() tests
    if grep -q "ToFloat_TemplateMethod" "$file"; then
        echo "✓ $file already has to<T>() tests (skipping)"
        continue
    fi
    
    echo "Adding tests to: $file"
    
    # The actual test additions would be done programmatically here
    # For now, this script just identifies the files
done

echo ""
echo "====================================================================================="
echo "NOTE: This script identified the files. The actual test additions should be done"
echo "using the replace_string_in_file tool to find the last test in each file and"
echo "insert the new to<T>() tests before the main() function."
echo "====================================================================================="
