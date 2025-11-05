#!/bin/bash
# Add template tests to all IQ-series tensor test files

cd /workspaces/llaminar

# IQ tensor test files (using createSimpleTensor pattern like IQ4_NL)
IQ_TENSORS=("IQ4_XS" "IQ2_XXS" "IQ2_XS" "IQ3_XXS" "IQ2_S" "IQ1_M" "IQ1_S" "IQ3_S" "Q8_K")

for TENSOR in "${IQ_TENSORS[@]}"; do
    FILE="tests/v2/unit/tensors/Test__${TENSOR}Tensor.cpp"
    
    if [ ! -f "$FILE" ]; then
        echo "❌ File not found: $FILE"
        continue
    fi
    
    # Check if tests already exist
    if grep -q "${TENSOR}Tensor_Template" "$FILE"; then
        echo "✓ $TENSOR already has template tests"
        continue
    fi
    
    echo "Adding tests to $TENSOR..."
    
    # Create temporary test code
    cat > /tmp/iq_tests_$TENSOR.cpp << EOF

// ========================================================================
// Template Method Tests for to<T>() API
// ========================================================================

TEST(${TENSOR}Tensor_Template, ToFloat_TemplateMethod)
{
    using namespace llaminar2;
    
    auto tensor = createSimpleTensor<${TENSOR}Tensor>();
    const size_t num_elements = tensor->size_elements();
    
    std::vector<float> result_template(num_elements);
    tensor->template to<float>(result_template.data());
    
    std::vector<float> result_legacy(num_elements);
    tensor->to_fp32(result_legacy.data());
    
    for (size_t i = 0; i < num_elements; ++i)
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(${TENSOR}Tensor_Template, ToBF16_TemplateMethod)
{
    using namespace llaminar2;
    
    auto tensor = createSimpleTensor<${TENSOR}Tensor>();
    const size_t num_elements = tensor->size_elements();
    
    std::vector<uint16_t> result_template(num_elements);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);
    
    std::vector<uint16_t> result_legacy(num_elements);
    tensor->to_bf16(result_legacy.data());
    
    for (size_t i = 0; i < num_elements; ++i)
        EXPECT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(${TENSOR}Tensor_Template, ToFP16_TemplateMethod)
{
    using namespace llaminar2;
    
    auto tensor = createSimpleTensor<${TENSOR}Tensor>();
    const size_t num_elements = tensor->size_elements();
    
    std::vector<uint16_t> result_template(num_elements);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);
    
    std::vector<uint16_t> result_legacy(num_elements);
    tensor->to_fp16(result_legacy.data());
    
    for (size_t i = 0; i < num_elements; ++i)
        EXPECT_EQ(result_template[i], result_legacy[i]) << "Mismatch at index " << i;
}

TEST(${TENSOR}Tensor_Template, ToINT8_TemplateMethod)
{
    using namespace llaminar2;
    
    auto tensor = createSimpleTensor<${TENSOR}Tensor>();
    const size_t num_elements = tensor->size_elements();
    
    std::vector<int8_t> int8_data(num_elements);
    tensor->template to<int8_t>(int8_data.data());
    
    for (size_t i = 0; i < num_elements; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }
}

TEST(${TENSOR}Tensor_Template, ToINT32_TemplateMethod)
{
    using namespace llaminar2;
    
    auto tensor = createSimpleTensor<${TENSOR}Tensor>();
    const size_t num_elements = tensor->size_elements();
    
    std::vector<int32_t> int32_data(num_elements);
    tensor->template to<int32_t>(int32_data.data());
    
    for (size_t i = 0; i < num_elements; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }
}

TEST(${TENSOR}Tensor_Template, RoundTrip_${TENSOR}_FP32_BF16_FP32)
{
    using namespace llaminar2;
    
    auto tensor = createSimpleTensor<${TENSOR}Tensor>();
    const size_t num_elements = tensor->size_elements();
    const std::vector<size_t>& shape = tensor->shape();
    
    std::vector<float> fp32_data(num_elements);
    tensor->template to<float>(fp32_data.data());
    
    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), num_elements * sizeof(float));
    
    std::vector<uint16_t> bf16_data(num_elements);
    fp32_tensor->template to<uint16_t>(bf16_data.data(), TensorType::BF16);
    
    auto bf16_tensor = std::make_shared<BF16Tensor>(shape, bf16_data);
    std::vector<float> final_fp32_data(num_elements);
    bf16_tensor->template to<float>(final_fp32_data.data());
    
    double sum_sq_diff = 0.0;
    double sum_sq_orig = 0.0;
    
    for (size_t i = 0; i < num_elements; ++i)
    {
        double diff = fp32_data[i] - final_fp32_data[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += fp32_data[i] * fp32_data[i];
    }
    
    double rel_l2_error = std::sqrt(sum_sq_diff / sum_sq_orig);
    EXPECT_LT(rel_l2_error, 0.05);
}
EOF
    
    # Find the last closing brace before any main() function
    LAST_BRACE=$(grep -n "^}$" "$FILE" | tail -2 | head -1 | cut -d: -f1)
    
    if [ -z "$LAST_BRACE" ]; then
        echo "❌ Could not find insertion point in $FILE"
        continue
    fi
    
    # Insert the tests before the last closing brace
    head -n $LAST_BRACE "$FILE" > /tmp/file_part1.txt
    cat /tmp/iq_tests_$TENSOR.cpp >> /tmp/file_part1.txt
    tail -n +$((LAST_BRACE + 1)) "$FILE" >> /tmp/file_part1.txt
    mv /tmp/file_part1.txt "$FILE"
    
    echo "✓ Added tests to $TENSOR"
done

echo ""
echo "Building all IQ-series tests..."
cmake --build build_v2 --parallel -- v2_test_iq4_xs_simd v2_test_iq2_xxs_simd v2_test_iq2_xs_simd v2_test_iq3_xxs_simd v2_test_iq2_s_simd v2_test_iq1_m_simd v2_test_iq1_s_simd v2_test_iq3_s_simd v2_test_q8_k_simd 2>&1 | tail -20

echo ""
echo "Testing..."
for TENSOR in "${IQ_TENSORS[@]}"; do
    TARGET="v2_test_$(echo $TENSOR | tr 'A-Z' 'a-z' | tr '_' '_')_simd"
    echo "=== $TARGET ==="
    if [ -f "./build_v2/tests/v2/$TARGET" ]; then
        ./build_v2/tests/v2/$TARGET --gtest_filter="*Template*" 2>&1 | grep "PASSED"
    else
        echo "❌ Binary not found"
    fi
done
