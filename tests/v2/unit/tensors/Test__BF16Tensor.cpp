/**
 * @file Test__BF16Tensor.cpp
 * @brief Unit tests for BF16Tensor class
 * @author David Sanftenberg
 *
 * Tests BF16 tensor operations including:
 * - Basic tensor creation and properties
 * - BF16 ↔ FP32 conversion accuracy
 * - GEMM correctness with BF16 weights
 * - Backend selection (MKL vs OpenBLAS)
 *
 * Naming convention: Test file and test suite are named after the class under test.
 * File: Test__BF16Tensor.cpp → Testing: BF16Tensor class
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include <memory>
#include <cmath>
#include <cstring>

using namespace llaminar2;

/**
 * @brief Helper to convert FP32 to BF16 (truncate mantissa)
 */
inline uint16_t fp32_to_bf16(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    // BF16: Keep sign (1 bit) + exponent (8 bits) + top 7 mantissa bits
    // Truncate bottom 16 bits of mantissa
    return static_cast<uint16_t>(bits >> 16);
}

/**
 * @brief Helper to convert BF16 to FP32 (zero-extend mantissa)
 */
inline float bf16_to_fp32(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float val;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
}

/**
 * @brief Test BF16 tensor creation and basic properties
 */
TEST(Test__BF16Tensor, BasicCreation)
{
    std::vector<size_t> shape = {3, 5}; // 3x5 matrix
    std::vector<uint16_t> data(15);

    // Initialize with BF16 representation of 1.0
    for (size_t i = 0; i < 15; ++i)
    {
        data[i] = fp32_to_bf16(1.0f);
    }

    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 3);
    EXPECT_EQ(tensor->shape()[1], 5);
    EXPECT_EQ(tensor->native_type(), TensorType::BF16);
    EXPECT_TRUE(tensor->home_device().is_cpu()); // Default to CPU
    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test BF16 ↔ FP32 conversion accuracy
 *
 * BF16 has ~3 decimal digits of precision (7 mantissa bits vs FP32's 23).
 * Conversion should preserve sign, exponent, and top 7 mantissa bits.
 */
TEST(Test__BF16Tensor, ConversionAccuracy)
{
    // Test values covering different ranges
    std::vector<float> test_values = {
        0.0f,     // Zero
        1.0f,     // Exact representation
        -1.0f,    // Negative
        3.14159f, // π (loses precision)
        0.125f,   // Power of 2 (exact)
        1234.5f,  // Larger value
        0.001f,   // Small value
        -99.99f   // Negative larger value
    };

    for (float original : test_values)
    {
        uint16_t bf16 = fp32_to_bf16(original);
        float converted = bf16_to_fp32(bf16);

        if (original == 0.0f)
        {
            EXPECT_EQ(converted, 0.0f);
        }
        else
        {
            // BF16 has ~0.8% relative precision (2^-7 ≈ 0.0078)
            float rel_error = std::abs((converted - original) / original);
            EXPECT_LT(rel_error, 0.01f) << "Value: " << original
                                        << " converted to: " << converted;
        }
    }
}

// Weight-based GEMM tests removed: weights no longer invoke GEMM on themselves.

/**
 * @brief Ensure BF16Tensor::from_int32_with_scales applies scaling and bias correctly.
 */
TEST(Test__BF16Tensor, FromInt32WithScalesProducesExpectedValues)
{
    const int rows = 2;
    const int cols = 2;
    auto tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    const std::vector<int32_t> accum = {
        64, -32,
        16, -8};
    const std::vector<float> row_scales = {0.5f, 0.25f};
    const std::vector<float> col_scales = {1.0f, 2.0f};
    const std::vector<float> bias = {0.0f, 1.0f};

    ASSERT_TRUE(tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        row_scales.data(),
        col_scales.data(),
        bias.data()));

    const float *fp32_view = tensor->data();
    ASSERT_NE(fp32_view, nullptr);

    const std::vector<float> expected = {
        32.0f,  // 64 * 0.5 * 1.0 + 0.0
        -31.0f, // -32 * 0.5 * 2.0 + 1.0
        4.0f,   // 16 * 0.25 * 1.0 + 0.0
        -3.0f   // -8 * 0.25 * 2.0 + 1.0
    };

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(fp32_view[i], expected[i]) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test BF16 precision loss is acceptable
 *
 * Verify that BF16 quantization error is within expected bounds (~1%).
 */
TEST(Test__BF16Tensor, PrecisionLoss)
{
    std::vector<float> original_values;
    std::vector<uint16_t> bf16_values;

    // Generate test values from -10 to 10 with 0.1 steps
    for (float val = -10.0f; val <= 10.0f; val += 0.1f)
    {
        original_values.push_back(val);
        bf16_values.push_back(fp32_to_bf16(val));
    }

    auto tensor = std::make_shared<BF16Tensor>(
        std::vector<size_t>{original_values.size()},
        bf16_values);

    // BF16Tensor stores converted FP32 values in data()
    const float *converted = tensor->data();

    for (size_t i = 0; i < original_values.size(); ++i)
    {
        float original = original_values[i];
        float after_conversion = converted[i];

        if (std::abs(original) < 1e-6f)
        {
            // Near-zero values
            EXPECT_NEAR(after_conversion, 0.0f, 1e-3f);
        }
        else
        {
            // Relative error should be < 1% for most values
            float rel_error = std::abs((after_conversion - original) / original);
            EXPECT_LT(rel_error, 0.015f) << "Value: " << original
                                         << " -> " << after_conversion;
        }
    }
}

// Further weight-based GEMM and createGemm tests removed: weight tensors no longer own GEMM paths.

// ========== View Tests ==========

/**
 * @brief Test basic view creation
 */
TEST(Test__BF16Tensor, BasicViewCreation)
{
    // Create parent tensor [10, 20] = 200 elements
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    // Fill with test data (0, 1, 2, ..., 199)
    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create a view of first 5 rows
    auto view = parent->create_view({5, 20}, 0);

    ASSERT_NE(view, nullptr) << "View creation failed";
    EXPECT_EQ(view->shape().size(), 2);
    EXPECT_EQ(view->shape()[0], 5);
    EXPECT_EQ(view->shape()[1], 20);
    EXPECT_TRUE(view->is_view());

    // Verify data pointer is valid
    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr) << "View data pointer is null";

    // Verify first element (BF16 tolerance)
    EXPECT_NEAR(view_data[0], 0.0f, 0.1f);
}

/**
 * @brief Test view creation with offset
 */
TEST(Test__BF16Tensor, ViewWithOffset)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    // Fill with test data
    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create a view starting at element 100 (row 5)
    auto view = parent->create_view({3, 20}, 100);

    ASSERT_NE(view, nullptr);
    EXPECT_TRUE(view->is_view());

    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr);

    // First element should be ~100 (BF16 tolerance)
    EXPECT_NEAR(view_data[0], 100.0f, 1.0f) << "First element should be ~100";
}

/**
 * @brief Test view bounds checking
 */
TEST(Test__BF16Tensor, ViewBoundsChecking)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    // Try to create a view that exceeds parent bounds
    auto view = parent->create_view({20, 20}, 0); // 400 elements > 200 available

    EXPECT_EQ(view, nullptr) << "View creation should fail for out-of-bounds request";
}

/**
 * @brief Test view with offset that exceeds bounds
 */
TEST(Test__BF16Tensor, ViewOffsetBoundsChecking)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    // Try to create a view with offset that exceeds bounds
    auto view = parent->create_view({5, 20}, 150); // offset 150 + 100 elements > 200

    EXPECT_EQ(view, nullptr) << "View creation should fail when offset + size exceeds bounds";
}

/**
 * @brief Test view lifetime (parent stays alive via shared_ptr)
 */
TEST(Test__BF16Tensor, ViewLifetime)
{
    std::shared_ptr<TensorBase> view;
    const float *view_data_ptr = nullptr;

    {
        auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});
        std::vector<float> test_data(200);
        for (size_t i = 0; i < 200; ++i)
        {
            test_data[i] = static_cast<float>(i * 2);
        }
        parent->from_fp32(test_data.data(), 200);

        // Create view in inner scope
        view = parent->create_view({5, 20}, 0);
        ASSERT_NE(view, nullptr);
        view_data_ptr = view->data();
        ASSERT_NE(view_data_ptr, nullptr);

        // parent goes out of scope here
    }

    // View still exists, should keep parent alive
    EXPECT_NE(view->data(), nullptr) << "View data should still be valid";
    EXPECT_TRUE(view->is_view());

    // Data should still be accessible (BF16 tolerance)
    const float *current_data = view->data();
    EXPECT_NEAR(current_data[0], 0.0f, 0.1f) << "View data should still be accessible";
}

/**
 * @brief Test view chaining (view of a view)
 */
TEST(Test__BF16Tensor, ViewChaining)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create first view (rows 2-6)
    auto view1 = parent->create_view({5, 20}, 40); // offset = 2 rows * 20 cols
    ASSERT_NE(view1, nullptr);
    EXPECT_TRUE(view1->is_view());

    // Cast to BF16Tensor to create view of view
    auto bf16_view1 = std::dynamic_pointer_cast<BF16Tensor>(view1);
    ASSERT_NE(bf16_view1, nullptr);

    // Create view of view (first 2 rows of view1)
    auto view2 = bf16_view1->create_view({2, 20}, 0);
    ASSERT_NE(view2, nullptr);
    EXPECT_TRUE(view2->is_view());

    // Verify view2 points to correct data in original parent
    const float *view2_data = view2->data();
    ASSERT_NE(view2_data, nullptr);

    // First element should be ~40 (offset 40 in parent, BF16 tolerance)
    EXPECT_NEAR(view2_data[0], 40.0f, 1.0f) << "Chained view should point to element 40";
}

/**
 * @brief Test view data modification affects parent
 */
TEST(Test__BF16Tensor, ViewModification)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200, 1.0f);
    parent->from_fp32(test_data.data(), 200);

    // Create view
    auto view = parent->create_view({5, 20}, 0);
    ASSERT_NE(view, nullptr);

    auto bf16_view = std::dynamic_pointer_cast<BF16Tensor>(view);
    ASSERT_NE(bf16_view, nullptr);

    // Modify through view
    std::vector<float> new_data(100, 42.0f);
    bf16_view->from_fp32(new_data.data(), 100);

    // Verify parent data changed (BF16 tolerance)
    const float *parent_data = parent->data();
    EXPECT_NEAR(parent_data[0], 42.0f, 0.5f) << "Parent should reflect view modification";
    EXPECT_NEAR(parent_data[99], 42.0f, 0.5f);
}

/**
 * @brief Test view with different shape (reshape)
 */
TEST(Test__BF16Tensor, ViewReshape)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create view with different shape but same total elements
    auto view = parent->create_view({20, 10}, 0);

    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], 20);
    EXPECT_EQ(view->shape()[1], 10);

    const float *view_data = view->data();
    ASSERT_NE(view_data, nullptr);

    // Data should still be accessible in new shape (BF16 tolerance)
    EXPECT_NEAR(view_data[0], 0.0f, 0.1f);
    EXPECT_NEAR(view_data[10], 10.0f, 0.5f);
}

/**
 * @brief Test view of subset with reshape
 */
TEST(Test__BF16Tensor, ViewSubsetReshape)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Take 120 elements starting at offset 40, reshape to [6, 20]
    auto view = parent->create_view({6, 20}, 40);

    ASSERT_NE(view, nullptr);
    EXPECT_EQ(view->shape()[0], 6);
    EXPECT_EQ(view->shape()[1], 20);

    const float *view_data = view->data();
    EXPECT_NEAR(view_data[0], 40.0f, 1.0f);
}

/**
 * @brief Test multiple views of same parent
 */
TEST(Test__BF16Tensor, MultipleViews)
{
    auto parent = std::make_shared<BF16Tensor>(std::vector<size_t>{10, 20});

    std::vector<float> test_data(200);
    for (size_t i = 0; i < 200; ++i)
    {
        test_data[i] = static_cast<float>(i);
    }
    parent->from_fp32(test_data.data(), 200);

    // Create multiple non-overlapping views
    auto view1 = parent->create_view({3, 20}, 0);   // rows 0-2
    auto view2 = parent->create_view({3, 20}, 60);  // rows 3-5
    auto view3 = parent->create_view({4, 20}, 120); // rows 6-9

    ASSERT_NE(view1, nullptr);
    ASSERT_NE(view2, nullptr);
    ASSERT_NE(view3, nullptr);

    EXPECT_NEAR(view1->data()[0], 0.0f, 0.1f);
    EXPECT_NEAR(view2->data()[0], 60.0f, 1.0f);
    EXPECT_NEAR(view3->data()[0], 120.0f, 2.0f);
}

/**
 * @brief Test BF16 activation-activation GEMM (Q @ K^T pattern)
 *
 * Tests multiply_activations with both A and B as FP32 activation buffers,
 * converted to BF16 internally for computation.
 * DEPRECATED: Uses old GEMM kernels (kernels/cpu/gemm). OneDNN v4 replaces this.
 */
TEST(Test__BF16Tensor, DISABLED_ActivationGemmQKT)
{
    // Small attention-like computation: Q @ K^T
    // Q: [4, 8] (seq_len=4, head_dim=8)
    // K: [4, 8]
    // scores: [4, 4]

    std::vector<float> Q_data = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
        3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
        4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

    std::vector<float> K_data = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    std::vector<float> scores_data(16, 0.0f);

    // Create dummy BF16 tensor just to get GEMM kernel
    auto dummy_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{1, 1});
    auto gemm = dummy_tensor->createGemm();

    // Execute: scores = Q @ K^T (transpose_B=true)
    bool success = gemm->multiply_activations_strided(
        Q_data.data(), K_data.data(), scores_data.data(),
        4, 4, 8, // m=4, n=4, k=8
        8, 8, 4, // lda=k, ldb=k, ldc=n
        true,    // transpose_B (K^T)
        1.0f,    // alpha
        0.0f,    // beta
        nullptr, -1);

    ASSERT_TRUE(success);

    // Expected results (computed with FP32):
    // scores[0,0] = Q[0] · K[0] = 1*1 + 2*1 + ... + 8*1 = 36
    // scores[0,1] = Q[0] · K[1] = 1*2 + 2*2 + ... + 8*2 = 72
    // scores[0,2] = Q[0] · K[2] = 1*3 + 2*3 + ... + 8*3 = 108
    // scores[0,3] = Q[0] · K[3] = 1*4 + 2*4 + ... + 8*4 = 144

    // BF16 tolerance: ~1-2% relative error for accumulated results
    EXPECT_NEAR(scores_data[0], 36.0f, 1.0f);
    EXPECT_NEAR(scores_data[1], 72.0f, 2.0f);
    EXPECT_NEAR(scores_data[2], 108.0f, 3.0f);
    EXPECT_NEAR(scores_data[3], 144.0f, 4.0f);
}

/**
 * @brief Test BF16 strided activation GEMM (multi-head attention pattern)
 *
 * Tests multiply_activations_strided for zero-copy multi-head operations.
 *
 * DISABLED: FloatingPointGemmKernel no longer supports strided GEMM.
 * Strided operations are handled by attention kernels directly.
 */
TEST(Test__BF16Tensor, DISABLED_ActivationGemmStrided)
{
    // Simulate 2 heads, seq_len=2, head_dim=4
    // V: [seq_len, n_heads, head_dim] = [2, 2, 4] interleaved
    const int seq_len = 2;
    const int n_heads = 2;
    const int head_dim = 4;

    std::vector<float> weights_data = {
        1.0f, 0.0f, // head 0: [2, 2] contiguous
        0.0f, 1.0f};

    std::vector<float> V_data = {
        // token 0: [head0: 1,2,3,4, head1: 5,6,7,8]
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        // token 1: [head0: 2,3,4,5, head1: 6,7,8,9]
        2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    std::vector<float> output_data(16, 0.0f); // [seq_len, n_heads, head_dim]

    // Create dummy BF16 tensor to get GEMM kernel
    auto dummy_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{1, 1});
    auto gemm = dummy_tensor->createGemm();

    // Process head 0 with strided GEMM
    const float *V_h0 = V_data.data() + 0; // First element of head 0
    float *output_h0 = output_data.data() + 0;

    const int lda = seq_len;            // weights contiguous
    const int ldb = n_heads * head_dim; // V stride between rows
    const int ldc = n_heads * head_dim; // output stride between rows

    bool success = gemm->multiply_activations_strided(
        weights_data.data(), V_h0, output_h0,
        seq_len, head_dim, seq_len, // m=2, n=4, k=2
        lda, ldb, ldc,
        false, // transpose_B=false (weights @ V)
        1.0f,  // alpha
        0.0f,  // beta
        nullptr, -1);

    ASSERT_TRUE(success);

    // Expected for head 0:
    // output[0,h0,:] = weights[0,:] @ V[:,h0,:] = 1.0 * V[0,h0,:] + 0.0 * V[1,h0,:] = [1,2,3,4]
    // output[1,h0,:] = weights[1,:] @ V[:,h0,:] = 0.0 * V[0,h0,:] + 1.0 * V[1,h0,:] = [2,3,4,5]

    // Check head 0 output (indices 0,1,2,3 and 8,9,10,11)
    EXPECT_NEAR(output_data[0], 1.0f, 0.1f);
    EXPECT_NEAR(output_data[1], 2.0f, 0.1f);
    EXPECT_NEAR(output_data[2], 3.0f, 0.1f);
    EXPECT_NEAR(output_data[3], 4.0f, 0.1f);

    EXPECT_NEAR(output_data[8], 2.0f, 0.1f);
    EXPECT_NEAR(output_data[9], 3.0f, 0.1f);
    EXPECT_NEAR(output_data[10], 4.0f, 0.1f);
    EXPECT_NEAR(output_data[11], 5.0f, 0.1f);
}

/**
 * @brief Test BF16Tensor to<float>() template method matches to_fp32()
 */
TEST(Test__BF16Tensor, ToFloat_TemplateMethod)
{
    // Create a simple BF16 tensor with known values
    std::vector<size_t> shape = {2, 32};
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i)
    {
        data[i] = fp32_to_bf16(static_cast<float>(i) * 0.5f);
    }
    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    // Test to<float>() template method
    std::vector<float> result_template(64);
    tensor->template to<float>(result_template.data());

    // Compare with legacy to_fp32()
    std::vector<float> result_legacy(64);
    tensor->to_fp32(result_legacy.data());

    // Should be identical
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test BF16Tensor to<uint16_t>(BF16) template method matches to_bf16()
 */
TEST(Test__BF16Tensor, ToBF16_TemplateMethod)
{
    // Create a simple BF16 tensor
    std::vector<size_t> shape = {2, 32};
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i)
    {
        data[i] = fp32_to_bf16(static_cast<float>(i) * 0.5f);
    }
    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    // Test to<uint16_t>() with BF16 format
    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::BF16);

    // Compare with legacy to_bf16()
    std::vector<uint16_t> result_legacy(64);
    tensor->to_bf16(result_legacy.data());

    // Should be identical (should be a no-op for BF16 tensor)
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test BF16Tensor to<uint16_t>(FP16) template method matches to_fp16()
 */
TEST(Test__BF16Tensor, ToFP16_TemplateMethod)
{
    // Create a simple BF16 tensor
    std::vector<size_t> shape = {2, 32};
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i)
    {
        data[i] = fp32_to_bf16(static_cast<float>(i) * 0.5f);
    }
    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    // Test to<uint16_t>() with FP16 format
    std::vector<uint16_t> result_template(64);
    tensor->template to<uint16_t>(result_template.data(), TensorType::FP16);

    // Compare with legacy to_fp16()
    std::vector<uint16_t> result_legacy(64);
    tensor->to_fp16(result_legacy.data());

    // Should be identical
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_EQ(result_template[i], result_legacy[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test BF16Tensor to<int8_t>() INT8 quantization
 */
TEST(Test__BF16Tensor, ToINT8_TemplateMethod)
{
    // Create a simple BF16 tensor
    std::vector<size_t> shape = {2, 32};
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i)
    {
        data[i] = fp32_to_bf16(static_cast<float>(i) * 0.5f);
    }
    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    // Test to<int8_t>() INT8 quantization
    const size_t total_elements = 64;
    std::vector<int8_t> int8_data(total_elements);

    tensor->template to<int8_t>(int8_data.data());

    // Verify all int8 values are in valid range [-127, 127]
    for (size_t i = 0; i < total_elements; ++i)
    {
        EXPECT_GE(int8_data[i], -127) << "Value at index " << i << " too low";
        EXPECT_LE(int8_data[i], 127) << "Value at index " << i << " too high";
    }
}

/**
 * @brief Test BF16Tensor to<int32_t>() INT32 conversion
 */
TEST(Test__BF16Tensor, ToINT32_TemplateMethod)
{
    // Create a simple BF16 tensor
    std::vector<size_t> shape = {2, 32};
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i)
    {
        data[i] = fp32_to_bf16(static_cast<float>(i) * 0.5f);
    }
    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    // Test to<int32_t>() INT32 conversion
    const size_t total_elements = 64;
    std::vector<int32_t> int32_data(total_elements);

    tensor->template to<int32_t>(int32_data.data());

    // Verify no overflow occurred (INT32 range is huge, so we just check it's in bounds)
    for (size_t i = 0; i < total_elements; ++i)
    {
        EXPECT_GE(int32_data[i], INT32_MIN);
        EXPECT_LE(int32_data[i], INT32_MAX);
    }

    // Verify values are non-zero for non-zero inputs (quantization preserves magnitude order)
    for (size_t i = 1; i < total_elements; ++i) // Skip i=0 which is 0*0.5=0
    {
        EXPECT_NE(int32_data[i], 0) << "Non-zero input at index " << i << " should produce non-zero output";
    }
}

/**
 * @brief Test round-trip conversion: BF16 → FP32 → FP16 → FP32
 */
TEST(Test__BF16Tensor, RoundTrip_BF16_FP32_FP16_FP32)
{
    // Create a simple BF16 tensor
    std::vector<size_t> shape = {2, 32};
    std::vector<uint16_t> data(64);
    for (size_t i = 0; i < 64; ++i)
    {
        data[i] = fp32_to_bf16(static_cast<float>(i) * 0.5f);
    }
    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    // Step 1: BF16 → FP32
    std::vector<float> fp32_data(64);
    tensor->template to<float>(fp32_data.data());

    // Step 2: FP32 → FP16 (create FP32Tensor, then convert)
    auto fp32_tensor = std::make_shared<FP32Tensor>(shape);
    std::memcpy(fp32_tensor->mutable_data(), fp32_data.data(), 64 * sizeof(float));

    std::vector<uint16_t> fp16_data(64);
    fp32_tensor->template to<uint16_t>(fp16_data.data(), TensorType::FP16);

    // Step 3: FP16 → FP32
    auto fp16_tensor = std::make_shared<FP16Tensor>(shape, fp16_data);
    std::vector<float> final_fp32_data(64);
    fp16_tensor->template to<float>(final_fp32_data.data());

    // Verify round-trip accuracy
    for (size_t i = 0; i < 64; ++i)
    {
        float original = bf16_to_fp32(data[i]);
        float final_value = final_fp32_data[i];

        if (std::abs(original) > 1e-6f) // Skip near-zero values
        {
            float rel_error = std::abs((final_value - original) / original);
            // Both BF16 and FP16 have reduced precision, so we expect ~1-2% error
            EXPECT_LT(rel_error, 0.02f)
                << "Round-trip error at index " << i
                << " original=" << original
                << " final=" << final_value;
        }
    }
}
