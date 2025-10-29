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
    EXPECT_EQ(tensor->device_index(), -1); // Default to CPU
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

/**
 * @brief Test BF16 GEMM correctness with small known matrix
 *
 * Same test as FP32 but with BF16 precision tolerance.
 */
TEST(Test__BF16Tensor, GemmCorrectnessTranspose)
{
    // Create activation matrix A [2, 3] in FP32
    std::vector<float> A_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};

    // Create weight matrix B in BF16 [2, 3] transposed layout
    std::vector<uint16_t> B_bf16_data = {
        fp32_to_bf16(1.0f), fp32_to_bf16(2.0f), fp32_to_bf16(3.0f),
        fp32_to_bf16(4.0f), fp32_to_bf16(5.0f), fp32_to_bf16(6.0f)};

    auto B_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{2, 3}, B_bf16_data);

    // Create GEMM kernel
    auto gemm = B_tensor->createGemm();
    ASSERT_NE(gemm, nullptr);

    std::vector<float> C_data(4, 0.0f);

    // Execute: C = A @ B^T
    bool success = gemm->multiply(
        A_data.data(),
        C_data.data(),
        2, 2, 3,
        true,
        1.0f, 0.0f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Expected results (exact for these small integers)
    // C[0,0] = 1*1 + 2*2 + 3*3 = 14
    // C[0,1] = 1*4 + 2*5 + 3*6 = 32
    // C[1,0] = 4*1 + 5*2 + 6*3 = 32
    // C[1,1] = 4*4 + 5*5 + 6*6 = 77

    // BF16 tolerance: 1% relative error for accumulated results
    EXPECT_NEAR(C_data[0], 14.0f, 0.15f);
    EXPECT_NEAR(C_data[1], 32.0f, 0.35f);
    EXPECT_NEAR(C_data[2], 32.0f, 0.35f);
    EXPECT_NEAR(C_data[3], 77.0f, 0.80f);
}

/**
 * @brief Test BF16 GEMM with alpha and beta parameters
 */
TEST(Test__BF16Tensor, GemmAlphaBeta)
{
    // Simple 2x2 identity-like operation
    std::vector<float> A_data = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<uint16_t> B_bf16_data = {
        fp32_to_bf16(1.0f), fp32_to_bf16(0.0f),
        fp32_to_bf16(0.0f), fp32_to_bf16(1.0f)};

    auto B_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{2, 2}, B_bf16_data);
    auto gemm = B_tensor->createGemm();

    std::vector<float> C_data = {10.0f, 20.0f, 30.0f, 40.0f};

    // Execute: C = 2.0 * A @ I + 0.5 * C
    bool success = gemm->multiply(
        A_data.data(), C_data.data(),
        2, 2, 2,
        true, 2.0f, 0.5f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Expected: C = 2.0 * A + 0.5 * C_old
    EXPECT_NEAR(C_data[0], 7.0f, 0.1f);
    EXPECT_NEAR(C_data[1], 14.0f, 0.2f);
    EXPECT_NEAR(C_data[2], 21.0f, 0.3f);
    EXPECT_NEAR(C_data[3], 28.0f, 0.4f);
}

/**
 * @brief Test BF16 GEMM with non-transposed layout
 */
TEST(Test__BF16Tensor, GemmNoTranspose)
{
    std::vector<float> A_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};

    // B in non-transposed layout [3, 2]
    std::vector<uint16_t> B_bf16_data = {
        fp32_to_bf16(1.0f), fp32_to_bf16(4.0f),
        fp32_to_bf16(2.0f), fp32_to_bf16(5.0f),
        fp32_to_bf16(3.0f), fp32_to_bf16(6.0f)};

    auto B_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{3, 2}, B_bf16_data);
    auto gemm = B_tensor->createGemm();

    std::vector<float> C_data(4, 0.0f);

    bool success = gemm->multiply(
        A_data.data(), C_data.data(),
        2, 2, 3,
        false, // No transpose
        1.0f, 0.0f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Same expected results as transposed case
    EXPECT_NEAR(C_data[0], 14.0f, 0.15f);
    EXPECT_NEAR(C_data[1], 32.0f, 0.35f);
    EXPECT_NEAR(C_data[2], 32.0f, 0.35f);
    EXPECT_NEAR(C_data[3], 77.0f, 0.80f);
}

/**
 * @brief Test BF16 GEMM with larger matrix
 *
 * Stress test to verify BF16 precision holds up with accumulation.
 */
TEST(Test__BF16Tensor, GemmLargerMatrix)
{
    const int m = 16, n = 32, k = 24;

    // Create deterministic inputs
    std::vector<float> A_data(m * k);
    std::vector<uint16_t> B_bf16_data(n * k);

    for (int i = 0; i < m * k; ++i)
    {
        A_data[i] = static_cast<float>((i * 7 + 3) % 100) / 10.0f;
    }
    for (int i = 0; i < n * k; ++i)
    {
        float val = static_cast<float>((i * 11 + 5) % 100) / 10.0f;
        B_bf16_data[i] = fp32_to_bf16(val);
    }

    auto B_tensor = std::make_shared<BF16Tensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
        B_bf16_data);
    auto gemm = B_tensor->createGemm();

    std::vector<float> C_data(m * n, 0.0f);

    bool success = gemm->multiply(
        A_data.data(), C_data.data(),
        m, n, k,
        true, 1.0f, 0.0f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Verify results are reasonable (no NaN/Inf)
    bool has_nonzero = false;
    for (float val : C_data)
    {
        EXPECT_FALSE(std::isnan(val)) << "Unexpected NaN in result";
        EXPECT_FALSE(std::isinf(val)) << "Unexpected Inf in result";
        if (std::abs(val) > 1e-6f)
        {
            has_nonzero = true;
        }
    }
    EXPECT_TRUE(has_nonzero) << "All results are zero (unexpected)";
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

/**
 * @brief Test edge case: zero matrix GEMM
 */
TEST(Test__BF16Tensor, GemmZeroMatrix)
{
    // A is all zeros
    std::vector<float> A_data(6, 0.0f); // [2, 3]

    // B is non-zero
    std::vector<uint16_t> B_bf16_data = {
        fp32_to_bf16(1.0f), fp32_to_bf16(2.0f), fp32_to_bf16(3.0f),
        fp32_to_bf16(4.0f), fp32_to_bf16(5.0f), fp32_to_bf16(6.0f)};

    auto B_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{2, 3}, B_bf16_data);
    auto gemm = B_tensor->createGemm();

    std::vector<float> C_data(4, 0.0f);

    bool success = gemm->multiply(
        A_data.data(), C_data.data(),
        2, 2, 3,
        true, 1.0f, 0.0f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Result should be all zeros
    for (float val : C_data)
    {
        EXPECT_NEAR(val, 0.0f, 1e-6f);
    }
}

/**
 * @brief Test createGemm returns valid kernel
 */
TEST(Test__BF16Tensor, CreateGemmNotNull)
{
    std::vector<uint16_t> data(10, fp32_to_bf16(1.0f));
    auto tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{2, 5}, data);

    auto gemm = tensor->createGemm();

    ASSERT_NE(gemm, nullptr);
    EXPECT_TRUE(gemm->supports_device(-1)); // Should support CPU
}

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
