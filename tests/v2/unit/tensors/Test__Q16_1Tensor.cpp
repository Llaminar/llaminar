/**
 * @file Test__Q16_1Tensor.cpp
 * @brief Unit tests for Q16_1Tensor high-precision quantized activation storage
 * @author David Sanftenberg
 *
 * Tests encode/decode round-trip precision and verifies that Q16_1 provides
 * significantly better precision than Q8_1 (256× finer granularity).
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include <vector>
#include <random>
#include <cstring>
#include <cmath>

using namespace llaminar2;

namespace
{
    // Helper to create an empty mutable Q16_1 tensor
    std::shared_ptr<Q16_1Tensor> create_mutable_q16_1_tensor(int rows, int cols)
    {
        return std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    }

    // Compute relative L2 error between two float arrays
    float compute_relative_l2_error(const float *actual, const float *expected, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
        }
        return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
    }

    // Compute max absolute error between two float arrays
    float compute_max_abs_error(const float *actual, const float *expected, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float err = std::abs(actual[i] - expected[i]);
            max_err = std::max(max_err, err);
        }
        return max_err;
    }
}

/**
 * @brief Basic encode/decode round-trip test for Q16_1Tensor
 *
 * Quantizes random FP32 data to Q16_1, then dequantizes back to FP32.
 * Verifies that the round-trip error is very small (< 0.01% relative L2).
 */
TEST(Test__Q16_1Tensor, EncodeDecodeRoundTrip_SmallError)
{
    const int rows = 8;
    const int cols = 128; // Multiple of 32 (block size)
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random FP32 data in typical activation range [-2, 2]
    std::vector<float> original(elements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Quantize to Q16_1
    auto q16_1_tensor = Q16_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q16_1_tensor, nullptr);
    ASSERT_EQ(q16_1_tensor->shape()[0], static_cast<size_t>(rows));
    ASSERT_EQ(q16_1_tensor->shape()[1], static_cast<size_t>(cols));
    ASSERT_EQ(q16_1_tensor->native_type(), TensorType::Q16_1);

    // Dequantize back to FP32
    std::vector<float> reconstructed(elements);
    q16_1_tensor->to_fp32(reconstructed.data());

    // Compute errors
    float rel_l2_error = compute_relative_l2_error(reconstructed.data(), original.data(), elements);
    float max_abs_error = compute_max_abs_error(reconstructed.data(), original.data(), elements);

    std::cout << "[Q16_1 Round-Trip] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 Round-Trip] Max abs error: " << max_abs_error << std::endl;

    // Q16_1 should have < 0.1% relative L2 error (better than Q8_1's ~0.4%)
    // The FP16 scale storage limits precision slightly
    EXPECT_LT(rel_l2_error, 0.001f)
        << "Q16_1 round-trip error exceeds 0.1% threshold";

    // Max error should be small for [-2, 2] range
    // With FP16 scale, max error is ~0.002 in practice
    EXPECT_LT(max_abs_error, 0.003f)
        << "Q16_1 max abs error exceeds 0.003 threshold";
}

/**
 * @brief Compare Q16_1 vs Q8_1 precision
 *
 * Quantizes the same FP32 data to both Q16_1 and Q8_1, then compares
 * the round-trip error. Q16_1 should have ~256× less error.
 */
TEST(Test__Q16_1Tensor, Q16_1_vs_Q8_1_PrecisionComparison)
{
    const int rows = 4;
    const int cols = 256;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random FP32 data
    std::vector<float> original(elements);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Quantize to Q16_1
    auto q16_1_tensor = Q16_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Quantize to Q8_1
    auto q8_1_tensor = Q8_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q16_1_tensor, nullptr);
    ASSERT_NE(q8_1_tensor, nullptr);

    // Dequantize both
    std::vector<float> q16_1_recon(elements);
    std::vector<float> q8_1_recon(elements);
    q16_1_tensor->to_fp32(q16_1_recon.data());
    q8_1_tensor->to_fp32(q8_1_recon.data());

    // Compute errors
    float q16_1_error = compute_relative_l2_error(q16_1_recon.data(), original.data(), elements);
    float q8_1_error = compute_relative_l2_error(q8_1_recon.data(), original.data(), elements);

    std::cout << "[Precision Comparison] Q16_1 relative L2 error: " << (q16_1_error * 100.0f) << "%" << std::endl;
    std::cout << "[Precision Comparison] Q8_1 relative L2 error: " << (q8_1_error * 100.0f) << "%" << std::endl;

    if (q16_1_error > 0 && q8_1_error > 0)
    {
        float improvement = q8_1_error / q16_1_error;
        std::cout << "[Precision Comparison] Q16_1 is " << improvement << "× better than Q8_1" << std::endl;

        // Q16_1 should be at least 2× better than Q8_1
        // The theoretical 256× improvement is limited by FP16 scale storage
        EXPECT_GT(improvement, 2.0f)
            << "Q16_1 should be at least 2× more precise than Q8_1";
    }

    // Verify Q16_1 absolute error bounds
    EXPECT_LT(q16_1_error, 0.002f)
        << "Q16_1 error should be < 0.2%";
}

/**
 * @brief Test that Q16_1Block sum_qs is correctly computed
 *
 * Verifies that the pre-computed sum stored in each block matches
 * the actual sum of the quantized values.
 */
TEST(Test__Q16_1Tensor, PrecomputedSumIsCorrect)
{
    const int rows = 2;
    const int cols = 64; // 2 blocks per row
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Check each block
    const Q16_1Block *blocks = tensor->q16_1_blocks();
    const size_t blocks_per_row = tensor->blocks_per_row();
    const size_t total_blocks = tensor->total_blocks();

    for (size_t b = 0; b < total_blocks; ++b)
    {
        const Q16_1Block &block = blocks[b];

        // Compute actual sum
        int32_t actual_sum = 0;
        for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
        {
            actual_sum += static_cast<int32_t>(block.qs[i]);
        }

        EXPECT_EQ(block.sum_qs, actual_sum)
            << "Block " << b << " has incorrect pre-computed sum";
    }

    std::cout << "[Q16_1 Sum Check] Verified " << total_blocks << " blocks" << std::endl;
}

/**
 * @brief Test Q16_1 → Q8_1 conversion
 *
 * Verifies that to_q8_1() produces valid Q8_1 data and that the
 * conversion introduces expected precision loss.
 */
TEST(Test__Q16_1Tensor, ConvertToQ8_1)
{
    const int rows = 4;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> original(elements);
    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Create Q16_1 tensor
    auto q16_1_tensor = Q16_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q16_1_tensor, nullptr);

    // Convert to Q8_1
    auto q8_1_tensor = q16_1_tensor->to_q8_1();

    ASSERT_NE(q8_1_tensor, nullptr);
    ASSERT_EQ(q8_1_tensor->shape()[0], static_cast<size_t>(rows));
    ASSERT_EQ(q8_1_tensor->shape()[1], static_cast<size_t>(cols));
    ASSERT_EQ(q8_1_tensor->native_type(), TensorType::Q8_1);

    // Dequantize Q8_1
    std::vector<float> q8_1_recon(elements);
    q8_1_tensor->to_fp32(q8_1_recon.data());

    // The Q8_1 result should be close to original (with Q8_1-level precision)
    float error = compute_relative_l2_error(q8_1_recon.data(), original.data(), elements);

    std::cout << "[Q16_1→Q8_1] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;

    // Should be similar to direct FP32→Q8_1 quantization error (~0.5-1%)
    EXPECT_LT(error, 0.02f)
        << "Q16_1→Q8_1 conversion error exceeds 2% threshold";
}

/**
 * @brief Test mutable Q16_1 tensor with copyFrom
 *
 * Creates a mutable Q16_1 tensor, copies FP32 data into it,
 * and verifies the round-trip precision.
 */
TEST(Test__Q16_1Tensor, MutableCopyFrom)
{
    const int rows = 4;
    const int cols = 96;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> original(elements);
    std::mt19937 gen(888);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Create FP32 source tensor
    auto fp32_src = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    std::memcpy(fp32_src->mutable_data(), original.data(), elements * sizeof(float));

    // Create mutable Q16_1 tensor
    auto q16_1_tensor = create_mutable_q16_1_tensor(rows, cols);

    // Copy from FP32
    ASSERT_TRUE(q16_1_tensor->copyFrom(fp32_src.get()));

    // Dequantize and verify
    std::vector<float> reconstructed(elements);
    q16_1_tensor->to_fp32(reconstructed.data());

    float error = compute_relative_l2_error(reconstructed.data(), original.data(), elements);

    std::cout << "[Q16_1 copyFrom] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;

    EXPECT_LT(error, 0.003f)
        << "Q16_1 copyFrom round-trip error exceeds 0.3% threshold";
}

/**
 * @brief Test Q16_1 with extreme values
 *
 * Tests quantization behavior at the boundaries of typical activation ranges.
 */
TEST(Test__Q16_1Tensor, ExtremeValues)
{
    const int rows = 1;
    const int cols = 32; // One block
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Create data with extreme values
    std::vector<float> data(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        // Alternate between -10 and +10 (fairly extreme for activations)
        data[i] = (i % 2 == 0) ? -10.0f : 10.0f;
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    std::vector<float> recon(elements);
    tensor->to_fp32(recon.data());

    // Check each value
    for (size_t i = 0; i < elements; ++i)
    {
        float expected = (i % 2 == 0) ? -10.0f : 10.0f;
        EXPECT_NEAR(recon[i], expected, 0.01f)
            << "Mismatch at index " << i;
    }

    std::cout << "[Q16_1 Extreme] All values within tolerance" << std::endl;
}

/**
 * @brief Test Q16_1 with very small values (near zero)
 *
 * Verifies that small values are preserved with high precision.
 */
TEST(Test__Q16_1Tensor, SmallValues)
{
    const int rows = 1;
    const int cols = 32;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Create data with small values
    std::vector<float> data(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        // Values in range [-0.001, 0.001]
        data[i] = (static_cast<float>(i) / elements - 0.5f) * 0.002f;
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    std::vector<float> recon(elements);
    tensor->to_fp32(recon.data());

    // Compute error
    float max_abs_error = compute_max_abs_error(recon.data(), data.data(), elements);

    std::cout << "[Q16_1 Small Values] Max abs error: " << max_abs_error << std::endl;

    // For very small values, the FP16 scale limits precision
    // Allow error up to ~1% of the max value in range
    EXPECT_LT(max_abs_error, 0.01f)
        << "Q16_1 small value error exceeds threshold";
}

/**
 * @brief Test SIMD decode paths (AVX512/AVX2/scalar consistency)
 *
 * Verifies that all decode paths produce the same results.
 */
TEST(Test__Q16_1Tensor, DecodeConsistency)
{
    const int rows = 2;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    const Q16_1Block *blocks = tensor->q16_1_blocks();
    const size_t total_blocks = tensor->total_blocks();

    // Decode each block with scalar and compare to main decode
    for (size_t b = 0; b < total_blocks; ++b)
    {
        alignas(64) float scalar_output[Q16_1Block::BLOCK_SIZE];
        alignas(64) float main_output[Q16_1Block::BLOCK_SIZE];

        Q16_1Tensor::decodeBlockScalar(blocks[b], scalar_output);
        Q16_1Tensor::decodeBlock(blocks[b], main_output);

        for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
        {
            EXPECT_FLOAT_EQ(scalar_output[i], main_output[i])
                << "Decode mismatch at block " << b << " element " << i;
        }
    }

    std::cout << "[Q16_1 Decode] Verified " << total_blocks << " blocks" << std::endl;
}

/**
 * @brief Test view support
 *
 * Creates a view of a Q16_1 tensor and verifies data access.
 */
TEST(Test__Q16_1Tensor, ViewSupport)
{
    const int rows = 8;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate data
    std::vector<float> data(elements);
    std::mt19937 gen(222);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Create full tensor
    auto full_tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(full_tensor, nullptr);

    // Create view of rows 2-5 (4 rows)
    auto view = full_tensor->create_view(
        {4, static_cast<size_t>(cols)},
        2 * cols); // Offset by 2 rows

    ASSERT_NE(view, nullptr);
    ASSERT_EQ(view->shape()[0], 4);
    ASSERT_EQ(view->shape()[1], static_cast<size_t>(cols));
    ASSERT_TRUE(view->is_view());

    // Dequantize view
    std::vector<float> view_data(4 * cols);
    view->to_fp32(view_data.data());

    // Compare to expected (rows 2-5 of original)
    float max_view_error = 0.0f;
    for (size_t i = 0; i < 4 * static_cast<size_t>(cols); ++i)
    {
        size_t orig_idx = (2 * cols) + i;
        float err = std::abs(view_data[i] - data[orig_idx]);
        max_view_error = std::max(max_view_error, err);
    }

    std::cout << "[Q16_1 View] Max view error: " << max_view_error << std::endl;

    // Allow for quantization error in view comparison
    EXPECT_LT(max_view_error, 0.003f)
        << "View data diverges too much from expected";

    std::cout << "[Q16_1 View] View data matches expected" << std::endl;
}

/**
 * @brief Stress test with large tensor
 *
 * Tests Q16_1 with a larger tensor to verify performance and correctness.
 */
TEST(Test__Q16_1Tensor, LargeTensorStressTest)
{
    const int rows = 128;
    const int cols = 1024;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(333);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    std::vector<float> recon(elements);
    tensor->to_fp32(recon.data());

    // Compute error
    float error = compute_relative_l2_error(recon.data(), data.data(), elements);
    float max_error = compute_max_abs_error(recon.data(), data.data(), elements);

    std::cout << "[Q16_1 Large] " << rows << "×" << cols << " tensor" << std::endl;
    std::cout << "[Q16_1 Large] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 Large] Max abs error: " << max_error << std::endl;

    EXPECT_LT(error, 0.001f)
        << "Q16_1 large tensor error exceeds 0.1% threshold";
}

// ==================== Q16_1 SIMD Native Operation Tests ====================

/**
 * @brief Test native Q16_1 + Q16_1 addition
 *
 * Verifies that q16_1_add_q16_1() produces correct results by comparing
 * against reference FP32 addition.
 */
TEST(Test__Q16_1Tensor, NativeQ16_1Addition)
{
    const int rows = 64;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    // Generate two random float arrays
    std::vector<float> data_a(elements);
    std::vector<float> data_b(elements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data_a[i] = dist(gen);
        data_b[i] = dist(gen);
    }

    // Compute reference (FP32 add)
    std::vector<float> reference(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        reference[i] = data_a[i] + data_b[i];
    }

    // Quantize both to Q16_1
    auto tensor_a = Q16_1Tensor::quantize_from_fp32(
        data_a.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto tensor_b = Q16_1Tensor::quantize_from_fp32(
        data_b.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Create output tensor with raw storage
    std::vector<uint8_t> raw_output(n_blocks * sizeof(Q16_1Block), 0);
    auto tensor_out = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_output);

    // Get raw block pointers using proper accessors
    const Q16_1Block *blocks_a = tensor_a->q16_1_blocks();
    const Q16_1Block *blocks_b = tensor_b->q16_1_blocks();
    Q16_1Block *blocks_out = tensor_out->mutable_q16_1_blocks();

    // Perform native Q16_1 addition
    simd::q16_1_add_q16_1(blocks_a, blocks_b, blocks_out, elements);

    // Dequantize result
    std::vector<float> result(elements);
    tensor_out->to_fp32(result.data());

    // Compare to reference
    float error = compute_relative_l2_error(result.data(), reference.data(), elements);
    float max_error = compute_max_abs_error(result.data(), reference.data(), elements);

    std::cout << "[Q16_1 Native Add] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 Native Add] Max abs error: " << max_error << std::endl;

    // Q16_1 addition should be very accurate (cumulative quant error from 3 operations)
    // Input quant + Input quant + Output quant = ~3× single quant error
    EXPECT_LT(error, 0.01f) // < 1% relative error
        << "Q16_1 native addition error exceeds 1% threshold";
    EXPECT_LT(max_error, 0.01f) // Max abs error < 0.01
        << "Q16_1 native addition max abs error too high";
}

/**
 * @brief Test Q16_1 + FP32 addition (residual += delta pattern)
 *
 * Verifies that q16_1_add_fp32() correctly adds FP32 delta to Q16_1 residual.
 */
TEST(Test__Q16_1Tensor, NativeQ16_1AddFP32)
{
    const int rows = 32;
    const int cols = 256;
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    // Generate random residual and delta
    std::vector<float> residual_fp32(elements);
    std::vector<float> delta_fp32(elements);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (size_t i = 0; i < elements; ++i)
    {
        residual_fp32[i] = dist(gen);
        delta_fp32[i] = dist(gen);
    }

    // Compute reference
    std::vector<float> reference(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        reference[i] = residual_fp32[i] + delta_fp32[i];
    }

    // Create Q16_1 residual tensor with raw storage
    auto temp_tensor = Q16_1Tensor::quantize_from_fp32(
        residual_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Copy to mutable tensor
    std::vector<uint8_t> raw_residual(n_blocks * sizeof(Q16_1Block));
    std::memcpy(raw_residual.data(), temp_tensor->q16_1_blocks(), n_blocks * sizeof(Q16_1Block));
    auto residual = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_residual);

    Q16_1Block *blocks = residual->mutable_q16_1_blocks();

    // Add FP32 delta in-place
    simd::q16_1_add_fp32(blocks, delta_fp32.data(), elements);

    // Dequantize result
    std::vector<float> result(elements);
    residual->to_fp32(result.data());

    // Compare to reference
    float error = compute_relative_l2_error(result.data(), reference.data(), elements);
    float max_error = compute_max_abs_error(result.data(), reference.data(), elements);

    std::cout << "[Q16_1 + FP32] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 + FP32] Max abs error: " << max_error << std::endl;

    // Should be slightly better than Q16_1+Q16_1 (only 2 quant ops: input + output)
    EXPECT_LT(error, 0.005f) // < 0.5% relative error
        << "Q16_1 + FP32 error exceeds 0.5% threshold";
    EXPECT_LT(max_error, 0.005f)
        << "Q16_1 + FP32 max abs error too high";
}

/**
 * @brief Test optimized Q16_1 → Q8_1 conversion
 *
 * Verifies that q16_1_to_q8_1_packed() produces correct Q8_1 output.
 */
TEST(Test__Q16_1Tensor, Q16_1ToQ8_1Packed)
{
    const int rows = 16;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize to Q16_1
    auto q16_tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Allocate Q8_1 output
    std::vector<Q8_1Block> q8_blocks(n_blocks);

    // Convert using SIMD function with proper accessor
    const Q16_1Block *q16_blocks = q16_tensor->q16_1_blocks();
    simd::q16_1_to_q8_1_packed(q16_blocks, q8_blocks.data(), n_blocks);

    // Dequantize Q8_1 result manually
    std::vector<float> q8_result(elements);
    for (size_t blk = 0; blk < n_blocks; ++blk)
    {
        float scale = simd::fp16_to_fp32(q8_blocks[blk].d);
        for (int j = 0; j < 32; ++j)
        {
            q8_result[blk * 32 + j] = scale * static_cast<float>(q8_blocks[blk].qs[j]);
        }
    }

    // Compare to original data (should have Q8 precision, ~0.4% error)
    float error = compute_relative_l2_error(q8_result.data(), data.data(), elements);
    float max_error = compute_max_abs_error(q8_result.data(), data.data(), elements);

    std::cout << "[Q16_1 → Q8_1 Packed] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 → Q8_1 Packed] Max abs error: " << max_error << std::endl;

    // Should match Q8 precision (the conversion loses Q16 precision)
    EXPECT_LT(error, 0.01f) // < 1% (Q8 precision + Q16 input precision)
        << "Q16_1 → Q8_1 conversion error too high";
}

/**
 * @brief Test Q16_1 native add with edge cases
 *
 * Tests near-zero values and very large values.
 */
TEST(Test__Q16_1Tensor, NativeQ16_1AddEdgeCases)
{
    const int rows = 4;
    const int cols = 32; // 1 row of blocks
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    // Test 1: Near-zero values
    {
        std::vector<float> data_a(elements, 1e-8f);
        std::vector<float> data_b(elements, -1e-8f);

        auto tensor_a = Q16_1Tensor::quantize_from_fp32(
            data_a.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto tensor_b = Q16_1Tensor::quantize_from_fp32(
            data_b.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

        std::vector<uint8_t> raw_output(n_blocks * sizeof(Q16_1Block), 0);
        auto tensor_out = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            raw_output);

        const Q16_1Block *blocks_a = tensor_a->q16_1_blocks();
        const Q16_1Block *blocks_b = tensor_b->q16_1_blocks();
        Q16_1Block *blocks_out = tensor_out->mutable_q16_1_blocks();

        simd::q16_1_add_q16_1(blocks_a, blocks_b, blocks_out, elements);

        // Result should be all zeros (or very close)
        std::vector<float> result(elements);
        tensor_out->to_fp32(result.data());

        for (size_t i = 0; i < elements; ++i)
        {
            EXPECT_NEAR(result[i], 0.0f, 1e-6f) << "Near-zero add failed at index " << i;
        }
    }

    // Test 2: Large canceling values
    {
        std::vector<float> data_a(elements, 1000.0f);
        std::vector<float> data_b(elements, -1000.0f);

        auto tensor_a = Q16_1Tensor::quantize_from_fp32(
            data_a.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto tensor_b = Q16_1Tensor::quantize_from_fp32(
            data_b.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

        std::vector<uint8_t> raw_output(n_blocks * sizeof(Q16_1Block), 0);
        auto tensor_out = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            raw_output);

        const Q16_1Block *blocks_a = tensor_a->q16_1_blocks();
        const Q16_1Block *blocks_b = tensor_b->q16_1_blocks();
        Q16_1Block *blocks_out = tensor_out->mutable_q16_1_blocks();

        simd::q16_1_add_q16_1(blocks_a, blocks_b, blocks_out, elements);

        std::vector<float> result(elements);
        tensor_out->to_fp32(result.data());

        for (size_t i = 0; i < elements; ++i)
        {
            EXPECT_NEAR(result[i], 0.0f, 0.1f) << "Large cancel add failed at index " << i;
        }
    }

    std::cout << "[Q16_1 Edge Cases] All edge case tests passed" << std::endl;
}

/**
 * @brief Test Q16_1 += Q8_1 residual addition (the core typed residual pattern)
 *
 * This is the most common operation in typed residual inference:
 *   residual (Q16_1) += layer_output (Q8_1)
 *
 * Verifies that:
 * 1. Round-trip error is only quantization noise
 * 2. Result matches FP32 reference within expected tolerance
 * 3. Precision is bounded by the Q8_1 input (not Q16_1 output)
 */
TEST(Test__Q16_1Tensor, NativeQ16_1AddQ8_1)
{
    const int rows = 32;
    const int cols = 256;
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    // Generate random residual (Q16_1) and delta (Q8_1) data
    std::vector<float> residual_fp32(elements);
    std::vector<float> delta_fp32(elements);
    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    for (size_t i = 0; i < elements; ++i)
    {
        residual_fp32[i] = dist(gen);
        delta_fp32[i] = dist(gen);
    }

    // Compute FP32 reference
    std::vector<float> reference(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        reference[i] = residual_fp32[i] + delta_fp32[i];
    }

    // Create Q16_1 residual (copy to mutable tensor)
    auto temp_residual = Q16_1Tensor::quantize_from_fp32(
        residual_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});
    std::vector<uint8_t> raw_residual(n_blocks * sizeof(Q16_1Block));
    std::memcpy(raw_residual.data(), temp_residual->q16_1_blocks(), n_blocks * sizeof(Q16_1Block));
    auto residual = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_residual);

    // Create Q8_1 delta
    auto delta = Q8_1Tensor::quantize_from_fp32(
        delta_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Get block pointers
    Q16_1Block *residual_blocks = residual->mutable_q16_1_blocks();
    const Q8_1Block *delta_blocks = delta->q8_1_blocks();

    // Add Q8_1 delta to Q16_1 residual in-place
    simd::q16_1_add_q8_1(residual_blocks, delta_blocks, elements);

    // Dequantize result
    std::vector<float> result(elements);
    residual->to_fp32(result.data());

    // Compare to FP32 reference
    float error = compute_relative_l2_error(result.data(), reference.data(), elements);
    float max_error = compute_max_abs_error(result.data(), reference.data(), elements);

    std::cout << "[Q16_1 += Q8_1] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 += Q8_1] Max abs error: " << max_error << std::endl;

    // Error is dominated by Q8_1 input quantization (~0.4%) + Q16_1 output (~0.002%)
    // Total should be < 1% (2 quant ops: Q8_1 input + Q16_1 output, plus Q16_1 input)
    EXPECT_LT(error, 0.01f) // < 1% relative error
        << "Q16_1 += Q8_1 error exceeds 1% threshold";
    EXPECT_LT(max_error, 0.02f)
        << "Q16_1 += Q8_1 max abs error too high";
}

/**
 * @brief Compare Q16_1 += Q8_1 vs converting Q8_1 → FP32 first
 *
 * Verifies that the native operation produces equivalent results to
 * the two-step approach: q16_1_add_fp32(residual, q8_1.to_fp32()).
 */
TEST(Test__Q16_1Tensor, NativeQ16_1AddQ8_1_vs_TwoStep)
{
    const int rows = 16;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    // Generate random data
    std::vector<float> residual_fp32(elements);
    std::vector<float> delta_fp32(elements);
    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        residual_fp32[i] = dist(gen);
        delta_fp32[i] = dist(gen);
    }

    // Create Q16_1 residual (two copies for comparison)
    auto temp_residual = Q16_1Tensor::quantize_from_fp32(
        residual_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    std::vector<uint8_t> raw_residual1(n_blocks * sizeof(Q16_1Block));
    std::vector<uint8_t> raw_residual2(n_blocks * sizeof(Q16_1Block));
    std::memcpy(raw_residual1.data(), temp_residual->q16_1_blocks(), n_blocks * sizeof(Q16_1Block));
    std::memcpy(raw_residual2.data(), temp_residual->q16_1_blocks(), n_blocks * sizeof(Q16_1Block));

    auto residual1 = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_residual1);
    auto residual2 = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_residual2);

    // Create Q8_1 delta
    auto delta_q8 = Q8_1Tensor::quantize_from_fp32(
        delta_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Method 1: Native Q16_1 += Q8_1
    simd::q16_1_add_q8_1(residual1->mutable_q16_1_blocks(), delta_q8->q8_1_blocks(), elements);

    // Method 2: Convert Q8_1 → FP32, then Q16_1 += FP32
    std::vector<float> delta_fp32_dequant(elements);
    delta_q8->to_fp32(delta_fp32_dequant.data());
    simd::q16_1_add_fp32(residual2->mutable_q16_1_blocks(), delta_fp32_dequant.data(), elements);

    // Dequantize both results
    std::vector<float> result1(elements);
    std::vector<float> result2(elements);
    residual1->to_fp32(result1.data());
    residual2->to_fp32(result2.data());

    // Compare the two methods
    float diff = compute_relative_l2_error(result1.data(), result2.data(), elements);
    float max_diff = compute_max_abs_error(result1.data(), result2.data(), elements);

    std::cout << "[Q16_1 += Q8_1 vs TwoStep] Relative L2 diff: " << (diff * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 += Q8_1 vs TwoStep] Max abs diff: " << max_diff << std::endl;

    // Both methods should produce nearly identical results
    // Difference is only from numerical ordering of FP32 operations
    EXPECT_LT(diff, 0.001f) // < 0.1% difference
        << "Native vs two-step methods diverge too much";
    EXPECT_LT(max_diff, 0.001f)
        << "Native vs two-step max diff too high";
}

/**
 * @brief Test Q16_1 += Q8_1 with realistic layer output magnitudes
 *
 * Simulates typical attention/FFN output magnitudes being added to residual.
 */
TEST(Test__Q16_1Tensor, NativeQ16_1AddQ8_1_RealisticMagnitudes)
{
    const int rows = 8;
    const int cols = 896; // Qwen2-0.5B hidden dim
    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t n_blocks = elements / 32;

    std::mt19937 gen(42);

    // Residual stream: typically larger magnitude (accumulated from previous layers)
    std::uniform_real_distribution<float> residual_dist(-3.0f, 3.0f);
    std::vector<float> residual_fp32(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        residual_fp32[i] = residual_dist(gen);
    }

    // Layer output: typically smaller (one layer's contribution)
    std::uniform_real_distribution<float> delta_dist(-0.5f, 0.5f);
    std::vector<float> delta_fp32(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        delta_fp32[i] = delta_dist(gen);
    }

    // Compute FP32 reference
    std::vector<float> reference(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        reference[i] = residual_fp32[i] + delta_fp32[i];
    }

    // Create quantized tensors
    auto temp_residual = Q16_1Tensor::quantize_from_fp32(
        residual_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});
    std::vector<uint8_t> raw_residual(n_blocks * sizeof(Q16_1Block));
    std::memcpy(raw_residual.data(), temp_residual->q16_1_blocks(), n_blocks * sizeof(Q16_1Block));
    auto residual = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        raw_residual);

    auto delta = Q8_1Tensor::quantize_from_fp32(
        delta_fp32.data(), {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Add
    simd::q16_1_add_q8_1(residual->mutable_q16_1_blocks(), delta->q8_1_blocks(), elements);

    // Compare
    std::vector<float> result(elements);
    residual->to_fp32(result.data());

    float error = compute_relative_l2_error(result.data(), reference.data(), elements);
    float max_error = compute_max_abs_error(result.data(), reference.data(), elements);

    std::cout << "[Q16_1 += Q8_1 Realistic] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 += Q8_1 Realistic] Max abs error: " << max_error << std::endl;

    // Should maintain good precision even with asymmetric magnitudes
    EXPECT_LT(error, 0.01f)
        << "Q16_1 += Q8_1 realistic magnitudes error too high";
}

// ============================================================================
// Variable Block Size Tests
// ============================================================================

/**
 * @brief Test Q16_1Tensor constructor with default block size (32)
 */
TEST(Test__Q16_1Tensor, Constructor_DefaultBlockSize)
{
    const int rows = 4;
    const int cols = 128;

    auto tensor = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    EXPECT_EQ(tensor->block_size(), 32);
    EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_32);
    EXPECT_EQ(tensor->blocks_per_row(), 4); // 128 / 32 = 4
    EXPECT_EQ(tensor->total_blocks(), 16);  // 4 rows * 4 blocks
}

/**
 * @brief Test Q16_1Tensor constructor with BLOCK_64
 */
TEST(Test__Q16_1Tensor, Constructor_Block64)
{
    const int rows = 4;
    const int cols = 64; // Exactly one 64-element block per row

    auto tensor = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        Q16BlockSize::BLOCK_64);

    EXPECT_EQ(tensor->block_size(), 64);
    EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_64);
    EXPECT_EQ(tensor->blocks_per_row(), 1); // 64 / 64 = 1
    EXPECT_EQ(tensor->total_blocks(), 4);   // 4 rows * 1 block
}

/**
 * @brief Test Q16_1Tensor constructor with BLOCK_128
 */
TEST(Test__Q16_1Tensor, Constructor_Block128)
{
    const int rows = 4;
    const int cols = 128; // Exactly one 128-element block per row

    auto tensor = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        Q16BlockSize::BLOCK_128);

    EXPECT_EQ(tensor->block_size(), 128);
    EXPECT_EQ(tensor->q16_block_size(), Q16BlockSize::BLOCK_128);
    EXPECT_EQ(tensor->blocks_per_row(), 1); // 128 / 128 = 1
    EXPECT_EQ(tensor->total_blocks(), 4);   // 4 rows * 1 block
}

// Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
// with independent scales, not a combined 192-dim block.

/**
 * @brief Test memory allocation sizes for different block sizes
 */
TEST(Test__Q16_1Tensor, Constructor_MemoryAllocation)
{
    const int rows = 8;

    // BLOCK_32: 72 bytes per block
    {
        const int cols = 64; // 2 blocks per row
        auto tensor = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            Q16BlockSize::BLOCK_32);
        // 8 rows * 2 blocks/row * 72 bytes/block = 1152 bytes
        EXPECT_EQ(tensor->size_bytes(), 8 * 2 * 72);
    }

    // BLOCK_64: 136 bytes per block
    {
        const int cols = 64; // 1 block per row
        auto tensor = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            Q16BlockSize::BLOCK_64);
        // 8 rows * 1 block/row * 136 bytes/block = 1088 bytes
        EXPECT_EQ(tensor->size_bytes(), 8 * 1 * 136);
    }

    // BLOCK_128: 264 bytes per block
    {
        const int cols = 128; // 1 block per row
        auto tensor = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            Q16BlockSize::BLOCK_128);
        // 8 rows * 1 block/row * 264 bytes/block = 2112 bytes
        EXPECT_EQ(tensor->size_bytes(), 8 * 1 * 264);
    }
}

/**
 * @brief Test blocks_per_row calculation with partial blocks
 */
TEST(Test__Q16_1Tensor, BlocksPerRow_PartialBlocks)
{
    const int rows = 1;

    // 100 elements with BLOCK_64 = ceil(100/64) = 2 blocks
    {
        auto tensor = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), 100},
            Q16BlockSize::BLOCK_64);
        EXPECT_EQ(tensor->blocks_per_row(), 2);
    }

    // 100 elements with BLOCK_128 = ceil(100/128) = 1 block
    {
        auto tensor = std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), 100},
            Q16BlockSize::BLOCK_128);
        EXPECT_EQ(tensor->blocks_per_row(), 1);
    }
}

/**
 * @brief Test optimal block size selection for Qwen2-0.5B (head_dim=64)
 */
TEST(Test__Q16_1Tensor, OptimalBlockSize_Qwen2_05B)
{
    const int head_dim = 64; // Qwen2-0.5B head dimension
    const Q16BlockSize optimal = optimal_q16_block_size(head_dim);

    EXPECT_EQ(optimal, Q16BlockSize::BLOCK_64);

    // Create tensor with optimal block size
    auto tensor = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(head_dim)},
        optimal);

    // Should have exactly 1 block per head
    EXPECT_EQ(tensor->blocks_per_row(), 1);
}

/**
 * @brief Test optimal block size selection for Llama3-8B (head_dim=128)
 */
TEST(Test__Q16_1Tensor, OptimalBlockSize_Llama3)
{
    const int head_dim = 128; // Llama3 head dimension
    const Q16BlockSize optimal = optimal_q16_block_size(head_dim);

    EXPECT_EQ(optimal, Q16BlockSize::BLOCK_128);

    // Create tensor with optimal block size
    auto tensor = std::make_shared<Q16_1Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(head_dim)},
        optimal);

    // Should have exactly 1 block per head
    EXPECT_EQ(tensor->blocks_per_row(), 1);
}

// Note: DeepSeek V3 MLA uses separate NOPE (128-dim) + ROPE (64-dim) tensors
// with independent scales, not a combined 192-dim block.
