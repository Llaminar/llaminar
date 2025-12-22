/**
 * @file Test__ResidualOpTyped.cpp
 * @brief Unit tests for ResidualOpTyped<P> typed residual operations
 * @author David Sanftenberg
 *
 * Tests the typed residual connection operations across precision modes:
 * - FP32: Standard float addition
 * - BF16: BF16 addition (dequant-add-requant)
 * - FP16: FP16 addition (dequant-add-requant)
 * - Q8_1: Native quantized addition via SIMD
 *
 * Key tests:
 * 1. Basic FP32 residual addition
 * 2. Q8_1 residual addition parity vs FP32 reference
 * 3. Batched residual with padding mask
 * 4. Pipeline-realistic dimensions (Qwen 2.5 0.5B)
 * 5. Multi-row residual with varying sequence lengths
 */

#include <gtest/gtest.h>

#include "../../../../src/v2/pipelines/ops/ResidualOp.h"
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/SIMDHelpers.h"
#include "../../../../src/v2/pipelines/RuntimeConfig.h"

#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ResidualOpTyped : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed RNG for reproducibility
        gen_.seed(42);
    }

    // Generate random FP32 data in range [-range, range]
    std::vector<float> generateRandomFP32(size_t count, float range = 1.0f)
    {
        std::vector<float> data(count);
        std::uniform_real_distribution<float> dist(-range, range);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen_);
        }
        return data;
    }

    // Compute FP32 reference: output = residual + projection
    std::vector<float> computeFP32Reference(
        const std::vector<float> &residual,
        const std::vector<float> &projection)
    {
        std::vector<float> output(residual.size());
        for (size_t i = 0; i < residual.size(); ++i)
        {
            output[i] = residual[i] + projection[i];
        }
        return output;
    }

    // Compute cosine similarity between two vectors
    float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0f;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12));
    }

    // Compute relative L2 error
    float relativeL2Error(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
        }
        return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
    }

    std::mt19937 gen_;
};

// =============================================================================
// FP32 Residual Tests
// =============================================================================

/**
 * @brief Test FP32 residual addition - basic correctness
 */
TEST_F(Test__ResidualOpTyped, FP32_BasicResidual)
{
    const int seq_len = 4;
    const int d_model = 64;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    // Generate random FP32 data
    auto residual_data = generateRandomFP32(elements);
    auto projection_data = generateRandomFP32(elements);
    auto expected = computeFP32Reference(residual_data, projection_data);

    // Create FP32 tensors (allocate then copy data)
    auto residual = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto projection = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    std::copy(residual_data.begin(), residual_data.end(), residual->mutable_data());
    std::copy(projection_data.begin(), projection_data.end(), projection->mutable_data());

    // Create FP32 residual op
    ResidualOpTyped<ActivationPrecision::FP32> op;

    // Apply residual
    ASSERT_TRUE(op.apply(residual.get(), projection.get(), output.get(), seq_len, d_model));

    // Verify output
    const float *out_data = output->data();
    for (size_t i = 0; i < elements; ++i)
    {
        EXPECT_FLOAT_EQ(out_data[i], expected[i])
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test FP32 residual with Qwen 2.5 0.5B dimensions (d_model=896)
 */
TEST_F(Test__ResidualOpTyped, FP32_QwenDimensions)
{
    const int seq_len = 9; // Same as E2E tests
    const int d_model = 896;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    auto residual_data = generateRandomFP32(elements);
    auto projection_data = generateRandomFP32(elements);
    auto expected = computeFP32Reference(residual_data, projection_data);

    auto residual = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto projection = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    std::copy(residual_data.begin(), residual_data.end(), residual->mutable_data());
    std::copy(projection_data.begin(), projection_data.end(), projection->mutable_data());

    ResidualOpTyped<ActivationPrecision::FP32> op;
    ASSERT_TRUE(op.apply(residual.get(), projection.get(), output.get(), seq_len, d_model));

    const float *out_data = output->data();
    float max_error = 0.0f;
    for (size_t i = 0; i < elements; ++i)
    {
        float error = std::abs(out_data[i] - expected[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[FP32 Qwen Dims] Max error: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-6f) << "FP32 should be exact";
}

// =============================================================================
// Q8_1 Residual Tests
// =============================================================================

/**
 * @brief Test Q8_1 residual addition parity vs FP32 reference
 *
 * This is the key test: verify that Q8_1 residual (dequant-add-requant)
 * produces output with high cosine similarity to FP32 reference.
 */
TEST_F(Test__ResidualOpTyped, Q8_1_ParityVsFP32)
{
    const int seq_len = 4;
    const int d_model = 64; // Must be multiple of 32 for Q8_1
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    // Generate random FP32 data
    auto residual_fp32 = generateRandomFP32(elements);
    auto projection_fp32 = generateRandomFP32(elements);
    auto expected = computeFP32Reference(residual_fp32, projection_fp32);

    // Quantize to Q8_1
    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto projection_q8 = Q8_1Tensor::quantize_from_fp32(
        projection_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    ASSERT_NE(residual_q8, nullptr);
    ASSERT_NE(projection_q8, nullptr);

    // Create output Q8_1 tensor
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        output_raw);

    // Create Q8_1 residual op
    ResidualOpTyped<ActivationPrecision::Q8_1> op;

    // Apply residual
    ASSERT_TRUE(op.apply(residual_q8.get(), projection_q8.get(), output_q8.get(), seq_len, d_model));

    // Dequantize output and compare
    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Q8_1 Parity] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 Parity] Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    // Q8_1 residual should have very high parity
    // 3 quantization ops: residual quant + projection quant + output requant
    EXPECT_GT(cosine, 0.999f) << "Q8_1 residual should have >0.999 cosine vs FP32";
    EXPECT_LT(rel_error, 0.05f) << "Q8_1 residual should have <5% relative L2 error";
}

/**
 * @brief Test Q8_1 residual with Qwen 2.5 0.5B dimensions (d_model=896)
 */
TEST_F(Test__ResidualOpTyped, Q8_1_QwenDimensions)
{
    const int seq_len = 9;   // Same as E2E tests
    const int d_model = 896; // Qwen 2.5 0.5B
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    // Generate random FP32 data (typical activation range)
    auto residual_fp32 = generateRandomFP32(elements, 2.0f);
    auto projection_fp32 = generateRandomFP32(elements, 2.0f);
    auto expected = computeFP32Reference(residual_fp32, projection_fp32);

    // Quantize to Q8_1
    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto projection_q8 = Q8_1Tensor::quantize_from_fp32(
        projection_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        output_raw);

    ResidualOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.apply(residual_q8.get(), projection_q8.get(), output_q8.get(), seq_len, d_model));

    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Q8_1 Qwen Dims] seq_len=" << seq_len << ", d_model=" << d_model << std::endl;
    std::cout << "[Q8_1 Qwen Dims] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 Qwen Dims] Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Q8_1 residual should have >0.999 cosine vs FP32";
    EXPECT_LT(rel_error, 0.05f) << "Q8_1 residual should have <5% relative L2 error";
}

/**
 * @brief Test Q8_1 residual with single-token decode (seq_len=1)
 *
 * This is the most common case during autoregressive decode.
 */
TEST_F(Test__ResidualOpTyped, Q8_1_SingleTokenDecode)
{
    const int seq_len = 1;
    const int d_model = 896;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    auto residual_fp32 = generateRandomFP32(elements, 2.0f);
    auto projection_fp32 = generateRandomFP32(elements, 2.0f);
    auto expected = computeFP32Reference(residual_fp32, projection_fp32);

    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto projection_q8 = Q8_1Tensor::quantize_from_fp32(
        projection_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        output_raw);

    ResidualOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.apply(residual_q8.get(), projection_q8.get(), output_q8.get(), seq_len, d_model));

    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);

    std::cout << "[Q8_1 Single Token] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Single-token Q8_1 residual should have >0.999 cosine";
}

/**
 * @brief Test Q8_1 residual with all-zeros input
 *
 * Edge case: ensure zero handling is correct.
 */
TEST_F(Test__ResidualOpTyped, Q8_1_ZeroInput)
{
    const int seq_len = 2;
    const int d_model = 64;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    // All zeros
    std::vector<float> zeros(elements, 0.0f);
    auto residual_fp32 = generateRandomFP32(elements);

    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto zeros_q8 = Q8_1Tensor::quantize_from_fp32(
        zeros.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        output_raw);

    ResidualOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.apply(residual_q8.get(), zeros_q8.get(), output_q8.get(), seq_len, d_model));

    // Output should be approximately equal to residual (adding zero)
    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    // Dequantize residual for comparison
    std::vector<float> residual_dequant(elements);
    residual_q8->to_fp32(residual_dequant.data());

    float cosine = cosineSimilarity(actual, residual_dequant);
    std::cout << "[Q8_1 Zero Input] Cosine vs residual: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    // Adding zero should preserve residual (within quantization error)
    EXPECT_GT(cosine, 0.999f) << "Adding zero should preserve residual";
}

// =============================================================================
// Batched Residual Tests
// =============================================================================

/**
 * @brief Test FP32 batched residual with padding mask
 */
TEST_F(Test__ResidualOpTyped, FP32_BatchedWithPadding)
{
    const int batch_size = 2;
    const int max_seq_len = 8;
    const int d_model = 64;
    std::vector<int> seq_lengths = {6, 4}; // First sequence 6 tokens, second 4 tokens

    const size_t elements = static_cast<size_t>(batch_size) * max_seq_len * d_model;

    auto residual_data = generateRandomFP32(elements);
    auto projection_data = generateRandomFP32(elements);

    auto residual = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(max_seq_len), static_cast<size_t>(d_model)});
    auto projection = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(max_seq_len), static_cast<size_t>(d_model)});
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size), static_cast<size_t>(max_seq_len), static_cast<size_t>(d_model)});

    std::copy(residual_data.begin(), residual_data.end(), residual->mutable_data());
    std::copy(projection_data.begin(), projection_data.end(), projection->mutable_data());

    ResidualOpTyped<ActivationPrecision::FP32> op;
    ASSERT_TRUE(op.batched(
        residual.get(), projection.get(), output.get(),
        batch_size, seq_lengths.data(), max_seq_len, d_model));

    const float *out_data = output->data();

    // Verify valid positions have correct values
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < max_seq_len; ++s)
        {
            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;

            if (s < seq_lengths[b])
            {
                // Valid position: should be residual + projection
                for (int d = 0; d < d_model; ++d)
                {
                    float expected = residual_data[row_offset + d] + projection_data[row_offset + d];
                    EXPECT_FLOAT_EQ(out_data[row_offset + d], expected)
                        << "Mismatch at batch=" << b << ", seq=" << s << ", dim=" << d;
                }
            }
            else
            {
                // Padding position: should be zero
                for (int d = 0; d < d_model; ++d)
                {
                    EXPECT_FLOAT_EQ(out_data[row_offset + d], 0.0f)
                        << "Padding not zeroed at batch=" << b << ", seq=" << s << ", dim=" << d;
                }
            }
        }
    }
}

/**
 * @brief Test Q8_1 batched residual with padding mask
 *
 * Note: Q8_1Tensor uses 2D shape (rows x cols) internally.
 * For batched operations, we flatten batch*seq_len into rows.
 */
TEST_F(Test__ResidualOpTyped, Q8_1_BatchedWithPadding)
{
    const int batch_size = 2;
    const int max_seq_len = 8;
    const int d_model = 64; // Must be multiple of 32
    std::vector<int> seq_lengths = {6, 4};

    // Q8_1 uses 2D layout: (batch*seq_len) x d_model
    const size_t total_rows = static_cast<size_t>(batch_size) * max_seq_len;
    const size_t elements = total_rows * d_model;

    auto residual_fp32 = generateRandomFP32(elements);
    auto projection_fp32 = generateRandomFP32(elements);

    // Quantize to Q8_1 with 2D shape
    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {total_rows, static_cast<size_t>(d_model)});
    auto projection_q8 = Q8_1Tensor::quantize_from_fp32(
        projection_fp32.data(),
        {total_rows, static_cast<size_t>(d_model)});

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{total_rows, static_cast<size_t>(d_model)},
        output_raw);

    ResidualOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.batched(
        residual_q8.get(), projection_q8.get(), output_q8.get(),
        batch_size, seq_lengths.data(), max_seq_len, d_model));

    // Dequantize output
    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    // Compute FP32 reference for valid positions
    std::vector<float> expected(elements, 0.0f);
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_lengths[b]; ++s)
        { // Only valid positions
            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;
            for (int d = 0; d < d_model; ++d)
            {
                expected[row_offset + d] = residual_fp32[row_offset + d] + projection_fp32[row_offset + d];
            }
        }
    }

    // Compute cosine for valid positions only
    std::vector<float> actual_valid, expected_valid;
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_lengths[b]; ++s)
        {
            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;
            for (int d = 0; d < d_model; ++d)
            {
                actual_valid.push_back(actual[row_offset + d]);
                expected_valid.push_back(expected[row_offset + d]);
            }
        }
    }

    float cosine = cosineSimilarity(actual_valid, expected_valid);
    std::cout << "[Q8_1 Batched] Cosine (valid positions): " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Q8_1 batched residual should have >0.999 cosine";

    // Verify padding positions are zeroed
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = seq_lengths[b]; s < max_seq_len; ++s)
        {
            const size_t row_offset = (static_cast<size_t>(b) * max_seq_len + s) * d_model;
            for (int d = 0; d < d_model; ++d)
            {
                EXPECT_NEAR(actual[row_offset + d], 0.0f, 1e-6f)
                    << "Padding not zeroed at batch=" << b << ", seq=" << s << ", dim=" << d;
            }
        }
    }
}

// =============================================================================
// Factory Function Tests
// =============================================================================

/**
 * @brief Test createResidualOp factory function
 */
TEST_F(Test__ResidualOpTyped, Factory_CreateAllPrecisions)
{
    // FP32
    auto fp32_op = createResidualOp(ActivationPrecision::FP32);
    ASSERT_NE(fp32_op, nullptr);
    EXPECT_EQ(fp32_op->precision(), ActivationPrecision::FP32);

    // BF16
    auto bf16_op = createResidualOp(ActivationPrecision::BF16);
    ASSERT_NE(bf16_op, nullptr);
    EXPECT_EQ(bf16_op->precision(), ActivationPrecision::BF16);

    // FP16
    auto fp16_op = createResidualOp(ActivationPrecision::FP16);
    ASSERT_NE(fp16_op, nullptr);
    EXPECT_EQ(fp16_op->precision(), ActivationPrecision::FP16);

    // Q8_1
    auto q8_1_op = createResidualOp(ActivationPrecision::Q8_1);
    ASSERT_NE(q8_1_op, nullptr);
    EXPECT_EQ(q8_1_op->precision(), ActivationPrecision::Q8_1);
}

/**
 * @brief Test factory-created Q8_1 op produces same results as direct instantiation
 */
TEST_F(Test__ResidualOpTyped, Factory_Q8_1_Consistency)
{
    const int seq_len = 4;
    const int d_model = 64;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    auto residual_fp32 = generateRandomFP32(elements);
    auto projection_fp32 = generateRandomFP32(elements);

    // Create Q8_1 tensors
    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto projection_q8 = Q8_1Tensor::quantize_from_fp32(
        projection_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    // Create two output tensors
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output1_raw(n_blocks * sizeof(Q8_1Block), 0);
    std::vector<uint8_t> output2_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output1 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        output1_raw);
    auto output2 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
        output2_raw);

    // Direct instantiation
    ResidualOpTyped<ActivationPrecision::Q8_1> direct_op;
    ASSERT_TRUE(direct_op.apply(residual_q8.get(), projection_q8.get(), output1.get(), seq_len, d_model));

    // Factory creation
    auto factory_op = createResidualOp(ActivationPrecision::Q8_1);
    ASSERT_TRUE(factory_op->apply(residual_q8.get(), projection_q8.get(), output2.get(), seq_len, d_model));

    // Compare outputs
    std::vector<float> result1(elements), result2(elements);
    output1->to_fp32(result1.data());
    output2->to_fp32(result2.data());

    float cosine = cosineSimilarity(result1, result2);
    std::cout << "[Factory Consistency] Direct vs Factory cosine: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_FLOAT_EQ(cosine, 1.0f) << "Factory and direct should produce identical results";
}

// =============================================================================
// Error Accumulation Tests
// =============================================================================

/**
 * @brief Test Q8_1 residual error doesn't accumulate excessively over multiple ops
 *
 * In a real pipeline, residual connections are applied many times.
 * This tests whether error accumulates reasonably.
 */
TEST_F(Test__ResidualOpTyped, Q8_1_ErrorAccumulationMultipleOps)
{
    const int seq_len = 4;
    const int d_model = 64;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;
    const int num_iterations = 24; // ~24 layers in Qwen 0.5B

    // Start with random FP32 data
    auto current_fp32 = generateRandomFP32(elements);
    auto current_q8 = Q8_1Tensor::quantize_from_fp32(
        current_fp32.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    ResidualOpTyped<ActivationPrecision::Q8_1> op;

    // Track FP32 reference through iterations
    std::vector<float> reference = current_fp32;

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        // Generate small random projection (typical of residual updates)
        auto projection_fp32 = generateRandomFP32(elements, 0.5f);
        auto projection_q8 = Q8_1Tensor::quantize_from_fp32(
            projection_fp32.data(),
            {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

        // Create new output tensor
        const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
        std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
        auto output_q8 = std::make_shared<Q8_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            output_raw);

        // Apply Q8_1 residual
        ASSERT_TRUE(op.apply(current_q8.get(), projection_q8.get(), output_q8.get(), seq_len, d_model));

        // Update FP32 reference
        for (size_t i = 0; i < elements; ++i)
        {
            reference[i] += projection_fp32[i];
        }

        // Move to next iteration
        current_q8 = output_q8;
    }

    // Final comparison
    std::vector<float> actual(elements);
    current_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, reference);
    float rel_error = relativeL2Error(actual, reference);

    std::cout << "[Q8_1 Accumulation] After " << num_iterations << " iterations:" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "  Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    // After 24 iterations, we expect some degradation but should still be reasonable
    EXPECT_GT(cosine, 0.95f) << "Error accumulated too much over " << num_iterations << " iterations";
    EXPECT_LT(rel_error, 0.20f) << "Relative error accumulated too much over " << num_iterations << " iterations";
}

// =============================================================================
// Type Safety Tests
// =============================================================================

/**
 * @brief Test that Q8_1 op rejects FP32 tensors
 */
TEST_F(Test__ResidualOpTyped, Q8_1_RejectsWrongTensorType)
{
    const int seq_len = 4;
    const int d_model = 64;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    auto fp32_data = generateRandomFP32(elements);
    auto fp32_tensor = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    std::copy(fp32_data.begin(), fp32_data.end(), fp32_tensor->mutable_data());

    ResidualOpTyped<ActivationPrecision::Q8_1> op;

    // Should return false because we're passing FP32 tensors to Q8_1 op
    EXPECT_FALSE(op.apply(fp32_tensor.get(), fp32_tensor.get(), fp32_tensor.get(), seq_len, d_model));
}

/**
 * @brief Test that FP32 op rejects Q8_1 tensors
 */
TEST_F(Test__ResidualOpTyped, FP32_RejectsWrongTensorType)
{
    const int seq_len = 4;
    const int d_model = 64;
    const size_t elements = static_cast<size_t>(seq_len) * d_model;

    auto fp32_data = generateRandomFP32(elements);
    auto q8_tensor = Q8_1Tensor::quantize_from_fp32(
        fp32_data.data(),
        {static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    ResidualOpTyped<ActivationPrecision::FP32> op;

    // Should return false because we're passing Q8_1 tensors to FP32 op
    EXPECT_FALSE(op.apply(q8_tensor.get(), q8_tensor.get(), q8_tensor.get(), seq_len, d_model));
}

// =============================================================================
// Pipeline Integration Tests
// =============================================================================

/**
 * @brief Test Q8_1 residual as used in Qwen2Pipeline attention block
 *
 * This test simulates exactly what happens in:
 *   TRY_OP(add_residual(buffers.residual, buffers.attn_proj, current_hidden_,
 *                       batch_size_, padded_seq_len_, d_model_,
 *                       sequence_lengths_, layer_prefix + "_ATTENTION_RESIDUAL"));
 *
 * Key aspects:
 * - Uses createResidualOp() factory (same as PipelineBase::initializeTypedOps)
 * - Uses batched() interface with sequence_lengths array
 * - Tests with Qwen 0.5B dimensions (d_model=896, seq_len=9)
 */
TEST_F(Test__ResidualOpTyped, Q8_1_PipelineIntegration_AttentionResidual)
{
    // Qwen 2.5 0.5B dimensions (same as E2E tests)
    const int batch_size = 1;
    const int padded_seq_len = 9;
    const int d_model = 896;
    std::vector<int> sequence_lengths = {9}; // Single sequence, no padding

    const size_t total_rows = static_cast<size_t>(batch_size) * padded_seq_len;
    const size_t elements = total_rows * d_model;

    // Simulate buffers.residual (saved from input_hidden before RMSNorm)
    auto residual_fp32 = generateRandomFP32(elements, 2.0f);
    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {total_rows, static_cast<size_t>(d_model)});

    // Simulate buffers.attn_proj (output of Wo projection)
    auto attn_proj_fp32 = generateRandomFP32(elements, 1.0f);
    auto attn_proj_q8 = Q8_1Tensor::quantize_from_fp32(
        attn_proj_fp32.data(),
        {total_rows, static_cast<size_t>(d_model)});

    // Simulate current_hidden_ (output buffer)
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto current_hidden_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{total_rows, static_cast<size_t>(d_model)},
        output_raw);

    // Create typed residual op exactly as pipeline does
    auto typed_residual_op = createResidualOp(ActivationPrecision::Q8_1);
    ASSERT_NE(typed_residual_op, nullptr);
    EXPECT_EQ(typed_residual_op->precision(), ActivationPrecision::Q8_1);

    // Call batched() exactly as pipeline does (const_cast pattern matches pipeline code)
    ASSERT_TRUE(typed_residual_op->batched(
        const_cast<TensorBase *>(static_cast<TensorBase *>(residual_q8.get())),
        const_cast<TensorBase *>(static_cast<TensorBase *>(attn_proj_q8.get())),
        current_hidden_q8.get(),
        batch_size, sequence_lengths.data(), padded_seq_len, d_model));

    // Compute FP32 reference
    std::vector<float> expected(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        expected[i] = residual_fp32[i] + attn_proj_fp32[i];
    }

    // Dequantize output and compare
    std::vector<float> actual(elements);
    current_hidden_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Pipeline Integration] Attention residual:" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "  Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Q8_1 attention residual should have >0.999 cosine vs FP32";
    EXPECT_LT(rel_error, 0.05f) << "Q8_1 attention residual should have <5% relative L2 error";
}

/**
 * @brief Test Q8_1 residual with the exact call pattern from add_residual()
 *
 * This test verifies that:
 * 1. const_cast pattern used in add_residual() doesn't break anything
 * 2. sequence_lengths array is handled correctly
 * 3. padded_seq_len vs actual lengths work
 */
TEST_F(Test__ResidualOpTyped, Q8_1_AddResidualCallPattern)
{
    // Same as pipeline: batch_size=1, padded but sequence_lengths < padded
    const int batch_size = 1;
    const int padded_seq_len = 16; // Padded to power of 2
    const int actual_seq_len = 9;  // Actual tokens
    const int d_model = 896;
    std::vector<int> sequence_lengths = {actual_seq_len};

    const size_t total_rows = static_cast<size_t>(batch_size) * padded_seq_len;
    const size_t elements = total_rows * d_model;

    // Create Q8_1 tensors
    auto residual_fp32 = generateRandomFP32(elements, 2.0f);
    auto attn_proj_fp32 = generateRandomFP32(elements, 1.0f);

    auto residual_q8 = Q8_1Tensor::quantize_from_fp32(
        residual_fp32.data(),
        {total_rows, static_cast<size_t>(d_model)});
    auto attn_proj_q8 = Q8_1Tensor::quantize_from_fp32(
        attn_proj_fp32.data(),
        {total_rows, static_cast<size_t>(d_model)});

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{total_rows, static_cast<size_t>(d_model)},
        output_raw);

    // Use factory to create op
    auto typed_residual_op = createResidualOp(ActivationPrecision::Q8_1);

    // Call with const TensorBase* inputs (like add_residual receives)
    const TensorBase *const_residual = residual_q8.get();
    const TensorBase *const_input = attn_proj_q8.get();

    // Exactly mimic add_residual() call pattern
    ASSERT_TRUE(typed_residual_op->batched(
        const_cast<TensorBase *>(const_residual),
        const_cast<TensorBase *>(const_input),
        output_q8.get(),
        batch_size, sequence_lengths.data(), padded_seq_len, d_model));

    // Compute FP32 reference for VALID positions only
    std::vector<float> expected_valid;
    for (int s = 0; s < actual_seq_len; ++s)
    {
        for (int d = 0; d < d_model; ++d)
        {
            size_t idx = static_cast<size_t>(s) * d_model + d;
            expected_valid.push_back(residual_fp32[idx] + attn_proj_fp32[idx]);
        }
    }

    // Dequantize output and extract valid positions
    std::vector<float> actual_all(elements);
    output_q8->to_fp32(actual_all.data());

    std::vector<float> actual_valid;
    for (int s = 0; s < actual_seq_len; ++s)
    {
        for (int d = 0; d < d_model; ++d)
        {
            size_t idx = static_cast<size_t>(s) * d_model + d;
            actual_valid.push_back(actual_all[idx]);
        }
    }

    float cosine = cosineSimilarity(actual_valid, expected_valid);

    std::cout << "[AddResidual Pattern] With padding (9 valid / 16 padded):" << std::endl;
    std::cout << "  Cosine similarity (valid positions): " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Valid positions should have >0.999 cosine";

    // Verify padding positions are zeroed
    int padding_zeros = 0;
    int padding_nonzeros = 0;
    for (int s = actual_seq_len; s < padded_seq_len; ++s)
    {
        for (int d = 0; d < d_model; ++d)
        {
            size_t idx = static_cast<size_t>(s) * d_model + d;
            if (std::abs(actual_all[idx]) < 1e-6f)
            {
                ++padding_zeros;
            }
            else
            {
                ++padding_nonzeros;
            }
        }
    }

    int expected_padding_elements = (padded_seq_len - actual_seq_len) * d_model;
    std::cout << "  Padding zeros: " << padding_zeros << " / " << expected_padding_elements << std::endl;
    EXPECT_EQ(padding_zeros, expected_padding_elements) << "All padding positions should be zeroed";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
