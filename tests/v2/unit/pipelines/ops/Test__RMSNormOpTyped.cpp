/**
 * @file Test__RMSNormOpTyped.cpp
 * @brief Unit tests for RMSNormOpTyped<P> typed RMSNorm operations
 * @author David Sanftenberg
 *
 * Tests the typed RMSNorm operations across precision modes:
 * - FP32: Standard float normalization
 * - Q8_1: Native quantized RMSNorm (dequant-norm-requant)
 *
 * Key tests:
 * 1. Basic FP32 RMSNorm correctness
 * 2. Q8_1 RMSNorm parity vs FP32 reference
 * 3. Qwen 2.5 0.5B dimensions (d_model=896)
 * 4. Pipeline integration pattern (how rms_norm() is called)
 * 5. Multi-row batched normalization
 */

#include <gtest/gtest.h>

#include "../../../../src/v2/pipelines/ops/RMSNormOp.h"
#include "../../../../src/v2/tensors/Tensors.h"
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

class Test__RMSNormOpTyped : public ::testing::Test
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

    // Generate gamma weights (typically close to 1.0 with small variance)
    std::vector<float> generateGammaWeights(size_t count)
    {
        std::vector<float> data(count);
        std::normal_distribution<float> dist(1.0f, 0.1f);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen_);
        }
        return data;
    }

    // Compute FP32 reference RMSNorm
    std::vector<float> computeFP32Reference(
        const std::vector<float> &input,
        const std::vector<float> &gamma,
        int rows, int cols, float eps)
    {
        std::vector<float> output(input.size());

        for (int row = 0; row < rows; ++row)
        {
            const float *in_row = input.data() + static_cast<size_t>(row) * cols;
            float *out_row = output.data() + static_cast<size_t>(row) * cols;

            // Compute RMS
            double sum_sq = 0.0;
            for (int c = 0; c < cols; ++c)
            {
                sum_sq += static_cast<double>(in_row[c]) * in_row[c];
            }
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + eps);
            float inv_rms = 1.0f / rms;

            // Normalize and scale by gamma
            for (int c = 0; c < cols; ++c)
            {
                out_row[c] = in_row[c] * inv_rms * gamma[c];
            }
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
    static constexpr float DEFAULT_EPS = 1e-6f;
};

// =============================================================================
// FP32 RMSNorm Tests
// =============================================================================

/**
 * @brief Test FP32 RMSNorm - basic correctness
 */
TEST_F(Test__RMSNormOpTyped, FP32_BasicCorrectness)
{
    const int rows = 4;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random input and gamma
    auto input_data = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);
    auto expected = computeFP32Reference(input_data, gamma_data, rows, cols, DEFAULT_EPS);

    // Create FP32 tensors
    auto input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    std::copy(input_data.begin(), input_data.end(), input->mutable_data());
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    // Create FP32 RMSNorm op
    RMSNormOpTyped<ActivationPrecision::FP32> op;

    // Execute
    ASSERT_TRUE(op.execute(input.get(), gamma.get(), output.get(), rows, cols, DEFAULT_EPS));

    // Verify output
    const float *out_data = output->data();
    float max_error = 0.0f;
    for (size_t i = 0; i < elements; ++i)
    {
        float error = std::abs(out_data[i] - expected[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[FP32 Basic] Max error: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-5f) << "FP32 RMSNorm should be very close to reference";
}

/**
 * @brief Test FP32 RMSNorm with Qwen 2.5 0.5B dimensions (d_model=896)
 */
TEST_F(Test__RMSNormOpTyped, FP32_QwenDimensions)
{
    const int rows = 9;   // Same as E2E tests
    const int cols = 896; // Qwen 2.5 0.5B d_model
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_data = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);
    auto expected = computeFP32Reference(input_data, gamma_data, rows, cols, DEFAULT_EPS);

    auto input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    std::copy(input_data.begin(), input_data.end(), input->mutable_data());
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    RMSNormOpTyped<ActivationPrecision::FP32> op;
    ASSERT_TRUE(op.execute(input.get(), gamma.get(), output.get(), rows, cols, DEFAULT_EPS));

    const float *out_data = output->data();
    float max_error = 0.0f;
    for (size_t i = 0; i < elements; ++i)
    {
        float error = std::abs(out_data[i] - expected[i]);
        max_error = std::max(max_error, error);
    }

    std::cout << "[FP32 Qwen Dims] Max error: " << max_error << std::endl;
    EXPECT_LT(max_error, 1e-5f) << "FP32 RMSNorm should be very close to reference";
}

// =============================================================================
// Q8_1 RMSNorm Tests
// =============================================================================

/**
 * @brief Test Q8_1 RMSNorm parity vs FP32 reference
 *
 * This is the key test: verify that Q8_1 RMSNorm (dequant-norm-requant)
 * produces output with high cosine similarity to FP32 reference.
 */
TEST_F(Test__RMSNormOpTyped, Q8_1_ParityVsFP32)
{
    const int rows = 4;
    const int cols = 64; // Must be multiple of 32 for Q8_1
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random FP32 data
    auto input_fp32 = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);
    auto expected = computeFP32Reference(input_fp32, gamma_data, rows, cols, DEFAULT_EPS);

    // Quantize input to Q8_1
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});
    ASSERT_NE(input_q8, nullptr);

    // Create gamma tensor (FP32 - gamma is typically not quantized)
    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    // Create output Q8_1 tensor
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output_raw);

    // Create Q8_1 RMSNorm op
    RMSNormOpTyped<ActivationPrecision::Q8_1> op;

    // Execute
    ASSERT_TRUE(op.execute(input_q8.get(), gamma.get(), output_q8.get(), rows, cols, DEFAULT_EPS));

    // Dequantize output and compare
    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Q8_1 Parity] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 Parity] Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    // Q8_1 RMSNorm involves:
    // 1. Dequant input (Q8_1 -> FP32)
    // 2. Compute RMS and normalize (FP32)
    // 3. Requant output (FP32 -> Q8_1)
    // We expect some quantization error but should be very high correlation
    EXPECT_GT(cosine, 0.999f) << "Q8_1 RMSNorm should have >0.999 cosine vs FP32";
    EXPECT_LT(rel_error, 0.05f) << "Q8_1 RMSNorm should have <5% relative L2 error";
}

/**
 * @brief Test Q8_1 RMSNorm with Qwen 2.5 0.5B dimensions (d_model=896)
 */
TEST_F(Test__RMSNormOpTyped, Q8_1_QwenDimensions)
{
    const int rows = 9;   // Same as E2E tests
    const int cols = 896; // Qwen 2.5 0.5B d_model
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_fp32 = generateRandomFP32(elements, 2.0f); // Typical activation range
    auto gamma_data = generateGammaWeights(cols);
    auto expected = computeFP32Reference(input_fp32, gamma_data, rows, cols, DEFAULT_EPS);

    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output_raw);

    RMSNormOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.execute(input_q8.get(), gamma.get(), output_q8.get(), rows, cols, DEFAULT_EPS));

    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Q8_1 Qwen Dims] rows=" << rows << ", cols=" << cols << std::endl;
    std::cout << "[Q8_1 Qwen Dims] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Q8_1 Qwen Dims] Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Q8_1 RMSNorm should have >0.999 cosine vs FP32";
    EXPECT_LT(rel_error, 0.05f) << "Q8_1 RMSNorm should have <5% relative L2 error";
}

/**
 * @brief Test Q8_1 RMSNorm with single row (decode case)
 */
TEST_F(Test__RMSNormOpTyped, Q8_1_SingleRow)
{
    const int rows = 1;
    const int cols = 896;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_fp32 = generateRandomFP32(elements, 2.0f);
    auto gamma_data = generateGammaWeights(cols);
    auto expected = computeFP32Reference(input_fp32, gamma_data, rows, cols, DEFAULT_EPS);

    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output_raw);

    RMSNormOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.execute(input_q8.get(), gamma.get(), output_q8.get(), rows, cols, DEFAULT_EPS));

    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);

    std::cout << "[Q8_1 Single Row] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Single-row Q8_1 RMSNorm should have >0.999 cosine";
}

/**
 * @brief Test Q8_1 RMSNorm with all-ones gamma (identity scale)
 */
TEST_F(Test__RMSNormOpTyped, Q8_1_UnityGamma)
{
    const int rows = 4;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_fp32 = generateRandomFP32(elements);
    std::vector<float> gamma_ones(cols, 1.0f); // All ones
    auto expected = computeFP32Reference(input_fp32, gamma_ones, rows, cols, DEFAULT_EPS);

    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::fill(gamma->mutable_data(), gamma->mutable_data() + cols, 1.0f);

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output_raw);

    RMSNormOpTyped<ActivationPrecision::Q8_1> op;
    ASSERT_TRUE(op.execute(input_q8.get(), gamma.get(), output_q8.get(), rows, cols, DEFAULT_EPS));

    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);

    std::cout << "[Q8_1 Unity Gamma] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Unity gamma Q8_1 RMSNorm should have >0.999 cosine";
}

// =============================================================================
// Factory Function Tests
// =============================================================================

/**
 * @brief Test createRMSNormOp factory function
 */
TEST_F(Test__RMSNormOpTyped, Factory_CreateAllPrecisions)
{
    // FP32
    auto fp32_op = createRMSNormOp(ActivationPrecision::FP32);
    ASSERT_NE(fp32_op, nullptr);
    EXPECT_EQ(fp32_op->precision(), ActivationPrecision::FP32);

    // BF16
    auto bf16_op = createRMSNormOp(ActivationPrecision::BF16);
    ASSERT_NE(bf16_op, nullptr);
    EXPECT_EQ(bf16_op->precision(), ActivationPrecision::BF16);

    // FP16
    auto fp16_op = createRMSNormOp(ActivationPrecision::FP16);
    ASSERT_NE(fp16_op, nullptr);
    EXPECT_EQ(fp16_op->precision(), ActivationPrecision::FP16);

    // Q8_1
    auto q8_1_op = createRMSNormOp(ActivationPrecision::Q8_1);
    ASSERT_NE(q8_1_op, nullptr);
    EXPECT_EQ(q8_1_op->precision(), ActivationPrecision::Q8_1);
}

/**
 * @brief Test factory-created Q8_1 op produces same results as direct instantiation
 */
TEST_F(Test__RMSNormOpTyped, Factory_Q8_1_Consistency)
{
    const int rows = 4;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_fp32 = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);

    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    // Create two output tensors
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output1_raw(n_blocks * sizeof(Q8_1Block), 0);
    std::vector<uint8_t> output2_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output1 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output1_raw);
    auto output2 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output2_raw);

    // Direct instantiation
    RMSNormOpTyped<ActivationPrecision::Q8_1> direct_op;
    ASSERT_TRUE(direct_op.execute(input_q8.get(), gamma.get(), output1.get(), rows, cols, DEFAULT_EPS));

    // Factory creation
    auto factory_op = createRMSNormOp(ActivationPrecision::Q8_1);
    ASSERT_TRUE(factory_op->execute(input_q8.get(), gamma.get(), output2.get(), rows, cols, DEFAULT_EPS));

    // Compare outputs
    std::vector<float> result1(elements), result2(elements);
    output1->to_fp32(result1.data());
    output2->to_fp32(result2.data());

    float cosine = cosineSimilarity(result1, result2);
    std::cout << "[Factory Consistency] Direct vs Factory cosine: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_FLOAT_EQ(cosine, 1.0f) << "Factory and direct should produce identical results";
}

// =============================================================================
// Factory Function Tests
// =============================================================================

/**
 * @brief Test createRMSNormOp() factory with Q8_1 tensors
 *
 * This tests that the factory-created op correctly handles Q8_1 tensors.
 * This is how the pipeline calls it via typed_rmsnorm_op_.
 */
TEST_F(Test__RMSNormOpTyped, Factory_Q8_1_RuntimeDispatch)
{
    const int rows = 9;
    const int cols = 896;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_fp32 = generateRandomFP32(elements, 2.0f);
    auto gamma_data = generateGammaWeights(cols);
    auto expected = computeFP32Reference(input_fp32, gamma_data, rows, cols, DEFAULT_EPS);

    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto output_q8 = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        output_raw);

    // Use factory-created Q8_1 op
    auto factory_op = createRMSNormOp(ActivationPrecision::Q8_1);
    ASSERT_TRUE(factory_op->execute(input_q8.get(), gamma.get(), output_q8.get(), rows, cols, DEFAULT_EPS));

    std::vector<float> actual(elements);
    output_q8->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Factory Q8_1] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "[Factory Q8_1] Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Factory-created RMSNormOp Q8_1 path should work correctly";
}

/**
 * @brief Verify factory op produces same results as typed RMSNormOpTyped<Q8_1>
 */
TEST_F(Test__RMSNormOpTyped, Factory_MatchesTyped_Q8_1)
{
    const int rows = 4;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_fp32 = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);

    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    std::copy(gamma_data.begin(), gamma_data.end(), gamma->mutable_data());

    // Two outputs
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> typed_raw(n_blocks * sizeof(Q8_1Block), 0);
    std::vector<uint8_t> factory_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto typed_output = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        typed_raw);
    auto factory_output = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
        factory_raw);

    // Typed op (direct instantiation)
    RMSNormOpTyped<ActivationPrecision::Q8_1> typed_op;
    ASSERT_TRUE(typed_op.execute(input_q8.get(), gamma.get(), typed_output.get(), rows, cols, DEFAULT_EPS));

    // Factory op
    auto factory_op = createRMSNormOp(ActivationPrecision::Q8_1);
    ASSERT_TRUE(factory_op->execute(input_q8.get(), gamma.get(), factory_output.get(), rows, cols, DEFAULT_EPS));

    // Compare
    std::vector<float> typed_result(elements), factory_result(elements);
    typed_output->to_fp32(typed_result.data());
    factory_output->to_fp32(factory_result.data());

    float cosine = cosineSimilarity(typed_result, factory_result);
    std::cout << "[Typed vs Factory] Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;

    EXPECT_FLOAT_EQ(cosine, 1.0f) << "Typed and factory should produce identical results";
}

// =============================================================================
// Pipeline Integration Tests
// =============================================================================

/**
 * @brief Test Q8_1 RMSNorm as used in Qwen2Pipeline attention block
 *
 * This simulates exactly what happens before QKV projection:
 *   TRY_OP(rms_norm(input_hidden, layer.attn_norm, buffers.normalized,
 *                   effective_seq_len, d_model_, rms_norm_eps_,
 *                   layer_prefix + "_ATTENTION_NORM", attn_device));
 */
TEST_F(Test__RMSNormOpTyped, Q8_1_PipelineIntegration_AttentionNorm)
{
    // Qwen 2.5 0.5B dimensions
    const int effective_seq_len = 9;
    const int d_model = 896;
    const float rms_norm_eps = 1e-6f;
    const size_t elements = static_cast<size_t>(effective_seq_len) * d_model;

    // Simulate input_hidden (Q8_1 from embedding or previous layer)
    auto input_fp32 = generateRandomFP32(elements, 2.0f);
    auto input_q8 = Q8_1Tensor::quantize_from_fp32(
        input_fp32.data(),
        {static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model)});

    // Simulate layer.attn_norm (FP32 gamma weights from GGUF)
    auto gamma_data = generateGammaWeights(d_model);
    auto attn_norm = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(d_model)});
    std::copy(gamma_data.begin(), gamma_data.end(), attn_norm->mutable_data());

    // Simulate buffers.normalized (Q8_1 output)
    const size_t n_blocks = elements / Q8_1Block::BLOCK_SIZE;
    std::vector<uint8_t> output_raw(n_blocks * sizeof(Q8_1Block), 0);
    auto normalized = std::make_shared<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model)},
        output_raw);

    // Use factory-created Q8_1 op (same as pipeline's typed_rmsnorm_op_)
    auto rmsnorm_op = createRMSNormOp(ActivationPrecision::Q8_1);
    ASSERT_TRUE(rmsnorm_op->execute(
        input_q8.get(), attn_norm.get(), normalized.get(),
        effective_seq_len, d_model, rms_norm_eps));

    // Compute FP32 reference
    auto expected = computeFP32Reference(input_fp32, gamma_data, effective_seq_len, d_model, rms_norm_eps);

    // Dequantize and compare
    std::vector<float> actual(elements);
    normalized->to_fp32(actual.data());

    float cosine = cosineSimilarity(actual, expected);
    float rel_error = relativeL2Error(actual, expected);

    std::cout << "[Pipeline Integration] Pre-attention RMSNorm:" << std::endl;
    std::cout << "  Cosine similarity: " << std::fixed << std::setprecision(6) << cosine << std::endl;
    std::cout << "  Relative L2 error: " << (rel_error * 100.0f) << "%" << std::endl;

    EXPECT_GT(cosine, 0.999f) << "Pipeline Q8_1 RMSNorm should have >0.999 cosine vs FP32";
    EXPECT_LT(rel_error, 0.05f) << "Pipeline Q8_1 RMSNorm should have <5% relative L2 error";
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/**
 * @brief Test that Q8_1 op rejects FP32 tensors
 */
TEST_F(Test__RMSNormOpTyped, Q8_1_RejectsWrongTensorType)
{
    const int rows = 4;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_data = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);

    // Create FP32 tensors
    auto fp32_input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto fp32_gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    auto fp32_output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    std::copy(input_data.begin(), input_data.end(), fp32_input->mutable_data());
    std::copy(gamma_data.begin(), gamma_data.end(), fp32_gamma->mutable_data());

    // Try to use Q8_1 op with FP32 tensors
    RMSNormOpTyped<ActivationPrecision::Q8_1> op;
    EXPECT_FALSE(op.execute(fp32_input.get(), fp32_gamma.get(), fp32_output.get(), rows, cols, DEFAULT_EPS));
}

/**
 * @brief Test that FP32 op works with FP32 tensors
 */
TEST_F(Test__RMSNormOpTyped, FP32_AcceptsFP32Tensors)
{
    const int rows = 4;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    auto input_data = generateRandomFP32(elements);
    auto gamma_data = generateGammaWeights(cols);

    auto fp32_input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    auto fp32_gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(cols)});
    auto fp32_output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    std::copy(input_data.begin(), input_data.end(), fp32_input->mutable_data());
    std::copy(gamma_data.begin(), gamma_data.end(), fp32_gamma->mutable_data());

    RMSNormOpTyped<ActivationPrecision::FP32> op;
    EXPECT_TRUE(op.execute(fp32_input.get(), fp32_gamma.get(), fp32_output.get(), rows, cols, DEFAULT_EPS));
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
