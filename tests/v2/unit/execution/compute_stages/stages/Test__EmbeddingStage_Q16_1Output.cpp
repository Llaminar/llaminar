/**
 * @file Test__EmbeddingStage_Q16_1Output.cpp
 * @brief Unit tests for EmbeddingStage Q16_1 output path
 *
 * @category unit/stages
 * @tested   EmbeddingStage::executeQ16_1Output()
 *
 * These tests verify that the EmbeddingStage correctly:
 * 1. Performs FP32 embedding lookup
 * 2. Quantizes to Q16_1 format
 * 3. Produces output that dequantizes back to near-original FP32
 *
 * This is critical for HybridQ16 mode where the Q16_1 residual stream
 * is initialized by embedding output.
 *
 * @author GitHub Copilot
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <numeric>

#include "backends/DeviceId.h"
#include "execution/compute_stages/ComputeStages.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

namespace
{
    using namespace llaminar2;

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__EmbeddingStage_Q16_1Output : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Qwen2 0.5B dimensions
            vocab_size_ = 151936;
            d_model_ = 896;
        }

        // Create a minimal FP32 embedding table with known values
        std::unique_ptr<FP32Tensor> createEmbeddingTable(int vocab_size, int d_model, float scale = 0.02f)
        {
            auto table = std::make_unique<FP32Tensor>(std::vector<size_t>{
                static_cast<size_t>(vocab_size), static_cast<size_t>(d_model)});

            float *data = table->mutable_data();
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, scale);

            for (int i = 0; i < vocab_size * d_model; ++i)
            {
                data[i] = dist(gen);
            }

            return table;
        }

        // Create Q16_1 output tensor
        std::unique_ptr<Q16_1Tensor> createQ16_1Output(int max_tokens, int d_model)
        {
            return std::make_unique<Q16_1Tensor>(std::vector<size_t>{
                static_cast<size_t>(max_tokens), static_cast<size_t>(d_model)});
        }

        // Create FP32 output tensor for reference
        std::unique_ptr<FP32Tensor> createFP32Output(int max_tokens, int d_model)
        {
            return std::make_unique<FP32Tensor>(std::vector<size_t>{
                static_cast<size_t>(max_tokens), static_cast<size_t>(d_model)});
        }

        // Compute cosine similarity
        double cosineSimilarity(const float *a, const float *b, size_t n)
        {
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            if (norm_a < 1e-12 || norm_b < 1e-12)
                return 0.0;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        // Compute max absolute difference
        double maxAbsDiff(const float *a, const float *b, size_t n)
        {
            double max_diff = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                max_diff = std::max(max_diff, std::abs(static_cast<double>(a[i] - b[i])));
            }
            return max_diff;
        }

        // Check for NaN/Inf
        bool hasNaNOrInf(const float *data, size_t n)
        {
            for (size_t i = 0; i < n; ++i)
            {
                if (std::isnan(data[i]) || std::isinf(data[i]))
                    return true;
            }
            return false;
        }

        int vocab_size_;
        int d_model_;
    };

    // ============================================================================
    // Basic Functionality Tests
    // ============================================================================

    TEST_F(Test__EmbeddingStage_Q16_1Output, SingleToken_ProducesValidOutput)
    {
        // Create embedding table
        auto embed_table = createEmbeddingTable(vocab_size_, d_model_);

        // Create Q16_1 output
        auto q16_output = createQ16_1Output(16, d_model_);

        // Token to look up
        std::vector<int> token_ids = {42};

        // Set up stage params
        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = 1;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        // Execute
        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        // Dequantize and verify
        std::vector<float> dequantized(d_model_);
        q16_output->to_fp32_row(0, dequantized.data());

        // Compare with expected FP32 embedding
        const float *expected = embed_table->data() + token_ids[0] * d_model_;

        double cosine = cosineSimilarity(expected, dequantized.data(), d_model_);
        double max_diff = maxAbsDiff(expected, dequantized.data(), d_model_);

        std::cout << "[SingleToken] Cosine: " << cosine << ", MaxDiff: " << max_diff << std::endl;

        EXPECT_GT(cosine, 0.9999) << "Q16_1 quantization should preserve embedding with high fidelity";
        EXPECT_LT(max_diff, 0.001) << "Max abs diff should be very small for Q16_1";
        EXPECT_FALSE(hasNaNOrInf(dequantized.data(), d_model_)) << "Output should not contain NaN/Inf";
    }

    TEST_F(Test__EmbeddingStage_Q16_1Output, MultipleTokens_ProducesValidOutput)
    {
        auto embed_table = createEmbeddingTable(vocab_size_, d_model_);
        auto q16_output = createQ16_1Output(32, d_model_);

        // Multiple tokens (simulating a prompt)
        std::vector<int> token_ids = {1, 42, 100, 500, 1000, 5000, 10000, 50000, 100000};
        const int num_tokens = static_cast<int>(token_ids.size());

        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = num_tokens;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        // Verify each token's embedding
        std::vector<float> dequantized(d_model_);
        double min_cosine = 1.0;
        double max_diff_overall = 0.0;

        for (int t = 0; t < num_tokens; ++t)
        {
            q16_output->to_fp32_row(t, dequantized.data());
            const float *expected = embed_table->data() + token_ids[t] * d_model_;

            double cosine = cosineSimilarity(expected, dequantized.data(), d_model_);
            double max_diff = maxAbsDiff(expected, dequantized.data(), d_model_);

            min_cosine = std::min(min_cosine, cosine);
            max_diff_overall = std::max(max_diff_overall, max_diff);

            EXPECT_FALSE(hasNaNOrInf(dequantized.data(), d_model_))
                << "Token " << t << " output contains NaN/Inf";
        }

        std::cout << "[MultiToken] MinCosine: " << min_cosine
                  << ", MaxDiff: " << max_diff_overall << std::endl;

        EXPECT_GT(min_cosine, 0.9999) << "All tokens should have high cosine similarity";
        EXPECT_LT(max_diff_overall, 0.001) << "Max diff across all tokens should be small";
    }

    TEST_F(Test__EmbeddingStage_Q16_1Output, RealPromptTokens_MatchesFP32Output)
    {
        // Simulate typical prompt tokens (e.g., "The capital of France is")
        // These are representative token IDs for such a prompt
        std::vector<int> token_ids = {785, 6864, 315, 9822, 374};
        const int num_tokens = static_cast<int>(token_ids.size());

        auto embed_table = createEmbeddingTable(vocab_size_, d_model_);
        auto q16_output = createQ16_1Output(32, d_model_);
        auto fp32_output = createFP32Output(32, d_model_);

        // Execute Q16_1 path
        {
            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = q16_output.get();
            params.num_tokens = num_tokens;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createEmbedding(params);
            ASSERT_TRUE(stage->execute(nullptr));
        }

        // Execute FP32 path
        {
            EmbeddingStage::Params params;
            params.embed_table = embed_table.get();
            params.token_ids = token_ids.data();
            params.output = fp32_output.get();
            params.num_tokens = num_tokens;
            params.d_model = d_model_;
            params.vocab_size = vocab_size_;
            params.device_id = DeviceId::cpu();

            auto stage = ComputeStageFactory::createEmbedding(params);
            ASSERT_TRUE(stage->execute(nullptr));
        }

        // Compare Q16_1 (dequantized) vs FP32
        std::vector<float> q16_dequant(num_tokens * d_model_);
        for (int t = 0; t < num_tokens; ++t)
        {
            q16_output->to_fp32_row(t, q16_dequant.data() + t * d_model_);
        }

        const float *fp32_data = fp32_output->data();
        double cosine = cosineSimilarity(fp32_data, q16_dequant.data(), num_tokens * d_model_);
        double max_diff = maxAbsDiff(fp32_data, q16_dequant.data(), num_tokens * d_model_);

        std::cout << "[FP32vQ16_1] Cosine: " << cosine << ", MaxDiff: " << max_diff << std::endl;

        EXPECT_GT(cosine, 0.9999) << "Q16_1 should match FP32 with high fidelity";
        EXPECT_LT(max_diff, 0.001) << "Max diff should be very small";
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    TEST_F(Test__EmbeddingStage_Q16_1Output, FirstAndLastVocabTokens)
    {
        auto embed_table = createEmbeddingTable(vocab_size_, d_model_);
        auto q16_output = createQ16_1Output(16, d_model_);

        // Test boundary tokens
        std::vector<int> token_ids = {0, vocab_size_ - 1};
        const int num_tokens = 2;

        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = num_tokens;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        // Verify both tokens
        std::vector<float> dequantized(d_model_);
        for (int t = 0; t < num_tokens; ++t)
        {
            q16_output->to_fp32_row(t, dequantized.data());
            const float *expected = embed_table->data() + token_ids[t] * d_model_;

            double cosine = cosineSimilarity(expected, dequantized.data(), d_model_);
            EXPECT_GT(cosine, 0.9999) << "Token " << token_ids[t] << " should have high fidelity";
            EXPECT_FALSE(hasNaNOrInf(dequantized.data(), d_model_));
        }
    }

    TEST_F(Test__EmbeddingStage_Q16_1Output, LargeEmbeddingValues_QuantizesCorrectly)
    {
        // Create embedding table with larger values to test quantization range
        auto embed_table = createEmbeddingTable(vocab_size_, d_model_, 0.5f); // Larger scale
        auto q16_output = createQ16_1Output(16, d_model_);

        std::vector<int> token_ids = {42};

        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = 1;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        std::vector<float> dequantized(d_model_);
        q16_output->to_fp32_row(0, dequantized.data());

        const float *expected = embed_table->data() + token_ids[0] * d_model_;
        double cosine = cosineSimilarity(expected, dequantized.data(), d_model_);

        std::cout << "[LargeValues] Cosine: " << cosine << std::endl;

        // Q16_1 should still handle larger values well
        EXPECT_GT(cosine, 0.999) << "Q16_1 should handle larger embedding values";
        EXPECT_FALSE(hasNaNOrInf(dequantized.data(), d_model_));
    }

    TEST_F(Test__EmbeddingStage_Q16_1Output, SmallEmbeddingValues_QuantizesCorrectly)
    {
        // Create embedding table with very small values
        auto embed_table = createEmbeddingTable(vocab_size_, d_model_, 0.001f); // Very small scale
        auto q16_output = createQ16_1Output(16, d_model_);

        std::vector<int> token_ids = {42};

        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = 1;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        std::vector<float> dequantized(d_model_);
        q16_output->to_fp32_row(0, dequantized.data());

        const float *expected = embed_table->data() + token_ids[0] * d_model_;
        double cosine = cosineSimilarity(expected, dequantized.data(), d_model_);

        std::cout << "[SmallValues] Cosine: " << cosine << std::endl;

        // Q16_1 should handle small values (16-bit precision is sufficient)
        EXPECT_GT(cosine, 0.999) << "Q16_1 should handle small embedding values";
        EXPECT_FALSE(hasNaNOrInf(dequantized.data(), d_model_));
    }

    // ============================================================================
    // Numerical Precision Tests
    // ============================================================================

    TEST_F(Test__EmbeddingStage_Q16_1Output, Q16_1_Roundtrip_PreservesValues)
    {
        // This tests the full roundtrip: FP32 → Q16_1 → FP32
        // to ensure quantization/dequantization works correctly

        auto embed_table = createEmbeddingTable(vocab_size_, d_model_);
        auto q16_output = createQ16_1Output(32, d_model_);

        // Use a variety of tokens
        std::vector<int> token_ids;
        for (int i = 0; i < 16; ++i)
        {
            token_ids.push_back(i * 10000); // Spread across vocab
        }
        const int num_tokens = static_cast<int>(token_ids.size());

        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = num_tokens;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        // Compute statistics across all tokens
        double total_mse = 0.0;
        double total_elements = 0.0;
        std::vector<float> dequantized(d_model_);

        for (int t = 0; t < num_tokens; ++t)
        {
            q16_output->to_fp32_row(t, dequantized.data());
            const float *expected = embed_table->data() + token_ids[t] * d_model_;

            for (int d = 0; d < d_model_; ++d)
            {
                double diff = dequantized[d] - expected[d];
                total_mse += diff * diff;
                total_elements += 1.0;
            }
        }

        double rmse = std::sqrt(total_mse / total_elements);
        std::cout << "[Roundtrip] RMSE: " << rmse << std::endl;

        // Q16_1 should have very low quantization error
        EXPECT_LT(rmse, 1e-4) << "RMSE should be very low for Q16_1 quantization";
    }

    // ============================================================================
    // Stress Tests
    // ============================================================================

    TEST_F(Test__EmbeddingStage_Q16_1Output, MaxSequenceLength_WorksCorrectly)
    {
        // Test with a longer sequence (e.g., 512 tokens)
        const int seq_len = 512;

        auto embed_table = createEmbeddingTable(vocab_size_, d_model_);
        auto q16_output = createQ16_1Output(seq_len, d_model_);

        // Generate random token IDs
        std::vector<int> token_ids(seq_len);
        std::mt19937 gen(123);
        std::uniform_int_distribution<int> dist(0, vocab_size_ - 1);
        for (int i = 0; i < seq_len; ++i)
        {
            token_ids[i] = dist(gen);
        }

        EmbeddingStage::Params params;
        params.embed_table = embed_table.get();
        params.token_ids = token_ids.data();
        params.output = q16_output.get();
        params.num_tokens = seq_len;
        params.d_model = d_model_;
        params.vocab_size = vocab_size_;
        params.device_id = DeviceId::cpu();

        auto stage = ComputeStageFactory::createEmbedding(params);
        ASSERT_TRUE(stage->execute(nullptr));

        // Spot check a few tokens
        std::vector<float> dequantized(d_model_);
        for (int t : {0, seq_len / 2, seq_len - 1})
        {
            q16_output->to_fp32_row(t, dequantized.data());
            const float *expected = embed_table->data() + token_ids[t] * d_model_;

            double cosine = cosineSimilarity(expected, dequantized.data(), d_model_);
            EXPECT_GT(cosine, 0.9999) << "Token at position " << t << " should have high fidelity";
            EXPECT_FALSE(hasNaNOrInf(dequantized.data(), d_model_));
        }
    }

} // anonymous namespace
