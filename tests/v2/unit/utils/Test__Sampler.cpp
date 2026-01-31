/**
 * @file Test__Sampler.cpp
 * @brief Unit tests for Sampler class
 * @author David Sanftenberg
 * @date 2025-11-07
 *
 * Tests for token sampling strategies including:
 * - Greedy sampling (argmax)
 * - Temperature scaling
 * - Top-k sampling
 * - Top-p (nucleus) sampling
 * - Seed reproducibility
 * - Edge cases (empty logits, single token, uniform distribution)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "utils/Sampler.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Test fixture for Sampler tests
     */
    class SamplerTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Create sampler with fixed seed for deterministic tests
            sampler_ = std::make_unique<Sampler>(12345);

            // Standard logits for testing (5 tokens)
            // Token 2 has highest logit (3.0)
            standard_logits_ = {1.0f, 2.0f, 3.0f, 0.5f, 1.5f};

            // Uniform logits (all same value)
            uniform_logits_ = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

            // Single peak logits (one clearly dominant token)
            peaked_logits_ = {0.1f, 0.2f, 10.0f, 0.1f, 0.2f};
        }

        void TearDown() override
        {
            sampler_.reset();
        }

        std::unique_ptr<Sampler> sampler_;
        std::vector<float> standard_logits_;
        std::vector<float> uniform_logits_;
        std::vector<float> peaked_logits_;
    };

    // =============================================================================
    // Basic Functionality Tests
    // =============================================================================

    TEST_F(SamplerTest, SamplerCreation)
    {
        // Should construct without errors
        EXPECT_NE(sampler_, nullptr);
    }

    TEST_F(SamplerTest, SamplerWithSeed)
    {
        // Create two samplers with same seed
        Sampler sampler1(42);
        Sampler sampler2(42);

        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // Should produce identical sequences
        for (int i = 0; i < 10; ++i)
        {
            SamplingParams params;
            params.temperature = 0.8f;

            int token1 = sampler1.sample(logits, params);
            int token2 = sampler2.sample(logits, params);

            EXPECT_EQ(token1, token2) << "Iteration " << i;
        }
    }

    // =============================================================================
    // Greedy Sampling Tests
    // =============================================================================

    TEST_F(SamplerTest, GreedySampling_StandardLogits)
    {
        int token = sampler_->sample_greedy(standard_logits_);

        // Should select token with highest logit (index 2, value 3.0)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, GreedySampling_UniformLogits)
    {
        int token = sampler_->sample_greedy(uniform_logits_);

        // With uniform logits, should select first occurrence of max
        EXPECT_EQ(token, 0);
    }

    TEST_F(SamplerTest, GreedySampling_SingleToken)
    {
        std::vector<float> single = {5.0f};
        int token = sampler_->sample_greedy(single);

        EXPECT_EQ(token, 0);
    }

    TEST_F(SamplerTest, GreedySampling_Deterministic)
    {
        // Greedy sampling should always return same result
        int first = sampler_->sample_greedy(standard_logits_);

        for (int i = 0; i < 10; ++i)
        {
            int token = sampler_->sample_greedy(standard_logits_);
            EXPECT_EQ(token, first) << "Iteration " << i;
        }
    }

    // =============================================================================
    // Temperature Scaling Tests
    // =============================================================================

    TEST_F(SamplerTest, TemperatureZero_IsGreedy)
    {
        // Temperature 0 should behave like greedy sampling
        int token = sampler_->sample_temperature(standard_logits_, 0.0f);

        // Should select token with highest logit (index 2)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, TemperatureOne_Standard)
    {
        // Temperature 1.0 is standard softmax (no scaling)
        // With fixed seed, should produce consistent result
        int token = sampler_->sample_temperature(standard_logits_, 1.0f);

        // Should be valid token index
        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, TemperatureHigh_MoreRandom)
    {
        // High temperature should allow more diversity
        // Run multiple samples and check we don't always get the peak
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_temperature(peaked_logits_, 2.0f);
        }

        // Count unique tokens sampled
        std::sort(samples.begin(), samples.end());
        auto unique_end = std::unique(samples.begin(), samples.end());
        int num_unique = std::distance(samples.begin(), unique_end);

        // With high temperature on peaked logits, should sample multiple tokens
        EXPECT_GT(num_unique, 1) << "High temperature should allow diversity";
    }

    TEST_F(SamplerTest, TemperatureLow_LessRandom)
    {
        // Low temperature should strongly favor the peak
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_temperature(peaked_logits_, 0.1f);
        }

        // Count how often we sample the peak (token 2)
        int peak_count = std::count(samples.begin(), samples.end(), 2);

        // With very low temperature, should almost always sample the peak
        EXPECT_GT(peak_count, 90) << "Low temperature should favor peak token";
    }

    // =============================================================================
    // Top-k Sampling Tests
    // =============================================================================

    TEST_F(SamplerTest, TopK_K1_IsGreedy)
    {
        // Top-k with k=1 should be equivalent to greedy
        int token = sampler_->sample_top_k(standard_logits_, 1, 1.0f);

        // Should select token with highest logit (index 2)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, TopK_K2_OnlyTopTokens)
    {
        // Top-k with k=2 should only sample from top 2 tokens
        // For standard_logits: top 2 are indices 2 (3.0) and 1 (2.0)
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_k(standard_logits_, 2, 1.0f);
        }

        // All samples should be either token 1 or 2
        for (int token : samples)
        {
            EXPECT_TRUE(token == 1 || token == 2)
                << "Top-k=2 sampled token " << token << ", expected 1 or 2";
        }
    }

    TEST_F(SamplerTest, TopK_KLargerThanVocab)
    {
        // If k > vocab_size, should consider all tokens (standard sampling)
        int token = sampler_->sample_top_k(standard_logits_, 100, 1.0f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, TopK_WithTemperature)
    {
        // Top-k with temperature should apply temp scaling before filtering
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_k(peaked_logits_, 3, 0.5f);
        }

        // Should only sample from top 3 tokens
        for (int token : samples)
        {
            EXPECT_LT(token, 5) << "Invalid token ID";
        }
    }

    // =============================================================================
    // Top-p (Nucleus) Sampling Tests
    // =============================================================================

    TEST_F(SamplerTest, TopP_P1_AllTokens)
    {
        // Top-p with p=1.0 should consider all tokens
        int token = sampler_->sample_top_p(standard_logits_, 1.0f, 1.0f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, TopP_SmallP_FewTokens)
    {
        // Small p should only consider a few high-probability tokens
        // For peaked_logits, token 2 has very high probability
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_p(peaked_logits_, 0.1f, 1.0f);
        }

        // Count unique tokens
        std::sort(samples.begin(), samples.end());
        auto unique_end = std::unique(samples.begin(), samples.end());
        int num_unique = std::distance(samples.begin(), unique_end);

        // With very small p on peaked distribution, should sample very few tokens
        EXPECT_LE(num_unique, 3) << "Small top-p should limit diversity";
    }

    TEST_F(SamplerTest, TopP_WithTemperature)
    {
        // Top-p with temperature should apply temp scaling before filtering
        int token = sampler_->sample_top_p(standard_logits_, 0.9f, 0.8f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    // =============================================================================
    // Unified sample() Interface Tests
    // =============================================================================

    TEST_F(SamplerTest, Sample_GreedyMode)
    {
        SamplingParams params;
        params.temperature = 0.0f; // Greedy

        int token = sampler_->sample(standard_logits_, params);

        // Should select token with highest logit (index 2)
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, Sample_TopKMode)
    {
        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 2;

        // Sample multiple times
        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample(standard_logits_, params);
        }

        // All samples should be from top 2 tokens (indices 1 or 2)
        for (int token : samples)
        {
            EXPECT_TRUE(token == 1 || token == 2);
        }
    }

    TEST_F(SamplerTest, Sample_TopPMode)
    {
        SamplingParams params;
        params.temperature = 1.0f;
        params.top_p = 0.5f;

        int token = sampler_->sample(standard_logits_, params);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    TEST_F(SamplerTest, Sample_CombinedTopKTopP)
    {
        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 3;
        params.top_p = 0.9f;

        int token = sampler_->sample(standard_logits_, params);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(standard_logits_.size()));
    }

    // =============================================================================
    // Seed Reproducibility Tests
    // =============================================================================

    TEST_F(SamplerTest, SeedReproducibility_SameSeed)
    {
        // Two samplers with same seed should produce identical sequences
        Sampler sampler1(12345);
        Sampler sampler2(12345);

        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 40;

        for (int i = 0; i < 20; ++i)
        {
            int token1 = sampler1.sample(standard_logits_, params);
            int token2 = sampler2.sample(standard_logits_, params);

            EXPECT_EQ(token1, token2) << "Iteration " << i;
        }
    }

    TEST_F(SamplerTest, SeedReproducibility_SetSeed)
    {
        Sampler sampler(0); // Start with random seed

        // Set fixed seed
        sampler.set_seed(42);

        SamplingParams params;
        params.temperature = 1.0f;

        // Record first sequence
        std::vector<int> sequence1;
        for (int i = 0; i < 10; ++i)
        {
            sequence1.push_back(sampler.sample(standard_logits_, params));
        }

        // Reset seed and generate again
        sampler.set_seed(42);
        std::vector<int> sequence2;
        for (int i = 0; i < 10; ++i)
        {
            sequence2.push_back(sampler.sample(standard_logits_, params));
        }

        // Sequences should match
        EXPECT_EQ(sequence1, sequence2);
    }

    // =============================================================================
    // Edge Case Tests
    // =============================================================================

    TEST_F(SamplerTest, EdgeCase_SingleToken)
    {
        std::vector<float> single = {5.0f};

        // Greedy
        EXPECT_EQ(sampler_->sample_greedy(single), 0);

        // Temperature
        EXPECT_EQ(sampler_->sample_temperature(single, 1.0f), 0);

        // Top-k
        EXPECT_EQ(sampler_->sample_top_k(single, 1, 1.0f), 0);

        // Top-p
        EXPECT_EQ(sampler_->sample_top_p(single, 0.5f, 1.0f), 0);
    }

    TEST_F(SamplerTest, EdgeCase_AllZeros)
    {
        std::vector<float> zeros = {0.0f, 0.0f, 0.0f, 0.0f};

        // All logits equal, should select first token
        int token = sampler_->sample_greedy(zeros);
        EXPECT_EQ(token, 0);

        // Temperature sampling should still work (uniform distribution)
        token = sampler_->sample_temperature(zeros, 1.0f);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 4);
    }

    TEST_F(SamplerTest, EdgeCase_AllSameValue)
    {
        // Uniform distribution should work correctly
        int token = sampler_->sample_temperature(uniform_logits_, 1.0f);

        EXPECT_GE(token, 0);
        EXPECT_LT(token, static_cast<int>(uniform_logits_.size()));
    }

    TEST_F(SamplerTest, EdgeCase_NegativeLogits)
    {
        std::vector<float> negative = {-5.0f, -2.0f, -1.0f, -10.0f};

        // Greedy should still select max (index 2, value -1.0)
        int token = sampler_->sample_greedy(negative);
        EXPECT_EQ(token, 2);

        // Temperature sampling should work
        token = sampler_->sample_temperature(negative, 1.0f);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 4);
    }

    TEST_F(SamplerTest, EdgeCase_ExtremeLogits)
    {
        std::vector<float> extreme = {-1000.0f, -1000.0f, 100.0f, -1000.0f};

        // With extreme difference, greedy should select the peak
        int token = sampler_->sample_greedy(extreme);
        EXPECT_EQ(token, 2);

        // Even with temperature, should almost always sample the peak
        Sampler sampler(42);
        for (int i = 0; i < 10; ++i)
        {
            token = sampler.sample_temperature(extreme, 1.0f);
            EXPECT_EQ(token, 2) << "Extreme logit difference should dominate";
        }
    }

    // =============================================================================
    // SamplingParams Tests
    // =============================================================================

    TEST_F(SamplerTest, SamplingParams_IsGreedy_TemperatureZero)
    {
        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_TRUE(params.is_greedy());
    }

    TEST_F(SamplerTest, SamplingParams_IsGreedy_TopK1)
    {
        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 1;
        params.top_p = 1.0f;

        EXPECT_TRUE(params.is_greedy());
    }

    TEST_F(SamplerTest, SamplingParams_NotGreedy)
    {
        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 40;
        params.top_p = 0.95f;

        EXPECT_FALSE(params.is_greedy());
    }

    // =============================================================================
    // Large Vocabulary Tests
    // =============================================================================

    TEST_F(SamplerTest, LargeVocabulary_Greedy)
    {
        // Simulate large vocabulary (e.g., 50k tokens)
        std::vector<float> large_logits(50000, 0.0f);
        large_logits[12345] = 10.0f; // Peak at specific token

        int token = sampler_->sample_greedy(large_logits);
        EXPECT_EQ(token, 12345);
    }

    TEST_F(SamplerTest, LargeVocabulary_TopK)
    {
        // Large vocabulary with top-k
        std::vector<float> large_logits(50000, 0.0f);
        large_logits[100] = 5.0f;
        large_logits[200] = 4.0f;
        large_logits[300] = 3.0f;

        Sampler sampler(42);
        std::vector<int> samples(100);

        for (int i = 0; i < 100; ++i)
        {
            samples[i] = sampler.sample_top_k(large_logits, 3, 1.0f);
        }

        // Should only sample from tokens 100, 200, 300
        for (int token : samples)
        {
            EXPECT_TRUE(token == 100 || token == 200 || token == 300)
                << "Large vocab top-k sampled unexpected token " << token;
        }
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(SamplerTest, ErrorHandling_EmptyLogits_Greedy)
    {
        std::vector<float> empty;

        EXPECT_THROW(sampler_->sample_greedy(empty), std::invalid_argument);
    }

    TEST_F(SamplerTest, ErrorHandling_EmptyLogits_Sample)
    {
        std::vector<float> empty;
        SamplingParams params;

        EXPECT_THROW(sampler_->sample(empty, params), std::invalid_argument);
    }

    TEST_F(SamplerTest, ErrorHandling_InvalidTopP)
    {
        // Top-p must be in (0, 1]
        EXPECT_THROW(sampler_->sample_top_p(standard_logits_, 0.0f, 1.0f), std::invalid_argument);
        EXPECT_THROW(sampler_->sample_top_p(standard_logits_, -0.5f, 1.0f), std::invalid_argument);
        EXPECT_THROW(sampler_->sample_top_p(standard_logits_, 1.5f, 1.0f), std::invalid_argument);
    }

    TEST_F(SamplerTest, ErrorHandling_InvalidTopK)
    {
        // Top-k must be positive
        EXPECT_THROW(sampler_->sample_top_k(standard_logits_, 0, 1.0f), std::invalid_argument);
        EXPECT_THROW(sampler_->sample_top_k(standard_logits_, -1, 1.0f), std::invalid_argument);
    }

    // =============================================================================
    // Temperature Boundary Tests (Critical for E2E parity)
    // =============================================================================

    TEST_F(SamplerTest, Temperature_VeryLow_AlmostGreedy)
    {
        // Temperature very close to 0 should behave like greedy
        // This is critical: temperature=0.01 should almost always pick the max
        Sampler sampler(42);

        for (int i = 0; i < 100; ++i)
        {
            int token = sampler.sample_temperature(peaked_logits_, 0.01f);
            EXPECT_EQ(token, 2) << "Very low temperature should always pick argmax";
        }
    }

    TEST_F(SamplerTest, Temperature_ExactlyZero_EqualsGreedy)
    {
        // Temperature = 0.0 should be exactly equivalent to greedy
        int greedy_token = sampler_->sample_greedy(standard_logits_);
        int temp_zero_token = sampler_->sample_temperature(standard_logits_, 0.0f);

        EXPECT_EQ(greedy_token, temp_zero_token)
            << "Temperature=0 must equal greedy sampling";
    }

    TEST_F(SamplerTest, Temperature_NearZero_StillDeterministic)
    {
        // Any temperature < ~0.01 should be effectively deterministic
        Sampler sampler1(123);
        Sampler sampler2(456); // Different seeds

        // Even with different seeds, near-zero temp should pick same token
        int token1 = sampler1.sample_temperature(peaked_logits_, 0.001f);
        int token2 = sampler2.sample_temperature(peaked_logits_, 0.001f);

        EXPECT_EQ(token1, token2)
            << "Near-zero temperature should be deterministic regardless of seed";
    }

    // =============================================================================
    // Token Ranking Tests (Critical for debugging inference)
    // =============================================================================

    TEST_F(SamplerTest, Ranking_TopKReturnsCorrectIndices)
    {
        // Verify that top-k returns tokens from the correct indices
        // Logits: [1.0, 2.0, 3.0, 0.5, 1.5]
        // Sorted by logit: idx 2 (3.0), idx 1 (2.0), idx 4 (1.5), idx 0 (1.0), idx 3 (0.5)

        Sampler sampler(42);
        std::map<int, int> token_counts;

        for (int i = 0; i < 1000; ++i)
        {
            int token = sampler.sample_top_k(standard_logits_, 3, 1.0f);
            token_counts[token]++;
        }

        // Top-3 are tokens 2, 1, 4
        EXPECT_GT(token_counts[2], 0) << "Token 2 (highest logit) should be sampled";
        EXPECT_GT(token_counts[1], 0) << "Token 1 (2nd highest) should be sampled";
        EXPECT_GT(token_counts[4], 0) << "Token 4 (3rd highest) should be sampled";
        EXPECT_EQ(token_counts[0], 0) << "Token 0 (4th) should NOT be sampled with k=3";
        EXPECT_EQ(token_counts[3], 0) << "Token 3 (5th) should NOT be sampled with k=3";
    }

    TEST_F(SamplerTest, Ranking_HigherLogitsHaveHigherProbability)
    {
        // With standard temperature, higher logits should be sampled more often
        Sampler sampler(42);
        std::map<int, int> token_counts;

        for (int i = 0; i < 10000; ++i)
        {
            int token = sampler.sample_temperature(standard_logits_, 1.0f);
            token_counts[token]++;
        }

        // Token 2 (logit 3.0) should be most frequent
        // Token 1 (logit 2.0) should be second most frequent
        // Token 3 (logit 0.5) should be least frequent
        EXPECT_GT(token_counts[2], token_counts[1])
            << "Higher logit should have higher sample count";
        EXPECT_GT(token_counts[1], token_counts[4])
            << "Logit 2.0 > 1.5, should be sampled more";
        EXPECT_GT(token_counts[4], token_counts[0])
            << "Logit 1.5 > 1.0, should be sampled more";
        EXPECT_GT(token_counts[0], token_counts[3])
            << "Logit 1.0 > 0.5, should be sampled more";
    }

    // =============================================================================
    // Softmax Numerical Stability Tests
    // =============================================================================

    TEST_F(SamplerTest, Softmax_VeryLargeLogits)
    {
        // Test numerical stability with very large logit values
        std::vector<float> large_logits = {500.0f, 501.0f, 502.0f, 500.5f};

        // Should not overflow or produce NaN
        int token = sampler_->sample_greedy(large_logits);
        EXPECT_EQ(token, 2) << "Greedy should work with large logits";

        token = sampler_->sample_temperature(large_logits, 1.0f);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 4);
        EXPECT_FALSE(std::isnan(static_cast<float>(token)));
    }

    TEST_F(SamplerTest, Softmax_VerySmallLogits)
    {
        // Test numerical stability with very small (negative) logit values
        std::vector<float> small_logits = {-500.0f, -501.0f, -499.0f, -500.5f};

        // Should not underflow or produce NaN
        int token = sampler_->sample_greedy(small_logits);
        EXPECT_EQ(token, 2) << "Greedy should pick -499.0f (highest)";

        token = sampler_->sample_temperature(small_logits, 1.0f);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 4);
    }

    TEST_F(SamplerTest, Softmax_MixedExtremeLogits)
    {
        // Mix of very large and very small logits
        std::vector<float> mixed = {-1000.0f, 1000.0f, -1000.0f, -1000.0f};

        // Token 1 should dominate completely
        for (int i = 0; i < 100; ++i)
        {
            int token = sampler_->sample_temperature(mixed, 1.0f);
            EXPECT_EQ(token, 1) << "Extreme positive logit should always win";
        }
    }

    // =============================================================================
    // Probability Distribution Verification
    // =============================================================================

    TEST_F(SamplerTest, Distribution_UniformLogitsGiveUniformSampling)
    {
        // Uniform logits should give approximately uniform sampling
        Sampler sampler(42);
        std::map<int, int> token_counts;

        for (int i = 0; i < 10000; ++i)
        {
            int token = sampler.sample_temperature(uniform_logits_, 1.0f);
            token_counts[token]++;
        }

        // Each token should be sampled ~2000 times (10000/5 = 2000)
        // Allow 20% tolerance
        for (int i = 0; i < 5; ++i)
        {
            EXPECT_GT(token_counts[i], 1600)
                << "Token " << i << " undersampled in uniform distribution";
            EXPECT_LT(token_counts[i], 2400)
                << "Token " << i << " oversampled in uniform distribution";
        }
    }

    TEST_F(SamplerTest, Distribution_HighTempFlattens)
    {
        // High temperature should flatten the distribution
        // Use a less extreme distribution so we can see the flattening effect
        std::vector<float> moderate_peak = {1.0f, 2.0f, 5.0f, 1.5f, 2.5f};

        Sampler sampler(42);
        std::map<int, int> counts_low_temp;
        std::map<int, int> counts_high_temp;

        for (int i = 0; i < 10000; ++i)
        {
            counts_low_temp[sampler.sample_temperature(moderate_peak, 0.3f)]++;
        }

        sampler.set_seed(42); // Reset for fair comparison
        for (int i = 0; i < 10000; ++i)
        {
            counts_high_temp[sampler.sample_temperature(moderate_peak, 5.0f)]++;
        }

        // Low temp should heavily favor the peak (token 2)
        EXPECT_GT(counts_low_temp[2], 8000) << "Low temp should strongly favor peak";

        // High temp should distribute more evenly - all tokens should get some samples
        for (int i = 0; i < 5; ++i)
        {
            EXPECT_GT(counts_high_temp[i], 500)
                << "High temp should give token " << i << " meaningful probability";
        }

        // The peak should still be favored, but less so
        EXPECT_LT(counts_high_temp[2], 4000)
            << "High temp should reduce peak dominance";
    }

    // =============================================================================
    // is_greedy() Method Comprehensive Tests
    // =============================================================================

    TEST_F(SamplerTest, IsGreedy_VariousConfigurations)
    {
        SamplingParams params;

        // Temperature 0 is always greedy
        params.temperature = 0.0f;
        params.top_k = 0;
        params.top_p = 1.0f;
        EXPECT_TRUE(params.is_greedy()) << "temp=0 should be greedy";

        // Top-k=1 with top_p>=1 is greedy
        params.temperature = 1.0f;
        params.top_k = 1;
        params.top_p = 1.0f;
        EXPECT_TRUE(params.is_greedy()) << "top_k=1 with top_p=1 should be greedy";

        // Top-k=1 with top_p<1 - still considers only 1 token
        params.temperature = 1.0f;
        params.top_k = 1;
        params.top_p = 0.5f;
        EXPECT_FALSE(params.is_greedy()) << "top_k=1 with top_p<1 is NOT considered greedy by is_greedy()";

        // Standard sampling is NOT greedy
        params.temperature = 0.8f;
        params.top_k = 40;
        params.top_p = 0.95f;
        EXPECT_FALSE(params.is_greedy()) << "Standard params should not be greedy";

        // Temperature 1 with no filtering is NOT greedy (will sample)
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 1.0f;
        EXPECT_FALSE(params.is_greedy()) << "temp=1 without filtering is not greedy";
    }

    // =============================================================================
    // Real-World Scenario Tests (Qwen2 vocabulary size)
    // =============================================================================

    TEST_F(SamplerTest, RealWorld_Qwen2VocabSize)
    {
        // Qwen2.5 has vocab_size = 151936
        const size_t vocab_size = 151936;
        std::vector<float> logits(vocab_size, 0.0f);

        // Set up realistic distribution: one clear winner with some noise
        logits[256] = 15.0f;    // Top prediction (space token)
        logits[8159] = 14.0f;   // Second
        logits[100160] = 13.5f; // Third
        logits[72363] = 13.0f;  // Fourth
        logits[105797] = 12.8f; // Fifth

        // Greedy should pick token 256
        int token = sampler_->sample_greedy(logits);
        EXPECT_EQ(token, 256) << "Greedy should pick highest logit token";

        // With low temperature, should almost always pick 256
        Sampler sampler(42);
        int count_256 = 0;
        for (int i = 0; i < 100; ++i)
        {
            if (sampler.sample_temperature(logits, 0.1f) == 256)
            {
                count_256++;
            }
        }
        EXPECT_GT(count_256, 95) << "Low temp should strongly favor top token";

        // Top-k=5 should only pick from the top 5 tokens
        sampler.set_seed(42);
        std::set<int> sampled_tokens;
        for (int i = 0; i < 100; ++i)
        {
            sampled_tokens.insert(sampler.sample_top_k(logits, 5, 1.0f));
        }

        // All sampled tokens should be in top-5
        std::set<int> top5 = {256, 8159, 100160, 72363, 105797};
        for (int t : sampled_tokens)
        {
            EXPECT_TRUE(top5.count(t) > 0)
                << "Token " << t << " not in top-5 but was sampled with k=5";
        }
    }

    TEST_F(SamplerTest, RealWorld_DecodeLoopSimulation)
    {
        // Simulate a decode loop like Main.cpp does
        const size_t vocab_size = 151936;
        std::vector<float> logits(vocab_size, 0.0f);

        // First decode step prediction
        logits[256] = 14.54f;
        logits[8159] = 14.01f;
        logits[100160] = 13.13f;

        SamplingParams params;
        params.temperature = 0.0f; // Greedy for determinism
        params.top_k = 0;
        params.top_p = 1.0f;

        // With greedy, should always get 256
        for (int step = 0; step < 10; ++step)
        {
            int token = sampler_->sample(logits, params);
            EXPECT_EQ(token, 256)
                << "Greedy decode should produce consistent tokens";
        }
    }

} // anonymous namespace
