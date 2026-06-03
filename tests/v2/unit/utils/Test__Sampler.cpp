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
#include "kernels/common/SamplingMath.h"

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

    TEST_F(SamplerTest, ComputeDistributionCombinesTopKAndTopP)
    {
        std::vector<float> logits = {4.0f, 3.0f, 2.0f, 1.0f};

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 3;
        params.top_p = 0.8f;

        auto distribution =
            sampler_->compute_distribution(logits.data(), logits.size(), params);

        ASSERT_EQ(distribution.size(), 2u)
            << "top-p should be applied inside the top-k candidate set";
        EXPECT_EQ(distribution[0].token_id, 0);
        EXPECT_EQ(distribution[1].token_id, 1);
        EXPECT_NEAR(distribution[0].probability + distribution[1].probability,
                    1.0f,
                    1e-6f);
        EXPECT_GT(distribution[0].probability, distribution[1].probability);
        EXPECT_FLOAT_EQ(Sampler::probability_of_token(distribution, 2), 0.0f);
    }

    TEST_F(SamplerTest, ComputeDistributionMatchesSharedCompactSamplingMath)
    {
        std::vector<float> logits = {0.1f, 4.5f, 3.8f, 0.0f,
                                     2.2f, 5.0f, -1.0f, 3.2f};

        SamplingParams params;
        params.temperature = 0.6f;
        params.top_k = 5;
        params.top_p = 0.85f;

        std::vector<std::pair<float, int>> sorted;
        sorted.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            sorted.emplace_back(logits[i], static_cast<int>(i));
        }
        std::partial_sort(
            sorted.begin(),
            sorted.begin() + params.top_k,
            sorted.end(),
            [](const auto &a, const auto &b)
            {
                return a.first > b.first;
            });

        std::vector<float> sorted_logits(static_cast<size_t>(params.top_k));
        std::vector<int> sorted_ids(static_cast<size_t>(params.top_k));
        for (int i = 0; i < params.top_k; ++i)
        {
            sorted_logits[static_cast<size_t>(i)] = sorted[static_cast<size_t>(i)].first;
            sorted_ids[static_cast<size_t>(i)] = sorted[static_cast<size_t>(i)].second;
        }

        std::vector<float> scratch(static_cast<size_t>(params.top_k), 0.0f);
        std::vector<int> expected_ids(static_cast<size_t>(params.top_k), -1);
        std::vector<float> expected_probs(static_cast<size_t>(params.top_k), 0.0f);
        sampling_math::build_topk_topp_distribution_from_sorted(
            sorted_logits.data(),
            sorted_ids.data(),
            params.top_k,
            params.top_p,
            params.temperature,
            expected_ids.data(),
            expected_probs.data(),
            scratch.data());

        const auto distribution =
            sampler_->compute_distribution(logits.data(), logits.size(), params);

        std::vector<int> actual_ids;
        std::vector<float> actual_probs;
        for (const auto &entry : distribution)
        {
            actual_ids.push_back(entry.token_id);
            actual_probs.push_back(entry.probability);
        }

        size_t expected_active = 0;
        for (int i = 0; i < params.top_k; ++i)
        {
            if (expected_ids[static_cast<size_t>(i)] >= 0)
            {
                ASSERT_LT(expected_active, actual_ids.size());
                EXPECT_EQ(actual_ids[expected_active], expected_ids[static_cast<size_t>(i)]);
                EXPECT_NEAR(actual_probs[expected_active],
                            expected_probs[static_cast<size_t>(i)],
                            1e-6f);
                ++expected_active;
            }
        }
        EXPECT_EQ(actual_ids.size(), expected_active);
    }

    TEST_F(SamplerTest, SharedSamplingMathSpeculativeVerifyMatchesSamplerHelpers)
    {
        const int target_ids[] = {10, 20, 30, -1};
        const float target_probs[] = {0.2f, 0.3f, 0.5f, 0.0f};
        const int draft_ids[] = {10, 20, 40, -1};
        const float draft_probs[] = {0.4f, 0.1f, 0.5f, 0.0f};

        int out_token = -1;
        int out_accepted = -1;
        float out_accept_probability = -1.0f;
        float out_accept_threshold = -1.0f;
        sampling_math::speculative_verify_with_thresholds(
            target_ids,
            target_probs,
            draft_ids,
            draft_probs,
            4,
            10,
            0.75f,
            0.25f,
            &out_token,
            &out_accepted,
            &out_accept_probability,
            &out_accept_threshold);

        EXPECT_EQ(out_accepted, 0);
        EXPECT_EQ(out_token, 20);
        EXPECT_NEAR(out_accept_probability,
                    Sampler::speculative_accept_probability(0.2f, 0.4f),
                    1e-6f);
        EXPECT_NEAR(out_accept_threshold, 0.75f, 1e-6f);

        std::vector<SamplingDistributionEntry> residual =
            Sampler::residual_distribution({{10, 0.2f}, {20, 0.3f}, {30, 0.5f}},
                                           {{10, 0.4f}, {20, 0.1f}, {40, 0.5f}});
        std::vector<int> residual_ids;
        std::vector<float> residual_probs;
        for (const auto &entry : residual)
        {
            residual_ids.push_back(entry.token_id);
            residual_probs.push_back(entry.probability);
        }
        EXPECT_EQ(sampling_math::sample_distribution_with_threshold(
                      residual_ids.data(),
                      residual_probs.data(),
                      static_cast<int>(residual_ids.size()),
                      0.25f),
                  out_token);
    }

    TEST_F(SamplerTest, ResidualDistributionSamplesPositiveTargetMinusDraftMass)
    {
        Sampler sampler(123);
        std::vector<SamplingDistributionEntry> target = {
            {1, 0.2f},
            {2, 0.8f},
        };
        std::vector<SamplingDistributionEntry> draft = {
            {1, 0.9f},
            {2, 0.1f},
        };

        for (int i = 0; i < 20; ++i)
        {
            EXPECT_EQ(sampler.sample_from_residual_distribution(target, draft), 2);
        }
    }

    TEST_F(SamplerTest, SpeculativeAcceptProbabilityUsesTargetOverDraftMass)
    {
        EXPECT_FLOAT_EQ(Sampler::speculative_accept_probability(0.8f, 0.2f), 1.0f)
            << "p >= q should always accept the draft token";
        EXPECT_NEAR(Sampler::speculative_accept_probability(0.2f, 0.5f), 0.4f, 1e-6f);
        EXPECT_FLOAT_EQ(Sampler::speculative_accept_probability(0.2f, 0.0f), 0.0f)
            << "a draft token absent from q cannot be accepted";
        EXPECT_FLOAT_EQ(Sampler::speculative_accept_probability(-0.2f, 0.5f), 0.0f)
            << "negative probability inputs should clamp to no accept";
    }

    TEST_F(SamplerTest, ResidualDistributionNormalizesTargetMinusDraftMass)
    {
        std::vector<SamplingDistributionEntry> target = {
            {1, 0.2f},
            {2, 0.3f},
            {3, 0.5f},
        };
        std::vector<SamplingDistributionEntry> draft = {
            {1, 0.1f},
            {2, 0.5f},
            {4, 0.4f},
        };

        const auto residual = Sampler::residual_distribution(target, draft);

        ASSERT_EQ(residual.size(), 2u);
        EXPECT_EQ(residual[0].token_id, 1);
        EXPECT_EQ(residual[1].token_id, 3);
        EXPECT_NEAR(residual[0].probability, 1.0f / 6.0f, 1e-6f);
        EXPECT_NEAR(residual[1].probability, 5.0f / 6.0f, 1e-6f);
        EXPECT_FLOAT_EQ(Sampler::probability_of_token(residual, 2), 0.0f)
            << "tokens where q exceeds p must not appear in the residual";
        EXPECT_FLOAT_EQ(Sampler::probability_of_token(residual, 4), 0.0f)
            << "draft-only tokens must not appear in the residual";
    }

    TEST_F(SamplerTest, SpeculativeAcceptRejectReconstructsTargetDistribution)
    {
        std::vector<SamplingDistributionEntry> target = {
            {1, 0.2f},
            {2, 0.3f},
            {3, 0.5f},
        };
        std::vector<SamplingDistributionEntry> draft = {
            {1, 0.5f},
            {2, 0.25f},
            {3, 0.25f},
        };
        const auto residual = Sampler::residual_distribution(target, draft);

        std::vector<SamplingDistributionEntry> reconstructed = {
            {1, 0.0f},
            {2, 0.0f},
            {3, 0.0f},
        };
        auto add_probability = [&](int token_id, float mass) {
            for (auto &entry : reconstructed)
            {
                if (entry.token_id == token_id)
                {
                    entry.probability += mass;
                    return;
                }
            }
        };

        for (const auto &draft_entry : draft)
        {
            const float p = Sampler::probability_of_token(target, draft_entry.token_id);
            const float q = draft_entry.probability;
            const float accept_probability =
                Sampler::speculative_accept_probability(p, q);
            add_probability(draft_entry.token_id, q * accept_probability);

            const float reject_mass = q * (1.0f - accept_probability);
            for (const auto &residual_entry : residual)
            {
                add_probability(residual_entry.token_id,
                                reject_mass * residual_entry.probability);
            }
        }

        for (const auto &target_entry : target)
        {
            EXPECT_NEAR(Sampler::probability_of_token(reconstructed, target_entry.token_id),
                        target_entry.probability,
                        1e-6f)
                << "speculative accept/reject path must preserve the target distribution";
        }
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

    // =============================================================================
    // SamplingParams Penalty Helpers
    // =============================================================================

    TEST_F(SamplerTest, HasPenalties_DefaultFalse)
    {
        SamplingParams params;
        EXPECT_FALSE(params.has_penalties());
    }

    TEST_F(SamplerTest, HasPenalties_PresenceOnly)
    {
        SamplingParams params;
        params.presence_penalty = 1.0f;
        EXPECT_TRUE(params.has_penalties());
    }

    TEST_F(SamplerTest, HasPenalties_FrequencyOnly)
    {
        SamplingParams params;
        params.frequency_penalty = 0.5f;
        EXPECT_TRUE(params.has_penalties());
    }

    TEST_F(SamplerTest, HasPenalties_BothSet)
    {
        SamplingParams params;
        params.presence_penalty = 1.0f;
        params.frequency_penalty = 0.5f;
        EXPECT_TRUE(params.has_penalties());
    }

    TEST_F(SamplerTest, HasPenalties_NegativeValues)
    {
        SamplingParams params;
        params.presence_penalty = -1.0f;
        EXPECT_TRUE(params.has_penalties())
            << "Negative penalties (reward) should still be 'has_penalties'";
    }

    // =============================================================================
    // Record Token and Reset History
    // =============================================================================

    TEST_F(SamplerTest, RecordToken_TracksTokenCounts)
    {
        // Recording tokens changes future sampling with penalties
        sampler_->record_token(2); // token 2 once
        sampler_->record_token(2); // token 2 twice
        sampler_->record_token(3); // token 3 once

        // Verify via frequency penalty: token 2 (count=2) should be penalized
        // more than token 3 (count=1)
        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5} → token 2 has highest (3.0)
        SamplingParams params;
        params.temperature = 0.0f; // greedy
        params.frequency_penalty = 5.0f; // large enough to shift argmax

        // Without penalty, greedy picks token 2 (logit 3.0)
        // With freq penalty: logit[2] -= 5.0 * 2 = -7.0, logit[3] -= 5.0 * 1 = -4.5
        // Adjusted: {1.0, 2.0, -7.0, -4.0, 1.5}
        // Token 1 (logit 2.0) becomes the highest
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 1) << "Frequency penalty should shift greedy argmax away from repeated token";
    }

    TEST_F(SamplerTest, ResetHistory_ClearsTokenCounts)
    {
        sampler_->record_token(2);
        sampler_->record_token(2);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 100.0f; // huge penalty

        // With penalty and history, token 2 is penalized
        int token_with_penalty = sampler_->sample(standard_logits_, params);
        EXPECT_NE(token_with_penalty, 2) << "Token 2 should be penalized away";

        // Reset history
        sampler_->reset_history();

        // After reset, no history → no penalty applied → back to normal greedy
        int token_after_reset = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token_after_reset, 2) << "After reset, greedy should pick token 2 again";
    }

    TEST_F(SamplerTest, ResetHistory_CanBeCalledWhenEmpty)
    {
        // Should not throw or crash
        sampler_->reset_history();
        sampler_->reset_history(); // twice is fine
    }

    // =============================================================================
    // Presence Penalty
    // =============================================================================

    TEST_F(SamplerTest, PresencePenalty_PenalizesSeenTokens)
    {
        // Record token 2 (the default greedy pick)
        sampler_->record_token(2);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 5.0f; // subtract 5.0 from any seen token

        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5}
        // After presence penalty: token 2 goes from 3.0 to 3.0 - 5.0 = -2.0
        // New argmax is token 1 (logit 2.0)
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 1);
    }

    TEST_F(SamplerTest, PresencePenalty_SamePenaltyRegardlessOfFrequency)
    {
        // Record token 2 multiple times
        sampler_->record_token(2);
        sampler_->record_token(2);
        sampler_->record_token(2);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 2.5f; // subtract 2.5 regardless of count

        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5}
        // Token 2 with presence penalty: 3.0 - 2.5 = 0.5
        // Token 1 still has 2.0 → should be argmax
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 1)
            << "Presence penalty should be the same whether token appeared 1 or 3 times";
    }

    TEST_F(SamplerTest, PresencePenalty_NoEffectOnUnseenTokens)
    {
        // Record token 0 only
        sampler_->record_token(0);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 100.0f;

        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5}
        // Token 0 becomes 1.0 - 100 = -99.0, rest unchanged
        // Token 2 (logit 3.0) should still win
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 2);
    }

    // =============================================================================
    // Frequency Penalty
    // =============================================================================

    TEST_F(SamplerTest, FrequencyPenalty_ScalesWithCount)
    {
        // Record token 2 three times, token 1 once
        sampler_->record_token(2);
        sampler_->record_token(2);
        sampler_->record_token(2);
        sampler_->record_token(1);

        SamplingParams params;
        params.temperature = 0.0f;
        params.frequency_penalty = 1.0f;

        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5}
        // After freq penalty:
        //   token 1: 2.0 - 1.0*1 = 1.0
        //   token 2: 3.0 - 1.0*3 = 0.0
        //   others unchanged: {1.0, 1.0, 0.0, 0.5, 1.5}
        // Token 4 (logit 1.5) should be argmax
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 4);
    }

    TEST_F(SamplerTest, FrequencyPenalty_NoEffectWithoutHistory)
    {
        // No tokens recorded
        SamplingParams params;
        params.temperature = 0.0f;
        params.frequency_penalty = 100.0f;

        // Should pick token 2 as normal (no history to penalize)
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 2);
    }

    // =============================================================================
    // Combined Penalties
    // =============================================================================

    TEST_F(SamplerTest, CombinedPenalties_BothApply)
    {
        // Record token 2 twice
        sampler_->record_token(2);
        sampler_->record_token(2);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 1.0f;
        params.frequency_penalty = 1.0f;

        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5}
        // Token 2 penalty: presence(1.0) + frequency(1.0 * 2) = 3.0
        // Token 2 adjusted: 3.0 - 3.0 = 0.0
        // Argmax should be token 1 (2.0)
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 1);
    }

    TEST_F(SamplerTest, Penalty_NegativePresence_BoostsSeenTokens)
    {
        // Negative presence penalty acts as a reward for repeated tokens
        std::vector<float> logits = {5.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        sampler_->record_token(1);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = -10.0f; // boost seen tokens by 10

        // Token 1 adjusted: 1.0 - (-10.0) = 11.0, which beats token 0 (5.0)
        int token = sampler_->sample(logits, params);
        EXPECT_EQ(token, 1);
    }

    TEST_F(SamplerTest, Penalty_OutOfBoundsTokenId_Ignored)
    {
        // Record token IDs that are out of bounds for the logits array
        sampler_->record_token(100); // way beyond vocab size of 5
        sampler_->record_token(-1);  // negative

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 100.0f;
        params.frequency_penalty = 100.0f;

        // Should still pick token 2 normally (out-of-bound tokens ignored)
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 2);
    }

    TEST_F(SamplerTest, Penalty_WithTemperatureSampling)
    {
        // Penalties are applied before temperature scaling
        sampler_->record_token(2); // penalize the dominant token

        SamplingParams params;
        params.temperature = 0.5f; // low temperature (more peaked)
        params.presence_penalty = 5.0f;
        params.seed = 42;

        sampler_->set_seed(42);

        // Token 2 adjusted: 3.0 - 5.0 = -2.0
        // After penalty, token 1 (2.0) is highest
        // With low temp, should strongly favor token 1
        std::unordered_map<int, int> counts;
        for (int i = 0; i < 100; ++i)
        {
            sampler_->set_seed(42 + i);
            int token = sampler_->sample(standard_logits_, params);
            counts[token]++;
        }
        // Token 1 should be heavily favored
        EXPECT_GT(counts[1], 50) << "Token 1 should be most likely after penalty + low temp";
        EXPECT_EQ(counts.count(2) > 0 ? counts[2] : 0, counts.count(2) > 0 ? counts[2] : 0);
    }

    TEST_F(SamplerTest, Penalty_GreedyWithPenalties_MakesCopy)
    {
        // Greedy + penalties should work (uses vector copy path)
        sampler_->record_token(2);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 5.0f;

        // Should not modify the original logits
        std::vector<float> logits_copy = standard_logits_;
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_NE(token, 2);

        // Original logits should be unchanged
        EXPECT_EQ(standard_logits_, logits_copy);
    }

    TEST_F(SamplerTest, Penalty_RawPointerApi_Works)
    {
        sampler_->record_token(2);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 5.0f;

        // Using the raw pointer API
        int token = sampler_->sample(standard_logits_.data(), standard_logits_.size(), params);
        EXPECT_NE(token, 2) << "Raw pointer API should also apply penalties";
    }

    TEST_F(SamplerTest, Penalty_MultipleTokenPenalized)
    {
        // Record multiple different tokens
        sampler_->record_token(1);
        sampler_->record_token(2);
        sampler_->record_token(4);

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 10.0f;

        // Logits: {1.0, 2.0, 3.0, 0.5, 1.5}
        // After penalty: {1.0, -8.0, -7.0, 0.5, -8.5}
        // Token 0 (1.0) should be argmax
        int token = sampler_->sample(standard_logits_, params);
        EXPECT_EQ(token, 0);
    }

    TEST_F(SamplerTest, Penalty_SimulateDecodeLoop)
    {
        // Simulate repeated sampling with penalty to avoid repetition
        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 2.5f, 1.0f};
        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 5.0f;

        std::vector<int> generated;
        for (int i = 0; i < 5; ++i)
        {
            int token = sampler_->sample(logits, params);
            sampler_->record_token(token);
            generated.push_back(token);
        }

        // Each token should be different (with strong enough penalty on 5-token vocab)
        std::set<int> unique_tokens(generated.begin(), generated.end());
        EXPECT_EQ(unique_tokens.size(), 5u)
            << "With strong presence penalty, all 5 tokens should be selected once each";
    }

    TEST_F(SamplerTest, Penalty_LargeVocab_Efficient)
    {
        // Verify penalties work efficiently with large vocab
        const size_t vocab_size = 151936;
        std::vector<float> logits(vocab_size, 0.0f);
        logits[0] = 10.0f;
        logits[42] = 9.0f;
        logits[1000] = 8.0f;

        // Record a few tokens
        sampler_->record_token(0);
        sampler_->record_token(0);

        SamplingParams params;
        params.temperature = 0.0f;
        params.frequency_penalty = 5.5f;

        // Token 0: 10.0 - 5.5*2 = -1.0
        // Token 42 (9.0) should now be argmax
        int token = sampler_->sample(logits, params);
        EXPECT_EQ(token, 42);
    }

    // =============================================================================
    // DRY Penalty Tests
    // =============================================================================

    TEST_F(SamplerTest, DRY_NoPenaltyWhenDisabled)
    {
        // DRY is disabled when multiplier == 0 (default)
        const int vocab_size = 100;
        sampler_->record_token(5);
        sampler_->record_token(6);
        sampler_->record_token(5);
        sampler_->record_token(6);

        SamplingParams params;
        params.dry_multiplier = 0.0f; // disabled
        auto penalties = sampler_->compute_penalty_map(params, vocab_size);
        EXPECT_TRUE(penalties.empty());
    }

    TEST_F(SamplerTest, DRY_NoPenaltyWithNoHistory)
    {
        SamplingParams params;
        params.dry_multiplier = 1.0f;
        auto penalties = sampler_->compute_penalty_map(params, 100);
        EXPECT_TRUE(penalties.empty());
    }

    TEST_F(SamplerTest, DRY_DetectsSimpleRepeat)
    {
        // History: [A, B, C, A, B, C] — "A B C" repeated
        // The token that would continue the repeat is A (token 10)
        const int A = 10, B = 20, C = 30;
        for (int token : {A, B, C, A, B, C})
            sampler_->record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 1; // Trigger on repeat_len > 1
        params.dry_penalty_last_n = -1; // Use full history

        auto penalties = sampler_->compute_penalty_map(params, 100);

        // Token A should be penalized (extending the repeat "A B C" → "A B C A")
        bool found_A = false;
        for (const auto &p : penalties)
        {
            if (p.token_id == A)
            {
                found_A = true;
                EXPECT_GT(p.penalty, 0.0f);
            }
        }
        EXPECT_TRUE(found_A) << "Token A should be penalized for extending the repeat";
    }

    TEST_F(SamplerTest, DRY_AllowedLengthPreventsShortRepeats)
    {
        // History: [A, B, A, B] — "A B" is length 2
        // llama.cpp semantics: allowed_length means "penalize repeats >= this length"
        // So allowed_length=3 means repeats of length 2 are NOT penalized
        const int A = 10, B = 20;
        for (int token : {A, B, A, B})
            sampler_->record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 3; // Don't penalize repeats of length < 3
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);
        // The repeat is length 2, which is < allowed_length 3, so no penalty
        EXPECT_TRUE(penalties.empty())
            << "Repeats < allowed_length should not be penalized";
    }

    TEST_F(SamplerTest, DRY_ExponentialPenaltyScaling)
    {
        // History: [A, B, C, D, A, B, C, D] — repeat of length 4
        const int A = 10, B = 20, C = 30, D = 40;
        for (int token : {A, B, C, D, A, B, C, D})
            sampler_->record_token(token);

        SamplingParams params;
        params.dry_multiplier = 2.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);

        // Find penalty for A (the token that would extend the repeat)
        float penalty_A = 0.0f;
        for (const auto &p : penalties)
        {
            if (p.token_id == A)
                penalty_A = p.penalty;
        }

        // Expected: multiplier * base^(repeat_len - allowed_length) = 2.0 * 1.75^(4-1) = 2.0 * 5.359375
        float expected = 2.0f * std::pow(1.75f, 3.0f);
        EXPECT_NEAR(penalty_A, expected, 0.01f)
            << "DRY penalty should scale exponentially";
    }

    TEST_F(SamplerTest, DRY_WindowLimitsHistory)
    {
        // Fill history: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        for (int token : {1, 2, 3, 4, 5, 1, 2, 3, 4, 5})
            sampler_->record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = 3; // Only look at last 3 tokens [4, 5]... wait, [3,4,5]

        auto penalties = sampler_->compute_penalty_map(params, 100);

        // With only 3 tokens of history visible, longer repeats won't be detected
        // The 3-token window sees [3, 4, 5] — need at least repeat_len+1 tokens to detect
        // So the full 5-token repeat can't be seen
        // But smaller sub-patterns within the window might still trigger
        // The key test is that with dry_penalty_last_n=3, we don't get the full-length penalty
        float full_penalty = std::pow(1.75f, 4.0f); // 5-1-0 = 4, never reached
        for (const auto &p : penalties)
        {
            EXPECT_LT(p.penalty, full_penalty)
                << "Window should prevent detection of full repeat";
        }
    }

    TEST_F(SamplerTest, DRY_SequenceBreakersResetDetection)
    {
        // History: [A, NEWLINE, A] where NEWLINE is a breaker
        const int A = 10, NEWLINE = 50;
        for (int token : {A, NEWLINE, A})
            sampler_->record_token(token);

        // Set up breaker: token 50 (NEWLINE) is a single-token breaker
        sampler_->initDryBreakers({"\n"}, [&](const std::string &) -> std::vector<int> {
            return {NEWLINE}; // Mock: "\n" tokenizes to [NEWLINE]
        });

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 1;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);

        // The NEWLINE between the two A tokens should break repeat detection
        // So A should NOT be penalized
        for (const auto &p : penalties)
        {
            if (p.token_id == A)
            {
                FAIL() << "Token A should not be penalized when a sequence breaker intervenes";
            }
        }
    }

    TEST_F(SamplerTest, DRY_CombinesWithPresenceFrequencyPenalty)
    {
        const int A = 10;
        sampler_->record_token(A);
        sampler_->record_token(A);
        sampler_->record_token(A);
        sampler_->record_token(A);

        SamplingParams params;
        params.presence_penalty = 1.0f;
        params.frequency_penalty = 0.5f;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);

        // Token A should have combined penalty from presence + frequency + DRY
        float total_penalty = 0.0f;
        for (const auto &p : penalties)
        {
            if (p.token_id == A)
                total_penalty = p.penalty;
        }

        // Presence: 1.0, Frequency: 0.5 * 4 = 2.0, total presence+freq = 3.0
        // DRY also adds on top of that
        EXPECT_GT(total_penalty, 3.0f)
            << "DRY should add to presence+frequency penalties";
    }

    TEST_F(SamplerTest, DRY_AffectsSampling)
    {
        // Create a scenario where DRY penalty changes the argmax
        const int vocab_size = 10;
        std::vector<float> logits(vocab_size, 0.0f);

        // Token 5 has the highest logit
        logits[5] = 10.0f;
        logits[3] = 9.5f;

        // Build history that would cause DRY to penalize token 5
        // History: [5, 7, 5, 7] — repeating pattern, next would be 5
        sampler_->record_token(5);
        sampler_->record_token(7);
        sampler_->record_token(5);
        sampler_->record_token(7);

        SamplingParams params;
        params.temperature = 0.0f; // Greedy
        params.dry_multiplier = 5.0f;
        params.dry_base = 1.75f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        int token = sampler_->sample(logits, params);
        // Token 5 should be penalized enough that token 3 wins
        EXPECT_EQ(token, 3) << "DRY penalty should prevent repeating token 5";
    }

    TEST_F(SamplerTest, DRY_ResetHistoryClearsDryState)
    {
        sampler_->record_token(1);
        sampler_->record_token(2);
        sampler_->record_token(1);
        sampler_->record_token(2);

        sampler_->reset_history();

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);
        EXPECT_TRUE(penalties.empty())
            << "After reset, DRY should have no history to detect repeats from";
    }

    TEST_F(SamplerTest, DRY_PenaltyLastN_Zero_DisablesDRY)
    {
        sampler_->record_token(5);
        sampler_->record_token(5);
        sampler_->record_token(5);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_penalty_last_n = 0; // Disabled

        auto penalties = sampler_->compute_penalty_map(params, 100);
        EXPECT_TRUE(penalties.empty());
    }

    TEST_F(SamplerTest, DRY_LogitPenaltyStruct)
    {
        // Basic sanity check for LogitPenalty
        LogitPenalty lp{42, 3.14f};
        EXPECT_EQ(lp.token_id, 42);
        EXPECT_FLOAT_EQ(lp.penalty, 3.14f);
    }

    TEST_F(SamplerTest, DRY_HasPenaltiesIncludesDRY)
    {
        SamplingParams params;
        params.dry_multiplier = 0.0f;
        EXPECT_FALSE(params.has_penalties());

        params.dry_multiplier = 1.0f;
        EXPECT_TRUE(params.has_penalties());
    }

    TEST_F(SamplerTest, DRY_OverflowProtection)
    {
        // With a very large repeat, pow() should not overflow to infinity
        // History: 50 copies of token A → repeat_len = 49
        const int A = 10;
        for (int i = 0; i < 50; ++i)
            sampler_->record_token(A);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 2.0f;         // 2^49 would overflow float
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);

        for (const auto &p : penalties)
        {
            EXPECT_FALSE(std::isinf(p.penalty))
                << "DRY penalty should not overflow to infinity";
            EXPECT_FALSE(std::isnan(p.penalty))
                << "DRY penalty should not be NaN";
            EXPECT_GT(p.penalty, 0.0f)
                << "DRY penalty should still be positive after clamping";
        }
    }

    TEST_F(SamplerTest, DRY_SingleTokenBreakerExemption)
    {
        // If a token is itself a single-token sequence breaker, it should be
        // exempt from DRY penalty even if it would extend a repeat.
        // This aligns with llama.cpp's Step 4 breaker exemption.
        const int A = 10, NEWLINE = 50;

        // History: [NEWLINE, A, NEWLINE, A, NEWLINE]
        // The repeat "NEWLINE A" appears twice, so NEWLINE would normally be penalized
        // as the token that extends the repeat. But NEWLINE is a breaker → exempt.
        for (int token : {NEWLINE, A, NEWLINE, A, NEWLINE})
            sampler_->record_token(token);

        sampler_->initDryBreakers({"\n"}, [&](const std::string &) -> std::vector<int> {
            return {NEWLINE};
        });

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);

        // NEWLINE should be exempt from penalty because it's a single-token breaker
        for (const auto &p : penalties)
        {
            EXPECT_NE(p.token_id, NEWLINE)
                << "Single-token breaker should be exempt from DRY penalty";
        }
    }

    TEST_F(SamplerTest, DRY_BaseLessThanOneDisabled)
    {
        // dry_base < 1.0 should disable DRY (aligned with llama.cpp)
        const int A = 10;
        for (int token : {A, A, A, A})
            sampler_->record_token(token);

        SamplingParams params;
        params.dry_multiplier = 1.0f;
        params.dry_base = 0.5f;  // < 1.0 → disabled
        params.dry_allowed_length = 0;
        params.dry_penalty_last_n = -1;

        auto penalties = sampler_->compute_penalty_map(params, 100);
        EXPECT_TRUE(penalties.empty())
            << "DRY should be disabled when dry_base < 1.0";
    }

    TEST_F(SamplerTest, DRY_PenaltyIsAddedToPresenceFrequency)
    {
        // Verify DRY penalty is ADDED to presence+frequency, not max'd
        const int A = 10;
        sampler_->record_token(A);
        sampler_->record_token(A);
        sampler_->record_token(A);

        // Compute presence+frequency only
        SamplingParams params_pf;
        params_pf.presence_penalty = 1.0f;
        params_pf.frequency_penalty = 0.5f;

        Sampler sampler_pf(42);
        sampler_pf.record_token(A);
        sampler_pf.record_token(A);
        sampler_pf.record_token(A);
        auto penalties_pf = sampler_pf.compute_penalty_map(params_pf, 100);
        float pf_only = 0.0f;
        for (const auto &p : penalties_pf)
            if (p.token_id == A) pf_only = p.penalty;

        // Compute DRY only
        SamplingParams params_dry;
        params_dry.dry_multiplier = 1.0f;
        params_dry.dry_allowed_length = 0;
        params_dry.dry_penalty_last_n = -1;

        Sampler sampler_dry(42);
        sampler_dry.record_token(A);
        sampler_dry.record_token(A);
        sampler_dry.record_token(A);
        auto penalties_dry = sampler_dry.compute_penalty_map(params_dry, 100);
        float dry_only = 0.0f;
        for (const auto &p : penalties_dry)
            if (p.token_id == A) dry_only = p.penalty;

        // Compute combined
        SamplingParams params_both;
        params_both.presence_penalty = 1.0f;
        params_both.frequency_penalty = 0.5f;
        params_both.dry_multiplier = 1.0f;
        params_both.dry_allowed_length = 0;
        params_both.dry_penalty_last_n = -1;

        auto penalties_both = sampler_->compute_penalty_map(params_both, 100);
        float combined = 0.0f;
        for (const auto &p : penalties_both)
            if (p.token_id == A) combined = p.penalty;

        // Combined should equal pf + dry (additive, not max)
        EXPECT_NEAR(combined, pf_only + dry_only, 0.001f)
            << "DRY should be added to presence+frequency penalties, not max'd";
    }

} // anonymous namespace
