/**
 * @file Test__MultiTurnSessionReset.cpp
 * @brief Integration test for multi-turn inference with session resets
 * @author David Sanftenberg
 *
 * Validates that clear_cache() correctly resets kernel dynamic state
 * so that a second inference request produces correct output (not garbage
 * from stale token IDs cached in embedding kernel GPU buffers).
 *
 * This test exercises the full inference pipeline:
 *   1. Create OrchestrationRunner with a real model
 *   2. Run prefill + decode for request R1
 *   3. Call clearCache() (simulates session boundary)
 *   4. Run prefill + decode for request R2
 *   5. Verify R2 output is valid (not NaN/Inf, reasonable logit distribution)
 *   6. Run R1 again and verify same output as step 2 (determinism)
 *
 * Requires: models/qwen2.5-0.5b-instruct-q4_0.gguf
 */

#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>

#include "utils/TestOrchestrationHelper.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"
#include "utils/Sampler.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__MultiTurnSessionReset : public ::testing::Test
{
protected:
    static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    std::unique_ptr<IOrchestrationRunner> runner_;

    void SetUp() override
    {
        // Skip if model file doesn't exist
        if (!std::ifstream(MODEL_PATH).good())
        {
            GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
        }

        DeviceManager::instance().initialize(-1);

        runner_ = TestOrchestrationHelper::createSimple(
            MODEL_PATH, DeviceId::cpu(), 512);

        ASSERT_NE(runner_, nullptr) << "Failed to create runner";
        ASSERT_TRUE(runner_->initialize()) << "Failed to initialize: " << runner_->lastError();

        // Use greedy sampling (temperature=0) for deterministic output
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        runner_->setSamplingParams(greedy);
    }

    void TearDown() override
    {
        if (runner_)
        {
            runner_->shutdown();
        }
    }

    // Helper: run prefill + N decode steps, return generated token IDs
    std::vector<int32_t> runInference(const std::vector<int32_t> &prompt_tokens, int decode_steps)
    {
        EXPECT_TRUE(runner_->prefill(prompt_tokens));

        std::vector<int32_t> generated;
        for (int i = 0; i < decode_steps; ++i)
        {
            auto result = runner_->decodeStep();
            if (!result.success() || result.is_complete)
                break;
            if (!result.tokens.empty())
                generated.push_back(result.tokens[0]);
        }
        return generated;
    }

    // Helper: check logits are valid (no NaN/Inf, reasonable range)
    bool logitsAreValid(const float *logits, int vocab_size)
    {
        if (!logits)
            return false;

        bool has_nan = false;
        bool has_inf = false;
        float max_val = -1e30f;
        float min_val = 1e30f;

        for (int i = 0; i < vocab_size; ++i)
        {
            if (std::isnan(logits[i]))
                has_nan = true;
            if (std::isinf(logits[i]))
                has_inf = true;
            max_val = std::max(max_val, logits[i]);
            min_val = std::min(min_val, logits[i]);
        }

        // Valid logits should have no NaN/Inf and a reasonable dynamic range
        return !has_nan && !has_inf && (max_val - min_val) > 0.1f;
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(Test__MultiTurnSessionReset, R2_After_ClearCache_ProducesValidOutput)
{
    // Typical chat tokens (small prompt)
    // "Hello" in Qwen2.5 tokenizer
    std::vector<int32_t> prompt_r1 = {9707}; // "Hello"
    std::vector<int32_t> prompt_r2 = {3838}; // "Hi"

    // R1: first request
    auto tokens_r1 = runInference(prompt_r1, 5);
    ASSERT_FALSE(tokens_r1.empty()) << "R1 produced no tokens";

    const float *logits_r1 = runner_->lastLogits();
    ASSERT_TRUE(logitsAreValid(logits_r1, runner_->vocabSize()))
        << "R1 logits are invalid";

    // Session boundary
    runner_->clearCache();

    // R2: second request (the historically buggy case)
    auto tokens_r2 = runInference(prompt_r2, 5);
    ASSERT_FALSE(tokens_r2.empty()) << "R2 produced no tokens after clearCache()";

    const float *logits_r2 = runner_->lastLogits();
    ASSERT_TRUE(logitsAreValid(logits_r2, runner_->vocabSize()))
        << "R2 logits are invalid (stale dynamic state bug)";
}

TEST_F(Test__MultiTurnSessionReset, R1_Repeat_After_ClearCache_ProducesValidOutput)
{
    std::vector<int32_t> prompt = {9707}; // "Hello"

    // R1: first request
    auto tokens_r1a = runInference(prompt, 5);
    ASSERT_FALSE(tokens_r1a.empty());

    int vocab = runner_->vocabSize();
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), vocab));

    // Session boundary
    runner_->clearCache();

    // R1 repeat: same prompt after clear — must produce identical tokens
    // (greedy sampling + deterministic GEMM on CPU = exact match)
    auto tokens_r1b = runInference(prompt, 5);
    ASSERT_FALSE(tokens_r1b.empty())
        << "Same prompt after clearCache() should still generate tokens";
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), vocab))
        << "Logits after clearCache() + re-run should be valid";
    ASSERT_EQ(tokens_r1a, tokens_r1b)
        << "Same prompt with greedy sampling must produce identical tokens after clearCache()";
}

TEST_F(Test__MultiTurnSessionReset, Three_Consecutive_Sessions_AllValid)
{
    std::vector<int32_t> prompt1 = {9707};  // "Hello"
    std::vector<int32_t> prompt2 = {3838};  // "Hi"
    std::vector<int32_t> prompt3 = {25402}; // "Tell"

    // Session 1
    auto tokens1 = runInference(prompt1, 3);
    ASSERT_FALSE(tokens1.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));

    runner_->clearCache();

    // Session 2
    auto tokens2 = runInference(prompt2, 3);
    ASSERT_FALSE(tokens2.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));

    runner_->clearCache();

    // Session 3
    auto tokens3 = runInference(prompt3, 3);
    ASSERT_FALSE(tokens3.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));
}
