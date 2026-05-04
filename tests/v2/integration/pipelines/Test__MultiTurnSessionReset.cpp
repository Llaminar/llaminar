/**
 * @file Test__MultiTurnSessionReset.cpp
 * @brief Integration test for multi-turn inference with session resets
 * @author David Sanftenberg
 *
 * Validates that clear_cache() correctly resets kernel dynamic state
 * so that a second inference request produces correct output (not garbage
 * from stale token IDs cached in embedding kernel GPU buffers).
 *
 * Parameterized across backends: CPU, CUDA, ROCm. Each GPU backend is
 * skipped if the corresponding hardware is not available.
 *
 * Regression coverage for:
 *   - Embedding stale dynamic_params_active_ after clearCache() (eeca83dd)
 *   - Graph cache invalidation on clear_cache() (eeca83dd)
 *   - KernelFactory::resetAllDynamicState() lifecycle (8666332f)
 *   - Session epoch reset (8666332f)
 *   - Stale activation buffer K/V in decode after graph cache reuse
 *     (AttentionComputeStage decode-mode KV cache override)
 *
 * Requires: models/qwen2.5-0.5b-instruct-q4_0.gguf
 */

#include <gtest/gtest.h>
#include <mpi.h>
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
// Backend Parameterization
// ============================================================================

enum class TestBackend
{
    CPU,
    CUDA,
    ROCm
};

std::string backendName(const ::testing::TestParamInfo<TestBackend> &info)
{
    switch (info.param)
    {
    case TestBackend::CPU:
        return "CPU";
    case TestBackend::CUDA:
        return "CUDA";
    case TestBackend::ROCm:
        return "ROCm";
    }
    return "Unknown";
}

// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class Test__MultiTurnSessionReset : public ::testing::TestWithParam<TestBackend>
{
protected:
    static constexpr const char *MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    std::unique_ptr<IOrchestrationRunner> runner_;

    DeviceId deviceForBackend()
    {
        switch (GetParam())
        {
        case TestBackend::CUDA:
            return DeviceId::cuda(0);
        case TestBackend::ROCm:
            return DeviceId::rocm(0);
        default:
            return DeviceId::cpu();
        }
    }

    void SetUp() override
    {
        if (!std::ifstream(MODEL_PATH).good())
        {
            GTEST_SKIP() << "Model file not found: " << MODEL_PATH;
        }

        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        // Skip if requested backend hardware is not available
        if (GetParam() == TestBackend::CUDA && dm.cuda_device_count() == 0)
            GTEST_SKIP() << "No CUDA GPU available";
        if (GetParam() == TestBackend::ROCm && dm.rocm_device_count() == 0)
            GTEST_SKIP() << "No ROCm GPU available";

        runner_ = TestOrchestrationHelper::createSimple(
            MODEL_PATH, deviceForBackend(), 512);

        ASSERT_NE(runner_, nullptr) << "Failed to create runner";
        ASSERT_TRUE(runner_->initialize()) << "Failed to initialize: " << runner_->lastError();

        // Greedy sampling for deterministic output
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

        return !has_nan && !has_inf && (max_val - min_val) > 0.1f;
    }
};

// ============================================================================
// Tests — parameterized across CPU, CUDA, ROCm
// ============================================================================

// Regression: second request after clearCache() must produce valid output.
// Root cause (eeca83dd): embedding kernel's dynamic_params_active_ remained
// true after clear_cache(), causing stale GPU-side token IDs to be used
// instead of re-uploading the new prompt.
TEST_P(Test__MultiTurnSessionReset, R2_After_ClearCache_ProducesValidOutput)
{
    std::vector<int32_t> prompt_r1 = {9707}; // "Hello"
    std::vector<int32_t> prompt_r2 = {3838}; // "Hi"

    auto tokens_r1 = runInference(prompt_r1, 5);
    ASSERT_FALSE(tokens_r1.empty()) << "R1 produced no tokens";
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()))
        << "R1 logits are invalid";

    runner_->clearCache();

    auto tokens_r2 = runInference(prompt_r2, 5);
    ASSERT_FALSE(tokens_r2.empty()) << "R2 produced no tokens after clearCache()";
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()))
        << "R2 logits are invalid (stale dynamic state bug)";
}

// Regression: same prompt after clearCache() must produce identical tokens.
// Verifies the full reset path: KernelFactory::resetAllDynamicState() →
// embedding kernel resetDynamicState() → memcmp guard forces H2D re-upload.
TEST_P(Test__MultiTurnSessionReset, R1_Repeat_After_ClearCache_IsDeterministic)
{
    std::vector<int32_t> prompt = {9707}; // "Hello"

    auto tokens_r1a = runInference(prompt, 5);
    ASSERT_FALSE(tokens_r1a.empty());

    int vocab = runner_->vocabSize();
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), vocab));

    runner_->clearCache();

    auto tokens_r1b = runInference(prompt, 5);
    ASSERT_FALSE(tokens_r1b.empty())
        << "Same prompt after clearCache() should still generate tokens";
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), vocab))
        << "Logits after clearCache() + re-run should be valid";
    ASSERT_EQ(tokens_r1a, tokens_r1b)
        << "Same prompt with greedy sampling must produce identical tokens after clearCache()";
}

// Regression: three consecutive sessions all produce valid output.
// Tests that arena_.reset() + re-initialization, graph cache invalidation,
// and KV cache clearing all work correctly across multiple session boundaries.
TEST_P(Test__MultiTurnSessionReset, Three_Consecutive_Sessions_AllValid)
{
    std::vector<int32_t> prompt1 = {9707};  // "Hello"
    std::vector<int32_t> prompt2 = {3838};  // "Hi"
    std::vector<int32_t> prompt3 = {25402}; // "Tell"

    auto tokens1 = runInference(prompt1, 3);
    ASSERT_FALSE(tokens1.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));

    runner_->clearCache();

    auto tokens2 = runInference(prompt2, 3);
    ASSERT_FALSE(tokens2.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));

    runner_->clearCache();

    auto tokens3 = runInference(prompt3, 3);
    ASSERT_FALSE(tokens3.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));
}

// Regression: different-length prompts across sessions must be deterministic.
// Root cause: the forward graph cache reuses a graph built during prefill
// (when cached_tokens=0, K/V wired to activation scratch buffers) for decode
// steps (when the attention needs full KV history from the KV cache).
// Fix: AttentionComputeStage/FusedAttentionWoStage override K/V from KV
// cache at execute time in decode mode.
TEST_P(Test__MultiTurnSessionReset, DifferentLengthPrompts_AcrossSessions)
{
    // Short prompt (1 token)
    std::vector<int32_t> short_prompt = {9707};
    // Longer prompt (3 tokens)
    std::vector<int32_t> long_prompt = {9707, 3838, 25402};

    // Session 1: Short prompt
    auto tokens_short = runInference(short_prompt, 3);
    ASSERT_FALSE(tokens_short.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));

    runner_->clearCache();

    // Session 2: Long prompt (different seq_len exercises graph cache paths)
    auto tokens_long = runInference(long_prompt, 3);
    ASSERT_FALSE(tokens_long.empty());
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()));

    runner_->clearCache();

    // Session 3: Short prompt again — must match session 1
    auto tokens_short2 = runInference(short_prompt, 3);
    ASSERT_FALSE(tokens_short2.empty())
        << "Short prompt after long prompt session must still generate tokens";
    ASSERT_TRUE(logitsAreValid(runner_->lastLogits(), runner_->vocabSize()))
        << "Logits must be valid after different-length session";
    EXPECT_EQ(tokens_short, tokens_short2)
        << "Same prompt with greedy sampling must produce identical tokens across sessions";
}

// ============================================================================
// Instantiate for all backends
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    Test__MultiTurnSessionReset,
    ::testing::Values(TestBackend::CPU, TestBackend::CUDA, TestBackend::ROCm),
    backendName);

// ============================================================================
// Custom main() — MPI must be initialized before the runner factory is called
// ============================================================================

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();

    // Use _exit() to skip static destructors — CUDA/ROCm driver cleanup
    // races with MPI teardown, causing hangs or SIGABRT on process exit.
    _exit(result);
}
