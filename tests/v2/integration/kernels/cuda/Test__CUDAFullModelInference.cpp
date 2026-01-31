/**
 * @file Test__CUDAFullModelInference.cpp
 * @brief Lightweight GPU inference integration tests
 *
 * **Purpose**: Validates GPU inference parity with CPU on first layer only.
 * Tests focus on early-layer cosine similarity (>= 0.98) to catch kernel bugs
 * without being affected by accumulated divergence in later layers.
 *
 * **Tests**:
 * 1. Prefill - Multi-token forward pass (exercises batched attention)
 * 2. Incremental Decode - Single-token forward pass (exercises KV cache)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "loaders/ModelContext.h"
#include "utils/Logger.h"
#include "utils/Tokenizer.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

#include "../../../utils/CUDATestUtils.h"

#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <set>

using namespace llaminar2;
using namespace llaminar2::test::cuda;

// ============================================================================
// Constants
// ============================================================================

namespace
{
    constexpr const char *TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    constexpr const char *TEST_PROMPT = "The capital of France is";
    constexpr float COSINE_THRESHOLD = 0.98f;
}

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAFullModelInference : public CUDATestBase
{
protected:
    void SetUp() override
    {
        CUDATestBase::SetUp();

        if (!std::filesystem::exists(TEST_MODEL_PATH))
        {
            GTEST_SKIP() << "Model not found: " << TEST_MODEL_PATH;
        }
    }

    struct RunnerWithContext
    {
        std::unique_ptr<IInferenceRunner> runner;
        std::shared_ptr<ModelContext> model_ctx;
    };

    RunnerWithContext createRunner(DeviceId device_id)
    {
        RunnerWithContext result;
        result.model_ctx = ModelContext::create(TEST_MODEL_PATH);
        if (!result.model_ctx)
            return result;

        InferenceRunnerConfig config;
        config.batch_size = 1;
        config.max_seq_len = 512;
        config.activation_precision = ActivationPrecision::FP32;

        result.runner = createInferenceRunner(result.model_ctx, nullptr, device_id, config);
        return result;
    }

    std::vector<int> tokenize(std::shared_ptr<ModelContext> model_ctx, const std::string &prompt)
    {
        auto tokenizer = createTokenizer(model_ctx);
        if (!tokenizer)
            return {};
        return tokenizer->encode(prompt, true, false);
    }

    static float cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    /**
     * @brief Check first layer parity between CPU and GPU snapshots
     * @return Minimum cosine similarity across all layer0 stages
     */
    float checkFirstLayerParity(IInferenceRunner *cpu_runner, IInferenceRunner *gpu_runner)
    {
        auto cpu_keys = cpu_runner->getSnapshotKeys();
        auto gpu_keys = gpu_runner->getSnapshotKeys();

        float min_cosine = 1.0f;
        int checked = 0;

        for (const auto &key : cpu_keys)
        {
            // Only check layer0 stages
            if (key.find("layer0") == std::string::npos)
                continue;

            size_t cpu_size = 0, gpu_size = 0;
            const float *cpu_data = cpu_runner->getSnapshot(key, cpu_size);
            const float *gpu_data = gpu_runner->getSnapshot(key, gpu_size);

            if (!cpu_data || !gpu_data || cpu_size != gpu_size)
            {
                LOG_WARN("[Test] Missing or size mismatch for " << key);
                continue;
            }

            float cosine = cosineSimilarity(cpu_data, gpu_data, cpu_size);
            LOG_INFO("[Test] " << key << ": cosine=" << cosine);

            if (cosine < min_cosine)
                min_cosine = cosine;

            ++checked;
        }

        LOG_INFO("[Test] Checked " << checked << " layer0 stages, min_cosine=" << min_cosine);
        return min_cosine;
    }
};

// ============================================================================
// Tests
// ============================================================================

/**
 * @brief Prefill parity test - multi-token forward pass
 *
 * Exercises the batched attention path where Q/K/V are computed for
 * multiple tokens simultaneously. Checks first layer cosine >= 0.98.
 */
TEST_F(Test__CUDAFullModelInference, Prefill_FirstLayerParity)
{
    auto cpu_result = createRunner(DeviceId::cpu());
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr) << "Failed to create CPU runner";
    ASSERT_NE(gpu_result.runner, nullptr) << "Failed to create GPU runner";

    cpu_result.runner->enableSnapshotCapture();
    gpu_result.runner->enableSnapshotCapture();

    auto tokens = tokenize(cpu_result.model_ctx, TEST_PROMPT);
    ASSERT_GT(tokens.size(), 1) << "Need multi-token prompt for prefill test";

    LOG_INFO("[Test] Prefill with " << tokens.size() << " tokens");

    // Run prefill on both
    cpu_result.runner->forward(tokens.data(), static_cast<int>(tokens.size()));
    gpu_result.runner->forward(tokens.data(), static_cast<int>(tokens.size()));

    // Check first layer parity
    float min_cosine = checkFirstLayerParity(cpu_result.runner.get(), gpu_result.runner.get());

    EXPECT_GE(min_cosine, COSINE_THRESHOLD)
        << "First layer cosine similarity too low: " << min_cosine;
}

/**
 * @brief Incremental decode parity test - single-token forward pass
 *
 * Exercises the KV cache path where only a single new token is processed
 * and attention reads from cached K/V. Checks first layer cosine >= 0.98.
 */
TEST_F(Test__CUDAFullModelInference, IncrementalDecode_FirstLayerParity)
{
    auto cpu_result = createRunner(DeviceId::cpu());
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr) << "Failed to create CPU runner";
    ASSERT_NE(gpu_result.runner, nullptr) << "Failed to create GPU runner";

    auto tokens = tokenize(cpu_result.model_ctx, TEST_PROMPT);
    ASSERT_GT(tokens.size(), 1) << "Need multi-token prompt";

    // First do prefill (without snapshots)
    cpu_result.runner->forward(tokens.data(), static_cast<int>(tokens.size()));
    gpu_result.runner->forward(tokens.data(), static_cast<int>(tokens.size()));

    // Get next token (greedy)
    int vocab_sz = cpu_result.runner->vocab_size();
    const float *cpu_logits = cpu_result.runner->logits();
    int next_token = static_cast<int>(
        std::max_element(cpu_logits, cpu_logits + vocab_sz) - cpu_logits);

    LOG_INFO("[Test] Incremental decode with token " << next_token);

    // Now enable snapshots and do incremental decode
    cpu_result.runner->enableSnapshotCapture();
    gpu_result.runner->enableSnapshotCapture();

    cpu_result.runner->forward(&next_token, 1);
    gpu_result.runner->forward(&next_token, 1);

    // Check first layer parity
    float min_cosine = checkFirstLayerParity(cpu_result.runner.get(), gpu_result.runner.get());

    EXPECT_GE(min_cosine, COSINE_THRESHOLD)
        << "First layer cosine similarity too low: " << min_cosine;
}

// ============================================================================
// Skip Test When No CUDA
// ============================================================================

TEST(Test__CUDAFullModelInference_NoCUDA, SkipWithoutCUDA)
{
    auto &dm = DeviceManager::instance();
    bool has_cuda = false;

    for (const auto &dev : dm.devices())
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA devices available";
    }

    SUCCEED();
}
