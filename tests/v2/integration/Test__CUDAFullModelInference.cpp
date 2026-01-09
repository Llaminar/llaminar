/**
 * @file Test__CUDAFullModelInference.cpp
 * @brief End-to-end GPU inference integration tests
 *
 * **Purpose**: Validates complete GPU inference pipeline:
 * 1. Model weights transfer to GPU
 * 2. Forward pass executes entirely on GPU
 * 3. Token predictions match CPU reference (parity)
 *
 * **Test Strategy**:
 * - Load small model (qwen2.5-0.5b-instruct-q4_0.gguf)
 * - Run identical forward passes on CPU and GPU
 * - Compare logits and token predictions
 * - Use greedy sampling (temperature=0) for determinism
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Project headers - use same pattern as other CUDA tests
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "execution/DeviceContext.h"
#include "execution/InferenceRunnerFactory.h"
#include "execution/GraphOrchestrator.h"
#include "loaders/ModelContext.h"
#include "loaders/ModelLoader.h"
#include "utils/Logger.h"
#include "utils/Tokenizer.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include <cuda_runtime.h>
#endif

// Test utilities
#include "../utils/CUDATestUtils.h"

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <set>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::test::cuda;

// ============================================================================
// Test Constants
// ============================================================================

namespace
{
    // Model path - uses small model for fast testing
    constexpr const char *TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // Test prompts
    constexpr const char *TEST_PROMPT_SIMPLE = "Hello";
    constexpr const char *TEST_PROMPT_LONGER = "The capital of France is";

    // Tolerance for logit comparison
    constexpr float LOGIT_RTOL = 1e-3f; // Relative tolerance
    constexpr float LOGIT_ATOL = 1e-4f; // Absolute tolerance

    // Number of tokens to generate in multi-token test
    constexpr int MULTI_TOKEN_COUNT = 10;
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

        // Check if model exists
        model_path_ = TEST_MODEL_PATH;
        if (!std::filesystem::exists(model_path_))
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }
    }

    /**
     * @brief Result of createRunner - includes both runner and model context
     *        The model context must stay alive while the runner is in use!
     */
    struct RunnerWithContext
    {
        std::unique_ptr<IInferenceRunner> runner;
        std::shared_ptr<ModelContext> model_ctx;
    };

    /**
     * @brief Create inference runner for specified device
     * @param device_id DeviceId (CPU or GPU)
     * @return Runner with its associated model context (both must stay alive)
     */
    RunnerWithContext createRunner(DeviceId device_id)
    {
        RunnerWithContext result;

        // Load model - create new context for each runner
        result.model_ctx = ModelContext::create(model_path_);
        if (!result.model_ctx)
        {
            return result;
        }

        // Configure inference
        InferenceRunnerConfig config;
        config.batch_size = 1;
        config.max_seq_len = 512;
        config.activation_precision = ActivationPrecision::FP32;

        result.runner = createInferenceRunner(result.model_ctx, nullptr, device_id, config);
        return result;
    }

    /**
     * @brief Tokenize a prompt string using given model context
     * @param model_ctx Model context with tokenizer
     * @param prompt Text prompt
     * @return Vector of token IDs
     */
    std::vector<int> tokenize(std::shared_ptr<ModelContext> model_ctx, const std::string &prompt)
    {
        if (!model_ctx)
        {
            LOG_ERROR("No model context available for tokenization");
            return {};
        }
        auto tokenizer = createTokenizer(model_ctx);
        if (!tokenizer)
        {
            LOG_ERROR("Failed to create tokenizer");
            return {};
        }
        return tokenizer->encode(prompt, true, false); // add_bos=true, add_eos=false
    }

    /**
     * @brief Run forward pass and get logits
     * @param runner Inference runner
     * @param tokens Input token IDs
     * @return Vector of logits for last token position
     */
    std::vector<float> forward(IInferenceRunner *runner, const std::vector<int> &tokens)
    {
        // Run forward pass
        runner->forward(tokens.data(), static_cast<int>(tokens.size()));

        // Get vocabulary size
        int vocab_sz = runner->vocab_size();

        // Get logits (returns pointer to vocab_size floats)
        const float *logits_ptr = runner->logits();
        if (!logits_ptr)
        {
            LOG_ERROR("Failed to get logits from runner");
            return {};
        }

        // Copy logits to vector
        return std::vector<float>(logits_ptr, logits_ptr + vocab_sz);
    }

    /**
     * @brief Get top-K token predictions from logits
     * @param logits Vocabulary logits
     * @param k Number of top predictions
     * @return Vector of (token_id, logit) pairs sorted by logit descending
     */
    std::vector<std::pair<int, float>> getTopK(const std::vector<float> &logits, int k)
    {
        // Create index vector
        std::vector<int> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Partial sort to get top-K
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&logits](int a, int b)
                          { return logits[a] > logits[b]; });

        // Build result
        std::vector<std::pair<int, float>> result;
        result.reserve(k);
        for (int i = 0; i < k; ++i)
        {
            result.emplace_back(indices[i], logits[indices[i]]);
        }
        return result;
    }

    /**
     * @brief Compare two logit vectors with tolerance
     * @return true if within tolerance
     */
    bool compareLogits(const std::vector<float> &expected,
                       const std::vector<float> &actual,
                       float rtol = LOGIT_RTOL,
                       float atol = LOGIT_ATOL)
    {
        if (expected.size() != actual.size())
        {
            return false;
        }

        for (size_t i = 0; i < expected.size(); ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            float ref = std::abs(expected[i]);
            if (diff > atol + rtol * ref)
            {
                LOG_DEBUG("Logit mismatch at index " << i
                                                     << ": expected=" << expected[i]
                                                     << ", actual=" << actual[i]
                                                     << ", diff=" << diff);
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Generate tokens greedily
     * @param runner Inference runner
     * @param prompt Initial prompt tokens
     * @param n_tokens Number of tokens to generate
     * @return Generated token IDs (excluding prompt)
     */
    std::vector<int> generate(IInferenceRunner *runner,
                              const std::vector<int> &prompt,
                              int n_tokens)
    {
        std::vector<int> generated;
        generated.reserve(n_tokens);

        // Clear cache before starting generation
        runner->clear_cache();

        // Prefill: forward pass with full prompt
        runner->forward(prompt.data(), static_cast<int>(prompt.size()));

        // Get first token prediction
        int vocab_sz = runner->vocab_size();
        const float *logits_ptr = runner->logits();
        if (!logits_ptr)
        {
            LOG_ERROR("Failed to get logits during prefill");
            return generated;
        }

        // Greedy: select argmax
        int next_token = static_cast<int>(
            std::max_element(logits_ptr, logits_ptr + vocab_sz) - logits_ptr);
        generated.push_back(next_token);

        // Decode: forward pass with single new token at a time
        for (int i = 1; i < n_tokens; ++i)
        {
            // Incremental decode: pass only the new token (seq_len=1)
            runner->forward(&next_token, 1);

            // Get logits and vocab size
            logits_ptr = runner->logits();
            if (!logits_ptr)
            {
                LOG_ERROR("Failed to get logits during decode step " << i);
                break;
            }

            // Greedy: select argmax
            next_token = static_cast<int>(
                std::max_element(logits_ptr, logits_ptr + vocab_sz) - logits_ptr);

            generated.push_back(next_token);
        }

        return generated;
    }

    std::string model_path_;
};

// ============================================================================
// Basic Setup Tests
// ============================================================================

TEST_F(Test__CUDAFullModelInference, CanCreateCPURunner)
{
    // Test that CPU inference runner can be created
    DeviceId cpu_id = DeviceId::cpu();

    auto result = createRunner(cpu_id);
    ASSERT_NE(result.runner, nullptr) << "Failed to create CPU inference runner";
}

TEST_F(Test__CUDAFullModelInference, CanCreateGPURunner)
{
    // Test that GPU inference runner can be created
    auto result = createRunner(gpu_device_);
    ASSERT_NE(result.runner, nullptr) << "Failed to create GPU inference runner";
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(Test__CUDAFullModelInference, SingleTokenPrediction_MatchesCPU)
{
    // This is the primary parity test
    DeviceId cpu_id = DeviceId::cpu();

    // Create runners (each with its own model context that must stay alive)
    auto cpu_result = createRunner(cpu_id);
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr) << "Failed to create CPU runner";
    ASSERT_NE(gpu_result.runner, nullptr) << "Failed to create GPU runner";

    // Tokenize prompt (use CPU model context)
    auto tokens = tokenize(cpu_result.model_ctx, TEST_PROMPT_SIMPLE);
    ASSERT_GT(tokens.size(), 0) << "Tokenization failed";

    LOG_INFO("[Test] Running forward pass with " << tokens.size() << " tokens");

    // Run forward passes
    auto cpu_logits = forward(cpu_result.runner.get(), tokens);
    auto gpu_logits = forward(gpu_result.runner.get(), tokens);

    ASSERT_EQ(cpu_logits.size(), gpu_logits.size())
        << "Logit sizes don't match: CPU=" << cpu_logits.size()
        << ", GPU=" << gpu_logits.size();

    // Compare top-5 predictions
    auto cpu_top5 = getTopK(cpu_logits, 5);
    auto gpu_top5 = getTopK(gpu_logits, 5);

    LOG_INFO("[Test] CPU top-5 predictions:");
    for (const auto &[token, logit] : cpu_top5)
    {
        LOG_INFO("  Token " << token << ": " << logit);
    }

    LOG_INFO("[Test] GPU top-5 predictions:");
    for (const auto &[token, logit] : gpu_top5)
    {
        LOG_INFO("  Token " << token << ": " << logit);
    }

    // Primary check: top-1 token should match
    EXPECT_EQ(cpu_top5[0].first, gpu_top5[0].first)
        << "Top-1 token mismatch: CPU=" << cpu_top5[0].first
        << ", GPU=" << gpu_top5[0].first;

    // Secondary check: top-5 tokens should match (order may vary slightly)
    std::vector<int> cpu_top5_tokens, gpu_top5_tokens;
    for (const auto &[token, logit] : cpu_top5)
        cpu_top5_tokens.push_back(token);
    for (const auto &[token, logit] : gpu_top5)
        gpu_top5_tokens.push_back(token);

    std::sort(cpu_top5_tokens.begin(), cpu_top5_tokens.end());
    std::sort(gpu_top5_tokens.begin(), gpu_top5_tokens.end());

    EXPECT_EQ(cpu_top5_tokens, gpu_top5_tokens)
        << "Top-5 token sets don't match";

    // Tertiary check: logit values should be close
    // (Only check top-100 to avoid noise in low-probability tokens)
    auto cpu_top100 = getTopK(cpu_logits, 100);
    auto gpu_top100 = getTopK(gpu_logits, 100);

    int mismatches = 0;
    for (int i = 0; i < 100; ++i)
    {
        float cpu_val = cpu_top100[i].second;
        float gpu_val = gpu_top100[i].second;
        float diff = std::abs(cpu_val - gpu_val);
        float ref = std::abs(cpu_val);

        if (diff > LOGIT_ATOL + LOGIT_RTOL * ref)
        {
            ++mismatches;
            if (mismatches <= 5)
            {
                LOG_WARN("[Test] Logit mismatch at rank " << i
                                                          << ": CPU=" << cpu_val
                                                          << ", GPU=" << gpu_val
                                                          << ", diff=" << diff);
            }
        }
    }

    EXPECT_LE(mismatches, 10)
        << "Too many logit mismatches in top-100: " << mismatches;
}

TEST_F(Test__CUDAFullModelInference, LongerPrompt_MatchesCPU)
{
    // Test with a longer prompt to exercise more of the pipeline
    DeviceId cpu_id = DeviceId::cpu();

    auto cpu_result = createRunner(cpu_id);
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr);
    ASSERT_NE(gpu_result.runner, nullptr);

    // Tokenize longer prompt
    auto tokens = tokenize(cpu_result.model_ctx, TEST_PROMPT_LONGER);
    ASSERT_GT(tokens.size(), 3) << "Longer prompt should have multiple tokens";

    LOG_INFO("[Test] Testing with " << tokens.size() << " token prompt");

    // Run forward passes
    auto cpu_logits = forward(cpu_result.runner.get(), tokens);
    auto gpu_logits = forward(gpu_result.runner.get(), tokens);

    // Compare top-1
    auto cpu_top1 = getTopK(cpu_logits, 1)[0];
    auto gpu_top1 = getTopK(gpu_logits, 1)[0];

    EXPECT_EQ(cpu_top1.first, gpu_top1.first)
        << "Top-1 mismatch on longer prompt: CPU=" << cpu_top1.first
        << ", GPU=" << gpu_top1.first;
}

// ============================================================================
// Multi-Token Generation Tests
// ============================================================================

TEST_F(Test__CUDAFullModelInference, MultiTokenGeneration_MatchesCPU)
{

    // Test that multi-token generation produces identical sequences
    DeviceId cpu_id = DeviceId::cpu();

    auto cpu_result = createRunner(cpu_id);
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr);
    ASSERT_NE(gpu_result.runner, nullptr);

    // Tokenize prompt
    auto prompt_tokens = tokenize(cpu_result.model_ctx, TEST_PROMPT_LONGER);

    LOG_INFO("[Test] Generating " << MULTI_TOKEN_COUNT << " tokens...");

    // Generate on CPU and GPU
    auto cpu_generated = generate(cpu_result.runner.get(), prompt_tokens, MULTI_TOKEN_COUNT);
    auto gpu_generated = generate(gpu_result.runner.get(), prompt_tokens, MULTI_TOKEN_COUNT);

    ASSERT_EQ(cpu_generated.size(), gpu_generated.size())
        << "Generated sequence lengths don't match";

    // Compare sequences
    int mismatches = 0;
    for (size_t i = 0; i < cpu_generated.size(); ++i)
    {
        if (cpu_generated[i] != gpu_generated[i])
        {
            ++mismatches;
            LOG_WARN("[Test] Token mismatch at position " << i
                                                          << ": CPU=" << cpu_generated[i]
                                                          << ", GPU=" << gpu_generated[i]);
        }
    }

    EXPECT_EQ(mismatches, 0)
        << "Generated sequences differ at " << mismatches << " positions";

    // Log the generated sequences for debugging
    LOG_INFO("[Test] CPU generated: ");
    for (int tok : cpu_generated)
        LOG_INFO("  " << tok);
    LOG_INFO("[Test] GPU generated: ");
    for (int tok : gpu_generated)
        LOG_INFO("  " << tok);
}

// ============================================================================
// Memory and Performance Tests
// ============================================================================

TEST_F(Test__CUDAFullModelInference, GPUMemoryUsage)
{
    // Test that GPU memory is properly allocated and doesn't leak
    auto result = createRunner(gpu_device_);
    ASSERT_NE(result.runner, nullptr);

    // Get initial memory
    size_t initial_free = 0, initial_total = 0;
#ifdef HAVE_CUDA
    cudaMemGetInfo(&initial_free, &initial_total);
#endif

    // Run a few forward passes
    auto tokens = tokenize(result.model_ctx, TEST_PROMPT_SIMPLE);
    for (int i = 0; i < 5; ++i)
    {
        forward(result.runner.get(), tokens);
    }

    // Check memory hasn't grown unexpectedly
    size_t final_free = 0, final_total = 0;
#ifdef HAVE_CUDA
    cudaMemGetInfo(&final_free, &final_total);
#endif

    // Allow some variance but flag major leaks
    size_t used_initial = initial_total - initial_free;
    size_t used_final = initial_total - final_free;

    LOG_INFO("[Test] GPU memory: initial=" << (used_initial / 1024 / 1024) << "MB"
                                           << ", final=" << (used_final / 1024 / 1024) << "MB");

    // Memory shouldn't grow more than 10% after warmup
    EXPECT_LE(used_final, used_initial * 1.1 + 10 * 1024 * 1024)
        << "Possible GPU memory leak detected";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__CUDAFullModelInference, SingleTokenPrompt)
{
    // Test with minimal single-token input
    DeviceId cpu_id = DeviceId::cpu();

    auto cpu_result = createRunner(cpu_id);
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr);
    ASSERT_NE(gpu_result.runner, nullptr);

    // Single token (BOS or similar)
    std::vector<int> single_token = {1}; // Assume token ID 1 exists

    auto cpu_logits = forward(cpu_result.runner.get(), single_token);
    auto gpu_logits = forward(gpu_result.runner.get(), single_token);

    auto cpu_top1 = getTopK(cpu_logits, 1)[0];
    auto gpu_top1 = getTopK(gpu_logits, 1)[0];

    EXPECT_EQ(cpu_top1.first, gpu_top1.first)
        << "Single token prediction mismatch";
}

// ============================================================================
// Diagnostic Test: Stage-by-Stage Comparison
// ============================================================================

/**
 * @brief Compute cosine similarity between two float vectors
 */
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
 * @brief Compute max absolute error between two float vectors
 */
static float maxAbsError(const float *a, const float *b, size_t n)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

TEST_F(Test__CUDAFullModelInference, DiagnosticStageByStageComparison)
{
    // This test captures snapshots at each stage for both CPU and GPU
    // and compares them to identify where divergence begins

    DeviceId cpu_id = DeviceId::cpu();

    auto cpu_result = createRunner(cpu_id);
    auto gpu_result = createRunner(gpu_device_);

    ASSERT_NE(cpu_result.runner, nullptr) << "Failed to create CPU runner";
    ASSERT_NE(gpu_result.runner, nullptr) << "Failed to create GPU runner";

    // Enable snapshot capture on both runners
    cpu_result.runner->enableSnapshotCapture();
    gpu_result.runner->enableSnapshotCapture();

    // Tokenize prompt
    auto tokens = tokenize(cpu_result.model_ctx, TEST_PROMPT_SIMPLE);
    ASSERT_GT(tokens.size(), 0) << "Tokenization failed";

    LOG_INFO("[Diagnostic] Running forward passes with " << tokens.size() << " tokens");

    // Run CPU forward
    cpu_result.runner->forward(tokens.data(), static_cast<int>(tokens.size()));
    auto cpu_keys = cpu_result.runner->getSnapshotKeys();

    // Run GPU forward
    gpu_result.runner->forward(tokens.data(), static_cast<int>(tokens.size()));
    auto gpu_keys = gpu_result.runner->getSnapshotKeys();

    LOG_INFO("[Diagnostic] CPU captured " << cpu_keys.size() << " snapshots");
    LOG_INFO("[Diagnostic] GPU captured " << gpu_keys.size() << " snapshots");

    // Sort keys for consistent ordering
    std::sort(cpu_keys.begin(), cpu_keys.end());
    std::sort(gpu_keys.begin(), gpu_keys.end());

    // Print header
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    STAGE-BY-STAGE CPU vs GPU COMPARISON                          ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Stage Key                           │ Elements  │ Cosine Sim │ Max Abs Err │ OK? ║\n";
    std::cout << "╠═════════════════════════════════════╪═══════════╪════════════╪═════════════╪═════╣\n";

    // Find common keys and compare
    std::set<std::string> all_keys(cpu_keys.begin(), cpu_keys.end());
    all_keys.insert(gpu_keys.begin(), gpu_keys.end());

    int pass_count = 0;
    int fail_count = 0;
    std::string first_failure_key;
    float first_failure_cosine = 1.0f;

    for (const auto &key : all_keys)
    {
        // Filter to layer 0 and layer 1 only for brevity (or all if small model)
        bool is_layer_0_or_1 = (key.find("layer0") != std::string::npos ||
                                key.find("layer1") != std::string::npos ||
                                key.find("EMBEDDING") != std::string::npos ||
                                key.find("FINAL") != std::string::npos ||
                                key.find("LM_HEAD") != std::string::npos);

        if (!is_layer_0_or_1)
            continue;

        size_t cpu_size = 0, gpu_size = 0;
        const float *cpu_data = cpu_result.runner->getSnapshot(key, cpu_size);
        const float *gpu_data = gpu_result.runner->getSnapshot(key, gpu_size);

        // Format key for display (truncate if too long)
        std::string display_key = key;
        if (display_key.length() > 35)
        {
            display_key = display_key.substr(0, 32) + "...";
        }

        if (!cpu_data || !gpu_data)
        {
            std::cout << "║ " << std::left << std::setw(35) << display_key
                      << " │ " << std::setw(9) << (cpu_data ? cpu_size : 0)
                      << " │ " << std::setw(10) << "MISSING"
                      << " │ " << std::setw(11) << "-"
                      << " │ " << std::setw(3) << "❌" << " ║\n";
            ++fail_count;
            continue;
        }

        if (cpu_size != gpu_size)
        {
            std::cout << "║ " << std::left << std::setw(35) << display_key
                      << " │ " << std::setw(9) << std::to_string(cpu_size) + "/" + std::to_string(gpu_size)
                      << " │ " << std::setw(10) << "SIZE DIFF"
                      << " │ " << std::setw(11) << "-"
                      << " │ " << std::setw(3) << "❌" << " ║\n";
            ++fail_count;
            continue;
        }

        float cosine = cosineSimilarity(cpu_data, gpu_data, cpu_size);
        float max_err = maxAbsError(cpu_data, gpu_data, cpu_size);

        // Consider pass if cosine > 0.999 and max_err < 0.1
        bool is_pass = (cosine > 0.999f && max_err < 0.1f);

        std::cout << "║ " << std::left << std::setw(35) << display_key
                  << " │ " << std::right << std::setw(9) << cpu_size
                  << " │ " << std::fixed << std::setprecision(6) << std::setw(10) << cosine
                  << " │ " << std::scientific << std::setprecision(3) << std::setw(11) << max_err
                  << " │ " << std::setw(3) << (is_pass ? "✓" : "❌") << " ║\n";

        if (is_pass)
        {
            ++pass_count;
        }
        else
        {
            ++fail_count;
            if (first_failure_key.empty())
            {
                first_failure_key = key;
                first_failure_cosine = cosine;
            }
        }
    }

    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ SUMMARY: " << pass_count << " passed, " << fail_count << " failed";
    if (!first_failure_key.empty())
    {
        std::cout << " (first failure: " << first_failure_key << ")";
    }
    std::cout << std::string(std::max(0, 52 - static_cast<int>(first_failure_key.length())), ' ') << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════╝\n\n";

    // This test is diagnostic - it reports results but doesn't fail
    // Uncomment the following to make it fail if divergence detected:
    // EXPECT_EQ(fail_count, 0) << "Stage divergence detected at: " << first_failure_key;

    // For now, just record what we found
    if (!first_failure_key.empty())
    {
        LOG_WARN("[Diagnostic] First divergence detected at: " << first_failure_key
                                                               << " (cosine=" << first_failure_cosine << ")");
    }
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
        GTEST_SKIP() << "No CUDA devices available - skipping GPU inference tests";
    }

    // If we get here, CUDA is available
    SUCCEED();
}
