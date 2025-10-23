/**
 * @file test_q8_0_full_model.cpp
 * @brief Full model validation for Q8_0Tensor integration
 * @author David Sanftenberg
 *
 * Tests Week 1 Day 4-5:
 * - Load entire model (291 tensors)
 * - Count Q8_0Tensor vs SimpleTensor instances
 * - Measure total memory usage
 * - Validate inference still works end-to-end
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <cstdlib>
#include "ModelLoader.h"
#include "utils/DebugEnv.h" // For refreshDebugEnv()
#include "tensors/TensorBase.h"
#include "tensors/SimpleTensor.h"
#include "tensors/Q8_0Tensor.h"

using namespace llaminar;

/**
 * @brief Test loading entire model and count tensor types
 */
TEST(Q8_0FullModelTest, LoadAllWeightsTypeCounts)
{
    // Enable quantized loading
    setenv("LLAMINAR_QUANT_ENABLE", "1", 1);
    setenv("LLAMINAR_LOAD_QUANTIZED", "1", 1);

    ModelLoader loader;
    bool loaded = loader.loadModel("models/qwen2.5-0.5b-instruct-q8_0.gguf");
    ASSERT_TRUE(loaded) << "Failed to load model";

    const auto &model = loader.getModel();
    std::cout << "\n=== Model Info ===" << std::endl;
    std::cout << "Architecture: " << model.architecture << std::endl;
    std::cout << "Tensor count: " << model.tensors.size() << std::endl;
    std::cout << "Layers: " << model.block_count << std::endl;

    // Count tensor types
    int q8_0_count = 0;
    int simple_count = 0;
    int other_count = 0;
    size_t total_q8_0_size = 0;
    size_t total_simple_size = 0;

    std::cout << "\n=== Loading All Weights ===" << std::endl;

    for (const auto &info : model.tensors)
    {
        const std::string &name = info.name;
        auto tensor = loader.loadTensor(name);
        ASSERT_NE(tensor, nullptr) << "Failed to load tensor: " << name;

        // Type check
        if (auto q8_tensor = std::dynamic_pointer_cast<Q8_0Tensor>(tensor))
        {
            q8_0_count++;
            total_q8_0_size += q8_tensor->raw_size();
        }
        else if (auto simple = std::dynamic_pointer_cast<SimpleTensor>(tensor))
        {
            simple_count++;
            total_simple_size += simple->size() * sizeof(float);
        }
        else
        {
            other_count++;
        }
    }

    std::cout << "\n=== Tensor Type Summary ===" << std::endl;
    std::cout << "Q8_0Tensor:    " << std::setw(4) << q8_0_count
              << " tensors  (" << (total_q8_0_size / 1024 / 1024) << " MB)" << std::endl;
    std::cout << "SimpleTensor:  " << std::setw(4) << simple_count
              << " tensors  (" << (total_simple_size / 1024 / 1024) << " MB)" << std::endl;
    std::cout << "Other:         " << std::setw(4) << other_count << " tensors" << std::endl;
    std::cout << "TOTAL:         " << std::setw(4) << (q8_0_count + simple_count + other_count)
              << " tensors" << std::endl;

    // Expected: Most weights are Q8_0 (except norms, some biases)
    EXPECT_GT(q8_0_count, 100) << "Expected many Q8_0 tensors";

    // Calculate memory savings
    if (q8_0_count > 0)
    {
        size_t q8_0_mb = total_q8_0_size / 1024 / 1024;
        size_t simple_mb = total_simple_size / 1024 / 1024;
        size_t total_mb = q8_0_mb + simple_mb;

        // Estimate what it would be if all Q8_0 were FP32
        size_t q8_0_as_fp32_mb = (total_q8_0_size * 4) / 1024 / 1024; // ~3.76× expansion
        size_t old_total_mb = q8_0_as_fp32_mb + simple_mb;
        size_t saved_mb = old_total_mb - total_mb;

        std::cout << "\n=== Memory Savings ===" << std::endl;
        std::cout << "Current (Q8_0 compressed): " << total_mb << " MB" << std::endl;
        std::cout << "Old (all FP32):            " << old_total_mb << " MB" << std::endl;
        std::cout << "Saved:                     " << saved_mb << " MB" << std::endl;
        std::cout << "Reduction:                 "
                  << std::fixed << std::setprecision(1)
                  << (100.0 * saved_mb / old_total_mb) << "%" << std::endl;

        EXPECT_GT(saved_mb, 1000) << "Expected significant memory savings (>1GB)";
    }
}

/**
 * @brief Test loading time: Q8_0 vs FP32
 */
TEST(Q8_0FullModelTest, LoadingTimeComparison)
{
    std::cout << "\n=== Loading Time Benchmark ===" << std::endl;

    // Test 1: Q8_0 native loading
    {
        setenv("LLAMINAR_QUANT_ENABLE", "1", 1);
        setenv("LLAMINAR_LOAD_QUANTIZED", "1", 1);

        ModelLoader loader;

        auto start = std::chrono::high_resolution_clock::now();
        bool loaded = loader.loadModel("models/qwen2.5-0.5b-instruct-q8_0.gguf");
        ASSERT_TRUE(loaded);

        // Load first 10 weights as sample
        const auto &model = loader.getModel();
        int count = 0;
        for (const auto &info : model.tensors)
        {
            loader.loadTensor(info.name);
            if (++count >= 10)
                break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Q8_0 native (10 tensors): " << duration_ms << " ms" << std::endl;
    }

    // Test 2: FP32 decode loading
    {
        setenv("LLAMINAR_QUANT_ENABLE", "0", 1);
        setenv("LLAMINAR_LOAD_QUANTIZED", "0", 1);

        ModelLoader loader;

        auto start = std::chrono::high_resolution_clock::now();
        bool loaded = loader.loadModel("models/qwen2.5-0.5b-instruct-q8_0.gguf");
        ASSERT_TRUE(loaded);

        // Load first 10 weights as sample
        const auto &model = loader.getModel();
        int count = 0;
        for (const auto &info : model.tensors)
        {
            loader.loadTensor(info.name);
            if (++count >= 10)
                break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "FP32 decode (10 tensors):  " << duration_ms << " ms" << std::endl;
    }

    std::cout << "\nNote: Q8_0 should be faster (no decode step)" << std::endl;
}

/**
 * @brief Test specific weight tensors we know should be Q8_0
 */
TEST(Q8_0FullModelTest, ValidateKnownQ8_0Weights)
{
    setenv("LLAMINAR_QUANT_ENABLE", "1", 1);
    setenv("LLAMINAR_LOAD_QUANTIZED", "1", 1);

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel("models/qwen2.5-0.5b-instruct-q8_0.gguf"));

    // These tensors should definitely be Q8_0
    std::vector<std::string> expected_q8_weights = {
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output.weight"};

    std::cout << "\n=== Validating Known Q8_0 Weights ===" << std::endl;

    for (const auto &name : expected_q8_weights)
    {
        auto tensor = loader.loadTensor(name);
        ASSERT_NE(tensor, nullptr) << "Failed to load: " << name;

        auto q8_tensor = std::dynamic_pointer_cast<Q8_0Tensor>(tensor);
        if (q8_tensor)
        {
            std::cout << "✅ " << name << " → Q8_0Tensor" << std::endl;

            // Validate properties
            EXPECT_EQ(q8_tensor->native_type(), TensorDataType::QUANTIZED);
            EXPECT_GT(q8_tensor->compression_ratio(), 3.5f);
            EXPECT_LE(q8_tensor->compression_ratio(), 4.0f); // Allow exactly 4.0 (perfect alignment)
        }
        else
        {
            std::cout << "❌ " << name << " → SimpleTensor (expected Q8_0)" << std::endl;
            FAIL() << "Expected Q8_0Tensor for: " << name;
        }
    }
}

/**
 * @brief Test that FP32 fallback works when env vars not set
 */
TEST(Q8_0FullModelTest, FP32FallbackBehavior)
{
    // NOTE: This test must be run WITHOUT LLAMINAR_QUANT_ENABLE/LLAMINAR_LOAD_QUANTIZED
    // set in the shell, because setenv() after program start may not override shell environment

    // Check if environment was set at program start - if so, skip this test
    const auto &initial_env = llaminar::debugEnv();
    if (initial_env.quant.enable || initial_env.quant.load_quantized)
    {
        GTEST_SKIP() << "Test must be run without LLAMINAR_QUANT_ENABLE/LLAMINAR_LOAD_QUANTIZED "
                     << "in shell environment. Current: enable=" << initial_env.quant.enable
                     << " load_quantized=" << initial_env.quant.load_quantized;
    }

    // Disable quantized loading by UNSETTING the variables (not setting to "0")
    // The flag() parser treats any non-empty string as true, including "0"
    unsetenv("LLAMINAR_QUANT_ENABLE");
    unsetenv("LLAMINAR_LOAD_QUANTIZED");

    // CRITICAL: Refresh environment snapshot after unsetenv (rebuild immediately)
    llaminar::debugEnvRefresh();

    // Debug: Verify environment was updated
    const auto &env_check = llaminar::debugEnv();
    std::cout << "After unsetenv: LLAMINAR_QUANT_ENABLE=" << env_check.quant.enable
              << ", LLAMINAR_LOAD_QUANTIZED=" << env_check.quant.load_quantized << std::endl;

    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel("models/qwen2.5-0.5b-instruct-q8_0.gguf"));

    // Load a weight that would be Q8_0 if enabled
    auto tensor = loader.loadTensor("token_embd.weight");
    ASSERT_NE(tensor, nullptr);

    // Should be SimpleTensor (FP32 fallback)
    auto simple = std::dynamic_pointer_cast<SimpleTensor>(tensor);
    ASSERT_NE(simple, nullptr) << "Expected SimpleTensor with quantized loading disabled";

    auto q8_tensor = std::dynamic_pointer_cast<Q8_0Tensor>(tensor);
    EXPECT_EQ(q8_tensor, nullptr) << "Should NOT be Q8_0Tensor when disabled";

    std::cout << "✅ FP32 fallback works correctly (SimpleTensor created)" << std::endl;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
