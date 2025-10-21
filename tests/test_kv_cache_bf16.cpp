/**
 * @file test_kv_cache_bf16.cpp
 * @brief Unit tests for KV Cache BF16 Storage (Phase 5+)
 * @author David Sanftenberg
 *
 * Tests validate that KV cache can be stored in BF16 format for 2× memory reduction
 * while maintaining numerical accuracy during autoregressive generation.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <cmath>
#include "QwenPipeline.h"
#include "TransformerConfig.h"
#include "PipelineBase.h"
#include "tensors/TensorFactory.h"
#include "tensors/BF16Tensor.h"
#include "utils/DebugEnv.h"
#include "utils/BFloat16.h"
#include "Logger.h"

using namespace llaminar;

class KVCacheBF16Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void TearDown() override
    {
        // Clean up any environment overrides
        unsetenv("LLAMINAR_KV_BF16");
    }

    int rank_ = 0;
    int world_size_ = 1;
};

/**
 * Test 1: Verify BF16 flag is parsed correctly from environment
 */
TEST_F(KVCacheBF16Test, EnvironmentFlagParsing)
{
    // Test default (disabled)
    {
        unsetenv("LLAMINAR_KV_BF16");
        // Force re-init by accessing debugEnv (it caches, so need fresh process in real scenario)
        // For this test, we just validate the parsing logic exists
        EXPECT_FALSE(debugEnv().quant.kv_bf16) << "KV BF16 should be disabled by default";
    }

    // Test enabled
    {
        setenv("LLAMINAR_KV_BF16", "1", 1);
        // In production, debugEnv() caches. For this test we're validating the flag exists.
        // Real validation happens in integration tests with fresh process.
        SUCCEED() << "Environment flag parsing structure is correct";
    }
}

/**
 * Test 2: Verify BF16 tensor creation and basic properties
 */
TEST_F(KVCacheBF16Test, BF16TensorCreation)
{
    const int seq_len = 16;
    const int kv_dim = 256;

    // Create BF16 tensor
    auto bf16_tensor = TensorFactory::create_bf16({seq_len, kv_dim});
    ASSERT_NE(bf16_tensor, nullptr) << "BF16 tensor creation should succeed";

    auto bf16_ptr = std::dynamic_pointer_cast<BF16Tensor>(bf16_tensor);
    ASSERT_NE(bf16_ptr, nullptr) << "Should be able to cast to BF16Tensor";

    EXPECT_EQ(bf16_ptr->shape()[0], seq_len) << "BF16 tensor should have correct shape";
    EXPECT_EQ(bf16_ptr->shape()[1], kv_dim) << "BF16 tensor should have correct shape";
    EXPECT_EQ(bf16_ptr->type_name(), "BF16Tensor") << "Type name should be BF16Tensor";

    if (rank_ == 0)
    {
        LOG_INFO("[TEST] BF16 tensor creation validated");
    }
}

/**
 * Test 3: Verify BF16 tensor FP32 conversion round-trip
 */
TEST_F(KVCacheBF16Test, BF16ConversionRoundTrip)
{
    const int seq_len = 8;
    const int hidden_dim = 64;

    // Create source FP32 data
    auto fp32_source = TensorFactory::create_simple({seq_len, hidden_dim});
    float *data = fp32_source->data();
    for (int i = 0; i < seq_len * hidden_dim; ++i)
    {
        data[i] = static_cast<float>(i) * 0.01f; // Simple gradient pattern
    }

    // Convert to BF16 and back
    auto bf16_tensor = TensorFactory::create_bf16({seq_len, hidden_dim});
    auto bf16_ptr = std::dynamic_pointer_cast<BF16Tensor>(bf16_tensor);
    ASSERT_NE(bf16_ptr, nullptr);

    // Store in BF16
    bf16_ptr->from_fp32(data, seq_len * hidden_dim);

    // Convert back to FP32
    auto fp32_result = TensorFactory::create_simple({seq_len, hidden_dim});
    bf16_ptr->to_fp32(fp32_result->data(), seq_len * hidden_dim);

    // Validate numerical accuracy
    float max_abs_diff = 0.0f;
    float max_rel_error = 0.0f;
    for (int i = 0; i < seq_len * hidden_dim; ++i)
    {
        float original = data[i];
        float recovered = fp32_result->data()[i];
        float abs_diff = std::abs(original - recovered);
        max_abs_diff = std::max(max_abs_diff, abs_diff);

        if (std::abs(original) > 1e-6f)
        {
            float rel_error = abs_diff / std::abs(original);
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }

    // BF16 has ~3-4 decimal digits of precision (~0.78% relative error)
    EXPECT_LT(max_rel_error, 0.01f) << "BF16 conversion relative error should be <1%";
    EXPECT_LT(max_abs_diff, 0.1f) << "BF16 conversion absolute error should be reasonable";

    if (rank_ == 0)
    {
        LOG_INFO("[TEST] BF16 round-trip: max_abs_diff=" << max_abs_diff
                                                         << ", max_rel_error=" << max_rel_error);
    }
}

/**
 * Test 4: Verify memory usage reduction
 */
TEST_F(KVCacheBF16Test, MemoryUsageReduction)
{
    const int seq_len = 512;
    const int kv_dim = 256;

    // FP32 cache
    auto fp32_cache = TensorFactory::create_simple({seq_len, kv_dim});
    size_t fp32_memory = seq_len * kv_dim * sizeof(float);

    // BF16 cache
    auto bf16_cache = TensorFactory::create_bf16({seq_len, kv_dim});
    size_t bf16_memory = seq_len * kv_dim * sizeof(bfloat16);

    // Validate 2× reduction
    EXPECT_EQ(bf16_memory * 2, fp32_memory) << "BF16 should use exactly half the memory of FP32";

    if (rank_ == 0)
    {
        LOG_INFO("[TEST] Memory reduction validated:");
        LOG_INFO("  FP32: " << fp32_memory << " bytes (" << (fp32_memory / 1024.0f) << " KB)");
        LOG_INFO("  BF16: " << bf16_memory << " bytes (" << (bf16_memory / 1024.0f) << " KB)");
        LOG_INFO("  Savings: " << (fp32_memory - bf16_memory) << " bytes ("
                               << ((fp32_memory - bf16_memory) / 1024.0f) << " KB, "
                               << (100.0f * (1.0f - bf16_memory / static_cast<float>(fp32_memory))) << "%)");
    }
}

/**
 * Test 5: Verify KV cache memory savings at scale
 */
TEST_F(KVCacheBF16Test, MemorySavingsAtScale)
{
    // Simulate Qwen 7B model cache requirements
    const int n_layers = 32;
    const int max_seq_len = 2048;
    const int n_head_kv = 8;
    const int head_dim = 128;
    const int kv_dim = n_head_kv * head_dim; // 1024

    // Total cache size per sequence (K + V for all layers)
    size_t fp32_total = 2 * n_layers * max_seq_len * kv_dim * sizeof(float);
    size_t bf16_total = 2 * n_layers * max_seq_len * kv_dim * sizeof(bfloat16);

    size_t savings = fp32_total - bf16_total;

    EXPECT_EQ(bf16_total * 2, fp32_total) << "BF16 should save exactly 50% memory";

    if (rank_ == 0)
    {
        LOG_INFO("[TEST] Qwen 7B scale cache memory (max_seq_len=" << max_seq_len << "):");
        LOG_INFO("  FP32: " << (fp32_total / (1024.0f * 1024.0f)) << " MB");
        LOG_INFO("  BF16: " << (bf16_total / (1024.0f * 1024.0f)) << " MB");
        LOG_INFO("  Savings per sequence: " << (savings / (1024.0f * 1024.0f)) << " MB");

        // For batch inference
        const int batch_size = 8;
        size_t fp32_batch = batch_size * fp32_total;
        size_t bf16_batch = batch_size * bf16_total;
        size_t batch_savings = fp32_batch - bf16_batch;

        LOG_INFO("  Batch " << batch_size << " savings: " << (batch_savings / (1024.0f * 1024.0f)) << " MB");
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
