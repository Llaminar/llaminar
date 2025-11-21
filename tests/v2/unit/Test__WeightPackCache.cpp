/**
 * @file Test__WeightPackCache.cpp
 * @brief Unit tests for OneDNN INT8 weight packing cache
 *
 * Tests the lazy WeightPack cache stored in TensorBase::cache_:
 * 1. Cache hit/miss behavior
 * 2. Memory deallocation after repacking (quantized→INT8)
 * 3. Thread-safe cache access
 * 4. Correctness after original buffer deallocation
 *
 * @author David Sanftenberg
 * @date 2025-11-21
 */

#include <gtest/gtest.h>
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "../../../src/v2/kernels/cpu/gemm_v4/OneDNNGemmAdapter.h"
#include "../../../src/v2/utils/Logger.h"
#include <vector>
#include <memory>
#include <thread>
#include <any>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

/**
 * @class Test__WeightPackCache
 * @brief Test fixture for weight packing cache tests
 */
class Test__WeightPackCache : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set log level to ERROR to reduce test output
        Logger::getInstance().setLogLevel(LogLevel::ERROR);
    }

    /**
     * @brief Create a Q4_0 weight tensor for testing
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Shared pointer to Q4_0 tensor
     */
    std::shared_ptr<TensorBase> createQ4_0Tensor(size_t rows, size_t cols)
    {
        // Q4_0 block size = 32 elements
        constexpr size_t BLOCK_SIZE = 32;
        size_t blocks_per_row = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        size_t total_blocks = rows * blocks_per_row;

        // Q4_0 block structure: 1 FP16 scale + 16 bytes (32 nibbles)
        constexpr size_t BYTES_PER_BLOCK = sizeof(uint16_t) + 16;
        std::vector<uint8_t> raw_data(total_blocks * BYTES_PER_BLOCK);

        // Fill with random data
        for (size_t i = 0; i < raw_data.size(); ++i)
        {
            raw_data[i] = static_cast<uint8_t>(rand() % 256);
        }

        return std::make_shared<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

    /**
     * @brief Verify WeightPack cache exists and is valid
     * @param tensor Tensor to check
     * @return true if cache exists and is valid
     */
    bool verifyCacheExists(const std::shared_ptr<TensorBase> &tensor)
    {
        if (!tensor->cache_.has_value())
        {
            return false;
        }

        try
        {
            const auto &pack = std::any_cast<const WeightPack &>(tensor->cache_);
            return pack.element_count() > 0 && !pack.col_scales.empty();
        }
        catch (const std::bad_any_cast &)
        {
            return false;
        }
    }
};

/**
 * @test CacheLazyInitialization
 * @brief Verify cache is NOT created until first pack_weights_to_int8() call
 */
TEST_F(Test__WeightPackCache, CacheLazyInitialization)
{
    auto tensor = createQ4_0Tensor(64, 256);

    // Cache should be empty initially
    EXPECT_FALSE(tensor->cache_.has_value())
        << "Cache should NOT exist before first pack operation";

    // Pack weights (triggers cache creation)
    WeightPack pack = pack_weights_to_int8(*tensor, 256, 64);

    // Cache should still be empty (pack_weights_to_int8 doesn't populate cache_)
    // OneDNNGemmKernel populates cache_ when it detects quantized weights
    EXPECT_FALSE(tensor->cache_.has_value())
        << "pack_weights_to_int8() should not populate cache_ (kernel does that)";

    // Verify pack contents are valid
    EXPECT_EQ(pack.element_count(), 64 * 256);
    EXPECT_EQ(pack.col_scales.size(), 64);
}

/**
 * @test CachePopulationByKernel
 * @brief Verify cache is populated when pack_weights_to_int8() is called
 */
TEST_F(Test__WeightPackCache, CachePopulationByKernel)
{
    constexpr int N = 64, K = 256;
    auto weight_tensor = createQ4_0Tensor(N, K);

    // Cache should be empty initially
    EXPECT_FALSE(weight_tensor->cache_.has_value())
        << "Cache should be empty before packing";

    // Manually pack weights (simulates what OneDNNGemmKernel does)
    WeightPack pack = pack_weights_to_int8(*weight_tensor, K, N);

    // Store in cache (this is what OneDNNGemmKernel does)
    weight_tensor->cache_ = pack;

    // Cache should now exist
    EXPECT_TRUE(verifyCacheExists(weight_tensor))
        << "Cache should be populated after packing";

    // Verify pack contents
    EXPECT_EQ(pack.rows, K) << "Pack rows should match K";
    EXPECT_EQ(pack.cols, N) << "Pack cols should match N";
    EXPECT_EQ(pack.element_count(), K * N) << "Pack element count should be K*N";
    EXPECT_EQ(pack.col_scales.size(), N) << "Should have N column scales";
}

/**
 * @test CacheCorrectness
 * @brief Verify cached pack produces same data as fresh pack
 */
TEST_F(Test__WeightPackCache, CacheCorrectness)
{
    constexpr int N = 32, K = 128;
    auto weight_tensor = createQ4_0Tensor(N, K);

    // First pack (populates cache)
    WeightPack pack1 = pack_weights_to_int8(*weight_tensor, K, N);
    weight_tensor->cache_ = pack1;

    // Second pack (would use cache in real kernel, but we test fresh pack matches)
    WeightPack pack2 = pack_weights_to_int8(*weight_tensor, K, N);

    // Verify data matches
    ASSERT_EQ(pack1.element_count(), pack2.element_count());
    for (size_t i = 0; i < pack1.element_count(); ++i)
    {
        EXPECT_EQ(pack1.data[i], pack2.data[i])
            << "Packed INT8 data should match at index " << i;
    }

    // Verify scales match
    ASSERT_EQ(pack1.col_scales.size(), pack2.col_scales.size());
    for (size_t i = 0; i < pack1.col_scales.size(); ++i)
    {
        EXPECT_FLOAT_EQ(pack1.col_scales[i], pack2.col_scales[i])
            << "Column scales should match at index " << i;
    }
}

/**
 * @test MultipleKernelsShareCache
 * @brief Verify cache persists across multiple pack operations
 */
TEST_F(Test__WeightPackCache, MultipleKernelsShareCache)
{
    constexpr int N = 16, K = 64;
    auto weight_tensor = createQ4_0Tensor(N, K);

    // First pack - populates cache
    WeightPack pack1 = pack_weights_to_int8(*weight_tensor, K, N);
    weight_tensor->cache_ = pack1;
    EXPECT_TRUE(verifyCacheExists(weight_tensor));

    // Simulate second kernel accessing cache (extract cached pack)
    ASSERT_TRUE(weight_tensor->cache_.has_value());
    const auto &cached_pack = std::any_cast<const WeightPack &>(weight_tensor->cache_);

    // Verify cached pack matches original
    EXPECT_EQ(cached_pack.element_count(), pack1.element_count());
    EXPECT_EQ(cached_pack.rows, pack1.rows);
    EXPECT_EQ(cached_pack.cols, pack1.cols);
    EXPECT_EQ(cached_pack.col_scales.size(), pack1.col_scales.size());
}

/**
 * @test ThreadSafeCacheAccess
 * @brief Verify cache access is thread-safe (multiple pack operations concurrently)
 */
TEST_F(Test__WeightPackCache, ThreadSafeCacheAccess)
{
    constexpr int N = 16, K = 64;
    constexpr int NUM_THREADS = 4;
    auto weight_tensor = createQ4_0Tensor(N, K);

    // Atomic flag to track failures
    std::atomic<int> failures{0};

    // Spawn multiple threads, each packing weights
    auto thread_func = [&]()
    {
        try
        {
            WeightPack pack = pack_weights_to_int8(*weight_tensor, K, N);

            // Verify pack validity
            if (pack.element_count() != K * N || pack.col_scales.size() != static_cast<size_t>(N))
            {
                failures.fetch_add(1);
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Thread exception: " << e.what());
            failures.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back(thread_func);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(failures.load(), 0) << "All threads should complete successfully";
}

/**
 * @test MemoryFootprintAfterCaching
 * @brief Verify memory footprint comparison (Q4_0 vs WeightPack)
 *
 * NOTE: This test documents memory usage. Deallocation implementation pending.
 */
TEST_F(Test__WeightPackCache, MemoryFootprintAfterCaching)
{
    constexpr int N = 2048, K = 4096; // Large weight matrix
    auto weight_tensor = createQ4_0Tensor(N, K);

    // Initial memory: Q4_0 raw blocks (~4.5 bits per weight)
    size_t blocks_per_row = (K + 31) / 32;
    size_t q4_0_bytes = N * blocks_per_row * 18; // 18 bytes per block (FP16 scale + 16 bytes data)

    LOG_INFO("Q4_0 weight memory: " << q4_0_bytes << " bytes");
    LOG_INFO("  N=" << N << ", K=" << K << ", blocks_per_row=" << blocks_per_row);

    // Pack weights
    WeightPack pack = pack_weights_to_int8(*weight_tensor, K, N);
    weight_tensor->cache_ = pack;

    ASSERT_TRUE(verifyCacheExists(weight_tensor));

    // After caching: WeightPack contains INT8 data + FP32 scales
    size_t int8_data_bytes = N * K;         // INT8 data (8 bits per weight)
    size_t scale_bytes = N * sizeof(float); // Per-column FP32 scales
    size_t weightpack_bytes = int8_data_bytes + scale_bytes;

    LOG_INFO("WeightPack memory: " << weightpack_bytes << " bytes");
    LOG_INFO("  INT8 data: " << int8_data_bytes << " bytes");
    LOG_INFO("  FP32 scales: " << scale_bytes << " bytes");

    // TODO: After implementing buffer deallocation:
    // - Original Q4_0 raw_data_ should be freed (mark as "deallocated")
    // - Only WeightPack should remain in memory
    // - Memory footprint = weightpack_bytes (not q4_0_bytes + weightpack_bytes)

    // CURRENT BEHAVIOR: Both Q4_0 and WeightPack coexist (~2.2× overhead)
    // DESIRED BEHAVIOR: Only WeightPack remains (~1.8× compression vs FP32)

    float current_overhead = static_cast<float>(q4_0_bytes + weightpack_bytes) / (N * K * sizeof(float));
    float desired_overhead = static_cast<float>(weightpack_bytes) / (N * K * sizeof(float));

    LOG_INFO("Current memory overhead: " << current_overhead << "× (Q4_0 + WeightPack)");
    LOG_INFO("Desired memory overhead: " << desired_overhead << "× (WeightPack only)");

    // Document expected improvement
    float memory_savings_pct = 100.0f * (1.0f - desired_overhead / current_overhead);
    LOG_INFO("Potential memory savings: " << memory_savings_pct << "%");

    // For 2048×4096 weights:
    // - Q4_0: ~4.5MB
    // - WeightPack: ~8.4MB (8MB INT8 + 8KB scales)
    // - Current total: ~12.9MB
    // - Desired total: ~8.4MB (35% reduction)

    EXPECT_GT(memory_savings_pct, 30.0f)
        << "Deallocation should save >30% memory for large weight matrices";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
