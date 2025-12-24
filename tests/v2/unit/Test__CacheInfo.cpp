/**
 * @file Test__CacheInfo.cpp
 * @brief Unit tests for CPU cache detection and batch size calculation
 */

#include <gtest/gtest.h>
#include "v2/utils/CPUFeatures.h"

namespace llaminar2::test
{

    class Test__CacheInfo : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // CacheInfo is a singleton, just get reference
        }
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Cache Detection Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__CacheInfo, L1CacheSizeIsReasonable)
    {
        uint32_t l1 = cpu_l1_cache_size();
        // L1 data cache is typically 32KB or 48KB
        EXPECT_GE(l1, 16 * 1024) << "L1 cache should be at least 16KB";
        EXPECT_LE(l1, 128 * 1024) << "L1 cache should be at most 128KB";
    }

    TEST_F(Test__CacheInfo, L2CacheSizeIsReasonable)
    {
        uint32_t l2 = cpu_l2_cache_size();
        // L2 cache is typically 256KB to 2MB per core
        EXPECT_GE(l2, 128 * 1024) << "L2 cache should be at least 128KB";
        EXPECT_LE(l2, 4 * 1024 * 1024) << "L2 cache should be at most 4MB per core";
    }

    TEST_F(Test__CacheInfo, L3CacheSizeIsReasonable)
    {
        uint32_t l3 = cpu_l3_cache_size();
        // L3 cache is typically 4MB to 64MB shared
        EXPECT_GE(l3, 2 * 1024 * 1024) << "L3 cache should be at least 2MB";
        EXPECT_LE(l3, 256 * 1024 * 1024) << "L3 cache should be at most 256MB";
    }

    TEST_F(Test__CacheInfo, CacheHierarchyIsOrdered)
    {
        uint32_t l1 = cpu_l1_cache_size();
        uint32_t l2 = cpu_l2_cache_size();
        uint32_t l3 = cpu_l3_cache_size();

        EXPECT_LT(l1, l2) << "L1 should be smaller than L2";
        EXPECT_LT(l2, l3) << "L2 should be smaller than L3";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CacheInfo Struct Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__CacheInfo, CacheInfoSingletonWorks)
    {
        const CacheInfo &info1 = cache_info();
        const CacheInfo &info2 = cache_info();

        // Should return same cached values
        EXPECT_EQ(info1.l1_size, info2.l1_size);
        EXPECT_EQ(info1.l2_size, info2.l2_size);
        EXPECT_EQ(info1.l3_size, info2.l3_size);
    }

    TEST_F(Test__CacheInfo, CacheInfoMatchesDirectCalls)
    {
        const CacheInfo &info = cache_info();

        EXPECT_EQ(info.l1_size, cpu_l1_cache_size());
        EXPECT_EQ(info.l2_size, cpu_l2_cache_size());
        EXPECT_EQ(info.l3_size, cpu_l3_cache_size());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Optimal Batch Size Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__CacheInfo, OptimalWoBatchSizeIsPositive)
    {
        const CacheInfo &info = cache_info();

        // Test with various model sizes
        EXPECT_GT(info.optimal_wo_batch_size(896), 0);  // Qwen2-0.5B
        EXPECT_GT(info.optimal_wo_batch_size(3584), 0); // Qwen2-7B
        EXPECT_GT(info.optimal_wo_batch_size(8192), 0); // Llama-70B
    }

    TEST_F(Test__CacheInfo, OptimalWoBatchSizeRespectsLimits)
    {
        const CacheInfo &info = cache_info();

        // Test with default limits (min=2, max=16)
        int batch = info.optimal_wo_batch_size(3584);
        EXPECT_GE(batch, 2) << "Batch size should be at least min_batch";
        EXPECT_LE(batch, 16) << "Batch size should be at most max_batch";
    }

    TEST_F(Test__CacheInfo, OptimalWoBatchSizeRespectsCustomLimits)
    {
        const CacheInfo &info = cache_info();

        // Test with custom limits
        int batch = info.optimal_wo_batch_size(3584, 0.25f, 4, 8);
        EXPECT_GE(batch, 4) << "Batch size should be at least custom min_batch";
        EXPECT_LE(batch, 8) << "Batch size should be at most custom max_batch";
    }

    TEST_F(Test__CacheInfo, OptimalWoBatchSizeIsPowerOfTwo)
    {
        const CacheInfo &info = cache_info();

        // The implementation rounds down to power of 2
        int batch = info.optimal_wo_batch_size(3584);
        bool is_power_of_two = (batch & (batch - 1)) == 0;
        EXPECT_TRUE(is_power_of_two) << "Batch size " << batch << " should be power of 2";
    }

    TEST_F(Test__CacheInfo, LargerModelGetsSmallerOrEqualBatch)
    {
        const CacheInfo &info = cache_info();

        // Larger models have larger context buffers, so may get smaller batch
        int batch_small = info.optimal_wo_batch_size(896);  // 3.5KB context
        int batch_large = info.optimal_wo_batch_size(8192); // 32KB context

        EXPECT_GE(batch_small, batch_large)
            << "Smaller model should get >= batch size of larger model";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Optimal KV Tile Size Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__CacheInfo, OptimalKvTileSizeIsReasonable)
    {
        const CacheInfo &info = cache_info();

        int tile = info.optimal_kv_tile_size(128);
        EXPECT_GE(tile, 4) << "KV tile should be at least 4";
        EXPECT_LE(tile, 16) << "KV tile should be at most 16";
    }

    TEST_F(Test__CacheInfo, OptimalKvTileSizeIsPowerOfTwo)
    {
        const CacheInfo &info = cache_info();

        int tile = info.optimal_kv_tile_size(128);
        bool is_power_of_two = (tile & (tile - 1)) == 0;
        EXPECT_TRUE(is_power_of_two) << "KV tile " << tile << " should be power of 2";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helper Method Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__CacheInfo, FitsInCacheMethods)
    {
        const CacheInfo &info = cache_info();

        // Small buffer should fit in all caches
        EXPECT_TRUE(info.fits_l1(1024));
        EXPECT_TRUE(info.fits_l2(1024));
        EXPECT_TRUE(info.fits_l3(1024));

        // Large buffer shouldn't fit in L1
        size_t large = 1024 * 1024; // 1MB
        EXPECT_FALSE(info.fits_l1(large));
        EXPECT_TRUE(info.fits_l2(large) || info.fits_l3(large));
    }

    TEST_F(Test__CacheInfo, SummaryIsNotEmpty)
    {
        const CacheInfo &info = cache_info();
        const char *summary = info.summary();

        EXPECT_NE(summary, nullptr);
        EXPECT_GT(strlen(summary), 0);
        EXPECT_NE(strstr(summary, "L1"), nullptr) << "Summary should mention L1";
        EXPECT_NE(strstr(summary, "L2"), nullptr) << "Summary should mention L2";
    }

} // namespace llaminar2::test
