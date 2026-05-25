#include <gtest/gtest.h>
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "../mocks/MockBackend.h"

/**
 * @file Test__WeightVRAMPool.cpp
 * @brief Unit tests for planning, allocation, and staging cleanup in WeightVRAMPool.
 *
 * These tests exercise the pool without a real GPU backend where possible, and use
 * MockBackend for allocation-lifetime checks that verify staging can be released
 * without invalidating persistent prepared-weight pointers.
 */

namespace llaminar2
{

    // Q4_0/Q4_K typical: 16 payload bytes per block, symmetric, no emins
    static constexpr int kQ4PayloadBytes = 16;
    // Q2_K: 16 payload bytes per block, asymmetric, has emins
    static constexpr int kQ2KPayloadBytes = 16;

    TEST(Test__WeightVRAMPool, PlanSingleWeight)
    {
        WeightVRAMPool pool;
        pool.planWeight("attn_q", /*N=*/1024, /*K=*/1024,
                        kQ4PayloadBytes, /*is_asymmetric=*/false, /*has_emins=*/false,
                        /*raw_gguf_bytes=*/524288);

        EXPECT_GT(pool.totalPlannedBytes(), 0u);
        EXPECT_EQ(pool.numPlannedWeights(), 1u);
    }

    TEST(Test__WeightVRAMPool, PlanMultipleWeights)
    {
        WeightVRAMPool pool;
        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 262144);
        pool.planWeight("w2", 1024, 2048, kQ4PayloadBytes, false, false, 524288);
        pool.planWeight("w3", 256, 512, kQ4PayloadBytes, false, false, 131072);

        EXPECT_EQ(pool.numPlannedWeights(), 3u);
        EXPECT_GT(pool.totalPlannedBytes(), 0u);

        // Total should be at least the sum of individual payload + scales
        // w1: blocks_per_row=32, payload=32*512*16=262144, scales=32*512*2=32768
        // w2: blocks_per_row=64, payload=64*1024*16=1048576, scales=64*1024*2=131072
        // w3: blocks_per_row=16, payload=16*256*16=65536, scales=16*256*2=8192
        size_t min_expected = (262144 + 32768) + (1048576 + 131072) + (65536 + 8192);
        EXPECT_GE(pool.totalPlannedBytes(), min_expected);
    }

    TEST(Test__WeightVRAMPool, GetSlotBeforeAllocate)
    {
        WeightVRAMPool pool;
        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 262144);

        auto slot = pool.getSlot("w1");
        EXPECT_FALSE(slot.has_value());
    }

    TEST(Test__WeightVRAMPool, PlanAsymmetricWeight)
    {
        WeightVRAMPool pool;

        // First plan a symmetric weight for comparison
        WeightVRAMPool pool_sym;
        pool_sym.planWeight("w_sym", 512, 1024, kQ4PayloadBytes, false, false, 262144);
        size_t sym_bytes = pool_sym.totalPlannedBytes();

        // Now plan an asymmetric weight (same dimensions)
        pool.planWeight("w_asym", 512, 1024, kQ4PayloadBytes, true, false, 262144);
        size_t asym_bytes = pool.totalPlannedBytes();

        // Asymmetric should have more bytes due to mins region
        EXPECT_GT(asym_bytes, sym_bytes);
    }

    TEST(Test__WeightVRAMPool, PlanWithEmins)
    {
        WeightVRAMPool pool;

        WeightVRAMPool pool_no_emins;
        pool_no_emins.planWeight("w1", 512, 1024, kQ2KPayloadBytes, true, false, 262144);
        size_t no_emins_bytes = pool_no_emins.totalPlannedBytes();

        pool.planWeight("w1", 512, 1024, kQ2KPayloadBytes, true, true, 262144);
        size_t with_emins_bytes = pool.totalPlannedBytes();

        EXPECT_GT(with_emins_bytes, no_emins_bytes);
    }

    TEST(Test__WeightVRAMPool, AlignmentIsCorrect)
    {
        WeightVRAMPool pool;
        // Plan multiple weights to force multiple region offsets
        pool.planWeight("w1", 100, 96, kQ4PayloadBytes, true, true, 50000);
        pool.planWeight("w2", 200, 128, kQ4PayloadBytes, false, false, 80000);

        // Allocate without GPU backend — d_base_ will be nullptr but allocation succeeds
        ASSERT_TRUE(pool.allocate(nullptr, 0, 0));

        // Verify total is aligned
        EXPECT_EQ(pool.totalPlannedBytes() % 256, 0u);
    }

    TEST(Test__WeightVRAMPool, EmptyPlanHasZeroBytes)
    {
        WeightVRAMPool pool;
        EXPECT_EQ(pool.totalPlannedBytes(), 0u);
        EXPECT_EQ(pool.numPlannedWeights(), 0u);
    }

    TEST(Test__WeightVRAMPool, DuplicateNameThrows)
    {
        WeightVRAMPool pool;
        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 262144);

        EXPECT_THROW(
            pool.planWeight("w1", 256, 512, kQ4PayloadBytes, false, false, 131072),
            std::runtime_error);
    }

    TEST(Test__WeightVRAMPool, NumPlannedWeights)
    {
        WeightVRAMPool pool;
        EXPECT_EQ(pool.numPlannedWeights(), 0u);

        pool.planWeight("a", 64, 64, kQ4PayloadBytes, false, false, 1024);
        EXPECT_EQ(pool.numPlannedWeights(), 1u);

        pool.planWeight("b", 64, 64, kQ4PayloadBytes, false, false, 1024);
        EXPECT_EQ(pool.numPlannedWeights(), 2u);

        pool.planWeight("c", 64, 64, kQ4PayloadBytes, false, false, 1024);
        EXPECT_EQ(pool.numPlannedWeights(), 3u);
    }

    TEST(Test__WeightVRAMPool, AllocateAndGetSlot)
    {
        WeightVRAMPool pool;
        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 262144);

        ASSERT_TRUE(pool.allocate(nullptr, 0, 0));
        EXPECT_TRUE(pool.isAllocated());
        EXPECT_EQ(pool.deviceId(), 0);

        auto slot = pool.getSlot("w1");
        ASSERT_TRUE(slot.has_value());
        EXPECT_GT(slot->payload_bytes, 0u);
        EXPECT_EQ(slot->staging_bytes, 262144u);

        // Without GPU backend, pointers are nullptr (d_base_ is nullptr)
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        EXPECT_EQ(slot->d_native_vnni_payload, nullptr);
#endif
    }

    TEST(Test__WeightVRAMPool, GetSlotUnknownName)
    {
        WeightVRAMPool pool;
        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 262144);
        ASSERT_TRUE(pool.allocate(nullptr, 0, 0));

        auto slot = pool.getSlot("nonexistent");
        EXPECT_FALSE(slot.has_value());
    }

    TEST(Test__WeightVRAMPool, StagingRegionIncluded)
    {
        WeightVRAMPool pool;
        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 100000);
        pool.planWeight("w2", 256, 512, kQ4PayloadBytes, false, false, 200000);

        size_t bytes_no_staging = pool.totalPlannedBytes();

        // Allocate with staging
        ASSERT_TRUE(pool.allocate(nullptr, 0, 3));

        // Total should include staging: max(100000, 200000) * 3 = 600000
        EXPECT_GE(pool.totalPlannedBytes(), bytes_no_staging + 600000);
    }

    TEST(Test__WeightVRAMPool, ReleaseStagingKeepsWeightSlotValid)
    {
        test::MockBackend backend(DeviceType::ROCm);
        WeightVRAMPool pool;

        pool.planWeight("w1", 512, 1024, kQ4PayloadBytes, false, false, 100000);
        pool.planWeight("w2", 256, 512, kQ4PayloadBytes, false, false, 200000);
        const size_t weight_bytes = pool.totalPlannedBytes();

        ASSERT_TRUE(pool.allocate(&backend, 0, 3));
        ASSERT_EQ(backend.getAllocationCount(), 2u);
        EXPECT_GE(pool.totalPlannedBytes(), weight_bytes + 600000);
        EXPECT_NE(pool.getStagingSlot(0), nullptr);

        auto before = pool.getSlot("w1");
        ASSERT_TRUE(before.has_value());
        ASSERT_NE(before->d_native_vnni_payload, nullptr);
        auto *payload_before = before->d_native_vnni_payload;

        pool.releaseStaging();

        EXPECT_TRUE(pool.isAllocated());
        EXPECT_EQ(pool.getStagingSlot(0), nullptr);
        EXPECT_EQ(pool.stagingSlotCount(), 0);
        EXPECT_EQ(pool.maxStagingSlotBytes(), 0u);
        EXPECT_EQ(backend.getAllocationCount(), 1u);
        EXPECT_EQ(backend.getTotalAllocatedBytes(), weight_bytes);
        EXPECT_EQ(pool.totalPlannedBytes(), weight_bytes);

        auto after = pool.getSlot("w1");
        ASSERT_TRUE(after.has_value());
        EXPECT_EQ(after->d_native_vnni_payload, payload_before);
    }

} // namespace llaminar2
