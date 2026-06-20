#include <gtest/gtest.h>
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "../mocks/MockBackend.h"

/**
 * @file Test__LoadOrchestrator.cpp
 * @brief Unit tests for GPU weight loading orchestration and staging finalization.
 *
 * The tests cover planning, VRAM preflight accounting, and the finalize() contract
 * that releases temporary staging resources while preserving persistent pool state.
 */

#include <cstdlib>

namespace llaminar2
{

    namespace
    {
        static constexpr size_t kMiB = 1024ULL * 1024ULL;

        class BudgetMockBackend : public test::MockBackend
        {
        public:
            BudgetMockBackend(size_t total_bytes, size_t free_bytes)
                : test::MockBackend(DeviceType::ROCm), total_bytes_(total_bytes), free_bytes_(free_bytes)
            {
            }

            void *allocate(size_t bytes, int device_id) override
            {
                ++allocate_calls_;
                last_allocate_bytes_ = bytes;
                return test::MockBackend::allocate(bytes, device_id);
            }

            void *allocatePinned(size_t bytes, int device_id) override
            {
                (void)device_id;
                return std::malloc(bytes);
            }

            void freePinned(void *ptr, int device_id) override
            {
                (void)device_id;
                std::free(ptr);
            }

            size_t deviceMemoryTotal(int device_id) const override
            {
                (void)device_id;
                return total_bytes_;
            }

            size_t deviceMemoryFree(int device_id) const override
            {
                (void)device_id;
                return free_bytes_;
            }

            int allocateCalls() const { return allocate_calls_; }
            size_t lastAllocateBytes() const { return last_allocate_bytes_; }

        private:
            size_t total_bytes_ = 0;
            size_t free_bytes_ = 0;
            int allocate_calls_ = 0;
            size_t last_allocate_bytes_ = 0;
        };
    } // namespace

    static constexpr int kQ4PayloadBytes = 16;

    TEST(Test__LoadOrchestrator, AddDevice)
    {
        LoadOrchestrator orch;
        EXPECT_EQ(orch.numDevices(), 0u);

        orch.addDevice(0);
        EXPECT_EQ(orch.numDevices(), 1u);

        orch.addDevice(1);
        EXPECT_EQ(orch.numDevices(), 2u);
    }

    TEST(Test__LoadOrchestrator, PlanWeightForDevice)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);

        orch.planWeight(0, "attn_q", 1024, 1024, kQ4PayloadBytes,
                        false, false, 524288);

        auto *pool = orch.getPool(0);
        ASSERT_NE(pool, nullptr);
        EXPECT_EQ(pool->numPlannedWeights(), 1u);
        EXPECT_GT(pool->totalPlannedBytes(), 0u);
    }

    TEST(Test__LoadOrchestrator, GetPoolReturnsCorrect)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);

        auto *pool = orch.getPool(0);
        EXPECT_NE(pool, nullptr);
    }

    TEST(Test__LoadOrchestrator, GetPoolUnknownDevice)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);

        auto *pool = orch.getPool(99);
        EXPECT_EQ(pool, nullptr);

        // Const version too
        const auto &const_orch = orch;
        EXPECT_EQ(const_orch.getPool(99), nullptr);
    }

    TEST(Test__LoadOrchestrator, MultipleDevices)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);
        orch.addDevice(1);

        orch.planWeight(0, "w_dev0", 512, 1024, kQ4PayloadBytes,
                        false, false, 262144);
        orch.planWeight(1, "w_dev1", 256, 512, kQ4PayloadBytes,
                        true, false, 131072);

        auto *pool0 = orch.getPool(0);
        auto *pool1 = orch.getPool(1);
        ASSERT_NE(pool0, nullptr);
        ASSERT_NE(pool1, nullptr);

        EXPECT_EQ(pool0->numPlannedWeights(), 1u);
        EXPECT_EQ(pool1->numPlannedWeights(), 1u);

        // Different sizes due to different weight dimensions and asymmetric flag
        EXPECT_NE(pool0->totalPlannedBytes(), pool1->totalPlannedBytes());
    }

    TEST(Test__LoadOrchestrator, AllocateSucceeds)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);
        orch.planWeight(0, "w1", 64, 64, kQ4PayloadBytes, false, false, 1024);

        ASSERT_NO_THROW(orch.allocate(1024, 3));

        auto *pool = orch.getPool(0);
        ASSERT_NE(pool, nullptr);
        EXPECT_TRUE(pool->isAllocated());
    }

    TEST(Test__LoadOrchestrator, AllocateRejectsPinnedStagingWithoutStreams)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);
        orch.planRawWeight(0, "raw_weight", 64, 64, 1024);

        // A raw upload requires a pinned ring slot and an H2D stream. Without
        // this guard, load() fails later with a misleading "pinned ring not
        // allocated" error after the pool has already been allocated.
        EXPECT_THROW(orch.allocate(1024, 0), std::runtime_error);
    }

    TEST(Test__LoadOrchestrator, AllocateFailsBeforeBackendAllocationWhenVramBudgetExceeded)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/640ULL * kMiB);
        LoadOrchestrator orch(&backend);
        orch.addDevice(0);
        orch.planRawWeight(0, "large_raw_weight", 1, 1, 128ULL * kMiB);

        // Required = 128 MiB planned + 32 MiB staging + 512 MiB safety margin,
        // which exceeds the reported 640 MiB free budget.
        EXPECT_THROW(orch.allocate(32ULL * kMiB, 1), std::runtime_error);
        EXPECT_EQ(backend.allocateCalls(), 0)
            << "VRAM preflight should fail before WeightVRAMPool calls backend->allocate()";
    }

    TEST(Test__LoadOrchestrator, AllocateSucceedsWhenVramBudgetHasHeadroom)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/1024ULL * kMiB);
        LoadOrchestrator orch(&backend);
        orch.addDevice(0);
        orch.planRawWeight(0, "large_raw_weight", 1, 1, 128ULL * kMiB);

        ASSERT_NO_THROW(orch.allocate(32ULL * kMiB, 1));
        EXPECT_GT(backend.allocateCalls(), 0);
        EXPECT_GT(backend.lastAllocateBytes(), 0u);

        auto *pool = orch.getPool(0);
        ASSERT_NE(pool, nullptr);
        EXPECT_TRUE(pool->isAllocated());
    }

    TEST(Test__LoadOrchestrator, FinalizeReleasesTemporaryStagingOnly)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/1024ULL * kMiB);
        LoadOrchestrator orch(&backend);
        orch.addDevice(0);
        orch.planRawWeight(0, "large_raw_weight", 1, 1, 128ULL * kMiB);

        ASSERT_NO_THROW(orch.allocate(32ULL * kMiB, 1));
        auto *pool = orch.getPool(0);
        ASSERT_NE(pool, nullptr);
        auto slot_before = pool->getSlot("large_raw_weight");
        ASSERT_TRUE(slot_before.has_value());
        ASSERT_NE(slot_before->d_native_vnni_payload, nullptr);
        auto *payload_before = slot_before->d_native_vnni_payload;
        EXPECT_NE(pool->getStagingSlot(0), nullptr);

        orch.finalize();

        EXPECT_TRUE(pool->isAllocated());
        EXPECT_EQ(pool->getStagingSlot(0), nullptr);
        EXPECT_EQ(pool->stagingSlotCount(), 0);
        EXPECT_EQ(backend.getAllocationCount(), 1u);
        auto slot_after = pool->getSlot("large_raw_weight");
        ASSERT_TRUE(slot_after.has_value());
        EXPECT_EQ(slot_after->d_native_vnni_payload, payload_before);
    }

    TEST(Test__LoadOrchestrator, LoadAndFinalizeWithNoJobs)
    {
        LoadOrchestrator orch;
        EXPECT_NO_THROW(orch.load());
        EXPECT_NO_THROW(orch.finalize());
    }

    TEST(Test__LoadOrchestrator, PlanWeightUnknownDeviceThrows)
    {
        LoadOrchestrator orch;
        orch.addDevice(0);

        EXPECT_THROW(
            orch.planWeight(99, "w", 64, 64, kQ4PayloadBytes, false, false, 1024),
            std::runtime_error);
    }

} // namespace llaminar2
