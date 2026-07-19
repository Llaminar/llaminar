#include <gtest/gtest.h>
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/GPUVramPreflight.h"
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

    TEST(Test__LoadOrchestrator, OversizedWeightIsSplitIntoBoundedRowChunks)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/2ULL * 1024ULL * kMiB);
        LoadOrchestrator orch(&backend);
        orch.addDevice(0);

        constexpr int rows = 8;
        constexpr int cols = 64;
        constexpr size_t bytes_per_row = cols * sizeof(float);
        constexpr size_t raw_bytes = rows * bytes_per_row;
        std::vector<float> raw(rows * cols, 1.0f);

        orch.planRawWeight(0, "chunked", rows, cols, raw_bytes);
        ASSERT_NO_THROW(orch.allocate(/*pinned_slot_size=*/2 * bytes_per_row,
                                      /*num_h2d_streams=*/2));

        WeightJob job;
        job.name = "chunked";
        job.host_raw_data = raw.data();
        job.raw_bytes = raw_bytes;
        job.format = RepackFormat::RAW_FP;
        job.N = rows;
        job.K = cols;
        job.is_asymmetric = false;
        ASSERT_NO_THROW(orch.addWeightJob(0, job));

        EXPECT_EQ(orch.pendingJobCount(0), 4u);
        EXPECT_EQ(orch.totalPendingBytes(0), raw_bytes);
        ASSERT_NE(orch.getPool(0), nullptr);
        EXPECT_EQ(orch.getPool(0)->maxStagingSlotBytes(), 2 * bytes_per_row);
    }

    TEST(Test__LoadOrchestrator, RejectsBudgetSmallerThanOneRawRow)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/2ULL * 1024ULL * kMiB);
        LoadOrchestrator orch(&backend);
        orch.addDevice(0);

        constexpr int rows = 2;
        constexpr int cols = 256;
        const size_t raw_bytes = static_cast<size_t>(rows) * cols * sizeof(float);
        std::vector<float> raw(static_cast<size_t>(rows) * cols, 1.0f);
        orch.planRawWeight(0, "too_wide", rows, cols, raw_bytes);
        ASSERT_NO_THROW(orch.allocate(/*pinned_slot_size=*/512,
                                      /*num_h2d_streams=*/1));

        WeightJob job;
        job.name = "too_wide";
        job.host_raw_data = raw.data();
        job.raw_bytes = raw_bytes;
        job.format = RepackFormat::RAW_FP;
        job.N = rows;
        job.K = cols;
        job.is_asymmetric = false;

        EXPECT_THROW(orch.addWeightJob(0, job), std::runtime_error);
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

    TEST(Test__LoadOrchestrator, DirectRebalanceMarginAllowsTightNoStagingArrival)
    {
        const size_t planned_bytes = 8ULL * kMiB;
        BudgetMockBackend backend(/*total_bytes=*/24ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/131ULL * kMiB);

        {
            LoadOrchestrator generic(&backend);
            generic.setVramPreflightSafetyMarginBytes(128ULL * kMiB);
            generic.addDevice(0);
            generic.planRawWeight(0, "arrival_generic_margin", 1, 1, planned_bytes);
            EXPECT_THROW(generic.allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0),
                         std::runtime_error);
        }
        EXPECT_EQ(backend.allocateCalls(), 0)
            << "The generic reserve should fail before any device allocation";

        LoadOrchestrator direct(&backend);
        direct.setVramPreflightSafetyMarginBytes(gpuDirectRebalanceVramSafetyMarginBytes());
        direct.addDevice(0);
        direct.planRawWeight(0, "arrival_direct_margin", 1, 1, planned_bytes);

        ASSERT_NO_THROW(direct.allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0));
        EXPECT_GT(backend.allocateCalls(), 0);
        EXPECT_GT(backend.lastAllocateBytes(), 0u);
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
