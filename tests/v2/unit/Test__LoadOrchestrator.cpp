#include <gtest/gtest.h>
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/GPUVramPreflight.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "../mocks/MockBackend.h"

/**
 * @file Test__LoadOrchestrator.cpp
 * @brief Unit tests for GPU weight loading orchestration and staging finalization.
 *
 * The tests cover planning, VRAM preflight accounting, and the finalize() contract
 * that releases temporary staging resources while preserving persistent pool state.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
                if (device_id == failing_device_id_)
                    return nullptr;
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

            /** @brief Make every subsequent allocation on one device fail. */
            void failAllocationsOnDevice(int device_id)
            {
                failing_device_id_ = device_id;
            }

        private:
            size_t total_bytes_ = 0;
            size_t free_bytes_ = 0;
            int allocate_calls_ = 0;
            size_t last_allocate_bytes_ = 0;
            int failing_device_id_ = -1;
        };

        /**
         * @brief Admit one exact rank-local GPU load before constructing its allocator.
         *
         * The helper intentionally consumes the same owner lines as
         * LoadOrchestrator. Tests therefore exercise the production contract:
         * aggregate admission happens once, and concrete allocation merely
         * materializes the certified bytes through RAII leases.
         */
        std::shared_ptr<PhysicalMemoryAuthority> makeLoadAuthority(
            DeviceId gpu,
            size_t gpu_total_bytes,
            size_t gpu_available_bytes,
            std::initializer_list<std::pair<PhysicalMemoryOwner, size_t>>
                gpu_charges,
            size_t host_staging_bytes = 0u)
        {
            PhysicalMemoryBOMBuilder gpu_bom({
                .world_rank = 0,
                .device = gpu,
                .total_bytes = gpu_total_bytes,
                .admission_available_bytes = gpu_available_bytes,
            });
            for (const auto &[owner, bytes] : gpu_charges)
            {
                if (bytes != 0u)
                    gpu_bom.add(owner, bytes);
            }

            PhysicalMemoryPlanBuilder plan;
            plan.add(gpu_bom.build());
            if (host_staging_bytes != 0u)
            {
                PhysicalMemoryBOMBuilder host_bom({
                    .world_rank = 0,
                    .device = DeviceId::cpu(),
                    .total_bytes = 4ULL * 1024ULL * kMiB,
                    .admission_available_bytes = 4ULL * 1024ULL * kMiB,
                });
                host_bom.add(
                    PhysicalMemoryOwner::WeightLoadStaging,
                    host_staging_bytes);
                plan.add(host_bom.build());
            }

            auto admission = std::make_shared<
                const PhysicalMemoryPlanAdmissionCertificate>(plan.build());
            return std::make_shared<PhysicalMemoryAuthority>(
                std::move(admission), 0);
        }
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
        LoadOrchestrator orch(&backend, kTestOnlyUnadmittedGPUAllocation);
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

    TEST(Test__LoadOrchestrator, OrdersJobsByMappedSourceAddressAndPreservesAliases)
    {
        std::array<uint8_t, 256> mapped_file{};
        std::vector<WeightJob> jobs(5);

        jobs[0].name = "late";
        jobs[0].host_raw_data = mapped_file.data() + 192;
        jobs[1].name = "early_alias_a";
        jobs[1].host_raw_data = mapped_file.data() + 32;
        jobs[2].name = "middle";
        jobs[2].host_raw_data = mapped_file.data() + 128;
        jobs[3].name = "early_alias_b";
        jobs[3].host_raw_data = mapped_file.data() + 32;
        jobs[4].name = "invalid";
        jobs[4].host_raw_data = nullptr;

        const size_t backward_jumps =
            orderWeightJobsForSequentialHostAccess(jobs);

        EXPECT_EQ(backward_jumps, 2u);
        ASSERT_EQ(jobs.size(), 5u);
        EXPECT_EQ(jobs[0].name, "early_alias_a");
        EXPECT_EQ(jobs[1].name, "early_alias_b")
            << "Stable source ordering must preserve tied-weight aliases";
        EXPECT_EQ(jobs[2].name, "middle");
        EXPECT_EQ(jobs[3].name, "late");
        EXPECT_EQ(jobs[4].name, "invalid")
            << "Null sources stay last for processJobs() validation";
    }

    TEST(Test__LoadOrchestrator, CoalescesViewsUsingNonOwningParentIdentity)
    {
        constexpr int rows_per_expert = 2;
        constexpr int columns = 32;
        constexpr size_t bytes_per_row = 16;
        constexpr size_t bytes_per_expert = rows_per_expert * bytes_per_row;
        std::array<uint8_t, 4 * bytes_per_expert> parent{};
        int parent_identity = 0;

        // Discovery order intentionally differs from source order. The returned
        // member indices must still refer to this original vector so graph-side
        // expert identity cannot be permuted by I/O coalescing.
        constexpr std::array<size_t, 4> discovery_order = {2, 0, 3, 1};
        std::vector<WeightJob> jobs;
        std::vector<const void *> source_identities;
        for (const size_t expert : discovery_order)
        {
            WeightJob job;
            job.name = "expert_" + std::to_string(expert);
            job.host_raw_data = parent.data() + expert * bytes_per_expert;
            job.raw_bytes = bytes_per_expert;
            job.format = RepackFormat::IQ3_S;
            job.N = rows_per_expert;
            job.K = columns;
            jobs.push_back(job);
            // The identity is deliberately an unrelated scalar, proving that
            // coalescing neither dereferences it nor treats it as byte ownership.
            source_identities.push_back(&parent_identity);
        }

        const auto runs = coalesceContiguousWeightJobs(jobs, source_identities);

        ASSERT_EQ(runs.size(), 1u);
        EXPECT_EQ(runs[0].job.host_raw_data, parent.data());
        EXPECT_EQ(runs[0].job.N, 4 * rows_per_expert);
        EXPECT_EQ(runs[0].job.full_N, 4 * rows_per_expert);
        EXPECT_EQ(runs[0].job.K, columns);
        EXPECT_EQ(runs[0].job.raw_bytes, parent.size());
        EXPECT_EQ(runs[0].job.packed_group_rows, rows_per_expert);
        ASSERT_EQ(runs[0].members.size(), 4u);

        for (size_t source_position = 0; source_position < discovery_order.size(); ++source_position)
        {
            const size_t expert = discovery_order[source_position];
            const auto member = std::find_if(
                runs[0].members.begin(), runs[0].members.end(),
                [source_position](const CoalescedWeightJobMember &candidate)
                { return candidate.source_job_index == source_position; });
            ASSERT_NE(member, runs[0].members.end());
            EXPECT_EQ(member->row_offset,
                      static_cast<int>(expert) * rows_per_expert);
        }
    }

    TEST(Test__LoadOrchestrator, CoalescingPreservesOwnerGapGeometryAndFormatBoundaries)
    {
        std::array<uint8_t, 512> source{};
        int owner_a = 0;
        int owner_b = 0;

        auto make_job = [&](const char *name,
                            size_t byte_offset,
                            RepackFormat format,
                            int rows,
                            int columns,
                            size_t raw_bytes)
        {
            WeightJob job;
            job.name = name;
            job.host_raw_data = source.data() + byte_offset;
            job.raw_bytes = raw_bytes;
            job.format = format;
            job.N = rows;
            job.K = columns;
            return job;
        };

        // Every adjacent pair differs in exactly one required contract. None may
        // share packed storage despite occupying nearby bytes.
        std::vector<WeightJob> jobs{
            make_job("base", 0, RepackFormat::Q4_0, 2, 32, 32),
            make_job("source_gap", 48, RepackFormat::Q4_0, 2, 32, 32),
            make_job("different_owner", 80, RepackFormat::Q4_0, 2, 32, 32),
            make_job("different_k", 112, RepackFormat::Q4_0, 1, 64, 32),
            make_job("different_n_base", 144, RepackFormat::Q4_0, 2, 32, 32),
            make_job("different_n", 176, RepackFormat::Q4_0, 1, 32, 16),
            make_job("different_format", 192, RepackFormat::Q5_0, 1, 32, 16),
        };
        std::vector<const void *> source_identities{
            &owner_a, &owner_a, &owner_b, &owner_b,
            &owner_b, &owner_b, &owner_b};

        const auto runs = coalesceContiguousWeightJobs(jobs, source_identities);
        ASSERT_EQ(runs.size(), jobs.size());
        for (const auto &run : runs)
        {
            ASSERT_EQ(run.members.size(), 1u);
            EXPECT_EQ(run.members[0].row_offset, 0);
        }
    }

    TEST(Test__LoadOrchestrator, CoalescingIsTotalAcrossEveryGpuRepackFormat)
    {
        constexpr std::array<RepackFormat, 22> formats = {
            RepackFormat::Q4_0, RepackFormat::IQ4_NL, RepackFormat::Q4_1,
            RepackFormat::Q5_0, RepackFormat::Q5_1, RepackFormat::Q4_K,
            RepackFormat::Q5_K, RepackFormat::Q6_K, RepackFormat::Q3_K,
            RepackFormat::Q2_K, RepackFormat::IQ4_XS, RepackFormat::IQ3_S,
            RepackFormat::IQ3_XXS, RepackFormat::IQ2_S, RepackFormat::IQ2_XS,
            RepackFormat::IQ2_XXS, RepackFormat::IQ1_S, RepackFormat::IQ1_M,
            RepackFormat::Q8_0, RepackFormat::Q8_1, RepackFormat::Q8_K,
            RepackFormat::RAW_FP,
        };
        std::array<uint8_t, 256> source{};
        int owner = 0;

        for (const RepackFormat format : formats)
        {
            std::vector<WeightJob> jobs(2);
            for (size_t expert = 0; expert < jobs.size(); ++expert)
            {
                jobs[expert].name = "format_expert_" + std::to_string(expert);
                jobs[expert].host_raw_data = source.data() + expert * 64;
                jobs[expert].raw_bytes = 64;
                jobs[expert].format = format;
                jobs[expert].N = 2;
                jobs[expert].K = 32;
            }

            const auto runs = coalesceContiguousWeightJobs(
                jobs, std::vector<const void *>(jobs.size(), &owner));
            ASSERT_EQ(runs.size(), 1u)
                << "format=" << static_cast<int>(format);
            EXPECT_EQ(runs[0].job.N, 4)
                << "format=" << static_cast<int>(format);
            EXPECT_EQ(runs[0].job.raw_bytes, 128u)
                << "format=" << static_cast<int>(format);
            EXPECT_EQ(runs[0].job.packed_group_rows, 2)
                << "format=" << static_cast<int>(format);
        }
    }

    TEST(Test__LoadOrchestrator, NativeVnniExpertSubregionsAreGapFreeForEveryCodebook)
    {
        constexpr std::array<const NativeVnniFormatInfo *, 21> formats = {
            &native_vnni_formats::IQ4_NL, &native_vnni_formats::Q8_0,
            &native_vnni_formats::Q8_1, &native_vnni_formats::Q4_0,
            &native_vnni_formats::Q4_1, &native_vnni_formats::Q5_0,
            &native_vnni_formats::Q5_1, &native_vnni_formats::Q6_K,
            &native_vnni_formats::Q2_K, &native_vnni_formats::Q5_K,
            &native_vnni_formats::Q3_K, &native_vnni_formats::Q4_K,
            &native_vnni_formats::Q8_K, &native_vnni_formats::IQ4_XS,
            &native_vnni_formats::IQ2_XXS, &native_vnni_formats::IQ2_XS,
            &native_vnni_formats::IQ3_XXS, &native_vnni_formats::IQ2_S,
            &native_vnni_formats::IQ3_S, &native_vnni_formats::IQ1_S,
            &native_vnni_formats::IQ1_M,
        };
        constexpr size_t expert_count = 15;
        constexpr size_t rows_per_expert = 7;
        constexpr size_t columns = 256;

        for (const auto *format : formats)
        {
            ASSERT_NE(format, nullptr);
            const auto slab = nativeVnniPackedRegionSizes(
                expert_count * rows_per_expert, columns, *format);
            const auto expert = nativeVnniPackedRegionSizes(
                rows_per_expert, columns, *format);

            for (size_t expert_index = 0; expert_index < expert_count; ++expert_index)
            {
                const auto offset = nativeVnniPackedRegionSizes(
                    expert_index * rows_per_expert, columns, *format);
                EXPECT_EQ(offset.payload_bytes, expert_index * expert.payload_bytes);
                EXPECT_EQ(offset.scales_bytes, expert_index * expert.scales_bytes);
                EXPECT_EQ(offset.mins_bytes, expert_index * expert.mins_bytes);
                EXPECT_EQ(offset.emins_bytes, expert_index * expert.emins_bytes);
            }

            EXPECT_EQ(expert_count * expert.payload_bytes, slab.payload_bytes);
            EXPECT_EQ(expert_count * expert.scales_bytes, slab.scales_bytes);
            EXPECT_EQ(expert_count * expert.mins_bytes, slab.mins_bytes);
            EXPECT_EQ(expert_count * expert.emins_bytes, slab.emins_bytes);
        }
    }

    TEST(Test__LoadOrchestrator, CoalescingRejectsMissingOwnershipAndPrechunkedInput)
    {
        std::array<uint8_t, 64> source{};
        WeightJob job;
        job.name = "invalid_planning_phase";
        job.host_raw_data = source.data();
        job.raw_bytes = source.size();
        job.format = RepackFormat::Q4_0;
        job.N = 2;
        job.K = 32;

        EXPECT_THROW(
            coalesceContiguousWeightJobs({job}, {nullptr}),
            std::invalid_argument);

        int owner = 0;
        job.row_offset = 1;
        job.full_N = 4;
        EXPECT_THROW(
            coalesceContiguousWeightJobs({job}, {&owner}),
            std::invalid_argument);

        job.row_offset = 0;
        job.full_N = job.N;
        job.packed_group_rows = job.N;
        EXPECT_THROW(
            coalesceContiguousWeightJobs({job}, {&owner}),
            std::invalid_argument);
    }

    TEST(Test__LoadOrchestrator, RejectsBudgetSmallerThanOneRawRow)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/2ULL * 1024ULL * kMiB);
        LoadOrchestrator orch(&backend, kTestOnlyUnadmittedGPUAllocation);
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

    TEST(Test__LoadOrchestrator, AdmissionFailsBeforeBackendAllocationWhenVramBudgetExceeded)
    {
        constexpr size_t weights = 128ULL * kMiB;
        constexpr size_t staging = 32ULL * kMiB;
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/weights + staging - 1u);

        // Aggregate admission—not the low-level allocator—rejects the exact
        // 128 MiB weight plus 32 MiB device staging requirement.
        EXPECT_THROW(
            (void)makeLoadAuthority(
                DeviceId::rocm(0),
                backend.deviceMemoryTotal(0),
                backend.deviceMemoryFree(0),
                {
                    {PhysicalMemoryOwner::PrimaryModelWeights, weights},
                    {PhysicalMemoryOwner::WeightLoadStaging, staging},
                },
                staging),
            std::invalid_argument);
        EXPECT_EQ(backend.allocateCalls(), 0)
            << "Admission must fail before WeightVRAMPool reaches the backend";
    }

    TEST(Test__LoadOrchestrator, AllocateSucceedsAtExactVramBoundary)
    {
        constexpr size_t weights = 128ULL * kMiB;
        constexpr size_t staging = 32ULL * kMiB;
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/weights + staging);
        auto authority = makeLoadAuthority(
            DeviceId::rocm(0),
            backend.deviceMemoryTotal(0),
            backend.deviceMemoryFree(0),
            {
                {PhysicalMemoryOwner::PrimaryModelWeights, weights},
                {PhysicalMemoryOwner::WeightLoadStaging, staging},
            },
            staging);
        LoadOrchestrator orch(
            &backend, authority, PhysicalMemoryOwner::PrimaryModelWeights);
        orch.addDevice(0);
        orch.planRawWeight(0, "large_raw_weight", 1, 1, weights);

        ASSERT_NO_THROW(orch.allocate(staging, 1));
        EXPECT_GT(backend.allocateCalls(), 0);
        EXPECT_GT(backend.lastAllocateBytes(), 0u);

        auto *pool = orch.getPool(0);
        ASSERT_NE(pool, nullptr);
        EXPECT_TRUE(pool->isAllocated());
    }

    TEST(Test__LoadOrchestrator, NoStagingArrivalUsesExactConcreteBill)
    {
        const size_t planned_bytes = 8ULL * kMiB;
        BudgetMockBackend backend(/*total_bytes=*/24ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/8ULL * kMiB);
        auto authority = makeLoadAuthority(
            DeviceId::rocm(0),
            backend.deviceMemoryTotal(0),
            backend.deviceMemoryFree(0),
            {{PhysicalMemoryOwner::PrimaryModelWeights, planned_bytes}});
        LoadOrchestrator direct(
            &backend, authority, PhysicalMemoryOwner::PrimaryModelWeights);
        direct.addDevice(0);
        direct.planRawWeight(0, "arrival_exact_bill", 1, 1, planned_bytes);

        ASSERT_NO_THROW(direct.allocate(/*pinned_slot_size=*/0, /*num_h2d_streams=*/0));
        EXPECT_GT(backend.allocateCalls(), 0);
        EXPECT_GT(backend.lastAllocateBytes(), 0u);
    }

    TEST(Test__LoadOrchestrator, FinalizeReleasesTemporaryStagingOnly)
    {
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/1024ULL * kMiB);
        constexpr size_t weights = 128ULL * kMiB;
        constexpr size_t staging = 32ULL * kMiB;
        auto authority = makeLoadAuthority(
            DeviceId::rocm(0),
            backend.deviceMemoryTotal(0),
            backend.deviceMemoryFree(0),
            {
                {PhysicalMemoryOwner::PrimaryModelWeights, weights},
                {PhysicalMemoryOwner::WeightLoadStaging, staging},
            },
            staging);
        LoadOrchestrator orch(
            &backend, authority, PhysicalMemoryOwner::PrimaryModelWeights);
        orch.addDevice(0);
        orch.planRawWeight(0, "large_raw_weight", 1, 1, weights);

        ASSERT_NO_THROW(orch.allocate(staging, 1));
        auto *pool = orch.getPool(0);
        ASSERT_NE(pool, nullptr);
        auto slot_before = pool->getSlot("large_raw_weight");
        ASSERT_TRUE(slot_before.has_value());
        ASSERT_NE(slot_before->d_native_vnni_payload, nullptr);
        auto *payload_before = slot_before->d_native_vnni_payload;
        EXPECT_NE(pool->getStagingSlot(0), nullptr);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            staging);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::cpu(),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            staging);

        orch.finalize();

        EXPECT_TRUE(pool->isAllocated());
        EXPECT_EQ(pool->getStagingSlot(0), nullptr);
        EXPECT_EQ(pool->stagingSlotCount(), 0);
        EXPECT_EQ(backend.getAllocationCount(), 1u);
        auto slot_after = pool->getSlot("large_raw_weight");
        ASSERT_TRUE(slot_after.has_value());
        EXPECT_EQ(slot_after->d_native_vnni_payload, payload_before);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            weights);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::cpu(),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);

        orch.release();
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
    }

    TEST(Test__LoadOrchestrator,
         MultiOwnerPlanAttributesEveryAlignmentByteAndRetiresByLifetime)
    {
        /*
         * The second weight begins at byte 512 after internal alignment. The
         * final pool allocation ends at byte 768, so the exact owner split is
         * 257 primary bytes and 511 routed-expert bytes.
         */
        constexpr size_t primary_bytes = 257u;
        constexpr size_t routed_logical_bytes = 17u;
        constexpr size_t routed_physical_bytes = 511u;
        constexpr size_t device_staging_bytes = 1024u;
        constexpr size_t host_staging_bytes = 514u;
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/1024ULL * kMiB);
        auto authority = makeLoadAuthority(
            DeviceId::rocm(0),
            backend.deviceMemoryTotal(0),
            backend.deviceMemoryFree(0),
            {
                {PhysicalMemoryOwner::PrimaryModelWeights, primary_bytes},
                {PhysicalMemoryOwner::RoutedExpertWeights,
                 routed_physical_bytes},
                {PhysicalMemoryOwner::WeightLoadStaging,
                 device_staging_bytes},
            },
            host_staging_bytes);
        LoadOrchestrator orch(
            &backend, authority, PhysicalMemoryOwner::PrimaryModelWeights);
        orch.addDevice(0);
        orch.planRawWeightForOwner(
            0,
            "primary",
            1,
            1,
            primary_bytes,
            PhysicalMemoryOwner::PrimaryModelWeights);
        orch.planRawWeightForOwner(
            0,
            "routed",
            1,
            1,
            routed_logical_bytes,
            PhysicalMemoryOwner::RoutedExpertWeights);

        EXPECT_EQ(
            orch.plannedPersistentBytes(
                0, PhysicalMemoryOwner::PrimaryModelWeights),
            primary_bytes);
        EXPECT_EQ(
            orch.plannedPersistentBytes(
                0, PhysicalMemoryOwner::RoutedExpertWeights),
            routed_physical_bytes);

        ASSERT_NO_THROW(orch.allocate(/*pinned_slot_size=*/257u,
                                      /*num_h2d_streams=*/2));
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            primary_bytes);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::RoutedExpertWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            routed_physical_bytes);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            device_staging_bytes);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::cpu(),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            host_staging_bytes);

        orch.finalize();
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::cpu(),
                PhysicalMemoryOwner::WeightLoadStaging,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
        EXPECT_EQ(backend.getAllocationCount(), 1u);

        orch.release();
        EXPECT_EQ(backend.getAllocationCount(), 0u);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::RoutedExpertWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
    }

    TEST(Test__LoadOrchestrator,
         LaterOwnerClaimFailureRollsBackEarlierClaimsBeforeAllocation)
    {
        constexpr size_t primary_bytes = 257u;
        constexpr size_t routed_physical_bytes = 511u;
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/1024ULL * kMiB);
        auto authority = makeLoadAuthority(
            DeviceId::rocm(0),
            backend.deviceMemoryTotal(0),
            backend.deviceMemoryFree(0),
            {
                {PhysicalMemoryOwner::PrimaryModelWeights, primary_bytes},
                {PhysicalMemoryOwner::RoutedExpertWeights,
                 routed_physical_bytes - 1u},
            });
        LoadOrchestrator orch(
            &backend, authority, PhysicalMemoryOwner::PrimaryModelWeights);
        orch.addDevice(0);
        orch.planRawWeightForOwner(
            0, "primary", 1, 1, primary_bytes,
            PhysicalMemoryOwner::PrimaryModelWeights);
        orch.planRawWeightForOwner(
            0, "routed", 1, 1, 17u,
            PhysicalMemoryOwner::RoutedExpertWeights);

        EXPECT_THROW(
            orch.allocate(/*pinned_slot_size=*/0,
                          /*num_h2d_streams=*/0),
            std::logic_error);
        EXPECT_EQ(backend.allocateCalls(), 0);
        EXPECT_EQ(backend.getAllocationCount(), 0u);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::PrimaryModelWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
        EXPECT_EQ(
            authority->claimedBytes(
                DeviceId::rocm(0),
                PhysicalMemoryOwner::RoutedExpertWeights,
                PhysicalMemoryMaterializationKind::NewAllocation),
            0u);
    }

    TEST(Test__LoadOrchestrator,
         LaterDeviceAllocationFailureRollsBackWholeTopologyTransaction)
    {
        constexpr size_t logical_bytes = 257u;
        constexpr size_t physical_bytes = 512u;
        BudgetMockBackend backend(/*total_bytes=*/4ULL * 1024ULL * kMiB,
                                  /*free_bytes=*/1024ULL * kMiB);
        PhysicalMemoryPlanBuilder plan;
        for (const int device_id : {0, 1})
        {
            PhysicalMemoryBOMBuilder bom({
                .world_rank = 0,
                .device = DeviceId::rocm(device_id),
                .total_bytes = backend.deviceMemoryTotal(device_id),
                .admission_available_bytes =
                    backend.deviceMemoryFree(device_id),
            });
            bom.add(
                PhysicalMemoryOwner::PrimaryModelWeights,
                physical_bytes);
            plan.add(bom.build());
        }
        auto admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(plan.build());
        auto authority = std::make_shared<PhysicalMemoryAuthority>(
            std::move(admission), 0);

        LoadOrchestrator orch(
            &backend, authority, PhysicalMemoryOwner::PrimaryModelWeights);
        for (const int device_id : {0, 1})
        {
            orch.addDevice(device_id);
            orch.planRawWeight(
                device_id,
                "weight_" + std::to_string(device_id),
                1,
                1,
                logical_bytes);
        }
        backend.failAllocationsOnDevice(1);

        EXPECT_THROW(
            orch.allocate(/*pinned_slot_size=*/0,
                          /*num_h2d_streams=*/0),
            std::runtime_error);
        EXPECT_EQ(backend.allocateCalls(), 2);
        EXPECT_EQ(backend.getAllocationCount(), 0u);
        for (const int device_id : {0, 1})
        {
            EXPECT_EQ(
                authority->claimedBytes(
                    DeviceId::rocm(device_id),
                    PhysicalMemoryOwner::PrimaryModelWeights,
                    PhysicalMemoryMaterializationKind::NewAllocation),
                0u);
        }
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
