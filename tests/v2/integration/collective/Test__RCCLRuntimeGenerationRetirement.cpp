/**
 * @file Test__RCCLRuntimeGenerationRetirement.cpp
 * @brief Real-device proof for RCCL ownership across HIP generation reset.
 *
 * A model may retire its final allocations while an inactive local-TP RCCL
 * coordinator remains pooled for the next runner.  The coordinator still owns
 * HIP streams, events, communicators, and library allocations; resetting the
 * primary context first leaves those handles stale and crashes when its owner
 * thread later exits.  This focused integration test proves the production
 * lifecycle without loading a model:
 *
 * 1. an active collective owner makes reset fail;
 * 2. backend shutdown parks that owner for reuse;
 * 3. TransferEngine retires the pooled owner before `hipDeviceReset()`; and
 * 4. a fresh RCCL clique performs byte-observable work on the new generation.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/BackendRouter.h"
#include "collective/DeviceGroup.h"
#include "collective/backends/RCCLBackend.h"
#include "transfer/TransferEngine.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr std::size_t kRetirementBytes = 64u * 1024u * 1024u;
        constexpr std::size_t kCollectiveElements = 256u;

        /** @return One rank-ordered two-device local ROCm collective group. */
        DeviceGroup twoDeviceROCmGroup()
        {
            return DeviceGroupBuilder()
                .setName("rccl_runtime_generation_retirement")
                .setScope(CollectiveScope::LOCAL)
                .setLocalRank(0)
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .build();
        }

        /**
         * @brief Capture and release one exact allocation before completion.
         * @return Move-only proof carrying the pre-release driver baseline.
         */
        ExclusiveModelRetirementTicket releasedAllocationTicket(
            IBackend &backend,
            DeviceId device)
        {
            void *const allocation = backend.allocate(
                kRetirementBytes,
                device.gpu_ordinal());
            if (!allocation)
                throw std::runtime_error("could not allocate retirement witness");

            auto ticket =
                TransferEngine::instance().beginExclusiveModelRetirement(
                    ModelDeviceMemoryRetention{
                        .device = device,
                        .prepared_weight_bytes = kRetirementBytes,
                        .reusable_workspace_bytes = 0u,
                    });
            backend.free(allocation, device.gpu_ordinal());
            return ticket;
        }

        /**
         * @brief Execute and verify a two-device FP32 sum on explicit streams.
         */
        void expectFreshCollectiveWorks(
            RCCLBackend &collective,
            IBackend &backend)
        {
            const std::size_t bytes =
                kCollectiveElements * sizeof(float);
            std::array<void *, 2> buffers{
                backend.allocate(bytes, 0),
                backend.allocate(bytes, 1),
            };
            ASSERT_NE(buffers[0], nullptr);
            ASSERT_NE(buffers[1], nullptr);

            auto &contexts = GPUDeviceContextPool::instance();
            std::array<void *, 2> streams{
                contexts.getContext(DeviceId::rocm(0)).defaultStream(),
                contexts.getContext(DeviceId::rocm(1)).defaultStream(),
            };
            ASSERT_NE(streams[0], nullptr);
            ASSERT_NE(streams[1], nullptr);
            collective.setComputeStreams({streams[0], streams[1]});

            const std::vector<float> first(kCollectiveElements, 1.0f);
            const std::vector<float> second(kCollectiveElements, 2.0f);
            ASSERT_TRUE(backend.hostToDevice(
                buffers[0], first.data(), bytes, 0, streams[0]));
            ASSERT_TRUE(backend.hostToDevice(
                buffers[1], second.data(), bytes, 1, streams[1]));

            ASSERT_TRUE(collective.allreduceMultiAndSynchronize(
                {buffers[0], buffers[1]},
                kCollectiveElements,
                CollectiveDataType::FLOAT32,
                CollectiveOp::ALLREDUCE_SUM))
                << collective.lastError();

            std::array<std::vector<float>, 2> observed{
                std::vector<float>(kCollectiveElements),
                std::vector<float>(kCollectiveElements),
            };
            ASSERT_TRUE(backend.deviceToHost(
                observed[0].data(), buffers[0], bytes, 0, streams[0]));
            ASSERT_TRUE(backend.deviceToHost(
                observed[1].data(), buffers[1], bytes, 1, streams[1]));
            ASSERT_TRUE(backend.synchronizeStream(streams[0], 0));
            ASSERT_TRUE(backend.synchronizeStream(streams[1], 1));

            for (std::size_t device = 0u; device < observed.size(); ++device)
            {
                for (std::size_t element = 0u;
                     element < kCollectiveElements;
                     ++element)
                {
                    EXPECT_FLOAT_EQ(observed[device][element], 3.0f)
                        << "device=" << device << " element=" << element;
                }
            }

            backend.free(buffers[0], 0);
            backend.free(buffers[1], 1);
        }
    } // namespace

    /**
     * @brief Prove active rejection, ordered pooled teardown, and fresh reuse.
     */
    TEST(RCCLRuntimeGenerationRetirement,
         ActiveOwnerIsRejectedAndPooledOwnerRetiresBeforeReset)
    {
        ensureAMDFactoryRegistered();
        auto &context_pool = GPUDeviceContextPool::instance();
        if (!context_pool.hasAMDSupport() ||
            context_pool.amdDeviceCount() < 2)
        {
            GTEST_SKIP() << "two ROCm devices are required";
        }

        GlobalBackendRouter::shutdown();
        IBackend *const backend = getBackendFor(DeviceId::rocm(0));
        ASSERT_NE(backend, nullptr);

        const DeviceGroup group = twoDeviceROCmGroup();
        RCCLBackend first_collective;
        ASSERT_TRUE(first_collective.initialize(group))
            << first_collective.lastError();

        /* The public transaction must consume, not interrupt, an active
         * collective owner. This makes caller ordering executable rather than
         * a comment attached to hipDeviceReset(). */
        auto blocked_ticket =
            releasedAllocationTicket(*backend, DeviceId::rocm(0));
        EXPECT_THROW(
            (void)TransferEngine::instance()
                .completeExclusiveModelRetirement(
                    std::move(blocked_ticket)),
            std::runtime_error);

        first_collective.shutdown();

        /* The inactive coordinator is now reusable but still belongs to the
         * current HIP generation. Completion must join it before reset. */
        const std::uint64_t generation_before =
            backend->deviceRuntimeGeneration(0);
        ASSERT_NE(generation_before, 0u);
        auto retirement_ticket =
            releasedAllocationTicket(*backend, DeviceId::rocm(0));
        const DeviceMemoryReclamationReceipt receipt =
            TransferEngine::instance().completeExclusiveModelRetirement(
                std::move(retirement_ticket));
        EXPECT_TRUE(receipt.runtime_reset_invoked);
        EXPECT_EQ(receipt.retired_runtime_generation, generation_before);
        EXPECT_EQ(receipt.active_runtime_generation, generation_before + 1u);

        RCCLBackend fresh_collective;
        ASSERT_TRUE(fresh_collective.initialize(group))
            << fresh_collective.lastError();
        expectFreshCollectiveWorks(fresh_collective, *backend);
        fresh_collective.shutdown();

        /* Retire process-owned resources in the same order used by production
         * shutdown so this test cannot defer cleanup to static destruction. */
        RCCLBackend::drainCoordinatorPool();
        context_pool.shutdown();
    }

} // namespace llaminar2::test
