/**
 * @file Test__TransferEngine.cpp
 * @brief Unit coverage for transfer planning, coherence, and exact-event ordering.
 *
 * GPU behavior is modeled with host allocations so these tests remain
 * device-free. The event-failure fixture proves that TransferEngine never
 * substitutes a stream/device synchronization for an exact dependency and
 * that queued H2D source lifetimes are protected at host reuse boundaries.
 */

#include <gtest/gtest.h>

#include "tensors/CoherenceState.h"
#include "transfer/TransferEngine.h"
#include "transfer/TransferMethod.h"

// For execute tests
#include "backends/DeviceId.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorSlice.h"
#include "../../mocks/MockBackend.h"
#include "../../mocks/MockWorkerGPUContext.h"
#include "../../utils/TestTensorFactory.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// planTransfer() tests — pure logic, no GPU needed
// ============================================================================

class Test__TransferEngine_Plan : public ::testing::Test
{
};

TEST(Test__TransferEngine_Plan, CpuToCuda_HostToDevice)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, CpuToRocm_HostToDevice)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::rocm(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, CudaToCpu_DeviceToHost)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_HOST);
}

TEST(Test__TransferEngine_Plan, RocmToCpu_DeviceToHost)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::cpu(), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_HOST);
}

TEST(Test__TransferEngine_Plan, CudaToCuda_SameBackend)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cuda(1), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
}

TEST(Test__TransferEngine_Plan, RocmToRocm_SameBackend)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::rocm(1), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
}

TEST(Test__TransferEngine_Plan, CudaToRocm_HostStaged)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_STAGED);
}

TEST(Test__TransferEngine_Plan, RocmToCuda_HostStaged)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::rocm(0), DeviceId::cuda(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::HOST_STAGED);
}

TEST(Test__TransferEngine_Plan, SameDevice_Noop)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cuda(0), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, CpuToCpu_Noop)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::cpu(), MemoryResidency::STANDARD);
    EXPECT_EQ(method, TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, Mapped_AlwaysMappedNoop)
{
    // Mapped memory is always a no-op regardless of devices
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::MAPPED),
              TransferMethod::MAPPED_NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::MAPPED),
              TransferMethod::MAPPED_NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::MAPPED),
              TransferMethod::MAPPED_NOOP);
}

TEST(Test__TransferEngine_Plan, HostResident_AlwaysNoop)
{
    // HOST_RESIDENT tensors never move to device — always NOOP regardless of direction
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::rocm(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cuda(0), DeviceId::rocm(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);

    // Same device also NOOP (both paths agree)
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::cpu(), DeviceId::cpu(), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, HostResident_PrecedesOtherChecks)
{
    // HOST_RESIDENT should NOOP even for cross-vendor transfers that would
    // normally be HOST_STAGED — residency takes priority.
    EXPECT_EQ(TransferEngine::planTransfer(
                  DeviceId::rocm(0), DeviceId::cuda(0), MemoryResidency::HOST_RESIDENT),
              TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, DescribeTransferPlan_HumanReadable)
{
    auto desc = TransferEngine::describeTransferPlan(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::STANDARD);

    // Should contain source, destination, method
    EXPECT_NE(desc.find("HOST_TO_DEVICE"), std::string::npos);
}

TEST(Test__TransferEngine_Plan, DescribeTransferPlan_HostResident)
{
    auto desc = TransferEngine::describeTransferPlan(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::HOST_RESIDENT);

    EXPECT_NE(desc.find("HOST_RESIDENT"), std::string::npos);
    EXPECT_NE(desc.find("NOOP"), std::string::npos);
}

namespace
{
    /** @brief Device-free allocation spy for persistent staging-slab ownership. */
    class TransferStagingBackendSpy final : public MockBackend
    {
    public:
        TransferStagingBackendSpy() : MockBackend(DeviceType::CUDA) {}

        /** @brief Model one canonical device-slab allocation. */
        void *allocate(size_t bytes, int device_id) override
        {
            ++device_allocations;
            last_device_bytes = bytes;
            last_device_ordinal = device_id;
            return MockBackend::allocate(bytes, device_id);
        }

        /** @brief Observe the final shared-owner device release. */
        void free(void *ptr, int device_id) override
        {
            if (ptr)
                ++device_frees;
            MockBackend::free(ptr, device_id);
        }

        /** @brief Model one backend-pinned host slab with ordinary test RAM. */
        void *allocatePinned(size_t bytes, int device_id) override
        {
            ++pinned_allocations;
            last_pinned_bytes = bytes;
            last_pinned_ordinal = device_id;
            return std::malloc(bytes);
        }

        /** @brief Observe the final shared-owner pinned release. */
        void freePinned(void *ptr, int) override
        {
            if (ptr)
            {
                ++pinned_frees;
                std::free(ptr);
            }
        }

        size_t pinned_allocations = 0u;
        size_t pinned_frees = 0u;
        size_t device_allocations = 0u;
        size_t device_frees = 0u;
        size_t last_pinned_bytes = 0u;
        size_t last_device_bytes = 0u;
        int last_pinned_ordinal = -1;
        int last_device_ordinal = -1;
    };

    TransferStagingBackendSpy *transfer_staging_backend_spy = nullptr;

    /** @return Active device-free staging backend for TransferEngine injection. */
    IBackend *resolveTransferStagingBackend(DeviceId)
    {
        return transfer_staging_backend_spy;
    }

    /** @brief Device-free backend spy for reclamation receipt semantics. */
    class ReclamationBackendSpy final : public MockBackend
    {
    public:
        ReclamationBackendSpy() : MockBackend(DeviceType::CUDA) {}

        /** @brief Return the configured raw backend result and record identity. */
        DeviceMemoryCacheReclamationResult
        trimUnusedDeviceMemoryCaches(int device_id) override
        {
            ++calls;
            last_device_id = device_id;
            return result;
        }

        /** @brief Return exact injected canonical allocation ownership. */
        DeviceAllocationAccounting
        deviceAllocationAccounting(int device_id) const override
        {
            ++allocation_accounting_calls;
            last_allocation_accounting_device_id = device_id;
            return allocation_accounting;
        }

        /** @brief Model one exact successful runtime-generation reset. */
        DeviceRuntimeGenerationRetirementResult
        retireExclusiveDeviceRuntimeGeneration(
            const DeviceRuntimeGenerationRetirementRequest &request) override
        {
            ++runtime_reset_calls;
            last_runtime_reset_device_id = request.deviceOrdinal();
            runtime_reset_device_ids.push_back(request.deviceOrdinal());

            /* A batch reset is correct only if every participant worker was
             * already destroyed and its acquisition marker remains installed.
             * Probe through the public pool API at the exact backend reset
             * boundary; only its typed logic_error proves exclusion. */
            if (!required_excluded_cuda_ordinals.empty())
            {
                bool all_participants_excluded = true;
                for (const int ordinal :
                     required_excluded_cuda_ordinals)
                {
                    try
                    {
                        (void)GPUDeviceContextPool::instance().getContext(
                            DeviceId::cuda(ordinal));
                        all_participants_excluded = false;
                    }
                    catch (const std::logic_error &)
                    {
                        // This is the one valid acquisition result in reset.
                    }
                    catch (const std::exception &)
                    {
                        all_participants_excluded = false;
                    }
                }
                batch_exclusion_observations.push_back(
                    all_participants_excluded);
            }

            DeviceRuntimeGenerationRetirementResult receipt;
            receipt.supported = true;
            receipt.success = runtime_reset_succeeds;
            receipt.reset_invoked = runtime_reset_succeeds;
            receipt.retired_generation = runtime_generation;
            receipt.successor_generation = runtime_reset_succeeds
                                                ? runtime_generation + 1u
                                                : 0u;
            receipt.post_reset_state =
                runtime_reset_succeeds
                    ? DeviceRuntimePostResetState::Quiescent
                    : DeviceRuntimePostResetState::Unverified;
            receipt.driver_free_bytes_before = driver_free_bytes;
            receipt.diagnostic = runtime_reset_succeeds
                                     ? "injected runtime reset"
                                     : "injected runtime reset failure";
            if (runtime_reset_succeeds)
                ++runtime_generation;
            return receipt;
        }

        /** @brief Return the pre-owner-release observation used by a ticket. */
        size_t deviceMemoryFree(int device_id) const override
        {
            ++free_memory_calls;
            last_free_memory_device_id = device_id;
            return driver_free_bytes;
        }

        DeviceMemoryCacheReclamationResult result; ///< Injected backend evidence.
        DeviceAllocationAccounting allocation_accounting{
            .supported = true,
            .active_allocations = 1u,
            .active_bytes = 384u,
        }; ///< Exact pre-release canonical ownership.
        size_t driver_free_bytes = 1000u; ///< Injected ticket baseline.
        int calls = 0; ///< Number of public-authority invocations.
        int last_device_id = -1; ///< Exact ordinal forwarded by TransferEngine.
        bool runtime_reset_succeeds = true; ///< Injected reset outcome.
        std::uint64_t runtime_generation = 1u; ///< Current fake runtime identity.
        int runtime_reset_calls = 0; ///< Exclusive runtime-reset invocations.
        int last_runtime_reset_device_id = -1; ///< Exact reset ordinal.
        /** Exact reset order observed by the backend authority. */
        std::vector<int> runtime_reset_device_ids;
        /** CUDA ordinals that must all be excluded at every batch reset. */
        std::vector<int> required_excluded_cuda_ordinals;
        /** One all-participant exclusion observation per runtime reset. */
        std::vector<bool> batch_exclusion_observations;
        mutable int free_memory_calls = 0; ///< Ticket baseline observations.
        mutable int last_free_memory_device_id = -1; ///< Observed GPU ordinal.
        mutable int allocation_accounting_calls = 0; ///< Ledger observations.
        mutable int last_allocation_accounting_device_id = -1; ///< Ledger GPU.
    };

    ReclamationBackendSpy *reclamation_backend_spy = nullptr;

    /** @return Active device-free backend used by the function-pointer resolver. */
    IBackend *resolveReclamationBackend(DeviceId)
    {
        return reclamation_backend_spy;
    }

    /** @return A complete successful raw receipt for test mutation. */
    DeviceMemoryCacheReclamationResult successfulRawReclamationResult()
    {
        DeviceMemoryCacheReclamationResult result;
        result.supported = true;
        result.success = true;
        result.graph_trim_invoked = true;
        result.async_pool_trim_invoked = true;
        result.before = {
            .driver_free_bytes = 1000u,
            .graph_used_bytes = 64u,
            .graph_reserved_bytes = 128u,
            .async_pool_used_bytes = 32u,
            .async_pool_reserved_bytes = 256u,
            .graph_accounting_available = true,
            .async_pool_accounting_available = true,
        };
        result.after = {
            .driver_free_bytes = 1384u,
            .graph_used_bytes = 0u,
            .graph_reserved_bytes = 0u,
            .async_pool_used_bytes = 0u,
            .async_pool_reserved_bytes = 0u,
            .graph_accounting_available = true,
            .async_pool_accounting_available = true,
        };
        return result;
    }
} // namespace

TEST(Test__TransferEngine_StagingSlab,
     OneAllocationOwnsEveryDisjointConcurrentLaneSlice)
{
    TransferStagingBackendSpy backend;
    transfer_staging_backend_spy = &backend;
    TransferEngine engine(&resolveTransferStagingBackend);

    constexpr size_t kSliceBytes = 4096u;
    constexpr size_t kSliceCount = 49u;
    {
        const auto slices =
            engine.allocatePersistentTransferStagingSlices(
                kSliceBytes,
                kSliceCount,
                DeviceId::cuda(3));

        ASSERT_EQ(slices.size(), kSliceCount);
        EXPECT_EQ(backend.pinned_allocations, 1u);
        EXPECT_EQ(backend.device_allocations, 1u);
        EXPECT_EQ(backend.last_pinned_bytes, kSliceBytes * kSliceCount);
        EXPECT_EQ(backend.last_device_bytes, kSliceBytes * kSliceCount);
        EXPECT_EQ(backend.last_pinned_ordinal, 3);
        EXPECT_EQ(backend.last_device_ordinal, 3);

        const auto *const pinned_base = static_cast<const std::uint8_t *>(
            slices.front().mutablePinnedData());
        const auto *const device_base = static_cast<const std::uint8_t *>(
            slices.front().mutableDeviceData());
        for (size_t index = 0u; index < slices.size(); ++index)
        {
            EXPECT_TRUE(slices[index].valid());
            EXPECT_EQ(slices[index].device(), DeviceId::cuda(3));
            EXPECT_EQ(slices[index].sizeBytes(), kSliceBytes);
            EXPECT_EQ(
                static_cast<const std::uint8_t *>(
                    slices[index].mutablePinnedData()),
                pinned_base + index * kSliceBytes);
            EXPECT_EQ(
                static_cast<const std::uint8_t *>(
                    slices[index].mutableDeviceData()),
                device_base + index * kSliceBytes);
        }
        EXPECT_EQ(backend.pinned_frees, 0u);
        EXPECT_EQ(backend.device_frees, 0u);
    }

    EXPECT_EQ(backend.pinned_frees, 1u);
    EXPECT_EQ(backend.device_frees, 1u);
    transfer_staging_backend_spy = nullptr;
}

TEST(Test__TransferEngine_StagingSlab, RejectsInvalidOrOverflowingGeometry)
{
    TransferStagingBackendSpy backend;
    transfer_staging_backend_spy = &backend;
    TransferEngine engine(&resolveTransferStagingBackend);

    EXPECT_THROW(
        (void)engine.allocatePersistentTransferStagingSlices(
            0u, 1u, DeviceId::cuda(0)),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.allocatePersistentTransferStagingSlices(
            1u, 0u, DeviceId::cuda(0)),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.allocatePersistentTransferStagingSlices(
            1u, 1u, DeviceId::cpu()),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.allocatePersistentTransferStagingSlices(
            std::numeric_limits<size_t>::max(),
            2u,
            DeviceId::cuda(0)),
        std::overflow_error);
    EXPECT_EQ(backend.pinned_allocations, 0u);
    EXPECT_EQ(backend.device_allocations, 0u);
    transfer_staging_backend_spy = nullptr;
}

TEST(Test__TransferEngine_ExecutionStreamPool,
     InvalidGeometryFailsBeforeAnyDeviceContextIsRequired)
{
    TransferEngine engine;
    const PersistentTransferExecutionLane invalid_lane;

    EXPECT_FALSE(invalid_lane.valid());
    EXPECT_EQ(invalid_lane.device(), DeviceId::invalid());
    EXPECT_THROW((void)invalid_lane.stream(), std::logic_error);
    EXPECT_THROW(
        (void)engine.allocatePersistentTransferExecutionLanes(
            0u,
            DeviceId::cuda(0),
            "unit_invalid_zero_width"),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.allocatePersistentTransferExecutionLanes(
            1u,
            DeviceId::cpu(),
            "unit_invalid_cpu_endpoint"),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.allocatePersistentTransferExecutionLanes(
            1u,
            DeviceId::cuda(0),
            ""),
        std::invalid_argument);
}

TEST(Test__TransferEngine_Reclamation, RequestFactoriesRejectInvalidOwnership)
{
    EXPECT_THROW(
        (void)DeviceMemoryReclamationRequest::retiredExecutionTopology(
            DeviceId::cpu()),
        std::invalid_argument);
    EXPECT_THROW(
        (void)DeviceMemoryReclamationRequest::retiredExecutionTopology(
            DeviceId::invalid()),
        std::invalid_argument);

    ReclamationBackendSpy backend;
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);
    EXPECT_THROW(
        (void)engine.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{
                .device = DeviceId::cpu(),
                .prepared_weight_bytes = 1u,
            }),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{
                .device = DeviceId::cuda(0),
            }),
        std::invalid_argument);
    EXPECT_THROW(
        (void)engine.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{
                .device = DeviceId::cuda(0),
                .prepared_weight_bytes =
                    std::numeric_limits<size_t>::max(),
                .reusable_workspace_bytes = 1u,
            }),
        std::invalid_argument);
    EXPECT_EQ(backend.free_memory_calls, 0);
    reclamation_backend_spy = nullptr;
}

TEST(Test__TransferEngine_Reclamation, PublicAuthorityReturnsCompleteReceipt)
{
    ReclamationBackendSpy backend;
    backend.result = successfulRawReclamationResult();
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);

    const DeviceMemoryReclamationReceipt receipt =
        engine.reclaimDeviceMemory(
            DeviceMemoryReclamationRequest::retiredExecutionTopology(
                DeviceId::cuda(3)));

    EXPECT_EQ(backend.calls, 1);
    EXPECT_EQ(backend.last_device_id, 3);
    EXPECT_EQ(receipt.device, DeviceId::cuda(3));
    EXPECT_EQ(receipt.reclaimedDriverBytes(), 384u);
    EXPECT_EQ(receipt.graph_reserved_bytes_before, 128u);
    EXPECT_EQ(receipt.graph_reserved_bytes_after, 0u);
    EXPECT_EQ(receipt.async_pool_reserved_bytes_before, 256u);
    EXPECT_EQ(receipt.async_pool_reserved_bytes_after, 0u);
    EXPECT_TRUE(receipt.graph_trim_invoked);
    EXPECT_TRUE(receipt.async_pool_trim_invoked);
    reclamation_backend_spy = nullptr;
}

TEST(Test__TransferEngine_Reclamation, BackendFailureCannotForgeReceipt)
{
    ReclamationBackendSpy backend;
    backend.result.diagnostic = "injected trim failure";
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);

    EXPECT_THROW(
        (void)engine.reclaimDeviceMemory(
            DeviceMemoryReclamationRequest::retiredExecutionTopology(
                DeviceId::cuda(0))),
        std::runtime_error);
    reclamation_backend_spy = nullptr;
}

TEST(Test__TransferEngine_Reclamation,
     ExclusiveRetirementRejectsBOMBeyondCanonicalOwnership)
{
    ReclamationBackendSpy backend;
    backend.result = successfulRawReclamationResult();
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);

    EXPECT_THROW(
        (void)engine.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{
                .device = DeviceId::cuda(0),
                .prepared_weight_bytes = 300u,
                .reusable_workspace_bytes = 85u,
            }),
        std::runtime_error);

    auto exact_ticket = engine.beginExclusiveModelRetirement(
        ModelDeviceMemoryRetention{
            .device = DeviceId::cuda(0),
            .prepared_weight_bytes = 300u,
            .reusable_workspace_bytes = 84u,
        });
    EXPECT_NO_THROW(
        (void)engine.completeExclusiveModelRetirement(
            std::move(exact_ticket)));
    EXPECT_EQ(backend.free_memory_calls, 1);
    EXPECT_EQ(backend.last_free_memory_device_id, 0);
    EXPECT_EQ(backend.allocation_accounting_calls, 2);
    EXPECT_EQ(backend.last_allocation_accounting_device_id, 0);
    EXPECT_EQ(backend.runtime_reset_calls, 1);
    EXPECT_EQ(backend.last_runtime_reset_device_id, 0);
    EXPECT_EQ(backend.calls, 0);
    reclamation_backend_spy = nullptr;
}

TEST(Test__TransferEngine_Reclamation,
     ExclusiveTicketUsesCanonicalLedgerInsteadOfDriverDelta)
{
    ReclamationBackendSpy backend;
    backend.driver_free_bytes = 1000u;
    backend.result = successfulRawReclamationResult();
    backend.result.before.driver_free_bytes = 1300u;
    backend.result.after.driver_free_bytes = 1384u;
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);

    auto ticket = engine.beginExclusiveModelRetirement(
        ModelDeviceMemoryRetention{
            .device = DeviceId::cuda(4),
            .prepared_weight_bytes = 320u,
            .reusable_workspace_bytes = 64u,
        });
    EXPECT_EQ(ticket.driverFreeBytesBeforeOwnerRelease(), 1000u);
    EXPECT_EQ(ticket.expectedRetiredBytes(), 384u);
    EXPECT_EQ(ticket.canonicalAllocationsBeforeOwnerRelease(), 1u);
    EXPECT_EQ(ticket.canonicalAllocationBytesBeforeOwnerRelease(), 384u);

    const auto receipt = engine.completeExclusiveModelRetirement(
        std::move(ticket));
    EXPECT_EQ(receipt.reclaimedDriverBytes(), 0u);
    EXPECT_EQ(receipt.driverBytesVisibleSinceOwnerRelease(), 0u);
    EXPECT_EQ(receipt.releasedCanonicalBytes(), 384u);
    EXPECT_EQ(receipt.expected_retired_bytes, 384u);
    EXPECT_EQ(receipt.canonical_allocations_before_owner_release, 1u);
    EXPECT_EQ(receipt.canonical_allocations_before_runtime_reset, 0u);
    EXPECT_EQ(receipt.driver_free_bytes_before_owner_release, 1000u);
    EXPECT_TRUE(receipt.runtime_reset_invoked);
    EXPECT_EQ(receipt.retired_runtime_generation, 1u);
    EXPECT_EQ(receipt.successor_runtime_generation, 2u);
    EXPECT_EQ(
        receipt.runtime_post_reset_state,
        DeviceRuntimePostResetState::Quiescent);
    EXPECT_EQ(backend.last_runtime_reset_device_id, 4);
    EXPECT_EQ(backend.calls, 0);
    EXPECT_FALSE(ticket.valid());
    EXPECT_THROW(
        (void)engine.completeExclusiveModelRetirement(
            std::move(ticket)),
        std::logic_error);
    reclamation_backend_spy = nullptr;
}

TEST(Test__TransferEngine_Reclamation,
     MultiDeviceBatchExcludesEveryWorkerBeforeFirstRuntimeReset)
{
    llaminar2::testing::installHardwareFreeGPUContextFactories();
    auto &pool = GPUDeviceContextPool::instance();
    constexpr int kFirstOrdinal = 61;
    constexpr int kSecondOrdinal = 62;
    ASSERT_TRUE(
        pool.getContext(DeviceId::cuda(kFirstOrdinal)).isInitialized());
    ASSERT_TRUE(
        pool.getContext(DeviceId::cuda(kSecondOrdinal)).isInitialized());

    ReclamationBackendSpy backend;
    backend.required_excluded_cuda_ordinals = {
        kFirstOrdinal,
        kSecondOrdinal,
    };
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);

    std::vector<ExclusiveModelRetirementTicket> tickets;
    tickets.reserve(2u);
    tickets.push_back(engine.beginExclusiveModelRetirement(
        ModelDeviceMemoryRetention{
            .device = DeviceId::cuda(kFirstOrdinal),
            .prepared_weight_bytes = 300u,
            .reusable_workspace_bytes = 84u,
        }));
    tickets.push_back(engine.beginExclusiveModelRetirement(
        ModelDeviceMemoryRetention{
            .device = DeviceId::cuda(kSecondOrdinal),
            .prepared_weight_bytes = 300u,
            .reusable_workspace_bytes = 84u,
        }));

    const auto receipts = engine.completeExclusiveModelRetirements(
        std::move(tickets));
    ASSERT_EQ(receipts.size(), 2u);
    EXPECT_EQ(receipts[0].device, DeviceId::cuda(kFirstOrdinal));
    EXPECT_EQ(receipts[1].device, DeviceId::cuda(kSecondOrdinal));
    EXPECT_EQ(
        backend.runtime_reset_device_ids,
        (std::vector<int>{kFirstOrdinal, kSecondOrdinal}));
    ASSERT_EQ(backend.batch_exclusion_observations.size(), 2u);
    EXPECT_TRUE(backend.batch_exclusion_observations[0]);
    EXPECT_TRUE(backend.batch_exclusion_observations[1]);

    /* The scopes release together after every reset and receipt is complete;
     * the next model generation must be able to acquire both endpoints. */
    EXPECT_TRUE(
        pool.getContext(DeviceId::cuda(kFirstOrdinal)).isInitialized());
    EXPECT_TRUE(
        pool.getContext(DeviceId::cuda(kSecondOrdinal)).isInitialized());
    (void)pool.retireExclusiveGeneration(DeviceId::cuda(kFirstOrdinal));
    (void)pool.retireExclusiveGeneration(DeviceId::cuda(kSecondOrdinal));
    reclamation_backend_spy = nullptr;
}

TEST(Test__TransferEngine_Reclamation,
     MultiDeviceBatchRejectsEmptyAndDuplicateParticipantSets)
{
    ReclamationBackendSpy backend;
    reclamation_backend_spy = &backend;
    TransferEngine engine(&resolveReclamationBackend);

    std::vector<ExclusiveModelRetirementTicket> empty;
    EXPECT_THROW(
        (void)engine.completeExclusiveModelRetirements(std::move(empty)),
        std::invalid_argument);

    std::vector<ExclusiveModelRetirementTicket> duplicates;
    duplicates.reserve(2u);
    for (int index = 0; index < 2; ++index)
    {
        duplicates.push_back(engine.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{
                .device = DeviceId::cuda(11),
                .prepared_weight_bytes = 300u,
                .reusable_workspace_bytes = 84u,
            }));
    }
    EXPECT_THROW(
        (void)engine.completeExclusiveModelRetirements(
            std::move(duplicates)),
        std::invalid_argument);
    EXPECT_EQ(backend.runtime_reset_calls, 0);
    reclamation_backend_spy = nullptr;
}

// ============================================================================
// execute() tests — uses MockBackend
// ============================================================================

class Test__TransferEngine_Execute : public ::testing::Test
{
protected:
    void SetUp() override
    {
        llaminar2::testing::installHardwareFreeGPUContextFactories();
        mock_backend_ = std::make_shared<MockBackend>(DeviceType::CUDA);

        // Create engine with custom resolver that returns our mock
        resolver_ = [](DeviceId) -> IBackend *
        {
            // The lambda captures nothing; we use a static pointer.
            return s_mock_;
        };
    }

    // Static mock pointer for the resolver lambda
    static MockBackend *s_mock_;
    std::shared_ptr<MockBackend> mock_backend_;
    TransferEngine::BackendResolver resolver_;
};

MockBackend *Test__TransferEngine_Execute::s_mock_ = nullptr;

TEST_F(Test__TransferEngine_Execute, HostToDevice_RecordsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Prepare host data
    float host_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Allocate "device" memory via mock
    void *device_ptr = mock_backend_->allocate(sizeof(host_data), 0);
    ASSERT_NE(device_ptr, nullptr);

    // Build request
    MemoryDescriptor desc;
    desc.host_ptr = host_data;
    desc.device = DeviceId::cpu();
    desc.size_bytes = sizeof(host_data);
    desc.residency = MemoryResidency::STANDARD;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::HOST_TO_DEVICE;
    req.target_ptr = device_ptr;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);

    // Verify mock recorded the transfer
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 1u);
    EXPECT_EQ(stats.h2d_bytes, sizeof(host_data));

    // Verify data was copied (MockBackend does real memcpy)
    auto *result_data = static_cast<float *>(device_ptr);
    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
    EXPECT_FLOAT_EQ(result_data[3], 4.0f);

    mock_backend_->free(device_ptr, 0);
}

TEST_F(Test__TransferEngine_Execute, DeviceToHost_RecordsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Allocate "device" memory and fill it
    constexpr size_t bytes = 4 * sizeof(float);
    void *device_ptr = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(device_ptr, nullptr);
    float device_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    std::memcpy(device_ptr, device_data, bytes);

    // Host destination
    float host_data[4] = {0};

    MemoryDescriptor desc;
    desc.host_ptr = host_data;
    desc.device_ptr = device_ptr;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = bytes;
    desc.residency = MemoryResidency::STANDARD;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cpu();
    req.method = TransferMethod::DEVICE_TO_HOST;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
    EXPECT_EQ(stats.d2h_bytes, bytes);

    // Verify data
    EXPECT_FLOAT_EQ(host_data[0], 10.0f);
    EXPECT_FLOAT_EQ(host_data[3], 40.0f);

    mock_backend_->free(device_ptr, 0);
}

TEST_F(Test__TransferEngine_Execute, HostStaged_RecordsBothTransfers)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Source "device" memory (simulating CUDA)
    constexpr size_t bytes = 4 * sizeof(float);
    void *src_device = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(src_device, nullptr);
    float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(src_device, src_data, bytes);

    // Target "device" memory (simulating ROCm)
    void *dst_device = mock_backend_->allocate(bytes, 0);
    ASSERT_NE(dst_device, nullptr);

    // Host bounce buffer
    float host_bounce[4] = {0};

    MemoryDescriptor desc;
    desc.host_ptr = host_bounce;
    desc.device_ptr = src_device;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = bytes;
    desc.residency = MemoryResidency::STANDARD;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::rocm(0);
    req.method = TransferMethod::HOST_STAGED;
    req.target_ptr = dst_device;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::HOST_STAGED);

    // Should have done D2H + H2D
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
    EXPECT_EQ(stats.h2d_count, 1u);

    // Verify data arrived at destination
    auto *result_data = static_cast<float *>(dst_device);
    EXPECT_FLOAT_EQ(result_data[0], 1.0f);
    EXPECT_FLOAT_EQ(result_data[3], 4.0f);

    mock_backend_->free(src_device, 0);
    mock_backend_->free(dst_device, 0);
}

TEST_F(Test__TransferEngine_Execute, Noop_NoBackendCall)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    MemoryDescriptor desc;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = 100;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::NOOP;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, MappedNoop_NoBackendCall)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    MemoryDescriptor desc;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = 100;
    desc.residency = MemoryResidency::MAPPED;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cpu();
    req.method = TransferMethod::MAPPED_NOOP;

    auto result = engine.execute(req);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::MAPPED_NOOP);

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

// ============================================================================
// HOST_RESIDENT high-level tests — verifies upload/uploadFull/transferActivation
// skip device allocation and transfer for HOST_RESIDENT tensors.
// All paths exit before any backend interaction, so MockBackend is sufficient.
// ============================================================================

TEST_F(Test__TransferEngine_Execute, Upload_HostResident_SkipsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Create a tensor and mark it HOST_RESIDENT
    auto tensor = TestTensorFactory::createFP32Random({4, 4});
    tensor->setHostResident();

    EXPECT_TRUE(tensor->isHostResident());

    // Upload should succeed as NOOP — no backend calls
    auto result = engine.upload(tensor.get(), DeviceId::cuda(0));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // Verify zero backend interaction
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, UploadFull_HostResident_SkipsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Random({8, 8});
    tensor->setHostResident();

    // uploadFull should also NOOP for HOST_RESIDENT
    auto result = engine.uploadFull(tensor.get(), DeviceId::rocm(0));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // Zero backend interaction
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, TransferActivation_HostResident_SkipsTransfer)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Random({2, 2});
    tensor->setHostResident();

    // transferActivation should NOOP for HOST_RESIDENT
    auto result = engine.transferActivation(tensor.get(), DeviceId::cuda(1));
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // Zero backend interaction
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, Upload_HostResident_HostDataStillAccessible)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setHostResident();

    // Upload does nothing for HOST_RESIDENT
    auto result = engine.upload(tensor.get(), DeviceId::cuda(0));
    EXPECT_TRUE(result.success);

    // Host data remains accessible and valid — no device buffer was allocated
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < 16; ++i)
    {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(Test__TransferEngine_Execute, Upload_HostResident_MultipleCallsStillNoop)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Random({4, 4});
    tensor->setHostResident();

    // Multiple uploads to different devices should all NOOP
    for (int i = 0; i < 3; ++i)
    {
        auto result = engine.upload(tensor.get(), DeviceId::rocm(i));
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.method_used, TransferMethod::NOOP);
    }

    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, Upload_TensorSliceMutatesBackingStorageOwner)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto inner = TestTensorFactory::createFP32Ones({4, 4});
    inner->setBackendForTesting(mock_backend_.get());
    TensorBase *inner_ptr = inner.get();

    SliceMetadata metadata{
        .mode = SliceMode::ROW_PARALLEL,
        .original_rows = 4,
        .original_cols = 4,
        .slice_start = 0,
        .slice_end = 4,
        .rank = 0,
        .world_size = 1,
        .inner_is_presliced = true,
    };
    std::unique_ptr<TensorBase> storage_owner = std::move(inner);
    TensorSlice slice(std::move(storage_owner), metadata);

    const auto result = engine.upload(&slice, DeviceId::cuda(0));

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_EQ(slice.transferStorageOwner(), inner_ptr);
    EXPECT_EQ(slice.current_device(), DeviceId::cuda(0));
    EXPECT_NE(slice.gpu_data_ptr(), nullptr);
    EXPECT_EQ(slice.gpu_data_ptr(), inner_ptr->gpu_data_ptr());

    const auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 1u);
    EXPECT_EQ(stats.d2h_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, Download_TensorSliceReadsBackingStorageOwner)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto inner = TestTensorFactory::createFP32Ones({4, 4});
    inner->setBackendForTesting(mock_backend_.get());
    TensorBase *inner_ptr = inner.get();

    SliceMetadata metadata{
        .mode = SliceMode::ROW_PARALLEL,
        .original_rows = 4,
        .original_cols = 4,
        .slice_start = 0,
        .slice_end = 4,
        .rank = 0,
        .world_size = 1,
        .inner_is_presliced = true,
    };
    std::unique_ptr<TensorBase> storage_owner = std::move(inner);
    TensorSlice slice(std::move(storage_owner), metadata);

    ASSERT_TRUE(engine.upload(&slice, DeviceId::cuda(0)).success);
    TransferEngine::publishDeviceWrite(
        &slice,
        DeviceId::cuda(0),
        reinterpret_cast<void *>(0x1234));
    mock_backend_->resetTransferStats();

    const auto result = engine.download(&slice);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_EQ(slice.transferStorageOwner(), inner_ptr);
    EXPECT_TRUE(slice.hostValid());

    const auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 1u);
}

// ============================================================================
// Error handling tests
// ============================================================================

TEST_F(Test__TransferEngine_Execute, HostToDevice_NullSourceHostPtr_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    MemoryDescriptor desc;
    desc.host_ptr = nullptr; // No host data!
    desc.size_bytes = 100;

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cuda(0);
    req.method = TransferMethod::HOST_TO_DEVICE;
    req.target_ptr = reinterpret_cast<void *>(0xDEAD);

    auto result = engine.execute(req);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST_F(Test__TransferEngine_Execute, DeviceToHost_NullDevicePtr_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    float host_buf[4];
    MemoryDescriptor desc;
    desc.host_ptr = host_buf;
    desc.device_ptr = nullptr; // No device data!
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = sizeof(host_buf);

    TransferRequest req;
    req.source = desc;
    req.target_device = DeviceId::cpu();
    req.method = TransferMethod::DEVICE_TO_HOST;

    auto result = engine.execute(req);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

// ============================================================================
// copyActivation() tests — tensor→tensor copy with transport auto-selection.
//
// GPU-buffer paths (same-device D2D, same-vendor peer copy, cross-vendor
// host-staging) require real device backends and are exercised in the
// integration suite (integration/transfer/Test__TransferEngine_CopyActivation).
// Here we cover the host/CPU-destination path and the argument guards, which
// are fully deterministic without a GPU.
// ============================================================================

TEST_F(Test__TransferEngine_Execute, CopyActivation_NullSrc_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto dst = TestTensorFactory::createFP32({4, 4});
    auto result = engine.copyActivation(nullptr, dst.get(), DeviceId::cpu(), 64);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST_F(Test__TransferEngine_Execute, CopyActivation_NullDst_Fails)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto src = TestTensorFactory::createFP32({4, 4});
    auto result = engine.copyActivation(src.get(), nullptr, DeviceId::cpu(), 64);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST_F(Test__TransferEngine_Execute, CopyActivation_ZeroBytes_Noop)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    auto src = TestTensorFactory::createFP32({4, 4});
    auto dst = TestTensorFactory::createFP32({4, 4});

    auto result = engine.copyActivation(src.get(), dst.get(), DeviceId::cpu(), 0);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::NOOP);

    // No backend interaction for a zero-byte copy
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
    EXPECT_EQ(stats.d2d_count, 0u);
}

TEST_F(Test__TransferEngine_Execute, CopyActivation_CpuDestination_HostMemcpy)
{
    s_mock_ = mock_backend_.get();
    TransferEngine engine(resolver_);

    // Source filled with a known pattern; destination starts zeroed.
    auto src = TestTensorFactory::createFP32({4, 4});
    auto dst = TestTensorFactory::createFP32({4, 4});
    float *src_data = src->mutable_data();
    float *dst_data = dst->mutable_data();
    for (size_t i = 0; i < src->numel(); ++i)
    {
        src_data[i] = static_cast<float>(i) + 0.5f;
        dst_data[i] = 0.0f;
    }

    const size_t bytes = src->numel() * sizeof(float);
    auto result = engine.copyActivation(src.get(), dst.get(), DeviceId::cpu(), bytes);

    EXPECT_TRUE(result.success);

    // CPU destination is a pure host-side copy: no device transfers occur.
    auto stats = mock_backend_->getTransferStats();
    EXPECT_EQ(stats.h2d_count, 0u);
    EXPECT_EQ(stats.d2h_count, 0u);
    EXPECT_EQ(stats.d2d_count, 0u);

    // Data must have landed in the destination host buffer.
    const float *out = dst->data();
    for (size_t i = 0; i < dst->numel(); ++i)
    {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 0.5f);
    }
}

// ============================================================================
// to_string tests
// ============================================================================

TEST(Test__TransferEngine_Strings, TransferMethodToString)
{
    EXPECT_EQ(to_string(TransferMethod::NOOP), "NOOP");
    EXPECT_EQ(to_string(TransferMethod::HOST_TO_DEVICE), "HOST_TO_DEVICE");
    EXPECT_EQ(to_string(TransferMethod::DEVICE_TO_HOST), "DEVICE_TO_HOST");
    EXPECT_EQ(to_string(TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND), "DEVICE_TO_DEVICE_SAME_BACKEND");
    EXPECT_EQ(to_string(TransferMethod::HOST_STAGED), "HOST_STAGED");
    EXPECT_EQ(to_string(TransferMethod::MAPPED_NOOP), "MAPPED_NOOP");
}

// ============================================================================
// MemoryDescriptor tests
// ============================================================================

TEST(Test__TransferEngine_Descriptor, Describe_ContainsDevice)
{
    MemoryDescriptor desc;
    desc.device = DeviceId::cuda(0);
    desc.size_bytes = 1024;
    desc.residency = MemoryResidency::STANDARD;

    std::string s = desc.describe();
    EXPECT_NE(s.find("1024"), std::string::npos);
    EXPECT_NE(s.find("STANDARD"), std::string::npos);
}

TEST(Test__TransferEngine_Descriptor, TransferResult_Ok)
{
    auto r = TransferResult::ok(TransferMethod::HOST_TO_DEVICE, 42);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_EQ(r.elapsed_ns, 42u);
    EXPECT_TRUE(r.error.empty());
}

TEST(Test__TransferEngine_Descriptor, TransferResult_Fail)
{
    auto r = TransferResult::fail(TransferMethod::DEVICE_TO_HOST, "test error");
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.error, "test error");
}

// ============================================================================
// GPU_ONLY planTransfer tests — GPU_ONLY uses standard transfer logic
// ============================================================================

TEST(Test__TransferEngine_Plan, GpuOnly_CpuToCuda_HostToDevice)
{
    // GPU_ONLY does NOT affect planTransfer — data still moves H2D normally.
    // The host release happens AFTER the transfer, not instead of it.
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::cuda(0), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, GpuOnly_CpuToRocm_HostToDevice)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cpu(), DeviceId::rocm(0), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::HOST_TO_DEVICE);
}

TEST(Test__TransferEngine_Plan, GpuOnly_SameDevice_Noop)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cuda(0), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::NOOP);
}

TEST(Test__TransferEngine_Plan, GpuOnly_GpuToCpu_DeviceToHost)
{
    auto method = TransferEngine::planTransfer(
        DeviceId::cuda(0), DeviceId::cpu(), MemoryResidency::GPU_ONLY);
    EXPECT_EQ(method, TransferMethod::DEVICE_TO_HOST);
}

// ============================================================================
// GPU_ONLY TensorBase API tests
// ============================================================================

TEST(Test__TransferEngine_GpuOnly, SetGpuOnly_SetsResidency)
{
    auto tensor = TestTensorFactory::createFP32Random({4, 4});
    EXPECT_FALSE(tensor->isGpuOnly());
    EXPECT_EQ(tensor->memoryResidency(), MemoryResidency::STANDARD);

    tensor->setGpuOnly();
    EXPECT_TRUE(tensor->isGpuOnly());
    EXPECT_EQ(tensor->memoryResidency(), MemoryResidency::GPU_ONLY);
}

TEST(Test__TransferEngine_GpuOnly, GpuOnly_MutuallyExclusive_WithHostResident)
{
    auto tensor = TestTensorFactory::createFP32Random({4, 4});

    tensor->setGpuOnly();
    EXPECT_TRUE(tensor->isGpuOnly());
    EXPECT_FALSE(tensor->isHostResident());

    tensor->setHostResident();
    EXPECT_FALSE(tensor->isGpuOnly());
    EXPECT_TRUE(tensor->isHostResident());
}

TEST(Test__TransferEngine_GpuOnly, MemoryResidency_ToString)
{
    EXPECT_EQ(to_string(MemoryResidency::GPU_ONLY), "GPU_ONLY");
}

TEST(Test__TransferEngine_GpuOnly, ReleaseHostWeightData_FreesMemory)
{
    auto tensor = TestTensorFactory::createFP32Ones({32, 32});
    EXPECT_FALSE(tensor->is_raw_data_released());

    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());
}

TEST(Test__TransferEngine_GpuOnly, ReleaseHostWeightData_Idempotent)
{
    auto tensor = TestTensorFactory::createFP32Ones({32, 32});

    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());

    // Second call is safe (no-op)
    tensor->release_host_weight_data();
    EXPECT_TRUE(tensor->is_raw_data_released());
}

// ============================================================================
// Event publication/wait failures — every required dependency fails closed.
// ============================================================================

namespace
{
    /// Non-null sentinel used as a mock producer stream without GPU work.
    void *mockProducerStream()
    {
        return reinterpret_cast<void *>(0x5055424C);
    }

    /**
     * @brief MockBackend subclass with configurable event wait failure.
     *
     * Allows tests to make waitForEvent() return false on demand,
     * simulating corrupted or invalid completion events (e.g., events
     * recorded during CUDA graph capture that are invalid for synchronize).
     */
    class FailableEventMockBackend : public MockBackend
    {
    public:
        FailableEventMockBackend() : MockBackend(DeviceType::CUDA) {}

        bool hostToDeviceOnStream(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream) override
        {
            async_h2d_count_++;
            return MockBackend::hostToDevice(
                dst, src, bytes, device_id, stream);
        }

        bool deviceToHostOnStream(
            void *dst,
            const void *src,
            size_t bytes,
            int device_id,
            void *stream) override
        {
            async_d2h_count_++;
            return MockBackend::deviceToHost(
                dst, src, bytes, device_id, stream);
        }

        /** @brief Allocate stable host storage for typed captured-copy tests. */
        void *allocatePinned(size_t bytes, int device_id) override
        {
            (void)device_id;
            ++pinned_allocation_count_;
            return std::malloc(bytes);
        }

        /** @brief Release one allocation created by @ref allocatePinned. */
        void freePinned(void *ptr, int device_id) override
        {
            (void)device_id;
            if (!ptr)
                return;
            ++pinned_free_count_;
            std::free(ptr);
        }

        bool waitForEvent(void *event, int device_id) override
        {
            // Still record the operation for test inspection
            MockBackend::waitForEvent(event, device_id);
            host_event_wait_count_++;
            return !fail_event_wait_;
        }

        bool streamWaitEvent(void *stream, void *event, int device_id) override
        {
            MockBackend::streamWaitEvent(stream, event, device_id);
            stream_event_wait_count_++;
            return !fail_stream_event_wait_;
        }

        void *createEvent(int device_id) override
        {
            if (fail_event_create_)
                return nullptr;
            return MockBackend::createEvent(device_id);
        }

        bool recordEvent(void *event, int device_id, void *stream = nullptr) override
        {
            MockBackend::recordEvent(event, device_id, stream);
            return !fail_event_record_;
        }

        bool synchronize(int device_id) override
        {
            // Track that synchronize was called as a fallback
            sync_fallback_count_++;
            return !fail_synchronize_;
        }

        /// Make waitForEvent() return false from now on
        void setEventWaitFails(bool fail) { fail_event_wait_ = fail; }

        /// Make streamWaitEvent() return false from now on
        void setStreamEventWaitFails(bool fail) { fail_stream_event_wait_ = fail; }

        /// Make createEvent() fail from now on
        void setEventCreateFails(bool fail) { fail_event_create_ = fail; }

        /// Make recordEvent() fail from now on
        void setEventRecordFails(bool fail) { fail_event_record_ = fail; }

        /// Make synchronize() return false from now on
        void setSynchronizeFails(bool fail) { fail_synchronize_ = fail; }

        /// Number of times synchronize() was called (for verifying fallback behavior)
        size_t getSyncFallbackCount() const { return sync_fallback_count_; }

        size_t getHostEventWaitCount() const { return host_event_wait_count_; }
        size_t getStreamEventWaitCount() const { return stream_event_wait_count_; }
        size_t getAsyncH2DCount() const { return async_h2d_count_; }
        size_t getAsyncD2HCount() const { return async_d2h_count_; }
        size_t getPinnedAllocationCount() const
        {
            return pinned_allocation_count_;
        }
        size_t getPinnedFreeCount() const { return pinned_free_count_; }

    private:
        bool fail_event_wait_ = false;
        bool fail_stream_event_wait_ = false;
        bool fail_event_create_ = false;
        bool fail_event_record_ = false;
        bool fail_synchronize_ = false;
        size_t sync_fallback_count_ = 0;
        size_t host_event_wait_count_ = 0;
        size_t stream_event_wait_count_ = 0;
        size_t async_h2d_count_ = 0;
        size_t async_d2h_count_ = 0;
        size_t pinned_allocation_count_ = 0;
        size_t pinned_free_count_ = 0;
    };
} // namespace

class Test__TransferEngine_EventFailure : public ::testing::Test
{
protected:
    /**
     * @brief Keep a persistent mock wait failure scoped to one assertion.
     *
     * A tensor with a queued H2D source may legitimately need to retire that
     * source in its destructor. Resetting the injected backend fault before
     * local tensors unwind keeps the test focused on the requested transfer
     * boundary instead of poisoning the independent lifetime cleanup path.
     */
    class ScopedEventWaitFailure final
    {
    public:
        explicit ScopedEventWaitFailure(FailableEventMockBackend &backend)
            : backend_(backend)
        {
            backend_.setEventWaitFails(true);
        }

        ~ScopedEventWaitFailure()
        {
            backend_.setEventWaitFails(false);
        }

        ScopedEventWaitFailure(const ScopedEventWaitFailure &) = delete;
        ScopedEventWaitFailure &operator=(const ScopedEventWaitFailure &) = delete;
        ScopedEventWaitFailure(ScopedEventWaitFailure &&) = delete;
        ScopedEventWaitFailure &operator=(ScopedEventWaitFailure &&) = delete;

    private:
        FailableEventMockBackend &backend_;
    };

    void SetUp() override
    {
        llaminar2::testing::installHardwareFreeGPUContextFactories();
        mock_ = std::make_shared<FailableEventMockBackend>();

        // Resolver returns our failable mock
        s_failable_mock_ = mock_.get();
        resolver_ = [](DeviceId) -> IBackend *
        {
            return s_failable_mock_;
        };
    }

    /// Helper: set up a tensor on CUDA device with a completion event.
    /// Returns the tensor in DEVICE_AUTHORITATIVE state with a valid event.
    std::unique_ptr<FP32Tensor> createTensorOnDeviceWithEvent()
    {
        auto tensor = TestTensorFactory::createFP32Ones({4, 4});
        tensor->setBackendForTesting(mock_.get());

        // Upload to device (allocates GPU buffer, sets state to SYNCED)
        bool ok = tensor->ensureOnDevice(DeviceId::cuda(0));
        EXPECT_TRUE(ok);

        // Publish DEVICE_AUTHORITATIVE with a completion event on the mock backend.
        TransferEngine::publishDeviceWrite(
            tensor,
            DeviceId::cuda(0),
            mockProducerStream());

        return tensor;
    }

    static FailableEventMockBackend *s_failable_mock_;
    std::shared_ptr<FailableEventMockBackend> mock_;
    TransferEngine::BackendResolver resolver_;
};

FailableEventMockBackend *Test__TransferEngine_EventFailure::s_failable_mock_ = nullptr;

// -----------------------------------------------------------------------------
// downloadFull: the host observes bytes only after the exact D2H event.
// -----------------------------------------------------------------------------

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitFail_ReturnsHardError)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Now make event wait fail — simulates corrupted event from graph capture
    ScopedEventWaitFailure event_wait_failure(*mock_);

    auto result = engine.downloadFull(tensor.get());

    // Must be a hard failure — no silent fallback
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_NE(
        result.error.find("host-publication event wait failed"),
        std::string::npos);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitFail_ErrorMessageMentionsInvalidEvent)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    tensor->setDebugName("test_attention_output");

    ScopedEventWaitFailure event_wait_failure(*mock_);

    auto result = engine.downloadFull(tensor.get());

    EXPECT_FALSE(result.success);
    // Error should mention "invalid" to help diagnosis
    EXPECT_NE(result.error.find("invalid"), std::string::npos);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    DownloadFull_EventWaitFail_QueuesOnlyTheExactD2HBeforeFailing)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    ScopedEventWaitFailure event_wait_failure(*mock_);
    mock_->resetTransferStats();

    engine.downloadFull(tensor.get());

    // Producer ordering is imported on-device, then only the D2H publication
    // event is host-observed. A failure at that final boundary cannot undo an
    // already accepted copy submission.
    auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
    EXPECT_EQ(mock_->getAsyncD2HCount(), 1u);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    DownloadFull_ExplicitProducerStreamStillWaitsForD2HPublication)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    ScopedEventWaitFailure event_wait_failure(*mock_);
    mock_->resetTransferStats();
    mock_->resetEventRecords();

    void *producer_stream = reinterpret_cast<void *>(0x1234);
    auto result = engine.downloadFull(tensor.get(), producer_stream);

    EXPECT_FALSE(result.success)
        << "Host bytes cannot be exposed until the exact D2H publication event.";
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 0u)
        << "The exact producer stream already carries device-side ordering.";
    EXPECT_EQ(mock_->getHostEventWaitCount(), 1u)
        << "Only the newly recorded D2H completion event is host-observed.";
    EXPECT_EQ(mock_->getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
    auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_EventWaitSuccess_TransferSucceeds)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Event wait succeeds (default) — download should work
    mock_->setEventWaitFails(false);
    mock_->resetTransferStats();

    auto result = engine.downloadFull(tensor.get());

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);

    auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 1u);
}

TEST_F(Test__TransferEngine_EventFailure, DownloadFull_NoEvent_FailsClosedWithoutTransfer)
{
    TransferEngine engine(resolver_);

    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    // Upload to device but DON'T set a completion event
    tensor->ensureOnDevice(DeviceId::cuda(0));
    TransferEngine::publishGraphOwnedDeviceWrite(tensor, DeviceId::cuda(0));

    mock_->resetTransferStats();

    auto result = engine.downloadFull(tensor.get());

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_HOST);
    EXPECT_NE(result.error.find("no completion event"), std::string::npos);

    const auto stats = mock_->getTransferStats();
    EXPECT_EQ(stats.d2h_count, 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

// -----------------------------------------------------------------------------
// uploadFull: a resident tensor never turns a dependency into a host wait.
// -----------------------------------------------------------------------------

TEST_F(
    Test__TransferEngine_EventFailure,
    UploadFull_ResidentTensorDoesNotHostWaitItsCompletionEvent)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // A host wait would fail, proving that success cannot be coming from one.
    ScopedEventWaitFailure event_wait_failure(*mock_);

    // No consumer stream was named, so this placement check must retain the
    // published event for a later exact device consumer.
    auto result = engine.uploadFull(tensor.get(), DeviceId::cuda(0));

    EXPECT_TRUE(result.success);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    UploadFull_ResidentTensorLeavesDependencyForExactConsumerStream)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    ScopedEventWaitFailure event_wait_failure(*mock_);

    auto result = engine.uploadFull(tensor.get(), DeviceId::cuda(0));
    ASSERT_TRUE(result.success);

    EXPECT_NO_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)));
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, UploadFull_StreamWaitFail_DoesNotHostWait)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();
    mock_->setStreamEventWaitFails(true);

    auto result = engine.uploadFull(
        tensor.get(),
        DeviceId::cuda(0),
        reinterpret_cast<void *>(0x1234));

    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("Stream event wait failed"), std::string::npos);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u)
        << "A failed device-side dependency must never degrade into a "
           "host-blocking event wait.";
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, PublicationCreateFail_DoesNotPublishDeviceAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(tensor.get(), DeviceId::cuda(0));
    const TensorCoherenceState state_before = tensor->coherenceState();

    mock_->setEventCreateFails(true);
    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor,
            DeviceId::cuda(0),
            mockProducerStream()),
        std::runtime_error);
    EXPECT_EQ(tensor->coherenceState(), state_before)
        << "Authority cannot become externally visible without its event.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    PrepareDeviceInputQueuesH2DAndPublishesExactEventWithoutHostBlocking)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    void *const consumer_stream = reinterpret_cast<void *>(0xA51C0001);

    mock_->resetTransferStats();
    mock_->resetEventRecords();
    TransferEngine::prepareDeviceInput(
        tensor.get(), DeviceId::cuda(0), consumer_stream);

    EXPECT_EQ(mock_->getAsyncH2DCount(), 1u);
    EXPECT_EQ(mock_->getEventCreateCount(), 1u);
    EXPECT_EQ(mock_->getEventRecordCount(), 1u);
    const auto stream_records =
        mock_->getEventRecordsForStream(consumer_stream);
    ASSERT_EQ(stream_records.size(), 1u);
    EXPECT_EQ(stream_records.front().type, MockBackend::EventRecord::RECORD);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
    EXPECT_TRUE(tensor->hostValid());
    EXPECT_TRUE(tensor->deviceValid());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    MutableHostAccessWaitsOnlyForPendingH2DSourceUse)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    void *const consumer_stream = reinterpret_cast<void *>(0xA51C0002);

    TransferEngine::prepareDeviceInput(
        tensor.get(), DeviceId::cuda(0), consumer_stream);
    ASSERT_EQ(mock_->getHostEventWaitCount(), 0u);

    float *const host_data = tensor->mutable_data();
    ASSERT_NE(host_data, nullptr);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 1u)
        << "Host storage reuse must wait only for the queued upload event.";
    EXPECT_EQ(mock_->getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
    EXPECT_TRUE(tensor->hostValid());
    EXPECT_FALSE(tensor->deviceValid());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    TensorDestructionWaitsOnlyForPendingH2DSourceUse)
{
    {
        auto tensor = TestTensorFactory::createFP32Ones({4, 4});
        tensor->setBackendForTesting(mock_.get());
        TransferEngine::prepareDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0xA51C0003));
        ASSERT_EQ(mock_->getHostEventWaitCount(), 0u);
    }

    EXPECT_EQ(mock_->getHostEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, NullProducerStream_DoesNotPublishDeviceAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();
    mock_->resetEventRecords();

    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor,
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_EQ(tensor->coherenceState(), state_before);
    EXPECT_EQ(mock_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, NullInputPreparationStreamFailsBeforePlacement)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    EXPECT_THROW(
        TransferEngine::prepareDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_FALSE(tensor->current_device().has_value());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputRejectsHostOnlyTensorWithoutAllocatingOrUploading)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    const size_t allocations_before = mock_->getAllocationCount();
    const auto transfers_before = mock_->getTransferStats();

    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)),
        std::runtime_error);

    EXPECT_EQ(mock_->getAllocationCount(), allocations_before);
    const auto transfers_after = mock_->getTransferStats();
    EXPECT_EQ(transfers_after.h2d_count, transfers_before.h2d_count);
    EXPECT_EQ(transfers_after.d2h_count, transfers_before.d2h_count);
    EXPECT_FALSE(tensor->current_device().has_value());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputJoinsProducerEventWithoutTransfer)
{
    auto tensor = createTensorOnDeviceWithEvent();
    const size_t allocations_before = mock_->getAllocationCount();
    const auto transfers_before = mock_->getTransferStats();

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        reinterpret_cast<void *>(0x1234));

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getAllocationCount(), allocations_before);
    const auto transfers_after = mock_->getTransferStats();
    EXPECT_EQ(transfers_after.h2d_count, transfers_before.h2d_count);
    EXPECT_EQ(transfers_after.d2h_count, transfers_before.d2h_count);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    PreCaptureJoinMakesExactEventAndStreamCaptureSafe)
{
    auto tensor = createTensorOnDeviceWithEvent();
    void *capture_stream = reinterpret_cast<void *>(0xCA970001);

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        capture_stream);
    ASSERT_EQ(mock_->getStreamEventWaitCount(), 1u);

    {
        GraphCaptureGuard capture_guard;
        EXPECT_NO_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(),
                DeviceId::cuda(0),
                capture_stream));
    }

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u)
        << "Capture must consume the prejoined dependency without importing "
           "the external producer event into the graph.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureRejectsUnpreparedOrDifferentConsumerStreamWithoutBackendWait)
{
    auto tensor = createTensorOnDeviceWithEvent();
    void *prepared_stream = reinterpret_cast<void *>(0xCA970001);
    void *different_stream = reinterpret_cast<void *>(0xCA970002);

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        prepared_stream);
    ASSERT_EQ(mock_->getStreamEventWaitCount(), 1u);

    {
        GraphCaptureGuard capture_guard;
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(),
                DeviceId::cuda(0),
                different_stream),
            std::runtime_error);
    }

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u)
        << "A missing pre-capture dependency is fatal; capture must never try "
           "the backend wait and hope the runtime accepts it.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    NewDevicePublicationInvalidatesPriorPreCaptureJoin)
{
    auto tensor = createTensorOnDeviceWithEvent();
    void *capture_stream = reinterpret_cast<void *>(0xCA970001);

    TransferEngine::requireDeviceInput(
        tensor.get(),
        DeviceId::cuda(0),
        capture_stream);
    ASSERT_EQ(mock_->getStreamEventWaitCount(), 1u);

    TransferEngine::publishDeviceWrite(
        tensor.get(),
        DeviceId::cuda(0),
        mockProducerStream());

    {
        GraphCaptureGuard capture_guard;
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(),
                DeviceId::cuda(0),
                capture_stream),
            std::runtime_error);
    }

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputFailsClosedWhenProducerEventCannotBeJoined)
{
    auto tensor = createTensorOnDeviceWithEvent();
    mock_->setStreamEventWaitFails(true);

    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)),
        std::runtime_error);

    EXPECT_EQ(mock_->getStreamEventWaitCount(), 1u);
    EXPECT_EQ(mock_->getHostEventWaitCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceInputRejectsNullConsumerStreamBeforeResidencyInspection)
{
    auto tensor = createTensorOnDeviceWithEvent();

    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_EQ(mock_->getStreamEventWaitCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, NullOutputPreparationStreamFailsBeforeAllocation)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());

    EXPECT_THROW(
        TransferEngine::prepareDeviceOutput(
            tensor.get(),
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);
    EXPECT_FALSE(tensor->current_device().has_value());
}

TEST_F(
    Test__TransferEngine_EventFailure,
    RequireDeviceOutputAcceptsInvalidPreallocatedBytesWithoutAllocation)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        tensor.get(),
        DeviceId::cuda(0));
    ASSERT_FALSE(tensor->deviceValid());
    const size_t allocations_before = mock_->getAllocationCount();

    EXPECT_NO_THROW(
        TransferEngine::requireDeviceOutput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x0A117001)));

    EXPECT_EQ(mock_->getAllocationCount(), allocations_before)
        << "Execution-time output validation must never allocate.";
    EXPECT_FALSE(tensor->deviceValid())
        << "Storage validation must not publish unwritten bytes.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureRejectsPlacementCapableInputEvenWhenDeviceBytesAreValid)
{
    auto tensor = createTensorOnDeviceWithEvent();
    GraphCaptureGuard capture_guard;

    EXPECT_THROW(
        TransferEngine::prepareDeviceInput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x0A117002)),
        std::logic_error)
        << "A no-op placement call can still import stale generation state into capture.";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureRejectsPlacementCapableOutputEvenWhenStorageExists)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        tensor.get(),
        DeviceId::cuda(0));
    const size_t allocations_before = mock_->getAllocationCount();
    GraphCaptureGuard capture_guard;

    EXPECT_THROW(
        TransferEngine::prepareDeviceOutput(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x0A117003)),
        std::logic_error);
    EXPECT_EQ(mock_->getAllocationCount(), allocations_before);
}

TEST_F(Test__TransferEngine_EventFailure, AllocationOnlyStorageDoesNotPublishAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    mock_->resetEventRecords();

    TransferEngine::allocateDeviceStorage(
        tensor.get(),
        DeviceId::cuda(0));

    ASSERT_TRUE(tensor->current_device().has_value());
    EXPECT_EQ(*tensor->current_device(), DeviceId::cuda(0));
    EXPECT_NE(tensor->gpu_data_ptr(), nullptr);
    EXPECT_FALSE(tensor->deviceValid())
        << "Allocation alone must not publish unwritten device bytes";
    EXPECT_EQ(mock_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    PinnedTransferFamilyCapturesBidirectionalCopiesWithoutSynchronization)
{
    TransferEngine engine(resolver_);
    constexpr size_t kTensorBytes = 16u * sizeof(float);
    constexpr size_t kFamilyBytes = 2u * kTensorBytes;
    void *const capture_stream = reinterpret_cast<void *>(0xCA970101);

    auto pinned = engine.declarePinnedHostBuffer(
        kFamilyBytes, DeviceId::cuda(0));
    ASSERT_NE(pinned, nullptr);
    EXPECT_FALSE(pinned->isBound());
    EXPECT_EQ(pinned->sizeBytes(), kFamilyBytes);
    EXPECT_EQ(pinned->registrationDevice(), DeviceId::cuda(0));
    EXPECT_TRUE(pinned->contains(kTensorBytes, kTensorBytes));
    EXPECT_FALSE(pinned->contains(kTensorBytes + 1u, kTensorBytes));
    EXPECT_THROW((void)pinned->mutableData(), std::out_of_range);

    engine.bindPinnedHostBuffer(*pinned);
    engine.bindPinnedHostBuffer(*pinned);
    ASSERT_TRUE(pinned->isBound());
    EXPECT_EQ(mock_->getPinnedAllocationCount(), 1u)
        << "Binding a declared graph identity must be idempotent.";

    auto *const source = static_cast<float *>(pinned->mutableData());
    for (size_t index = 0; index < 16u; ++index)
        source[index] = static_cast<float>(index) + 0.25f;

    auto tensor = TestTensorFactory::createFP32Zeros({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        tensor.get(), DeviceId::cuda(0));
    ASSERT_FALSE(tensor->deviceValid());

    int input_publish_stage = 0;
    int output_publish_stage = 0;
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages;
    stages.push_back({
        .stage_identity = &input_publish_stage,
        .stage_name = "captured_pinned_h2d",
        .outputs = {tensor->transferStorageOwner()},
    });
    stages.push_back({
        .stage_identity = &output_publish_stage,
        .stage_name = "captured_pinned_d2h",
        .internal_inputs = {{
            .tensor = tensor->transferStorageOwner(),
            .producer_stage_index = 0,
        }},
    });
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0),
        capture_stream,
        std::move(stages),
        "pinned_bidirectional_transaction");

    mock_->resetTransferStats();
    mock_->resetEventRecords();
    {
        GraphCaptureGuard capture_guard(&ledger);
        {
            ScopedGraphCaptureStage input_scope(&input_publish_stage);
            EXPECT_NO_THROW(engine.enqueuePinnedHostToDevice(
                *pinned,
                /*source_offset=*/0,
                tensor.get(),
                /*destination_offset=*/0,
                kTensorBytes,
                DeviceId::cuda(0),
                capture_stream));
            input_scope.complete();
        }
        {
            ScopedGraphCaptureStage output_scope(&output_publish_stage);
            EXPECT_NO_THROW(engine.enqueueDeviceToPinnedHost(
                tensor.get(),
                /*source_offset=*/0,
                *pinned,
                /*destination_offset=*/kTensorBytes,
                kTensorBytes,
                DeviceId::cuda(0),
                capture_stream));
            output_scope.complete();
        }
    }

    EXPECT_EQ(mock_->getAsyncH2DCount(), 1u);
    EXPECT_EQ(mock_->getAsyncD2HCount(), 1u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u)
        << "Captured transfer nodes must use their graph completion event.";
    EXPECT_EQ(mock_->getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
    EXPECT_EQ(
        std::memcmp(
            pinned->data(/*offset=*/0),
            pinned->data(/*offset=*/kTensorBytes),
            kTensorBytes),
        0);
    EXPECT_FALSE(tensor->deviceValid())
        << "Recording a graph transaction must not publish unexecuted bytes.";

    EXPECT_THROW(
        engine.enqueuePinnedHostToDevice(
            *pinned,
            kTensorBytes + 1u,
            tensor.get(),
            0,
            kTensorBytes,
            DeviceId::cuda(0),
            capture_stream),
        std::out_of_range);
    EXPECT_THROW(
        engine.enqueueDeviceToPinnedHost(
            tensor.get(),
            0,
            *pinned,
            kTensorBytes,
            kTensorBytes,
            DeviceId::cuda(0),
            nullptr),
        std::invalid_argument);

    pinned.reset();
    EXPECT_EQ(mock_->getPinnedFreeCount(), 1u);
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerAdmitsOnlyAnEarlierRecordedInternalProducer)
{
    auto internal = TestTensorFactory::createFP32Ones({4, 4});
    auto external = TestTensorFactory::createFP32Ones({4, 4});
    internal->setBackendForTesting(mock_.get());
    external->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(internal.get(), DeviceId::cuda(0));
    TransferEngine::allocateDeviceStorage(external.get(), DeviceId::cuda(0));
    ASSERT_FALSE(internal->deviceValid());
    ASSERT_FALSE(external->deviceValid());

    int producer_stage = 0;
    int consumer_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970001);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages;
    stages.push_back({
        .stage_identity = &producer_stage,
        .stage_name = "producer",
        .outputs = {internal->transferStorageOwner()},
    });
    stages.push_back({
        .stage_identity = &consumer_stage,
        .stage_name = "consumer",
        .external_inputs = {external->transferStorageOwner()},
        .internal_inputs = {{
            .tensor = internal->transferStorageOwner(),
            .producer_stage_index = 0,
        }},
    });
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages), "unit_capture");

    {
        GraphCaptureGuard capture_guard(&ledger);
        {
            ScopedGraphCaptureStage producer_scope(&producer_stage);
            EXPECT_NO_THROW(TransferEngine::publishDeviceWrite(
                internal.get(), DeviceId::cuda(0), capture_stream));
            EXPECT_FALSE(internal->deviceValid())
                << "Recording a producer must not publish unexecuted bytes";
            EXPECT_EQ(mock_->getEventRecordCount(), 0u)
                << "Captured stage publication must not create per-tensor events";
            producer_scope.complete();
        }
        {
            ScopedGraphCaptureStage consumer_scope(&consumer_stage);
            EXPECT_NO_THROW(TransferEngine::requireDeviceInput(
                internal.get(), DeviceId::cuda(0), capture_stream));
            EXPECT_THROW(
                TransferEngine::requireDeviceInput(
                    external.get(), DeviceId::cuda(0), capture_stream),
                std::runtime_error)
                << "An allocated external tensor still needs globally valid bytes";
            consumer_scope.complete();
        }
    }

    EXPECT_FALSE(internal->deviceValid());
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            internal.get(), DeviceId::cuda(0), capture_stream),
        std::runtime_error)
        << "The internal-edge proof must not escape its capture transaction";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerAdmitsOnlyTypedRetainedParentImports)
{
    auto retained_input = TestTensorFactory::createFP32Ones({4, 4});
    retained_input->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        retained_input.get(), DeviceId::cuda(0));
    ASSERT_FALSE(retained_input->deviceValid());

    int child_consumer_stage = 0;
    void *const capture_stream = reinterpret_cast<void *>(0xCA970007);
    void *const unrelated_stream = reinterpret_cast<void *>(0xCA970008);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {{
        .stage_identity = &child_consumer_stage,
        .stage_name = "retained_child_consumer",
        .retained_parent_inputs = {
            retained_input->transferStorageOwner()},
    }};
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0),
        capture_stream,
        std::move(stages),
        "retained_parent_import");

    {
        GraphCaptureGuard capture_guard(&ledger);
        ScopedGraphCaptureStage consumer_scope(&child_consumer_stage);
        EXPECT_NO_THROW(TransferEngine::requireDeviceInput(
            retained_input.get(), DeviceId::cuda(0), capture_stream));
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                retained_input.get(),
                DeviceId::cuda(0),
                unrelated_stream),
            std::logic_error)
            << "A retained-parent proof belongs to one exact capture stream";
        consumer_scope.complete();
    }

    EXPECT_FALSE(retained_input->deviceValid())
        << "A child template must not publish its parent's unexecuted bytes";
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            retained_input.get(), DeviceId::cuda(0), capture_stream),
        std::runtime_error)
        << "The retained-parent proof must not escape its capture transaction";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    SetupCaptureAdmitsOnlyDeclaredArenaFrontierAddresses)
{
    auto declared_frontier = TestTensorFactory::createFP32Ones({4, 4});
    auto undeclared_metadata = TestTensorFactory::createFP32Ones({4, 4});
    declared_frontier->setBackendForTesting(mock_.get());
    undeclared_metadata->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(
        declared_frontier.get(), DeviceId::cuda(0));
    TransferEngine::allocateDeviceStorage(
        undeclared_metadata.get(), DeviceId::cuda(0));
    ASSERT_FALSE(declared_frontier->deviceValid());
    ASSERT_FALSE(undeclared_metadata->deviceValid());

    int setup_stage = 0;
    void *const capture_stream = reinterpret_cast<void *>(0xCA970009);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {{
        .stage_identity = &setup_stage,
        .stage_name = "setup_frontier_consumer",
        .external_inputs = {
            declared_frontier->transferStorageOwner()},
    }};
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0),
        capture_stream,
        std::move(stages),
        "setup_address_only_frontier",
        GraphCaptureDependencyLedger::ExternalInputAuthority::
            BindDeclaredAddressesOnly);

    {
        GraphCaptureGuard capture_guard(&ledger);
        ScopedGraphCaptureStage stage_scope(&setup_stage);
        EXPECT_NO_THROW(TransferEngine::requireDeviceInput(
            declared_frontier.get(), DeviceId::cuda(0), capture_stream));
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                undeclared_metadata.get(), DeviceId::cuda(0), capture_stream),
            std::runtime_error)
            << "Setup may bind only a typed graph-frontier address; weights "
               "and undeclared metadata remain strict external inputs";
        stage_scope.complete();
    }

    EXPECT_FALSE(declared_frontier->deviceValid())
        << "Setup recording must not publish request payload authority";
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            declared_frontier.get(), DeviceId::cuda(0), capture_stream),
        std::runtime_error)
        << "The address-only proof must not escape its setup capture transaction";
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerAdmitsSameStageScratchOnlyAfterExactPublication)
{
    auto scratch = TestTensorFactory::createFP32Ones({4, 4});
    scratch->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(scratch.get(), DeviceId::cuda(0));
    ASSERT_FALSE(scratch->deviceValid());

    int compound_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970005);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {{
        .stage_identity = &compound_stage,
        .stage_name = "compound_gate_up_down",
        .outputs = {scratch->transferStorageOwner()},
    }};
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages),
        "same_stage_scratch");

    GraphCaptureGuard capture_guard(&ledger);
    ScopedGraphCaptureStage stage_scope(&compound_stage);
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            scratch.get(), DeviceId::cuda(0), capture_stream),
        std::logic_error)
        << "A pure stage output cannot be consumed before its producer is recorded";
    EXPECT_NO_THROW(TransferEngine::publishDeviceWrite(
        scratch.get(), DeviceId::cuda(0), capture_stream));
    EXPECT_NO_THROW(TransferEngine::requireDeviceInput(
        scratch.get(), DeviceId::cuda(0), capture_stream));
    EXPECT_FALSE(scratch->deviceValid())
        << "Intra-stage capture ordering must not publish unexecuted bytes globally";
    EXPECT_EQ(mock_->getEventRecordCount(), 0u)
        << "Intra-stage capture ordering must not add event nodes";
    stage_scope.complete();
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerRejectsPublicationOutsideDeclaredStageOutputs)
{
    auto declared = TestTensorFactory::createFP32Ones({4, 4});
    auto undeclared = TestTensorFactory::createFP32Ones({4, 4});
    declared->setBackendForTesting(mock_.get());
    undeclared->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(declared.get(), DeviceId::cuda(0));
    TransferEngine::allocateDeviceStorage(undeclared.get(), DeviceId::cuda(0));

    int stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970006);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {{
        .stage_identity = &stage,
        .stage_name = "strict_publication_contract",
        .outputs = {declared->transferStorageOwner()},
    }};
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages),
        "undeclared_publication");

    GraphCaptureGuard capture_guard(&ledger);
    ScopedGraphCaptureStage stage_scope(&stage);
    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            undeclared.get(), DeviceId::cuda(0), capture_stream),
        std::logic_error);
    stage_scope.complete();
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerRejectsInternalInputWhoseProducerIsNotEarlier)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(tensor.get(), DeviceId::cuda(0));

    int invalid_consumer_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970002);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {{
        .stage_identity = &invalid_consumer_stage,
        .stage_name = "consumer_before_producer",
        .internal_inputs = {{
            .tensor = tensor->transferStorageOwner(),
            .producer_stage_index = 0,
        }},
    }};
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages), "invalid_order");

    GraphCaptureGuard capture_guard(&ledger);
    ScopedGraphCaptureStage consumer_scope(&invalid_consumer_stage);
    EXPECT_THROW(
        TransferEngine::requireDeviceInput(
            tensor.get(), DeviceId::cuda(0), capture_stream),
        std::logic_error);
    consumer_scope.complete();
}

TEST_F(
    Test__TransferEngine_EventFailure,
    CaptureLedgerRejectsDifferentConsumerStream)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    TransferEngine::allocateDeviceStorage(tensor.get(), DeviceId::cuda(0));

    int producer_stage = 0;
    int consumer_stage = 0;
    void *capture_stream = reinterpret_cast<void *>(0xCA970003);
    void *different_stream = reinterpret_cast<void *>(0xCA970004);
    std::vector<GraphCaptureDependencyLedger::StagePlan> stages = {
        {
            .stage_identity = &producer_stage,
            .stage_name = "producer",
            .outputs = {tensor->transferStorageOwner()},
        },
        {
            .stage_identity = &consumer_stage,
            .stage_name = "consumer",
            .internal_inputs = {{
                .tensor = tensor->transferStorageOwner(),
                .producer_stage_index = 0,
            }},
        },
    };
    GraphCaptureDependencyLedger ledger(
        DeviceId::cuda(0), capture_stream, std::move(stages), "stream_identity");

    GraphCaptureGuard capture_guard(&ledger);
    {
        ScopedGraphCaptureStage producer_scope(&producer_stage);
        producer_scope.complete();
    }
    {
        ScopedGraphCaptureStage consumer_scope(&consumer_stage);
        EXPECT_THROW(
            TransferEngine::requireDeviceInput(
                tensor.get(), DeviceId::cuda(0), different_stream),
            std::logic_error);
        consumer_scope.complete();
    }
}

TEST_F(Test__TransferEngine_EventFailure, CurrentDevicePublicationRejectsNullStream)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();
    mock_->resetEventRecords();

    EXPECT_THROW(
        TransferEngine::publishCurrentDeviceWrite(tensor, nullptr),
        std::invalid_argument);
    EXPECT_EQ(tensor->coherenceState(), state_before);
    EXPECT_EQ(mock_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_->getEventRecordCount(), 0u);
}

TEST_F(Test__TransferEngine_EventFailure, PublicationRecordFail_DoesNotPublishDeviceAuthority)
{
    auto tensor = TestTensorFactory::createFP32Ones({4, 4});
    tensor->setBackendForTesting(mock_.get());
    ASSERT_TRUE(tensor->ensureOnDevice(DeviceId::cuda(0)));
    const TensorCoherenceState state_before = tensor->coherenceState();

    mock_->setEventRecordFails(true);
    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor.get(),
            DeviceId::cuda(0),
            reinterpret_cast<void *>(0x1234)),
        std::runtime_error);
    EXPECT_EQ(tensor->coherenceState(), state_before)
        << "A failed record must not expose an eventless GPU write.";
}

TEST_F(Test__TransferEngine_EventFailure, UploadFull_EventWaitSuccess_Succeeds)
{
    TransferEngine engine(resolver_);

    auto tensor = createTensorOnDeviceWithEvent();

    // Event wait succeeds
    mock_->setEventWaitFails(false);

    auto result = engine.uploadFull(tensor.get(), DeviceId::cuda(0));

    // Should succeed without needing synchronize fallback
    EXPECT_TRUE(result.success);
    EXPECT_EQ(mock_->getSyncFallbackCount(), 0u);
}
