/**
 * @file Test__WorkspaceRetention.cpp
 * @brief Real-device retirement of heterogeneous workspace backing.
 *
 * A GPU graph owner can retain both GPU and CPU scratch. The reuse inventory
 * must preserve those physical allocation domains before TransferEngine binds
 * a final-owner retirement ticket. These tests allocate only small workspace
 * blocks, prove their canonical claims, then exercise real CUDA/HIP retirement.
 * No model, synthetic inference path, or alternate accounting ledger is used.
 */

#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include "execution/local_execution/device/ReusableExecutionWorkspace.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "transfer/TransferEngine.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

using namespace llaminar2;

namespace
{
    /** @brief Declare one fixed scratch region through the production interface. */
    class ScratchConsumer final : public IWorkspaceConsumer
    {
    public:
        /** @param bytes Immutable physical requirement of this consumer. */
        explicit ScratchConsumer(size_t bytes) : bytes_(bytes) {}

        /** @return A row-independent descriptor; request geometry cannot grow it. */
        WorkspaceRequirements getWorkspaceRequirements(int, int = 0, int = 0) const override
        {
            return {.buffers = {{"retention_scratch", bytes_, 256u, true}}};
        }

        /** @brief Retain the allocator-published manager without owning its storage. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        /** @brief End the consumer's borrow before backing is sealed. */
        void unbindWorkspace() override { workspace_ = nullptr; }
        /** @return Whether the allocator has published the scratch manager. */
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        /** @return Exact manager supplied by the production allocator. */
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

    private:
        size_t bytes_;
        DeviceWorkspaceManager *workspace_ = nullptr;
    };

    /**
     * @brief Prove physical attribution and complete final-owner retirement.
     * @param gpu Exact backend under test; an unavailable backend is a failure.
     */
    void verifyHeterogeneousRetention(DeviceId gpu)
    {
        ASSERT_NE(getBackendFor(gpu), nullptr) << gpu.toString();
        {
            /*
             * The retirement ledger must follow this exact pool allocation,
             * not a LoadOrchestrator that may outlive it during a movement
             * transaction.  Keep this small and model-free so the same
             * invariant runs in every backend's production preflight lane.
             */
            WeightVRAMPool pool;
            pool.planRawWeight(
                "retirement_pool_probe", 16, 16,
                16u * 16u * sizeof(float));
            ASSERT_TRUE(pool.allocate(
                getBackendFor(gpu), gpu.gpu_ordinal(), 0));
            const std::weak_ptr<void> allocation_lifetime =
                pool.persistentAllocationLifetime();
            EXPECT_FALSE(allocation_lifetime.expired());
            pool.release();
            EXPECT_TRUE(allocation_lifetime.expired());
        }
        if (!hasCPUBackend())
            initCPUBackend(-1);
        constexpr size_t gpu_bytes = 4096u;
        constexpr size_t cpu_bytes = 8192u;
        PhysicalMemoryPlanBuilder builder;
        for (const DeviceId device : {gpu, DeviceId::cpu()})
        {
            builder.add(
                PhysicalMemoryResource{
                    .world_rank = 0,
                    .device = device,
                    .total_bytes = 1024u * 1024u,
                    .admission_available_bytes = 1024u * 1024u,
                },
                PhysicalMemoryOwner::ExecutionWorkspace,
                device == gpu ? gpu_bytes : cpu_bytes);
        }
        auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
        auto authority = std::make_shared<PhysicalMemoryAuthority>(admission, 0);
        auto registry = std::make_shared<ReusableExecutionWorkspaceRegistry>();
        {
            std::string error;
            auto lease = registry->acquire(
                {.device = gpu, .first_layer = 0, .last_layer = 0},
                authority, &error);
            ASSERT_NE(lease, nullptr) << error;
            {
                ComputeGraph graph;
                ScratchConsumer on_gpu(gpu_bytes);
                ScratchConsumer on_cpu(cpu_bytes);
                WorkspaceSizingHints hints;
                hints.max_seq_len = hints.n_heads = hints.head_dim = 1;
                hints.d_model = hints.batch_size = hints.vocab_size = 1;
                ASSERT_TRUE(lease->allocator()->allocateForGraph(
                    graph, hints,
                    {{.consumer = &on_gpu, .device = gpu, .m = 1, .n = 1, .k = 1},
                     {.consumer = &on_cpu, .device = DeviceId::cpu(), .m = 1, .n = 1, .k = 1}}));
                EXPECT_TRUE(on_gpu.hasWorkspace());
                EXPECT_TRUE(on_cpu.hasWorkspace());
            }
            // Destroy graph/consumer borrows before publishing backing-only state.
            ASSERT_TRUE(lease->publishReusable(&error)) << error;
        }
        ASSERT_EQ(registry->retainedPrimaryBytes(gpu), gpu_bytes);
        ASSERT_EQ(registry->retainedPrimaryBytes(DeviceId::cpu()), cpu_bytes);
        for (const DeviceId device : {gpu, DeviceId::cpu()})
        {
            EXPECT_EQ(authority->claimedBytes(
                device, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryMaterializationKind::NewAllocation),
                device == gpu ? gpu_bytes : cpu_bytes);
        }

        auto &transfers = TransferEngine::instance();
        auto ticket = transfers.beginExclusiveModelRetirement({
            .device = gpu,
            .reusable_workspace_bytes = registry->retainedPrimaryBytes(gpu),
        });
        registry.reset();
        for (const DeviceId device : {gpu, DeviceId::cpu()})
        {
            EXPECT_EQ(authority->claimedBytes(
                device, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        }
        const auto receipt = transfers.completeExclusiveModelRetirement(std::move(ticket));
        EXPECT_EQ(receipt.expected_retired_bytes, gpu_bytes);
        EXPECT_EQ(receipt.releasedCanonicalBytes(), gpu_bytes);
        EXPECT_TRUE(receipt.runtime_reset_invoked);
    }
} // namespace

/** @test CPU scratch in a CUDA-owned graph cannot be charged as CUDA VRAM. */
TEST(Test__WorkspaceRetention, CUDA) { verifyHeterogeneousRetention(DeviceId::cuda(0)); }

/** @test The identical physical-domain contract holds for ROCm graph owners. */
TEST(Test__WorkspaceRetention, ROCm) { verifyHeterogeneousRetention(DeviceId::rocm(0)); }
