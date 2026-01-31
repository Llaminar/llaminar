/**
 * @file Test__CollectiveBufferAllocation.cpp
 * @brief Unit tests for collective buffer allocation via GraphBufferManager
 *
 * Tests the Phase 3 Buffer Registration API integration where buffers
 * marked as participating in collective operations can be allocated
 * from the BAR region when a PCIeBARBackend is active.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/execution/local_execution/graph/GraphBufferManager.h"
#include "v2/execution/local_execution/collective/CollectiveContext.h"
#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/BackendRouter.h"
#include "v2/collective/ICollectiveBackend.h"
#include "v2/collective/IBufferRegistration.h" // For RegisteredBuffer
#include "v2/tensors/TensorFactory.h"
#include "v2/utils/MPIContext.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Mock Backend Router for Testing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * @brief Mock backend router that returns a configurable backend
     */
    class MockBackendRouter : public IBackendRouter
    {
    public:
        explicit MockBackendRouter(ICollectiveBackend *backend = nullptr)
            : mock_backend_(backend) {}

        ICollectiveBackend *getBackend(const DeviceGroup & /*group*/) override
        {
            return mock_backend_;
        }

        ICollectiveBackend *getBackend(CollectiveBackendType type) override
        {
            if (!mock_backend_)
                return nullptr;
            return mock_backend_->type() == type ? mock_backend_ : nullptr;
        }

        BackendSelection selectBackend(const DeviceGroup & /*group*/) const override
        {
            BackendSelection sel;
            sel.type = mock_backend_ ? mock_backend_->type() : CollectiveBackendType::HOST;
            sel.reason = "mock selection";
            return sel;
        }

        bool isAvailable(CollectiveBackendType type) const override
        {
            if (!mock_backend_)
                return false;
            return mock_backend_->type() == type;
        }

        bool executeHeterogeneousAllReduce(
            const DeviceGroup & /*group*/,
            void * /*buffer*/,
            size_t /*count*/,
            CollectiveDataType /*dtype*/,
            CollectiveOp /*op*/) override
        {
            return true;
        }

        bool executeHeterogeneousAllGather(
            const DeviceGroup & /*group*/,
            const void * /*send_buf*/,
            void * /*recv_buf*/,
            size_t /*send_count*/,
            CollectiveDataType /*dtype*/) override
        {
            return true;
        }

        void setMockBackend(ICollectiveBackend *backend)
        {
            mock_backend_ = backend;
        }

    private:
        ICollectiveBackend *mock_backend_;
    };

    /**
     * @brief Mock backend that mimics PCIeBARBackend for registration testing
     */
    class MockPCIeBarLikeBackend : public ICollectiveBackend
    {
    public:
        // Identity
        CollectiveBackendType type() const override { return CollectiveBackendType::HOST; }
        std::string name() const override { return "MockPCIeBarLike"; }

        // Capability queries
        bool supportsDevice(DeviceType /*type*/) const override { return true; }
        bool supportsDirectTransfer(DeviceId /*src*/, DeviceId /*dst*/) const override { return true; }
        bool isAvailable() const override { return true; }

        // Lifecycle
        bool initialize(const DeviceGroup & /*group*/) override { return true; }
        bool isInitialized() const override { return true; }
        void shutdown() override {}

        // Collective operations
        bool allreduce(void * /*buffer*/, size_t /*count*/,
                       CollectiveDataType /*dtype*/, CollectiveOp /*op*/) override
        {
            return true;
        }

        bool allgather(const void * /*send_buf*/, void * /*recv_buf*/,
                       size_t /*send_count*/, CollectiveDataType /*dtype*/) override
        {
            return true;
        }

        bool reduceScatter(const void * /*send_buf*/, void * /*recv_buf*/,
                           size_t /*recv_count*/, CollectiveDataType /*dtype*/,
                           CollectiveOp /*op*/) override
        {
            return true;
        }

        bool broadcast(void * /*buffer*/, size_t /*count*/,
                       CollectiveDataType /*dtype*/, int /*root_rank*/) override
        {
            return true;
        }

        bool synchronize() override { return true; }

        // Override to indicate registration is required
        bool requiresBufferRegistration() const override { return true; }

        // Track registrations for testing
        bool registerBuffer(const std::string &collective_id,
                            DeviceId device, void *buffer, size_t size) override
        {
            registrations_[collective_id] = {device, buffer, size};
            return true;
        }

        void unregisterBuffer(const std::string &collective_id, DeviceId /*device*/) override
        {
            registrations_.erase(collective_id);
        }

        std::optional<RegisteredBuffer> getBuffer(const std::string &collective_id,
                                                  DeviceId device) const override
        {
            auto it = registrations_.find(collective_id);
            if (it != registrations_.end() && it->second.device == device)
            {
                return RegisteredBuffer(it->second.device, it->second.buffer, it->second.size);
            }
            return std::nullopt;
        }

        // Test helper
        bool hasRegistration(const std::string &collective_id) const
        {
            return registrations_.find(collective_id) != registrations_.end();
        }

    private:
        struct Registration
        {
            DeviceId device;
            void *buffer;
            size_t size;
        };
        std::unordered_map<std::string, Registration> registrations_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class CollectiveBufferAllocationTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            mpi_ctx_ = std::make_unique<MPIContext>(0, 1, MPI_COMM_WORLD);
            factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
            manager_ = std::make_unique<GraphBufferManager>(factory_.get(), mpi_ctx_.get());
        }

        void TearDown() override
        {
            manager_.reset();
            factory_.reset();
            mpi_ctx_.reset();
        }

        std::unique_ptr<MPIContext> mpi_ctx_;
        std::unique_ptr<TensorFactory> factory_;
        std::unique_ptr<GraphBufferManager> manager_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // BufferDescriptor Collective API Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST(Test__BufferDescriptor, ForCollective_SetsFields)
    {
        BufferDescriptor desc = BufferDescriptor::output("test", {16, 32})
                                    .forCollective("my_collective");

        EXPECT_TRUE(desc.participates_in_collective);
        EXPECT_EQ(desc.collective_id, "my_collective");
        EXPECT_TRUE(desc.isCollectiveBuffer());
    }

    TEST(Test__BufferDescriptor, ForCollective_Chainable)
    {
        BufferDescriptor desc = BufferDescriptor::output("test", {16, 32})
                                    .forCollective("my_collective")
                                    .withLayout(TensorLayout::ROW_MAJOR_2D);

        EXPECT_TRUE(desc.participates_in_collective);
        EXPECT_EQ(desc.collective_id, "my_collective");
        EXPECT_EQ(desc.expected_layout, TensorLayout::ROW_MAJOR_2D);
    }

    TEST(Test__BufferDescriptor, IsCollectiveBuffer_RequiresBothFields)
    {
        BufferDescriptor desc1;
        desc1.participates_in_collective = true;
        desc1.collective_id = "";
        EXPECT_FALSE(desc1.isCollectiveBuffer()); // Empty collective_id

        BufferDescriptor desc2;
        desc2.participates_in_collective = false;
        desc2.collective_id = "some_id";
        EXPECT_FALSE(desc2.isCollectiveBuffer()); // participates_in_collective is false

        BufferDescriptor desc3;
        desc3.participates_in_collective = true;
        desc3.collective_id = "valid_id";
        EXPECT_TRUE(desc3.isCollectiveBuffer()); // Both set
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CollectiveContext Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(CollectiveBufferAllocationTest, CollectiveContext_RequiresBufferRegistration_Default)
    {
        // Default single-device context should not require registration
        auto ctx = CollectiveContextFactory::createSingleDevice();
        EXPECT_FALSE(ctx->requiresBufferRegistration());
    }

    TEST_F(CollectiveBufferAllocationTest, CollectiveContext_GetPCIeBarBackend_NoRouter)
    {
        // Single-device context without router should return nullptr
        auto ctx = CollectiveContextFactory::createSingleDevice();
        EXPECT_EQ(ctx->getPCIeBarBackend(), nullptr);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GraphBufferManager Collective Integration Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(CollectiveBufferAllocationTest, SetCollectiveContext_Works)
    {
        EXPECT_EQ(manager_->collectiveContext(), nullptr);

        auto ctx = CollectiveContextFactory::createSingleDevice();
        auto raw_ptr = ctx.get();
        manager_->setCollectiveContext(std::move(ctx));

        EXPECT_EQ(manager_->collectiveContext().get(), raw_ptr);
    }

    TEST_F(CollectiveBufferAllocationTest, AllocateCollectiveBuffer_WithoutContext_Succeeds)
    {
        // Collective buffer without context should allocate normally
        BufferDescriptor desc = BufferDescriptor::output("collective_out", {64, 128})
                                    .forCollective("test_allreduce");

        EXPECT_TRUE(manager_->allocateBuffer("node", desc));
        EXPECT_TRUE(manager_->hasBuffer("node", "collective_out"));

        auto *buffer = manager_->getBuffer("node", "collective_out");
        ASSERT_NE(buffer, nullptr);
        EXPECT_EQ(buffer->numel(), 64u * 128u);
    }

    TEST_F(CollectiveBufferAllocationTest, AllocateCollectiveBuffer_CPUDevice_NoBar)
    {
        // CPU device should never use BAR allocation
        auto ctx = CollectiveContextFactory::createSingleDevice();
        manager_->setCollectiveContext(std::move(ctx));

        BufferDescriptor desc = BufferDescriptor::output("collective_out", {64, 128})
                                    .forCollective("test_allreduce");
        desc.device = DeviceId::cpu();

        EXPECT_TRUE(manager_->allocateBuffer("node", desc));
        EXPECT_TRUE(manager_->hasBuffer("node", "collective_out"));
    }

    TEST_F(CollectiveBufferAllocationTest, AllocateRegularBuffer_WithContext_NoBar)
    {
        // Regular (non-collective) buffer should not use BAR even with context
        auto ctx = CollectiveContextFactory::createSingleDevice();
        manager_->setCollectiveContext(std::move(ctx));

        BufferDescriptor desc = BufferDescriptor::output("regular_out", {64, 128});
        // NOT marked as collective

        EXPECT_TRUE(manager_->allocateBuffer("node", desc));
        EXPECT_TRUE(manager_->hasBuffer("node", "regular_out"));
    }

    TEST_F(CollectiveBufferAllocationTest, AllocateMixedBuffers_Graph)
    {
        // Graph with mix of regular and collective buffers
        auto ctx = CollectiveContextFactory::createSingleDevice();
        manager_->setCollectiveContext(std::move(ctx));

        // Simulate allocation via direct calls (graph allocation would do same)
        BufferDescriptor regular = BufferDescriptor::output("regular", {32, 64});
        BufferDescriptor collective = BufferDescriptor::output("collective", {32, 64})
                                          .forCollective("layer0_allreduce");
        BufferDescriptor scratch = BufferDescriptor::scratch("workspace", {256});

        EXPECT_TRUE(manager_->allocateBuffer("stage1", regular));
        EXPECT_TRUE(manager_->allocateBuffer("stage1", collective));
        EXPECT_TRUE(manager_->allocateBuffer("stage1", scratch));

        EXPECT_EQ(manager_->bufferCount(), 3u);
        EXPECT_TRUE(manager_->hasBuffer("stage1", "regular"));
        EXPECT_TRUE(manager_->hasBuffer("stage1", "collective"));
        EXPECT_TRUE(manager_->hasBuffer("stage1", "workspace"));
    }

    TEST_F(CollectiveBufferAllocationTest, AllocateCollectiveBuffer_CUDADevice_NoBar)
    {
        // CUDA device should not use BAR allocation (BAR is for ROCm)
        auto ctx = CollectiveContextFactory::createSingleDevice();
        manager_->setCollectiveContext(std::move(ctx));

        BufferDescriptor desc = BufferDescriptor::output("collective_out", {64, 128})
                                    .forCollective("test_allreduce");
        desc.device = DeviceId::cuda(0);

        EXPECT_TRUE(manager_->allocateBuffer("node", desc));
        EXPECT_TRUE(manager_->hasBuffer("node", "collective_out"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistics Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(CollectiveBufferAllocationTest, Stats_TrackCollectiveBuffers)
    {
        BufferDescriptor collective = BufferDescriptor::output("collective", {100})
                                          .forCollective("test_allreduce");
        BufferDescriptor regular = BufferDescriptor::output("regular", {50});

        EXPECT_TRUE(manager_->allocateBuffer("node1", collective));
        EXPECT_TRUE(manager_->allocateBuffer("node2", regular));

        const auto &stats = manager_->stats();
        EXPECT_EQ(stats.total_buffers, 2u);
        // Both should be tracked as OUTPUT bytes
        EXPECT_EQ(stats.output_bytes, (100u + 50u) * sizeof(float));
    }

} // namespace llaminar2::test
