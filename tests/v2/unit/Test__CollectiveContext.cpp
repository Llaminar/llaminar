/**
 * @file Test__CollectiveContext.cpp
 * @brief Unit tests for CollectiveContext
 *
 * Tests runtime context for collective operations using mock backends.
 * Verifies:
 * - Single-device vs multi-device/multi-rank detection
 * - Delegation to router for backend selection
 * - Delegation to backend for collective operations
 * - Factory methods for creating contexts
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/CollectiveContext.h"
#include "collective/test/CollectiveTestMocks.h"
#include "config/TPDomain.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__CollectiveContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock router
        mock_router_ = std::make_unique<MockBackendRouter>();
        mock_router_raw_ = mock_router_.get();

        // Create mock backend
        mock_backend_ = std::make_unique<MockCollectiveBackend>(CollectiveBackendType::MPI);
        mock_backend_raw_ = mock_backend_.get();

        // Configure router to return mock backend
        mock_router_raw_->setDefaultBackend(mock_backend_raw_);
    }

    // Helper to create context with mock router (transfers ownership)
    std::unique_ptr<CollectiveContext> createContextWithMockRouter(
        std::shared_ptr<MPIContext> mpi_ctx = nullptr,
        std::vector<DeviceId> devices = {DeviceId::cpu()})
    {
        return CollectiveContextFactory::createWithRouter(
            std::move(mock_router_),
            std::move(mpi_ctx),
            std::move(devices));
    }

    // Helper to create mock MPI context
    std::shared_ptr<MPIContext> createMockMPIContext(int rank, int world_size)
    {
        // Note: MPIContext requires MPI to be initialized, so for unit tests
        // we use nullptr and rely on world_size parameter in the context
        // In a real scenario, we'd need to either:
        // 1. Actually initialize MPI
        // 2. Create a mock/fake MPIContext
        // For now, return nullptr and test the paths that don't require real MPI
        return nullptr;
    }

    std::unique_ptr<MockBackendRouter> mock_router_;
    MockBackendRouter *mock_router_raw_ = nullptr;

    std::unique_ptr<MockCollectiveBackend> mock_backend_;
    MockCollectiveBackend *mock_backend_raw_ = nullptr;
};

// =============================================================================
// Single Device / Multi Device Tests
// =============================================================================

TEST_F(Test__CollectiveContext, SingleDeviceDoesNotRequireCollectives)
{
    // Single CPU device, no MPI
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    EXPECT_FALSE(ctx->requiresCollectives());
    EXPECT_EQ(ctx->worldSize(), 1);
    EXPECT_EQ(ctx->rank(), 0);
}

TEST_F(Test__CollectiveContext, MultiDeviceRequiresCollectives)
{
    // Multiple local devices require collectives
    auto ctx = createContextWithMockRouter(
        nullptr,
        {DeviceId::cuda(0), DeviceId::cuda(1)});

    EXPECT_TRUE(ctx->requiresCollectives());
}

TEST_F(Test__CollectiveContext, SingleDeviceWithGPU)
{
    // Single GPU device, no MPI - no collectives needed
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0)});

    EXPECT_FALSE(ctx->requiresCollectives());
    EXPECT_EQ(ctx->worldSize(), 1);
}

// =============================================================================
// World Size and Rank Tests
// =============================================================================

TEST_F(Test__CollectiveContext, WorldSizeReturnsCorrectValue)
{
    // Without MPI, world_size should be 1
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    EXPECT_EQ(ctx->worldSize(), 1);
}

TEST_F(Test__CollectiveContext, RankReturnsCorrectValue)
{
    // Without MPI context, rank should be 0
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    EXPECT_EQ(ctx->rank(), 0);
}

// =============================================================================
// Backend Delegation Tests
// =============================================================================

TEST_F(Test__CollectiveContext, ExecuteAllreduceDelegatesToRouter)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Create a buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce
    bool result = ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu());

    // Verify router was called
    EXPECT_GE(mock_router_raw_->getBackendCallCount(), 1);
    EXPECT_TRUE(result);
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceDelegatesToBackend)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Create a buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce
    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu());

    // Verify backend's allreduce was called
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 1);
    EXPECT_EQ(mock_backend_raw_->lastAllreduceCount(), 16);
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceUsesBufferNumelWhenCountIsZero)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Create a 4x4 buffer (16 elements)
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce with count=0 (should use buffer->numel())
    ctx->executeAllreduce(buffer.get(), 0, DeviceId::cpu());

    // Verify count was derived from buffer
    EXPECT_EQ(mock_backend_raw_->lastAllreduceCount(), 16);
}

TEST_F(Test__CollectiveContext, ExecuteAllgatherDelegatesToBackend)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Create input and output buffers
    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 8});
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 16});

    // Execute allgather
    ctx->executeAllgather(local_input.get(), full_output.get(), 2, DeviceId::cpu());

    // Verify backend's allgather was called
    EXPECT_EQ(mock_backend_raw_->allgatherCallCount(), 1);
}

TEST_F(Test__CollectiveContext, ExecuteBroadcastDelegatesToBackend)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Create a buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute broadcast from rank 0
    ctx->executeBroadcast(buffer.get(), 16, 0, DeviceId::cpu());

    // Verify backend's broadcast was called
    EXPECT_EQ(mock_backend_raw_->broadcastCallCount(), 1);
}

// =============================================================================
// Backend Availability Tests
// =============================================================================

TEST_F(Test__CollectiveContext, IsBackendAvailableDelegatesToRouter)
{
    // Configure availability
    mock_router_raw_->setAvailable(CollectiveBackendType::MPI, true);
    mock_router_raw_->setAvailable(CollectiveBackendType::NCCL, false);

    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::MPI));
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::NCCL));
}

TEST_F(Test__CollectiveContext, IsBackendAvailableReturnsFalseWithoutRouter)
{
    // Create context without router (via factory that doesn't set one)
    CollectiveContext::Config config;
    auto ctx = std::make_unique<CollectiveContext>(config);

    // Without a router, all backends should report unavailable
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::MPI));
    EXPECT_FALSE(ctx->isBackendAvailable(CollectiveBackendType::NCCL));
}

// =============================================================================
// Local Devices Tests
// =============================================================================

TEST_F(Test__CollectiveContext, LocalDevicesReturnsConfiguredDevices)
{
    std::vector<DeviceId> expected_devices = {
        DeviceId::cuda(0),
        DeviceId::cuda(1),
        DeviceId::cpu()};

    auto ctx = createContextWithMockRouter(nullptr, expected_devices);

    const auto &devices = ctx->localDevices();
    ASSERT_EQ(devices.size(), 3);
    EXPECT_TRUE(devices[0].is_cuda());
    EXPECT_EQ(devices[0].ordinal, 0);
    EXPECT_TRUE(devices[1].is_cuda());
    EXPECT_EQ(devices[1].ordinal, 1);
    EXPECT_TRUE(devices[2].is_cpu());
}

// =============================================================================
// Device Group Building Tests
// =============================================================================

TEST_F(Test__CollectiveContext, BuildDeviceGroupSetsCorrectScope_Local)
{
    // Single rank (world_size=1) should create LOCAL scope groups
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});

    // Execute operation to trigger group building
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});
    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0));

    // Verify group was built with LOCAL scope
    const auto &group = mock_router_raw_->lastGroup();
    EXPECT_EQ(group.scope, CollectiveScope::LOCAL);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(Test__CollectiveContext, ExecuteAllreduceFailsWithoutRouter)
{
    // Create context without router
    CollectiveContext::Config config;
    auto ctx = std::make_unique<CollectiveContext>(config);

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should fail gracefully
    EXPECT_FALSE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu()));
}

TEST_F(Test__CollectiveContext, ExecuteAllgatherFailsWithoutRouter)
{
    // Create context without router
    CollectiveContext::Config config;
    auto ctx = std::make_unique<CollectiveContext>(config);

    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 8});
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 16});

    // Should fail gracefully
    EXPECT_FALSE(ctx->executeAllgather(local_input.get(), full_output.get(), 2, DeviceId::cpu()));
}

TEST_F(Test__CollectiveContext, ExecuteBroadcastFailsWithoutRouter)
{
    // Create context without router
    CollectiveContext::Config config;
    auto ctx = std::make_unique<CollectiveContext>(config);

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should fail gracefully
    EXPECT_FALSE(ctx->executeBroadcast(buffer.get(), 16, 0, DeviceId::cpu()));
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceFailsWhenBackendReturnsNull)
{
    // Configure router to return null backend
    mock_router_raw_->setDefaultBackend(nullptr);

    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should fail gracefully when backend is null
    EXPECT_FALSE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu()));
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceFailsWhenBackendFails)
{
    // Configure backend to fail
    mock_backend_raw_->setAllreduceResult(false);

    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should propagate backend failure
    EXPECT_FALSE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu()));
}

// =============================================================================
// Factory Method Tests
// =============================================================================

TEST_F(Test__CollectiveContext, FactoryCreateSingleDevice)
{
    auto ctx = CollectiveContextFactory::createSingleDevice();

    EXPECT_FALSE(ctx->requiresCollectives());
    EXPECT_EQ(ctx->worldSize(), 1);
    EXPECT_EQ(ctx->rank(), 0);
}

TEST_F(Test__CollectiveContext, FactoryCreateWithConfig)
{
    CollectiveContext::Config config;
    config.verbose = true;

    auto ctx = CollectiveContextFactory::create(config);

    EXPECT_EQ(ctx->worldSize(), 1);
    EXPECT_EQ(ctx->rank(), 0);
}

TEST_F(Test__CollectiveContext, FactoryCreateWithRouter)
{
    // Create and configure mock
    auto router = std::make_unique<MockBackendRouter>();
    router->setAvailable(CollectiveBackendType::HOST, true);

    auto ctx = CollectiveContextFactory::createWithRouter(
        std::move(router),
        nullptr,
        {DeviceId::cuda(0), DeviceId::cuda(1)});

    EXPECT_TRUE(ctx->requiresCollectives());
    EXPECT_TRUE(ctx->isBackendAvailable(CollectiveBackendType::HOST));
}

// =============================================================================
// Router Access Tests
// =============================================================================

TEST_F(Test__CollectiveContext, RouterAccessReturnsInjectedRouter)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Should be able to access router
    EXPECT_NE(ctx->router(), nullptr);
}

TEST_F(Test__CollectiveContext, RouterAccessReturnsNullWithoutRouter)
{
    CollectiveContext::Config config;
    auto ctx = std::make_unique<CollectiveContext>(config);

    EXPECT_EQ(ctx->router(), nullptr);
}

// =============================================================================
// MPI Context Access Tests
// =============================================================================

TEST_F(Test__CollectiveContext, MPIContextAccessReturnsNull)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Without MPI, context should be null
    EXPECT_EQ(ctx->mpiContext(), nullptr);
}

// =============================================================================
// Collective Operation Type Tests
// =============================================================================

TEST_F(Test__CollectiveContext, ExecuteAllreduceUsesCorrectOp)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Test with different operations
    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_SUM);
    EXPECT_EQ(mock_backend_raw_->lastAllreduceOp(), CollectiveOp::ALLREDUCE_SUM);

    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_MAX);
    EXPECT_EQ(mock_backend_raw_->lastAllreduceOp(), CollectiveOp::ALLREDUCE_MAX);
}

// =============================================================================
// Domain-Aware Collective Operation Tests
// =============================================================================

TEST_F(Test__CollectiveContext, ExecuteAllreduceInDomainWithNullFallsBack)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute with nullptr domain - should fall back to regular allreduce
    bool result = ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_SUM, nullptr);

    EXPECT_TRUE(result);
    // Regular allreduce should have been called via the router
    EXPECT_GE(mock_router_raw_->getBackendCallCount(), 1);
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 1);
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceInDomainUsesCorrectBackend)
{
    // Configure router to support domain routing
    mock_router_raw_->setHasDomainSupport(true);
    mock_router_raw_->setDomainBackend(mock_backend_raw_);

    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::rocm(0)});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create a GPU domain for testing
    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "test_gpu_domain";
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
    gpu_domain.communicator = MPI_COMM_NULL;

    // Execute with domain
    bool result = ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cuda(0), CollectiveOp::ALLREDUCE_SUM, &gpu_domain);

    EXPECT_TRUE(result);
    // Domain-aware backend selection should have been used
    EXPECT_GE(mock_router_raw_->selectDomainCallCount(), 1);
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 1);
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceInDomainSkipsTrivialDomain)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create a trivial domain (size 1)
    TPDomain trivial_domain;
    trivial_domain.type = TPDomainType::GPU_INTRA_RANK;
    trivial_domain.name = "trivial_domain";
    trivial_domain.domain_size = 1;
    trivial_domain.local_rank_in_domain = 0;
    trivial_domain.devices = {DeviceId::cpu()};
    trivial_domain.communicator = MPI_COMM_NULL;

    // Execute with trivial domain - should return true without any backend call
    bool result = ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_SUM, &trivial_domain);

    EXPECT_TRUE(result);
    // No backend should have been called for trivial domain
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 0);
}

TEST_F(Test__CollectiveContext, ExecuteAllgatherInDomainWithNullFallsBack)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 8});
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 16});

    // Execute with nullptr domain - should fall back to regular allgather
    bool result = ctx->executeAllgatherInDomain(
        local_input.get(), full_output.get(), 2, DeviceId::cpu(), nullptr);

    EXPECT_TRUE(result);
    EXPECT_EQ(mock_backend_raw_->allgatherCallCount(), 1);
}

TEST_F(Test__CollectiveContext, ExecuteAllgathervInDomainWithNullFallsBack)
{
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto local_input = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 8});
    auto full_output = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 16});
    std::vector<int> recv_counts = {8, 8};
    std::vector<int> displacements = {0, 8};

    // Execute with nullptr domain - should fall back to regular allgatherv
    bool result = ctx->executeAllgathervInDomain(
        local_input.get(), full_output.get(), recv_counts, displacements,
        2, DeviceId::cpu(), nullptr);

    EXPECT_TRUE(result);
    EXPECT_EQ(mock_backend_raw_->allgathervCallCount(), 1);
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceInDomainFailsWithoutRouter)
{
    // Create context without router
    CollectiveContext::Config config;
    auto ctx = std::make_unique<CollectiveContext>(config);

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create a non-trivial domain
    TPDomain domain;
    domain.type = TPDomainType::GPU_INTRA_RANK;
    domain.name = "test_domain";
    domain.domain_size = 2;
    domain.local_rank_in_domain = 0;
    domain.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    domain.communicator = MPI_COMM_NULL;

    // Should fail gracefully because fallback path also has no router
    EXPECT_FALSE(ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_SUM, &domain));
}

TEST_F(Test__CollectiveContext, ExecuteAllreduceInDomainFailsWhenBackendReturnsNull)
{
    // Configure router to return null for domain
    mock_router_raw_->setHasDomainSupport(true);
    mock_router_raw_->setDomainBackend(nullptr);

    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Create a non-trivial domain
    TPDomain domain;
    domain.type = TPDomainType::GPU_INTRA_RANK;
    domain.name = "test_domain";
    domain.domain_size = 2;
    domain.local_rank_in_domain = 0;
    domain.devices = {DeviceId::cuda(0), DeviceId::cuda(1)};
    domain.communicator = MPI_COMM_NULL;

    // Should fail when backend is null
    EXPECT_FALSE(ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_SUM, &domain));
}
