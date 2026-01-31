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
#include <gmock/gmock.h>
#include "execution/local_execution/collective/CollectiveContext.h"
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

TEST_F(Test__CollectiveContext, SingleDeviceAllreduceIsNoOp)
{
    // Single-device context - allreduce should be a no-op (nothing to reduce)
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});

    // Create a buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce
    bool result = ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu());

    // Single-device allreduce should succeed (no-op) without calling backend
    EXPECT_TRUE(result);
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 0);
}

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceDelegatesToRouter)
{
    // Multi-device context requires actual collective
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});

    // Create a buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce
    bool result = ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0));

    // Verify router was called
    EXPECT_GE(mock_router_raw_->getBackendCallCount(), 1);
    EXPECT_TRUE(result);
}

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceDelegatesToBackend)
{
    // Multi-device context requires actual collective
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});

    // Create a buffer
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce
    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0));

    // Verify backend's allreduce was called
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 1);
    EXPECT_EQ(mock_backend_raw_->lastAllreduceCount(), 16);
}

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceUsesBufferNumelWhenCountIsZero)
{
    // Multi-device context to test actual allreduce behavior
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});

    // Create a 4x4 buffer (16 elements)
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute allreduce with count=0 (should use buffer->numel())
    ctx->executeAllreduce(buffer.get(), 0, DeviceId::cuda(0));

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

TEST_F(Test__CollectiveContext, SingleDeviceAllreduceSucceedsWithoutRouter)
{
    // Single-device context without router - allreduce is a no-op
    // Use the factory method for single-device creation
    auto ctx = CollectiveContextFactory::createSingleDevice();

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Single-device allreduce should succeed (no-op) even without router
    EXPECT_TRUE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cpu()));
}

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceFailsWithoutRouter)
{
    // Multi-device context without router - need to create via createWithRouter with null router
    // This is a tricky case - createWithRouter requires a router
    // Let's test via the factory with a null backend set
    auto router = std::make_unique<MockBackendRouter>();
    router->setDefaultBackend(nullptr); // No backend available

    auto ctx = CollectiveContextFactory::createWithRouter(
        std::move(router),
        nullptr,
        {DeviceId::cuda(0), DeviceId::cuda(1)});

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should fail gracefully - backend required for multi-device
    EXPECT_FALSE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0)));
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

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceFailsWhenBackendReturnsNull)
{
    // Configure router to return null backend
    mock_router_raw_->setDefaultBackend(nullptr);

    // Multi-device context to trigger actual collective
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should fail gracefully when backend is null
    EXPECT_FALSE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0)));
}

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceFailsWhenBackendFails)
{
    // Configure backend to fail
    mock_backend_raw_->setAllreduceResult(false);

    // Multi-device context to trigger actual collective
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});

    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Should propagate backend failure
    EXPECT_FALSE(ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0)));
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

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceUsesCorrectOp)
{
    // Multi-device context to test operation type forwarding
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Test with different operations
    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0), CollectiveOp::ALLREDUCE_SUM);
    EXPECT_EQ(mock_backend_raw_->lastAllreduceOp(), CollectiveOp::ALLREDUCE_SUM);

    ctx->executeAllreduce(buffer.get(), 16, DeviceId::cuda(0), CollectiveOp::ALLREDUCE_MAX);
    EXPECT_EQ(mock_backend_raw_->lastAllreduceOp(), CollectiveOp::ALLREDUCE_MAX);
}

// =============================================================================
// Domain-Aware Collective Operation Tests
// =============================================================================

TEST_F(Test__CollectiveContext, SingleDeviceAllreduceInDomainWithNullIsNoOp)
{
    // Single-device context - even domain allreduce is a no-op
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cpu()});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute with nullptr domain - single-device should be no-op
    bool result = ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cpu(), CollectiveOp::ALLREDUCE_SUM, nullptr);

    EXPECT_TRUE(result);
    // Single-device: no backend call expected
    EXPECT_EQ(mock_backend_raw_->allreduceCallCount(), 0);
}

TEST_F(Test__CollectiveContext, MultiDeviceAllreduceInDomainWithNullFallsBack)
{
    // Multi-device context to test fallback to regular allreduce
    auto ctx = createContextWithMockRouter(nullptr, {DeviceId::cuda(0), DeviceId::cuda(1)});
    auto buffer = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});

    // Execute with nullptr domain - should fall back to regular allreduce
    bool result = ctx->executeAllreduceInDomain(
        buffer.get(), 16, DeviceId::cuda(0), CollectiveOp::ALLREDUCE_SUM, nullptr);

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

// =============================================================================
// tensorToCollectiveDataType Tests
// =============================================================================

TEST(Test__TensorToCollectiveDataType, ThrowsOnNullTensor)
{
    EXPECT_THROW(tensorToCollectiveDataType(nullptr), std::invalid_argument);

    // Verify the exception message
    try
    {
        tensorToCollectiveDataType(nullptr);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("nullptr"));
    }
}

TEST(Test__TensorToCollectiveDataType, ReturnsFLOAT32ForFP32Tensor)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{4, 4});
    EXPECT_EQ(tensorToCollectiveDataType(tensor.get()), CollectiveDataType::FLOAT32);
}

TEST(Test__TensorToCollectiveDataType, ReturnsFLOAT16ForFP16Tensor)
{
    auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{4, 4});
    EXPECT_EQ(tensorToCollectiveDataType(tensor.get()), CollectiveDataType::FLOAT16);
}

TEST(Test__TensorToCollectiveDataType, ReturnsBFLOAT16ForBF16Tensor)
{
    auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{4, 4});
    EXPECT_EQ(tensorToCollectiveDataType(tensor.get()), CollectiveDataType::BFLOAT16);
}

TEST(Test__TensorToCollectiveDataType, ReturnsINT32ForINT32Tensor)
{
    auto tensor = std::make_unique<INT32Tensor>(std::vector<size_t>{4, 4});
    EXPECT_EQ(tensorToCollectiveDataType(tensor.get()), CollectiveDataType::INT32);
}

TEST(Test__TensorToCollectiveDataType, ReturnsINT8ForINT8Tensor)
{
    auto tensor = std::make_unique<INT8Tensor>(std::vector<size_t>{4, 4});
    EXPECT_EQ(tensorToCollectiveDataType(tensor.get()), CollectiveDataType::INT8);
}

TEST(Test__TensorToCollectiveDataType, ThrowsForQuantizedQ8_0Tensor)
{
    // Q8_0 block size is 32 elements, 34 bytes per block (32 int8 + 2 bytes for scale)
    // 4 rows x 32 cols = 4 blocks = 4 * 34 = 136 bytes
    std::vector<uint8_t> raw_data(4 * 34, 0);
    auto tensor = std::make_unique<Q8_0Tensor>(std::vector<size_t>{4, 32}, raw_data);
    EXPECT_THROW(tensorToCollectiveDataType(tensor.get()), std::invalid_argument);

    // Verify the exception message includes the tensor type
    try
    {
        tensorToCollectiveDataType(tensor.get());
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Q8_0"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("unsupported"));
    }
}

TEST(Test__TensorToCollectiveDataType, ThrowsForQuantizedQ4_0Tensor)
{
    // Q4_0 block size is 32 elements, 18 bytes per block (16 nibbles + 2 bytes for scale)
    // 4 rows x 32 cols = 4 blocks = 4 * 18 = 72 bytes
    std::vector<uint8_t> raw_data(4 * 18, 0);
    auto tensor = std::make_unique<Q4_0Tensor>(std::vector<size_t>{4, 32}, raw_data);
    EXPECT_THROW(tensorToCollectiveDataType(tensor.get()), std::invalid_argument);

    // Verify the exception message includes the tensor type
    try
    {
        tensorToCollectiveDataType(tensor.get());
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Q4_0"));
    }
}

TEST(Test__TensorToCollectiveDataType, ThrowsForQuantizedIQ4_NLTensor)
{
    // IQ4_NL block size is 32 elements, 18 bytes per block
    // 4 rows x 32 cols = 4 blocks = 4 * 18 = 72 bytes
    std::vector<uint8_t> raw_data(4 * 18, 0);
    auto tensor = std::make_unique<IQ4_NLTensor>(std::vector<size_t>{4, 32}, raw_data);
    EXPECT_THROW(tensorToCollectiveDataType(tensor.get()), std::invalid_argument);

    // Verify the exception message includes the tensor type
    try
    {
        tensorToCollectiveDataType(tensor.get());
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_THAT(e.what(), ::testing::HasSubstr("IQ4_NL"));
    }
}
