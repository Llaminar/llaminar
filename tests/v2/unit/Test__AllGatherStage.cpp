/**
 * @file Test__AllGatherStage.cpp
 * @brief Unit tests for AllGatherStage compute stage
 * @author David Sanftenberg
 * @date January 2025
 *
 * Tests for MPI AllGather stage functionality:
 * - Basic execution with mock MPI communicator
 * - Stage type and properties
 * - Buffer requirements
 *
 * Note: Full MPI testing is done in integration tests.
 * This tests the stage interface and validation logic.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "backends/DeviceId.h"
#include "config/TPDomain.h"
#include "../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__AllGatherStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU);

        // Create a simple mock MPI context for testing
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
    }

    std::unique_ptr<MockDeviceContext> ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// =============================================================================
// AllGatherStage Unit Tests
// =============================================================================

/**
 * @test Verify AllGatherStage type returns ALLGATHER
 */
TEST_F(Test__AllGatherStage, StageTypeIsAllGather)
{
    // Create minimal params
    AllGatherStage::Params params;
    params.local_input = nullptr;
    params.full_output = nullptr;
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllGatherStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::ALLGATHER);
    // AllGather is an MPI collective operation, so it requires collective sync
    EXPECT_TRUE(stage->requiresAllreduce());
}

/**
 * @test Verify AllGatherStage supports all backends
 */
TEST_F(Test__AllGatherStage, SupportsAllBackends)
{
    AllGatherStage::Params params;
    params.local_input = nullptr;
    params.full_output = nullptr;
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllGatherStage>(params);

    // AllGather should support all backends (it's MPI-based, not compute-bound)
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_VULKAN));
}

/**
 * @test Verify AllGatherStage fails with null input buffer
 */
TEST_F(Test__AllGatherStage, FailsWithNullLocalInput)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto full_output = factory.createFP32({4, 2}, DeviceId::cpu()); // [seq_len, vocab_size]

    AllGatherStage::Params params;
    params.local_input = nullptr; // NULL
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllGatherStage>(params);

    // Should fail due to null input
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherStage fails with null output buffer
 */
TEST_F(Test__AllGatherStage, FailsWithNullFullOutput)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 1}, DeviceId::cpu()); // [seq_len, vocab_local]

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = nullptr; // NULL
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllGatherStage>(params);

    // Should fail due to null output
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherStage fails with null MPI communicator
 */
TEST_F(Test__AllGatherStage, FailsWithNullMpiComm)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 1}, DeviceId::cpu()); // [seq_len, vocab_local]
    auto full_output = factory.createFP32({4, 2}, DeviceId::cpu()); // [seq_len, vocab_size]

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = nullptr; // NULL

    auto stage = std::make_unique<AllGatherStage>(params);

    // Should fail due to null MPI context
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherStage fails with null MPI context (world_size now from mpi_ctx)
 */
TEST_F(Test__AllGatherStage, FailsWithNullMpiContext)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 1}, DeviceId::cpu()); // [seq_len, vocab_local]
    auto full_output = factory.createFP32({4, 2}, DeviceId::cpu()); // [seq_len, vocab_size]

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = nullptr; // Invalid - null context

    auto stage = std::make_unique<AllGatherStage>(params);

    // Should fail due to null mpi_ctx
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherStage buffer requirements are correct
 */
TEST_F(Test__AllGatherStage, BufferRequirementsAreCorrect)
{
    // Create test buffers with known shapes
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 128}, DeviceId::cpu()); // [seq_len, vocab_local]
    auto full_output = factory.createFP32({4, 256}, DeviceId::cpu()); // [seq_len, vocab_size]

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllGatherStage>(params);

    auto reqs = stage->getBufferRequirements();

    // Should have input and output buffers
    EXPECT_EQ(reqs.buffers.size(), 2);
}

/**
 * @test Verify factory creates AllGatherStage correctly
 */
TEST_F(Test__AllGatherStage, FactoryCreatesStage)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 128}, DeviceId::cpu()); // [seq_len, vocab_local]
    auto full_output = factory.createFP32({4, 256}, DeviceId::cpu()); // [seq_len, vocab_size]

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = ComputeStageFactory::createAllGather(params);

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::ALLGATHER);
}

// =============================================================================
// TPDomain Support Tests (Phase 4.1a)
// =============================================================================

/**
 * @test Verify getDomain() returns nullptr by default
 */
TEST_F(Test__AllGatherStage, GetDomainReturnsNullByDefault)
{
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 128}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 256}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    // domain not set - should default to nullptr

    auto stage = std::make_unique<AllGatherStage>(params);

    EXPECT_EQ(stage->getDomain(), nullptr);
}

/**
 * @test Verify getDomain() returns the configured domain
 */
TEST_F(Test__AllGatherStage, GetDomainReturnsConfiguredDomain)
{
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 128}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 256}, DeviceId::cpu());

    // Create a test domain
    TPDomain test_domain;
    test_domain.type = TPDomainType::GPU_INTRA_RANK;
    test_domain.domain_size = 2;
    test_domain.local_rank_in_domain = 0;
    test_domain.name = "test_gpu_domain";
    test_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.domain = &test_domain;

    auto stage = std::make_unique<AllGatherStage>(params);

    ASSERT_NE(stage->getDomain(), nullptr);
    EXPECT_EQ(stage->getDomain(), &test_domain);
    EXPECT_EQ(stage->getDomain()->name, "test_gpu_domain");
    EXPECT_EQ(stage->getDomain()->domain_size, 2);
}

/**
 * @test Verify Params stores domain field correctly
 */
TEST_F(Test__AllGatherStage, ParamsStoreDomain)
{
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.domain_size = 4;
    cpu_domain.name = "cpu_cross_rank_domain";

    AllGatherStage::Params params;
    params.domain = &cpu_domain;

    // Verify the field is stored
    EXPECT_EQ(params.domain, &cpu_domain);
    EXPECT_EQ(params.domain->type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_TRUE(params.domain->isCrossRank());
}
