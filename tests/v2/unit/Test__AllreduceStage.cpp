/**
 * @file Test__AllreduceStage.cpp
 * @brief Unit tests for AllreduceStage compute stage
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests for MPI Allreduce stage functionality:
 * - Stage type and properties
 * - Buffer requirements
 * - TPDomain support (Phase 4.1a)
 *
 * Note: Full MPI testing is done in integration tests.
 * This tests the stage interface and validation logic.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/DeviceContext.h"
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

class Test__AllreduceStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU);
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
    }

    std::unique_ptr<MockDeviceContext> ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// =============================================================================
// AllreduceStage Basic Tests
// =============================================================================

/**
 * @test Verify AllreduceStage type returns ALLREDUCE
 */
TEST_F(Test__AllreduceStage, StageTypeIsAllreduce)
{
    AllreduceStage::Params params;
    params.buffer = nullptr;
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllreduceStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::ALLREDUCE);
    EXPECT_TRUE(stage->requiresAllreduce());
}

/**
 * @test Verify AllreduceStage supports all backends
 */
TEST_F(Test__AllreduceStage, SupportsAllBackends)
{
    AllreduceStage::Params params;
    params.buffer = nullptr;
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllreduceStage>(params);

    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_VULKAN));
}

/**
 * @test Verify AllreduceStage fails with null buffer
 */
TEST_F(Test__AllreduceStage, FailsWithNullBuffer)
{
    AllreduceStage::Params params;
    params.buffer = nullptr;
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllreduceStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllreduceStage fails with null MPI context
 */
TEST_F(Test__AllreduceStage, FailsWithNullMpiContext)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = nullptr;

    auto stage = std::make_unique<AllreduceStage>(params);

    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllreduceStage buffer requirements are correct
 */
TEST_F(Test__AllreduceStage, BufferRequirementsAreCorrect)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllreduceStage>(params);

    auto reqs = stage->getBufferRequirements();

    // Should have one in-place buffer (both input and output)
    EXPECT_GE(reqs.buffers.size(), 1);
}

/**
 * @test Verify coherence policy is NONE for MPI stages
 */
TEST_F(Test__AllreduceStage, CoherencePolicyIsNone)
{
    AllreduceStage::Params params;
    params.mpi_ctx = mpi_ctx_.get();

    auto stage = std::make_unique<AllreduceStage>(params);

    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

// =============================================================================
// TPDomain Support Tests (Phase 4.1a)
// =============================================================================

/**
 * @test Verify getDomain() returns nullptr by default
 */
TEST_F(Test__AllreduceStage, GetDomainReturnsNullByDefault)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = mpi_ctx_.get();
    // domain not set - should default to nullptr

    auto stage = std::make_unique<AllreduceStage>(params);

    EXPECT_EQ(stage->getDomain(), nullptr);
}

/**
 * @test Verify getDomain() returns the configured domain
 */
TEST_F(Test__AllreduceStage, GetDomainReturnsConfiguredDomain)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    // Create a test domain
    TPDomain test_domain;
    test_domain.type = TPDomainType::GPU_INTRA_RANK;
    test_domain.domain_size = 2;
    test_domain.local_rank_in_domain = 0;
    test_domain.name = "test_gpu_domain";
    test_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.domain = &test_domain;

    auto stage = std::make_unique<AllreduceStage>(params);

    ASSERT_NE(stage->getDomain(), nullptr);
    EXPECT_EQ(stage->getDomain(), &test_domain);
    EXPECT_EQ(stage->getDomain()->name, "test_gpu_domain");
    EXPECT_EQ(stage->getDomain()->domain_size, 2);
}

/**
 * @test Verify Params stores domain field correctly
 */
TEST_F(Test__AllreduceStage, ParamsStoreDomain)
{
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.domain_size = 4;
    cpu_domain.name = "cpu_cross_rank_domain";

    AllreduceStage::Params params;
    params.domain = &cpu_domain;

    // Verify the field is stored
    EXPECT_EQ(params.domain, &cpu_domain);
    EXPECT_EQ(params.domain->type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_TRUE(params.domain->isCrossRank());
}

/**
 * @test Verify backward compatibility - stage works without domain
 */
TEST_F(Test__AllreduceStage, BackwardCompatibleWithoutDomain)
{
    TensorFactory factory(*mpi_ctx_);
    auto buffer = factory.createFP32({4, 128}, DeviceId::cpu());

    // Legacy params without domain
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = mpi_ctx_.get();
    // params.domain intentionally not set

    auto stage = std::make_unique<AllreduceStage>(params);

    // Stage should be valid and domain should be null
    EXPECT_EQ(stage->getDomain(), nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::ALLREDUCE);
}
