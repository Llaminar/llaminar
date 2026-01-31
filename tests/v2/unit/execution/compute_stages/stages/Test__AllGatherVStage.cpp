/**
 * @file Test__AllGatherVStage.cpp
 * @brief Unit tests for AllGatherVStage compute stage
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests for variable-sized AllGather stage functionality:
 * - Stage type and properties validation
 * - Buffer requirements
 * - Parameter validation (null checks, size mismatches)
 * - Variable recv_counts and displacements handling
 *
 * Note: Full MPI testing with actual data exchange is done in integration tests.
 * This tests the stage interface and validation logic.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/compute_stages/stages/AllGatherVStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "backends/DeviceId.h"
#include "config/TPDomain.h"
#include "../../../../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__AllGatherVStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU);

        // Create a simple mock MPI context for testing (rank 0 of 2)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
    }

    std::unique_ptr<MockDeviceContext> ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// =============================================================================
// AllGatherVStage Unit Tests
// =============================================================================

/**
 * @test Verify AllGatherVStage type returns ALLGATHER_V
 */
TEST_F(Test__AllGatherVStage, StageTypeIsAllGatherV)
{
    // Create minimal params
    AllGatherVStage::Params params;
    params.local_input = nullptr;
    params.full_output = nullptr;
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::ALLGATHER_V);
    // AllGatherV is an MPI collective operation, so it requires collective sync
    EXPECT_TRUE(stage->requiresAllreduce());
}

/**
 * @test Verify AllGatherVStage supports all backends
 */
TEST_F(Test__AllGatherVStage, SupportsAllBackends)
{
    AllGatherVStage::Params params;
    params.local_input = nullptr;
    params.full_output = nullptr;
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    // AllGatherV should support all backends (it's MPI-based, not compute-bound)
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_ROCM));
    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::GPU_VULKAN));
}

/**
 * @test Verify AllGatherVStage fails with null input buffer
 */
TEST_F(Test__AllGatherVStage, FailsWithNullLocalInput)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu()); // [seq_len, total_dim]

    AllGatherVStage::Params params;
    params.local_input = nullptr; // NULL
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Should fail due to null input
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherVStage fails with null output buffer
 */
TEST_F(Test__AllGatherVStage, FailsWithNullFullOutput)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu()); // [seq_len, local_dim]

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = nullptr; // NULL
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Should fail due to null output
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherVStage fails with empty recv_counts
 */
TEST_F(Test__AllGatherVStage, FailsWithEmptyRecvCounts)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {}; // Empty!
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Should fail due to empty recv_counts
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherVStage fails with empty displacements
 */
TEST_F(Test__AllGatherVStage, FailsWithEmptyDisplacements)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {}; // Empty!

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Should fail due to empty displacements
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherVStage fails when recv_counts and displacements size mismatch
 */
TEST_F(Test__AllGatherVStage, FailsWithSizeMismatch)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};       // Size 2
    params.displacements = {0, 64, 128}; // Size 3 - MISMATCH!

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Should fail due to size mismatch
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherVStage fails with null MPI context
 */
TEST_F(Test__AllGatherVStage, FailsWithNullMpiContext)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = nullptr; // NULL - no collective_ctx either
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Should fail due to null mpi_ctx (and no collective_ctx)
    EXPECT_FALSE(stage->execute(ctx_.get()));
}

/**
 * @test Verify AllGatherVStage buffer requirements are correct
 */
TEST_F(Test__AllGatherVStage, BufferRequirementsAreCorrect)
{
    // Create test buffers with known shapes
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());  // [seq_len, local_dim]
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu()); // [seq_len, total_dim]

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    auto reqs = stage->getBufferRequirements();

    // Should have input and output buffers
    EXPECT_EQ(reqs.buffers.size(), 2);
}

/**
 * @test Verify factory creates AllGatherVStage correctly
 */
TEST_F(Test__AllGatherVStage, FactoryCreatesStage)
{
    // Create test buffers
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = ComputeStageFactory::createAllGatherV(params);

    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::ALLGATHER_V);
}

/**
 * @test Verify AllGatherVStage reports correct coherence policy
 */
TEST_F(Test__AllGatherVStage, CoherencePolicyIsNone)
{
    AllGatherVStage::Params params;
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    // MPI stages handle their own synchronization
    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Verify variable recv_counts params for heterogeneous TP scenario
 *
 * This simulates the real use case: NVIDIA GPU with 20 heads (1280 dim)
 * and AMD GPU with 8 heads (512 dim) participating in allgatherv.
 */
TEST_F(Test__AllGatherVStage, HeterogeneousHeadCountParams)
{
    // Create MPI context for 2 ranks
    auto mpi_ctx_2rank = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
    TensorFactory factory(*mpi_ctx_2rank);

    // Rank 0: NVIDIA with 20 heads * 64 head_dim = 1280 elements per token
    // Rank 1: AMD with 8 heads * 64 head_dim = 512 elements per token
    // Total: 1792 elements per token
    int nvidia_dim = 20 * 64;             // 1280
    int amd_dim = 8 * 64;                 // 512
    int total_dim = nvidia_dim + amd_dim; // 1792

    auto local_input = factory.createFP32({4, static_cast<size_t>(nvidia_dim)}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, static_cast<size_t>(total_dim)}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_2rank.get();
    params.recv_counts = {nvidia_dim, amd_dim}; // Different counts!
    params.displacements = {0, nvidia_dim};     // AMD starts at offset 1280

    auto stage = std::make_unique<AllGatherVStage>(params);

    // Stage should be created successfully with heterogeneous params
    EXPECT_EQ(stage->type(), ComputeStageType::ALLGATHER_V);

    auto reqs = stage->getBufferRequirements();
    EXPECT_EQ(reqs.buffers.size(), 2);
}

/**
 * @test Verify usesCollectiveContext() reports correctly
 */
TEST_F(Test__AllGatherVStage, UsesCollectiveContextFlagWhenNoContext)
{
    AllGatherVStage::Params params;
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};
    params.collective_ctx = nullptr; // No CollectiveContext

    auto stage = std::make_unique<AllGatherVStage>(params);

    EXPECT_FALSE(stage->usesCollectiveContext());
}

/**
 * @test Verify getDumpInfo() returns valid information
 */
TEST_F(Test__AllGatherVStage, DumpInfoIsValid)
{
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};

    auto stage = std::make_unique<AllGatherVStage>(params);

    auto dump_info = stage->getDumpInfo();

    // Should have input and output tensors in dump info
    EXPECT_FALSE(dump_info.inputs.empty());
    EXPECT_FALSE(dump_info.outputs.empty());

    // Check scalars include expected parameters
    EXPECT_GE(dump_info.scalars.size(), 2u); // At least world_size and actual_seq_len
}

// =============================================================================
// TPDomain Support Tests (Phase 4.1a)
// =============================================================================

/**
 * @test Verify getDomain() returns nullptr by default
 */
TEST_F(Test__AllGatherVStage, GetDomainReturnsNullByDefault)
{
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {64, 64};
    params.displacements = {0, 64};
    // domain not set - should default to nullptr

    auto stage = std::make_unique<AllGatherVStage>(params);

    EXPECT_EQ(stage->getDomain(), nullptr);
}

/**
 * @test Verify getDomain() returns the configured domain
 */
TEST_F(Test__AllGatherVStage, GetDomainReturnsConfiguredDomain)
{
    TensorFactory factory(*mpi_ctx_);
    auto local_input = factory.createFP32({4, 64}, DeviceId::cpu());
    auto full_output = factory.createFP32({4, 128}, DeviceId::cpu());

    // Create a test domain for heterogeneous GPU TP
    TPDomain test_domain;
    test_domain.type = TPDomainType::GPU_INTRA_RANK;
    test_domain.domain_size = 2;
    test_domain.local_rank_in_domain = 0;
    test_domain.name = "hetero_gpu_domain";
    test_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};

    AllGatherVStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {80, 48}; // Different sizes for heterogeneous heads
    params.displacements = {0, 80};
    params.domain = &test_domain;

    auto stage = std::make_unique<AllGatherVStage>(params);

    ASSERT_NE(stage->getDomain(), nullptr);
    EXPECT_EQ(stage->getDomain(), &test_domain);
    EXPECT_EQ(stage->getDomain()->name, "hetero_gpu_domain");
    EXPECT_EQ(stage->getDomain()->domain_size, 2);
}

/**
 * @test Verify Params stores domain field correctly
 */
TEST_F(Test__AllGatherVStage, ParamsStoreDomain)
{
    TPDomain cpu_domain;
    cpu_domain.type = TPDomainType::CPU_CROSS_RANK;
    cpu_domain.domain_size = 4;
    cpu_domain.name = "cpu_cross_rank_domain";

    AllGatherVStage::Params params;
    params.domain = &cpu_domain;

    // Verify the field is stored
    EXPECT_EQ(params.domain, &cpu_domain);
    EXPECT_EQ(params.domain->type, TPDomainType::CPU_CROSS_RANK);
    EXPECT_TRUE(params.domain->isCrossRank());
}
