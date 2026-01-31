/**
 * @file Test__CollectiveStageIntegration.cpp
 * @brief Integration tests for collective stages with CollectiveContext
 *
 * Tests that AllreduceStage and AllGatherStage correctly integrate with
 * the new collective infrastructure (CollectiveContext, BackendRouter).
 *
 * Test cases:
 * 1. AllreduceStage with CollectiveContext - verifies delegation to context
 * 2. AllreduceStage without context - verifies MPI backward compatibility
 * 3. AllGatherStage with CollectiveContext - verifies delegation to context
 * 4. AllGatherStage without context - verifies MPI backward compatibility
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>

#include "execution/compute_stages/stages/AllreduceStage.h"
#include "execution/compute_stages/stages/AllGatherStage.h"
#include "execution/local_execution/collective/CollectiveContext.h"
#include "collective/test/CollectiveTestMocks.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__CollectiveStageIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create mock MPI context (not a real MPI environment)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);

        // Create mock router and backend
        mock_backend_ = std::make_unique<MockCollectiveBackend>(CollectiveBackendType::HOST);
        mock_router_ = std::make_unique<MockBackendRouter>();

        // Configure mock router to return our mock backend
        mock_router_->setDefaultBackend(mock_backend_.get());

        // Create CollectiveContext with injected mock router
        std::vector<DeviceId> local_devices = {DeviceId::cpu()};
        collective_ctx_ = CollectiveContextFactory::createWithRouter(
            std::move(mock_router_),
            mpi_ctx_,
            local_devices);

        // Create TensorFactory for creating test tensors
        tensor_factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<MockCollectiveBackend> mock_backend_;
    std::unique_ptr<MockBackendRouter> mock_router_;
    std::unique_ptr<CollectiveContext> collective_ctx_;
    std::unique_ptr<TensorFactory> tensor_factory_;

    // Helper to get mock backend after ownership transfer
    MockCollectiveBackend *getMockBackend()
    {
        return mock_backend_.get();
    }
};

// =============================================================================
// AllreduceStage with CollectiveContext Tests
// =============================================================================

/**
 * @test Verify AllreduceStage delegates to CollectiveContext when provided
 */
TEST_F(Test__CollectiveStageIntegration, AllreduceStage_UsesCollectiveContext)
{
    // Create test buffer
    auto buffer = tensor_factory_->createFP32({100}, DeviceId::cpu());
    float *data = buffer->mutable_data();
    for (size_t i = 0; i < 100; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Create stage WITH CollectiveContext
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = nullptr; // Not needed when using CollectiveContext
    params.count = 100;
    params.collective_ctx = collective_ctx_.get();

    AllreduceStage stage(params);

    // Verify stage knows it's using CollectiveContext
    EXPECT_TRUE(stage.usesCollectiveContext());

    // Configure mock to track calls
    bool callback_invoked = false;
    size_t callback_count = 0;
    getMockBackend()->setAllreduceCallback(
        [&](void *buf, size_t count, CollectiveDataType dtype, CollectiveOp op)
        {
            callback_invoked = true;
            callback_count = count;
            EXPECT_EQ(dtype, CollectiveDataType::FLOAT32);
            EXPECT_EQ(op, CollectiveOp::ALLREDUCE_SUM);
        });

    // Execute stage
    bool result = stage.execute(nullptr);
    EXPECT_TRUE(result);

    // Verify CollectiveContext/backend was called
    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(callback_count, 100u);
    EXPECT_EQ(getMockBackend()->allreduceCallCount(), 1);
}

/**
 * @test Verify AllreduceStage handles CollectiveContext failure
 */
TEST_F(Test__CollectiveStageIntegration, AllreduceStage_HandlesContextFailure)
{
    auto buffer = tensor_factory_->createFP32({50}, DeviceId::cpu());

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.collective_ctx = collective_ctx_.get();

    AllreduceStage stage(params);

    // Configure mock to fail
    getMockBackend()->setAllreduceResult(false);

    // Execute should fail
    bool result = stage.execute(nullptr);
    EXPECT_FALSE(result);
}

/**
 * @test Verify AllreduceStage uses buffer size when count is 0
 */
TEST_F(Test__CollectiveStageIntegration, AllreduceStage_UsesBufferSizeWhenCountZero)
{
    auto buffer = tensor_factory_->createFP32({200}, DeviceId::cpu());

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.count = 0; // Should use buffer->numel()
    params.collective_ctx = collective_ctx_.get();

    AllreduceStage stage(params);

    size_t actual_count = 0;
    getMockBackend()->setAllreduceCallback(
        [&](void *buf, size_t count, CollectiveDataType dtype, CollectiveOp op)
        {
            actual_count = count;
        });

    stage.execute(nullptr);

    EXPECT_EQ(actual_count, 200u);
}

// =============================================================================
// AllreduceStage Backward Compatibility Tests
// =============================================================================

/**
 * @test Verify AllreduceStage without CollectiveContext reports correct state
 */
TEST_F(Test__CollectiveStageIntegration, AllreduceStage_WithoutContext_ReportsCorrectState)
{
    auto buffer = tensor_factory_->createFP32({100}, DeviceId::cpu());

    // Create stage WITHOUT CollectiveContext
    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_ctx = mpi_ctx_.get(); // Using legacy MPI path
    params.count = 100;
    params.collective_ctx = nullptr; // No collective context

    AllreduceStage stage(params);

    // Should NOT be using CollectiveContext
    EXPECT_FALSE(stage.usesCollectiveContext());

    // Stage type and properties should still be correct
    EXPECT_EQ(stage.type(), ComputeStageType::ALLREDUCE);
    EXPECT_TRUE(stage.requiresAllreduce());
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Verify AllreduceStage fails gracefully with null buffer
 */
TEST_F(Test__CollectiveStageIntegration, AllreduceStage_FailsWithNullBuffer)
{
    AllreduceStage::Params params;
    params.buffer = nullptr;
    params.collective_ctx = collective_ctx_.get();

    AllreduceStage stage(params);

    // Should fail due to null buffer
    bool result = stage.execute(nullptr);
    EXPECT_FALSE(result);
}

// =============================================================================
// AllGatherStage with CollectiveContext Tests
// =============================================================================

/**
 * @test Verify AllGatherStage delegates to CollectiveContext when provided
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_UsesCollectiveContext)
{
    // Create test tensors: [seq_len, vocab_local] and [seq_len, vocab_full]
    auto local_input = tensor_factory_->createFP32({4, 32}, DeviceId::cpu()); // [4, 32] per rank
    auto full_output = tensor_factory_->createFP32({4, 64}, DeviceId::cpu()); // [4, 64] gathered (2 ranks)

    // Fill local input with test data
    float *data = local_input->mutable_data();
    for (size_t i = 0; i < local_input->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Create stage WITH CollectiveContext
    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = nullptr; // Not needed when using CollectiveContext
    params.actual_seq_len = 4;
    params.collective_ctx = collective_ctx_.get();

    AllGatherStage stage(params);

    // Verify stage knows it's using CollectiveContext
    EXPECT_TRUE(stage.usesCollectiveContext());

    // Configure mock to track calls
    bool callback_invoked = false;
    size_t callback_count = 0;
    getMockBackend()->setAllgatherCallback(
        [&](const void *send_buf, void *recv_buf, size_t count, CollectiveDataType dtype)
        {
            callback_invoked = true;
            callback_count = count;
            EXPECT_EQ(dtype, CollectiveDataType::FLOAT32);
        });

    // Execute stage
    bool result = stage.execute(nullptr);
    EXPECT_TRUE(result);

    // Verify CollectiveContext/backend was called
    EXPECT_TRUE(callback_invoked);
    EXPECT_EQ(getMockBackend()->allgatherCallCount(), 1);
}

/**
 * @test Verify AllGatherStage handles CollectiveContext failure
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_HandlesContextFailure)
{
    auto local_input = tensor_factory_->createFP32({4, 16}, DeviceId::cpu());
    auto full_output = tensor_factory_->createFP32({4, 32}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.collective_ctx = collective_ctx_.get();

    AllGatherStage stage(params);

    // Configure mock to fail
    getMockBackend()->setAllgatherResult(false);

    // Execute should fail
    bool result = stage.execute(nullptr);
    EXPECT_FALSE(result);
}

// =============================================================================
// AllGatherStage Backward Compatibility Tests
// =============================================================================

/**
 * @test Verify AllGatherStage without CollectiveContext reports correct state
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_WithoutContext_ReportsCorrectState)
{
    auto local_input = tensor_factory_->createFP32({4, 16}, DeviceId::cpu());
    auto full_output = tensor_factory_->createFP32({4, 32}, DeviceId::cpu());

    // Create stage WITHOUT CollectiveContext
    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_ctx = mpi_ctx_.get(); // Using legacy MPI path
    params.actual_seq_len = 4;
    params.collective_ctx = nullptr; // No collective context

    AllGatherStage stage(params);

    // Should NOT be using CollectiveContext
    EXPECT_FALSE(stage.usesCollectiveContext());

    // Stage type and properties should still be correct
    EXPECT_EQ(stage.type(), ComputeStageType::ALLGATHER);
    EXPECT_TRUE(stage.requiresAllreduce());
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Verify AllGatherStage fails gracefully with null local_input
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_FailsWithNullLocalInput)
{
    auto full_output = tensor_factory_->createFP32({4, 32}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = nullptr;
    params.full_output = full_output.get();
    params.collective_ctx = collective_ctx_.get();

    AllGatherStage stage(params);

    // Should fail due to null local_input
    bool result = stage.execute(nullptr);
    EXPECT_FALSE(result);
}

/**
 * @test Verify AllGatherStage fails gracefully with null full_output
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_FailsWithNullFullOutput)
{
    auto local_input = tensor_factory_->createFP32({4, 16}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = nullptr;
    params.collective_ctx = collective_ctx_.get();

    AllGatherStage stage(params);

    // Should fail due to null full_output
    bool result = stage.execute(nullptr);
    EXPECT_FALSE(result);
}

/**
 * @test Verify AllGatherStage rejects non-2D tensors
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_Rejects1DTensors)
{
    // Create 1D tensors (invalid for AllGather which expects [seq_len, dim])
    auto local_input = tensor_factory_->createFP32({100}, DeviceId::cpu());
    auto full_output = tensor_factory_->createFP32({200}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.collective_ctx = collective_ctx_.get();

    AllGatherStage stage(params);

    // Should fail due to non-2D tensors
    bool result = stage.execute(nullptr);
    EXPECT_FALSE(result);
}

// =============================================================================
// Stage Properties Tests
// =============================================================================

/**
 * @test Verify AllreduceStage buffer requirements
 */
TEST_F(Test__CollectiveStageIntegration, AllreduceStage_BufferRequirements)
{
    auto buffer = tensor_factory_->createFP32({128, 64}, DeviceId::cpu());

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.collective_ctx = collective_ctx_.get();

    AllreduceStage stage(params);

    auto reqs = stage.getBufferRequirements();
    // Allreduce operates in-place, so it should have buffers
    EXPECT_FALSE(reqs.buffers.empty());
}

/**
 * @test Verify AllGatherStage buffer requirements
 */
TEST_F(Test__CollectiveStageIntegration, AllGatherStage_BufferRequirements)
{
    auto local_input = tensor_factory_->createFP32({4, 32}, DeviceId::cpu());
    auto full_output = tensor_factory_->createFP32({4, 64}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.collective_ctx = collective_ctx_.get();

    AllGatherStage stage(params);

    auto reqs = stage.getBufferRequirements();
    // AllGather has separate input and output, so should have multiple buffers
    EXPECT_GE(reqs.buffers.size(), 2u);
}

/**
 * @test Verify both stages support all backends
 */
TEST_F(Test__CollectiveStageIntegration, CollectiveStages_SupportAllBackends)
{
    auto buffer = tensor_factory_->createFP32({100}, DeviceId::cpu());
    auto local = tensor_factory_->createFP32({4, 16}, DeviceId::cpu());
    auto full = tensor_factory_->createFP32({4, 32}, DeviceId::cpu());

    AllreduceStage::Params allreduce_params;
    allreduce_params.buffer = buffer.get();
    AllreduceStage allreduce_stage(allreduce_params);

    AllGatherStage::Params allgather_params;
    allgather_params.local_input = local.get();
    allgather_params.full_output = full.get();
    AllGatherStage allgather_stage(allgather_params);

    // Both collective stages should support all backends
    // (they're MPI-based, not compute-bound)
    EXPECT_TRUE(allreduce_stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(allreduce_stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(allreduce_stage.supportsBackend(ComputeBackendType::GPU_ROCM));

    EXPECT_TRUE(allgather_stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(allgather_stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_TRUE(allgather_stage.supportsBackend(ComputeBackendType::GPU_ROCM));
}
